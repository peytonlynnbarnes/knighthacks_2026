#!/usr/bin/env python3
"""IK node: /ball_position target -> SO-ARM101 joint angles on /arm/<joint>.

Input:  /ball_position  std_msgs/Float64MultiArray [x, y, z, roll, pitch, yaw]
        OFFSET FROM HOME. Home = the end-effector pose at the base pose (all
        joints 0 rad = servo 2048, base_pose.py). [0,0,0,0,0,0] = stay home.
        xyz: meters along the base_link axes (+X forward, +Y left, +Z up).
        rpy: rotation (base_link axes, R = Rz Ry Rx) applied to the home
        gripper orientation. Target = (home_p + xyz, R(rpy) @ home_R).
Output: /arm/shoulder_pan, /arm/shoulder_lift, /arm/elbow_flex,
        /arm/wrist_flex, /arm/wrist_roll   std_msgs/Float64, radians (URDF)

The arm has 5 DOF: position is always solved first, orientation is best
effort with the remaining DOF (in practice: approach pitch in the arm's
vertical plane + roll about the approach axis; yaw is tied to position).

Safety/robustness (the real arm follows this output):
    - target EMA filter (alpha) and a workspace box reject NaN / wild targets
    - warm start from the last solution; random restarts only if that fails
    - unreachable (pos_err > max_pos_err) or stale targets: publish nothing,
      so downstream holds the last good pose
    - joint limits from the URDF
    The hardware node still rate-limits and clamps; this is not the last line.

Usage:
    ros2 run so101_mirror ik_node.py
    ros2 run so101_mirror ik_node.py --ros-args -p use_orientation:=false

Parameters:
    urdf_path        xacro/urdf file (default: installed so101.urdf.xacro)
    use_orientation  track ball rpy (true) or position only (false)
    (topic) /wrist_roll_cmd  Float64 rad: while fresh (< stale_s), wrist_roll is
                     held at this value and the other joints solve position +
                     pitch around it (two-hand mode: the other hand turns the roll)
    jump_deg, jump_confirm  a solution more than jump_deg (any joint) from the
                     last published one is held back until jump_confirm
                     consecutive solves agree (stops IK branch flips and
                     glitches sending the arm on multi-second detours)
    free_yaw         ignore the target yaw and use the yaw the pan joint gives
                     at that position (true): a 5-DOF arm can't set yaw on its
                     own, and an unreachable yaw used to make the whole pose
                     fall back to position-only (gripper flips down)
    rot_weight       orientation task gain (0..1)
    alpha            EMA factor on the target, 1 = no filtering
    max_pos_err      reject solutions worse than this (m)
    rate             solve/publish rate (Hz)
    stale_s          stop publishing when no target for this long (s)
    workspace_min/max  box for the ABSOLUTE target (home + offset) in
                     base_link (m); outside -> rejected
"""
import math

import numpy as np
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64, Float64MultiArray

from so101_mirror import ARM_JOINTS, ARM_TOPIC_PREFIX, BALL_TOPIC, WRIST_ROLL_TOPIC
from so101_mirror.kinematics import (BranchGuard, rpy_to_matrix, solve_ik, solve_ik_restarts,
                                    solve_relaxed_pitch)
from so101_mirror.model import default_urdf_path, home_pose, load_chain


class IKNode(Node):
    def __init__(self):
        super().__init__("ik_node")
        self.declare_parameter("urdf_path", default_urdf_path())
        self.declare_parameter("use_orientation", True)
        self.declare_parameter("free_yaw", True)
        self.declare_parameter("jump_deg", 30.0)
        self.declare_parameter("jump_confirm", 3)
        self.declare_parameter("rot_weight", 0.5)
        self.declare_parameter("alpha", 0.5)
        self.declare_parameter("max_pos_err", 0.01)
        self.declare_parameter("rate", 30.0)
        self.declare_parameter("stale_s", 0.5)
        self.declare_parameter("workspace_min", [-0.45, -0.45, -0.05])
        self.declare_parameter("workspace_max", [0.45, 0.45, 0.50])
        gp = lambda n: self.get_parameter(n).value  # noqa: E731

        self.chain = load_chain(gp("urdf_path"))
        if tuple(self.chain.active) != ARM_JOINTS:
            raise RuntimeError(f"URDF chain joints {self.chain.active} != {ARM_JOINTS}")
        self.home_p, self.home_R = home_pose(self.chain)
        _, frames = self.chain.fk(np.zeros(len(self.chain.active)), return_frames=True)
        self.pan_xy = frames[0][0][:2]  # pan axis location in base_link
        self.ws_min = np.array(gp("workspace_min"))
        self.ws_max = np.array(gp("workspace_max"))

        self.q = np.zeros(len(ARM_JOINTS))  # warm start = last published (zero = L-pose, all 2048)
        self.published = False
        self.guard = BranchGuard(math.radians(gp("jump_deg")), int(gp("jump_confirm")))
        self.target = None                   # filtered offset [x y z r p y]
        self.target_t = None
        self.pubs = {j: self.create_publisher(Float64, ARM_TOPIC_PREFIX + j, 10)
                     for j in ARM_JOINTS}
        self.create_subscription(Float64MultiArray, BALL_TOPIC, self.on_target, 10)
        self.roll_cmd = None  # (rad, time) from WRIST_ROLL_TOPIC
        self.roll_idx = ARM_JOINTS.index("wrist_roll")
        self.create_subscription(Float64, WRIST_ROLL_TOPIC, self.on_roll, 10)
        self.create_timer(1.0 / gp("rate"), self.tick)
        self.get_logger().info(f"IK ready: {BALL_TOPIC} (offset from home EE "
                               f"{np.round(self.home_p, 4)}) -> {ARM_TOPIC_PREFIX}<joint>, "
                               f"limits lo={np.round(self.chain.lower, 2)} hi={np.round(self.chain.upper, 2)}")

    def on_roll(self, msg):
        if math.isfinite(msg.data):
            self.roll_cmd = (msg.data, self.get_clock().now())

    def on_target(self, msg):
        d = np.array(msg.data, dtype=float)
        if d.shape != (6,) or not np.all(np.isfinite(d)):
            self.get_logger().warn(f"bad {BALL_TOPIC} message {list(msg.data)}; expected 6 finite floats",
                                   throttle_duration_sec=2.0)
            return
        p_abs = self.home_p + d[:3]
        if np.any(p_abs < self.ws_min) or np.any(p_abs > self.ws_max):
            self.get_logger().warn(f"target {np.round(p_abs, 3)} (offset {np.round(d[:3], 3)}) "
                                   "outside workspace box, ignored",
                                   throttle_duration_sec=2.0)
            return
        a = self.get_parameter("alpha").value
        if self.target is None:
            self.target = d
        else:
            # EMA; angles are filtered through their wrapped difference
            diff = d - self.target
            diff[3:] = (diff[3:] + math.pi) % (2 * math.pi) - math.pi
            self.target = self.target + a * diff
        self.target_t = self.get_clock().now()

    def tick(self):
        if self.target is None:
            return
        age = (self.get_clock().now() - self.target_t).nanoseconds * 1e-9
        if age > self.get_parameter("stale_s").value:
            self.get_logger().warn(f"target stale ({age:.1f} s), holding", throttle_duration_sec=5.0)
            return
        p = self.home_p + self.target[:3]
        lock = None
        if self.roll_cmd is not None and \
                (self.get_clock().now() - self.roll_cmd[1]).nanoseconds * 1e-9 <= self.get_parameter("stale_s").value:
            i = self.roll_idx
            lock = {i: float(np.clip(self.roll_cmd[0], self.chain.lower[i], self.chain.upper[i]))}
        R = None
        if self.get_parameter("use_orientation").value:
            roll, pitch, yaw = self.target[3:]
            if lock is not None:
                roll = -lock[self.roll_idx]  # the locked joint's roll (tool roll = -wrist_roll in rpy)
            if self.get_parameter("free_yaw").value:
                yaw = math.atan2(p[1] - self.pan_xy[1], p[0] - self.pan_xy[0])
            R = rpy_to_matrix(roll, pitch, yaw) @ self.home_R
        kw = dict(rot_weight=self.get_parameter("rot_weight").value, lock=lock)
        tol = self.get_parameter("max_pos_err").value

        res = solve_ik(self.chain, p, R, q0=self.q, max_iters=50, **kw)
        if res.pos_err > tol:  # warm start stuck: try fresh seeds
            res = solve_ik_restarts(self.chain, p, R, q0=self.q, accept_tol=tol / 2, **kw)
        if res.pos_err > tol and R is not None:
            # pitch unreachable here: use the nearest pitch that is, from the current
            # pose, before dropping orientation (which can flip the gripper over)
            r2, used = solve_relaxed_pitch(self.chain, p, roll, pitch, yaw, self.home_R, self.q, tol, **kw)
            if r2 is not None:
                res = r2
                self.get_logger().warn(f"pitch {math.degrees(pitch):.0f} deg unreachable, "
                                       f"using {math.degrees(used):.0f}", throttle_duration_sec=2.0)
        if res.pos_err > tol and R is not None:
            # orientation is best effort: give it up before giving up position.
            # From the current pose first: random seeds land on far-away branches.
            res = solve_ik(self.chain, p, None, q0=self.q, max_iters=150, lock=lock)
            if res.pos_err > tol:
                res = solve_ik_restarts(self.chain, p, None, q0=self.q, accept_tol=tol / 2, lock=lock)
            if res.pos_err <= tol:
                self.get_logger().warn("pose unreachable with orientation, using position only",
                                       throttle_duration_sec=2.0)
        if res.pos_err > tol:
            self.get_logger().warn(f"unreachable target {np.round(p, 3)} "
                                   f"(best err {res.pos_err * 1000:.0f} mm), holding",
                                   throttle_duration_sec=1.0)
            return

        if not self.guard(self.q if self.published else None, res.q):
            jump = math.degrees(np.max(np.abs(res.q - self.q)))
            self.get_logger().warn(f"solution jumps {jump:.0f} deg from the last one: holding until it "
                                   "persists (branch flip / glitch filter)", throttle_duration_sec=2.0)
            return
        self.q = res.q
        self.published = True
        for j, v in zip(ARM_JOINTS, self.q):
            self.pubs[j].publish(Float64(data=float(v)))
        self.get_logger().info(
            "q(deg)=" + " ".join(f"{j}:{math.degrees(v):+.0f}" for j, v in zip(ARM_JOINTS, self.q))
            + f"  pos_err={res.pos_err * 1000:.1f}mm rot_err={math.degrees(res.rot_err):.0f}deg",
            throttle_duration_sec=2.0)


def main():
    rclpy.init()
    node = IKNode()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
