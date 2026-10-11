#!/usr/bin/env python3
"""IK node: /ball_position target -> SO-ARM101 joint angles on /arm/<joint>.

Input:  /ball_position  std_msgs/Float64MultiArray [x, y, z, roll, pitch, yaw]
        OFFSET FROM HOME. Home = the mid-workspace pose (so101_mirror.model:
        joint-zero end-effector + HOME_OFFSET, 10 cm back, 5 cm down, gripper
        level). [0,0,0,0,0,0] = stay home.
        xyz: meters along the base_link axes (+X forward, +Y left, +Z up).
        rpy: rotation (base_link axes, R = Rz Ry Rx) applied to the home
        gripper orientation. Target = (home_p + xyz, R(rpy) @ home_R).
        /wrist_roll_cmd std_msgs/Float64, rad (optional, two-hand mode): while
                fresh (< stale_s) wrist_roll is held at it and the other joints
                solve position + pitch around it
Output: /arm/shoulder_pan, /arm/shoulder_lift, /arm/elbow_flex,
        /arm/wrist_flex, /arm/wrist_roll   std_msgs/Float64, radians (URDF)

The arm has 5 DOF: position is always solved first, orientation is best
effort with the remaining DOF (in practice: approach pitch in the arm's
vertical plane + roll about the approach axis; yaw is tied to position).

Solving is so101_mirror.ik_solver.TargetSolver (pure, see its docstring):
warm start from the last solution, limit-aware task-priority IK (position
first, orientation best effort), deterministic reseeds and a time budget when
the warm start misses, and a branch guard against sudden joint flips.

Safety/robustness (the real arm follows this output):
    - targets are interpolated to the solve rate (so101_mirror.target_filter):
      camera-rate targets arrive irregularly, and solving on the latest one
      gave the arm uneven steps; a workspace box rejects NaN / wild targets
    - unreachable (pos_err > max_pos_err), stale or branch-flipping targets:
      publish nothing, so downstream holds the last good pose
    - joint limits: URDF intersected with the real arm's soft limits
      (control/so101_bus.py SOFT_LIMITS, which feetech_mirror clamps to), so
      the IK never relies on a pose the arm will refuse
    The hardware node still rate-limits and clamps; this is not the last line.

Usage:
    ros2 run so101_mirror ik_node.py
    ros2 run so101_mirror ik_node.py --ros-args -p use_orientation:=false

Parameters:
    urdf_path        xacro/urdf file (default: installed so101.urdf.xacro)
    use_orientation  track ball rpy (true) or position only (false)
    free_yaw         ignore the target yaw and use the yaw the pan joint gives
                     at that position (true): a 5-DOF arm can't set yaw on its
                     own, and an unreachable yaw used to make the whole pose
                     fall back to position-only (gripper flips down)
    rot_weight       orientation task gain (0..1)
    rot_damping      orientation task damping: higher = smoother where the
                     target orientation is out of reach, less exact
    jump_deg, jump_confirm  branch guard: a solution moving any joint more
                     than jump_deg is published only once jump_confirm
                     consecutive ticks agree
    solve_budget_s   time for reseeding when the warm start misses
    hw_limits_file   so101_bus.py to read SOFT_LIMITS from ('' = URDF only)
    interp_delay     sample the target path this far behind real time (s),
                     linearly between the targets around that moment; ~one
                     normal gap between targets. 0 = latest target, as is.
                     With interp_auto_max > 0 this is the minimum
    interp_auto_max  > 0: the delay follows the target rate (0.9 x the median
                     recent gap, at most this, s); 0 (default) = fixed interp_delay
                     (the auto delay added ~50 ms of lag in two-hand mode)
    smooth_tau       extra time-based smoothing of the target (s), 0 = off
                     (smoothing belongs in the CV node; this replaced the
                     per-message EMA `alpha`, which smoothed more at higher fps)
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
from so101_mirror.ik_solver import TargetSolver
from so101_mirror.target_filter import TargetInterpolator
from so101_mirror.model import (HW_LIMITS_FILE, apply_limits, default_urdf_path, hardware_limits,
                                home_pose, load_chain)


class IKNode(Node):
    def __init__(self):
        super().__init__("ik_node")
        self.declare_parameter("urdf_path", default_urdf_path())
        self.declare_parameter("use_orientation", True)
        self.declare_parameter("free_yaw", True)
        self.declare_parameter("rot_weight", 0.5)
        self.declare_parameter("rot_damping", 0.1)
        self.declare_parameter("jump_deg", 30.0)
        self.declare_parameter("jump_confirm", 3)
        self.declare_parameter("solve_budget_s", 0.02)
        self.declare_parameter("hw_limits_file", HW_LIMITS_FILE)
        self.declare_parameter("interp_delay", 0.04)
        self.declare_parameter("interp_auto_max", 0.0)
        self.declare_parameter("smooth_tau", 0.0)
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
        changed = apply_limits(self.chain, hardware_limits(gp("hw_limits_file")))
        if changed:
            self.get_logger().info("joint limits narrowed to the arm's soft limits: " + ", ".join(
                f"{j} {math.degrees(lo):+.0f}..{math.degrees(hi):+.0f}" for j, (lo, hi) in changed.items()))
        elif gp("hw_limits_file"):
            self.get_logger().warn(f"no soft limits from {gp('hw_limits_file')!r}: URDF limits only")
        self.solver = TargetSolver(
            self.chain, self.home_p, self.home_R, use_orientation=gp("use_orientation"),
            free_yaw=gp("free_yaw"), rot_weight=gp("rot_weight"), rot_damping=gp("rot_damping"),
            max_pos_err=gp("max_pos_err"), jump_deg=gp("jump_deg"), confirm=gp("jump_confirm"),
            budget_s=gp("solve_budget_s"))
        self.ws_min = np.array(gp("workspace_min"))
        self.ws_max = np.array(gp("workspace_max"))

        self.interp = TargetInterpolator(delay=gp("interp_delay"), tau=gp("smooth_tau"),
                                         auto_max=gp("interp_auto_max"))
        self.target = None                   # latest accepted offset [x y z r p y]
        self.target_t = None
        self.pubs = {j: self.create_publisher(Float64, ARM_TOPIC_PREFIX + j, 10)
                     for j in ARM_JOINTS}
        self.create_subscription(Float64MultiArray, BALL_TOPIC, self.on_target, 10)
        self.roll_cmd = None  # (rad, time) from WRIST_ROLL_TOPIC
        self.create_subscription(Float64, WRIST_ROLL_TOPIC, self.on_roll, 10)
        self.create_timer(1.0 / gp("rate"), self.tick)
        self.get_logger().info(f"IK ready: {BALL_TOPIC} (offset from home EE "
                               f"{np.round(self.home_p, 4)}) -> {ARM_TOPIC_PREFIX}<joint>, "
                               f"limits lo={np.round(self.chain.lower, 2)} hi={np.round(self.chain.upper, 2)}")

    def on_roll(self, msg):
        if math.isfinite(msg.data):
            self.roll_cmd = (float(msg.data), self.get_clock().now())

    def roll_lock(self):
        """The other hand's wrist_roll command while fresh, else None (IK picks the roll)."""
        if self.roll_cmd is None:
            return None
        age = (self.get_clock().now() - self.roll_cmd[1]).nanoseconds * 1e-9
        return self.roll_cmd[0] if age <= self.get_parameter("stale_s").value else None

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
        now = self.get_clock().now()
        if self.target_t is not None and \
                (now - self.target_t).nanoseconds * 1e-9 > self.get_parameter("stale_s").value:
            self.interp.reset()  # don't interpolate across a long gap (stale) from an old target
        self.target = d
        self.target_t = now
        self.interp.add(now.nanoseconds * 1e-9, d)

    def tick(self):
        if self.target is None:
            return
        age = (self.get_clock().now() - self.target_t).nanoseconds * 1e-9
        if age > self.get_parameter("stale_s").value:
            self.get_logger().warn(f"target stale ({age:.1f} s), holding", throttle_duration_sec=5.0)
            return
        target = self.interp.sample(self.get_clock().now().nanoseconds * 1e-9)
        q = self.solver.solve(target, roll_lock=self.roll_lock())
        info = self.solver.info
        if q is None:
            p = self.home_p + target[:3]
            self.get_logger().warn(f"{info['how']}: target {np.round(p, 3)} "
                                   f"(best err {info['pos_err'] * 1000:.0f} mm), holding",
                                   throttle_duration_sec=1.0)
            return
        for j, v in zip(ARM_JOINTS, q):
            self.pubs[j].publish(Float64(data=float(v)))
        self.get_logger().info(
            "q(deg)=" + " ".join(f"{j}:{math.degrees(v):+.0f}" for j, v in zip(ARM_JOINTS, q))
            + f"  pos_err={info['pos_err'] * 1000:.1f}mm rot_err={math.degrees(info['rot_err']):.0f}deg"
            f" [{info['how']}, {info['ms']:.0f} ms"
            + ("" if info["roll_lock"] is None else f", roll locked {math.degrees(info['roll_lock']):+.0f}")
            + "]",
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
