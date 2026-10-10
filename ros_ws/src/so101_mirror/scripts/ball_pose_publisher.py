#!/usr/bin/env python3
"""Publish the Gazebo ball's pose, as an offset from home, on /ball_position.

Input:  /sim/ball_pose  geometry_msgs/PoseStamped, world frame (bridged from
        gz /model/ball/pose by sim.launch.py)
Output: /ball_position  std_msgs/Float64MultiArray
        data = [x, y, z, roll, pitch, yaw]  (meters, radians), OFFSET FROM HOME:
        xyz = ball position minus the home end-effector position (base pose,
        all joints 2048), along the base_link axes. rpy = the ball's rotation
        (base_link axes, R = Rz(yaw) Ry(pitch) Rx(roll)); the ball spawns at
        home with rpy 0, so the first message is [0, 0, 0, 0, 0, 0].

The vision pipeline replaces THIS node: it publishes the same message on the
same topic and nothing downstream changes (see ~/kh/INTEGRATION.md).

Usage:
    ros2 run so101_mirror ball_pose_publisher.py
    ros2 run so101_mirror ball_pose_publisher.py --ros-args -p rate:=30.0 \
        -p base_xyz:="[0.0, 0.0, 0.0]" -p base_yaw:=0.0

Parameters:
    rate      publish rate in Hz (re-publishes the latest pose)
    base_xyz  base_link origin in the world frame (the launch file spawns
              the arm at the world origin, so the default is zeros)
    base_yaw  base_link yaw in the world frame
    stale_s   stop publishing if no Gazebo pose arrived for this long
"""
import math

import numpy as np
import rclpy
from geometry_msgs.msg import PoseStamped
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray

from so101_mirror import BALL_TOPIC
from so101_mirror.kinematics import matrix_to_rpy, rot_z
from so101_mirror.model import home_pose, load_chain


def quat_to_matrix(x, y, z, w):
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


class BallPosePublisher(Node):
    def __init__(self):
        super().__init__("ball_pose_publisher")
        self.declare_parameter("rate", 30.0)
        self.declare_parameter("base_xyz", [0.0, 0.0, 0.0])
        self.declare_parameter("base_yaw", 0.0)
        self.declare_parameter("stale_s", 1.0)
        self.base_p = np.array(self.get_parameter("base_xyz").value, dtype=float)
        self.base_R = rot_z(self.get_parameter("base_yaw").value)
        self.stale_s = self.get_parameter("stale_s").value
        self.home_p, _ = home_pose(load_chain())

        self.latest = None
        self.latest_t = None
        self.pub = self.create_publisher(Float64MultiArray, BALL_TOPIC, 10)
        self.create_subscription(PoseStamped, "/sim/ball_pose", self.on_pose, 10)
        self.create_timer(1.0 / self.get_parameter("rate").value, self.tick)
        self.get_logger().info(f"waiting for /sim/ball_pose, publishing {BALL_TOPIC} "
                               f"as offset from home EE {np.round(self.home_p, 4)}")

    def on_pose(self, msg):
        p, q = msg.pose.position, msg.pose.orientation
        R_world = quat_to_matrix(q.x, q.y, q.z, q.w)
        # world -> base_link
        p_base = self.base_R.T @ (np.array([p.x, p.y, p.z]) - self.base_p)
        R_base = self.base_R.T @ R_world
        self.latest = [*(p_base - self.home_p), *matrix_to_rpy(R_base)]
        self.latest_t = self.get_clock().now()

    def tick(self):
        if self.latest is None:
            self.get_logger().warn("no ball pose from Gazebo yet", throttle_duration_sec=5.0)
            return
        age = (self.get_clock().now() - self.latest_t).nanoseconds * 1e-9
        if age > self.stale_s:
            self.get_logger().warn(f"ball pose stale ({age:.1f} s), not publishing",
                                   throttle_duration_sec=5.0)
            return
        self.pub.publish(Float64MultiArray(data=[float(v) for v in self.latest]))
        x, y, z, r, pi, yw = self.latest
        self.get_logger().info(
            f"ball offset xyz=({x:+.3f}, {y:+.3f}, {z:+.3f}) m  "
            f"rpy=({math.degrees(r):+.0f}, {math.degrees(pi):+.0f}, {math.degrees(yw):+.0f}) deg",
            throttle_duration_sec=2.0)


def main():
    rclpy.init()
    node = BallPosePublisher()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
