#!/usr/bin/env python3
"""Move the Gazebo ball along a slow figure-8 with a wobbling orientation.

For testing without the GUI (or hands-free). It calls the bridged
/world/mirror_world/set_pose service; ball_pose_publisher.py then picks up
the new pose like it would after a GUI drag.

Usage:
    ros2 run so101_mirror ball_sweep.py
    ros2 run so101_mirror ball_sweep.py --ros-args -p center:="[0.30, 0.0, 0.20]" \
        -p amplitude:="[0.0, 0.10, 0.05]" -p period:=12.0 -p tilt_deg:=30.0
"""
import math

import rclpy
from geometry_msgs.msg import Pose
from rclpy.node import Node
from ros_gz_interfaces.msg import Entity
from ros_gz_interfaces.srv import SetEntityPose


def rpy_to_quat(r, p, y):
    cr, sr = math.cos(r / 2), math.sin(r / 2)
    cp, sp = math.cos(p / 2), math.sin(p / 2)
    cy, sy = math.cos(y / 2), math.sin(y / 2)
    return (sr * cp * cy - cr * sp * sy, cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy, cr * cp * cy + sr * sp * sy)


class BallSweep(Node):
    def __init__(self):
        super().__init__("ball_sweep")
        self.declare_parameter("world", "mirror_world")
        self.declare_parameter("center", [0.30, 0.0, 0.20])
        self.declare_parameter("amplitude", [0.0, 0.10, 0.05])
        self.declare_parameter("period", 12.0)
        self.declare_parameter("tilt_deg", 30.0)
        self.declare_parameter("rate", 20.0)
        world = self.get_parameter("world").value
        self.cli = self.create_client(SetEntityPose, f"/world/{world}/set_pose")
        self.t0 = self.get_clock().now()
        self.pending = None
        self.create_timer(1.0 / self.get_parameter("rate").value, self.tick)

    def tick(self):
        if not self.cli.service_is_ready():
            self.get_logger().warn("waiting for set_pose service", throttle_duration_sec=5.0)
            return
        if self.pending is not None and not self.pending.done():
            return  # don't queue up requests
        t = (self.get_clock().now() - self.t0).nanoseconds * 1e-9
        w = 2 * math.pi * t / self.get_parameter("period").value
        c = self.get_parameter("center").value
        a = self.get_parameter("amplitude").value
        tilt = math.radians(self.get_parameter("tilt_deg").value)
        req = SetEntityPose.Request()
        req.entity = Entity(name="ball", type=Entity.MODEL)
        pose = Pose()
        pose.position.x = c[0] + a[0] * math.sin(w)
        pose.position.y = c[1] + a[1] * math.sin(w)
        pose.position.z = c[2] + a[2] * math.sin(2 * w)
        qx, qy, qz, qw = rpy_to_quat(tilt * math.sin(w), tilt * math.sin(2 * w) * 0.5, 0.0)
        pose.orientation.x, pose.orientation.y, pose.orientation.z, pose.orientation.w = qx, qy, qz, qw
        req.pose = pose
        self.pending = self.cli.call_async(req)


def main():
    rclpy.init()
    node = BallSweep()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
