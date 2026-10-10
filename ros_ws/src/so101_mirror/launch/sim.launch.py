"""End-effector mirror: Gazebo world with only the target ball -> IK -> /arm/<joint>.

There is no simulated arm. The IK output on /arm/<joint> is for the real
arm (~/kh/control/feetech_mirror.py, run separately, dry run by default).

    ros2 launch so101_mirror sim.launch.py                 # GUI, drag the ball
    ros2 launch so101_mirror sim.launch.py sweep:=true     # ball moves by itself
    ros2 launch so101_mirror sim.launch.py headless:=true sweep:=true
    ros2 launch so101_mirror sim.launch.py ball_source:=false   # vision publishes /ball_position
    ros2 launch so101_mirror sim.launch.py ik:=false            # something else publishes /arm/*
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.conditions import IfCondition
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PythonExpression
from launch_ros.actions import Node

WORLD_NAME = "mirror_world"


def generate_launch_description():
    pkg = get_package_share_directory("so101_mirror")
    world_file = os.path.join(pkg, "worlds", "mirror_world.sdf")

    headless = LaunchConfiguration("headless")
    sweep = LaunchConfiguration("sweep")
    ball_source = LaunchConfiguration("ball_source")
    ik = LaunchConfiguration("ik")
    use_orientation = LaunchConfiguration("use_orientation")

    args = [
        DeclareLaunchArgument("headless", default_value="false",
                              description="Gazebo server only, no GUI"),
        DeclareLaunchArgument("sweep", default_value="false",
                              description="move the ball automatically (ball_sweep.py)"),
        DeclareLaunchArgument("ball_source", default_value="true",
                              description="publish /ball_position from the Gazebo ball; "
                                          "false when vision publishes it"),
        DeclareLaunchArgument("ik", default_value="true", description="run ik_node.py"),
        DeclareLaunchArgument("use_orientation", default_value="true",
                              description="IK tracks ball orientation (best effort)"),
    ]

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(os.path.join(
            get_package_share_directory("ros_gz_sim"), "launch", "gz_sim.launch.py")),
        launch_arguments={"gz_args": [
            PythonExpression(["'-r -s --headless-rendering ' if '", headless,
                              "'.lower() in ('true', '1') else '-r '"]),
            world_file]}.items())

    bridge = Node(
        package="ros_gz_bridge", executable="parameter_bridge",
        arguments=[
            "/model/ball/pose@geometry_msgs/msg/PoseStamped[gz.msgs.Pose",
            f"/world/{WORLD_NAME}/set_pose@ros_gz_interfaces/srv/SetEntityPose",
        ],
        remappings=[("/model/ball/pose", "/sim/ball_pose")],
        output="screen")

    ball_pub = Node(package="so101_mirror", executable="ball_pose_publisher.py",
                    condition=IfCondition(ball_source), output="screen")
    ball_sweep = Node(package="so101_mirror", executable="ball_sweep.py",
                      parameters=[{"world": WORLD_NAME}],
                      condition=IfCondition(sweep), output="screen")
    ik_node = Node(package="so101_mirror", executable="ik_node.py",
                   parameters=[{"use_orientation": use_orientation}],
                   condition=IfCondition(ik), output="screen")

    return LaunchDescription(args + [
        gazebo,
        bridge,
        TimerAction(period=3.0, actions=[ball_pub, ball_sweep, ik_node]),
    ])
