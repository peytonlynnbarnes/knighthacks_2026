"""SO-ARM101 end-effector mirror (see ~/kh/INTEGRATION.md)."""

# Topic contract. Change here and in ~/kh/control/feetech_mirror.py together.
BALL_TOPIC = "/ball_position"     # std_msgs/Float64MultiArray [x, y, z, roll, pitch, yaw]
ARM_TOPIC_PREFIX = "/arm/"        # /arm/<joint>, std_msgs/Float64, radians (URDF convention)
WRIST_ROLL_TOPIC = "/wrist_roll_cmd"  # std_msgs/Float64, rad (URDF): optional direct wrist_roll override
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
