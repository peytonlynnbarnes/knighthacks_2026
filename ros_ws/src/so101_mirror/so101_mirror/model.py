"""Load the SO-101 chain from the installed URDF and compute the home pose.

Home = all arm joints at 0 rad = servo 2048 (base_pose.py), the LeRobot
calibration L-pose. /ball_position is an offset from the home end-effector
pose, so both ball_pose_publisher.py and ik_node.py use home_pose() from here.
"""
import os

import numpy as np

from so101_mirror.kinematics import Chain


def default_urdf_path():
    from ament_index_python.packages import get_package_share_directory
    return os.path.join(get_package_share_directory("so101_mirror"),
                        "description", "so101.urdf.xacro")


def load_chain(path=None):
    path = path or default_urdf_path()
    if path.endswith(".xacro"):
        import xacro
        xml = xacro.process_file(path).toxml()
    else:
        with open(path) as f:
            xml = f.read()
    return Chain.from_urdf(xml)


def home_pose(chain):
    """(position (3,), rotation (3,3)) of the tool frame at q = 0, base_link frame."""
    T = chain.fk(np.zeros(len(chain.active)))
    return T[:3, 3].copy(), T[:3, :3].copy()
