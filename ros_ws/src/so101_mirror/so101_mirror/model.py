"""Load the SO-101 chain from the installed URDF and compute the home pose.

Home = the mid-workspace pose: the end-effector of the joint-zero pose (all
arm joints 0 rad = servo 2048 = base_pose.py, the LeRobot calibration
L-pose, near full forward reach) moved by HOME_OFFSET, gripper level and
pointing +X as at joint zero. From there there's room to move every way.
/ball_position is an offset from home, so both ball_pose_publisher.py and
ik_node.py use home_pose() from here (and mirror_world.sdf spawns the ball
there).
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


# home relative to the joint-zero end-effector (m, base_link): 10 cm back, 5 cm
# down = the middle of the reachable box. IK there (level gripper):
# pan 0, lift -51, elbow 64, wrist_flex -13, roll 0 deg.
HOME_OFFSET = np.array([-0.10, 0.0, -0.05])


HW_LIMITS_FILE = os.path.expanduser("~/kh/control/so101_bus.py")


def hardware_limits(path=HW_LIMITS_FILE):
    """{joint: (lo, hi) rad} from the real arm's SOFT_LIMITS (ticks) and JOINT_CAL in
    control/so101_bus.py, or {} if the file is missing.

    feetech_mirror clamps every goal to these, so an IK solution outside them
    puts the real arm somewhere else than the IK thinks. The file is parsed,
    not imported (it needs scservo_sdk, which the ROS python lacks).
    """
    import ast
    if not path or not os.path.exists(path):
        return {}
    env = {}
    for node in ast.parse(open(path).read()).body:
        if (isinstance(node, ast.Assign) and len(node.targets) == 1
                and isinstance(node.targets[0], ast.Name)
                and node.targets[0].id in ("TICKS_PER_REV", "CENTER", "JOINT_CAL", "SOFT_LIMITS")):
            env[node.targets[0].id] = eval(compile(ast.Expression(node.value), path, "eval"),
                                           {"__builtins__": {}}, env)
    out = {}
    for name, (sign, zero) in env.get("JOINT_CAL", {}).items():
        if name in env.get("SOFT_LIMITS", {}):
            a, b = (sign * (t - zero) * 2 * np.pi / env["TICKS_PER_REV"] for t in env["SOFT_LIMITS"][name])
            out[name] = (min(a, b), max(a, b))
    return out


def apply_limits(chain, limits, margin=np.radians(2.0)):
    """Narrow the chain's joint limits to their intersection with `limits`
    ({joint: (lo, hi) rad}), minus a margin. Returns {joint: (lo, hi)} changed."""
    changed = {}
    for i, name in enumerate(chain.active):
        if name in limits:
            lo = max(chain.lower[i], limits[name][0] + margin)
            hi = min(chain.upper[i], limits[name][1] - margin)
            if (lo, hi) != (chain.lower[i], chain.upper[i]) and lo < hi:
                chain.lower[i], chain.upper[i] = lo, hi
                changed[name] = (lo, hi)
    return changed


def zero_pose(chain):
    """(position (3,), rotation (3,3)) of the tool frame at q = 0, base_link frame."""
    T = chain.fk(np.zeros(len(chain.active)))
    return T[:3, 3].copy(), T[:3, :3].copy()


def home_pose(chain):
    """(position (3,), rotation (3,3)) of home (mid-workspace), base_link frame."""
    p, R = zero_pose(chain)
    return p + HOME_OFFSET, R
