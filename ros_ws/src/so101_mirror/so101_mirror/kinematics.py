"""SO-ARM101 forward/inverse kinematics, pure numpy, no ROS.

The chain is read from the URDF (so101.urdf.xacro is the single source of
truth for link lengths and joint axes). Angles are radians in the URDF
convention, positions are meters in the base_link frame.

The arm has 5 arm joints, so a full 6-DOF pose is generally not reachable.
IK is task-priority damped least squares:
    1. position (always, top priority)
    2. orientation, solved only in the null space of the position task
       (best effort: whatever is left of the 2 spare DOF)
    3. if orientation is disabled, a posture pull toward q_rest instead
Every iterate is clamped to the URDF joint limits. The caller decides
whether a result is good enough via IKResult.pos_err.

Quick test (no ROS needed beyond numpy):
    python3 -m so101_mirror.kinematics path/to/so101.urdf
"""
import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass

import numpy as np

ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")


# ---------------------------------------------------------------- rotations

def rot_x(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def rot_y(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rpy_to_matrix(roll, pitch, yaw):
    """URDF/ROS convention: R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    return rot_z(yaw) @ rot_y(pitch) @ rot_x(roll)


def matrix_to_rpy(R):
    pitch = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    if abs(math.cos(pitch)) < 1e-9:  # gimbal lock: put everything in yaw
        return 0.0, pitch, math.atan2(-R[0, 1], R[1, 1])
    return math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0])


def axis_angle(axis, a):
    """Rotation matrix for angle a about unit vector axis (Rodrigues)."""
    x, y, z = axis
    c, s, C = math.cos(a), math.sin(a), 1 - math.cos(a)
    return np.array([[c + x * x * C, x * y * C - z * s, x * z * C + y * s],
                     [y * x * C + z * s, c + y * y * C, y * z * C - x * s],
                     [z * x * C - y * s, z * y * C + x * s, c + z * z * C]])


def rotvec_error(R_target, R):
    """Rotation vector (base frame) that takes R to R_target."""
    E = R_target @ R.T
    cos_a = max(-1.0, min(1.0, (np.trace(E) - 1) / 2))
    a = math.acos(cos_a)
    v = np.array([E[2, 1] - E[1, 2], E[0, 2] - E[2, 0], E[1, 0] - E[0, 1]])
    if a < 1e-6:
        return 0.5 * v
    if math.pi - a < 1e-4:  # near 180 deg: axis from the symmetric part
        k = int(np.argmax(np.diag(E)))
        axis = (E[:, k] + np.eye(3)[:, k]) / math.sqrt(2 * (1 + E[k, k]))
        return a * axis
    return a * v / (2 * math.sin(a))


def transform(xyz, rpy):
    T = np.eye(4)
    T[:3, :3] = rpy_to_matrix(*rpy)
    T[:3, 3] = xyz
    return T


# ---------------------------------------------------------------- chain

@dataclass
class Joint:
    name: str
    origin: np.ndarray        # 4x4 parent->joint frame at q = 0
    axis: np.ndarray          # unit axis in the joint frame (zeros for fixed)
    lower: float = -math.pi
    upper: float = math.pi


class Chain:
    """Serial chain from `base` link to `tip` link of a URDF."""

    def __init__(self, joints, active):
        self.joints = joints                    # ordered base -> tip, incl. fixed
        self.active = active                    # names of movable joints, in order
        lims = {j.name: (j.lower, j.upper) for j in joints}
        self.lower = np.array([lims[n][0] for n in active])
        self.upper = np.array([lims[n][1] for n in active])

    @classmethod
    def from_urdf(cls, urdf_xml, base="base_link", tip="gripper_frame_link"):
        root = ET.fromstring(urdf_xml)
        by_child = {}
        for j in root.findall("joint"):
            by_child[j.find("child").get("link")] = j
        chain, link = [], tip
        while link != base:
            if link not in by_child:
                raise ValueError(f"no path from {base} to {tip} (stuck at {link})")
            j = by_child[link]
            chain.append(j)
            link = j.find("parent").get("link")
        chain.reverse()

        joints, active = [], []
        for j in chain:
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz", "0 0 0") if o is not None else "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy", "0 0 0") if o is not None else "0 0 0").split()]
            jtype = j.get("type")
            if jtype in ("revolute", "continuous"):
                ax = j.find("axis")
                axis = np.array([float(v) for v in (ax.get("xyz") if ax is not None else "1 0 0").split()])
                axis /= np.linalg.norm(axis)
                lim = j.find("limit")
                lo, hi = -math.pi, math.pi
                if jtype == "revolute" and lim is not None:
                    lo, hi = float(lim.get("lower")), float(lim.get("upper"))
                joints.append(Joint(j.get("name"), transform(xyz, rpy), axis, lo, hi))
                active.append(j.get("name"))
            elif jtype == "fixed":
                joints.append(Joint(j.get("name"), transform(xyz, rpy), np.zeros(3)))
            else:
                raise ValueError(f"unsupported joint type {jtype} ({j.get('name')})")
        return cls(joints, active)

    def fk(self, q, return_frames=False):
        """Tip pose (4x4) for active joint vector q. Optionally the world
        origin/axis of every active joint (for the Jacobian)."""
        T = np.eye(4)
        frames = []
        i = 0
        for j in self.joints:
            T = T @ j.origin
            if j.axis.any():
                frames.append((T[:3, 3].copy(), T[:3, :3] @ j.axis))
                R = np.eye(4)
                R[:3, :3] = axis_angle(j.axis, q[i])
                T = T @ R
                i += 1
        return (T, frames) if return_frames else T

    def jacobian(self, q):
        T, frames = self.fk(q, return_frames=True)
        p = T[:3, 3]
        J = np.zeros((6, len(self.active)))
        for k, (o, z) in enumerate(frames):
            J[:3, k] = np.cross(z, p - o)
            J[3:, k] = z
        return T, J


# ---------------------------------------------------------------- IK

@dataclass
class IKResult:
    q: np.ndarray
    pos_err: float      # meters
    rot_err: float      # radians
    iters: int


def _dls(J, damping):
    """Damped least-squares pseudo-inverse."""
    m = J.shape[0]
    return J.T @ np.linalg.inv(J @ J.T + damping ** 2 * np.eye(m))


def solve_ik(chain, p_target, R_target=None, q0=None, q_rest=None,
             max_iters=150, pos_tol=1e-4, damping=0.005, rot_weight=0.5,
             posture_gain=0.1, max_step=0.2, lock=None):
    """Task-priority IK. R_target=None means position only.
    lock: {joint index: value} joints held fixed (the others solve around them).

    Returns IKResult; it is the caller's job to reject large pos_err."""
    n = len(chain.active)
    q = np.zeros(n) if q0 is None else np.array(q0, dtype=float)
    free = np.ones(n, dtype=bool)
    for i, v in (lock or {}).items():
        q[i], free[i] = v, False
    q = np.clip(q, chain.lower, chain.upper)
    q_rest = np.zeros(n) if q_rest is None else np.asarray(q_rest, dtype=float)
    p_target = np.asarray(p_target, dtype=float)
    I = np.eye(n)
    it = 0
    for it in range(1, max_iters + 1):
        T, J = chain.jacobian(q)
        J[:, ~free] = 0.0  # locked joints contribute nothing
        Jp, Jw = J[:3], J[3:]
        ep = p_target - T[:3, 3]
        Jp_pinv = _dls(Jp, damping)
        dq = Jp_pinv @ ep
        N = I - Jp_pinv @ Jp
        if R_target is not None:
            ew = rot_weight * rotvec_error(R_target, T[:3, :3])
            JwN = Jw @ N
            dq += N @ (_dls(JwN, damping) @ (ew - Jw @ dq))
        else:
            dq += N @ (posture_gain * (q_rest - q))
        dq[~free] = 0.0
        step = np.max(np.abs(dq))
        if step > max_step:
            dq *= max_step / step
        q = np.clip(q + dq, chain.lower, chain.upper)
        if np.linalg.norm(ep) < pos_tol and step < 1e-4:
            break
    T = chain.fk(q)
    pos_err = float(np.linalg.norm(p_target - T[:3, 3]))
    rot_err = 0.0 if R_target is None else float(np.linalg.norm(rotvec_error(R_target, T[:3, :3])))
    return IKResult(q, pos_err, rot_err, it)


def solve_ik_restarts(chain, p_target, R_target=None, q0=None, restarts=8,
                      accept_tol=2e-3, seed=0, **kw):
    """solve_ik from q0, then from random seeds while pos_err > accept_tol.
    Returns the best result (lowest pos_err, then rot_err)."""
    rng = np.random.default_rng(seed)
    best = solve_ik(chain, p_target, R_target, q0=q0, **kw)
    for _ in range(restarts):
        if best.pos_err <= accept_tol:
            break
        r = solve_ik(chain, p_target, R_target,
                     q0=rng.uniform(chain.lower, chain.upper), **kw)
        if (r.pos_err, r.rot_err) < (best.pos_err, best.rot_err):
            best = r
    return best


PITCH_STEPS = [math.radians(d) for d in (10, -10, 20, -20, 30, -30, 45, -45, 60, -60)]


def solve_relaxed_pitch(chain, p_target, roll, pitch, yaw, home_R, q0, tol, **kw):
    """Target pose unreachable: try the nearest reachable pitch instead, solving
    from q0 (the current pose) so the result stays on the current branch.
    Returns (IKResult, pitch used) or (None, None)."""
    for dp in PITCH_STEPS:
        R = rpy_to_matrix(roll, pitch + dp, yaw) @ home_R
        res = solve_ik(chain, p_target, R, q0=q0, max_iters=50, **kw)
        if res.pos_err <= tol:
            return res, pitch + dp
    return None, None


class BranchGuard:
    """Hold off sudden joint-space jumps (IK branch flips, glitchy inputs).

    A solution within jump (rad, every joint) of the last accepted one passes.
    A bigger jump passes only after `confirm` consecutive solutions agree with
    each other (within agree rad): a real move persists, a flip back and forth
    or a one-frame glitch doesn't. Call reset() after a hold/re-anchor if needed.
    """

    def __init__(self, jump=math.radians(30), confirm=3, agree=math.radians(10)):
        self.jump, self.confirm, self.agree = jump, confirm, agree
        self.pending, self.n = None, 0

    def reset(self):
        self.pending, self.n = None, 0

    def __call__(self, q_last, q_new):
        """True if q_new may be published (q_last = last published, or None)."""
        q_new = np.asarray(q_new, dtype=float)
        if q_last is None or np.max(np.abs(q_new - q_last)) <= self.jump:
            self.reset()
            return True
        if self.pending is not None and np.max(np.abs(q_new - self.pending)) <= self.agree:
            self.n += 1
        else:
            self.n = 1
        self.pending = q_new
        if self.n >= self.confirm:
            self.reset()
            return True
        return False


# ---------------------------------------------------------------- self-test

def _selftest(urdf_path):
    with open(urdf_path) as f:
        chain = Chain.from_urdf(f.read())
    print("active joints:", chain.active)
    T0 = chain.fk(np.zeros(len(chain.active)))
    print("tip at q=0:", np.round(T0[:3, 3], 4), "rpy", np.round(matrix_to_rpy(T0[:3, :3]), 3))
    rng = np.random.default_rng(0)
    for label, use_rot in (("pose", True), ("position-only", False)):
        worst, fails, rot = 0.0, 0, []
        for _ in range(200):
            q_true = rng.uniform(chain.lower * 0.8, chain.upper * 0.8)
            T = chain.fk(q_true)
            r = solve_ik_restarts(chain, T[:3, 3], T[:3, :3] if use_rot else None,
                                  q0=np.zeros(len(q_true)))
            worst = max(worst, r.pos_err)
            fails += r.pos_err > 2e-3
            rot.append(r.rot_err)
        print(f"{label}: 200 random reachable targets, worst pos err {worst * 1000:.2f} mm, "
              f"{fails} over 2 mm, median rot err {np.degrees(np.median(rot)):.2f} deg")


if __name__ == "__main__":
    import sys
    _selftest(sys.argv[1])
