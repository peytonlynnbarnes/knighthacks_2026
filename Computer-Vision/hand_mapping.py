#!/home/peyton/kh/.venv/bin/python
"""Hand pose (cam frame) -> /ball_position offset [x y z roll pitch yaw]. Pure, no ROS.

out_p = anchor_out_p + scale * A (p - anchor_hand_p)          (meters, base_link axes)
out_R = (A R_hand anchor_hand_R^T A^T)^rot_scale  @ anchor_out_R
then clamped to the offset box and slew-limited (speed caps) so a jump in
the vision input (re-acquired hand, bad frame) can't make the target jump.

Initially anchor_hand = the calibrated neutral hand and anchor_out =
neutral_offset (the arm target for the neutral hand). The clutch (fist)
re-anchors on release: the current hand maps to the last output, so the
arm doesn't move when you reposition your hand.

Self-test: .venv/bin/python Computer-Vision/hand_mapping.py
"""
import math

import numpy as np

from hand3d import matrix_to_rotvec, matrix_to_rpy, rotvec_to_matrix

# conservative offset box from INTEGRATION.md (home is near full forward reach)
BOX_MIN = np.array([-0.20, -0.15, -0.15])
BOX_MAX = np.array([0.00, 0.15, 0.05])


class OneEuro:
    """One-euro filter (Casiez et al. 2012) on a vector. min_cutoff Hz, beta s/m."""

    def __init__(self, min_cutoff=1.0, beta=1.0, d_cutoff=1.0):
        self.min_cutoff, self.beta, self.d_cutoff = min_cutoff, beta, d_cutoff
        self.x = self.dx = self.t = None

    @staticmethod
    def _alpha(cutoff, dt):
        tau = 1.0 / (2 * math.pi * cutoff)
        return 1.0 / (1.0 + tau / dt)

    def reset(self):
        self.x = self.dx = self.t = None

    def __call__(self, x, t):
        x = np.asarray(x, dtype=float)
        if self.x is None or t <= self.t:
            self.x, self.dx, self.t = x, np.zeros_like(x), t
            return x
        dt = t - self.t
        dx = (x - self.x) / dt
        self.dx = self.dx + self._alpha(self.d_cutoff, dt) * (dx - self.dx)
        cutoff = self.min_cutoff + self.beta * np.linalg.norm(self.dx)
        self.x = self.x + self._alpha(cutoff, dt) * (x - self.x)
        self.t = t
        return self.x


class HandMapper:
    MAX_DT = 0.1  # s

    def __init__(self, A, neutral_p, neutral_R, scale=1.1, rot_scale=1.0,
                 neutral_offset=(-0.10, 0.0, -0.05), box_min=BOX_MIN, box_max=BOX_MAX,
                 max_lin_speed=0.25, max_ang_speed=math.radians(120),
                 pos_filter=(1.0, 2.0), rot_filter=(1.0, 0.5)):
        self.A = np.asarray(A, dtype=float)
        self.scale, self.rot_scale = scale, rot_scale
        self.box_min, self.box_max = np.asarray(box_min), np.asarray(box_max)
        self.max_lin, self.max_ang = max_lin_speed, max_ang_speed
        self.hand_p, self.hand_R = np.asarray(neutral_p, float), np.asarray(neutral_R, float)
        self.out_p = np.clip(np.asarray(neutral_offset, float), self.box_min, self.box_max)
        self.out_R = np.eye(3)
        self.f_pos = OneEuro(*pos_filter)
        self.f_rot = OneEuro(*rot_filter)
        # slew-limited output state; starts at home (= where feetech_mirror --live homes to)
        self.cur_p = np.zeros(3)
        self.cur_R = np.eye(3)
        self.cur_t = None
        self.last_raw = None  # (p, R) of the latest unclutched target, for re-anchoring

    def raw_target(self, p, R_hand):
        """Unfiltered (offset p (3,), rotation (3,3)) for a hand pose in cam frame."""
        p_out = self.out_p + self.scale * (self.A @ (np.asarray(p) - self.hand_p))
        dR = self.A @ (R_hand @ self.hand_R.T) @ self.A.T
        if self.rot_scale != 1.0:
            dR = rotvec_to_matrix(self.rot_scale * matrix_to_rotvec(dR))
        return np.clip(p_out, self.box_min, self.box_max), dR @ self.out_R

    def reanchor(self, p, R_hand):
        """Clutch release: the current hand now maps to the current output."""
        self.hand_p, self.hand_R = np.asarray(p, float), np.asarray(R_hand, float)
        self.out_p, self.out_R = self.cur_p.copy(), self.cur_R.copy()
        self.f_pos.reset()
        self.f_rot.reset()

    def update(self, p, R_hand, t):
        """Hand pose at time t (s) -> [x y z roll pitch yaw] offset for /ball_position."""
        tp, tR = self.raw_target(p, R_hand)
        tp = self.f_pos(tp, t)
        # filter rotation as a rotvec (offset from home orientation; continuous below 180 deg)
        tR = rotvec_to_matrix(self.f_rot(matrix_to_rotvec(tR), t))
        # dt capped: after the hand was lost for a while, the step is still small
        dt = 0.0 if self.cur_t is None else min(max(0.0, t - self.cur_t), self.MAX_DT)
        self.cur_t = t
        # slew limit position
        dp = tp - self.cur_p
        n = np.linalg.norm(dp)
        if n > self.max_lin * dt:
            dp *= self.max_lin * dt / n
        self.cur_p = self.cur_p + dp
        # slew limit rotation
        drv = matrix_to_rotvec(tR @ self.cur_R.T)
        a = np.linalg.norm(drv)
        if a > self.max_ang * dt:
            drv *= self.max_ang * dt / a
        self.cur_R = rotvec_to_matrix(drv) @ self.cur_R
        return np.concatenate([self.cur_p, matrix_to_rpy(self.cur_R)])

    def hold(self, t):
        """Output while clutched: the current target, unchanged."""
        self.cur_t = t
        return np.concatenate([self.cur_p, matrix_to_rpy(self.cur_R)])


def _selftest():
    from hand3d import robot_axes, rpy_to_matrix
    A = robot_axes([0, 0, 1])  # moving the hand away from the cameras = robot +X
    p0 = np.array([0.1, 0.0, 0.6])
    m = HandMapper(A, p0, np.eye(3), scale=0.5, max_lin_speed=10, max_ang_speed=100)
    t = 0.0
    for _ in range(200):  # neutral hand, settle
        out = m.update(p0, np.eye(3), t)
        t += 1 / 30
    assert np.allclose(out, [-0.10, 0, -0.05, 0, 0, 0], atol=1e-6), out
    # hand 10 cm toward cameras (cam -z) = robot -X: arm 5 cm back
    for _ in range(200):
        out = m.update(p0 + [0, 0, -0.10], np.eye(3), t)
        t += 1 / 30
    assert np.allclose(out[:3], [-0.15, 0, -0.05], atol=1e-4), out
    # hand up 10 cm (cam -y) -> +z 5 cm, clamped at box top 0.05? -0.05+0.05 = 0
    for _ in range(200):
        out = m.update(p0 + [0, -0.10, 0], np.eye(3), t)
        t += 1 / 30
    assert np.allclose(out[:3], [-0.10, 0, 0.0], atol=1e-4), out
    # wrist rotation: hand rotated about cam -y (robot +Z) by 30 deg -> yaw 30
    Rc = A.T @ rpy_to_matrix(0, 0, math.radians(30)) @ A
    for _ in range(300):
        out = m.update(p0, Rc, t)
        t += 1 / 30
    assert np.allclose(out[3:], [0, 0, math.radians(30)], atol=1e-3), out
    # clutch: hold, move hand far, release -> no jump
    held = m.hold(t)
    m.reanchor(p0 + [0.2, 0.2, 0.2], np.eye(3))
    out = m.update(p0 + [0.2, 0.2, 0.2], np.eye(3), t + 0.03)
    assert np.allclose(out, held, atol=1e-6), (out, held)
    # slew limit: default 0.25 m/s, a 10 cm jump takes > 0.3 s
    m2 = HandMapper(A, p0, np.eye(3))
    out = m2.update(p0, np.eye(3), 0.0)
    assert np.allclose(out, 0)  # starts at home, ramps to the neutral offset
    out = m2.update(p0, np.eye(3), 0.1)
    assert np.linalg.norm(out[:3]) <= 0.025 + 1e-9
    # hand lost for 5 s, reappears far away: still only MAX_DT worth of motion
    before = m2.update(p0, np.eye(3), 0.2)
    out = m2.update(p0 + [0.3, 0.3, 0.3], np.eye(3), 5.2)
    assert np.linalg.norm(out[:3] - before[:3]) <= 0.25 * HandMapper.MAX_DT + 1e-9
    print("hand_mapping self-test OK")


if __name__ == "__main__":
    _selftest()
