"""Target offset [x y z roll pitch yaw] -> arm joint angles, with continuity. Pure numpy.

Used by scripts/ik_node.py; testable without ROS. Strategy per target:

1. Warm start from the last published solution (position exact, orientation
   best effort in the null space; joints at limits are handled in solve_ik).
   Almost every tick ends here, and the result is continuous with the last pose.
2. Only if that misses the position: a few deterministic seeds (home pose,
   last pose with the wrist relaxed), then random seeds, within a time budget.
   Candidates are ranked by orientation error plus distance from the last pose,
   so a far-away branch only wins when nothing near works.
3. Branch guard: a solution that moves any joint more than jump_deg from the
   last published one (a branch flip, or a glitch in the target) is held back
   until confirm consecutive ticks agree on it.

Claw down (claw_down = (z_full, z_top, deg_top)): the target pitch comes from the tip
height, not the hand's tilt: straight down (90 deg) at z <= z_full, linearly less down
above it (deg_top at z_top, same slope beyond, never above level). Pick and place: the
claw hangs over the object near the table. The default line (4 cm -> 90, 20 cm -> 50)
follows what the arm can reach on the centre of the workspace (soft limits, measured
2026-10-10: 90 / 79 / 69 / 59 / 48 deg at 2 / 8 / 12 / 16 / 20 cm); where it can't (far
out), the claw tilts as little as it can (orientation is the secondary task; position
stays exact). floor_z: the target tip height never goes below this (m, base_link; the
table is at z ~ 0), so normal use hovers just above the table without touching it.

Two-hand mode: solve(offset, roll_lock=rad) holds wrist_roll at the other hand's
command and solves position + pitch around it; the orientation target takes its
roll from the lock, and the branch guard ignores wrist_roll (a commanded change).

There's no separate position-only fallback any more: with limit handling, the
orientation task already yields to position, so an unreachable pitch comes out
as the nearest reachable one instead of the gripper swinging free.
"""
import math
import time

import numpy as np

from so101_mirror.kinematics import matrix_to_rpy, rpy_to_matrix, solve_ik

# IK at home (mid-workspace, gripper level), deg: see model.HOME_OFFSET
HOME_Q_DEG = (0.0, -51.0, 63.8, -12.8, 0.0)


class TargetSolver:
    def __init__(self, chain, home_p, home_R, use_orientation=True, free_yaw=True,
                 rot_weight=0.5, max_pos_err=0.01, jump_deg=30.0, confirm=3,
                 budget_s=0.02, seed=0, rot_damping=0.1, claw_down=None, floor_z=None):
        self.chain, self.home_p, self.home_R = chain, np.asarray(home_p), np.asarray(home_R)
        _, frames = chain.fk(np.zeros(len(chain.active)), return_frames=True)
        self.pan_xy = frames[0][0][:2]  # pan axis location in base_link
        self.use_orientation, self.free_yaw = use_orientation, free_yaw
        self.rot_damping = rot_damping
        self.claw_down = claw_down  # (z_full m, z_top m, deg_top) or None: pitch from the target
        self.floor_z = floor_z      # m, lowest target tip height, or None
        self.rot_weight, self.tol = rot_weight, max_pos_err
        self.jump, self.confirm, self.budget = math.radians(jump_deg), confirm, budget_s
        self.q_home = np.clip(np.radians(HOME_Q_DEG), chain.lower, chain.upper)
        self.q = self.q_home.copy()      # last published solution
        self.pending = []                 # held-back far solutions (branch guard)
        self.rng = np.random.default_rng(seed)
        self.info = {}
        # wrist_roll -> tool roll in target rpy: same or opposite sign, from the model
        self.roll_idx = list(chain.active).index("wrist_roll")
        q1 = self.q_home.copy()
        q1[self.roll_idx] += 0.1
        R0, R1 = chain.fk(self.q_home)[:3, :3], chain.fk(q1)[:3, :3]
        self.roll_sign = float(np.sign(matrix_to_rpy(R1 @ R0.T)[0]))

    def target_pose(self, offset, roll_lock=None):
        """Absolute (p, R or None) for an offset from home, yaw freed if set; with
        roll_lock (wrist_roll, rad) the roll is the one that joint angle gives."""
        p = self.home_p + np.asarray(offset[:3], dtype=float)
        if self.floor_z is not None and p[2] < self.floor_z:
            p[2] = self.floor_z
        if not self.use_orientation:
            return p, None
        roll, pitch, yaw = offset[3:6]
        if self.claw_down is not None:
            pitch = self.claw_pitch(p[2])
        if roll_lock is not None:
            roll = self.roll_sign * roll_lock
        if self.free_yaw:
            yaw = math.atan2(p[1] - self.pan_xy[1], p[0] - self.pan_xy[0])
        return p, rpy_to_matrix(roll, pitch, yaw) @ self.home_R

    def claw_pitch(self, z):
        """Claw-down target pitch (rad, + = down) at tip height z (m)."""
        z_full, z_top, deg_top = self.claw_down
        deg = 90.0 + (deg_top - 90.0) * (z - z_full) / (z_top - z_full)
        return math.radians(min(90.0, max(0.0, deg)))

    def _ik(self, p, R, q0, iters, lock=None):
        posture = dict(q_rest=self.q_home) if R is None else {}
        return solve_ik(self.chain, p, R, q0=q0, max_iters=iters, lock=lock,
                        rot_weight=self.rot_weight, rot_damping=self.rot_damping, **posture)

    def _score(self, r):
        return r.rot_err + 0.5 * float(np.max(np.abs(r.q - self.q)))

    def solve(self, offset, roll_lock=None):
        """Joint angles (5,) to publish, or None to hold. self.info explains.
        roll_lock: hold wrist_roll at this angle (rad, clipped to its limits)."""
        t0 = time.perf_counter()
        lock = None
        if roll_lock is not None:
            i = self.roll_idx
            roll_lock = float(np.clip(roll_lock, self.chain.lower[i], self.chain.upper[i]))
            lock = {i: roll_lock}
        p, R = self.target_pose(offset, roll_lock)
        best = self._ik(p, R, self.q, 60, lock)
        how = "warm"
        if best.pos_err > self.tol:
            wrist_relaxed = self.q.copy()
            wrist_relaxed[3] = 0.0
            seeds = [self.q_home, wrist_relaxed]
            cands = [r for r in (self._ik(p, R, s, 100, lock) for s in seeds) if r.pos_err <= self.tol]
            while not cands and time.perf_counter() - t0 < self.budget:
                r = self._ik(p, R, self.rng.uniform(self.chain.lower, self.chain.upper), 100, lock)
                if r.pos_err <= self.tol:
                    cands.append(r)
            if cands:
                best, how = min(cands, key=self._score), "reseeded"
        self.info = dict(how=how, pos_err=best.pos_err, rot_err=best.rot_err,
                         ms=(time.perf_counter() - t0) * 1000, roll_lock=roll_lock)
        if best.pos_err > self.tol:
            self.info["how"] = "unreachable"
            self.pending = []
            return None
        guarded = np.ones(len(best.q), dtype=bool)
        if lock:
            guarded[self.roll_idx] = False  # a commanded roll change isn't a branch flip
        if float(np.max(np.abs(best.q - self.q)[guarded])) > self.jump:
            # far jump: accept only once `confirm` consecutive ticks agree
            if self.pending and float(np.max(np.abs(best.q - self.pending[-1])[guarded])) > self.jump:
                self.pending = []
            self.pending.append(best.q)
            if len(self.pending) < self.confirm:
                self.info["how"] = f"jump held ({len(self.pending)}/{self.confirm})"
                return None
        self.pending = []
        self.q = best.q
        return best.q


def _selftest():
    """Roll lock on the real chain + soft limits: wrist_roll stays at the command, position
    stays solved, pitch still follows; no lock = unchanged; locked roll changes pass the guard."""
    from so101_mirror.model import (HW_LIMITS_FILE, apply_limits, hardware_limits, home_pose,
                                    load_chain)
    chain = load_chain()
    apply_limits(chain, hardware_limits(HW_LIMITS_FILE))
    home_p, home_R = home_pose(chain)
    rng = np.random.default_rng(1)
    i = list(chain.active).index("wrist_roll")
    worst = dict(roll=0.0, pos=0.0, pos_free=0.0)
    n_ok = n_free_ok = 0
    for _ in range(60):
        off = np.r_[rng.uniform([-0.08, -0.12, -0.06], [0.08, 0.12, 0.08]), 0.0,
                    rng.uniform(-0.6, 0.9), 0.0]
        lock = rng.uniform(-1.5, 0.8)
        a, b = (TargetSolver(chain, home_p, home_R, jump_deg=360.0) for _ in range(2))  # guard: below
        qa, qb = a.solve(off, roll_lock=lock), b.solve(off)
        if qa is not None:
            n_ok += 1
            worst["roll"] = max(worst["roll"], abs(qa[i] - lock))
            worst["pos"] = max(worst["pos"], a.info["pos_err"])
        if qb is not None:
            n_free_ok += 1
            worst["pos_free"] = max(worst["pos_free"], b.info["pos_err"])
    assert worst["roll"] < 1e-9, worst
    assert worst["pos"] <= 0.01 and worst["pos_free"] <= 0.01, worst
    assert n_ok >= 0.9 * n_free_ok, (n_ok, n_free_ok)
    # the orientation target uses the locked roll: at home, a locked roll gives no rot error
    s = TargetSolver(chain, home_p, home_R)
    q = s.solve(np.zeros(6), roll_lock=0.5)
    assert q is not None and s.info["rot_err"] < 0.02, s.info
    # sweeping the locked roll 0 -> -1.4 rad in one step isn't held back as a branch flip
    s = TargetSolver(chain, home_p, home_R)
    s.solve(np.zeros(6), roll_lock=0.0)
    q = s.solve(np.zeros(6), roll_lock=-1.4)
    assert q is not None and abs(q[i] + 1.4) < 1e-9, s.info
    # claw down: pitch from the height (the hand's pitch is ignored), floor, roll lock
    claw = (0.04, 0.20, 50.0)
    s = TargetSolver(chain, home_p, home_R, claw_down=claw, floor_z=0.02, jump_deg=360.0)
    assert abs(math.degrees(s.claw_pitch(0.04)) - 90) < 1e-9 and abs(math.degrees(s.claw_pitch(0.20)) - 50) < 1e-9
    assert math.degrees(s.claw_pitch(0.0)) == 90.0 and math.degrees(s.claw_pitch(0.60)) == 0.0
    low = np.r_[np.array([0.22, 0.0, 0.04]) - home_p, 0.0, -0.5, 0.0]
    q = s.solve(low)
    assert q is not None and s.info["rot_err"] < math.radians(3), s.info  # straight down
    tilts = []
    for z in (0.08, 0.12, 0.16):  # centre of the workspace: on the schedule
        q = s.solve(np.r_[np.array([0.27, 0.0, z]) - home_p, 0.0, 0.0, 0.0])
        assert q is not None and s.info["pos_err"] < 0.005, (z, s.info)
        tilts.append(math.degrees(s.info["rot_err"]))
    assert max(tilts) < 5, tilts
    q = s.solve(np.r_[np.array([0.22, 0.0, -0.03]) - home_p, 0.0, 0.0, 0.0])  # below the floor
    assert q is not None and abs(chain.fk(q)[2, 3] - 0.02) < 0.005, chain.fk(q)[2, 3]
    s2 = TargetSolver(chain, home_p, home_R, claw_down=claw, jump_deg=360.0)
    q = s2.solve(low, roll_lock=-0.7)  # with the left hand's roll: still down, roll exact
    assert q is not None and abs(q[i] + 0.7) < 1e-9 and s2.info["rot_err"] < math.radians(3), s2.info
    print(f"ik_solver self-test OK: roll lock exact, {n_ok}/{n_free_ok} targets solved (locked/free), "
          f"pos err max {worst['pos'] * 1000:.2f} / {worst['pos_free'] * 1000:.2f} mm, "
          f"roll sign {s.roll_sign:+.0f}")


if __name__ == "__main__":
    _selftest()
