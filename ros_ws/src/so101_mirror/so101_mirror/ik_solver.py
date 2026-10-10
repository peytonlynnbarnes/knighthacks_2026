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

There's no separate position-only fallback any more: with limit handling, the
orientation task already yields to position, so an unreachable pitch comes out
as the nearest reachable one instead of the gripper swinging free.
"""
import math
import time

import numpy as np

from so101_mirror.kinematics import rpy_to_matrix, solve_ik

# IK at home (mid-workspace, gripper level), deg: see model.HOME_OFFSET
HOME_Q_DEG = (0.0, -51.0, 63.8, -12.8, 0.0)


class TargetSolver:
    def __init__(self, chain, home_p, home_R, use_orientation=True, free_yaw=True,
                 rot_weight=0.5, max_pos_err=0.01, jump_deg=30.0, confirm=3,
                 budget_s=0.02, seed=0, rot_damping=0.1):
        self.chain, self.home_p, self.home_R = chain, np.asarray(home_p), np.asarray(home_R)
        _, frames = chain.fk(np.zeros(len(chain.active)), return_frames=True)
        self.pan_xy = frames[0][0][:2]  # pan axis location in base_link
        self.use_orientation, self.free_yaw = use_orientation, free_yaw
        self.rot_damping = rot_damping
        self.rot_weight, self.tol = rot_weight, max_pos_err
        self.jump, self.confirm, self.budget = math.radians(jump_deg), confirm, budget_s
        self.q_home = np.clip(np.radians(HOME_Q_DEG), chain.lower, chain.upper)
        self.q = self.q_home.copy()      # last published solution
        self.pending = []                 # held-back far solutions (branch guard)
        self.rng = np.random.default_rng(seed)
        self.info = {}

    def target_pose(self, offset):
        """Absolute (p, R or None) for an offset from home, yaw freed if set."""
        p = self.home_p + np.asarray(offset[:3], dtype=float)
        if not self.use_orientation:
            return p, None
        roll, pitch, yaw = offset[3:6]
        if self.free_yaw:
            yaw = math.atan2(p[1] - self.pan_xy[1], p[0] - self.pan_xy[0])
        return p, rpy_to_matrix(roll, pitch, yaw) @ self.home_R

    def _ik(self, p, R, q0, iters):
        posture = dict(q_rest=self.q_home) if R is None else {}
        return solve_ik(self.chain, p, R, q0=q0, max_iters=iters,
                        rot_weight=self.rot_weight, rot_damping=self.rot_damping, **posture)

    def _score(self, r):
        return r.rot_err + 0.5 * float(np.max(np.abs(r.q - self.q)))

    def solve(self, offset):
        """Joint angles (5,) to publish, or None to hold. self.info explains."""
        t0 = time.perf_counter()
        p, R = self.target_pose(offset)
        best = self._ik(p, R, self.q, 60)
        how = "warm"
        if best.pos_err > self.tol:
            wrist_relaxed = self.q.copy()
            wrist_relaxed[3] = 0.0
            seeds = [self.q_home, wrist_relaxed]
            cands = [r for r in (self._ik(p, R, s, 100) for s in seeds) if r.pos_err <= self.tol]
            while not cands and time.perf_counter() - t0 < self.budget:
                r = self._ik(p, R, self.rng.uniform(self.chain.lower, self.chain.upper), 100)
                if r.pos_err <= self.tol:
                    cands.append(r)
            if cands:
                best, how = min(cands, key=self._score), "reseeded"
        self.info = dict(how=how, pos_err=best.pos_err, rot_err=best.rot_err,
                         ms=(time.perf_counter() - t0) * 1000)
        if best.pos_err > self.tol:
            self.info["how"] = "unreachable"
            self.pending = []
            return None
        if float(np.max(np.abs(best.q - self.q))) > self.jump:
            # far jump: accept only once `confirm` consecutive ticks agree
            if self.pending and float(np.max(np.abs(best.q - self.pending[-1]))) > self.jump:
                self.pending = []
            self.pending.append(best.q)
            if len(self.pending) < self.confirm:
                self.info["how"] = f"jump held ({len(self.pending)}/{self.confirm})"
                return None
        self.pending = []
        self.q = best.q
        return best.q
