#!/usr/bin/env python3
"""Hand pose (cam frame) -> /ball_position offset [x y z roll pitch yaw]. Pure, no ROS.

out_p = anchor_out_p + scale * A (p - anchor_hand_p)          (meters, base_link axes)
out_R = shape(filter(A R_hand neutral_R^T A^T))
then clamped to the offset box and slew-limited (speed caps) so a jump in
the vision input (re-acquired hand, bad frame) can't make the target jump.

Orientation (hand rotation from the calibrated neutral, robot axes):
  1. Frames where the palm is turned nearly edge-on to the cameras (where
     the landmarks get noisy, then the hand is lost) count less, and the rotation
     holds once the palm is past FACING_DEG[0] from facing the cameras.
  2. One-euro filter on the hand rotation (before the gain, so the filter
     sees real hand speed): strong smoothing when still, little when turning.
  3. RotationShaper: per-axis gain (roll, pitch up / down, yaw) so a hand
     turn that stays trackable covers the gripper's range, a soft deadband
     (a still hand stays level), and the arm's reachable limits.

Initially anchor_hand = the calibrated neutral hand and anchor_out =
neutral_offset (the arm target for the neutral hand). The clutch re-anchors
POSITION only on release: the current hand position maps to the last output,
so the arm doesn't move when you reposition your hand. Orientation is never
re-anchored: it stays relative to the calibrated neutral hand (flat hand =
level gripper). Re-anchoring it baked the filter lag and the noise of the
first frame back into a permanent offset on every clutch, so rpy drifted
(simulated: ~10-100 deg after 20 clutches); a turned hand now just rotates
the gripper back at the slew limit.

Self-test: python Computer-Vision/common/hand_mapping.py
"""
import math

import numpy as np

from hand3d import matrix_to_rotvec, matrix_to_rpy, rotvec_to_matrix, rpy_to_matrix

# offset box around home = mid-workspace (home z = 17.6 cm above base_link).
# z floor -0.13 = 4.6 cm above base_link: the -0.10 floor left the gripper ~40 mm
# above the table (2026-10-10); this leaves ~10 mm, since the real arm sits 1.5-2 cm
# below its command (elbow sag). hand_target_node --z-min overrides it.
BOX_MIN = np.array([-0.10, -0.15, -0.13])
BOX_MAX = np.array([0.10, 0.15, 0.10])

# gripper rotation limits (deg, offsets from level at home; +pitch = tip down).
# Roll: one direction only, 0 (level) .. +90 = a quarter turn clockwise seen
# from behind the gripper. The real wrist_roll soft limits are -157..+50 deg
# (control/so101_bus.py, as the IK reads them) and +joint = -gripper roll, so
# gripper roll -50..+157: + is the side with room for 90. Pitch -85 (up) ..
# +85 (down): at +-90 roll and yaw become the same axis and the rpy split of
# the hand rotation turns unstable. Up reaches ~-86 at home with wrist_flex at
# its -119 deg servo limit (URDF widened to match); where it can't, the IK
# relaxes the pitch. Yaw is replaced by the IK (free_yaw).
ROT_LIMITS_DEG = ((0.0, 90.0), (-85.0, 85.0), (-180.0, 180.0))
# palm angle from facing the cameras (deg): full trust below FACING_DEG[1],
# none (rotation held) above FACING_DEG[0]
FACING_DEG = (75.0, 55.0)


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


class StereoKalman:
    """Constant-velocity Kalman filter on the hand position, in the cam0 frame.

    State [p, v]. Measurement noise follows stereo geometry: sideways
    sigma = z / f * sigma_px, along the viewing ray z^2 / (f B) * ray_px
    (~4x worse at 0.8 m with the 20 cm baseline; measured on a still hand
    2026-10-10: 0.32 mm sideways, 1.19 mm along the ray, both ~0.35 px).
    So depth is smoothed harder than sideways motion. The px defaults are much
    higher than that still-hand figure: landmarks are noisier on a moving hand.
    Along the ray the noise also grows with speed (measured: 1.35 mm still,
    3.7 mm at 0.1-0.2 m/s, 4.5 at 0.2-0.4; sideways stays ~0.3-0.5 mm; not
    camera skew, which is corrected): speed_k (m per m/s) adds speed_k * |v|
    to the along-ray sigma. Worse frames count less: the noise grows with
    (err / err_ref)^2 above err_ref px, and x mono_scale^2 for single-camera poses.

    Process noise q (m^2/s^3, white acceleration): higher follows fast moves
    with less lag, lower smooths more (it must be ~0.001: 0.3 already lets the
    filter expect 2 mm per frame, so it hardly smooths). A frame whose
    innovation is beyond gate (Mahalanobis^2) is a maneuver the model didn't
    expect (a reversal, a reach starting): the predicted covariance is inflated
    by d^2 / gate so the filter catches up at once instead of lagging. Frames
    are never dropped here: one bad frame and a sudden real move look the same,
    and dropping them (then restarting) added jumps on real motion (tested
    2026-10-10). Bad frames are caught upstream (hand3d reprojection / bone / fit gates).

    v_tau (s, 0 = off): the velocity decays with this time constant between
    measurements (damped constant velocity). Cut overshoot little and cost
    50-200 ms of lag in tests: off.

    StereoIMM (below) runs two of these and is the default; alone, sigma 4 /
    q 0.001 vs the one-euro on the recordings: lag 20-40 ms instead of 85-90,
    sweep jitter -22%, acceleration -36%, ~1 mm more overshoot at stops.
    """

    def __init__(self, f, baseline, sigma_px=4.0, q=0.001, err_ref=3.0, mono_scale=4.0,
                 gate=16.0, v0_sigma=0.5, v_tau=0.0, ray_px=None, speed_k=0.0):
        self.f, self.B = f, baseline
        self.sigma_px, self.q, self.err_ref, self.mono_scale = sigma_px, q, err_ref, mono_scale
        self.ray_px = sigma_px if ray_px is None else ray_px
        self.gate, self.v0_sigma, self.v_tau, self.speed_k = gate, v0_sigma, v_tau, speed_k
        self.reset()

    def reset(self):
        self.x = self.P = self.t = None
        self.maneuver = False  # last update was inflated

    def meas_cov(self, p, err=None, mono=False, speed=0.0):
        z = max(float(p[2]), 0.1)
        s_lat = z / self.f * self.sigma_px
        s_ray = z * z / (self.f * self.B) * self.ray_px + self.speed_k * speed
        r = np.asarray(p, float) / np.linalg.norm(p)
        Rm = s_lat ** 2 * np.eye(3) + (s_ray ** 2 - s_lat ** 2) * np.outer(r, r)
        k = 1.0
        if err is not None and np.isfinite(err) and err > self.err_ref:
            k *= (err / self.err_ref) ** 2
        if mono:
            k *= self.mono_scale ** 2
        return k * Rm

    def start_state(self, p, Rm):
        x = np.concatenate([p, np.zeros(3)])
        P = np.zeros((6, 6))
        P[:3, :3] = Rm
        P[3:, 3:] = self.v0_sigma ** 2 * np.eye(3)
        return x, P

    def predict(self, x, P, dt):
        I3 = np.eye(3)
        damp = math.exp(-dt / self.v_tau) if self.v_tau > 0 else 1.0
        F = np.block([[I3, dt * I3], [np.zeros((3, 3)), damp * I3]])
        Q = self.q * np.block([[dt ** 3 / 3 * I3, dt ** 2 / 2 * I3], [dt ** 2 / 2 * I3, dt * I3]])
        return F @ x, F @ P @ F.T + Q

    def update(self, x, P, p, err=None, mono=False, maneuver=True):
        """(x, P, likelihood of p) after measuring p. maneuver: inflate on a big innovation."""
        Rm = self.meas_cov(p, err, mono, float(np.linalg.norm(x[3:])))
        y = p - x[:3]
        S = P[:3, :3] + Rm
        Si = np.linalg.inv(S)
        d2 = float(y @ Si @ y)
        lik = math.exp(-0.5 * min(d2, 200.0)) / math.sqrt(np.linalg.det(2 * math.pi * S))
        self.maneuver = maneuver and d2 > self.gate
        if self.maneuver:
            P = P * (d2 / self.gate)
            S = P[:3, :3] + Rm
            Si = np.linalg.inv(S)
        K = P[:, :3] @ Si
        P = P - K @ S @ K.T
        return x + K @ y, 0.5 * (P + P.T), lik

    def __call__(self, p, t, err=None, mono=False):
        """(filtered p, v) in cam frame for a measured hand position at time t."""
        p = np.asarray(p, float)
        if self.x is None or t <= self.t:
            self.x, self.P = self.start_state(p, self.meas_cov(p, err, mono))
            self.t = t
            return self.x[:3].copy(), self.x[3:].copy()
        x, P = self.predict(self.x, self.P, min(t - self.t, 0.1))
        self.t = t
        self.x, self.P, _ = self.update(x, P, p, err, mono)
        return self.x[:3].copy(), self.x[3:].copy()


# StereoIMM / single StereoKalman settings per landmark backend, tuned offline on the
# recordings (scored against a zero-lag reference, kalman_bench.py). WiLoR (tuned 2026-10-10
# on recordings/wilor/, WiLoR's own frames at its live ~15 Hz, kalman_bench --ref-s 0.25): its
# noise is higher on a still hand but barely grows with speed, so low, flat noise (speed_k 0)
# and a very sticky still model. vs the earlier MediaPipe smooth preset (sigma 4, ray 8,
# speed_k 0.005, p_stay 0.8) on the same WiLoR data:
#   smooth:     tracking error about halved (sweep 5.4 -> 2.7 mm, drop-out 10.1 -> 3.3),
#               lag 20-56 -> 5-13 ms, still-hand wobble 0.56 -> 0.31 mm
#   responsive: drop-out error a further -10%, ~5 ms less lag, still wobble 0.40 mm
IMM_PRESETS = {
    "wilor": {
        "smooth": dict(sigma_px=1.5, ray_px=1.5, speed_k=0.0, q=(1e-6, 0.01), p_stay=0.995),
        "responsive": dict(sigma_px=1.0, ray_px=1.0, speed_k=0.0, q=(1e-6, 0.01), p_stay=0.99),
    },
}
# single StereoKalman (--pos-filter kalman) defaults per backend
KALMAN_DEFAULTS = {
    "wilor": dict(sigma_px=1.0, q=0.0001),  # vs (4, 0.001): moving error -45..65%, still equal
}


class StereoIMM:
    """Interacting multiple models (IMM): a "still" constant-velocity model (q ~0,
    a hand that isn't moving) and a "moving" one (higher q) run side by side;
    each frame their states are mixed by the Markov switching probabilities
    (p_stay on the diagonal), each updates, and the output is the mix weighted
    by how well each predicted the measurement. A still hand gets the still
    model's heavy smoothing, a moving one the moving model's quick response,
    and a stop switches back quickly, so less overshoot than one model with a
    compromise q. Same interface as StereoKalman: (p, v) = imm(p, t, err, mono).
    The single-model maneuver inflation is off here (the moving model covers it).
    """

    def __init__(self, f, baseline, q=(1e-6, 0.01), p_stay=0.8, **kw):
        self.models = [StereoKalman(f, baseline, q=qi, **kw) for qi in q]
        n = len(q)
        self.T = np.full((n, n), (1.0 - p_stay) / (n - 1))
        np.fill_diagonal(self.T, p_stay)
        self.reset()

    def reset(self):
        n = len(self.models)
        self.mu = np.full(n, 1.0 / n)  # model probabilities
        self.xs = self.Ps = None
        self.t = None
        self.maneuver = False

    def __call__(self, p, t, err=None, mono=False):
        p = np.asarray(p, float)
        n = len(self.models)
        if self.xs is None or t <= self.t:
            x, P = self.models[0].start_state(p, self.models[0].meas_cov(p, err, mono))
            self.xs, self.Ps = [x.copy() for _ in range(n)], [P.copy() for _ in range(n)]
            self.mu, self.t = np.full(n, 1.0 / n), t
            return x[:3].copy(), x[3:].copy()
        dt = min(t - self.t, 0.1)
        self.t = t
        c = self.T.T @ self.mu  # predicted model probabilities
        lik = np.zeros(n)
        xs, Ps = [], []
        for j, m in enumerate(self.models):
            w = self.T[:, j] * self.mu / c[j]  # mixing weights into model j
            x0 = sum(w[i] * self.xs[i] for i in range(n))
            P0 = sum(w[i] * (self.Ps[i] + np.outer(self.xs[i] - x0, self.xs[i] - x0)) for i in range(n))
            x, P = m.predict(x0, P0, dt)
            x, P, lik[j] = m.update(x, P, p, err, mono, maneuver=False)
            xs.append(x)
            Ps.append(P)
        self.xs, self.Ps = xs, Ps
        mu = c * lik
        self.mu = mu / mu.sum() if mu.sum() > 0 else c
        x = sum(self.mu[j] * xs[j] for j in range(n))
        return x[:3].copy(), x[3:].copy()


class StillHold:
    """Freezes the position while the hand is still, for a steady arm when lining up
    a grasp. Engages after the filtered speed stays under v_on for hold_s; releases
    once the hand drifts more than release_m from the held point or speeds past v_off.
    On release the held-to-live gap fades out with time constant fade_s instead of
    stepping (a 5-10 mm step per release showed up as acceleration spikes)."""

    def __init__(self, v_on=0.01, v_off=0.03, hold_s=0.15, release_m=0.005, fade_s=0.1):
        self.v_on, self.v_off, self.hold_s, self.release_m = v_on, v_off, hold_s, release_m
        self.fade_s = fade_s
        self.reset()

    def reset(self):
        self.held = None
        self.slow_since = None
        self.gap = None  # (held - live at release, release time)

    def __call__(self, p, v, t):
        p = np.asarray(p, float)
        sp = float(np.linalg.norm(v))
        if self.held is not None:
            if sp > self.v_off or np.linalg.norm(p - self.held) > self.release_m:
                self.gap = (self.held - p, t)
                self.held, self.slow_since = None, None
            else:
                return self.held.copy()
        if sp < self.v_on:
            if self.slow_since is None:
                self.slow_since = t
            elif t - self.slow_since >= self.hold_s:
                self.held = self._faded(p, t)
                self.gap = None
                return self.held.copy()
        else:
            self.slow_since = None
        return self._faded(p, t)

    def _faded(self, p, t):
        if self.gap is None:
            return p
        k = math.exp(-(t - self.gap[1]) / self.fade_s)
        if k < 1e-3:
            self.gap = None
            return p
        return p + k * self.gap[0]


class RotationShaper:
    """Hand rotation -> gripper rotation, per axis in robot rpy (roll about X,
    pitch about Y, + = down, yaw about Z), each a gain with a soft deadband:
        out = clip(gain * a^3 / (a^2 + deadband^2), limits)
    ~0 for |a| << deadband, ~gain * a (minus gain * deadband^2 / a) beyond it,
    smooth in between. Each gain is a number or a (negative, positive)
    direction pair: pitch up (-) / down (+), since the arm tilts the gripper
    up only ~30 deg but down 90, and roll, whose range is lopsided too."""

    def __init__(self, roll_gain=1.0, pitch_gain=(1.0, 1.0), yaw_gain=1.0,
                 deadband=0.0, limits_deg=ROT_LIMITS_DEG, scale=1.0):
        pair = lambda g: tuple(float(v) * scale for v in (np.broadcast_to(g, 2)))  # noqa: E731
        self.gains = (pair(roll_gain), pair(pitch_gain), pair(yaw_gain))
        self.deadband = deadband
        self.limits = np.radians(np.asarray(limits_deg, dtype=float))

    def axis(self, a, gain, lim):
        d = self.deadband
        if d > 0:
            a = a ** 3 / (a * a + d * d)
        return float(np.clip(gain * a, lim[0], lim[1]))

    def __call__(self, dR):
        """(gripper rotation (3,3), its rpy) for a hand rotation dR (robot axes)."""
        out = tuple(self.axis(a, g[1] if a > 0 else g[0], lim)
                    for a, g, lim in zip(matrix_to_rpy(dR), self.gains, self.limits))
        return rpy_to_matrix(*out), out


def gains_from_range(rng, limits_deg=ROT_LIMITS_DEG, use=0.85, lo=0.7, hi=3.0):
    """Gains from the calibrated trackable hand range, so `use` of the range
    reaches the gripper's limit; magnitudes clipped to [lo, hi] (more than ~3x
    amplifies the tracking noise too much). rng: {"roll": (neg, pos), "pitch":
    (up, down)} hand angles in deg (magnitudes; None = not measured).
    Pitch: {"pitch": (g_up, g_down)}, None per unmeasured direction.
    Roll is calibrated one way only (the hand's clockwise turn, whichever sign
    that is in robot axes) and drives the gripper's 0..+90: {"roll": (g, g)},
    negative if the hand's clockwise is negative roll, so the turn back the
    other way clips at 0. (None, None) if not measured."""
    out = {}
    up, down = rng.get("pitch", (None, None))
    out["pitch"] = tuple(None if r is None or r <= 0 else
                         float(np.clip(abs(limits_deg[1][i]) / (use * r), lo, hi))
                         for i, r in enumerate((up, down)))
    neg, pos = rng.get("roll", (None, None))
    r, sign = (pos, 1.0) if pos else (neg, -1.0)
    g = None if not r or r <= 0 else sign * float(np.clip(limits_deg[0][1] / (use * r), lo, hi))
    out["roll"] = (g, g)
    return out


def facing_weight(p, R_hand, facing_deg=FACING_DEG):
    """(trust 0..1, palm angle deg): angle between the palm normal (hand z) and
    the ray from cam0 to the hand; 0 = palm facing the cameras, 90 = edge-on."""
    ray = np.asarray(p, float) / np.linalg.norm(p)
    ang = math.degrees(math.acos(min(1.0, abs(float(R_hand[:, 2] @ ray)))))
    lo, hi = facing_deg
    return float(np.clip((lo - ang) / (lo - hi), 0.0, 1.0)), ang


class GripFilter:
    """Finger curl -> gripper 0 (open) .. 1 (closed), made robust for pick and place:
    - hold: while the palm is turned more than hold_deg from the cameras, or there is
      no curl (mono), the grip keeps its value. Curl is unreliable there: in a live
      WiLoR run (2026-10-10) a still hand tilted forward (palm 53-61 deg) read curl
      0.70..1.18 frame to frame, so the gripper flipped open/closed.
    - backlash of `band` (curl units): the output target only moves once the curl
      has moved by more than half the band (flat-hand noise was about +-0.03).
    - time-based smoothing (tau), the same at any frame rate.
    grip = f(curl, palm_angle_deg, t) -> float, or None until the first curl."""

    def __init__(self, open_curl, closed_curl, hold_deg=55.0, band=0.06, tau=0.04, max_dt=0.1):
        self.open, self.closed = float(open_curl), float(closed_curl)
        self.hold_deg, self.band, self.tau, self.max_dt = hold_deg, band, tau, max_dt
        self.curl = None    # backlash-filtered curl
        self.grip = None    # output
        self.t = None
        self.held = False

    def __call__(self, curl, palm_angle, t):
        self.held = curl is None or palm_angle > self.hold_deg
        dt = 0.0 if self.t is None else min(max(0.0, t - self.t), self.max_dt)
        self.t = t
        if self.held:
            return self.grip
        h = self.band / 2
        self.curl = curl if self.curl is None else min(max(self.curl, curl - h), curl + h)
        g = min(1.0, max(0.0, (self.open - self.curl) / (self.open - self.closed)))
        self.grip = g if self.grip is None else self.grip + (1.0 - math.exp(-dt / self.tau)) * (g - self.grip)
        return self.grip


class HandMapper:
    MAX_DT = 0.1  # s

    def __init__(self, A, neutral_p, neutral_R, scale=1.0, rot_scale=1.0,
                 neutral_offset=(0.0, 0.0, 0.0), box_min=BOX_MIN, box_max=BOX_MAX,
                 max_lin_speed=0.25, max_ang_speed=math.radians(120),
                 pos_filter=(1.0, 2.0), rot_filter=(1.0, 0.5), shaper=None,
                 facing_deg=FACING_DEG, kalman=None, still_hold=None, lead_s=0.0):
        self.A = np.asarray(A, dtype=float)
        self.scale, self.rot_scale = scale, rot_scale
        self.box_min, self.box_max = np.asarray(box_min), np.asarray(box_max)
        self.max_lin, self.max_ang = max_lin_speed, max_ang_speed
        self.hand_p, self.hand_R = np.asarray(neutral_p, float), np.asarray(neutral_R, float)
        self.out_p = np.clip(np.asarray(neutral_offset, float), self.box_min, self.box_max)
        self.out_R = np.eye(3)
        self.f_pos = OneEuro(*pos_filter)
        # kalman (StereoKalman) replaces the one-euro position filter when given:
        # it filters in the cam frame, before the mapping, where the noise is known
        self.kalman, self.still_hold, self.lead_s = kalman, still_hold, lead_s
        self.f_rot = OneEuro(*rot_filter)  # on the hand rotation (rotvec), before shaping
        # rot_scale = plain uniform gain (old behavior) unless a shaper is given
        self.shaper = shaper or RotationShaper(scale=rot_scale)
        self.facing_deg = facing_deg
        self.rv_hand = None              # trust-weighted hand rotvec (held when edge-on)
        self.hand_rpy = np.zeros(3)      # filtered hand rotation, for display
        self.palm_angle, self.trust = 0.0, 1.0
        # slew-limited output state; starts at home (offset 0, the mid-workspace pose)
        self.cur_p = np.zeros(3)
        self.cur_R = np.eye(3)
        self.cur_t = None
        self.last_raw = None  # (p, R) of the latest unclutched target, for re-anchoring

    def raw_target(self, p, R_hand):
        """Unfiltered (offset p (3,), rotation (3,3)) for a hand pose in cam frame."""
        p_out = self.out_p + self.scale * (self.A @ (np.asarray(p) - self.hand_p))
        return np.clip(p_out, self.box_min, self.box_max), self.shaper(self.hand_rotation(R_hand))[0]

    def hand_rotation(self, R_hand):
        """Hand rotation from the calibrated neutral, in robot axes."""
        return self.A @ (R_hand @ self.hand_R.T) @ self.A.T

    def reanchor(self, p, R_hand=None):
        """Clutch release: the current hand position now maps to the current
        output position. Orientation keeps its calibrated anchor (no drift);
        R_hand is accepted for compatibility and ignored."""
        self.hand_p = np.asarray(p, float)
        self.out_p = self.cur_p.copy()
        self.f_pos.reset()
        for f in (self.kalman, self.still_hold):
            if f is not None:
                f.reset()

    def update(self, p, R_hand, t, err=None, mono=False):
        """Hand pose at time t (s) -> [x y z roll pitch yaw] offset for /ball_position.
        err (reprojection px) and mono (single-camera pose) weight the Kalman filter."""
        if self.kalman is not None:
            pc, vc = self.kalman(p, t, err, mono)
            if self.still_hold is not None:
                pc = self.still_hold(pc, vc, t) + (0.0 if self.still_hold.held is not None
                                                   else self.lead_s * vc)
            else:
                pc = pc + self.lead_s * vc
            p_out = self.out_p + self.scale * (self.A @ (pc - self.hand_p))
            tp = np.clip(p_out, self.box_min, self.box_max)
        else:
            p_out = self.out_p + self.scale * (self.A @ (np.asarray(p) - self.hand_p))
            tp = self.f_pos(np.clip(p_out, self.box_min, self.box_max), t)
        # hand rotation as a rotvec (continuous below 180 deg); a palm turned toward
        # edge-on moves it only partly (trust), fully edge-on holds it
        rv = matrix_to_rotvec(self.hand_rotation(R_hand))
        self.trust, self.palm_angle = facing_weight(p, R_hand, self.facing_deg)
        if self.rv_hand is not None:
            rv = self.rv_hand + self.trust * (rv - self.rv_hand)
        self.rv_hand = rv
        dR = rotvec_to_matrix(self.f_rot(rv, t))
        self.hand_rpy = np.array(matrix_to_rpy(dR))
        tR = self.shaper(dR)[0] @ self.out_R
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
    assert np.allclose(out, 0, atol=1e-6), out  # neutral hand -> home
    # hand 10 cm toward cameras (cam -z) = robot -X: arm 5 cm back
    for _ in range(200):
        out = m.update(p0 + [0, 0, -0.10], np.eye(3), t)
        t += 1 / 30
    assert np.allclose(out[:3], [-0.05, 0, 0], atol=1e-4), out
    # hand up 10 cm (cam -y) -> +z 5 cm
    for _ in range(200):
        out = m.update(p0 + [0, -0.10, 0], np.eye(3), t)
        t += 1 / 30
    assert np.allclose(out[:3], [0, 0, 0.05], atol=1e-4), out
    # wrist rotation: hand rotated about cam -y (robot +Z) by 30 deg -> yaw 30
    Rc = A.T @ rpy_to_matrix(0, 0, math.radians(30)) @ A
    for _ in range(300):
        out = m.update(p0, Rc, t)
        t += 1 / 30
    assert np.allclose(out[3:], [0, 0, math.radians(30)], atol=1e-3), out
    # clutch: hold, move hand far, release -> no jump
    held = m.hold(t)
    m.reanchor(p0 + [0.2, 0.2, 0.2], np.eye(3))
    out = m.update(p0 + [0.2, 0.2, 0.2], Rc, t + 0.03)
    assert np.allclose(out, held, atol=1e-6), (out, held)
    # orientation is not re-anchored: many clutches with a turned hand and
    # noise, then the hand back at the calibrated neutral -> output rpy ~0
    rng = np.random.default_rng(0)
    for k in range(20):
        Rk = A.T @ rpy_to_matrix(*rng.normal(0, 0.4, 3)) @ A
        for _ in range(10):
            m.update(p0, Rk, t)
            t += 1 / 30
        t += 0.5
        m.reanchor(p0, Rk @ rotvec_to_matrix(rng.normal(0, 0.05, 3)))
    for _ in range(300):
        out = m.update(p0, np.eye(3), t)
        t += 1 / 30
    assert np.allclose(out[3:], 0, atol=1e-3), out
    # slew limit: default 0.25 m/s, a 10 cm jump takes > 0.3 s
    m2 = HandMapper(A, p0, np.eye(3), neutral_offset=(-0.10, 0.0, -0.05))
    out = m2.update(p0, np.eye(3), 0.0)
    assert np.allclose(out, 0)  # starts at home, ramps to the neutral offset
    out = m2.update(p0, np.eye(3), 0.1)
    assert np.linalg.norm(out[:3]) <= 0.025 + 1e-9
    # hand lost for 5 s, reappears far away: still only MAX_DT worth of motion
    before = m2.update(p0, np.eye(3), 0.2)
    out = m2.update(p0 + [0.3, 0.3, 0.3], np.eye(3), 5.2)
    assert np.linalg.norm(out[:3] - before[:3]) <= 0.25 * HandMapper.MAX_DT + 1e-9
    # RotationShaper: per-axis gains, pitch up/down, deadband, limits
    sh = RotationShaper(roll_gain=1.5, pitch_gain=(1.0, 2.0), deadband=math.radians(3))
    d = lambda *rpy: np.degrees(sh(rpy_to_matrix(*np.radians(rpy)))[1])  # noqa: E731
    assert abs(d(0, 30, 0)[1] - 60 * (1 - 9 / 909)) < 1e-6        # down x2 (minus deadband)
    assert abs(d(0, -30, 0)[1] + 30 * (1 - 9 / 909)) < 1e-6       # up x1
    assert -60 < d(0, -60, 0)[1] < -59.8 and abs(d(0, 60, 0)[1] - 85) < 1e-6  # pitch up not cut at -35
    assert abs(d(80, 0, 0)[0] - 90) < 1e-6 and d(-40, 0, 0)[0] == 0  # roll 0..90, one way
    # roll calibrated one way: hand clockwise = -roll here -> gripper +roll
    g = gains_from_range({"roll": (60.0, None), "pitch": (30.0, 45.0)})
    assert g["roll"][0] == g["roll"][1] < 0 and abs(g["roll"][0] + 90 / 51) < 1e-9, g
    assert abs(g["pitch"][0] - 3.0) < 1e-9 and abs(g["pitch"][1] - 85 / 38.25) < 1e-9, g
    sr = RotationShaper(roll_gain=g["roll"])
    dr = lambda a: math.degrees(sr(rpy_to_matrix(math.radians(a), 0, 0))[1][0])  # noqa: E731
    assert abs(dr(-51) - 90) < 1e-6 and abs(dr(-20) - 20 * 90 / 51) < 1e-6 and dr(20) == 0
    assert gains_from_range({})["roll"] == (None, None)
    assert np.all(np.abs(d(1, -1, 1)) < 0.2)                      # inside the deadband ~0

    # mapper with the shaper: hand pitched 30 deg down -> gripper ~60 down
    ms = HandMapper(A, p0, np.eye(3), shaper=sh, max_ang_speed=math.radians(360))
    Rp = A.T @ rpy_to_matrix(0, math.radians(30), 0) @ A
    for k in range(150):
        out = ms.update(p0, Rp, k / 30)
    assert abs(math.degrees(out[4]) - 59.4) < 0.1, out
    # palm turned edge-on to the cameras: rotation held, not followed
    assert ms.trust == 1.0
    Re = A.T @ rpy_to_matrix(0, math.radians(85), 0) @ A
    for k in range(150, 300):
        out = ms.update(p0, Re, k / 30)
    assert ms.trust == 0.0 and abs(math.degrees(out[4]) - 59.4) < 0.1, (ms.palm_angle, out)
    w, ang = facing_weight(p0, A.T @ rpy_to_matrix(0, math.radians(65), 0) @ A)
    assert 0 < w < 1, (w, ang)                                    # partly trusted in between
    # StereoKalman: depth (along the ray) smoothed harder, a ramp followed without
    # lag, a sudden 10 cm move caught up quickly (maneuver inflation)
    kf = StereoKalman(950.0, 0.2)
    pk = np.array([0.0, 0.0, 0.8])
    sig = kf.meas_cov(pk) * (0.35 / 6.0) ** 2  # still-hand noise, as measured
    noise = rng.multivariate_normal(np.zeros(3), sig, 600)
    est = np.array([kf(pk + n, k / 30)[0] for k, n in enumerate(noise)])[100:]
    ratio = np.std(est, axis=0) / np.std(noise[100:], axis=0)  # ~0.43 sideways, ~0.34 depth
    assert ratio[2] < 0.4 and ratio[2] < ratio[0] < 0.5, ratio
    kf.reset()
    vr = np.array([0.2, 0.0, 0.0])
    for k in range(60):
        e, v = kf(pk + vr * k / 30, k / 30)
    assert np.linalg.norm(e - (pk + vr * 59 / 30)) < 1e-3 and np.linalg.norm(v - vr) < 0.02, (e, v)
    kf.reset()
    for k in range(30):
        kf(pk, k / 30)
    for k in range(30, 36):  # 10 cm step, 0.2 s later
        e, _ = kf(pk + [0.1, 0, 0], k / 30)
    assert abs(e[0] - 0.1) < 0.012, e                             # caught up, small overshoot
    for k in range(36, 90):
        e, _ = kf(pk + [0.1, 0, 0], k / 30)
    assert abs(e[0] - 0.1) < 0.002, e                              # settled
    # StereoIMM: smoother than one model on a still hand, follows a ramp, catches a step
    # up quickly and settles with less overshoot
    for backend, preset in [(b, kw) for b, ps in IMM_PRESETS.items() for kw in ps.values()]:
        imm = StereoIMM(950.0, 0.2, **preset)
        est_i = np.array([imm(pk + n, k / 30)[0] for k, n in enumerate(noise)])[100:]
        assert np.all(np.std(est_i, axis=0) < np.std(noise[100:], axis=0)), preset
        imm.reset()
        for k in range(60):
            e, v = imm(pk + vr * k / 30, k / 30)
        assert np.linalg.norm(e - (pk + vr * 59 / 30)) < 2e-3 and np.linalg.norm(v - vr) < 0.03, (preset, e, v)
        imm.reset()
        for k in range(30):
            imm(pk, k / 30)
        xs = [imm(pk + [0.1, 0, 0], k / 30)[0][0] for k in range(30, 60)]
        # WiLoR's low noise settings overshoot this instant 10 cm step by ~12 mm (on its real
        # reaches they overshoot less than the old MediaPipe presets did: 4.6 vs 7.2 mm)
        max_ov = 0.115
        assert abs(xs[5] - 0.1) < 0.01 and max(xs) < max_ov and abs(xs[-1] - 0.1) < 0.002, (preset, xs)
    # StillHold: freezes a still hand, releases on motion without a step
    sh = StillHold()
    outs = [sh(np.array([0.0, 0.0, 0.0]) + 0.0005 * rng.normal(size=3), np.zeros(3), k / 30) for k in range(30)]
    assert sh.held is not None and np.allclose(outs[-1], outs[-2])
    pr = np.array(outs[-1])
    for k in range(30, 60):  # moving off at 0.1 m/s
        o = sh(np.array([0.1 * (k - 30) / 30, 0, 0]), np.array([0.1, 0, 0]), k / 30)
        assert np.linalg.norm(o - pr) < 0.006, (k, o, pr)
        pr = o
    assert sh.held is None and abs(o[0] - 0.1 * 29 / 30) < 1e-3
    # GripFilter: noise inside the band doesn't move it, a real close does, edge-on holds
    gf = GripFilter(1.0, 0.75)
    gs = [gf(1.0 + 0.025 * math.sin(k), 20.0, k / 16) for k in range(40)]  # flat hand, +-0.025 noise
    assert max(gs) - min(gs) < 0.02, (min(gs), max(gs))
    for k in range(40, 60):  # fist
        g = gf(0.6, 20.0, k / 16)
    assert g > 0.99, g
    for k in range(60, 70):  # tilted forward: curl garbage, grip holds
        assert gf(1.2 if k % 2 else 0.7, 70.0, k / 16) == g and gf.held
    for k in range(70, 90):  # palm back, hand opens
        g = gf(1.05, 30.0, k / 16)
    assert g < 0.01, g
    assert GripFilter(1.0, 0.75)(None, 0.0, 0.0) is None  # no curl yet
    print("hand_mapping self-test OK")


if __name__ == "__main__":
    _selftest()
