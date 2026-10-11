#!/home/peyton/kh/.venv/bin/python
"""Stereo hand geometry: pure numpy/OpenCV math, no cameras, no ROS.

Frames:
    cam   = camera 0 (reference) optical frame: x right, y down, z forward, meters.
            Camera 1 is at X1 = R @ X0 + t (OpenCV stereo convention).
    robot = SO-101 base_link axes: +X forward, +Y left, +Z up.
            A (3x3, rows = robot axes expressed in cam coords) maps
            cam vectors to robot vectors: v_robot = A @ v_cam.

Hand frame (from the backend's 21 landmarks, OpenPose order, triangulated to 3D):
    origin = centroid of palm points {0 wrist, 5, 9, 13, 17 MCPs}
    x = wrist -> middle MCP (0 -> 9), where the hand points
    z = palm normal (5 - 0) x (17 - 0); the sign flips with chirality,
        which is fine: only rotations relative to the calibrated neutral are used
    y = z x x

Self-test (synthetic stereo, no hardware):
    python Computer-Vision/common/hand3d.py
"""
import math

import cv2
import numpy as np

WRIST, THUMB_TIP, INDEX_MCP, MIDDLE_MCP, RING_MCP, PINKY_MCP = 0, 4, 5, 9, 13, 17
PALM = [0, 5, 9, 13, 17]
FINGERTIPS = [8, 12, 16, 20]  # index..pinky; thumb is unreliable for fist detection
# rigid-ish bones used to reject bad triangulations (palm only; fingers bend)
BONES = [(0, 5), (0, 9), (0, 13), (0, 17), (5, 9), (9, 13), (13, 17), (5, 17)]


# ---------- rotations (ROS fixed-axis convention, same as so101_mirror.kinematics) ----------

def rpy_to_matrix(roll, pitch, yaw):
    """R = Rz(yaw) @ Ry(pitch) @ Rx(roll)."""
    cr, sr, cp, sp, cy, sy = (math.cos(roll), math.sin(roll), math.cos(pitch),
                              math.sin(pitch), math.cos(yaw), math.sin(yaw))
    return np.array([[cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
                     [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
                     [-sp, cp * sr, cp * cr]])


def matrix_to_rpy(R):
    pitch = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    if abs(math.cos(pitch)) < 1e-9:  # gimbal lock: put everything in yaw
        return 0.0, pitch, math.atan2(-R[0, 1], R[1, 1])
    return math.atan2(R[2, 1], R[2, 2]), pitch, math.atan2(R[1, 0], R[0, 0])


def rotvec_to_matrix(v):
    return cv2.Rodrigues(np.asarray(v, dtype=float).reshape(3, 1))[0]


def matrix_to_rotvec(R):
    return cv2.Rodrigues(np.asarray(R, dtype=float))[0].ravel()


def project_to_so3(M):
    """Nearest rotation matrix (SVD), e.g. for averaging rotations."""
    U, _, Vt = np.linalg.svd(M)
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


# ---------- camera model ----------

def intrinsics_from_fov(width, height, hfov_deg):
    """Pinhole K from a horizontal field of view (used when there is no board calibration)."""
    f = 0.5 * width / math.tan(math.radians(hfov_deg) / 2)
    return np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])


def normalize_points(px, K, dist):
    """Pixel (N,2) -> undistorted normalized image coords (N,2)."""
    px = np.asarray(px, dtype=np.float64).reshape(-1, 1, 2)
    return cv2.undistortPoints(px, K, dist).reshape(-1, 2)


def triangulate(n0, n1, R, t):
    """Normalized coords in cam0/cam1 (N,2) -> 3D points in cam0 frame (N,3)."""
    P0 = np.hstack([np.eye(3), np.zeros((3, 1))])
    P1 = np.hstack([R, np.asarray(t, dtype=float).reshape(3, 1)])
    X = cv2.triangulatePoints(P0, P1, n0.T.astype(np.float64), n1.T.astype(np.float64))
    return (X[:3] / X[3]).T


def reprojection_error(X, n0, n1, R, t, f0, f1):
    """Per-point reprojection error in (approximate) pixels, max over both views."""
    def proj(Xc):
        return Xc[:, :2] / Xc[:, 2:3]
    e0 = np.linalg.norm(proj(X) - n0, axis=1) * f0
    X1 = X @ R.T + np.asarray(t).reshape(1, 3)
    e1 = np.linalg.norm(proj(X1) - n1, axis=1) * f1
    bad = (X[:, 2] <= 0) | (X1[:, 2] <= 0)  # behind a camera
    e = np.maximum(e0, e1)
    e[bad] = np.inf
    return e


# ---------- hand geometry ----------

def palm_center(P):
    return P[PALM].mean(axis=0)


def hand_frame(P):
    """(origin (3,), R (3,3) columns = hand x/y/z axes in the points' frame)."""
    x = P[MIDDLE_MCP] - P[WRIST]
    z = np.cross(P[INDEX_MCP] - P[WRIST], P[PINKY_MCP] - P[WRIST])
    x = x / np.linalg.norm(x)
    z = z - x * (z @ x)
    z = z / np.linalg.norm(z)
    return palm_center(P), np.column_stack([x, np.cross(z, x), z])


def bone_lengths(P):
    return np.array([np.linalg.norm(P[a] - P[b]) for a, b in BONES])


def bones_ok(P, ref, tol):
    """True if every palm bone is within ±tol (fraction) of the calibrated length."""
    r = bone_lengths(P) / ref
    r = r[np.isfinite(r)]  # bones to a dropped (NaN) point aren't checked
    return bool(np.all(np.abs(r - 1) < tol))


def curl(P):
    """Finger curl: median fingertip (index..pinky) distance from the palm center,
    in palm lengths (wrist -> middle MCP). ~1.4+ open hand, < ~0.9 fist.
    NaN if fewer than 3 fingertips are known (dropped landmarks are NaN)."""
    c = palm_center(P)
    palm_len = np.linalg.norm(P[MIDDLE_MCP] - P[WRIST])
    d = np.linalg.norm(P[FINGERTIPS] - c, axis=1)
    d = d[np.isfinite(d)]
    return float(np.median(d) / palm_len) if len(d) >= 3 else float("nan")


def is_fist(P):
    """Fingertips (index..pinky) curled close to the palm center."""
    c = palm_center(P)
    palm_len = np.linalg.norm(P[MIDDLE_MCP] - P[WRIST])
    d = np.linalg.norm(P[FINGERTIPS] - c, axis=1)
    return bool(np.all(d < 0.9 * palm_len))


def palm_pnp(n_px, K, dist, template, R_cv, t_cv, guess=None):
    """Single-view fallback: pose of the rigid palm template from one camera.

    n_px: (21,2) pixels in that camera. template: (5,3) palm points in the
    hand frame (from calibration). R_cv, t_cv: that camera's pose relative to
    cam0 (identity/zero for cam0). Returns the 5 palm points in cam0 frame,
    or None. guess: previous (rvec, tvec) in that camera, for stability.
    """
    img = np.asarray(n_px, dtype=np.float64)[PALM].reshape(-1, 1, 2)
    obj = template.astype(np.float64).reshape(-1, 1, 3)
    if guess is not None:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, guess[0].copy(), guess[1].copy(),
                                      useExtrinsicGuess=True, flags=cv2.SOLVEPNP_ITERATIVE)
    else:
        ok, rvec, tvec = cv2.solvePnP(obj, img, K, dist, flags=cv2.SOLVEPNP_SQPNP)
    if not ok:
        return None, None
    Rh = rotvec_to_matrix(rvec)
    Xc = template @ Rh.T + tvec.reshape(1, 3)              # in that camera
    X0 = (Xc - np.asarray(t_cv).reshape(1, 3)) @ R_cv       # back to cam0: R^T (X - t)
    if np.any(X0[:, 2] <= 0):
        return None, None
    return X0, (rvec, tvec)


def palm_template(P):
    """Palm points (5,3) expressed in the hand frame, for palm_pnp() and palm_fit()."""
    o, R = hand_frame(P)
    return (P[PALM] - o) @ R


def palm_fit(X, template):
    """Least-squares rigid fit (Kabsch) of the palm template to measured palm points.

    X: (5,3) palm points (PALM order). template: (5,3) from palm_template().
    Returns (origin, R, rms): X ~= template @ R.T + origin, R columns = hand
    axes (same convention as hand_frame()), rms = fit residual (m). Uses all
    5 palm points, so one noisy landmark tilts the frame much less than in
    hand_frame(), which builds it from 3.
    """
    X = np.asarray(X, dtype=float)
    T = np.asarray(template, dtype=float)
    cx, ct = X.mean(axis=0), T.mean(axis=0)
    U, _, Vt = np.linalg.svd((T - ct).T @ (X - cx))
    s = 1.0 if np.linalg.det(Vt.T @ U.T) >= 0 else -1.0  # no reflections
    R = Vt.T @ np.diag([1.0, 1.0, s]) @ U.T
    o = cx - R @ ct
    rms = float(np.sqrt(np.mean(np.sum((T @ R.T + o - X) ** 2, axis=1))))
    return o, R, rms


def twist_about(R, axis):
    """Twist angle (rad) of rotation R about a unit axis, in R's frame (swing-twist split)."""
    v = matrix_to_rotvec(R)
    th = float(np.linalg.norm(v))
    if th < 1e-12:
        return 0.0
    return 2.0 * math.atan2(math.sin(th / 2) * float(v @ np.asarray(axis, float)) / th, math.cos(th / 2))


def twist_x(R):
    """Twist angle (rad) of rotation R about its body x axis (swing-twist split)."""
    return twist_about(R, (1.0, 0.0, 0.0))


def rotation_angle(R):
    """Angle (rad) of a rotation matrix."""
    return math.acos(max(-1.0, min(1.0, (np.trace(R) - 1) / 2)))


class RotationGate:
    """Rejects one-off orientation spikes (e.g. |roll| or |yaw| glitching > 90 deg).

    A rotation more than jump_deg from the last accepted one is held back;
    it's accepted once `confirm` consecutive frames agree with each other
    (a real fast turn). After reset_s without an accepted frame (hand lost),
    the next frame is accepted as is. jump_deg <= 0 disables the gate.
    """

    def __init__(self, jump_deg=45.0, confirm=3, reset_s=0.3):
        self.jump = math.radians(jump_deg)
        self.confirm, self.reset_s = confirm, reset_s
        self.last = self.last_t = None
        self.pending = []

    def __call__(self, R, t=None):
        """(accepted, angle from the last accepted rotation in rad)."""
        stale = t is not None and self.last_t is not None and t - self.last_t > self.reset_s
        ang = 0.0 if self.last is None else rotation_angle(R @ self.last.T)
        if self.jump <= 0 or self.last is None or stale or ang <= self.jump:
            self.pending = []
        else:
            if self.pending and rotation_angle(R @ self.pending[-1].T) > self.jump:
                self.pending = []
            self.pending.append(R)
            if len(self.pending) < self.confirm:
                return False, ang
            self.pending = []
        self.last, self.last_t = R, t
        return True, ang


def robot_axes(forward_cam, up_cam=(0.0, -1.0, 0.0)):
    """A (rows = robot X, Y, Z in cam coords) from a measured forward direction.

    up defaults to image-up (cameras mounted level). forward is made
    perpendicular to up; robot Y = Z x X (left).
    """
    up = np.asarray(up_cam, dtype=float)
    up = up / np.linalg.norm(up)
    f = np.asarray(forward_cam, dtype=float)
    f = f - up * (f @ up)
    n = np.linalg.norm(f)
    if n < 1e-6:
        raise ValueError("forward direction is (nearly) vertical")
    f = f / n
    return np.vstack([f, np.cross(up, f), up])


class SkewCorrector:
    """Moves cam1's landmarks to cam0's capture time.

    The cameras free-run, so a pair is up to ~17 ms apart, and a moving hand is
    then in different places in the two views: that triangulates mostly into
    depth error (~4 mm at 0.25 m/s). cam1's landmark velocity from its last two
    detections extrapolates them by (t0 - t1). Landmarks pass unchanged without
    a recent previous detection or when the implied speed is implausible.
    """

    def __init__(self, max_gap=0.1, max_speed_px=4000.0):
        self.max_gap, self.max_speed = max_gap, max_speed_px
        self.prev = None  # (landmarks, t1)

    def __call__(self, px1, t1, t0):
        prev = self.prev
        if px1 is None:
            self.prev = None
            return None
        if prev is not None and prev[1] == t1:  # same cam1 frame paired twice
            return px1 + prev[2] * (t0 - t1)
        v = np.zeros_like(px1)
        if prev is not None and 1e-3 < t1 - prev[1] < self.max_gap:
            v = (px1 - prev[0]) / (t1 - prev[1])
            if np.median(np.linalg.norm(v, axis=1)) > self.max_speed:
                v = np.zeros_like(px1)
        self.prev = (px1, t1, v)
        return px1 + v * (t0 - t1)



# ---------- per-frame estimator ----------

class StereoHand:
    """Landmark pixels from both cameras -> validated 3D hand pose in cam0 frame.

    estimate() returns (pose, reason). pose is a dict:
        P     (21,3) points (NaN where unknown: non-palm points in mono mode, and
              in stereo single landmarks dropped for reprojection error > reproj_px;
              a dropped palm point is filled in from the fitted template)
        p, R  palm center and hand rotation (palm_fit() once calibrated with
              frame="fit", else hand_frame())
        fist  bool (stereo only; False in mono mode)
        curl  finger curl, or None (mono, or < 3 fingertips left)
        mode  "stereo" | "mono0" | "mono1"
        err   median reprojection error of the kept points (px), stereo only
    or None with a short reason ("no hand", "reproj 23px", "bones", "fit 14mm",
    "rot jump 120deg", ...).
    Bone/template/fit checks are active after set_hand() (hand calibration).
    """

    def __init__(self, rig, reproj_px=15.0, bone_tol=0.3, mono_fallback=False,
                 frame="fit", fit_mm=15.0, jump_deg=45.0):
        self.K0, self.d0, self.K1, self.d1 = rig["K0"], rig["d0"], rig["K1"], rig["d1"]
        self.R, self.t = rig["R"], rig["t"].ravel()
        self.reproj_px, self.bone_tol, self.mono = reproj_px, bone_tol, mono_fallback
        if frame not in ("fit", "3pt"):
            raise ValueError(f"frame must be 'fit' or '3pt', not {frame!r}")
        self.frame, self.fit_m = frame, fit_mm / 1000
        self.gate = RotationGate(jump_deg)
        self.bones = self.template = None
        self._guess = [None, None]

    def set_hand(self, bones, template):
        self.bones = np.asarray(bones)
        self.template = np.asarray(template)

    def hand_pose(self, P):
        """(p, R, fit rms m) of a stereo hand: palm_fit() once calibrated with
        frame="fit", else hand_frame() (rms 0). p is the palm center either way."""
        if self.template is not None and self.frame == "fit":
            ok = np.isfinite(P[PALM]).all(axis=1)  # a dropped palm point: fit the other 4
            return palm_fit(P[PALM][ok], self.template[ok])
        p, R = hand_frame(P)
        return p, R, 0.0

    def estimate(self, px0, px1, t=None):
        """t: frame time (s), lets the rotation gate reset after the hand was lost."""
        pose, reason = self._estimate(px0, px1)
        if pose is None:
            return None, reason
        ok, ang = self.gate(pose["R"], t)
        if not ok:
            return None, f"rot jump {math.degrees(ang):.0f}deg"
        return pose, ""

    def _drop_bad_points(self, P, e):
        """Per-landmark check instead of all-or-nothing: points whose reprojection
        error is over reproj_px become NaN. One bad palm point is tolerated once
        calibrated with frame="fit" (the fit uses the other 4); otherwise every
        palm point must be good. Returns (P, median error of the good points, reason)."""
        good = e <= self.reproj_px
        err = float(np.median(e[good])) if good.any() else float("inf")
        n_bad_palm = int(np.count_nonzero(~good[PALM]))
        allowed = 1 if self.template is not None and self.frame == "fit" else 0
        if n_bad_palm > allowed or good.sum() < len(good) // 2:
            return P, err, f"reproj {float(np.median(e)):.0f}px ({len(good) - good.sum()} pts)"
        P = P.copy()
        P[~good] = np.nan
        return P, err, ""

    def _check_pose(self, P):
        """(p, R, reason): bone lengths, then the palm fit residual."""
        if self.bones is not None and not bones_ok(P, self.bones, self.bone_tol):
            return None, None, "bones"
        p, R, rms = self.hand_pose(P)
        if rms > self.fit_m:
            return None, None, f"fit {rms * 1000:.0f}mm"
        return p, R, ""

    def _palm_pose(self, P):
        """(P, p, R, reason). If the checks fail with all 5 palm points, retry
        without the one whose removal fits the template best (leave-one-out):
        a landmark that's off along the epipolar line (horizontal here) passes
        the reprojection check but lands at the wrong depth."""
        p, R, reason = self._check_pose(P)
        if (reason and self.template is not None and self.frame == "fit"
                and np.isfinite(P[PALM]).all()):
            best = None
            for i in PALM:
                Q = P.copy()
                Q[i] = np.nan
                rms = self.hand_pose(Q)[2]
                if best is None or rms < best[0]:
                    best = (rms, Q)
            q_p, q_R, q_reason = self._check_pose(best[1])
            if not q_reason:
                return best[1], q_p, q_R, ""
        return P, p, R, reason

    def _fill_palm(self, P, p, R):
        """A dropped palm point is replaced by the fitted template's, so palm-based
        quantities (curl's palm center and length) stay defined."""
        bad = np.isnan(P[PALM]).any(axis=1)
        if bad.any():
            P = P.copy()
            P[np.array(PALM)[bad]] = (self.template @ R.T + p)[bad]
        return P

    def _estimate(self, px0, px1):
        if px0 is not None and px1 is not None:
            n0 = normalize_points(px0, self.K0, self.d0)
            n1 = normalize_points(px1, self.K1, self.d1)
            P = triangulate(n0, n1, self.R, self.t)
            e = reprojection_error(P, n0, n1, self.R, self.t, self.K0[0, 0], self.K1[0, 0])
            P, err, reason = self._drop_bad_points(P, e)
            if not reason:
                P, p, R, reason = self._palm_pose(P)
            if not reason:
                P = self._fill_palm(P, p, R)
                c = curl(P)
                return dict(P=P, p=p, R=R, fist=is_fist(P), curl=None if np.isnan(c) else c,
                            mode="stereo", err=err), ""
            # stereo failed validation: fall through to mono if allowed
        else:
            reason = "no hand" if px0 is None and px1 is None else "one view"
        if not self.mono or self.template is None:
            return None, reason
        for k, (px, K, d, Rc, tc) in enumerate([(px0, self.K0, self.d0, np.eye(3), np.zeros(3)),
                                                 (px1, self.K1, self.d1, self.R, self.t)]):
            if px is None:
                self._guess[k] = None
                continue
            X, self._guess[k] = palm_pnp(px, K, d, self.template, Rc, tc, self._guess[k])
            if X is None:
                continue
            P = np.full((21, 3), np.nan)
            P[PALM] = X
            p, R = hand_frame(P)
            return dict(P=P, p=p, R=R, fist=False, curl=None, mode=f"mono{k}", err=float("nan")), ""
        return None, reason


# ---------- self-calibration of stereo extrinsics from hand landmarks ----------

def stereo_from_correspondences(n0, n1, baseline_m):
    """Relative pose of cam1 from matched normalized points (N,2) each.

    Essential matrix + cheirality; translation scaled to the known baseline.
    Returns (R, t, inlier_fraction).
    """
    n0 = np.asarray(n0, dtype=np.float64)
    n1 = np.asarray(n1, dtype=np.float64)
    E, mask = cv2.findEssentialMat(n0, n1, np.eye(3), method=cv2.RANSAC,
                                   prob=0.999, threshold=2e-3)
    if E is None or E.shape != (3, 3):
        raise RuntimeError("essential matrix estimation failed (move the hand around more)")
    _, R, t, mask2 = cv2.recoverPose(E, n0, n1, np.eye(3), mask=mask)
    inl = float(np.count_nonzero(mask2)) / len(n0)
    return R, t.ravel() / np.linalg.norm(t) * baseline_m, inl


# ---------- self-test ----------

def _synthetic_hand():
    """21 landmarks, open right hand, palm facing -z (toward a camera), meters."""
    P = np.zeros((21, 3))
    mcp_x = [-0.035, -0.012, 0.010, 0.030]  # index..pinky across the palm (x), fingers up (-y)
    P[0] = [0.0, 0.0, 0.0]
    for k, mx in enumerate(mcp_x):
        base = 5 + 4 * k
        P[base] = [mx, -0.085 + abs(mx) * 0.2, 0.0]
        for j in range(1, 4):
            P[base + j] = P[base] + [0, -0.025 * j, 0.004 * j]
    for j in range(1, 5):
        P[j] = [-0.03 - 0.015 * j, -0.02 - 0.012 * j, 0.002 * j]
    return P


def _selftest():
    rng = np.random.default_rng(0)
    # rpy round trip
    for _ in range(100):
        rpy = rng.uniform([-3, -1.5, -3], [3, 1.5, 3])
        assert np.allclose(matrix_to_rpy(rpy_to_matrix(*rpy)), rpy, atol=1e-9)
    # stereo rig: 0.279 m baseline, cam1 to the right and toed in 5 deg
    K = intrinsics_from_fov(640, 480, 65)
    dist = np.zeros(5)
    R_true = rpy_to_matrix(0.0, math.radians(-5), 0.0)
    C1 = np.array([0.279, 0.0, 0.0])        # cam1 center in cam0 frame
    t_true = -R_true @ C1

    def view(X, R, t):
        Xc = X @ R.T + t
        px = Xc[:, :2] / Xc[:, 2:3] * K[0, 0] + K[:2, 2]
        return px + rng.normal(0, 0.5, px.shape)  # 0.5 px noise

    hand0 = _synthetic_hand()
    # many hand poses in front of the cameras -> self-calibration
    c0, c1, poses = [], [], []
    for _ in range(60):
        Rh = rotvec_to_matrix(rng.normal(0, 0.4, 3))
        p = np.array([rng.uniform(-0.05, 0.3), rng.uniform(-0.1, 0.1), rng.uniform(0.4, 0.8)])
        X = hand0 @ Rh.T + p
        poses.append(X)
        c0.append(normalize_points(view(X, np.eye(3), np.zeros(3)), K, dist))
        c1.append(normalize_points(view(X, R_true, t_true), K, dist))
    R_est, t_est, inl = stereo_from_correspondences(np.vstack(c0), np.vstack(c1), 0.279)
    rot_err = math.degrees(np.linalg.norm(matrix_to_rotvec(R_est @ R_true.T)))
    t_err = np.linalg.norm(t_est - t_true)
    print(f"self-calib: rot err {rot_err:.2f} deg, t err {t_err * 1000:.1f} mm, inliers {inl:.0%}")
    assert rot_err < 1.0 and t_err < 0.02

    # triangulation + hand frame + bones with the estimated extrinsics
    X = poses[0]
    n0, n1 = c0[0], c1[0]
    P = triangulate(n0, n1, R_est, t_est)
    err = np.linalg.norm(P - X, axis=1)
    rep = reprojection_error(P, n0, n1, R_est, t_est, K[0, 0], K[0, 0])
    print(f"triangulation: median 3D err {np.median(err) * 1000:.1f} mm, "
          f"median reproj {np.median(rep):.2f} px")
    assert np.median(err) < 0.01
    assert bones_ok(P, bone_lengths(X), 0.15)

    # relative rotation recovered through the hand frame
    _, R_a = hand_frame(X)
    dR = rotvec_to_matrix([0.3, -0.2, 0.1])
    _, R_b = hand_frame((X - palm_center(X)) @ dR.T + palm_center(X))
    assert np.allclose(R_b @ R_a.T, dR, atol=1e-9)

    # palm_fit: exact without noise, same convention as hand_frame
    tmpl = palm_template(X)
    o_f, R_f, rms = palm_fit(X[PALM], tmpl)
    o_3, R_3 = hand_frame(X)
    assert rms < 1e-9 and np.allclose(R_f, R_3, atol=1e-9) and np.allclose(o_f, o_3)
    X_moved = (X - palm_center(X)) @ dR.T + palm_center(X) + [0.1, -0.05, 0.02]
    o_f, R_f, _ = palm_fit(X_moved[PALM], tmpl)
    assert np.allclose(R_f, dR @ R_3, atol=1e-9) and np.allclose(o_f, palm_center(X_moved))

    # palm_fit vs hand_frame under 3 mm landmark noise
    tmpl0 = palm_template(hand0)
    e3, ef = [], []
    for _ in range(300):
        Xt = hand0 @ rotvec_to_matrix(rng.normal(0, 0.5, 3)).T + [0.0, 0.0, 0.5]
        R_t = hand_frame(Xt)[1]
        Xn = Xt + rng.normal(0, 0.003, Xt.shape)
        e3.append(rotation_angle(hand_frame(Xn)[1] @ R_t.T))
        ef.append(rotation_angle(palm_fit(Xn[PALM], tmpl0)[1] @ R_t.T))
    e3, ef = np.degrees(e3), np.degrees(ef)
    print(f"rotation err, 3 mm noise: 3-point median {np.median(e3):.1f} deg "
          f"(p95 {np.percentile(e3, 95):.1f}), palm_fit median {np.median(ef):.1f} deg "
          f"(p95 {np.percentile(ef, 95):.1f})")
    assert np.median(ef) < np.median(e3) and np.percentile(ef, 95) < np.percentile(e3, 95)

    # rotation gate: one-frame spike held back, a sustained turn accepted after 3 frames
    g = RotationGate(45.0, confirm=3, reset_s=0.3)
    Rz = lambda deg: rotvec_to_matrix([0.0, 0.0, math.radians(deg)])  # noqa: E731
    assert g(Rz(0), 0.0)[0] and g(Rz(10), 0.033)[0]
    assert not g(Rz(130), 0.067)[0]                       # spike
    assert g(Rz(12), 0.1)[0]                              # back to normal
    assert not g(Rz(130), 0.133)[0] and not g(Rz(132), 0.167)[0]
    assert g(Rz(134), 0.2)[0]                             # 3rd agreeing frame: real turn
    assert not g(Rz(0), 0.233)[0]
    assert g(Rz(0), 0.6)[0]                               # hand was lost > reset_s: accept
    assert RotationGate(0)(Rz(0))[0] and RotationGate(0)(Rz(170))[0]  # disabled

    # PnP fallback from cam1 only
    px1 = view(X, R_true, t_true)
    X0, _ = palm_pnp(px1, K, dist, tmpl, R_true, t_true)
    e = np.linalg.norm(X0 - X[PALM], axis=1)
    print(f"palm PnP (cam1 only): median err {np.median(e) * 1000:.1f} mm")
    assert np.median(e) < 0.02

    # StereoHand: stereo path, then mono fallback when cam0 loses the hand
    rig = dict(K0=K, d0=dist, K1=K, d1=dist, R=R_est, t=t_est)
    sh = StereoHand(rig, mono_fallback=True)
    sh.set_hand(bone_lengths(X), tmpl)
    px0 = view(X, np.eye(3), np.zeros(3))
    pose, why = sh.estimate(px0, px1)
    assert pose and pose["mode"] == "stereo" and not pose["fist"], why
    sh3 = StereoHand(rig, frame="3pt")
    sh3.set_hand(bone_lengths(X), tmpl)
    pose3, why = sh3.estimate(px0, px1)
    assert pose3, why
    print(f"StereoHand rotation err: fit {math.degrees(rotation_angle(pose['R'] @ R_3.T)):.2f} deg, "
          f"3pt {math.degrees(rotation_angle(pose3['R'] @ R_3.T)):.2f} deg")
    assert rotation_angle(pose["R"] @ R_3.T) < math.radians(5)
    sh_strict = StereoHand(rig, fit_mm=0.01)              # fit gate rejects (noise > 0.01 mm)
    sh_strict.set_hand(bone_lengths(X), tmpl)
    assert sh_strict.estimate(px0, px1)[1].startswith("fit")
    pose_m, why = sh.estimate(None, px1)
    assert pose_m and pose_m["mode"] == "mono1", why
    assert np.linalg.norm(pose_m["p"] - pose["p"]) < 0.03
    assert sh.estimate(None, None) == (None, "no hand")
    shrunk = px0.copy()
    shrunk[PALM] = shrunk[0] + 0.5 * (shrunk[PALM] - shrunk[0])  # inconsistent views
    assert sh.estimate(shrunk, px1)[0]["mode"] == "mono0"  # stereo rejected, PnP on cam0

    # single bad landmarks are dropped, not the frame
    shp = StereoHand(rig)
    shp.set_hand(bone_lengths(X), tmpl)
    one = px1.copy()
    one[RING_MCP] += [0.0, 40.0]                         # one palm point off 40 px across the epipolar line
    pose_1, why = shp.estimate(px0, one)
    assert pose_1 and pose_1["mode"] == "stereo", why
    assert rotation_angle(pose_1["R"] @ R_3.T) < math.radians(5)
    assert np.linalg.norm(pose_1["p"] - pose["p"]) < 0.005 and np.isfinite(pose_1["P"][PALM]).all()
    tip = px1.copy()
    tip[FINGERTIPS[0]] += [0.0, 60.0]                    # one fingertip off: curl from the other 3
    pose_t, why = shp.estimate(px0, tip)
    assert pose_t and abs(pose_t["curl"] - pose["curl"]) < 0.1, why
    two = one.copy()
    two[PINKY_MCP] += [0.0, 40.0]                        # two palm points off: rejected
    assert shp.estimate(px0, two)[1].startswith("reproj")
    along = px1.copy()
    along[RING_MCP] += [25.0, 0.0]                       # along the epipolar line: reprojection can't
    assert StereoHand(rig).estimate(px0, along)[0] is not None  # (uncalibrated: no checks to fail)
    pose_a, why = shp.estimate(px0, along)               # ...bones/fit catch it, leave-one-out fixes it
    assert pose_a and rotation_angle(pose_a["R"] @ R_3.T) < math.radians(5), why
    assert np.linalg.norm(pose_a["p"] - pose["p"]) < 0.005
    print(f"dropped landmark: 1 bad palm point -> rotation err "
          f"{math.degrees(rotation_angle(pose_1['R'] @ R_3.T)):.2f} deg, "
          f"position {np.linalg.norm(pose_1['p'] - pose['p']) * 1000:.1f} mm")

    # cam1 skew correction: a hand moving at constant pixel speed lands on cam0's time
    sk = SkewCorrector()
    v = np.array([300.0, -100.0])                        # px/s
    for k in range(3):
        t1 = k / 30
        out = sk(px1 + v * t1, t1, t1 + 0.012)
    assert np.allclose(out, px1 + v * (2 / 30 + 0.012))
    assert sk(None, 0.1, 0.11) is None and np.allclose(sk(px1, 0.2, 0.21), px1)  # reset after loss

    # fist detection
    assert not is_fist(hand0)
    fist = hand0.copy()
    c = palm_center(hand0)
    fist[FINGERTIPS] = c + (fist[FINGERTIPS] - c) * 0.3
    assert is_fist(fist)

    # robot axes: forward along cam +z (hand moves away from cameras), level cams
    A = robot_axes([0.0, 0.1, 1.0])
    assert np.allclose(A @ [0, 0, 1], [1, 0, 0], atol=0.01)   # cam z -> robot +X
    assert np.allclose(A @ [0, -1, 0], [0, 0, 1])             # image up -> robot +Z
    assert np.allclose(A @ [-1, 0, 0], [0, 1, 0], atol=1e-9)  # cam -x -> robot +Y (left)
    assert np.isclose(np.linalg.det(A), 1.0)
    print("hand3d self-test OK")


if __name__ == "__main__":
    _selftest()
