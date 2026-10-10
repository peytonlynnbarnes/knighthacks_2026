#!/home/peyton/kh/.venv/bin/python
"""One-time stereo calibration -> Computer-Vision/calib/stereo.npz.

Redo it whenever a camera is bumped or re-aimed, and to change resolution.

Resolution: --size (default 640x480, a 4:3 center crop, ~38 deg; 1280x720 is
the full ~50 deg view but cam0 only manages ~5 fps at it in normal light,
USB bandwidth through usbipd, so check the fps the startup line prints). The
rig file records the size and hand_target_node.py opens the cameras at it. Focus: find the sharpest value for a hand at working
distance with stereo_cam.py (f key), then calibrate with --focus N. It's
locked during calibration and saved in the rig, and the node locks the same
value (autofocus changes the focal length the calibration measured).

Mode 1 (default, no equipment): calibrate from your hand.
    .venv/bin/python Computer-Vision/stereo_calib.py [--hfov 38]   (default 38 at 640x480, 50 at 1280x720)
    Move an open hand slowly all around the shared view for ~25 s: left/right,
    up/down, near/far (30-90 cm), and tilt it. The essential matrix from those
    landmark matches gives cam1's rotation and direction; the measured
    baseline (--baseline-in, default 9.5 in) gives the scale. Intrinsics are a pinhole guess from --hfov (no lens
    distortion), so absolute depth can be off by the hfov error; relative
    motion (what the arm uses, 1:1) is what matters.

Mode 2 (more accurate): a chessboard shown on a phone/tablet screen.
    .venv/bin/python Computer-Vision/stereo_calib.py --board
        -> writes calib/chessboard.png (9x6 inner corners = 10x7 squares) and exits.
    Show it full screen on the phone at max brightness, measure the width of
    all 10 squares in a row with a ruler and divide by 10, then:
    .venv/bin/python Computer-Vision/stereo_calib.py --board --square-mm 9.8
    Hold the phone still for each grab (blur kills detection). Two kinds of views:
      1. per camera (15 each, 6 of them TILTED 20-40 deg): hold the phone close
         (15-40 cm) to one camera so it fills much of that view; corners of the
         frame too. These fix the lens (focal length + distortion). Tilt matters:
         flat-on views can't tell a narrow lens from a far board.
      2. pairs (12): where BOTH cameras see it (farther away), varied spots/tilts.
    Grabs are automatic once the board has moved since the last one and is then
    held still for a moment ("HOLD STILL" on screen until it is; space = grab now).
    The window shows each camera's count and the current tilt.
    Sanity gates (hfov 30-100 deg, rms, baseline within 7% of --baseline-in, both hfovs agree,
    principal point near the center) refuse to save a bad result; corners are saved to
    calib/board_corners.npz, re-solve with --from-corners (no recapture). The baseline
    comes out of the calibration; it's compared with --baseline-in.
    Solving drops outlier views (per-view error > 2.5x the median and > 1 px; blurred
    or mis-detected grabs) and re-solves, per camera and for the pairs; the
    principal point is free (--fix-center to pin it to the image center).

Both modes print a sanity check: baseline, rotation between the cameras,
reprojection error, and your palm size.
"""
import argparse
import math
import os
import sys
import time

import cv2
import numpy as np

import hand3d as h3
from stereo_cam import (BASELINE_M, HERE, RIG_PATH, StereoCam, StereoLandmarks,
                        draw_hand, fit_view, save_rig)

BOARD = (9, 6)  # inner corners


def _show(f0, f1, lines):
    view = np.hstack([f0, f1])
    for i, s in enumerate(lines):
        cv2.putText(view, s, (10, 25 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    cv2.imshow("stereo calibration (q aborts)", fit_view(view))
    return cv2.waitKey(1) & 0xFF


def report(R, t, K0, d0, K1, d1, samples):
    """Print rig sanity numbers; samples = list of (px0, px1) hand landmark pairs."""
    rot = math.degrees(np.linalg.norm(h3.matrix_to_rotvec(R)))
    C1 = -R.T @ t  # cam1 center in cam0 frame
    side = "right" if C1[0] > 0 else "left"
    print(f"\ncam1 is {np.linalg.norm(C1) * 100:.1f} cm from cam0, to its {side} "
          f"(direction {np.round(C1 / np.linalg.norm(C1), 2)}; expect ~[±1, 0, 0]), "
          f"rotated {rot:.1f} deg relative to it")
    if not samples:
        return
    errs, widths, depths = [], [], []
    for a, b in samples:
        n0, n1 = h3.normalize_points(a, K0, d0), h3.normalize_points(b, K1, d1)
        P = h3.triangulate(n0, n1, R, t)
        errs.append(np.median(h3.reprojection_error(P, n0, n1, R, t, K0[0, 0], K1[0, 0])))
        widths.append(np.linalg.norm(P[5] - P[17]))
        depths.append(h3.palm_center(P)[2])
    print(f"hand: median reprojection {np.median(errs):.1f} px, "
          f"index-pinky MCP width {np.median(widths) * 100:.1f} cm (typical adult 6.5-9 cm), "
          f"depth range {min(depths) * 100:.0f}-{max(depths) * 100:.0f} cm")
    print("If the width is far off, the --hfov guess is wrong (depth scales with it).")


def calib_hand(cam, args):
    K = h3.intrinsics_from_fov(cam.size[0], cam.size[1], args.hfov)
    dist = np.zeros(5)
    lms = StereoLandmarks()
    samples, last = [], None
    t_end = None
    try:
        while True:
            pr = cam.pair()
            if pr is None:
                continue
            f0, f1, t = pr
            a, b = lms(f0, f1, t)
            draw_hand(f0, a)
            draw_hand(f1, b)
            if a is not None and b is not None:
                if t_end is None:
                    t_end = time.monotonic() + args.seconds
                # only keep frames where the hand moved, for diverse geometry
                if last is None or np.median(np.linalg.norm(a - last, axis=1)) > 12:
                    samples.append((a, b))
                    last = a
            left = 0 if t_end is None else max(0.0, t_end - time.monotonic())
            k = _show(f0, f1, [f"move your open hand around, near/far, tilt it",
                               f"samples {len(samples)}  "
                               + (f"{left:.0f} s left" if t_end else "waiting for hand in both views")])
            if k in (ord("q"), 27):
                sys.exit("aborted")
            if t_end is not None and left <= 0:
                break
    finally:
        lms.close()
        cv2.destroyAllWindows()
    if len(samples) < 40:
        sys.exit(f"only {len(samples)} samples; move the hand more and keep it in both views")
    n0 = np.vstack([h3.normalize_points(a, K, dist) for a, _ in samples])
    n1 = np.vstack([h3.normalize_points(b, K, dist) for _, b in samples])
    R, t, inl = h3.stereo_from_correspondences(n0, n1, args.baseline_in * 0.0254)
    print(f"{len(samples)} hand samples, {len(n0)} point matches, {inl:.0%} RANSAC inliers")
    if inl < 0.5:
        print("WARNING: few inliers; the result may be poor. Retry with slower, wider motion.")
    report(R, t, K, dist, K, dist, samples[::max(1, len(samples) // 50)])
    return K, dist, K, dist, R, t, "hand"


BOARD_PNG = os.path.join(HERE, "calib", "chessboard.png")


def write_board_png():
    """10x7 squares with a 1-square white margin (needed for detection)."""
    os.makedirs(os.path.dirname(BOARD_PNG), exist_ok=True)
    sq = 120
    img = np.full(((BOARD[1] + 3) * sq, (BOARD[0] + 3) * sq), 255, np.uint8)
    for r in range(BOARD[1] + 1):
        for c in range(BOARD[0] + 1):
            if (r + c) % 2 == 0:
                img[(r + 1) * sq:(r + 2) * sq, (c + 1) * sq:(c + 2) * sq] = 0
    cv2.imwrite(BOARD_PNG, img)


TILT_DEG = 20.0  # a view counts as "tilted" above this (tilt pins down focal length)
STILL_PX = 1.0  # board counts as still when its corners move less than this per frame...
STILL_FRAMES = 4  # ...for this many frames in a row (~130 ms at 30 fps)
CORNERS_PATH = os.path.join(HERE, "calib", "board_corners.npz")


def board_points(square_mm):
    obj = np.zeros((BOARD[0] * BOARD[1], 3), np.float32)
    obj[:, :2] = np.mgrid[0:BOARD[0], 0:BOARD[1]].T.reshape(-1, 2) * square_mm / 1000
    return obj


def board_tilt(c, obj, K):
    """Angle (deg) between the board normal and the optical axis (rough, from a K guess)."""
    ok, rvec, _ = cv2.solvePnP(obj, c, K, None)
    if not ok:
        return 0.0
    n = h3.rotvec_to_matrix(rvec)[:, 2]
    return math.degrees(math.acos(min(1.0, abs(n[2]))))


def capture_board(cam, args):
    """Single-camera views (intrinsics) + both-camera pairs (extrinsics) of the board."""
    obj = board_points(args.square_mm)
    Kg = h3.intrinsics_from_fov(cam.size[0], cam.size[1], args.hfov)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    views, tilts, pairs = ([], []), ([], []), []
    last = [(0.0, None), (0.0, None), (0.0, None)]  # per cam0, cam1, pair: (time, corners)
    prev, still_n = [None, None], [0, 0]  # per camera: last frame's corners, frames held still

    def fresh(slot, c, now, key, still):
        t0, c0 = last[slot]
        moved = c0 is None or np.median(np.linalg.norm(c - c0, axis=2)) > 25
        return key == ord(" ") or (still and now - t0 > 1.0 and moved)

    def done_cam(k):
        return len(views[k]) >= args.views and sum(a >= TILT_DEG for a in tilts[k]) >= args.tilted

    try:
        while not (done_cam(0) and done_cam(1) and len(pairs) >= args.pairs):
            pr = cam.pair()
            if pr is None:
                continue
            f0, f1, _ = pr
            found, tilt = [], []
            for f in (f0, f1):
                g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
                ok, c = cv2.findChessboardCorners(g, BOARD, flags)
                if ok:
                    c = cv2.cornerSubPix(g, c, (5, 5), (-1, -1), crit)
                    cv2.drawChessboardCorners(f, BOARD, c, ok)
                found.append(c if ok else None)
                tilt.append(board_tilt(c, obj, Kg) if ok else None)
            # only grab a held-still board: motion blur and rolling shutter bend the
            # grid by pixels, and the two cameras' frames can be up to 50 ms apart
            for k in (0, 1):
                c = found[k]
                if c is not None and prev[k] is not None \
                        and np.median(np.linalg.norm(c - prev[k], axis=2)) < STILL_PX:
                    still_n[k] += 1
                else:
                    still_n[k] = 0
                prev[k] = c
            still = [n >= STILL_FRAMES for n in still_n]
            status = []
            for k in (0, 1):
                nt = sum(a >= TILT_DEG for a in tilts[k])
                status.append(f"cam{k}: {len(views[k])}/{args.views} views, {nt}/{args.tilted} tilted"
                              + (f"  (now {tilt[k]:.0f} deg{'' if still[k] else ', HOLD STILL'})"
                                 if tilt[k] is not None else ""))
            hint = ("fill each camera's view up close; TILT the phone 20-40 deg"
                    if not (done_cam(0) and done_cam(1))
                    else "now hold it where BOTH cameras see it, vary position/tilt")
            key = _show(f0, f1, [hint] + status + [f"pairs (both cams): {len(pairs)}/{args.pairs}",
                                                   "space = grab now, q = abort"])
            if key in (ord("q"), 27):
                sys.exit("aborted")
            now = time.monotonic()
            for k in (0, 1):
                c = found[k]
                if c is None or not fresh(k, c, now, key, still[k]):
                    continue
                # once the view count is reached, only take tilted views (if still short)
                if len(views[k]) >= args.views and (done_cam(k) or tilt[k] < TILT_DEG):
                    continue
                views[k].append(c)
                tilts[k].append(tilt[k])
                last[k] = (now, c)
            if found[0] is not None and found[1] is not None and len(pairs) < args.pairs \
                    and fresh(2, found[0], now, key, still[0] and still[1]):
                pairs.append((found[0], found[1]))
                last[2] = (now, found[0])
                print(f"pair {len(pairs)}")
    finally:
        cv2.destroyAllWindows()
    os.makedirs(os.path.dirname(CORNERS_PATH), exist_ok=True)
    np.savez(CORNERS_PATH, v0=np.array(views[0]), v1=np.array(views[1]),
             p0=np.array([p[0] for p in pairs]), p1=np.array([p[1] for p in pairs]),
             square_mm=args.square_mm, size=np.array(cam.size),
             focus=-1 if args.focus is None else args.focus)
    print(f"saved corners to {CORNERS_PATH} (re-solve with --from-corners)")
    return [np.array(v) for v in views], pairs, cam.size


OUTLIER_X, OUTLIER_PX = 2.5, 1.0  # a view is an outlier above max(2.5 x median, 1 px)


def _outliers(errs):
    errs = np.asarray(errs)
    return errs > max(OUTLIER_X * np.median(errs), OUTLIER_PX)


def solve_board(views, pairs, size, args):
    """Constrained intrinsics per camera, then stereo extrinsics, each re-solved
    without outlier views. Returns (rig..., problems)."""
    obj = board_points(args.square_mm)
    size = tuple(int(v) for v in size)
    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6)
    # few free parameters: focal (one, square pixels), center, k1/k2 only
    cflags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO |
              cv2.CALIB_ZERO_TANGENT_DIST | cv2.CALIB_FIX_K3)
    if args.fix_center:
        cflags |= cv2.CALIB_FIX_PRINCIPAL_POINT
    problems, Ks, fovs = [], [], []
    for k in (0, 1):
        v = list(views[k]) + [p[k] for p in pairs]
        n0 = len(v)
        for _ in range(3):  # solve, drop outlier views, re-solve
            K = h3.intrinsics_from_fov(size[0], size[1], args.hfov)
            rms, K, d, _, _, _, _, per = cv2.calibrateCameraExtended(
                [obj] * len(v), v, size, K, np.zeros(5), flags=cflags, criteria=crit)
            bad = _outliers(per.ravel())
            if not bad.any() or len(v) - bad.sum() < 10:
                break
            v = [x for x, b in zip(v, bad) if not b]
        hfov = math.degrees(2 * math.atan(size[0] / 2 / K[0, 0]))
        fovs.append(hfov)
        print(f"cam{k}: {len(v)} views ({n0 - len(v)} outliers dropped), rms {rms:.2f} px, "
              f"fx {K[0, 0]:.0f} (hfov {hfov:.0f} deg), center {K[0, 2]:.0f},{K[1, 2]:.0f}, "
              f"k1 {d.ravel()[0]:+.3f} k2 {d.ravel()[1]:+.3f}")
        off = np.abs(K[:2, 2] - np.array(size) / 2) / np.array(size)
        if off.max() > 0.12:
            problems.append(f"cam{k} principal point {K[0, 2]:.0f},{K[1, 2]:.0f} is far from the "
                            "center (more varied views, or --fix-center)")
        if not 30 <= hfov <= 100:  # Cybertrack H4 at 640x480 measures ~38 deg (fov_check.py)
            problems.append(f"cam{k} hfov {hfov:.0f} deg is implausible for a webcam "
                            "(need more tilted, close-up views)")
        if rms > 1.5:
            problems.append(f"cam{k} rms {rms:.2f} px > 1.5 (blur? wrong board size?)")
        Ks += [K, d]
    if abs(fovs[0] - fovs[1]) > 6:
        problems.append(f"same-model cameras disagree: hfov {fovs[0]:.0f} vs {fovs[1]:.0f} deg "
                        "(need more tilted, close-up views)")
    pairs, n0 = list(pairs), len(pairs)
    for _ in range(3):
        res = cv2.stereoCalibrateExtended(
            [obj] * len(pairs), [p[0] for p in pairs], [p[1] for p in pairs],
            Ks[0], Ks[1], Ks[2], Ks[3], size, np.eye(3), np.zeros((3, 1)),
            flags=cv2.CALIB_FIX_INTRINSIC, criteria=crit)
        rms, K0, d0, K1, d1, R, t, per = *res[:7], res[-1]
        bad = _outliers(per.max(axis=1))
        if not bad.any() or len(pairs) - bad.sum() < 8:
            break
        pairs = [x for x, b in zip(pairs, bad) if not b]
    t = t.ravel()
    b, b_meas = np.linalg.norm(t), args.baseline_in * 0.0254
    print(f"stereo: {len(pairs)} pairs ({n0 - len(pairs)} outliers dropped), rms {rms:.2f} px "
          f"(worst pair {per.max():.2f}); baseline {b * 100:.1f} cm (you measured {b_meas * 100:.1f} cm)")
    if rms > 2.0:
        problems.append(f"stereo rms {rms:.2f} px > 2 (cameras not synced while moving? hold still)")
    if abs(b / b_meas - 1) > 0.07:
        problems.append(f"baseline {b * 100:.1f} cm vs measured {b_meas * 100:.1f} cm (>7% off: "
                        "check --square-mm and --baseline-in)")
    report(R, t, K0, d0, K1, d1, [])
    return (K0, d0, K1, d1, R, t, "board"), problems


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cams", type=int, nargs=2, default=[0, 2])
    ap.add_argument("--size", type=int, nargs=2, default=[640, 480], metavar=("W", "H"),
                    help="capture size; the node uses whatever the rig was calibrated at")
    ap.add_argument("--focus", type=int, default=None,
                    help="lock focus 0..1023 (from stereo_cam.py's f sweep) and save it in the "
                         "rig (default autofocus, not recommended)")
    ap.add_argument("--fix-center", action="store_true",
                    help="board mode: pin the principal point to the image center")
    ap.add_argument("--hfov", type=float, default=None,
                    help="horizontal field of view guess (deg): hand mode intrinsics, board mode start")
    ap.add_argument("--seconds", type=float, default=25.0, help="hand mode: capture time")
    ap.add_argument("--board", action="store_true", help="use a chessboard on a screen")
    ap.add_argument("--square-mm", type=float, help="board mode: measured square size")
    ap.add_argument("--views", type=int, default=15, help="board mode: single-camera views per camera")
    ap.add_argument("--tilted", type=int, default=6, help=f"board mode: of those, tilted > {TILT_DEG:.0f} deg")
    ap.add_argument("--pairs", type=int, default=12, help="board mode: views seen by both cameras")
    ap.add_argument("--from-corners", action="store_true",
                    help="board mode: re-solve from the saved corners, no cameras")
    ap.add_argument("--force", action="store_true", help="save even if sanity checks fail")
    ap.add_argument("--baseline-in", type=float, default=BASELINE_M / 0.0254,
                    help="camera separation in inches (hand mode scale; board mode check)")
    ap.add_argument("--out", default=RIG_PATH)
    args = ap.parse_args()
    if args.hfov is None:  # measured: ~38 deg at 640x480 (center crop), ~50 at 1280x720
        args.hfov = 38.0 if args.size[0] <= 640 else 50.0
    if args.board and not args.square_mm:
        write_board_png()
        sys.exit(f"wrote {BOARD_PNG}\nShow it full screen on your phone, measure 10 squares "
                 "in a row (mm) / 10, then rerun with --board --square-mm <that>")
    problems = []
    if args.board and args.from_corners:
        z = np.load(CORNERS_PATH)
        if float(z["square_mm"]) != args.square_mm:
            print(f"note: corners were captured with --square-mm {float(z['square_mm'])}")
        res, problems = solve_board([z["v0"], z["v1"]], list(zip(z["p0"], z["p1"])), z["size"], args)
        size = tuple(int(v) for v in z["size"])
        if "focus" in z.files and args.focus is None and int(z["focus"]) >= 0:
            args.focus = int(z["focus"])
    else:
        cam = StereoCam(args.cams, size=args.size)
        size = cam.size
        print("cameras:", cam.configure(focus=args.focus))
        try:
            if args.board:
                views, pairs, _ = capture_board(cam, args)
                res, problems = solve_board(views, pairs, size, args)
            else:
                res = calib_hand(cam, args)
        finally:
            cam.close()
    if problems:
        print("\nPROBLEMS:\n  " + "\n  ".join(problems))
        if not args.force:
            sys.exit(f"not saved (fix and rerun; or --force). Re-solve without recapturing: "
                     f"--board --square-mm {args.square_mm} --from-corners")
    K0, d0, K1, d1, R, t, method = res
    save_rig(args.out, K0, d0, K1, d1, R, t, size, method, focus=args.focus)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
