#!/usr/bin/env python3
"""Stereo hand tracking -> /ball_position [x y z roll pitch yaw] for the SO-101 IK,
plus /gripper (0 open .. 1 closed) from how far the fingers are curled.

Pipeline (see ~/kh/INTEGRATION.md for the /ball_position contract):
    2 cameras (macOS AVFoundation, --cams LEFT RIGHT; resolution from
    calib/stereo.npz, auto exposure and white balance, manual focus rings) ->
    pair nearest frames -> 21 landmarks per view (WiLoR's MANO hand model on the
    Apple GPU, see ../WILOR_PLAN.md) -> cam1 landmarks moved to cam0's time
    -> triangulate -> drop single bad landmarks, validate (reprojection,
    palm bone lengths, template fit; one bad palm point is tolerated) -> hand frame
    -> relative to the calibrated neutral hand -> robot axes, x scale
    -> position filter (cam frame, depth smoothed harder: still/moving Kalman
    pair by default, --pos-filter kalman / euro for one model / the old
    one-euro), offset box clamp, speed limit -> /ball_position

Startup: a ~30 s hand calibration (hand_calib.py; instructions in the
window; ~10 s with --no-range-calib, which skips the roll/tilt range steps), every run: wherever the hand is held then becomes the origin, so
control is consistent from session to session. Needs calib/stereo.npz from
stereo_calib.py.

Run (the conda env with ROS 2 Jazzy + torch, see README.md):
    conda activate ros2_jazzy
    python Computer-Vision/common/hand_target_node.py --cams 1 2 --no-ros   # print only
    python Computer-Vision/common/hand_target_node.py --cams 1 2            # publish
    # IK (no Gazebo needed):
    ros2 run so101_mirror ik_node.py

Controls:
    calibration hand = origin: arm at home (the mid-workspace pose, see
    INTEGRATION.md; --neutral-offset shifts it) with the gripper level (rpy 0).
    Hand motion from there moves the arm 1:1 (--scale 1.0). Orientation is
    amplified so a hand turn that stays trackable (palm not edge-on to the
    cameras) covers the gripper's range: pitch -85 (up) .. +85 (down), roll
    one way only, 0 .. +90 (a clockwise quarter turn of the hand; turning the
    other way stays at 0). The gains come from the calibration's range steps
    (roll clockwise 90 deg, tilt each way as far as it tracks: 85% of that
    reaches the gripper's limit, gains 0.7..3); a direction it didn't measure
    uses ROLL_GAIN / PITCH_GAIN, and --roll-gain / --pitch-gain override. A 3 deg soft deadband (--rot-deadband) keeps a still
    hand level; the rotation is filtered before the gain (--rot-filter) and
    held while the palm is edge-on. The status line shows hand vs gripper
    angles and the palm's angle to the cameras.
    Close the hand = close the gripper (proportional: flat open hand = fully
    open, fist = closed). "Fully open" is the flat-hand curl measured in the
    calibration's neutral step (x GRIP_OPEN_MARGIN, so a flat hand reliably
    reaches 0); override with --grip-open-curl. --grip-closed-curl sets the
    fist end (the printout shows the current curl).
    Hide the hand = clutch: nothing is published, the arm and gripper hold
    (IK after 0.5 s). When the hand comes back after >= --reanchor-s it
    re-anchors the position: the arm continues from where it is, no jump.
    Orientation is not re-anchored (it would drift with every clutch): it
    always follows the hand relative to the calibration pose, so a hand that
    comes back turned rotates the gripper back at --max-ang. Shorter
    dropouts (bad frames) don't re-anchor.
    Keys: q/Esc quit, c recalibrate the hand.

Filter testing (Phase 0, see filter_eval.py):
    --record rec.npz   save the raw landmarks + capture times of every frame
                       (start recording from launch: replay redoes the hand
                       calibration from the recorded frames)
    --replay rec.npz   no cameras / no GPU: run the recording through the
                       same estimator + mapper (pass the same --frame /
                       --no-range-calib). With --no-ros it runs as fast as it
                       can; publishing, it plays back in real time. The
                       file records its --backend (only WiLoR recordings
                       replay on this branch).
    --record-frames D  save both camera streams instead (JPEG, common/frames.py)
    --replay-frames D  run them through the backend (identical video, e.g.
                       to compare settings); --replay-live drops the pairs a
                       live run would have missed
    --log out.npz     per frame: hand pose, unfiltered target, published
                       target, grip (live or replayed) -> filter_eval.py

Safety: the target starts at home (all zeros, where feetech_mirror --live
homes) and moves at most --max-lin m/s and --max-ang deg/s, then the IK
(interpolation, workspace box) and feetech_mirror (rate limit, clamps) still apply.
"""
import argparse
import json
import math
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np

import hand3d as h3
from frames import FrameRecorder, FrameReplay, ReplayEnd
from hand_calib import HandCalibrator, LeftHandCalibrator, hand_path, save_hand
from hand_mapping import (BOX_MIN, IMM_PRESETS, KALMAN_DEFAULTS, GripFilter, HandMapper, RollFilter, RotationShaper,
                          StereoIMM, StereoKalman, StillHold, facing_weight, gains_from_range)
from stereo_cam import (BACKENDS, DEFAULT_CAMS, RIG_PATH, StereoCam, draw_hand, fit_view, load_rig,
                        make_landmarks)

BALL_TOPIC = "/ball_position"  # must match so101_mirror.BALL_TOPIC
GRIP_TOPIC = "/gripper"        # std_msgs/Float64, 0 = open .. 1 = closed (feetech_mirror --gripper)
WRIST_ROLL_TOPIC = "/wrist_roll_cmd"  # std_msgs/Float64 rad, two-hand mode; must match so101_mirror
LEFT_COLOR = (0, 140, 255)     # BGR orange: the left hand in the preview (two-hand mode)
GRIP_TAU = 0.04                # s, time constant of the grip smoothing (was 0.5 per frame ~ 25 fps)
ROLL_GAIN = (1.5, 1.5)         # (neg, pos) when the range calibration didn't measure the roll
PITCH_GAIN = (1.0, 2.0)        # (up, down) likewise
GRIP_OPEN_CURL = 1.35          # fully-open curl when the hand calibration has none
GRIP_OPEN_MARGIN = 0.9         # calibrated flat-hand curl x this = fully open


class Publisher:
    def __init__(self, enabled):
        self.node = None
        if not enabled:
            return
        import rclpy
        from std_msgs.msg import Float64, Float64MultiArray
        rclpy.init()
        self.rclpy, self.Msg, self.F64 = rclpy, Float64MultiArray, Float64
        self.node = rclpy.create_node("hand_target_node")
        self.pub = self.node.create_publisher(Float64MultiArray, BALL_TOPIC, 10)
        self.grip_pub = self.node.create_publisher(Float64, GRIP_TOPIC, 10)
        self.roll_pub = self.node.create_publisher(Float64, WRIST_ROLL_TOPIC, 10)

    def __call__(self, target):
        if self.node is not None:
            self.pub.publish(self.Msg(data=[float(v) for v in target]))

    def grip(self, v):
        if self.node is not None:
            self.grip_pub.publish(self.F64(data=float(v)))

    def roll(self, v):
        if self.node is not None:
            self.roll_pub.publish(self.F64(data=float(v)))

    def close(self):
        if self.node is not None:
            self.node.destroy_node()
            self.rclpy.try_shutdown()


class Jitter:
    """Raw (pre-filter) pose spread over the last window_s: position RMS (mm) and
    rotation RMS (deg) about the window mean. Hold the hand still to read noise."""

    def __init__(self, window_s=2.0):
        self.window_s, self.buf = window_s, deque()

    def add(self, t, p, R):
        self.buf.append((t, p, R))
        while t - self.buf[0][0] > self.window_s:
            self.buf.popleft()

    def clear(self):
        self.buf.clear()

    def read(self):
        """(pos mm, rot deg), or None until the window is mostly full."""
        if len(self.buf) < 5 or self.buf[-1][0] - self.buf[0][0] < 0.8 * self.window_s:
            return None
        ps = np.array([b[1] for b in self.buf])
        Rm = h3.project_to_so3(np.mean([b[2] for b in self.buf], axis=0))
        pos = np.sqrt(np.mean(np.sum((ps - ps.mean(axis=0)) ** 2, axis=1)))
        rot = np.sqrt(np.mean([h3.rotation_angle(b[2] @ Rm.T) ** 2 for b in self.buf]))
        return pos * 1000, math.degrees(rot)


NAN_PX = np.full((21, 2), np.nan)


class Replay:
    """A --record file in place of StereoCam + the landmark backend: pair() and
    __call__() hand back the recorded capture times and landmark pixels."""

    def __init__(self, path, realtime):
        z = np.load(path)
        self.t, self.t0s, self.t1s = z["t"], z["t0"], z["t1"]
        self.px0, self.px1 = z["px0"], z["px1"]
        self.recal = set(int(i) for i in z["recal"])
        self.argv = str(z["argv"])
        self.backend = str(z["backend"]) if "backend" in z.files else "mediapipe"  # older files
        self.realtime, self.i, self.start = realtime, -1, None
        self.t0 = self.t1 = 0.0

    def pair(self):
        self.i += 1
        if self.i >= len(self.t):
            raise ReplayEnd
        t = float(self.t[self.i])
        if self.realtime:
            if self.start is None:
                self.start = time.monotonic() - t
            time.sleep(max(0.0, self.start + t - time.monotonic()))
        self.t0, self.t1 = float(self.t0s[self.i]), float(self.t1s[self.i])
        return None, None, t

    def __call__(self, f0, f1, t):
        px = lambda a: None if np.isnan(a[0, 0]) else a.copy()  # noqa: E731
        return px(self.px0[self.i]), px(self.px1[self.i])

    def recalibrate_now(self):
        return self.i in self.recal

    def close(self):
        pass


class Recorder:
    """--record: raw landmarks + capture times per frame, plus when 'c' was pressed."""

    def __init__(self, path, backend):
        self.path, self.rows, self.recal, self.backend = path, [], [], backend

    def add(self, t, t0, t1, px0, px1):
        self.rows.append((t, t0, t1, NAN_PX if px0 is None else px0, NAN_PX if px1 is None else px1))

    def mark_recal(self):
        self.recal.append(len(self.rows) - 1)

    def save(self):
        if not self.rows:
            return
        t, t0, t1, px0, px1 = (np.array(c, dtype=float) for c in zip(*self.rows))
        np.savez_compressed(self.path, t=t, t0=t0, t1=t1, px0=px0, px1=px1,
                            recal=np.array(self.recal, dtype=int), argv=" ".join(sys.argv[1:]),
                            backend=self.backend)
        print(f"recorded {len(t)} frames ({t[-1] - t[0]:.0f} s) -> {self.path}")


class FrameLog:
    """--log: per frame t, state (0 calibrating, 1 tracking, 2 no hand), hand p (cam),
    hand rpy (cam), unfiltered target, published target (NaN if none), grip,
    reprojection error (px, NaN in mono) and mono (1 = single-camera pose)."""

    def __init__(self, path, args):
        self.path, self.rows, self.args = path, [], vars(args)

    def add(self, t, state, pose=None, raw=None, out=None, grip=None):
        nan3, nan6 = [np.nan] * 3, [np.nan] * 6
        self.rows.append([t, state]
                         + (list(pose["p"]) if pose is not None else nan3)
                         + (list(h3.matrix_to_rpy(pose["R"])) if pose is not None else nan3)
                         + (list(raw) if raw is not None else nan6)
                         + (list(out) if out is not None else nan6)
                         + [np.nan if grip is None else grip]
                         + ([pose.get("err", np.nan), float(pose["mode"] != "stereo")]
                            if pose is not None else [np.nan, np.nan]))

    def save(self):
        if not self.rows:
            return
        a = np.array(self.rows, dtype=float)
        np.savez_compressed(self.path, t=a[:, 0], state=a[:, 1], hand_p=a[:, 2:5],
                            hand_rpy=a[:, 5:8], raw=a[:, 8:14], out=a[:, 14:20], grip=a[:, 20],
                            err=a[:, 21], mono=a[:, 22],
                            args=json.dumps(self.args, default=str))
        print(f"logged {len(a)} frames -> {self.path}")


def draw_axes(img, pose, K, d, scale=0.05):
    """Hand frame axes (x red, y green, z blue) projected into cam0."""
    o = pose["p"]
    pts = np.array([o, o + scale * pose["R"][:, 0], o + scale * pose["R"][:, 1],
                    o + scale * pose["R"][:, 2]])
    if np.any(pts[:, 2] <= 0):
        return
    px, _ = cv2.projectPoints(pts, np.zeros(3), np.zeros(3), K, d)
    px = [tuple(int(v) for v in p) for p in px.reshape(-1, 2)]
    for k, c in zip(range(1, 4), [(0, 0, 255), (0, 255, 0), (255, 0, 0)]):
        cv2.line(img, px[0], px[k], c, 3)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cams", type=int, nargs=2, default=DEFAULT_CAMS, metavar=("LEFT", "RIGHT"),
                    help="AVFoundation indices: cam0 = left camera, cam1 = right, as calibrated "
                         "(find them with stereo_cam.py --list)")
    ap.add_argument("--rig", default=RIG_PATH)
    ap.add_argument("--no-ros", action="store_true", help="don't publish, just print")
    ap.add_argument("--backend", choices=BACKENDS, default="wilor",
                    help="landmarks: wilor (MANO model, Apple GPU); see Computer-Vision/WILOR_PLAN.md")
    ap.add_argument("--wilor-hand", choices=["any", "right", "left"], default="any",
                    help="wilor: track only this hand (default: whichever it finds, then keep it)")
    ap.add_argument("--wilor-yolo-every", type=int, default=4,
                    help="wilor: run the hand detector every N frames (in between, boxes follow the keypoints)")
    ap.add_argument("--wilor-fp32", action="store_true", help="wilor: float32 (~3x slower)")
    ap.add_argument("--scale", type=float, default=1.0, help="arm m per hand m")
    ap.add_argument("--rot-scale", type=float, default=1.0,
                    help="multiplies all the rotation gains below")
    ap.add_argument("--roll-gain", type=float, nargs=2, default=None, metavar=("NEG", "POS"),
                    help="gripper roll per hand roll (about the forward axis), each direction; "
                         "gripper roll is 0..+90, so negative gains make -hand roll the "
                         "clockwise one (default: from the range calibration)")
    ap.add_argument("--pitch-gain", type=float, nargs=2, default=None, metavar=("UP", "DOWN"),
                    help="gripper pitch per hand pitch, tilting up / down (default: from the "
                         "range calibration)")
    ap.add_argument("--no-range-calib", action="store_true",
                    help="skip the roll/tilt range steps of the calibration (default gains)")
    ap.add_argument("--yaw-gain", type=float, default=1.0,
                    help="yaw (the IK replaces it with the pan direction by default)")
    ap.add_argument("--rot-deadband", type=float, default=3.0,
                    help="soft deadband around the neutral hand (deg)")
    ap.add_argument("--rot-filter", type=float, nargs=3, default=[0.3, 1.0, 0.3],
                    metavar=("MIN_HZ", "BETA", "D_HZ"),
                    help="one-euro filter on the hand rotation: cutoff when still, increase "
                         "per rad/s, speed-estimate cutoff (lower = smoother, laggier)")
    ap.add_argument("--no-orientation", action="store_true", help="send roll/pitch/yaw = 0")
    ap.add_argument("--neutral-offset", type=float, nargs=3, default=[0.0, 0.0, 0.0],
                    help="arm offset from home for the neutral hand (m)")
    ap.add_argument("--z-min", type=float, default=float(BOX_MIN[2]),
                    help="lowest arm offset below home (m); the table guard. Lower it in "
                         "small steps (home is 17.6 cm above base_link)")
    ap.add_argument("--max-lin", type=float, default=1.0, help="target speed limit (m/s); 0.25 = old, smoother but ~200 ms more lag")
    ap.add_argument("--max-ang", type=float, default=360.0,
                    help="target rotation speed limit (deg/s, gripper; gains make it turn faster "
                         "than the hand)")
    ap.add_argument("--reproj-px", type=float, default=15.0,
                    help="drop landmarks whose reprojection error is above this (px)")
    ap.add_argument("--no-skew-correct", action="store_true",
                    help="don't move cam1's landmarks to cam0's capture time")
    ap.add_argument("--frame", choices=["fit", "3pt"], default="fit",
                    help="hand rotation: fit = least-squares fit of the 5-point palm template "
                         "(needs a calibration made with it), 3pt = old wrist/index/pinky frame")
    ap.add_argument("--fit-mm", type=float, default=15.0,
                    help="reject frames whose palm fits the template worse than this (RMS mm)")
    ap.add_argument("--jump-deg", type=float, default=45.0,
                    help="hold back hand rotations jumping more than this until 3 frames agree "
                         "(0 = off)")
    ap.add_argument("--mono-fallback", action="store_true",
                    help="track the palm from one camera when the other loses the hand")
    ap.add_argument("--no-view", action="store_true", help="no preview window")
    ap.add_argument("--calib-up", action="store_true",
                    help="hand calibration: add a 'move ~15 cm straight up' step after forward, so robot up "
                         "comes from the hand instead of assuming level cameras (prints the camera tilt)")
    ap.add_argument("--grip-open-curl", type=float, default=None,
                    help="finger curl (palm lengths) that counts as fully open "
                         f"(default: from the hand calibration's flat hand, else {GRIP_OPEN_CURL})")
    ap.add_argument("--grip-closed-curl", type=float, default=0.85,
                    help="finger curl that counts as fully closed (fist)")
    ap.add_argument("--grip-hold-deg", type=float, default=55.0,
                    help="gripper holds while the palm is turned more than this from the cameras "
                         "(finger curl is unreliable there)")
    ap.add_argument("--grip-band", type=float, default=0.06,
                    help="gripper backlash in curl units: curl changes smaller than half this are "
                         "ignored (noise); 0 = off")
    ap.add_argument("--pos-filter", choices=["imm", "kalman", "euro"], default="imm",
                    help="position filter: still/moving model pair (hand_mapping.StereoIMM), a single "
                         "stereo-aware Kalman (StereoKalman, --kf-*) or the old one-euro (--euro-pos)")
    ap.add_argument("--imm", choices=sorted(IMM_PRESETS["wilor"]), default="smooth",
                    help="IMM setting: smooth (default) or responsive (less tracking error, overshoot "
                         "and lag, rougher); tuned separately per --backend (hand_mapping.IMM_PRESETS)")
    ap.add_argument("--kf-q", type=float, default=None,
                    help="Kalman process noise (m^2/s^3): higher = less lag, less smoothing "
                         "(default per backend: hand_mapping.KALMAN_DEFAULTS)")
    ap.add_argument("--kf-sigma-px", type=float, default=None,
                    help="Kalman landmark noise (px): higher = smoother, more lag and overshoot "
                         "(a still hand measures ~0.35, moving hands are noisier; 3-6 is the useful range)")
    ap.add_argument("--kf-gate", type=float, default=16.0,
                    help="Kalman maneuver threshold (Mahalanobis^2): beyond it the filter catches up faster")
    ap.add_argument("--euro-pos", type=float, nargs=2, default=[1.0, 2.0], metavar=("MIN_CUTOFF", "BETA"),
                    help="one-euro position filter (--pos-filter euro): Hz, s/m")
    ap.add_argument("--kf-vtau", type=float, default=0.0,
                    help="Kalman velocity decay time constant (s, 0 = off): less overshoot on stops")
    ap.add_argument("--lead", type=float, default=0.0,
                    help="Kalman: predict this far ahead (s) to offset latency (0 = off)")
    ap.add_argument("--still-hold", type=float, nargs="?", const=0.01, default=None, metavar="M_S",
                    help="freeze the position while the hand is slower than M_S (default 0.01 m/s) "
                         "for 0.15 s; releases past 5 mm or 3x the speed")
    ap.add_argument("--record", metavar="NPZ", help="save raw landmarks for --replay")
    ap.add_argument("--replay", metavar="NPZ", help="run a --record file instead of the cameras")
    ap.add_argument("--log", metavar="NPZ", help="save per-frame targets for filter_eval.py")
    ap.add_argument("--record-frames", metavar="DIR",
                    help="save both camera streams (JPEG) for --replay-frames (common/frames.py)")
    ap.add_argument("--replay-frames", metavar="DIR",
                    help="run a --record-frames dir instead of the cameras, through --backend (compare "
                         "backends on the same video)")
    ap.add_argument("--replay-live", action="store_true",
                    help="with --replay-frames: drop the pairs a live run would have missed while the "
                         "backend was busy (fair speed comparison; otherwise every pair is processed)")
    ap.add_argument("--reanchor-s", type=float, default=0.3,
                    help="hand hidden at least this long (s) -> re-anchor when it returns")
    ap.add_argument("--two-hands", action="store_true",
                    help="right hand = position + pitch, left hand = wrist_roll (/wrist_roll_cmd) + "
                         "gripper (TWO_HAND_PLAN.md); after the right-hand calibration, 2 left-hand steps")
    ap.add_argument("--roll-sign", type=float, default=1.0, choices=[1.0, -1.0],
                    help="two hands: -1 if wrist_roll turns the opposite way to the left hand")
    args = ap.parse_args()
    if args.two_hands and (args.record or args.replay):
        raise SystemExit("--two-hands: --record / --replay store one hand; use --record-frames / --replay-frames")
    if args.replay:
        args.no_view = True
    if args.replay and args.replay_frames:
        raise SystemExit("--replay and --replay-frames: pick one")
    replaying = bool(args.replay or args.replay_frames)

    try:
        rig = load_rig(args.rig)
    except FileNotFoundError as e:
        raise SystemExit(str(e))
    est = h3.StereoHand(rig, reproj_px=args.reproj_px, mono_fallback=args.mono_fallback,
                        frame=args.frame, fit_mm=args.fit_mm, jump_deg=args.jump_deg)
    size = tuple(int(v) for v in rig["size"])
    print(f"stereo rig: {args.rig} ({rig['method']}), {size[0]}x{size[1]}, baseline "
          f"{np.linalg.norm(rig['t']) * 100:.1f} cm")
    calib, mapper, grip_f = None, None, None
    # two hands: the left hand's own estimator / calibration / roll + grip filters
    two = args.two_hands
    est_l = h3.StereoHand(rig, reproj_px=args.reproj_px, mono_fallback=False, frame=args.frame,
                          fit_mm=args.fit_mm, jump_deg=args.jump_deg) if two else None
    lcal = LeftHandCalibrator() if two else None
    roll_f = grip_l = None
    left_axis = None  # robot forward in cam coords (right-hand calibration): the left roll's axis

    def start_tracking(res):
        nonlocal mapper, grip_f, left_axis
        left_axis = np.asarray(res["A"], float)[0]
        if lcal is not None:
            lcal.axis = left_axis
        grip_open = GRIP_OPEN_CURL
        est.set_hand(res["bones"], res["template"])
        if args.grip_open_curl is not None:
            grip_open = args.grip_open_curl
        elif res.get("open_curl") is not None:
            grip_open = float(res["open_curl"]) * GRIP_OPEN_MARGIN
        # keep a usable range even if the calibration hand wasn't really flat
        grip_open = max(grip_open, args.grip_closed_curl + 0.15)
        print(f"gripper: fully open at curl >= {grip_open:.2f}, closed at <= {args.grip_closed_curl:.2f}; "
              f"holds while the palm is > {args.grip_hold_deg:.0f} deg from the cameras")
        old_grip = grip_f.grip if grip_f is not None else None
        grip_f = GripFilter(grip_open, args.grip_closed_curl, hold_deg=args.grip_hold_deg,
                            band=args.grip_band, tau=GRIP_TAU, max_dt=HandMapper.MAX_DT)
        grip_f.grip = old_grip  # recalibration: the gripper doesn't jump
        rng = {k: tuple(None if not np.isfinite(v) else float(v) for v in res.get(f"range_{k}", (np.nan,) * 2))
               for k in ("roll", "pitch")}
        cal = gains_from_range(rng)
        gains = {}
        for k, user, dflt in (("roll", args.roll_gain, ROLL_GAIN), ("pitch", args.pitch_gain, PITCH_GAIN)):
            gains[k] = tuple(user[i] if user else (cal[k][i] if cal[k][i] is not None else dflt[i])
                             for i in range(2))
            src = "option" if user else ", ".join(
                f"{'+-'[i == 0]}{rng[k][i]:.0f}deg" if rng[k][i] is not None else "default" for i in (0, 1))
            print(f"{k} gain -/+ {gains[k][0]:.2f} / {gains[k][1]:.2f}  (hand range {src})")
        old = mapper
        kalman = None
        f_px, base_m = float(rig["K0"][0, 0]), float(np.linalg.norm(rig["t"]))
        if args.pos_filter == "imm":
            kalman = StereoIMM(f_px, base_m, **IMM_PRESETS[args.backend][args.imm])
        elif args.pos_filter == "kalman":
            kf = KALMAN_DEFAULTS[args.backend]
            kalman = StereoKalman(f_px, base_m, sigma_px=kf["sigma_px"] if args.kf_sigma_px is None else args.kf_sigma_px,
                                  q=kf["q"] if args.kf_q is None else args.kf_q, gate=args.kf_gate,
                                  v_tau=args.kf_vtau)
        still = None if args.still_hold is None else StillHold(args.still_hold, 3 * args.still_hold)
        mapper = HandMapper(res["A"], res["neutral_p"], res["neutral_R"], scale=args.scale,
                            rot_filter=tuple(args.rot_filter),
                            shaper=RotationShaper(gains["roll"], gains["pitch"], args.yaw_gain,
                                                  math.radians(args.rot_deadband), scale=args.rot_scale),
                            neutral_offset=args.neutral_offset,
                            max_lin_speed=args.max_lin, max_ang_speed=math.radians(args.max_ang),
                            box_min=np.array([BOX_MIN[0], BOX_MIN[1], args.z_min]),
                            pos_filter=tuple(args.euro_pos),
                            kalman=kalman, still_hold=still, lead_s=args.lead)
        if old is not None:  # recalibration: continue (speed-limited) from the current target
            mapper.cur_p, mapper.cur_R, mapper.cur_t = old.cur_p, old.cur_R, old.cur_t

    calib = HandCalibrator(args.frame, not args.no_range_calib, args.calib_up, not args.two_hands)  # every run: calibration hand = origin

    pub = Publisher(not args.no_ros)
    replay = Replay(args.replay, realtime=not args.no_ros) if args.replay else None
    if replay is not None:  # the landmarks are in the file; it says which backend made them
        if replay.backend not in BACKENDS:
            raise SystemExit(f"{args.replay} was recorded with {replay.backend}, which this branch "
                             f"doesn't have (filter presets: {', '.join(BACKENDS)})")
        args.backend = replay.backend
    rec = Recorder(args.record, args.backend) if args.record else None
    frec = FrameRecorder(args.record_frames) if args.record_frames else None
    log = FrameLog(args.log, args) if args.log else None
    if replay is not None:
        cam = lms = replay
        print(f"replaying {args.replay} ({len(cam.t)} frames, {cam.backend}); recorded with: "
              f"{cam.argv or '(defaults)'}")
    else:
        # backend first: WiLoR takes ~15 s to load, and missing packages fail before the cameras open
        lms = make_landmarks(args.backend, hand=args.wilor_hand, yolo_every=args.wilor_yolo_every,
                             fp32=args.wilor_fp32, warmup_size=size,  # MPS compiles per input shape
                             hands=2 if two else 1)
        print(f"landmarks: {args.backend}")
        if args.replay_frames:
            cam = FrameReplay(args.replay_frames, realtime=not args.no_ros and not args.replay_live,
                              live=args.replay_live)
            print(f"replaying frames {args.replay_frames} ({len(cam.t)} pairs) through {args.backend}; "
                  f"recorded with: {cam.argv or '(defaults)'}")
        else:
            cam = StereoCam(args.cams, size=size)  # the intrinsics only hold at the calibrated size
            print("cameras:", cam.configure())
    skew = None if args.no_skew_correct else h3.SkewCorrector()
    skew_l = None if args.no_skew_correct or not two else h3.SkewCorrector()
    lost_since, grip = None, None  # when the hand disappeared; filtered grip 0..1
    last_print, fps_n, fps_t, fps = 0.0, 0, time.monotonic(), 0.0
    target, why = None, ""
    jitter = Jitter()

    def start_left(res):
        """Left-hand calibration done: roll + grip filters (a recalibration keeps both commands)."""
        nonlocal roll_f, grip_l
        est_l.set_hand(res["bones"], res["template"])
        grip_open = args.grip_open_curl if args.grip_open_curl is not None else (
            res["open_curl"] * GRIP_OPEN_MARGIN if res["open_curl"] is not None else GRIP_OPEN_CURL)
        grip_open = max(grip_open, args.grip_closed_curl + 0.15)
        old_roll = roll_f.cmd if roll_f is not None else 0.0
        # a clockwise hand turn = gripper roll + (as one-handed) = wrist_roll - (tool roll =
        # -wrist_roll, ik_solver.roll_sign); --roll-sign -1 flips it if the arm disagrees
        roll_f = RollFilter(sign=-res["cw_sign"] * args.roll_sign, start=old_roll, axis=left_axis)
        old_grip = grip_l.grip if grip_l is not None else None
        grip_l = GripFilter(grip_open, args.grip_closed_curl, hold_deg=args.grip_hold_deg,
                            band=args.grip_band, tau=GRIP_TAU, max_dt=HandMapper.MAX_DT)
        grip_l.grip = old_grip
        print(f"left hand: wrist_roll from its twist ({lcal.msg}), gripper open at curl >= {grip_open:.2f}, "
              f"closed at <= {args.grip_closed_curl:.2f}", flush=True)

    def process_left(t):
        """Two hands: the left hand -> wrist_roll + gripper. Returns ((px0, px1), status)."""
        nonlocal lcal
        l0, l1 = lms.other if lms.other_fresh else (None, None)
        lpose = None
        if lms.other_fresh:
            l1s = l1 if skew_l is None else skew_l(l1, cam.t1, cam.t0)
            lpose, _ = est_l.estimate(l0, l1s, t)
        if calib is not None:
            return (l0, l1), "left: calibrates after the right hand"
        if lcal is not None:
            if not lcal.feed(lpose, t):
                return (l0, l1), lcal.instructions()
            start_left(lcal.result)
            lcal = None
        if lpose is not None and lpose["mode"] == "stereo":
            palm = facing_weight(lpose["p"], lpose["R"])[1]
            cmd, st = roll_f.update(lpose["R"], lpose["curl"], palm, t)
            g = grip_l(lpose["curl"], palm, t)
            if g is not None:
                pub.grip(g)
            line = f"LEFT {st:<6} roll {math.degrees(cmd):+4.0f} deg" + (
                f"  grip {g:.2f}{'H' if grip_l.held else ' '}" if g is not None else "")
        else:
            if lms.other_fresh:
                roll_f.lost(t)
            line = f"LEFT none   roll {math.degrees(roll_f.cmd):+4.0f} deg (holding)"
        pub.roll(roll_f.cmd)  # every pair, also while hidden: the IK keeps wrist_roll locked
        return (l0, l1), line

    def recalibrate():
        nonlocal calib, target, lcal
        calib, target = HandCalibrator(args.frame, not args.no_range_calib, args.calib_up, not args.two_hands), None  # stops publishing
        if two:
            lcal = LeftHandCalibrator(axis=left_axis)  # start_tracking updates it after the right hand
        for r in (rec, frec):
            if r is not None:
                r.mark_recal()

    def process_pair():
        """One stereo pair: landmarks -> pose -> calibration / mapping -> publish + terminal
        line. Returns what the preview draws, or None when no pair came."""
        nonlocal calib, lost_since, grip, last_print, fps_n, fps_t, fps, target, why
        pr = cam.pair()
        if pr is None:
            return None
        f0, f1, t = pr
        if frec is not None:
            frec.add(f0, f1, t, cam.t0, cam.t1)
        px0, px1_raw = lms(f0, f1, t)
        if rec is not None:
            rec.add(t, cam.t0, cam.t1, px0, px1_raw)
        px1 = px1_raw if skew is None else skew(px1_raw, cam.t1, cam.t0)
        pose, why = est.estimate(px0, px1, t)
        status = ""
        published = False

        if calib is not None:
            if calib.feed(pose, t):
                if replaying:  # don't overwrite the live calibration
                    print(f"hand calibrated ({calib.msg})")
                else:
                    save_hand(calib.result, hand_path(args.backend))
                    print(f"hand calibrated ({calib.msg}); saved {hand_path(args.backend)}")
                start_tracking(calib.result)
                calib = None
                lost_since = None
            else:
                status = calib.instructions()
            if log is not None:
                log.add(t, 0, pose)
        elif pose is not None:
            if lost_since is not None and t - lost_since >= args.reanchor_s:
                mapper.reanchor(pose["p"], pose["R"])  # clutch release: no jump
            lost_since = None
            jitter.add(t, pose["p"], pose["R"])
            raw_p, raw_R = mapper.raw_target(pose["p"], pose["R"])
            target = mapper.update(pose["p"], pose["R"], t, pose.get("err"), pose["mode"] != "stereo")
            out = target.copy()
            if args.no_orientation:
                out[3:] = 0.0
            pub(out)
            published = True
            status = f"TRACKING [{pose['mode']}]"
            # mono (no curl) or palm turned away -> the gripper holds
            grip = None if two else grip_f(pose["curl"], mapper.palm_angle, t)  # two hands: the left's
            if grip is not None:
                pub.grip(grip)
                curl_s = "  -- " if pose["curl"] is None else f"{pose['curl']:.2f}"
                status += f" grip {grip:.2f}{'H' if grip_f.held else ' '}(curl {curl_s})"
            if log is not None:
                log.add(t, 1, pose, np.concatenate([raw_p, h3.matrix_to_rpy(raw_R)]), out, grip)
        else:
            if lost_since is None:
                lost_since = t
            if log is not None:
                log.add(t, 2)
            jitter.clear()
            status = f"no hand (clutch: holding): {why}"

        left_px, left_line = process_left(t) if two else ((None, None), "")

        if time.monotonic() - last_print > 0.5:
            last_print = time.monotonic()
            line = f"{'PUB ' if published and not args.no_ros else '    '}{status:<34} {fps:3.0f} fps"
            if target is not None and calib is None:
                x, y, z, r, p, yw = target
                line += (f"  xyz cm {x * 100:+6.1f} {y * 100:+6.1f} {z * 100:+6.1f}  "
                         f"rpy deg {math.degrees(r):+5.0f} {math.degrees(p):+5.0f} "
                         f"{math.degrees(yw):+5.0f}")
                hr = np.degrees(mapper.hand_rpy)
                line += (f"  hand {hr[0]:+4.0f} {hr[1]:+4.0f} {hr[2]:+4.0f}"
                         f"  palm {mapper.palm_angle:2.0f}deg{'' if mapper.trust > 0 else ' HELD'}")
                jit = jitter.read()
                if jit is not None:
                    line += f"  jitter {jit[0]:4.1f} mm {jit[1]:4.1f} deg"
            print(line + (f"  | {left_line}" if two else ""), flush=True)

        if replaying and cam.recalibrate_now():
            calib, target = HandCalibrator(args.frame, not args.no_range_calib, args.calib_up, not args.two_hands), None
        fps_n += 1
        if time.monotonic() - fps_t > 1.0:
            fps, fps_n, fps_t = fps_n / (time.monotonic() - fps_t), 0, time.monotonic()
        return dict(f0=f0, f1=f1, px0=px0, px1=px1_raw, pose=pose, status=status, fps=fps,
                    skew_ms=(cam.t1 - cam.t0) * 1000,
                    target=target if calib is None else None, left_px=left_px, left_line=left_line)

    def show(f0, f1, snap, cam_fps=None):
        """Draw snap's landmarks / pose / text on f0, f1 and show them. Returns the key."""
        draw_hand(f0, snap["px0"])
        draw_hand(f1, snap["px1"])
        draw_hand(f0, snap["left_px"][0], LEFT_COLOR)
        draw_hand(f1, snap["left_px"][1], LEFT_COLOR)
        if snap["pose"] is not None:
            draw_axes(f0, snap["pose"], rig["K0"], rig["d0"])
        view = np.hstack([f0, f1])
        rate = f"{snap['fps']:.0f} fps" if cam_fps is None else \
            f"model {snap['fps']:.0f} pairs/s, view {cam_fps:.0f} fps"
        lines = [snap["status"], f"{rate}, skew {snap['skew_ms']:+.0f} ms"]
        if snap["target"] is not None:
            x, y, z, r, p, yw = snap["target"]
            lines.append(f"offset cm x{x * 100:+.1f} y{y * 100:+.1f} z{z * 100:+.1f}  "
                         f"rpy deg {math.degrees(r):+.0f} {math.degrees(p):+.0f} "
                         f"{math.degrees(yw):+.0f}")
        if snap["left_line"]:
            lines.append(snap["left_line"])
        for i, s in enumerate(lines):
            cv2.putText(view, s, (10, 25 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                        (0, 0, 255) if snap["pose"] is None else (0, 255, 255), 2)
        cv2.imshow("hand -> /ball_position (q quits, c recalibrates)", fit_view(view))
        return cv2.waitKey(1) & 0xFF

    # live cameras with a window: WiLoR and everything after it run in a worker thread, the
    # main thread (macOS: windows only work there) shows the newest camera frames at camera
    # rate with the newest result drawn on them. Replays and --no-view stay single-threaded
    # (deterministic, same output as before).
    threaded = not args.no_view and not replaying and isinstance(cam, StereoCam)
    worker, stop = None, threading.Event()
    try:
        if not threaded:
            while True:
                snap = process_pair()
                if snap is None or args.no_view:
                    continue
                k = show(snap["f0"], snap["f1"], snap)
                if k in (ord("q"), 27):
                    break
                if k == ord("c"):
                    recalibrate()
        else:
            lock, latest, failed, want_recal = threading.Lock(), [None], [], threading.Event()

            def work():
                try:
                    while not stop.is_set():
                        if want_recal.is_set():
                            want_recal.clear()
                            recalibrate()
                        snap = process_pair()
                        if snap is not None:
                            with lock:
                                latest[0] = snap
                except BaseException as e:  # shown and re-raised by the main thread
                    failed.append(e)
                    stop.set()

            worker = threading.Thread(target=work, name="hand-pipeline", daemon=True)
            worker.start()
            seq, view_n, view_t, view_fps = -1, 0, time.monotonic(), 0.0
            while not stop.is_set():
                f0, _, s0 = cam.c0.wait_new(seq, timeout=0.2)
                if f0 is None or s0 == seq:
                    continue
                seq = s0
                f1 = cam.c1.latest()[0]
                with lock:
                    snap = latest[0]
                if f1 is None or snap is None:
                    continue
                view_n += 1
                if time.monotonic() - view_t > 1.0:
                    view_fps, view_n, view_t = view_n / (time.monotonic() - view_t), 0, time.monotonic()
                # copies: the worker may be reading these same frames
                k = show(f0.copy(), f1.copy(), snap, view_fps)
                if k in (ord("q"), 27):
                    break
                if k == ord("c"):
                    want_recal.set()
            if failed:
                raise failed[0]
    except (KeyboardInterrupt, ReplayEnd):
        pass
    finally:
        stop.set()
        if worker is not None:
            worker.join(timeout=5.0)  # finish the pair in flight before closing cameras / ROS
        for r in (rec, frec):
            if r is not None:
                r.save()
        if log is not None:
            log.save()
        if getattr(cam, "skipped", 0):
            print(f"live replay: skipped {cam.skipped} of {len(cam.t)} pairs while busy")
        lms.close()
        cam.close()
        pub.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
