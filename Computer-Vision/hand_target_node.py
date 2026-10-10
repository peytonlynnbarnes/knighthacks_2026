#!/home/peyton/kh/.venv/bin/python
"""Stereo hand tracking -> /ball_position [x y z roll pitch yaw] for the SO-101 IK,
plus /gripper (0 open .. 1 closed) from how far the fingers are curled.

Pipeline (see ~/kh/INTEGRATION.md for the /ball_position contract):
    2 cameras -> MediaPipe landmarks per view -> triangulate (calib/stereo.npz)
    -> validate (reprojection, palm bone lengths) -> hand frame
    -> relative to the calibrated neutral hand -> robot axes, x scale
    -> one-euro filter, offset box clamp, speed limit -> /ball_position

Startup: a ~10 s hand calibration (hand_calib.py; instructions in the
window), unless --reuse-calib. Needs calib/stereo.npz from stereo_calib.py.

Run (venv python + ROS on PYTHONPATH, like feetech_mirror.py):
    source /opt/ros/jazzy/setup.bash
    .venv/bin/python Computer-Vision/hand_target_node.py --no-ros   # print only
    .venv/bin/python Computer-Vision/hand_target_node.py            # publish
    # IK without the Gazebo ball:
    ros2 launch so101_mirror sim.launch.py ball_source:=false   (or ros2 run so101_mirror ik_node.py)

Controls:
    neutral hand -> arm at --neutral-offset (default 10 cm back, 5 cm down from
    home, the middle of the reachable box). Hand motion x --scale (1.1).
    Close the gripping hand (main in --one-hand mode, else the other hand)
    = close the gripper (proportional: open hand = open,
    fist = closed). Tune with --grip-open-curl / --grip-closed-curl (the
    printout shows the current curl).
    Two hands (default; --one-hand for the old single-hand mode):
      MAIN hand (--main-hand, default right): arm position only; its own
        rotation is ignored and the gripper is kept level (rpy 0).
      OTHER hand: the gripper (finger curl, as above) and the wrist_roll
        motor via /wrist_roll_cmd. Twist it
        about its own finger axis (doorknob); relative: when it appears the
        roll stays put, then the twist turns it (x --roll-scale, flip with
        --roll-sign -1). Out of view = the roll holds. Steadied by
        hand_mapping.RollFilter: frozen while the fingers open/close, held
        while the palm is edge-on, one-frame jumps ignored, smoothed and
        limited to --max-roll-speed. The status shows its state. The IK holds
        wrist_roll at that angle and solves the other joints around it.
        Main hidden + other visible = position frozen and steady; the other
        hand still rolls the wrist and opens/closes the gripper (grabbing).
      Hands are told apart by MediaPipe handedness (frames are unmirrored);
      if the preview labels them the wrong way round, pass --swap-hands.
    Hide the main hand = clutch: no arm target is published and the arm holds
    (IK after 0.5 s); a hidden gripping hand = the gripper holds. When the hand comes back after >= --reanchor-s it
    re-anchors: the arm continues from where it is, no jump. Shorter
    dropouts (bad frames) don't re-anchor.
    Keys: q/Esc quit, c recalibrate the hand.

Safety: the target starts at home (all zeros, where feetech_mirror --live
homes) and moves at most --max-lin m/s and --max-ang deg/s, then the IK
(EMA, workspace box) and feetech_mirror (rate limit, clamps) still apply.
"""
import argparse
import math
import os
import time

import cv2
import numpy as np

import hand3d as h3
from hand_calib import HAND_PATH, HandCalibrator, load_hand, save_hand
from hand_mapping import HandMapper, RollFilter
from stereo_cam import RIG_PATH, StereoCam, StereoLandmarks, draw_hand, load_rig

BALL_TOPIC = "/ball_position"  # must match so101_mirror.BALL_TOPIC
GRIP_TOPIC = "/gripper"        # std_msgs/Float64, 0 = open .. 1 = closed (feetech_mirror --gripper)
ROLL_TOPIC = "/wrist_roll_cmd"  # must match so101_mirror.WRIST_ROLL_TOPIC (Float64 rad, URDF)
GRIP_ALPHA = 0.5               # EMA on the grip value, per frame


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
        self.roll_pub = self.node.create_publisher(Float64, ROLL_TOPIC, 10)

    def roll(self, v):
        if self.node is not None:
            self.roll_pub.publish(self.F64(data=float(v)))

    def __call__(self, target):
        if self.node is not None:
            self.pub.publish(self.Msg(data=[float(v) for v in target]))

    def grip(self, v):
        if self.node is not None:
            self.grip_pub.publish(self.F64(data=float(v)))

    def close(self):
        if self.node is not None:
            self.node.destroy_node()
            self.rclpy.try_shutdown()


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
    ap.add_argument("--cams", type=int, nargs=2, default=[0, 2])
    ap.add_argument("--rig", default=RIG_PATH)
    ap.add_argument("--no-ros", action="store_true", help="don't publish, just print")
    ap.add_argument("--reuse-calib", action="store_true", help=f"load {HAND_PATH} instead of calibrating")
    ap.add_argument("--scale", type=float, default=1.1, help="arm m per hand m")
    ap.add_argument("--rot-scale", type=float, default=1.0, help="arm rad per hand rad")
    ap.add_argument("--no-orientation", action="store_true", help="send roll/pitch/yaw = 0")
    ap.add_argument("--neutral-offset", type=float, nargs=3, default=[-0.10, 0.0, -0.05],
                    help="arm offset from home for the neutral hand (m)")
    ap.add_argument("--max-lin", type=float, default=0.25, help="target speed limit (m/s)")
    ap.add_argument("--max-ang", type=float, default=120.0, help="target speed limit (deg/s)")
    ap.add_argument("--reproj-px", type=float, default=15.0, help="reject frames above this")
    ap.add_argument("--mono-fallback", action="store_true",
                    help="track the palm from one camera when the other loses the hand")
    ap.add_argument("--no-view", action="store_true", help="no preview window")
    ap.add_argument("--grip-open-curl", type=float, default=1.35,
                    help="finger curl (palm lengths) that counts as fully open")
    ap.add_argument("--grip-closed-curl", type=float, default=0.85,
                    help="finger curl that counts as fully closed (fist)")
    ap.add_argument("--reanchor-s", type=float, default=0.3,
                    help="hand hidden at least this long (s) -> re-anchor when it returns")
    ap.add_argument("--one-hand", action="store_true",
                    help="single-hand mode: the main hand also sets the wrist orientation")
    ap.add_argument("--main-hand", choices=("right", "left"), default="right",
                    help="hand for position + gripper; the other one turns the wrist")
    ap.add_argument("--swap-hands", action="store_true",
                    help="MediaPipe handedness is reversed (mirrored frames)")
    ap.add_argument("--roll-scale", type=float, default=1.0, help="wrist_roll rad per hand twist rad")
    ap.add_argument("--roll-sign", type=float, choices=(1.0, -1.0), default=1.0,
                    help="-1 if twisting the hand turns the gripper the wrong way")
    ap.add_argument("--roll-edge-cos", type=float, default=0.2,
                    help="hold the roll while |palm normal . line of sight| is below this "
                         "(palm edge-on, twist unreliable); 0 disables")
    ap.add_argument("--max-roll-speed", type=float, default=90.0, help="wrist_roll command speed limit (deg/s)")
    ap.add_argument("--roll-range", type=float, nargs=2, default=[-2.70, 0.90],
                    help="wrist_roll command limits (rad); default = URDF limits combined with "
                         "the arm's usable range (SOFT_LIMITS 0..2645 ticks = -180..+52 deg)")
    args = ap.parse_args()

    try:
        rig = load_rig(args.rig)
        if args.reuse_calib:
            load_hand()
    except FileNotFoundError as e:
        raise SystemExit(f"missing {e.filename or e}" if args.reuse_calib and os.path.exists(args.rig)
                         else str(e))
    two = not args.one_hand
    est = h3.StereoHand(rig, reproj_px=args.reproj_px, mono_fallback=args.mono_fallback)
    est_w = h3.StereoHand(rig, reproj_px=args.reproj_px)  # wrist hand: stereo only
    print(f"stereo rig: {args.rig} ({rig['method']}), baseline "
          f"{np.linalg.norm(rig['t']) * 100:.1f} cm")
    calib, mapper = None, None
    # two-hand mode: the wrist hand's twist, accumulated across its appearances
    roll_f = RollFilter(sign=args.roll_sign, scale=args.roll_scale, lo=args.roll_range[0],
                        hi=args.roll_range[1], max_speed=math.radians(args.max_roll_speed),
                        edge_cos=args.roll_edge_cos, reanchor_s=args.reanchor_s)
    roll_cmd, roll_state = 0.0, "-"

    def start_tracking(res):
        nonlocal mapper
        est.set_hand(res["bones"], res["template"])
        est_w.set_hand(res["bones"], res["template"])  # left/right palms: same bone lengths
        old = mapper
        mapper = HandMapper(res["A"], res["neutral_p"], res["neutral_R"], scale=args.scale,
                            rot_scale=args.rot_scale, neutral_offset=args.neutral_offset,
                            max_lin_speed=args.max_lin, max_ang_speed=math.radians(args.max_ang))
        if old is not None:  # recalibration: continue (speed-limited) from the current target
            mapper.cur_p, mapper.cur_R, mapper.cur_t = old.cur_p, old.cur_R, old.cur_t
        if two:
            mapper.hand_R = np.eye(3)  # rotation input fixed at identity: gripper stays level
            roll_f.anchor = None  # re-anchor the wrist hand; the roll keeps its value

    if args.reuse_calib:
        start_tracking(load_hand())
        print(f"hand calibration loaded from {HAND_PATH}")
    else:
        calib = HandCalibrator()

    pub = Publisher(not args.no_ros)
    cam = StereoCam(args.cams)
    lms = StereoLandmarks(num_hands=2 if two else 1)
    lost_since, grip = None, None  # when the hand disappeared; filtered grip 0..1
    last_print, fps_n, fps_t, fps = 0.0, 0, time.monotonic(), 0.0
    target, why = None, ""
    try:
        while True:
            pr = cam.pair()
            if pr is None:
                continue
            f0, f1, t = pr
            if two:
                (px0, px1), (wpx0, wpx1) = h3.assign_hands(*lms.all(f0, f1, t), main=args.main_hand,
                                                           mirrored=args.swap_hands)
            else:
                (px0, px1), wpx0, wpx1 = lms(f0, f1, t), None, None
            pose, why = est.estimate(px0, px1)
            wpose = est_w.estimate(wpx0, wpx1)[0] if two else None
            if two and calib is None:  # wrist hand twist -> steady roll_cmd (see RollFilter)
                if wpose is not None:
                    view = wpose["p"] / np.linalg.norm(wpose["p"])  # cam0 -> hand
                    facing = abs(float(wpose["R"][:, 2] @ view))  # palm normal vs line of sight
                    roll_cmd, roll_state = roll_f.update(wpose["R"], wpose["curl"], facing, t)
                else:
                    roll_f.lost(t)
                    roll_state = "hidden"
            rot_in = np.eye(3) if two else (pose["R"] if pose is not None else None)
            status = ""
            published = False

            if calib is not None:
                if calib.feed(pose, t):
                    save_hand(calib.result)
                    print(f"hand calibrated ({calib.msg}); saved {HAND_PATH}")
                    start_tracking(calib.result)
                    calib = None
                    lost_since = None
                else:
                    status = calib.instructions()
            elif pose is not None:
                if lost_since is not None and t - lost_since >= args.reanchor_s:
                    mapper.reanchor(pose["p"], rot_in)  # clutch release: no jump
                lost_since = None
                target = mapper.update(pose["p"], rot_in, t)
                out = target.copy()
                if args.no_orientation:
                    out[3:] = 0.0
                pub(out)
                if two:
                    pub.roll(roll_cmd)
                published = True
                status = f"TRACKING [{pose['mode']}]"
            else:
                if lost_since is None:
                    lost_since = t
                status = f"no hand (clutch: holding): {why}"
                if wpose is not None and mapper is not None:  # roll only, position frozen
                    target = mapper.hold(t)
                    pub(target)
                    pub.roll(roll_cmd)
                    published = True
                    status = "main hidden: position held, roll + grip"
            gpose = wpose if two else pose  # two hands: the wrist hand grips
            if calib is None and gpose is not None and gpose["curl"] is not None:  # stereo only
                g = (args.grip_open_curl - gpose["curl"]) / (args.grip_open_curl - args.grip_closed_curl)
                g = min(1.0, max(0.0, g))
                grip = g if grip is None else grip + GRIP_ALPHA * (g - grip)
                pub.grip(grip)  # independent of the arm target: works with the main hand hidden
                status += f" grip {grip:.2f} (curl {gpose['curl']:.2f})"
            if two and calib is None:
                status += (f" | roll {math.degrees(roll_cmd):+.0f} deg [{roll_state}] "
                           + ("(wrist hand ON)" if wpose is not None else "(wrist hand off)"))

            if time.monotonic() - last_print > 0.5:
                last_print = time.monotonic()
                line = f"{'PUB ' if published and not args.no_ros else '    '}{status:<34} {fps:3.0f} fps"
                if target is not None and calib is None:
                    x, y, z, r, p, yw = target
                    line += (f"  xyz cm {x * 100:+6.1f} {y * 100:+6.1f} {z * 100:+6.1f}  "
                             f"rpy deg {math.degrees(r):+5.0f} {math.degrees(p):+5.0f} "
                             f"{math.degrees(yw):+5.0f}")
                print(line, flush=True)

            fps_n += 1
            if time.monotonic() - fps_t > 1.0:
                fps, fps_n, fps_t = fps_n / (time.monotonic() - fps_t), 0, time.monotonic()
            if args.no_view:
                continue
            draw_hand(f0, px0)
            draw_hand(f1, px1)
            draw_hand(f0, wpx0, (255, 128, 0))  # wrist hand in blue
            draw_hand(f1, wpx1, (255, 128, 0))
            for img, mp_, wp_ in ((f0, px0, wpx0), (f1, px1, wpx1)):
                for px, name in ((mp_, "MAIN"), (wp_, "WRIST")):
                    if px is not None:
                        cv2.putText(img, name, tuple(int(v) for v in px[0]), cv2.FONT_HERSHEY_SIMPLEX,
                                    0.7, (255, 255, 255), 2)
            if pose is not None:
                draw_axes(f0, pose, rig["K0"], rig["d0"])
            view = np.hstack([f0, f1])
            lines = [status, f"{fps:.0f} fps"]
            if target is not None and calib is None:
                x, y, z, r, p, yw = target
                lines.append(f"offset cm x{x * 100:+.1f} y{y * 100:+.1f} z{z * 100:+.1f}  "
                             f"rpy deg {math.degrees(r):+.0f} {math.degrees(p):+.0f} "
                             f"{math.degrees(yw):+.0f}")
            for i, s in enumerate(lines):
                cv2.putText(view, s, (10, 25 + 28 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.65,
                            (0, 0, 255) if pose is None else (0, 255, 255), 2)
            cv2.imshow("hand -> /ball_position (q quits, c recalibrates)", view)
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("c"):
                calib, target = HandCalibrator(), None  # stops publishing until done
    except KeyboardInterrupt:
        pass
    finally:
        lms.close()
        cam.close()
        pub.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
