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
    Close the hand = close the gripper (proportional: open hand = open,
    fist = closed). Tune with --grip-open-curl / --grip-closed-curl (the
    printout shows the current curl).
    Hide the hand = clutch: nothing is published, the arm and gripper hold
    (IK after 0.5 s). When the hand comes back after >= --reanchor-s it
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
from hand_mapping import HandMapper
from stereo_cam import RIG_PATH, StereoCam, StereoLandmarks, draw_hand, load_rig

BALL_TOPIC = "/ball_position"  # must match so101_mirror.BALL_TOPIC
GRIP_TOPIC = "/gripper"        # std_msgs/Float64, 0 = open .. 1 = closed (feetech_mirror --gripper)
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


class HandTarget:
    """Per-frame logic: hand calibration, then pose -> /ball_position + /gripper.

    Shared by main() and the app window (ui/pipeline.py). args are the
    parsed build_parser() options.
    """

    def __init__(self, est, pub, args):
        self.est, self.pub, self.args = est, pub, args
        self.calib = self.mapper = self.target = None
        self.lost_since, self.grip = None, None  # when the hand disappeared; filtered grip 0..1

    def recalibrate(self):
        """(Re)start the hand calibration; nothing is published until it's done."""
        self.calib, self.target = HandCalibrator(), None

    def start_tracking(self, res):
        a = self.args
        self.est.set_hand(res["bones"], res["template"])
        old = self.mapper
        self.mapper = HandMapper(res["A"], res["neutral_p"], res["neutral_R"], scale=a.scale,
                                 rot_scale=a.rot_scale, neutral_offset=a.neutral_offset,
                                 max_lin_speed=a.max_lin, max_ang_speed=math.radians(a.max_ang))
        if old is not None:  # recalibration: continue (speed-limited) from the current target
            self.mapper.cur_p, self.mapper.cur_R, self.mapper.cur_t = old.cur_p, old.cur_R, old.cur_t

    def step(self, pose, why, t):
        """One frame's estimate -> (status line, whether a target was published)."""
        a = self.args
        if self.calib is not None:
            if not self.calib.feed(pose, t):
                return self.calib.instructions(), False
            save_hand(self.calib.result)
            print(f"hand calibrated ({self.calib.msg}); saved {HAND_PATH}")
            self.start_tracking(self.calib.result)
            self.calib, self.lost_since = None, None
            return "", False
        if self.mapper is None:
            return "not calibrated", False
        if pose is None:
            if self.lost_since is None:
                self.lost_since = t
            return f"no hand (clutch: holding): {why}", False

        if self.lost_since is not None and t - self.lost_since >= a.reanchor_s:
            self.mapper.reanchor(pose["p"], pose["R"])  # clutch release: no jump
        self.lost_since = None
        self.target = self.mapper.update(pose["p"], pose["R"], t)
        out = self.target.copy()
        if a.no_orientation:
            out[3:] = 0.0
        self.pub(out)
        status = f"TRACKING [{pose['mode']}]"
        if pose["curl"] is not None:  # stereo only; mono -> gripper holds
            g = (a.grip_open_curl - pose["curl"]) / (a.grip_open_curl - a.grip_closed_curl)
            g = min(1.0, max(0.0, g))
            self.grip = g if self.grip is None else self.grip + GRIP_ALPHA * (g - self.grip)
            self.pub.grip(self.grip)
            status += f" grip {self.grip:.2f} (curl {pose['curl']:.2f})"
        return status, True


def build_parser():
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
    return ap


def main():
    args = build_parser().parse_args()

    try:
        rig = load_rig(args.rig)
        if args.reuse_calib:
            load_hand()
    except FileNotFoundError as e:
        raise SystemExit(f"missing {e.filename or e}" if args.reuse_calib and os.path.exists(args.rig)
                         else str(e))
    est = h3.StereoHand(rig, reproj_px=args.reproj_px, mono_fallback=args.mono_fallback)
    print(f"stereo rig: {args.rig} ({rig['method']}), baseline "
          f"{np.linalg.norm(rig['t']) * 100:.1f} cm")

    pub = Publisher(not args.no_ros)
    ht = HandTarget(est, pub, args)
    if args.reuse_calib:
        ht.start_tracking(load_hand())
        print(f"hand calibration loaded from {HAND_PATH}")
    else:
        ht.recalibrate()

    cam = StereoCam(args.cams)
    lms = StereoLandmarks()
    last_print, fps_n, fps_t, fps = 0.0, 0, time.monotonic(), 0.0
    try:
        while True:
            pr = cam.pair()
            if pr is None:
                continue
            f0, f1, t = pr
            px0, px1 = lms(f0, f1, t)
            pose, why = est.estimate(px0, px1)
            status, published = ht.step(pose, why, t)
            target = ht.target

            if time.monotonic() - last_print > 0.5:
                last_print = time.monotonic()
                line = f"{'PUB ' if published and not args.no_ros else '    '}{status:<34} {fps:3.0f} fps"
                if target is not None and ht.calib is None:
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
            if pose is not None:
                draw_axes(f0, pose, rig["K0"], rig["d0"])
            view = np.hstack([f0, f1])
            lines = [status, f"{fps:.0f} fps"]
            if target is not None and ht.calib is None:
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
                ht.recalibrate()  # stops publishing until done
    except KeyboardInterrupt:
        pass
    finally:
        lms.close()
        cam.close()
        pub.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
