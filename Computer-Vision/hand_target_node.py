#!/home/peyton/kh/.venv/bin/python
"""Stereo hand tracking -> /ball_position [x y z roll pitch yaw] for the SO-101 IK.

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
    home, the middle of the reachable box). Hand motion x --scale (0.8).
    FIST = clutch: the arm holds; open the hand anywhere and it continues
    from there (re-anchors, no jump).
    Hand lost / bad frame: nothing is published; the IK holds after 0.5 s.
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
FIST_FRAMES = 3                # debounce for clutch on/off


class Publisher:
    def __init__(self, enabled):
        self.node = None
        if not enabled:
            return
        import rclpy
        from std_msgs.msg import Float64MultiArray
        rclpy.init()
        self.rclpy, self.Msg = rclpy, Float64MultiArray
        self.node = rclpy.create_node("hand_target_node")
        self.pub = self.node.create_publisher(Float64MultiArray, BALL_TOPIC, 10)

    def __call__(self, target):
        if self.node is not None:
            self.pub.publish(self.Msg(data=[float(v) for v in target]))

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
    ap.add_argument("--scale", type=float, default=0.8, help="arm m per hand m")
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
    args = ap.parse_args()

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
    calib, mapper = None, None

    def start_tracking(res):
        nonlocal mapper
        est.set_hand(res["bones"], res["template"])
        old = mapper
        mapper = HandMapper(res["A"], res["neutral_p"], res["neutral_R"], scale=args.scale,
                            rot_scale=args.rot_scale, neutral_offset=args.neutral_offset,
                            max_lin_speed=args.max_lin, max_ang_speed=math.radians(args.max_ang))
        if old is not None:  # recalibration: continue (speed-limited) from the current target
            mapper.cur_p, mapper.cur_R, mapper.cur_t = old.cur_p, old.cur_R, old.cur_t

    if args.reuse_calib:
        start_tracking(load_hand())
        print(f"hand calibration loaded from {HAND_PATH}")
    else:
        calib = HandCalibrator()

    pub = Publisher(not args.no_ros)
    cam = StereoCam(args.cams)
    lms = StereoLandmarks()
    clutched, fist_n, open_n = False, 0, 0
    last_print, fps_n, fps_t, fps = 0.0, 0, time.monotonic(), 0.0
    target, why = None, ""
    try:
        while True:
            pr = cam.pair()
            if pr is None:
                continue
            f0, f1, t = pr
            px0, px1 = lms(f0, f1, t)
            pose, why = est.estimate(px0, px1)
            status = ""
            published = False

            if calib is not None:
                if calib.feed(pose, t):
                    save_hand(calib.result)
                    print(f"hand calibrated ({calib.msg}); saved {HAND_PATH}")
                    start_tracking(calib.result)
                    calib = None
                    clutched, fist_n, open_n = False, 0, 0
                else:
                    status = calib.instructions()
            elif pose is not None:
                fist_n, open_n = (fist_n + 1, 0) if pose["fist"] else (0, open_n + 1)
                if not clutched and fist_n >= FIST_FRAMES:
                    clutched = True
                elif clutched and open_n >= FIST_FRAMES:
                    clutched = False
                    mapper.reanchor(pose["p"], pose["R"])
                if clutched:
                    target = mapper.hold(t)
                else:
                    target = mapper.update(pose["p"], pose["R"], t)
                out = target.copy()
                if args.no_orientation:
                    out[3:] = 0.0
                pub(out)
                published = True
                status = ("CLUTCH (fist): holding" if clutched else "TRACKING") + f" [{pose['mode']}]"
            else:
                status = f"no target: {why}"

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
                            (0, 0, 255) if clutched else (0, 255, 255), 2)
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
