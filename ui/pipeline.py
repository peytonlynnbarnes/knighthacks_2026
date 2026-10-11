"""Camera + hand-control loop behind the app window (ui/app.py).

Opens the stereo pair like hand_target_node.py and shows cam0. start()
runs hand_target_node.HandTarget, the node's own per-frame logic: the hand
calibration (saved to calib/hand.json), then tracking that publishes
/ball_position and /gripper, exactly as the node does. Pressing it again
recalibrates (the node's `c` key). Options are the node's (build_parser()).

The ROS publisher is created up front, on the main thread. MediaPipe and the
rig file are only loaded on the first start(). If the stereo pair won't open,
cam0 alone still gives a preview and start() reports why it can't run.
"""
import os
import sys
import threading
import time

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "Computer-Vision"))

import hand3d as h3  # noqa: E402
from hand_calib import load_hand  # noqa: E402
from hand_target_node import HandTarget, Publisher, draw_axes  # noqa: E402
from stereo_cam import Camera, StereoCam, StereoLandmarks, draw_hand, load_rig  # noqa: E402

ERROR_S = 6.0  # how long error notes stay up


def short(e):
    """First line of an exception, with repo paths made relative."""
    return str(e).splitlines()[0].replace(ROOT + os.sep, "")


class Pipeline:
    """latest() -> (cam0 frame, frame seq, status) for the UI to poll.

    status is None or (text, detail, kind), kind one of "calib" (step
    instructions), "track" (publishing), "hold" (hand lost, arm holding),
    "error".
    """

    def __init__(self, args):
        self.args = args
        self.stereo, self.cam, self.stereo_err = None, None, None
        try:
            self.stereo = StereoCam(args.cams)
        except RuntimeError as e:
            self.stereo_err = str(e)
            try:
                self.cam = Camera(args.cams[0])
            except RuntimeError:
                pass
        self.pub, self.ros_err = None, None
        try:
            self.pub = Publisher(not args.no_ros)
        except ImportError as e:  # rclpy / std_msgs: ROS not sourced
            self.ros_err = short(e)
        self.lock = threading.Lock()
        self.frame, self.seq = None, 0
        self.status, self.status_until = None, 0.0
        self.rig = self.est = self.lms = self.ht = None
        self.want_start = False
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    @property
    def has_camera(self):
        return self.stereo is not None or self.cam is not None

    def start(self):
        """Calibrate, then control the arm (or recalibrate); runs on the camera thread."""
        self.want_start = True

    def latest(self):
        with self.lock:
            if self.status and self.status[2] == "error" and time.monotonic() > self.status_until:
                self.status = None
            return self.frame, self.seq, self.status

    def _set_status(self, text, detail="", kind="calib"):
        with self.lock:
            self.status = (text, detail, kind)
            self.status_until = time.monotonic() + ERROR_S

    def _run(self):
        seq = 0
        while self.running:
            f0 = f1 = None
            t = time.monotonic()
            if self.stereo is not None:
                pr = self.stereo.pair()
                if pr is not None:
                    f0, f1, t = pr
            elif self.cam is not None:
                f, t, s = self.cam.wait_new(seq, timeout=0.2)
                if s != seq:
                    f0, seq = f, s
            else:
                time.sleep(0.05)

            if self.want_start:
                self.want_start = False
                self._start()
            if self.ht is not None and f0 is not None:
                self._step(f0, f1, t)
            if f0 is not None:
                with self.lock:
                    self.frame, self.seq = f0, self.seq + 1

    def _start(self):
        if self.stereo is None:
            self._set_status("Needs both cameras", self.stereo_err or "", "error")
            return
        if self.pub is None:
            self._set_status("Can't control the arm: ROS 2 isn't available",
                             f"{self.ros_err}. Source /opt/ros/jazzy/setup.bash first, "
                             "or pass --no-ros to track without publishing", "error")
            return
        if self.ht is not None:
            self.ht.recalibrate()
            print("hand recalibration started")
            return
        a = self.args
        try:
            self.rig = load_rig(a.rig)
            self.est = h3.StereoHand(self.rig, reproj_px=a.reproj_px, mono_fallback=a.mono_fallback)
            self.lms = StereoLandmarks()
            ht = HandTarget(self.est, self.pub, a)
            if a.reuse_calib:
                ht.start_tracking(load_hand())
            else:
                ht.recalibrate()
        except (ImportError, OSError, RuntimeError) as e:  # rig/hand file, mediapipe, or its model
            print(f"can't start: {e}", file=sys.stderr)
            self._set_status("Can't start", short(e), "error")
            return
        self.ht = ht
        print("hand loaded from calib/hand.json; tracking" if a.reuse_calib
              else "hand calibration started")

    def _step(self, f0, f1, t):
        px0, px1 = self.lms(f0, f1, t)
        pose, why = self.est.estimate(px0, px1)
        status, published = self.ht.step(pose, why, t)
        draw_hand(f0, px0)
        if pose is not None:
            draw_axes(f0, pose, self.rig["K0"], self.rig["d0"])

        if self.ht.calib is not None:
            self._set_status(status, why)
        elif published:
            self._set_status("Not publishing (--no-ros)" if self.args.no_ros else "Controlling the arm",
                             status, "track")
        elif self.ht.target is not None:
            self._set_status("Hand lost: arm holding", status, "hold")
        else:  # just calibrated; the next frame with a hand starts the arm
            self._set_status("Hand calibrated", "show your hand to start", "hold")

    def close(self):
        self.running = False
        self.thread.join(timeout=2.0)
        if self.lms is not None:
            self.lms.close()
        for c in (self.stereo, self.cam):
            if c is not None:
                c.close()
        if self.pub is not None:
            self.pub.close()
