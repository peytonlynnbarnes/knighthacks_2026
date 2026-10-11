#!/usr/bin/env python3
"""Two-camera capture (macOS, AVFoundation), the stereo rig file, and a preview with hand
landmarks per view (the landmark backend, WiLoR, lives in ../WiLoR/wilor_landmarks.py).

Each camera runs in its own thread and keeps its latest two frames with
capture timestamps. pair() waits for a new frame from cam0 and pairs it with
cam1's nearest frame in time (previous, latest, or the next one if that will
be closer); the pair is rejected if they're more than max_dt apart. The
cameras aren't hardware-synced (both free-run at CAM_FPS, 60: a pair is up to
~8 ms apart); hand3d.SkewCorrector shifts cam1's landmarks to cam0's time for the rest.

Camera settings: macOS (AVFoundation through OpenCV) can't set exposure, white
balance or focus, so the cameras run on auto exposure / auto white balance,
which is what this project wants anyway (locking them broke hand detection on
Linux, 2026-10-10). The Adesso CyberTrack H4s have manual focus rings: set them
by hand with the sharpness numbers in the preview, tape them, and recalibrate
the rig after turning one.

Cameras: AVFoundation numbers them 0, 1, 2, ... in its own order. The MacBook's
built-in camera and an iPhone (Continuity Camera) take numbers too, and two
identical H4s can swap between plug-ins. Find them with --list, then pass
--cams LEFT RIGHT (cam0 = the left camera, cam1 = the one to its right, as
calibrated) to every script. Each camera on its own USB port, no shared hub.
The terminal app needs camera access (System Settings > Privacy & Security >
Camera) the first time.

Preview (no calibration needed; checks both cameras + landmarks):
    python Computer-Vision/common/stereo_cam.py --list        # which index is which camera
    python Computer-Vision/common/stereo_cam.py --cams 1 2 [--size 640 480]
    q / Esc quits. The sharpness of the center of each view is shown: turn a
    focus ring until its number peaks, with the same textured target in both views.
"""
import argparse
import os
import sys
import threading
import time

import cv2
import numpy as np

from timing import NULL as NULL_TIMERS

HERE = os.path.dirname(os.path.abspath(__file__))
CV_DIR = os.path.dirname(HERE)  # Computer-Vision/: common/ (this), WiLoR/
BACKENDS = ("wilor",)  # landmark backends: ../WiLoR/
DEFAULT_CAMS = [1, 2]  # the built-in camera is usually 0; check with --list


def make_landmarks(backend="wilor", **kw):
    """The landmark backend: StereoLandmarks(f0, f1, t) -> (px0, px1), (21, 2) pixels
    per view or None. wilor = ../WiLoR/wilor_landmarks.py (MANO model on the Apple GPU,
    MPS; kw: hand, yolo_every, fp32, warmup_size = the capture size: MPS compiles
    kernels per input shape, so a mismatch stalls ~0.3 s when the hand first appears)."""
    if backend not in BACKENDS:
        raise ValueError(f"backend {backend!r}: one of {BACKENDS}")
    d = os.path.join(CV_DIR, "WiLoR")
    if d not in sys.path:
        sys.path.insert(0, d)
    try:
        from wilor_landmarks import StereoLandmarks
        return StereoLandmarks(**kw)
    except ImportError as e:
        raise SystemExit(f"{backend} backend: {e}. Install torch, ultralytics and "
                         "Computer-Vision/WiLoR/repo/requirements.txt into the conda env (README.md)")

RIG_PATH = os.path.join(HERE, "calib", "stereo.npz")
BASELINE_M = 0.21  # cameras mounted 21 cm apart, lens to lens (measured 2026-10-10) (stereo_calib.py --baseline-in)

CAM_FPS = 60      # requested capture rate (Elgato Facecam MK.2: 60 fps at 1280x720 and 1920x1080)
CAPTURE_SIZE = (1920, 1080)  # live capture size; the rig file is scaled to it (scale_rig)
MIN_FPS = 20.0    # configure() warns below this (measured per camera)
OPEN_WAIT_S = 3.0  # first frame after opening (camera start-up, the permission prompt)

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


def open_capture(index, width, height, fps=CAM_FPS):
    """(cv2.VideoCapture, first frame) for an AVFoundation camera at width x height,
    or (None, None) if it doesn't open or delivers nothing within OPEN_WAIT_S."""
    cap = cv2.VideoCapture(index, cv2.CAP_AVFOUNDATION)
    if not cap.isOpened():
        return None, None
    # AVFoundation picks the pixel format and buffering itself (no MJPG fourcc,
    # no buffer size); width/height select its capture preset
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, fps)
    t_end = time.monotonic() + OPEN_WAIT_S
    while time.monotonic() < t_end:
        ok, f = cap.read()
        if ok and f is not None:
            return cap, f
        time.sleep(0.05)
    cap.release()
    return None, None


class Camera:
    def __init__(self, index, width=640, height=480, fps=CAM_FPS, timers=NULL_TIMERS, name=None):
        self.index = index
        self.timers, self._tick = timers, f"{name or f'camera {index}'} frames"

        self.cap, f = open_capture(index, width, height, fps)
        if self.cap is None:
            raise RuntimeError(f"camera {index}: no frames (plugged in? camera access allowed for this "
                               "terminal? find the indices with stereo_cam.py --list)")
        self.size = (f.shape[1], f.shape[0])  # what AVFoundation actually delivers
        if self.size != (width, height):
            self.cap.release()
            raise RuntimeError(f"camera {index} gave {self.size[0]}x{self.size[1]}, asked for {width}x{height}")
        self.cond = threading.Condition()
        self.frame, self.t, self.seq = None, 0.0, 0
        self.prev, self.prev_t = None, 0.0
        self.period = 1.0 / fps  # measured frame interval (EMA)
        self.running = True
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        while self.running:
            ok, f = self.cap.read()
            t = time.monotonic()
            if not ok:
                time.sleep(0.01)
                continue
            with self.cond:
                if self.frame is not None:
                    self.period += 0.1 * (min(max(t - self.t, 1 / 120), 0.2) - self.period)
                self.prev, self.prev_t = self.frame, self.t
                self.frame, self.t, self.seq = f, t, self.seq + 1
                self.cond.notify_all()
            self.timers.tick(self._tick)

    def latest(self):
        with self.cond:
            return self.frame, self.t, self.seq

    def latest2(self):
        """(frame, t, seq, previous frame, its t)."""
        with self.cond:
            return self.frame, self.t, self.seq, self.prev, self.prev_t

    def wait_new(self, last_seq, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq, timeout)
            return self.frame, self.t, self.seq

    def settled_frame(self, frames=5):
        """A frame captured after `frames` new ones."""
        _, _, seq = self.latest()
        f = None
        for _ in range(frames):
            f, _, seq = self.wait_new(seq)
        return f

    def measure_fps(self, secs=1.0):
        _, _, s = self.latest()
        t = time.monotonic()
        time.sleep(secs)
        return (self.latest()[2] - s) / (time.monotonic() - t)

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


class StereoCam:
    def __init__(self, cams=DEFAULT_CAMS, size=(640, 480), **kw):
        if cams[0] == cams[1]:
            raise RuntimeError(f"--cams {cams[0]} {cams[1]}: two different cameras needed")
        self.c0 = Camera(cams[0], *size, name="cam0", **kw)  # timers: by role, not AVFoundation index
        try:
            self.c1 = Camera(cams[1], *size, name="cam1", **kw)
        except Exception:
            self.c0.close()
            raise
        self.size = self.c0.size
        self._seq = 0
        self.t0 = self.t1 = 0.0  # capture times of the last pair

    def configure(self):
        """Nothing to set on macOS (auto exposure / white balance, manual focus rings).
        Returns a summary line with the measured frame rates."""
        return "exposure auto, wb auto, focus manual (rings)" + self._fps_check()

    def _fps_check(self):
        fps = [c.measure_fps(0.7) for c in (self.c0, self.c1)]
        msg = f"; fps {fps[0]:.0f}/{fps[1]:.0f}"
        if min(fps) < MIN_FPS:
            msg += (f"  SLOW CAMERA (< {MIN_FPS:.0f} fps): both on one USB hub? a dark room (auto "
                    "exposure gets longer)? a size the camera can't do at --cam-fps?")
        return msg

    def pair(self, max_dt=0.04, timeout=1.0):
        """(f0, f1, t) with a new cam0 frame and cam1's nearest frame in time, or
        None (timeout / too far apart). t = cam0's capture time; self.t0 and
        self.t1 are both capture times (for hand3d.SkewCorrector)."""
        f0, t0, self._seq = self.c0.wait_new(self._seq, timeout)
        if f0 is None:
            return None
        f1, t1, s1, pf1, pt1 = self.c1.latest2()
        cands = [(f1, t1), (pf1, pt1)]
        if f1 is not None and t0 - t1 > 0.5 * self.c1.period:
            # cam1's next frame is due closer to t0 than its latest one: wait for it,
            # at most until it could still pair (it's normally there within half a
            # period; longer only when cam1 dropped a frame)
            wait = t0 + max_dt - time.monotonic()
            if wait > 0:
                g1, u1, s = self.c1.wait_new(s1, timeout=wait)
                if s != s1:
                    cands.append((g1, u1))
        cands = [c for c in cands if c[0] is not None]
        if not cands:
            return None
        f1, t1 = min(cands, key=lambda c: abs(c[1] - t0))
        if abs(t0 - t1) > max_dt:
            return None
        self.t0, self.t1 = t0, t1
        return f0, f1, t0

    def close(self):
        self.c0.close()
        self.c1.close()


def draw_hand(img, px, color=(0, 255, 0)):
    if px is None:
        return
    pts = [tuple(int(v) for v in p) for p in px]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(img, p, 3, (0, 0, 255), -1)


def fit_view(img, max_w=1600):
    """Downscale a side-by-side preview that would be wider than the screen. Linear, not
    INTER_AREA: 2560x720 -> 1600 took 15 ms with INTER_AREA (a quarter of a 60 fps frame)."""
    if img.shape[1] <= max_w:
        return img
    s = max_w / img.shape[1]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_LINEAR)


def sharpness(img):
    """Laplacian variance of the center half of the image: higher = sharper. Only
    comparable between views of the same textured target (a board, printed text)."""
    h, w = img.shape[:2]
    g = cv2.cvtColor(img[h // 4:3 * h // 4, w // 4:3 * w // 4], cv2.COLOR_BGR2GRAY)
    return float(cv2.Laplacian(g, cv2.CV_64F).var())


def list_cameras(size=(640, 480), max_index=8):
    """Open every AVFoundation index, print what delivers frames and show them labeled."""
    found = []
    for i in range(max_index):
        cap, f = open_capture(i, *size)
        if cap is None:
            continue
        print(f"camera {i}: {f.shape[1]}x{f.shape[0]}")
        cap.release()
        f = cv2.resize(f, (320, 240))
        cv2.putText(f, f"--cams index {i}", (8, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        found.append(f)
    if not found:
        print("no camera delivered frames (camera access allowed for this terminal?)")
        return
    print("cover one lens to tell the two H4s apart; cam0 (first --cams value) = the left camera")
    cv2.imshow("cameras (any key closes)", np.hstack(found))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# ---------- rig file ----------

def save_rig(path, K0, d0, K1, d1, R, t, size, method):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    # focus -1: no software focus (manual rings); older rig files store the V4L2 value
    np.savez(path, K0=K0, d0=d0, K1=K1, d1=d1, R=R, t=t, size=np.array(size), method=method, focus=-1)


def load_rig(path=RIG_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run stereo_calib.py first")
    z = np.load(path)
    return {k: z[k] for k in z.files}


def scale_rig(rig, size):
    """The rig at another capture size with the same aspect ratio. The Elgato Facecam MK.2's
    16:9 modes are the same view at another scale (1280x720 -> 1920x1080 measured by feature
    matching: x1.4999 / x1.4998, offset 0.25 px = pixel centers, 2026-10-11), so the
    intrinsics scale (pixel centers kept) and distortion, R, t stay. Another aspect ratio is
    a different crop of the sensor: recalibrate at that size instead. rig["calib_size"] keeps
    the calibrated size (px settings tuned there scale with size / calib_size)."""
    w0, h0 = (int(v) for v in rig["size"])
    w, h = (int(v) for v in size)
    out = dict(rig)
    out.setdefault("calib_size", np.array((w0, h0)))
    if (w, h) == (w0, h0):
        return out
    s = w / w0
    if abs(h / h0 - s) > 1e-3:
        raise ValueError(f"rig calibrated at {w0}x{h0}, capture {w}x{h}: another aspect ratio "
                         "(a different crop) needs its own calibration (stereo_calib.py --size W H)")
    for k in ("K0", "K1"):
        K = np.array(rig[k], dtype=float)
        K[0, 0] *= s
        K[0, 1] *= s
        K[1, 1] *= s
        K[0, 2] = s * (K[0, 2] + 0.5) - 0.5
        K[1, 2] = s * (K[1, 2] + 0.5) - 0.5
        out[k] = K
    out["size"] = np.array((w, h))
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cams", type=int, nargs=2, default=DEFAULT_CAMS, metavar=("LEFT", "RIGHT"),
                    help="AVFoundation indices: cam0 = left camera, cam1 = right (find them with --list)")
    ap.add_argument("--list", action="store_true", help="show every camera with its index, then quit")
    ap.add_argument("--size", type=int, nargs=2, default=list(CAPTURE_SIZE), metavar=("W", "H"))
    ap.add_argument("--backend", choices=BACKENDS, default="wilor", help="landmark backend")
    ap.add_argument("--no-landmarks", action="store_true",
                    help="cameras only (no WiLoR load): quick check, focusing the rings")
    args = ap.parse_args()
    if args.list:
        list_cameras(tuple(args.size))
        return
    cam = StereoCam(args.cams, size=args.size)
    print("cameras:", cam.configure())
    lms = None if args.no_landmarks else make_landmarks(args.backend, warmup_size=tuple(args.size))
    n, t_fps, fps = 0, time.monotonic(), 0.0
    try:
        while True:
            pr = cam.pair()
            if pr is None:
                continue
            f0, f1, t = pr
            sharp = (sharpness(f0), sharpness(f1))
            p0, p1 = lms(f0, f1, t) if lms is not None else (None, None)
            draw_hand(f0, p0)
            draw_hand(f1, p1)
            n += 1
            if time.monotonic() - t_fps > 1.0:
                fps, n, t_fps = n / (time.monotonic() - t_fps), 0, time.monotonic()
            view = np.hstack([f0, f1])
            cv2.putText(view, f"{fps:.1f} pairs/s  skew {(cam.t1 - cam.t0) * 1000:+.0f} ms  "
                        f"cam0:{'hand' if p0 is not None else '-'}  "
                        f"cam1:{'hand' if p1 is not None else '-'}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.putText(view, f"sharpness cam0 {sharp[0]:.0f}  cam1 {sharp[1]:.0f}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("stereo preview (q quits)", fit_view(view))
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
    finally:
        if lms is not None:
            lms.close()
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
