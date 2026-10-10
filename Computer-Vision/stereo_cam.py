#!/home/peyton/kh/.venv/bin/python
"""Two-camera capture + MediaPipe hand landmarks per view, and the stereo rig file.

Each camera runs in its own thread and keeps its latest two frames with
capture timestamps. pair() waits for a new frame from cam0 and pairs it with
cam1's nearest frame in time (previous, latest, or the next one if that will
be closer); the pair is rejected if they're more than max_dt apart. The
cameras aren't hardware-synced (both free-run at 30 fps, so a pair is up to
~17 ms apart); hand3d.SkewCorrector shifts cam1's landmarks to cam0's time for the rest.

Camera settings (exposure, focus, white balance) persist in the camera between
runs, so configure() sets them explicitly every start. Defaults: auto exposure
and auto white balance (they don't affect the stereo geometry, and MediaPipe
needs a bright, natural-colored hand: a locked 4600 K white balance under warm
room light turned the image green and a 20 ms exposure cap left it dark, and
MediaPipe then found the hand in 0% of frames vs 86% on auto, 2026-10-10).
Focus is locked when the rig records one (autofocus hunting changes the focal
length the calibration measured). Dynamic framerate is always off (auto
exposure may not drop below 30 fps). The driver reports stale values in auto
mode, so "lock what auto chose" isn't possible.

Cameras: two Adesso CyberTrack H4 show up as /dev/video0 and /dev/video2
(video1/video3 are their metadata nodes). In WSL attach them with usbipd.

Preview (no calibration needed; checks both cameras + landmarks):
    .venv/bin/python Computer-Vision/stereo_cam.py [--cams 0 2] [--size 640 480]
    q / Esc quits. f = focus sweep: hold an open hand still where you'll use it;
    prints the sharpest --focus value (for stereo_calib.py).
"""
import argparse
import fcntl
import os
import struct
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")
RIG_PATH = os.path.join(HERE, "calib", "stereo.npz")
BASELINE_M = 0.20  # cameras mounted 20 cm apart, lens to lens (stereo_calib.py --baseline-in)

EXPOSURE_TARGET = 100    # mean gray level configure(exposure="match") aims for (0..255)
MAX_EXPOSURE = 300       # 100 us units = 30 ms (> 33 ms drops below 30 fps)
FOCUS_RANGE = (0, 1023)  # CyberTrack H4 focus_absolute
MIN_FPS = 20.0           # configure() warns below this (measured per camera)

V4L2_CID_EXPOSURE_AUTO_PRIORITY = 0x009A0903  # "Exposure, Dynamic Framerate"; not in OpenCV
VIDIOC_S_CTRL = (3 << 30) | (8 << 16) | (ord("V") << 8) | 28


def v4l2_set_ctrl(index, cid, value):
    """Set a V4L2 control OpenCV has no property for. False if it failed."""
    try:
        fd = os.open(f"/dev/video{index}", os.O_RDWR)
        try:
            fcntl.ioctl(fd, VIDIOC_S_CTRL, bytearray(struct.pack("Ii", cid, value)))
        finally:
            os.close(fd)
        return True
    except OSError:
        return False

HAND_CONNECTIONS = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8),
                    (5, 9), (9, 10), (10, 11), (11, 12), (9, 13), (13, 14), (14, 15),
                    (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]


class Camera:
    def __init__(self, index, width=640, height=480, fps=30):
        self.index = index
        self.cap = cv2.VideoCapture(index, cv2.CAP_V4L2)
        if not self.cap.isOpened():
            raise RuntimeError(f"cannot open /dev/video{index} (attached with usbipd?)")
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        self.cap.set(cv2.CAP_PROP_FPS, fps)
        # 2, not 1: with a single V4L2 buffer the camera has nowhere to put the next
        # frame while one is being read, and every other frame is dropped (15 vs 30
        # fps measured). _run drains continuously, so 2 adds no latency.
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 2)
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        if self.size != (width, height):
            self.cap.release()
            raise RuntimeError(f"/dev/video{index} gave {self.size}, asked for {width}x{height}")
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
                    self.period += 0.1 * (min(max(t - self.t, 1 / 60), 0.2) - self.period)
                self.prev, self.prev_t = self.frame, self.t
                self.frame, self.t, self.seq = f, t, self.seq + 1
                self.cond.notify_all()

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

    def set_controls(self, exposure=None, focus=None, wb=None):
        """None = auto. exposure in 100 us units, focus 0..1023, wb in kelvin.
        Auto is set explicitly too: the camera keeps the last run's settings."""
        c = self.cap
        # dynamic framerate lets the camera drop below 30 fps for longer exposures
        # (both cameras had it on, persisted from somewhere); always off
        v4l2_set_ctrl(self.index, V4L2_CID_EXPOSURE_AUTO_PRIORITY, 0)
        if exposure is None:
            c.set(cv2.CAP_PROP_AUTO_EXPOSURE, 3)  # V4L2 aperture priority = auto
        else:
            c.set(cv2.CAP_PROP_AUTO_EXPOSURE, 1)  # manual
            c.set(cv2.CAP_PROP_EXPOSURE, float(exposure))
        c.set(cv2.CAP_PROP_GAIN, 0)
        if focus is None:
            c.set(cv2.CAP_PROP_AUTOFOCUS, 1)
        else:
            c.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            c.set(cv2.CAP_PROP_FOCUS, float(focus))
        if wb is None:
            c.set(cv2.CAP_PROP_AUTO_WB, 1)
        else:
            c.set(cv2.CAP_PROP_AUTO_WB, 0)
            c.set(cv2.CAP_PROP_WB_TEMPERATURE, float(wb))

    def settled_frame(self, frames=5):
        """A frame captured after `frames` new ones (lets a setting take effect)."""
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


def _gray_mean(f):
    return float(cv2.cvtColor(f[::4, ::4], cv2.COLOR_BGR2GRAY).mean())


class StereoCam:
    def __init__(self, cams=(0, 2), size=(640, 480), **kw):
        self.c0 = Camera(cams[0], *size, **kw)
        try:
            self.c1 = Camera(cams[1], *size, **kw)
        except Exception:
            self.c0.close()
            raise
        self.size = self.c0.size
        self._seq = 0
        self.t0 = self.t1 = 0.0  # capture times of the last pair

    def configure(self, exposure=None, focus=None, wb=None,
                  target=EXPOSURE_TARGET, max_exposure=MAX_EXPOSURE):
        """Set both cameras alike. exposure: None (auto, default), "match" (the
        fixed exposure giving `target` brightness, capped at max_exposure) or a
        number (100 us units). focus / wb: a number or None (auto). Returns a
        summary line."""
        if exposure != "match":
            for c in (self.c0, self.c1):
                c.set_controls(exposure, focus, wb)
            return (f"exposure {exposure or 'auto'}, focus {focus or 'auto'}, wb {wb or 'auto'}"
                    + self._fps_check())
        e, b = 100.0, 0.0
        for _ in range(6):  # brightness ~ exposure, so a few scaled steps converge
            for c in (self.c0, self.c1):
                c.set_controls(e, focus, wb)
            # the camera ramps to a new exposure over several frames (5 read ~60% high)
            b = 0.5 * (_gray_mean(self.c0.settled_frame(15)) + _gray_mean(self.c1.settled_frame(15)))
            if abs(b - target) < 8:
                break
            e_new = min(max(e * target / max(b, 1.0), 3.0), max_exposure)
            if e_new == e:
                break
            e = e_new
        msg = (f"exposure {e:.0f} ({e / 10:.1f} ms, brightness {b:.0f}, target {target}), "
               f"focus {focus or 'auto'}, wb {wb or 'auto'}")
        if b < 0.6 * target:
            msg += f"  DARK: exposure is capped at {max_exposure / 10:.0f} ms, add light"
        return msg + self._fps_check()

    def _fps_check(self):
        """Frame rates after configuring. Bright, detailed images make bigger MJPEG
        frames: at 1280x720 cam0 drops to ~5 fps (measured through usbipd), 640x480 is fine."""
        fps = [c.measure_fps(0.7) for c in (self.c0, self.c1)]
        msg = f"; fps {fps[0]:.0f}/{fps[1]:.0f}"
        if min(fps) < MIN_FPS:
            msg += (f"  SLOW CAMERA (< {MIN_FPS:.0f} fps): USB bandwidth at this size? "
                    "use 640x480 or a shorter --exposure")
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
            # period; longer only when cam1 dropped a frame, ~25 fps here)
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


class Landmarker:
    """MediaPipe HandLandmarker (Tasks API, VIDEO mode), one hand."""

    def __init__(self, min_conf=0.5):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"{MODEL_PATH} missing. Download it:\n"
                                    f"  curl -L -o {MODEL_PATH} {MODEL_URL}")
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO, num_hands=1,
            min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf)
        self.lm = vision.HandLandmarker.create_from_options(opts)
        self.last_ms = -1

    def detect(self, frame_bgr, t):
        """(21,2) landmark pixels or None."""
        import mediapipe as mp
        ms = max(int(t * 1000), self.last_ms + 1)  # VIDEO mode needs increasing stamps
        self.last_ms = ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        res = self.lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)
        if not res.hand_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        return np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[0]])

    def close(self):
        self.lm.close()


class StereoLandmarks:
    """One Landmarker per camera, run in parallel."""

    def __init__(self, min_conf=0.5):
        self.lms = (Landmarker(min_conf), Landmarker(min_conf))
        self.pool = ThreadPoolExecutor(2)

    def __call__(self, f0, f1, t):
        a = self.pool.submit(self.lms[0].detect, f0, t)
        b = self.pool.submit(self.lms[1].detect, f1, t)
        return a.result(), b.result()

    def close(self):
        self.pool.shutdown()
        for lm in self.lms:
            lm.close()


def draw_hand(img, px, color=(0, 255, 0)):
    if px is None:
        return
    pts = [tuple(int(v) for v in p) for p in px]
    for a, b in HAND_CONNECTIONS:
        cv2.line(img, pts[a], pts[b], color, 2)
    for p in pts:
        cv2.circle(img, p, 3, (0, 0, 255), -1)


def fit_view(img, max_w=1600):
    """Downscale a side-by-side preview that would be wider than the screen."""
    if img.shape[1] <= max_w:
        return img
    s = max_w / img.shape[1]
    return cv2.resize(img, None, fx=s, fy=s, interpolation=cv2.INTER_AREA)


def hand_box(px, size, pad=0.25):
    """(x0, y0, x1, y1) pixel box around landmarks, padded."""
    lo, hi = px.min(axis=0), px.max(axis=0)
    d = (hi - lo) * pad
    lo, hi = np.maximum(lo - d, 0), np.minimum(hi + d, size)
    return int(lo[0]), int(lo[1]), int(hi[0]), int(hi[1])


def focus_sweep(cam, box0, box1, steps=17):
    """Sharpest focus, same on both cameras: Laplacian variance inside each
    camera's hand box, summed. Leaves the best focus set. Returns (focus, scores)."""
    scores = []
    for f in np.linspace(*FOCUS_RANGE, steps).astype(int):
        for c in (cam.c0, cam.c1):
            c.cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)
            c.cap.set(cv2.CAP_PROP_FOCUS, float(f))
        s = 0.0
        for c, (x0, y0, x1, y1) in ((cam.c0, box0), (cam.c1, box1)):
            g = cv2.cvtColor(c.settled_frame(6)[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY)
            s += cv2.Laplacian(g, cv2.CV_64F).var()
        scores.append((int(f), s))
    best = max(scores, key=lambda x: x[1])[0]
    for c in (cam.c0, cam.c1):
        c.cap.set(cv2.CAP_PROP_FOCUS, float(best))
    return best, scores


# ---------- rig file ----------

def save_rig(path, K0, d0, K1, d1, R, t, size, method, focus=None):
    """focus: the locked focus the calibration was done at (None = autofocus).
    The intrinsics only hold at that focus, so the node reuses it."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, K0=K0, d0=d0, K1=K1, d1=d1, R=R, t=t, size=np.array(size), method=method,
             focus=-1 if focus is None else int(focus))


def load_rig(path=RIG_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run stereo_calib.py first")
    z = np.load(path)
    return {k: z[k] for k in z.files}


def rig_focus(rig):
    """The focus the rig was calibrated at, or None (autofocus / older rig file)."""
    f = int(rig.get("focus", -1))
    return None if f < 0 else f


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cams", type=int, nargs=2, default=[0, 2])
    ap.add_argument("--size", type=int, nargs=2, default=[640, 480], metavar=("W", "H"))
    ap.add_argument("--focus", type=int, help="locked focus 0..1023 (default autofocus)")
    args = ap.parse_args()
    cam = StereoCam(args.cams, size=args.size)
    print("cameras:", cam.configure(focus=args.focus))
    lms = StereoLandmarks()
    n, t_fps, fps = 0, time.monotonic(), 0.0
    try:
        while True:
            pr = cam.pair()
            if pr is None:
                continue
            f0, f1, t = pr
            p0, p1 = lms(f0, f1, t)
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
            cv2.imshow("stereo preview (q quits, f focus sweep)", fit_view(view))
            k = cv2.waitKey(1) & 0xFF
            if k in (ord("q"), 27):
                break
            if k == ord("f"):
                if p0 is None or p1 is None:
                    print("focus sweep needs the hand in both views")
                    continue
                print("focus sweep: hold the hand still...")
                best, scores = focus_sweep(cam, hand_box(p0, cam.size), hand_box(p1, cam.size))
                print("  " + "  ".join(f"{f}:{s:.0f}" for f, s in scores))
                print(f"sharpest: --focus {best} (locked now; use it for stereo_calib.py)")
    finally:
        lms.close()
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
