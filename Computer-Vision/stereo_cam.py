#!/home/peyton/kh/.venv/bin/python
"""Two-camera capture + MediaPipe hand landmarks per view, and the stereo rig file.

Each camera runs in its own thread and keeps only its latest frame with a
capture timestamp. pair() waits for a new frame from cam0 and takes cam1's
latest; the pair is rejected if they're more than max_dt apart (the cameras
aren't hardware-synced, so fast motion still skews triangulation a bit).

Cameras: two Adesso CyberTrack H4 show up as /dev/video0 and /dev/video2
(video1/video3 are their metadata nodes). In WSL attach them with usbipd.

Preview (no calibration needed; checks both cameras + landmarks):
    .venv/bin/python Computer-Vision/stereo_cam.py [--cams 0 2]
    q / Esc quits.
"""
import argparse
import os
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
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        self.size = (int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                     int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        self.cond = threading.Condition()
        self.frame, self.t, self.seq = None, 0.0, 0
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
                self.frame, self.t, self.seq = f, t, self.seq + 1
                self.cond.notify_all()

    def latest(self):
        with self.cond:
            return self.frame, self.t, self.seq

    def wait_new(self, last_seq, timeout=1.0):
        with self.cond:
            self.cond.wait_for(lambda: self.seq != last_seq, timeout)
            return self.frame, self.t, self.seq

    def close(self):
        self.running = False
        self.thread.join(timeout=1.0)
        self.cap.release()


class StereoCam:
    def __init__(self, cams=(0, 2), **kw):
        self.c0 = Camera(cams[0], **kw)
        try:
            self.c1 = Camera(cams[1], **kw)
        except Exception:
            self.c0.close()
            raise
        if self.c0.size != self.c1.size:
            raise RuntimeError(f"camera sizes differ: {self.c0.size} vs {self.c1.size}")
        self.size = self.c0.size
        self._seq = 0

    def pair(self, max_dt=0.05, timeout=1.0):
        """(f0, f1, t) with a new cam0 frame, or None (timeout / too far apart)."""
        f0, t0, self._seq = self.c0.wait_new(self._seq, timeout)
        f1, t1, s1 = self.c1.latest()
        if f1 is not None and t0 - t1 > 0.5 * max_dt:
            # cam1's frame is old: its next one is likely closer to t0
            g1, u1, _ = self.c1.wait_new(s1, timeout=2 * max_dt)
            if g1 is not None and abs(u1 - t0) < abs(t1 - t0):
                f1, t1 = g1, u1
        if f0 is None or f1 is None or abs(t0 - t1) > max_dt:
            return None
        return f0, f1, 0.5 * (t0 + t1)

    def close(self):
        self.c0.close()
        self.c1.close()


class Landmarker:
    """MediaPipe HandLandmarker (Tasks API, VIDEO mode), up to num_hands hands."""

    def __init__(self, min_conf=0.5, num_hands=1):
        from mediapipe.tasks.python import vision
        from mediapipe.tasks.python.core.base_options import BaseOptions
        if not os.path.exists(MODEL_PATH):
            raise FileNotFoundError(f"{MODEL_PATH} missing. Download it:\n"
                                    f"  curl -L -o {MODEL_PATH} {MODEL_URL}")
        opts = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=MODEL_PATH),
            running_mode=vision.RunningMode.VIDEO, num_hands=num_hands,
            min_hand_detection_confidence=min_conf, min_hand_presence_confidence=min_conf,
            min_tracking_confidence=min_conf)
        self.lm = vision.HandLandmarker.create_from_options(opts)
        self.last_ms = -1

    def _run(self, frame_bgr, t):
        import mediapipe as mp
        ms = max(int(t * 1000), self.last_ms + 1)  # VIDEO mode needs increasing stamps
        self.last_ms = ms
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        return self.lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb), ms)

    def detect(self, frame_bgr, t):
        """(21,2) landmark pixels of the first hand, or None."""
        res = self._run(frame_bgr, t)
        if not res.hand_landmarks:
            return None
        h, w = frame_bgr.shape[:2]
        return np.array([[p.x * w, p.y * h] for p in res.hand_landmarks[0]])

    def detect_all(self, frame_bgr, t):
        """[(px (21,2), MediaPipe handedness label "Left"/"Right", score), ...]."""
        res = self._run(frame_bgr, t)
        h, w = frame_bgr.shape[:2]
        return [(np.array([[p.x * w, p.y * h] for p in lm]), hd[0].category_name, hd[0].score)
                for lm, hd in zip(res.hand_landmarks, res.handedness)]

    def close(self):
        self.lm.close()


class StereoLandmarks:
    """One Landmarker per camera, run in parallel."""

    def __init__(self, min_conf=0.5, num_hands=1):
        self.lms = (Landmarker(min_conf, num_hands), Landmarker(min_conf, num_hands))
        self.pool = ThreadPoolExecutor(2)

    def __call__(self, f0, f1, t):
        a = self.pool.submit(self.lms[0].detect, f0, t)
        b = self.pool.submit(self.lms[1].detect, f1, t)
        return a.result(), b.result()

    def all(self, f0, f1, t):
        """Per camera, every detected hand: see Landmarker.detect_all."""
        a = self.pool.submit(self.lms[0].detect_all, f0, t)
        b = self.pool.submit(self.lms[1].detect_all, f1, t)
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


# ---------- rig file ----------

def save_rig(path, K0, d0, K1, d1, R, t, size, method):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    np.savez(path, K0=K0, d0=d0, K1=K1, d1=d1, R=R, t=t, size=np.array(size), method=method)


def load_rig(path=RIG_PATH):
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run stereo_calib.py first")
    z = np.load(path)
    return {k: z[k] for k in z.files}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cams", type=int, nargs=2, default=[0, 2])
    args = ap.parse_args()
    cam = StereoCam(args.cams)
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
            cv2.putText(view, f"{fps:.1f} pairs/s  cam0:{'hand' if p0 is not None else '-'}  "
                        f"cam1:{'hand' if p1 is not None else '-'}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("stereo preview (q quits)", view)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        lms.close()
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
