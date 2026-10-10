"""MediaPipe landmark backend: 21 hand landmarks (pixels) per camera view.

StereoLandmarks(f0, f1, t) -> (px0, px1), each a (21, 2) array in MediaPipe /
OpenPose joint order or None. common/hand3d.StereoHand takes it from there, so
any other backend (e.g. ../WiLoR) only has to match this call.
Model: models/hand_landmarker.task (gitignored; download URL below).
"""
import os
from concurrent.futures import ThreadPoolExecutor

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "models", "hand_landmarker.task")
MODEL_URL = ("https://storage.googleapis.com/mediapipe-models/hand_landmarker/"
             "hand_landmarker/float16/latest/hand_landmarker.task")


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
