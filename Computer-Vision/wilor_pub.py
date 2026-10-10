"""Single-camera WiLoR hand viewer: like xy_pub.py, but WiLoR instead of MediaPipe.

Draws the 21 keypoints, a red dot at the palm center and the hand's orientation
axes (x red = wrist -> middle MCP, y green, z blue = palm normal), and prints the
palm, fingertips and roll/pitch/yaw for every hand, every frame.

The orientation comes from WiLoR's single-view 3D hand fit (plan B), not from
triangulating two cameras: hand3d.hand_frame() runs on WiLoR's 3D palm joints,
which MANO places from the hand's root rotation alone, so it is WiLoR's
rotation expressed in the same hand frame the rest of the pipeline uses.
roll/pitch/yaw here are relative to the camera (x right, y down, z forward).

Setup (once, in the venv). Install AMD's ROCm PyTorch first ("Install PyTorch
for Radeon GPUs on WSL"); --no-deps keeps WiLoR-mini from replacing it:
    pip install --no-deps git+https://github.com/warmshao/WiLoR-mini
    pip install smplx==0.1.28 timm einops ultralytics==8.1.34 huggingface_hub \
        scikit-image roma opencv-python dill
    pip install --no-build-isolation git+https://github.com/mattloper/chumpy
The first run downloads the models (~2.5 GB) from Hugging Face.

Run:
    .venv/bin/python Computer-Vision/wilor_pub.py             # camera 0
    .venv/bin/python Computer-Vision/wilor_pub.py --cam 2
    .venv/bin/python Computer-Vision/wilor_pub.py --bgr       # if keypoints look off
q quits.
"""
import argparse
import math
import time

import cv2
import numpy as np

import hand3d as h3
from stereo_cam import HAND_CONNECTIONS

# 0 = wrist, 5 = index MCP, 9 = middle MCP, 13 = ring MCP, 17 = pinky MCP
PALM_POINTS = [0, 5, 9, 13, 17]

FINGERTIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}

AXIS_M = 0.05  # drawn axis length (m)


class WiLoRHand:
    """One hand from WiLoR: 2D keypoints, palm center and 3D orientation."""

    def __init__(self, det, frame_shape):
        pred = det["wilor_preds"]
        self.is_right = bool(det["is_right"])
        # (21,2) pixels, same joint order as MediaPipe (OpenPose)
        self.points = np.asarray(pred["pred_keypoints_2d"][0], dtype=float)
        # (21,3) meters, camera axes; pred_cam_t_full is the translation WiLoR
        # projects with (its focal length is assumed, so the depth isn't metric)
        self.P = np.asarray(pred["pred_keypoints_3d"][0], dtype=float)
        self.cam_t = np.asarray(pred["pred_cam_t_full"], dtype=float).reshape(3)
        self.focal = float(pred["scaled_focal_length"])
        h, w = frame_shape[:2]
        self.center = np.array([w / 2, h / 2])
        self.origin, self.R = h3.hand_frame(self.P)
        self.palm_center = self._compute_palm_center()

    # grab the midpoint between all palm points for x
    # then midpoint between wrist and middle finger for y
    def _compute_palm_center(self):
        x = np.mean(self.points[PALM_POINTS, 0])
        y = (self.points[0, 1] + self.points[9, 1]) / 2
        return int(x), int(y)

    # project WiLoR 3D points (no translation yet) to pixels, the way WiLoR does
    def _project(self, X):
        X = np.asarray(X, dtype=float) + self.cam_t
        return X[:, :2] / X[:, 2:3] * self.focal + self.center

    # roll, pitch, yaw (deg) of the hand frame relative to the camera
    def rpy(self):
        return tuple(math.degrees(a) for a in h3.matrix_to_rpy(self.R))

    # draw the points, the palm dot and the orientation axes
    def draw(self, frame):
        pts = [tuple(int(v) for v in p) for p in self.points]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(frame, p, 3, (255, 255, 255), -1)
        o, x, y, z = self._project([self.origin] + [self.origin + AXIS_M * self.R[:, k] for k in range(3)])
        for tip, color in ((x, (0, 0, 255)), (y, (0, 255, 0)), (z, (255, 0, 0))):
            cv2.line(frame, tuple(int(v) for v in o), tuple(int(v) for v in tip), color, 3)
        cv2.circle(frame, self.palm_center, 8, (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, "R" if self.is_right else "L", (pts[0][0] + 10, pts[0][1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # get fingertips from hand
    def fingertips(self):
        return {name: tuple(int(v) for v in self.points[i]) for name, i in FINGERTIPS.items()}

    # print the palm, finger tips and orientation
    def print_coords(self, label):
        parts = [f"palm={self.palm_center}"]
        parts += [f"{name}={pt}" for name, pt in self.fingertips().items()]
        parts.append("rpy deg=({:+.0f}, {:+.0f}, {:+.0f})".format(*self.rpy()))
        print(f"{label} ({'right' if self.is_right else 'left'}): " + " ".join(parts))


class WiLoRTracker:
    """Wraps WiLoR-mini and returns WiLoRHand objects for each frame."""

    def __init__(self, rgb=True):
        import os
        import torch
        from wilor_mini.pipelines.wilor_hand_pose3d_estimation_pipeline import (
            WiLorHandPose3dEstimationPipeline)
        gpu = torch.cuda.is_available()  # ROCm builds of PyTorch show the AMD GPU as "cuda"
        if not gpu:
            print("WARNING: PyTorch sees no GPU, running WiLoR on the CPU (slow)")
        # WiLoR's detector.pt was pickled by an old ultralytics and fails PyTorch
        # >= 2.6's weights-only torch.load; allow a full load for WiLoR's own files only
        old = os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD")
        os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
        try:
            self.pipe = WiLorHandPose3dEstimationPipeline(
                device=torch.device("cuda" if gpu else "cpu"),
                dtype=torch.float16 if gpu else torch.float32, verbose=False)
        finally:
            if old is None:
                del os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"]
            else:
                os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = old
        self.rgb = rgb

    def process(self, frame):
        img = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB) if self.rgb else frame
        return [WiLoRHand(det, frame.shape) for det in self.pipe.predict(img)]

    def close(self):
        pass


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam", type=int, default=0, help="camera index (/dev/videoN)")
    ap.add_argument("--bgr", action="store_true", help="feed WiLoR BGR instead of RGB")
    args = ap.parse_args()

    cap = cv2.VideoCapture(args.cam)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    if not cap.isOpened():
        print("Cannot open camera")
        return

    tracker = WiLoRTracker(rgb=not args.bgr)
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            hands = tracker.process(frame)
            ms = (time.perf_counter() - t0) * 1000

            for i, hand in enumerate(hands):
                hand.draw(frame)
                hand.print_coords(f"hand {i}")

            cv2.putText(frame, f"WiLoR {ms:.0f} ms", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                        (0, 255, 255), 2)
            cv2.imshow("WiLoR Hand Tracking", frame)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
    finally:
        tracker.close()
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
