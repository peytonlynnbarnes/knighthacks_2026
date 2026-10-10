#!/usr/bin/env python3
"""MediaPipe hand landmarks from a webcam, video file or image. No ROS.

    .venv/bin/python mediapipe/hand_tracker.py                 # webcam 0
    .venv/bin/python mediapipe/hand_tracker.py --source 1      # another camera
    .venv/bin/python mediapipe/hand_tracker.py --source clip.mp4
    .venv/bin/python mediapipe/hand_tracker.py --source hand.jpg --out annotated.jpg
    add --no-show to skip the window (headless), --print to dump landmarks

`HandTracker` is the reusable part: feed it BGR frames, get `Hand`s back.
Landmark indices: 0 wrist, 4 thumb tip, 5/9/13/17 MCPs, 8 index tip, 12 middle tip
(https://ai.google.dev/edge/mediapipe/solutions/vision/hand_landmarker).
"""
import argparse
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

MODEL = Path(__file__).resolve().parent / "models" / "hand_landmarker.task"
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}
CONNECTIONS = [(c.start, c.end) for c in vision.HandLandmarksConnections.HAND_CONNECTIONS]


@dataclass
class Hand:
    handedness: str      # "Left" / "Right" as MediaPipe labels it (assumes a mirrored, selfie-view image)
    score: float
    image: np.ndarray    # (21, 3): x, y normalized to [0, 1] by image width/height; z relative depth (wrist ≈ 0, smaller = closer)
    world: np.ndarray    # (21, 3): meters, origin at the hand's approximate geometric center


class HandTracker:
    def __init__(self, num_hands=1, video=True, model=MODEL,
                 min_detection=0.5, min_presence=0.5, min_tracking=0.5):
        self.video = video
        self._t0 = time.monotonic()
        self._last_ms = -1
        self._landmarker = vision.HandLandmarker.create_from_options(vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model)),
            running_mode=vision.RunningMode.VIDEO if video else vision.RunningMode.IMAGE,
            num_hands=num_hands,
            min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_presence,
            min_tracking_confidence=min_tracking,
        ))

    def process(self, bgr, timestamp_ms=None):
        """BGR frame -> list[Hand]. In video mode timestamps must strictly increase."""
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
        if not self.video:
            result = self._landmarker.detect(image)
        else:
            if timestamp_ms is None:
                timestamp_ms = int((time.monotonic() - self._t0) * 1000)
            timestamp_ms = max(int(timestamp_ms), self._last_ms + 1)
            self._last_ms = timestamp_ms
            result = self._landmarker.detect_for_video(image, timestamp_ms)
        return [
            Hand(handedness=h[0].category_name, score=h[0].score,
                 image=np.array([[p.x, p.y, p.z] for p in lm]),
                 world=np.array([[p.x, p.y, p.z] for p in wlm]))
            for h, lm, wlm in zip(result.handedness, result.hand_landmarks, result.hand_world_landmarks)
        ]

    def close(self):
        self._landmarker.close()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def draw(bgr, hands):
    h, w = bgr.shape[:2]
    for hand in hands:
        pts = [(int(x * w), int(y * h)) for x, y, _ in hand.image]
        for a, b in CONNECTIONS:
            cv2.line(bgr, pts[a], pts[b], (255, 255, 255), 2)
        for p in pts:
            cv2.circle(bgr, p, 4, (0, 0, 255), -1)
        cv2.putText(bgr, f"{hand.handedness} {hand.score:.2f}", (pts[0][0], pts[0][1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return bgr


def print_hands(hands):
    for hand in hands:
        wx, wy, _ = hand.image[0]
        print(f"{hand.handedness:5s} {hand.score:.2f}  wrist=({wx:.3f}, {wy:.3f})  "
              f"index_tip=({hand.image[8, 0]:.3f}, {hand.image[8, 1]:.3f})")


def run_image(path, args):
    bgr = cv2.imread(str(path))
    if bgr is None:
        raise SystemExit(f"can't read image {path}")
    with HandTracker(num_hands=args.num_hands, video=False) as tracker:
        hands = tracker.process(bgr)
    print(f"{len(hands)} hand(s)")
    print_hands(hands)
    draw(bgr, hands)
    if args.out:
        cv2.imwrite(args.out, bgr)
    if not args.no_show:
        cv2.imshow("hands", bgr)
        cv2.waitKey(0)


def run_stream(source, args):
    cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    if not cap.isOpened():
        raise SystemExit(f"can't open {source!r} (WSL has no webcam unless it's attached with usbipd)")
    is_file = not source.isdigit()
    fps_n, fps_t = 0, time.monotonic()
    with HandTracker(num_hands=args.num_hands) as tracker:
        while True:
            ok, bgr = cap.read()
            if not ok:
                break
            if not is_file:
                bgr = cv2.flip(bgr, 1)  # selfie view, so handedness labels come out right
            ts = cap.get(cv2.CAP_PROP_POS_MSEC) if is_file else None
            hands = tracker.process(bgr, ts)
            if args.print:
                print_hands(hands)
            fps_n += 1
            if time.monotonic() - fps_t >= 1.0:
                print(f"{fps_n / (time.monotonic() - fps_t):.1f} fps, {len(hands)} hand(s)")
                fps_n, fps_t = 0, time.monotonic()
            if not args.no_show:
                cv2.imshow("hands (q to quit)", draw(bgr, hands))
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
    cap.release()
    cv2.destroyAllWindows()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default="0", help="camera index, video file or image (default: 0)")
    ap.add_argument("--num-hands", type=int, default=1)
    ap.add_argument("--no-show", action="store_true", help="no OpenCV window")
    ap.add_argument("--print", action="store_true", help="print landmarks every frame")
    ap.add_argument("--out", help="image mode: write the annotated image here")
    args = ap.parse_args()
    if Path(args.source).suffix.lower() in IMAGE_EXTS:
        run_image(args.source, args)
    else:
        run_stream(args.source, args)


if __name__ == "__main__":
    main()
