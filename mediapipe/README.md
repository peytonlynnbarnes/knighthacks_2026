# mediapipe/ — hand landmarks

- `models/hand_landmarker.task`: MediaPipe Hand Landmarker, float16
  (palm detector + 21-landmark model, 7.8 MB). Source:
  `https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task`
  (sha256 `fbc2a300…cde1`)
- `hand_tracker.py`: `HandTracker` (BGR frame → list of `Hand`, with image
  and world landmarks) plus a CLI for webcam, video or image. No ROS.

```bash
.venv/bin/python mediapipe/hand_tracker.py                    # webcam 0, q quits
.venv/bin/python mediapipe/hand_tracker.py --source hand.jpg --num-hands 2 --out out.jpg
.venv/bin/python mediapipe/hand_tracker.py --source clip.mp4 --no-show --print
```

Uses the venv's mediapipe 1.1.0 (Tasks API; the old `mp.solutions` API is gone).
WSL has no `/dev/video*` by default. Attach the webcam with `usbipd`, or run
the tracker on Windows and send the results over.

Next step (see `INTEGRATION.md`, "How to wire CV in", option 2): a ROS node that
maps `Hand` → `/ball_position`.
