# Plan: WiLoR (MANO) hand model in the hand → arm pipeline

Written 2026-10-10 on `ik-improvements`. Contract and run order: `~/kh/INTEGRATION.md`.
Pipeline details, measured hardware and filter history: `~/kh/CLAUDE.md`.

## Status (2026-10-10)

| Phase | State |
|---|---|
| 0 measure | **done**: passes the gate, barely (below) |
| 1 backend | **done**: `WiLoR/wilor_landmarks.py`, self-test passes |
| 2 node wiring | **done**; run live with a real hand for the phase 3 recordings (11–16 pairs/s) |
| 3 compare | **done**: 5 takes, both backends, `common/recordings/wilor/README.md` |
| 4 retune | position filter **done** (separate WiLoR presets, below); gates and gripper not yet |
| 5–6 | not started |

Phase 4 filter result: `hand_mapping.IMM_PRESETS["wilor"]` / `KALMAN_DEFAULTS["wilor"]`, picked
automatically from `--backend` (and by a replay from its file). MediaPipe's presets are unchanged
(replays bit-identical). On the WiLoR takes vs the MediaPipe smooth preset: tracking error about
halved, lag 20–56 → 5–13 ms, still hand 0.53 → 0.22 mm in the full pipeline (= MediaPipe), but
~1.6× the acceleration while moving. Details: `common/recordings/wilor/README.md`.

Phase 0, `WiLoR/bench_wilor.py` on the RTX 2060 (fp16, cudnn.benchmark):

| | median | p90 | rate |
|---|---|---|---|
| WiLoR, 1 crop | 33.7 ms | 52 ms | 30 Hz |
| **WiLoR, 2 crops (both views)** | **47.5 ms** | 51 ms | **21 Hz** |
| WiLoR, 4 crops | 70 ms | 76 ms | 14 Hz |
| YOLO detector, 320 px | 9.4 ms | 11 ms | |
| peak VRAM | 4.4 GB | | |

`torch.compile`: no gain (47.0 ms). fp32: 136 ms (7 Hz). Attention is already
`scaled_dot_product_attention`: the ViT-L backbone is simply compute-bound
here. So WiLoR runs at ~20 Hz with one hand (MediaPipe gets ~31 pairs/s), and
two hands (4 crops) would be 14 Hz, which is too slow.

Phase 1 results:
- Self-test (`repo/demo_img`, same image in both views + a mirrored copy):
  8/8 images give 21 points inside the image, and the mirrored run matches
  the original to 0.2–1.3 % of the hand size, so the left-hand flip is right.
- Crops follow upstream: square, 2× the box, warped (zero padding at edges).
- No hand in view (12 s of live frames): YOLO gave 63 boxes ≥ 0.3 in 364 pairs
  (6 ≥ 0.5, never in both views together). Box tracking kept each alive for
  frames, so WiLoR reported hands in 113 view-frames. Hysteresis fixed most
  of it: a new track needs 0.5, an existing one is kept down to 0.3. Now 25
  view-frames, none in both views at once (MediaPipe on the same frames: 7).
  A one-view pose needs `--mono-fallback`, which is off, so these publish
  nothing. Check `conf_start` against a real hand in phase 2.
- Live idle run: the node runs on the cameras with `--backend wilor`, ~32 fps
  with no hand (detector only).

**Commands used for phases 2–3** (rerun to add takes):
```bash
source /opt/ros/jazzy/setup.bash
# 1. live check, print only: calibrate the hand, watch fps + tracking
Computer-Vision/WiLoR/.venv/bin/python Computer-Vision/common/hand_target_node.py --backend wilor --no-ros
# 2. phase 3 recordings: record each take from launch (still, sweep, reach, drop_out, edge_on)
.venv/bin/python Computer-Vision/common/hand_target_node.py --no-ros --record-frames ~/kh/frames_<take>
# 3. both backends on the same frames
.venv/bin/python Computer-Vision/common/hand_target_node.py --no-ros --no-view --replay-frames ~/kh/frames_<take> --log mp_<take>.npz
Computer-Vision/WiLoR/.venv/bin/python Computer-Vision/common/hand_target_node.py --backend wilor --no-ros --no-view --replay-frames ~/kh/frames_<take> --log wl_<take>.npz
.venv/bin/python Computer-Vision/common/filter_eval.py mp_<take>.npz wl_<take>.npz
```
A frame recording is ~2 MB/s (JPEG 95, both cameras).

## Goal

Use WiLoR as a second **landmark backend** behind the existing stereo pipeline,
not as a separate tracker. Everything after the landmarks stays as it is and is
shared: rig calibration, skew correction, triangulation and its gates, hand
calibration, the IMM filter, rotation shaping, the offset box, speed caps, the
gripper, the clutch, `--record` / `--replay` / `--log` and `filter_eval.py`.
Downstream (`ik_node`, `feetech_mirror`) doesn't change at all.

After that, use what MediaPipe can't give: the MANO wrist rotation (good when
the palm is edge-on, where MediaPipe loses the hand) and per-view metric 3D
joints (a better single-camera fallback).

MediaPipe stays the default until WiLoR beats it on the replay scores (bottom).

## Layout (done 2026-10-10)

```
Computer-Vision/
  common/      shared pipeline: stereo_cam.py (cameras, rig file, preview), stereo_calib.py,
               fov_check.py, hand3d.py, hand_calib.py, hand_mapping.py, hand_target_node.py,
               filter_eval.py, kalman_bench.py, calib/, recordings/, POTENTIAL_IMPROVEMENTS.md
  MediaPipe/   mediapipe_landmarks.py (Landmarker, StereoLandmarks), models/ (gitignored), xy_pub.py
  WiLoR/       stereo_hand_tracker.py (macOS), stereo_hand_tracker_ubuntu.py (CUDA),
               stereo_calibrate.py, repo/ (upstream clone + weights + MANO, gitignored),
               .venv/ (gitignored)
```

`common/stereo_cam.py` adds `../MediaPipe` to `sys.path` and re-exports
`StereoLandmarks`. Verified after the move: all self-tests pass, and
`--replay rec_reach.npz` gives exactly the committed `recordings/phase2_imm` scores.

## What we have (checked 2026-10-10)

**The seam.** `StereoLandmarks(f0, f1, t) -> (px0, px1)`: each is a (21, 2)
array of full-frame pixels (distorted image coordinates; `hand3d` undistorts
with the rig's `d0`/`d1`) or `None`. `hand_target_node.py` calls it once per
camera pair. A WiLoR class with the same call is all the node needs.

**WiLoR outputs** (`repo/wilor/models/wilor.py`), per 256×256 crop:
- `pred_keypoints_2d`: 21 joints. `mano_wrapper.mano_to_openpose` puts them in
  **OpenPose order, the same as MediaPipe's** (wrist, thumb 1–4, index 5–8, …),
  so `hand3d` indices (palm points, bones, curl) work unchanged.
- `pred_keypoints_3d`: MANO joints (metric, crop camera frame); `pred_cam_t`: translation.
- `pred_mano_params`: `global_orient` (3×3, wrist rotation), `hand_pose` (15×3×3), `betas`.
- `pred_vertices`: the 778-vertex mesh (not needed).

**Environment.** `WiLoR/.venv`: Python 3.12.3, numpy 1.26.4, torch 2.5.1+cu124,
OpenCV 4.11; `torch.cuda.is_available()` is True on the RTX 2060 (6 GB) in WSL2.
numpy matches ROS Jazzy, so this venv can run `rclpy` with ROS sourced, and it
imports `common/` (verified). mediapipe isn't installed there, but `Landmarker`
imports it lazily, so that's fine. Weights: `repo/pretrained_models/`
(`wilor_final.ckpt` 2.5 GB, `detector.pt` YOLO with a left/right class), MANO
in `repo/mano_data/`.

**The current trackers** (`stereo_hand_tracker*.py`) are standalone demos:
own camera threads without capture timestamps or pairing, preview only, inches,
nothing published. Problems to fix in the backend rather than copy:
1. **Left hands aren't flipped.** WiLoR only models right hands. Upstream
   (`vitdet_dataset.py`, `demo.py`) mirrors the crop for detector class 0 and
   mirrors x back after. Without that, left-hand keypoints are wrong.
2. **Crops at the image edge are clipped**, so they aren't square, and
   resizing them to 256×256 distorts the aspect. Pad instead (`copyMakeBorder`).
3. **Hands are matched across views by sorting on x.** That breaks when one
   view sees only one hand. Use the detector's left/right class plus the
   epipolar check (`hand3d` reprojection) instead.
4. `load_wilor` hardcodes `./mano_data/`, so the Ubuntu script `chdir`s into
   `repo/` for good. In the node that breaks relative paths (`--record x.npz`).
   chdir only while loading, then go back.
5. A leftover module-level `def filter(...)` (dead code) and a one-euro on
   top: the shared IMM replaces both.

## Phases

Each phase ends with something runnable. Same rule as the smoothing plan:
judge it on replay scores, and it may add at most ~20 ms of lag.

### 0. Measure WiLoR on this machine (≈ 1 h)

Run `Computer-Vision/WiLoR/.venv/bin/python Computer-Vision/WiLoR/stereo_hand_tracker_ubuntu.py`
(fp16, then `--fp32`) and record: model Hz with both views batched, YOLO cost,
VRAM, per-step latency. Only Apple-silicon numbers (~19 Hz) exist so far;
the 2060 hasn't been measured.
**Gate:** ≥ 20 Hz for a 2-crop batch (the IMM and the IK interpolation are
tuned for the ~30 Hz MediaPipe rate; `interp_delay` 0.04 s covers one gap at 25 fps).
Below that, try in order: larger `--yolo_interval` (track boxes from keypoints),
fp16 + `cudnn.benchmark`, `torch.compile`, a smaller detector `imgsz`. If none
of these reach it, stop here and stay on MediaPipe.

### 1. `WiLoR/wilor_landmarks.py`: the backend (≈ 3 h)

`class StereoLandmarks` with the MediaPipe interface: `__call__(f0, f1, t) -> (px0, px1)`,
`close()`.
- Load once (chdir into `repo/` only during `load_wilor`), fp16 on CUDA, eval, `inference_mode`.
- Boxes per view: YOLO every N calls or when tracking is lost; otherwise the
  previous keypoints' bounding box. Pad to square.
- One hand, like MediaPipe's `num_hands=1`: the one nearest the last tracked
  hand, else the largest. Keep the detector's `is_right` and flip left crops.
- Both views in one forward pass (batch of 2).
- Keypoints back to full-frame pixels: one exact mapping from the crop's
  box, scale and flip, not the current min/max guessing in `to_frame`.
- Extras for later phases on `self.last`: per view `global_orient`,
  `keypoints_3d`, `cam_t`, `is_right`, detector confidence.
- Self-test with no cameras: run `repo/demo_img/` images through it and check
  21 finite points inside the image, and that left/right flip agree.

### 2. Wire it into the node (≈ 1 h)

- `common/stereo_cam.py`: replace the fixed import with
  `make_landmarks(backend)` that loads `MediaPipe/` or `WiLoR/` from `CV_DIR`.
- `--backend mediapipe|wilor` on `hand_target_node.py`, `stereo_cam.py`
  (preview) and `stereo_calib.py` (hand-based calibration).
- Run (WiLoR venv, ROS sourced):
  `Computer-Vision/WiLoR/.venv/bin/python Computer-Vision/common/hand_target_node.py --backend wilor --no-ros`.
  Then without `--no-ros`, with `ros2 run so101_mirror ik_node.py` and
  `control/feetech_mirror.py` in **dry run**.
- `--record` saves landmarks, so `--replay` works for WiLoR recordings as is.
  Store the backend name in the record file and refuse a mismatched `--frame`
  or backend on replay.
- The hand calibration's bone lengths, palm template and flat-hand curl depend
  on the backend: write `calib/hand_<backend>.json`.

### 3. Compare on the same frames (≈ 2 h)

The landmark recordings can't compare backends, since each one holds a
different motion. Add `--record-frames DIR` (both MJPG streams + capture
times) and a frame-replay source in place of `StereoCam`, so both backends run
on identical video. Record still / sweep / reach / drop_out again, plus a new
**edge_on** take (palm turning away, MediaPipe's weak spot). Score both with
`filter_eval.py` (same windows as `recordings/baseline/README.md`) and
`kalman_bench.py` (noise by speed). Save logs in `common/recordings/wilor/`.

### 4. Retune the shared stages for WiLoR noise (≈ 2 h)

- `StereoKalman` / `StereoIMM`: measure WiLoR's sideways and along-ray px noise
  with `kalman_bench.py` and set `sigma_px` / `ray_px` / `speed_k` per backend
  (`IMM_PRESETS` keyed by backend). MANO keypoints are smoother but may be
  biased (prior pulls toward the mean hand).
- Gates: `--reproj-px`, bone tolerance, `--fit-mm`. MANO bones are nearly
  constant, so the bone check catches less. Rely on reprojection between views.
- Gripper: `GRIP_OPEN_CURL` / `--grip-closed-curl` from the WiLoR calibration.

### 5. MANO wrist rotation (the main gain, ≈ 3 h)

- Per view: `global_orient` is in that crop's virtual camera. Correct for the
  crop's offset from the principal point (rotate by the ray through the box
  center), un-mirror left hands, rotate cam1's into cam0 with the rig `R`.
- Fuse the two views (weighted average on SO(3), weight by detector confidence
  and how frontal each view is), then use it as the hand rotation in place of
  the 5-point palm fit, behind `--rot-source mano|fit`. Keep the palm fit and
  `RotationGate` as a cross-check: big disagreement → hold, like now.
- If it tracks edge-on palms, the edge-on hold (`FACING_DEG` 75°) can relax and
  the rotation gains can come down toward 1:1 (they're amplified now only
  because MediaPipe loses the hand before the gripper's range is covered).
  Re-run the range calibration steps to check.
- Done when, on the edge_on take: fewer held frames, lower rotation jitter,
  and no added |roll|/|yaw| > 90° glitches.

### 6. Optional

- Mono fallback from `keypoints_3d` + `cam_t` (metric from one view) instead of
  palm PnP when one camera loses the hand.
- Two hands: on `Second-Hand-Controls-Wrist`, the detector's left/right class
  replaces MediaPipe handedness (`hand3d.assign_hands`). Do this after the branches merge.
- WiLoR on a second machine (the Mac): publish its landmarks over ROS
  (same `ROS_DOMAIN_ID`; ask the user before choosing any non-ROS transport).

## Acceptance (WiLoR becomes the default when all hold)

On the same recorded frames, vs MediaPipe:
- end-to-end pair rate ≥ 20 Hz on the 2060
- tracking-frame share ≥ MediaPipe on every take, clearly higher on edge_on
- position jitter on still ≤ MediaPipe; rotation jitter lower
- lag ≤ MediaPipe + 20 ms; overshoot not worse
- then: IK-only check → `feetech_mirror.py` dry run → `--live` only with the user present

## Risks

- **Speed on the 2060 is unknown** (phase 0 decides). The research notes guess
  10–15 fps for a ViT-H mesh model on this GPU. WiLoR (ViT-L, fp16) should be faster, but that's unmeasured.
- VRAM 6 GB: WiLoR + YOLO in fp16 should fit; MediaPipe runs on CPU.
- Licensing: MANO is non-commercial. Never commit `repo/` (gitignored).
- Two venvs: the MediaPipe pipeline runs on `~/kh/.venv`, WiLoR on
  `Computer-Vision/WiLoR/.venv`. Both must keep numpy 1.26.4 for ROS.
- Branches: this layout exists only on `ik-improvements`. Moving files makes the
  merge with `Second-Hand-Controls-Wrist` harder (git follows renames, but
  check `hand_target_node.py` / `hand3d.py` conflicts by hand).
