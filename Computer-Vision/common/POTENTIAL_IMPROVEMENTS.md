# Potential improvements: hand-tracking accuracy

Ranked by expected gain per effort. Before changing anything, take a baseline: hold the
hand still ~10 s and log position/rotation std, then do a slow sweep and log per-point
reprojection error. Re-measure after each change.

## 1. Fit the hand rotation to all 5 palm points, not 3

**Status: implemented, not yet tested on the cameras.** Code: `hand3d.palm_fit`,
`RotationGate`, `--frame fit|3pt`, `--fit-mm`, `--jump-deg`, plus a jitter readout on the
status line. Synthetic results: 3 mm noise gives 4.7° → 4.1° median error (p95 8.5° → 7.3°).
With one landmark 15 mm off, middle MCP improves 8.9° → 3.9°, but index/ring outliers get
slightly worse. A plain least-squares fit spreads a single bad point over the whole frame.
Possible next step: refit leaving out each palm point and keep the best fit.

`hand3d.hand_frame()` builds the palm normal from only wrist, index MCP and pinky MCP, so
one noisy landmark tilts the whole frame. This is a likely cause of the known
|roll|/|yaw| > 90° glitch.

- Least-squares rigid fit (Kabsch/SVD) of the 5-point palm template (already saved in
  `calib/hand.json`) to the triangulated palm points.
- Continuity gate: reject a frame whose rotation jumps more than ~45° from the last one.

## 2. Reject single bad landmarks, not whole frames

**Implemented** (`hand3d.StereoHand._drop_bad_points` and `_palm_pose`):
- Landmarks with reprojection error above `--reproj-px` become NaN.
- One bad palm point is tolerated: the fit uses the other 4, and the point is filled back in from the template.
- Curl needs at least 3 fingertips, otherwise the gripper holds.
- A point that's off along the epipolar line passes the reprojection check but lands at the wrong depth.
  The bone/fit checks catch it, and then the leave-one-out fit drops it. Self-test: 3.5° and 0.7 mm with one palm point 40 px off.

`StereoHand.estimate()` accepts or rejects the entire frame on the median reprojection error.
Instead, drop individual points with high error and keep the frame if at least 4 of 5
palm points are good. This gives fewer dropouts and keeps bad points out of the pose.

## 3. Camera sync

**Implemented** (`stereo_cam.StereoCam.pair`, `hand3d.SkewCorrector`):
- Fixed a bug: `CAP_PROP_BUFFERSIZE=1` dropped every other frame. Pair rate went from 13 to 31 Hz.
- cam1 is paired with its nearest frame. Median skew went from 23 to 8 ms.
- cam1's landmarks are extrapolated to cam0's time (`--no-skew-correct` turns this off).
- cam1 only does ~25 fps (it drops frames), so p95 skew is still ~24 ms. The correction handles that.

The cameras aren't hardware-synced. Pairs can be up to `max_dt = 50 ms` apart, and
timestamps are taken after `cap.read()` returns, so they include USB/decode jitter.
About 16 ms of skew with the hand at 0.5 m/s gives ~8 mm of error, almost all of it in depth.

- Tighten `max_dt` to ~20 ms.
- Use the driver's capture timestamp (`CAP_PROP_POS_MSEC`).
- Better: interpolate cam1's landmarks to cam0's timestamp from the last two cam1 detections.

## 4. Camera settings

**Partly reverted:**
- Locking exposure and white balance broke MediaPipe. With 4600 K under warm room light the image turned
  green, and with the exposure capped at 20 ms it came out dark. Hand detection fell from 86% to 0% of
  frames (measured 2026-10-10).
- Defaults are now auto exposure and auto white balance. `--exposure match` is an option for a bright room.
- What stays: dynamic framerate forced off, focus locked when the rig records one, and settings applied
  explicitly at every start.

Auto-exposure in indoor light means long exposures, and the motion blur shifts landmarks.
On both cameras, identically:

- Lock exposure short (raise gain or add light).
- Lock focus and white balance.

cam0 is already noted as noisier (lighting/focus). Set this up with `v4l2-ctl` or `cap.set(...)` in
`stereo_cam.Camera.__init__`.

## 5. Recalibrate the stereo pair

**Solver improved, recapture still needed:**
- `stereo_calib.py` drops outlier views and re-solves, and frees the principal point (`--fix-center` pins it).
- On the saved corners, cam0 improves from 1.45 to 0.18 px: 10 of its 27 views were bad.
- The stereo error stays at 1.5 px, because almost every saved pair is off by 2–3 px.
  These were captured at 13 fps with up to 50 ms skew and autofocus, all fixed now. Recapture with `--focus N`.

The current stereo RMS is 1.55 px; a good board calibration is under 0.5 px. Use more views, hold the board at
the hand's working distance, and cover the frame corners. Depth error scales with z²/(f·B), so
calibration error becomes depth error directly.

## 6. Camera geometry (needs remounting and recalibration)

**Resolution follows the rig file:**
- `stereo_calib.py --size W H` sets it; the node opens the cameras at the rig's size.
- 1280×720 is **not usable right now**: cam0 drops to ~5 fps once the image is properly bright.
  Larger MJPEG frames exceed its USB bandwidth through usbipd; cam1 still does 26 fps.
  At startup the node prints both cameras' fps and warns below 20.

- A wider baseline (e.g. 30 cm) improves depth precision proportionally.
- 1280×720 shows the full ~50° view instead of the 37–39° crop. The hand then stays in both views more
  often, with fewer mono fallbacks.

## 7. Orientation range: amplify, filter, hold when edge-on (implemented)

A hand can't rotate as far as the gripper and stay tracked: MediaPipe loses it as the palm turns edge-on
to the cameras. `hand_mapping` now does three things:
- Filters the hand rotation before applying any gain (one-euro, `--rot-filter 0.3 1.0 0.3`).
- Shapes each axis (`RotationShaper`):
  - roll ×1.5
  - pitch ×1 up and ×2 down: 45° of hand tilt gives a gripper pointing straight down
  - a 3° soft deadband
  - output clipped to what the arm can reach: roll −45..+150° (the wrist_roll soft limits), pitch −35..+90°
- Holds the rotation while the palm is more than 75° from facing the cameras, ramping trust down from 55°.
  The status line shows hand vs gripper angles and the palm angle.

The speed limit is now 360°/s (`--max-ang`).

Offline test (`scratchpad/rot/rot_eval.py`), 2° per-axis hand noise, old 1:1 setup vs new:

| | old 1:1 | new, double the gain |
|---|---|---|
| jitter, hand still at level | 0.63° | 0.09° |
| jitter, hand still tilted 30° | 0.76° | 0.71° |
| lag following a smooth turn | 67 ms | 67 ms |
| time to 90% of a step | 233 ms (30° step) | 200 ms (60° step) |

The IK test with roll up to +150° shows no position errors and 0–3 flips, the same as ±45°.

The gains are calibrated per run. After the 3 position steps, `hand_calib` has you roll and tilt a flat hand each
way as far as it tracks (steps 4–7). 85% of each range then maps to the gripper's limit in that direction, with
gains capped at 0.7–3. A step that times out keeps the default for that direction (roll 1.5, pitch 1 up / 2 down).
Overrides: `--roll-gain NEG POS`, `--pitch-gain UP DOWN`. `--no-range-calib` skips the steps.

## Not accuracy, for reference

- The one-euro filter settings (`pos_filter=(1.0, 2.0)` in `hand_mapping.py`) trade jitter for
  lag; they don't make tracking more accurate.
- GPU: the MediaPipe GPU delegate in WSL2 runs through Mesa D3D12 and was measured at 61 ms/frame
  vs 9.4 ms on CPU, so stay on CPU. A CUDA hand-mesh model (WiLoR/HaMeR) on the RTX 2060 could
  improve orientation (~20–40 ms/frame), but try items 1–2 first.
