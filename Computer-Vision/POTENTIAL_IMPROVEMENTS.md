# Potential improvements: hand-tracking accuracy

Ranked by expected gain per effort. Before changing anything, take a baseline: hold the
hand still ~10 s and log position/rotation std, then do a slow sweep and log per-point
reprojection error. Re-measure after each change.

## 1. Fit the hand rotation to all 5 palm points, not 3

`hand3d.hand_frame()` builds the palm normal from only wrist, index MCP and pinky MCP, so
one noisy landmark tilts the whole frame. This is a likely cause of the known
|roll|/|yaw| > 90° glitch.

- Least-squares rigid fit (Kabsch/SVD) of the 5-point palm template (already saved in
  `calib/hand.json`) to the triangulated palm points.
- Continuity gate: reject a frame whose rotation jumps more than ~45° from the last one.

## 2. Reject single bad landmarks, not whole frames

`StereoHand.estimate()` accepts or rejects the entire frame on the median reprojection error.
Instead, drop individual points with high error and keep the frame if at least 4 of 5
palm points are good. This gives fewer dropouts and keeps bad points out of the pose.

## 3. Camera sync

The cameras aren't hardware-synced. Pairs can be up to `max_dt = 50 ms` apart, and
timestamps are taken after `cap.read()` returns, so they include USB/decode jitter.
About 16 ms of skew with the hand at 0.5 m/s gives ~8 mm of error, almost all of it in depth.

- Tighten `max_dt` to ~20 ms.
- Use the driver's capture timestamp (`CAP_PROP_POS_MSEC`).
- Better: interpolate cam1's landmarks to cam0's timestamp from the last two cam1 detections.

## 4. Camera settings

Auto-exposure in indoor light means long exposures, and the motion blur shifts landmarks.
On both cameras, identically:

- Lock exposure short (raise gain or add light).
- Lock focus and white balance.

cam0 is already noted as noisier (lighting/focus). Set this up with `v4l2-ctl` or `cap.set(...)` in
`stereo_cam.Camera.__init__`.

## 5. Recalibrate the stereo pair

The current stereo RMS is 1.55 px; a good board calibration is under 0.5 px. Use more views, hold the board at
the hand's working distance, and cover the frame corners. Depth error scales with z²/(f·B), so
calibration error becomes depth error directly.

## 6. Camera geometry (needs remounting and recalibration)

- A wider baseline (e.g. 30 cm) improves depth precision proportionally.
- 1280×720 shows the full ~50° view instead of the 37–39° crop. The hand then stays in both views more
  often, with fewer mono fallbacks.

## Not accuracy, for reference

- The one-euro filter settings (`pos_filter=(1.0, 2.0)` in `hand_mapping.py`) trade jitter for
  lag; they don't make tracking more accurate.
- GPU: the MediaPipe GPU delegate in WSL2 runs through Mesa D3D12 and was measured at 61 ms/frame
  vs 9.4 ms on CPU, so stay on CPU. A CUDA hand-mesh model (WiLoR/HaMeR) on the RTX 2060 could
  improve orientation (~20–40 ms/frame), but try items 1–2 first.
