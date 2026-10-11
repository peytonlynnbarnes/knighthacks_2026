# Plan: tracking accuracy on the Mac rig (Phase 4)

Written 2026-10-10 on `mac-version`. Context: the pipeline works end to end on the MacBook
(Elgato Facecam MK.2 stereo rig, WiLoR on MPS), but the arm is hard to control accurately.
Phases 1–3 of the accuracy work are summarized below; this file details Phase 4.

## Where things stand

| Phase | What | State |
|---|---|---|
| 0 | baseline measurements (still hand, axis moves) | proposed |
| 1 | setup: hand at ~50 cm instead of 80 cm; optionally remount so arm-forward isn't camera depth; live driver settings | proposed, **do before Phase 4** |
| 2 | code: IK interpolation delay follows the target rate (`interp_auto_max`), opt-in `--calib-up` step, WiLoR warm-up at the capture size | **done** |
| 3 | arm calibration check (`JOINT_CAL` zero offsets + signs, INTEGRATION.md "Calibration") | proposed, needs the user at the arm |
| 4 | retune the position filter for this rig; gripper rotation from WiLoR's MANO wrist | **this plan** |

Why Phase 1 comes first: noise depends on the geometry. The last hand calibration had the hand
at 0.80 m, with the arm's forward axis almost exactly along camera depth (`A[0]` = cam
[0.20, 0, −0.98]). Depth noise grows with z² and WiLoR's along-ray noise is ~3× its sideways
noise (2.35 vs 0.85 mm, still hand, H4 rig), so tuning on the current setup would be wasted.

Measured on this machine (M5 Pro, 2026-10-10): WiLoR 79 ms per pair (model 77 ms; YOLO every
4th pair adds 24 ms), node 10–12 pairs/s with a hand; rig 1280×720, rms 0.09/0.08/0.14 px,
baseline 26.2 cm, hfov 75°.

## 4.0 New recordings (~15 min, needs the user)

At the final working distance and mounting, 5 takes, raw video + landmarks:

```bash
python Computer-Vision/common/hand_target_node.py --cams 1 0 --no-ros \
    --record-frames ~/kh_rec/<take> --record Computer-Vision/common/recordings/wilor_mac/lm_<take>.npz
```

| take | what the user does | measures |
|---|---|---|
| still | hand still ~15 s after calibration | noise floor |
| sweep | slow left/right, up/down | lag, sideways noise |
| reach | forward/back reaches with stops | depth noise, overshoot |
| drop_out | hide/show the hand a few times | clutch, re-anchor |
| edge_on | tilt and roll to the limits | rotation dropouts |

- Video stays outside the repo (~1 GB, `~/kh_rec/`). Commit only `lm_*.npz` + `windows.json`
  (scoring window per take, s). Nothing MANO-related is saved.
- The video is what makes 4B testable offline: `--record` files hold 2D landmarks only, not
  WiLoR's `global_orient`.
- First check that the replay tools (`--replay`, `--replay-frames`, `filter_eval.py`,
  `kalman_bench.py`) run on this env (conda OpenCV 5.0, ultralytics 8.1.34) using the old
  `recordings/wilor/lm_wl_*.npz` + the 2513a49 rig.

## 4A. Retune the position filter (~3 h, no arm)

1. `--replay lm_<take>.npz --no-ros --no-view --log wl_mac/<take>.npz` for every take.
2. `kalman_bench.py --logs recordings/wilor_mac/wl_mac --backend wilor`: measured noise by hand
   speed (along-ray / sideways) for this rig. Compare with the H4 table in
   `recordings/wilor/README.md`.
3. Set the noise model (`sigma_px`, `ray_px`, `speed_k`) from it.
4. Grid search with a **committed** `common/tune_filters.py` (the last one was a scratch
   script): IMM `sigma_px` / `ray_px` / `speed_k` / moving `q` / `p_stay`, ~2000 settings, plus
   single-Kalman. Score = mean ratio vs the current preset of each moving take's error + still
   wobble; reject any setting with > +20 ms lag. Prefer the middle of a flat ridge (neighbours
   score almost as well) over a single best point: 5 takes are easy to overfit.
5. Gates on the new data: `--reproj-px` 15, `--fit-mm` 15, bone tolerance. Count false rejects
   (frames dropped while the hand is clearly tracked) vs glitches let through.
6. Rotation one-euro (`--rot-filter`) from the measured rotation noise.
7. Update `hand_mapping.IMM_PRESETS["wilor"]` / `KALMAN_DEFAULTS["wilor"]` on `mac-version`
   (the H4 values stay on `ik-improvements` and in the README). Results in
   `recordings/wilor_mac/README.md`.

Acceptance: on the new takes, still-hand jitter and tracking error below the current preset;
lag ≤ +20 ms; overshoot not worse. Then IK + `feetech_mirror.py` dry run; `--live` by the user.

## 4B. Gripper rotation from WiLoR's MANO wrist (~4–5 h, no arm until the end)

Today the gripper rotation is a Kabsch fit of 5 triangulated palm points (`hand3d.palm_fit`).
Depth noise tilts it, and `--pitch-gain` up to 2.4 amplifies it. WiLoR already returns a full
hand rotation per view (`lms.last[v]["global_orient"]`, 3×3), unused so far.

1. Per view → cam0 frame:
   - WiLoR predicts in the crop's virtual camera (optical axis through the crop center).
     Rotate by the rotation taking that axis to the real ray through the crop center
     (`K⁻¹ [cx, cy, 1]`, from `metas`).
   - Un-mirror left-hand crops (`M R M`, `M = diag(−1, 1, 1)`).
   - cam1 → cam0 with the rig: `R0 = Rrigᵀ R1`.
2. Fuse: weighted chordal mean (`project_to_so3(w0 R0 + w1 R1)`), weights = detector
   confidence × how frontal each view sees the palm. One view only: use it.
3. Constant offset between MANO's wrist frame and the palm-fit frame: no extra math needed,
   because rotation is always relative to the neutral pose. `HandCalibrator` must measure
   `neutral_R` from the same source.
4. Wire in: `--rot-source fit|mano` (default `fit` until accepted). Cross-check: if MANO and
   the palm fit disagree by > ~30°, hold the rotation (like `RotationGate`).
5. Evaluate on the 4.0 videos with `--replay-frames`, both sources on identical frames:
   - rotation jitter on still (deg RMS)
   - held-frame share on edge_on
   - |roll| / |yaw| > 90° glitches
   - lag on sweep (vs an acausal reference, as `kalman_bench` does for position)
6. If MANO tracks edge-on palms: relax the edge-on hold (`FACING_DEG` 75°) and bring the
   rotation gains toward 1:1 (they're amplified only because tracking drops before the
   gripper's range is covered). Re-run the range calibration to check.

Acceptance (`WILOR_PLAN.md` phase 5): on edge_on fewer held frames and lower rotation jitter,
no new > 90° glitches. Then dry run; `--live` by the user.

## Risks

- MANO's rotation may be biased (the prior pulls toward a mean hand) or the two views may
  disagree. It stays behind `--rot-source` until the numbers say otherwise.
- Overfitting 5 takes (see the ridge rule in 4A.4).
- The two-hand work (left hand → wrist_roll + gripper, `TWO_HAND_PLAN.md`) changes which hand
  drives rotation: 4B then applies to the left hand's roll, and the right hand's pitch only.
  Record 4.0 with one hand first; add two-hand takes when that plan lands.
