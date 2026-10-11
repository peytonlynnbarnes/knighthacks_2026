# Changes: `mac-version` and `Lock-Arm-Facing-Down`

Both branches start from `main` (`b58cd24 gripper`). `Lock-Arm-Facing-Down` added two commits
(sections 8–9) on top of `mac-version` (`c31d847`, sections 1–7). Then `mac-version` was
fast-forwarded to it (2026-10-11), so **both branches are now the same commit** (the one that
adds this file) and contain everything below.

| branch | was | now | commits since `main` |
|---|---|---|---|
| `mac-version` | `c31d847` | same as `Lock-Arm-Facing-Down` | 24 |
| `Lock-Arm-Facing-Down` | `b706682` (local only) | pushed to origin | 24 |

Some changes live in the servos' EEPROM rather than in git: see
[Arm settings stored in the servos](#arm-settings-stored-in-the-servos-not-in-git).

---

## `mac-version`

The pipeline (stereo cameras → WiLoR hand tracking → ROS 2 IK → SO-ARM101) ported to a MacBook
(Apple silicon), then tuned on the real arm for accuracy, latency and smoothness.

### 1. Filtering and evaluation tools (`6e3bdb8`, from `ik-improvements`)
- Position filters: `StereoKalman` and `StereoIMM` (still / moving models), which replace the
  one-euro filter; still-hand hold; palm fit and tracking gates in `hand3d.py`.
- Tools: `--record` / `--replay` / `--log` on the hand node, `filter_eval.py`, `kalman_bench.py`,
  and baseline recordings with their scores.

### 2. WiLoR hand-tracking backend (`2513a49`, `7963a39`)
- `WiLoR/wilor_landmarks.py`: the WiLoR (MANO) hand model as a second landmark backend
  (`--backend wilor`), using YOLO hand boxes and mirroring left-hand crops.
- `Computer-Vision/` split into `common/`, `MediaPipe/` and `WiLoR/`.
- `WILOR_PLAN.md`, `CAMERA_PLAN.md`, WiLoR filter presets and tuning recordings, and the
  19 Hz Apple stereo demo.

### 3. macOS port (`cb435a4`, `0b4e7af`, `821757e`, `36f49a2`, `84e2910`)
- AVFoundation cameras (`--cams LEFT RIGHT`, `stereo_cam.py --list`), MPS for WiLoR, and
  automatic detection of the servo port (`/dev/cu.usbmodem*`).
- MediaPipe removed (WiLoR only); conda ROS 2 Jazzy environment; step-by-step `README.md`;
  requirements follow conda's numpy.

### 4. Mac bring-up (`0d454b4` … `c04cb03`)
- Stereo rig: 2× Elgato Facecam MK.2 at 1280×720, `--cams 1 0`, baseline 26.2 cm; OpenCV 5
  corner-shape fix and an optional `--free-k3` in `stereo_calib.py`.
- `bench_wilor.py` loads through `load_models`; WiLoR warms up at the capture size.
- Hand node: the preview runs at camera rate, with the pipeline on worker threads.
- `so101_bus.SOFT_LIMITS` = the measured range of motion ∩ the EEPROM limits.
- IK interpolation delay that follows the target rate (`interp_auto_max`).
- Optional `--calib-up` step in the hand calibration.
- Plans: `ACCURACY_PLAN.md`, `TWO_HAND_PLAN.md`.
- **Two-hand mode, first part:** the IK can lock `wrist_roll` from `/wrist_roll_cmd`;
  `RollFilter` and `LeftHandCalibrator`; WiLoR tracks both hands.

### 5. Latency and arm control (`1a584b7`, `896ed59`)
Measured on the real arm: target → arm lag went from **570 ms to about 220 ms**.
- **Driver bug fixed:** `feetech_mirror.py` handled only one ROS message per 50 Hz tick against
  150+ per second. Joint goals moved in pulses (the arm shook), using commands up to about
  0.3 s old. ROS callbacks now run on their own thread, so each tick uses the newest command.
- One bus read per tick for every servo's position, load and status flags. The loop runs on a
  fixed schedule: macOS oversleeping had held it at 44 Hz instead of 50. `--max-speed` now
  counts real elapsed time.
- **Blocked joints** (pushing into something: error and load toward the goal, not moving) and
  **stalls** (stuck with little load) stop only that joint, in that direction only. Moving the
  command back frees it. There's no abort, so the arm can always recover. Lifting needs a
  larger error and load to count, because gravity sag looks the same. Limits tested live by
  pushing into the table and with fast free moves.
- **Gripper hold:** contact means load toward closing while the jaws are still. The squeeze
  is then regulated to 18 % load, under the servo's 25 % overload threshold, which used to trip
  and weaken the grip. Closing in free air no longer stutters.
- `--lift-speed`; homing at a per-joint peak speed; `/hw/status` (JSON) for the hand app;
  `--log` (per-tick log); `--selftest`.
- `set_gains.py` (read and write P/D/I, with read-back) and `tune_gains.py` (step-response
  sweeps), used for the gains in the table below.
- Faster defaults: hand node `--max-lin 1.0` (was 0.25), driver `--alpha 0.6` (was 0.3).

### 6. IK (`fbc7ed1`)
- **Claw points down near the table:** the pitch comes from the tip height (straight down
  below 4 cm, 50° at 20 cm), and `floor_z` keeps the tip 2 cm above the table.
- **Smooth targets:** default `interp_auto_max 0.12` and `smooth_tau 0.03`. A fixed 40 ms delay
  ran past the newest target at ~13 targets/s and turned the path into a 13 Hz staircase.
  IK-output jitter fell from 35 % to 6 %.
- `/wrist_roll_cmd` is interpolated in time like the position target.

### 7. Hand app (`c31d847`)
- Two-hand mode (`--two-hands`): the left hand drives `wrist_roll` and the gripper.
- A status line from the driver in the preview: loop rate, worst lag, and a red banner when a
  joint is blocked or stalled, or the driver stops.
- Left-roll gain from the calibration's quarter turn: a 90° hand turn had read as ~45°.
- Runs as a "Hand Tracker" macOS app (`macos_app.py`); `--timing` stage timers (`timing.py`);
  1080p capture with the rig scaled to it.
- README: libomp fix, iCloud file eviction, timing expectations.

---

## `Lock-Arm-Facing-Down` (sections 8–9; now also on `mac-version`)

### 8. `wrist_roll`: full 333° range (`4cc0c2b`)
- The wrist's ~341° of travel crossed the servo's 0/4095 encoder seam, which position mode
  can't drive across, so only ~205° was usable.
- The hard stop was measured by hand (torque off), then probed live with the block detection
  (ends at 110 / 3987, stopped at ~30 % load). The servo's `Homing_Offset` went
  1731 → **1542**, which puts the seam in the middle of the stop.
- `so101_bus`: wrist_roll zero is 2237 (the same physical orientation as before),
  `SOFT_LIMITS` 135…3962 (**−185° … +152°**); `HOME` = each joint's zero; bus reads retry when
  the SDK returns a truncated reply.
- `model.apply_limits`: wrist_roll's measured range replaces the URDF's ±157°.
- `set_homing_offset.py`: dry-run-first tool for the offset.

### 9. Hand control (`b706682`)
- **Pan as an angle** (`--pan-mapping cylinder`, now the default): hand left/right turns the
  base (±20 cm of hand travel → ±75°, clipped to the arm's range), and forward/back is reach.
  The x/y box had capped pan at about ±30°; the IK now reaches its whole −64° … +80°.
  `--pan-span`, `--pan-deg`, and `--pan-mapping cartesian` for the old behaviour.
- **Absolute, full-range wrist roll:** the left hand's twist from its calibrated neutral,
  ±90° (after the quarter-turn correction), covers wrist_roll's whole range. Nothing re-anchors:
  gripping, or hiding the hand, just holds the roll.
- **The wrist turns with only the left hand in view:** when the position target is stale but
  roll commands keep coming, the IK holds the last target and follows the roll.
- **Saved hand calibration:** both hands' calibrations are saved
  (`calib/hand_wilor.json`, `calib/hand_wilor_left.json`) with the camera rig's signature, and
  reused at startup. Press `c`, or start with `--calibrate`, to make new ones.

---

## Arm settings stored in the servos (not in git)

These live in the servos' EEPROM. A replaced or re-calibrated servo needs them set again.

| joint | P | D | I | other |
|---|---|---|---|---|
| shoulder_pan | 24 | 30 | 0 | |
| shoulder_lift | 40 | 30 | 2 | |
| elbow_flex | 40 | 15 | 2 | |
| wrist_flex | 24 | 30 | 0 | |
| wrist_roll | 24 | 30 | 0 | `Homing_Offset` **1542** (LeRobot's was 1731; must match `so101_bus.JOINT_CAL`) |
| gripper | 24 | 32 | 0 | |

Before tuning, P was 6 / 6 / 6 / 10 / 15 / 10 (pan … gripper). The soft gains made
shoulder_lift sag 7–12° under the arm's weight; that sag looked like lag and stalls.
Set the gains with `control/set_gains.py`, measure them with `control/tune_gains.py`
(step-response sweeps), and set the offset with `control/set_homing_offset.py`.

## Running (Mac, conda env `ros2_jazzy`, workspace sourced)

```bash
ros2 run so101_mirror ik_node.py
python control/feetech_mirror.py --gripper --live --max-speed 800 --lift-speed 450   # someone at the arm
python Computer-Vision/common/hand_target_node.py --cams 1 0 --two-hands --z-min -0.16 --grip-closed-curl 0.75
```
