# Plan: left hand drives wrist_roll (joint 5) and the gripper (joint 6)

Written 2026-10-10 on `mac-version`. Not started.

## Goal

| hand | drives | how |
|---|---|---|
| **right** | joints 1–4: shoulder_pan, shoulder_lift, elbow_flex, wrist_flex | position → IK; hand tilt → gripper pitch (as now) |
| **left** | joint 5 wrist_roll | twist of the left hand about its finger axis, relative |
| **left** | joint 6 gripper | left-hand finger curl (flat = open, fist = closed) |

Either hand hidden = that hand's joints hold (clutch per hand). One hand only (`--two-hands` off,
the default until accepted) = today's behaviour, unchanged.

## What already exists

### Prototype: `~/Documents/GitHub/WiLoR/stereo_hand_tracker.py` (standalone, publishes nothing)

- Two `HandTrack`s; YOLO hand boxes in both views paired by image row (`match_stereo_pairs`,
  thresholds in px scaled by resolution), NMS with distance, a collision guard when tracks merge.
- **Left/right labelling with a second YOLO model:** `yolov8n-pose.pt` (COCO body pose) on cam1
  at 320×240 gives the person's wrists; a hand box within 0.20 (normalized) of a wrist takes
  that side. Fallback when the body isn't visible: 3D thumb-side test (palm normal · thumb) mixed
  with a 2D one, smoothed with hysteresis; plus "the two hands must differ".
- Inference: `interleaved` (one hand per step, both views) or `batched` (all crops at once).
- Gaps we must not copy:
  - It **ignores the hand detector's own class** (`detector.pt` has left = 0 / right = 1) and
    **doesn't mirror left-hand crops**. WiLoR only models right hands, so its left-hand
    keypoints are degraded. `wilor_landmarks.py` already does both correctly.
  - `model._skip_mesh_inference = True` is a no-op: nothing in WiLoR reads it.
  - Body pose runs on one view only; plain-crop resize (not the upstream warp); its own
    calibration file and a pixel-scaled intrinsics hack.

### Branch `Second-Hand-Controls-Wrist` (ba9e37a, MediaPipe era, flat layout)

- ROS contract: `/wrist_roll_cmd` (`std_msgs/Float64`, rad, URDF). While fresh (< `stale_s`)
  `ik_node` **locks** wrist_roll at it and solves the other joints for position + pitch around
  it (`solve_ik(lock=...)`; the tip is ~11 mm off the roll axis, so roll can't just be
  overwritten after the IK).
- `hand_mapping.RollFilter`: relative twist (`hand3d.twist_x`) since the hand appeared;
  re-anchors while fingers move (gripping can't twist the wrist), holds when the palm is
  edge-on, ignores one-frame jumps (3-frame confirm), one-euro, speed cap, clip to
  `--roll-range`.
- `hand3d.assign_hands`: MediaPipe-label based (not reusable as is), with an image-x fallback.
- Port the pieces, **don't merge the branch**: it uses the old IK (`solve_relaxed_pitch`),
  one-euro filtering, and a different `/ball_position` home (off by `HOME_OFFSET`).

## Design

### 1. Two hands in `WiLoR/wilor_landmarks.py`

- `StereoLandmarks(hands=2)`: one track per (view, side), side from the detector class
  (already read, `cls == 1` = right) with the existing hysteresis and IoU re-matching. Crops of
  both hands in one batch; left crops mirrored (already done).
- Label checks, cheapest first, each one measured before adding the next:
  1. the detector class, kept per track (flips ~1 % per `wilor_landmarks` comment);
  2. same side in both views (the stereo pair must agree; else the more confident wins);
  3. two hands can't share a side (complementarity, as in the prototype);
  4. image-x prior: cameras facing the user → the user's right hand is on image-left;
  5. only if 1–4 leave label errors: the prototype's body-pose wrist check (`yolov8n-pose`),
     run every N pairs, on both views.
- Return per side: `{"right": (px0, px1), "left": (px0, px1)}`; `.last` keeps
  `global_orient` / `kp3d` per side (for the roll, see 3).

### 2. Rate budget (measured, M5 Pro, fp16)

WiLoR batch 2 = 76 ms, batch 4 = 115 ms (`bench_wilor.py`). Both hands in both views every
pair → ~8 pairs/s (now 10–12). Options, decide on measurements:

- **A. Batch 4 every pair**: simplest; ~8 Hz for position, ~30 % more lag. 
- **B. Left hand every 2nd pair** (prototype's interleave idea, but only for the left hand):
  batch 2 / 4 alternating, ~10 Hz position, ~5 Hz roll/grip. Roll and grip are slow
  signals; the gripper already smooths over 40 ms (`GRIP_TAU`).
- **C. Left hand in one view only**: roll from WiLoR's per-view `global_orient`
  (monocular; ties in with `ACCURACY_PLAN.md` 4B), curl from its `kp3d`. Batch 3 every pair.

Start with B (no new geometry), measure, try C if the roll from the palm fit is too noisy.

### 3. Left-hand signals (`common/hand_mapping.py`, pure, self-tested)

- Roll: port `RollFilter` (relative twist about the finger axis, grip re-anchor, edge-on
  hold, jump confirm, smoothing, speed cap). Range = wrist_roll's usable range
  (soft limits −157..+50°, never across the 0/4095 seam), gain from a roll range step in the
  calibration. The rotation source is the left hand's palm fit, or MANO (`--rot-source`, 4B).
- Grip: the existing `GripFilter`, fed by the left hand's curl instead of the right's.
- Clutch: left hand hidden ≥ `--reanchor-s` → roll and gripper hold; roll re-anchors on return.
- Right hand: position + pitch only (its roll is ignored; the IK locks wrist_roll).

### 4. ROS (`so101_mirror`)

- `WRIST_ROLL_TOPIC = "/wrist_roll_cmd"` in `so101_mirror/__init__.py`.
- `kinematics.solve_ik(lock={joint index: rad})`: locked joint fixed, its Jacobian column
  zeroed; `ik_solver.TargetSolver.solve(offset, lock=...)` passes it through, random reseeds
  keep the locked value; the orientation target uses the locked roll (as the branch did).
- `ik_node`: subscribe, lock while fresh, else today's behaviour. Branch guard unchanged.
- `feetech_mirror.py`: unchanged (wrist_roll stays an `/arm/*` joint from the IK; the
  gripper stays on `/gripper`).

### 5. Hand calibration (`common/hand_calib.py`)

Two-hand mode adds two steps after today's right-hand steps (the roll step moves to the left hand):
- `LEFT`: left hand flat, palm to the cameras, hold 2 s → neutral + open curl.
- `LEFT ROLL`: roll it a quarter turn, back → roll gain and sign.
Right-hand tilt steps stay (pitch gains).

### 6. Node + display (`common/hand_target_node.py`)

- `--two-hands` (off by default), `--swap-hands` (if labels come out reversed), `--roll-range`.
- Preview: right hand green, left orange (as the prototype), per-hand status on two lines.
- Terminal line adds `roll deg  grip` for the left hand.
- `--record` / `--replay` store both hands (new file version; old files still replay).

## Steps and checks

| # | Step | Check |
|---|---|---|
| 1 | IK lock (kinematics, TargetSolver, ik_node) | self-test: FK roll = locked value, position error unchanged vs unlocked; replay of `lm_wl_reach` bit-identical with no `/wrist_roll_cmd` |
| 2 | Port `RollFilter` + self-test | grip re-anchor, edge hold, jump confirm, seam-safe clip |
| 3 | Two-hand `wilor_landmarks` + label checks 1–4 | record 2 takes (hands apart, hands crossing/near): label accuracy per frame, swap events |
| 4 | Node wiring, calibration steps, display | `--no-ros`: printed roll/grip follow the left hand, position follows the right |
| 5 | Rate option B (then C if needed) | pairs/s, roll jitter (deg RMS) still / moving |
| 6 | Full pipeline dry run | `feetech_mirror.py --gripper` table: wrist_roll and gripper follow the left hand only |
| 7 | Live | by the user, someone at the arm, `--max-speed 400` first |

Single-hand mode must stay bit-identical at every step (the `lm_wl_reach` replay).

Effort: ~1–1.5 days (1–2 ½ day each for 1–2, 3, 4–5; 6–7 with the user).

## Risks

- **Rate**: two hands cost ~25–35 % of the pair rate (section 2), adding lag for the right hand.
- **Labels**: hands crossing or touching. Mitigation: per-track hysteresis, both views must
  agree, complementarity; if a frame is ambiguous, both hands hold (fail safe: no motion).
- **Wrist roll seam**: wrist_roll can't pass 0/4095; keep the clip to the soft limits.
- **At 80 cm** both hands are small in the image; Phase 1 of `ACCURACY_PLAN.md` (hand at
  ~50 cm) matters even more here, and both hands must fit in both views (75° hfov: ~77 cm
  wide at 50 cm).
- The gripper moving from the right hand to the left changes how a pick feels: plan a short
  practice run in dry run before going live.
