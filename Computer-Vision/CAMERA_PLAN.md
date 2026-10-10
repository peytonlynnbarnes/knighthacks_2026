# Plan: switch the stereo rig to Elgato Facecam MK.2 (1080p60)

Written 2026-10-10 on `ik-improvements`. Replaces the 2× Adesso CyberTrack H4 (640×480 MJPG,
~38° real HFOV, 30 fps, autofocus). Pipeline and measured H4 facts: `~/kh/CLAUDE.md`.

## Status

| Phase | State |
|---|---|
| 0 measure the cameras through usbipd | not started (**gate: it decides everything below**) |
| 1 camera profiles in code | not started |
| 2 mount + calibrate | not started |
| 3 record + compare against the H4 | not started |
| 4 retune for 60 Hz | not started |
| 5 docs + cleanup | not started |

## What changes (Elgato spec vs what we have)

| | CyberTrack H4 (now) | Facecam MK.2 (spec, unverified on our setup) |
|---|---|---|
| Modes | MJPG 640×480 @ 30 (720p drops cam0 to ~5 fps via usbipd) | uncompressed up to 1080p60, 720p120, 540p120 on USB 3; **MJPEG only on USB 2** |
| FOV | ~38° real at 640×480 (center crop) | 84° (probably diagonal: ~76° horizontal at 16:9) |
| Focus | autofocus, `focus_absolute` 0..1023, we lock it | **fixed focus**, sharp at 30–120 cm |
| USB | 2.0 | USB-C, 3.x for uncompressed |
| Image processing | auto exp / WB | auto exp / WB, HDR, more processing set in Elgato Camera Hub (Windows) |

What this means for us:

1. **Bandwidth is the main risk.** Uncompressed 1080p60 YUY2 is ~250 MB/s per camera, and
   usbipd tunnels USB over a virtual network. That won't work for two cameras. We need MJPEG,
   and Elgato says the camera only sends MJPEG on USB 2, so on a USB 3 port V4L2 may not list
   MJPG at all. Phase 0 checks this first.
2. **A wider FOV means fewer pixels on the hand.** At the same resolution, focal length in px
   drops ~2.6× (640 px wide: fx ≈ 930 → ≈ 355–410). Stereo depth noise goes as z²/(f·B), so
   it gets ~2.6× worse, and MediaPipe/WiLoR see a smaller hand crop. Ways to get it back,
   in order of preference: a higher capture resolution (1280×720: fx ≈ 710–830; 1080p:
   ≈ 1070–1250), cameras closer to the hand (fixed focus allows 30 cm and up), a wider
   baseline, or the camera's digital zoom (a crop, so it costs resolution like the H4's did).
   The benefit: the hand stays in both views over a much bigger workspace.
3. **60 fps** halves frame age (~17 → ~8 ms) and the worst pair skew, and gives the filters
   twice the samples. But MediaPipe on CPU takes ~9.4 ms per view (~19 ms per pair), so the
   node will run at ~45–50 pairs/s at best; WiLoR stays at ~21 Hz. The camera rate is useful
   even when the node can't keep up (fresher frames, smaller skew).
4. **No focus control.** Everything built around `focus_absolute` (the `stereo_cam.py` `f`
   sweep, `--focus`, the rig's saved focus, `FOCUS_RANGE`) has to be optional. The upside:
   intrinsics can't drift with autofocus.
5. **A stronger lens distortion at 84°.** `stereo_calib.py` currently fixes k3
   (`CALIB_FIX_K3`) and has a no-distortion `--hfov` pinhole path. Both are likely too weak
   at this FOV.

## Phase 0: measure (no code changes; ~1 h)

Attach both through usbipd (`usbipd list` / `attach --wsl`), then for each camera:

- `v4l2-ctl --list-devices`, `v4l2-ctl -d /dev/videoN --list-formats-ext` on a **USB 3 port**
  and again on a **USB 2 port / USB 2 cable**: which modes offer MJPG?
- `v4l2-ctl -d /dev/videoN -l`: which controls exist (exposure, WB, gain, zoom/pan/tilt,
  HDR, power-line frequency, no focus expected).
- `ls -l /dev/v4l/by-id/ /dev/v4l/by-path/`: are the two cameras told apart (serial numbers)?
  Two identical models get `/dev/video*` numbers in attach order, so cam0/cam1 can swap.
- Real fps, **both cameras at once**, with `stereo_cam.py --size W H` after the fps param
  in phase 1 (or a 10-line OpenCV script for now) for MJPG 640×480@60, 1280×720@60 and
  1920×1080@30/60, with a bright, detailed scene (bigger JPEGs). The `_fps_check` warning
  is the measurement.
- CPU per frame: MJPEG decode at 720p / 1080p (`cv2.imdecode` timing), MediaPipe time per view
  at 1280×720 vs 640×480.
- `fov_check.py` at each candidate size: real HFOV (crop or scaled sensor?).

**Gate:** both cameras at the same mode, ≥ 50 fps each, ≤ ~60 % of one CPU core for capture +
decode. Pick the largest mode that passes (my guess: 1280×720 MJPG @ 60).

If nothing passes through usbipd: (a) a USB 2 port to force MJPEG, (b) capture on the
Windows side (small Python/OpenCV script, MSMF backend) and send JPEG frames into WSL over
localhost TCP (frames are already JPEG, so no re-encode; `frames.py` has the JPEG I/O),
(c) a native Linux boot. (a) first, (b) is maybe half a day.

## Phase 1: camera profiles (code, `common/stereo_cam.py` and users)

Keep the H4 working: the recordings, calibrations and filter tuning all came from it.

- `CAMERA_PROFILES = {"h4": {...}, "facecam": {...}}` in `stereo_cam.py`: size, fps,
  `focus_range` (None for the Facecam), default HFOV for `stereo_calib.py`, `max_exposure`,
  device paths. `--camera h4|facecam` on `stereo_cam.py`, `stereo_calib.py`, `fov_check.py`,
  `hand_target_node.py`. The default stays `h4` until phase 3 passes.
- `Camera(fps=...)` already exists. Pass it from the profile and check the delivered fps the
  same way `size` is checked.
- `--cams` takes paths as well as indexes (`/dev/v4l/by-id/...`), so cam0/cam1 can't swap.
- Timing constants that assume 30 fps:
  - `Camera._run` clamps the period EMA at `1 / 60`: make it `1 / 240`.
  - `MAX_EXPOSURE = 300` (30 ms) caps fps at ~33: make it per profile, ≤ 160 (16 ms) at 60 fps.
  - `pair(max_dt=0.04)`: fine, could tighten to ~0.025 at 60 fps.
  - `stereo_calib.STILL_FRAMES = 4` (≈130 ms at 30 fps): make it seconds-based.
  - `GRIP_TAU`, `HandMapper`, Kalman/IMM are already time-based (check `MAX_DT`).
  - Old docstrings/comments that say "30 fps" / "~31 Hz".
- Focus: `set_controls(focus=None)` must not touch `CAP_PROP_AUTOFOCUS` when the profile has
  no focus control; `focus_sweep` / the `f` key / `--focus` print "fixed focus" and do nothing;
  `save_rig` writes focus −1.
- The rig file records `camera`, `size` and `fps`; `hand_target_node.py` refuses a rig from a
  different camera/size (an H4 rig with Facecam frames would triangulate garbage).
- Camera settings: set power-line frequency (60 Hz) and turn HDR off if V4L2 exposes it (HDR
  merges frames; motion artifacts on a moving hand). Keep auto exposure + auto WB
  (locking them broke MediaPipe on the H4). If some settings exist only in Camera Hub, set
  them once on Windows and check if they persist on the camera.
- Second calibration file per profile: `calib/stereo_facecam.npz` and keep
  `calib/stereo.npz` (H4). Same for `calib/hand*.json` if the neutral pose ends up different.

## Phase 2: mount + calibrate

- Mounting: with ~76° HFOV, toe-in is hardly needed. Start parallel (or ~5°), 20 cm baseline
  (`BASELINE_M`), cameras ~50–70 cm from the hand workspace (inside the 30–120 cm sharp range).
  If phase 0 had to stay at 640×480, use a wider baseline (30 cm+) to make up for depth
  noise, and check that the hand stays in both views at the workspace edges.
- `stereo_calib.py`: free k3 for this profile (or try `CALIB_RATIONAL_MODEL`) and don't use
  the pinhole `--hfov` path for the final rig. Print the board for the full view (use
  larger squares if needed), cover the corners of the image (that's where distortion is
  measured). The FOV gate (30–100°) passes.
- Target: stereo rms < 1 px (the H4 rig was 1.55 px), baselines within 7 %.
- `fov_check.py` ruler test to confirm the calibrated HFOV.
- Hand calibration (`hand_calib.py`) once per backend.

## Phase 3: compare against the H4 (go / no-go)

Record the same takes as `common/recordings/baseline/` (still, sweep, reach, drop_out) with
`--record`, and `--record-frames` for WiLoR replays. Score with `filter_eval.py` and
`kalman_bench.py` (same scoring windows as the baseline README). Put results in
`common/recordings/facecam/README.md`.

Compare: detection rate per view, still-hand jitter (mm), reprojection error, measured depth
noise vs speed (the IMM `speed_k` model), lag, pairs/s, skew. **Switch the default when:**
still jitter and depth noise are no worse than the H4, detection ≥ the H4's 86 %, pairs/s
≥ 31. If jitter is worse at the chosen size, go back to phase 0/2 (bigger size, closer
cameras, wider baseline) before tuning filters.

Then a live check with `feetech_mirror.py` (dry run; `--live` only with the user present,
see `control/CLAUDE.md`).

## Phase 4: retune for 60 Hz

The IMM / Kalman presets (`IMM_PRESETS`, `KALMAN_DEFAULTS`) were tuned on 30 Hz H4
recordings. Measurement noise per frame changes with the camera, and twice the frame rate
with the same `q` gives a different steady state. Add a `facecam` dimension to the presets
(keyed by backend × camera), tune on the phase 3 recordings with `kalman_bench.py`, the same
way as `recordings/wilor/README.md`. Also re-check `--reproj-px` (pixels are worth different
amounts now), `RotationGate`, the gripper curl thresholds, and the `SkewCorrector`.

Same rule as the motion-smoothing phases: must beat the H4 baseline on replay scores without
adding > ~20 ms of lag. Downstream (`ik_node` interpolation, `feetech_mirror` safety limits):
no changes expected; never loosen the safety limits.

## Phase 5: docs + cleanup

- `~/kh/CLAUDE.md` hardware facts (cameras, devices, real HFOV, fps, rms, bandwidth result),
  `INTEGRATION.md` calibration status, `README.md` usbipd steps for the cameras.
- Make `facecam` the default profile; keep `h4` for the old recordings.
- `Second-Hand-Controls-Wrist` has its own flat `Computer-Vision/` copy of `stereo_cam.py`: port
  the profile change there separately if that branch is still used.
- The WiLoR demo scripts (`stereo_hand_tracker*.py`, `stereo_calibrate.py`) hardcode 640×480
  and their own camera setup: update or leave as demos.

## Open questions (answered in phase 0)

- Does the camera list MJPG on a USB 3 port under Linux, or only on USB 2?
- Is 84° diagonal or horizontal, and is 640×480 a crop or the full sensor scaled?
- Do two cameras get distinct serials in `/dev/v4l/by-id`?
- Which Camera Hub settings (HDR, noise reduction, zoom) exist as V4L2 controls, and which
  persist on the camera?
