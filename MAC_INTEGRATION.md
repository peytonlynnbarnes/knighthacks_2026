# Moving the project to a MacBook Pro (M5 Pro): integration guide

Written 2026-10-10 on `ik-improvements`; the code changes below are done on the
**`mac-version`** branch (macOS only, MediaPipe removed), not yet run on the Mac. Everything in **Verify** boxes is an assumption until it's checked on the machine.
The topic contract, safety rules and pipeline don't change: read `INTEGRATION.md`,
`CLAUDE.md` and `control/CLAUDE.md` as before (dry run first, ask before the first motion,
never write EEPROM, `--live` only with the user present, never loosen the hardware limits).

## What changes, in short

| | Now (Windows + WSL2, RTX 2060) | MacBook Pro M5 Pro |
|---|---|---|
| OS / ROS 2 Jazzy | Ubuntu 24.04 in WSL2, apt packages | macOS, **ROS 2 Jazzy installed through conda** (RoboStack / conda-forge, native arm64) |
| USB devices | `usbipd attach` for each camera + the servo board | native USB, no usbipd |
| Cameras | OpenCV `CAP_V4L2`, `/dev/video0` + `/dev/video2`, MJPG, V4L2 controls | OpenCV `CAP_AVFOUNDATION`, index numbers (order not guaranteed), no V4L2 controls |
| Servo board | `/dev/ttyACM0` | `/dev/cu.usbmodem…` (or `/dev/cu.usbserial-…`) |
| Landmark backend | MediaPipe (default) or WiLoR | **WiLoR only** (MediaPipe is not used on the Mac) |
| WiLoR | CUDA fp16, ~21 Hz model-only, 11–16 pairs/s in the node | PyTorch **MPS**; speed unknown, measure |
| Gazebo sim | `ros_gz`, works | optional; may not be available or may need server/GUI run separately |

Not changed: the ROS topics and message types, the IK, the filters (WiLoR presets), `JOINT_CAL` /
`SOFT_LIMITS`, the safety limits, the hand calibration flow, the recordings and replays
(`--replay`, `--replay-frames` need no cameras: these are the first things to run on the Mac).

**MediaPipe is removed on `mac-version`**: WiLoR is the only (and default) backend.
`stereo_cam.py --no-landmarks` previews the cameras without loading WiLoR. The MediaPipe
recordings (`rec_*.npz`, `recordings/baseline/`, `phase2*/`, the `mp` half of
`recordings/wilor/`) and `calib/hand.json` are deleted on this branch (still on
`ik-improvements`); a MediaPipe landmark file given to `--replay` makes the node stop with a message.

## 0. Before leaving the Windows machine

1. **Nothing on `ik-improvements` is committed yet** (the Computer-Vision split, WiLoR backend,
   `frames.py`, the WiLoR tuning, `GripFilter`, the new `calib/stereo.npz`). Either commit and
   push the branch (ask the owner first), or copy the whole folder:
   ```bash
   rsync -a --exclude .venv --exclude 'Computer-Vision/WiLoR/.venv' \
         --exclude 'ros_ws/build' --exclude 'ros_ws/install' --exclude 'ros_ws/log' \
         --exclude '__pycache__' ~/kh/ <mac>:~/kh/
   ```
2. **Files git never has, copy them by hand:**
   - `Computer-Vision/WiLoR/repo/`: the WiLoR code, `pretrained_models/` (2.5 GB checkpoint +
     YOLO `detector.pt`) and `mano_data/`. MANO is under a non-commercial license: copy it,
     never commit or upload it.
   - `CLAUDE.md`, `INTEGRATION.md`, `control/CLAUDE.md` (gitignored, so a `git clone` won't have them).
   - Optional: the frame recordings (`~/kh/{still,sweep,reach,drop-out,edge-on,cam_bug}/`,
     ~540 MB, gitignored), for `--replay-frames` tests on the Mac.
3. Don't move the cameras or their focus rings while unplugging them if you want to try the
   current stereo calibration on the Mac (you'll most likely redo it anyway, see step 5).

## 1. ROS 2 Jazzy (installed through conda)

ROS 2 Jazzy is already installed in a conda environment (RoboStack, conda-forge `osx-arm64`).
It runs natively, so it can see the USB cameras and the serial port (Docker or a VM on macOS
couldn't: no USB passthrough). Activate that environment in every terminal; everything below
(Python packages, the workspace build, all scripts) runs inside it.

Check that it has what this project uses:
```bash
conda activate ros2_jazzy            # or micromamba / mamba, whichever installed it
python --version                    # every pip package below must match this Python
python -c "import rclpy, std_msgs.msg, numpy; print('ok', numpy.__version__)"
ros2 pkg list | grep -E "^(std_msgs|geometry_msgs|xacro)$"
```
For the real arm the IK needs only `rclpy`, `std_msgs` and `xacro`. Missing ones:
`conda install -c robostack-jazzy -c conda-forge ros-jazzy-xacro` (same for `ros-jazzy-std-msgs`,
`ros-jazzy-geometry-msgs`), plus the build tools if they aren't there yet:
`conda install -c conda-forge compilers cmake pkg-config make ninja colcon-common-extensions`.

> **Verify:** if `xacro` isn't available for `osx-arm64`: `pip install xacro`, or generate the
> URDF once on Linux (`xacro so101.urdf.xacro > so101.urdf`) and start the IK with
> `-p urdf_path:=…/so101.urdf` (`model.load_chain` reads plain URDFs too).
> `ros-jazzy-ros-gz` (the Gazebo bridge) is only for the sim.

Build the workspace inside that environment:
```bash
cd ~/kh/ros_ws && colcon build --symlink-install
source install/setup.zsh          # macOS default shell is zsh: setup.zsh, not setup.bash
```
`package.xml` lists `ros_gz_*` as exec dependencies. That's fine for the build. Skip
`rosdep install` if it tries to pull Gazebo; only `ik_node.py` is needed for the arm.

**Python packages go into the same conda env.** On Ubuntu the venv pinned `numpy==1.26.4` to
match the apt-built ROS messages. Here ROS and numpy come from the same env, so let numpy stay
at the version conda installed (`requirements.txt`'s pins are for the Ubuntu setup (on ik-improvements the Mac list is `requirements-mac.txt`); don't apply
them here):
```bash
pip install feetech-servo-sdk pyyaml opencv-contrib-python    # servo SDK, cameras
python -c "import numpy, cv2, rclpy; print(numpy.__version__, cv2.__version__)"
conda list numpy                  # check pip did not replace conda's numpy
```
If pip wants to change numpy, pick an `opencv-contrib-python` built for that numpy (4.12+ for
numpy 2, < 4.12 for numpy 1.x). WiLoR's packages are in step 4.

DDS on macOS: if nodes on the same Mac don't see each other's topics, set
`export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST` in every terminal (Jazzy's replacement for
`ROS_LOCALHOST_ONLY`). Allow the incoming-connections prompt from the macOS firewall.

## 2. Servo board (control/)

- macOS has a built-in CDC-ACM driver, so the board (`/dev/ttyACM0` on Linux) should show up as
  `/dev/cu.usbmodemXXXX` with no driver install. Find it: `ls /dev/cu.*` before and after
  plugging it in. Use the `cu.` device, not `tty.` (`tty.` waits for a carrier signal).
  If it shows as `/dev/cu.usbserial-…` it's a CH34x/FTDI bridge: recent macOS has drivers for
  both; otherwise install the vendor driver.
- Every control script (`feetech_mirror.py`, `base_pose.py`, `jog.py`, `servo_raw.py`,
  `torque_off.py`, `record_range.py`) finds the port itself when exactly one USB serial device
  is plugged in (`so101_bus.default_port`); with none or several it stops and lists them, then
  pass `--port /dev/cu.usbmodemXXXX`.
- Shebangs are `#!/usr/bin/env python3`, so `./control/feetech_mirror.py` uses the active env;
  `python control/feetech_mirror.py …` works too.
- Nothing about the servos changes: same baud rate, `JOINT_CAL`, `SOFT_LIMITS`, EEPROM limits.
  Re-run the same safety order as the first time: read-only dry run (prints the ticks table),
  check the table against the arm, then `--live` only with the user present.

## 3. Cameras

The code opens cameras with `cv2.CAP_V4L2` and sets V4L2 controls (`stereo_cam.Camera`,
`stereo_cam.v4l2_set_ctrl`, `configure`). None of that exists on macOS:

| Setting | Linux / V4L2 | macOS / AVFoundation |
|---|---|---|
| Open | `VideoCapture(index, CAP_V4L2)`, `/dev/video0`, `/dev/video2` | `VideoCapture(index, CAP_AVFOUNDATION)`; indices 0, 1, … (built-in camera and iPhone Continuity Camera take numbers too) |
| MJPG fourcc | set explicitly | ignored: AVFoundation picks the camera format and delivers decoded frames |
| `CAP_PROP_BUFFERSIZE` 2 | needed (1 buffer halved the fps) | not supported; check latency / dropped frames |
| Exposure, white balance | V4L2 controls | not settable from OpenCV (auto). Auto is what we use anyway |
| Focus (`focus_absolute`) | accepted, but barely changes the H4 image | not settable. Doesn't matter: the **H4s have manual focus rings** (set by hand, keep them taped) |
| Permission | none | macOS asks for camera access the first time; allow your terminal app (System Settings → Privacy & Security → Camera) |

Steps:
1. Plug each camera into its **own USB-C port** directly (no shared hub): two USB 2 webcams on
   one hub share 480 Mbit/s, and if AVFoundation asks for uncompressed frames that's too little.
2. Turn off iPhone Continuity Camera while working (System Settings → General → AirDrop &
   Handoff), so it doesn't take a camera index.
3. Find the indices: `python Computer-Vision/common/stereo_cam.py --list` opens every index and
   shows the images labeled with their number. **Two identical H4s can swap** between
   plug-ins: cover one lens to tell them apart, and pass `--cams LEFT RIGHT` (default `1 2`) so
   cam0 is the left camera (cam1 is to its right, as calibrated), to `stereo_cam.py`,
   `stereo_calib.py` and `hand_target_node.py`. `ffmpeg -f avfoundation -list_devices true -i ""`
   shows the names in (usually) the same order.
4. **Check the real frame format.** On Linux, 640×480 was a 4:3 center crop of the 16:9 sensor
   (real HFOV ~37–39°). AVFoundation may produce 640×480 differently (crop vs scale), which
   changes the intrinsics: run `fov_check.py` or just recalibrate (step 5).
5. **720p may work now.** 1280×720 was unusable only because cam0 dropped to ~5 fps through
   usbipd. Natively it may run both cameras at 30 fps, which gives a wider view (~50°) and more
   pixels on the hand. Measure both cameras at once (`stereo_cam.py --size 1280 720 --cams L R`,
   watch the fps on the startup line). A new size needs a new stereo calibration and new hand calibration.
   WiLoR crops the hand and resizes it to 256 px, so more pixels mostly help when the hand is
   far away; the YOLO detector runs at 320 px either way.

## 4. WiLoR backend (CUDA → Apple GPU via MPS)

- Install it into the same conda env as ROS (one env, so the node can publish): `requirements.txt`
  has torch 2.5.1 / torchvision 0.20.1 (the versions that ran on Linux; newer torch defaults
  `torch.load` to `weights_only=True`), ultralytics 8.1.34 and the parts of WiLoR's own
  requirements the model imports (not gradio, hydra, webdataset, xtcocotools: demo/training
  only). chumpy (for loading MANO) comes from git. See `README.md` step 3.
- Device: `wilor_landmarks.best_device()` picks MPS, then CUDA, then the CPU (printing a
  warning), and the YOLO detector gets the device explicitly (ultralytics never picks MPS on its
  own). The old macOS demo (`WiLoR/stereo_hand_tracker.py`, ~19 Hz on Apple `mps`) shows MPS
  works for this model.
- fp16 on MPS: try it first (same as CUDA). If an op isn't implemented on MPS, run with
  `PYTORCH_ENABLE_MPS_FALLBACK=1` (that op runs on the CPU) or `--wilor-fp32`.
- `bench_wilor.py` times MPS (or CUDA): run it first to get the model-only rate, as in phase 0
  of `WILOR_PLAN.md` (Windows: 47.5 ms for both views = ~21 Hz).
- Copy `repo/` intact; `load_wilor` hardcodes `./mano_data/` (the loader changes directory
  while loading, as on Linux).

> **Verify:** fps of `bench_wilor.py` on MPS, then of the node with `--cams L R --no-ros`.
> The WiLoR filter presets were tuned at ~15 pairs/s. If the Mac runs much faster (e.g. ≥ 25
> pairs/s), retune them on new recordings (`recordings/wilor/README.md` has the procedure).

## 5. Calibrate on the Mac

Even if nothing moved, recalibrate: the capture backend may produce frames with a different
crop/scaling than V4L2 did.
```bash
python Computer-Vision/common/stereo_calib.py --board --square-mm 20 --cams L R   # L = left camera
```
Targets from the last Linux calibration (refocused cams, 2026-10-10): rms ~0.2 px per camera,
stereo ~0.4 px, baseline ~20.9 cm (measured lens to lens: 21 cm). There is no `--focus` on
this branch (rings). The hand calibration runs at every node start, as before.
The old recordings need the rig they were made with: commit 2513a49's
`Computer-Vision/calib/stereo.npz` (baseline 20.4 cm). `HEAD`'s is the refocused Linux rig.

## 6. Bring-up order on the Mac

No hardware needed first:
1. Self-tests: `python Computer-Vision/common/hand_mapping.py`, `hand3d.py`, `frames.py`,
   `python -m so101_mirror.target_filter`, and `WiLoR/wilor_landmarks.py`.
2. Replay of a WiLoR landmark recording (no GPU needed; deterministic, so it should match the
   Linux numbers closely). It was recorded with the old rig, so pass that one (commit 2513a49):
   ```bash
   git show 2513a49:Computer-Vision/calib/stereo.npz > /tmp/old_stereo.npz
   python Computer-Vision/common/hand_target_node.py --no-ros --no-view --rig /tmp/old_stereo.npz \
       --replay Computer-Vision/common/recordings/wilor/lm_wl_reach.npz --log out.npz
   python Computer-Vision/common/filter_eval.py out.npz --window 111 148
   ```
   Linux (2026-10-10, WiLoR smooth preset): jitter out 2.46 mm, lag 16 ms, overshoot 3.5 mm.
3. WiLoR on recorded video (GPU, no cameras): `--replay-frames ~/kh/reach
   --rig /tmp/old_stereo.npz --no-ros --no-view` (needs the copied frame recordings). This is
   the first real test of MPS speed and of the keypoints.
4. ROS without hardware: `ros2 run so101_mirror ik_node.py` + `feetech_mirror.py --no-bus` +
   the node with the replay from step 2 (without `--no-ros`). Watch `ros2 topic echo /arm/elbow_flex`.
5. Cameras: `stereo_cam.py --list`, then `stereo_cam.py --cams L R` preview → calibrate (step 5) →
   `hand_target_node.py --cams L R --no-ros`.
6. Arm, dry run: `feetech_mirror.py --gripper --port /dev/cu.usbmodemXXXX` (reads only).
7. Arm live, with the user present: `--live --max-speed 400` first, and the node with the
   pick-and-place settings from `CLAUDE.md` ("Run"):
   `--cams L R --z-min -0.08 --pitch-gain 1.4 2.4 --grip-closed-curl 0.75`.

## Code changes (done on `mac-version`)

| # | Where | What |
|---|---|---|
| C1 | `control/so101_bus.py` (`default_port`), every control script's `--port` | default: the one `/dev/cu.usbmodem*` / `cu.usbserial*` / `cu.wchusbserial*` device; none or several → stop and list them. Error texts without usbipd |
| C2 | `common/stereo_cam.py` | `CAP_AVFOUNDATION`; no MJPG / buffer size / V4L2 controls / focus sweep; the real frame size is checked on the first frame; `configure()` only reports (auto exposure and WB, manual focus); `--list`; sharpness of each view in the preview; `--no-landmarks`. `stereo_calib.py` and the node lost `--focus` / `--exposure`; `fov_check.py` uses AVFoundation indices |
| C3 | `WiLoR/wilor_landmarks.py` (`best_device`), `WiLoR/bench_wilor.py` | MPS → CUDA → CPU; YOLO gets the device explicitly; `cudnn.benchmark` only on CUDA; MPS sync and memory in the benchmark |
| C4 | shebangs | `#!/usr/bin/env python3` everywhere; run hints say `python …` |
| C5 | `README.md`, `requirements.txt`, this file | macOS setup (conda env, numpy pin from conda, WiLoR deps); Ubuntu steps are on `ik-improvements` |
| C6 | `stereo_cam.py --list` | pick the cameras by looking at them (OpenCV can't read AVFoundation names). Names-based selection: not done |
| C7 | MediaPipe | removed: `Computer-Vision/MediaPipe/` code, the `mediapipe` backend, its filter presets (`IMM_PRESETS` / `KALMAN_DEFAULTS` are WiLoR only), `kalman_bench` defaults to the WiLoR logs. Also removed: `WiLoR/stereo_hand_tracker_ubuntu.py` (CUDA/V4L2 demo) |

Checked on the Linux machine before the move (not on the Mac): every file compiles; the
`hand_mapping`, `hand3d`, `frames` and `wilor_landmarks` self-tests pass; a replay of
`lm_wl_reach.npz` gives bit-identical output to `ik-improvements`; `kalman_bench.py` runs.
Not checkable there: AVFoundation capture, MPS, `/dev/cu.*` ports.

## Risks and open questions

- **ROS packages:** ROS is installed, but check the few packages this project needs (step 1).
  If one is missing for `osx-arm64` and can't be replaced (xacro has a pip/URDF workaround),
  don't switch to another transport between the CV node, IK and servo driver without asking
  the owner first (rule from `control/CLAUDE.md`).
- **Camera timing:** AVFoundation's buffering and timestamps differ from V4L2. Check pair rate
  and skew (`stereo_cam.py` preview prints skew; Linux: ~31 pairs/s, median skew ~8 ms) before
  trusting the filter presets, which were tuned on that timing.
- **Two identical cameras** can swap indices: a swapped rig triangulates garbage (the node's
  checks will reject most frames, so it fails safe, but it looks like "no hand").
- **Gazebo:** optional. On macOS the Gazebo GUI and server may need to run as separate processes
  (`gz sim -s` / `gz sim -g`), if `ros-jazzy-ros-gz` exists for `osx-arm64` at all.
- **Performance numbers** for the M5 Pro (WiLoR on MPS: model-only and in the node) are unknown:
  measure them (step 4, bring-up step 3) and write them into `CLAUDE.md` "Hardware facts".
- **Branch `Second-Hand-Controls-Wrist`** has its own flat `Computer-Vision/` with the same V4L2
  camera code: it needs C1/C2 separately if it's still used on the Mac.
- **numpy 2:** if the conda env's numpy is 2.x, chumpy (MANO loading) and ultralytics 8.1.34 may
  not import. The fallback is a conda env with numpy 1.26 (ROS packages built for it).
