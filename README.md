# kh: hand-tracked SO-ARM101 robot arm (macOS)

Two webcams watch your hand. The WiLoR hand model (on the Mac's GPU) finds it in both views, the
pipeline triangulates it to 3D, and the SO-ARM101 arm follows: hand position → gripper position,
hand tilt/roll → gripper tilt/roll, closing your fingers → closing the gripper.

```
2 webcams ──► hand_target_node.py ──/ball_position──► ik_node.py ──/arm/<joint>──► feetech_mirror.py ──► arm
              (WiLoR, Apple GPU)    ──/gripper──────────────────────────────────►  (servo driver)
```

This is the **`mac-version`** branch: macOS on Apple silicon only (AVFoundation cameras, PyTorch
MPS, `/dev/cu.*` serial ports), WiLoR as the only hand model. The Linux/WSL version is on
`ik-improvements`.

| Folder / file | What |
|---|---|
| `Computer-Vision/common/` | cameras, stereo + hand calibration, triangulation, filters, `hand_target_node.py` |
| `Computer-Vision/WiLoR/` | the WiLoR hand-model backend (`wilor_landmarks.py`) + `repo/` (weights, not in git) |
| `ros_ws/src/so101_mirror/` | ROS 2 package: `ik_node.py` (target → joint angles), optional Gazebo ball sim |
| `control/` | servo scripts: `feetech_mirror.py` (ROS → servos), `servo_raw.py`, `jog.py`, … |
| `INTEGRATION.md`, `MAC_INTEGRATION.md` | topic contract and design notes; what changed for the Mac |
| `control/CLAUDE.md` | **safety rules for the real arm: read before moving it** |

---

## What you need

- A Mac with Apple silicon (written for a MacBook Pro M5 Pro).
- ROS 2 Jazzy in a conda env named `ros2_jazzy` (RoboStack, `robostack-jazzy` + `conda-forge`).
- 2× Adesso CyberTrack H4 webcams, ~21 cm apart, angled in slightly, at hand height. Plug each
  into **its own USB port** (not a shared hub).
- The SO-ARM101 with its USB servo board and power supply.
- A phone or tablet to show the calibration chessboard, and a ruler.
- `Computer-Vision/WiLoR/repo/` from the old machine (~2.5 GB: WiLoR code, `pretrained_models/`,
  `mano_data/`). It isn't in git. **MANO has a non-commercial license: never commit or upload it.**

---

## One-time setup

### 1. Get the code and the WiLoR weights

```bash
cd ~
git clone git@github.com:peytonlynnbarnes/knighthacks_2026.git kh
cd kh
git checkout mac-version
```

Copy `repo/` from the old machine into `~/kh/Computer-Vision/WiLoR/repo/`, over the network or with a
USB drive. Example from the old machine (WSL), with `<mac>` = the Mac's hostname or IP:
```bash
rsync -a --progress ~/kh/Computer-Vision/WiLoR/repo/ VJ@<mac>:~/kh/Computer-Vision/WiLoR/repo/
```
Check on the Mac: `ls ~/kh/Computer-Vision/WiLoR/repo/pretrained_models` should list
`wilor_final.ckpt`, `model_config.yaml` and `detector.pt`, and `repo/mano_data/` should exist.

### 2. Check the ROS environment

Every terminal you use for this project starts with `conda activate ros2_jazzy`.

```bash
conda activate ros2_jazzy
python --version                                   # 3.12.x
python -c "import rclpy, std_msgs.msg; print('rclpy ok')"
ros2 pkg list | grep -E "^(std_msgs|geometry_msgs|xacro)$"   # all three must be listed
pip show numpy | head -2                           # note the version (step 3 keeps it)
```
Missing build tools: `conda install -c conda-forge compilers cmake pkg-config make ninja colcon-common-extensions`.

### 3. Install the Python packages (into the same env)

ROS's message packages are built against conda's numpy, so pip must not replace it. The first
line pins it to whatever version conda installed:

```bash
cd ~/kh
python -c "import numpy; print(f'numpy=={numpy.__version__}')" > /tmp/numpy-pin.txt
pip install -r requirements.txt -c /tmp/numpy-pin.txt
pip install --no-build-isolation git+https://github.com/mattloper/chumpy   # needed to load MANO
# torch ships its own OpenMP runtime, which clashes with conda's ("OMP: Error #15" and an abort):
T=$(python -c "import os, torch; print(os.path.dirname(torch.__file__))")/lib
mv $T/libomp.dylib $T/libomp.dylib.orig && ln -s $CONDA_PREFIX/lib/libomp.dylib $T/libomp.dylib
python -c "import torch; print('MPS available:', torch.backends.mps.is_available())"   # must be True
```
Redo the two `libomp` lines after every reinstall of torch.

`requirements.txt` does not pin numpy itself: the constraint file holds it at conda's version
(RoboStack Jazzy ships numpy 2.x, e.g. 2.2.6; torch 2.14.1 and OpenCV 4.10+ support it).
If `conda list numpy` ever shows two numpy entries or a different version than step 2, pip
replaced it: `pip uninstall -y numpy`, `conda install -y --force-reinstall numpy=<step 2's version>`,
then run this step again.

### 4. Build the ROS workspace

```bash
cd ~/kh/ros_ws
colcon build --symlink-install
source install/setup.zsh          # zsh is the macOS default shell; setup.bash for bash
```
`--symlink-install` makes Python edits take effect without rebuilding (rebuild after adding files).

### 5. Allow camera access

The first time a script opens a camera, macOS asks whether your terminal app (Terminal, iTerm,
VS Code…) may use the camera: allow it, then run the script again. If you missed it: System
Settings → Privacy & Security → Camera → enable your terminal app.

`hand_target_node.py` with its window runs as its own app, **Hand Tracker** (a Dock icon, Cmd-Tab
reaches its window; a window started from a terminal otherwise belongs to the terminal app). It
relaunches itself through `~/Library/Application Support/kh/Hand Tracker.app` (generated by
`Computer-Vision/common/macos_app.py`), keeps printing in the terminal, and stops on Ctrl-C there.
That app asks for camera access once, separately from the terminal. `--no-app` runs it the old way.

**iCloud:** if `~/Documents` syncs with iCloud ("Desktop & Documents Folders") and the disk gets
full, macOS evicts files to the cloud, including the 2.5 GB WiLoR weights and source files: a
script then hangs at start while they download (`ls -lO` shows `dataless`). In Finder, right-click
the repo folder → **Keep Downloaded**, and keep some disk space free.

Also turn off iPhone Continuity Camera while working (System Settings → General → AirDrop &
Handoff), so the phone doesn't show up as a camera.

### 6. Check that each part works (no arm motion)

```bash
cd ~/kh
python Computer-Vision/WiLoR/wilor_landmarks.py      # WiLoR self-test on demo images: "self-test OK"
python Computer-Vision/WiLoR/bench_wilor.py          # speed on the Apple GPU ("device mps")
python control/servo_raw.py                          # arm plugged in + powered: lists servos 1–6
```
`servo_raw.py` finds the servo board's port itself (`/dev/cu.usbmodem…`). If it says there are
none or several, run `ls /dev/cu.*` and pass `--port /dev/cu.usbmodemXXXX`.

---

## Camera setup (once, and again whenever a camera or a focus ring moves)

### 7. Find the camera numbers

```bash
python Computer-Vision/common/stereo_cam.py --list
```
A window shows every camera with its number. The MacBook's own camera is usually `0`. Cover one
H4 lens to see which number is which. **LEFT** = the camera on the left when you stand behind the
cameras looking the same way they do; **RIGHT** = the other one. Every command below takes
`--cams LEFT RIGHT` (default `1 2`). The two H4s can swap numbers when you replug them: check
again after replugging.

### 8. Focus the cameras

The H4s have manual focus rings (software focus does nothing).
```bash
python Computer-Vision/common/stereo_cam.py --cams LEFT RIGHT --no-landmarks
```
Hold a textured target (the chessboard, printed text) at your hand's working distance in the
middle of both views. The top line shows each view's **sharpness**: turn a ring slowly until its
number peaks, for both cameras. Tape the rings afterwards.

### 9. Calibrate the stereo rig

```bash
python Computer-Vision/common/stereo_calib.py --board           # writes calib/chessboard.png, exits
```
Show `Computer-Vision/common/calib/chessboard.png` full screen on the phone at max brightness,
measure 10 squares in a row with a ruler (mm) and divide by 10. Then:
```bash
python Computer-Vision/common/stereo_calib.py --board --square-mm <size> --cams LEFT RIGHT
```
Follow the window: hold the phone still for each grab. It takes 15 close-up views per camera
(6 of them tilted 20–40°) and 12 views both cameras see. Good results (from the Linux machine):
rms ~0.2 px per camera, stereo ~0.4 px, baseline ~21 cm. It refuses to save a bad calibration.
The result is `Computer-Vision/common/calib/stereo.npz`.

### 10. Check the cameras with hand tracking

```bash
python Computer-Vision/common/stereo_cam.py --cams LEFT RIGHT
```
You should see the hand skeleton in both views, `pairs/s` around 15–30 and a small skew.
`q` quits.

---

## Running it

Open **three terminals**. Start each one with:
```bash
conda activate ros2_jazzy
cd ~/kh
source ros_ws/install/setup.zsh
```
If the terminals don't see each other's topics, also run `export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`
in each (and allow the macOS firewall prompt).

### Step A: hand tracking only (prints, publishes nothing)

```bash
python Computer-Vision/common/hand_target_node.py --cams LEFT RIGHT --no-ros
```
WiLoR loads (~15 s), then the **hand calibration** starts. Follow the text in the window, with
your hand in view of both cameras:

1. open hand, palm to the cameras, hold still 2 s
2. move the hand ~15 cm in the **arm's forward** direction, hold
3. bring it back to the start
4. flat hand, roll it clockwise a quarter turn (like a doorknob), then back
5. tilt the hand as far as it still tracks (fingertips toward or away from the cameras), then back
6. tilt the other way, then back
7. hand back at the start, palm to the cameras

Where your hand was in step 1 is the arm's home. After that, the terminal prints the target it
would send (`xyz cm`, `rpy deg`, `grip`). `c` recalibrates, `q` quits.

### Step B: the full pipeline, arm in dry run (no motion)

```bash
# terminal 1: IK
ros2 run so101_mirror ik_node.py

# terminal 2: servo driver, DRY RUN: reads the arm, prints the planned ticks, no torque
python control/feetech_mirror.py --gripper

# terminal 3: hand tracking, publishing
python Computer-Vision/common/hand_target_node.py --cams LEFT RIGHT \
    --z-min -0.08 --pitch-gain 1.4 2.4 --grip-closed-curl 0.75
```
Move your hand and watch terminal 2's table: the planned ticks should follow your hand
(forward → arm reaches forward, up → arm goes up, fist → gripper closes).

### Step C: live (the arm moves). Only with someone at the arm.

Read `control/CLAUDE.md` first. Before going live:
- lift the arm by hand out of its folded rest (shoulder_lift / elbow_flex must be inside their
  limits) and keep wrist_roll on its usable side; `--live` refuses to start otherwise and says which joint.
- clear the table around the arm.

Stop terminal 2 (Ctrl-C) and restart it live, slowly first:
```bash
python control/feetech_mirror.py --gripper --live --max-speed 400
```
It asks for confirmation, turns torque on, homes all joints to the base pose at 400 ticks/s, then
moves on its own to the working home (~11 cm, slowly) and starts following your hand.

While running:
- **Hide your hand** (≥ 0.3 s) = clutch: the arm and gripper hold. When the hand comes back the
  arm continues from where it is (no jump), so you can reposition your hand.
- **Gripper:** flat hand = open, fist = closed. It holds its value while your palm is turned more
  than 55° away from the cameras; open/close with the hand facing the cameras more.
- **Stop:** Ctrl-C in terminal 2. The arm holds its pose (torque stays on). `python control/torque_off.py`
  makes it go limp: **support the arm first, it drops**.

### Useful options (`hand_target_node.py`)

| Option | What |
|---|---|
| `--z-min -0.08` | lowest the arm goes below home (m). −0.05 was too high to reach the table; go lower in small steps after checking clearance |
| `--pitch-gain UP DOWN` | how much the gripper tilts per degree of hand tilt. `1.4 2.4` points the gripper down with ~35° of hand tilt (bigger hand tilts lose tracking) |
| `--grip-closed-curl 0.75` | finger curl that counts as a closed fist (WiLoR reads a flat hand ~1.0) |
| `--imm responsive` | less lag, slightly shakier still hand (default `smooth`) |
| `--scale 1.3` | arm moves 1.3 cm per cm of hand motion (default 1.0) |
| `--no-orientation` | position only, gripper stays level |
| `--record-frames DIR` | save both camera streams (for replaying a problem later: `--replay-frames DIR`) |

`python Computer-Vision/common/hand_target_node.py --help` lists everything.

---

## Testing without hardware

```bash
# replay a recorded WiLoR session (no cameras, no GPU); it was made with the old rig
git show 2513a49:Computer-Vision/calib/stereo.npz > /tmp/old_stereo.npz
python Computer-Vision/common/hand_target_node.py --no-ros --no-view --rig /tmp/old_stereo.npz \
    --replay Computer-Vision/common/recordings/wilor/lm_wl_reach.npz --log /tmp/out.npz
python Computer-Vision/common/filter_eval.py /tmp/out.npz --window 111 148
# expected (Linux, 2026-10-10): jitter out 2.46 mm, lag 16 ms, overshoot 3.5 mm

# self-tests
python Computer-Vision/common/hand_mapping.py
python Computer-Vision/common/hand3d.py
python Computer-Vision/common/frames.py

# IK + servo driver without the arm (prints the ticks the IK output maps to)
ros2 run so101_mirror ik_node.py          # terminal 1
python control/feetech_mirror.py --no-bus # terminal 2
```
Without the replay's `--no-ros`, the replay publishes in real time and drives terminals 1–2.

The Gazebo ball sim (`ros2 launch so101_mirror sim.launch.py`) needs `ros-jazzy-ros-gz`, which is
optional and may not be available for macOS.

---

## Troubleshooting

| Problem | Fix |
|---|---|
| `ros2: command not found` / `No module named rclpy` | `conda activate ros2_jazzy` in this terminal |
| `Package 'so101_mirror' not found` | `source ros_ws/install/setup.zsh`, or the build (step 4) failed |
| numpy / `_ARRAY_API` errors importing `rclpy` | pip replaced conda's numpy: `conda install numpy=<the original version>`, then redo step 3 with the pin |
| `camera N: no frames` | wrong number (`stereo_cam.py --list`), camera access not allowed (step 5), or another app has the camera |
| `camera N gave WxH, asked for 640x480` | that camera can't do the size: check `--list`, or calibrate at the size it gives (`--size W H`) |
| capture size | `hand_target_node.py` captures at 1920x1080 (60 fps) and scales `calib/stereo.npz` to it: the Facecam's 16:9 modes are the same view scaled, so a 1280x720 calibration holds (px settings scale along). `--size 1280 720` goes back; another aspect ratio (e.g. 640x480) needs its own calibration |
| `SLOW CAMERA` | cameras on one hub (use separate ports), or a dark room (more light) |
| tracking jumps, `bones` / `rot jump` lines | one view blurry (step 8), a camera moved (redo step 9), or LEFT/RIGHT swapped |
| `WARNING: WiLoR on the CPU` / very slow | torch has no MPS (step 3 check); an op missing on MPS: run with `PYTORCH_ENABLE_MPS_FALLBACK=1`, or `--wilor-fp32` |
| model under ~20 pairs/s, preview under 60 fps | `--timing` prints each stage's time every 2 s (thread.stage; the busiest thread is the limit). Expected (M5 Pro, one hand): `wilor.forward` ~48 ms, everything else < 2 ms. Two hands (`--two-hands`): 4 crops per pass, ~75 ms, ~13 pairs/s. torch older than 2.14: ~83 ms |
| `OMP: Error #15` and an abort at start | torch's own `libomp.dylib`: redo the two `libomp` lines of step 3 |
| script hangs at start, Ctrl-C shows `zipimport` / `torch.load` | iCloud evicted the files (see step 5): Keep Downloaded, wait for the download |
| `no USB serial device` / `several USB serial devices` | arm board unplugged or unpowered; or pass `--port` (`ls /dev/cu.*`) |
| `could not open /dev/cu…` | another script has the port open (stop it), or replug the board |
| `--live` refuses: joint outside its usable range | move that joint by hand into range (see Step C) |
| IK jumps between solutions | two IK nodes running: stop one, or `export ROS_DOMAIN_ID=77` in all terminals |
