# kh: hand-tracked SO-ARM101 (macOS, `mac-version` branch)

Computer vision tracks a hand, and the SO-ARM101 robot arm follows it.

- `control/`: talks to the arm's Feetech STS3215 servos.
- `ros_ws/src/so101_mirror/`: ROS 2 Jazzy package: target → IK → `/arm/<joint>` commands (+ an optional Gazebo ball sim).
- `Computer-Vision/`: stereo hand tracking with the WiLoR (MANO) hand model → `/ball_position` + `/gripper`.
- `INTEGRATION.md`: topic contract, how the pieces connect, calibration status. `MAC_INTEGRATION.md`: what changed for the Mac and the bring-up order.

This branch runs on macOS only (Apple silicon: AVFoundation cameras, the Apple GPU via PyTorch MPS,
`/dev/cu.*` serial ports). The Linux/WSL setup is on `ik-improvements`.

## Requirements

- A Mac with Apple silicon (written for a MacBook Pro M5 Pro).
- ROS 2 Jazzy in a conda environment (RoboStack, `robostack-jazzy` + `conda-forge` channels).
- The arm's USB servo board and two USB webcams (Adesso CyberTrack H4), each on its own USB port.
- `Computer-Vision/WiLoR/repo/` (the WiLoR code, `pretrained_models/`, `mano_data/`): not in git,
  copy it from the old machine. MANO is under a non-commercial license: never commit or upload it.

## 1. ROS 2 Jazzy (conda)

Every terminal: `conda activate ros2_jazzy` (or `micromamba activate`). Check it has what's needed:

```bash
python -c "import rclpy, std_msgs.msg, numpy; print('ok', numpy.__version__)"
ros2 pkg list | grep -E "^(std_msgs|geometry_msgs|xacro)$"
# missing ones / build tools:
conda install -c robostack-jazzy -c conda-forge ros-jazzy-xacro ros-jazzy-geometry-msgs \
    compilers cmake pkg-config make ninja colcon-common-extensions
```

If nodes on the same Mac don't see each other's topics: `export ROS_AUTOMATIC_DISCOVERY_RANGE=LOCALHOST`
in every terminal. Allow the macOS firewall prompt.

## 2. Build the ROS workspace

```bash
cd ~/kh/ros_ws
colcon build --symlink-install
source install/setup.zsh     # every new terminal, after activating the env (zsh is the macOS default)
```

`--symlink-install` makes edits to Python scripts take effect without rebuilding. Rebuild after you add
new files. Gazebo (`ros-jazzy-ros-gz`) is only needed for the ball sim (`sim.launch.py`), not for the arm.

## 3. Python packages (same conda env)

```bash
cd ~/kh
python -c "import numpy; print(f'numpy=={numpy.__version__}')" > /tmp/numpy-pin.txt
pip install -r requirements.txt -c /tmp/numpy-pin.txt
pip install --no-build-isolation git+https://github.com/mattloper/chumpy
python -c "import torch; print('MPS', torch.backends.mps.is_available())"   # must say True
```

The numpy pin keeps conda's numpy: ROS's messages are built against it, and pip must not replace it.

## 4. Devices

- **Servo board:** shows up as `/dev/cu.usbmodem…` (or `/dev/cu.usbserial…`). The control scripts find
  it automatically when it's the only one; otherwise pass `--port /dev/cu.usbmodemXXXX` (`ls /dev/cu.*`).
- **Cameras:** the terminal app needs camera access (System Settings → Privacy & Security → Camera).
  Turn off iPhone Continuity Camera while working. Find the camera numbers:
  ```bash
  python Computer-Vision/common/stereo_cam.py --list
  ```
  Then pass `--cams LEFT RIGHT` to `stereo_cam.py`, `stereo_calib.py` and `hand_target_node.py`
  (default `1 2`; cam0 = the left camera). Two identical H4s can swap numbers between plug-ins: check.
- **Focus:** the H4s have manual focus rings. Set them in the preview
  (`stereo_cam.py --cams L R --no-landmarks` shows each view's sharpness: same textured target in
  both, turn until the number peaks), then tape them. Recalibrate after turning one.

## 5. Check that it works

```bash
# servos: scan IDs 1–6 and print position/load/voltage/temp (no motion)
python control/servo_raw.py

# WiLoR on the Apple GPU (demo images, no cameras), then its speed
python Computer-Vision/WiLoR/wilor_landmarks.py
python Computer-Vision/WiLoR/bench_wilor.py

# cameras + hand landmarks
python Computer-Vision/common/stereo_cam.py --cams L R

# stereo calibration (once per camera mount / focus ring change; see the script's docstring)
python Computer-Vision/common/stereo_calib.py --board --square-mm 20 --cams L R

# IK → servo ticks, dry run (doesn't move the arm)
ros2 run so101_mirror ik_node.py
python control/feetech_mirror.py --no-bus
```

For the full pipeline, the `--live` mode and calibration, see `INTEGRATION.md` and
`MAC_INTEGRATION.md`. Before you run anything that moves the arm, read the safety rules in
`control/CLAUDE.md`.

## Troubleshooting

- **`ros2: command not found`**: the conda env isn't active in this terminal.
- **`Package 'so101_mirror' not found`**: you didn't source `ros_ws/install/setup.zsh`, or the build failed.
- **numpy / `_ARRAY_API` errors when importing `rclpy`**: pip replaced conda's numpy. Reinstall it
  with conda (`conda install numpy=<the version ROS came with>`) and redo step 3 with the pin.
- **`camera N: no frames`**: wrong index (`--list`), camera access not granted, or the camera is in
  use by another app.
- **`SLOW CAMERA`**: both cameras on one hub, or a dark room (auto exposure gets longer).
- **WiLoR very slow / `WARNING: WiLoR on the CPU`**: torch has no MPS (`torch.backends.mps.is_available()`).
  An op missing on MPS: run with `PYTORCH_ENABLE_MPS_FALLBACK=1`, or `--wilor-fp32`.
- **IK jumps between solutions**: two IK nodes / sims are running. Use `export ROS_DOMAIN_ID=77` to isolate one.
