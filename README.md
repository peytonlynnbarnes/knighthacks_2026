# kh: hand-tracked SO-ARM101

xD
Computer vision tracks a hand, and the SO-ARM101 robot arm follows it.

- `control/`: talks to the arm's Feetech STS3215 servos (Python venv).
- `ros_ws/src/so101_mirror/`: ROS 2 Jazzy package: Gazebo target ball → IK → `/arm/<joint>` commands.
- `INTEGRATION.md`: topic contract, how the pieces connect, calibration status. Read it after setup.

## Requirements

- Ubuntu 24.04 (native or WSL2). ROS 2 Jazzy targets 24.04 only.
- Python 3.12 (the Ubuntu 24.04 system Python).
- The arm's USB servo board, if you're driving real hardware.

## 1. ROS 2 Jazzy + Gazebo

Follows the [official Jazzy install guide](https://docs.ros.org/en/jazzy/Installation/Ubuntu-Install-Debs.html).

```bash
# locale + universe repo
sudo apt update && sudo apt install -y locales software-properties-common curl
sudo locale-gen en_US en_US.UTF-8
sudo update-locale LC_ALL=en_US.UTF-8 LANG=en_US.UTF-8
sudo add-apt-repository -y universe

# ROS apt source
export ROS_APT_SOURCE_VERSION=$(curl -s https://api.github.com/repos/ros-infrastructure/ros-apt-source/releases/latest | grep -F "tag_name" | awk -F\" '{print $4}')
curl -L -o /tmp/ros2-apt-source.deb "https://github.com/ros-infrastructure/ros-apt-source/releases/download/${ROS_APT_SOURCE_VERSION}/ros2-apt-source_${ROS_APT_SOURCE_VERSION}.$(. /etc/os-release && echo $VERSION_CODENAME)_all.deb"
sudo dpkg -i /tmp/ros2-apt-source.deb

# ROS, Gazebo (Harmonic, via ros_gz), and the packages so101_mirror uses
sudo apt update
sudo apt install -y ros-jazzy-desktop ros-dev-tools \
  ros-jazzy-ros-gz ros-jazzy-xacro python3-numpy
```

Load ROS in every new terminal (or add it to `~/.bashrc`):

```bash
source /opt/ros/jazzy/setup.bash
```

## 2. Build the ROS workspace

```bash
cd ~/kh/ros_ws
rosdep init 2>/dev/null; rosdep update          # first time only (`sudo rosdep init` if it asks)
rosdep install --from-paths src --ignore-src -y  # pulls any missing deps from package.xml
colcon build --symlink-install
source install/setup.bash                        # every new terminal, after /opt/ros/jazzy
```

`--symlink-install` makes edits to Python scripts take effect without rebuilding. Rebuild after you add new files.

## 3. Python venv (servo control + hand tracking)

```bash
cd ~/kh
sudo apt install -y python3-venv
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
```

`requirements.txt` pins `numpy==1.26.4` to match the numpy ROS Jazzy was built against. Don't upgrade numpy or OpenCV in the venv, or ROS messages break when the venv runs a ROS node (`feetech_mirror.py`).

The `control/` scripts have the shebang `#!/home/peyton/kh/.venv/bin/python`. If your checkout is somewhere else, run them as `.venv/bin/python control/<script>.py` instead.

## 4. Connect the arm (WSL2 only)

WSL can't see USB devices until you pass them in. In an **admin PowerShell** on Windows:

```powershell
winget install usbipd      # first time only
usbipd list                # find the servo board's BUSID
usbipd bind --busid <BUSID>     # first time only
usbipd attach --wsl --busid <BUSID>
```

Then in WSL:

```bash
ls /dev/ttyACM0             # missing = board not attached
sudo chmod 666 /dev/ttyACM0 # needed after every attach
```

On native Linux, add yourself to the `dialout` group instead (`sudo usermod -aG dialout $USER`, then log out and back in).

## 5. Check that it works

```bash
# servos: scan IDs 1–6 and print position/load/voltage/temp (no motion)
.venv/bin/python control/servo_raw.py

# sim: Gazebo ball → IK (drag the ball with the T/R tools)
ros2 launch so101_mirror sim.launch.py

# second terminal: IK → servo ticks, dry run (doesn't move the arm)
source /opt/ros/jazzy/setup.bash
.venv/bin/python control/feetech_mirror.py --no-bus
```

For running the full pipeline, the `--live` mode, and calibration, see `INTEGRATION.md`. Before you run anything that moves the arm, read the safety rules in `control/CLAUDE.md`.

## Troubleshooting

- **`ros2: command not found`**: you didn't source `/opt/ros/jazzy/setup.bash` in this terminal.
- **`Package 'so101_mirror' not found`**: you didn't source `ros_ws/install/setup.bash`, or the build failed.
- **numpy / `_ARRAY_API` errors when a venv script imports `rclpy`**: numpy in the venv isn't 1.26.x. Run `.venv/bin/pip install -r requirements.txt` again.
- **IK jumps between solutions**: two sims are running on the same machine. Use `export ROS_DOMAIN_ID=77 GZ_PARTITION=test` to isolate one.
