#!/usr/bin/env python3
"""Drive the real SO-ARM101 from the ROS topics /arm/<joint> (radians, URDF).

Subscribes to /arm/shoulder_pan, /arm/shoulder_lift, /arm/elbow_flex,
/arm/wrist_flex, /arm/wrist_roll (std_msgs/Float64), converts them to ticks
(so101_bus.JOINT_CAL), and sync-writes Goal_Position at --rate Hz.
Publishes the measured pose as sensor_msgs/JointState on /hw/joint_states.
The gripper (ID 6) is only touched with --gripper: then /gripper
(std_msgs/Float64, 0 = open .. 1 = closed, from hand_target_node.py) maps
linearly to so101_bus.GRIPPER_OPEN .. GRIPPER_CLOSED ticks.

With --live the arm first eases to its base pose (every arm joint at 2048,
like base_pose.py). Only then does it start following /arm/*. IK home
(/ball_position [0,0,0,0,0,0]) is the mid-workspace pose 10 cm back and 5 cm
down from there, so with the target at home the IK output moves the arm there
(lift -51, elbow +64, wrist_flex -13 deg) right after homing, rate limited.

Run it with the conda env that has ROS 2 Jazzy (macOS), from any folder:
    conda activate ros2_jazzy      # the servo port is found automatically (/dev/cu.usbmodem*),
                                  # or pass --port /dev/cu.usbmodemXXXX
    ./feetech_mirror.py                     # DRY RUN (default): reads the arm,
                                            # prints planned ticks, no torque
    ./feetech_mirror.py --live              # asks, torque on, homes, then follows
    ./feetech_mirror.py --live --max-speed 400 --alpha 0.2   # slower, smoother
    ./feetech_mirror.py --live --lift-speed 350              # shoulder_lift slower still
    ./feetech_mirror.py --no-bus            # no arm attached: print the
                                            # ticks the /arm/* topics map to
    ./feetech_mirror.py --selftest          # stall detector test, no bus

Safety:
    - JOINT_CAL signs/offsets are NOT verified on hardware yet. Do the
      calibration check in ~/kh/INTEGRATION.md before the first --live run.
    - Torque on per CLAUDE.md: read present, goal = present, low servo
      accel/speed, then Torque_Enable. Refused if a joint is outside its
      usable range (EEPROM limits, read only, never written, combined with
      SOFT_LIMITS).
    - Homing: smoothstep ramp, every joint at most --home-speed ticks/s
      (shoulder_lift at most --lift-speed), with the same
      lag/load abort as base_pose.py.
    - Every goal: EMA filter (--alpha), per-tick rate limit (--max-speed
      ticks/s), clamp to EEPROM limits AND so101_bus.SOFT_LIMITS (measured
      range; keeps wrist_roll off the 0/4095 seam).
    - No message for --stale-s on any joint: hold the current goal.
    - While following, the goal never leads the present position by more
      than max(MIN_LEAD, LEAD_TIME * speed) ticks (like jog.py's MAX_LEAD),
      so a joint slower than --max-speed just moves at its own pace.
    - Lag > max(MAX_LAG, LAG_TIME * speed) (only possible if a joint is
      pushed away or drops) or |load| > LOAD_ABORT: hold at present pose
      and exit.
    - Stall (a joint > STALL_ERR ticks from its goal, not moving toward it
      for STALL_S, little load; the lead limit keeps the lag abort from seeing
      it): handled like a block, below.
      Stalls, the loop rate and each servo's tripped protections (Status
      register: overload, temperature, ...) go out on /hw/status (JSON) for
      hand_target_node's preview.
    - Blocked (pushing into something: far from the goal, not moving, load in the
      goal's direction, BLOCK_*; lifting needs more, BLOCK_*_UP, as gravity sag
      looks the same): only that joint stops, and only in that direction;
      the command moving back unblocks it. No abort however long it lasts (the
      arm stays recoverable; the load abort still applies). On /hw/status as
      "blocked" (the hand app shows it).
    - --max-speed is per second of real loop time (the loop can run below
      --rate); one sync read per tick for every servo's position/load/status.
    - Default following speed is 75 % of the servo's no-load maximum
      (2250 ticks/s ~ 200 deg/s). Use a lower --max-speed for first tests.
    - Ctrl-C / exit: goal = present, torque stays ON (the arm holds, it
      does not go limp). Use torque_off.py to release it.
    - Gripper (--gripper): torque on holding where it is (no homing), clamped
      to its usable range, rate limited (--grip-speed), holds when /gripper
      goes stale. It isn't in the lag/load abort (gripping an object stalls
      it by design); instead, on contact (load + jaws still, GRIP_CONTACT_*) its goal
      stays a squeeze past where the jaws stopped, regulated to GRIP_HOLD_LOAD (under
      the servo's overload threshold); opening past the contact lets go.
"""
import argparse
import json
import math
import sys
import threading
import time

import numpy as np

try:
    import rclpy
    from rclpy.executors import SingleThreadedExecutor
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64, String
except ImportError as e:
    sys.exit(f"{e}\nactivate the conda env with ROS 2 Jazzy first (and pip install -r ~/kh/requirements.txt in it)")

import so101_bus as sb

ARM_TOPIC_PREFIX = "/arm/"  # must match so101_mirror.ARM_TOPIC_PREFIX
ARM_JOINTS = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")

# STS3215 12 V (C018): 0.222 s/60 deg no-load = ~3070 ticks/s; the servos'
# factory Maximum_Velocity_Limit reads 65 (x50) = 3250 ticks/s.
FULL_SPEED = 3000    # ticks/s, ~270 deg/s
SERVO_SPEED = FULL_SPEED  # servo-side cap; the software rate limit is what really paces motion
SERVO_ACCEL = 100    # 0-254, LeRobot uses 254
MAX_SPEED_CAP = FULL_SPEED  # refuse --max-speed / --home-speed above this
DEFAULT_SPEED = 2250  # 75 % of full speed (~200 deg/s), user's choice 2026-10-09
LIFT_SPEED = 450     # shoulder_lift (~40 deg/s): live 2026-10-11 it never went faster than
                     # ~35-40 deg/s and stalled (stuck up to 60 s) when driven at 800
MAX_LAG = 250        # ticks between goal and present before aborting, at low speed
LAG_TIME = 0.25      # ...grows with speed: limit = max(MAX_LAG, LAG_TIME * speed)
MIN_LEAD = 150       # max goal lead over present while following (ticks), at low speed
LEAD_TIME = 0.15     # ...grows with speed: lead = max(MIN_LEAD, LEAD_TIME * speed)
LOAD_ABORT = 700     # 0.1 % units (70 %)
HOME_TOL = 15        # ticks to count as arrived at home
GRIP_TOPIC = "/gripper"  # must match hand_target_node.GRIP_TOPIC
# Gripper on an object (GripGuard). Contact = load > GRIP_CONTACT_LOAD toward closing with
# the jaws still for GRIP_CONTACT_S (load alone fired while closing in free air: 38-42 %).
# Then the goal stays GRIP_SQUEEZE ticks past the contact, the squeeze adjusted every tick
# so the holding load settles at GRIP_HOLD_LOAD: the servo's own overload protection
# (gripper Overload_Torque 25 % for 2 s -> output cut to 20 %) tripped at a fixed 40-tick
# squeeze with P 24 (2026-10-11), and the old load-threshold rule then pumped 31..50 %.
GRIP_CONTACT_LOAD = 300  # 0.1 %
GRIP_CONTACT_S = 0.15
GRIP_HOLD_LOAD = 180     # 0.1 % (18 %, under the 25 % overload threshold)
GRIP_SQUEEZE = 20        # ticks past the contact to start with...
GRIP_SQUEEZE_MIN, GRIP_SQUEEZE_MAX = 5, 40  # ...kept in this range
GRIP_SQUEEZE_K = 0.05    # ticks per tick per % of load error
GRIP_RELEASE = 30        # ticks the command must open past the contact to let go
# Stall: a joint more than STALL_ERR ticks from its goal that hasn't moved STALL_MOVE ticks
# toward it for STALL_S (seen live: shoulder_lift stuck 60 s, 34 deg off, no abort: the
# lead limit keeps the goal within MIN_LEAD of a stuck joint, so the lag abort can't fire).
# Then it's handled like a block (BlockGuard): that joint stops pushing that way and
# recovers when the command comes back. No abort, so the arm can always recover.
STALL_ERR = 60       # ticks (~5 deg)
STALL_MOVE = 8       # ticks
STALL_S = 0.5
# Blocked: pushing into something (table, object, person). A joint > BLOCK_ERR from its goal,
# not moving toward it (STALL_MOVE) for BLOCK_S, with its load > BLOCK_LOAD in the goal's
# direction (the load's sign is opposite the push: -sign(load) matched sign(goal - present)
# in ~90 % of the tuning samples, 2026-10-11). Then only that joint's goal stops: at most
# BLOCK_PRELOAD past where it stopped, in that direction only; moves away still follow. It
# unblocks once the command comes back BLOCK_RELEASE ticks from the block. No abort for a
# long block (the arm must stay recoverable): the 70 % load abort is the only hard stop.
# A stuck joint with LOW load is a stall (above), handled the same way.
BLOCK_ERR = 25       # ticks (~2.2 deg); at 40 the table push test reached 57 % load before
                     # the block fired (2026-10-11), at 25 it fires near 27 %
BLOCK_LOAD = 200     # 0.1 % (20 %); with BLOCK_ERR and no motion, gravity alone stays below
BLOCK_S = 0.3
BLOCK_PRELOAD = 10   # ticks the goal stays past the block (keeps contact, no sag away)
BLOCK_RELEASE = 25   # ticks (~2 deg) the command must come back to unblock
# Against gravity (lifting) a joint also sags: stretched out, shoulder_lift held 33 ticks
# short at 20 % load and read as blocked "up" (2026-10-11 table test). Lifting directions
# need a clearly bigger error and load; pushing down keeps the tight limits (no sag that way).
BLOCK_ERR_UP = 60    # ticks (~5.3 deg)
BLOCK_LOAD_UP = 350  # 0.1 % (35 %)
# what +/- ticks mean for the preview (from the gravity sag at the tuning pose)
DIR_NAMES = {"shoulder_lift": {1: "down", -1: "up"}, "elbow_flex": {1: "down", -1: "up"}}
STATUS_TOPIC = "/hw/status"  # std_msgs/String JSON; must match hand_target_node.HW_STATUS_TOPIC
STATUS_EVERY_S = 0.2


class StallDetector:
    """Per joint: stalled = farther than STALL_ERR from the goal without moving STALL_MOVE
    toward it for STALL_S. Pure (feed it ticks and times), self-tested."""

    def __init__(self, err=STALL_ERR, move=STALL_MOVE, after_s=STALL_S):
        self.err, self.move, self.after_s = err, move, after_s
        self.since = {}  # joint -> (start time, present then, sign of the error)

    def update(self, joint, goal, present, t):
        """Seconds this joint has been stuck (0 = fine). Stalled once >= after_s."""
        e = goal - present
        if abs(e) <= self.err:
            self.since.pop(joint, None)
            return 0.0
        s = 1 if e > 0 else -1
        start = self.since.get(joint)
        if start is None or start[2] != s or (present - start[1]) * s > self.move:
            self.since[joint] = start = (t, present, s)  # (re)start: new, flipped or moving
        return t - start[0]

    def stalled(self, joint, goal, present, t):
        return self.update(joint, goal, present, t) >= self.after_s

    def reset(self):
        self.since.clear()


class BlockGuard:
    """Per joint: detects pushing into an obstacle and then stops the goal there, in that
    direction only. Pure (ticks, load in 0.1 %, times), self-tested."""

    def __init__(self, err=BLOCK_ERR, load=BLOCK_LOAD, after_s=BLOCK_S, preload=BLOCK_PRELOAD,
                 release=BLOCK_RELEASE, up=None, err_up=BLOCK_ERR_UP, load_up=BLOCK_LOAD_UP):
        """up: {joint: tick direction (+1/-1) that lifts against gravity}; those directions
        use err_up / load_up."""
        self.det = StallDetector(err, STALL_MOVE, after_s)
        self.det_up = StallDetector(err_up, STALL_MOVE, after_s)
        self.det_stall = StallDetector(STALL_ERR, STALL_MOVE, STALL_S)  # stuck with little load
        self.load, self.preload, self.release = load, preload, release
        self.up, self.load_up = up or {}, load_up
        self.blocked = {}  # joint -> (direction +1/-1 in ticks, tick where it stopped, since, why)

    def update(self, joint, goal, present, load, t):
        """Every tick, before the goals are computed. "pushing" (into something) or
        "stalled" (stuck far from the goal without the load to show for it: a servo
        protection or weak servo) when `joint` just got blocked, else None."""
        if joint in self.blocked:
            return None
        e = goal - present
        lifting = self.up.get(joint) == (1 if e > 0 else -1)
        det, lim = (self.det_up, self.load_up) if lifting else (self.det, self.load)
        (self.det if lifting else self.det_up).since.pop(joint, None)
        pushing = abs(load) > lim and (load < 0) == (e > 0)  # load sign is opposite the push
        if not pushing:
            det.since.pop(joint, None)
            if self.det_stall.stalled(joint, goal, present, t):
                self.blocked[joint] = (1 if e > 0 else -1, present, t, "stalled")
                self.det_stall.since.pop(joint, None)
                return "stalled"
            return None
        self.det_stall.since.pop(joint, None)
        if det.stalled(joint, goal, present, t):
            self.blocked[joint] = (1 if e > 0 else -1, present, t, "pushing")
            det.since.pop(joint, None)
            return "pushing"
        return None

    def clamp(self, joint, g, cmd):
        """Goal `g` for `joint`: if blocked, at most `preload` past where it stopped (moves
        away pass). Unblocks once the command `cmd` is `release` ticks back from there."""
        b = self.blocked.get(joint)
        if b is None:
            return g
        s, at = b[:2]
        if (cmd - at) * s < -self.release:
            del self.blocked[joint]
            return g
        return at + s * self.preload if (g - at) * s > self.preload else g

    def hold_goal(self, joint):
        s, at = self.blocked[joint][:2]
        return at + s * self.preload


class GripGuard:
    """Gripper goal while it holds something: detects contact (pushing, jaws still) and
    then keeps the goal a regulated squeeze past where the jaws stopped. Pure, self-tested."""

    def __init__(self, close_dir, contact_load=GRIP_CONTACT_LOAD, contact_s=GRIP_CONTACT_S,
                 hold_load=GRIP_HOLD_LOAD, squeeze=GRIP_SQUEEZE, k=GRIP_SQUEEZE_K, release=GRIP_RELEASE):
        self.d, self.contact_load, self.hold_load = close_dir, contact_load, hold_load
        self.squeeze0, self.k, self.release = squeeze, k, release
        self.det = StallDetector(err=20, move=6, after_s=contact_s)
        self.contact, self.squeeze = None, squeeze

    def update(self, g, cmd, present, load, t):
        """g: the rate-limited goal, cmd: the commanded position, present, load (0.1 %).
        Returns the goal to send."""
        if self.contact is None:
            pushing = abs(load) > self.contact_load and (load < 0) == (self.d > 0)  # sign opposite the push
            if not (pushing and (g - present) * self.d > 0):
                self.det.since.pop("g", None)
                return g
            if not self.det.stalled("g", g, present, t):
                return g
            self.contact, self.squeeze = present, self.squeeze0
            self.det.since.pop("g", None)
        if (cmd - self.contact) * self.d < -self.release:  # opened past the contact: let go
            self.contact = None
            return g
        if (present - self.contact) * self.d > self.squeeze + 10:  # closed on through: object gone
            self.contact = None
            return g
        self.squeeze = min(GRIP_SQUEEZE_MAX, max(GRIP_SQUEEZE_MIN,
                                                 self.squeeze + self.k * (self.hold_load - abs(load)) / 10))
        hold = self.contact + self.d * self.squeeze
        return g if (g - hold) * self.d < 0 else hold  # a less-closed command still passes


class Mirror(Node):
    def __init__(self, args):
        super().__init__("feetech_mirror")
        self.args = args
        self.cmd = {}      # joint -> (rad, monotonic time)
        for j in ARM_JOINTS:
            self.create_subscription(Float64, ARM_TOPIC_PREFIX + j,
                                     lambda m, j=j: self.on_cmd(j, m.data), 10)
        self.state_pub = self.create_publisher(JointState, "/hw/joint_states", 10)
        self.status_pub = self.create_publisher(String, STATUS_TOPIC, 10)
        self.grip = None   # (0..1, monotonic time)
        if args.gripper:
            self.create_subscription(Float64, GRIP_TOPIC, self.on_grip, 10)

    def on_grip(self, m):
        if math.isfinite(m.data):
            self.grip = (min(1.0, max(0.0, m.data)), time.monotonic())

    def fresh_grip(self):
        """0 (open) .. 1 (closed) if recent, else None (hold)."""
        if self.grip is None or time.monotonic() - self.grip[1] > self.args.stale_s:
            return None
        return self.grip[0]

    def on_cmd(self, joint, rad):
        if math.isfinite(rad):
            self.cmd[joint] = (rad, time.monotonic())

    def fresh_targets(self):
        """{joint: rad} if every joint has a recent command, else None."""
        now = time.monotonic()
        cmd = dict(self.cmd)  # snapshot: the callbacks write from the ROS thread
        if len(cmd) < len(ARM_JOINTS):
            return None
        if any(now - t > self.args.stale_s for _, t in cmd.values()):
            return None
        return {j: cmd[j][0] for j in ARM_JOINTS}

    def publish_state(self, present):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        arm = {j: t for j, t in present.items() if j in sb.JOINT_CAL}  # gripper has no rad cal
        msg.name = list(arm)
        msg.position = [sb.ticks_to_rad(j, t) for j, t in arm.items()]
        self.state_pub.publish(msg)

    def publish_status(self, status):
        self.status_pub.publish(String(data=json.dumps(status)))


def print_table(targets, present, limits, goals):
    print(f"{'joint':<14} {'cmd deg':>8} {'cmd tick':>8} {'clamped':>8} "
          f"{'present':>8} {'filt':>6}  limits")
    for j in ARM_JOINTS:
        rad = targets.get(j) if targets else None
        raw = sb.rad_to_ticks(j, rad) if rad is not None else None
        lo, hi = limits.get(j, (0, 4095))
        clamped = max(lo, min(hi, raw)) if raw is not None else None
        f = lambda v, w=8, d=0: f"{v:>{w}.{d}f}" if v is not None else f"{'-':>{w}}"  # noqa: E731
        print(f"{j:<14} {f(math.degrees(rad) if rad is not None else None, 8, 1)} {f(raw)} "
              f"{f(clamped)} {f(present.get(j))} {f(goals.get(j), 6)}  [{lo}, {hi}]")
    print()


def lag_limit(speed):
    return max(MAX_LAG, LAG_TIME * speed)


def sync_read_works(bus, all_sids):
    """One trial sync read; False (with a note) = use the old per-servo reads."""
    try:
        bus.read_states(all_sids.values())
        return True
    except RuntimeError as e:
        print(f"sync read failed ({e}); reading each servo separately (slower loop)")
        return False


def selftest():
    sd = StallDetector(err=60, move=8, after_s=0.5)
    t = 0.0
    for _ in range(40):  # far from the goal but moving: never stalled
        assert not sd.stalled("a", 1000, 500 + t * 400, t)
        t += 0.02
    sd.reset()
    for i in range(30):  # stuck 150 ticks off (the shoulder_lift case): stalled after 0.5 s
        t = i * 0.02
        assert sd.stalled("b", 2150, 2000 + (i % 3), t) == (t >= 0.5 - 1e-9), t
    assert not sd.stalled("b", 2030, 2000, 0.62)  # within STALL_ERR: fine, resets
    for i in range(30):  # creeping 2 ticks per tick = moving: never stalled
        assert not sd.stalled("c", 3000, 2000 + 2 * i, i * 0.02)
    sd.reset()  # error flips sign: restarts the clock
    assert sd.update("d", 2100, 2000, 0.0) == 0.0 and sd.update("d", 2100, 2000, 0.4) > 0.39
    assert sd.update("d", 1900, 2000, 0.45) == 0.0
    bg = BlockGuard(err=40, load=300, after_s=0.3, preload=10, release=25)  # explicit values: the test is about the logic
    t = 0.0
    for i in range(10):  # moving into it with high load: not blocked while it still moves
        assert not bg.update("j", 2200, 2000 + 20 * i, -400, t)
        t += 0.05
    for i in range(10):  # stopped 60 ticks short, load pushing toward the goal: blocked after 0.3 s
        hit = bg.update("j", 2240, 2180, -400, t)
        t += 0.05
        if hit:
            break
    assert "j" in bg.blocked and bg.blocked["j"][:2] == (1, 2180), bg.blocked
    assert bg.clamp("j", 2300, 2400) == 2190  # can't go further that way
    assert bg.clamp("j", 2100, 2400) == 2100  # moving away passes
    assert bg.clamp("j", 2300, 2170) == 2190 and "j" in bg.blocked  # command back 10: still blocked
    assert bg.clamp("j", 2140, 2150) == 2140 and "j" not in bg.blocked  # back 30: unblocked
    bg2 = BlockGuard(err=40, load=300, after_s=0.3)
    for i in range(20):  # stuck but low load (gravity sag / weak servo): not a block
        assert not bg2.update("k", 2240, 2180, -100, i * 0.05)
    for i in range(20):  # high load AWAY from the goal (being pushed back by hand): not a block
        assert not bg2.update("k", 2240, 2180, +400, i * 0.05)
    bg5 = BlockGuard(err=25, load=200, after_s=0.3)
    why = [bg5.update("m", 2300, 2150, -50, i * 0.05) for i in range(15)]  # 150 off, 5 % load
    assert "stalled" in why and bg5.blocked["m"][3] == "stalled" and bg5.clamp("m", 2400, 2400) == 2160
    bg3 = BlockGuard(err=25, load=200, after_s=0.3, up={"lift": -1}, err_up=60, load_up=350)
    for i in range(20):  # sagging 33 ticks below a goal above it at 20 % (stretched-out hold): not a block
        assert not bg3.update("lift", 2379, 2412, 200, i * 0.05)
    hit = [bg3.update("lift", 2600, 2560, -360, i * 0.05) for i in range(20)]  # pushing down 40 ticks, 36 %
    assert any(hit) and bg3.blocked["lift"][0] == 1
    bg4 = BlockGuard(err=25, load=200, after_s=0.3, up={"lift": -1}, err_up=60, load_up=350)
    assert any(bg4.update("lift", 2300, 2380, 400, i * 0.05) for i in range(20))  # held down hard: blocked up
    # gripper: closing (d = -1) at 30 ticks/tick in free air with 40 % load: no contact;
    # an object at 2139: contact, then the squeeze settles so the load is ~18 %
    gg = GripGuard(close_dir=-1)
    pos, sent, t = 2900.0, 2900.0, 0.0
    for i in range(400):
        load = 400 if pos > 2139 + 1 else int(max(0, (pos - sent)) * 11)  # ~1.1 % per tick past (P 24)
        sent = gg.update(max(2040.0, sent - 30), 2040, round(pos), load, t)
        pos = max(2139.0, pos - 25) if pos > 2139 else pos  # jaws stop at the object
        if pos > 2139 + 1:
            assert gg.contact is None, (i, pos)  # free motion: never a contact
        t += 0.02
    assert gg.contact is not None and abs((pos - sent) * 11 - 180) < 30, (gg.contact, gg.squeeze, pos - sent)
    assert gg.update(2400, 2400, 2139, 100, t) == 2400 and gg.contact is None  # opened: let go
    assert sb.sign_mag(0x400 | 25, 10) == -25 and sb.sign_mag(25, 10) == 25
    assert sb.sign_mag(0x8000 | 300, 15) == -300
    print("feetech_mirror self-test OK")


def max_lead(speed):
    """Goal lead over present while following; kept below lag_limit so the
    lag abort only fires when a joint is pushed away or drops, not when it's
    merely slower than requested (load abort still catches blocking)."""
    return max(MIN_LEAD, LEAD_TIME * speed)


def home(bus, sids, start, speed, dt, lift_speed=None):
    """Ease every joint from `start` to sb.HOME together (smoothstep), like
    base_pose.py. Aborts (holds, exits) on lag/load. Returns the final goals."""
    target = {j: float(sb.HOME[j]) for j in sids}
    biggest = max(abs(target[j] - start[j]) for j in sids)
    # `speed` is every joint's PEAK speed (smoothstep peaks at 1.5x the average; at
    # 1.5 x 400 = 600 ticks/s shoulder_lift fell 250 ticks behind and aborted, 2026-10-11),
    # and shoulder_lift's is at most lift_speed
    peak = {j: min(speed, lift_speed) if j == "shoulder_lift" and lift_speed else speed for j in sids}
    duration = max(max(1.5 * abs(target[j] - start[j]) / peak[j] for j in sids), 1.0)
    print(f"homing: largest move {biggest:.0f} ticks ({biggest * 360 / 4096:.0f} deg), ~{duration:.1f} s")
    t0 = time.monotonic()
    while True:
        s = min((time.monotonic() - t0) / duration, 1.0)
        e = s * s * (3 - 2 * s)
        goals = {j: start[j] + (target[j] - start[j]) * e for j in sids}
        bus.send_goals({sids[j]: goals[j] for j in sids})
        for j, sid in sids.items():
            lag = abs(goals[j] - bus.position(sid))
            load = bus.load(sid)
            if lag > lag_limit(peak[j]) or abs(load) > LOAD_ABORT:
                held = bus.hold(list(sids.values()))
                sys.exit(f"ABORT while homing: {j} lag {lag:.0f} ticks, load {load / 10:.0f}%. "
                         f"Holding at {held}.")
        if s >= 1.0:
            break
        time.sleep(dt)
    t1 = time.monotonic()
    while time.monotonic() - t1 < 2.0:
        if all(abs(bus.position(sid) - target[j]) <= HOME_TOL for j, sid in sids.items()):
            break
        time.sleep(0.05)
    print("home:", {j: bus.position(sid) for j, sid in sids.items()})
    return target


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help=sb.PORT_HELP)
    ap.add_argument("--live", action="store_true", help="enable torque and move (default: dry run)")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt (with --live)")
    ap.add_argument("--no-bus", action="store_true", help="dry run without the arm attached")
    ap.add_argument("--rate", type=float, default=50.0, help="control loop Hz")
    ap.add_argument("--alpha", type=float, default=0.6, help="EMA factor on targets (0..1]; lower = smoother, more lag")
    ap.add_argument("--lift-speed", type=float, default=LIFT_SPEED,
                    help=f"shoulder_lift's own rate limit, ticks/s (default {LIFT_SPEED}; it carries "
                         f"the arm's weight and stalled when driven at --max-speed); capped at --max-speed")
    ap.add_argument("--max-speed", type=float, default=DEFAULT_SPEED,
                    help=f"per-joint goal rate limit, ticks/s (default {DEFAULT_SPEED} = 75%% of "
                         f"full speed, max {MAX_SPEED_CAP})")
    ap.add_argument("--stale-s", type=float, default=0.5, help="hold if no command for this long")
    ap.add_argument("--gripper", action="store_true",
                    help=f"also drive the gripper (ID 6) from {GRIP_TOPIC}")
    ap.add_argument("--grip-speed", type=float, default=1500.0, help="gripper rate limit, ticks/s")
    ap.add_argument("--home-speed", type=float, default=400.0,
                    help=f"peak ticks/s of any joint while homing (shoulder_lift: at most "
                         f"--lift-speed; max {MAX_SPEED_CAP})")
    ap.add_argument("--selftest", action="store_true", help="test the stall detector, no ROS or bus")
    ap.add_argument("--log", default=None, help="save every tick (time, goal, present, load, blocked "
                                                "per arm joint) to this .npz on exit")
    args = ap.parse_args()
    if args.selftest:
        return selftest()

    for name in ("max_speed", "home_speed", "grip_speed", "lift_speed"):
        if not 0 < getattr(args, name) <= MAX_SPEED_CAP:
            sys.exit(f"--{name.replace('_', '-')} must be in (0, {MAX_SPEED_CAP}]")
    if not 0 < args.alpha <= 1:
        sys.exit("--alpha must be in (0, 1]")
    if args.live and args.no_bus:
        sys.exit("--live needs the bus")

    rclpy.init()
    node = Mirror(args)
    # Callbacks on their own thread: they only store the newest command per joint. The loop
    # used to call spin_once() once per tick, which handles ONE message: 5 joint topics at
    # 30 Hz = 150 msg/s against 50 handled/s, so each joint's command updated every few ticks
    # (the goal moved in pulses: shaky) and queued commands were up to ~0.3 s old (lag).
    executor = SingleThreadedExecutor()
    executor.add_node(node)
    def spin():  # exits quietly when the executor is shut down at exit
        try:
            executor.spin()
        except Exception:
            pass
    threading.Thread(target=spin, name="ros", daemon=True).start()
    bus = None
    tick_log = []  # --log rows
    sids = {j: sb.IDS[j] for j in ARM_JOINTS}
    gsid = sb.IDS["gripper"] if args.gripper else None
    all_sids = dict(sids, gripper=gsid) if gsid is not None else dict(sids)
    hold_ids = list(all_sids.values())
    grip_close_dir = 1 if sb.GRIPPER_CLOSED > sb.GRIPPER_OPEN else -1
    grip_guard = GripGuard(grip_close_dir)
    gload = 0
    limits, eeprom, present, goals = {}, {}, {}, {}
    torque = False
    dt = 1.0 / args.rate
    # per joint: shoulder_lift slower (it holds the arm up; at full pace it stalled, likely
    # its overload protection), the rest at --max-speed
    speed = {j: min(args.max_speed, args.lift_speed) if j == "shoulder_lift" else args.max_speed
             for j in ARM_JOINTS}
    lead = {j: max_lead(v) for j, v in speed.items()}
    try:
        if not args.no_bus:
            bus = sb.Bus(args.port)
            for j, sid in all_sids.items():
                if not bus.ping(sid):
                    sys.exit(f"servo {sid} ({j}) not responding")
                eeprom[j] = bus.limits(sid)
                limits[j] = sb.effective_limits(j, eeprom[j])
                present[j] = bus.position(sid)
                # usable range = EEPROM limits combined with the measured range. Each
                # contains home (2048), so homing from inside stays inside.
                lo, hi = limits[j]
                if not lo <= present[j] <= hi:
                    why = ("the servo would jump to the EEPROM limit on torque-on"
                           if not eeprom[j][0] <= present[j] <= eeprom[j][1] else
                           "homing from here would leave the measured range of motion")
                    if j == "wrist_roll":
                        why += " (wrist_roll is past the 0/4095 seam: turn it back toward the middle)"
                    msg = (f"{j} at {present[j]} is outside its usable range [{lo}, {hi}]; "
                           f"{why}. Move it into range by hand first")
                    if args.live:
                        sys.exit(msg)
                    print("WARNING (would refuse --live): " + msg)
            print("present pose (rad, via JOINT_CAL):",
                  {j: round(sb.ticks_to_rad(j, present[j]), 3) for j in ARM_JOINTS})
            print("homing plan (present -> base pose):")
            for j in ARM_JOINTS:
                d = sb.HOME[j] - present[j]
                print(f"  {j:<14} {present[j]:>5} -> {sb.HOME[j]:>5}  ({d:+5d} ticks, "
                      f"{d * 360 / 4096:+6.1f} deg)  limits {list(limits[j])}")

        if args.live:
            print(f"LIVE: torque will be enabled on {', '.join(all_sids)}, the arm will "
                  f"move to its base pose (plan above, {args.home_speed:.0f} ticks/s), then "
                  f"follow /arm/* at <= {args.max_speed:.0f} ticks/s (shoulder_lift "
                  f"{speed['shoulder_lift']:.0f}) "
                  f"({args.max_speed * 360 / 4096:.0f} deg/s).\n"
                  + (f"The gripper follows {GRIP_TOPIC} (holds where it is until then).\n"
                     if gsid is not None else "")
                  + "JOINT_CAL must be verified first. Keep the area clear; Ctrl-C holds.")
            if not args.yes and input("Proceed? [y/N] ").strip().lower() != "y":
                sys.exit("aborted, nothing changed")
            for j, sid in all_sids.items():
                goals[j] = float(bus.torque_on_holding(sid, SERVO_ACCEL, SERVO_SPEED))
            torque = True
            grip_start = goals.get("gripper")
            goals = home(bus, sids, {j: goals[j] for j in sids}, args.home_speed, dt, args.lift_speed)
            if grip_start is not None:
                goals["gripper"] = grip_start  # the gripper isn't homed; it holds
        else:
            print("DRY RUN: no torque, no goal writes. Pass --live to move.\n")

        # EMA-filtered target ticks; starts at home so following begins there
        filt = {j: float(sb.HOME[j]) for j in ARM_JOINTS} if torque else {}
        last_print = 0.0
        waiting_logged = False
        sync_read = bus is not None and sync_read_works(bus, all_sids)
        block = BlockGuard(up={j: d for j, names in DIR_NAMES.items() for d, n in names.items() if n == "up"})
        n_blocks, n_stalls = 0, 0
        stalled = {}           # (status field, always empty: stalls are blocks now)
        loads, flags = {}, {}
        last_t, loop_hz, last_status, last_key, next_t = None, 0.0, 0.0, None, None
        while rclpy.ok():
            t0 = time.monotonic()
            # Speed limit from the real loop period: per-servo reads ran the loop at 44 Hz,
            # not 50, so --max-speed delivered ~12 % less. Capped at 2 periods so a slow
            # tick (bus retry) can't become one big step.
            period = dt if last_t is None else t0 - last_t
            if last_t is not None:
                loop_hz = 1 / period if not loop_hz else 0.9 * loop_hz + 0.1 / period
            last_t = t0
            max_step = {j: v * min(period, 2 * dt) for j, v in speed.items()}
            grip_step = args.grip_speed * min(period, 2 * dt)
            targets = node.fresh_targets()

            if bus is not None:
                if sync_read:  # one bus round trip for every servo's position, load, status
                    st = bus.read_states(all_sids.values())
                    for j, sid in all_sids.items():
                        present[j], loads[j], flags[j] = st[sid]["pos"], st[sid]["load"], st[sid]["flags"]
                else:
                    for j, sid in all_sids.items():
                        present[j] = bus.position(sid)
                        if torque or j == "gripper":
                            loads[j] = bus.load(sid)
                gload = loads.get("gripper", 0)
                node.publish_state(present)

            holding = False  # (status field kept; stalls no longer hold the whole arm)
            if torque:
                for j in ARM_JOINTS:
                    why = block.update(j, goals[j], present[j], loads.get(j, 0), t0)
                    if why:
                        d, at = block.blocked[j][:2]
                        goals[j] = block.hold_goal(j)
                        way = DIR_NAMES.get(j, {}).get(d, f"{d:+d} ticks")
                        extra = f" [{'/'.join(flags[j])}]" if flags.get(j) else ""
                        if why == "pushing":
                            n_blocks += 1
                            print(f"BLOCKED: {j} pushing {way} at {at} "
                                  f"({sb.ticks_to_rad(j, at) * 180 / math.pi:+.0f} deg), load "
                                  f"{loads.get(j, 0) / 10:.0f}%{extra}; won't push further that way", flush=True)
                        else:
                            n_stalls += 1
                            print(f"STALLED: {j} not moving {way} at {at} "
                                  f"({sb.ticks_to_rad(j, at) * 180 / math.pi:+.0f} deg), load "
                                  f"{loads.get(j, 0) / 10:.0f}%{extra}; won't push further that way", flush=True)

            if targets is None:
                if not waiting_logged:
                    print("no fresh /arm/* commands: holding")
                    waiting_logged = True
            else:
                waiting_logged = False
                for j in ARM_JOINTS:
                    lo, hi = limits.get(j, (0, 4095))
                    raw = max(lo, min(hi, sb.rad_to_ticks(j, targets[j])))
                    filt[j] = raw if j not in filt else filt[j] + args.alpha * (raw - filt[j])
                    if torque and not holding:
                        step = max(-max_step[j], min(max_step[j], filt[j] - goals[j]))
                        g = goals[j] + step
                        # never lead the joint by more than `lead` ticks: a slow
                        # joint then moves at its own best speed instead of the goal
                        # running away and tripping the lag abort
                        g = max(present[j] - lead[j], min(present[j] + lead[j], g))
                        was = j in block.blocked
                        g = block.clamp(j, g, filt[j])  # blocked: not further into the obstacle
                        if was and j not in block.blocked:
                            print(f"unblocked: {j} (command moved back)", flush=True)
                        goals[j] = max(lo, min(hi, g))
                if torque:
                    bus.send_goals({sids[j]: goals[j] for j in ARM_JOINTS})

            grip = node.fresh_grip() if gsid is not None else None
            grip_raw = None
            if grip is not None:
                lo, hi = limits.get("gripper", (0, 4095))
                grip_raw = max(lo, min(hi, sb.GRIPPER_OPEN + (sb.GRIPPER_CLOSED - sb.GRIPPER_OPEN) * grip))
                if torque:
                    step = max(-grip_step, min(grip_step, grip_raw - goals["gripper"]))
                    g = goals["gripper"] + step
                    if abs(gload) > LOAD_ABORT:
                        g = present["gripper"]  # way overloaded: stop pushing entirely
                    else:
                        held = grip_guard.contact is not None
                        g = grip_guard.update(g, grip_raw, present["gripper"], gload, t0)
                        if held != (grip_guard.contact is not None):
                            print(f"gripper: {'holding an object at ' + str(grip_guard.contact) if not held else 'let go'}",
                                  flush=True)
                    goals["gripper"] = max(lo, min(hi, g))
                    bus.send_goals({gsid: goals["gripper"]})

            if torque:
                for j in ARM_JOINTS:
                    lag = abs(goals[j] - present[j])
                    load = loads[j]
                    if lag > lag_limit(speed[j]) or abs(load) > LOAD_ABORT:
                        held = bus.hold(hold_ids)
                        sys.exit(f"ABORT: {j} lag {lag:.0f} ticks, load {load / 10:.0f}%. "
                                 f"Holding at {held}.")

            if args.log and bus is not None:
                tick_log.append([t0] + [goals.get(j, np.nan) for j in ARM_JOINTS] + [present[j] for j in ARM_JOINTS]
                                + [loads.get(j, 0) for j in ARM_JOINTS]
                                + [float(j in block.blocked) for j in ARM_JOINTS])
            # status for hand_target_node's preview: now on a change, else every STATUS_EVERY_S
            key = (holding, tuple(stalled), tuple(block.blocked), tuple(j for j in flags if flags[j]))
            if key != last_key or t0 - last_status >= STATUS_EVERY_S:
                node.publish_status({
                    "t": time.time(), "live": torque, "hz": round(loop_hz, 1),
                    "holding": holding, "hold_left": 0.0,
                    "stalled": {j: round(s, 2) for j, s in stalled.items()}, "stalls": n_stalls,
                    "blocked": {j: DIR_NAMES.get(j, {}).get(b[0], f"{b[0]:+d} ticks")
                                + ("" if b[3] == "pushing" else ", stalled")
                                for j, b in block.blocked.items()}, "blocks": n_blocks,
                    "flags": {j: f for j, f in flags.items() if f},
                    "err_deg": {j: round((goals[j] - present[j]) * 360 / 4096, 1)
                                for j in ARM_JOINTS if torque},
                    "load": {j: round(v / 10) for j, v in loads.items()}})
                last_status, last_key = t0, key

            if not args.live and time.monotonic() - last_print > 0.5:
                print_table(targets, present, limits, filt)
                if gsid is not None:
                    print(f"gripper: cmd {'-' if grip is None else f'{grip:.2f}'} -> ticks "
                          f"{'-' if grip_raw is None else f'{grip_raw:.0f}'}, present "
                          f"{present.get('gripper', '-')}, load {gload / 10:.0f}%  "
                          f"(open {sb.GRIPPER_OPEN}, closed {sb.GRIPPER_CLOSED})\n")
                last_print = time.monotonic()

            # sleep to a fixed schedule: macOS oversleeps ~4.5 ms per 20 ms sleep, so sleeping
            # the rest of each tick ran the loop at 44 Hz; deadlines keep the average at --rate
            next_t = (next_t or t0) + dt
            if next_t < time.monotonic() - dt:  # more than a tick behind: don't catch up in a burst
                next_t = time.monotonic()
            time.sleep(max(0.0, next_t - time.monotonic()))
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:  # bus error that survived the retries
        print(f"\nBUS ERROR: {e}")
    finally:
        if args.log and tick_log:
            np.savez(args.log, ticks=np.array(tick_log, dtype=float), joints=np.array(ARM_JOINTS))
            print(f"tick log: {args.log} ({len(tick_log)} ticks)")
        if bus is not None:
            if torque:
                bus.port.clearPort()  # Ctrl-C can land mid-transaction: drop the half packet
                time.sleep(0.02)
                print(f"\nstopped, holding at {bus.hold(hold_ids)} (torque ON)")
            if bus.retries:
                print(f"bus: {bus.retries} transaction(s) needed a retry this run")
            bus.close()
        executor.shutdown()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
