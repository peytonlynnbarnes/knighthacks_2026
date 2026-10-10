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
    ./feetech_mirror.py --no-bus            # no arm attached: print the
                                            # ticks the /arm/* topics map to

Safety:
    - JOINT_CAL signs/offsets are NOT verified on hardware yet. Do the
      calibration check in ~/kh/INTEGRATION.md before the first --live run.
    - Torque on per CLAUDE.md: read present, goal = present, low servo
      accel/speed, then Torque_Enable. Refused if a joint is outside its
      usable range (EEPROM limits, read only, never written, combined with
      SOFT_LIMITS).
    - Homing: smoothstep ramp at --home-speed ticks/s with the same
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
    - Default following speed is 75 % of the servo's no-load maximum
      (2250 ticks/s ~ 200 deg/s). Use a lower --max-speed for first tests.
    - Ctrl-C / exit: goal = present, torque stays ON (the arm holds, it
      does not go limp). Use torque_off.py to release it.
    - Gripper (--gripper): torque on holding where it is (no homing), clamped
      to its usable range, rate limited (--grip-speed), holds when /gripper
      goes stale. It isn't in the lag/load abort (gripping an object stalls
      it by design); instead, above GRIP_LOAD it stops closing further and
      keeps the goal only GRIP_SQUEEZE ticks past where the jaws stopped.
"""
import argparse
import math
import sys
import time

try:
    import rclpy
    from rclpy.node import Node
    from sensor_msgs.msg import JointState
    from std_msgs.msg import Float64
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
MAX_LAG = 250        # ticks between goal and present before aborting, at low speed
LAG_TIME = 0.25      # ...grows with speed: limit = max(MAX_LAG, LAG_TIME * speed)
MIN_LEAD = 150       # max goal lead over present while following (ticks), at low speed
LEAD_TIME = 0.15     # ...grows with speed: lead = max(MIN_LEAD, LEAD_TIME * speed)
LOAD_ABORT = 700     # 0.1 % units (70 %)
HOME_TOL = 15        # ticks to count as arrived at home
GRIP_TOPIC = "/gripper"  # must match hand_target_node.GRIP_TOPIC
GRIP_LOAD = 400      # 0.1 % units: above this the gripper is holding something...
GRIP_SQUEEZE = 40    # ...so its goal stays at most this many ticks past present


class Mirror(Node):
    def __init__(self, args):
        super().__init__("feetech_mirror")
        self.args = args
        self.cmd = {}      # joint -> (rad, monotonic time)
        for j in ARM_JOINTS:
            self.create_subscription(Float64, ARM_TOPIC_PREFIX + j,
                                     lambda m, j=j: self.on_cmd(j, m.data), 10)
        self.state_pub = self.create_publisher(JointState, "/hw/joint_states", 10)
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
        if len(self.cmd) < len(ARM_JOINTS):
            return None
        if any(now - t > self.args.stale_s for _, t in self.cmd.values()):
            return None
        return {j: self.cmd[j][0] for j in ARM_JOINTS}

    def publish_state(self, present):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        arm = {j: t for j, t in present.items() if j in sb.JOINT_CAL}  # gripper has no rad cal
        msg.name = list(arm)
        msg.position = [sb.ticks_to_rad(j, t) for j, t in arm.items()]
        self.state_pub.publish(msg)


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


def max_lead(speed):
    """Goal lead over present while following; kept below lag_limit so the
    lag abort only fires when a joint is pushed away or drops, not when it's
    merely slower than requested (load abort still catches blocking)."""
    return max(MIN_LEAD, LEAD_TIME * speed)


def home(bus, sids, start, speed, dt):
    """Ease every joint from `start` to sb.HOME together (smoothstep), like
    base_pose.py. Aborts (holds, exits) on lag/load. Returns the final goals."""
    target = {j: float(sb.HOME[j]) for j in sids}
    biggest = max(abs(target[j] - start[j]) for j in sids)
    duration = max(biggest / speed, 1.0)
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
            if lag > lag_limit(speed) or abs(load) > LOAD_ABORT:
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
    ap.add_argument("--alpha", type=float, default=0.3, help="EMA factor on targets (0..1]")
    ap.add_argument("--max-speed", type=float, default=DEFAULT_SPEED,
                    help=f"per-joint goal rate limit, ticks/s (default {DEFAULT_SPEED} = 75%% of "
                         f"full speed, max {MAX_SPEED_CAP})")
    ap.add_argument("--stale-s", type=float, default=0.5, help="hold if no command for this long")
    ap.add_argument("--gripper", action="store_true",
                    help=f"also drive the gripper (ID 6) from {GRIP_TOPIC}")
    ap.add_argument("--grip-speed", type=float, default=1500.0, help="gripper rate limit, ticks/s")
    ap.add_argument("--home-speed", type=float, default=400.0,
                    help=f"ticks/s for the largest joint while homing (max {MAX_SPEED_CAP})")
    args = ap.parse_args()

    for name in ("max_speed", "home_speed", "grip_speed"):
        if not 0 < getattr(args, name) <= MAX_SPEED_CAP:
            sys.exit(f"--{name.replace('_', '-')} must be in (0, {MAX_SPEED_CAP}]")
    if not 0 < args.alpha <= 1:
        sys.exit("--alpha must be in (0, 1]")
    if args.live and args.no_bus:
        sys.exit("--live needs the bus")

    rclpy.init()
    node = Mirror(args)
    bus = None
    sids = {j: sb.IDS[j] for j in ARM_JOINTS}
    gsid = sb.IDS["gripper"] if args.gripper else None
    all_sids = dict(sids, gripper=gsid) if gsid is not None else dict(sids)
    hold_ids = list(all_sids.values())
    grip_close_dir = 1 if sb.GRIPPER_CLOSED > sb.GRIPPER_OPEN else -1
    gload = 0
    limits, eeprom, present, goals = {}, {}, {}, {}
    torque = False
    dt = 1.0 / args.rate
    max_step = args.max_speed * dt
    grip_step = args.grip_speed * dt
    lead = max_lead(args.max_speed)
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
                  f"follow /arm/* at <= {args.max_speed:.0f} ticks/s "
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
            goals = home(bus, sids, {j: goals[j] for j in sids}, args.home_speed, dt)
            if grip_start is not None:
                goals["gripper"] = grip_start  # the gripper isn't homed; it holds
        else:
            print("DRY RUN: no torque, no goal writes. Pass --live to move.\n")

        # EMA-filtered target ticks; starts at home so following begins there
        filt = {j: float(sb.HOME[j]) for j in ARM_JOINTS} if torque else {}
        last_print = 0.0
        waiting_logged = False
        while rclpy.ok():
            t0 = time.monotonic()
            rclpy.spin_once(node, timeout_sec=0)
            targets = node.fresh_targets()

            if bus is not None:
                for j, sid in sids.items():
                    present[j] = bus.position(sid)
                if gsid is not None:
                    present["gripper"] = bus.position(gsid)
                    gload = bus.load(gsid)
                node.publish_state(present)

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
                    if torque:
                        step = max(-max_step, min(max_step, filt[j] - goals[j]))
                        g = goals[j] + step
                        # never lead the joint by more than `lead` ticks: a slow
                        # joint then moves at its own best speed instead of the goal
                        # running away and tripping the lag abort
                        g = max(present[j] - lead, min(present[j] + lead, g))
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
                    past = (g - present["gripper"]) * grip_close_dir  # how far the goal is past the jaws
                    if abs(gload) > LOAD_ABORT:
                        g = present["gripper"]  # way overloaded: stop pushing entirely
                    elif abs(gload) > GRIP_LOAD and past > GRIP_SQUEEZE:
                        g = present["gripper"] + grip_close_dir * GRIP_SQUEEZE  # holding: don't squeeze harder
                    goals["gripper"] = max(lo, min(hi, g))
                    bus.send_goals({gsid: goals["gripper"]})

            if torque:
                for j, sid in sids.items():
                    lag = abs(goals[j] - present[j])
                    load = bus.load(sid)
                    if lag > lag_limit(args.max_speed) or abs(load) > LOAD_ABORT:
                        held = bus.hold(hold_ids)
                        sys.exit(f"ABORT: {j} lag {lag:.0f} ticks, load {load / 10:.0f}%. "
                                 f"Holding at {held}.")

            if not args.live and time.monotonic() - last_print > 0.5:
                print_table(targets, present, limits, filt)
                if gsid is not None:
                    print(f"gripper: cmd {'-' if grip is None else f'{grip:.2f}'} -> ticks "
                          f"{'-' if grip_raw is None else f'{grip_raw:.0f}'}, present "
                          f"{present.get('gripper', '-')}, load {gload / 10:.0f}%  "
                          f"(open {sb.GRIPPER_OPEN}, closed {sb.GRIPPER_CLOSED})\n")
                last_print = time.monotonic()

            time.sleep(max(0.0, dt - (time.monotonic() - t0)))
    except KeyboardInterrupt:
        pass
    except RuntimeError as e:  # bus error that survived the retries
        print(f"\nBUS ERROR: {e}")
    finally:
        if bus is not None:
            if torque:
                print(f"\nstopped, holding at {bus.hold(hold_ids)} (torque ON)")
            if bus.retries:
                print(f"bus: {bus.retries} transaction(s) needed a retry this run")
            bus.close()
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == "__main__":
    main()
