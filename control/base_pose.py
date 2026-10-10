#!/home/peyton/kh/.venv/bin/python
"""Move the SO-ARM101 to its base pose: every joint at 2048 (calibration middle).

All joints move together along a straight line in joint space at a slow,
capped speed, so they start and finish at the same time.

Usage:
    python base_pose.py --dry-run          # show the plan, no motion
    python base_pose.py                    # show the plan, ask, then move
    python base_pose.py --joints 1 2 3     # only some joints
    python base_pose.py --speed 100        # ticks/s for the largest move

Safety:
    - Reads Present_Position first; torque is enabled holding that pose with
      low servo speed/accel before any goal changes.
    - Targets are clamped to the EEPROM Min/Max_Position_Limit (read only).
    - Aborts and holds the current pose if a joint lags its goal by more than
      MAX_LAG ticks (blocked/collision) or its load exceeds LOAD_ABORT, or on
      Ctrl-C. Torque stays ON at the end so the arm holds the pose.
"""
import argparse
import sys
import time

import scservo_sdk as scs

BAUD = 1_000_000
NAMES = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex",
         4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}
BASE_POSE = {sid: 2048 for sid in NAMES}

# STS3215 control table (address)
MIN_POSITION_LIMIT = 9
MAX_POSITION_LIMIT = 11
TORQUE_ENABLE = 40
ACCELERATION = 41
GOAL_POSITION = 42
GOAL_SPEED = 46
PRESENT_POSITION = 56
PRESENT_LOAD = 60

SERVO_SPEED = 600   # servo-side speed cap (ticks/s); 0 would mean max
SERVO_ACCEL = 20
MAX_SPEED = 300     # refuse --speed above this (ticks/s)
DT = 0.02           # goal update period (s)
MAX_LAG = 200       # ticks between goal and present before aborting
LOAD_ABORT = 700    # 0.1 % units (70 %)
TOL = 15            # ticks to count as arrived


class Bus:
    def __init__(self, port_name):
        self.port = scs.PortHandler(port_name)
        self.pkt = scs.PacketHandler(0)
        if not self.port.openPort():
            sys.exit(f"could not open {port_name} (attached via usbipd? chmod 666?)")
        if not self.port.setBaudRate(BAUD):
            sys.exit(f"could not set baud {BAUD}")
        self.sync = scs.GroupSyncWrite(self.port, self.pkt, GOAL_POSITION, 2)

    def r2(self, sid, addr):
        v, res, _ = self.pkt.read2ByteTxRx(self.port, sid, addr)
        if res != scs.COMM_SUCCESS:
            raise RuntimeError(f"read addr {addr} id {sid}: {self.pkt.getTxRxResult(res)}")
        return v

    def w1(self, sid, addr, val):
        res, _ = self.pkt.write1ByteTxRx(self.port, sid, addr, val)
        if res != scs.COMM_SUCCESS:
            raise RuntimeError(f"write addr {addr} id {sid}: {self.pkt.getTxRxResult(res)}")

    def w2(self, sid, addr, val):
        res, _ = self.pkt.write2ByteTxRx(self.port, sid, addr, val)
        if res != scs.COMM_SUCCESS:
            raise RuntimeError(f"write addr {addr} id {sid}: {self.pkt.getTxRxResult(res)}")

    def load(self, sid):
        v = self.r2(sid, PRESENT_LOAD)  # sign-magnitude, sign in bit 10
        return -(v & 0x3FF) if v & 0x400 else v

    def send_goals(self, goals):
        self.sync.clearParam()
        for sid, g in goals.items():
            g = int(round(g))
            self.sync.addParam(sid, [scs.SCS_LOBYTE(g), scs.SCS_HIBYTE(g)])
        res = self.sync.txPacket()
        if res != scs.COMM_SUCCESS:
            raise RuntimeError(f"sync write: {self.pkt.getTxRxResult(res)}")

    def hold(self, sids):
        """Set goal = present position for each joint (stops motion)."""
        goals = {}
        for sid in sids:
            try:
                goals[sid] = self.r2(sid, PRESENT_POSITION)
            except RuntimeError:
                pass
        if goals:
            self.send_goals(goals)
        return goals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--joints", type=int, nargs="+", default=list(NAMES),
                    help="joint IDs to move (default: all)")
    ap.add_argument("--speed", type=float, default=150,
                    help=f"ticks/s for the largest move (max {MAX_SPEED})")
    ap.add_argument("--dry-run", action="store_true", help="print the plan only")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    if not 0 < args.speed <= MAX_SPEED:
        sys.exit(f"--speed must be in (0, {MAX_SPEED}]")
    bad = [j for j in args.joints if j not in NAMES]
    if bad:
        sys.exit(f"unknown joint IDs: {bad}")

    bus = Bus(args.port)
    sids = []
    try:
        start, target = {}, {}
        print(f"{'id':>2} {'joint':<14} {'now':>5} {'target':>6} {'delta':>6} {'deg':>7}  limits")
        for sid in args.joints:
            _, res, _ = bus.pkt.ping(bus.port, sid)
            if res != scs.COMM_SUCCESS:
                sys.exit(f"servo {sid} ({NAMES[sid]}) not responding")
            pos = bus.r2(sid, PRESENT_POSITION)
            lo, hi = bus.r2(sid, MIN_POSITION_LIMIT), bus.r2(sid, MAX_POSITION_LIMIT)
            if lo >= hi:
                lo, hi = 0, 4095
            if not lo <= pos <= hi:
                sys.exit(f"{NAMES[sid]} at {pos} is outside its limits [{lo}, {hi}]; "
                         "the servo would jump to the limit on torque-on. Move it by hand first.")
            tgt = max(lo, min(hi, BASE_POSE[sid]))
            start[sid], target[sid] = pos, tgt
            d = tgt - pos
            print(f"{sid:>2} {NAMES[sid]:<14} {pos:>5} {tgt:>6} {d:>+6} {d * 360 / 4096:>+7.1f}  [{lo}, {hi}]")
        sids = list(start)

        biggest = max(abs(target[s] - start[s]) for s in sids)
        duration = max(biggest / args.speed, 0.5)
        print(f"\nlargest move {biggest} ticks ({biggest * 360 / 4096:.0f} deg), "
              f"~{duration:.1f} s at {args.speed:.0f} ticks/s")
        if biggest <= TOL:
            print("already at base pose")
            return
        if args.dry_run:
            return
        if not args.yes:
            print("Torque will be enabled on these joints and the arm will move. "
                  "Keep the area clear; Ctrl-C stops and holds.")
            if input("Proceed? [y/N] ").strip().lower() != "y":
                print("aborted, nothing changed")
                return

        # enable torque holding the current pose, with gentle servo-side limits
        for sid in sids:
            bus.w1(sid, ACCELERATION, SERVO_ACCEL)
            bus.w2(sid, GOAL_SPEED, SERVO_SPEED)
            bus.w2(sid, GOAL_POSITION, start[sid])
            bus.w1(sid, TORQUE_ENABLE, 1)

        # linear interpolation in joint space with smoothstep easing
        t0 = time.time()
        while True:
            s = min((time.time() - t0) / duration, 1.0)
            e = s * s * (3 - 2 * s)
            goals = {sid: start[sid] + (target[sid] - start[sid]) * e for sid in sids}
            bus.send_goals(goals)
            for sid in sids:
                pos = bus.r2(sid, PRESENT_POSITION)
                lag = abs(goals[sid] - pos)
                load = bus.load(sid)
                if lag > MAX_LAG or abs(load) > LOAD_ABORT:
                    held = bus.hold(sids)
                    sys.exit(f"ABORT: {NAMES[sid]} lag {lag:.0f} ticks, load {load / 10:.0f}%. "
                             f"Holding at {held}.")
            if s >= 1.0:
                break
            time.sleep(DT)

        # let the servos settle on the final goal
        t1 = time.time()
        while time.time() - t1 < 2.0:
            if all(abs(bus.r2(s, PRESENT_POSITION) - target[s]) <= TOL for s in sids):
                break
            time.sleep(0.05)

        print("\nfinal:")
        for sid in sids:
            pos = bus.r2(sid, PRESENT_POSITION)
            print(f"{sid:>2} {NAMES[sid]:<14} {pos:>5}  (target {target[sid]}, err {pos - target[sid]:+d}, "
                  f"load {bus.load(sid) / 10:.0f}%)")
        print("torque left ON (holding). Turning it off will drop the arm.")
    except KeyboardInterrupt:
        if sids:
            print(f"\nstopped, holding at {bus.hold(sids)}")
    finally:
        bus.port.closePort()


if __name__ == "__main__":
    main()
