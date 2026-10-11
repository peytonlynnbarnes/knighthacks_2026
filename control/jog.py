#!/home/peyton/kh/.venv/bin/python
"""Keyboard jog tool for the SO-ARM101 (Feetech STS3215, scservo_sdk).

Keys:
    1-6          select joint
    left/right   jog selected joint -/+ by step   (also a/d)
    up/down      bigger/smaller step              (also w/s)
    h            hold: set goal = present position (stops a moving joint)
    t            torque ON for selected joint (holds current pose)
    x x          torque OFF for selected joint (press twice; the joint goes limp!)
    q            quit (torque left as-is, so held joints keep holding)

Safety:
    - Torque is never enabled at startup; the first jog of a joint enables it
      after writing Goal_Position = Present_Position, with low speed/accel.
    - Goals are clamped to the EEPROM Min/Max_Position_Limit (read only, never
      written). Torque is refused if the joint is currently outside them, since
      the servo would clamp the goal and jump to the limit.
    - The goal never runs more than MAX_LEAD ticks ahead of the present
      position, so holding a key against an obstacle doesn't wind up.
"""
import argparse
import curses
import time

import scservo_sdk as scs

from so101_bus import OPEN_HINT, PORT_HELP, default_port

BAUD = 1_000_000
NAMES = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex",
         4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}

# STS3215 control table (address)
MIN_POSITION_LIMIT = 9
MAX_POSITION_LIMIT = 11
TORQUE_ENABLE = 40
ACCELERATION = 41
GOAL_POSITION = 42
GOAL_SPEED = 46
PRESENT_POSITION = 56
PRESENT_LOAD = 60
PRESENT_VOLTAGE = 62
PRESENT_TEMPERATURE = 63

STEPS = [5, 10, 20, 50, 100]
SPEED = 300       # ticks/s (0 would mean max speed)
ACCEL = 20
MAX_LEAD = 150    # max |goal - present| in ticks
LOAD_WARN = 500   # 0.1 % units (50 %)
TEMP_WARN = 55    # deg C


class Arm:
    def __init__(self, port_name):
        self.port = scs.PortHandler(port_name)
        self.pkt = scs.PacketHandler(0)
        if not self.port.openPort():
            raise SystemExit(f"could not open {port_name} ({OPEN_HINT})")
        if not self.port.setBaudRate(BAUD):
            raise SystemExit(f"could not set baud {BAUD}")
        self.joints = {}
        for sid in NAMES:
            _, res, _ = self.pkt.ping(self.port, sid)
            if res != scs.COMM_SUCCESS:
                continue
            j = {"min": self.r2(sid, MIN_POSITION_LIMIT),
                 "max": self.r2(sid, MAX_POSITION_LIMIT),
                 "torque": self.r1(sid, TORQUE_ENABLE) == 1,
                 "goal": None, "pos": None, "load": 0, "volt": 0.0, "temp": 0}
            if j["min"] >= j["max"]:  # unset/odd limits: fall back to full range
                j["min"], j["max"] = 0, 4095
            self.joints[sid] = j
        if not self.joints:
            raise SystemExit("no servos responded")
        self.refresh()
        for sid, j in self.joints.items():
            if j["torque"]:  # already holding from a previous run
                j["goal"] = self.r2(sid, GOAL_POSITION)

    def r1(self, sid, addr):
        v, res, _ = self.pkt.read1ByteTxRx(self.port, sid, addr)
        return v if res == scs.COMM_SUCCESS else None

    def r2(self, sid, addr):
        v, res, _ = self.pkt.read2ByteTxRx(self.port, sid, addr)
        return v if res == scs.COMM_SUCCESS else None

    def w1(self, sid, addr, val):
        res, _ = self.pkt.write1ByteTxRx(self.port, sid, addr, val)
        return res == scs.COMM_SUCCESS

    def w2(self, sid, addr, val):
        res, _ = self.pkt.write2ByteTxRx(self.port, sid, addr, val)
        return res == scs.COMM_SUCCESS

    def refresh(self):
        for sid, j in self.joints.items():
            pos = self.r2(sid, PRESENT_POSITION)
            if pos is not None:
                j["pos"] = pos
            load = self.r2(sid, PRESENT_LOAD)
            if load is not None:  # sign-magnitude, sign in bit 10
                j["load"] = -(load & 0x3FF) if load & 0x400 else load
            volt = self.r1(sid, PRESENT_VOLTAGE)
            if volt is not None:
                j["volt"] = volt / 10
            temp = self.r1(sid, PRESENT_TEMPERATURE)
            if temp is not None:
                j["temp"] = temp

    def in_range(self, sid):
        j = self.joints[sid]
        return j["min"] <= j["pos"] <= j["max"]

    def torque_on(self, sid):
        """Enable torque holding the current pose. Returns an error string or None."""
        j = self.joints[sid]
        pos = self.r2(sid, PRESENT_POSITION)
        if pos is None:
            return "read failed"
        j["pos"] = pos
        if not self.in_range(sid):
            return (f"pos {pos} outside limits [{j['min']}, {j['max']}]: "
                    "move it into range by hand first")
        self.w1(sid, ACCELERATION, ACCEL)
        self.w2(sid, GOAL_SPEED, SPEED)
        self.w2(sid, GOAL_POSITION, pos)
        if not self.w1(sid, TORQUE_ENABLE, 1):
            return "torque write failed"
        j["torque"], j["goal"] = True, pos
        return None

    def torque_off(self, sid):
        if self.w1(sid, TORQUE_ENABLE, 0):
            j = self.joints[sid]
            j["torque"], j["goal"] = False, None

    def hold(self, sid):
        j = self.joints[sid]
        pos = self.r2(sid, PRESENT_POSITION)
        if j["torque"] and pos is not None:
            j["goal"] = pos
            self.w2(sid, GOAL_POSITION, pos)

    def jog(self, sid, delta):
        """Move goal by delta. Returns an error string or None."""
        j = self.joints[sid]
        if not j["torque"]:
            err = self.torque_on(sid)
            if err:
                return err
        pos = self.r2(sid, PRESENT_POSITION)
        if pos is None:
            return "read failed"
        goal = j["goal"] + delta
        goal = max(pos - MAX_LEAD, min(pos + MAX_LEAD, goal))
        goal = max(j["min"], min(j["max"], goal))
        j["goal"] = goal
        if not self.w2(sid, GOAL_POSITION, goal):
            return "goal write failed"
        return None

    def close(self):
        self.port.closePort()


def draw(scr, arm, sel, step, msg):
    scr.erase()
    scr.addstr(0, 0, "SO-ARM101 jog   1-6 select  <-/-> jog  up/down step  "
                     "h hold  t torque on  x x torque off  q quit")
    scr.addstr(2, 0, f"   {'id':>2} {'joint':<14} {'pos':>5} {'goal':>5} "
                     f"{'min':>5} {'max':>5} {'deg':>7} {'load%':>6} {'V':>5} "
                     f"{'C':>3}  torque")
    for row, (sid, j) in enumerate(arm.joints.items(), start=3):
        deg = (j["pos"] - 2048) * 360 / 4096
        goal = "-" if j["goal"] is None else str(j["goal"])
        warn = ""
        if not arm.in_range(sid):
            warn += "  OUT OF LIMITS"
        if abs(j["load"]) >= LOAD_WARN:
            warn += "  HIGH LOAD"
        if j["temp"] >= TEMP_WARN:
            warn += "  HOT"
        line = (f"{'>' if sid == sel else ' '}  {sid:>2} {NAMES[sid]:<14} "
                f"{j['pos']:>5} {goal:>5} {j['min']:>5} {j['max']:>5} "
                f"{deg:>7.1f} {j['load'] / 10:>6.1f} {j['volt']:>5.1f} "
                f"{j['temp']:>3}  {'ON ' if j['torque'] else 'off'}{warn}")
        attr = curses.A_REVERSE if sid == sel else curses.A_NORMAL
        if warn:
            attr |= curses.A_BOLD
        scr.addstr(row, 0, line, attr)
    base = 4 + len(arm.joints)
    scr.addstr(base, 0, f"step: {step} ticks ({step * 360 / 4096:.1f} deg)   "
                        f"speed {SPEED}  accel {ACCEL}")
    scr.addstr(base + 2, 0, msg[:curses.COLS - 1], curses.A_BOLD)
    scr.refresh()


def run(scr, arm):
    curses.curs_set(0)
    scr.timeout(50)  # also sets the refresh rate (~10-20 Hz with bus reads)
    sel = next(iter(arm.joints))
    step_i = 2
    msg = "torque is off on all joints until you jog or press t"
    off_armed_at = 0.0
    while True:
        arm.refresh()
        draw(scr, arm, sel, STEPS[step_i], msg)
        k = scr.getch()
        if k == -1:
            continue
        if k != ord("x"):
            off_armed_at = 0.0
        if k == ord("q"):
            return
        elif ord("1") <= k <= ord("6") and (k - ord("0")) in arm.joints:
            sel = k - ord("0")
            msg = f"selected {NAMES[sel]}"
        elif k in (curses.KEY_LEFT, ord("a"), curses.KEY_RIGHT, ord("d")):
            sign = 1 if k in (curses.KEY_RIGHT, ord("d")) else -1
            err = arm.jog(sel, sign * STEPS[step_i])
            msg = f"{NAMES[sel]}: {err}" if err else ""
        elif k in (curses.KEY_UP, ord("w")):
            step_i = min(step_i + 1, len(STEPS) - 1)
        elif k in (curses.KEY_DOWN, ord("s")):
            step_i = max(step_i - 1, 0)
        elif k == ord("h"):
            arm.hold(sel)
            msg = f"{NAMES[sel]}: holding"
        elif k == ord("t"):
            err = arm.torque_on(sel)
            msg = f"{NAMES[sel]}: {err or 'torque ON'}"
        elif k == ord("x"):
            if time.time() - off_armed_at < 2.0:
                arm.torque_off(sel)
                msg = f"{NAMES[sel]}: torque OFF (limp)"
                off_armed_at = 0.0
            else:
                off_armed_at = time.time()
                msg = (f"WARNING: {NAMES[sel]} will go limp and may drop. "
                       "Support it, then press x again within 2 s.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help=PORT_HELP)
    args = ap.parse_args()
    args.port = args.port or default_port()
    arm = Arm(args.port)
    try:
        curses.wrapper(run, arm)
    finally:
        arm.refresh()
        for sid, j in arm.joints.items():
            print(f"{sid} {NAMES[sid]:<14} pos {j['pos']:>5}  "
                  f"torque {'ON (holding)' if j['torque'] else 'off'}")
        arm.close()


if __name__ == "__main__":
    main()
