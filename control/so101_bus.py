"""Shared SO-ARM101 servo helpers (Feetech STS3215 via scservo_sdk).

Constants, a small bus wrapper and the tick <-> radian conversion, factored
out of base_pose.py for the real-time loop (feetech_mirror.py). The older
standalone scripts keep their own copies and still work unchanged.

Not a script; import it:
    from so101_bus import Bus, NAMES, rad_to_ticks, ticks_to_rad
"""
import math
import sys

import scservo_sdk as scs

BAUD = 1_000_000
NAMES = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex",
         4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}
IDS = {name: sid for sid, name in NAMES.items()}

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

RETRIES = 2  # extra attempts per bus transaction before raising

TICKS_PER_REV = 4096
CENTER = 2048

# URDF joint angle (rad) <-> servo ticks, per joint: (sign, zero_tick).
#   ticks = zero_tick + sign * rad * 4096 / 2pi
# Assumes the URDF zero pose (L-pose: upper arm ~vertical, forearm and
# gripper straight forward along +X) is the LeRobot-calibrated middle (2048,
# what base_pose.py moves to) and the URDF axis turns the same way as
# increasing ticks. NOT VERIFIED ON HARDWARE YET. Check each joint before
# trusting IK output (procedure in ~/kh/INTEGRATION.md, "Calibration").
JOINT_CAL = {
    "shoulder_pan": (+1, CENTER),
    "shoulder_lift": (+1, CENTER),
    "elbow_flex": (+1, CENTER),
    "wrist_flex": (+1, CENTER),
    "wrist_roll": (+1, CENTER),
}


# Home = base pose (base_pose.py): every arm joint at 2048. Visually checked
# on hardware (2026-10-09) to match the URDF zero L-pose.
HOME = {name: CENTER for name in JOINT_CAL}

# Soft limits (ticks) from the hand-measured range of motion
# (calib/ranges_session1.json), applied on top of the EEPROM limits.
# wrist_roll's travel crosses the 0/4095 seam (2878 -> 4095/0 -> 2645), and
# position mode can't pass the seam, so it's kept on the side containing
# home (2048): 0..2645.
SOFT_LIMITS = {
    "shoulder_pan": (1272, 3000),
    "shoulder_lift": (815, 3199),
    "elbow_flex": (944, 3176),
    "wrist_flex": (694, 3071),
    "wrist_roll": (0, 2645),
    "gripper": (2007, 3523),
}

# Gripper command range (ticks), for feetech_mirror --gripper. Closed jaws read
# ~2030 (user-confirmed 2026-10-10), so low ticks = closed. Open stops short of
# the EEPROM max (3298). Closing past an object is limited by load, not by this.
GRIPPER_CLOSED = 2040
GRIPPER_OPEN = 2900


def effective_limits(name, eeprom):
    """Intersection of the EEPROM limits and SOFT_LIMITS (ticks)."""
    lo, hi = eeprom
    slo, shi = SOFT_LIMITS.get(name, (0, TICKS_PER_REV - 1))
    return max(lo, slo), min(hi, shi)


def rad_to_ticks(name, rad):
    sign, zero = JOINT_CAL[name]
    return zero + sign * rad * TICKS_PER_REV / (2 * math.pi)


def ticks_to_rad(name, ticks):
    sign, zero = JOINT_CAL[name]
    return sign * (ticks - zero) * 2 * math.pi / TICKS_PER_REV


class Bus:
    def __init__(self, port_name):
        self.port = scs.PortHandler(port_name)
        self.pkt = scs.PacketHandler(0)  # protocol 0 = STS series
        try:
            opened = self.port.openPort()
        except Exception as e:  # pyserial raises instead of returning False
            print(e)
            opened = False
        if not opened:
            sys.exit(f"could not open {port_name} (attached via usbipd? chmod 666?)")
        if not self.port.setBaudRate(BAUD):
            sys.exit(f"could not set baud {BAUD}")
        self.sync = scs.GroupSyncWrite(self.port, self.pkt, GOAL_POSITION, 2)
        self.retries = 0  # transactions that needed a retry (for reporting)

    def ping(self, sid):
        _, res, _ = self.pkt.ping(self.port, sid)
        return res == scs.COMM_SUCCESS

    def _retry(self, what, fn):
        """Run one bus transaction, retrying up to RETRIES times. A single
        missed status packet (seen while several servos move, presumably a
        supply/USB hiccup) shouldn't abort a run; persistent failure still raises."""
        for attempt in range(RETRIES + 1):
            out = fn()
            res = out[1] if isinstance(out, tuple) and len(out) == 3 else out[0]
            if res == scs.COMM_SUCCESS:
                if attempt:
                    self.retries += attempt
                return out
        raise RuntimeError(f"{what}: {self.pkt.getTxRxResult(res)} ({RETRIES + 1} tries)")

    def r1(self, sid, addr):
        return self._retry(f"read addr {addr} id {sid}",
                           lambda: self.pkt.read1ByteTxRx(self.port, sid, addr))[0]

    def r2(self, sid, addr):
        return self._retry(f"read addr {addr} id {sid}",
                           lambda: self.pkt.read2ByteTxRx(self.port, sid, addr))[0]

    def w1(self, sid, addr, val):
        self._retry(f"write addr {addr} id {sid}",
                    lambda: self.pkt.write1ByteTxRx(self.port, sid, addr, val))

    def w2(self, sid, addr, val):
        self._retry(f"write addr {addr} id {sid}",
                    lambda: self.pkt.write2ByteTxRx(self.port, sid, addr, val))

    def position(self, sid):
        return self.r2(sid, PRESENT_POSITION)

    def load(self, sid):
        v = self.r2(sid, PRESENT_LOAD)  # sign-magnitude, sign in bit 10, 0.1 %
        return -(v & 0x3FF) if v & 0x400 else v

    def limits(self, sid):
        """EEPROM Min/Max_Position_Limit (read only). Odd/unset -> full range."""
        lo, hi = self.r2(sid, MIN_POSITION_LIMIT), self.r2(sid, MAX_POSITION_LIMIT)
        return (lo, hi) if lo < hi else (0, TICKS_PER_REV - 1)

    def torque_on_holding(self, sid, accel, speed):
        """Enable torque without a jump: goal = present first, gentle limits."""
        pos = self.position(sid)
        self.w1(sid, ACCELERATION, accel)
        self.w2(sid, GOAL_SPEED, speed)
        self.w2(sid, GOAL_POSITION, pos)
        self.w1(sid, TORQUE_ENABLE, 1)
        return pos

    def send_goals(self, goals):
        """Sync-write {sid: ticks} so the joints move together."""
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
                goals[sid] = self.position(sid)
            except RuntimeError:
                pass
        if goals:
            self.send_goals(goals)
        return goals

    def close(self):
        self.port.closePort()
