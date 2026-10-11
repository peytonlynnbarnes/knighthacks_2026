"""Shared SO-ARM101 servo helpers (Feetech STS3215 via scservo_sdk).

Constants, a small bus wrapper and the tick <-> radian conversion, factored
out of base_pose.py for the real-time loop (feetech_mirror.py). The older
standalone scripts keep their own copies and still work unchanged.

Not a script; import it:
    from so101_bus import Bus, NAMES, rad_to_ticks, ticks_to_rad, default_port
"""
import glob
import math
import sys

import scservo_sdk as scs

BAUD = 1_000_000

# macOS names USB serial adapters /dev/cu.* (use cu., not tty.: tty. waits for a carrier
# signal). CDC-ACM boards are cu.usbmodem*, CH34x / FTDI bridges cu.usbserial* / cu.wchusbserial*.
PORT_GLOBS = ("/dev/cu.usbmodem*", "/dev/cu.usbserial*", "/dev/cu.wchusbserial*")
PORT_HELP = "servo board serial port (default: the one USB serial device in /dev/cu.*)"


def default_port():
    """The servo board's serial port: the only USB serial device present. Exits with
    the candidates if there are none or several (then pass --port)."""
    found = sorted({p for g in PORT_GLOBS for p in glob.glob(g)})
    if len(found) == 1:
        return found[0]
    if not found:
        sys.exit("no USB serial device (/dev/cu.usbmodem*, /dev/cu.usbserial*): is the servo board "
                 "plugged in and powered? (ls /dev/cu.*)")
    sys.exit(f"several USB serial devices, pick one with --port: {', '.join(found)}")
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
PRESENT_SPEED = 58
PRESENT_LOAD = 60
PRESENT_VOLTAGE = 62
PRESENT_TEMPERATURE = 63
STATUS = 65
# read_states(): one sync read of 56..65 (position, speed, load, voltage, temperature, -, status)
STATE_LEN = STATUS - PRESENT_POSITION + 1
# Status (65) bits, Feetech STS manual: a set bit = that protection has tripped
STATUS_BITS = ("voltage", "sensor", "temperature", "current", "angle", "overload")

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
    # wrist_roll's Homing_Offset was moved 1731 -> 1542 (2026-10-11, set_homing_offset.py):
    # its reading is +189 ticks from before, so the URDF zero (was 2048) is now 2237, and the
    # 0/4095 seam sits inside the wrist's hard stop (see SOFT_LIMITS)
    "wrist_roll": (+1, CENTER + 189),
}


# Home = base pose: every arm joint at its URDF zero (JOINT_CAL; 2048 except wrist_roll,
# whose reading was shifted). Visually checked on hardware (2026-10-09) to match the URDF
# zero L-pose. (base_pose.py / jog.py keep their own 2048: don't use them on wrist_roll.)
HOME = {name: zero for name, (_, zero) in JOINT_CAL.items()}

# Soft limits (ticks): the usable range = the hand-measured range of motion
# (calib/ranges_session1.json) intersected with the EEPROM limits read from the
# servos (2026-10-09), so the IK (which can't read the bus) plans exactly what
# feetech_mirror allows. Measured, where EEPROM is narrower: shoulder_lift 815,
# elbow_flex 3176, wrist_flex 694, gripper 3523.
# wrist_roll: its hard stop (~27 deg) was measured by hand, torque off (2026-10-11): the
# travel ends read 993 and 1298 with Homing_Offset 396, i.e. ~333 deg of travel that crossed
# the 0/4095 seam (position mode can't), so only ~205 deg was usable (calib/ranges_session1
# .json's arc was wrong about where the stop is). Offset 1542 puts the seam in the middle of
# the stop. Probed live with block detection (slow, stops at ~30 % load): the travel reads
# 110..3987 (341 deg; the stop's middle = the seam); kept 25 ticks inside: -185..+152 deg.
SOFT_LIMITS = {
    "shoulder_pan": (1272, 3000),
    "shoulder_lift": (1067, 3199),
    "elbow_flex": (944, 2879),
    "wrist_flex": (833, 3071),
    "wrist_roll": (135, 3962),
    "gripper": (2007, 3298),
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


def sign_mag(v, bit):
    """Feetech sign-magnitude register value: the sign is bit `bit`."""
    return -(v & ((1 << bit) - 1)) if v & (1 << bit) else v


def rad_to_ticks(name, rad):
    sign, zero = JOINT_CAL[name]
    return zero + sign * rad * TICKS_PER_REV / (2 * math.pi)


def ticks_to_rad(name, ticks):
    sign, zero = JOINT_CAL[name]
    return sign * (ticks - zero) * 2 * math.pi / TICKS_PER_REV


class Bus:
    def __init__(self, port_name=None):
        port_name = port_name or default_port()
        self.port = scs.PortHandler(port_name)
        self.pkt = scs.PacketHandler(0)  # protocol 0 = STS series
        try:
            opened = self.port.openPort()
        except Exception as e:  # pyserial raises instead of returning False
            print(e)
            opened = False
        if not opened:
            sys.exit(f"could not open {port_name} (in use by another script? unplug and replug the board)")
        if not self.port.setBaudRate(BAUD):
            sys.exit(f"could not set baud {BAUD}")
        self.sync = scs.GroupSyncWrite(self.port, self.pkt, GOAL_POSITION, 2)
        self.sync_read = None  # GroupSyncRead for read_states, built for the ids asked
        self.retries = 0  # transactions that needed a retry (for reporting)

    def ping(self, sid):
        _, res, _ = self.pkt.ping(self.port, sid)
        return res == scs.COMM_SUCCESS

    def _retry(self, what, fn):
        """Run one bus transaction, retrying up to RETRIES times. A single
        missed status packet (seen while several servos move, presumably a
        supply/USB hiccup) shouldn't abort a run; persistent failure still raises."""
        for attempt in range(RETRIES + 1):
            try:
                out = fn()
            except IndexError:  # the SDK on a truncated reply (seen right after EEPROM writes)
                out = (None, scs.COMM_RX_CORRUPT, 0)
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
        return sign_mag(self.r2(sid, PRESENT_LOAD), 10)  # sign in bit 10, 0.1 %

    def read_states(self, sids):
        """Position, speed (ticks/s), load (0.1 %) and tripped protections of several
        servos in one sync read: one bus round trip instead of one per register and
        servo. Returns {sid: {"pos", "speed", "load", "flags"}}; raises RuntimeError
        like the single reads."""
        sids = list(sids)
        if self.sync_read is None or list(self.sync_read.data_dict) != sids:
            self.sync_read = scs.GroupSyncRead(self.port, self.pkt, PRESENT_POSITION, STATE_LEN)
            for sid in sids:
                self.sync_read.addParam(sid)
        g = self.sync_read
        self._retry(f"sync read ids {sids}", lambda: (g.txRxPacket(),))
        out = {}
        for sid in sids:
            if not g.isAvailable(sid, PRESENT_POSITION, STATE_LEN):
                raise RuntimeError(f"sync read: no data from id {sid}")
            status = g.getData(sid, STATUS, 1)
            out[sid] = {"pos": g.getData(sid, PRESENT_POSITION, 2),
                        "speed": sign_mag(g.getData(sid, PRESENT_SPEED, 2), 15),
                        "load": sign_mag(g.getData(sid, PRESENT_LOAD, 2), 10),
                        "flags": [n for i, n in enumerate(STATUS_BITS) if status >> i & 1]}
        return out

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
