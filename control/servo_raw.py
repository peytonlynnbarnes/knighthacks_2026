#!/home/peyton/kh/.venv/bin/python
"""Low-level Feetech STS3215 control for the SO-ARM101 via scservo_sdk.

Scans IDs 1-6, prints state, then nudges one joint by a small step and
returns it to where it started.

Usage:
    python servo_raw.py                      # scan only
    python servo_raw.py --id 1 --step 100    # scan + nudge shoulder_pan ~9 deg and back
"""
import argparse
import sys
import time

import scservo_sdk as scs

from so101_bus import OPEN_HINT, PORT_HELP, default_port

BAUD = 1_000_000
NAMES = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex",
         4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}

# STS3215 control table (address)
TORQUE_ENABLE = 40
ACCELERATION = 41
GOAL_POSITION = 42
GOAL_SPEED = 46
PRESENT_POSITION = 56
PRESENT_LOAD = 60
PRESENT_VOLTAGE = 62
PRESENT_TEMPERATURE = 63


def check(pkt, res, err, what):
    if res != scs.COMM_SUCCESS:
        raise RuntimeError(f"{what}: {pkt.getTxRxResult(res)}")
    if err:
        print(f"  warning ({what}): {pkt.getRxPacketError(err)}")


def read_state(pkt, port, sid):
    pos, res, err = pkt.read2ByteTxRx(port, sid, PRESENT_POSITION)
    check(pkt, res, err, f"read position id {sid}")
    load, _, _ = pkt.read2ByteTxRx(port, sid, PRESENT_LOAD)
    volt, _, _ = pkt.read1ByteTxRx(port, sid, PRESENT_VOLTAGE)
    temp, _, _ = pkt.read1ByteTxRx(port, sid, PRESENT_TEMPERATURE)
    return pos, load, volt / 10, temp


def move_and_wait(pkt, port, sid, goal, tol=15, timeout=3.0):
    res, err = pkt.write2ByteTxRx(port, sid, GOAL_POSITION, goal)
    check(pkt, res, err, "write goal")
    t0 = time.time()
    while time.time() - t0 < timeout:
        pos, _, _ = pkt.read2ByteTxRx(port, sid, PRESENT_POSITION)
        if abs(pos - goal) <= tol:
            return pos
        time.sleep(0.05)
    return pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help=PORT_HELP)
    ap.add_argument("--id", type=int, help="servo ID to nudge (omit to scan only)")
    ap.add_argument("--step", type=int, default=100, help="ticks to nudge (4096 = 360 deg)")
    args = ap.parse_args()
    args.port = args.port or default_port()

    if args.id is not None and abs(args.step) > 300:
        sys.exit("refusing step > 300 ticks (~26 deg); edit the script if you really mean it")

    port = scs.PortHandler(args.port)
    pkt = scs.PacketHandler(0)  # protocol 0 = STS series
    try:
        opened = port.openPort()
    except Exception as e:
        opened = False
        print(e)
    if not opened:
        sys.exit(f"could not open {args.port} ({OPEN_HINT})")
    if not port.setBaudRate(BAUD):
        sys.exit(f"could not set baud {BAUD}")

    try:
        print(f"{'id':>2} {'joint':<14} {'pos':>5} {'load':>5} {'volt':>5} {'temp':>4}")
        found = []
        for sid, name in NAMES.items():
            _, res, _ = pkt.ping(port, sid)
            if res != scs.COMM_SUCCESS:
                print(f"{sid:>2} {name:<14}  -- no response")
                continue
            pos, load, volt, temp = read_state(pkt, port, sid)
            print(f"{sid:>2} {name:<14} {pos:>5} {load:>5} {volt:>5.1f} {temp:>4}")
            found.append(sid)

        if args.id is None:
            return
        if args.id not in found:
            sys.exit(f"servo {args.id} not found on bus")

        sid = args.id
        start, *_ = read_state(pkt, port, sid)
        target = max(0, min(4095, start + args.step))
        print(f"\nnudging {NAMES[sid]} (id {sid}): {start} -> {target} -> {start}")

        pkt.write1ByteTxRx(port, sid, ACCELERATION, 20)
        pkt.write2ByteTxRx(port, sid, GOAL_SPEED, 300)
        # hold current pose before enabling torque so it doesn't jump
        pkt.write2ByteTxRx(port, sid, GOAL_POSITION, start)
        res, err = pkt.write1ByteTxRx(port, sid, TORQUE_ENABLE, 1)
        check(pkt, res, err, "torque on")

        print(f"  reached {move_and_wait(pkt, port, sid, target)}")
        time.sleep(0.5)
        print(f"  back at {move_and_wait(pkt, port, sid, start)}")
        # torque left ON so the joint holds; write 0 to TORQUE_ENABLE to go limp
    finally:
        port.closePort()


if __name__ == "__main__":
    main()
