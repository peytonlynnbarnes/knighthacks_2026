#!/usr/bin/env python3
"""Shift one servo's position reading by changing its Homing_Offset (EEPROM 31, sign-magnitude,
sign in bit 11; Present_Position = raw - Homing_Offset, mod 4096).

Why: wrist_roll travels ~333 deg between the two sides of a ~27 deg hard stop, but that
travel crossed the 0/4095 seam, which position mode can't drive across, so only ~205 deg was
usable. Find the stop first (torque off, turn it by hand to each side, read the position),
then shift the reading so the seam sits in the middle of the stop. Done 2026-10-11:
offset 1731 (LeRobot) -> 1542; so101_bus JOINT_CAL / SOFT_LIMITS match it.

    ./set_homing_offset.py --joint wrist_roll --shift -1146           # dry run: prints the plan
    ./set_homing_offset.py --joint wrist_roll --set 1542 --write      # asks, then writes
    ./set_homing_offset.py --joint wrist_roll --set 1731 --write      # LeRobot's value back

Safety: EEPROM, persistent. The joint's torque is turned OFF first (the reading jumps when
the offset changes; with torque on the servo would drive to its old goal in the new
coordinates) and stays off: the arm-side constants (so101_bus JOINT_CAL / SOFT_LIMITS) must
match the new offset before feetech_mirror enables it again. Only that joint goes limp.
"""
import argparse
import sys
import time

import so101_bus as sb

HOMING_OFFSET, LOCK = 31, 55


def encode(v):
    """Signed offset -> sign-magnitude register value (bit 11)."""
    if not -2047 <= v <= 2047:
        sys.exit(f"offset {v} out of range (-2047..2047)")
    return (abs(v) | 0x800) if v < 0 else v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help=sb.PORT_HELP)
    ap.add_argument("--joint", choices=list(sb.IDS), required=True)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--shift", type=int, help="ticks to add to the position reading")
    g.add_argument("--set", type=int, help="write this offset (signed ticks)")
    ap.add_argument("--write", action="store_true", help="write the EEPROM (default: dry run)")
    ap.add_argument("--yes", action="store_true")
    args = ap.parse_args()

    sid = sb.IDS[args.joint]
    bus = sb.Bus(args.port)
    try:
        old = sb.sign_mag(bus.r2(sid, HOMING_OFFSET), 11)
        new = old - args.shift if args.shift is not None else args.set
        p0 = bus.position(sid)
        expect = (p0 + (old - new)) % 4096
        print(f"{args.joint}: Homing_Offset {old} -> {new}; position reads {p0} now, ~{expect} after; "
              f"torque {'on' if bus.r1(sid, sb.TORQUE_ENABLE) else 'off'}")
        if not args.write:
            print("DRY RUN: nothing written. Add --write to apply.")
            return
        if not args.yes and input(f"{args.joint} torque goes OFF and stays off. Someone at the arm? [y/N] ") \
                .strip().lower() != "y":
            sys.exit("aborted, nothing written")
        bus.w1(sid, sb.TORQUE_ENABLE, 0)
        time.sleep(0.1)
        p_off = bus.position(sid)
        bus.w1(sid, LOCK, 0)
        try:
            bus.w2(sid, HOMING_OFFSET, encode(new))
        finally:
            bus.w1(sid, LOCK, 1)
        for _ in range(3):
            time.sleep(0.05)
            got = sb.sign_mag(bus.r2(sid, HOMING_OFFSET), 11)
            if got == new:
                break
        p1 = bus.position(sid)
        want = (p_off + (old - new)) % 4096
        print(f"written: offset reads {got}; position {p_off} -> {p1} (expected ~{want}); torque OFF")
        if got != new:
            sys.exit("offset read back wrong")
        if min(abs(p1 - want), 4096 - abs(p1 - want)) > 20:
            print("WARNING: the reading didn't move as expected (the register works the other way?). "
                  f"Put it back with --set {old} --write and tell Claude.")
    finally:
        bus.close()


if __name__ == "__main__":
    main()
