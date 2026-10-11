#!/usr/bin/env python3
"""Read, and with --write change, the STS3215 position-loop gains (P/D/I_Coefficient,
EEPROM 21/22/23) of selected joints.

Why: on 2026-10-11 shoulder_pan / shoulder_lift / elbow_flex read P = 6 (Feetech default
32, LeRobot's SO-101 setting 16). A position servo pushes ~P x error, so at 6 shoulder_lift
sagged 7-12 deg below its goal under the arm's weight (load only 15-20 %), homing stopped
140 ticks short and feetech_mirror's stall check fired. Higher P = less sag, faster
response; too high = buzzing / oscillation.

    ./set_gains.py                                   # read every joint's gains (no writes)
    ./set_gains.py --joints shoulder_pan shoulder_lift elbow_flex --p 16            # dry run
    ./set_gains.py --joints shoulder_pan shoulder_lift elbow_flex --p 16 --write    # asks first
    ./set_gains.py --joints shoulder_pan shoulder_lift elbow_flex --p 6 --write     # revert
    ./set_gains.py --joints shoulder_lift --d 30 --i 2 --write

Tuned 2026-10-11 (tune_gains.py, step responses, ~/kh_rec/tune/): shoulder_pan P32 D30 I0,
shoulder_lift P40 D30 I2, elbow_flex P40 D15 I2 (were P6 D30/15/15 I0).
shoulder_pan then P24: spline runs (~/kh_rec/shake/) shook less at P24 (servo jitter = its
goal's) for +10 ms lag; P16 smoother but +35 ms lag and more tracking error.
wrist_flex / wrist_roll P10 / P15 -> 24 / 24 (step tests clean to 40; wrist spline runs: tracking
error -34 % / -20 %, full travel; P32 doubled the roll jitter).
gripper P10 -> 24 (open/close cycles: 90 % reached 270 -> 180 ms after the goal, stopped
10-11 -> 3-5 ticks short; P32 no better). Grip force ~ P x GRIP_SQUEEZE: 2.4x stronger.

Safety: EEPROM writes are persistent (they survive power-off); run only with the user's
OK, someone at the arm, and no other script on the port. Before changing a joint whose
torque is on, its goal is set to its present position, so the stronger loop doesn't jump
to an old goal. Each write is unlocked (Lock 55 = 0), written, re-locked (Lock 55 = 1) and
read back.
"""
import argparse
import sys
import time

import so101_bus as sb

P_COEF, D_COEF, I_COEF = 21, 22, 23
LOCK = 55


def show(bus, names):
    print(f"{'joint':<14} {'P':>4} {'D':>4} {'I':>4}  torque  goal-present")
    for j in names:
        sid = sb.IDS[j]
        gains = [bus.r1(sid, a) for a in (P_COEF, D_COEF, I_COEF)]
        torque = bus.r1(sid, sb.TORQUE_ENABLE)
        err = bus.r2(sid, sb.GOAL_POSITION) - bus.position(sid)
        print(f"{j:<14} {gains[0]:>4} {gains[1]:>4} {gains[2]:>4}  {'on' if torque else 'off':>6}  {err:+5d}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help=sb.PORT_HELP)
    ap.add_argument("--joints", nargs="+", choices=list(sb.IDS), default=list(sb.IDS))
    ap.add_argument("--p", type=int, help="new P_Coefficient (1..64)")
    ap.add_argument("--d", type=int, help="new D_Coefficient (0..64)")
    ap.add_argument("--i", type=int, help="new I_Coefficient (0..16; it removes gravity sag, too high hunts)")
    ap.add_argument("--write", action="store_true", help="write the EEPROM (default: dry run)")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()
    if args.p is not None and not 1 <= args.p <= 64:
        sys.exit("--p must be 1..64")
    if args.d is not None and not 0 <= args.d <= 64:
        sys.exit("--d must be 0..64")
    if args.i is not None and not 0 <= args.i <= 16:
        sys.exit("--i must be 0..16")
    new = {a: v for a, v in ((P_COEF, args.p), (D_COEF, args.d), (I_COEF, args.i)) if v is not None}

    bus = sb.Bus(args.port)
    try:
        print("now:")
        show(bus, args.joints)
        if not new:
            return
        what = ", ".join(f"{n} -> {v}" for n, v in (("P", args.p), ("D", args.d), ("I", args.i)) if v is not None)
        print(f"\nplan: {what} on {', '.join(args.joints)} (EEPROM, persistent)")
        if not args.write:
            print("DRY RUN: nothing written. Add --write to apply.")
            return
        if not args.yes and input("Write the EEPROM? Someone at the arm? [y/N] ").strip().lower() != "y":
            sys.exit("aborted, nothing written")
        for j in args.joints:
            sid = sb.IDS[j]
            if bus.r1(sid, sb.TORQUE_ENABLE):
                bus.w2(sid, sb.GOAL_POSITION, bus.position(sid))  # hold here, no jump
            bus.w1(sid, LOCK, 0)
            try:
                for a, v in new.items():
                    bus.w1(sid, a, v)
            finally:
                bus.w1(sid, LOCK, 1)
            for _ in range(3):  # a read right after an EEPROM write came back garbled once (251)
                time.sleep(0.05)
                got = {a: bus.r1(sid, a) for a in new}
                if got == new:
                    break
            if got != new:
                sys.exit(f"{j}: read back {got}, expected {new}")
        print("\nwritten and verified:")
        show(bus, args.joints)
    finally:
        bus.close()


if __name__ == "__main__":
    main()
