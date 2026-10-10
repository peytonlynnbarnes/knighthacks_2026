#!/home/peyton/kh/.venv/bin/python
"""Turn torque OFF on the SO-ARM101 servos. The arm goes limp and WILL DROP.

Usage:
    ./torque_off.py              # all joints, asks first
    ./torque_off.py --joints 6   # only the gripper
    ./torque_off.py --yes        # no prompt
"""
import argparse
import sys

import scservo_sdk as scs

BAUD = 1_000_000
NAMES = {1: "shoulder_pan", 2: "shoulder_lift", 3: "elbow_flex",
         4: "wrist_flex", 5: "wrist_roll", 6: "gripper"}
TORQUE_ENABLE = 40


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyACM0")
    ap.add_argument("--joints", type=int, nargs="+", default=list(NAMES))
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()

    names = ", ".join(NAMES.get(j, str(j)) for j in args.joints)
    if not args.yes:
        print(f"Torque OFF for: {names}\nThe arm will go limp and drop. Support it first.")
        if input("Proceed? [y/N] ").strip().lower() != "y":
            sys.exit("aborted, torque unchanged")

    port = scs.PortHandler(args.port)
    pkt = scs.PacketHandler(0)
    if not port.openPort() or not port.setBaudRate(BAUD):
        sys.exit(f"could not open {args.port}")
    try:
        for sid in args.joints:
            res, _ = pkt.write1ByteTxRx(port, sid, TORQUE_ENABLE, 0)
            ok = res == scs.COMM_SUCCESS
            print(f"{sid} {NAMES.get(sid, '?'):<14} {'off' if ok else 'FAILED: ' + pkt.getTxRxResult(res)}")
    finally:
        port.closePort()


if __name__ == "__main__":
    main()
