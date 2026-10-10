#!/usr/bin/env python3
"""Record SO-ARM101 joint positions to CSV while you move the arm by hand.

READ ONLY: never writes any register, never touches torque. Torque should
already be off on the joints you move (check the startup line). If it's ON,
the joint will fight you. Use torque_off.py first (the arm drops!).

Usage:
    ./record_range.py                          # writes calib/range_<time>.csv
    ./record_range.py --out my.csv --rate 20
    ./record_range.py --summary calib/range_X.csv [--since T0 --until T1]
                                               # min/max per joint (unix time window)
Ctrl-C stops recording and prints min/max per joint.
"""
import argparse
import csv
import os
import sys
import time

import so101_bus as sb


def summarize(path, since=None, until=None):
    lo, hi, n = {}, {}, 0
    with open(path) as f:
        for row in csv.DictReader(f):
            t = float(row["t"])
            if (since and t < since) or (until and t > until):
                continue
            n += 1
            for name in sb.NAMES.values():
                v = row.get(name)
                if v in (None, ""):
                    continue
                v = int(v)
                lo[name] = min(lo.get(name, v), v)
                hi[name] = max(hi.get(name, v), v)
    print(f"{n} samples")
    print(f"{'joint':<14} {'min':>5} {'max':>5} {'span':>5} {'span deg':>8}")
    for name in sb.NAMES.values():
        if name in lo:
            span = hi[name] - lo[name]
            print(f"{name:<14} {lo[name]:>5} {hi[name]:>5} {span:>5} {span * 360 / 4096:>8.1f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help=sb.PORT_HELP)
    ap.add_argument("--out", help="CSV path (default calib/range_<time>.csv next to this script)")
    ap.add_argument("--rate", type=float, default=20.0)
    ap.add_argument("--summary", metavar="CSV", help="print min/max of an existing recording and exit")
    ap.add_argument("--since", type=float)
    ap.add_argument("--until", type=float)
    args = ap.parse_args()

    if args.summary:
        summarize(args.summary, args.since, args.until)
        return

    out = args.out or os.path.join(os.path.dirname(os.path.abspath(__file__)), "calib",
                                   time.strftime("range_%Y%m%d_%H%M%S.csv"))
    os.makedirs(os.path.dirname(out), exist_ok=True)
    bus = sb.Bus(args.port)
    sids = [sid for sid in sb.NAMES if bus.ping(sid)]
    if not sids:
        sys.exit("no servos responded")
    for sid in sids:
        lo, hi = bus.limits(sid)
        print(f"{sid} {sb.NAMES[sid]:<14} pos {bus.position(sid):>5}  "
              f"torque {'ON' if bus.r1(sid, sb.TORQUE_ENABLE) else 'off'}  EEPROM limits [{lo}, {hi}]",
              flush=True)
    print(f"recording to {out} at {args.rate:.0f} Hz (Ctrl-C to stop)", flush=True)

    dt = 1.0 / args.rate
    try:
        with open(out, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["t"] + [sb.NAMES[s] for s in sids])
            while True:
                t0 = time.time()
                row = [f"{t0:.3f}"]
                for sid in sids:
                    try:
                        row.append(bus.position(sid))
                    except RuntimeError:
                        row.append("")  # missed read; skipped in summary
                w.writerow(row)
                f.flush()
                time.sleep(max(0.0, dt - (time.time() - t0)))
    except KeyboardInterrupt:
        pass
    finally:
        bus.close()
    print()
    summarize(out)


if __name__ == "__main__":
    main()
