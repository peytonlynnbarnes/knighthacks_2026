#!/home/peyton/kh/.venv/bin/python
"""Step-response sweep of the STS3215 position-loop gain (P_Coefficient) per joint, to
pick gains from measurements instead of guesses.

For each joint, from the pose the arm is holding: for each candidate P (ascending), set
the goal to the present position, write P, then step the goal +STEP / back / -STEP / back
(REPS times), sampling every arm joint's position and load at ~100 Hz with one sync read.
Per P it scores rise time (10-90 %), overshoot, steady-state error (sag: gravity against
a soft P), ripple after settling and peak load. The sweep stops going higher at the first
P that overshoots or rings. The recommendation is the highest clean P one step back.

    ./tune_gains.py --yes                                   # pan, elbow, lift; restores gains
    ./tune_gains.py --joints shoulder_lift --p 6 10 16 24 --yes
    ./tune_gains.py --apply --yes                           # also keep the recommended P

Safety (the arm moves: someone at the arm, the user's OK, nothing else on the port):
    - steps are +-STEP ticks (~5 deg), one joint at a time, goals kept inside SOFT_LIMITS
    - every arm joint must already hold with torque on, or is turned on holding (goal =
      present first); every other joint's goal is its present position
    - abort (that joint's original P back, every joint holds where it is, torque stays on):
      any load > LOAD_ABORT, any other joint pushed > DRIFT ticks, the tested joint
      > STEP + RUNAWAY ticks from its base, ripple > RIPPLE_ABORT after settling, Ctrl-C
    - original gains are restored at the end unless --apply
    - each gain write: Lock 55 = 0, write, Lock 55 = 1, read back
Results: ~/kh_rec/tune/<time>.npz (raw samples) + the printed table.
"""
import argparse
import os
import sys
import time

import numpy as np

import so101_bus as sb

LOCK = 55
REGS = {"P": 21, "D": 22, "I": 23}  # P/D/I_Coefficient (EEPROM)
DEFAULT_VALUES = {"P": [6, 8, 12, 16, 20, 24, 32], "D": [0, 8, 15, 30, 45, 60], "I": [0, 1, 2, 4, 8]}
ARM = ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")
STEP = 60            # ticks (~5.3 deg)
HOLD_S = 1.2         # per segment
REPS = 2
RATE = 100.0         # Hz sampling
LOAD_ABORT = 600     # 0.1 % (60 %)
DRIFT = 25           # ticks another joint may move (~2 deg)
RUNAWAY = 40         # ticks past the step target
RIPPLE_ABORT = 20    # ticks peak-to-peak over a settled window
OVERSHOOT_MAX = 0.15  # fraction of the step: above = not clean
RIPPLE_MAX = 6       # ticks p-p: above = not clean


class Abort(Exception):
    pass


def write_gain(bus, sid, addr, v):
    bus.w1(sid, LOCK, 0)
    try:
        bus.w1(sid, addr, v)
    finally:
        bus.w1(sid, LOCK, 1)
    for _ in range(3):  # a read right after an EEPROM write came back garbled once
        time.sleep(0.05)
        got = bus.r1(sid, addr)
        if got == v:
            break
    if got != v:
        raise RuntimeError(f"id {sid} addr {addr}: read back {got}, wrote {v}")


def run_segment(bus, sids, joint, goal, held, base, dur, log):
    """Hold `goal` on `joint` for dur s, sampling; abort checks every sample (the drift
    check only when `held` is given)."""
    bus.send_goals({sids[joint]: goal})
    t_end = time.monotonic() + dur
    nxt = time.monotonic()
    while time.monotonic() < t_end:
        st = bus.read_states(sids.values())
        t = time.monotonic()
        pos = {j: st[s]["pos"] for j, s in sids.items()}
        load = {j: st[s]["load"] for j, s in sids.items()}
        log.append((t, goal, *[pos[j] for j in ARM], *[load[j] for j in ARM]))
        for j in ARM:
            if abs(load[j]) > LOAD_ABORT:
                raise Abort(f"{j} load {load[j] / 10:.0f}%")
            if held is not None and j != joint and abs(pos[j] - held[j]) > DRIFT:
                raise Abort(f"{j} moved {pos[j] - held[j]:+d} ticks while {joint} was tested")
        if abs(pos[joint] - base) > max(STEP, abs(goal - base)) + RUNAWAY:
            raise Abort(f"{joint} ran {pos[joint] - base:+d} ticks from its base")
        nxt += 1 / RATE
        time.sleep(max(0.0, nxt - time.monotonic()))


def ramp(bus, sids, joint, target, speed=100.0):
    """Move one joint to `target` slowly (ticks/s), the others holding; same aborts."""
    st = bus.read_states(sids.values())
    held = {j: st[s]["pos"] for j, s in sids.items()}
    start = held[joint]
    n = max(1, int(abs(target - start) / speed * RATE))
    log = []
    for i in range(1, n + 1):
        run_segment(bus, sids, joint, start + (target - start) * i / n, held, start, 1 / RATE, log)
    run_segment(bus, sids, joint, target, None, start, 1.5, log)  # settle (no drift check)
    return bus.position(sids[joint])


def score(seg, joint_i, start, target):
    """Metrics of one segment (rows of the log) stepping from start to target."""
    t = seg[:, 0] - seg[0, 0]
    x = seg[:, 2 + joint_i]
    d = target - start
    out = {"steady": float(np.mean(target - x[t > HOLD_S - 0.3]))}
    m = t > HOLD_S - 0.5  # settled window; ripple = p-p around a straight line (a slow
    w = x[m] - np.polyval(np.polyfit(t[m], x[m], 1), t[m])  # creep isn't oscillation)
    out["ripple"] = float(np.ptp(w))
    if d != 0:
        f = (x - start) / d
        t10 = t[np.argmax(f >= 0.1)] if (f >= 0.1).any() else np.nan
        t90 = t[np.argmax(f >= 0.9)] if (f >= 0.9).any() else np.nan
        out["rise_ms"] = float((t90 - t10) * 1000)
        out["overshoot"] = float(max(0.0, f.max() - 1.0))
        out["reach"] = float(f[t > HOLD_S - 0.3].mean())  # fraction of the step reached
    out["load"] = float(np.abs(seg[:, 2 + len(ARM) + joint_i]).max() / 10)
    return out


def main():
    global HOLD_S
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default=None, help=sb.PORT_HELP)
    ap.add_argument("--joints", nargs="+", choices=ARM, default=["shoulder_pan", "elbow_flex", "shoulder_lift"])
    ap.add_argument("--reg", choices=list(REGS), default="P", help="which gain to sweep (the others stay)")
    ap.add_argument("--values", "--p", nargs="+", type=int, default=None,
                    help=f"values to try, ascending (default {DEFAULT_VALUES})")
    ap.add_argument("--hold", type=float, default=HOLD_S, help="s per step segment (longer for I)")
    ap.add_argument("--apply", action="store_true", help="keep the recommended value (else restore)")
    ap.add_argument("--pre-move", nargs="+", default=[], metavar="JOINT=TICKS",
                    help="first move these joints by TICKS (relative, slowly), e.g. elbow_flex=-150")
    ap.add_argument("--goto", nargs="+", default=[], metavar="JOINT=TICKS",
                    help="first move these joints TO TICKS (absolute, slowly, in the order given)")
    ap.add_argument("--yes", action="store_true", help="skip the confirmation prompt")
    args = ap.parse_args()
    HOLD_S = args.hold
    reg, ADDR = args.reg, REGS[args.reg]
    values = args.values or DEFAULT_VALUES[reg]

    bus = sb.Bus(args.port)
    sids = {j: sb.IDS[j] for j in ARM}
    orig = {j: {r: bus.r1(s, a) for r, a in REGS.items()} for j, s in sids.items()}
    st = bus.read_states(sids.values())
    held = {j: st[s]["pos"] for j, s in sids.items()}
    for j in ARM:
        lo, hi = sb.effective_limits(j, bus.limits(sids[j]))
        if not lo <= held[j] <= hi:
            bus.close()
            sys.exit(f"{j} at {held[j]} is outside [{lo}, {hi}]; move it into range first")
    print("pose (ticks):", held)
    print("gains now:", orig)
    print(f"plan: {', '.join(args.joints)}; {reg} {values}; steps +-{STEP} ticks "
          f"({STEP * 360 / 4096:.1f} deg), {REPS}x, {HOLD_S} s each")
    if not args.yes and input("The arm will move. Someone at the arm? [y/N] ").strip().lower() != "y":
        sys.exit("aborted")

    os.makedirs(os.path.expanduser("~/kh_rec/tune"), exist_ok=True)
    out_path = os.path.expanduser(f"~/kh_rec/tune/{time.strftime('%Y%m%d_%H%M%S')}.npz")
    logs, results, applied = {}, {}, {}
    try:
        for j in ARM:  # torque on holding where off; every goal = present
            if not bus.r1(sids[j], sb.TORQUE_ENABLE):
                bus.torque_on_holding(sids[j], 100, 3000)
        bus.send_goals({sids[j]: held[j] for j in ARM})
        moves = [(spec, False) for spec in args.pre_move] + [(spec, True) for spec in args.goto]
        for spec, absolute in moves:
            j, d = spec.split("=")
            lo, hi = sb.effective_limits(j, bus.limits(sids[j]))
            now = bus.position(sids[j])
            tgt = max(lo + 20, min(hi - 20, int(d) if absolute else now + int(d)))
            held[j] = now
            print(f"pre-move: {j} {held[j]} -> {tgt}", flush=True)
            try:
                print(f"  now at {ramp(bus, sids, j, tgt)}", flush=True)
            except Abort as e:
                print(f"pre-move ABORT ({e}); holding, nothing tested")
                return
        st = bus.read_states(sids.values())
        ref = {j: st[s]["pos"] for j, s in sids.items()}  # fixed bases for the whole run
        print("reference pose:", ref, flush=True)
        for joint in args.joints:
            sid, ji = sids[joint], ARM.index(joint)
            lo, hi = sb.effective_limits(joint, bus.limits(sid))
            results[joint] = {}
            for p in values:
                # back to the run's reference (not where the joint sagged to): stepping
                # around the present position ratcheted shoulder_lift ~11 deg down over
                # several runs, into the table (2026-10-11)
                base = ref[joint]
                bus.send_goals({sids[j]: ref[j] for j in ARM})
                held = dict(ref)
                write_gain(bus, sid, ADDR, p)
                log = []
                segs = []
                try:
                    run_segment(bus, sids, joint, base, None, base, 0.6, log)  # settle at base
                    st = bus.read_states(sids.values())  # the other joints' reference: after settling
                    held = {j: st[s]["pos"] for j, s in sids.items()}
                    plan = [s for s in (base + STEP, base, base - STEP, base) * REPS]
                    prev = base
                    for g in plan:
                        if not lo + 20 <= g <= hi - 20:
                            continue
                        i0 = len(log)
                        run_segment(bus, sids, joint, g, held, base, HOLD_S, log)
                        segs.append((i0, len(log), prev, g))
                        prev = g
                except Abort as e:
                    write_gain(bus, sid, ADDR, orig[joint][reg])
                    bus.hold(list(sids.values()))
                    print(f"  {reg} {p:2d}: ABORT ({e}); {joint} back to {reg} {orig[joint][reg]}, holding")
                    results[joint][p] = {"abort": str(e)}
                    logs[f"{joint}_P{p}"] = np.array(log)
                    break
                L = np.array(log)
                logs[f"{joint}_P{p}"] = L
                sc = [score(L[a:b], ji, s, g) for a, b, s, g in segs]
                steps = [m for m in sc if "rise_ms" in m]
                r = {k: float(np.nanmean([m[k] for m in steps])) for k in ("rise_ms", "overshoot", "reach")}
                r["sag"] = float(np.max([abs(m["steady"]) for m in sc]))
                r["ripple"] = float(np.max([m["ripple"] for m in sc]))
                r["load"] = float(np.max([m["load"] for m in sc]))
                r["up"] = float(np.mean([m["steady"] for (a, b, s, g), m in zip(segs, sc) if g > s]))
                r["down"] = float(np.mean([m["steady"] for (a, b, s, g), m in zip(segs, sc) if g < s]))
                results[joint][p] = r
                clean = r["overshoot"] <= OVERSHOOT_MAX and r["ripple"] <= RIPPLE_MAX
                print(f"  {joint:13s} {reg} {p:2d}: rise {r['rise_ms']:4.0f} ms, overshoot {r['overshoot'] * 100:3.0f}%, "
                      f"reach {r['reach'] * 100:3.0f}%, sag max {r['sag']:3.0f} ticks "
                      f"(+step {r['up']:+.0f}, -step {r['down']:+.0f}), ripple {r['ripple']:2.0f}, "
                      f"load {r['load']:3.0f}%{'' if clean else '  NOT CLEAN'}", flush=True)
                if r["ripple"] > RIPPLE_ABORT:
                    print(f"  ripple {r['ripple']:.0f} ticks: stop raising P")
                    break
                if not clean:
                    break
            bus.send_goals({sid: bus.position(sid)})
            write_gain(bus, sid, ADDR, orig[joint][reg])
            clean = [p for p, r in results[joint].items() if "abort" not in r
                     and r["overshoot"] <= OVERSHOOT_MAX and r["ripple"] <= RIPPLE_MAX]
            stopped = len(clean) < len(results[joint])  # the sweep hit an unclean / aborted P
            rec = None
            if clean and reg == "P":
                rec = clean[-2] if stopped and len(clean) > 1 else clean[-1]
            elif clean:  # D / I: the clean value with the least sag + overshoot + ripple
                cost = lambda v: (results[joint][v]["sag"] + STEP * results[joint][v]["overshoot"]  # noqa: E731
                                  + 2 * results[joint][v]["ripple"])
                best = min(cost(v) for v in clean)
                rec = min(v for v in clean if cost(v) <= best + 2)  # near-ties: the lowest gain
            results[joint]["recommend"] = rec
            print(f"{joint}: clean {reg} {clean}; recommended {rec} (was {orig[joint][reg]})", flush=True)
        if args.apply:
            for joint in args.joints:
                rec = results[joint].get("recommend")
                if rec:
                    bus.send_goals({sids[joint]: bus.position(sids[joint])})
                    write_gain(bus, sids[joint], ADDR, rec)
                    applied[joint] = rec
            print("applied:", applied)
    except KeyboardInterrupt:
        print("\nCtrl-C")
    finally:
        try:
            for j in args.joints:
                if j not in applied and bus.r1(sids[j], ADDR) != orig[j][reg]:
                    bus.send_goals({sids[j]: bus.position(sids[j])})
                    write_gain(bus, sids[j], ADDR, orig[j][reg])
            print("holding at", bus.hold(list(sids.values())), "(torque ON)")
            print("gains now (P, D, I):", {j: tuple(bus.r1(s, a) for a in REGS.values()) for j, s in sids.items()})
        finally:
            np.savez(out_path, results=np.array([results], dtype=object), orig=np.array([orig], dtype=object),
                     **logs)
            print("saved", out_path)
            bus.close()


if __name__ == "__main__":
    main()
