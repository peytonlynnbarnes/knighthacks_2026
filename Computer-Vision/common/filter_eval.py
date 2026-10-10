#!/home/peyton/kh/.venv/bin/python
"""Score hand -> target smoothing from hand_target_node.py --log files. No cameras, no ROS.

    .venv/bin/python Computer-Vision/common/filter_eval.py out.npz [more.npz ...] [--window 20 37]

Record a test motion once (hand_target_node.py --record rec.npz), then replay it
through each filter variant (--replay rec.npz --no-ros --log variant.npz) and
compare the logs side by side. Only tracking frames count (calibration and
no-hand frames are skipped; a gap > GAP_S splits the track into segments).

Scores (lower is better unless noted):
    jitter      RMS of the frame-to-frame wobble, i.e. the distance from a
                centered moving average over JITTER_WIN_S, for the unfiltered
                target (raw) and the published one (out), mm / deg. Hold the
                hand still for a pure noise number.
    lag         time shift that best lines the output up with the raw target
                (ms), only when the hand moved more than LAG_MIN_M
    accel       RMS and 99th percentile of the output's acceleration (m/s^2):
                jerky starts and stops and leftover noise both show up here
    at cap      share of frames moving at the --max-lin speed cap (the cap
                then shapes the motion, not the filter)
    overshoot   at each stop (raw speed > STOP_FROM then < STOP_TO for STOP_HOLD_S),
                how far the output went past where the raw target stopped, along
                the direction of motion (mm): mean and max over the stops
    dt          time between published targets (ms): mean, std, max
"""
import argparse
import json
import math

import numpy as np

GAP_S = 0.15          # a gap longer than this between tracking frames splits a segment
JITTER_WIN_S = 0.3    # centered moving-average window for jitter
LAG_MIN_M = 0.02      # need this much raw motion to estimate lag
LAG_MAX_S = 0.4
GRID_HZ = 200.0       # resampling rate for the lag search
STOP_FROM, STOP_TO, STOP_HOLD_S = 0.15, 0.03, 0.25  # m/s, m/s, s: what counts as a stop


def segments(t, ok):
    """Index arrays of runs of ok frames with no gap > GAP_S."""
    idx = np.flatnonzero(ok)
    if len(idx) == 0:
        return []
    cut = np.flatnonzero(np.diff(t[idx]) > GAP_S) + 1
    return [s for s in np.split(idx, cut) if len(s) >= 5]


def wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def residual(t, x, angles=False):
    """x minus its centered moving average over JITTER_WIN_S (edges skipped)."""
    h = JITTER_WIN_S / 2
    res = []
    for i in range(len(t)):
        if t[i] - t[0] < h or t[-1] - t[i] < h:
            continue
        w = np.abs(t - t[i]) <= h
        d = x[w] - x[i]
        if angles:
            d = wrap(d)
        res.append(-d.mean(axis=0))
    return np.array(res)


def best_lag(t, raw, out):
    """(lag s, rms m) minimizing |out(t) - raw(t - lag)| on a uniform grid."""
    g = np.arange(t[0], t[-1], 1.0 / GRID_HZ)
    if len(g) < 10:
        return None
    r = np.stack([np.interp(g, t, raw[:, k]) for k in range(3)], axis=1)
    o = np.stack([np.interp(g, t, out[:, k]) for k in range(3)], axis=1)
    best = None
    for n in range(int(LAG_MAX_S * GRID_HZ) + 1):
        if len(g) - n < 10:
            break
        e = math.sqrt(np.mean(np.sum((o[n:] - r[:len(g) - n]) ** 2, axis=1)))
        if best is None or e < best[1]:
            best = (n / GRID_HZ, e)
    return best


def overshoots(t, raw, out):
    """Overshoot (m) past each stop of the raw target, see the module doc."""
    g = np.arange(t[0], t[-1], 1.0 / 100)
    if len(g) < 20:
        return []
    r = np.stack([np.interp(g, t, raw[:, k]) for k in range(3)], axis=1)
    o = np.stack([np.interp(g, t, out[:, k]) for k in range(3)], axis=1)
    # speed over 50 ms, so single-frame noise doesn't count as motion
    n = 5
    sp = np.zeros(len(g))
    sp[n:] = np.linalg.norm(r[n:] - r[:-n], axis=1) / (n / 100)
    res, i, hold = [], n, int(STOP_HOLD_S * 100)
    while i < len(g) - hold:
        if sp[i] > STOP_FROM:
            j = i
            while j < len(g) - hold and not np.all(sp[j:j + hold] < STOP_TO):
                j += 1
            if j >= len(g) - hold:
                break
            d = r[j] - r[max(n, i - 20)]
            if np.linalg.norm(d) > 0.02:
                d /= np.linalg.norm(d)
                end = r[j:j + hold].mean(axis=0)
                res.append(max(0.0, float(np.max((o[j:j + hold + 30] - end) @ d))))
            i = j + hold
        else:
            i += 1
    return res


def score(path, window=None):
    """window: (start, end) s from the first frame of the log, or None = all."""
    z = np.load(path)
    t, state, raw, out = z["t"], z["state"], z["raw"], z["out"]
    args = json.loads(str(z["args"])) if "args" in z.files else {}
    ok = (state == 1) & np.all(np.isfinite(out), axis=1)
    if window is not None:
        rel = t - t[0]
        inside = (rel >= window[0]) & (rel <= window[1])
        ok &= inside
        state = np.where(inside, state, -1)  # excluded from the no-hand share too
    segs = segments(t, ok)
    s = {"file": path, "frames": int(ok.sum()), "segments": len(segs),
         "lost": float(np.mean(state[state > 0] == 2)) if np.any(state > 0) else float("nan")}
    if not segs:
        return s
    jr_p, jo_p, jr_a, jo_a, acc, dts, caps, lags, ovs = [], [], [], [], [], [], [], [], []
    max_lin = args.get("max_lin")
    for ix in segs:
        ts = t[ix]
        jr_p.append(residual(ts, raw[ix, :3]))
        jo_p.append(residual(ts, out[ix, :3]))
        jr_a.append(residual(ts, raw[ix, 3:], True))
        jo_a.append(residual(ts, out[ix, 3:], True))
        dt = np.diff(ts)
        dts.append(dt)
        v = np.diff(out[ix, :3], axis=0) / dt[:, None]
        sp = np.linalg.norm(v, axis=1)
        if max_lin:
            caps.append(sp >= 0.95 * max_lin)
        if len(v) > 1:
            acc.append(np.linalg.norm(np.diff(v, axis=0), axis=1) / (0.5 * (dt[1:] + dt[:-1])))
        ovs += overshoots(ts, raw[ix, :3], out[ix, :3])
        span = np.ptp(raw[ix, :3], axis=0).max()
        if span > LAG_MIN_M:
            lg = best_lag(ts, raw[ix, :3], out[ix, :3])
            if lg is not None:
                lags.append((lg[0], len(ix)))

    def rms_norm(parts, scale):
        a = np.vstack([p for p in parts if len(p)]) if any(len(p) for p in parts) else None
        return float("nan") if a is None else scale * math.sqrt(np.mean(np.sum(a ** 2, axis=1)))

    s["jit_raw_mm"], s["jit_out_mm"] = rms_norm(jr_p, 1000), rms_norm(jo_p, 1000)
    s["jit_raw_deg"], s["jit_out_deg"] = rms_norm(jr_a, 180 / math.pi), rms_norm(jo_a, 180 / math.pi)
    a = np.concatenate(acc) if acc else np.array([np.nan])
    s["acc_rms"], s["acc_p99"] = float(np.sqrt(np.mean(a ** 2))), float(np.percentile(a, 99))
    d = np.concatenate(dts) * 1000
    s["dt_mean"], s["dt_std"], s["dt_max"] = float(d.mean()), float(d.std()), float(d.max())
    s["at_cap"] = float(np.mean(np.concatenate(caps))) if caps else float("nan")
    s["stops"] = len(ovs)
    s["ov_mean"] = 1000 * float(np.mean(ovs)) if ovs else float("nan")
    s["ov_max"] = 1000 * float(np.max(ovs)) if ovs else float("nan")
    if lags:
        w = np.array([n for _, n in lags], dtype=float)
        s["lag_ms"] = 1000 * float(np.average([lg for lg, _ in lags], weights=w))
    else:
        s["lag_ms"] = float("nan")
    return s


ROWS = (("frames", "tracking frames", "{:.0f}"), ("segments", "segments", "{:.0f}"),
        ("lost", "no-hand share", "{:.0%}"),
        ("jit_raw_mm", "jitter raw (mm)", "{:.2f}"), ("jit_out_mm", "jitter out (mm)", "{:.2f}"),
        ("jit_raw_deg", "jitter raw (deg)", "{:.2f}"), ("jit_out_deg", "jitter out (deg)", "{:.2f}"),
        ("lag_ms", "lag (ms)", "{:.0f}"),
        ("acc_rms", "accel RMS (m/s2)", "{:.2f}"), ("acc_p99", "accel p99 (m/s2)", "{:.2f}"),
        ("at_cap", "at speed cap", "{:.0%}"),
        ("stops", "stops", "{:.0f}"), ("ov_mean", "overshoot mean (mm)", "{:.1f}"),
        ("ov_max", "overshoot max (mm)", "{:.1f}"),
        ("dt_mean", "dt mean (ms)", "{:.1f}"), ("dt_std", "dt std (ms)", "{:.1f}"),
        ("dt_max", "dt max (ms)", "{:.0f}"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+")
    ap.add_argument("--window", type=float, nargs=2, metavar=("START", "END"),
                    help="only score this span (s from the log's first frame)")
    args = ap.parse_args()
    scores = [score(p, args.window) for p in args.logs]
    names = [s["file"].rsplit("/", 1)[-1] for s in scores]
    w = max(12, *(len(n) for n in names))
    print(f"{'':<20}" + "".join(f"{n:>{w + 2}}" for n in names))
    for key, label, fmt in ROWS:
        cells = []
        for s in scores:
            v = s.get(key)
            cells.append("-" if v is None or (isinstance(v, float) and math.isnan(v)) else fmt.format(v))
        print(f"{label:<20}" + "".join(f"{c:>{w + 2}}" for c in cells))


if __name__ == "__main__":
    main()
