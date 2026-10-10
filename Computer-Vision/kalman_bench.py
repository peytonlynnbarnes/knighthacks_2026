#!/home/peyton/kh/.venv/bin/python
"""Offline benchmark of the position filters (hand_mapping) on logged hand positions.

    .venv/bin/python Computer-Vision/kalman_bench.py [--logs DIR]

Runs each filter directly on the raw cam-frame hand positions in --log files
(hand_target_node.py --replay rec_<name>.npz --no-ros --log DIR/<name>.npz),
without the mapping / box / speed caps after it, so it isolates the filter.
filter_eval.py scores the whole published output instead.

Reference: a zero-lag (acausal) local cubic fit over 9 frames (~0.3 s), which
a live filter can't do because it sees future frames. Scores against it:
    still jit   RMS error while the reference is slower than 3 cm/s (mm)
    err         RMS error to the reference while tracking (mm): noise + lag
    lag         time shift that best lines the filter up with the reference (ms)
    ov          mean overshoot past the reference at stops (mm), filter_eval.overshoots
    acc         RMS acceleration of the filter output (m/s^2): roughness
Also prints the measured noise (raw - reference) by hand speed, which is what
StereoKalman's noise model (ray_px, sigma_px, speed_k) should match.

Edit FILTERS below to compare settings.
"""
import argparse
import math
import os

import numpy as np

import filter_eval as fe
from hand_mapping import IMM_PRESETS, OneEuro, StereoIMM, StereoKalman
from stereo_cam import HERE, RIG_PATH, load_rig

# scoring windows (s from the log's first frame), see recordings/baseline/README.md
WINDOWS = {"still": (24.5, 38.5), "sweep": (35.5, 53), "reach": (25.1, 58.6), "drop_out": (28.5, 47)}
REF_HALF, REF_ORDER = 4, 3  # reference fit: +-4 frames, cubic
SLOW = 0.03  # m/s


def local_fit(t, x, deriv=0):
    """Acausal local polynomial fit (Savitzky-Golay on irregular times) of x (N, 3)."""
    out = np.full_like(x, np.nan)
    for i in range(len(t)):
        a, b = max(0, i - REF_HALF), min(len(t), i + REF_HALF + 1)
        if b - a < REF_ORDER + 2:
            continue
        V = np.vander(t[a:b] - t[i], REF_ORDER + 1, increasing=True)
        c, *_ = np.linalg.lstsq(V, x[a:b], rcond=None)
        out[i] = c[deriv]
    return out


def load_segments(logs):
    segs = []
    for name, (a, b) in WINDOWS.items():
        path = os.path.join(logs, name + ".npz")
        if not os.path.exists(path):
            print(f"(no {path})")
            continue
        z = np.load(path)
        t = z["t"] - z["t"][0]
        ok = (z["state"] == 1) & (t >= a) & (t <= b)
        idx = np.flatnonzero(ok)
        for seg in np.split(idx, np.flatnonzero(np.diff(z["t"][idx]) > fe.GAP_S) + 1):
            if len(seg) >= 15:
                ts, p = z["t"][seg], z["hand_p"][seg]
                segs.append(dict(name=name, t=ts, p=p, err=z["err"][seg], mono=z["mono"][seg],
                                 ref=local_fit(ts, p),
                                 speed=np.linalg.norm(local_fit(ts, p, deriv=1), axis=1)))
    return segs


def noise_by_speed(segs):
    rows, k = [], slice(REF_HALF + 1, -REF_HALF - 1)
    for s in segs:
        ray = s["ref"] / np.linalg.norm(s["ref"], axis=1)[:, None]
        r = (s["p"] - s["ref"])[k]
        along = np.sum(r * ray[k], axis=1)
        lat = np.linalg.norm(r - along[:, None] * ray[k], axis=1) / math.sqrt(2)
        rows.append(np.stack([s["speed"][k], along, lat], axis=1))
    a = np.vstack(rows)
    print("measured noise (raw - reference):  speed m/s     n   along-ray mm   sideways mm")
    for lo, hi in ((0, 0.03), (0.03, 0.1), (0.1, 0.2), (0.2, 0.4), (0.4, 9)):
        m = (a[:, 0] >= lo) & (a[:, 0] < hi)
        if m.sum() > 10:
            print(f"{'':36s}{lo:4.2f}-{hi:<5.2f} {m.sum():5d}   {1e3 * math.sqrt(np.mean(a[m, 1] ** 2)):8.2f}"
                  f"   {1e3 * math.sqrt(np.mean(a[m, 2] ** 2)):9.2f}")
    print()


def bench(segs, make):
    """Scores per recording for filter factory make() -> f(p, t, err, mono) -> p."""
    acc = {}
    k = slice(REF_HALF + 1, -REF_HALF - 1)
    for s in segs:
        f = make()
        out = np.array([f(p, t, e, bool(m)) for p, t, e, m in zip(s["p"], s["t"], s["err"], s["mono"])])
        r = acc.setdefault(s["name"], dict(err=[], jit=[], ov=[], acc=[], lag=[]))
        d = (out - s["ref"])[k]
        r["err"].append(d)
        r["jit"].append(d[s["speed"][k] < SLOW])
        r["ov"] += fe.overshoots(s["t"], s["ref"], out)
        v = np.diff(out, axis=0) / np.diff(s["t"])[:, None]
        r["acc"].append(np.linalg.norm(np.diff(v, axis=0), axis=1) / np.diff(s["t"])[1:])
        if np.ptp(s["ref"], axis=0).max() > fe.LAG_MIN_M:
            lg = fe.best_lag(s["t"], s["ref"], out)
            if lg is not None:
                r["lag"].append((lg[0], len(s["t"])))
    rms = lambda parts: (1e3 * math.sqrt(np.mean(np.sum(np.vstack(parts) ** 2, axis=1)))  # noqa: E731
                         if sum(len(x) for x in parts) else float("nan"))
    res = {}
    for n, r in acc.items():
        a = np.concatenate(r["acc"])
        res[n] = dict(jit=rms(r["jit"]), err=rms(r["err"]),
                      ov=1e3 * float(np.mean(r["ov"])) if r["ov"] else float("nan"),
                      acc=float(math.sqrt(np.mean(a ** 2))),
                      lag=1e3 * float(np.average([x for x, _ in r["lag"]], weights=[w for _, w in r["lag"]]))
                      if r["lag"] else float("nan"))
    return res


def filters(f, B):
    def euro():
        e = OneEuro(1.0, 2.0)
        return lambda p, t, err=None, mono=False: e(p, t)

    def wrap(make):
        def factory():
            m = make()
            return lambda p, t, err=None, mono=False: m(p, t, err, mono)[0]
        return factory

    out = [("one-euro (1 Hz, beta 2)", euro),
           ("kalman (sigma 4, q 0.001)", wrap(lambda: StereoKalman(f, B)))]
    out += [(f"imm {name}", wrap(lambda kw=kw: StereoIMM(f, B, **kw))) for name, kw in IMM_PRESETS.items()]
    return out


COLS = (("still", "jit"), ("sweep", "err"), ("sweep", "lag"), ("sweep", "acc"), ("reach", "err"),
        ("reach", "lag"), ("reach", "ov"), ("reach", "acc"), ("drop_out", "err"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default=os.path.join(HERE, "recordings", "baseline"))
    args = ap.parse_args()
    rig = load_rig(RIG_PATH)
    f, B = float(rig["K0"][0, 0]), float(np.linalg.norm(rig["t"]))
    segs = load_segments(args.logs)
    noise_by_speed(segs)
    names = sorted({s["name"] for s in segs}, key=list(WINDOWS).index)
    cols = [c for c in COLS if c[0] in names]
    print(f"{'filter':28s}" + "".join(f"{n[:5] + ' ' + k:>11s}" for n, k in cols))
    for label, make in filters(f, B):
        r = bench(segs, make)
        print(f"{label:28s}" + "".join(f"{r[n][k]:11.2f}" for n, k in cols))


if __name__ == "__main__":
    main()
