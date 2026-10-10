#!/home/peyton/kh/.venv/bin/python
"""Offline benchmark of the position filters (hand_mapping) on logged hand positions.

    .venv/bin/python Computer-Vision/common/kalman_bench.py [--logs DIR] [--backend wilor]

Windows: DIR/windows.json ({"take": [start, end] or null}) if present, else
WINDOWS (the baseline recordings). An explicit window scores every frame with a
hand pose in it, calibration frames too; a take without one is scored over
everything after the hand calibration.

Runs each filter directly on the raw cam-frame hand positions in --log files
(hand_target_node.py --replay rec_<name>.npz --no-ros --log DIR/<name>.npz),
without the mapping / box / speed caps after it, so it isolates the filter.
filter_eval.py scores the whole published output instead.

Reference: a zero-lag (acausal) local polynomial fit over +-REF_S seconds
(9 frames, cubic, at MediaPipe's ~30 Hz; 5 frames, quadratic, at WiLoR's
~16 Hz), which a live filter can't do because it sees future frames. The
window is in seconds so both frame rates get the same reference smoothness. Scores against it:
    still jit   RMS error while the reference is slower than 3 cm/s (mm)
    still wob   output jitter on the still take, as filter_eval measures it (mm):
                at a low frame rate the reference keeps much of the noise
                itself, so "still jit" understates the smoothing there
    err         RMS error to the reference while tracking (mm): noise + lag
    lag         time shift that best lines the filter up with the reference (ms)
    ov          mean overshoot past the reference at stops (mm), filter_eval.overshoots
    acc         RMS acceleration of the filter output (m/s^2): roughness
Also prints the measured noise (raw - reference) by hand speed, which is what
StereoKalman's noise model (ray_px, sigma_px, speed_k) should match.

Edit FILTERS below to compare settings.
"""
import argparse
import json
import math
import os

import numpy as np

import filter_eval as fe
from hand_mapping import IMM_PRESETS, KALMAN_DEFAULTS, OneEuro, StereoIMM, StereoKalman
from stereo_cam import HERE, RIG_PATH, load_rig

# scoring windows (s from the log's first frame), see recordings/baseline/README.md
WINDOWS = {"still": (24.5, 38.5), "sweep": (35.5, 53), "reach": (25.1, 58.6), "drop_out": (28.5, 47)}
REF_S = 0.135  # reference fit half-width (s): +-4 frames at 30 Hz, as before
SLOW = 0.03  # m/s


def ref_half(t):
    """Reference half-width in frames for this frame rate (>= 2)."""
    return max(2, int(round(REF_S / float(np.median(np.diff(t))))))


def local_fit(t, x, deriv=0):
    """Acausal local polynomial fit (Savitzky-Golay on irregular times) of x (N, 3):
    +-ref_half frames, cubic when that's >= 7 points, else quadratic."""
    out = np.full_like(x, np.nan)
    half = ref_half(t)
    order = 3 if 2 * half + 1 >= 7 else 2
    for i in range(len(t)):
        a, b = max(0, i - half), min(len(t), i + half + 1)
        if b - a < order + 2:
            continue
        V = np.vander(t[a:b] - t[i], order + 1, increasing=True)
        c, *_ = np.linalg.lstsq(V, x[a:b], rcond=None)
        out[i] = c[deriv]
    return out


def windows(logs):
    path = os.path.join(logs, "windows.json")
    if not os.path.exists(path):
        return WINDOWS
    with open(path) as f:
        return {k: (None, None) if v is None else tuple(v) for k, v in json.load(f).items()}


def take_names(logs):
    """Names with a window first (in that order), then any other .npz in logs."""
    found = sorted(f[:-4] for f in os.listdir(logs) if f.endswith(".npz"))
    win = windows(logs)
    return [n for n in win if n in found] + [n for n in found if n not in win]


def load_segments(logs):
    segs = []
    win = windows(logs)
    for name in take_names(logs):
        z = np.load(os.path.join(logs, name + ".npz"))
        t = z["t"] - z["t"][0]
        a, b = win.get(name, (None, None))
        if a is None:  # everything after the hand calibration
            a, b = (t[np.argmax(z["state"] == 1)] if (z["state"] == 1).any() else np.inf), np.inf
            ok = z["state"] == 1
        else:  # an explicit window: every frame with a hand pose, also while calibrating
            ok = np.isfinite(z["hand_p"][:, 0])  # (a take whose calibration never finished)
        ok &= (t >= a) & (t <= b)
        idx = np.flatnonzero(ok)
        for seg in np.split(idx, np.flatnonzero(np.diff(z["t"][idx]) > fe.GAP_S) + 1):
            if len(seg) >= 15:
                ts, p = z["t"][seg], z["hand_p"][seg]
                segs.append(dict(name=name, t=ts, p=p, err=z["err"][seg], mono=z["mono"][seg],
                                 ref=local_fit(ts, p),
                                 speed=np.linalg.norm(local_fit(ts, p, deriv=1), axis=1)))
    return segs


def noise_by_speed(segs):
    rows = []
    for s in segs:
        k = slice(ref_half(s["t"]) + 1, -ref_half(s["t"]) - 1)
        ray = s["ref"] / np.linalg.norm(s["ref"], axis=1)[:, None]
        r = (s["p"] - s["ref"])[k]
        along = np.sum(r * ray[k], axis=1)
        lat = np.linalg.norm(r - along[:, None] * ray[k], axis=1) / math.sqrt(2)
        rows.append(np.stack([s["speed"][k], along, lat], axis=1))
    if not rows:
        print("no tracking segments")
        return
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
    for s in segs:
        k = slice(ref_half(s["t"]) + 1, -ref_half(s["t"]) - 1)
        f = make()
        out = np.array([f(p, t, e, bool(m)) for p, t, e, m in zip(s["p"], s["t"], s["err"], s["mono"])])
        r = acc.setdefault(s["name"], dict(err=[], jit=[], ov=[], acc=[], lag=[], wob=[]))
        r["wob"].append(fe.residual(s["t"], out))
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
        res[n] = dict(jit=rms(r["jit"]), err=rms(r["err"]), wob=rms(r["wob"]),
                      ov=1e3 * float(np.mean(r["ov"])) if r["ov"] else float("nan"),
                      acc=float(math.sqrt(np.mean(a ** 2))),
                      lag=1e3 * float(np.average([x for x, _ in r["lag"]], weights=[w for _, w in r["lag"]]))
                      if r["lag"] else float("nan"))
    return res


def filters(f, B, backend="mediapipe"):
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
    out += [(f"imm {name}", wrap(lambda kw=kw: StereoIMM(f, B, **kw))) for name, kw in IMM_PRESETS["mediapipe"].items()]
    if backend != "mediapipe":  # its own tuning next to the MediaPipe one
        kf = KALMAN_DEFAULTS[backend]
        out += [(f"kalman {backend} (s {kf['sigma_px']:g}, q {kf['q']:g})", wrap(lambda: StereoKalman(f, B, **kf)))]
        out += [(f"imm {backend} {name}", wrap(lambda kw=kw: StereoIMM(f, B, **kw)))
                for name, kw in IMM_PRESETS[backend].items()]
    return out


COLS = (("still", "jit"), ("still", "wob"), ("sweep", "err"), ("sweep", "lag"), ("sweep", "acc"), ("reach", "err"),
        ("reach", "lag"), ("reach", "ov"), ("reach", "acc"), ("drop_out", "err"), ("edge_on", "err"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--logs", default=os.path.join(HERE, "recordings", "baseline"))
    ap.add_argument("--ref-s", type=float, default=REF_S, help="reference fit half-width (s)")
    ap.add_argument("--backend", default="mediapipe", choices=sorted(IMM_PRESETS),
                    help="also score this backend's own presets (hand_mapping.IMM_PRESETS)")
    args = ap.parse_args()
    globals()["REF_S"] = args.ref_s
    rig = load_rig(RIG_PATH)
    f, B = float(rig["K0"][0, 0]), float(np.linalg.norm(rig["t"]))
    segs = load_segments(args.logs)
    noise_by_speed(segs)
    names = [n for n in take_names(args.logs) if n in {s["name"] for s in segs}]
    cols = [c for c in COLS if c[0] in names]
    print(f"{'filter':28s}" + "".join(f"{n[:5] + ' ' + k:>11s}" for n, k in cols))
    for label, make in filters(f, B, args.backend):
        r = bench(segs, make)
        print(f"{label:28s}" + "".join(f"{r[n][k]:11.2f}" for n, k in cols))


if __name__ == "__main__":
    main()
