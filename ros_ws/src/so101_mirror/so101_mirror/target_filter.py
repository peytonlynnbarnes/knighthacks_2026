"""Irregular /ball_position targets -> evenly paced IK targets. Pure numpy.

Targets arrive at the camera rate (13-30 Hz, irregular: dropped frames,
landmark model time varies), while the IK solves on a fixed timer. Taking the
latest target each tick re-solves the same target on some ticks and skips one
on others, so the arm gets uneven steps. TargetInterpolator samples the target
path `delay` seconds behind real time instead, linearly between the two
targets around that moment, so every tick moves a fair share. The delay should
cover one normal gap between targets (~40 ms at 25 fps); a later target just
holds the last one until it arrives (no extrapolation).

Optional time-based smoothing (tau, s): an EMA whose strength depends on
elapsed time, not on how many targets arrived (the old per-message alpha
smoothed more when the camera ran faster).

Self-test: python3 -m so101_mirror.target_filter
"""
import math
from collections import deque

import numpy as np


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


class TargetInterpolator:
    """[x y z roll pitch yaw] targets in, evenly paced samples out (angles wrap)."""

    def __init__(self, delay=0.04, tau=0.0, max_dt=0.1, keep=16):
        self.delay, self.tau, self.max_dt = delay, tau, max_dt
        self.buf = deque(maxlen=keep)  # (t, target)
        self.out = self.out_t = None

    def reset(self):
        self.buf.clear()
        self.out = self.out_t = None

    def add(self, t, d):
        d = np.asarray(d, dtype=float)
        if self.buf and t <= self.buf[-1][0]:  # same stamp (or clock went back): replace
            self.buf.pop()
        self.buf.append((t, d))

    def _at(self, ts):
        if ts <= self.buf[0][0]:
            return self.buf[0][1].copy()
        if ts >= self.buf[-1][0]:
            return self.buf[-1][1].copy()
        for (ta, a), (tb, b) in zip(self.buf, list(self.buf)[1:]):
            if ta <= ts <= tb:
                u = (ts - ta) / (tb - ta)
                diff = b - a
                diff[3:] = _wrap(diff[3:])
                out = a + u * diff
                out[3:] = _wrap(out[3:])
                return out
        return self.buf[-1][1].copy()

    def sample(self, t):
        """Target for time t (sampled at t - delay, then smoothed), or None before any target."""
        if not self.buf:
            return None
        x = self._at(t - self.delay)
        if self.tau > 0 and self.out is not None:
            dt = min(max(0.0, t - self.out_t), self.max_dt)
            diff = x - self.out
            diff[3:] = _wrap(diff[3:])
            x = self.out + (1.0 - math.exp(-dt / self.tau)) * diff
            x[3:] = _wrap(x[3:])
        self.out, self.out_t = x, t
        return x.copy()


def _selftest():
    rng = np.random.default_rng(0)
    # a ramp arriving at irregular times: samples at a fixed rate land on the ramp,
    # delay behind, and evenly spaced (unlike taking the latest target)
    v = np.array([0.1, 0.0, 0.0, 0.5, 0.0, 0.0])  # m/s, rad/s
    ti = TargetInterpolator(delay=0.05)
    t_arr, t = [], 0.0
    while t < 2.0:
        t += rng.uniform(0.025, 0.05)
        t_arr.append(t)
    i, latest, steps_i, steps_l, prev_i, prev_l = 0, None, [], [], None, None
    for k in range(int(2.0 * 30)):
        now = k / 30.0
        while i < len(t_arr) and t_arr[i] <= now:
            ti.add(t_arr[i], v * t_arr[i])
            latest = v * t_arr[i]
            i += 1
        s = ti.sample(now)
        if s is None or now < 0.2:
            continue
        assert np.allclose(s, v * (now - 0.05), atol=1e-9), (now, s)
        if prev_i is not None:
            steps_i.append(s[0] - prev_i)
            steps_l.append(latest[0] - prev_l)
        prev_i, prev_l = s[0], latest[0]
    print(f"step std at 30 Hz: interpolated {np.std(steps_i) * 1000:.3f} mm, "
          f"latest {np.std(steps_l) * 1000:.3f} mm")
    assert np.std(steps_i) < 1e-9 < np.std(steps_l)
    # holds the last target when they stop, and angles interpolate across +-pi
    for ta in t_arr[i:]:
        ti.add(ta, v * ta)
    assert np.allclose(ti.sample(10.0), v * t_arr[-1])
    ti = TargetInterpolator(delay=0.0)
    ti.add(0.0, [0, 0, 0, math.pi - 0.1, 0, 0])
    ti.add(1.0, [0, 0, 0, -math.pi + 0.1, 0, 0])
    assert abs(abs(ti.sample(0.5)[3]) - math.pi) < 1e-9
    # time-based smoothing: same result at any sample rate
    ends = []
    for hz in (25.0, 50.0, 100.0):
        ti = TargetInterpolator(delay=0.0, tau=0.1)
        ti.add(0.0, np.zeros(6))
        ti.sample(0.0)
        ti.add(0.001, np.ones(6))
        for k in range(1, int(0.2 * hz) + 1):
            out = ti.sample(k / hz)
        ends.append(out[0])
    print(f"tau 0.1 s after 0.2 s at 25/50/100 Hz: {np.round(ends, 4)}")
    assert np.ptp(ends) < 0.01 and abs(ends[0] - (1 - math.exp(-2))) < 0.01
    print("target_filter self-test OK")


if __name__ == "__main__":
    _selftest()
