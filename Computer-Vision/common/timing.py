"""Debug timers: where the time goes, per pipeline stage, across threads.

    T = Timers()
    with T("wilor.forward"):
        ...
    T.add("cam0.read", dt)          # a duration measured elsewhere (s)
    T.tick("pairs")                 # a rate counter, no duration
    print(T.report())               # table since the last report, then reset

Stage names are "thread.stage" by convention, so the report groups by thread. Per stage:
calls per second, mean / p90 / max ms, and busy % = the share of wall time that stage
took (the stages of one thread add up to <= 100 %; the busiest thread is the bottleneck).
Thread-safe; the overhead is ~1 us per measurement. A disabled Timers (enabled=False)
does nothing, so callers don't need ifs.
"""
import threading
import time
from collections import defaultdict
from contextlib import contextmanager

import numpy as np


class Timers:
    def __init__(self, enabled=True):
        self.enabled = enabled
        self._lock = threading.Lock()
        self._d = defaultdict(list)    # stage -> durations (s) since the last report
        self._n = defaultdict(int)     # counter -> ticks since the last report
        self._t0 = time.perf_counter()
        self.last = {}                 # stage -> mean ms in the last report (for the preview)

    @contextmanager
    def __call__(self, name):
        if not self.enabled:
            yield
            return
        t = time.perf_counter()
        try:
            yield
        finally:
            self.add(name, time.perf_counter() - t)

    def add(self, name, dt):
        if self.enabled:
            with self._lock:
                self._d[name].append(dt)

    def tick(self, name):
        if self.enabled:
            with self._lock:
                self._n[name] += 1

    def report(self):
        """Table of everything since the last report (then reset), busiest stage first
        within each thread."""
        with self._lock:
            d, n, self._d, self._n = self._d, self._n, defaultdict(list), defaultdict(int)
            t1 = time.perf_counter()
            wall, self._t0 = t1 - self._t0, t1
        if wall <= 0:
            return ""
        rows, last = [], {}
        for name, v in d.items():
            a = np.asarray(v) * 1000
            last[name] = float(a.mean())
            rows.append((name.split(".")[0], -a.sum(), name, len(a) / wall, a.mean(),
                         np.percentile(a, 90), a.max(), a.sum() / 10 / wall))
        self.last = last
        rows.sort()
        out = [f"--- timing over {wall:.1f} s "
               + "  ".join(f"{k} {c / wall:.1f}/s" for k, c in sorted(n.items())),
               f"{'stage':<24}{'/s':>6}{'mean':>8}{'p90':>8}{'max':>8}{'busy':>7}  (ms)"]
        for _, _, name, rate, mean, p90, mx, busy in rows:
            out.append(f"{name:<24}{rate:6.1f}{mean:8.1f}{p90:8.1f}{mx:8.1f}{busy:6.0f}%")
        return "\n".join(out)

    def summary(self, names):
        """'name mean' for the given stages from the last report (preview line)."""
        return "  ".join(f"{k.split('.')[-1]} {self.last[k]:.0f}" for k in names if k in self.last)


NULL = Timers(enabled=False)


if __name__ == "__main__":
    T = Timers()
    for _ in range(20):
        with T("a.sleep5"):
            time.sleep(0.005)
        T.add("b.fake", 0.002)
        T.tick("pairs")
    r = T.report()
    print(r)
    assert "a.sleep5" in r and "pairs" in r and 4.5 < T.last["a.sleep5"] < 20, T.last
    assert T.report().count("\n") == 1  # reset: header only
    print("timing self-test OK")
