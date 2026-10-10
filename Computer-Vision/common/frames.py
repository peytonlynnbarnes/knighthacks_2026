"""Record / replay both camera streams, so landmark backends can be compared on identical video.

--record (hand_target_node.py) saves landmarks, so it can only replay the backend
that made them. --record-frames DIR saves the frames themselves (JPEG, quality 95)
plus both capture times per pair; --replay-frames DIR feeds them back in place
of StereoCam and the chosen --backend runs on them live (see ../WILOR_PLAN.md,
phase 3).

DIR layout: cam0.jpgs / cam1.jpgs (concatenated JPEGs, streamed to disk while
recording) and index.npz (t, t0, t1, byte offsets, recal = pair indices where
'c' was pressed, argv).
"""
import os
import sys
import time

import cv2
import numpy as np


class ReplayEnd(Exception):
    pass


class FrameRecorder:
    def __init__(self, path, quality=95):
        os.makedirs(path, exist_ok=True)
        self.path, self.q = path, [int(cv2.IMWRITE_JPEG_QUALITY), quality]
        self.files = [open(os.path.join(path, f"cam{i}.jpgs"), "wb") for i in (0, 1)]
        self.offs = ([0], [0])
        self.t, self.t0, self.t1, self.recal = [], [], [], []

    def add(self, f0, f1, t, t0, t1):
        for i, f in enumerate((f0, f1)):
            ok, buf = cv2.imencode(".jpg", f, self.q)
            self.files[i].write(buf.tobytes())
            self.offs[i].append(self.offs[i][-1] + len(buf))
        self.t.append(t)
        self.t0.append(t0)
        self.t1.append(t1)

    def mark_recal(self):
        self.recal.append(len(self.t) - 1)

    def save(self):
        for f in self.files:
            f.close()
        if not self.t:
            return
        np.savez(os.path.join(self.path, "index.npz"), t=self.t, t0=self.t0, t1=self.t1,
                 off0=self.offs[0], off1=self.offs[1], recal=np.array(self.recal, dtype=int),
                 argv=" ".join(sys.argv[1:]))
        mb = (self.offs[0][-1] + self.offs[1][-1]) / 1e6
        print(f"recorded {len(self.t)} frame pairs ({self.t[-1] - self.t[0]:.0f} s, {mb:.0f} MB) -> {self.path}")


class FrameReplay:
    """In place of StereoCam: pair() -> (f0, f1, t) from a --record-frames dir; t0/t1 as recorded.

    live=True drops pairs the way the live loop does: a virtual clock starts at the
    first pair and advances by the real time spent between pair() calls (the
    backend + the rest of the loop, on this machine); pair() then returns the
    first recorded pair at or after that clock, as live pair() waits for the
    next camera frame. A slow backend (WiLoR ~16 Hz) then sees the frames it
    would have seen live, not all ~31/s of a full-rate recording. The timing
    is measured, so live replays vary a little from run to run."""

    def __init__(self, path, realtime, live=False):
        z = np.load(os.path.join(path, "index.npz"))
        self.t, self.t0s, self.t1s = z["t"], z["t0"], z["t1"]
        self.offs = (z["off0"], z["off1"])
        self.recal = set(int(i) for i in z["recal"])
        self.argv = str(z["argv"])
        self.data = [np.memmap(os.path.join(path, f"cam{i}.jpgs"), dtype=np.uint8, mode="r") for i in (0, 1)]
        self.realtime, self.i, self.start = realtime, -1, None
        self.live, self.clock, self.wall = live, None, None
        self.prev_i, self.skipped = -1, 0
        self.t0 = self.t1 = 0.0

    def frame(self, cam, i):
        o = self.offs[cam]
        return cv2.imdecode(np.asarray(self.data[cam][o[i]:o[i + 1]]), cv2.IMREAD_COLOR)

    def pair(self):
        self.prev_i = self.i
        self.i += 1
        if self.live and self.wall is not None:
            self.clock += time.monotonic() - self.wall
            j = min(int(np.searchsorted(self.t, self.clock)), len(self.t))  # first pair at/after the clock
            if j > self.i:
                self.skipped += j - self.i
                self.i = j
        if self.i >= len(self.t):
            raise ReplayEnd
        t = float(self.t[self.i])
        if self.live:
            self.clock = max(self.clock or t, t)
        if self.realtime:
            if self.start is None:
                self.start = time.monotonic() - t
            time.sleep(max(0.0, self.start + t - time.monotonic()))
        self.t0, self.t1 = float(self.t0s[self.i]), float(self.t1s[self.i])
        f0, f1 = self.frame(0, self.i), self.frame(1, self.i)
        self.wall = time.monotonic()  # decoding isn't part of the live loop
        return f0, f1, t

    def recalibrate_now(self):
        """'c' was pressed at this pair, or at a pair the live mode skipped."""
        return any(self.prev_i < k <= self.i for k in self.recal)

    def close(self):
        pass


def _selftest():
    import tempfile
    d = tempfile.mkdtemp()
    rec = FrameRecorder(os.path.join(d, "r"))
    yy, xx = np.mgrid[0:48, 0:64]
    img = lambda k: np.dstack([(xx * 4 + k * 10) % 256, yy * 5, (xx + yy) * 2]).astype(np.uint8)  # noqa: E731
    frames = [(img(k), img(k + 7)) for k in range(5)]
    for k, (a, b) in enumerate(frames):
        rec.add(a, b, 10.0 + k / 30, 10.0 + k / 30, 10.005 + k / 30)
    rec.mark_recal()
    rec.save()
    rep = FrameReplay(os.path.join(d, "r"), realtime=False)
    for k, (a, b) in enumerate(frames):
        f0, f1, t = rep.pair()
        assert f0.shape == a.shape and abs(t - (10.0 + k / 30)) < 1e-9 and abs(rep.t1 - rep.t0 - 0.005) < 1e-9
        assert np.mean(np.abs(f0.astype(int) - a)) < 3  # JPEG: lossy but close
    assert rep.recalibrate_now()
    try:
        rep.pair()
        raise AssertionError("expected ReplayEnd")
    except ReplayEnd:
        pass
    # live: 50 ms of "processing" per pair at 30 pairs/s -> every other pair or so is dropped
    rec = FrameRecorder(os.path.join(d, "l"))
    for k in range(60):
        rec.add(frames[0][0], frames[0][1], k / 30, k / 30, k / 30)
        if k == 30:
            rec.mark_recal()  # mid-run: must fire even when live mode skips over pair 30
    rec.save()
    rep = FrameReplay(os.path.join(d, "l"), realtime=False, live=True)
    seen, recal = [], False
    try:
        while True:
            seen.append(rep.pair()[2])
            recal |= rep.recalibrate_now()
            time.sleep(0.05)
    except ReplayEnd:
        pass
    gaps = np.diff(seen)
    assert 15 <= len(seen) <= 30 and np.all(gaps >= 0.05 - 1e-6), (len(seen), gaps.min())
    assert recal and rep.skipped == 60 - len(seen)
    print(f"frames self-test OK (live: {len(seen)} of 60 pairs at 50 ms each)")


if __name__ == "__main__":
    _selftest()
