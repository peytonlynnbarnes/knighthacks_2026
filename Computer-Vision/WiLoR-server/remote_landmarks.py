"""Remote WiLoR landmark backend (--backend wilor-remote): the model runs in wilor_server.py on
the cloud GPU, this file runs on the laptop in the main venv (OpenCV + numpy + stdlib only).

Same interface as ../WiLoR/wilor_landmarks.py:

    StereoLandmarks(f0, f1, t) -> (px0, px1), each a (21, 2) array of full-frame
    pixels (OpenPose joint order = MediaPipe's) or None.

Called like that, every pair is one blocking round trip (fine for stereo_calib.py /
stereo_cam.py previews, too slow for tracking: ~240 ms per pair to the current server).
hand_target_node.py instead wraps its camera with `cam = lms.wrap(cam)`: pair() then
sends each new camera pair to the server and returns the oldest pair whose answer is
back, with `inflight` requests on the wire at once. Throughput is then ~inflight /
round trip pairs/s, but every result is still one round trip old. The returned pair
keeps its own capture times (t, cam.t0, cam.t1), so the filters, skew correction,
recordings and the preview stay consistent; only the whole loop runs late.

The server returns every hand it sees per view. Like wilor_landmarks.py (one hand,
MediaPipe num_hands=1) this keeps one per view: the hand overlapping the last one
(keeping its left/right label), else the largest, preferring the other view's
handedness; `hand` = right/left keeps only that hand. A view whose hand suddenly
changes size (x max_size_jump) or jumps (> max_shift hand sizes) vs the last good
frame is dropped once (the collapsed-hand failure, see wilor_landmarks.py).
Two hands (hands=2, hand_target_node.py --two-hands): one hand per side and view, split
by the server's left/right label, each side tracked as above. __call__ returns the main
hand (right unless hand="left"); self.other = the other side's (px0, px1) or
(None, None), self.other_fresh = True (it is picked on every answer), like
wilor_landmarks.py with other_every=1.
Not available remotely: YOLO confidence (no conf_start hysteresis; triangulation
gates catch most one-view false positives) and the MANO extras (`last` has R only).

Setup: wilor_server.py running on the GPU machine, the SSH tunnel open
(ssh -N -i ~/.ssh/<key> -p <port> -L 8000:localhost:8000 root@<host>), and
WILOR_TOKEN set to the server's secret (kept out of argv: --record files store it).
Self-test (needs the tunnel + server): .venv/bin/python Computer-Vision/WiLoR-server/remote_landmarks.py [URL]
"""
import http.client
import json
import math
import os
import sys
import threading
import time
from urllib.parse import urlparse

import cv2
import numpy as np


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
    return inter / max(area(a) + area(b) - inter, 1e-9)


def _box(p):
    return p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max()


def _size(b):
    return max(b[2] - b[0], b[3] - b[1], 1e-6)


class Client:
    """One kept-alive HTTP connection to wilor_server.py (one per worker thread)."""

    def __init__(self, url, token, timeout):
        u = urlparse(url)
        self.host, self.port, self.timeout = u.hostname, u.port or 80, timeout
        self.headers = {"X-Token": token}
        self.conn = None

    def request(self, method, path, body=None, headers=None):
        for attempt in (0, 1):  # a kept-alive connection the server closed fails once
            if self.conn is None:
                self.conn = http.client.HTTPConnection(self.host, self.port, timeout=self.timeout)
            try:
                self.conn.request(method, path, body=body, headers=dict(self.headers, **(headers or {})))
                r = self.conn.getresponse()
                data = r.read()
            except (OSError, http.client.HTTPException):
                self.conn.close()
                self.conn = None
                if attempt:
                    raise
                continue
            if r.status == 401:
                raise SystemExit("wilor server rejected the token: set the same WILOR_TOKEN here and on the server")
            if r.status != 200:
                raise OSError(f"wilor server: HTTP {r.status} {data[:200]!r}")
            return json.loads(data)

    def detect(self, jpgs):
        return self.request("POST", "/detect", b"".join(jpgs),
                            {"Content-Type": "application/octet-stream",
                             "X-Sizes": ",".join(str(len(j)) for j in jpgs)})


class StereoLandmarks:
    """Remote WiLoR on both views, one hand. hand: 'any', 'right' or 'left' (the user's hand)."""

    def __init__(self, url="http://localhost:8000", token=None, hand="any", inflight=4, quality=80,
                 timeout=3.0, max_size_jump=1.55, max_shift=0.6, hands=1):
        self.url, self.token = url, token if token is not None else os.environ.get("WILOR_TOKEN", "")
        if not self.token:
            raise SystemExit("wilor-remote: set WILOR_TOKEN to the server's secret")
        self.hand, self.inflight, self.quality, self.timeout = hand, max(1, inflight), quality, timeout
        self.max_size_jump, self.max_shift = max_size_jump, max_shift
        self.track = [None, None]  # per view: {"box", "is_right"} of the hand picked last
        self.good = [None, None]   # per view: last accepted keypoint box
        self.rejected = [0, 0]     # dropped (sudden-change) frames per view
        self.last = [None, None]
        # two hands: per side, per view (same meaning as track / good above)
        self.hands = hands
        self.main = "left" if hand == "left" else "right"
        self.other_side = "left" if self.main == "right" else "right"
        self.side_track = {s: [None, None] for s in ("right", "left")}
        self.side_good = {s: [None, None] for s in ("right", "left")}
        self.other, self.other_fresh = (None, None), False
        self.sync = Client(url, self.token, timeout)
        try:
            info = self.sync.request("GET", "/health")
        except OSError as e:
            raise SystemExit(f"wilor-remote: can't reach {url} ({e}). Is the SSH tunnel open "
                             "(ssh -N -L 8000:localhost:8000 ...) and wilor_server.py running?")
        print(f"wilor-remote: server GPU {info.get('gpu')}, {info.get('precision', '?')}", flush=True)
        self.pending = None  # (f0, f1) -> server views, set by the wrapper for the pair it returns

    def encode(self, frames):
        return [cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, self.quality])[1].tobytes() for f in frames]

    def wrap(self, cam):
        """Pipelined camera: see the module docstring."""
        return PipelinedCam(cam, self)

    def __call__(self, f0, f1, t=None):
        if self.pending is not None and self.pending[0] is f0 and self.pending[1] is f1:
            views = self.pending[2]
        else:  # not wrapped: one blocking round trip
            try:
                views = self.sync.detect(self.encode((f0, f1)))["views"]
            except OSError as e:
                print(f"wilor-remote: {e}", flush=True)
                views = None
        self.pending = None
        return self.pick(views)

    # -- one hand per view -------------------------------------------------------
    def pick(self, views):
        if self.hands == 2:
            return self.pick_two(views)
        self.last = [None, None]
        if views is None:  # request failed: no hand this pair (the node holds the arm)
            self.track = [None, None]
            return None, None
        hands = []
        for v in (0, 1):
            hs = [h for h in views[v]["hands"] if self.hand == "any" or h["is_right"] == (self.hand == "right")]
            for h in hs:
                h["px"] = np.asarray(h["points"], dtype=float)
                h["box"] = _box(h["px"])
            hands.append([h for h in hs if np.isfinite(h["px"]).all()])
        out = [None, None]
        for v in (0, 1):
            tr, other = self.track[v], self.track[1 - v]
            h = self._pick(hands[v], tr, other and other["is_right"])
            if h is None:
                self.track[v] = self.good[v] = None
                continue
            is_right = tr["is_right"] if tr is not None and _iou(h["box"], tr["box"]) > 0.1 else h["is_right"]
            self.track[v] = {"box": h["box"], "is_right": is_right}
            if self._sudden(self.good[v], h["box"]):
                self.good[v] = None  # drop once, accept the next frame (a real fast move)
                self.rejected[v] += 1
                continue
            self.good[v] = h["box"]
            out[v] = h["px"]
            self.last[v] = {"box": h["box"], "is_right": is_right, "R": np.asarray(h["R"], dtype=float)}
        return out[0], out[1]

    def pick_two(self, views):
        """Two hands: one per side (the server's label) and view. Returns the main side's."""
        self.last = [None, None]
        self.other_fresh = True
        out = {"right": [None, None], "left": [None, None]}
        if views is None:
            for side in out:
                self.side_track[side] = [None, None]
                self.side_good[side] = [None, None]
            self.other = (None, None)
            return None, None
        for v in (0, 1):
            hs = []
            for h in views[v]["hands"]:
                h["px"] = np.asarray(h["points"], dtype=float)
                if np.isfinite(h["px"]).all():
                    h["box"] = _box(h["px"])
                    hs.append(h)
            for side in ("right", "left"):
                cand = [h for h in hs if h["is_right"] == (side == "right")]
                h = self._pick(cand, self.side_track[side][v], None)
                if h is None:
                    self.side_track[side][v] = self.side_good[side][v] = None
                    continue
                self.side_track[side][v] = {"box": h["box"], "is_right": side == "right"}
                if self._sudden(self.side_good[side][v], h["box"]):
                    self.side_good[side][v] = None
                    self.rejected[v] += 1
                    continue
                self.side_good[side][v] = h["box"]
                out[side][v] = h["px"]
                if side == self.main:
                    self.last[v] = {"box": h["box"], "is_right": side == "right",
                                    "R": np.asarray(h["R"], dtype=float)}
        self.other = tuple(out[self.other_side])
        return out[self.main][0], out[self.main][1]

    @staticmethod
    def _pick(hands, track, prefer_right):
        if not hands:
            return None
        if track is not None:
            best = max(hands, key=lambda h: _iou(h["box"], track["box"]))
            if _iou(best["box"], track["box"]) > 0.1:
                return best
        if prefer_right is not None and any(h["is_right"] == prefer_right for h in hands):
            hands = [h for h in hands if h["is_right"] == prefer_right]
        return max(hands, key=lambda h: (h["box"][2] - h["box"][0]) * (h["box"][3] - h["box"][1]))

    def _sudden(self, ref, box):
        if ref is None:
            return False
        r = _size(box) / _size(ref)
        shift = math.hypot((box[0] + box[2] - ref[0] - ref[2]) / 2, (box[1] + box[3] - ref[1] - ref[3]) / 2)
        return not (1 / self.max_size_jump < r < self.max_size_jump) or shift > self.max_shift * _size(ref)

    def close(self):
        pass


class PipelinedCam:
    """StereoCam / FrameReplay in front of the remote model. pair() grabs the next camera
    pair, queues it for the server, and returns the oldest pair whose answer is back
    (or None). t0/t1 are those of the returned pair. Live cameras: the newest pair
    replaces any not yet sent, and a late answer older than one already returned is
    dropped, so lag doesn't pile up. Replays (FrameReplay): every pair is processed,
    in order, and the replay waits for the server."""

    def __init__(self, cam, lms):
        self.cam, self.lms = cam, lms
        self.replay = hasattr(cam, "recalibrate_now")
        self.cond = threading.Condition()
        self.queue = []      # [seq, ...] waiting for a worker (live: at most one)
        self.items = {}      # seq -> {"f0", "f1", "t", "t0", "t1", "jpgs", "recal", "views", "done"}
        self.seq = 0         # last submitted
        self.next = 1        # next seq to hand out
        self.running = True
        self.ended = None    # ReplayEnd from the replay, raised once everything is out
        self.t0 = self.t1 = 0.0
        self._recal = False
        self.rtt, self.gpu, self.errors, self.dropped, self.stat_t = [], [], 0, 0, time.monotonic()
        self.workers = [threading.Thread(target=self._work, daemon=True) for _ in range(lms.inflight)]
        for w in self.workers:
            w.start()

    def __getattr__(self, name):  # t, skipped, argv, ... of the wrapped camera
        return getattr(self.cam, name)

    def _outstanding(self):
        return self.seq - self.next + 1

    def pair(self):
        # live: one new camera pair per call; replay: fill every free slot before waiting
        while self.ended is None:
            try:
                pr = self.cam.pair()
            except Exception as e:  # ReplayEnd: finish what's in flight first
                if self._outstanding() == 0:
                    raise
                self.ended, pr = e, None
            if pr is not None:
                self._submit(pr)
            if not self.replay or self._outstanding() >= self.lms.inflight:
                break
        if self.ended is not None and self._outstanding() == 0:
            raise self.ended
        with self.cond:
            if self.replay or self.ended is not None:  # in order, waiting for the next one
                self.cond.wait_for(lambda: self.items.get(self.next, {}).get("done"), timeout=0.5)
            done = sorted(s for s, it in self.items.items() if it["done"] and s >= self.next)
            if not done or (self.replay and done[0] != self.next):
                return None
            s = done[0]
            for old in [k for k in self.items if k < s]:  # live: skip answers that are still out
                self.items.pop(old)
                if old in self.queue:
                    self.queue.remove(old)
                self.dropped += 1
            it = self.items.pop(s)
            self.next = s + 1
        self._stats()
        self.t0, self.t1 = it["t0"], it["t1"]
        self._recal = self._recal or it["recal"]
        self.lms.pending = (it["f0"], it["f1"], it["views"])
        return it["f0"], it["f1"], it["t"]

    def _submit(self, pr):
        f0, f1, t = pr
        # encode here, not in the workers: StereoCam can hand out the same cam1 frame twice,
        # and the node draws on frames it got back
        jpgs = self.lms.encode((f0, f1))
        with self.cond:
            self.seq += 1
            self.items[self.seq] = {"f0": f0, "f1": f1, "t": t, "t0": self.cam.t0, "t1": self.cam.t1,
                                    "jpgs": jpgs, "views": None, "done": False,
                                    "recal": self.replay and self.cam.recalibrate_now()}
            if not self.replay:
                for old in self.queue:  # never sent: replaced by this newer pair
                    self.items.pop(old, None)
                    self.dropped += 1
                self.queue.clear()
                self.next = min(self.items) if self.items else self.seq
            self.queue.append(self.seq)
            self.cond.notify_all()

    def _work(self):
        client = Client(self.lms.url, self.lms.token, self.lms.timeout)
        while True:
            with self.cond:
                self.cond.wait_for(lambda: self.queue or not self.running)
                if not self.running:
                    return
                s = self.queue.pop(0)
                jpgs = self.items[s]["jpgs"]
            t = time.perf_counter()
            try:
                out = client.detect(jpgs)
                views, rtt = out["views"], (time.perf_counter() - t) * 1000
            except OSError as e:
                out, views, rtt = None, None, None
                err = str(e)
            with self.cond:
                if out is None:
                    self.errors += 1
                    if self.errors == 1 or self.errors % 50 == 0:
                        print(f"wilor-remote: request failed ({err}); {self.errors} so far", flush=True)
                else:
                    self.rtt.append(rtt)
                    self.gpu.append(out.get("infer_ms", 0.0))
                it = self.items.get(s)
                if it is not None:  # None: skipped meanwhile
                    it["views"], it["done"] = views, True
                    it["jpgs"] = None
                self.cond.notify_all()
            if out is None:
                time.sleep(0.2)  # server / tunnel down: back off, the camera keeps running

    def _stats(self):
        now = time.monotonic()
        if now - self.stat_t < 5.0 or not self.rtt:
            return
        rtt, gpu = np.median(self.rtt), np.median(self.gpu)
        print(f"wilor-remote: round trip {rtt:.0f} ms median (GPU {gpu:.0f} ms), {len(self.rtt) / (now - self.stat_t):.1f} "
              f"answers/s, {self.dropped} pairs dropped, {self.errors} errors, {sum(self.lms.rejected)} sudden-change "
              "views dropped", flush=True)
        self.rtt, self.gpu, self.dropped, self.stat_t = [], [], 0, now

    def recalibrate_now(self):
        r, self._recal = self._recal, False
        return r

    def close(self):
        with self.cond:
            self.running = False
            self.cond.notify_all()
        self.cam.close()


def _selftest():
    """Against the live server: demo images through the blocking call and the pipeline."""
    import glob
    here = os.path.dirname(os.path.abspath(__file__))
    imgs = []
    for path in sorted(glob.glob(os.path.join(here, "..", "WiLoR", "repo", "demo_img", "*"))):
        img = cv2.imread(path)
        if img is not None:
            imgs.append(cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1]))))
    if not imgs:  # no WiLoR repo here: a synthetic frame only checks the round trip
        imgs = [np.full((480, 640, 3), 128, np.uint8)]
    url = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8000"
    lms = StereoLandmarks(url=url, inflight=4)
    t = time.perf_counter()
    p0, p1 = lms(imgs[0], imgs[0])
    print(f"blocking call: {(time.perf_counter() - t) * 1000:.0f} ms, hand: {p0 is not None}")
    if p0 is not None:
        # fp16 on the server: the same image in two batch slots differs by up to ~3 px (measured 2.7)
        assert p0.shape == (21, 2) and np.allclose(p0, p1, atol=4.0), "same image in both views must agree"

    class FakeCam:  # a replay of the demo images, 30 times over
        def __init__(self):
            self.i, self.t0, self.t1 = -1, 0.0, 0.0

        def pair(self):
            self.i += 1
            if self.i >= 30 * len(imgs):
                raise EOFError
            self.t0 = self.t1 = self.i / 30
            f = imgs[self.i % len(imgs)]
            return f, f, self.t0

        def recalibrate_now(self):
            return False

        def close(self):
            pass

    cam = lms.wrap(FakeCam())
    n, last_t, t = 0, -1.0, time.perf_counter()
    try:
        while True:
            pr = cam.pair()
            if pr is None:
                continue
            assert pr[2] > last_t and cam.t0 == pr[2], "pairs must come back in order with their own times"
            last_t = pr[2]
            lms(*pr)
            n += 1
    except EOFError:
        pass
    dt = time.perf_counter() - t
    cam.close()
    assert n == 30 * len(imgs), f"replay must return every pair ({n})"
    print(f"pipelined: {n} pairs in {dt:.1f} s = {n / dt:.1f} pairs/s with {lms.inflight} in flight")
    print("remote_landmarks self-test OK")


if __name__ == "__main__":
    _selftest()
