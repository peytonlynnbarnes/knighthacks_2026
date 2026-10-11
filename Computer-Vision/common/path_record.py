"""Record and play back an arm path: what hand_target_node.py publishes, with its timing.

The node's three commands go through PathRecorder.live(kind, value) instead of straight
to ROS ("target" = /ball_position [x y z r p y], "roll" = /wrist_roll_cmd, "grip" =
/gripper). Recording stores them as they are sent. Playback sends a take back out at its
own timing on a thread, and live commands are dropped meanwhile. The IK and
feetech_mirror run unchanged, so their speed limits, block detection and table floor
still apply: a played path can't do anything a hand couldn't.

No jumps: playback first glides from the last sent command to the take's first one
(GLIDE_S), and when it ends (or is stopped) live commands are blended back in from where
the take left the arm, over GLIDE_S.

    rec = PathRecorder(send)        # send(kind, value) publishes for real
    rec.live("target", out)         # instead of publishing
    rec.toggle_record(); rec.toggle_play(); rec.status()

Takes are saved to recordings/paths/last.json (and a timestamped copy) and loaded at
start, so the last take survives a restart.

Self-test: python path_record.py
"""
import json
import math
import os
import threading
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PATH_DIR = os.path.join(HERE, "recordings", "paths")
LAST = os.path.join(PATH_DIR, "last.json")
KINDS = ("target", "roll", "grip")
GLIDE_S = 1.5      # to the take's start, and back to the live hand after it
GLIDE_HZ = 30.0    # command rate while gliding to the start
MIN_TAKE_S = 0.5   # shorter takes (or ones with no arm target) are discarded


def _blend(a, b, u):
    """a -> b at u in 0..1 (floats or lists); smoothstep, so the glide starts and ends slowly."""
    u = u * u * (3.0 - 2.0 * u)
    if isinstance(a, list):
        return [x + (y - x) * u for x, y in zip(a, b)]
    return a + (b - a) * u


class PathRecorder:
    def __init__(self, send, path=LAST, clock=time.monotonic, sleep=time.sleep, load=True):
        self.send_raw, self.path, self.clock, self.sleep = send, path, clock, sleep
        self.lock = threading.Lock()
        self.mode = "idle"            # idle | rec | glide | play
        self.take, self.rec_t0 = [], None
        self.last = {}                # kind -> last value sent (live or played)
        self.events, self.play_t0, self.stop_ev, self.thread = [], None, threading.Event(), None
        self.back = None              # (t0, {kind: value the take left}) blend back to live
        self.note = ""                # last message for the status line
        if load:
            self.events = self.load(path)

    # ---- the node's commands --------------------------------------------------------------

    def _send(self, kind, v):
        self.last[kind] = v
        self.send_raw(kind, v)

    def live(self, kind, v):
        """A live command from the hand: sent (and recorded), dropped while a take plays."""
        v = [float(x) for x in v] if np.ndim(v) else float(v)
        with self.lock:
            if self.mode in ("glide", "play"):
                return
            if self.mode == "rec":
                self.take.append((self.clock() - self.rec_t0, kind, v))
            if self.back is not None:  # just after a take: blend from where it left the arm
                u = (self.clock() - self.back[0]) / GLIDE_S
                if u >= 1.0:
                    self.back = None
                elif kind in self.back[1]:
                    v = _blend(self.back[1][kind], v, u)
            self._send(kind, v)

    # ---- record -----------------------------------------------------------------------------

    def toggle_record(self):
        with self.lock:
            mode = self.mode
        if mode == "rec":
            return self.stop_record()
        if mode in ("glide", "play"):
            self.stop_play()
        with self.lock:
            self.mode, self.take, self.rec_t0, self.note = "rec", [], self.clock(), ""
        print("path: recording (r / Record again to stop)", flush=True)
        return True

    def stop_record(self):
        with self.lock:
            if self.mode != "rec":
                return False
            self.mode, take = "idle", self.take
        dur = take[-1][0] if take else 0.0
        if dur < MIN_TAKE_S or not any(k == "target" for _, k, _ in take):
            self.note = f"take too short or no arm target ({dur:.1f} s): kept the previous one"
            print(f"path: {self.note}", flush=True)
            return False
        self.events = take
        try:
            self.save(take, self.path)
            self.note = f"recorded {dur:.1f} s"
        except OSError as e:
            self.note = f"recorded {dur:.1f} s (not saved: {e})"
        print(f"path: {self.note}, {len(take)} commands -> {self.path}", flush=True)
        return True

    # ---- play -------------------------------------------------------------------------------

    def toggle_play(self):
        with self.lock:
            mode = self.mode
        if mode in ("glide", "play"):
            return self.stop_play()
        if mode == "rec":
            self.stop_record()
        if not self.events:
            self.note = "nothing recorded yet"
            print(f"path: {self.note}", flush=True)
            return False
        self.stop_ev.clear()
        with self.lock:
            self.mode, self.back, self.play_t0, self.note = "glide", None, self.clock(), ""
        self.thread = threading.Thread(target=self._play, args=(list(self.events),), name="path.play",
                                       daemon=True)
        self.thread.start()
        print(f"path: playing {self.events[-1][0]:.1f} s (p / Play again to stop)", flush=True)
        return True

    def stop_play(self, wait=True):
        self.stop_ev.set()
        th = self.thread
        if wait and th is not None and th is not threading.current_thread():
            th.join(timeout=2.0)
        return True

    def _play(self, events):
        firsts = {}
        for _, k, v in events:
            firsts.setdefault(k, v)
        try:
            # glide from the last sent command to the take's start
            starts = {k: self.last[k] for k in firsts if k in self.last}
            t0 = self.clock()
            while starts and not self.stop_ev.is_set():
                u = min(1.0, (self.clock() - t0) / GLIDE_S)
                with self.lock:
                    for k, v in firsts.items():
                        self._send(k, _blend(starts[k], v, u) if k in starts else v)
                if u >= 1.0:
                    break
                self.sleep(1.0 / GLIDE_HZ)
            with self.lock:
                self.mode, self.play_t0 = "play", self.clock()
            t0 = self.play_t0
            for t, k, v in events:
                while not self.stop_ev.is_set():
                    wait = t - (self.clock() - t0)
                    if wait <= 0:
                        break
                    self.sleep(min(wait, 0.05))
                if self.stop_ev.is_set():
                    break
                with self.lock:
                    self._send(k, v)
            self.note = "stopped" if self.stop_ev.is_set() else "played"
        finally:
            with self.lock:
                self.mode = "idle"
                self.back = (self.clock(), dict(self.last))
            print(f"path: {self.note}; back to the hand (blending over {GLIDE_S:.1f} s)", flush=True)

    # ---- status / files ---------------------------------------------------------------------

    def status(self):
        """{"mode", "t" (s into the take / recording), "len" (take length), "note"} for the UI."""
        with self.lock:
            mode, now = self.mode, self.clock()
            length = self.events[-1][0] if self.events else 0.0
            t = (now - self.rec_t0 if mode == "rec" else now - self.play_t0 if mode == "play" else 0.0)
        return {"mode": mode, "t": round(t, 2), "len": round(length, 2), "note": self.note}

    def line(self):
        """One status line for the preview, or '' when there's nothing to say."""
        s = self.status()
        if s["mode"] == "rec":
            return f"REC {s['t']:.1f} s  (r / Record to stop)"
        if s["mode"] == "glide":
            return "PLAY: moving to the path's start"
        if s["mode"] == "play":
            return f"PLAY {min(s['t'], s['len']):.1f} / {s['len']:.1f} s  (p / Play to stop)"
        return f"path: {s['note']}" if s["note"] else ""

    @staticmethod
    def save(events, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        data = {"saved_at": time.strftime("%Y-%m-%d %H:%M:%S"), "kinds": list(KINDS),
                "events": [[round(t, 4), k, v] for t, k, v in events]}
        tmp = path + ".tmp"
        with open(tmp, "w") as f:
            json.dump(data, f)
        os.replace(tmp, path)
        if os.path.basename(path) == "last.json":  # keep every take
            stamp = os.path.join(os.path.dirname(path), time.strftime("take_%Y%m%d_%H%M%S.json"))
            with open(stamp, "w") as f:
                json.dump(data, f)

    @staticmethod
    def load(path):
        try:
            with open(path) as f:
                data = json.load(f)
            return [(float(t), k, v) for t, k, v in data["events"] if k in KINDS]
        except (OSError, ValueError, KeyError, TypeError):
            return []


def _selftest():
    import tempfile

    class Clock:  # fake time: sleep advances it
        t = 0.0

        def __call__(self):
            return self.t

        def sleep(self, s):
            self.t += max(s, 1e-4)

    clk = Clock()
    sent = []
    tmp = os.path.join(tempfile.mkdtemp(), "last.json")
    rec = PathRecorder(lambda k, v: sent.append((clk.t, k, v)), path=tmp, clock=clk, sleep=clk.sleep)
    assert rec.events == [] and not rec.toggle_play()  # nothing to play
    # live before recording: sent, not recorded
    rec.live("target", np.zeros(6))
    rec.toggle_record()
    for i in range(40):  # 2 s: x ramps 0 -> 0.1, roll and grip too
        clk.t += 0.05
        rec.live("target", np.array([0.0025 * (i + 1), 0, 0, 0, 0, 0]))
        rec.live("roll", -math.radians(4 * (i + 1)))
        rec.live("grip", (i + 1) / 40)
    assert rec.status()["mode"] == "rec" and abs(rec.status()["t"] - 2.0) < 1e-9
    assert rec.stop_record() and abs(rec.events[-1][0] - 2.0) < 1e-9
    assert PathRecorder.load(tmp) == [(round(t, 4), k, v) for t, k, v in rec.events]
    # the arm is now elsewhere (live), then play: glide there, then the take at its timing
    rec.live("target", np.array([-0.05, 0, 0, 0, 0, 0]))
    rec.live("roll", 0.5)
    rec.live("grip", 0.0)
    n0 = len(sent)
    rec.toggle_play()
    rec.live("target", np.array([9.0, 9, 9, 9, 9, 9]))  # live while playing: dropped
    rec.thread.join(timeout=5)
    out = sent[n0:]
    assert all(v[0] != 9.0 for _, k, v in out if k == "target")
    xs = [v[0] for _, k, v in out if k == "target"]
    steps = np.abs(np.diff(xs))
    assert xs[0] < -0.04 and abs(xs[-1] - 0.1) < 1e-9 and steps.max() < 0.01, (xs[:3], steps.max())
    # the glide takes GLIDE_S (+ one step), then the take keeps its own timing
    t_end = max(t for t, k, _ in out if k == "target")
    assert abs(t_end - (out[0][0] + GLIDE_S + rec.events[-1][0])) < 0.1, t_end
    assert rec.status()["mode"] == "idle"
    # back to live: blended from where the take ended, not a jump to the hand
    clk.t += 0.1
    rec.live("target", np.array([-0.1, 0, 0, 0, 0, 0]))
    assert sent[-1][2][0] > 0.09, sent[-1]
    clk.t += GLIDE_S
    rec.live("target", np.array([-0.1, 0, 0, 0, 0, 0]))
    assert abs(sent[-1][2][0] + 0.1) < 1e-9
    # a new recorder loads the take; a too-short take keeps it
    rec2 = PathRecorder(lambda k, v: None, path=tmp, clock=clk, sleep=clk.sleep)
    assert len(rec2.events) == len(rec.events)
    rec2.toggle_record()
    clk.t += 0.1
    rec2.live("target", np.zeros(6))
    assert not rec2.stop_record() and len(rec2.events) == len(rec.events)
    # stop mid-play (a sleep that also takes real time, so the stop lands mid-take)
    rec2.sleep = lambda s: (time.sleep(0.002), clk.sleep(s))
    rec2.toggle_play()
    time.sleep(0.02)
    rec2.stop_play()
    assert rec2.status()["mode"] == "idle" and rec2.note == "stopped"
    print("path_record self-test OK")


if __name__ == "__main__":
    _selftest()
