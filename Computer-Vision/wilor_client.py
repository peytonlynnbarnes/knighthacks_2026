"""Laptop side of cloud WiLoR: camera(s) -> wilor_server.py on the cloud GPU -> hands on screen.

Like wilor_pub.py, but the model runs on the server. Needs only OpenCV, numpy and
requests on the laptop (no PyTorch). Shows the 21 keypoints, the red palm dot and
the orientation axes, and prints palm, fingertips, roll/pitch/yaw and latency.

Run:
    ssh -N -L 8000:localhost:8000 <user>@<cloud-ip>   # another terminal; keep it open
    export WILOR_TOKEN=<secret>                       # same as on the server
    python3 Computer-Vision/wilor_client.py                     # camera 0
    python3 Computer-Vision/wilor_client.py --cams 0 2          # both stereo cameras
    python3 Computer-Vision/wilor_client.py --image hand.jpg --count 20 --no-view
                                                      # connection test, no camera
q quits. The video is shown with the result it belongs to, so it lags the live
camera by one round trip.

The status line shows the round trip (and the server's share of it) and the
results per second over the last 2 seconds. 4 requests are kept in flight
(--inflight): if the round trip climbs well above its usual value, the upload is
full; then use --inflight 3 or --quality 60.
"""
import argparse
import os
import sys
import threading
import time
from collections import deque

import cv2
import numpy as np
import requests

from stereo_cam import HAND_CONNECTIONS

FINGERTIPS = {"thumb": 4, "index": 8, "middle": 12, "ring": 16, "pinky": 20}
RATE_WINDOW_S = 2.0  # results/s is measured over this many recent seconds


class RemoteHand:
    """One hand from the server's JSON."""

    def __init__(self, d):
        self.is_right = d["is_right"]
        self.points = np.asarray(d["points"], dtype=float)  # (21,2) px, MediaPipe order
        self.palm_center = tuple(d["palm"])
        self.R = np.asarray(d["R"], dtype=float)
        self.rpy = d["rpy_deg"]

    # skeleton, palm dot, and the hand axes (WiLoR's camera is nearly orthographic,
    # so an axis's image direction is just its x/y in camera coordinates)
    def draw(self, frame):
        pts = [tuple(int(v) for v in p) for p in self.points]
        for a, b in HAND_CONNECTIONS:
            cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
        for p in pts:
            cv2.circle(frame, p, 3, (255, 255, 255), -1)
        o = self.points[[0, 5, 9, 13, 17]].mean(axis=0)
        length = 0.6 * np.linalg.norm(self.points[9] - self.points[0])
        for k, color in enumerate(((0, 0, 255), (0, 255, 0), (255, 0, 0))):
            tip = o + length * self.R[:2, k]
            cv2.line(frame, tuple(int(v) for v in o), tuple(int(v) for v in tip), color, 3)
        cv2.circle(frame, self.palm_center, 8, (0, 0, 255), cv2.FILLED)
        cv2.putText(frame, "R" if self.is_right else "L", (pts[0][0] + 10, pts[0][1] + 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    def print_coords(self, label):
        parts = [f"palm={self.palm_center}"]
        parts += [f"{name}={tuple(int(v) for v in self.points[i])}" for name, i in FINGERTIPS.items()]
        parts.append("rpy deg=({:+.0f}, {:+.0f}, {:+.0f})".format(*self.rpy))
        print(f"{label} ({'right' if self.is_right else 'left'}): " + " ".join(parts))


class CloudWiLoR:
    """Sends frames from `inflight` worker threads, always the newest one, and keeps
    only the newest result, so a slow network drops frames instead of piling up lag."""

    def __init__(self, url, token, inflight=4, quality=80, timeout=3.0):
        self.url, self.headers, self.quality, self.timeout = url.rstrip("/"), {"X-Token": token}, quality, timeout
        self.cond = threading.Condition()
        self.pending = None   # (seq, frames) newest frame(s) not sent yet
        self.result = None    # (seq, frames, views, rtt_ms, infer_ms)
        self.error = ""
        self.seq = 0
        self.running = True
        self.threads = [threading.Thread(target=self._work, daemon=True) for _ in range(inflight)]
        for th in self.threads:
            th.start()

    def check(self):
        """Raise SystemExit with a clear message unless the tunnel, server and token are OK."""
        try:
            r = requests.get(self.url + "/health", headers=self.headers, timeout=10)
        except requests.ConnectionError:
            sys.exit(f"can't reach {self.url}. Is the SSH tunnel open "
                     "(ssh -N -L 8000:localhost:8000 <user>@<cloud-ip>) and wilor_server.py running?")
        except requests.Timeout:
            sys.exit(f"{self.url} didn't answer within 10 s (server still loading, or the tunnel is stuck)")
        if r.status_code == 401:
            sys.exit("server rejected the token: set the same WILOR_TOKEN on the laptop and the server")
        r.raise_for_status()
        return r.json()

    def submit(self, frames):
        with self.cond:
            self.seq += 1
            self.pending = (self.seq, [f.copy() for f in frames])  # replaces any older unsent frame
            self.cond.notify()

    def latest(self):
        with self.cond:
            return self.result

    def _work(self):
        session = requests.Session()  # one kept-alive connection per worker
        while self.running:
            with self.cond:
                self.cond.wait_for(lambda: self.pending is not None or not self.running)
                if not self.running:
                    return
                seq, frames = self.pending
                self.pending = None
            jpgs = [cv2.imencode(".jpg", f, [cv2.IMWRITE_JPEG_QUALITY, self.quality])[1].tobytes()
                    for f in frames]
            headers = dict(self.headers, **{"Content-Type": "application/octet-stream",
                                            "X-Sizes": ",".join(str(len(j)) for j in jpgs)})
            t = time.perf_counter()
            try:
                r = session.post(self.url + "/detect", data=b"".join(jpgs), headers=headers,
                                 timeout=self.timeout)
                r.raise_for_status()
                out = r.json()
            except requests.RequestException as e:
                with self.cond:
                    self.error = f"server error: {type(e).__name__}"
                time.sleep(0.5)  # back off; the camera keeps running
                continue
            rtt_ms = (time.perf_counter() - t) * 1000
            with self.cond:
                self.error = ""
                if self.result is None or seq > self.result[0]:
                    self.result = (seq, frames, out["views"], rtt_ms, out["infer_ms"])

    def close(self):
        with self.cond:
            self.running = False
            self.cond.notify_all()


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", default="http://localhost:8000", help="server URL (the tunnel's local end)")
    ap.add_argument("--token", default=os.environ.get("WILOR_TOKEN", ""), help="default: $WILOR_TOKEN")
    ap.add_argument("--cams", type=int, nargs="+", default=[0], help="camera indices (/dev/videoN)")
    ap.add_argument("--image", help="send this image instead of a camera (connection test)")
    ap.add_argument("--inflight", type=int, default=4,
                    help="requests in flight: more = more results/s, same round trip, until the "
                         "upload is full (then the round trip grows: use fewer, or a lower --quality)")
    ap.add_argument("--quality", type=int, default=80, help="JPEG quality sent to the server")
    ap.add_argument("--count", type=int, default=0, help="stop after this many results (0 = run until q)")
    ap.add_argument("--no-view", action="store_true", help="no window, just print")
    args = ap.parse_args()
    if not args.token:
        sys.exit("set WILOR_TOKEN (or --token) to the server's secret")

    cloud = CloudWiLoR(args.url, args.token, args.inflight, args.quality)
    info = cloud.check()
    extra = f" | {info['precision']}, TunableOp {'on' if info['tunableop'] else 'off'}" if "precision" in info else ""
    print(f"connected: server GPU {info['gpu']}{extra}", flush=True)

    caps = []
    if args.image:
        still = cv2.imread(args.image)
        if still is None:
            sys.exit(f"can't read {args.image}")
    else:
        for i in args.cams:
            cap = cv2.VideoCapture(i)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
            if not cap.isOpened():
                sys.exit(f"Cannot open camera {i}")
            caps.append(cap)

    shown, results, last_seq = None, 0, 0
    recent = deque()  # arrival times of results in the last RATE_WINDOW_S seconds
    try:
        while True:
            if args.image:
                frames = [still]
                time.sleep(1 / 30)
            else:
                frames = []
                for cap in caps:
                    ok, f = cap.read()
                    if not ok:
                        sys.exit("camera stopped delivering frames")
                    frames.append(f)
            cloud.submit(frames)

            res = cloud.latest()
            if res is not None and res[0] != last_seq:
                last_seq, results = res[0], results + 1
                seq, sent, views, rtt_ms, infer_ms = res
                shown = [f.copy() for f in sent]
                for v, (frame, view) in enumerate(zip(shown, views)):
                    for i, d in enumerate(view["hands"]):
                        hand = RemoteHand(d)
                        hand.draw(frame)
                        hand.print_coords(f"cam {v} hand {i}")
                now = time.perf_counter()
                recent.append(now)
                while now - recent[0] > RATE_WINDOW_S:
                    recent.popleft()
                rate = (len(recent) - 1) / (now - recent[0]) if len(recent) > 1 else 0.0
                status = f"round trip {rtt_ms:.0f} ms (GPU {infer_ms:.0f} ms)  {rate:.1f} results/s"
                print(status, flush=True)
                for frame in shown:
                    cv2.putText(frame, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
                if args.count and results >= args.count:
                    break

            if not args.no_view:
                view = np.hstack(shown if shown is not None else frames).copy()
                if cloud.error:
                    cv2.putText(view, cloud.error, (10, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                elif shown is None:
                    cv2.putText(view, "waiting for the server ...", (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, (0, 255, 255), 2)
                cv2.imshow("WiLoR (cloud)", view)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            elif cloud.error:
                print(cloud.error, flush=True)
                time.sleep(0.5)
    except KeyboardInterrupt:
        pass
    finally:
        cloud.close()
        for cap in caps:
            cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
