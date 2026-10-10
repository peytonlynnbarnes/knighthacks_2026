"""WiLoR hand-tracking server: runs on the cloud GPU; wilor_client.py runs on the laptop.

The laptop POSTs JPEG frames, this runs WiLoR (wilor_pub.WiLoRTracker) on them and
answers with JSON. It listens on localhost only: the laptop reaches it through an
SSH tunnel, which encrypts the traffic and needs no open ports or certificates.

Cloud setup (once): bash Computer-Vision/cloud_setup.sh
Run (on the cloud machine, from Computer-Vision/):
    export WILOR_TOKEN=<secret>        # the same secret goes on the laptop
    ~/wilor-venv/bin/python wilor_server.py          # 127.0.0.1:8000
Laptop:
    ssh -N -L 8000:localhost:8000 <user>@<cloud-ip>  # keep this open
    WILOR_TOKEN=<secret> python3 Computer-Vision/wilor_client.py

API (header X-Token: <secret> on every request):
    GET  /health   -> {"ok": true, "gpu": "...", "warmup_ms": ...}
    POST /detect   body = 1+ JPEG frames back to back, header X-Sizes: "n0,n1,..."
                   (byte length of each; omit for a single frame)
                   -> {"views": [{"hands": [hand, ...]}, ...], "infer_ms": ...}
    hand = {"is_right", "points" (21 [x, y] pixels, MediaPipe order), "palm" [x, y],
            "R" (3x3 hand frame in camera axes, see wilor_pub.py), "rpy_deg"}

Hands are tracked between requests per frame position (camera 0, camera 1, ...),
so this serves one client at a time; the first start also tunes the GPU's matrix
kernels (TunableOp, saved in ~/.cache/wilor/), later starts reuse that.
"""
import argparse
import os
import secrets
import sys
import threading
import time

import cv2
import numpy as np
import uvicorn
from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.concurrency import run_in_threadpool

from wilor_pub import WiLoRTracker


def hand_json(hand):
    return {"is_right": hand.is_right,
            "points": np.round(hand.points, 1).tolist(),
            "palm": list(hand.palm_center),
            "R": np.round(hand.R, 5).tolist(),
            "rpy_deg": [round(a, 1) for a in hand.rpy()]}


def create_app(token, precision="fp16", tunable=True):
    import torch
    tracker = WiLoRTracker(precision=precision, tunable=tunable)
    gpu = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU (no GPU found)"
    warmup_ms = tracker.warm_up(max_crops=4, max_views=2)  # 2 cameras x 2 hands
    lock = threading.Lock()  # one model on one GPU: one request at a time
    app = FastAPI()

    def check(x_token):
        if not secrets.compare_digest(x_token.encode(), token.encode()):
            raise HTTPException(401, "wrong X-Token (WILOR_TOKEN differs between laptop and server)")

    @app.get("/health")
    def health(x_token: str = Header("")):
        check(x_token)
        return {"ok": True, "gpu": gpu, "warmup_ms": round(warmup_ms),
                "precision": tracker.precision, "tunableop": tracker.tunable}

    @app.post("/detect")
    async def detect(request: Request, x_token: str = Header(""), x_sizes: str = Header("")):
        check(x_token)
        body = await request.body()
        try:
            sizes = [int(s) for s in x_sizes.split(",")] if x_sizes else [len(body)]
        except ValueError:
            raise HTTPException(400, "X-Sizes must be comma-separated byte counts")
        if sum(sizes) != len(body) or min(sizes) <= 0:
            raise HTTPException(400, f"X-Sizes adds up to {sum(sizes)} bytes, body has {len(body)}")
        frames, offset = [], 0
        for n in sizes:
            frame = cv2.imdecode(np.frombuffer(body, np.uint8, n, offset), cv2.IMREAD_COLOR)
            if frame is None:
                raise HTTPException(400, f"frame {len(frames)} is not a valid JPEG")
            frames.append(frame)
            offset += n

        def run():
            with lock:  # all views: one detector call (when needed) and one model call
                t = time.perf_counter()
                views = [{"hands": [hand_json(h) for h in hands]} for hands in tracker.process_many(frames)]
                return views, (time.perf_counter() - t) * 1000

        views, infer_ms = await run_in_threadpool(run)  # keeps /health responsive meanwhile
        return {"views": views, "infer_ms": round(infer_ms, 1)}

    return app


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1",
                    help="listen address (keep 127.0.0.1 and use the SSH tunnel)")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--precision", default="fp16", choices=["fp16", "bf16-backbone", "bf16"],
                    help="GPU number format; run wilor_bench.py once to find the fastest on this GPU")
    ap.add_argument("--no-tunableop", action="store_true", help="don't tune matrix kernels")
    args = ap.parse_args()
    token = os.environ.get("WILOR_TOKEN", "")
    if not token:
        sys.exit("set WILOR_TOKEN to a secret first (the laptop needs the same value)")
    print(f"loading WiLoR ({args.precision}, TunableOp {'off' if args.no_tunableop else 'on'}) "
          "and warming up the GPU ...", flush=True)
    app = create_app(token, args.precision, not args.no_tunableop)
    print(f"WiLoR server ready on http://{args.host}:{args.port}  (Ctrl-C stops)", flush=True)
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()