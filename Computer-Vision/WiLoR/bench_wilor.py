#!/usr/bin/env python3
"""Phase 0 of ../WILOR_PLAN.md: how fast WiLoR + its YOLO detector run here, no cameras.

Times, per call, what the stereo backend will do every frame: one WiLoR forward
pass on a batch of crops (2 = one hand in both views), and YOLO hand detection
at 320 px on one view (run every few frames). Inputs are the repo's demo images.

    python Computer-Vision/WiLoR/bench_wilor.py [--fp32] [--n 50]
"""
import argparse
import glob
import os
import sys
import time

import cv2
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "repo")
sys.path.insert(0, REPO)


def timed(fn, n, sync):
    for _ in range(5):  # warm-up (cudnn.benchmark picks kernels here)
        fn()
    sync()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        sync()
        ts.append(time.perf_counter() - t0)
    ts = np.array(ts) * 1000
    return f"median {np.median(ts):6.1f} ms  p90 {np.percentile(ts, 90):6.1f} ms  ({1000 / np.median(ts):5.1f} Hz)"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fp32", action="store_true")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--compile", action="store_true", help="torch.compile the model first")
    args = ap.parse_args()

    from wilor_landmarks import CROP, load_models

    # same loader as the node (device choice, dtype, MPS contiguity fix)
    model, detector, dev, dtype = load_models(fp32=args.fp32)
    if args.compile:
        model = torch.compile(model)
    size = CROP
    name = {"cuda": lambda: torch.cuda.get_device_name(), "mps": lambda: "Apple GPU"}.get(dev.type, lambda: "cpu")()
    print(f"device {dev} ({name}), {dtype}, image size {size}")

    imgs = [cv2.imread(p) for p in sorted(glob.glob(os.path.join(REPO, "demo_img", "*")))]
    imgs = [cv2.resize(i, (640, 480)) for i in imgs if i is not None]
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev, dtype=dtype).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=dev, dtype=dtype).view(1, 3, 1, 1)
    sync = {"cuda": torch.cuda.synchronize, "mps": torch.mps.synchronize}.get(dev.type, lambda: None)

    def batch(k):
        crops = [cv2.cvtColor(cv2.resize(imgs[i % len(imgs)], (size, size)), cv2.COLOR_BGR2RGB)
                 for i in range(k)]
        return torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2)

    with torch.inference_mode():
        for k in (1, 2, 4):
            cpu = batch(k)

            def run():
                x = (cpu.to(dev, non_blocking=True).to(dtype) / 255.0 - mean) / std
                out = model({"img": x})
                out["pred_keypoints_2d"].float().cpu()
            print(f"WiLoR batch {k}: {timed(run, args.n, sync)}")
        small = cv2.resize(imgs[0], (320, 240))
        print(f"YOLO 320 px  : "
              f"{timed(lambda: detector(small, verbose=False, conf=0.3, imgsz=320, device=dev), args.n, sync)}")
    if dev.type == "cuda":
        print(f"peak VRAM {torch.cuda.max_memory_allocated() / 2**20:.0f} MiB")
    elif dev.type == "mps":
        print(f"MPS memory {torch.mps.driver_allocated_memory() / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
