"""WiLoR landmark backend: 21 hand landmarks (pixels) per camera view, from the MANO hand model.

The landmark backend seam of common/ (stereo_cam.make_landmarks), so common/hand3d.StereoHand
and everything after it are backend-independent (see ../WILOR_PLAN.md):

    StereoLandmarks(f0, f1, t) -> (px0, px1), each a (21, 2) array of full-frame
    pixels (distorted image, OpenPose joint order, the same as MediaPipe's) or None.

Per call, for each view:
  1. Hand box: YOLO (repo/pretrained_models/detector.pt, also gives left/right)
     every `yolo_every` calls, when that view has no track, or when the track
     looks wrong; otherwise the box around the last keypoints. A keypoint-
     tracked view whose hand suddenly changes size or jumps (vs the last good
     frame) is dropped for that frame and re-detected on the next: a blurry
     frame once collapsed it into a tiny hand at the wrist, and the box from
     those keypoints kept it there until the next YOLO pass. One hand, like
     num_hands=1: the detection overlapping the tracked hand (kept
     down to conf 0.3, and keeping the track's left/right label), else the
     largest with conf >= 0.5 (both views prefer the same handedness).
  2. Square crop of `rescale` x the box, warped to 256x256 (zero padding at the
     image edge, never squashed). WiLoR only models right hands: left-hand
     crops are mirrored, as upstream does (vitdet_dataset.py).
  3. Both views in one forward pass (fp16 on the GPU: Apple MPS, or CUDA). Keypoints come back in
     crop units (-0.5..0.5 around the center) and map back exactly.

Extras for later phases (MANO rotation, single-view 3D): self.last[view] =
{"box", "is_right", "conf", "global_orient" (3x3, in the crop's camera, mirrored
for left hands), "kp3d" (21x3 m, same frame), "cam_t"} or None.

Speed: on the old RTX 2060 (fp16, bench_wilor.py) 47 ms for both views (~21 Hz) + ~9 ms
for a YOLO pass; on the Mac (MPS) not measured yet: run bench_wilor.py. Needs torch,
ultralytics and repo/'s requirements in the conda env, and repo/ (weights + mano_data,
gitignored). If an op isn't implemented on MPS: PYTORCH_ENABLE_MPS_FALLBACK=1 (that op runs
on the CPU) or fp32=True.

Self-test (no cameras, demo images): python Computer-Vision/WiLoR/wilor_landmarks.py
"""
import math
import os
import sys

import cv2
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.join(HERE, "repo")
CROP = 256
MEAN = np.array([0.485, 0.456, 0.406]) * 255.0
STD = np.array([0.229, 0.224, 0.225]) * 255.0


def best_device():
    """'mps' (Apple GPU), else 'cuda', else 'cpu'."""
    import torch
    if torch.backends.mps.is_available():
        return "mps"
    return "cuda" if torch.cuda.is_available() else "cpu"


def load_models(device=None, fp32=False):
    """(wilor model, YOLO detector, torch device, dtype). Loads from repo/; load_wilor
    hardcodes ./mano_data/, so the cwd is changed only while loading."""
    import torch
    import ultralytics.nn.tasks
    from ultralytics import YOLO
    if REPO not in sys.path:
        sys.path.insert(0, REPO)
    from wilor.models import load_wilor

    dev = torch.device(device or best_device())
    dtype = torch.float32 if (fp32 or dev.type == "cpu") else torch.float16
    if dev.type == "cpu":
        print("WARNING: WiLoR on the CPU (no MPS/CUDA): very slow", flush=True)
    if dev.type == "cuda":
        torch.backends.cudnn.benchmark = True
    torch.serialization.add_safe_globals([ultralytics.nn.tasks.PoseModel])
    cwd = os.getcwd()
    os.chdir(REPO)
    try:
        model, cfg = load_wilor(checkpoint_path="./pretrained_models/wilor_final.ckpt",
                                cfg_path="./pretrained_models/model_config.yaml")
        detector = YOLO("./pretrained_models/detector.pt")
    finally:
        os.chdir(cwd)
    assert cfg.MODEL.IMAGE_SIZE == CROP, cfg.MODEL.IMAGE_SIZE
    if dev.type == "mps":
        # torch 2.5 MPS conv2d fails on non-contiguous inputs ("view size is not compatible");
        # WiLoR's refine_net gets the permuted ViT feature map, so make every conv input contiguous
        for m in model.modules():
            if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d)):
                m.register_forward_pre_hook(lambda _m, args: tuple(a.contiguous() for a in args))
    detector.to(dev)
    return model.to(device=dev, dtype=dtype).eval(), detector, dev, dtype


def square_box(x1, y1, x2, y2, rescale):
    """(cx, cy, size) of the square crop around a box."""
    return (x1 + x2) / 2.0, (y1 + y2) / 2.0, rescale * max(x2 - x1, y2 - y1)


def crop_matrix(cx, cy, size, flip):
    """2x3 affine frame px -> crop px: u = CROP * (s (x - cx) / size + 0.5), s = -1 if flip."""
    k = CROP / size
    s = -1.0 if flip else 1.0
    return np.array([[s * k, 0.0, CROP / 2.0 - s * k * cx],
                     [0.0, k, CROP / 2.0 - k * cy]])


def crop_to_frame(kp, cx, cy, size, flip):
    """WiLoR pred_keypoints_2d (crop units, -0.5..0.5 around the center) -> frame px."""
    s = -1.0 if flip else 1.0
    return np.stack([cx + s * kp[:, 0] * size, cy + kp[:, 1] * size], axis=1)


def _iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = lambda r: (r[2] - r[0]) * (r[3] - r[1])  # noqa: E731
    return inter / max(area(a) + area(b) - inter, 1e-9)


class StereoLandmarks:
    """WiLoR on both views, one hand. hand: 'any', 'right' or 'left' (the user's hand)."""

    def __init__(self, yolo_every=4, conf=0.3, conf_start=0.5, rescale=2.0, hand="any", yolo_size=320,
                 device=None, fp32=False, warmup_size=(640, 480), max_size_jump=1.55, max_shift=0.6):
        import torch
        self.torch = torch
        self.model, self.detector, self.dev, self.dtype = load_models(device, fp32)
        self.mean = torch.tensor(MEAN, device=self.dev, dtype=self.dtype).view(1, 3, 1, 1)
        self.std = torch.tensor(STD, device=self.dev, dtype=self.dtype).view(1, 3, 1, 1)
        # hysteresis: a new track needs conf_start, an existing one is kept down to conf
        # (no hand in view, 2026-10-10: 63 boxes >= 0.3 in 364 pairs, 6 >= 0.5)
        self.yolo_every, self.conf, self.conf_start, self.rescale = yolo_every, conf, conf_start, rescale
        self.hand, self.yolo_size = hand, yolo_size
        self.track = [None, None]  # per view: {"box": (x1 y1 x2 y2), "is_right", "conf"}
        self.last = [None, None]
        self.n = 0
        # sudden-change check of keypoint-tracked views, vs the last good keypoint box. Normal
        # frame to frame (cam_bug recording, 16 Hz, 99%): size x0.81..1.40, center shift < 0.26
        # of the hand size; the collapsed frames were x0.33 / x3.3
        self.max_size_jump, self.max_shift = max_size_jump, max_shift
        self.good = [None, None]       # per view: last accepted keypoint box
        self.redetect = [False, False]  # run YOLO on this view next call
        self.rejected = [0, 0]          # dropped frames per view (diagnostics)
        if warmup_size:
            self.warmup(warmup_size)

    def warmup(self, size):
        """Run every input shape once: cudnn.benchmark tunes kernels on the first call per
        shape, which stalled the first live frames (5.1 s at start, 0.8 + 1.3 s when the hand
        first appeared, 2026-10-10)."""
        import time
        t0 = time.perf_counter()
        torch = self.torch
        blank = np.zeros((size[1], size[0], 3), np.uint8)
        for k in (1, 2):  # YOLO on one or both views
            self.detector([blank] * k, verbose=False, conf=self.conf, imgsz=self.yolo_size, device=self.dev)
        with torch.inference_mode():
            for k in (1, 2):  # WiLoR on the hand in one or both views
                x = torch.zeros((k, 3, CROP, CROP), device=self.dev, dtype=self.dtype)
                self.model({"img": x})["pred_keypoints_2d"].float().cpu()
        print(f"WiLoR warm-up {time.perf_counter() - t0:.1f} s", flush=True)

    # -- detection ---------------------------------------------------------
    def _detect(self, frames):
        """Per frame: list of (box, is_right, conf), YOLO in one batch."""
        # device explicitly: ultralytics picks CUDA or the CPU by itself, never MPS
        res = self.detector(list(frames), verbose=False, conf=self.conf, imgsz=self.yolo_size, device=self.dev)
        out = []
        for r in res:
            b = r.boxes
            xyxy = b.xyxy.cpu().numpy().astype(float)
            cls = b.cls.cpu().numpy().astype(int)
            cf = b.conf.cpu().numpy().astype(float)
            dets = [(tuple(xyxy[i]), bool(cls[i] == 1), float(cf[i])) for i in range(len(cf))]
            if self.hand != "any":
                dets = [d for d in dets if d[1] == (self.hand == "right")]
            out.append(dets)
        return out

    def _pick(self, dets, track, prefer_right=None):
        if not dets:
            return None
        if track is not None:
            best = max(dets, key=lambda d: _iou(d[0], track["box"]))
            if _iou(best[0], track["box"]) > 0.1:  # same hand: keep its label (YOLO's flips ~1%)
                return best[0], track["is_right"], best[2]
        dets = [d for d in dets if d[2] >= self.conf_start]
        if not dets:
            return None
        if prefer_right is not None and any(d[1] == prefer_right for d in dets):
            dets = [d for d in dets if d[1] == prefer_right]
        return max(dets, key=lambda d: (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]))

    def _plausible(self, track, shape):
        """A keypoint-tracked box that shrank to nothing or left the image -> re-detect."""
        x1, y1, x2, y2 = track["box"]
        h, w = shape[:2]
        size = max(x2 - x1, y2 - y1)
        return 15 < size < 0.9 * max(w, h) and x2 > 0 and y2 > 0 and x1 < w and y1 < h

    # -- main --------------------------------------------------------------
    def __call__(self, f0, f1, t=None):
        frames = (f0, f1)
        self.n += 1
        need = [self.track[v] is None or self.n % self.yolo_every == 1 or self.redetect[v]
                or not self._plausible(self.track[v], frames[v].shape) for v in (0, 1)]
        self.redetect = [False, False]
        if any(need):
            dets = self._detect([frames[v] for v in (0, 1) if need[v]])
            it = iter(dets)
            found = [next(it) if need[v] else None for v in (0, 1)]
            # same handedness in both views when the other view already knows it
            for v in (0, 1):
                if need[v]:
                    other = self.track[1 - v]
                    pick = self._pick(found[v], self.track[v], other and other["is_right"])
                    self.track[v] = (None if pick is None else
                                     {"box": pick[0], "is_right": pick[1], "conf": pick[2]})

        views = [v for v in (0, 1) if self.track[v] is not None]
        for v in (0, 1):
            if self.track[v] is None:
                self.good[v] = None
        self.last = [None, None]
        if not views:
            return None, None
        crops, metas = [], []
        for v in views:
            tr = self.track[v]
            cx, cy, size = square_box(*tr["box"], self.rescale)
            flip = not tr["is_right"]
            img = cv2.warpAffine(frames[v], crop_matrix(cx, cy, size, flip), (CROP, CROP),
                                 flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
            crops.append(img[:, :, ::-1])  # BGR -> RGB
            metas.append((cx, cy, size, flip))

        torch = self.torch
        with torch.inference_mode():
            x = torch.from_numpy(np.ascontiguousarray(np.stack(crops))).to(self.dev)
            x = (x.permute(0, 3, 1, 2).to(self.dtype) - self.mean) / self.std
            out = self.model({"img": x})
            kp2 = out["pred_keypoints_2d"].float().cpu().numpy()
            kp3 = out["pred_keypoints_3d"].float().cpu().numpy()
            rot = out["pred_mano_params"]["global_orient"].float().cpu().numpy().reshape(len(views), 3, 3)
            cam_t = out["pred_cam_t"].float().cpu().numpy()

        px = [None, None]
        for i, v in enumerate(views):
            cx, cy, size, flip = metas[i]
            p = crop_to_frame(kp2[i], cx, cy, size, flip)
            if not np.all(np.isfinite(p)):
                self.track[v] = None
                continue
            tr = self.track[v]
            box = (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())
            if not need[v] and self._sudden(self.good[v], box):
                # keep the last good box, drop this view's keypoints, re-detect next call
                tr["box"] = self.good[v]
                self.redetect[v] = True
                self.rejected[v] += 1
                continue
            self.good[v] = box
            px[v] = p
            self.last[v] = {"box": tr["box"], "is_right": tr["is_right"], "conf": tr["conf"],
                            "global_orient": rot[i], "kp3d": kp3[i], "cam_t": cam_t[i]}
            # next call's box: around these keypoints (YOLO checks it every yolo_every calls)
            tr["box"] = box
        return px[0], px[1]

    def _sudden(self, ref, box):
        """Keypoint box changed size or moved too much since the last good frame."""
        if ref is None:
            return False
        size = lambda b: max(b[2] - b[0], b[3] - b[1], 1e-6)  # noqa: E731
        r = size(box) / size(ref)
        shift = math.hypot((box[0] + box[2] - ref[0] - ref[2]) / 2, (box[1] + box[3] - ref[1] - ref[3]) / 2)
        return not (1 / self.max_size_jump < r < self.max_size_jump) or shift > self.max_shift * size(ref)

    def close(self):
        pass


def _selftest():
    import glob
    lms = StereoLandmarks()
    imgs = sorted(glob.glob(os.path.join(REPO, "demo_img", "*")))
    ok = 0
    for path in imgs:
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (640, int(640 * img.shape[0] / img.shape[1])))
        h, w = img.shape[:2]
        lms.track = [None, None]
        p0, p1 = lms(img, img)
        if p0 is None:
            print(f"{os.path.basename(path)}: no hand")
            continue
        assert p0.shape == (21, 2) and np.allclose(p0, p1), "same image in both views must agree"
        inside = np.mean((p0[:, 0] >= -5) & (p0[:, 0] < w + 5) & (p0[:, 1] >= -5) & (p0[:, 1] < h + 5))
        # mirrored image: the detector should call it the other hand, and the
        # keypoints should come back mirrored (checks the left-hand flip path)
        # seed the mirrored run with the mirrored box, so it picks the same hand
        # when there are several (test4)
        x1, y1, x2, y2 = lms.last[0]["box"]
        seed = {"box": (w - 1 - x2, y1, w - 1 - x1, y2), "is_right": not lms.last[0]["is_right"], "conf": 1.0}
        lms.track, lms.n = [dict(seed), dict(seed)], 0  # n = 0: the detector runs on the next call
        m0, _ = lms(img[:, ::-1].copy(), img[:, ::-1].copy())
        side = "right" if lms.last[0]["is_right"] else "left"
        if m0 is None:
            print(f"{os.path.basename(path)}: {side}, mirrored: no hand")
            continue
        m0[:, 0] = (w - 1) - m0[:, 0]
        size = max(np.ptp(p0[:, 0]), np.ptp(p0[:, 1]))
        err = np.median(np.linalg.norm(m0 - p0, axis=1))
        print(f"{os.path.basename(path)}: hand {size:.0f} px, inside {inside:.0%}, "
              f"mirror check {err:.1f} px median ({100 * err / size:.1f}% of hand), mirrored is {side}")
        assert inside > 0.9 and err < 0.1 * size, "keypoints outside the image or flip broken"
        ok += 1
    assert ok >= 3, f"only {ok} demo images gave a hand"
    print(f"wilor_landmarks self-test OK ({ok} images)")


if __name__ == "__main__":
    _selftest()
