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

Two hands (hands=2, ../TWO_HAND_PLAN.md): tracks per (side, view). Existing tracks
re-match by box overlap (highest first, so two hands never share a detection) and keep
their side; new ones take the detector's class (conf >= conf_start); one new hand
labelled differently in the two views is one hand (the more confident label wins).
__call__ still returns the main hand (`hand`: right unless "left"); the other hand is
self.other = (px0, px1) or (None, None), updated every `other_every` calls
(self.other_fresh says whether it ran this call: not run != lost). Default every call:
on MPS, alternating batch sizes (2 / 4) costs ~15-25 ms per switch, so every 2nd call
gave 8.4 pairs/s vs 8.0 (M5 Pro, 1280x720); steady batches: 2 = 75, 3 = 94, 4 = 121 ms.

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
                 device=None, fp32=False, warmup_size=(640, 480), max_size_jump=1.55, max_shift=0.6,
                 hands=1, other_every=1):
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
        # two hands: per side, per view (same meaning as track / good / redetect above)
        self.hands, self.other_every = hands, max(1, int(other_every))
        self.main = "left" if hand == "left" else "right"
        self.other_side = "left" if self.main == "right" else "right"
        self.tr2 = {s: [None, None] for s in ("right", "left")}
        self.good2 = {s: [None, None] for s in ("right", "left")}
        self.redetect2 = {s: [False, False] for s in ("right", "left")}
        self.last2 = {s: [None, None] for s in ("right", "left")}
        self.other, self.other_fresh = (None, None), False
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
            for k in range(1, 2 * self.hands + 1):  # WiLoR on 1..2 hands x 1..2 views
                x = torch.zeros((k, 3, CROP, CROP), device=self.dev, dtype=self.dtype)
                self.model({"img": x})["pred_keypoints_2d"].float().cpu()
        print(f"WiLoR warm-up {time.perf_counter() - t0:.1f} s", flush=True)

    # -- detection ---------------------------------------------------------
    def _detect(self, frames, hand=None):
        """Per frame: list of (box, is_right, conf), YOLO in one batch. hand: filter
        ("any" / "right" / "left"), default self.hand."""
        hand = self.hand if hand is None else hand
        # device explicitly: ultralytics picks CUDA or the CPU by itself, never MPS
        res = self.detector(list(frames), verbose=False, conf=self.conf, imgsz=self.yolo_size, device=self.dev)
        out = []
        for r in res:
            b = r.boxes
            xyxy = b.xyxy.cpu().numpy().astype(float)
            cls = b.cls.cpu().numpy().astype(int)
            cf = b.conf.cpu().numpy().astype(float)
            dets = [(tuple(xyxy[i]), bool(cls[i] == 1), float(cf[i])) for i in range(len(cf))]
            if hand != "any":
                dets = [d for d in dets if d[1] == (hand == "right")]
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
        if self.hands == 2:
            return self._call2((f0, f1))
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

    # -- two hands -----------------------------------------------------------
    def _assign2(self, dets, v, fresh_sides):
        """Detections of view v -> self.tr2[side][v]. Existing tracks first, by overlap
        (highest IoU first, each detection used once); then new tracks by class."""
        pairs = sorted(((_iou(d[0], self.tr2[s][v]["box"]), s, i) for s in ("right", "left")
                        if self.tr2[s][v] is not None for i, d in enumerate(dets)), reverse=True)
        used, matched = set(), set()
        for iou, s, i in pairs:
            if iou <= 0.1 or s in matched or i in used:
                continue
            tr = self.tr2[s][v]
            tr["box"], tr["conf"] = dets[i][0], dets[i][2]  # keeps its side (class flips ~1 %)
            used.add(i)
            matched.add(s)
        for s in ("right", "left"):
            if s in matched:
                continue
            cands = [d for i, d in enumerate(dets) if i not in used and d[1] == (s == "right")
                     and d[2] >= self.conf_start]
            if not cands:
                self.tr2[s][v] = None
                continue
            d = max(cands, key=lambda d: (d[0][2] - d[0][0]) * (d[0][3] - d[0][1]))
            used.add(next(i for i, e in enumerate(dets) if e is d))
            self.tr2[s][v] = {"box": d[0], "is_right": s == "right", "conf": d[2]}
            fresh_sides[v].add(s)

    def _agree2(self, fresh_sides):
        """One new hand labelled right in one view and left in the other is one hand:
        the more confident label wins (the other view's track moves over)."""
        for a in (0, 1):
            b = 1 - a
            for s in ("right", "left"):
                o = "left" if s == "right" else "right"
                if (s in fresh_sides[a] and o in fresh_sides[b] and self.tr2[s][b] is None
                        and self.tr2[o][a] is None):
                    keep, move = (s, o) if self.tr2[s][a]["conf"] >= self.tr2[o][b]["conf"] else (o, s)
                    src = b if keep == s else a
                    tr = self.tr2[move][src]
                    tr["is_right"] = keep == "right"
                    self.tr2[keep][src], self.tr2[move][src] = tr, None
                    return

    def _call2(self, frames):
        self.n += 1
        sides = ("right", "left")
        # YOLO on a view when its main-hand track is missing or suspect, or a tracked hand
        # needs it; a missing other hand waits for the regular pass (else one hand in view
        # would cost a YOLO pass on every call)
        need = [self.tr2[self.main][v] is None or self.n % self.yolo_every == 1
                or any(self.tr2[s][v] is not None and (self.redetect2[s][v]
                       or not self._plausible(self.tr2[s][v], frames[v].shape)) for s in sides)
                for v in (0, 1)]
        for s in sides:
            self.redetect2[s] = [False, False]
        if any(need):
            dets = self._detect([frames[v] for v in (0, 1) if need[v]], hand="any")
            it = iter(dets)
            fresh_sides = [set(), set()]
            for v in (0, 1):
                if need[v]:
                    self._assign2(next(it), v, fresh_sides)
            self._agree2(fresh_sides)
        run_other = self.n % self.other_every == 0 or all(o is None for o in self.other)
        jobs = [(s, v) for s in sides for v in (0, 1) if self.tr2[s][v] is not None
                and (s == self.main or run_other)]
        for s in sides:
            for v in (0, 1):
                if self.tr2[s][v] is None:
                    self.good2[s][v] = None
                if s == self.main or run_other:
                    self.last2[s][v] = None
        px = {s: [None, None] for s in sides}
        if jobs:
            crops, metas = [], []
            for s, v in jobs:
                cx, cy, size = square_box(*self.tr2[s][v]["box"], self.rescale)
                flip = s == "left"
                img = cv2.warpAffine(frames[v], crop_matrix(cx, cy, size, flip), (CROP, CROP),
                                     flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
                crops.append(img[:, :, ::-1])
                metas.append((cx, cy, size, flip))
            torch = self.torch
            with torch.inference_mode():
                x = torch.from_numpy(np.ascontiguousarray(np.stack(crops))).to(self.dev)
                x = (x.permute(0, 3, 1, 2).to(self.dtype) - self.mean) / self.std
                out = self.model({"img": x})
                kp2 = out["pred_keypoints_2d"].float().cpu().numpy()
                kp3 = out["pred_keypoints_3d"].float().cpu().numpy()
                rot = out["pred_mano_params"]["global_orient"].float().cpu().numpy().reshape(len(jobs), 3, 3)
                cam_t = out["pred_cam_t"].float().cpu().numpy()
            for i, (s, v) in enumerate(jobs):
                p = crop_to_frame(kp2[i], *metas[i])
                tr = self.tr2[s][v]
                if not np.all(np.isfinite(p)):
                    self.tr2[s][v] = None
                    continue
                box = (p[:, 0].min(), p[:, 1].min(), p[:, 0].max(), p[:, 1].max())
                if not need[v] and self._sudden(self.good2[s][v], box):
                    tr["box"] = self.good2[s][v]
                    self.redetect2[s][v] = True
                    self.rejected[v] += 1
                    continue
                self.good2[s][v] = box
                px[s][v] = p
                self.last2[s][v] = {"box": tr["box"], "is_right": s == "right", "conf": tr["conf"],
                                    "global_orient": rot[i], "kp3d": kp3[i], "cam_t": cam_t[i]}
                tr["box"] = box
        self.last = self.last2[self.main]
        self.track = self.tr2[self.main]
        self.other_fresh = run_other
        if run_other:
            self.other = tuple(px[self.other_side])
        return px[self.main][0], px[self.main][1]

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
    _selftest_two(imgs)


def _selftest_two(imgs):
    """Two hands: a patch holding one demo hand next to its mirror = a right and a left
    hand in one frame (same frame in both views). Each side must come back on its half,
    the left one the mirror of the right, and the left hand only every other_every calls.
    The demo photos have several hands each, so patches around each confident hand are
    tried and kept only if they hold exactly one."""
    lms = StereoLandmarks(hands=2, other_every=2, warmup_size=None)
    tested = 0
    for path in imgs:
        img = cv2.imread(path)
        if img is None:
            continue
        img = cv2.resize(img, (480, int(480 * img.shape[0] / img.shape[1])))
        h, w0 = img.shape[:2]
        for (x1, y1, x2, y2), is_right, conf in lms._detect([img], hand="any")[0]:
            if conf < 0.6:
                continue
            cx, cy, half = (x1 + x2) / 2, (y1 + y2) / 2, 0.9 * max(x2 - x1, y2 - y1)
            patch = np.ascontiguousarray(img[int(max(0, cy - half)):int(min(h, cy + half)),
                                             int(max(0, cx - half)):int(min(w0, cx + half))])
            if min(patch.shape[:2]) < 60:
                continue
            patch = cv2.resize(patch, (2 * patch.shape[1], 2 * patch.shape[0]))  # small hands
            if sum(e[2] >= 0.5 for e in lms._detect([patch], hand="any")[0]) != 1:
                continue
            right = patch if is_right else patch[:, ::-1].copy()  # the right hand goes left
            w = right.shape[1]
            frame = np.hstack([right, right[:, ::-1]])  # right hand in the left half, left in the right
            if sum(e[2] >= 0.5 for e in lms._detect([frame], hand="any")[0]) != 2:
                continue  # part of a neighbour hand at the patch edge became a hand at the seam
            lms.tr2 = {s: [None, None] for s in ("right", "left")}
            lms.good2 = {s: [None, None] for s in ("right", "left")}
            lms.other, lms.n = (None, None), 0
            fresh = []
            for _ in range(4):
                m0, _ = lms(frame, frame)
                fresh.append(lms.other_fresh)
            o0 = lms.other[0]
            name = f"{os.path.basename(path)} hand at ({cx:.0f},{cy:.0f})"
            if m0 is None or o0 is None:
                print(f"  two hands, {name}: right {'ok' if m0 is not None else 'missing'}, "
                      f"left {'ok' if o0 is not None else 'missing'}")
                continue
            # the main (right) hand must be on the half the detector calls right in this
            # frame (its class can differ from the photo's: a test of the tracker, not YOLO)
            fd = [e for e in lms._detect([frame], hand="any")[0] if e[2] >= 0.5]
            r_half = [(e[0][0] + e[0][2]) / 2 < w for e in fd if e[1]]
            m_left, o_left = np.median(m0[:, 0]) < w, np.median(o0[:, 0]) < w
            assert m_left != o_left, f"both hands on one half ({name})"
            assert not r_half or m_left == r_half[0], f"right hand on the wrong half ({name})"
            if m_left:  # make the mirror check independent of which half is which
                mirrored = o0.copy()
            else:
                m0, o0 = o0, m0
                mirrored = o0.copy()
            mirrored[:, 0] = (2 * w - 1) - mirrored[:, 0]
            size = max(np.ptp(m0[:, 0]), np.ptp(m0[:, 1]))
            err = np.median(np.linalg.norm(mirrored - m0, axis=1))
            assert err < 0.1 * size, f"left hand is not the mirror of the right ({err:.1f} px, {name})"
            assert fresh == [True, True, False, True], fresh  # other hand: first call, then every 2nd
            print(f"  two hands, {name}: right + left on their halves, mirror check "
                  f"{err:.1f} px ({100 * err / size:.1f}% of hand)")
            tested += 1
    assert tested >= 3, f"only {tested} demo hands gave two-hand frames"
    print(f"wilor_landmarks two-hand self-test OK ({tested} hands; left updated on calls 1, 2, 4)")


if __name__ == "__main__":
    _selftest()
