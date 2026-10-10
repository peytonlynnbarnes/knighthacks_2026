import math

class OneEuroFilter:
    def __init__(self, t0, x0, min_cutoff=1.0, beta=0.007, d_cutoff=1.0):
        self.min_cutoff = float(min_cutoff)
        self.beta = float(beta)
        self.d_cutoff = float(d_cutoff)
        self.x_prev = np.array(x0, dtype=np.float32)
        self.dx_prev = np.zeros_like(self.x_prev)
        self.t_prev = float(t0)

    def _alpha(self, rate, cutoff):
        tau = 1.0 / (2.0 * math.pi * cutoff)
        te = 1.0 / rate
        return 1.0 / (1.0 + tau / te)

    def filter(self, t, x):
        t = float(t)
        dt = t - self.t_prev
        if dt <= 1e-5:
            return self.x_prev

        rate = 1.0 / dt

        # Estimate derivative (speed) with low-pass filtering
        dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(rate, self.d_cutoff)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev

        # Adaptive cutoff: low cutoff when still, high cutoff when moving fast
        speed = np.linalg.norm(dx_hat, axis=-1, keepdims=True)
        cutoff = self.min_cutoff + self.beta * speed
        alpha = self._alpha(rate, cutoff)

        # Filtered output
        x_hat = alpha * x + (1.0 - alpha) * self.x_prev

        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

import argparse
import os
import subprocess
import sys
import threading
import time
import cv2
import numpy as np
import torch
import ultralytics.nn.tasks
from ultralytics import YOLO
from wilor.models import load_wilor

def activate_macos_window():
    if sys.platform != "darwin":
        return
    cmd = f"""osascript -e 'tell application "System Events" to set frontmost of the first process whose unix id is {os.getpid()} to true'"""
    try:
        subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

parser = argparse.ArgumentParser(description="High-Speed Async Dual-Hand Stereo WiLoR")
parser.add_argument("--cam1", type=int, default=1, help="Primary camera index")
parser.add_argument("--cam2", type=int, default=3, help="Secondary camera index")
parser.add_argument("--calib", type=str, default="stereo_camera_params.npz", help="Calibration file")
parser.add_argument("--yolo_interval", type=int, default=8, help="Run full YOLO check every N worker steps")
args = parser.parse_args()

cv2.namedWindow("Dual-Hand Stereo WiLoR (Async)", cv2.WINDOW_AUTOSIZE)
activate_macos_window()

# 1. Calibration Parameters
if not os.path.exists(args.calib):
    raise FileNotFoundError(f"Missing {args.calib}. Run stereo_calibrate.py first.")

calib = np.load(args.calib)
K1, D1 = calib['K1'], calib['D1']
K2, D2 = calib['K2'], calib['D2']
R, T = calib['R'], calib['T']

P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K2 @ np.hstack((R, T))
M_TO_INCHES = 39.3700787

# 2. Asynchronous Threaded Cameras
class ThreadedCamera:
    def __init__(self, src, target_w=640, target_h=480):
        self.cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
        self.target_w = target_w
        self.target_h = target_h
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if frame.shape[0] != self.target_h or frame.shape[1] != self.target_w:
                    frame = cv2.resize(frame, (self.target_w, self.target_h))
                with self.lock:
                    self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, self.frame

    def release(self):
        self.running = False
        self.cap.release()

cam1 = ThreadedCamera(args.cam1)
cam2 = ThreadedCamera(args.cam2)

# 3. Model Setup in Native FP16 on MPS
torch.serialization.add_safe_globals([ultralytics.nn.tasks.PoseModel])
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Loading WiLoR on compute device: {device}...")

checkpoint_path = './pretrained_models/wilor_final.ckpt'
cfg_path = './pretrained_models/model_config.yaml'
model, model_cfg = load_wilor(checkpoint_path=checkpoint_path, cfg_path=cfg_path)
model = model.to(device).half()
model.eval()

detector = YOLO('./pretrained_models/detector.pt').to(device)
target_size = model_cfg.MODEL.IMAGE_SIZE  # 256

mean_t = torch.tensor([0.485, 0.456, 0.406], device=device, dtype=torch.float16).view(1, 3, 1, 1)
std_t  = torch.tensor([0.229, 0.224, 0.225], device=device, dtype=torch.float16).view(1, 3, 1, 1)

SKELETON_BONES = [
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (0, 9), (9, 10), (10, 11), (11, 12),
    (0, 13), (13, 14), (14, 15), (15, 16),
    (0, 17), (17, 18), (18, 19), (19, 20)
]

def get_square_crop_meta(box, w, h):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    size = int(max(x2 - x1, y2 - y1) * 1.45)
    x1_s = max(0, int(cx - size / 2.0))
    y1_s = max(0, int(cy - size / 2.0))
    x2_s = min(w, int(cx + size / 2.0))
    y2_s = min(h, int(cy + size / 2.0))
    return x1_s, y1_s, x2_s, y2_s

def triangulate_landmarks(pts1, pts2):
    pts1_undist = cv2.undistortPoints(np.expand_dims(pts1, axis=1).astype(np.float32), K1, D1, P=K1)
    pts2_undist = cv2.undistortPoints(np.expand_dims(pts2, axis=1).astype(np.float32), K2, D2, P=K2)
    pts_4d = cv2.triangulatePoints(P1, P2, pts1_undist, pts2_undist)
    pts_3d_m = (pts_4d[:3] / pts_4d[3]).T
    return pts_3d_m * M_TO_INCHES

# Thread-safe result cache
state_lock = threading.Lock()
shared_results = {
    'hands': [],
    'inference_fps': 0.0
}
is_running = True

# 4. Asynchronous High-Speed Inference Worker
def inference_worker():
    global is_running
    infer_history = []
    worker_step = 0

    cached_boxes1 = []
    cached_boxes2 = []

    with torch.inference_mode():
        while is_running:
            t0 = time.perf_counter()
            r1, f1 = cam1.read()
            r2, f2 = cam2.read()
            if not r1 or not r2 or f1 is None or f2 is None:
                time.sleep(0.003)
                continue

            h, w = f1.shape[:2]
            worker_step += 1

            # Run YOLO periodically OR when tracking drops
            need_yolo = (worker_step % args.yolo_interval == 1) or (len(cached_boxes1) == 0) or (len(cached_boxes2) == 0)

            if need_yolo:
                f1_sm = cv2.resize(f1, (320, 240))
                f2_sm = cv2.resize(f2, (320, 240))
                det1 = detector(f1_sm, verbose=False, conf=0.30, imgsz=320)[0].boxes.xyxy.cpu().numpy()
                det2 = detector(f2_sm, verbose=False, conf=0.30, imgsz=320)[0].boxes.xyxy.cpu().numpy()

                cached_boxes1 = sorted((det1 * 2.0)[:2], key=lambda b: (b[0] + b[2]) / 2.0) if len(det1) > 0 else []
                cached_boxes2 = sorted((det2 * 2.0)[:2], key=lambda b: (b[0] + b[2]) / 2.0) if len(det2) > 0 else []

            n_hands = min(len(cached_boxes1), len(cached_boxes2))
            new_hands = []

            if n_hands > 0:
                crops = []
                metas = []
                for i in range(n_hands):
                    x1_a, y1_a, x2_a, y2_a = get_square_crop_meta(cached_boxes1[i], w, h)
                    x1_b, y1_b, x2_b, y2_b = get_square_crop_meta(cached_boxes2[i], w, h)
                    c1 = f1[y1_a:y2_a, x1_a:x2_a]
                    c2 = f2[y1_b:y2_b, x1_b:x2_b]

                    if c1.size > 0 and c2.size > 0:
                        c1_res = cv2.resize(cv2.cvtColor(c1, cv2.COLOR_BGR2RGB), (target_size, target_size))
                        c2_res = cv2.resize(cv2.cvtColor(c2, cv2.COLOR_BGR2RGB), (target_size, target_size))
                        crops.extend([c1_res, c2_res])
                        metas.append(((x1_a, y1_a, x2_a - x1_a, y2_a - y1_a),
                                      (x1_b, y1_b, x2_b - x1_b, y2_b - y1_b)))

                if len(crops) > 0:
                    t_batch = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).to(device, non_blocking=True).half() / 255.0
                    t_norm = (t_batch - mean_t) / std_t
                    out = model({'img': t_norm})

                    if 'pred_keypoints_2d' in out:
                        preds = out['pred_keypoints_2d'].cpu().numpy()

                        def to_frame(pts, cw, ch, xs, ys):
                            if pts.min() < 0:
                                pts = (pts + 0.5) * np.array([cw, ch])
                            elif pts.max() <= 1.0:
                                pts = pts * np.array([cw, ch])
                            else:
                                pts = pts * (np.array([cw, ch]) / float(target_size))
                            return pts + np.array([xs, ys])

                        next_b1 = []
                        next_b2 = []

                        for i in range(len(metas)):
                            meta1, meta2 = metas[i]
                            pts1 = to_frame(preds[2 * i], meta1[2], meta1[3], meta1[0], meta1[1])
                            pts2 = to_frame(preds[2 * i + 1], meta2[2], meta2[3], meta2[0], meta2[1])

                            # Update tracked bounding boxes from keypoints (bypasses YOLO)
                            next_b1.append([pts1[:, 0].min(), pts1[:, 1].min(), pts1[:, 0].max(), pts1[:, 1].max()])
                            next_b2.append([pts2[:, 0].min(), pts2[:, 1].min(), pts2[:, 0].max(), pts2[:, 1].max()])

                            joints_3d = triangulate_landmarks(pts1, pts2)
                            new_hands.append({'pts1': pts1, 'pts2': pts2, 'joints_3d': joints_3d})

                        # Update box cache for next iteration
                        cached_boxes1 = next_b1
                        cached_boxes2 = next_b2
            else:
                cached_boxes1 = []
                cached_boxes2 = []

            dt = time.perf_counter() - t0
            fps_inf = 1.0 / dt if dt > 0 else 0
            infer_history.append(fps_inf)
            if len(infer_history) > 10:
                infer_history.pop(0)

            with state_lock:
                shared_results['hands'] = new_hands
                shared_results['inference_fps'] = sum(infer_history) / len(infer_history)

def filter(self, t, x):
    t = float(t)
    dt = t - self.t_prev
    if dt <= 1e-5:
        return self.x_prev
    if dt > 0.2:  # Tracking was lost or hand reappeared: reset filter state
        self.x_prev = np.array(x, dtype=np.float32)
        self.dx_prev = np.zeros_like(self.x_prev)
        self.t_prev = t
        return self.x_prev

threading.Thread(target=inference_worker, daemon=True).start()

# 5. UI Render Loop
print("Stereo Hand Tracker active. Running at full display speed.")
ui_fps_history = []

# PERSISTENT FILTER INSTANCES (Must stay OUTSIDE the while loop)
filters_2d_c1 = {}
filters_2d_c2 = {}
filters_3d = {}

while True:
    t_ui_start = time.perf_counter()
    r1, f1 = cam1.read()
    r2, f2 = cam2.read()
    if not r1 or not r2 or f1 is None or f2 is None:
        continue

    h, w = f1.shape[:2]

    with state_lock:
        active_hands = list(shared_results['hands'])
        inf_fps = shared_results['inference_fps']

    t_now = time.perf_counter()

    # Draw hand skeletons
    for i, hand in enumerate(active_hands):
        raw_p1 = hand['pts1']
        raw_p2 = hand['pts2']
        raw_j3d = hand['joints_3d']

        # Initialize filter once per hand ID
        if i not in filters_3d:
            filters_2d_c1[i] = OneEuroFilter(t_now, raw_p1, min_cutoff=1.2, beta=0.01)
            filters_2d_c2[i] = OneEuroFilter(t_now, raw_p2, min_cutoff=1.2, beta=0.01)
            filters_3d[i]    = OneEuroFilter(t_now, raw_j3d, min_cutoff=0.6, beta=0.004)

        # Smooth 2D projections onto video
        pts1_smooth = filters_2d_c1[i].filter(t_now, raw_p1).astype(int)
        pts2_smooth = filters_2d_c2[i].filter(t_now, raw_p2).astype(int)

        # Smooth 3D metric coordinates (inches)
        joints_3d_smooth = filters_3d[i].filter(t_now, raw_j3d)

        color = (0, 255, 0) if i == 0 else (255, 140, 0)
        for pts, frame in [(pts1_smooth, f1), (pts2_smooth, f2)]:
            for j1, j2 in SKELETON_BONES:
                cv2.line(frame, tuple(pts[j1]), tuple(pts[j2]), color, 2, cv2.LINE_AA)
            for pt in pts:
                cv2.circle(frame, tuple(pt), 3, (0, 0, 255), -1)

        wrist_xyz = joints_3d_smooth[0]
        pinch_in = np.linalg.norm(joints_3d_smooth[4] - joints_3d_smooth[8])

        y_off = 40 + i * 80
        cv2.putText(f1, f"Hand {i+1} Wrist: [{wrist_xyz[0]:+.1f}, {wrist_xyz[1]:+.1f}, {wrist_xyz[2]:+.1f}] in",
                    (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(f1, f"Hand {i+1} Pinch: {pinch_in:.2f} in",
                    (20, y_off + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    dt_ui = time.perf_counter() - t_ui_start
    ui_fps = 1.0 / dt_ui if dt_ui > 0 else 0
    ui_fps_history.append(ui_fps)
    if len(ui_fps_history) > 30:
        ui_fps_history.pop(0)
    avg_ui_fps = sum(ui_fps_history) / len(ui_fps_history)

    cv2.putText(f1, f"Display FPS: {avg_ui_fps:.1f} | Model Rate: {inf_fps:.1f} Hz", 
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    combined = np.hstack([f1, f2])
    cv2.imshow("Dual-Hand Stereo WiLoR (Async)", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        is_running = False
        break

cam1.release()
cam2.release()
cv2.destroyAllWindows()