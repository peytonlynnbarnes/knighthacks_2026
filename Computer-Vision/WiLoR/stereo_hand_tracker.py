import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

import argparse
import subprocess
import sys
import threading
import time
import math
import cv2
import numpy as np
import torch
import torch.nn.functional as F
import ultralytics.nn.tasks
from ultralytics import YOLO
from wilor.models import load_wilor

class YoloPoseTracker:
    """
    Tracks full-body kinematics (arms and shoulders) using Ultralytics YOLOv8 Pose.
    This entirely bypasses the MediaPipe Apple Silicon import bugs and runs natively 
    on the MPS GPU alongside WiLoR!
    """
    def __init__(self, device):
        print("Loading YOLOv8n-pose (Arm Kinematics) onto GPU...")
        # Loads a tiny, ultra-fast pose model. Downloads automatically if missing.
        self.model = YOLO('yolov8n-pose.pt').to(device)
        self.enabled = True

    def get_arm_keypoints(self, bgr_frame):
        if not self.enabled:
            return None, None, None, None, False
            
        # Heavy downsample for blazing fast inference
        h, w = bgr_frame.shape[:2]
        f_sm = cv2.resize(bgr_frame, (320, 240))
        
        # YOLO inference
        results = self.model(f_sm, verbose=False, conf=0.45, imgsz=320)[0]
        
        left_w, right_w = None, None
        left_arm, right_arm = None, None
        body_found = False
        
        if results.keypoints is not None and len(results.keypoints.xy) > 0:
            body_found = True
            # Get the first/largest detected person
            kpts = results.keypoints.xy[0].cpu().numpy()  # Shape: (17, 2)
            conf = results.keypoints.conf[0].cpu().numpy() if results.keypoints.conf is not None else np.ones(17)
            
            # Normalize coordinates to [0.0 - 1.0] to match the pipeline math
            norm = np.array([320.0, 240.0], dtype=np.float32)
            
            # YOLOv8 Pose Mapping:
            # 5: L_Shoulder, 7: L_Elbow, 9: L_Wrist
            # 6: R_Shoulder, 8: R_Elbow, 10: R_Wrist
            if conf[9] > 0.3:
                left_w = kpts[9] / norm
            if conf[10] > 0.3:
                right_w = kpts[10] / norm
                
            if conf[5] > 0.3 and conf[7] > 0.3 and conf[9] > 0.3:
                left_arm = np.array([kpts[5], kpts[7], kpts[9]], dtype=np.float32) / norm
                
            if conf[6] > 0.3 and conf[8] > 0.3 and conf[10] > 0.3:
                right_arm = np.array([kpts[6], kpts[8], kpts[10]], dtype=np.float32) / norm
                
        return left_w, right_w, left_arm, right_arm, body_found

def activate_macos_window():
    if sys.platform != "darwin":
        return
    cmd = f"""osascript -e 'tell application "System Events" to set frontmost of the first process whose unix id is {os.getpid()} to true'"""
    try:
        subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

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
        if dt > 0.25:
            self.x_prev = np.array(x, dtype=np.float32)
            self.dx_prev = np.zeros_like(self.x_prev)
            self.t_prev = t
            return self.x_prev

        rate = 1.0 / dt
        dx = (x - self.x_prev) / dt
        alpha_d = self._alpha(rate, self.d_cutoff)
        dx_hat = alpha_d * dx + (1.0 - alpha_d) * self.dx_prev

        speed = np.linalg.norm(dx_hat, axis=-1, keepdims=True)
        cutoff = self.min_cutoff + self.beta * speed
        alpha = self._alpha(rate, cutoff)

        x_hat = alpha * x + (1.0 - alpha) * self.x_prev
        self.x_prev = x_hat
        self.dx_prev = dx_hat
        self.t_prev = t
        return x_hat

parser = argparse.ArgumentParser(description="Instrumented Fully Batched Dual-Hand Stereo WiLoR")
parser.add_argument("--cam1", type=int, default=1, help="Primary camera index")
parser.add_argument("--cam2", type=int, default=3, help="Secondary camera index")
parser.add_argument("--width", type=int, default=1920, help="Camera capture width (default: 1920)")
parser.add_argument("--height", type=int, default=1080, help="Camera capture height (default: 1080)")
parser.add_argument("--calib", type=str, default="stereo_camera_params.npz", help="Calibration file")
parser.add_argument("--yolo_interval", type=int, default=8, help="Run full YOLO check every N worker steps")
parser.add_argument("--mode", type=str, choices=["batched", "interleaved"], default="interleaved", help="Inference mode")
args = parser.parse_args()

cv2.namedWindow("Dual-Hand Stereo WiLoR", cv2.WINDOW_AUTOSIZE)
activate_macos_window()

if not os.path.exists(args.calib):
    raise FileNotFoundError(f"Missing {args.calib}. Run stereo_calibrate.py first.")

calib = np.load(args.calib)
K1, D1 = calib['K1'].copy(), calib['D1'].copy()
K2, D2 = calib['K2'].copy(), calib['D2'].copy()
R, T = calib['R'], calib['T']

# Scale calibration matrices if runtime capture res differs from calibration res
calib_w = float(calib['img_size'][0]) if 'img_size' in calib else (float(calib['image_size'][0]) if 'image_size' in calib else 640.0)
calib_h = float(calib['img_size'][1]) if 'img_size' in calib else (float(calib['image_size'][1]) if 'image_size' in calib else 480.0)

if args.width != int(calib_w) or args.height != int(calib_h):
    scale_x = float(args.width) / calib_w
    scale_y = float(args.height) / calib_h
    print(f"Rescaling calibration intrinsics from ({int(calib_w)}x{int(calib_h)}) to ({args.width}x{args.height}) by factor ({scale_x:.3f}, {scale_y:.3f})")
    K1[0, 0] *= scale_x
    K1[0, 2] *= scale_x
    K1[1, 1] *= scale_y
    K1[1, 2] *= scale_y

    K2[0, 0] *= scale_x
    K2[0, 2] *= scale_x
    K2[1, 1] *= scale_y
    K2[1, 2] *= scale_y

P1 = K1 @ np.hstack((np.eye(3), np.zeros((3, 1))))
P2 = K2 @ np.hstack((R, T))
M_TO_INCHES = 39.3700787

class LockFreeCamera:
    def __init__(self, src, target_w=1920, target_h=1080):
        self.cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, target_w)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, target_h)
        self.target_w = target_w
        self.target_h = target_h

        ret, frame = self.cap.read()
        self.ret = ret
        self.frame = frame if (ret and frame.shape[:2] == (target_h, target_w)) else np.zeros((target_h, target_w, 3), dtype=np.uint8)
        self.running = True
        threading.Thread(target=self._worker, daemon=True).start()

    def _worker(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                if frame.shape[0] != self.target_h or frame.shape[1] != self.target_w:
                    frame = cv2.resize(frame, (self.target_w, self.target_h))
                self.frame = frame
                self.ret = True

    def read(self):
        return self.ret, self.frame

    def release(self):
        self.running = False
        self.cap.release()

cam1 = LockFreeCamera(args.cam1, target_w=args.width, target_h=args.height)
cam2 = LockFreeCamera(args.cam2, target_w=args.width, target_h=args.height)

torch.serialization.add_safe_globals([ultralytics.nn.tasks.PoseModel])
device = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
print(f"Loading Models onto {device}...")

checkpoint_path = './pretrained_models/wilor_final.ckpt'
cfg_path = './pretrained_models/model_config.yaml'
model, model_cfg = load_wilor(checkpoint_path=checkpoint_path, cfg_path=cfg_path)
model._skip_mesh_inference = True  # Disables the 778 MANO vertex calculation
model = model.to(device).half()
model.eval()

detector = YOLO('./pretrained_models/detector.pt').to(device)
pose_tracker = YoloPoseTracker(device)

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

HAND_COLORS = {
    'Right': (0, 255, 0),      # Green
    'Left': (0, 140, 255),     # Orange (BGR)
    'Unknown': (200, 200, 200)
}

def make_square_crop_meta(box, w, h, expansion=1.35, min_size=110.0):
    x1, y1, x2, y2 = box
    cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
    size = max(x2 - x1, y2 - y1, min_size) * expansion
    size = min(size, min(w, h))

    x1_s = max(0, int(cx - size / 2.0))
    y1_s = max(0, int(cy - size / 2.0))
    x2_s = min(w, int(cx + size / 2.0))
    y2_s = min(h, int(cy + size / 2.0))
    return x1_s, y1_s, x2_s, y2_s

def tight_box_from_landmarks(pts, w, h):
    xmin = max(0, int(pts[:, 0].min()))
    ymin = max(0, int(pts[:, 1].min()))
    xmax = min(w, int(pts[:, 0].max()))
    ymax = min(h, int(pts[:, 1].max()))
    return [xmin, ymin, xmax, ymax]

def triangulate_landmarks(pts1, pts2):
    pts1_undist = cv2.undistortPoints(np.expand_dims(pts1, axis=1).astype(np.float32), K1, D1, P=K1)
    pts2_undist = cv2.undistortPoints(np.expand_dims(pts2, axis=1).astype(np.float32), K2, D2, P=K2)
    pts_4d = cv2.triangulatePoints(P1, P2, pts1_undist, pts2_undist)
    pts_3d_m = (pts_4d[:3] / pts_4d[3]).T
    return pts_3d_m * M_TO_INCHES

def box_center(b):
    return np.array([(b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0])

def box_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interArea = max(0.0, float(xB - xA)) * max(0.0, float(yB - yA))
    boxAArea = max(0.0, float(boxA[2] - boxA[0])) * max(0.0, float(boxA[3] - boxA[1]))
    boxBArea = max(0.0, float(boxB[2] - boxB[0])) * max(0.0, float(boxB[3] - boxB[1]))
    denom = boxAArea + boxBArea - interArea
    if denom <= 0.0:
        return 0.0
    return interArea / denom

def nms_boxes(boxes, iou_thresh=0.35, res_scale=1.0):
    if len(boxes) == 0:
        return []
    sorted_b = sorted(boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]), reverse=True)
    kept = []
    # Relaxed from 65.0 to 35.0 to allow hands to cross closer together
    min_dist = 35.0 * res_scale
    for b in sorted_b:
        if not any(box_iou(b, k) > iou_thresh or np.linalg.norm(box_center(b) - box_center(k)) < min_dist for k in kept):
            kept.append(b)
    return kept

def match_stereo_pairs(boxes1, boxes2, res_scale_x=1.0, res_scale_y=1.0):
    clean1 = nms_boxes(boxes1, res_scale=res_scale_x)
    clean2 = nms_boxes(boxes2, res_scale=res_scale_x)
    if len(clean1) == 0 or len(clean2) == 0:
        return []

    y_thresh = 80.0 * res_scale_y
    sep_thresh = 45.0 * res_scale_x  # Relaxed separation threshold
    size_thresh = 150.0 * res_scale_y

    clean1 = sorted(clean1, key=lambda b: (b[0] + b[2]) / 2.0)
    clean2 = sorted(clean2, key=lambda b: (b[0] + b[2]) / 2.0)

    if len(clean1) >= 2 and len(clean2) >= 2:
        c1_0, c2_0 = box_center(clean1[0]), box_center(clean2[0])
        c1_1, c2_1 = box_center(clean1[1]), box_center(clean2[1])
        if abs(c1_0[1] - c2_0[1]) < y_thresh and abs(c1_1[1] - c2_1[1]) < y_thresh:
            if np.linalg.norm(c1_0 - c1_1) > sep_thresh and np.linalg.norm(c2_0 - c2_1) > sep_thresh:
                return [(clean1[0], clean2[0]), (clean1[1], clean2[1])]

    pairs = []
    used2 = set()
    for b1 in clean1:
        c1 = box_center(b1)
        h1 = b1[3] - b1[1]
        best_j = None
        best_score = float('inf')
        for j, b2 in enumerate(clean2):
            if j in used2:
                continue
            c2 = box_center(b2)
            h2 = b2[3] - b2[1]
            y_diff = abs(c1[1] - c2[1])
            size_diff = abs(h1 - h2)
            if y_diff < y_thresh and size_diff < size_thresh:
                score = y_diff + 0.3 * size_diff
                if score < best_score:
                    best_score = score
                    best_j = j
        if best_j is not None:
            pairs.append((b1, clean2[best_j]))
            used2.add(best_j)
    return pairs

state_lock = threading.Lock()
shared_results = {
    'hands': {},
    'arms': {'left': None, 'right': None, 'timestamp': 0.0, 'detected': False},
    'inference_fps': 0.0,
    'wilor_gpu_ms': 0.0,
    'total_ms': 0.0
}
is_running = True

class HandTrack:
    def __init__(self, track_id):
        self.track_id = track_id
        self.box1 = None
        self.box2 = None
        self.active = False
        self.misses = 0
        self.chirality = 'Unknown'
        self.chirality_score = 0.0
        self.stable_count = 0

    def reset(self):
        self.box1 = None
        self.box2 = None
        self.active = False
        self.misses = 0
        self.chirality = 'Unknown'
        self.chirality_score = 0.0
        self.stable_count = 0

tracks = [HandTrack(0), HandTrack(1)]

def compute_2d_chirality_score(pts2d):
    p0, p9, p4 = pts2d[0], pts2d[9], pts2d[4]
    v_axis = p9 - p0
    v_thumb = p4 - p0
    cross_thumb = v_axis[0] * v_thumb[1] - v_axis[1] * v_thumb[0]
    denom = (np.linalg.norm(v_axis) * np.linalg.norm(v_thumb) + 1e-5)
    return float(np.clip(cross_thumb / denom, -1.0, 1.0))

def compute_multi_view_chirality(pts1, pts2, joints_3d):
    p0, p5, p9, p17 = joints_3d[0], joints_3d[5], joints_3d[9], joints_3d[17]
    thumb_tip = joints_3d[4]

    v_long = p9 - p0
    v_trans = p5 - p17
    normal = np.cross(v_long, v_trans)
    v_thumb = thumb_tip - p0

    n_norm = np.linalg.norm(normal)
    t_norm = np.linalg.norm(v_thumb)
    score_3d = float(np.dot(normal, v_thumb) / (n_norm * t_norm)) if (n_norm > 1e-4 and t_norm > 1e-4) else 0.0

    score_2d_c1 = compute_2d_chirality_score(pts1)
    score_2d_c2 = compute_2d_chirality_score(pts2)
    avg_2d = 0.5 * (score_2d_c1 + score_2d_c2)

    combined = 0.65 * score_3d + 0.35 * avg_2d
    return np.clip(combined, -1.0, 1.0)

def inference_worker():
    global is_running
    infer_times = []
    worker_step = 0

    with torch.inference_mode():
        while is_running:
            t0 = time.perf_counter()
            r1, f1 = cam1.read()
            r2, f2 = cam2.read()
            if not r1 or not r2 or f1 is None or f2 is None:
                time.sleep(0.002)
                continue

            h, w = f1.shape[:2]
            worker_step += 1

            # Get full-body arm kinematics for absolute hand association
            mp_left_w, mp_right_w, mp_left_arm, mp_right_arm, pose_detected = pose_tracker.get_arm_keypoints(f1)
            
            with state_lock:
                shared_results['arms']['left'] = mp_left_arm
                shared_results['arms']['right'] = mp_right_arm
                shared_results['arms']['timestamp'] = time.perf_counter()
                shared_results['arms']['detected'] = pose_detected

            need_yolo = (worker_step % args.yolo_interval == 1) or not any(t.active for t in tracks)

            if need_yolo:
                f1_sm = cv2.resize(f1, (320, 240))
                f2_sm = cv2.resize(f2, (320, 240))
                det_results = detector([f1_sm, f2_sm], verbose=False, conf=0.30, imgsz=320)
                det1 = det_results[0].boxes.xyxy.cpu().numpy()
                det2 = det_results[1].boxes.xyxy.cpu().numpy()

                sx = w / 320.0
                sy = h / 240.0
                scale_vec = np.array([sx, sy, sx, sy], dtype=np.float32)

                raw_boxes1 = (det1 * scale_vec) if len(det1) > 0 else []
                raw_boxes2 = (det2 * scale_vec) if len(det2) > 0 else []

                stereo_pairs = match_stereo_pairs(raw_boxes1, raw_boxes2, res_scale_x=sx, res_scale_y=sy)
                n_pairs = len(stereo_pairs)

                match_prox = 140.0 * sx
                col_prox = 90.0 * sx
                pair_sep = 80.0 * sx

                if n_pairs == 1:
                    cand_b1, cand_b2 = stereo_pairs[0]
                    c_cand = box_center(cand_b1)

                    assigned = False
                    for t in tracks:
                        if t.active and t.box1 is not None:
                            if np.linalg.norm(box_center(t.box1) - c_cand) < match_prox:
                                t.box1, t.box2 = cand_b1, cand_b2
                                t.misses = 0
                                assigned = True
                                break

                    if not assigned:
                        active_collision = any(
                            t.active and t.box1 is not None and np.linalg.norm(box_center(t.box1) - c_cand) < col_prox
                            for t in tracks
                        )
                        if not active_collision:
                            target = tracks[1] if tracks[0].active else tracks[0]
                            target.box1, target.box2 = cand_b1, cand_b2
                            target.active = True
                            target.misses = 0

                elif n_pairs >= 2:
                    p0, p1 = stereo_pairs[0], stereo_pairs[1]
                    c_p0, c_p1 = box_center(p0[0]), box_center(p1[0])

                    if np.linalg.norm(c_p0 - c_p1) > pair_sep:
                        if tracks[0].active and tracks[1].active and tracks[0].box1 is not None and tracks[1].box1 is not None:
                            c0 = box_center(tracks[0].box1)
                            c1 = box_center(tracks[1].box1)
                            cost_direct = np.linalg.norm(c_p0 - c0) + np.linalg.norm(c_p1 - c1)
                            cost_swap = np.linalg.norm(c_p1 - c0) + np.linalg.norm(c_p0 - c1)
                            if cost_direct <= cost_swap:
                                tracks[0].box1, tracks[0].box2 = p0[0], p0[1]
                                tracks[1].box1, tracks[1].box2 = p1[0], p1[1]
                            else:
                                tracks[0].box1, tracks[0].box2 = p1[0], p1[1]
                                tracks[1].box1, tracks[1].box2 = p0[0], p0[1]
                        else:
                            sorted_pairs = sorted([p0, p1], key=lambda p: (p[0][0] + p[0][2]) / 2.0)
                            tracks[0].box1, tracks[0].box2 = sorted_pairs[0][0], sorted_pairs[0][1]
                            tracks[1].box1, tracks[1].box2 = sorted_pairs[1][0], sorted_pairs[1][1]

                        tracks[0].active = True
                        tracks[0].misses = 0
                        tracks[1].active = True
                        tracks[1].misses = 0

            # Guard against track collision onto identical hand
            if tracks[0].active and tracks[1].active and tracks[0].box1 is not None and tracks[1].box1 is not None:
                dist = np.linalg.norm(box_center(tracks[0].box1) - box_center(tracks[1].box1))
                iou = box_iou(tracks[0].box1, tracks[1].box1)
                
                # Drastically relaxed collision guard. 
                # Only drops a track if hands are nearly on top of each other.
                guard_dist = 40.0 * (w / 320.0) 
                if dist < guard_dist or iou > 0.45:
                    drop_slot = 1 if tracks[0].misses <= tracks[1].misses else 0
                    tracks[drop_slot].reset()
                    with state_lock:
                        shared_results['hands'].pop(drop_slot, None)

            active_tracks = [t for t in tracks if t.active]
            t_gpu = 0.0

            if len(active_tracks) > 0:
                target_tracks = [active_tracks[worker_step % len(active_tracks)]] if args.mode == "interleaved" else active_tracks

                crops = []
                metas = []

                for t in target_tracks:
                    # Slightly increased crop expansion to 1.50x to prevent fast hands from slipping out of the bounding box
                    x1_a, y1_a, x2_a, y2_a = make_square_crop_meta(t.box1, w, h, expansion=1.50, min_size=110.0)
                    x1_b, y1_b, x2_b, y2_b = make_square_crop_meta(t.box2, w, h, expansion=1.50, min_size=110.0)
                    cw1, ch1 = x2_a - x1_a, y2_a - y1_a
                    cw2, ch2 = x2_b - x1_b, y2_b - y1_b

                    c1 = f1[y1_a:y2_a, x1_a:x2_a]
                    c2 = f2[y1_b:y2_b, x1_b:x2_b]

                    if c1.size > 0 and c2.size > 0 and cw1 > 40 and ch1 > 40 and cw2 > 40 and ch2 > 40:
                        c1_res = cv2.resize(cv2.cvtColor(c1, cv2.COLOR_BGR2RGB), (target_size, target_size))
                        c2_res = cv2.resize(cv2.cvtColor(c2, cv2.COLOR_BGR2RGB), (target_size, target_size))
                        crops.extend([c1_res, c2_res])
                        metas.append((t, (x1_a, y1_a, cw1, ch1), (x1_b, y1_b, cw2, ch2)))
                    else:
                        t.misses += 1
                        if t.misses > 10:
                            t.reset()
                            with state_lock:
                                shared_results['hands'].pop(t.track_id, None)

                if len(crops) > 0:
                    t_batch = torch.from_numpy(np.stack(crops)).permute(0, 3, 1, 2).to(device, non_blocking=True).half() / 255.0
                    t_norm = (t_batch - mean_t) / std_t

                    tg_start = time.perf_counter()
                    out = model({'img': t_norm})
                    if torch.backends.mps.is_available():
                        torch.mps.synchronize()
                    t_gpu = (time.perf_counter() - tg_start) * 1000.0

                    if 'pred_keypoints_2d' in out:
                        preds = out['pred_keypoints_2d'].cpu().numpy()

                        def to_frame(pts, cw, ch, xs, ys):
                            pts = pts.copy()
                            if pts.min() < 0:
                                pts = (pts + 0.5) * np.array([cw, ch], dtype=np.float32)
                            elif pts.max() <= 1.0:
                                pts = pts * np.array([cw, ch], dtype=np.float32)
                            else:
                                pts = pts * (np.array([cw, ch], dtype=np.float32) / float(target_size))
                            return pts + np.array([xs, ys], dtype=np.float32)

                        new_joints = {}
                        for idx, (t, m1, m2) in enumerate(metas):
                            pts1 = to_frame(preds[2 * idx], m1[2], m1[3], m1[0], m1[1])
                            pts2 = to_frame(preds[2 * idx + 1], m2[2], m2[3], m2[0], m2[1])

                            pts1[:, 0] = np.clip(pts1[:, 0], 0, w - 1)
                            pts1[:, 1] = np.clip(pts1[:, 1], 0, h - 1)
                            pts2[:, 0] = np.clip(pts2[:, 0], 0, w - 1)
                            pts2[:, 1] = np.clip(pts2[:, 1], 0, h - 1)

                            span1 = max(pts1[:, 0].max() - pts1[:, 0].min(), pts1[:, 1].max() - pts1[:, 1].min())
                            span2 = max(pts2[:, 0].max() - pts2[:, 0].min(), pts2[:, 1].max() - pts2[:, 1].min())

                            min_span = 50.0 * (w / 1920.0)
                            if span1 >= min_span and span2 >= min_span:
                                t.box1 = tight_box_from_landmarks(pts1, w, h)
                                t.box2 = tight_box_from_landmarks(pts2, w, h)
                                t.misses = 0

                                joints_3d = triangulate_landmarks(pts1, pts2)
                                
                                # 1. Primary Classification: Arm Kinematics Ground Truth
                                c1_norm = box_center(t.box1) / np.array([w, h], dtype=np.float32)
                                dist_l = np.linalg.norm(c1_norm - mp_left_w) if mp_left_w is not None else float('inf')
                                dist_r = np.linalg.norm(c1_norm - mp_right_w) if mp_right_w is not None else float('inf')

                                pose_label = None
                                if min(dist_l, dist_r) < 0.20:  # WiLoR hand is near a tracked shoulder/arm chain
                                    pose_label = 'Left' if dist_l < dist_r else 'Right'

                                if pose_label:
                                    t.chirality = pose_label
                                    t.chirality_score = 1.0 if pose_label == 'Left' else -1.0
                                    t.stable_count = 5  # Lock it firmly
                                else:
                                    # 2. Fallback Classification: 3D Multi-View Bone Geometry (if body is out of frame)
                                    raw_score = compute_multi_view_chirality(pts1, pts2, joints_3d)

                                    if t.stable_count == 0:
                                        t.chirality_score = raw_score
                                        t.chirality = 'Right' if raw_score <= 0.0 else 'Left'
                                    else:
                                        t.chirality_score = 0.80 * t.chirality_score + 0.20 * raw_score

                                    t.stable_count += 1

                                    if t.chirality == 'Right' and t.chirality_score > 0.12:
                                        t.chirality = 'Left'
                                    elif t.chirality == 'Left' and t.chirality_score < -0.12:
                                        t.chirality = 'Right'

                                new_joints[t.track_id] = (pts1, pts2, joints_3d)
                            else:
                                t.misses += 1
                                if t.misses > 10:
                                    t.reset()
                                    with state_lock:
                                        shared_results['hands'].pop(t.track_id, None)

                        # Bimanual complementarity enforcement
                        if tracks[0].active and tracks[1].active and tracks[0].box1 is not None and tracks[1].box1 is not None:
                            if tracks[0].chirality == tracks[1].chirality and tracks[0].chirality != 'Unknown':
                                if tracks[0].chirality_score < tracks[1].chirality_score:
                                    tracks[0].chirality, tracks[1].chirality = 'Right', 'Left'
                                else:
                                    tracks[0].chirality, tracks[1].chirality = 'Left', 'Right'

                        with state_lock:
                            for tid, (p1, p2, j3) in new_joints.items():
                                shared_results['hands'][tid] = {
                                    'pts1': p1,
                                    'pts2': p2,
                                    'joints_3d': j3,
                                    'chirality': tracks[tid].chirality,
                                    'timestamp': time.perf_counter()
                                }
            else:
                with state_lock:
                    shared_results['hands'].clear()

            dt_total = time.perf_counter() - t0
            infer_times.append(dt_total)
            if len(infer_times) > 15:
                infer_times.pop(0)

            avg_cycle = sum(infer_times) / len(infer_times) if len(infer_times) > 0 else 0.05
            fps = 1.0 / avg_cycle if avg_cycle > 0 else 0.0

            with state_lock:
                shared_results['inference_fps'] = fps
                shared_results['wilor_gpu_ms'] = t_gpu
                shared_results['total_ms'] = avg_cycle * 1000.0

threading.Thread(target=inference_worker, daemon=True).start()

# ... existing code ...
# 5. UI Render Loop
print("Stereo Hand Tracker running in single GPU thread. Press 'q' to exit.")
ui_fps_history = []

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
    t_now = time.perf_counter()

    # Create mirrored frames for the UI presentation (acts like a natural mirror)
    f1_vis = cv2.flip(f1, 1)
    f2_vis = cv2.flip(f2, 1)

    with state_lock:
        active_hands = dict(shared_results['hands'])
        arms_data = dict(shared_results['arms'])
        inf_fps = shared_results['inference_fps']
        gpu_ms = shared_results['wilor_gpu_ms']
        total_ms = shared_results['total_ms']
        
    # Draw YOLO Pose Arms on the mirrored visual frame
    if t_now - arms_data.get('timestamp', 0.0) < 0.40:
        for side, color in [('left', HAND_COLORS['Left']), ('right', HAND_COLORS['Right'])]:
            arm_pts = arms_data.get(side)
            if arm_pts is not None:
                pts = (arm_pts * np.array([w, h], dtype=np.float32)).astype(int)
                
                # Mirror the X coordinates for the visual overlay
                pts[:, 0] = w - pts[:, 0]
                
                # Draw Shoulder -> Elbow -> Wrist links
                cv2.line(f1_vis, tuple(pts[0]), tuple(pts[1]), color, 4, cv2.LINE_AA)
                cv2.line(f1_vis, tuple(pts[1]), tuple(pts[2]), color, 4, cv2.LINE_AA)
                # Draw joint nodes
                for pt in pts:
                    cv2.circle(f1_vis, tuple(pt), 6, (255, 255, 255), -1)
                    cv2.circle(f1_vis, tuple(pt), 4, color, -1)

    for i in list(active_hands.keys()):
        hand = active_hands[i]
        if t_now - hand['timestamp'] > 0.40:
            filters_3d.pop(i, None)
            filters_2d_c1.pop(i, None)
            filters_2d_c2.pop(i, None)
            continue

        raw_p1 = hand['pts1']
        raw_p2 = hand['pts2']
        raw_j3d = hand['joints_3d']
        hand_label = hand.get('chirality', f'Hand {i+1}')
        color = HAND_COLORS.get(hand_label, (0, 255, 0) if i == 0 else (0, 140, 255))

        if i not in filters_3d:
            filters_2d_c1[i] = OneEuroFilter(t_now, raw_p1, min_cutoff=1.2, beta=0.01)
            filters_2d_c2[i] = OneEuroFilter(t_now, raw_p2, min_cutoff=1.2, beta=0.01)
            filters_3d[i]    = OneEuroFilter(t_now, raw_j3d, min_cutoff=0.6, beta=0.004)

        pts1_smooth = filters_2d_c1[i].filter(t_now, raw_p1).astype(int)
        pts2_smooth = filters_2d_c2[i].filter(t_now, raw_p2).astype(int)
        joints_3d_smooth = filters_3d[i].filter(t_now, raw_j3d)

        # Mirror X coordinates for drawing on the flipped visual frames
        pts1_draw = pts1_smooth.copy()
        pts1_draw[:, 0] = w - pts1_draw[:, 0]
        
        pts2_draw = pts2_smooth.copy()
        pts2_draw[:, 0] = w - pts2_draw[:, 0]

        for pts_draw, frame_vis in [(pts1_draw, f1_vis), (pts2_draw, f2_vis)]:
            for j1, j2 in SKELETON_BONES:
                cv2.line(frame_vis, tuple(pts_draw[j1]), tuple(pts_draw[j2]), color, 2, cv2.LINE_AA)
            for pt in pts_draw:
                cv2.circle(frame_vis, tuple(pt), 3, (0, 0, 255), -1)
            tag_pos = (max(10, pts_draw[0][0] - 40), max(25, pts_draw[0][1] - 15))
            cv2.putText(frame_vis, hand_label, tag_pos, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

        wrist_xyz = joints_3d_smooth[0]
        pinch_in = np.linalg.norm(joints_3d_smooth[4] - joints_3d_smooth[8])

        y_off = 40 + i * 80
        cv2.putText(f1_vis, f"{hand_label} Wrist: [{wrist_xyz[0]:+.1f}, {wrist_xyz[1]:+.1f}, {wrist_xyz[2]:+.1f}] in",
                    (20, y_off), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)
        cv2.putText(f1_vis, f"{hand_label} Pinch: {pinch_in:.2f} in",
                    (20, y_off + 25), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)

    dt_ui = time.perf_counter() - t_ui_start
    ui_fps = 1.0 / dt_ui if dt_ui > 0 else 0
    ui_fps_history.append(ui_fps)
    if len(ui_fps_history) > 30:
        ui_fps_history.pop(0)
    avg_ui_fps = sum(ui_fps_history) / len(ui_fps_history)

    # Profiler HUD readout
    cv2.rectangle(f2_vis, (15, h-95), (315, h-10), (0, 0, 0), -1)
    cv2.rectangle(f2_vis, (15, h-95), (315, h-10), (100, 100, 100), 1)

    cv2.putText(f2_vis, f"Model Rate: {inf_fps:.1f} Hz ({total_ms:.1f}ms)", (25, h-75),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 255, 255), 1)
    cv2.putText(f2_vis, f"  WiLoR GPU: {gpu_ms:.1f}ms", (25, h-55),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1)
    cv2.putText(f2_vis, f"  Active Hands: {len(active_hands)} ({args.mode})", (25, h-35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (200, 200, 200), 1)
    cv2.putText(f2_vis, f"  Display FPS: {avg_ui_fps:.1f}", (25, h-17),
                cv2.FONT_HERSHEY_SIMPLEX, 0.40, (0, 255, 0), 1)

    cv2.putText(f1_vis, f"Display FPS: {avg_ui_fps:.1f} | Model: {inf_fps:.1f} Hz", 
                (20, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    combined = np.hstack([f1_vis, f2_vis])
    cv2.imshow("Dual-Hand Stereo WiLoR", combined)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        is_running = False
        break

cam1.release()
cam2.release()
cv2.destroyAllWindows()