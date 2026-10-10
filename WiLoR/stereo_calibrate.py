import os
import subprocess
import sys

def activate_macos_window(window_title="Stereo Hand Calibration"):
    """Brings Python to the foreground Dock and gives it focus safely without ctypes ABI crashes."""
    if sys.platform != "darwin":
        return
    # Run via osascript to trigger Cocoa window server activation safely
    cmd = f"""
    osascript -e '
    tell application "System Events"
        set frontmost of the first process whose unix id is {os.getpid()} to true
    end tell'
    """
    try:
        subprocess.Popen(cmd, shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except Exception:
        pass

import argparse
import os
import threading
import time
import cv2
import numpy as np

cv2.namedWindow("Stereo Hand Calibration", cv2.WINDOW_AUTOSIZE)
# Activate process and surface window
activate_macos_window()

parser = argparse.ArgumentParser(description="High-FPS Multi-Threaded Stereo Calibrator")
parser.add_argument("--cam1", type=int, default=0, help="Primary camera index")
parser.add_argument("--cam2", type=int, default=1, help="Secondary camera index")
parser.add_argument("--square_size", type=float, default=0.025, help="Checkerboard square size in meters")
parser.add_argument("--detect_interval", type=int, default=2, help="Check corners every N frames")
args = parser.parse_args()

CHECKERBOARD = (8, 6)

objp = np.zeros((CHECKERBOARD[0] * CHECKERBOARD[1], 3), np.float32)
objp[:, :2] = np.mgrid[0:CHECKERBOARD[0], 0:CHECKERBOARD[1]].T.reshape(-1, 2) * args.square_size

objpoints = []
imgpoints1 = []
imgpoints2 = []

# 1. Asynchronous Multi-threaded Camera Ingestion
class ThreadedCamera:
    def __init__(self, src):
        self.cap = cv2.VideoCapture(src, cv2.CAP_AVFOUNDATION)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        self.ret, self.frame = self.cap.read()
        self.running = True
        self.lock = threading.Lock()
        threading.Thread(target=self._reader, daemon=True).start()

    def _reader(self):
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                with self.lock:
                    self.ret, self.frame = ret, frame

    def read(self):
        with self.lock:
            return self.ret, (self.frame.copy() if self.ret else None)

    def release(self):
        self.running = False
        self.cap.release()

cam1 = ThreadedCamera(args.cam1)
cam2 = ThreadedCamera(args.cam2)

subpix_criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
fast_flags = cv2.CALIB_CB_ADAPTIVE_THRESH + cv2.CALIB_CB_NORMALIZE_IMAGE + cv2.CALIB_CB_FAST_CHECK

print("Optimized Stereo Calibrator active.")
print("Commands: [S] = Capture sample | [C] = Calculate & Save | [Q] = Quit")

cached_c1 = None
cached_c2 = None
found1 = False
found2 = False
frame_count = 0
fps_history = []

while True:
    t_start = time.perf_counter()
    r1, f1 = cam1.read()
    r2, f2 = cam2.read()
    if not r1 or not r2 or f1 is None or f2 is None:
        continue

    frame_count += 1
    h, w = f1.shape[:2]

    # Pre-convert to grayscale once
    g1 = cv2.cvtColor(f1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(f2, cv2.COLOR_BGR2GRAY)

    # 2. Interleaved Corner Detection
    # Fast prescreening on downsampled image to prevent UI stutter
    if frame_count % args.detect_interval == 0:
        g1_small = cv2.resize(g1, (w // 2, h // 2))
        g2_small = cv2.resize(g2, (w // 2, h // 2))

        pre1, _ = cv2.findChessboardCorners(g1_small, CHECKERBOARD, fast_flags)
        pre2, _ = cv2.findChessboardCorners(g2_small, CHECKERBOARD, fast_flags)

        if pre1 and pre2:
            found1, c1 = cv2.findChessboardCorners(g1, CHECKERBOARD, fast_flags)
            found2, c2 = cv2.findChessboardCorners(g2, CHECKERBOARD, fast_flags)
            if found1 and found2:
                cached_c1 = c1
                cached_c2 = c2
        else:
            found1, found2 = False, False
            cached_c1, cached_c2 = None, None

    # Render detection overlays
    disp1 = f1.copy()
    disp2 = f2.copy()
    if cached_c1 is not None and cached_c2 is not None:
        cv2.drawChessboardCorners(disp1, CHECKERBOARD, cached_c1, True)
        cv2.drawChessboardCorners(disp2, CHECKERBOARD, cached_c2, True)

    dt = time.perf_counter() - t_start
    fps = 1.0 / dt if dt > 0 else 0
    fps_history.append(fps)
    if len(fps_history) > 30:
        fps_history.pop(0)
    avg_fps = sum(fps_history) / len(fps_history)

    combined = np.hstack([disp1, disp2])
    status_color = (0, 255, 0) if (cached_c1 is not None and cached_c2 is not None) else (0, 0, 255)
    status_text = "READY TO CAPTURE" if (cached_c1 is not None and cached_c2 is not None) else "SEARCHING"

    cv2.putText(combined, f"Status: {status_text} | Samples: {len(objpoints)} | FPS: {avg_fps:.1f}", 
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
    cv2.imshow("Stereo Hand Calibration", combined)

    key = cv2.waitKey(1) & 0xFF

    # 3. Capture and Sub-pixel refinement
    if key == ord('s'):
        if found1 and found2 and cached_c1 is not None and cached_c2 is not None:
            c1_sub = cv2.cornerSubPix(g1, cached_c1, (11, 11), (-1, -1), subpix_criteria)
            c2_sub = cv2.cornerSubPix(g2, cached_c2, (11, 11), (-1, -1), subpix_criteria)
            objpoints.append(objp)
            imgpoints1.append(c1_sub)
            imgpoints2.append(c2_sub)
            print(f"Sample {len(objpoints)} successfully saved.")
        else:
            print("Both cameras must detect the full checkerboard before capturing.")

    elif key == ord('c'):
        if len(objpoints) >= 10:
            break
        print(f"Capture at least 10 samples before computing (current: {len(objpoints)}).")

    elif key == ord('q'):
        cam1.release()
        cam2.release()
        cv2.destroyAllWindows()
        exit()

cam1.release()
cam2.release()
cv2.destroyAllWindows()

# 4. Calibration Optimization Solver
print(f"\nComputing camera intrinsics across {len(objpoints)} point sets...")
img_shape = (w, h)
_, K1, D1, _, _ = cv2.calibrateCamera(objpoints, imgpoints1, img_shape, None, None)
_, K2, D2, _, _ = cv2.calibrateCamera(objpoints, imgpoints2, img_shape, None, None)

print("Solving stereo extrinsics (R, T)...")
flags = cv2.CALIB_FIX_INTRINSIC
criteria_stereo = (cv2.TERM_CRITERIA_MAX_ITER + cv2.TERM_CRITERIA_EPS, 100, 1e-5)
ret, K1, D1, K2, D2, R, T, E, F = cv2.stereoCalibrate(
    objpoints, imgpoints1, imgpoints2, K1, D1, K2, D2, img_shape,
    criteria=criteria_stereo, flags=flags
)

np.savez("stereo_camera_params.npz", K1=K1, D1=D1, K2=K2, D2=D2, R=R, T=T)
print(f"Calibration complete! RMS Reprojection Error: {ret:.4f}")
print("Parameters saved to stereo_camera_params.npz.")