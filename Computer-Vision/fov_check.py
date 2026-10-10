#!/home/peyton/kh/.venv/bin/python
"""Measure a camera's real horizontal FOV at the stream resolution, with a ruler.

Independent of calibrateCamera: hold the chessboard flat-on (facing the camera
squarely), centered, at a measured distance D from the lens, then:
    .venv/bin/python Computer-Vision/fov_check.py --cam 0 --square-mm 19.5 --dist-cm 40
Uses the center squares' pixel size (least lens distortion):
    fx = square_px * D / square,  hfov = 2 atan(width / 2 / fx).
Prints a running estimate; q quits. dx/dy near 1.00 means the board is square-on.
"""
import argparse
import math

import cv2
import numpy as np

from stereo_cam import Camera

BOARD = (9, 6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cam", type=int, default=0, help="V4L2 index (cam0 = 0, cam1 = 2)")
    ap.add_argument("--square-mm", type=float, required=True)
    ap.add_argument("--dist-cm", type=float, required=True, help="lens to board distance")
    ap.add_argument("--size", type=int, nargs=2, default=[640, 480], metavar=("W", "H"))
    args = ap.parse_args()
    cam = Camera(args.cam, *args.size)
    W = cam.size[0]
    seq = 0
    try:
        while True:
            f, _, seq = cam.wait_new(seq)
            if f is None:
                continue
            g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
            ok, c = cv2.findChessboardCorners(g, BOARD, cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE)
            line = "no board"
            if ok:
                c = cv2.cornerSubPix(g, c, (5, 5), (-1, -1),
                                     (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3))
                cv2.drawChessboardCorners(f, BOARD, c, ok)
                grid = c.reshape(BOARD[1], BOARD[0], 2)
                mid = grid[1:5, 2:7]  # central squares only
                dx = np.linalg.norm(np.diff(mid, axis=1), axis=2).mean()
                dy = np.linalg.norm(np.diff(mid, axis=0), axis=2).mean()
                fx = 0.5 * (dx + dy) * args.dist_cm * 10 / args.square_mm
                hfov = math.degrees(2 * math.atan(W / 2 / fx))
                line = f"fx {fx:.0f}  hfov {hfov:.1f} deg  (dx/dy {dx / dy:.2f})"
                print(line)
            cv2.putText(f, line, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            cv2.imshow("fov check (q quits)", f)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    finally:
        cam.close()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
