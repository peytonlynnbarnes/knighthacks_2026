"""Startup hand calibration, fed one stereo estimate per frame (used by hand_target_node.py).

Steps (instructions are shown in the preview window):
 1. NEUTRAL: open hand, palm toward the cameras, at camera height, held
    still for 2 s. Records the neutral position/orientation (-> arm at
    --neutral-offset), the palm bone lengths (bad-frame check) and the palm
    template (single-view fallback). Sanity-checks the palm width.
 2. FORWARD: move the hand ~15 cm in the arm's forward direction (+X, the
    way the gripper points at home) and hold. That direction + image-up
    (cameras mounted level) define the camera -> robot axes.
 3. RETURN: bring the hand back near the start. Then tracking begins.

Saved to calib/hand.json; hand_target_node.py --reuse-calib skips this.
The axes depend on how the cameras are mounted, so redo it if they move.
"""
import json
import os
from collections import deque

import numpy as np

import hand3d as h3
from stereo_cam import HERE

HAND_PATH = os.path.join(HERE, "calib", "hand.json")


class HandCalibrator:
    HOLD_S = 2.0          # neutral hold time
    STILL_M = 0.01        # palm center std while holding (m)
    FWD_MIN_M = 0.08      # forward step must move at least this far
    FWD_HOLD_S = 0.7
    RETURN_M = 0.04

    def __init__(self):
        self.step = "NEUTRAL"
        self.buf = deque()
        self.result = None
        self.msg = ""

    def _still(self, t, hold_s):
        while self.buf and t - self.buf[0][0] > hold_s:
            self.buf.popleft()
        if len(self.buf) < 5 or t - self.buf[0][0] < 0.9 * hold_s:
            return False
        ps = np.array([b[1]["p"] for b in self.buf])
        return float(np.max(np.std(ps, axis=0))) < self.STILL_M

    def instructions(self):
        return {
            "NEUTRAL": "CALIB 1/3: open hand, palm to cameras, hold still 2 s",
            "FORWARD": "CALIB 2/3: move hand ~15 cm in the ARM's forward (+X) direction, hold",
            "RETURN": "CALIB 3/3: bring hand back to the start position",
            "DONE": "calibrated",
        }[self.step] + (f"  [{self.msg}]" if self.msg else "")

    def feed(self, pose, t):
        """Returns True once calibration is complete (self.result is set)."""
        if pose is None or pose["mode"] != "stereo":
            self.buf.clear()  # hold must be continuous
            return False
        self.buf.append((t, pose))
        if self.step == "NEUTRAL" and self._still(t, self.HOLD_S):
            poses = [b[1] for b in self.buf]
            Ps = np.array([q["P"] for q in poses])
            P = np.median(Ps, axis=0)
            width = np.linalg.norm(P[5] - P[17])
            if not 0.03 < width < 0.20:
                self.msg = (f"palm width {width * 100:.0f} cm is implausible: "
                            "bad stereo calibration? retrying")
                self.buf.clear()
                return False
            self.msg = f"palm width {width * 100:.1f} cm" + (
                "  (unusual: check --hfov / stereo calib)" if not 0.055 < width < 0.11 else "")
            self.neutral_p = np.median([q["p"] for q in poses], axis=0)
            # same median hand as the template, so the 5-point fit at neutral is exactly this
            self.neutral_R = h3.hand_frame(P)[1]
            self.bones = np.median([h3.bone_lengths(q["P"]) for q in poses], axis=0)
            self.template = h3.palm_template(P)
            self.step, self.buf = "FORWARD", deque()
        elif self.step == "FORWARD":
            d = pose["p"] - self.neutral_p
            if np.linalg.norm(d) < self.FWD_MIN_M:
                self.buf.clear()
            elif self._still(t, self.FWD_HOLD_S):
                fwd = np.median([b[1]["p"] for b in self.buf], axis=0) - self.neutral_p
                try:
                    self.A = h3.robot_axes(fwd)
                except ValueError:
                    self.msg = "that was vertical; move horizontally"
                    self.buf.clear()
                    return False
                tilt = np.degrees(np.arcsin(abs(fwd @ [0, -1, 0]) / np.linalg.norm(fwd)))
                self.msg = f"forward = cam {np.round(fwd / np.linalg.norm(fwd), 2)}, {tilt:.0f} deg off level"
                self.step, self.buf = "RETURN", deque()
        elif self.step == "RETURN" and np.linalg.norm(pose["p"] - self.neutral_p) < self.RETURN_M:
            self.step = "DONE"
            self.result = dict(neutral_p=self.neutral_p, neutral_R=self.neutral_R,
                               bones=self.bones, template=self.template, A=self.A)
            return True
        return False


def save_hand(res, path=HAND_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({k: np.asarray(v).tolist() for k, v in res.items()}, f, indent=1)


def load_hand(path=HAND_PATH):
    with open(path) as f:
        return {k: np.array(v) for k, v in json.load(f).items()}
