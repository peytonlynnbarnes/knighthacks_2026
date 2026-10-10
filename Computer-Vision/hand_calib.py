"""Startup hand calibration, fed one stereo estimate per frame (used by hand_target_node.py).

Steps (instructions are shown in the preview window):
 1. NEUTRAL: open hand, palm toward the cameras, at camera height, held
    still for 2 s. Records the neutral position/orientation (-> arm at
    --neutral-offset), the palm bone lengths (bad-frame check), the palm
    template (single-view fallback) and the flat-hand finger curl (gripper
    fully open). Sanity-checks the palm width.
 2. FORWARD: move the hand ~15 cm in the arm's forward direction (+X, the
    way the gripper points at home) and hold. That direction + image-up
    (cameras mounted level) define the camera -> robot axes.
 3. RETURN: bring the hand back near the start.
 4-6. RANGE (skipped with range_steps=False):
    4. ROLL: one direction only: turn the flat hand a quarter turn (90 deg)
       CLOCKWISE, then back. That turn = gripper roll 0..+90; its sign in
       robot axes is whatever the hand measures, so it doesn't depend on the
       camera mounting. The other roll direction isn't used (clips at 0).
    5-6. TILT: tilt one way as far as it tracks, back, then the other way
       ("the other way" = the opposite sign); drives gripper pitch -85..+85.
    Records the largest angle per direction while the palm still faces the
    cameras well enough (hand_mapping.FACING_DEG); a step ends when the hand
    comes back (or tracking drops) after >= RANGE_MIN_DEG, or after
    RANGE_TIMEOUT_S (that direction then keeps the default gain). The node
    turns the range into rotation gains (hand_mapping.gains_from_range).
 7. FINAL: hand back at the start, palm to the cameras. Then tracking begins.

Runs at every hand_target_node.py start (the calibration hand is the origin);
calib/hand.json is a record of the last one, not reloaded.
The axes depend on how the cameras are mounted, so redo it if they move.
"""
import json
import os
from collections import deque

import numpy as np

import hand3d as h3
from hand_mapping import FACING_DEG, facing_weight
from stereo_cam import HERE

HAND_PATH = os.path.join(HERE, "calib", "hand.json")


class HandCalibrator:
    HOLD_S = 2.0          # neutral hold time
    STILL_M = 0.01        # palm center std while holding (m)
    FWD_MIN_M = 0.08      # forward step must move at least this far
    FWD_HOLD_S = 0.7
    RETURN_M = 0.04
    RANGE_MIN_DEG = 15.0    # a range step counts once the hand turned this far
    RANGE_BACK = 0.5        # ...and then came back below this fraction of its max
    RANGE_LOST_S = 0.3      # ...or tracking dropped this long
    RANGE_TIMEOUT_S = 12.0  # per range step
    FINAL_DEG = 20.0        # final step: palm back within this of neutral
    # (step, axis index in robot rpy, which sign: "a" = whichever comes first, "b" = opposite)
    RANGE_STEPS = (("ROLL", 0, "a"), ("TILT_A", 1, "a"), ("TILT_B", 1, "b"))

    def __init__(self, frame="fit", range_steps=True):
        self.frame = frame    # how tracking computes the hand rotation (StereoHand frame=)
        self.range_steps = range_steps
        self.step = "NEUTRAL"
        self.buf = deque()
        self.result = None
        self.msg = ""
        self.range = {"roll": [None, None], "pitch": [None, None]}  # (neg, pos) deg
        self._first_sign = {}

    def _still(self, t, hold_s):
        while self.buf and t - self.buf[0][0] > hold_s:
            self.buf.popleft()
        if len(self.buf) < 5 or t - self.buf[0][0] < 0.9 * hold_s:
            return False
        ps = np.array([b[1]["p"] for b in self.buf])
        return float(np.max(np.std(ps, axis=0))) < self.STILL_M

    def instructions(self):
        n = 7 if self.range_steps else 3
        return {
            "NEUTRAL": f"CALIB 1/{n}: open hand, palm to cameras, hold still 2 s",
            "FORWARD": f"CALIB 2/{n}: move hand ~15 cm in the ARM's forward (+X) direction, hold",
            "RETURN": f"CALIB 3/{n}: bring hand back to the start position",
            "ROLL": f"CALIB 4/{n}: flat hand, slowly ROLL it CLOCKWISE a quarter turn (90 deg, like a doorknob), then back",
            "TILT_A": f"CALIB 5/{n}: slowly TILT the hand (fingertips toward or away from the cameras) as far as it tracks, then back",
            "TILT_B": f"CALIB 6/{n}: now TILT the other way as far as it tracks, then back",
            "FINAL": f"CALIB 7/{n}: hand back at the start, palm to cameras",
            "DONE": "calibrated",
        }[self.step] + (f"  [{self.msg}]" if self.msg else "")

    def _hand_rpy(self, pose):
        """Hand rotation from neutral in robot axes (deg), as tracking measures it."""
        dR = self.A @ (pose["R"] @ self.neutral_R.T) @ self.A.T
        return np.degrees(h3.matrix_to_rpy(dR))

    def _start_range(self, t):
        self.step, self.buf = self.RANGE_STEPS[0][0], deque()
        self._range_t0, self._peak, self._lost_t, self._recent = t, 0.0, None, deque(maxlen=3)

    def _next_range(self, t, i):
        name, axis, which = self.RANGE_STEPS[i]
        key = ("roll", "pitch")[axis]
        if abs(self._peak) >= self.RANGE_MIN_DEG:
            self.range[key][int(self._peak > 0)] = abs(self._peak)
            if which == "a":
                self._first_sign[axis] = np.sign(self._peak)
            self.msg = f"{key} {'+' if self._peak > 0 else '-'}{abs(self._peak):.0f} deg"
        else:
            self.msg = f"{key}: no range measured (timeout), default gain"
        if i + 1 < len(self.RANGE_STEPS):
            self.step = self.RANGE_STEPS[i + 1][0]
            self._range_t0, self._peak, self._lost_t, self._recent = t, 0.0, None, deque(maxlen=3)
        else:
            self.step = "FINAL"

    def _feed_range(self, pose, t):
        i = [s[0] for s in self.RANGE_STEPS].index(self.step)
        _, axis, which = self.RANGE_STEPS[i]
        timeout = t - self._range_t0 > self.RANGE_TIMEOUT_S
        if pose is None or pose["mode"] != "stereo":
            if self._lost_t is None:
                self._lost_t = t
            if (abs(self._peak) >= self.RANGE_MIN_DEG and t - self._lost_t >= self.RANGE_LOST_S) or timeout:
                self._next_range(t, i)   # tracking dropped at the edge: the peak so far is the range
            return
        self._lost_t = None
        a = self._hand_rpy(pose)[axis]
        self._recent.append(a)
        a_med = float(np.median(self._recent))  # one-frame spikes don't set the peak
        want = self._first_sign.get(axis) if which == "b" else None  # "b": opposite of "a"
        ok_dir = want is None or np.sign(a_med) == -want
        if ok_dir and facing_weight(pose["p"], pose["R"])[1] < FACING_DEG[0] \
                and abs(a_med) > abs(self._peak):
            self._peak = a_med
        came_back = abs(self._peak) >= self.RANGE_MIN_DEG and abs(a_med) < self.RANGE_BACK * abs(self._peak)
        if came_back or timeout:
            self._next_range(t, i)

    def feed(self, pose, t):
        """Returns True once calibration is complete (self.result is set)."""
        if self.step in [s[0] for s in self.RANGE_STEPS]:
            self._feed_range(pose, t)
            return False
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
            self.neutral_R = h3.project_to_so3(np.mean([q["R"] for q in poses], axis=0))
            self.bones = np.median([h3.bone_lengths(q["P"]) for q in poses], axis=0)
            self.template = h3.palm_template(P)
            curls = [q["curl"] for q in poses if q["curl"] is not None]  # None: fingertips dropped
            self.open_curl = float(np.median(curls)) if curls else None
            if self.frame == "fit":  # neutral rotation measured the way tracking will measure it
                self.neutral_R = h3.project_to_so3(np.mean(
                    [h3.palm_fit(q["P"][h3.PALM], self.template)[1] for q in poses], axis=0))
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
            if self.range_steps:
                self._start_range(t)
            else:
                return self._done()
        elif self.step == "FINAL" and np.linalg.norm(pose["p"] - self.neutral_p) < self.RETURN_M \
                and np.max(np.abs(self._hand_rpy(pose)[:2])) < self.FINAL_DEG:
            return self._done()
        return False

    def _done(self):
        self.step = "DONE"
        self.result = dict(neutral_p=self.neutral_p, neutral_R=self.neutral_R,
                           bones=self.bones, template=self.template, A=self.A,
                           frame=self.frame, open_curl=self.open_curl,
                           range_roll=[np.nan if v is None else v for v in self.range["roll"]],
                           range_pitch=[np.nan if v is None else v for v in self.range["pitch"]])
        return True


def save_hand(res, path=HAND_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        json.dump({k: np.asarray(v).tolist() for k, v in res.items()}, f, indent=1)


def load_hand(path=HAND_PATH):
    with open(path) as f:
        return {k: np.array(v) for k, v in json.load(f).items()}


def _selftest():
    """Synthetic hand through all steps, incl. a roll that loses tracking near the
    edge (peak so far counts), a wrong-way tilt (ignored) and a skipped tilt (timeout)."""
    import math
    rng = np.random.default_rng(0)
    hand0, A = h3._synthetic_hand(), h3.robot_axes([0, 0, 1])
    base, c = np.array([0.0, 0.0, 0.55]), h3.palm_center(h3._synthetic_hand())
    cal, t = HandCalibrator("fit"), [0.0]

    def pose(p=base, rpy=(0, 0, 0)):
        P = (hand0 - c) @ (A.T @ h3.rpy_to_matrix(*np.radians(rpy)) @ A).T + p
        P = P + rng.normal(0, 0.0015, P.shape)
        o, R = h3.hand_frame(P)
        if hasattr(cal, "template"):
            o, R, _ = h3.palm_fit(P[h3.PALM], cal.template)
        return dict(P=P, p=o, R=R, mode="stereo", curl=h3.curl(P))

    def run(secs, f):
        for k in range(int(secs * 30)):
            cal.feed(f(k / (secs * 30)), t[0])
            t[0] += 1 / 30

    def sweep(axis, peak, lose=False):
        def f(s):
            a = peak * math.sin(math.pi * s)
            if lose and abs(a) > 0.9 * abs(peak):
                return None
            return pose(rpy=[a if i == axis else 0 for i in range(3)])
        return f

    run(2.5, lambda s: pose())
    run(1.5, lambda s: pose(base + [0, 0, 0.12 * min(1, 2 * s)]))
    run(1.0, lambda s: pose())
    assert cal.step == "ROLL", cal.step
    run(3.0, sweep(0, -40, lose=True))    # clockwise = -roll here; lost near the edge: -36 counts
    assert cal.step == "TILT_A", cal.step
    run(3.0, sweep(1, 45))
    run(3.0, sweep(1, 30))                # same direction again: TILT_B ignores it
    assert cal.step == "TILT_B", cal.step
    run(13.0, lambda s: pose())           # never tilted the other way: timeout
    run(1.0, lambda s: pose())
    assert cal.step == "DONE", cal.step
    r, p = cal.result["range_roll"], cal.result["range_pitch"]
    assert np.isnan(r[1]) and 30 < r[0] < 37 and np.isnan(p[0]) and abs(p[1] - 45) < 3, (r, p)
    print(f"hand_calib self-test OK (roll clockwise = -{r[0]:.0f}, tilt down {p[1]:.0f} deg)")


if __name__ == "__main__":
    _selftest()
