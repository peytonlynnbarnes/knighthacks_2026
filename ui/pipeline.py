"""The three pipeline processes behind the app window (ui/app.py).

start() launches, each in its own process group, with the app's python / environment:
    ik       ros2 run so101_mirror ik_node.py
    arm      control/feetech_mirror.py --gripper ... (--live --yes unless dry run)
    hand     Computer-Vision/common/hand_target_node.py ... --ui-port <port>
The hand node connects back to a localhost socket here (Computer-Vision/common/ui_link.py)
and streams its preview (both cameras side by side, hands drawn) plus status lines.
calibrate() sends it 'c' (or starts everything with --calibrate when stopped).
stop() asks the hand node to quit ('q'), then Ctrl-C (SIGINT) to feetech_mirror (it
leaves the arm holding its pose) and the IK; SIGTERM / SIGKILL only if one hangs.
torque_off() makes the arm go limp (it DROPS; the app confirms first): while running it
stops like stop() but sends feetech_mirror SIGUSR1 (exit with torque OFF) instead of
Ctrl-C; when stopped, or if the driver didn't say it released, control/torque_off.py --yes.
If any process exits on its own, the others are stopped and its last output line is
shown. Children's output goes to this terminal, prefixed with their name.
"""
import os
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
from collections import deque

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
COMMON = os.path.join(ROOT, "Computer-Vision", "common")
sys.path.insert(0, COMMON)

from ui_link import recv_frame  # noqa: E402

ERROR_S = 8.0     # how long error notes stay up
QUIT_S = 6.0      # wait this long for each stop step ('q' / SIGINT, SIGTERM) before the next
PREFIX_W = 5
NAMES = {"ik": "IK node", "arm": "Arm driver", "hand": "Hand tracker"}


class Pipeline:
    """latest() -> (frame, seq, status, state) for the UI to poll.

    frame: the hand node's preview (BGR, both views side by side) or None.
    status: None or (text, detail, kind), kind "calib" | "track" | "hold" | "lost" | "error" |
    "info"; arm: None or (text, alarm) from feetech_mirror's /hw/status.
    state: "idle" | "starting" | "running" | "stopping".
    """

    def __init__(self, hand_args, arm_args):
        self.hand_args, self.arm_args = list(hand_args), list(arm_args)
        self.lock = threading.Lock()
        self.frame, self.seq, self.status, self.status_until, self.arm = None, 0, None, 0.0, None
        self.state = "idle"
        self.procs = {}           # name -> Popen
        self.tails = {}           # name -> last output lines
        self.conn = None          # the hand node's socket
        self.streaming = False    # its first frame came: it reads 'q' (before, it's loading)
        self.gen = 0              # start() count: threads of an old run stop touching state
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind(("127.0.0.1", 0))
        self.server.listen(1)
        self.port = self.server.getsockname()[1]
        threading.Thread(target=self._accept_loop, daemon=True).start()

    # ---- UI calls --------------------------------------------------------------------------

    def latest(self):
        with self.lock:
            if self.status and self.status[2] == "error" and time.monotonic() > self.status_until:
                self.status = None
            return self.frame, self.seq, self.status, self.arm, self.state

    def alive(self):
        """{name: still running} for the started processes (empty when stopped)."""
        with self.lock:
            procs = dict(self.procs)
        return {name: p.poll() is None for name, p in procs.items()}

    def start(self, calibrate=False):
        with self.lock:
            if self.state != "idle":
                return
            self.state, self.gen = "starting", self.gen + 1
            self.frame, self.arm, self.streaming = None, None, False
        err = self._preflight()
        if err:
            self._set_state("idle")
            self._set_status(*err, kind="error")
            return
        gen = self.gen
        hand = [sys.executable, "-u", os.path.join(COMMON, "hand_target_node.py"), *self.hand_args,
                "--ui-port", str(self.port)] + (["--calibrate"] if calibrate else [])
        arm = [sys.executable, "-u", os.path.join(ROOT, "control", "feetech_mirror.py"), *self.arm_args]
        ik = ["ros2", "run", "so101_mirror", "ik_node.py"]
        self._set_status("Starting…", "IK, arm driver and hand tracker", "info")
        for name, cmd in (("ik", ik), ("arm", arm), ("hand", hand)):
            print(f"[ui] start {name}: {' '.join(cmd)}", flush=True)
            try:
                p = subprocess.Popen(cmd, cwd=ROOT, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE,
                                     stderr=subprocess.STDOUT, start_new_session=True,
                                     env=dict(os.environ, PYTHONUNBUFFERED="1"))
            except OSError as e:
                self._set_status(f"Can't start {name}", str(e), "error")
                self.stop()
                return
            self.procs[name], self.tails[name] = p, deque(maxlen=20)
            threading.Thread(target=self._pump, args=(name, p, gen), daemon=True).start()

    def calibrate(self):
        with self.lock:
            state, conn = self.state, self.conn
        if state == "idle":
            self.start(calibrate=True)
        elif conn is not None:
            self._send(b"c")
            print("[ui] hand recalibration requested", flush=True)

    def stop(self, release=False):
        with self.lock:
            if self.state in ("idle", "stopping"):
                return
            self.state = "stopping"
        threading.Thread(target=self._stop_all, args=(release,), daemon=True).start()

    def torque_off(self):
        """Arm limp (it drops). Running: stop with the driver releasing; stopped: torque_off.py."""
        if "--no-bus" in self.arm_args:
            self._set_status("No arm attached", "started with --no-bus: nothing to turn off", "error")
            return
        with self.lock:
            state = self.state
            if state == "idle":
                self.state = "stopping"  # buttons off while torque_off.py runs
        if state in ("starting", "running"):
            self.stop(release=True)
        elif state == "idle":
            threading.Thread(target=self._torque_off_script, daemon=True).start()

    def close(self):
        """App quitting: stop everything and wait for it (bounded)."""
        self.stop()
        t_end = time.monotonic() + 3 * QUIT_S
        while self.state != "idle" and time.monotonic() < t_end:
            time.sleep(0.05)
        self.server.close()

    # ---- internals -------------------------------------------------------------------------

    def _preflight(self):
        if shutil.which("ros2") is None:
            return ("ROS 2 isn't available", "conda activate ros2_jazzy first")
        if "so101_mirror" not in os.environ.get("AMENT_PREFIX_PATH", ""):
            return ("The so101_mirror package isn't sourced", "source ros_ws/install/setup.zsh first")
        return None

    def _set_status(self, text, detail="", kind="info", secs=ERROR_S):
        with self.lock:
            self.status = (text, detail, kind)
            self.status_until = time.monotonic() + secs

    def notice(self, text, detail="", secs=ERROR_S):
        """A note in the error style for `secs` (e.g. the Torque Off confirmation)."""
        self._set_status(text, detail, "error", secs)

    def clear_notice(self, text):
        with self.lock:
            if self.status and self.status[0] == text:
                self.status = None

    def _torque_off_script(self, why=""):
        """control/torque_off.py --yes on every servo (the driver isn't running to do it)."""
        cmd = [sys.executable, "-u", os.path.join(ROOT, "control", "torque_off.py"), "--yes"]
        if "--port" in self.arm_args:
            cmd += self.arm_args[self.arm_args.index("--port"):][:2]
        print(f"[ui] torque off{why}: {' '.join(cmd)}", flush=True)
        try:
            r = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, timeout=10)
            out = (r.stdout + r.stderr).strip()
            ok = r.returncode == 0 and "FAILED" not in out
        except (OSError, subprocess.TimeoutExpired) as e:
            out, ok = str(e), False
        for line in out.splitlines():
            print(f"[off] {line}", flush=True)
        if ok:
            self._set_status("Torque off: the arm is limp", "Start homes it again", "info")
        else:
            self._set_status("Torque off FAILED", (out.splitlines() or [""])[-1][-160:], "error")
        self._set_state("idle")

    def _set_state(self, s):
        with self.lock:
            self.state = s

    def _send(self, b):
        try:
            self.conn.sendall(b)
        except (OSError, AttributeError):
            pass

    def _pump(self, name, p, gen):
        """Print a child's output; if it exits while running, stop the rest and say why."""
        tail = self.tails[name]
        for raw in p.stdout:
            line = raw.decode(errors="replace").rstrip()
            tail.append(line)
            print(f"[{name:<{PREFIX_W - 2}}] {line}", flush=True)
        code = p.wait()
        with self.lock:
            unexpected = gen == self.gen and self.state in ("starting", "running")
        if unexpected:
            last = next((ln for ln in reversed(tail) if ln.strip()), "")
            print(f"[ui] {name} exited ({code}); stopping the pipeline", flush=True)
            self._set_status(f"{NAMES[name]} stopped (exit {code})", last[-160:], "error")
            self.stop()

    def _accept_loop(self):
        while True:
            try:
                conn, _ = self.server.accept()
            except OSError:
                return
            conn.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            with self.lock:
                old, self.conn, gen = self.conn, conn, self.gen
            if old is not None:
                old.close()
            threading.Thread(target=self._recv_loop, args=(conn, gen), daemon=True).start()

    def _recv_loop(self, conn, gen):
        try:
            while True:
                meta, img = recv_frame(conn)
                with self.lock:
                    if gen != self.gen:
                        break
                    if img is None:
                        if "loading" in meta:
                            self.status = (meta["loading"], "", "info")
                        continue
                    self.streaming = True
                    if self.state == "starting":  # first frame: WiLoR, cameras, ROS all up
                        self.state = "running"
                    self.frame = np.frombuffer(img, np.uint8).reshape(meta["h"], meta["w"], 3)
                    self.seq += 1
                    lines = [ln for ln in meta.get("lines", []) if ln]
                    if not (self.status and self.status[2] == "error" and time.monotonic() < self.status_until):
                        self.status = (lines[0] if lines else "", "\n".join(lines[1:]), meta.get("kind", "hold"))
                    self.arm = meta.get("arm")
        except (OSError, ConnectionError, ValueError):
            pass
        conn.close()

    def _stop_all(self, release=False):
        print(f"[ui] stopping the pipeline{', torque OFF' if release else ''}", flush=True)
        # the hand node finishes its pair and closes cameras / ROS on 'q' (Ctrl-C until its
        # first frame: while loading it doesn't read the socket); Ctrl-C: feetech_mirror holds
        # the arm's pose, SIGUSR1: it turns torque off instead
        with self.lock:
            streaming = self.streaming
        self._send(b"q")
        for name in ("arm", "ik") if streaming else ("hand", "arm", "ik"):
            self._signal(name, signal.SIGUSR1 if release and name == "arm" else signal.SIGINT)
        for name in ("hand", "arm", "ik"):
            p = self.procs.get(name)
            for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGKILL):
                try:
                    p is not None and p.wait(QUIT_S)
                    break
                except subprocess.TimeoutExpired:
                    print(f"[ui] {name} didn't stop: {sig.name}", flush=True)
                    self._signal(name, sig)
        # the driver prints "torque OFF" once it released; it may not have (killed, still
        # starting up: SIGUSR1's default action ends it as is), so then do it from here
        def released():
            return any("torque OFF" in ln for ln in list(self.tails.get("arm", ())))

        t_end = time.monotonic() + 1.0  # its output thread may still be reading the last lines
        while release and not released() and time.monotonic() < t_end:
            time.sleep(0.05)
        released = released()
        with self.lock:
            self.procs, self.frame, self.arm = {}, None, None
            if not release:
                self.state = "idle"
            if self.conn is not None:
                self.conn.close()
                self.conn = None
            if not (self.status and self.status[2] == "error"):
                self.status = None
        print("[ui] stopped", flush=True)
        if release and released:
            self._set_status("Torque off: the arm is limp", "Start homes it again", "info")
            self._set_state("idle")
        elif release:
            self._torque_off_script(" (the driver didn't)")

    def _signal(self, name, sig):
        p = self.procs.get(name)
        if p is not None and p.poll() is None:
            try:
                os.killpg(p.pid, sig)
            except ProcessLookupError:
                pass
