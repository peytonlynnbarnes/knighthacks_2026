"""Make `python script.py` show up as its own macOS app: a Dock icon, an entry in Cmd-Tab,
its OpenCV windows coming to the front.

Why a relaunch: a process started from a shell belongs to the terminal app (macOS's
"responsible process"; lsappinfo shows it with bundleID com.apple.Terminal), so its
cv2.imshow windows are folded into Terminal: no Dock icon of their own, Cmd-Tab goes to
Terminal. Unsetting __CFBundleIdentifier or setting the activation policy doesn't change
that (tried 2026-10-11). Only a process launched by LaunchServices from a bundle gets its
own identity, so relaunch_in_app() re-runs the same command through a small generated
app bundle (~/Library/Application Support/kh/<name>.app, outside iCloud):

    open -W -n <name>.app --stdout <this tty> --stderr <this tty> --args <python argv>

The bundle's script restores this shell's environment (conda, ROS), the working directory,
and execs the same python (linked into the bundle). Output stays in the terminal; Ctrl-C here is forwarded to the
app. The app asks for camera access once (its own entry in Privacy & Security > Camera).
register() (in the app) sets the Dock name / icon and brings it to the front. Plain ctypes
on the Objective-C runtime: no PyObjC.

    import macos_app
    macos_app.relaunch_in_app("Hand Tracker", "com.kh.handtracker")  # first thing in main()
    ...
    macos_app.register("Hand Tracker")  # main thread, before the first cv2.imshow

No-ops on other platforms; never raises over cosmetics (a failure prints one line).
"""
import ctypes
import ctypes.util
import os
import plistlib
import shlex
import signal
import subprocess
import sys
import tempfile

IN_APP_ENV = "KH_IN_APP"  # set inside the relaunched app
APP_DIR = os.path.expanduser("~/Library/Application Support/kh")
# environment variables not carried into the app (the terminal's identity / session)
SKIP_ENV = ("__CF", "XPC_", "TERM_SESSION_ID", "SECURITYSESSIONID", "LaunchInstanceID")

_objc = None


def _setup():
    global _objc, _send_addr
    if _objc is None:
        _objc = ctypes.cdll.LoadLibrary(ctypes.util.find_library("objc"))
        ctypes.cdll.LoadLibrary("/System/Library/Frameworks/AppKit.framework/AppKit")
        _objc.objc_getClass.restype = ctypes.c_void_p
        _objc.objc_getClass.argtypes = [ctypes.c_char_p]
        _objc.sel_registerName.restype = ctypes.c_void_p
        _objc.sel_registerName.argtypes = [ctypes.c_char_p]
        _send_addr = ctypes.cast(_objc.objc_msgSend, ctypes.c_void_p).value


def _msg(obj, sel, *args, restype=ctypes.c_void_p, argtypes=()):
    """objc_msgSend with an exact prototype (required on arm64: it isn't variadic there)."""
    fn = ctypes.CFUNCTYPE(restype, ctypes.c_void_p, ctypes.c_void_p, *argtypes)(_send_addr)
    return fn(obj, _objc.sel_registerName(sel.encode()), *args)


def _cls(name):
    return _objc.objc_getClass(name.encode())


def _nsstring(s):
    return _msg(_cls("NSString"), "stringWithUTF8String:", s.encode(), argtypes=(ctypes.c_char_p,))


def _icon_png():
    """A 256 px icon (hand skeleton on a dark rounded square), written once to the temp dir."""
    path = os.path.join(tempfile.gettempdir(), "kh_hand_tracker_icon.png")
    if os.path.exists(path):
        return path
    import cv2
    import numpy as np
    s = 256
    img = np.zeros((s, s, 4), np.uint8)  # transparent, then a rounded square
    cv2.rectangle(img, (48, 8), (s - 48, s - 8), (40, 40, 40, 255), -1)
    cv2.rectangle(img, (8, 48), (s - 8, s - 48), (40, 40, 40, 255), -1)
    for cx, cy in ((48, 48), (s - 48, 48), (48, s - 48), (s - 48, s - 48)):
        cv2.circle(img, (cx, cy), 40, (40, 40, 40, 255), -1, cv2.LINE_AA)
    # 21 landmarks of an open right hand (MediaPipe / OpenPose order), palm facing the viewer
    pts = [(128, 212), (98, 196), (78, 172), (64, 150), (52, 130), (100, 130), (94, 96), (90, 74),
           (88, 54), (124, 124), (124, 86), (124, 62), (124, 40), (146, 128), (152, 94), (156, 72),
           (158, 52), (166, 138), (178, 112), (186, 94), (192, 78)]
    bones = [(0, 1), (1, 2), (2, 3), (3, 4), (0, 5), (5, 6), (6, 7), (7, 8), (5, 9), (9, 10), (10, 11),
             (11, 12), (9, 13), (13, 14), (14, 15), (15, 16), (13, 17), (0, 17), (17, 18), (18, 19), (19, 20)]
    for a, b in bones:
        cv2.line(img, pts[a], pts[b], (80, 230, 80, 255), 7, cv2.LINE_AA)
    for p in pts:
        cv2.circle(img, p, 8, (60, 60, 240, 255), -1, cv2.LINE_AA)
    cv2.imwrite(path, img)
    return path


def register(name="Hand Tracker"):
    """Regular app with `name` and the hand icon, brought to the front. Main thread only."""
    if sys.platform != "darwin":
        return
    try:
        _setup()
        app = _msg(_cls("NSApplication"), "sharedApplication")
        _msg(app, "setActivationPolicy:", 0, restype=ctypes.c_bool, argtypes=(ctypes.c_long,))  # Regular
        info = _msg(_cls("NSProcessInfo"), "processInfo")
        _msg(info, "setProcessName:", _nsstring(name), restype=None, argtypes=(ctypes.c_void_p,))
        img = _msg(_msg(_cls("NSImage"), "alloc"), "initWithContentsOfFile:", _nsstring(_icon_png()),
                   argtypes=(ctypes.c_void_p,))
        if img:
            _msg(app, "setApplicationIconImage:", img, restype=None, argtypes=(ctypes.c_void_p,))
        _msg(app, "activateIgnoringOtherApps:", True, restype=None, argtypes=(ctypes.c_bool,))
    except Exception as e:  # cosmetic: never stop the program over it
        print(f"macos_app: couldn't register as an app ({e})", flush=True)


def _bundle(name, bundle_id):
    """Write the app bundle if it's missing or different (an unchanged bundle keeps its
    camera permission). Returns its path."""
    app = os.path.join(APP_DIR, f"{name}.app")
    files = {
        os.path.join(app, "Contents", "Info.plist"): plistlib.dumps({
            "CFBundleIdentifier": bundle_id, "CFBundleName": name, "CFBundleDisplayName": name,
            "CFBundleExecutable": "run", "CFBundlePackageType": "APPL", "CFBundleVersion": "1",
            "LSMinimumSystemVersion": "13.0", "NSHighResolutionCapable": True,
            "NSCameraUsageDescription": "Stereo hand tracking for the SO-ARM101 (two webcams).",
        }),
        os.path.join(app, "Contents", "MacOS", "run"): (
            "#!/bin/bash\n"
            "# generated by Computer-Vision/common/macos_app.py: the shell's environment, its\n"
            "# working directory, then the same python command line\n"
            'source "$KH_ENV_FILE" && rm -f "$KH_ENV_FILE"\n'
            'echo $$ > "$KH_PID_FILE"\n'
            '# python through a link inside the bundle: macOS finds the bundle (its name and the\n'
            '# camera usage text) from the executable path; exec of the conda python directly\n'
            '# left camera access "not determined" with no prompt\n'
            'exec "$(dirname "$0")/python" "$@"\n').encode(),
    }
    for path, data in files.items():
        try:
            with open(path, "rb") as f:
                if f.read() == data:
                    continue
        except OSError:
            pass
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(data)
        if path.endswith("run"):
            os.chmod(path, 0o755)
    link = os.path.join(app, "Contents", "MacOS", "python")
    if os.path.realpath(link) != os.path.realpath(sys.executable):
        if os.path.lexists(link):
            os.remove(link)
        os.symlink(os.path.realpath(sys.executable), link)
    return app


def relaunch_in_app(name, bundle_id):
    """Re-run this command as its own app (see the module docstring) and exit with it.
    Returns (does nothing) when not on macOS, already in the app, or without a terminal
    to show the output in."""
    if sys.platform != "darwin" or os.environ.get(IN_APP_ENV):
        return
    try:
        tty = os.ttyname(sys.stdout.fileno())
    except OSError:
        return  # piped / redirected output: stay a plain process
    try:
        app = _bundle(name, bundle_id)
        fd, env_file = tempfile.mkstemp(prefix="kh_env_", suffix=".sh")
        pid_file = env_file[:-3] + ".pid"
        env = {k: v for k, v in os.environ.items() if not k.startswith(SKIP_ENV) and k.isidentifier()}
        env[IN_APP_ENV] = "1"
        with os.fdopen(fd, "w") as f:
            f.writelines(f"export {k}={shlex.quote(v)}\n" for k, v in env.items())
            f.write(f"cd {shlex.quote(os.getcwd())}\n")
        cmd = ["open", "-W", "-n", app, "--stdout", tty, "--stderr", tty,
               "--env", f"KH_ENV_FILE={env_file}", "--env", f"KH_PID_FILE={pid_file}",
               "--args", *sys.argv]
        proc = subprocess.Popen(cmd)
    except Exception as e:
        print(f"macos_app: couldn't start as an app ({e}); running in the terminal", flush=True)
        return
    print(f"running as the '{name}' app (Dock, Cmd-Tab); Ctrl-C here stops it, --no-app runs "
          "in the terminal instead", flush=True)

    def forward(sig, _frame):
        try:
            with open(pid_file) as f:
                os.kill(int(f.read().strip()), sig)
        except (OSError, ValueError):
            proc.send_signal(sig)  # app not started yet: stop the launch

    for sig in (signal.SIGINT, signal.SIGTERM, signal.SIGHUP):
        signal.signal(sig, forward)
    while True:
        try:
            rc = proc.wait()
            break
        except KeyboardInterrupt:
            continue
    for p in (pid_file, env_file):
        try:
            os.remove(p)
        except OSError:
            pass
    sys.exit(rc)
