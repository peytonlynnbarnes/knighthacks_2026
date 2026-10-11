#!/usr/bin/env python3
"""Mimisis: the app window for the whole hand -> arm pipeline.

    conda activate ros2_jazzy && source ros_ws/install/setup.zsh
    python ui/app.py                 # LIVE: Start homes the arm, then it follows the hand
    python ui/app.py --dry-run       # feetech_mirror dry run (reads the arm, no torque)
    python ui/app.py --dry-run --no-bus   # no arm attached
    python ui/app.py -- --timing     # anything after -- goes to hand_target_node.py
    q / Esc quits (stops everything first). Ctrl-C in the terminal too.

Started from a terminal, it re-runs itself as the "Mimisis" app (its own Dock icon, Cmd-Tab
entry and camera permission; output stays in the terminal; macos_app.py, as the hand
tracker does); --no-app runs it in the terminal's process.

One Start / Stop button. Record (while running) records the arm's path: what the hand
commands, with its timing; click again to stop. Play sends the last take back to the arm
(it glides to the start first, then back to the hand when done; click again to stop).
The take is saved and comes back at the next start (Computer-Vision/common/path_record.py). Start (idle) runs the IK node, feetech_mirror.py (--gripper,
live unless --dry-run) and hand_target_node.py (two hands; the saved hand calibration
if there is one, else the calibration steps show in the status bar); the button then
turns into Stop, which ends all three (the arm holds its pose). Calibrate redoes the hand calibration (starting everything if stopped). Torque
Off makes the arm go limp and DROP: click it twice within a few seconds (support the
arm first); it stops everything, and Start homes the arm again. The
camera box shows the hand node's preview: both cameras side by side, the right hand
green, the left orange, the palm axes. See pipeline.py.

Look: an enchanted forest at night (the hackathon theme: purples and greens). Deep plum
sky, a lilac glow behind the camera card; the original vines from the top corners
(vines.py) over pixel-art scenery (forest.py): mushrooms bottom left, flowers bottom
right, stars, fireflies, trunks at the edges.
Cinzel Decorative for the name, Nunito for the rest (fixed-width digits in readouts); both
bundled in ui/fonts (OFL). --contrast prints the text contrast ratios.

Layout: the logo + name in the top margin; the camera card (preview, keeping its aspect
ratio, with the status bar attached under it: state, readout, arm driver, process dots,
LIVE / DRY RUN) and the button row centered below: Start/Stop, Calibrate, Record, Play,
and Torque Off in red.
"""
import argparse
import os
import random
import signal
import sys
import time

import forest
from pipeline import Pipeline

HERE = os.path.dirname(os.path.abspath(__file__))

# The cv2 pip wheel points Qt at its bundled Qt5 plugins and fonts on import;
# PySide6 is Qt6 and must not pick those up.
for _var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
    if "cv2" in os.environ.get(_var, ""):
        del os.environ[_var]
# The conda env's bin/qt.conf sends Qt to the env's own (Qt5) plugins: "Could not find the
# Qt platform plugin cocoa". Use the plugins that came with the PySide6 wheel.
import PySide6  # noqa: E402

_plugins = os.path.join(os.path.dirname(PySide6.__file__), "Qt", "plugins")
if os.path.isdir(_plugins):
    os.environ["QT_PLUGIN_PATH"] = _plugins
    os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = os.path.join(_plugins, "platforms")

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer  # noqa: E402
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QFontMetricsF, QIcon, QImage,  # noqa: E402
                           QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QRadialGradient)
from PySide6.QtWidgets import QAbstractButton, QApplication, QWidget  # noqa: E402

APP_NAME = "Mimisis"
FONT_PATH = os.path.join(HERE, "fonts", "CinzelDecorative-Bold.ttf")  # the name only
UI_FONT_PATH = os.path.join(HERE, "fonts", "Nunito-Variable.ttf")  # everything else
LOGO_PATH = os.path.join(HERE, "logo.png")  # white with green vines (title, window + Dock icon)

# Enchanted forest at night (the hackathon theme: purples and greens). Text/fill pairs
# are >= 4.5:1 contrast (python ui/app.py --contrast prints them).
NIGHT_TOP = QColor("#140A23")     # background gradient, top to bottom
NIGHT_MID = QColor("#2B1747")
NIGHT_BOTTOM = QColor("#190D2C")
GLOW = QColor(160, 115, 225, 70)  # moonlight behind the camera card
CARD = QColor("#221338")          # camera card / status bar
CARD_EDGE = QColor(198, 171, 234, 110)
TEXT = QColor("#F5EFFF")
MUTED = QColor("#C2B2DF")         # secondary text
LILAC = QColor("#C6ABEA")         # outline button, calibrating
PLUM = QColor("#6A4699")          # Stop button
GRASS = QColor("#8FCF68")         # Start button, tracking, vine leaves
MOSS = QColor("#10230A")          # text on GRASS
EMBER = QColor("#C73A55")         # danger: Torque Off, errors, LIVE
FIREFLY = QColor("#F4C95D")       # warnings: hand lost, confirm, DRY RUN
FIREFLY_TEXT = QColor("#2A1A00")  # text on FIREFLY
WHITE = QColor("#FFFFFF")
SHADOW = QColor(6, 2, 14)
KIND_COLOR = {"calib": LILAC, "hold": MUTED, "lost": FIREFLY, "info": LILAC, "track": GRASS, "error": EMBER}
KIND_LABEL = {"idle": "Idle", "starting": "Starting…", "running": "Waiting for a hand", "stopping": "Stopping…"}

# Base sizes at a 1400 x 820 window; everything scales from these.
BUTTON_SIZE = (190, 60)  # five in a row
BUTTON_GAP = 28          # even gaps: the row is symmetric about the window's center
STATUS_H = 58            # status bar under the camera
STACK_GAP = 30           # status bar -> buttons
BOTTOM_MARGIN = 30
TORQUE_CONFIRM_MS = 4000  # Torque Off: the second click must come within this
RUN_SWITCH_GUARD_S = 0.5  # Start/Stop ignores clicks this long after it switched
TORQUE_WARNING = "Click Torque Off again to drop the arm"
PREVIEW_ASPECT = 32 / 9  # two 16:9 views side by side, until the first frame says otherwise

# hand_target_node.py settings for this setup (CLAUDE.md "Run"); extra ones go after --
HAND_ARGS = ["--cams", "1", "0", "--two-hands", "--z-min", "-0.16", "--grip-closed-curl", "0.75"]
ARM_ARGS = ["--gripper", "--max-speed", "800", "--lift-speed", "450"]


def brand_font(px):
    """Cinzel Decorative: the app name."""
    f = QFont(brand_font.family)
    f.setBold(True)
    f.setPixelSize(max(1, round(px)))
    return f


def app_font(px, weight=QFont.Bold, digits=False):
    """Nunito: buttons and text. digits=True: fixed-width figures (readouts don't jitter)."""
    f = QFont(app_font.family)
    f.setWeight(weight)
    f.setPixelSize(max(1, round(px)))
    if digits:
        f.setFeature(QFont.Tag("tnum"), 1)
    return f


brand_font.family = "Cinzel Decorative"
app_font.family = "Nunito"


def load_fonts():
    """Register the bundled fonts (after the QApplication exists); system font if one fails."""
    for path, fn in ((FONT_PATH, brand_font), (UI_FONT_PATH, app_font)):
        fid = QFontDatabase.addApplicationFont(path)
        if fid >= 0:
            fn.family = QFontDatabase.applicationFontFamilies(fid)[0]
        else:
            print(f"warning: couldn't load {path}; using the system font", file=sys.stderr)


def with_alpha(c, a):
    c = QColor(c)
    c.setAlphaF(a)
    return c


def contrast(fg, bg):
    """WCAG contrast ratio of two opaque colors."""
    def lum(c):
        ch = [x / 255 for x in (c.red(), c.green(), c.blue())]
        ch = [x / 12.92 if x <= 0.03928 else ((x + 0.055) / 1.055) ** 2.4 for x in ch]
        return 0.2126 * ch[0] + 0.7152 * ch[1] + 0.0722 * ch[2]
    a, b = sorted((lum(fg), lum(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


CONTRAST_PAIRS = [("text on card", TEXT, CARD), ("muted on card", MUTED, CARD), ("text on night", TEXT, NIGHT_MID),
                  ("Start", MOSS, GRASS), ("Stop", WHITE, PLUM), ("Calibrate", LILAC, NIGHT_MID),
                  ("Torque Off", WHITE, EMBER), ("Confirm?", FIREFLY_TEXT, FIREFLY)]


def soft_shadow(p, rect, radius, spread, alpha=0.30, dy=0.0, steps=10):
    """Blurred-looking shadow outside `rect`: stacked translucent rounded rects."""
    outside = QPainterPath()
    outside.addRect(rect.adjusted(-2 * spread, -2 * spread, 2 * spread, 2 * spread + dy))
    inside = QPainterPath()
    inside.addRoundedRect(rect, radius, radius)
    p.save()
    p.setClipPath(outside.subtracted(inside))
    p.setPen(Qt.NoPen)
    for i in range(steps):
        g = spread * (i + 1) / steps
        p.setBrush(with_alpha(SHADOW, alpha / steps))
        p.drawRoundedRect(rect.adjusted(-g, -g + dy, g, g + dy), radius + g, radius + g)
    p.restore()


def draw_icon(p, kind, r, color, width):
    """Line icons in the square `r`: play, stop, target, power, record."""
    p.save()
    pen = QPen(color, width, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin)
    c, k = r.center(), r.width() / 2
    if kind == "play":
        path = QPainterPath(QPointF(c.x() - 0.55 * k, c.y() - 0.75 * k))
        path.lineTo(c.x() + 0.8 * k, c.y())
        path.lineTo(c.x() - 0.55 * k, c.y() + 0.75 * k)
        path.closeSubpath()
        p.setPen(pen)
        p.setBrush(color)
        p.drawPath(path)
    elif kind == "stop":
        p.setPen(pen)
        p.setBrush(color)
        p.drawRoundedRect(QRectF(c.x() - 0.62 * k, c.y() - 0.62 * k, 1.24 * k, 1.24 * k), 0.2 * k, 0.2 * k)
    elif kind == "target":
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        p.drawEllipse(c, 0.62 * k, 0.62 * k)
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            p.drawLine(QPointF(c.x() + 0.62 * k * dx, c.y() + 0.62 * k * dy),
                       QPointF(c.x() + 1.0 * k * dx, c.y() + 1.0 * k * dy))
        p.setBrush(color)
        p.drawEllipse(c, 0.14 * k, 0.14 * k)
    elif kind == "record":
        p.setPen(pen)
        p.setBrush(color)
        p.drawEllipse(c, 0.55 * k, 0.55 * k)
    elif kind == "power":
        p.setPen(pen)
        p.setBrush(Qt.NoBrush)
        ring = QRectF(c.x() - 0.78 * k, c.y() - 0.7 * k, 1.56 * k, 1.56 * k)
        p.drawArc(ring, 120 * 16, -240 * 16)  # open at the top
        p.drawLine(QPointF(c.x(), c.y() - 1.0 * k), QPointF(c.x(), c.y() + 0.05 * k))
    p.restore()


class ForestButton(QAbstractButton):
    """Rounded button with an icon. role: "primary" (green), "neutral" (plum),
    "outline" (lilac edge) or "danger" (ember); `armed` (Torque Off waiting for its
    second click) turns it firefly amber with a countdown ring (progress 1 -> 0)."""

    STYLE = {"primary": (GRASS, MOSS), "neutral": (PLUM, WHITE), "outline": (None, LILAC),
             "danger": (EMBER, WHITE)}

    def __init__(self, text, icon, role, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.icon, self.role = icon, role
        self.scale, self.armed, self.progress = 1.0, False, 0.0
        self.setCursor(Qt.PointingHandCursor)
        self.setFocusPolicy(Qt.TabFocus)  # keyboard focus only: a click doesn't leave a ring
        self.setAttribute(Qt.WA_Hover)    # repaint on hover enter/leave

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        s = self.scale
        ring = max(2.0, 3.0 * s)  # room for the focus ring
        r = QRectF(self.rect()).adjusted(ring + 1, ring + 1, -ring - 1, -ring - 1)
        radius = 0.32 * r.height()
        fill, ink = self.STYLE[self.role]
        icon = self.icon
        if self.armed:
            fill, ink, icon = FIREFLY, FIREFLY_TEXT, "power"
        if not self.isEnabled():
            p.setOpacity(0.35)

        hover = self.underMouse() and self.isEnabled()
        if fill is None:  # outline
            p.setBrush(with_alpha(LILAC, 0.22 if self.isDown() else 0.12 if hover else 0.0))
            p.setPen(QPen(LILAC, max(1.5, 2.2 * s)))
        else:
            c = fill.darker(112) if self.isDown() else fill.lighter(108) if hover else fill
            grad = QLinearGradient(r.topLeft(), r.bottomLeft())
            grad.setColorAt(0.0, c.lighter(107))
            grad.setColorAt(1.0, c.darker(106))
            p.setBrush(grad)
            p.setPen(QPen(with_alpha(WHITE, 0.25), max(1.0, 1.2 * s)))
        p.drawRoundedRect(r, radius, radius)

        # icon + label, centered together
        font = app_font(0.34 * r.height(), QFont.ExtraBold)
        tw = QFontMetricsF(font).horizontalAdvance(self.text())
        isz = 0.36 * r.height()
        gap = (0.42 if self.armed else 0.22) * r.height()  # armed: room for the countdown ring
        x = r.center().x() - (isz + gap + tw) / 2
        ir = QRectF(x, r.center().y() - isz / 2, isz, isz)
        draw_icon(p, icon, ir, EMBER if icon == "record" and fill is None else ink, max(1.5, 0.09 * isz))
        if self.armed:  # countdown ring around the icon
            p.setPen(QPen(ink, max(1.5, 0.08 * isz), Qt.SolidLine, Qt.RoundCap))
            p.setBrush(Qt.NoBrush)
            p.drawArc(ir.adjusted(-0.3 * isz, -0.3 * isz, 0.3 * isz, 0.3 * isz), 90 * 16,
                      round(-360 * 16 * self.progress))
        p.setFont(font)
        p.setPen(ink)
        p.drawText(QRectF(x + isz + gap, r.top(), tw + 4, r.height()), Qt.AlignVCenter | Qt.AlignLeft, self.text())

        if self.hasFocus():
            p.setOpacity(1.0)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(with_alpha(LILAC, 0.9), max(1.5, 2.0 * s)))
            p.drawRoundedRect(r.adjusted(-ring, -ring, ring, ring), radius + ring, radius + ring)


class MainWindow(QWidget):
    def __init__(self, pipeline, live, has_bus=True):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        logo = QImage(LOGO_PATH)
        self.logo = None if logo.isNull() else logo
        self.setMinimumSize(720, 480)
        self.pipeline, self.live = pipeline, live
        self.frame, self.seq, self.status, self.arm, self.state = None, 0, None, None, "idle"
        self.alive = {}
        self.aspect = PREVIEW_ASPECT
        self.bg = None
        self.cam_rect = QRectF()   # the camera image
        self.bar_rect = QRectF()   # the status bar under it (one card with the image)
        self.s = 1.0

        # one Start / Stop button: what it does (and looks like) follows the pipeline state
        self.run_btn = ForestButton("Start", "play", "primary", self)
        self.calib_btn = ForestButton("Calibrate", "target", "outline", self)
        self.rec_btn = ForestButton("Record", "record", "outline", self)
        self.play_btn = ForestButton("Play", "play", "outline", self)
        self.torque_btn = ForestButton("Torque Off", "power", "danger", self)
        self.buttons = (self.run_btn, self.calib_btn, self.rec_btn, self.play_btn, self.torque_btn)
        self.path = None  # the hand node's record / play state
        self.rec_btn.clicked.connect(self.pipeline.record)
        self.play_btn.clicked.connect(self.pipeline.play)
        self.run_btn.clicked.connect(self.run_clicked)
        self.run_switched = 0.0  # when it last turned from Start to Stop or back
        self.calib_btn.clicked.connect(self.pipeline.calibrate)
        self.torque_btn.clicked.connect(self.torque_clicked)
        self.has_bus = has_bus
        self.torque_armed = QTimer(self)  # running = the first click happened, waiting for the second
        self.torque_armed.setSingleShot(True)
        self.torque_armed.timeout.connect(self.torque_disarm)
        self.update_buttons()
        # the window starts with the focus (q / Esc work, no focus ring on a button until Tab)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setFocus()

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_pipeline)
        self.timer.start(15)

    # ---- layout -------------------------------------------------------------

    def resizeEvent(self, e):
        self.layout_widgets()
        super().resizeEvent(e)

    def layout_widgets(self):
        """Title in the top margin; below it the camera card (image + status bar) and the
        button row, centered together in the space left."""
        w, h = self.width(), self.height()
        self.s = s = min(w / 1400, h / 820)
        top = self.header_height()               # logo + title, vines
        side = 0.08 * w                          # vines hang in the side margins
        bar, gap, bh_btn, bottom = STATUS_H * s, STACK_GAP * s, BUTTON_SIZE[1] * s, BOTTOM_MARGIN * s
        cw = w - 2 * side
        ch = min(cw / self.aspect, h - top - bar - gap - bh_btn - bottom)
        cw = ch * self.aspect
        y = top + 0.3 * max(0.0, h - top - bottom - (ch + bar + gap + bh_btn))  # spare room: mostly below
        self.cam_rect = QRectF((w - cw) / 2, y, cw, ch)
        self.bar_rect = QRectF(self.cam_rect.left(), self.cam_rect.bottom(), cw, bar)

        sw, sh, bgap = BUTTON_SIZE[0] * s, bh_btn, BUTTON_GAP * s
        n = len(self.buttons)
        x = w / 2 - (n * sw + (n - 1) * bgap) / 2
        y = self.bar_rect.bottom() + gap
        for b in self.buttons:
            b.setGeometry(QRectF(x, y, sw, sh).toRect())
            b.scale = s
            x += sw + bgap

        self.bg = self.render_background()
        self.update()

    def header_height(self):
        """The top band for the logo and title (the camera card starts below it)."""
        return min(0.115 * self.width(), 0.15 * self.height())

    def card_rect(self):
        return self.cam_rect.united(self.bar_rect)

    def update_buttons(self):
        st = self.state
        b = self.run_btn
        role = "primary" if st == "idle" else "neutral"
        if role != b.role:
            b.role, b.icon = role, "play" if role == "primary" else "stop"
            self.run_switched = time.monotonic()
        b.setText({"idle": "Start", "stopping": "Stopping…"}.get(st, "Stop"))
        b.setEnabled(st != "stopping")
        b.update()
        self.calib_btn.setEnabled(st in ("idle", "running"))
        # Record / Play: while running; the button of what's going on turns into its Stop
        path = self.path or {}
        mode = path.get("mode", "idle")
        for btn, active, text, icon, role in (
                (self.rec_btn, mode == "rec", "Record", "record", "outline"),
                (self.play_btn, mode in ("glide", "play"), "Play", "play", "outline")):
            btn.setEnabled(st == "running" and (btn is self.rec_btn or active or path.get("len", 0) > 0))
            btn.setText(("Stop Rec" if btn is self.rec_btn else "Stop Play") if active else text)
            btn.icon, btn.role = ("stop", "danger" if btn is self.rec_btn else "neutral") if active else (icon, role)
            btn.update()
        self.torque_btn.setEnabled(self.has_bus and st in ("idle", "starting", "running"))
        if not self.torque_btn.isEnabled() and self.torque_armed.isActive():
            self.torque_disarm()
        self.torque_btn.armed = self.torque_armed.isActive()
        self.torque_btn.setText("Confirm?" if self.torque_btn.armed else "Torque Off")

    def run_clicked(self):
        """Start when idle, Stop while starting / running. Clicks right after it switched are
        ignored, so a double-click on Start doesn't land on the Stop it just became."""
        if time.monotonic() - self.run_switched < RUN_SWITCH_GUARD_S:
            return
        if self.state == "idle":
            self.pipeline.start()
        elif self.state in ("starting", "running"):
            self.pipeline.stop()
        self.poll_pipeline()  # show the new state right away

    def torque_clicked(self):
        """First click arms (warning in the status bar), a second within TORQUE_CONFIRM_MS drops the arm."""
        if self.torque_armed.isActive():
            self.torque_armed.stop()
            self.pipeline.clear_notice(TORQUE_WARNING)
            self.pipeline.torque_off()
        else:
            self.torque_armed.start(TORQUE_CONFIRM_MS)
            self.pipeline.notice(TORQUE_WARNING, "The arm goes limp and falls: support it first",
                                 TORQUE_CONFIRM_MS / 1000)
        self.update_buttons()

    def torque_disarm(self):
        self.torque_armed.stop()
        self.pipeline.clear_notice(TORQUE_WARNING)
        self.torque_btn.armed = False
        self.torque_btn.setText("Torque Off")
        self.torque_btn.update()

    def card_radius(self):
        return 0.03 * min(self.width(), self.height())

    def render_background(self):
        """Everything that doesn't change per frame: night sky, glow, the forest
        (forest.py), fireflies, title, shadows."""
        w, h = self.width(), self.height()
        dpr = self.devicePixelRatioF()
        pm = QPixmap(round(w * dpr), round(h * dpr))
        pm.setDevicePixelRatio(dpr)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)

        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, NIGHT_TOP)
        grad.setColorAt(0.55, NIGHT_MID)
        grad.setColorAt(1.0, NIGHT_BOTTOM)
        p.fillRect(QRectF(0, 0, w, h), grad)
        card = self.card_rect()
        glow = QRadialGradient(card.center(), 0.75 * max(card.width(), card.height()))
        glow.setColorAt(0.0, GLOW)
        glow.setColorAt(1.0, with_alpha(GLOW, 0.0))
        p.fillRect(QRectF(0, 0, w, h), glow)

        # title geometry first: the scenery keeps clear of it, the card and the buttons
        m = self.header_height()
        font = brand_font(max(16.0, min(0.5 * m, 0.068 * h)))
        px = font.pixelSize()
        text_w = QFontMetricsF(font).horizontalAdvance(APP_NAME)
        logo = self.logo
        side = min(1.75 * px, 0.92 * m) if logo is not None else 0.0
        gap = 0.35 * px if logo is not None else 0.0
        x0 = (w - (side + gap + text_w)) / 2
        title_area = QRectF(x0 - px, 0, side + gap + text_w + 2 * px, m)
        row = QRectF(self.buttons[0].geometry()).united(QRectF(self.buttons[-1].geometry()))
        clear = [title_area, card, row]

        forest.draw_back(p, w, h, clear)
        forest.draw_fireflies(p, w, h, clear, random.Random(3))
        soft_shadow(p, card, self.card_radius(), 22 * self.s, alpha=0.55, dy=10 * self.s)
        for b in self.buttons:
            r = QRectF(b.geometry())
            soft_shadow(p, r.adjusted(4, 4, -4, -4), 0.3 * r.height(), 10 * self.s, alpha=0.4, dy=5 * self.s)
        forest.draw_front(p, w, h, clear)

        # title: the logo, then the name, centered together in the top margin
        if logo is not None:
            lr = QRectF(x0, (m - side) / 2, side, side)
            p.drawImage(lr, logo.scaled(round(side * dpr), round(side * dpr), Qt.KeepAspectRatio,
                                        Qt.SmoothTransformation))
        title = QRectF(x0 + side + gap, 0, text_w + 2, m)
        p.setFont(font)
        p.setPen(with_alpha(LILAC, 0.35))  # soft lilac halo
        p.drawText(title.translated(0, max(1.0, 2 * self.s)), Qt.AlignVCenter | Qt.AlignLeft, APP_NAME)
        p.setPen(TEXT)
        p.drawText(title, Qt.AlignVCenter | Qt.AlignLeft, APP_NAME)
        p.end()
        return pm

    # ---- camera + status ----------------------------------------------------

    def poll_pipeline(self):
        f, seq, status, arm, state = self.pipeline.latest()
        alive = self.pipeline.alive()
        path = self.pipeline.path()
        path_key = None if path is None else (path.get("mode"), path.get("len", 0) > 0)
        if state != self.state or path_key != getattr(self, "path_key", None):
            self.state, self.path, self.path_key = state, path, path_key
            self.update_buttons()
        self.path = path
        if self.torque_btn.armed:  # countdown ring
            self.torque_btn.progress = self.torque_armed.remainingTime() / TORQUE_CONFIRM_MS
            self.torque_btn.update()
        if (seq == self.seq and status == self.status and arm == self.arm and alive == self.alive
                and (f is None) == (self.frame is None)):
            return
        self.frame, self.seq, self.status, self.arm, self.alive = f, seq, status, arm, alive
        if f is not None and abs(f.shape[1] / f.shape[0] - self.aspect) > 0.01:
            self.aspect = f.shape[1] / f.shape[0]  # the card follows the preview's shape
            self.layout_widgets()
        pad = 4 * max(1.0, self.s)
        self.update(self.card_rect().adjusted(-pad, -pad, pad, pad).toAlignedRect())

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        if self.bg is not None:
            p.drawPixmap(0, 0, self.bg)

        card, box, radius = self.card_rect(), self.cam_rect, self.card_radius()
        outline = QPainterPath()
        outline.addRoundedRect(card, radius, radius)
        p.fillPath(outline, CARD)

        p.save()
        p.setClipPath(outline)
        p.setClipRect(box, Qt.IntersectClip)  # the image: the card's top, square bottom corners
        if self.frame is not None:
            f = self.frame
            img = QImage(f.data, f.shape[1], f.shape[0], f.strides[0], QImage.Format_BGR888)
            # cover-fit (the box has the frame's aspect, so at most a pixel is cropped)
            k = max(box.width() / img.width(), box.height() / img.height())
            sw, sh = box.width() / k, box.height() / k
            p.drawImage(box, img, QRectF((img.width() - sw) / 2, (img.height() - sh) / 2, sw, sh))
        else:
            self.paint_empty(p, box)
        p.restore()

        self.paint_bar(p)
        p.setPen(QPen(with_alpha(CARD_EDGE, 0.35), max(1.0, 1.2 * self.s)))
        p.drawLine(QPointF(box.left(), box.bottom()), QPointF(box.right(), box.bottom()))
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(CARD_EDGE, max(1.5, 2.2 * self.s)))
        p.drawPath(outline)

    def paint_empty(self, p, box):
        """No preview yet: the logo, faint, and what to do."""
        glow = QRadialGradient(box.center(), 0.6 * box.height())
        glow.setColorAt(0.0, with_alpha(LILAC, 0.10))
        glow.setColorAt(1.0, with_alpha(LILAC, 0.0))
        p.fillRect(box, glow)
        side = 0.42 * box.height()
        msg = {"idle": "Press Start", "stopping": "Stopping…"}.get(self.state, "Getting ready…")
        font = app_font(max(12.0, 22 * self.s), QFont.Bold)
        th = QFontMetricsF(font).height()
        y = box.center().y() - (side + 0.4 * th + th) / 2
        if self.logo is not None:
            p.setOpacity(0.28)
            p.drawImage(QRectF(box.center().x() - side / 2, y, side, side), self.logo)
            p.setOpacity(1.0)
        p.setFont(font)
        p.setPen(MUTED)
        p.drawText(QRectF(box.left(), y + side + 0.4 * th, box.width(), th), Qt.AlignCenter, msg)

    def chip(self, p, x_right, cy, text, fill, ink, font, edge=None):
        """Pill ending at x_right, centered on cy; returns its left edge."""
        fm = QFontMetricsF(font)
        s = self.s
        w, h = fm.horizontalAdvance(text) + 22 * s, fm.height() + 8 * s
        r = QRectF(x_right - w, cy - h / 2, w, h)
        p.setBrush(fill)
        p.setPen(QPen(edge, max(1.0, 1.2 * s)) if edge is not None else Qt.NoPen)
        p.drawRoundedRect(r, h / 2, h / 2)
        p.setFont(font)
        p.setPen(ink)
        p.drawText(r, Qt.AlignCenter, text)
        return r.left()

    def paint_bar(self, p):
        """Status bar: [dot  status  · detail]   [arm chip] [IK · Arm · Hand dots] [LIVE]."""
        s, bar = self.s, self.bar_rect
        cy = bar.center().y()
        pad = 18 * s
        text, detail, kind = self.status if self.status else ("", "", "info")
        if not (text or detail):
            text, detail, kind = KIND_LABEL.get(self.state, ""), "", "hold" if self.state == "running" else "info"
            if self.state == "idle":
                kind = "hold"
        color = KIND_COLOR.get(kind, LILAC)
        if kind in ("error", "lost"):  # tint the whole bar
            p.save()
            outline = QPainterPath()
            outline.addRoundedRect(self.card_rect(), self.card_radius(), self.card_radius())
            p.setClipPath(outline)
            p.fillRect(bar, with_alpha(color, 0.16))
            p.restore()

        # right side, laid out right to left
        small = app_font(max(9.0, 12.5 * s), QFont.ExtraBold)
        x = bar.right() - pad
        x = self.chip(p, x, cy, "LIVE" if self.live else "DRY RUN", EMBER if self.live else FIREFLY,
                      WHITE if self.live else FIREFLY_TEXT, small) - 16 * s
        dot_font = app_font(max(9.0, 12.5 * s), QFont.DemiBold)
        fm = QFontMetricsF(dot_font)
        for name, label in (("hand", "Hand"), ("arm", "Arm"), ("ik", "IK")):
            up = self.alive.get(name, False)
            tw = fm.horizontalAdvance(label)
            p.setFont(dot_font)
            p.setPen(TEXT if up else MUTED)
            p.drawText(QRectF(x - tw, bar.top(), tw + 2, bar.height()), Qt.AlignVCenter | Qt.AlignLeft, label)
            x -= tw + 6 * s
            p.setPen(Qt.NoPen)
            p.setBrush(GRASS if up else with_alpha(MUTED, 0.45))
            p.drawEllipse(QPointF(x - 4 * s, cy), 4 * s, 4 * s)
            x -= 8 * s + 12 * s
        if self.arm is not None:
            arm_text, alarm = self.arm
            arm_font = app_font(max(9.0, 13 * s), QFont.Bold, digits=True)
            room = max(0.0, x - bar.left() - 0.45 * bar.width())
            arm_text = QFontMetricsF(arm_font).elidedText(arm_text, Qt.ElideRight, room)
            if arm_text:
                x = self.chip(p, x, cy, arm_text, EMBER if alarm else with_alpha(LILAC, 0.14),
                              WHITE if alarm else TEXT, arm_font,
                              edge=None if alarm else with_alpha(LILAC, 0.35)) - 14 * s

        # left side: state dot, status, detail on the same line (detail lines joined)
        lx = bar.left() + pad
        r = 6 * s
        if kind in ("track", "lost", "error"):  # a soft halo around the dot
            halo = QRadialGradient(QPointF(lx + r, cy), 3 * r)
            halo.setColorAt(0.0, with_alpha(color, 0.55))
            halo.setColorAt(1.0, with_alpha(color, 0.0))
            p.setPen(Qt.NoPen)
            p.setBrush(halo)
            p.drawEllipse(QPointF(lx + r, cy), 3 * r, 3 * r)
        p.setPen(Qt.NoPen)
        p.setBrush(color)
        p.drawEllipse(QPointF(lx + r, cy), r, r)
        lx += 2 * r + 12 * s
        room = x - lx - 8 * s
        main_font = app_font(max(11.0, 17 * s), QFont.ExtraBold)
        fm = QFontMetricsF(main_font)
        main = fm.elidedText(text, Qt.ElideRight, room)
        p.setFont(main_font)
        p.setPen(TEXT)
        p.drawText(QRectF(lx, bar.top(), room, bar.height()), Qt.AlignVCenter | Qt.AlignLeft, main)
        lx += fm.horizontalAdvance(main) + 14 * s
        detail = "   ·   ".join(ln for ln in detail.split("\n") if ln.strip())
        if detail and x - lx > 40 * s:
            dfont = app_font(max(10.0, 14 * s), QFont.DemiBold, digits=True)
            p.setFont(dfont)
            p.setPen(MUTED)
            p.drawText(QRectF(lx, bar.top(), x - lx - 8 * s, bar.height()), Qt.AlignVCenter | Qt.AlignLeft,
                       QFontMetricsF(dfont).elidedText(detail, Qt.ElideRight, x - lx - 8 * s))

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e):
        self.timer.stop()
        self.pipeline.close()
        super().closeEvent(e)


def main():
    argv = sys.argv[1:]
    extra = argv[argv.index("--") + 1:] if "--" in argv else []  # for hand_target_node.py
    argv = argv[:argv.index("--")] if "--" in argv else argv
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dry-run", action="store_true",
                    help="feetech_mirror without --live: reads the arm, no torque, nothing moves")
    ap.add_argument("--no-bus", action="store_true", help="dry run without the arm attached (implies --dry-run)")
    ap.add_argument("--cams", nargs=2, default=None, metavar=("LEFT", "RIGHT"),
                    help="camera indices (default 1 0; stereo_cam.py --list)")
    ap.add_argument("--contrast", action="store_true", help="print the palette's text contrast ratios and exit")
    ap.add_argument("--no-app", action="store_true",
                    help="run in the terminal's process instead of as the Mimisis app (no Dock icon / Cmd-Tab "
                         "entry of its own; camera access then belongs to the terminal)")
    args = ap.parse_args(argv)
    if args.contrast:
        for name, fg, bg in CONTRAST_PAIRS:
            r = contrast(fg, bg)
            print(f"{name:<15} {fg.name()} on {bg.name()}  {r:4.1f}:1  {'ok' if r >= 4.5 else 'LOW'}")
        return
    if not args.no_app:  # its own app (Dock, Cmd-Tab, camera permission); returns only inside it
        import macos_app  # Computer-Vision/common, on the path via pipeline.py
        macos_app.relaunch_in_app(APP_NAME, "com.kh.mimisis", icon=LOGO_PATH)
    live = not (args.dry_run or args.no_bus)
    hand = list(HAND_ARGS)
    if args.cams:
        hand[1:3] = args.cams
    arm = ARM_ARGS + (["--live", "--yes"] if live else ["--no-bus"] if args.no_bus else [])

    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    if os.path.exists(LOGO_PATH):
        app.setWindowIcon(QIcon(LOGO_PATH))  # the Dock icon on macOS
    load_fonts()
    app.setFont(app_font(14))

    pipeline = Pipeline(hand + extra, arm)
    print(f"[ui] {'LIVE: Start homes the arm, then it follows the hand' if live else 'dry run: the arm does not move'}",
          flush=True)

    win = MainWindow(pipeline, live, has_bus=not args.no_bus)
    geo = app.primaryScreen().availableGeometry()
    win.resize(min(1400, int(0.92 * geo.width())), min(820, int(0.9 * geo.height())))
    win.show()
    # Ctrl-C in the terminal closes the window (which stops the pipeline): the children run in
    # their own sessions, so they don't get it; the timer lets Python see the signal
    signal.signal(signal.SIGINT, lambda *_: win.close())
    tick = QTimer()
    tick.timeout.connect(lambda: None)
    tick.start(200)
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
