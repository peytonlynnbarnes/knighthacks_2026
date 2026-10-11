#!/usr/bin/env python3
"""Main window: live camera preview with a Start button.

    python3 ui/app.py [--no-ros] [any other hand_target_node.py option]
    q / Esc quits.

Start does what hand_target_node.py does: the hand calibration (steps shown
over the preview, saved to calib/hand.json), then tracking that publishes
/ball_position and /gripper for the IK node and feetech_mirror.py. Pressing
it again recalibrates. Needs ROS sourced, like the node. See pipeline.py.

Layout for a W x H window: the camera box is 0.8W x 0.6H and sits 0.1W from
the top, left and right edges; the button is centered in the space below.
The app name goes in the top margin, and vines hang from both top corners.
"""
import math
import os
import random
import sys

from pipeline import Pipeline  # imports cv2
from hand_target_node import build_parser

HERE = os.path.dirname(os.path.abspath(__file__))

# The cv2 pip wheel points Qt at its bundled Qt5 plugins and fonts on import;
# PySide6 is Qt6 and must not pick those up.
for _var in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
    if "cv2" in os.environ.get(_var, ""):
        del os.environ[_var]

from PySide6.QtCore import QPointF, QRectF, Qt, QTimer  # noqa: E402
from PySide6.QtGui import (QColor, QFont, QFontDatabase, QFontMetricsF, QImage,  # noqa: E402
                           QLinearGradient, QPainter, QPainterPath, QPen, QPixmap)
from PySide6.QtWidgets import QAbstractButton, QApplication, QWidget  # noqa: E402

APP_NAME = "Mimesis"
FONT_PATH = os.path.join(HERE, "fonts", "Comfortaa-Bold.otf")

LILAC_TOP = QColor("#9069C2")     # darker lilac
LILAC_MID = QColor("#C6ABEA")
LILAC_BOTTOM = QColor("#FBF8FF")  # very, very light purple
GRASS = QColor("#8FCF68")         # Start button, vine leaves
GRASS_DARK = QColor("#6BB247")    # darker leaves
STEM = QColor("#4F8D36")          # vine stems and leaf veins
WHITE = QColor("#FFFFFF")
SHADOW = QColor(58, 28, 104)      # purple-tinted drop shadow
STATUS_BG = {"calib": QColor(58, 28, 104, 190),   # calibration step instructions
             "hold": QColor(58, 28, 104, 190),    # hand lost, arm holding
             "track": QColor(83, 150, 52, 225),   # publishing to the arm
             "error": QColor(150, 48, 92, 220)}

# Base sizes at a 900 x 900 window; everything scales from these.
START_SIZE = (250, 68)


def app_font(px):
    f = QFont(app_font.family)
    f.setBold(True)
    f.setPixelSize(max(1, round(px)))
    return f


app_font.family = "Comfortaa"


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
        c = QColor(SHADOW)
        c.setAlphaF(alpha / steps)
        p.setBrush(c)
        p.drawRoundedRect(rect.adjusted(-g, -g + dy, g, g + dy), radius + g, radius + g)
    p.restore()


class RoundedButton(QAbstractButton):
    """Green rounded-square button with a white outline and white label."""

    def __init__(self, text, color, parent=None):
        super().__init__(parent)
        self.setText(text)
        self.color = QColor(color)
        self.scale = 1.0
        self.setCursor(Qt.PointingHandCursor)
        self.setAttribute(Qt.WA_Hover)  # repaint on hover enter/leave

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        border = max(2.0, 3.0 * self.scale)
        r = QRectF(self.rect()).adjusted(border / 2, border / 2, -border / 2, -border / 2)
        radius = 0.3 * r.height()

        c = self.color
        if self.isDown():
            c = c.darker(110)
        elif self.underMouse():
            c = c.lighter(107)
        fill = QLinearGradient(r.topLeft(), r.bottomLeft())
        fill.setColorAt(0.0, c.lighter(106))
        fill.setColorAt(1.0, c.darker(104))
        p.setBrush(fill)
        p.setPen(QPen(WHITE, border))
        p.drawRoundedRect(r, radius, radius)

        p.setFont(app_font(0.38 * r.height()))
        p.setPen(QColor(40, 90, 25, 90))  # soft shadow keeps white text readable on light green
        p.drawText(r.translated(0, max(1.0, 1.5 * self.scale)), Qt.AlignCenter, self.text())
        p.setPen(WHITE)
        p.drawText(r, Qt.AlignCenter, self.text())


class MainWindow(QWidget):
    def __init__(self, pipeline, cam0):
        super().__init__()
        self.setWindowTitle(APP_NAME)
        self.setMinimumSize(480, 480)
        self.pipeline, self.cam0 = pipeline, cam0
        self.frame, self.seq, self.status = None, 0, None
        self.bg = None
        self.cam_rect = QRectF()
        self.s = 1.0

        self.start_btn = RoundedButton("Start", GRASS, self)
        self.start_btn.clicked.connect(self.pipeline.start)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.poll_pipeline)
        self.timer.start(15)

    # ---- layout -------------------------------------------------------------

    def resizeEvent(self, e):
        w, h = self.width(), self.height()
        m = 0.1 * w
        self.cam_rect = QRectF(m, m, 0.8 * w, 0.6 * h)
        room = h - self.cam_rect.bottom()
        self.s = s = min(w / 900, h / 900, room / 190)

        sw, sh = START_SIZE[0] * s, START_SIZE[1] * s
        y = self.cam_rect.bottom() + (room - sh) / 2
        self.start_btn.setGeometry(QRectF(w / 2 - sw / 2, y, sw, sh).toRect())
        self.start_btn.scale = s

        self.bg = self.render_background()
        super().resizeEvent(e)

    def cam_radius(self):
        return 0.035 * min(self.width(), self.height())

    def border_width(self):
        return max(3.0, 5.0 * self.s)

    def render_background(self):
        """Everything that doesn't change per frame: gradient, title, vines, shadows."""
        w, h = self.width(), self.height()
        dpr = self.devicePixelRatioF()
        pm = QPixmap(round(w * dpr), round(h * dpr))
        pm.setDevicePixelRatio(dpr)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)

        grad = QLinearGradient(0, 0, 0, h)
        grad.setColorAt(0.0, LILAC_TOP)
        grad.setColorAt(0.5, LILAC_MID)
        grad.setColorAt(1.0, LILAC_BOTTOM)
        p.fillRect(QRectF(0, 0, w, h), grad)

        soft_shadow(p, self.cam_rect, self.cam_radius(), 18 * self.s, alpha=0.35, dy=8 * self.s)
        r = QRectF(self.start_btn.geometry())
        soft_shadow(p, r, 0.3 * r.height(), 10 * self.s, alpha=0.22, dy=5 * self.s)

        m = self.cam_rect.top()
        title = QRectF(0, 0, w, m)
        p.setFont(app_font(max(14.0, min(0.45 * m, 0.055 * h))))
        p.setPen(QColor(SHADOW.red(), SHADOW.green(), SHADOW.blue(), 80))
        p.drawText(title.translated(0, max(1.0, 2 * self.s)), Qt.AlignCenter, APP_NAME)
        p.setPen(WHITE)
        p.drawText(title, Qt.AlignCenter, APP_NAME)

        draw_corner_vines(p, w, h, random.Random(7))
        p.translate(w, 0)
        p.scale(-1, 1)  # right corner: mirror image, different leaf jitter
        draw_corner_vines(p, w, h, random.Random(11))
        p.end()
        return pm

    # ---- camera -------------------------------------------------------------

    def poll_pipeline(self):
        f, seq, status = self.pipeline.latest()
        if seq == self.seq and status == self.status:
            return
        self.frame, self.seq, self.status = f, seq, status
        pad = self.border_width()
        self.update(self.cam_rect.adjusted(-pad, -pad, pad, pad).toAlignedRect())

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHints(QPainter.Antialiasing | QPainter.SmoothPixmapTransform)
        if self.bg is not None:
            p.drawPixmap(0, 0, self.bg)

        box = self.cam_rect
        radius = self.cam_radius()
        clip = QPainterPath()
        clip.addRoundedRect(box, radius, radius)

        if self.frame is not None:
            f = self.frame
            img = QImage(f.data, f.shape[1], f.shape[0], f.strides[0], QImage.Format_BGR888)
            # cover-fit: fill the box, cropping the frame's overflow evenly
            k = max(box.width() / img.width(), box.height() / img.height())
            sw, sh = box.width() / k, box.height() / k
            src = QRectF((img.width() - sw) / 2, (img.height() - sh) / 2, sw, sh)
            p.save()
            p.setClipPath(clip)
            p.drawImage(box, img, src)
            p.restore()
        else:
            p.fillPath(clip, QColor(255, 255, 255, 45))
            msg = ("Starting camera…" if self.pipeline.has_camera
                   else f"Camera /dev/video{self.cam0} not available")
            p.setFont(app_font(0.032 * min(self.width(), self.height())))
            p.setPen(WHITE)
            p.drawText(box, Qt.AlignCenter, msg)

        if self.status is not None:
            self.paint_status(p, *self.status)

        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(WHITE, self.border_width()))
        p.drawRoundedRect(box, radius, radius)

    def paint_status(self, p, text, detail, kind):
        """Rounded card along the bottom of the camera box: a main line, plus a smaller detail line."""
        s, box = self.s, self.cam_rect
        pad_x, pad_y, gap = 22 * s, 12 * s, 4 * s
        max_w = box.width() - 2 * (24 * s + pad_x)
        flags = Qt.AlignCenter | Qt.TextWordWrap
        lines = [(app_font(max(12.0, 19 * s)), text, WHITE)]
        if detail:
            lines.append((app_font(max(10.0, 14 * s)), detail, QColor(255, 255, 255, 210)))
        rects = [QFontMetricsF(f).boundingRect(QRectF(0, 0, max_w, 1e4), flags, t) for f, t, _ in lines]

        w = max(r.width() for r in rects) + 2 * pad_x
        h = sum(r.height() for r in rects) + gap * (len(rects) - 1) + 2 * pad_y
        card = QRectF(box.center().x() - w / 2, box.bottom() - 20 * s - h, w, h)
        p.setBrush(STATUS_BG[kind])
        p.setPen(QPen(QColor(255, 255, 255, 200), max(1.5, 2 * s)))
        p.drawRoundedRect(card, min(h / 2, 16 * s), min(h / 2, 16 * s))

        y = card.top() + pad_y
        for (font, t, color), r in zip(lines, rects):
            p.setFont(font)
            p.setPen(color)
            p.drawText(QRectF(card.left() + pad_x, y, w - 2 * pad_x, r.height()), flags, t)
            y += r.height() + gap

    def keyPressEvent(self, e):
        if e.key() in (Qt.Key_Q, Qt.Key_Escape):
            self.close()
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e):
        self.timer.stop()
        self.pipeline.close()
        super().closeEvent(e)


# ---- vines ----------------------------------------------------------------

def leaf_path(length, width):
    """Almond leaf with its base at the origin, pointing down local +y."""
    path = QPainterPath(QPointF(0, 0))
    path.cubicTo(width, 0.15 * length, 0.75 * width, 0.7 * length, 0, length)
    path.cubicTo(-0.75 * width, 0.7 * length, -width, 0.15 * length, 0, 0)
    return path


def draw_leaf(p, at, stem_deg, splay_deg, length, color):
    """Leaf on a short stalk at `at`, splayed `splay_deg` off the stem direction."""
    width = 0.42 * length
    stalk = 0.12 * length
    p.save()
    p.translate(at)
    p.rotate(stem_deg - 90 + splay_deg)  # local +y now points along the leaf
    p.setPen(QPen(STEM, max(1.0, 0.07 * width), Qt.SolidLine, Qt.RoundCap))
    p.drawLine(QPointF(0, 0), QPointF(0, stalk))
    p.translate(0, stalk)

    fill = QLinearGradient(-width, 0, width, 0)
    fill.setColorAt(0.0, color.lighter(118))
    fill.setColorAt(1.0, color.darker(106))
    edge = QColor(STEM)
    edge.setAlpha(120)
    p.setBrush(fill)
    p.setPen(QPen(edge, 1.0))
    p.drawPath(leaf_path(length, width))

    vein = QColor(STEM)
    vein.setAlpha(150)
    p.setPen(QPen(vein, max(1.0, 0.06 * width), Qt.SolidLine, Qt.RoundCap))
    p.drawLine(QPointF(0, 0.08 * length), QPointF(0, 0.8 * length))
    p.restore()


def draw_stem(p, pts, w0, w1):
    """Polyline that tapers from width w0 to w1."""
    n = len(pts) - 1
    for i in range(n):
        p.setPen(QPen(STEM, w0 + (w1 - w0) * i / n, Qt.SolidLine, Qt.RoundCap))
        p.drawLine(pts[i], pts[i + 1])


def draw_tendril(p, at, size, turn):
    """Small curl starting at `at`; turn = +1 / -1 picks the spin direction."""
    a0 = math.pi if turn > 0 else 0.0
    center = QPointF(at.x() - size * math.cos(a0), at.y() - size * math.sin(a0))
    path = QPainterPath(at)
    for i in range(1, 30):
        t = i / 29
        a = a0 + turn * 2.6 * math.pi * t
        r = size * (1 - 0.8 * t)
        path.lineTo(center.x() + r * math.cos(a), center.y() + r * math.sin(a))
    p.setBrush(Qt.NoBrush)
    p.setPen(QPen(STEM, max(1.0, 0.12 * size), Qt.SolidLine, Qt.RoundCap))
    p.drawPath(path)


def stem_angle(pts, i):
    a, b = pts[max(0, i - 1)], pts[min(len(pts) - 1, i + 1)]
    return math.degrees(math.atan2(b.y() - a.y(), b.x() - a.x()))


def leaf_color(rng):
    r = rng.random()
    return GRASS_DARK if r < 0.35 else GRASS.lighter(108) if r > 0.85 else GRASS


def draw_corner_vines(p, w, h, rng):
    """Vines for the top-left corner (the caller mirrors them for the right).

    A vine drapes along the top edge and strands hang from it. Strands beside
    the camera stay inside the side margin; the rest end above the camera.
    """
    u = 0.1 * w           # side/top margin
    k = 0.1 * min(w, h)   # leaf size unit (<= u, so leaves fit in the margin)

    # Swag: enters off the left edge, sags, and leaves off the top edge.
    x0, x1 = -0.3 * u, 2.9 * u

    def swag(t):
        return QPointF(x0 + (x1 - x0) * t, 0.05 * u - 0.15 * u * t + 0.36 * u * math.sin(math.pi * t))

    swag_pts = [swag(i / 59) for i in range(60)]

    # (x, length, sway, phase): the first hangs beside the camera, the rest above it.
    strand_specs = [(0.38 * u, 0.45 * h, 0.06 * u, 0.0),
                    (0.80 * u, 0.36 * u, 0.04 * u, 1.3),
                    (1.45 * u, 0.36 * u, 0.05 * u, 2.1),
                    (2.15 * u, 0.24 * u, 0.04 * u, 0.7)]
    strands = []
    for x, length, sway, phase in strand_specs:
        top = swag((x - x0) / (x1 - x0))
        n = max(8, int(length / (0.04 * k)))
        strands.append([QPointF(top.x() + sway * math.sin(phase + 5.0 * t) * t, top.y() + length * t)
                         for t in (i / (n - 1) for i in range(n))])

    draw_stem(p, swag_pts, 0.075 * k, 0.03 * k)
    for pts in strands:
        draw_stem(p, pts, 0.05 * k, 0.022 * k)
    draw_tendril(p, strands[0][len(strands[0]) * 2 // 5], 0.11 * k, 1)
    draw_tendril(p, swag_pts[38], 0.09 * k, -1)
    draw_tendril(p, strands[2][len(strands[2]) // 2], 0.08 * k, -1)

    # Leaves along the swag: alternate hanging below and perking up above.
    side = 1
    t = 0.02
    while t < 0.97:
        i = int(t * 59)
        splay = rng.uniform(50, 75) if side > 0 else -rng.uniform(30, 50)
        size = 0.30 * k * rng.uniform(0.85, 1.1) * (1.0 if side > 0 else 0.8)
        draw_leaf(p, swag_pts[i], stem_angle(swag_pts, i), splay, size, leaf_color(rng))
        side = -side
        t += 0.045 * rng.uniform(0.85, 1.15)

    # Leaves along each strand: alternate sides, plus one at the tip.
    for pts in strands:
        length = pts[-1].y() - pts[0].y()
        side = 1 if rng.random() < 0.5 else -1
        d = 0.18 * k
        while d < length - 0.12 * k:
            i = int(d / length * (len(pts) - 1))
            splay = side * rng.uniform(35, 60)
            draw_leaf(p, pts[i], stem_angle(pts, i), splay, 0.28 * k * rng.uniform(0.85, 1.15),
                      leaf_color(rng))
            side = -side
            d += 0.27 * k * rng.uniform(0.85, 1.15)
        draw_leaf(p, pts[-1], stem_angle(pts, len(pts) - 1), rng.uniform(-10, 10), 0.24 * k,
                  leaf_color(rng))


def main():
    ap = build_parser()  # the node's options: --cams, --no-ros, --scale, ...
    ap.description = __doc__
    args = ap.parse_args()

    app = QApplication(sys.argv[:1])
    app.setApplicationName(APP_NAME)
    fid = QFontDatabase.addApplicationFont(FONT_PATH)
    if fid >= 0:
        app_font.family = QFontDatabase.applicationFontFamilies(fid)[0]
    else:
        print(f"warning: couldn't load {FONT_PATH}; using the system font", file=sys.stderr)
    app.setFont(app_font(14))

    pipeline = Pipeline(args)
    if pipeline.stereo is None:
        print(f"warning: stereo pair unavailable ({pipeline.stereo_err}); preview only", file=sys.stderr)
    if pipeline.ros_err:
        print(f"warning: ROS 2 unavailable ({pipeline.ros_err}); Start can't control the arm "
              "(source /opt/ros/jazzy/setup.bash, or pass --no-ros)", file=sys.stderr)

    win = MainWindow(pipeline, args.cams[0])
    side = min(900, int(0.9 * app.primaryScreen().availableGeometry().height()))
    win.resize(side, side)
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
