"""Stylized corner vines for the app background (ui/app.py).

Flat, graphic botanical style: a few clean curves and evenly spaced leaves, no
outlines, gloss or gradients. Per top corner: one sweeping stem that dips along the
top and climbs off the top edge, one long strand hanging beside the camera card and one short
one, each ending in a curl. Leaves are almond shaped with a curved pointed tip, split
along the midrib into a light and a dark green half, alternating, angled toward the
tip and getting smaller toward it. The right corner mirrors the left.

    draw(p, w, h, clear)   both top corners; `clear` = rects to stay out of (title,
                           camera card, buttons), in window coordinates
"""
import math

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainterPath, QPen

LIGHT = QColor("#9ED476")  # leaf, lit half
DARK = QColor("#5E9F45")   # leaf, shaded half
STEM = QColor("#4F8D36")


def draw(p, w, h, clear=()):
    for side in (0, 1):
        cl = [QRectF(w - r.right(), r.top(), r.width(), r.height()) for r in clear] if side else list(clear)
        p.save()
        if side:
            p.translate(w, 0)
            p.scale(-1, 1)
        corner(p, w, h, cl)
        p.restore()


# ---- geometry -----------------------------------------------------------------------------

def sample(path, n):
    return [path.pointAtPercent(i / n) for i in range(n + 1)]


def heading(pts, i):
    a, b = pts[max(0, i - 1)], pts[min(len(pts) - 1, i + 1)]
    return math.atan2(b.y() - a.y(), b.x() - a.x())


def with_curl(pts, radius, turn, laps=1.1):
    """Continue `pts` past its end into a tightening spiral (the curl at a vine's tip)."""
    hd = heading(pts, len(pts) - 1)
    x, y = pts[-1].x(), pts[-1].y()
    out = list(pts)
    n = 48
    total = laps * 2 * math.pi
    for i in range(1, n + 1):
        t = i / n
        r = radius * (1 - 0.75 * t)
        dtheta = total / n
        hd += turn * dtheta
        x += r * dtheta * math.cos(hd)
        y += r * dtheta * math.sin(hd)
        out.append(QPointF(x, y))
    return out


def trim(pts, clear, pad):
    """Cut a path before it enters a clear rect."""
    for i, q in enumerate(pts):
        if any(r.adjusted(-pad, -pad, pad, pad).contains(q) for r in clear):
            return pts[:i]
    return pts


# ---- one corner ---------------------------------------------------------------------------

def corner(p, w, h, clear):
    k = min(w, h)
    card_top = min((r.top() for r in clear if r.left() < 0.3 * w and r.top() > 0.05 * h), default=0.22 * h)
    card_left = min((r.left() for r in clear if r.top() < 0.6 * h and r.bottom() > 0.3 * h), default=0.08 * w)
    low = min((r.top() for r in clear if r.left() < card_left and r.top() > card_top + 0.1 * h), default=0.75 * h)

    # the main stem: in from the corner, a gentle dip along the top, then climbing off the
    # top edge (it continues out of view, like a vine growing up into the canopy)
    main = QPainterPath(QPointF(-0.02 * w, 0.015 * h))
    main.cubicTo(0.08 * w, 0.145 * h, 0.21 * w, 0.11 * h, 0.3 * w, -0.04 * h)
    main_pts = trim(sample(main, 120), clear, 0.01 * k)

    # the long strand: leaves the stem near the corner and hangs beside the card
    x_hang = max(0.02 * w, card_left - 0.05 * k)
    root = main.pointAtPercent(0.16)
    long_end_y = min(0.72 * h, low - 0.06 * k)
    strand = QPainterPath(root)
    strand.cubicTo(root.x() + 0.03 * w, root.y() + 0.12 * h, x_hang + 0.012 * w, 0.4 * h, x_hang, long_end_y)
    strand_pts = trim(with_curl(sample(strand, 140), 0.016 * k, 1), clear, 0.01 * k)

    # the short strand: from the middle of the stem, ending well above the card
    root2 = main.pointAtPercent(0.6)
    end2_y = min(root2.y() + 0.09 * h, card_top - 0.05 * k)
    short_pts = []
    if end2_y > root2.y() + 0.03 * k:
        short = QPainterPath(root2)
        short.cubicTo(root2.x() + 0.01 * w, root2.y() + 0.04 * h, root2.x() - 0.015 * w, end2_y - 0.02 * h,
                      root2.x() - 0.005 * w, end2_y)
        short_pts = trim(with_curl(sample(short, 60), 0.01 * k, -1, laps=0.9), clear, 0.01 * k)

    for pts, w0, w1 in ((strand_pts, 0.007, 0.0025), (short_pts, 0.005, 0.002), (main_pts, 0.012, 0.003)):
        stem(p, pts, w0 * k, w1 * k)

    # leaves: (path points, share of it that carries leaves, spacing, first size)
    leaf_row(p, main_pts, 0.04, 1.0, 0.05 * k, 0.05 * k, first_side=1, shrink=0.25)
    leaf_row(p, strand_pts, 0.06, 0.82, 0.048 * k, 0.044 * k, first_side=-1)
    leaf_row(p, short_pts, 0.15, 0.7, 0.04 * k, 0.034 * k, first_side=1)


# ---- drawing ------------------------------------------------------------------------------

def stem(p, pts, w0, w1):
    n = len(pts) - 1
    if n < 1:
        return
    p.save()
    for i in range(n):
        p.setPen(QPen(STEM, max(0.8, w0 + (w1 - w0) * i / n), Qt.SolidLine, Qt.RoundCap))
        p.drawLine(pts[i], pts[i + 1])
    p.restore()


def leaf_row(p, pts, start, stop, spacing, size, first_side=1, shrink=0.45):
    """Alternating leaves from `start` to `stop` (shares of the path's length), angled
    toward the tip, shrinking along the way."""
    if len(pts) < 3:
        return
    cum = [0.0]
    for a, b in zip(pts, pts[1:]):
        cum.append(cum[-1] + math.hypot(b.x() - a.x(), b.y() - a.y()))
    total = cum[-1]
    d, side, j = start * total, first_side, 0
    while d < stop * total:
        while j < len(cum) - 2 and cum[j + 1] < d:
            j += 1
        t = (d - start * total) / max(1e-6, (stop - start) * total)
        ang = math.degrees(heading(pts, j)) + side * 48  # 48 deg off the stem, toward the tip
        leaf(p, pts[j], ang, size * (1 - shrink * t), side)
        side = -side
        d += spacing * (1 - 0.25 * t)


def leaf(p, at, deg, ln, side):
    """Almond leaf from `at` pointing `deg` (0 = +x, 90 = down), tip curving away from
    the stem, light and dark halves split along the midrib."""
    wd = 0.42 * ln
    bend = 0.1 * ln * side
    tip = QPointF(bend, ln)
    path = QPainterPath(QPointF(0, 0))
    path.cubicTo(0.75 * wd, 0.2 * ln, 0.62 * wd + bend, 0.72 * ln, tip.x(), tip.y())
    path.cubicTo(-0.55 * wd + bend, 0.74 * ln, -0.7 * wd, 0.22 * ln, 0, 0)
    mid = QPainterPath(QPointF(0, 0))  # the midrib, curving with the tip
    mid.quadTo(0.0, 0.55 * ln, tip.x(), tip.y())
    half = QPainterPath(mid)  # one side of it, to clip the lit half
    half.lineTo(2 * ln, 2 * ln)
    half.lineTo(2 * ln, -ln)
    half.closeSubpath()

    p.save()
    p.translate(at)
    p.rotate(deg - 90)  # local +y = the leaf's direction
    p.setPen(Qt.NoPen)
    p.setBrush(DARK)
    p.drawPath(path)
    p.setBrush(LIGHT)
    p.drawPath(path.intersected(half))
    p.restore()
