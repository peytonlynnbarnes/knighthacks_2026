"""Pixel-art enchanted-forest scenery for the app background (ui/app.py).

Everything is drawn on a low-resolution canvas (one art pixel = PX screen pixels,
~5 at 1400 x 820) and scaled up without smoothing, so the pixels stay crisp. Sprites
are ASCII grids (see SPRITES / PAL); stems, mist and glows are generated
pixel by pixel. Seeded: the same window size always gives the same picture.

    draw_back(p, w, h, clear)    stars, tree trunks at the edges, dithered mist, the ground
    draw_fireflies(p, w, h, clear, rng)
    draw_front(p, w, h, clear)   mushrooms in the bottom left, flowers in the bottom right,
                                 then the smooth corner vines (vines.py)

`clear`: rects (screen coords) to keep free: title, camera card, button row.
"""
import math
import random

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QColor, QImage, QPainter

import vines

PAL = {
    "D": QColor("#1E4419"),  # leaf outline
    "d": QColor("#2F6A28"),  # stem
    "g": QColor("#5FA543"),  # leaf shade
    "G": QColor("#8FCF68"),  # leaf
    "H": QColor("#C4F0A0"),  # leaf highlight
    "q": QColor("#331C57"),  # mushroom outline
    "p": QColor("#6E44A8"),  # cap shade
    "P": QColor("#A574E0"),  # cap
    "h": QColor("#C9A6F2"),  # cap highlight
    "W": QColor("#F6EEFF"),  # spots, petals
    "w": QColor("#CDBCE6"),  # petal shade
    "S": QColor("#D9CCEB"),  # stalk
    "s": QColor("#9C88BC"),  # stalk shade
    "C": QColor("#B8F5A0"),  # glowing gills
    "Y": QColor("#F4C95D"),  # flower centers
    "y": QColor("#C9952E"),
    "R": QColor("#EE9AC4"),  # tulip
    "r": QColor("#B85C8C"),
    "L": QColor("#E6D6FA"),  # bellflower / lavender, light to dark
    "l": QColor("#B994EC"),
    "V": QColor("#8A5CCB"),
    "v": QColor("#5E3A93"),
}
MOON = QColor(198, 171, 234)
TRUNK = QColor(13, 6, 26)
GROUND = QColor(16, 8, 30)
GLOW = QColor("#B8F5A0")
FIREFLY = QColor("#F4C95D")
BAYER = [[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]]  # ordered dither

SPRITES = {
    "mushroom_big": [
        "......qqqqqq......",
        "....qqPPPPhhqq....",
        "...qPPWWPPPPhhq...",
        "..qPPWWWPPPPPPPq..",
        ".qPPPPWPPPPWWPPPq.",
        ".qPPPPPPPPPWWWPPq.",
        "qPPWWPPPPPPPWPPPPq",
        "qPPWWPPPPWWPPPPPpq",
        "qpPPPPPPPWWPPPPppq",
        "qppPPPPPPPPPPPpppq",
        ".qqppppppppppppqq.",
        "..qCCCCCCCCCCCCq..",
        "...qqqsSSSSqqqq...",
        "......sSSSSs......",
        "......sSSSSs......",
        "......sSSSSs......",
        ".....sSSSSSSs.....",
        "....ssSSSSSSss....",
    ],
    "mushroom_mid": [
        "...qqqqqq...",
        "..qPPhhPPq..",
        ".qPWWPPPPPq.",
        "qPPWWPPPWWPq",
        "qpPPPPPPWWpq",
        ".qppppppppq.",
        "..qCCCCCCq..",
        "....sSSs....",
        "....sSSs....",
        "...ssSSss...",
    ],
    "mushroom_small": [
        "..qqqq..",
        ".qPWhPq.",
        "qPPPPWPq",
        ".qppppq.",
        "..qCCq..",
        "...sS...",
        "..ssSs..",
    ],
    "daisy": [
        "...W...",
        ".W.W.W.",
        "..wYw..",
        "WWYYYWW",
        "..wYw..",
        ".W.W.W.",
        "...W...",
    ],
    "tulip": [
        ".R.R.R.",
        ".RRRRR.",
        ".RrRrR.",
        ".RRRRR.",
        "..rRr..",
    ],
    "bell": [
        "..V..",
        ".lVl.",
        "lllll",
        "lLlLl",
        "l.l.l",
    ],
    "bud": [
        ".R.",
        "RrR",
        ".r.",
    ],
    "fern": [
        ".....d.....",
        "...d.d.d...",
        "..dGdGdGd..",
        ".d.GdGdG.d.",
        ".GdGGdGGdG.",
        "..GGgdgGG..",
        ".dGGgdgGGd.",
        "dGGg.d.gGGd",
        ".Gg..d..gG.",
        ".....d.....",
        ".....d.....",
    ],
    "tuft": [
        ".G...G.",
        ".Gg..G.",
        "gGg.gGg",
        "dgdgdgd",
    ],
    "stem_leaf_right": [
        "GG.",
        "dGG",
    ],
}


def flip(rows):
    return [r[::-1] for r in rows]


SPRITES["stem_leaf_left"] = flip(SPRITES["stem_leaf_right"])


def alpha(c, a):
    c = QColor(c)
    c.setAlphaF(max(0.0, min(1.0, a)))
    return c


def mix(a, b, t):
    return QColor.fromRgbF(a.redF() + (b.redF() - a.redF()) * t, a.greenF() + (b.greenF() - a.greenF()) * t,
                           a.blueF() + (b.blueF() - a.blueF()) * t)


class Canvas:
    """Low-res pixel canvas over the window; `mirror` flips x (for the right side)."""

    def __init__(self, w, h, clear):
        self.px = max(3, round(min(w, h) / 165))
        self.cw, self.ch = math.ceil(w / self.px), math.ceil(h / self.px)
        self.img = QImage(self.cw, self.ch, QImage.Format_ARGB32_Premultiplied)
        self.img.fill(Qt.transparent)
        self.p = QPainter(self.img)  # no antialiasing: every fill is a whole pixel
        self.mirror = False
        self.clear = [QRectF(r.left() / self.px - 1, r.top() / self.px - 1, r.width() / self.px + 2,
                             r.height() / self.px + 2) for r in clear]

    def blocked(self, x, y):
        if self.mirror:
            x = self.cw - 1 - x
        return any(r.left() <= x <= r.right() and r.top() <= y <= r.bottom() for r in self.clear)

    def put(self, x, y, color):
        x, y = int(round(x)), int(round(y))
        if self.mirror:
            x = self.cw - 1 - x
        if 0 <= x < self.cw and 0 <= y < self.ch and not any(
                r.left() <= x <= r.right() and r.top() <= y <= r.bottom() for r in self.clear):
            self.p.fillRect(x, y, 1, 1, color)

    def stamp(self, name, x, y, a=1.0):
        """Sprite with its top-left at (x, y)."""
        for j, row in enumerate(SPRITES[name]):
            for i, ch in enumerate(row):
                if ch != ".":
                    self.put(x + i, y + j, PAL[ch] if a >= 1 else alpha(PAL[ch], a))

    def fits(self, name, x, y):
        rows = SPRITES[name]
        return not any(self.blocked(x + i, y + j) for j in (0, len(rows) - 1) for i in (0, len(rows[0]) - 1))

    def dither_disc(self, cx, cy, r, color, a):
        """Glow as an ordered-dither disc: denser in the middle."""
        for y in range(int(cy - r), int(cy + r) + 1):
            for x in range(int(cx - r), int(cx + r) + 1):
                d = math.hypot(x - cx, y - cy) / r
                if d < 1 and BAYER[y % 4][x % 4] / 16 < (1 - d) ** 2 * 0.55:
                    self.put(x, y, alpha(color, a))

    def blit(self, p):
        self.p.end()
        p.save()
        p.setRenderHint(QPainter.SmoothPixmapTransform, False)
        p.drawImage(QRectF(0, 0, self.cw * self.px, self.ch * self.px), self.img)
        p.restore()


# ---- public ------------------------------------------------------------------------------

def draw_back(p, w, h, clear):
    c = Canvas(w, h, clear)
    rng = random.Random(5)
    for _ in range(int(c.cw * c.ch / 260)):  # stars in the upper sky
        x, y = rng.randrange(c.cw), rng.randrange(int(0.55 * c.ch))
        a = rng.uniform(0.25, 0.7)
        c.put(x, y, alpha(MOON if rng.random() < 0.7 else QColor("#FFFFFF"), a))
        if rng.random() < 0.12:  # a twinkle
            for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                c.put(x + dx, y + dy, alpha(MOON, 0.5 * a))
    for side in (0, 1):
        c.mirror = bool(side)
        trunk(c, -2, 7, c.ch, 0.85, random.Random(21 + side))
        trunk(c, 9, 3, int(0.45 * c.ch), 0.45, random.Random(31 + side))
    c.mirror = False
    for y in range(int(0.62 * c.ch), c.ch):  # mist: ordered dither, thicker toward the ground
        t = (y - 0.62 * c.ch) / (0.38 * c.ch)
        for x in range(c.cw):
            if BAYER[y % 4][x % 4] / 16 < 0.3 * t * t:
                c.put(x, y, alpha(MOON, 0.07))
    ground(c, clear)
    c.blit(p)


def draw_fireflies(p, w, h, clear, rng, n=40):
    c = Canvas(w, h, clear)
    placed = 0
    for _ in range(n * 6):
        if placed >= n:
            break
        x, y = rng.randrange(c.cw), rng.randrange(int(0.06 * c.ch), int(0.92 * c.ch))
        if c.blocked(x, y) or c.blocked(x + 2, y + 2) or c.blocked(x - 2, y - 2):
            continue
        col = PAL["G"] if rng.random() < 0.25 else FIREFLY
        for dx, dy in ((1, 0), (-1, 0), (0, 1), (0, -1)):
            c.put(x + dx, y + dy, alpha(col, 0.3))
        c.put(x, y, col)
        placed += 1
    c.blit(p)


def draw_front(p, w, h, clear):
    c = Canvas(w, h, clear)
    gy = ground_y(c, clear)
    room = max(14, int(0.16 * c.cw))
    for side in (0, 1):
        c.mirror = bool(side)
        grass(c, gy, room, random.Random(41 + side))
    c.mirror = False
    mushrooms(c, gy, room, random.Random(3))
    flowers(c, gy, room, random.Random(9))
    c.blit(p)
    vines.draw(p, w, h, clear)  # smooth grown vines (vines.py), over the pixel art


# ---- back layer ---------------------------------------------------------------------------

def trunk(c, x0, width, height, a, rng):
    """A dark trunk from the bottom edge, moonlit on its inner edge, a little bark."""
    top = c.ch - height
    for y in range(top, c.ch):
        flare = 2 if y > c.ch - 4 else 1 if y > c.ch - 8 else 0  # roots
        for x in range(x0 - flare, x0 + width + flare):
            c.put(x, y, alpha(TRUNK, a))
        c.put(x0 + width + flare - 1, y, alpha(mix(TRUNK, MOON, 0.35), a))
        if rng.random() < 0.08:
            c.put(x0 + rng.randrange(max(1, width - 1)), y, alpha(mix(TRUNK, MOON, 0.15), a))


def ground_y(c, clear):
    floor = max((r.bottom() for r in c.clear), default=0) + 2
    return max(int(floor), c.ch - 7)


def ground(c, clear):
    gy = ground_y(c, clear)
    for x in range(c.cw):
        top = gy + (1 if (x // 9) % 3 == 0 else 0) - (1 if (x // 13) % 4 == 1 else 0)
        c.put(x, top, alpha(MOON, 0.18))
        for y in range(top + 1, c.ch):
            c.put(x, y, alpha(GROUND, 0.92))


# ---- bottom corners -------------------------------------------------------------------------

def grass(c, gy, room, rng):
    for x in range(0, room + 4):
        if rng.random() < 0.7:
            hgt = rng.choice((1, 2, 2, 3, 3, 4))
            for k in range(hgt):
                c.put(x, gy - k, PAL["d"] if k == 0 else PAL["g"] if k < hgt - 1 else PAL["G"])


def mushrooms(c, gy, room, rng):
    """Bottom left: a big toadstool, a medium and small ones, glowing gills."""
    for name, x in (("fern", 0), ("fern", 15), ("fern", 30), ("tuft", 11), ("tuft", 26), ("tuft", 38)):
        rows = SPRITES[name]
        if x + len(rows[0]) <= room + 6:
            c.stamp(name, x, gy - len(rows) + 1)
    plan = (("mushroom_big", 4), ("mushroom_mid", 22), ("mushroom_small", 34), ("mushroom_small", 0),
            ("mushroom_mid", 40))
    for name, x in plan:
        rows = SPRITES[name]
        y = gy - len(rows) + 1
        if x + len(rows[0]) > room + 6 or not c.fits(name, x, y):
            continue
        c.dither_disc(x + len(rows[0]) / 2, gy - 0.35 * len(rows), 0.75 * len(rows[0]), GLOW, 0.16)
    for name, x in plan:
        rows = SPRITES[name]
        y = gy - len(rows) + 1
        if x + len(rows[0]) <= room + 6 and c.fits(name, x, y):
            c.stamp(name, x, y)
    for x in range(1, room):  # a few glowing spores in the air
        if rng.random() < 0.12:
            c.put(x, gy - rng.randint(18, 28), alpha(GLOW, 0.5))


def flowers(c, gy, room, rng):
    """Bottom right: stems with daisies, tulips, bellflowers, lavender and buds."""
    x0 = c.cw - room - 2
    kinds = ["daisy", "tulip", "lavender", "bell", "daisy", "tulip", "bud", "lavender", "bell", "daisy"]
    x = x0 + 1
    i = 0
    while x < c.cw - 2:
        kind = kinds[i % len(kinds)]
        stem_h = rng.randint(8, 20)
        top = gy - stem_h
        if c.blocked(x, top - 7):
            x += 3
            i += 1
            continue
        for y in range(top, gy + 1):  # stem
            c.put(x, y, PAL["d"])
        if stem_h > 9:  # a leaf or two on the stem
            ly = rng.randint(top + 3, gy - 4)
            if rng.random() < 0.5:
                c.stamp("stem_leaf_right", x, ly)
            else:
                c.stamp("stem_leaf_left", x - 2, ly)
        if kind == "lavender":
            for k in range(7):
                col = PAL["l"] if k % 2 else PAL["V"]
                c.put(x, top - k, col)
                if k < 5 and k % 2 == 0:
                    c.put(x - 1, top - k, PAL["l"])
                    c.put(x + 1, top - k, PAL["L"])
        else:
            rows = SPRITES[kind]
            wd, hg = len(rows[0]), len(rows)
            c.stamp(kind, x - wd // 2, top - hg + 1)
        x += rng.randint(3, 6)
        i += 1
