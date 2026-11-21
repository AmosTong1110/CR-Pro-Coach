"""Original code-drawn art for the arena, units and card faces (QPainter vectors only, no image assets).

Everything here is CRbot's own look: a role glyph per unit type (sword, bow, wings, turret, bomb, star, bolt),
a stable per-card colour, stone towers with a team roof and a grass / river / wooden-bridge arena. No game
images, characters or icons are used or imitated. Rendering only: nothing here reads or changes engine state
beyond the card records it is given.
"""
from __future__ import annotations

import hashlib
import math
import random
from typing import Any, Dict, Mapping, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import (QColor, QFont, QLinearGradient, QPainter, QPainterPath, QPen, QPixmap, QPolygonF,
                           QRadialGradient)

from backend_engine.simulator import geometry as G

# -- palette ----------------------------------------------------------------------------------------------------
BLUE, RED = QColor("#3d8bff"), QColor("#ff5247")
GOLD, VIOLET = QColor("#ffd23f"), QColor("#a77bff")
GRASS_TOP, GRASS_MID, GRASS_LOW = QColor("#5e9f44"), QColor("#68ad4b"), QColor("#5a9940")
BANK, WATER_EDGE, WATER_MID = QColor("#3e6e2c"), QColor("#2c79c0"), QColor("#48a2e6")
WOOD, WOOD_DARK, WOOD_RAIL = QColor("#a87b4a"), QColor("#7a5530"), QColor("#6b4524")
STONE_LIGHT, STONE_MID, STONE_DARK = QColor("#c9ccd2"), QColor("#8b9099"), QColor("#5d6169")
ELIXIR_TOP, ELIXIR_LOW = QColor("#f39bff"), QColor("#a21cc4")

ROLES = ("melee", "ranged", "air", "building", "kamikaze", "champion", "spell")


def side_color(side: int) -> QColor:
    return QColor(BLUE if side == 0 else RED)


# -- card classification ----------------------------------------------------------------------------------------
def _kinds(card: Mapping[str, Any]) -> set:
    return {m["kind"] if isinstance(m, dict) else m for m in card.get("mechanics", [])}


def role(card: Mapping[str, Any]) -> str:
    """The unit type shown by the glyph, read from the card record (rarity, type, mechanics, movement, attack)."""
    k = _kinds(card)
    if card.get("rarity") == "champion" or "champion_ability" in k:
        return "champion"
    kind = card.get("type")
    if kind == "spell":
        return "spell"
    if kind == "building":
        return "building"
    if "suicide" in k:
        return "kamikaze"
    if card.get("movement", {}).get("layer") == "air" or "air_unit" in k:
        return "air"
    if card.get("attack", {}).get("range_tiles", 0) >= 2.0:
        return "ranged"
    return "melee"


def card_color(card_id: str) -> QColor:
    """A stable colour per card id (hash -> hue); CRbot's own scheme, not tied to any game rarity colours."""
    h = int(hashlib.md5(card_id.encode()).hexdigest()[:4], 16) % 360
    return QColor.fromHsl(h, 125, 118)


# -- glyphs (unit box -1..1) ------------------------------------------------------------------------------------
def _poly(pts) -> QPainterPath:
    path = QPainterPath()
    path.addPolygon(QPolygonF([QPointF(x, y) for x, y in pts]))
    path.closeSubpath()
    return path


def _glyph(kind: str) -> QPainterPath:
    p = QPainterPath()
    if kind == "melee":                                     # sword
        p = _poly([(0, -0.95), (0.13, -0.78), (0.13, 0.28), (-0.13, 0.28), (-0.13, -0.78)])
        p.addRect(QRectF(-0.45, 0.28, 0.9, 0.15))
        p.addRect(QRectF(-0.08, 0.43, 0.16, 0.32))
        p.addEllipse(QPointF(0, 0.85), 0.13, 0.13)
    elif kind == "ranged":                                  # bow + arrow
        bow = QPainterPath()
        bow.moveTo(-0.15, -0.85)
        bow.cubicTo(0.75, -0.55, 0.75, 0.55, -0.15, 0.85)
        bow.cubicTo(0.5, 0.45, 0.5, -0.45, -0.15, -0.85)
        p.addPath(bow)
        p.addRect(QRectF(-0.8, -0.05, 1.3, 0.1))
        p.addPath(_poly([(0.85, 0), (0.5, -0.2), (0.5, 0.2)]))
        p.addPath(_poly([(-0.8, -0.05), (-0.95, -0.22), (-0.65, -0.05)]))
        p.addPath(_poly([(-0.8, 0.05), (-0.95, 0.22), (-0.65, 0.05)]))
    elif kind == "air":                                     # wings
        for s in (-1, 1):
            w = QPainterPath()
            w.moveTo(0.08 * s, -0.05)
            w.cubicTo(0.5 * s, -0.75, 0.95 * s, -0.55, 0.95 * s, -0.35)
            w.cubicTo(0.75 * s, -0.25, 0.85 * s, 0.0, 0.62 * s, 0.05)
            w.cubicTo(0.5 * s, 0.2, 0.4 * s, 0.25, 0.08 * s, 0.25)
            w.closeSubpath()
            p.addPath(w)
        p.addEllipse(QPointF(0, 0.1), 0.14, 0.32)
    elif kind == "building":                                # turret block with a door
        p = _poly([(-0.7, 0.8), (-0.7, -0.35), (-0.45, -0.35), (-0.45, -0.65), (-0.15, -0.65), (-0.15, -0.35),
                   (0.15, -0.35), (0.15, -0.65), (0.45, -0.65), (0.45, -0.35), (0.7, -0.35), (0.7, 0.8)])
        door = QPainterPath()
        door.addRoundedRect(QRectF(-0.18, 0.2, 0.36, 0.6), 0.18, 0.18)
        p = p.subtracted(door)
    elif kind == "kamikaze":                                # bomb with a spark
        p.addEllipse(QPointF(-0.1, 0.2), 0.62, 0.62)
        p.addRect(QRectF(0.2, -0.55, 0.22, 0.25))
        for a in range(0, 360, 60):
            r = math.radians(a)
            p.addEllipse(QPointF(0.5 + 0.28 * math.cos(r), -0.72 + 0.28 * math.sin(r)), 0.07, 0.07)
    elif kind == "champion":                                # five-point star
        pts = []
        for i in range(10):
            r = 0.95 if i % 2 == 0 else 0.42
            a = math.radians(-90 + i * 36)
            pts.append((r * math.cos(a), r * math.sin(a) + 0.05))
        p = _poly(pts)
    elif kind == "spell":                                   # lightning bolt
        p = _poly([(0.2, -0.95), (-0.55, 0.12), (-0.05, 0.12), (-0.25, 0.95), (0.55, -0.15), (0.05, -0.15),
                   (0.3, -0.95)])
    elif kind == "crown":
        p = _poly([(-0.8, 0.5), (-0.8, -0.3), (-0.4, 0.05), (0, -0.6), (0.4, 0.05), (0.8, -0.3), (0.8, 0.5)])
    return p


_GLYPHS: Dict[str, QPainterPath] = {}


def glyph(kind: str) -> QPainterPath:
    if kind not in _GLYPHS:
        _GLYPHS[kind] = _glyph(kind)
    return _GLYPHS[kind]


def draw_glyph(p: QPainter, kind: str, c: QPointF, r: float, color: QColor = QColor("white"),
               shadow: bool = True) -> None:
    g = glyph(kind)
    p.save()
    p.translate(c)
    p.scale(r, r)
    p.setPen(Qt.NoPen)
    if shadow:
        p.setBrush(QColor(0, 0, 0, int(90 * color.alphaF())))
        p.translate(0.06, 0.08)
        p.drawPath(g)
        p.translate(-0.06, -0.08)
    p.setBrush(color)
    p.drawPath(g)
    p.restore()


def hp_bar(p: QPainter, cx: float, top: float, width: float, frac: float, side: int, h: float = 4.0) -> QRectF:
    """Rounded health bar in the side colour on a dark track; returns the bar rect."""
    frac = max(0.0, min(1.0, frac))
    r = QRectF(cx - width / 2, top, width, h)
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(10, 12, 16, 210))
    p.drawRoundedRect(r.adjusted(-1, -1, 1, 1), h / 2 + 1, h / 2 + 1)
    if frac > 0:
        col = side_color(side)
        g = QLinearGradient(r.topLeft(), r.bottomLeft())
        g.setColorAt(0, col.lighter(145))
        g.setColorAt(1, col)
        p.setBrush(g)
        p.drawRoundedRect(QRectF(r.left(), r.top(), max(h, r.width() * frac), h), h / 2, h / 2)
    return r


# -- arena background (drawn once per size, then cached as a pixmap) ----------------------------------------------
def arena_pixmap(board, bg: QColor) -> QPixmap:
    """Grass with soft stripes and tufts, worn lanes, banks, river with ripples, two plank bridges, vignette."""
    dpr = board.devicePixelRatioF()
    pm = QPixmap(int(board.width() * dpr), int(board.height() * dpr))
    pm.setDevicePixelRatio(dpr)
    pm.fill(bg)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    ts, _, _ = board._geom()
    full = board.tile_rect(0, 0, G.W, G.H)
    rnd = random.Random(42)                       # fixed seed: the same arena every time
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#0e1013"))
    p.drawRoundedRect(full.adjusted(-5, -5, 5, 5), 9, 9)
    for y0, y1 in ((0, G.RIVER_Y0), (G.RIVER_Y1, G.H)):
        r = board.tile_rect(0, y0, G.W, y1 - y0)
        g = QLinearGradient(r.topLeft(), r.bottomLeft())
        g.setColorAt(0, GRASS_TOP)
        g.setColorAt(0.5, GRASS_MID)
        g.setColorAt(1, GRASS_LOW)
        p.setBrush(g)
        p.drawRect(r)
    p.setBrush(QColor(255, 255, 255, 10))         # mowing stripes every other row (no hard grid)
    for y in range(0, int(G.H), 2):
        if not (G.RIVER_Y0 <= y < G.RIVER_Y1):
            p.drawRect(board.tile_rect(0, y, G.W, 1))
    for bx in G.BRIDGE_X:                         # worn lanes from the bridges toward the towers
        for ya, yb in ((2.5, G.RIVER_Y0), (G.RIVER_Y1, G.H - 2.5)):
            p.setBrush(QColor(196, 170, 112, 30))
            p.drawRoundedRect(board.tile_rect(bx - 0.9, ya, 1.8, yb - ya), ts * 0.8, ts * 0.8)
    tuft_w = max(1.0, ts * 0.06)
    dark, light = QColor(40, 90, 30, 70), QColor(150, 210, 110, 60)
    for _ in range(int(G.W * G.H * 1.4)):
        x, y = rnd.uniform(0, G.W), rnd.uniform(0, G.H)
        shade = dark if rnd.random() < 0.5 else light
        if G.RIVER_Y0 - 0.3 < y < G.RIVER_Y1 + 0.3:
            continue
        c = board.to_px(x, y)
        p.setPen(QPen(shade, tuft_w, Qt.SolidLine, Qt.RoundCap))
        for dx in (-0.08, 0, 0.08):
            p.drawLine(QPointF(c.x() + dx * ts, c.y()), QPointF(c.x() + dx * ts * 1.8, c.y() - ts * 0.18))
    p.setPen(Qt.NoPen)
    rr = board.tile_rect(0, G.RIVER_Y0, G.W, G.RIVER_Y1 - G.RIVER_Y0)
    p.setBrush(BANK)
    p.drawRect(rr.adjusted(0, -ts * 0.18, 0, ts * 0.18))
    g = QLinearGradient(rr.topLeft(), rr.bottomLeft())
    g.setColorAt(0, WATER_EDGE)
    g.setColorAt(0.5, WATER_MID)
    g.setColorAt(1, WATER_EDGE.darker(106))
    p.setBrush(g)
    p.drawRect(rr)
    p.setPen(QPen(QColor(255, 255, 255, 60), max(1.0, ts * 0.07), Qt.SolidLine, Qt.RoundCap))
    for _ in range(26):
        x, y = rnd.uniform(0.3, G.W - 1), rnd.uniform(G.RIVER_Y0 + 0.3, G.RIVER_Y1 - 0.3)
        c = board.to_px(x, y)
        p.drawLine(c, QPointF(c.x() + ts * rnd.uniform(0.4, 0.9), c.y()))
    p.setPen(Qt.NoPen)
    for bx in G.BRIDGE_X:
        br = board.tile_rect(bx - G.BRIDGE_HALF_W - 0.15, G.RIVER_Y0 - 0.35, 2 * G.BRIDGE_HALF_W + 0.3,
                             G.RIVER_Y1 - G.RIVER_Y0 + 0.7)
        p.setBrush(QColor(0, 0, 0, 70))
        p.drawRoundedRect(br.translated(ts * 0.12, ts * 0.18), 4, 4)
        p.setBrush(WOOD)
        p.drawRoundedRect(br, 4, 4)
        p.setPen(QPen(WOOD_DARK, max(1.0, ts * 0.06)))
        for i in range(1, 6):
            y = br.top() + br.height() * i / 6
            p.drawLine(QPointF(br.left() + 2, y), QPointF(br.right() - 2, y))
        p.setPen(Qt.NoPen)
        p.setBrush(WOOD_RAIL)
        p.drawRect(QRectF(br.left(), br.top(), ts * 0.22, br.height()))
        p.drawRect(QRectF(br.right() - ts * 0.22, br.top(), ts * 0.22, br.height()))
    vg = QRadialGradient(full.center(), max(full.width(), full.height()) * 0.75)
    vg.setColorAt(0.6, QColor(0, 0, 0, 0))
    vg.setColorAt(1, QColor(0, 0, 0, 70))
    p.setBrush(vg)
    p.drawRect(full)
    p.end()
    return pm


# -- towers -----------------------------------------------------------------------------------------------------
def draw_tower(p: QPainter, c: QPointF, r: float, ts: float, side: int, king: bool, active: bool,
               hp: float, max_hp: float, hurt: float = 0.0) -> None:
    """Stone tower with a darker front face (depth), crenellations, a team roof and an HP bar with the number."""
    col = side_color(side)
    depth = r * 0.42
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(0, 0, 0, 80))
    p.drawEllipse(QPointF(c.x() + r * 0.15, c.y() + r * 0.85), r * 1.05, r * 0.38)
    body = QRectF(c.x() - r * 0.92, c.y() - r * 0.95, r * 1.84, r * 1.84)
    front = QRectF(body.left(), body.bottom() - depth, body.width(), depth + r * 0.2)
    g = QLinearGradient(front.topLeft(), front.topRight())
    g.setColorAt(0, STONE_DARK.lighter(110))
    g.setColorAt(0.5, STONE_MID)
    g.setColorAt(1, STONE_DARK)
    p.setBrush(g)
    p.drawRoundedRect(front, r * 0.2, r * 0.2)
    top = QRectF(body.left(), body.top(), body.width(), body.height() - depth)
    g = QLinearGradient(top.topLeft(), top.bottomRight())
    g.setColorAt(0, STONE_LIGHT)
    g.setColorAt(1, STONE_MID.lighter(112))
    p.setBrush(g)
    p.drawRoundedRect(top, r * 0.22, r * 0.22)
    p.setPen(QPen(QColor(90, 94, 102, 120), max(1.0, ts * 0.05)))
    for i in (1, 2):
        y = front.top() + front.height() * i / 3
        p.drawLine(QPointF(front.left() + 2, y), QPointF(front.right() - 2, y))
    p.setPen(Qt.NoPen)
    p.setBrush(QColor("#d8dbe0"))
    n = 5 if king else 4
    mw = top.width() / (2 * n - 1)
    for i in range(n):
        p.drawRect(QRectF(top.left() + i * 2 * mw, top.top() - r * 0.12, mw, r * 0.2))
    inner = top.adjusted(r * 0.28, r * 0.3, -r * 0.28, -r * 0.18)
    base = col if active else col.darker(150)
    g = QRadialGradient(inner.center() - QPointF(r * 0.2, r * 0.2), inner.width())
    g.setColorAt(0, base.lighter(135))
    g.setColorAt(1, base.darker(120))
    p.setBrush(g)
    p.setPen(QPen(base.darker(170), max(1.0, ts * 0.08)))
    p.drawRoundedRect(inner, r * 0.18, r * 0.18)
    if king:
        p.save()
        p.translate(inner.center())
        s = inner.width() * 0.3
        p.scale(s, s)
        p.setPen(Qt.NoPen)
        p.setBrush(GOLD if active else QColor("#b8a46a"))
        p.drawPath(glyph("crown"))
        p.restore()
        if not active:
            f = QFont()
            f.setPixelSize(max(9, int(ts * 0.5)))
            f.setBold(True)
            p.setFont(f)
            p.setPen(QColor(255, 255, 255, 210))
            p.drawText(QRectF(inner.right() - ts * 0.9, inner.top() - ts * 0.95, ts * 1.6, ts), Qt.AlignCenter, "z z")
    else:
        draw_glyph(p, "ranged", inner.center(), inner.width() * 0.3)
    if hurt > 0:                                  # brief white flash when hit
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(255, 255, 255, int(110 * hurt)))
        p.drawRoundedRect(body.adjusted(0, -r * 0.12, 0, r * 0.2), r * 0.22, r * 0.22)
    bw, bh = r * 2.1, max(10.0, ts * 0.5)
    top_y = body.top() - r * 0.2 - bh
    hp_bar(p, c.x(), top_y, bw, hp / max_hp if max_hp else 0.0, side, h=bh)
    f = QFont()
    f.setPixelSize(max(8, int(bh * 0.85)))
    f.setBold(True)
    p.setFont(f)
    p.setPen(QColor(255, 255, 255, 240))
    p.drawText(QRectF(c.x() - bw / 2, top_y, bw, bh), Qt.AlignCenter, f"{int(hp)}")


def draw_rubble(p: QPainter, c: QPointF, r: float, side: int) -> None:
    """A destroyed tower: scorched base with stones and a small side flag."""
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(30, 30, 30, 90))
    p.drawEllipse(c, r * 1.0, r * 0.75)
    rnd = random.Random(int(c.x() * 7 + c.y() * 13))
    for _ in range(9):
        dx, dy = rnd.uniform(-0.7, 0.7) * r, rnd.uniform(-0.45, 0.45) * r
        s = rnd.uniform(0.14, 0.26) * r
        p.setBrush(STONE_MID if rnd.random() < 0.5 else STONE_DARK)
        p.drawRoundedRect(QRectF(c.x() + dx - s, c.y() + dy - s * 0.7, 2 * s, 1.4 * s), s * 0.4, s * 0.4)
    p.setBrush(side_color(side).darker(130))
    p.drawEllipse(c, r * 0.12, r * 0.12)


# -- unit tokens (cached pixmaps) -------------------------------------------------------------------------------
class TokenCache:
    """Pre-rendered unit tokens keyed by (card, side, role, size, variant): a cheap drawPixmap per unit per frame."""

    def __init__(self, limit: int = 600):
        self.limit = limit
        self._pm: Dict[Tuple, QPixmap] = {}

    def clear(self) -> None:
        self._pm.clear()

    def get(self, card_id: str, side: int, kind: str, r: float, dpr: float, ring: str = "") -> QPixmap:
        key = (card_id, side, kind, int(r * 4), round(dpr, 2), ring)
        pm = self._pm.get(key)
        if pm is None:
            if len(self._pm) >= self.limit:
                self._pm.clear()
            pm = self._pm[key] = _render_token(card_id, side, kind, int(r * 4) / 4, dpr, ring)
        return pm


def token_extent(r: float) -> float:
    """Half-size of a token pixmap (room for the outline, champion ring and card dot)."""
    return r * 1.32 + 3


def _render_token(card_id: str, side: int, kind: str, r: float, dpr: float, ring: str) -> QPixmap:
    half = token_extent(r)
    size = int(math.ceil(2 * half * dpr))
    pm = QPixmap(size, size)
    pm.setDevicePixelRatio(dpr)
    pm.fill(Qt.transparent)
    p = QPainter(pm)
    p.setRenderHint(QPainter.Antialiasing)
    c = QPointF(half, half)
    col = side_color(side)
    body = QColor.fromHsl(col.hslHue(), int(col.hslSaturation() * 0.75), int(col.lightness() * 0.78))
    edge = col.lighter(125)
    pen_w = max(2.0, r * 0.16)
    if kind == "building":
        rect = QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r)
        g = QLinearGradient(rect.topLeft(), rect.bottomLeft())
        g.setColorAt(0, body.lighter(130))
        g.setColorAt(1, body.darker(120))
        p.setBrush(g)
        p.setPen(QPen(edge, pen_w))
        p.drawRoundedRect(rect, r * 0.3, r * 0.3)
    else:
        g = QRadialGradient(c - QPointF(r * 0.35, r * 0.4), r * 1.4)
        g.setColorAt(0, body.lighter(150))
        g.setColorAt(1, body.darker(125))
        p.setBrush(g)
        p.setPen(QPen(edge, pen_w))
        p.drawEllipse(c, r, r)
    if ring:
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(GOLD if ring == "champion" else VIOLET, max(1.5, r * 0.1)))
        p.drawEllipse(c, r * 1.2, r * 1.2)
    draw_glyph(p, kind, c, r * 0.56)
    dot = card_color(card_id)                    # card-colour dot: tells cards apart inside a side
    dr = max(2.5, r * 0.3)
    dc = QPointF(c.x() + r * 0.72, c.y() + r * 0.72)
    p.setPen(QPen(QColor(15, 17, 22), max(1.0, dr * 0.35)))
    p.setBrush(dot)
    p.drawEllipse(dc, dr, dr)
    p.end()
    return pm


# -- card faces (hand + next) -----------------------------------------------------------------------------------
def elixir_drop(p: QPainter, bx: float, by: float, s: float, cost: int) -> None:
    d = QPainterPath()
    d.moveTo(bx, by - s * 1.25)
    d.cubicTo(bx + s * 0.9, by - s * 0.2, bx + s, by + s * 0.9, bx, by + s)
    d.cubicTo(bx - s, by + s * 0.9, bx - s * 0.9, by - s * 0.2, bx, by - s * 1.25)
    g = QLinearGradient(QPointF(bx, by - s), QPointF(bx, by + s))
    g.setColorAt(0, ELIXIR_TOP)
    g.setColorAt(1, ELIXIR_LOW)
    p.setBrush(g)
    p.setPen(QPen(QColor(40, 0, 50, 170), 1))
    p.drawPath(d)
    f = QFont()
    f.setPixelSize(max(9, int(s * 1.2)))
    f.setWeight(QFont.Black)
    p.setFont(f)
    p.setPen(QColor("white"))
    p.drawText(QRectF(bx - s, by - s * 0.55, 2 * s, 1.5 * s), Qt.AlignCenter, str(cost))


def draw_card_face(p: QPainter, rect: QRectF, card_id: str, name: str, cost: int, kind: str, *, afford: bool = True,
                   selected: bool = False, evo: bool = False, enabled: bool = True, compact: bool = False) -> None:
    """Card colour + type glyph + elixir drop + name strip. Unaffordable or locked cards turn grey."""
    r = QRectF(rect).adjusted(3, 4, -3, -3)
    if selected:
        r.translate(0, -3)
    lit = afford and enabled
    col = card_color(card_id)
    if not lit:
        col = QColor.fromHsl(col.hslHue(), 22, 78)
    rad = 9.0
    p.setPen(Qt.NoPen)
    p.setBrush(QColor(0, 0, 0, 90))
    p.drawRoundedRect(r.translated(0, 3), rad, rad)
    g = QLinearGradient(r.topLeft(), r.bottomLeft())
    g.setColorAt(0, col.lighter(130))
    g.setColorAt(1, col.darker(145))
    p.setBrush(g)
    border = GOLD if selected else (VIOLET if evo else QColor(255, 255, 255, 45))
    p.setPen(QPen(border, 2.5 if (selected or evo) else 1))
    p.drawRoundedRect(r, rad, rad)
    strip_h = 0.0 if compact else min(18.0, r.height() * 0.26)
    icon_r = max(6.0, min((r.height() - strip_h) * 0.34, r.width() * 0.22))
    draw_glyph(p, kind, QPointF(r.center().x(), r.top() + (r.height() - strip_h) * 0.54), icon_r,
               QColor(255, 255, 255, 235 if lit else 125))
    f = QFont()
    f.setBold(True)
    if strip_h:
        strip = QRectF(r.left() + 1, r.bottom() - strip_h, r.width() - 2, strip_h - 1)
        path = QPainterPath()
        path.addRoundedRect(strip, rad - 1, rad - 1)
        sq = QPainterPath()
        sq.addRect(QRectF(strip.left(), strip.top(), strip.width(), strip.height() / 2))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 115))
        p.drawPath(path.united(sq))
        f.setPixelSize(11)
        p.setFont(f)
        p.setPen(QColor(255, 255, 255, 235 if lit else 150))
        text = p.fontMetrics().elidedText(name, Qt.ElideRight, int(strip.width() - 6))
        p.drawText(strip, Qt.AlignCenter, text)
    if evo:
        f.setPixelSize(9)
        p.setFont(f)
        tag = QRectF(r.right() - 30, r.top() + 4, 26, 13)
        p.setPen(Qt.NoPen)
        p.setBrush(VIOLET)
        p.drawRoundedRect(tag, 6, 6)
        p.setPen(QColor("white"))
        p.drawText(tag, Qt.AlignCenter, "EVO")
    s = 8.0 if compact else 10.0
    elixir_drop(p, r.left() + s + 4, r.top() + s * 1.25 + 3, s, cost)
