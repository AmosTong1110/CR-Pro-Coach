"""Arena board widget: draws the 18x32 arena with QPainter and turns clicks into tile coordinates."""
from __future__ import annotations

from typing import List, Optional, Set, Tuple

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QBrush, QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QWidget

from backend_engine.simulator import geometry as G
from frontend_app.qt.session import AI, HUMAN, GameSession

SIDE_COLOR = {HUMAN: QColor("#2f7de1"), AI: QColor("#e0453a")}
GRASS_A, GRASS_B = QColor("#73b04f"), QColor("#6aa748")
RIVER, BRIDGE = QColor("#3b8fd9"), QColor("#a7814f")
LEGAL, ILLEGAL = QColor(80, 255, 120, 70), QColor(255, 60, 60, 110)


def short_name(name: str) -> str:
    words = name.replace("The ", "").split()
    return "".join(w[0] for w in words[:2]) if len(words) > 1 else name[:2]


class BoardWidget(QWidget):
    tileClicked = Signal(float, float)
    cancelled = Signal()

    def __init__(self, session: GameSession, parent=None):
        super().__init__(parent)
        self.session = session
        self.selected_card: Optional[str] = None
        self.legal: Set[Tuple[float, float]] = set()
        self.hover: Optional[Tuple[float, float]] = None
        self.show_ai_vision = True
        self.overlay_text: Optional[str] = None   # e.g. the 3-2-1 countdown after a rewind
        # review ghosts: (card_id, x, y, "actual" | "best")
        self.ghosts: List[Tuple[str, float, float, str]] = []
        self.banner: Optional[str] = None         # small text at the top of the board
        self.setMouseTracking(True)
        self.setMinimumSize(int(G.W * 15), int(G.H * 15))

    # -- geometry ----------------------------------------------------------
    def _geom(self) -> Tuple[float, float, float]:
        ts = min(self.width() / G.W, self.height() / G.H)
        return ts, (self.width() - G.W * ts) / 2, (self.height() - G.H * ts) / 2

    def to_px(self, x: float, y: float) -> QPointF:
        ts, ox, oy = self._geom()
        return QPointF(ox + x * ts, oy + (G.H - y) * ts)

    def to_tile(self, px: float, py: float) -> Tuple[float, float]:
        ts, ox, oy = self._geom()
        return (px - ox) / ts, G.H - (py - oy) / ts

    def tile_rect(self, x0: float, y0: float, w: float, h: float) -> QRectF:
        """Rect for tile-space box with lower-left (x0, y0)."""
        ts, _, _ = self._geom()
        tl = self.to_px(x0, y0 + h)
        return QRectF(tl.x(), tl.y(), w * ts, h * ts)

    # -- interaction -------------------------------------------------------
    def set_selected(self, card: Optional[str]) -> None:
        self.selected_card = card
        self.legal = set(self.session.legal_tiles(card)) if card else set()
        self.update()

    def refresh_legal(self) -> None:
        if self.selected_card:
            self.legal = set(self.session.legal_tiles(self.selected_card))

    def mouseMoveEvent(self, e) -> None:
        x, y = self.to_tile(e.position().x(), e.position().y())
        self.hover = G.snap(x, y) if G.in_bounds(x, y) else None
        if self.selected_card:
            self.update()

    def leaveEvent(self, e) -> None:
        self.hover = None
        self.update()

    def mousePressEvent(self, e) -> None:
        if e.button() == Qt.RightButton:
            self.cancelled.emit()
            return
        x, y = self.to_tile(e.position().x(), e.position().y())
        if G.in_bounds(x, y):
            self.tileClicked.emit(x, y)

    # -- painting ----------------------------------------------------------
    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#1e2a1a"))
        ts, _, _ = self._geom()
        self._paint_arena(p)
        self._paint_placement(p, ts)
        m = self.session.match
        for u in sorted(m.units.values(), key=lambda u: (u.layer == "air", u.uid)):
            if u.kind == "tower":
                self._paint_tower(p, u, ts)
            else:
                self._paint_unit(p, u, ts)
        self._paint_spells(p, ts)
        self._paint_ghosts(p, ts)
        if not self.ghosts:
            self._paint_result(p)
        self._paint_overlay(p)
        self._paint_banner(p)
        p.end()

    def _paint_ghosts(self, p: QPainter, ts: float) -> None:
        styles = {"actual": (QColor(235, 70, 70, 120), QColor("#ff8a80"), "You"),
                  "best": (QColor(60, 220, 100, 120), QColor("#b9f6ca"), "Best")}
        for card_id, x, y, kind in self.ghosts:
            fill, edge, tag = styles[kind]
            card = self.session.db.cards[card_id]
            c = self.to_px(x, y)
            p.setBrush(fill)
            p.setPen(QPen(edge, 2, Qt.DashLine))
            spell = card.get("spell", {})
            if "radius_tiles" in spell:
                r = spell["radius_tiles"] * ts
                p.drawEllipse(c, r, r)
            elif "rolling_range_tiles" in spell:
                w = spell["width_tiles"]
                p.drawRect(self.tile_rect(x - w / 2, y, w, spell["rolling_range_tiles"]))
            else:
                r = max(card.get("collision_radius_tiles", 0.5), 0.5) * ts * 1.2
                p.drawEllipse(c, r, r)
            f = QFont()
            f.setPixelSize(max(10, int(ts * 0.6)))
            f.setBold(True)
            p.setFont(f)
            p.setPen(QColor("white"))
            dy = -1.7 * ts if kind == "actual" else 0.8 * ts   # labels never collide on the same tile
            label = QRectF(c.x() - 3 * ts, c.y() + dy, 6 * ts, ts)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 150))
            p.drawRoundedRect(label.adjusted(ts, 0, -ts, 0), 4, 4)
            p.setPen(edge)
            p.drawText(label, Qt.AlignCenter, f"{tag} {card['name']}")

    def _paint_banner(self, p: QPainter) -> None:
        if not self.banner:
            return
        f = QFont()
        f.setPixelSize(13)
        f.setBold(True)
        p.setFont(f)
        r = QRectF(8, 8, self.width() - 16, 26)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 170))
        p.drawRoundedRect(r, 6, 6)
        p.setPen(QColor("#ffd400"))
        p.drawText(r, Qt.AlignCenter, self.banner)

    def _paint_overlay(self, p: QPainter) -> None:
        if not self.overlay_text:
            return
        p.fillRect(self.rect(), QColor(0, 0, 0, 70))
        f = QFont()
        f.setPixelSize(max(48, self.width() // 4))
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor(0, 0, 0, 140))
        p.drawText(self.rect().adjusted(4, 4, 4, 4), Qt.AlignCenter, self.overlay_text)
        p.setPen(QColor("#ffd400"))
        p.drawText(self.rect(), Qt.AlignCenter, self.overlay_text)

    def _paint_arena(self, p: QPainter) -> None:
        p.setPen(Qt.NoPen)
        for y in range(int(G.H)):
            for x in range(int(G.W)):
                p.setBrush(GRASS_A if (x + y) % 2 == 0 else GRASS_B)
                p.drawRect(self.tile_rect(x, y, 1, 1))
        p.setBrush(RIVER)
        p.drawRect(self.tile_rect(0, G.RIVER_Y0, G.W, G.RIVER_Y1 - G.RIVER_Y0))
        p.setBrush(BRIDGE)
        for bx in G.BRIDGE_X:
            p.drawRect(self.tile_rect(bx - G.BRIDGE_HALF_W, G.RIVER_Y0 - 0.2, 2 * G.BRIDGE_HALF_W,
                                      G.RIVER_Y1 - G.RIVER_Y0 + 0.4))

    def _paint_placement(self, p: QPainter, ts: float) -> None:
        if not self.selected_card:
            return
        p.setPen(Qt.NoPen)
        p.setBrush(LEGAL)
        for (x, y) in self.legal:
            p.drawRect(self.tile_rect(x - 0.5, y - 0.5, 1, 1))
        if self.hover is None:
            return
        ok = self.hover in self.legal
        card = self.session.db.cards[self.selected_card]
        color = QColor(60, 220, 100, 150) if ok else ILLEGAL
        p.setBrush(color)
        p.setPen(QPen(QColor(255, 255, 255, 200), 1.5))
        c = self.to_px(*self.hover)
        spell = card.get("spell", {})
        if "radius_tiles" in spell:
            r = spell["radius_tiles"] * ts
            p.drawEllipse(c, r, r)
        elif "rolling_range_tiles" in spell:
            w = spell["width_tiles"]
            p.drawRect(self.tile_rect(self.hover[0] - w / 2, self.hover[1], w, spell["rolling_range_tiles"]))
        else:
            r = max(card.get("collision_radius_tiles", 0.5), 0.45) * ts
            p.drawEllipse(c, r, r)
            rng = card.get("attack", {}).get("range_tiles", 0)
            if rng > 1.5:  # show attack range for ranged cards
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(255, 255, 255, 110), 1, Qt.DashLine))
                p.drawEllipse(c, rng * ts, rng * ts)

    def _hp_bar(self, p: QPainter, cx: float, top: float, width: float, frac: float, side: int) -> None:
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(0, 0, 0, 160))
        p.drawRect(QRectF(cx - width / 2, top, width, 4))
        p.setBrush(SIDE_COLOR[side].lighter(130))
        p.drawRect(QRectF(cx - width / 2, top, width * max(0.0, min(1.0, frac)), 4))

    def _paint_tower(self, p: QPainter, u, ts: float) -> None:
        r = u.radius * ts
        c = self.to_px(u.x, u.y)
        base = SIDE_COLOR[u.side]
        p.setPen(QPen(base.darker(160), 2))
        p.setBrush(base if u.active else base.darker(140))
        p.drawRoundedRect(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), 4, 4)
        p.setPen(QColor("white"))
        f = QFont()
        f.setPixelSize(max(9, int(ts * 0.55)))
        f.setBold(True)
        p.setFont(f)
        p.drawText(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), Qt.AlignCenter, f"{int(u.hp)}")
        if u.tower_role == "king" and not u.active:
            f.setPixelSize(max(8, int(ts * 0.4)))
            p.setFont(f)
            p.drawText(QRectF(c.x() - r, c.y() + r * 0.25, 2 * r, r * 0.7), Qt.AlignCenter, "zzz")
        self._hp_bar(p, c.x(), c.y() - r - 6, 2 * r, u.hp / u.max_hp, u.side)

    def _paint_unit(self, p: QPainter, u, ts: float) -> None:
        r = max(u.radius, 0.4) * ts
        c = self.to_px(u.x, u.y)
        col = QColor(SIDE_COLOR[u.side])
        if u.deploy_ms > 0:
            col.setAlpha(110)
        unseen = self.show_ai_vision and not self.session.ai_has_seen(u)
        if u.layer == "air":  # shadow under flyers
            p.setPen(Qt.NoPen)
            p.setBrush(QColor(0, 0, 0, 60))
            p.drawEllipse(QPointF(c.x() + r * 0.3, c.y() + r * 0.5), r * 0.8, r * 0.5)
        p.setBrush(col)
        pen = QPen(QColor("#ffd400") if unseen else col.darker(170), 2.5 if unseen else 1.5)
        if unseen:
            pen.setStyle(Qt.DashLine)
        p.setPen(pen)
        if u.kind == "building":
            p.drawRect(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r))
        else:
            p.drawEllipse(c, r, r)
        p.setPen(QColor("white"))
        f = QFont()
        f.setPixelSize(max(8, int(r * 0.85)))
        f.setBold(True)
        p.setFont(f)
        name = self.session.db.cards[u.card_id]["name"]
        p.drawText(QRectF(c.x() - r, c.y() - r, 2 * r, 2 * r), Qt.AlignCenter, short_name(name))
        if u.hp < u.max_hp:
            self._hp_bar(p, c.x(), c.y() - r - 5, 2 * r, u.hp / u.max_hp, u.side)

    def _paint_spells(self, p: QPainter, ts: float) -> None:
        m = self.session.match
        for sp in m.spells:
            c = self.to_px(sp.x, sp.y)
            p.setPen(QPen(QColor("#ff9b2f"), 2, Qt.DashLine))
            p.setBrush(QColor(255, 140, 40, 60))
            p.drawEllipse(c, sp.radius * ts, sp.radius * ts)
        for rs in m.rolling:
            p.setPen(QPen(QColor("#5a3b1c"), 2))
            p.setBrush(QColor("#8b5a2b"))
            p.drawRect(self.tile_rect(rs.x - rs.width / 2, rs.y - 0.4, rs.width, 0.8))

    def _paint_result(self, p: QPainter) -> None:
        r = self.session.result
        if r is None:
            return
        p.fillRect(self.rect(), QColor(0, 0, 0, 120))
        if r.winner is None:
            text, col = "Draw", QColor("#dddddd")
        elif r.winner == HUMAN:
            text, col = "Victory", QColor("#ffd400")
        else:
            text, col = "Defeat", QColor("#ff6b5e")
        f = QFont()
        f.setPixelSize(max(24, self.width() // 10))
        f.setBold(True)
        p.setFont(f)
        p.setPen(col)
        p.drawText(self.rect().adjusted(0, -40, 0, -40), Qt.AlignCenter, text)
        f.setPixelSize(max(14, self.width() // 26))
        p.setFont(f)
        p.setPen(QColor("white"))
        reason = {"king_destroyed": "3 Crowns", "crowns": "Crown lead",
                  "tiebreaker": "Lowest tower HP", "draw": ""}[r.reason]
        p.drawText(self.rect().adjusted(0, 40, 0, 40), Qt.AlignCenter, f"{r.crowns[0]} : {r.crowns[1]}   {reason}")
