"""Small reusable CRbot widgets: card panels, deck grids, stat tiles."""
from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

from PySide6.QtCore import QRectF, Qt
from PySide6.QtGui import QPainter
from PySide6.QtWidgets import QFrame, QGridLayout, QLabel, QPushButton, QSizePolicy, QVBoxLayout, QWidget

from frontend_app.qt import art

from frontend_app.qt.theme import C, cls

_CHIP = (f"background:{C['panel2']}; border:1px solid {{border}}; border-radius:6px; padding:3px 2px;"
         f"font-size:11px; color:{C['text']};")
CHIP_STYLE, CHIP_EVO_STYLE = _CHIP.format(border=C["line"]), _CHIP.format(border=C["violet"])


def card_frame(spacing: int = 8, margins: int = 12) -> QFrame:
    """A rounded panel (QSS class `card`) with a vertical layout."""
    f = QFrame()
    cls(f, "card")
    f.setAttribute(Qt.WA_StyledBackground, True)
    lay = QVBoxLayout(f)
    lay.setContentsMargins(margins, margins, margins, margins)
    lay.setSpacing(spacing)
    return f


def label(text: str = "", style: str = "", wrap: bool = False, rich: bool = False) -> QLabel:
    lab = QLabel(text)
    if style:
        cls(lab, style)
    lab.setWordWrap(wrap)
    if rich:
        lab.setTextFormat(Qt.RichText)
    return lab


class DeckView(QWidget):
    """Eight cards as a 4x2 grid of chips: cost (elixir colour) + name; the evolution is outlined in violet."""

    def __init__(self, columns: int = 4, parent=None):
        super().__init__(parent)
        self.columns = columns
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.grid.setSpacing(4)
        self.chips = []
        for i in range(8):
            lab = QLabel()
            lab.setTextFormat(Qt.RichText)
            lab.setWordWrap(True)
            lab.setAlignment(Qt.AlignCenter)
            lab.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
            lab.setMinimumWidth(10)
            self.grid.addWidget(lab, i // columns, i % columns)
            self.chips.append(lab)
        for col in range(columns):
            self.grid.setColumnStretch(col, 1)

    def set_deck(self, cards: Sequence[str], evo: Sequence[str], name_of: Callable[[str], str],
                 cost_of: Callable[[str], Optional[int]]) -> None:
        for i, lab in enumerate(self.chips):
            if i >= len(cards):
                lab.setText("")
                lab.setVisible(False)
                continue
            c = cards[i]
            cost = cost_of(c)
            hero = f"{c}#hero" in evo
            evo_tag = (f"<span style='color:{C['violet']}'>Hero </span>" if hero else
                       f"<span style='color:{C['violet']}'>Evo </span>" if c in evo else "")
            cost_tag = f"<span style='color:{C['elixir']}; font-weight:700'>{cost}</span>&nbsp;" if cost else ""
            lab.setText(f"{cost_tag}{evo_tag}{name_of(c)}")
            style = CHIP_EVO_STYLE if (c in evo or hero) else CHIP_STYLE
            if lab.styleSheet() != style:
                lab.setStyleSheet(style)
            lab.setVisible(True)
            lab.updateGeometry()


class StatTile(QFrame):
    """A number with a caption under it (Analysis page header)."""

    def __init__(self, caption: str, parent=None):
        super().__init__(parent)
        cls(self, "card")
        self.setAttribute(Qt.WA_StyledBackground, True)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(14, 10, 14, 10)
        lay.setSpacing(0)
        self.value = label("—", "stat")
        self.caption = label(caption, "sub")
        lay.addWidget(self.value)
        lay.addWidget(self.caption)

    def set(self, value: str, color: Optional[str] = None) -> None:
        self.value.setText(value)
        self.value.setStyleSheet(f"color:{color};" if color else "")


def deck_names(db_cards: Dict[str, dict]) -> Callable[[str], str]:
    return lambda c: db_cards.get(c, {}).get("name", c.replace("_", " ").title())


def deck_costs(db_cards: Dict[str, dict]) -> Callable[[str], Optional[int]]:
    return lambda c: db_cards.get(c, {}).get("elixir")


class CardButton(QPushButton):
    """A hand slot painted as an original card face (art.draw_card_face): card colour, type glyph, elixir drop,
    name strip. It stays a checkable QPushButton (keys, clicks, text for accessibility); only the look is custom."""

    _QSS = "QPushButton{min-height:%dpx; max-height:%dpx; padding:0; border:none; background:transparent;}"

    def __init__(self, height: int = 76, compact: bool = False, parent=None):
        super().__init__(parent)
        self.compact = compact
        self.face: Optional[tuple] = None
        self.setStyleSheet(self._QSS % (height, height))
        self.setCursor(Qt.PointingHandCursor)

    def set_face(self, card_id: str, name: str, cost: int, kind: str, *, afford: bool = True,
                 selected: bool = False, evo: bool = False, enabled: bool = True) -> None:
        face = (card_id, name, cost, kind, afford, selected, evo, enabled)
        if face != self.face:                     # repaint only when something on the face changed
            self.face = face
            self.setText(name)
            self.update()

    def paintEvent(self, e) -> None:
        if self.face is None:
            return
        card_id, name, cost, kind, afford, selected, evo, enabled = self.face
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        art.draw_card_face(p, QRectF(self.rect()), card_id, name, cost, kind, afford=afford, selected=selected,
                           evo=evo, enabled=enabled, compact=self.compact)
        p.end()
