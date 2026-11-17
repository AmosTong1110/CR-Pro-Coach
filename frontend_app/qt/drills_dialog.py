"""
Tactical Drills Selection Dialog.
Presents curated defense micro-challenges and tactical positioning puzzles.
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog, QFrame, QHBoxLayout, QLabel, QPushButton, QScrollArea,
    QVBoxLayout, QWidget,
)

from backend_engine.drills.curated_drills import get_curated_drills
from backend_engine.drills.drill_spec import DrillSpec
from backend_engine.meta.archetypes import pretty
from frontend_app.qt.theme import C, cls, dark_palette
from frontend_app.qt.widgets import card_frame, label

_cls = cls


class DrillsDialog(QDialog):
    """Catalog dialog of curated tactical defense drills & micro-challenges."""

    drillSelected = Signal(object)  # DrillSpec

    def __init__(self, parent=None, on_start: Optional[Callable[[DrillSpec], None]] = None):
        super().__init__(parent)
        self.on_start = on_start
        self.setWindowTitle("Tactical Defense Drills")
        self.setPalette(dark_palette())
        self.resize(760, 680)

        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        # Header
        head = QVBoxLayout()
        head.setSpacing(4)
        t = label("🎯 Tactical Defense Drills", "h1")
        sub = label(
            "Sharpen your micro-placement, pull & kiting, and tower defense in isolated scenarios. "
            "Get immediate feedback and 3-star ratings on your execution.",
            "sub",
            wrap=True,
        )
        head.addWidget(t)
        head.addWidget(sub)
        root.addLayout(head)

        # Scrollable list of drills
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        inner = QWidget()
        lay = QVBoxLayout(inner)
        lay.setContentsMargins(0, 0, 4, 0)
        lay.setSpacing(12)

        drills = get_curated_drills()
        for d in drills:
            lay.addWidget(self._create_drill_card(d))

        lay.addStretch(1)
        scroll.setWidget(inner)
        root.addWidget(scroll, 1)

        # Bottom bar
        bot = QHBoxLayout()
        close_btn = _cls(QPushButton("Close"), "ghost")
        close_btn.clicked.connect(self.reject)
        bot.addStretch(1)
        bot.addWidget(close_btn)
        root.addLayout(bot)

    def _create_drill_card(self, drill: DrillSpec) -> QWidget:
        card = card_frame(spacing=8, margins=14)
        cl = card.layout()

        # Top row: Title + Difficulty + Category
        top = QHBoxLayout()
        t = QLabel(drill.title)
        t.setStyleSheet(f"font-size:16px; font-weight:800; color:{C['text']};")
        top.addWidget(t)
        top.addSpacing(8)

        # Difficulty badge
        diff_names = {1: ("★☆☆ Beginner", "#5ad15a"), 2: ("★★☆ Intermediate", "#f39c12"), 3: ("★★★ Expert", "#e74c3c")}
        d_name, d_col = diff_names.get(drill.difficulty, ("Unknown", "#888"))
        d_badge = QLabel(f" {d_name} ")
        d_badge.setStyleSheet(f"color:{d_col}; background:{C['panel2']}; border:1px solid {d_col}; border-radius:4px; font-size:11px; font-weight:bold;")
        top.addWidget(d_badge)

        # Category badge
        cat_badge = QLabel(f" {drill.category.replace('_', ' ').title()} ")
        cat_badge.setStyleSheet(f"color:{C['teal']}; background:{C['panel2']}; border:1px solid {C['teal']}; border-radius:4px; font-size:11px;")
        top.addWidget(cat_badge)
        top.addStretch(1)

        # Start button
        start_btn = _cls(QPushButton("▶ Start Drill"), "primary")
        start_btn.setMinimumWidth(110)
        start_btn.clicked.connect(lambda _=False, d=drill: self._start_drill(d))
        top.addWidget(start_btn)
        cl.addLayout(top)

        # Description
        desc = label(drill.description, wrap=True)
        desc.setStyleSheet(f"color:{C['muted']}; font-size:12px; line-height:1.4;")
        cl.addWidget(desc)

        # Hand / Cards preview & Advice
        bot_row = QHBoxLayout()
        cards_str = " · ".join(pretty(c) for c in drill.player_hand)
        cards_lbl = QLabel(f"<span style='color:{C['muted']}'>Hand:</span> <span style='color:{C['text']}'>{cards_str}</span> "
                           f"<span style='color:{C['elixir']}; font-weight:bold;'>({drill.player_elixir:.0f} elixir)</span>")
        cards_lbl.setTextFormat(Qt.RichText)
        cards_lbl.setStyleSheet("font-size:11px;")
        bot_row.addWidget(cards_lbl, 1)

        if drill.custom_advice:
            tip = QLabel(f"<span style='color:{C['gold']}'>💡 Tip:</span> <span style='color:{C['muted']}'>{drill.custom_advice}</span>")
            tip.setTextFormat(Qt.RichText)
            tip.setStyleSheet("font-size:11px;")
            cl.addWidget(tip)

        cl.addLayout(bot_row)
        return card

    def _start_drill(self, drill: DrillSpec) -> None:
        self.drillSelected.emit(drill)
        if self.on_start:
            self.on_start(drill)
        self.accept()
