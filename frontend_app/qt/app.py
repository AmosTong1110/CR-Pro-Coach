"""CRbot Fight page: the 1v1 practice arena (PySide6).

    python -m frontend_app.qt.app          (opens the CRbot shell with this page as "Fight")

The page has four states, and each shows only what you need:

* setup   pick decks (Deck Builder: 30 ready-made decks), AI persona, level, elixir counter mode → Start
* live    the board, your hand, the clock, the elixir counter and the play log. No buttons.
* ended   the result, Review (R) and New match (N)
* review  coach column on the right; win-rate graph and a timeline slider under the board.
          Drag the slider (or click the graph) to jump anywhere, Take over to play on from there,
          Exit review to leave, Stop (or Esc) to cancel a running analysis.

Keys: 1-4 select a card, click the board to play it, right click / Esc cancel, Space start, N new match,
E elixir quiz guess, R review (after the match), U / F back / forward 10 s, double-click a play-log line to take
over from just before that play.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

from PySide6.QtCore import QEvent, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFrame, QHBoxLayout,
    QInputDialog, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QPushButton, QScrollArea, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from backend_engine.contracts import LatencyClass
from backend_engine.meta.archetypes import archetype_name, fmt_avg
from backend_engine.meta.deck_link import format_copy_deck_link, parse_deck_link
from backend_engine.meta.presets import AI_PUSH_CARDS, find_preset, playable
from backend_engine.simulator.geometry import TICK_MS
from frontend_app.qt.board import BoardWidget
from frontend_app.qt.review import ReviewPanel, mmss
from frontend_app.qt.session import AI, HUMAN, PERSONAS, persona_label, GameSession, LogEntry
from frontend_app.qt.theme import C, cls
from frontend_app.qt.widgets import DeckView, card_frame, deck_costs, deck_names, label

# Why a play was refused (kept in English; tests match on these)
REASONS = {
    "not_enough_elixir": "Not enough elixir",
    "enemy_half": "Can't deploy on enemy side",
    "river": "Can't deploy in the river",
    "blocked_by_structure": "Blocked by a building",
    "out_of_bounds": "Out of bounds",
    "not_in_hand": "Card not in hand",
    "match_over": "Match is over",
    "ability_not_ready": "That ability isn't ready (champion not on the field, still deploying, or already used)",
    "no_ability": "That card has no ability",
}
LATENCY_NAMES = {
    LatencyClass.SURPRISE: "Surprise", LatencyClass.ANTICIPATED: "Anticipated",
    LatencyClass.ROUTINE: "Routine", LatencyClass.HOVER_REFLEX: "Hover reflex",
}
# Simulation speeds for the Account page default (the Fight page has no speed control).
SPEEDS = [("1x", 1), ("2x", 2), ("4x", 4), ("8x", 8)]
# Real-time 3-2-1 countdown after a take over / when you come back to a running match.
COUNTDOWN_MS = 3000
SETUP, LIVE, ENDED, REVIEW = "setup", "live", "ended", "review"
RESULT_REASON = {"king_destroyed": "3 crowns", "crowns": "crown lead", "tiebreaker": "lowest tower HP", "draw": ""}


class AnalysisCancelled(Exception):
    """Raised from the analysis progress callback when you press Stop."""


def deck_label(cards, evo) -> str:
    p = find_preset(cards, evo)
    return p.name if p else archetype_name(cards, evo)


def avg_cost(cards, cost_of) -> str:
    vals = [cost_of(c) or 0 for c in cards]
    return fmt_avg(sum(vals) / len(vals)) if vals else "—"


# ---------------------------------------------------------------------------
class ElixirBar(QWidget):
    """Ten elixir segments: full ones bright, the charging one dark violet, current value on the left."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0.0
        self.setMinimumHeight(24)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_value(self, v: float) -> None:
        self.value = v
        self.update()

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        label_w = 34
        seg = (w - label_w - 4) / 10
        for i in range(10):
            r = QRectF(label_w + i * seg + 1, 3, seg - 2, h - 6)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#3a1f45"))
            p.drawRoundedRect(r, 3, 3)
            fill = max(0.0, min(1.0, self.value - i))
            if fill > 0:
                p.setBrush(QColor("#d43cf0") if fill >= 1 else QColor("#8e3aa6"))
                p.drawRoundedRect(QRectF(r.x(), r.y(), r.width() * fill, r.height()), 3, 3)
        f = QFont()
        f.setPixelSize(17)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor("#e8a7ff"))
        p.drawText(QRectF(0, 0, label_w, h), Qt.AlignCenter, str(int(self.value + 1e-6)))
        p.end()


class HandBar(QWidget):
    """Next card + the 4 hand slots (keys 1-4) + the elixir bar."""

    def __init__(self, on_select, parent=None, on_ability=None):
        super().__init__(parent)
        self.on_select = on_select
        self.on_ability = on_ability or (lambda: None)
        self.group = QButtonGroup(self)
        self.group.setExclusive(False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(6)
        row = QHBoxLayout()
        row.setSpacing(8)
        self.next_label = QLabel()
        self.next_label.setAlignment(Qt.AlignCenter)
        self.next_label.setFixedWidth(68)
        self.next_label.setWordWrap(True)
        self.next_label.setStyleSheet(f"color:{C['muted']}; font-size:11px;")
        row.addWidget(self.next_label)
        self.buttons: List[QPushButton] = []
        self._styles: Dict[int, str] = {}
        for i in range(4):
            b = QPushButton()
            b.setCheckable(True)
            b.setMinimumHeight(56)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.setToolTip(f"Key {i + 1}")
            b.clicked.connect(lambda _=False, i=i: self.on_select(i))
            self.group.addButton(b, i)
            self.buttons.append(b)
            row.addWidget(b)
        lay.addLayout(row)
        self.elixir = ElixirBar()
        self.ability_btn = QPushButton()                  # only visible while one of your champions can use its ability
        self.ability_btn.setCursor(Qt.PointingHandCursor)
        self.ability_btn.setToolTip("Use your champion's ability (A): once per champion")
        self.ability_btn.setStyleSheet(f"QPushButton{{font-size:12px; font-weight:700; border-radius:8px; padding:4px 10px;"
                                       f"background:{C['violet']}; color:white; border:none;}}")
        self.ability_btn.clicked.connect(lambda: self.on_ability())
        self.ability_btn.hide()
        bottom = QHBoxLayout()
        bottom.setSpacing(8)
        bottom.addWidget(self.elixir, 1)
        bottom.addWidget(self.ability_btn)
        lay.addLayout(bottom)

    def refresh(self, hand: List[str], next_card: str, names, costs, elixir: float,
                selected: Optional[str], evo_ready: set, enabled: bool = True,
                abilities: Sequence[Tuple[str, int]] = ()) -> None:
        self.ability_btn.setVisible(enabled and bool(abilities))
        if abilities:
            card, cost = abilities[0]
            self.ability_btn.setText(f"⚡ {names[card]} ability · {cost}  (A)")
        for i, (b, c) in enumerate(zip(self.buttons, hand)):
            evo = c in evo_ready
            b.setText(f"{names[c]}{'  ★ Evo' if evo else ''}\n{costs[c]} elixir")
            b.setChecked(c == selected)
            b.setEnabled(enabled)
            afford = enabled and costs[c] <= elixir + 1e-9
            border = C["gold"] if c == selected else (C["violet"] if evo else ("#3a5277" if afford else C["line"]))
            style = (f"QPushButton{{font-size:13px; font-weight:600; border-radius:8px; padding:4px;"
                     f"background:{'#22324a' if afford else C['panel']}; color:{C['text'] if afford else C['muted']};"
                     f"border:2px solid {border};}}")
            if self._styles.get(i) != style:          # restyling every tick is slow; only on change
                self._styles[i] = style
                b.setStyleSheet(style)
        self.next_label.setText(f"NEXT\n{names[next_card]}")
        self.elixir.set_value(elixir)


# ---------------------------------------------------------------------------
class DeckDialog(QDialog):
    """Deck Builder: pick one of the ready-made 8-card decks for you and for the AI (or build your own)."""

    def __init__(self, session: GameSession, parent=None, side: int = HUMAN):
        super().__init__(parent)
        self.setWindowTitle("Deck Builder")
        self.session = session
        db = session.db
        self.names, self.costs = deck_names(db.cards), deck_costs(db.cards)
        self.presets = playable(db.cards)
        cfg = session.cfg
        self.choice = {HUMAN: (tuple(cfg.player_deck), tuple(cfg.player_evo)),
                       AI: (tuple(cfg.ai_deck), tuple(cfg.ai_evo))}
        self.side = side

        seg = QHBoxLayout()
        grp = QButtonGroup(self)
        grp.setExclusive(True)
        self.side_btns = {}
        for s, text in ((HUMAN, "Your deck"), (AI, "AI deck")):
            b = cls(QPushButton(text), "seg")
            b.setCheckable(True)
            b.clicked.connect(lambda _=False, s=s: self.set_side(s))
            grp.addButton(b)
            seg.addWidget(b)
            self.side_btns[s] = b
        seg.addSpacing(12)
        self.side_summary = label("", "sub", rich=True)
        seg.addWidget(self.side_summary, 1)

        self.list = QListWidget()
        self.list.setMinimumWidth(270)
        self.list.setStyleSheet("QListWidget::item{padding:7px 8px;}")
        for p in self.presets:
            self.list.addItem(QListWidgetItem(f"{p.name}\n{avg_cost(p.cards, self.costs)} avg · {p.style}"))
        self.list.currentRowChanged.connect(self._picked)

        prev = card_frame(spacing=8, margins=16)
        self.p_name = label("", "h1", wrap=True)
        self.p_meta = label("", "sub", wrap=True)
        self.p_tip = label("", wrap=True)
        self.p_view = DeckView()
        self.p_note = label("", wrap=True)
        self.p_note.setStyleSheet(f"color:{C['warn']}; font-size:12px;")
        for w in (self.p_name, self.p_meta, self.p_view, self.p_tip, self.p_note):
            prev.layout().addWidget(w)
        prev.layout().addStretch(1)

        body = QHBoxLayout()
        body.setSpacing(12)
        body.addWidget(self.list, 2)
        body.addWidget(prev, 3)

        bottom = QHBoxLayout()
        custom = cls(QPushButton("Build your own…"), "ghost")
        custom.setToolTip("Tick 8 cards yourself for the selected side")
        custom.clicked.connect(self._custom)
        import_btn = cls(QPushButton("Import link…"), "ghost")
        import_btn.setToolTip("Import deck from official clashroyale:// link, RoyaleAPI, or text")
        import_btn.clicked.connect(self._import_link)
        copy_btn = cls(QPushButton("Copy link"), "ghost")
        copy_btn.setToolTip("Copy official clashroyale:// deck link for the selected side")
        copy_btn.clicked.connect(self._copy_link)
        cancel = cls(QPushButton("Cancel"), "ghost")
        cancel.clicked.connect(self.reject)
        ok = cls(QPushButton("Use these decks"), "primary")
        ok.setDefault(True)
        ok.clicked.connect(self.accept)
        bottom.addWidget(custom)
        bottom.addWidget(import_btn)
        bottom.addWidget(copy_btn)
        bottom.addStretch(1)
        bottom.addWidget(cancel)
        bottom.addWidget(ok)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(18, 16, 18, 16)
        lay.setSpacing(12)
        title = label(f"Choose a deck · {len(self.presets)} ready-made decks", "h2")
        lay.addWidget(title)
        lay.addLayout(seg)
        lay.addLayout(body, 1)
        lay.addLayout(bottom)
        self.resize(860, 560)
        self.set_side(side)

    def _name(self, side: int) -> str:
        cards, evo = self.choice[side]
        p = find_preset(cards, evo, self.presets)
        return p.name if p else "Custom deck"

    def set_side(self, side: int) -> None:
        self.side = side
        self.side_btns[side].setChecked(True)
        cards, evo = self.choice[side]
        p = find_preset(cards, evo, self.presets)
        self.list.blockSignals(True)
        self.list.setCurrentRow(self.presets.index(p) if p else -1)
        if p is None:
            self.list.clearSelection()
        self.list.blockSignals(False)
        self._preview()

    def _picked(self, row: int) -> None:
        if 0 <= row < len(self.presets):
            p = self.presets[row]
            self.choice[self.side] = (p.cards, p.evo)
            self._preview()

    def _preview(self) -> None:
        cards, evo = self.choice[self.side]
        p = find_preset(cards, evo, self.presets)
        self.p_name.setText(p.name if p else "Custom deck")
        style = p.style if p else archetype_name(cards, evo)
        self.p_meta.setText(f"{'Your deck' if self.side == HUMAN else 'AI deck'}  ·  "
                            f"{avg_cost(cards, self.costs)} average elixir  ·  {style}")
        self.p_tip.setText(p.tip if p else "Your own selection of 8 cards.")
        self.p_view.set_deck(cards, evo, self.names, self.costs)
        weak_ai = self.side == AI and not any(c in AI_PUSH_CARDS for c in cards)
        self.p_note.setText("Note: today's AI only attacks with Hog Rider, so it will mostly defend with this "
                            "deck. Pick a Hog deck for a tougher opponent." if weak_ai else "")
        self.p_note.setVisible(weak_ai)
        self.side_summary.setText(f"<span style='color:{C['you']}'>You</span>: {self._name(HUMAN)}"
                                  f"&nbsp;&nbsp;·&nbsp;&nbsp;<span style='color:{C['ai']}'>AI</span>: {self._name(AI)}")

    def _import_link(self) -> None:
        clip = QApplication.clipboard().text().strip()
        default_val = clip if ("clashroyale" in clip or "royaleapi" in clip or "," in clip or ";" in clip) else ""
        text, ok = QInputDialog.getText(
            self,
            "Import Deck Link",
            f"Paste deck link or card list for {'Your deck' if self.side == HUMAN else 'AI deck'}:\n"
            f"(Supports clashroyale://copyDeck, link.clashroyale.com, RoyaleAPI, or card names)",
            text=default_val,
        )
        if not ok or not text.strip():
            return
        parsed = parse_deck_link(text.strip(), self.session.db)
        if not parsed.playable:
            msg = parsed.warning or "Deck is incomplete or contains unsimulated cards."
            if parsed.missing:
                msg += f"\n\nMissing / unsimulated: {', '.join(parsed.missing)}"
            QMessageBox.warning(self, "Cannot Import Deck", msg)
            return

        self.choice[self.side] = (parsed.cards, parsed.evo)
        self.set_side(self.side)
        if parsed.warning:
            QMessageBox.information(self, "Deck Imported", f"Imported {parsed.archetype or 'deck'}.\n\nNote: {parsed.warning}")

    def _copy_link(self) -> None:
        cards, evo = self.choice[self.side]
        try:
            link = format_copy_deck_link(cards, evo, db=self.session.db)
        except Exception as e:
            QMessageBox.warning(self, "Copy Deck Link", f"Could not generate deck link: {e}")
            return
        QApplication.clipboard().setText(link)
        side_name = "Your deck" if self.side == HUMAN else "AI deck"
        QMessageBox.information(
            self,
            "Deck Link Copied",
            f"Official Clash Royale link for {side_name} copied to clipboard:\n\n{link}",
        )

    def _custom(self) -> None:
        cards, evo = self.choice[self.side]
        dlg = CustomDeckDialog(self.session.db, cards, evo, "Your deck" if self.side == HUMAN else "AI deck", self)
        if dlg.exec():
            self.choice[self.side] = dlg.result_deck()
            self.set_side(self.side)

    def result_config(self) -> dict:
        (pd, pe), (ad, ae) = self.choice[HUMAN], self.choice[AI]
        return dict(player_deck=tuple(pd), ai_deck=tuple(ad), player_evo=tuple(pe), ai_evo=tuple(ae))


class CustomDeckDialog(QDialog):
    """Build your own deck for one side: tick exactly 8 cards, optionally one evolution or hero."""

    def __init__(self, db, deck, evo, title: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Build your own · {title}")
        self.db = db
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 14, 16, 14)
        self.count = label("", "sub")
        lay.addWidget(label(title, "h2"))
        lay.addWidget(self.count)
        self.lw = QListWidget()
        for cid, c in sorted(db.cards.items(), key=lambda kv: (kv[1]["elixir"], kv[1]["name"])):
            it = QListWidgetItem(f"{c['name']}   ({c['elixir']})")
            it.setData(Qt.UserRole, cid)
            it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
            it.setCheckState(Qt.Checked if cid in deck else Qt.Unchecked)
            self.lw.addItem(it)
        self.lw.itemChanged.connect(lambda _: self._count())
        lay.addWidget(self.lw, 1)
        self.evo_box = QComboBox()
        self.evo_box.addItem("No evolution / hero", None)
        for cid, c in db.cards.items():
            for var, tag, slot in (("evo", "Evo", cid), ("hero", "Hero", f"{cid}#hero")):
                if db.variant_simulated(cid, var):
                    self.evo_box.addItem(f"{tag} {c['name']}", slot)
                    if slot in evo:
                        self.evo_box.setCurrentIndex(self.evo_box.count() - 1)
        lay.addWidget(self.evo_box)

        btn_bar = QHBoxLayout()
        import_btn = cls(QPushButton("Import link…"), "ghost")
        import_btn.setToolTip("Import deck from official clashroyale:// link, RoyaleAPI, or text")
        import_btn.clicked.connect(self._import_link)
        copy_btn = cls(QPushButton("Copy link"), "ghost")
        copy_btn.setToolTip("Copy official clashroyale:// deck link of selected cards")
        copy_btn.clicked.connect(self._copy_link)
        btn_bar.addWidget(import_btn)
        btn_bar.addWidget(copy_btn)
        btn_bar.addStretch(1)

        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        btn_bar.addWidget(bb)
        lay.addLayout(btn_bar)

        self.resize(380, 580)
        self._count()

    def _import_link(self) -> None:
        clip = QApplication.clipboard().text().strip()
        default_val = clip if ("clashroyale" in clip or "royaleapi" in clip or "," in clip or ";" in clip) else ""
        text, ok = QInputDialog.getText(
            self,
            "Import Deck Link",
            "Paste deck link or card list (clashroyale://, link.clashroyale.com, RoyaleAPI, or names):",
            text=default_val,
        )
        if not ok or not text.strip():
            return
        parsed = parse_deck_link(text.strip(), self.db)
        if parsed.missing:
            QMessageBox.warning(
                self,
                "Import Warning",
                f"Some cards could not be resolved or are not simulated:\n{', '.join(parsed.missing)}",
            )
        if not parsed.cards:
            return

        self.lw.blockSignals(True)
        for j in range(self.lw.count()):
            it = self.lw.item(j)
            cid = it.data(Qt.UserRole)
            it.setCheckState(Qt.Checked if cid in parsed.cards else Qt.Unchecked)
        self.lw.blockSignals(False)

        if parsed.evo:
            target_evo = parsed.evo[0]
            for idx in range(self.evo_box.count()):
                d = self.evo_box.itemData(idx)
                if d and (d == target_evo or d.split("#")[0] == target_evo):
                    self.evo_box.setCurrentIndex(idx)
                    break
        else:
            self.evo_box.setCurrentIndex(0)
        self._count()

    def _copy_link(self) -> None:
        deck = self.chosen()
        if len(deck) != 8:
            QMessageBox.warning(self, "Copy Deck Link", f"A deck needs exactly 8 cards to generate a link (you have {len(deck)}).")
            return
        evo = self.evo_box.currentData()
        evo_tuple = (evo,) if evo else ()
        try:
            link = format_copy_deck_link(deck, evo_tuple, db=self.db)
        except Exception as e:
            QMessageBox.warning(self, "Copy Deck Link", f"Could not generate deck link: {e}")
            return
        QApplication.clipboard().setText(link)
        QMessageBox.information(self, "Deck Link Copied", f"Official deck link copied to clipboard:\n\n{link}")

    def chosen(self):
        return tuple(self.lw.item(j).data(Qt.UserRole) for j in range(self.lw.count())
                     if self.lw.item(j).checkState() == Qt.Checked)

    def _count(self) -> None:
        n = len(self.chosen())
        self.count.setText(f"{n} of 8 cards selected")

    def _accept(self) -> None:
        deck = self.chosen()
        if len(deck) != 8:
            QMessageBox.warning(self, "Deck", f"A deck needs exactly 8 cards (you have {len(deck)}).")
            return
        evo = self.evo_box.currentData()
        if evo and evo.split("#")[0] not in deck:
            QMessageBox.warning(self, "Deck", "The evolution / hero card must be in the deck.")
            return
        self.accept()

    def result_deck(self):
        evo = self.evo_box.currentData()
        return self.chosen(), ((evo,) if evo else ())


# ---------------------------------------------------------------------------
class CounterPanel(QFrame):
    """Opponent elixir + cycle tracker. Quiz mode hides the number; press E to guess it."""

    def __init__(self, session: GameSession, on_guess, parent=None):
        super().__init__(parent)
        cls(self, "card")
        self.setAttribute(Qt.WA_StyledBackground, True)
        self.session = session
        self.quiz = False
        lay = QVBoxLayout(self)
        lay.setContentsMargins(12, 10, 12, 10)
        lay.setSpacing(4)
        top = QHBoxLayout()
        top.addWidget(label("ELIXIR COUNTER", "section"))
        top.addStretch(1)
        self.guess_btn = cls(QPushButton("Guess (E)"), "ghost")
        self.guess_btn.clicked.connect(lambda: on_guess())
        self.guess_btn.setVisible(False)
        top.addWidget(self.guess_btn)
        lay.addLayout(top)
        self.value = QLabel()
        self.value.setStyleSheet(f"font-size:22px; font-weight:800; color:{C['elixir']};")
        self.adv = QLabel()
        row = QHBoxLayout()
        row.addWidget(self.value)
        row.addStretch(1)
        row.addWidget(self.adv)
        lay.addLayout(row)
        self.cycle = label("", wrap=True)
        self.cycle.setStyleSheet("font-size:12px;")
        lay.addWidget(self.cycle)
        self.stats = label("", "muted", wrap=True)
        lay.addWidget(self.stats)

    @property
    def quiz_mode(self) -> bool:
        return self.quiz

    def set_quiz(self, on: bool) -> None:
        self.quiz = on
        self.guess_btn.setVisible(on)
        self.refresh()

    def refresh(self) -> None:
        s = self.session
        names = {cid: c["name"] for cid, c in s.db.cards.items()}
        n, mae, within = s.quiz_stats()
        self.stats.setText(f"Quiz: {n} guesses · average error {mae:.2f} · within 0.5: {within:.0%}" if n
                           else ("Quiz: no guesses yet" if self.quiz else ""))
        self.stats.setVisible(bool(self.stats.text()))
        if self.quiz:
            self.value.setText("AI ≈ ???")
            self.adv.setText("")
            self.cycle.setText("Track it yourself: what did the AI play, and which card comes back next?")
            return
        e = s.opp_estimate()
        self.value.setText(f"AI ≈ {e.point:.1f}")
        adv = s.elixir_advantage()
        if abs(adv) < 0.5:
            self.adv.setText("Even")
            self.adv.setStyleSheet(f"font-size:13px; font-weight:700; color:{C['draw']};")
        else:
            self.adv.setText(f"{'Ahead' if adv > 0 else 'Behind'} {adv:+.1f}")
            self.adv.setStyleSheet(f"font-size:13px; font-weight:700; color:{C['win'] if adv > 0 else C['loss']};")
        q = [names[c] if c else "?" for c in e.queue]
        hand = [names[c] for c in e.known_in_hand] + ["?"] * e.unknown_hand_slots
        lines = [f"In hand: {' · '.join(hand)}", f"Comes back next: {' → '.join(q)}"]
        evo = [f"{names[c]} {k}/{s.db.evo_cycles(c)}" for c, k in e.evo_progress.items()]
        if evo:
            lines.append("Evo cycles: " + ", ".join(evo))
        if e.leaked > 0.05:
            lines.append(f"AI leaked {e.leaked:.1f} elixir")
        g = s.opp_deck_guess()
        if g is not None and g.card_probs:
            pretty = lambda c: names.get(c, c.replace("_", " ").title())  # noqa: E731
            lines.append("Likely cards: " + " · ".join(f"{pretty(c)} {p:.0%}" for c, p in g.top(4)))
            if g.likely_decks and g.likely_decks[0][1] >= 0.3 and len(g.revealed) >= 2 and g.matching_weight >= 5:
                deck, p = g.likely_decks[0]
                lines.append(f"Most likely deck ({p:.0%}): " + ", ".join(pretty(c) for c in deck))
        self.cycle.setText("\n".join(lines))


# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    """The Fight page (also runs standalone). The CRbot shell listens to the signals to save matches."""
    matchEnded = Signal()        # the match reached a result
    reviewReady = Signal()       # a coach review finished
    aboutToReset = Signal()      # the current match is about to be replaced (new match, new settings, reopen)
    busyChanged = Signal(bool)   # coach analysis running
    modeChanged = Signal(str)    # setup / live / ended / review (the shell hides its sidebar while live)

    def __init__(self, session: Optional[GameSession] = None):
        super().__init__()
        self.setWindowTitle("CRbot — Fight")
        self.session = session or GameSession()
        self.auto_review = True
        self.mode = SETUP
        self.running = False
        self.speed = 1                 # set from the Account page default; no control on this page
        self.selected: Optional[str] = None
        self.countdown_ms = 0
        self.analyzing = False
        self._cancel = False
        self.away_paused = False
        self.playing = False                    # review playback (watch the match like a video)
        self.showing_me = False                 # "Show me" animated rollout of recommended line
        self._show_me_origin_session: Optional[GameSession] = None
        self._show_me_start_ms = 0
        self._show_me_end_ms = 0
        self._show_me_move_index: Optional[int] = None
        self._show_me_move: Optional[MoveReview] = None
        self.play_speed = 1
        self._play_actions = None
        self._banner_until: Optional[int] = None

        # -- left: board + (hand | review graph and timeline) -------------------------------------------------
        self.board = BoardWidget(self.session)
        self.board.tileClicked.connect(self.on_tile)
        self.board.cancelled.connect(lambda: self.select(None))
        self.hand = HandBar(self.on_hand_slot, on_ability=self.use_ability)

        names = lambda c: self.session.db.cards.get(c, {}).get("name", c)  # noqa: E731
        self.review_panel = ReviewPanel(names, self.run_review, self.take_over_review, self.exit_review,
                                        self.cancel_analysis, on_show_me=self.toggle_show_me)
        self.review_panel.moveSelected.connect(self.show_review_move)
        graph = self.review_panel.graph
        graph.setMinimumHeight(110)
        graph.setMaximumHeight(170)
        graph.timeClicked.connect(self.scrub_to)

        self.rew_btn = cls(QPushButton("⏪ 10s"), "ghost")
        self.rew_btn.setToolTip("Back 10 seconds (U)")
        self.rew_btn.clicked.connect(lambda: self.step(-10))
        self.fwd_btn = cls(QPushButton("10s ⏩"), "ghost")
        self.fwd_btn.setToolTip("Forward 10 seconds (F)")
        self.fwd_btn.clicked.connect(lambda: self.step(10))
        self.play_btn = cls(QPushButton("▶  Play"), "primary")
        self.play_btn.setToolTip("Watch the match from this moment (Space). Press again to stop.")
        self.play_btn.setMinimumWidth(92)
        self.play_btn.clicked.connect(self.toggle_playback)
        self.speed_btn = cls(QPushButton("1x"), "ghost")
        self.speed_btn.setToolTip("Playback speed: 1x · 2x · 4x")
        self.speed_btn.setMaximumWidth(56)
        self.speed_btn.clicked.connect(self._cycle_play_speed)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setSingleStep(1000)
        self.slider.setPageStep(10_000)
        self.slider.setToolTip("Drag to jump to any moment of the match")
        self.slider.valueChanged.connect(self._slider_moved)
        self.slider_label = label("0:00", "sub")
        self.slider_label.setMinimumWidth(82)
        self.slider_label.setAlignment(Qt.AlignCenter)
        tl = QHBoxLayout()
        tl.setSpacing(8)
        for w in (self.rew_btn, self.play_btn, self.slider, self.slider_label, self.speed_btn, self.fwd_btn):
            tl.addWidget(w, 1 if w is self.slider else 0)
        self.review_strip = QWidget()
        vs = QVBoxLayout(self.review_strip)
        vs.setContentsMargins(0, 0, 0, 0)
        vs.setSpacing(6)
        vs.addWidget(graph)
        vs.addLayout(tl)

        left = QVBoxLayout()
        left.setSpacing(10)
        left.addWidget(self.board, 1)
        left.addWidget(self.hand)
        left.addWidget(self.review_strip)

        # -- right column: header + one panel per state + play log + status ----------------------------------
        right = QVBoxLayout()
        right.setContentsMargins(0, 0, 0, 0)
        right.setSpacing(10)
        self.header = self._build_header()
        right.addWidget(self.header)
        self.setup_box = self._build_setup()
        right.addWidget(self.setup_box, 1)
        self.counter = CounterPanel(self.session, self.quiz_guess)
        right.addWidget(self.counter)
        self.ended_box = self._build_ended()
        right.addWidget(self.ended_box)
        right.addWidget(self.review_panel, 1)
        self.log_box = card_frame(spacing=6, margins=10)
        self.log_box.layout().addWidget(label("PLAY LOG · double-click a line to take over from before it",
                                              "section", wrap=True))
        self.log_list = QListWidget()
        self.log_list.setStyleSheet("QListWidget{border:none; background:transparent; font-size:12px;}")
        self.log_list.itemDoubleClicked.connect(self.take_over)
        self.log_box.layout().addWidget(self.log_list, 1)
        right.addWidget(self.log_box, 1)
        self.status = label("", "status", wrap=True)
        right.addWidget(self.status)
        self.stop_btn = cls(QPushButton("■  Stop match"), "stop")
        self.stop_btn.setToolTip("End the match right now (you can still review it)")
        self.stop_btn.clicked.connect(self.stop_match)
        self.stop_btn.setVisible(False)
        right.addWidget(self.stop_btn)
        for w in (self.header, self.counter, self.ended_box):
            w.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        rw = QWidget()
        rw.setLayout(right)
        rw.setFixedWidth(360)

        root = QWidget()
        root.setObjectName("FightRoot")
        root.setAttribute(Qt.WA_StyledBackground, True)
        h = QHBoxLayout(root)
        h.setContentsMargins(16, 14, 16, 14)
        h.setSpacing(16)
        h.addLayout(left, 1)
        h.addWidget(rw)
        self.setCentralWidget(root)
        self.resize(1000, 860)

        # -- keys (ignored while the coach is analyzing) ------------------------------------------------------
        def key(seq, fn) -> None:
            QShortcut(QKeySequence(seq), self, activated=lambda: None if self.analyzing else fn())
        for i in range(4):
            key(str(i + 1), lambda i=i: self.on_hand_slot(i))
        def _on_space():
            if self.mode != REVIEW:
                self.start_match()
            elif self.showing_me:
                self.stop_show_me()
            else:
                self.toggle_playback()
        key(Qt.Key_Space, _on_space)
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=self._escape)
        key("A", self.use_ability)
        key("N", self.new_match)
        key("E", self.quiz_guess)
        key("U", lambda: self.step(-10))
        key("F", lambda: self.step(10))
        key("R", self.run_review)
        key("S", self.toggle_show_me)

        self.timer = QTimer(self)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start()
        self._sync_setup()
        self.add_log([])
        self._enter_loaded_mode()
        if self.mode == SETUP:
            self.flash("Choose your settings, then press Start")

    # -- building blocks --------------------------------------------------------------------------------------
    def _build_header(self) -> QWidget:
        head = card_frame(spacing=2, margins=12)
        row = QHBoxLayout()
        col = QVBoxLayout()
        col.setSpacing(0)
        self.clock = label("3:00", "clock")
        self.phase = label(" ", "phase")
        col.addWidget(self.clock)
        col.addWidget(self.phase)
        row.addLayout(col)
        row.addStretch(1)
        side = QVBoxLayout()
        side.setSpacing(4)
        self.mode_tag = QLabel()
        self.mode_tag.setAlignment(Qt.AlignRight)
        self.crowns = label("", rich=True)
        self.crowns.setAlignment(Qt.AlignRight)
        self.crowns.setStyleSheet("font-size:15px; font-weight:700;")
        side.addWidget(self.mode_tag, 0, Qt.AlignRight)
        side.addWidget(self.crowns)
        row.addLayout(side)
        head.layout().addLayout(row)
        return head

    def _build_setup(self) -> QWidget:
        box = QWidget()
        v = QVBoxLayout(box)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(10)
        inner = QWidget()
        iv = QVBoxLayout(inner)
        iv.setContentsMargins(0, 0, 0, 0)
        iv.setSpacing(10)

        decks = card_frame(spacing=6)
        dl = decks.layout()
        dl.addWidget(label("DECKS", "section"))
        self.deck_title: Dict[int, QLabel] = {}
        self.deck_view: Dict[int, DeckView] = {}
        for side in (HUMAN, AI):
            t = label("", rich=True, wrap=True)
            dv = DeckView()
            dl.addWidget(t)
            dl.addWidget(dv)
            self.deck_title[side], self.deck_view[side] = t, dv
        self.deck_btn = cls(QPushButton("Deck Builder…"), "ghost")
        self.deck_btn.setToolTip("Choose from 30 ready-made decks for you and the AI")
        self.deck_btn.clicked.connect(lambda: self.edit_decks())
        dl.addWidget(self.deck_btn)
        iv.addWidget(decks)

        opts = card_frame(spacing=6)
        ol = opts.layout()
        ol.addWidget(label("AI PERSONA", "section"))
        self.persona_box = QComboBox()
        for k, p in PERSONAS.items():
            self.persona_box.addItem(persona_label(k), k)
        self.persona_box.setCurrentIndex(list(PERSONAS).index(self.session.cfg.persona))
        self.persona_box.currentIndexChanged.connect(lambda _: self.restart(persona=self.persona_box.currentData()))
        ol.addWidget(self.persona_box)
        ol.addSpacing(4)
        ol.addWidget(label("CARD LEVEL", "section"))
        lv = QHBoxLayout()
        lv.setSpacing(6)
        self.lv11, self.lv16 = cls(QPushButton("11 · Tournament"), "seg"), cls(QPushButton("16 · Max"), "seg")
        self.level_group = QButtonGroup(self)
        self.level_group.setExclusive(True)
        for b in (self.lv11, self.lv16):
            b.setCheckable(True)
            self.level_group.addButton(b)
        (self.lv16 if self.session.cfg.level == 16 else self.lv11).setChecked(True)
        self.lv11.toggled.connect(lambda on: on and self.restart(level=11))
        self.lv16.toggled.connect(lambda on: on and self.restart(level=16))
        lv.addWidget(self.lv11, 1)
        lv.addWidget(self.lv16, 1)
        ol.addLayout(lv)
        ol.addSpacing(4)
        ol.addWidget(label("ELIXIR COUNTER", "section"))
        self.counter_mode = QComboBox()
        self.counter_mode.addItem("Show the AI's elixir", "show")
        self.counter_mode.addItem("Quiz: hidden, press E to guess", "quiz")
        self.counter_mode.currentIndexChanged.connect(
            lambda _: self.counter.set_quiz(self.counter_mode.currentData() == "quiz"))
        ol.addWidget(self.counter_mode)
        self.vision = QCheckBox("Outline units the AI hasn't seen yet")
        self.vision.setChecked(True)
        self.vision.toggled.connect(self._toggle_vision)
        ol.addWidget(self.vision)
        iv.addWidget(opts)
        iv.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setWidget(inner)
        v.addWidget(scroll, 1)
        self.start_btn = cls(QPushButton("▶  Start match  (Space)"), "big")
        self.start_btn.clicked.connect(self.start_match)
        v.addWidget(self.start_btn)
        v.addWidget(label("AI: scripted baseline (a learned model replaces it later). "
                          "Card data: game files as of 2023-10.", "muted", wrap=True))
        return box

    def _build_ended(self) -> QWidget:
        box = card_frame(spacing=8, margins=16)
        lay = box.layout()
        self.result_title = QLabel()
        self.result_sub = label("", "sub", wrap=True)
        self.review_btn = cls(QPushButton("Review (R)"), "big")
        self.review_btn.setToolTip("Grade every play and replay the match with the coach")
        self.review_btn.clicked.connect(self.run_review)
        self.new_btn = cls(QPushButton("New match (N)"), "ghost")
        self.new_btn.clicked.connect(self.new_match)
        lay.addWidget(self.result_title)
        lay.addWidget(self.result_sub)
        lay.addSpacing(6)
        lay.addWidget(self.review_btn)
        lay.addWidget(self.new_btn)
        return box

    # -- states -----------------------------------------------------------------------------------------------
    def _set_mode(self, mode: str) -> None:
        if mode != REVIEW:
            self.stop_playback()
        self.mode = mode
        self.stop_btn.setVisible(mode == LIVE)
        self.setup_box.setVisible(mode == SETUP)
        self.counter.setVisible(mode == LIVE)
        self.ended_box.setVisible(mode == ENDED)
        self.review_panel.setVisible(mode == REVIEW)
        self.log_box.setVisible(mode in (LIVE, ENDED))
        self.hand.setVisible(mode in (SETUP, LIVE))
        self.review_strip.setVisible(mode == REVIEW)
        tag = {SETUP: ("SETUP", C["muted"]), LIVE: ("● LIVE", C["loss"]), ENDED: ("MATCH OVER", C["muted"]),
               REVIEW: ("REVIEW", C["teal"])}[mode]
        self.mode_tag.setText(tag[0])
        self.mode_tag.setStyleSheet(f"font-size:11px; font-weight:800; letter-spacing:1px; color:{tag[1]};")
        if mode == ENDED:
            self._show_result()
        if mode == LIVE:
            self.counter.refresh()
        self.refresh()
        QApplication.sendPostedEvents(None, QEvent.LayoutRequest)   # lay the new panels out now, not a frame later
        self.modeChanged.emit(mode)

    def _enter_loaded_mode(self) -> None:
        """Pick the state for the session on screen (fresh → setup, played → ended, reviewed → review)."""
        s = self.session
        if s.review is not None:
            self.review_panel.set_review(s.review, self._end_ms())
            self._set_mode(REVIEW)
        elif s.match.t_ms > 0 or s.result is not None:
            self._set_mode(ENDED)
        else:
            self._set_mode(SETUP)

    def _show_result(self) -> None:
        r = self.session.result
        m = self.session.match
        if r is None:
            text, col = "Match unfinished", C["draw"]
            sub = f"Stopped at {mmss(m.t_ms)} · {m.sides[HUMAN].crowns} – {m.sides[AI].crowns}"
        else:
            text, col = (("Draw", C["draw"]) if r.winner is None else
                         ("Victory", C["gold"]) if r.winner == HUMAN else ("Defeat", C["loss"]))
            why = RESULT_REASON.get(r.reason, "")
            sub = f"{r.crowns[0]} – {r.crowns[1]}" + (f" · {why}" if why else "") + f" · {mmss(m.t_ms)}"
        self.result_title.setText(text)
        self.result_title.setStyleSheet(f"font-size:30px; font-weight:800; color:{col};")
        has_plays = self.session.has_human_plays()
        self.result_sub.setText(sub + ("" if has_plays else "\nYou didn't play a card, so there is nothing to review."))
        self.review_btn.setEnabled(has_plays)

    def start_match(self) -> None:
        """Setup → live. There is no pause: the match runs until it ends (it only waits while you're on
        another page, and resumes with a 3-2-1 countdown when you come back)."""
        if self.mode != SETUP or self.analyzing:
            return
        self._set_mode(LIVE)
        self.running = True
        self.flash("")

    def stop_match(self) -> None:
        """Stop button: end the match instantly. It stays on the result screen (unfinished), saved, reviewable."""
        if self.mode != LIVE:
            return
        self.running = False
        self._stop_countdown()
        self.board.ghosts, self.board.banner = [], None
        self.select(None)
        self._set_mode(ENDED)
        self.flash("Match stopped")
        self.matchEnded.emit()

    def pause_for_away(self) -> None:
        """Called by the shell when you leave the Fight page: never let a match run unseen."""
        self.stop_playback()
        if self.mode == LIVE and (self.running or self.countdown_ms > 0):
            self.running = False
            self._stop_countdown()
            self.away_paused = True
            self.flash("Paused while you were away")

    def resume_after_away(self) -> None:
        """Called by the shell when you come back: resume with a 3-2-1 countdown."""
        if self.away_paused and self.mode == LIVE and self.session.result is None:
            self._start_countdown()
            self.flash("Welcome back · resuming in 3")
        self.away_paused = False

    # -- simulation loop --------------------------------------------------------------------------------------
    def on_timer(self) -> None:
        if self.showing_me:
            self._show_me_tick()
            return
        if self.playing:
            self._playback_tick()
            return
        if self.countdown_ms > 0:            # real-time 3-2-1; the match clock is frozen meanwhile
            self.countdown_ms -= TICK_MS
            if self.countdown_ms <= 0:
                self.board.overlay_text = None
                self.board.ghosts, self.board.banner = [], None
                self.running = True
                self.flash("")
            else:
                self.board.overlay_text = str(-(-self.countdown_ms // 1000))
            self.board.update()
            return
        if not self.running or self.mode != LIVE or self.session.result is not None:
            return
        self.board.ghosts, self.board.banner = [], None
        self.add_log(self.session.tick(self.speed))
        if self.session.result is not None:
            self._match_over()
            return
        self.board.refresh_legal()
        self.refresh()

    def _match_over(self) -> None:
        self.running = False
        self.select(None)
        self._set_mode(ENDED)
        self.matchEnded.emit()
        if self.auto_review and self.session.has_human_plays():
            self.run_review()

    # -- coach review -----------------------------------------------------------------------------------------
    def _end_ms(self) -> int:
        r = self.session.review if self.review_panel.review is None else self.review_panel.review
        last = r.curve[-1][0] if r is not None and r.curve else 0
        return max(self.session.max_known_ms(), last, 1000)

    def _goto_end(self) -> None:
        if self.session.can_forward():
            self.session.seek(self.session.max_known_ms())
            self._reload_log()

    def _set_busy(self, busy: bool) -> None:
        self.review_panel.set_busy(busy)
        if busy:
            self.stop_playback()
        for w in (self.board, self.rew_btn, self.fwd_btn, self.slider, self.review_panel.graph, self.play_btn,
                  self.speed_btn):
            w.setEnabled(not busy)
        self.busyChanged.emit(busy)

    def run_review(self) -> None:
        """Open the review (only after a match) and grade every play: each one is replayed against ~30-40
        alternatives for 8 s. Stop (or Esc) cancels; the review screen stays open either way."""
        if self.analyzing or self.mode in (SETUP, LIVE):
            return
        if not self.session.has_human_plays():
            self.flash("Play some cards first, then the coach can grade them")
            return
        self._stop_countdown()
        self.running = False
        self.select(None)
        self.board.ghosts, self.board.banner = [], None
        self.review_panel.clear_selection()
        self._set_mode(REVIEW)
        self._goto_end()
        self.refresh()
        self.analyzing, self._cancel = True, False
        self._set_busy(True)
        self.flash("Analyzing… every play is simulated 8 s ahead · Stop or Esc to cancel")
        panel = self.review_panel

        def progress(done: int, total: int) -> None:
            panel.progress.setMaximum(max(total, 1))
            panel.progress.setValue(done)
            panel.progress_label.setText(f"Analyzing… {100 * done // max(total, 1)}%")
            QApplication.processEvents()
            if self._cancel:
                raise AnalysisCancelled()

        review = None
        try:
            review = self.session.analyze(progress=progress)
        except AnalysisCancelled:
            review = None
        finally:
            self.analyzing = False
            self._set_busy(False)
        if review is None:
            self.flash("Analysis stopped. Press Analyze to grade your plays.")
            self.refresh()
            return
        panel.set_review(review, self._end_ms())
        self.flash(f"Review ready: accuracy {review.accuracy:.0f}% · click a play to see it on the board")
        self.refresh()
        self.reviewReady.emit()

    def cancel_analysis(self) -> None:
        if self.analyzing:
            self._cancel = True
            self.review_panel.progress_label.setText("Stopping…")

    def exit_review(self) -> None:
        """Leave the review: back to the final position and the result screen."""
        if self.showing_me:
            self.stop_show_me()
        if self.analyzing:
            self.cancel_analysis()
            return
        if self.mode != REVIEW:
            return
        self.board.ghosts, self.board.banner = [], None
        self.review_panel.clear_selection()
        self._goto_end()
        self._set_mode(ENDED)
        self.flash("")

    def _escape(self) -> None:
        if self.showing_me:
            self.stop_show_me()
        elif self.analyzing:
            self.cancel_analysis()
        else:
            self.select(None)

    def show_review_move(self, i: int) -> None:
        """Show graded play i on the board: just before it, your play in red, the recommendation in green."""
        if self.showing_me:
            self.stop_show_me()
        self.stop_playback()
        review = self.review_panel.review
        mv = review.moves[i]
        if self.session.restore_before_event(mv.event_id) is None:
            self.flash("That play is no longer on this timeline; re-analyze")
            return
        self._stop_countdown()
        self.running = False
        self._reload_log()
        self.select(None)
        a, b = mv.actual, mv.best
        ghosts = [(a.card_id, a.x, a.y, "actual")]
        if not b.is_wait and (b.card_id, b.x, b.y) != (a.card_id, a.x, a.y):
            ghosts.append((b.card_id, b.x, b.y, "best"))
        self.board.ghosts = ghosts
        tip = "Best: wait" if b.is_wait else ("Already the best play" if mv.loss <= 0.01
                                              else "Red = your play · Green = recommended")
        self.board.banner = f"{mmss(mv.t_ms)}  {mv.label_text} {mv.symbol}  ·  {tip}"
        self.refresh()

    # -- show me (animate recommended line) --------------------------------------------------------------------
    def toggle_show_me(self) -> None:
        """Toggle Show Me simulation of the recommended line (S)."""
        if getattr(self, "showing_me", False):
            self.stop_show_me()
        else:
            self.start_show_me()

    def start_show_me(self) -> None:
        """Simulate the coach's recommended alternative 8 seconds ahead in an isolated fork."""
        if self.mode != REVIEW or self.analyzing:
            return
        mv = self.review_panel.selected_move()
        if mv is None:
            self.flash("Select a move first, then press Show me")
            return
        self.stop_playback()
        self._stop_countdown()

        cp = self.session.pre_play.get(mv.event_id)
        if cp is None:
            self.flash("That play is no longer on this timeline; re-analyze")
            return

        origin = self.session
        forked = origin.fork_session(cp)
        self._show_me_origin_session = origin
        self._show_me_move_index = self.review_panel.selected_move_index
        self._show_me_move = mv
        self._show_me_start_ms = forked.match.t_ms
        self._show_me_end_ms = self._show_me_start_ms + 8000
        self.showing_me = True

        b, a = mv.best, mv.actual
        if not b.is_wait:
            res = forked.play(b.card_id, b.x, b.y)
            if not res.ok:
                self.flash(f"Could not simulate {b.card_id}: {res.reason}")
                self.stop_show_me()
                return
            rec_name = self.session.db.cards.get(b.card_id, {}).get("name", b.card_id)
        else:
            rec_name = "Wait (hold elixir)"

        self.session = forked
        self.board.session = forked
        self.counter.session = forked
        self.board.ghosts = [(a.card_id, a.x, a.y, "actual")]
        diff_str = f"+{mv.loss:.0%} win chance" if mv.loss > 0.01 else "best move"
        self.board.banner = f"👁 Show me: {rec_name} ({diff_str})  ·  0.0s / 8.0s  ·  Stop (S/Space) to exit"
        self.review_panel.show_me_btn.setText("■  Stop (S)")
        self.review_panel.show_me_btn.setStyleSheet(f"background:{C['loss']}; color:white;")
        self.flash(f"Simulating recommended line for 8s ({rec_name}) · Press Stop or S to exit")
        self.refresh()

    def _show_me_tick(self) -> None:
        cur_t = self.session.match.t_ms
        mv = self._show_me_move
        if mv is None:
            self.stop_show_me()
            return
        b = mv.best
        rec_name = "Wait (hold elixir)" if b.is_wait else self.session.db.cards.get(b.card_id, {}).get("name", b.card_id)
        diff_str = f"+{mv.loss:.0%} win chance" if mv.loss > 0.01 else "optimal"

        if cur_t >= self._show_me_end_ms or self.session.result is not None:
            self.board.banner = f"✓ Recommended line complete: win chance {mv.v_best:.0%} ({diff_str})  ·  Take over or Stop to exit"
            self.review_panel.show_me_btn.setText("↺  Exit preview")
            self.refresh()
            return

        self.add_log(self.session.tick(self.play_speed))
        cur_t = self.session.match.t_ms
        elapsed = min(8.0, max(0.0, (cur_t - self._show_me_start_ms) / 1000.0))
        self.board.banner = f"👁 Show me: {rec_name} ({diff_str})  ·  {elapsed:.1f}s / 8.0s  ·  Stop (S/Space) to exit"
        self.refresh()

    def stop_show_me(self) -> None:
        if not getattr(self, "showing_me", False):
            return
        origin = self._show_me_origin_session
        idx = self._show_me_move_index
        self.showing_me = False
        self._show_me_origin_session = None
        self._show_me_move_index = None
        self._show_me_move = None
        self.review_panel.show_me_btn.setText("👁  Show me (S)")
        self.review_panel.show_me_btn.setStyleSheet("")
        if origin is not None:
            self.session = origin
            self.board.session = origin
            self.counter.session = origin
            self.board.ghosts = []
            self.board.banner = None
            if idx is not None:
                self.show_review_move(idx)
            else:
                self.refresh()
            self.flash("Returned to review")

    # -- review playback ---------------------------------------------------------------------------------------
    def toggle_playback(self) -> None:
        """Play / Stop: watch the reviewed match from the moment on the board (Space)."""
        if self.showing_me:
            self.stop_show_me()
            return
        if self.playing:
            self.stop_playback()
            self.flash("Stopped · Take over from here, or press Play to continue")
            return
        if self.mode != REVIEW or self.analyzing:
            return
        s = self.session
        if s.result is not None or s.match.t_ms >= s.max_known_ms():
            s.seek(0)                                   # at the end: watch from the start
            self._reload_log()
        self._play_actions = s.review_actions()
        self.board.ghosts, self.board.banner = [], None
        self.review_panel.clear_selection()
        self.select(None)
        self.playing = True
        self._banner_until = None
        self.play_btn.setText("■  Stop")
        self.flash(f"Playing at {self.play_speed}x · Space or Stop to pause")
        self.refresh()

    def stop_playback(self) -> None:
        if not getattr(self, "playing", False):
            return
        self.playing = False
        self._play_actions = None
        self.play_btn.setText("▶  Play")

    def _cycle_play_speed(self) -> None:
        self.play_speed = {1: 2, 2: 4, 4: 1}[self.play_speed]
        self.speed_btn.setText(f"{self.play_speed}x")

    def _playback_tick(self) -> None:
        s = self.session
        before = s.match.t_ms
        t = s.replay_to(before + TICK_MS * self.play_speed, self._play_actions)
        review = self.review_panel.review
        if review is not None:                          # passing a graded play: show its grade for ~2.5 s
            for i, mv in enumerate(review.moves):
                if before < mv.t_ms <= t:
                    name = s.db.cards.get(mv.actual.card_id, {}).get("name", mv.actual.card_id)
                    self.board.banner = f"{mmss(mv.t_ms)}  {name}  {mv.symbol} {mv.label_text}  (−{mv.loss:.0%})"
                    self.review_panel.graph.selected = i
                    self._banner_until = t + 2500
        if self._banner_until is not None and t >= self._banner_until:
            self.board.banner, self._banner_until = None, None
            self.review_panel.graph.selected = None
        self.add_log([])
        self.refresh()
        if s.result is not None or t >= s.max_known_ms():
            self.stop_playback()
            self.flash("End of the match · drag the timeline or press Play to watch again")

    def review_goto(self, t_ms: int) -> None:
        """Review mode: select the graded play at t_ms if there is one, else jump to that moment."""
        if self.mode != REVIEW:
            return
        r = self.review_panel.review
        idx = next((i for i, mv in enumerate(r.moves) if mv.t_ms == t_ms), None) if r else None
        if idx is not None:
            self.review_panel.select(idx)
        else:
            self.scrub_to(t_ms)

    def scrub_to(self, t_ms: int) -> None:
        """Review mode: jump straight to t_ms (no countdown, nothing runs)."""
        if self.mode != REVIEW or self.analyzing:
            return
        if self.showing_me:
            self.stop_show_me()
        self.stop_playback()
        if self.session.seek(int(t_ms)) is not None:
            self._after_scrub()

    def _after_scrub(self) -> None:
        self.board.ghosts, self.board.banner = [], None
        self.review_panel.clear_selection()
        self._reload_log()
        self.select(None)
        self.refresh()

    def _slider_moved(self, value: int) -> None:
        if self.mode == REVIEW:
            self.scrub_to(value)

    def step(self, seconds: int) -> None:
        """U / F and the ⏪ / ⏩ buttons. Review: jump. Live: take over from there after a countdown."""
        if self.analyzing or self.mode not in (LIVE, REVIEW):
            return
        if self.showing_me:
            self.stop_show_me()
        self.stop_playback()
        s = self.session
        restored = s.rewind(-seconds) if seconds < 0 else s.forward(seconds)
        if self.mode == REVIEW:
            if restored is not None:
                self._after_scrub()
            return
        self.jump(restored, f"{'Back' if seconds < 0 else 'Forward'} {abs(seconds)} s")

    def take_over_review(self) -> None:
        if self.analyzing:
            return
        if self.showing_me:
            self.showing_me = False
            self._show_me_origin_session = None
            self._show_me_move_index = None
            self.review_panel.show_me_btn.setText("👁  Show me (S)")
            self.review_panel.show_me_btn.setStyleSheet("")
            self.board.ghosts, self.board.banner = [], None
            self.review_panel.mark_diverged()
        else:
            self.stop_playback()
        if self.session.result is not None:
            self.flash("This is the end of the match. Pick an earlier moment on the timeline first.")
            return
        self.jump(self.session.match.t_ms, "Take over")

    # -- match setup ------------------------------------------------------------------------------------------
    def restart(self, **changes) -> None:
        if self.analyzing:
            return
        self.aboutToReset.emit()
        self.session.reconfigure(**changes)
        self._reset_view()
        self.flash("Settings updated · press Start when ready")

    def new_match(self) -> None:
        if self.analyzing:
            return
        self.aboutToReset.emit()
        self.session.new_match()
        self._reset_view()
        self.flash("New match · choose your settings, then press Start")

    def _reset_view(self) -> None:
        self._stop_countdown()
        self.board.ghosts, self.board.banner = [], None
        self.review_panel.set_review(None, 0)
        self.running = False
        self.away_paused = False
        self.log_list.clear()
        self.select(None)
        self._sync_setup()
        self._set_mode(SETUP)

    def _sync_setup(self) -> None:
        """Show the session's settings in the setup panel (without triggering restarts)."""
        cfg = self.session.cfg
        db = self.session.db
        names, costs = deck_names(db.cards), deck_costs(db.cards)
        self.persona_box.blockSignals(True)
        self.persona_box.setCurrentIndex(list(PERSONAS).index(cfg.persona))
        self.persona_box.blockSignals(False)
        for rb, on in ((self.lv11, cfg.level != 16), (self.lv16, cfg.level == 16)):
            rb.blockSignals(True)
            rb.setChecked(on)
            rb.blockSignals(False)
        for side, deck, evo in ((HUMAN, cfg.player_deck, cfg.player_evo), (AI, cfg.ai_deck, cfg.ai_evo)):
            who, col = ("YOU", C["you"]) if side == HUMAN else ("AI", C["ai"])
            self.deck_title[side].setText(
                f"<span style='color:{col}; font-weight:800'>{who}</span>&nbsp;&nbsp;{deck_label(deck, evo)}"
                f"&nbsp;<span style='color:{C['muted']}'>· {avg_cost(deck, costs)} avg</span>")
            self.deck_view[side].set_deck(deck, evo, names, costs)

    def set_session(self, session: GameSession) -> None:
        """Show another session (e.g. a match reopened from Analysis), with its review if it has one."""
        if self.analyzing:
            return
        if self.showing_me:
            self.stop_show_me()
        self.aboutToReset.emit()
        self.session = session
        self.board.session = session
        self.counter.session = session
        self._reset_view()
        self.add_log([])
        self._enter_loaded_mode()

    def edit_decks(self, side: int = HUMAN) -> None:
        if self.analyzing:
            return
        dlg = DeckDialog(self.session, self, side)
        if dlg.exec():
            cfg = dlg.result_config()
            if any(tuple(getattr(self.session.cfg, k)) != tuple(v) for k, v in cfg.items()):
                self.restart(**cfg)

    def quiz_guess(self, guess: Optional[float] = None) -> None:
        """Elixir quiz: the match waits while you guess the AI's elixir, then you see the error."""
        if self.mode != LIVE or self.analyzing:
            return
        was_running = self.running
        self.running = False
        if guess is None or isinstance(guess, bool):
            guess, ok = QInputDialog.getDouble(self, "Elixir quiz", "How much elixir does the AI have right now?",
                                               5.0, 0.0, 10.0, 1)
            if not ok:
                self.running = was_running
                return
        r = self.session.quiz_guess(guess)
        verdict = "Perfect" if r.error <= 0.3 else "Close" if r.error <= 1.0 else "Far off"
        self.flash(f"AI had {r.truth:.1f} · you guessed {r.guess:.1f} · error {r.error:.1f} ({verdict})")
        self.running = was_running
        self.refresh()

    def _toggle_vision(self, on: bool) -> None:
        self.board.show_ai_vision = on
        self.board.update()

    # -- timeline jumps with a countdown (live play) ------------------------------------------------------------
    def _start_countdown(self) -> None:
        self.running = False
        self.countdown_ms = COUNTDOWN_MS
        self.board.overlay_text = "3"
        self.board.update()

    def jump(self, restored: Optional[int], what: str) -> None:
        """Continue live play from `restored` after a 3-2-1 countdown (take over / back / forward)."""
        if restored is None:
            self.flash("Nothing to jump to")
            self.refresh()
            return
        self._reload_log()
        self.select(None)
        if self.mode != LIVE:
            self._set_mode(LIVE)
        self._start_countdown()
        self.flash(f"{what} → {mmss(restored)} · get ready")
        self.refresh()

    def take_over(self, item: QListWidgetItem) -> None:
        """Double-click on a play-log line: play on from just before that play."""
        if self.analyzing or self.mode not in (LIVE, ENDED):
            return
        i = self.log_list.row(item)
        e = self.session.log[i]
        who = "you" if e.side == HUMAN else "the AI"
        name = self.session.db.cards[e.card_id]["name"]
        self.jump(self.session.take_over_before(i), f"Take over before {who} played {name}")

    def _stop_countdown(self) -> None:
        self.countdown_ms = 0
        self.board.overlay_text = None
        self.board.update()

    # -- input ------------------------------------------------------------------------------------------------
    def on_hand_slot(self, i: int) -> None:
        if self.mode != LIVE or self.analyzing:
            return
        hand = self.session.match.sides[HUMAN].hand
        if i < len(hand):
            self.select(None if self.selected == hand[i] else hand[i])

    def use_ability(self) -> None:
        if self.mode != LIVE or self.analyzing:
            return
        if self.countdown_ms > 0:
            self.flash("Wait for the countdown to finish")
            return
        cards = self.session.ability_cards()
        if not cards:
            self.flash("No champion ability is ready")
            return
        res = self.session.use_ability(cards[0])
        if res.ok:
            self.review_panel.mark_diverged()
            self.flash("")
            self.add_log([])
        else:
            self.flash(REASONS.get(res.reason, res.reason))

    def select(self, card: Optional[str]) -> None:
        self.selected = card
        self.board.set_selected(card)
        self.refresh()

    def on_tile(self, x: float, y: float) -> None:
        if not self.selected or self.mode != LIVE:
            return
        if self.countdown_ms > 0:
            self.flash("Wait for the countdown to finish before playing a card")
            return
        res = self.session.play(self.selected, x, y)
        if res.ok:
            self.review_panel.mark_diverged()
            self.flash("")
            self.select(None)
            self.add_log([])
        else:
            self.flash(REASONS.get(res.reason, res.reason))

    # -- view sync --------------------------------------------------------------------------------------------
    def flash(self, msg: str) -> None:
        self.status.setText(msg)
        self.status.setVisible(bool(msg))

    def _reload_log(self) -> None:
        self.log_list.clear()
        self.add_log([])

    def add_log(self, _new: List[LogEntry]) -> None:
        shown = self.log_list.count()
        names = {cid: c["name"] for cid, c in self.session.db.cards.items()}
        for e in self.session.log[shown:]:
            who = "You" if e.side == HUMAN else "AI"
            evo = " ★Evo" if e.evolved else ""
            what = "⚡ ability" if e.ability else f"({e.x:.1f}, {e.y:.1f})"
            text = f"{mmss(e.t_ms)}  {who}  {names[e.card_id]}{evo}  {what}"
            if e.ai_latency:
                cls_, ms = e.ai_latency
                text += f"\n        → AI noticed after {ms / 1000:.2f}s ({LATENCY_NAMES.get(cls_, cls_.value)})"
            it = QListWidgetItem(text)
            it.setForeground(QColor(C["you"]) if e.side == HUMAN else QColor(C["ai"]))
            self.log_list.addItem(it)
        self.log_list.scrollToBottom()

    def refresh(self) -> None:
        s = self.session
        m = s.match
        st = m.sides[HUMAN]
        clock, phase = s.clock()
        self.clock.setText(clock)
        self.phase.setText(phase or " ")
        self.crowns.setText(f"👑 <span style='color:{C['you']}'>You {st.crowns}</span>&nbsp; : &nbsp;"
                            f"<span style='color:{C['ai']}'>{m.sides[AI].crowns} AI</span>")
        if self.mode in (SETUP, LIVE):
            names = {cid: c["name"] for cid, c in s.db.cards.items()}
            costs = {cid: c["elixir"] for cid, c in s.db.cards.items()}
            evo_ready = {c for c in st.evo_cards if st.evo_counter[c] >= s.db.evo_cycles(c)}
            self.hand.refresh(st.hand, st.queue[0], names, costs, st.elixir, self.selected, evo_ready,
                              enabled=self.mode == LIVE,
                              abilities=[(c, s.ability_cost(c)) for c in s.ability_cards()] if self.mode == LIVE else ())
        if self.mode == LIVE:
            self.counter.refresh()
        end = max(s.max_known_ms(), 1000)
        if not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setRange(0, end)
            self.slider.setValue(m.t_ms)
            self.slider.blockSignals(False)
        self.slider_label.setText(f"{mmss(m.t_ms)} / {mmss(end)}")
        self.rew_btn.setEnabled(s.can_rewind() and not self.analyzing)
        self.fwd_btn.setEnabled(s.can_forward() and not self.analyzing)
        if self.mode == REVIEW:
            g = self.review_panel.graph
            if self.review_panel.review is None:
                g.end_ms = end
            g.set_cursor(m.t_ms)
            self.review_panel.take_btn.setEnabled(s.result is None and not self.analyzing)
        self.board.update()


def main() -> None:
    """Entry point: the CRbot shell (Fight / Analysis / Decks / Account) with this arena as the Fight page."""
    from frontend_app.qt.shell import main as shell_main
    shell_main()


if __name__ == "__main__":
    main()
