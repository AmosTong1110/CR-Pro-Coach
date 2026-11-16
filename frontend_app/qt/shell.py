"""CRbot app shell: sidebar navigation with Fight / Analysis / Decks / Account.

    python -m frontend_app.qt.app        (or: python -m frontend_app.qt.shell)

* Fight     the practice arena (frontend_app.qt.app.MainWindow, embedded as a page)
* Analysis  every saved match (user_data/matches/<id>/); open one to get the full review back
* Decks     most played decks from the replay data, marked by what CRbot can simulate today
* Account   local profile and settings (never leaves this Mac) plus data freshness

Matches are saved automatically when they end, when a review finishes, before a new match replaces the
current one, and when the app closes (only if you played at least one card).
"""
from __future__ import annotations

import sys
from datetime import datetime
from pathlib import Path
from typing import List, Optional

from PySide6.QtCore import QSize, Qt, QUrl
from PySide6.QtGui import QColor, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QPushButton, QScrollArea, QSlider, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from backend_engine.meta.archetypes import avg_elixir, fmt_avg, pretty
from backend_engine.meta.recommend import coverage, meta_info, recommend
from backend_engine.simulator.engine import ENGINE_VERSION
from backend_engine.simulator.match_runner import DEFAULT_DECK
from backend_engine.storage import default_data_dir
from backend_engine.storage.match_store import MatchMeta, MatchStore, MatchStoreError
from backend_engine.storage.profile import Profile, load_profile, save_profile
from frontend_app.qt.app import SPEEDS, MainWindow, avg_cost, deck_label
from frontend_app.qt.review import LABEL_COLOR, WinRateGraph, grade_chips_html, mmss
from frontend_app.qt.session import PERSONAS, persona_label, GameSession, ReplayMismatch, SessionConfig
from frontend_app.qt.theme import C, STYLE, cls, dark_palette  # noqa: F401  (C/STYLE/dark_palette re-exported)
from frontend_app.qt.widgets import DeckView, StatTile, card_frame, deck_costs, deck_names

# Palette + stylesheet live in theme.py (shared with the Fight page and dialogs); re-exported here.

NAV = (("fight", "⚔️   Fight"), ("analysis", "📈   Analysis"), ("decks", "🃏   Decks"), ("account", "👤   Account"))
OUTCOME_TEXT = {"win": "Win", "loss": "Loss", "draw": "Draw", None: "Unfinished"}


_cls = cls


def _label(text: str, cls: str = "", wrap: bool = False) -> QLabel:
    lab = QLabel(text)
    if cls:
        _cls(lab, cls)
    lab.setWordWrap(wrap)
    return lab


def _page() -> QWidget:
    w = QWidget()
    _cls(w, "page")
    w.setAttribute(Qt.WA_StyledBackground, True)
    return w


def _when(iso: str) -> str:
    try:
        return datetime.fromisoformat(iso).strftime("%b %d, %H:%M")
    except ValueError:
        return iso


def match_title(m: MatchMeta) -> str:
    """'CRbot Starter vs Hog 2.6 Cycle': preset names where the deck is a preset, else the archetype."""
    return f"{deck_label(m.player_deck, m.player_evo)} vs {deck_label(m.opp_deck, m.opp_evo)}"


def deck_text(deck, evo=()) -> str:
    return ", ".join(("Evo " if c in evo else "") + pretty(c) for c in deck)


# ---------------------------------------------------------------------------
class MatchRow(QWidget):
    """One saved match in the Analysis list: outcome stripe, result, when, title, length and accuracy."""

    def __init__(self, m: MatchMeta):
        super().__init__()
        self.setAttribute(Qt.WA_TransparentForMouseEvents)      # the list handles clicks and selection
        col = C[m.outcome] if m.outcome else C["muted"]
        h = QHBoxLayout(self)
        h.setContentsMargins(8, 8, 10, 8)
        h.setSpacing(10)
        bar = QFrame()
        bar.setFixedWidth(4)
        bar.setStyleSheet(f"background:{col}; border-radius:2px;")
        h.addWidget(bar)
        v = QVBoxLayout()
        v.setSpacing(2)
        top = QHBoxLayout()
        res = QLabel(f"{OUTCOME_TEXT[m.outcome]}  {m.crowns[0]}–{m.crowns[1]}")
        res.setStyleSheet(f"color:{col}; font-weight:800; font-size:13px; background:transparent;")
        when = QLabel(_when(m.created))
        when.setStyleSheet(f"color:{C['muted']}; font-size:11px; background:transparent;")
        top.addWidget(res)
        top.addStretch(1)
        top.addWidget(when)
        title = QLabel(match_title(m))
        title.setStyleSheet(f"color:{C['text']}; font-size:12px; background:transparent;")
        acc = f"accuracy {m.accuracy:.0f}%" if m.accuracy is not None else "not analyzed"
        sub = QLabel(f"{mmss(m.duration_ms)}  ·  {m.n_plays} plays  ·  {acc}")
        sub.setStyleSheet(f"color:{C['muted']}; font-size:11px; background:transparent;")
        v.addLayout(top)
        v.addWidget(title)
        v.addWidget(sub)
        h.addLayout(v, 1)


class AnalysisPage(QWidget):
    """Saved matches (newest first) on the left; the selected match on the right: summary, decks, win-rate graph
    with a time slider, and Open review (at the chosen moment) / Analyze / Delete."""

    def __init__(self, shell: "CRbotWindow"):
        super().__init__()
        self.shell = shell
        self.metas: List[MatchMeta] = []
        self.review = None
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        head = QHBoxLayout()
        head.addWidget(_label("Analysis", "h1"))
        head.addStretch(1)
        self.filter = QComboBox()
        for text, mode in (("All matches", None), ("Practice", "practice"), ("Recordings", "recording"),
                           ("Drills", "drill")):
            self.filter.addItem(text, mode)
        self.filter.currentIndexChanged.connect(lambda _: self.reload())
        head.addWidget(self.filter)
        root.addLayout(head)

        tiles = QHBoxLayout()
        tiles.setSpacing(12)
        self.t_matches, self.t_winrate = StatTile("matches"), StatTile("win rate")
        self.t_record, self.t_acc = StatTile("wins · losses · draws"), StatTile("average accuracy")
        for t in (self.t_matches, self.t_winrate, self.t_record, self.t_acc):
            tiles.addWidget(t, 1)
        root.addLayout(tiles)
        self.stats = _label("", "sub")          # one-line summary (kept for tests / screen readers)
        self.stats.setVisible(False)
        root.addWidget(self.stats)

        body = QHBoxLayout()
        body.setSpacing(16)
        left = QVBoxLayout()
        left.setSpacing(6)
        left.addWidget(_label("SAVED MATCHES", "section"))
        self.list = QListWidget()
        self.list.setFixedWidth(330)
        self.list.setSpacing(2)
        self.list.setStyleSheet("QListWidget::item{padding:0px;}")
        self.list.currentRowChanged.connect(self.show_match)
        self.list.itemDoubleClicked.connect(lambda _: self.open_selected())
        left.addWidget(self.list, 1)
        body.addLayout(left)

        # -- detail (scrolls instead of squeezing on small windows) --
        inner = QWidget()
        dv = QVBoxLayout(inner)
        dv.setContentsMargins(0, 0, 4, 0)
        dv.setSpacing(12)

        self.head_card = card_frame(spacing=8, margins=16)
        hl = self.head_card.layout()
        tr = QHBoxLayout()
        self.title = _label("", "h1", wrap=True)
        self.badge = QLabel()
        tr.addWidget(self.title, 1)
        tr.addWidget(self.badge, 0, Qt.AlignTop)
        hl.addLayout(tr)
        self.info = _label("", "sub", wrap=True)
        self.info.setTextFormat(Qt.RichText)
        hl.addWidget(self.info)
        self.deck_titles, self.deck_views = {}, {}
        for side in ("you", "ai"):
            t = QLabel()
            t.setTextFormat(Qt.RichText)
            dvw = DeckView()
            hl.addWidget(t)
            hl.addWidget(dvw)
            self.deck_titles[side], self.deck_views[side] = t, dvw
        dv.addWidget(self.head_card)

        self.graph_card = card_frame(spacing=8, margins=14)
        gl = self.graph_card.layout()
        gl.addWidget(_label("WIN CHANCE OVER TIME", "section"))
        self.grades = QLabel()
        self.grades.setTextFormat(Qt.RichText)
        self.grades.setWordWrap(True)
        self.grades.setStyleSheet("font-size:13px;")
        gl.addWidget(self.grades)
        self.graph = WinRateGraph()
        self.graph.setFixedHeight(220)
        self.graph.empty_text = "Not analyzed yet. Press Analyze to grade every play."
        self.graph.timeClicked.connect(lambda t: self.slider.setValue(t))
        self.graph.moveClicked.connect(self._move_clicked)
        gl.addWidget(self.graph)
        sr = QHBoxLayout()
        sr.addWidget(_label("Jump to", "sub"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setSingleStep(1000)
        self.slider.setPageStep(10_000)
        self.slider.setToolTip("Pick the moment to open the review at (or click the graph)")
        self.slider.valueChanged.connect(self._time_changed)
        sr.addWidget(self.slider, 1)
        self.time_label = _label("0:00", "sub")
        self.time_label.setMinimumWidth(80)
        self.time_label.setAlignment(Qt.AlignCenter)
        sr.addWidget(self.time_label)
        gl.addLayout(sr)
        self.moment = QLabel()
        self.moment.setTextFormat(Qt.RichText)
        self.moment.setWordWrap(True)
        self.moment.setStyleSheet("font-size:12px;")
        gl.addWidget(self.moment)
        dv.addWidget(self.graph_card)

        btns = QHBoxLayout()
        self.open_btn = _cls(QPushButton("▶  Open review"), "primary")
        self.open_btn.setToolTip("Open this match in Fight with the coach review, at the chosen moment")
        self.open_btn.clicked.connect(lambda: self.open_selected())
        self.analyze_btn = _cls(QPushButton("Analyze"), "ghost")
        self.analyze_btn.clicked.connect(lambda: self.open_selected(analyze=True))
        self.delete_btn = _cls(QPushButton("Delete"), "danger")
        self.delete_btn.clicked.connect(self.delete_selected)
        btns.addWidget(self.open_btn)
        btns.addWidget(self.analyze_btn)
        btns.addStretch(1)
        btns.addWidget(self.delete_btn)
        dv.addLayout(btns)
        self.notice = _label("", "sub", wrap=True)
        self.notice.setStyleSheet(f"color:{C['warn']}; font-size:12px;")
        dv.addWidget(self.notice)
        self.foot = _label("", "muted", wrap=True)
        dv.addWidget(self.foot)
        dv.addStretch(1)

        self.detail = QScrollArea()
        self.detail.setWidgetResizable(True)
        self.detail.setFrameShape(QFrame.NoFrame)
        self.detail.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.detail.setWidget(inner)

        self.empty = card_frame(spacing=10, margins=24)
        self.empty.layout().addWidget(_label("No saved matches yet", "h2"))
        self.empty.layout().addWidget(_label("Play a match in Fight: it is saved automatically when it ends or "
                                             "when you start a new one.", "sub", wrap=True))
        go = _cls(QPushButton("Go to Fight"), "primary")
        go.clicked.connect(lambda: self.shell.show_page("fight"))
        self.empty.layout().addWidget(go, 0, Qt.AlignLeft)
        self.empty.layout().addStretch(1)

        right = QVBoxLayout()
        right.addWidget(self.detail, 1)
        right.addWidget(self.empty, 1)
        body.addLayout(right, 1)
        root.addLayout(body, 1)

    # ------------------------------------------------------------------
    def reload(self, select_id: Optional[str] = None) -> None:
        self.metas = self.shell.store.list(self.filter.currentData())
        self.list.blockSignals(True)
        self.list.clear()
        for m in self.metas:
            it = QListWidgetItem()
            it.setData(Qt.UserRole, m.match_id)
            row = MatchRow(m)
            it.setSizeHint(row.sizeHint())
            self.list.addItem(it)
            self.list.setItemWidget(it, row)
        self.list.blockSignals(False)
        w = sum(m.outcome == "win" for m in self.metas)
        l_ = sum(m.outcome == "loss" for m in self.metas)
        d = sum(m.outcome == "draw" for m in self.metas)
        decided = w + l_ + d
        accs = [m.accuracy for m in self.metas if m.accuracy is not None]
        self.t_matches.set(str(len(self.metas)))
        self.t_winrate.set(f"{w / decided:.0%}" if decided else "—",
                           C["win"] if decided and w >= l_ else (C["loss"] if decided else None))
        self.t_record.set(f"{w} · {l_} · {d}")
        self.t_acc.set(f"{sum(accs) / len(accs):.0f}%" if accs else "—")
        self.stats.setText(f"{len(self.metas)} matches  ·  {w}W {l_}L {d}D"
                           + (f"  ·  average accuracy {sum(accs) / len(accs):.0f}%" if accs else ""))
        row = 0
        if select_id is not None:
            row = next((i for i, m in enumerate(self.metas) if m.match_id == select_id), 0)
        if self.metas:
            self.list.setCurrentRow(row)
        self.show_match(self.list.currentRow())

    def selected(self) -> Optional[MatchMeta]:
        i = self.list.currentRow()
        return self.metas[i] if 0 <= i < len(self.metas) else None

    def show_match(self, i: int) -> None:
        m = self.metas[i] if 0 <= i < len(self.metas) else None
        has = m is not None
        self.detail.setVisible(has)
        self.empty.setVisible(not has)
        if not has:
            return
        self.title.setText(match_title(m))
        col = C[m.outcome] if m.outcome else C["muted"]
        self.badge.setText(f"  {OUTCOME_TEXT[m.outcome]}  {m.crowns[0]}–{m.crowns[1]}  ")
        self.badge.setStyleSheet(f"color:#0b1f1d; background:{col}; border-radius:6px; padding:4px 6px;"
                                 "font-weight:800; font-size:13px;")
        persona = PERSONAS.get(m.persona or "", {}).get("label", m.persona or "—")
        why = f" ({m.reason.replace('_', ' ')})" if m.reason else ""
        self.info.setText(f"{_when(m.created)} · {mmss(m.duration_ms)}{why} · level {m.level} · "
                          f"AI: {persona} · {m.n_plays} plays")
        db = self.shell.fight.session.db.cards
        names, costs = deck_names(db), deck_costs(db)
        for side, deck, evo, who, c in (("you", m.player_deck, m.player_evo, "YOU", C["you"]),
                                        ("ai", m.opp_deck, m.opp_evo, "AI", C["ai"])):
            self.deck_titles[side].setText(f"<span style='color:{c}; font-weight:800'>{who}</span>&nbsp;&nbsp;"
                                           f"{deck_label(deck, evo)}&nbsp;<span style='color:{C['muted']}'>· "
                                           f"{avg_cost(deck, costs)} avg</span>")
            self.deck_views[side].set_deck(deck, evo, names, costs)
        review = None
        try:
            review = self.shell.store.load_review(m.match_id)
        except MatchStoreError:
            pass
        self.review = review
        self.graph.set_review(review, m.duration_ms)
        if review is not None:
            self.grades.setText(f"<b>Accuracy {review.accuracy:.0f}%</b>&nbsp;&nbsp;&nbsp;&nbsp;"
                                f"{grade_chips_html(review)}")
        else:
            self.grades.setText(f"<span style='color:{C['muted']}'>Not analyzed yet</span>")
        self.analyze_btn.setText("Re-analyze" if review is not None else "Analyze")
        self.slider.blockSignals(True)
        self.slider.setRange(0, max(m.duration_ms, 1000))
        self.slider.setValue(m.duration_ms)
        self.slider.blockSignals(False)
        self._time_changed(self.slider.value())
        replayable = m.mode == "practice" and m.engine_version == ENGINE_VERSION
        self.open_btn.setEnabled(replayable)
        self.analyze_btn.setEnabled(replayable)
        self.slider.setEnabled(replayable)
        self.notice.setText("" if m.engine_version == ENGINE_VERSION else
                            f"Made with an older engine (v{m.engine_version}): summary and saved review only, "
                            "it can't be replayed.")
        self.notice.setVisible(bool(self.notice.text()))
        self.foot.setText(f"card data {m.data_version or 'unknown'} · id {m.match_id}")

    def _move_clicked(self, i: int) -> None:
        if self.review and 0 <= i < len(self.review.moves):
            self.graph.selected = i
            self.slider.setValue(self.review.moves[i].t_ms)

    def _time_changed(self, t: int) -> None:
        m = self.selected()
        end = m.duration_ms if m else 0
        self.graph.set_cursor(t)
        self.time_label.setText(f"{mmss(t)} / {mmss(end)}")
        at_end = t >= end
        self.open_btn.setText("▶  Open review" if at_end else f"▶  Open review at {mmss(t)}")
        mv = None
        if self.review is not None and self.review.moves:
            near = min(self.review.moves, key=lambda x: abs(x.t_ms - t))
            mv = near if abs(near.t_ms - t) <= 1500 else None
        if mv is not None:
            col = LABEL_COLOR[mv.label].name()
            name = self.shell.fight.session.db.cards.get(mv.actual.card_id, {}).get("name", mv.actual.card_id)
            self.moment.setText(f"{mmss(mv.t_ms)} · your {name} — <span style='color:{col}; font-weight:700'>"
                                f"{mv.label_text} {mv.symbol}</span> (−{mv.loss * 100:.0f}% win chance)")
        else:
            self.moment.setText(f"<span style='color:{C['muted']}'>Drag the slider or click the graph to choose "
                                "where the review opens. Click a dot to open at that play.</span>")
        if self.graph.selected is not None and self.review is not None and (
                self.graph.selected >= len(self.review.moves) or self.review.moves[self.graph.selected].t_ms != t):
            self.graph.selected = None
            self.graph.update()

    def open_selected(self, analyze: bool = False) -> None:
        m = self.selected()
        if m is None:
            return
        t = self.slider.value()
        self.shell.open_match(m.match_id, analyze=analyze, at_ms=t if t < m.duration_ms else None)

    def delete_selected(self) -> None:
        m = self.selected()
        if m is None:
            return
        if QMessageBox.question(self, "Delete match", f"Delete “{m.title}” ({_when(m.created)})?\n"
                                "This removes its saved review too.") != QMessageBox.Yes:
            return
        self.shell.store.delete(m.match_id)
        self.reload()


# ---------------------------------------------------------------------------
class DecksPage(QWidget):
    """Most played decks, marked by whether CRbot can simulate every card."""

    def __init__(self, shell: "CRbotWindow"):
        super().__init__()
        self.shell = shell
        self.rows = []
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.addWidget(_label("Recommended decks", "h1"))
        self.sub = _label("", "sub", wrap=True)
        root.addWidget(self.sub)
        bar = QHBoxLayout()
        self.only_ready = QCheckBox("Only decks CRbot can simulate")
        self.only_ready.toggled.connect(lambda _: self.reload())
        bar.addWidget(self.only_ready)
        bar.addStretch(1)
        self.vs_btn = _cls(QPushButton("Fight against it"), "ghost")
        self.vs_btn.clicked.connect(lambda: self.use(as_ai=True))
        self.use_btn = _cls(QPushButton("Fight with this deck"), "primary")
        self.use_btn.clicked.connect(lambda: self.use(as_ai=False))
        builder = _cls(QPushButton("Deck builder…"), "ghost")
        builder.clicked.connect(self.shell.open_deck_builder)
        for b in (builder, self.vs_btn, self.use_btn):
            bar.addWidget(b)
        root.addLayout(bar)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Archetype", "Cards", "Avg", "Play rate", "Status"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.table.setWordWrap(True)
        hh = self.table.horizontalHeader()
        for col in (1, 4):
            hh.setSectionResizeMode(col, QHeaderView.Stretch)
        for col in (0, 2, 3):
            hh.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        self.table.setStyleSheet(f"QTableWidget{{background:{C['panel']}; color:{C['text']}; border:1px solid "
                                 f"{C['line']}; border-radius:8px; gridline-color:{C['line']};}}"
                                 f"QHeaderView::section{{background:{C['panel2']}; color:{C['muted']}; border:none;"
                                 f"padding:6px;}} QTableWidget::item:selected{{background:#2b3a4a; color:{C['text']};}}")
        self.table.itemSelectionChanged.connect(self._sel_changed)
        root.addWidget(self.table, 1)
        self.reload()

    def reload(self) -> None:
        sim = self.shell.db_cards()
        info = meta_info()
        ok, total = coverage(sim)
        self.sub.setText(
            f"Most played decks in {info.get('battles') or 0:,} replays ({info.get('as_of') or '?'}), "
            f"ranked by play rate. CRbot simulates {len(sim)} cards today, so {ok} of {total:,} decks are fully "
            f"playable; the rest unlock as cards are added. The starter deck is always playable.")
        starter = ("CRbot starter", tuple(DEFAULT_DECK), avg_elixir(DEFAULT_DECK), None, ())
        recs = recommend(sim, limit=40, playable_only=self.only_ready.isChecked())
        self.rows = [starter] + [(r.archetype, r.cards, r.avg_elixir, r.share, r.missing) for r in recs]
        self.table.setRowCount(len(self.rows))
        for i, (name, cards, avg, share, missing) in enumerate(self.rows):
            vals = [name, ", ".join(pretty(c) for c in cards), fmt_avg(avg) if avg else "—",
                    f"{share:.1%}" if share is not None else "—",
                    "Ready" if not missing else "Needs " + ", ".join(pretty(c) for c in missing)]
            for col, v in enumerate(vals):
                it = QTableWidgetItem(v)
                if col == 4:
                    it.setForeground(QColor(C["win"] if not missing else C["muted"]))
                self.table.setItem(i, col, it)
        self.table.resizeRowsToContents()
        self.table.selectRow(0)
        self._sel_changed()

    def _selected_row(self):
        rows = self.table.selectionModel().selectedRows() if self.table.selectionModel() else []
        return self.rows[rows[0].row()] if rows else None

    def _sel_changed(self) -> None:
        r = self._selected_row()
        ready = r is not None and not r[4]
        self.use_btn.setEnabled(ready)
        self.vs_btn.setEnabled(ready)

    def use(self, as_ai: bool) -> None:
        r = self._selected_row()
        if r is None or r[4]:
            return
        self.shell.fight_with(tuple(r[1]), as_ai=as_ai)


# ---------------------------------------------------------------------------
class AccountPage(QWidget):
    """Local profile, defaults, Mode B paths and data freshness."""

    def __init__(self, shell: "CRbotWindow"):
        super().__init__()
        self.shell = shell
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.addWidget(_label("Account", "h1"))
        root.addWidget(_label("Your profile and settings stay on this Mac. CRbot never sends your player tag "
                              "anywhere.", "sub", wrap=True))
        cols = QHBoxLayout()

        prof = QGroupBox("Profile && defaults")
        f = QFormLayout(prof)
        self.name = QLineEdit()
        self.tag = QLineEdit()
        self.tag.setPlaceholderText("optional, e.g. #2PQ8…")
        self.level = QComboBox()
        self.level.addItem("Level 11 (tournament)", 11)
        self.level.addItem("Level 16 (max)", 16)
        self.persona = QComboBox()
        for key, p in PERSONAS.items():
            self.persona.addItem(persona_label(key), key)
        self.speed = QComboBox()
        for label, v in SPEEDS:
            self.speed.addItem(label, v)
        self.autosave = QCheckBox("Save every match automatically")
        self.autoreview = QCheckBox("Analyze automatically when a match ends")
        f.addRow("Display name", self.name)
        f.addRow("Player tag", self.tag)
        f.addRow("Default level", self.level)
        f.addRow("Default AI persona", self.persona)
        f.addRow("Default speed", self.speed)
        f.addRow(self.autosave)
        f.addRow(self.autoreview)
        cols.addWidget(prof, 1)

        right = QVBoxLayout()
        live = QGroupBox("Mode B (screen companion)")
        g = QFormLayout(live)
        self.detector = QLineEdit()
        self.detector.setPlaceholderText("auto: first .onnx in pretrained_weights/")
        self.calib = QLineEdit()
        self.calib.setPlaceholderText("none yet")
        g.addRow("Detector model", self._with_browse(self.detector, "ONNX (*.onnx)"))
        g.addRow("Calibration", self._with_browse(self.calib, "JSON (*.json)"))
        right.addWidget(live)
        fresh = QGroupBox("Data freshness")
        v = QVBoxLayout(fresh)
        self.fresh = QLabel()
        self.fresh.setTextFormat(Qt.RichText)
        self.fresh.setWordWrap(True)
        v.addWidget(self.fresh)
        open_dir = _cls(QPushButton("Open data folder"), "ghost")
        open_dir.clicked.connect(lambda: QDesktopServices.openUrl(QUrl.fromLocalFile(str(self.shell.data_dir))))
        v.addWidget(open_dir, 0, Qt.AlignLeft)
        right.addWidget(fresh)
        right.addStretch(1)
        cols.addLayout(right, 1)
        root.addLayout(cols)

        row = QHBoxLayout()
        self.saved = _label("", "sub")
        row.addWidget(self.saved)
        row.addStretch(1)
        save = _cls(QPushButton("Save settings"), "primary")
        save.clicked.connect(self.save)
        row.addWidget(save)
        root.addLayout(row)
        root.addStretch(1)
        self.load()

    def _with_browse(self, edit: QLineEdit, filt: str) -> QWidget:
        w = QWidget()
        h = QHBoxLayout(w)
        h.setContentsMargins(0, 0, 0, 0)
        h.addWidget(edit, 1)
        b = _cls(QPushButton("Browse…"), "ghost")
        b.clicked.connect(lambda: self._browse(edit, filt))
        h.addWidget(b)
        return w

    def _browse(self, edit: QLineEdit, filt: str) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Choose file", edit.text() or str(Path.home()), filt)
        if path:
            edit.setText(path)

    def load(self) -> None:
        p = self.shell.profile
        self.name.setText(p.display_name)
        self.tag.setText(f"#{p.player_tag}" if p.player_tag else "")
        self.level.setCurrentIndex(max(0, self.level.findData(p.default_level)))
        self.persona.setCurrentIndex(max(0, self.persona.findData(p.default_persona)))
        self.speed.setCurrentIndex(max(0, self.speed.findData(p.default_speed)))
        self.autosave.setChecked(p.autosave_matches)
        self.autoreview.setChecked(p.auto_review)
        self.detector.setText(p.detector_model)
        self.calib.setText(p.calibration_file)
        self.refresh_freshness()

    def refresh_freshness(self) -> None:
        db = self.shell.fight.session.db
        dates = sorted({c.get("data_as_of", "?") for c in db.cards.values()})
        info = meta_info()
        n_matches = len(self.shell.store.list())
        self.fresh.setText(
            f"<b>Card stats</b>: {db.raw.get('data_version', '?')} · balance numbers as of "
            f"{dates[0] if dates else '?'}{' – ' + dates[-1] if len(dates) > 1 else ''} "
            f"<span style='color:{C['loss']}'>(stale: needs a current source)</span><br>"
            f"<b>Simulated cards</b>: {len(db.cards)}<br>"
            f"<b>Meta decks</b>: {info.get('battles') or 0:,} battles, as of {info.get('as_of') or '?'}<br>"
            f"<b>Saved matches</b>: {n_matches}<br>"
            f"<b>Data folder</b>: {self.shell.data_dir}")

    def save(self) -> None:
        raw_tag = self.tag.text().strip()
        tag = Profile.clean_tag(raw_tag) if raw_tag else ""
        if raw_tag and not tag:
            QMessageBox.warning(self, "Player tag", "That doesn't look like a player tag (letters 0289PYLQGRJCUV).")
            return
        p = Profile(display_name=self.name.text().strip(), player_tag=tag,
                    default_level=self.level.currentData(), default_persona=self.persona.currentData(),
                    default_speed=self.speed.currentData(), autosave_matches=self.autosave.isChecked(),
                    auto_review=self.autoreview.isChecked(), detector_model=self.detector.text().strip(),
                    calibration_file=self.calib.text().strip())
        self.shell.apply_profile(p, persist=True)
        self.load()
        self.saved.setText(f"Saved {datetime.now():%H:%M:%S}")


# ---------------------------------------------------------------------------
class CRbotWindow(QMainWindow):
    def __init__(self, session: Optional[GameSession] = None, data_dir: Optional[Path] = None):
        super().__init__()
        self.setWindowTitle("CRbot")
        self.data_dir = Path(data_dir) if data_dir is not None else default_data_dir()
        self.store = MatchStore(self.data_dir / "matches")
        self.profile = load_profile(self.data_dir)
        if session is None:
            session = GameSession(cfg=SessionConfig(level=self.profile.default_level,
                                                    persona=self.profile.default_persona))
        self.fight = MainWindow(session)
        self.fight.setWindowFlags(Qt.Widget)
        self.fight.matchEnded.connect(lambda: self.save_current())
        self.fight.reviewReady.connect(lambda: self.save_current())
        self.fight.aboutToReset.connect(lambda: self.save_current())
        self.fight.busyChanged.connect(self._fight_busy)
        self.fight.modeChanged.connect(lambda _m: self._sync_sidebar())
        self._pending_page: Optional[str] = None
        self._fight_shortcuts = self.fight.findChildren(QShortcut)

        self.pages = QStackedWidget()
        self.pages.setObjectName("Pages")
        self.analysis = AnalysisPage(self)
        self.decks = DecksPage(self)
        self.account = AccountPage(self)
        self.page_index = {}
        for key, w in (("fight", self.fight), ("analysis", self.analysis), ("decks", self.decks),
                       ("account", self.account)):
            if w is not self.fight:
                _cls(w, "page")
                w.setAttribute(Qt.WA_StyledBackground, True)
            self.page_index[key] = self.pages.addWidget(w)

        side = self.sidebar = QWidget()
        side.setObjectName("Sidebar")
        side.setAttribute(Qt.WA_StyledBackground, True)
        side.setFixedWidth(190)
        sv = QVBoxLayout(side)
        sv.setContentsMargins(0, 0, 0, 12)
        sv.setSpacing(0)
        brand = QLabel("CRbot")
        brand.setObjectName("Brand")
        sub = QLabel("SPARRING · COACH")
        sub.setObjectName("BrandSub")
        sv.addWidget(brand)
        sv.addWidget(sub)
        line = QFrame()
        line.setFrameShape(QFrame.HLine)
        line.setStyleSheet(f"color:{C['line']};")
        sv.addWidget(line)
        sv.addSpacing(8)
        self.nav = QButtonGroup(self)
        self.nav.setExclusive(True)
        self.nav_buttons = {}
        for i, (key, text) in enumerate(NAV):
            b = _cls(QPushButton(text), "nav")
            b.setCheckable(True)
            b.setCursor(Qt.PointingHandCursor)
            b.setToolTip(f"{text.split()[-1]} (Ctrl+{i + 1})")
            b.clicked.connect(lambda _=False, k=key: self.show_page(k))
            self.nav.addButton(b)
            self.nav_buttons[key] = b
            if key == "account":
                sv.addStretch(1)
            sv.addWidget(b)
            QShortcut(QKeySequence(f"Ctrl+{i + 1}"), self, activated=lambda k=key: self.show_page(k))

        root = QWidget()
        root.setObjectName("ShellRoot")
        h = QHBoxLayout(root)
        h.setContentsMargins(0, 0, 0, 0)
        h.setSpacing(0)
        h.addWidget(side)
        h.addWidget(self.pages, 1)
        self.setCentralWidget(root)
        self.setPalette(dark_palette())
        self.setStyleSheet(STYLE)
        self.resize(self._fit_size(QSize(1200, 900)))
        self.apply_profile(self.profile, persist=False)
        self.show_page("fight")

    # -- navigation ----------------------------------------------------------
    def current_page(self) -> str:
        i = self.pages.currentIndex()
        return next(k for k, v in self.page_index.items() if v == i)

    @staticmethod
    def _fit_size(want: QSize) -> QSize:
        """The preferred size, shrunk to the screen (a 13" laptop is ~800 px tall) so nothing gets squeezed."""
        screen = QApplication.primaryScreen()
        if screen is None:
            return want
        avail = screen.availableGeometry()
        return QSize(max(900, min(want.width(), avail.width() - 40)), max(640, min(want.height(), avail.height() - 40)))

    def _sync_sidebar(self) -> None:
        """During a live match the window is just the match: no Fight / Analysis / Decks / Account list."""
        live = self.current_page() == "fight" and self.fight.mode == "live"
        self.sidebar.setVisible(not live)

    def _fight_busy(self, busy: bool) -> None:
        """Analysis done or stopped: go to the page you asked for while it was running."""
        if not busy and self._pending_page:
            key, self._pending_page = self._pending_page, None
            self.show_page(key)

    def show_page(self, key: str) -> None:
        if self.fight.analyzing and key != "fight":
            self._pending_page = key            # never locked out: stop the analysis, then go there
            self.fight.cancel_analysis()
            return
        if key != "fight":
            self.fight.pause_for_away()         # never let a match run unseen
        for sc in self._fight_shortcuts:        # fight hotkeys (Space, 1-4, N, …) only on the Fight page
            sc.setEnabled(key == "fight")
        if key == "analysis":
            self.save_current()
            self.analysis.reload(select_id=self.fight.session.match_id)
        elif key == "account":
            self.account.refresh_freshness()
        self.nav_buttons[key].setChecked(True)
        self.pages.setCurrentIndex(self.page_index[key])
        if key == "fight":
            self.fight.resume_after_away()      # 3-2-1, then the match continues
        self._sync_sidebar()

    # -- profile ---------------------------------------------------------------
    def apply_profile(self, p: Profile, persist: bool) -> None:
        self.profile = p
        if persist:
            save_profile(p, self.data_dir)
        self.fight.auto_review = p.auto_review
        self.fight.speed = p.default_speed

    def db_cards(self):
        return self.fight.session.db.cards

    # -- saving and reopening ----------------------------------------------------
    def save_current(self) -> Optional[str]:
        """Save the Fight page's match if you played at least one card. Returns the match id."""
        s = self.fight.session
        if not self.profile.autosave_matches or not s.has_human_plays():
            return None
        meta, record = s.to_record("practice")
        review = s.review if s.review_is_current() else None   # a review of an abandoned branch is dropped
        self.store.save(meta, record, review)
        return meta.match_id

    def open_match(self, match_id: str, analyze: bool = False, at_ms: Optional[int] = None) -> bool:
        """Rebuild a saved match in the Fight page, with its review (or analyze it now). `at_ms` opens the
        review at that moment (or at the play made then)."""
        try:
            meta = self.store.load_meta(match_id)
            record = self.store.load_record(match_id)
            review = self.store.load_review(match_id)
        except MatchStoreError as e:
            QMessageBox.warning(self, "Open match", str(e))
            return False
        if meta.engine_version != ENGINE_VERSION:
            QMessageBox.information(self, "Open match", f"This match was played on an older engine "
                                    f"(v{meta.engine_version}, now v{ENGINE_VERSION}), so it can't be replayed. "
                                    "Its summary and saved review stay available in Analysis.")
            return False
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            session = GameSession.rebuild(record, db=self.fight.session.db)
        except (ReplayMismatch, KeyError, ValueError) as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.information(self, "Open match",
                                    "This match can't be replayed with the current card data, so it can't be "
                                    f"opened in Fight. Its saved summary and review stay available here.\n\n({e})")
            return False
        QApplication.restoreOverrideCursor()
        session.mark_loaded(meta, review)
        self.fight.set_session(session)
        self.show_page("fight")
        if analyze or review is None:
            self.fight.run_review()            # reviewReady -> save_current stores review.json
        else:
            self.fight.flash(f"Opened {meta.title} · click a play or drag the timeline, then Take over")
        if at_ms is not None:
            self.fight.review_goto(at_ms)
        return True

    def fight_with(self, deck, as_ai: bool = False) -> None:
        key = "ai_deck" if as_ai else "player_deck"
        evo_key = "ai_evo" if as_ai else "player_evo"
        cfg = self.fight.session.cfg
        evo = tuple(e for e in getattr(cfg, evo_key) if e in deck)
        self.fight.restart(**{key: tuple(deck), evo_key: evo})
        self.show_page("fight")
        self.fight.flash(("AI now plays" if as_ai else "You now play") + f" {deck_text(deck)} · Space to start")

    def open_deck_builder(self) -> None:
        self.show_page("fight")
        self.fight.edit_decks()

    def closeEvent(self, e) -> None:
        self.fight.cancel_analysis()
        try:
            self.save_current()
        finally:
            super().closeEvent(e)


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("CRbot")
    app.setPalette(dark_palette())          # dialogs (deck builder, quiz) get it too
    app.setStyleSheet(STYLE)
    w = CRbotWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
