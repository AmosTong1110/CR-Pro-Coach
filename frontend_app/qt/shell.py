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

# Ensure project root is in sys.path when executed directly as a script
_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from typing import List, Optional

from PySide6.QtCore import QSize, QThread, Qt, QUrl, Signal
from PySide6.QtGui import QColor, QDesktopServices, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QFormLayout, QFrame,
    QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QProgressBar, QPushButton, QScrollArea, QSlider, QStackedWidget, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from backend_engine.brain.coach_analyzer import DIAGNOSTIC_META, MatchReview, diagnose_move
from backend_engine.meta.archetypes import avg_elixir, fmt_avg, pretty
from backend_engine.meta.deck_link import format_copy_deck_link, parse_deck_link
from backend_engine.meta.recommend import coverage, meta_info, recommend
from backend_engine.simulator.engine import ENGINE_VERSION
from backend_engine.simulator.match_runner import DEFAULT_DECK
from backend_engine.storage import default_data_dir
from backend_engine.storage.match_store import MatchMeta, MatchStore, MatchStoreError
from backend_engine.storage.profile import Profile, load_profile, save_profile
from frontend_app.qt.app import SPEEDS, MainWindow, avg_cost, deck_label
from frontend_app.qt.review import LABEL_COLOR, WinRateGraph, grade_chips_html, mmss, tooltip_html
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
        if m.accuracy is not None:
            blunders = m.counts.get("blunder", 0) if m.counts else 0
            mistakes = m.counts.get("mistake", 0) if m.counts else 0
            blunder_note = ""
            if blunders > 0:
                blunder_note = f" ({blunders} blunder{'s' if blunders > 1 else ''})"
            elif mistakes > 0:
                blunder_note = f" ({mistakes} mistake{'s' if mistakes > 1 else ''})"
            acc = f"accuracy {m.accuracy:.0f}%{blunder_note}"
        else:
            acc = "not analyzed"
        sub = QLabel(f"{mmss(m.duration_ms)}  ·  {m.n_plays} plays  ·  {acc}")
        sub.setStyleSheet(f"color:{C['muted']}; font-size:11px; background:transparent;")
        v.addLayout(top)
        v.addWidget(title)
        v.addWidget(sub)
        h.addLayout(v, 1)


class AnalysisCancelled(Exception):
    """Raised when in-place analysis is stopped by the user."""
    pass


class AnalysisWorker(QThread):
    progress = Signal(int, int)               # done, total
    analysisComplete = Signal(object, object) # MatchMeta, MatchReview
    analysisFailed = Signal(str)
    analysisCancelled = Signal()

    def __init__(self, store: MatchStore, match_id: str, db, parent=None):
        super().__init__(parent)
        self.store = store
        self.match_id = match_id
        self.db = db
        self._cancelled = False

    def cancel(self) -> None:
        self._cancelled = True

    def run(self) -> None:
        try:
            meta = self.store.load_meta(self.match_id)
            record = self.store.load_record(self.match_id)
            session = GameSession.rebuild(record, db=self.db)
            if not session.has_human_plays():
                self.analysisFailed.emit("No cards were played in this match, so there is nothing to analyze.")
                return

            def on_progress(done: int, total: int) -> None:
                if self._cancelled:
                    raise AnalysisCancelled()
                self.progress.emit(done, total)

            review = session.analyze(progress=on_progress)
            if self._cancelled:
                self.analysisCancelled.emit()
                return
            updated_meta = self.store.save_review(self.match_id, review)
            self.analysisComplete.emit(updated_meta, review)
        except AnalysisCancelled:
            self.analysisCancelled.emit()
        except Exception as e:
            self.analysisFailed.emit(str(e))


class AnalysisPage(QWidget):
    """Saved matches (newest first) on the left; the selected match on the right: summary, decks, win-rate graph
    with a time slider, and Open review (at the chosen moment) / Analyze / Delete."""

    def __init__(self, shell: "CRbotWindow"):
        super().__init__()
        self.shell = shell
        self.metas: List[MatchMeta] = []
        self.review = None
        self.worker: Optional[AnalysisWorker] = None
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        root.setSpacing(14)

        head = QHBoxLayout()
        head.addWidget(_label("Analysis", "h1"))
        self.drills_btn = _cls(QPushButton("🎯 Tactical Drills…"), "ghost")
        self.drills_btn.setToolTip("Open the curated tactical defense drills & micro-challenges dialog")
        self.drills_btn.clicked.connect(self.shell.open_drills_dialog)
        head.addWidget(self.drills_btn)
        head.addStretch(1)
        self.filter = QComboBox()
        self.filter.setToolTip("Filter by game mode")
        for text, mode in (("All matches", None), ("Practice", "practice"), ("Drills", "drill")):
            self.filter.addItem(text, mode)
        self.filter.currentIndexChanged.connect(lambda _: self.reload())
        head.addWidget(self.filter)

        self.outcome_filter = QComboBox()
        self.outcome_filter.setToolTip("Filter by match outcome")
        for text, out in (("All results", None), ("Wins", "win"), ("Losses", "loss"), ("Draws", "draw")):
            self.outcome_filter.addItem(text, out)
        self.outcome_filter.currentIndexChanged.connect(lambda _: self.reload())
        head.addWidget(self.outcome_filter)

        self.sort_box = QComboBox()
        self.sort_box.setToolTip("Sort matches list")
        for text, key in (
            ("Newest first", "newest"),
            ("Oldest first", "oldest"),
            ("Lowest accuracy", "worst_acc"),
            ("Highest accuracy", "best_acc"),
            ("Most mistakes", "most_mistakes"),
            ("Longest duration", "longest"),
            ("Shortest duration", "shortest"),
        ):
            self.sort_box.addItem(text, key)
        self.sort_box.currentIndexChanged.connect(lambda _: self.reload())
        head.addWidget(self.sort_box)
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
        self.graph.names = self._card_name
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

        self.moments_card = card_frame(spacing=8, margins=14)
        ml = self.moments_card.layout()
        m_head = QHBoxLayout()
        m_head.addWidget(_label("KEY MOMENTS", "section"))
        m_head.addStretch(1)
        self.mistakes_only = QCheckBox("Mistakes only")
        self.mistakes_only.setChecked(True)
        self.mistakes_only.setToolTip("Show only blunders, mistakes and inaccuracies")
        self.mistakes_only.toggled.connect(lambda _: self._populate_moments())
        m_head.addWidget(self.mistakes_only)
        ml.addLayout(m_head)

        self.moments_list = QListWidget()
        self.moments_list.setFixedHeight(120)
        self.moments_list.setStyleSheet(
            f"QListWidget{{background:{C['panel2']}; border:1px solid {C['line']}; border-radius:6px;}} "
            f"QListWidget::item{{padding:4px 8px; font-size:12px;}} "
            f"QListWidget::item:selected{{background:#2b3a4a;}}"
        )
        self.moments_list.currentRowChanged.connect(self._moment_selected)
        self.moments_list.itemDoubleClicked.connect(lambda _: self.open_selected())
        ml.addWidget(self.moments_list)

        mrow = QHBoxLayout()
        self.practice_moment_btn = _cls(QPushButton("🎯 Practice this moment"), "ghost")
        self.practice_moment_btn.setToolTip("Create an isolated sandbox drill from this exact moment and practice defending it cleanly")
        self.practice_moment_btn.setEnabled(False)
        self.practice_moment_btn.clicked.connect(self.practice_selected_moment)
        mrow.addWidget(self.practice_moment_btn)
        mrow.addStretch(1)
        ml.addLayout(mrow)

        self.leak_banner = _label("", wrap=True)
        self.leak_banner.setVisible(False)
        ml.addWidget(self.leak_banner)

        self.moments_empty = _label("Not analyzed yet. Press Analyze to find key moments.", "sub")
        ml.addWidget(self.moments_empty)
        dv.addWidget(self.moments_card)

        btns = QHBoxLayout()
        self.open_btn = _cls(QPushButton("▶  Open review"), "primary")
        self.open_btn.setToolTip("Open this match in Fight with the coach review, at the chosen moment")
        self.open_btn.clicked.connect(lambda: self.open_selected())
        self.analyze_btn = _cls(QPushButton("Analyze"), "ghost")
        self.analyze_btn.clicked.connect(self.start_in_place_analysis)
        self.delete_btn = _cls(QPushButton("Delete"), "danger")
        self.delete_btn.clicked.connect(self.delete_selected)
        btns.addWidget(self.open_btn)
        btns.addWidget(self.analyze_btn)
        btns.addStretch(1)
        btns.addWidget(self.delete_btn)
        dv.addLayout(btns)

        self.busy_box = QWidget()
        bl = QVBoxLayout(self.busy_box)
        bl.setContentsMargins(0, 4, 0, 4)
        bl.setSpacing(4)
        brow = QHBoxLayout()
        self.progress_label = _label("Analyzing… 0%", "sub")
        self.stop_btn = _cls(QPushButton("■  Stop"), "danger")
        self.stop_btn.setToolTip("Stop the running analysis (Esc)")
        self.stop_btn.clicked.connect(self.cancel_analysis)
        brow.addWidget(self.progress_label, 1)
        brow.addWidget(self.stop_btn)
        bl.addLayout(brow)
        self.progress_bar = QProgressBar()
        self.progress_bar.setTextVisible(False)
        bl.addWidget(self.progress_bar)
        self.busy_box.setVisible(False)
        dv.addWidget(self.busy_box)

        QShortcut(QKeySequence("Escape"), self, activated=self._escape)

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
        self.empty_title = _label("No saved matches yet", "h2")
        self.empty.layout().addWidget(self.empty_title)
        self.empty_sub = _label("Play a match in Fight: it is saved automatically when it ends or "
                                "when you start a new one.", "sub", wrap=True)
        self.empty.layout().addWidget(self.empty_sub)
        self.empty_go = _cls(QPushButton("Go to Fight"), "primary")
        self.empty_go.clicked.connect(lambda: self.shell.show_page("fight"))
        self.empty.layout().addWidget(self.empty_go, 0, Qt.AlignLeft)
        self.empty.layout().addStretch(1)

        right = QVBoxLayout()
        right.addWidget(self.detail, 1)
        right.addWidget(self.empty, 1)
        body.addLayout(right, 1)
        root.addLayout(body, 1)

    # ------------------------------------------------------------------
    def _card_name(self, cid: str) -> str:
        db = self.shell.fight.session.db
        rec = db.cards.get(cid) or (getattr(db, "tokens", {}).get(cid))
        return rec.get("name", cid) if rec else cid

    def reload(self, select_id: Optional[str] = None) -> None:
        target_id = select_id
        if target_id is None:
            sel = self.selected()
            if sel is not None:
                target_id = sel.match_id

        raw_metas = self.shell.store.list(self.filter.currentData())
        outcome = self.outcome_filter.currentData()
        if outcome is not None:
            filtered = [m for m in raw_metas if m.outcome == outcome]
        else:
            filtered = list(raw_metas)

        sort_mode = self.sort_box.currentData() if hasattr(self, "sort_box") else "newest"
        if sort_mode == "oldest":
            filtered.sort(key=lambda m: (m.created, m.match_id))
        elif sort_mode == "worst_acc":
            filtered.sort(key=lambda m: (
                m.accuracy is None,
                m.accuracy if m.accuracy is not None else 1e9,
                -m.duration_ms,
            ))
        elif sort_mode == "best_acc":
            filtered.sort(key=lambda m: (
                m.accuracy is None,
                -(m.accuracy if m.accuracy is not None else -1.0),
                -m.duration_ms,
            ))
        elif sort_mode == "most_mistakes":
            def _mistake_key(m: MatchMeta):
                if m.accuracy is None:
                    return (1, 0, 0, 0)
                blunders = m.counts.get("blunder", 0) if m.counts else 0
                mistakes = m.counts.get("mistake", 0) if m.counts else 0
                inacc = m.counts.get("inaccuracy", 0) if m.counts else 0
                return (0, -(blunders + mistakes), -blunders, -inacc)
            filtered.sort(key=_mistake_key)
        elif sort_mode == "longest":
            filtered.sort(key=lambda m: (m.duration_ms, m.created), reverse=True)
        elif sort_mode == "shortest":
            filtered.sort(key=lambda m: (m.duration_ms, m.created))
        else:  # "newest"
            filtered.sort(key=lambda m: (m.created, m.match_id), reverse=True)

        self.metas = filtered
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
        w = sum(m.outcome == "win" for m in raw_metas)
        l_ = sum(m.outcome == "loss" for m in raw_metas)
        d = sum(m.outcome == "draw" for m in raw_metas)
        decided = w + l_ + d
        accs = [m.accuracy for m in raw_metas if m.accuracy is not None]
        self.t_matches.set(str(len(raw_metas)))
        self.t_winrate.set(f"{w / decided:.0%}" if decided else "—",
                           C["win"] if decided and w >= l_ else (C["loss"] if decided else None))
        self.t_record.set(f"{w} · {l_} · {d}")
        self.t_acc.set(f"{sum(accs) / len(accs):.0f}%" if accs else "—")
        filtered_note = f"  ·  showing {len(self.metas)} {outcome}s" if outcome else ""
        self.stats.setText(f"{len(raw_metas)} matches  ·  {w}W {l_}L {d}D"
                           + (f"  ·  average accuracy {sum(accs) / len(accs):.0f}%" if accs else "")
                           + filtered_note)
        mode = self.filter.currentData()
        if not self.metas:
            if outcome is not None:
                self.empty_title.setText(f"No {outcome} matches found")
                self.empty_sub.setText(f"You don't have any recorded {outcome} matches under the current filter.")
                self.empty_go.setVisible(False)
            elif mode == "drill":
                self.empty_title.setText("No drill matches yet")
                self.empty_sub.setText("Scenario drills and targeted practice are coming in a future update.")
                self.empty_go.setVisible(False)
            else:
                self.empty_title.setText("No saved matches yet")
                self.empty_sub.setText("Play a match in Fight: it is saved automatically when it ends or "
                                       "when you start a new one.")
                self.empty_go.setVisible(True)
        row = 0
        if target_id is not None:
            row = next((i for i, m in enumerate(self.metas) if m.match_id == target_id), 0)
        if self.metas:
            self.list.setCurrentRow(row)
        self.show_match(self.list.currentRow())

    def selected(self) -> Optional[MatchMeta]:
        i = self.list.currentRow()
        return self.metas[i] if 0 <= i < len(self.metas) else None

    def show_match(self, i: int) -> None:
        if self.is_analyzing():
            return
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
        self._populate_moments()
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
        has_plays = m.n_plays > 0
        self.open_btn.setEnabled(replayable and not self.is_analyzing())
        self.analyze_btn.setEnabled(replayable and has_plays and not self.is_analyzing())
        self.slider.setEnabled(m.duration_ms > 0)
        if not has_plays:
            self.analyze_btn.setToolTip("No cards were played in this match, so there is nothing to analyze.")
        elif not replayable:
            self.analyze_btn.setToolTip("Matches from an older engine version cannot be re-analyzed.")
        else:
            self.analyze_btn.setToolTip("Grade every play and find best alternatives")
        if not replayable:
            self.open_btn.setToolTip("Matches from an older engine version cannot be opened in Fight.")
        else:
            self.open_btn.setToolTip("Open this match in Fight with the coach review, at the chosen moment")
        self.notice.setText("" if m.engine_version == ENGINE_VERSION else
                            f"Made with an older engine (v{m.engine_version}): summary and saved review only, "
                            "it can't be replayed.")
        self.notice.setVisible(bool(self.notice.text()))
        self.foot.setText(f"card data {m.data_version or 'unknown'} · id {m.match_id}")

    def _populate_moments(self) -> None:
        self.moments_list.blockSignals(True)
        self.moments_list.clear()
        if self.review is None or not self.review.moves:
            self.moments_empty.setText("Not analyzed yet. Press Analyze to find key moments.")
            self.moments_empty.setVisible(True)
            self.moments_list.setVisible(False)
            self.mistakes_only.setVisible(False)
            self.leak_banner.setVisible(False)
            self.practice_moment_btn.setVisible(False)
            self.practice_moment_btn.setEnabled(False)
            self.moments_list.blockSignals(False)
            return

        self.moments_empty.setVisible(False)
        self.moments_list.setVisible(True)
        self.mistakes_only.setVisible(True)
        self.practice_moment_btn.setVisible(True)
        self.practice_moment_btn.setEnabled(False)
        moves = self.review.moves
        if self.mistakes_only.isChecked():
            targets = [mv for mv in moves if mv.label in ("blunder", "mistake", "inaccuracy", "brilliant")]
        else:
            targets = list(moves)

        if not targets:
            it = QListWidgetItem("★ Clean match — no mistakes found!")
            it.setForeground(QColor("#5ad15a"))
            self.moments_list.addItem(it)
            self.leak_banner.setVisible(False)
        else:
            db_cards = self.shell.db_cards()
            for mv in targets:
                col = LABEL_COLOR.get(mv.label, QColor(C["text"]))
                name = self._card_name(mv.actual.card_id)
                swing = f"−{mv.loss * 100:.0f}%" if mv.loss > 0.005 else "±0%"
                sym = mv.symbol or "·"
                d_key = getattr(mv, "diagnostic", None) or diagnose_move(mv, db_cards)
                diag_str = f"[{DIAGNOSTIC_META[d_key]['name']}]" if d_key in DIAGNOSTIC_META else ""
                it = QListWidgetItem(f"{mmss(mv.t_ms)}   {name:<14} {sym} {mv.label_text:<10} {diag_str:<16} {swing}")
                it.setData(Qt.UserRole, mv.t_ms)
                it.setForeground(col)
                it.setToolTip(tooltip_html(mv, self._card_name))
                self.moments_list.addItem(it)

            breakdown = self.review.diagnostics_breakdown(db_cards)
            leak = self.review.primary_leak_summary(db_cards)
            if breakdown:
                chips = []
                for k, count in sorted(breakdown.items(), key=lambda kv: -kv[1]):
                    meta = DIAGNOSTIC_META.get(k, {})
                    chips.append(
                        f"<span style='background:{meta.get('badge_bg', '#333')}; color:{meta.get('color', '#fff')}; "
                        f"padding:2px 6px; border-radius:4px; font-weight:bold; font-size:11px;'>"
                        f"{meta.get('icon', '')} {meta.get('name', k)}: {count}</span>"
                    )
                chips_html = " &nbsp;".join(chips)
                banner_html = (
                    f"<div style='margin-top:6px; padding:6px 10px; background:{C['panel2']}; border:1px solid {C['line']}; "
                    f"border-radius:6px; font-size:11px;'>"
                    f"<b>Mistake breakdown:</b>&nbsp;&nbsp;{chips_html}"
                )
                if leak:
                    banner_html += f"<br><span style='color:#e67e22;'><b>💡 {leak}</b></span>"
                banner_html += "</div>"
                self.leak_banner.setText(banner_html)
                self.leak_banner.setVisible(True)
            else:
                self.leak_banner.setText("")
                self.leak_banner.setVisible(False)
        self.moments_list.blockSignals(False)

    def _moment_selected(self, row: int) -> None:
        if row < 0 or not self.review or not self.review.moves:
            self.practice_moment_btn.setEnabled(False)
            return
        it = self.moments_list.item(row)
        if not it:
            self.practice_moment_btn.setEnabled(False)
            return
        t_ms = it.data(Qt.UserRole)
        self.practice_moment_btn.setEnabled(t_ms is not None)
        if t_ms is not None:
            self.slider.setValue(t_ms)

    def practice_selected_moment(self) -> None:
        from backend_engine.drills.moment_extractor import drill_from_move_review
        from backend_engine.simulator.engine import Match
        m = self.selected()
        if m is None or not self.review or not self.review.moves:
            return
        row = self.moments_list.currentRow()
        if row < 0:
            return
        it = self.moments_list.item(row)
        if not it:
            return
        t_ms = it.data(Qt.UserRole)
        if t_ms is None:
            return
        mv = next((mv for mv in self.review.moves if mv.t_ms == t_ms), None)
        if mv is None:
            return
        try:
            record = self.shell.store.load_record(m.match_id)
            session = GameSession.rebuild(record, db=self.shell.fight.session.db)
            cp = session.pre_play.get(mv.event_id)
            if cp is not None:
                match_before = Match(session.match.cfg, session.db)
                match_before.restore(cp.match)
            else:
                session.seek(max(0, mv.t_ms - 1))
                match_before = session.match
            spec = drill_from_move_review(mv, match_before)
            self.shell.start_drill(spec)
        except Exception as e:
            QMessageBox.warning(self, "Practice Drill", f"Could not create drill from this moment:\n{e}")

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
            name = self._card_name(mv.actual.card_id)
            self.moment.setText(f"{mmss(mv.t_ms)} · your {name} — <span style='color:{col}; font-weight:700'>"
                                f"{mv.label_text} {mv.symbol}</span> (−{mv.loss * 100:.0f}% win chance)")
            self.moments_list.blockSignals(True)
            matched = False
            for r in range(self.moments_list.count()):
                item = self.moments_list.item(r)
                if item and item.data(Qt.UserRole) == mv.t_ms:
                    self.moments_list.setCurrentRow(r)
                    matched = True
                    break
            if not matched:
                self.moments_list.setCurrentRow(-1)
            self.moments_list.blockSignals(False)
            idx = next((k for k, m_ in enumerate(self.review.moves) if m_.t_ms == mv.t_ms), None)
            if idx is not None and self.graph.selected != idx:
                self.graph.selected = idx
                self.graph.update()
        else:
            self.moment.setText(f"<span style='color:{C['muted']}'>Drag the slider or click the graph to choose "
                                "where the review opens. Click a dot to open at that play.</span>")
            self.moments_list.blockSignals(True)
            self.moments_list.setCurrentRow(-1)
            self.moments_list.blockSignals(False)
        if self.graph.selected is not None and self.review is not None and (
                self.graph.selected >= len(self.review.moves) or self.review.moves[self.graph.selected].t_ms != t):
            self.graph.selected = None
            self.graph.update()

    def open_selected(self, analyze: bool = False) -> None:
        m = self.selected()
        if m is None:
            return
        if analyze and m.n_plays == 0:
            QMessageBox.information(self, "Analyze match",
                                    "No cards were played in this match, so there is nothing to analyze.")
            return
        t = self.slider.value()
        self.shell.open_match(m.match_id, analyze=analyze, at_ms=t if t < m.duration_ms else None)

    def delete_selected(self) -> None:
        if self.is_analyzing():
            return
        m = self.selected()
        if m is None:
            return
        if QMessageBox.question(self, "Delete match", f"Delete “{m.title}” ({_when(m.created)})?\n"
                                "This removes its saved review too.") != QMessageBox.Yes:
            return
        self.shell.store.delete(m.match_id)
        if self.shell.fight.session.match_id == m.match_id:
            self.shell.fight.session.match_id = None
            self.shell.fight.session.loaded_from = None
        self.reload()

    def is_analyzing(self) -> bool:
        return self.worker is not None and self.worker.isRunning()

    def wait_analysis(self, timeout_ms: int = 2000) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.worker.wait(timeout_ms)

    def cancel_analysis(self) -> None:
        if self.worker is not None and self.worker.isRunning():
            self.progress_label.setText("Stopping…")
            self.worker.cancel()

    def _escape(self) -> None:
        if self.is_analyzing():
            self.cancel_analysis()

    def start_in_place_analysis(self) -> None:
        m = self.selected()
        if m is None or self.is_analyzing():
            return
        if m.n_plays == 0:
            QMessageBox.information(self, "Analyze match",
                                    "No cards were played in this match, so there is nothing to analyze.")
            return
        if m.engine_version != ENGINE_VERSION:
            QMessageBox.information(self, "Analyze match",
                                    f"This match was played on an older engine (v{m.engine_version}, now v{ENGINE_VERSION}), "
                                    "so it cannot be re-analyzed.")
            return

        self._set_analyzing(True)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_label.setText(f"Analyzing {m.title}… 0%")
        self.notice.setText("")

        worker = AnalysisWorker(self.shell.store, m.match_id, self.shell.fight.session.db, parent=self)
        worker.progress.connect(self._on_worker_progress)
        worker.analysisComplete.connect(self._on_worker_complete)
        worker.analysisFailed.connect(self._on_worker_failed)
        worker.analysisCancelled.connect(self._on_worker_cancelled)
        self.worker = worker
        worker.start()

    def _set_analyzing(self, busy: bool) -> None:
        self.busy_box.setVisible(busy)
        self.analyze_btn.setEnabled(not busy)
        self.open_btn.setEnabled(not busy)
        self.delete_btn.setEnabled(not busy)
        self.filter.setEnabled(not busy)
        self.outcome_filter.setEnabled(not busy)
        self.list.setEnabled(not busy)

    def _on_worker_progress(self, done: int, total: int) -> None:
        total = max(total, 1)
        pct = 100 * done // total
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(done)
        self.progress_label.setText(f"Analyzing plays 8 s ahead… {pct}% ({done}/{total})")

    def _on_worker_complete(self, meta: MatchMeta, review: MatchReview) -> None:
        self._set_analyzing(False)
        self.worker = None
        if self.shell.fight.session.match_id == meta.match_id:
            self.shell.fight.session.review = review
            self.shell.fight.review_panel.set_review(review, meta.duration_ms)
        self.reload(select_id=meta.match_id)
        self.notice.setText(f"✓ Analysis complete: accuracy {review.accuracy:.0f}%")
        self.notice.setStyleSheet(f"color:{C['teal']}; font-size:12px;")
        self.notice.setVisible(True)

    def _on_worker_cancelled(self) -> None:
        self._set_analyzing(False)
        self.worker = None
        self.notice.setText("Analysis stopped. Press Analyze to grade your plays.")
        self.notice.setStyleSheet(f"color:{C['muted']}; font-size:12px;")
        self.notice.setVisible(True)

    def _on_worker_failed(self, err: str) -> None:
        self._set_analyzing(False)
        self.worker = None
        self.notice.setText(f"Analysis failed: {err}")
        self.notice.setStyleSheet(f"color:{C['warn']}; font-size:12px;")
        self.notice.setVisible(True)
        QMessageBox.warning(self, "Analysis failed", str(err))


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
        self.import_btn = _cls(QPushButton("Import link…"), "ghost")
        self.import_btn.setToolTip("Import deck from official clashroyale:// link, RoyaleAPI, or text")
        self.import_btn.clicked.connect(self._import_link)
        self.copy_btn = _cls(QPushButton("Copy link"), "ghost")
        self.copy_btn.setToolTip("Copy official clashroyale:// link for the selected deck")
        self.copy_btn.clicked.connect(self._copy_link)
        self.vs_btn = _cls(QPushButton("Fight against it"), "ghost")
        self.vs_btn.clicked.connect(lambda: self.use(as_ai=True))
        self.use_btn = _cls(QPushButton("Fight with this deck"), "primary")
        self.use_btn.clicked.connect(lambda: self.use(as_ai=False))
        builder = _cls(QPushButton("Deck builder…"), "ghost")
        builder.clicked.connect(self.shell.open_deck_builder)
        for b in (self.import_btn, self.copy_btn, builder, self.vs_btn, self.use_btn):
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
        self.copy_btn.setEnabled(ready)

    def _copy_link(self) -> None:
        r = self._selected_row()
        if r is None or r[4]:
            return
        cards = tuple(r[1])
        try:
            link = format_copy_deck_link(cards, db=self.shell.fight.session.db)
        except Exception as e:
            QMessageBox.warning(self, "Copy Deck Link", f"Could not generate deck link: {e}")
            return
        QApplication.clipboard().setText(link)
        QMessageBox.information(self, "Deck Link Copied", f"Official deck link for {r[0]} copied to clipboard:\n\n{link}")

    def _import_link(self) -> None:
        clip = QApplication.clipboard().text().strip()
        default_val = clip if ("clashroyale" in clip or "royaleapi" in clip or "," in clip or ";" in clip) else ""
        text, ok = QInputDialog.getText(
            self,
            "Import Deck Link",
            "Paste official deck link, RoyaleAPI URL, or card list:",
            text=default_val,
        )
        if not ok or not text.strip():
            return
        db = self.shell.fight.session.db
        parsed = parse_deck_link(text.strip(), db)
        if not parsed.playable:
            msg = parsed.warning or "Deck is incomplete or contains unsimulated cards."
            if parsed.missing:
                msg += f"\n\nMissing / unsimulated: {', '.join(parsed.missing)}"
            QMessageBox.warning(self, "Cannot Import Deck", msg)
            return

        box = QMessageBox(self)
        box.setWindowTitle("Deck Imported")
        avg_str = f"Average elixir: {parsed.avg_elixir:.1f}" if parsed.avg_elixir else ""
        box.setText(f"Successfully parsed '{parsed.archetype or 'Custom Deck'}' (8/8 cards playable)!\n{avg_str}")
        play_btn = box.addButton("Fight with this deck", QMessageBox.AcceptRole)
        vs_btn = box.addButton("Fight against it", QMessageBox.ActionRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.exec()
        clicked = box.clickedButton()
        if clicked == play_btn:
            self.shell.fight_with(parsed.cards, as_ai=False, evo=parsed.evo)
        elif clicked == vs_btn:
            self.shell.fight_with(parsed.cards, as_ai=True, evo=parsed.evo)

    def use(self, as_ai: bool) -> None:
        r = self._selected_row()
        if r is None or r[4]:
            return
        self.shell.fight_with(tuple(r[1]), as_ai=as_ai)


# ---------------------------------------------------------------------------
class AccountPage(QWidget):
    """Local profile, defaults and data freshness."""

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

    def load(self) -> None:
        p = self.shell.profile
        self.name.setText(p.display_name)
        self.tag.setText(f"#{p.player_tag}" if p.player_tag else "")
        self.level.setCurrentIndex(max(0, self.level.findData(p.default_level)))
        self.persona.setCurrentIndex(max(0, self.persona.findData(p.default_persona)))
        self.speed.setCurrentIndex(max(0, self.speed.findData(p.default_speed)))
        self.autosave.setChecked(p.autosave_matches)
        self.autoreview.setChecked(p.auto_review)
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
                    auto_review=self.autoreview.isChecked())
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
            if getattr(self.fight, "showing_me", False):
                self.fight.stop_show_me()
            self.fight.pause_for_away()         # never let a match run unseen
        for sc in self._fight_shortcuts:        # fight hotkeys (Space, 1-4, N, …) only on the Fight page
            sc.setEnabled(key == "fight")
        if key == "analysis":
            self.save_current()
            if not self.analysis.is_analyzing():
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

    def fight_with(self, deck, as_ai: bool = False, evo: Optional[Sequence[str]] = None) -> None:
        key = "ai_deck" if as_ai else "player_deck"
        evo_key = "ai_evo" if as_ai else "player_evo"
        cfg = self.fight.session.cfg
        if evo is None:
            chosen_evo = tuple(e for e in getattr(cfg, evo_key) if e in deck)
        else:
            chosen_evo = tuple(e for e in evo if e.split("#")[0] in deck)
        self.fight.restart(**{key: tuple(deck), evo_key: chosen_evo})
        self.show_page("fight")
        self.fight.flash(("AI now plays" if as_ai else "You now play") + f" {deck_text(deck, chosen_evo)} · Space to start")

    def open_deck_builder(self) -> None:
        self.show_page("fight")
        self.fight.edit_decks()

    def open_drills_dialog(self) -> None:
        from frontend_app.qt.drills_dialog import DrillsDialog
        dlg = DrillsDialog(self, on_start=self.start_drill)
        dlg.exec()

    def start_drill(self, drill) -> None:
        self.show_page("fight")
        self.fight.start_drill(drill)

    def closeEvent(self, e) -> None:
        if getattr(self.fight, "showing_me", False):
            self.fight.stop_show_me()
        self.fight.cancel_analysis()
        if hasattr(self, "analysis") and self.analysis.is_analyzing():
            self.analysis.cancel_analysis()
            self.analysis.wait_analysis(2000)
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
