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
from PySide6.QtGui import QColor, QDesktopServices, QKeySequence, QPalette, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView, QApplication, QButtonGroup, QCheckBox, QComboBox, QFileDialog, QFormLayout, QFrame,
    QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMainWindow,
    QMessageBox, QPushButton, QStackedWidget, QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

from backend_engine.meta.archetypes import avg_elixir, fmt_avg, pretty
from backend_engine.meta.recommend import coverage, meta_info, recommend
from backend_engine.simulator.match_runner import DEFAULT_DECK
from backend_engine.storage import default_data_dir
from backend_engine.storage.match_store import MatchMeta, MatchStore, MatchStoreError
from backend_engine.storage.profile import Profile, load_profile, save_profile
from frontend_app.qt.app import SPEEDS, MainWindow
from frontend_app.qt.review import WinRateGraph, mmss
from frontend_app.qt.session import PERSONAS, GameSession, ReplayMismatch, SessionConfig

# -- CRbot palette (own identity: charcoal + teal / violet) -------------------
C = {
    "bg": "#15171c", "panel": "#1d2027", "panel2": "#252932", "line": "#2f3440",
    "text": "#e6e8ec", "muted": "#8a909c", "teal": "#2ec4b6", "violet": "#8b5cf6",
    "win": "#3ddc97", "loss": "#ff6b6b", "draw": "#c9ced8",
}
STYLE = f"""
QWidget#ShellRoot, QStackedWidget#Pages, QWidget.page {{ background:{C['bg']}; color:{C['text']}; }}
QWidget#Sidebar {{ background:{C['panel']}; border-right:1px solid {C['line']}; }}
QLabel#Brand {{ color:{C['text']}; font-size:22px; font-weight:800; padding:18px 16px 2px 18px; }}
QLabel#BrandSub {{ color:{C['teal']}; font-size:11px; padding:0 16px 14px 18px; letter-spacing:1px; }}
QPushButton.nav {{ text-align:left; padding:10px 16px; margin:2px 10px; border:none; border-radius:8px;
                   color:{C['muted']}; font-size:14px; background:transparent; }}
QPushButton.nav:hover {{ background:{C['panel2']}; color:{C['text']}; }}
QPushButton.nav:checked {{ background:{C['panel2']}; color:{C['text']}; border-left:3px solid {C['teal']}; }}
QLabel.h1 {{ font-size:22px; font-weight:700; color:{C['text']}; }}
QLabel.sub {{ font-size:12px; color:{C['muted']}; }}
QPushButton.primary {{ background:{C['teal']}; color:#0b1f1d; border:none; border-radius:7px;
                       padding:7px 14px; font-weight:700; }}
QPushButton.primary:disabled {{ background:{C['line']}; color:{C['muted']}; }}
QPushButton.ghost {{ background:transparent; color:{C['text']}; border:1px solid {C['line']}; border-radius:7px;
                     padding:7px 14px; }}
QPushButton.ghost:hover {{ border-color:{C['violet']}; }}
QPushButton.danger {{ background:transparent; color:{C['loss']}; border:1px solid {C['line']}; border-radius:7px;
                      padding:7px 14px; }}
"""

def dark_palette() -> QPalette:
    """CRbot palette for all standard widgets (Fight page, dialogs, lists), so nothing is dark-on-dark."""
    p = QPalette()
    roles = {
        QPalette.Window: C["bg"], QPalette.WindowText: C["text"], QPalette.Base: C["panel"],
        QPalette.AlternateBase: C["panel2"], QPalette.Text: C["text"], QPalette.Button: C["panel2"],
        QPalette.ButtonText: C["text"], QPalette.ToolTipBase: C["panel2"], QPalette.ToolTipText: C["text"],
        QPalette.Highlight: C["teal"], QPalette.HighlightedText: "#0b1f1d", QPalette.PlaceholderText: C["muted"],
        QPalette.Link: C["teal"], QPalette.BrightText: C["loss"],
    }
    for role, col in roles.items():
        p.setColor(role, QColor(col))
    for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
        p.setColor(QPalette.Disabled, role, QColor(C["muted"]))
    return p


NAV = (("fight", "⚔️   Fight"), ("analysis", "📈   Analysis"), ("decks", "🃏   Decks"), ("account", "👤   Account"))
OUTCOME_TEXT = {"win": "Win", "loss": "Loss", "draw": "Draw", None: "Unfinished"}


def _cls(w: QWidget, name: str) -> QWidget:
    """Attach a stylesheet class (QSS `.name`) to a widget."""
    w.setProperty("class", name)
    return w


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


def deck_text(deck, evo=()) -> str:
    return ", ".join(("Evo " if c in evo else "") + pretty(c) for c in deck)


# ---------------------------------------------------------------------------
class AnalysisPage(QWidget):
    """Saved matches (newest first) and a summary of the selected one."""

    def __init__(self, shell: "CRbotWindow"):
        super().__init__()
        self.shell = shell
        self.metas: List[MatchMeta] = []
        root = QVBoxLayout(self)
        root.setContentsMargins(24, 20, 24, 20)
        head = QHBoxLayout()
        head.addWidget(_label("Analysis", "h1"))
        head.addStretch(1)
        self.filter = QComboBox()
        for label, mode in (("All matches", None), ("Practice", "practice"), ("Recordings", "recording"),
                            ("Drills", "drill")):
            self.filter.addItem(label, mode)
        self.filter.currentIndexChanged.connect(lambda _: self.reload())
        head.addWidget(self.filter)
        root.addLayout(head)
        self.stats = _label("", "sub")
        root.addWidget(self.stats)

        body = QHBoxLayout()
        self.list = QListWidget()
        self.list.setMinimumWidth(320)
        self.list.setSpacing(2)
        self.list.setStyleSheet(f"QListWidget{{background:{C['panel']}; border:1px solid {C['line']};"
                                f"border-radius:8px;}}"
                                f"QListWidget::item{{padding:8px;}}"
                                f"QListWidget::item:selected{{background:{C['panel2']};}}")
        self.list.currentRowChanged.connect(self.show_match)
        self.list.itemDoubleClicked.connect(lambda _: self.open_selected())
        body.addWidget(self.list, 2)

        detail = QVBoxLayout()
        self.title = _label("", "h1", wrap=True)
        self.info = _label("", "sub", wrap=True)
        self.info.setTextFormat(Qt.RichText)
        self.graph = WinRateGraph()
        self.graph.setMinimumHeight(180)
        self.grades = QLabel()
        self.grades.setStyleSheet("font-size:13px;")
        btns = QHBoxLayout()
        self.open_btn = _cls(QPushButton("Open review in Fight"), "primary")
        self.open_btn.clicked.connect(self.open_selected)
        self.analyze_btn = _cls(QPushButton("Analyze"), "ghost")
        self.analyze_btn.clicked.connect(lambda: self.open_selected(analyze=True))
        self.delete_btn = _cls(QPushButton("Delete"), "danger")
        self.delete_btn.clicked.connect(self.delete_selected)
        for b in (self.open_btn, self.analyze_btn):
            btns.addWidget(b)
        btns.addStretch(1)
        btns.addWidget(self.delete_btn)
        self.empty = _label("No saved matches yet. Play a match in Fight: it is saved automatically when it "
                            "ends or when you start a new one.", "sub", wrap=True)
        for w in (self.title, self.info, self.graph, self.grades):
            detail.addWidget(w)
        detail.addLayout(btns)
        detail.addWidget(self.empty)
        detail.addStretch(1)
        dw = QWidget()
        dw.setLayout(detail)
        body.addWidget(dw, 3)
        root.addLayout(body, 1)

    # ------------------------------------------------------------------
    def reload(self, select_id: Optional[str] = None) -> None:
        self.metas = self.shell.store.list(self.filter.currentData())
        self.list.blockSignals(True)
        self.list.clear()
        for m in self.metas:
            acc = f"Acc {m.accuracy:.0f}%" if m.accuracy is not None else "Not analyzed"
            line2 = (f"{OUTCOME_TEXT[m.outcome]} {m.crowns[0]}–{m.crowns[1]}  ·  {mmss(m.duration_ms)}  ·  "
                     f"{acc}  ·  {_when(m.created)}")
            it = QListWidgetItem(f"{m.title}\n{line2}")
            it.setData(Qt.UserRole, m.match_id)
            it.setForeground(QColor(C[m.outcome] if m.outcome else C["text"]))
            self.list.addItem(it)
        self.list.blockSignals(False)
        w = sum(m.outcome == "win" for m in self.metas)
        l_ = sum(m.outcome == "loss" for m in self.metas)
        d = sum(m.outcome == "draw" for m in self.metas)
        accs = [m.accuracy for m in self.metas if m.accuracy is not None]
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
        for w in (self.title, self.info, self.graph, self.grades, self.open_btn, self.analyze_btn, self.delete_btn):
            w.setVisible(has)
        self.empty.setVisible(not has)
        if not has:
            return
        self.title.setText(m.title)
        persona = PERSONAS.get(m.persona or "", {}).get("label", m.persona or "—")
        res = OUTCOME_TEXT[m.outcome] + (f" ({m.reason.replace('_', ' ')})" if m.reason else "")
        self.info.setText(
            f"<b>{res}</b> · crowns {m.crowns[0]}–{m.crowns[1]} · {mmss(m.duration_ms)} · level {m.level} · "
            f"{m.mode} · {_when(m.created)}<br>"
            f"<span style='color:{C['teal']}'>You</span>: {deck_text(m.player_deck, m.player_evo)}<br>"
            f"<span style='color:{C['violet']}'>AI</span> ({persona}): {deck_text(m.opp_deck, m.opp_evo)}<br>"
            f"{m.n_plays} plays · card data {m.data_version or 'unknown'} · id {m.match_id}")
        review = None
        try:
            review = self.shell.store.load_review(m.match_id)
        except MatchStoreError:
            pass
        self.graph.set_review(review, m.duration_ms)
        if review is not None:
            c = review.counts()
            self.grades.setText(f"Accuracy {review.accuracy:.0f}%   !! {c['brilliant']}   ★ {c['best']}   "
                                f"Good {c['good']}   ?! {c['inaccuracy']}   ? {c['mistake']}   ?? {c['blunder']}")
        else:
            self.grades.setText("Not analyzed yet. Press Analyze to grade every play.")
        self.analyze_btn.setText("Re-analyze" if review is not None else "Analyze")
        self.open_btn.setEnabled(m.mode == "practice")

    def open_selected(self, analyze: bool = False) -> None:
        m = self.selected()
        if m is not None:
            self.shell.open_match(m.match_id, analyze=analyze)

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
            self.persona.addItem(p["label"], key)
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

        side = QWidget()
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
        self.resize(QSize(1180, 920))
        self.apply_profile(self.profile, persist=False)
        self.show_page("fight")

    # -- navigation ----------------------------------------------------------
    def current_page(self) -> str:
        i = self.pages.currentIndex()
        return next(k for k, v in self.page_index.items() if v == i)

    def show_page(self, key: str) -> None:
        if key != "fight" and self.fight.running:
            self.fight.toggle_run()             # pause: never let a match run unseen
            self.fight.flash("Paused while you were away · Space to resume")
        for sc in self._fight_shortcuts:        # fight hotkeys (Space, 1-4, N, …) only on the Fight page
            sc.setEnabled(key == "fight")
        if key == "analysis":
            self.save_current()
            self.analysis.reload(select_id=self.fight.session.match_id)
        elif key == "account":
            self.account.refresh_freshness()
        self.nav_buttons[key].setChecked(True)
        self.pages.setCurrentIndex(self.page_index[key])

    # -- profile ---------------------------------------------------------------
    def apply_profile(self, p: Profile, persist: bool) -> None:
        self.profile = p
        if persist:
            save_profile(p, self.data_dir)
        self.fight.auto_review = p.auto_review
        self.fight.speed_box.setCurrentIndex(max(0, self.fight.speed_box.findData(p.default_speed)))

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

    def open_match(self, match_id: str, analyze: bool = False) -> bool:
        """Rebuild a saved match in the Fight page, with its review (or analyze it now)."""
        try:
            meta = self.store.load_meta(match_id)
            record = self.store.load_record(match_id)
            review = self.store.load_review(match_id)
        except MatchStoreError as e:
            QMessageBox.warning(self, "Open match", str(e))
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
            self.fight.flash(f"Opened {meta.title} · click a move to see it, or Take over from there")
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
        try:
            self.save_current()
        finally:
            super().closeEvent(e)


def main() -> None:
    app = QApplication.instance() or QApplication(sys.argv)
    app.setStyle("Fusion")
    app.setApplicationName("CRbot")
    app.setPalette(dark_palette())          # dialogs (deck builder, quiz) get it too
    w = CRbotWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
