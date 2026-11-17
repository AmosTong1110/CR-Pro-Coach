"""Post-match coach review widgets.

* WinRateGraph: win-rate curve over match time, phase lines (x2 / OT / x3), one dot per graded play and a
  time cursor. Click a dot to select that play, click or drag anywhere else to jump to that moment.
* ReviewPanel: the coach column of the Fight page in review mode (accuracy, graded plays, explanation,
  Take over / Analyze, progress with a Stop button, Exit review). Its `graph` is placed by the owner
  (under the board on the Fight page).
"""
from __future__ import annotations

from typing import Callable, Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QProgressBar, QPushButton, QSizePolicy, QToolTip, QVBoxLayout,
    QWidget,
)

from backend_engine.brain.coach_analyzer import MatchReview, MoveReview
from frontend_app.qt.theme import C, cls

# Grade colours (same convention as chess review tools)
LABEL_COLOR = {
    "brilliant": QColor("#1bc7d6"),
    "best": QColor("#5ad15a"),
    "good": QColor("#9ccc65"),
    "inaccuracy": QColor("#f2c94c"),
    "mistake": QColor("#f2994a"),
    "blunder": QColor("#eb5757"),
}
GRADE_CHIPS = (("brilliant", "!!"), ("best", "★"), ("good", "Good"), ("inaccuracy", "?!"), ("mistake", "?"),
               ("blunder", "??"))
# drawing / labelling priority on the graph (worst on top, labelled first)
SEVERITY = {"blunder": 5, "mistake": 4, "inaccuracy": 3, "brilliant": 2, "best": 1, "good": 0}
LABELLED = ("blunder", "mistake", "inaccuracy", "brilliant")


def mmss(ms: int) -> str:
    s = max(0, ms) // 1000
    return f"{s // 60}:{s % 60:02d}"


def grade_chips_html(review: MatchReview) -> str:
    """Grade counts as coloured text (used by the Fight review panel and the Analysis page)."""
    c = review.counts()
    return "&nbsp;&nbsp;&nbsp;".join(
        f"<span style='color:{LABEL_COLOR[k].name()}; font-weight:700'>{sym}</span>&nbsp;{c.get(k, 0)}"
        for k, sym in GRADE_CHIPS)


class WinRateGraph(QWidget):
    """Win-rate curve (0-100%) over match time with graded plays and a time cursor."""
    moveClicked = Signal(int)   # index into review.moves
    timeClicked = Signal(int)   # sim ms under the mouse (click or drag away from a dot)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.review: Optional[MatchReview] = None
        self.end_ms = 300_000
        self.selected: Optional[int] = None
        self.cursor_ms: Optional[int] = None
        self.empty_text = "Press Analyze to grade your plays and draw the win-rate curve"
        self.names: Optional[Callable[[str], str]] = None   # card id -> display name (set by ReviewPanel)
        self.setMinimumHeight(120)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setCursor(Qt.PointingHandCursor)

    def set_review(self, review: Optional[MatchReview], end_ms: int) -> None:
        self.review, self.end_ms, self.selected = review, max(end_ms, 1000), None
        self.update()

    def set_cursor(self, t_ms: Optional[int]) -> None:
        self.cursor_ms = t_ms
        self.update()

    def _plot_rect(self) -> QRectF:
        return QRectF(38, 12, self.width() - 50, self.height() - 32)

    def _pt(self, t: int, v: float) -> QPointF:
        r = self._plot_rect()
        return QPointF(r.left() + r.width() * t / self.end_ms, r.top() + r.height() * (1 - v))

    def _t_at(self, x: float) -> int:
        r = self._plot_rect()
        frac = (x - r.left()) / max(r.width(), 1.0)
        return int(max(0.0, min(1.0, frac)) * self.end_ms)

    def _curve_at(self, t: int) -> float:
        return min(self.review.curve, key=lambda p: abs(p[0] - t))[1]

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(C["panel"]))
        p.drawRoundedRect(QRectF(self.rect()).adjusted(0.5, 0.5, -0.5, -0.5), 10, 10)
        r = self._plot_rect()
        f = QFont()
        f.setPixelSize(10)
        p.setFont(f)
        for t0, text in ((120_000, "x2"), (180_000, "OT"), (240_000, "x3")):
            if t0 < self.end_ms:
                x = self._pt(t0, 0).x()
                p.setPen(QPen(QColor(255, 255, 255, 40), 1, Qt.DashLine))
                p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
                p.setPen(QColor(C["violet"]))
                p.drawText(QPointF(x + 3, r.top() + 10), text)
        p.setPen(QColor(C["muted"]))
        for v, txt in ((1.0, "100%"), (0.5, "50%"), (0.0, "0%")):
            p.drawText(QRectF(0, self._pt(0, v).y() - 6, 33, 12), Qt.AlignRight, txt)
        p.setPen(QPen(QColor(255, 255, 255, 60), 1, Qt.DashLine))
        p.drawLine(self._pt(0, 0.5), self._pt(self.end_ms, 0.5))
        step = 30_000 if self.end_ms <= 200_000 else 60_000
        for t in range(0, self.end_ms + 1, step):
            p.setPen(QColor(C["muted"]))
            p.drawText(QRectF(self._pt(t, 0).x() - 20, r.bottom() + 4, 40, 14), Qt.AlignCenter, mmss(t))
        if self.review and self.review.curve:
            self._paint_curve(p, f)
        else:
            p.setPen(QColor(C["muted"]))
            p.drawText(r, Qt.AlignCenter | Qt.TextWordWrap, self.empty_text)
        if self.cursor_ms is not None:
            x = self._pt(min(self.cursor_ms, self.end_ms), 0).x()
            p.setPen(QPen(QColor(C["teal"]), 2))
            p.drawLine(QPointF(x, r.top() - 4), QPointF(x, r.bottom()))
            p.setBrush(QColor(C["teal"]))
            p.setPen(Qt.NoPen)
            p.drawEllipse(QPointF(x, r.top() - 4), 4, 4)
        p.end()

    def _paint_curve(self, p: QPainter, f: QFont) -> None:
        pts = [self._pt(t, v) for t, v in self.review.curve]
        mid = self._pt(0, 0.5).y()
        area = QPainterPath(QPointF(pts[0].x(), mid))
        for q in pts:
            area.lineTo(q)
        area.lineTo(QPointF(pts[-1].x(), mid))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(46, 196, 182, 50))
        p.drawPath(area)
        line = QPainterPath(pts[0])
        for q in pts[1:]:
            line.lineTo(q)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor("#7fd8cf"), 2))
        p.drawPath(line)
        f.setBold(True)
        f.setPixelSize(11)
        p.setFont(f)
        moves = self.review.moves
        order = sorted(range(len(moves)), key=lambda i: (SEVERITY[moves[i].label], i == self.selected))
        for i in order:                                   # dots: worst grades drawn last (on top)
            mv = moves[i]
            c = self._pt(mv.t_ms, self._curve_at(mv.t_ms))
            rad = 6 if mv.label in LABELLED else 3.5
            if i == self.selected:
                p.setPen(QPen(QColor("white"), 2))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(c, rad + 4, rad + 4)
            p.setPen(QPen(QColor(C["panel"]), 1))
            p.setBrush(LABEL_COLOR[mv.label])
            p.drawEllipse(c, rad, rad)
        for i, rect in self.label_rects():                 # symbols: stacked so none is dropped
            p.setPen(LABEL_COLOR[moves[i].label])
            p.drawText(rect, Qt.AlignCenter, moves[i].symbol)

    def label_rects(self):
        """(move index, rect) for every graded symbol (?!, ?, ??, !!): placed above its dot; when that spot is
        taken it shifts sideways, stacks higher or goes below, always inside the widget (worst grades first)."""
        if not self.review or not self.review.curve:
            return []
        moves = self.review.moves
        bounds = QRectF(self.rect()).adjusted(1, 1, -1, -1)
        placed = []
        order = sorted((i for i, mv in enumerate(moves) if mv.label in LABELLED and mv.symbol),
                       key=lambda i: (-(i == self.selected), -SEVERITY[moves[i].label], moves[i].t_ms))
        for i in order:
            c = self._pt(moves[i].t_ms, self._curve_at(moves[i].t_ms))
            spots = [(dx, dy) for dy in (-22, -36, -50, 8, 22, 36) for dx in (0, -12, 12, -24, 24)]
            for dx, dy in spots:
                r = QRectF(c.x() - 11 + dx, c.y() + dy, 22, 14)
                if r.top() < bounds.top():                 # near 100%: slide down instead of leaving the widget
                    r.moveTop(bounds.top())
                if r.bottom() > bounds.bottom():
                    r.moveBottom(bounds.bottom())
                if r.left() < bounds.left():
                    r.moveLeft(bounds.left())
                if r.right() > bounds.right():
                    r.moveRight(bounds.right())
                if r.contains(c) or any(r.intersects(q) for _, q in placed):
                    continue
                placed.append((i, r))
                break
        return placed

    def _move_near(self, pos) -> Optional[int]:
        if not self.review or not self.review.curve:
            return None
        best, dist = None, 14.0
        for i, mv in enumerate(self.review.moves):
            c = self._pt(mv.t_ms, self._curve_at(mv.t_ms))
            d = ((c.x() - pos.x()) ** 2 + (c.y() - pos.y()) ** 2) ** 0.5
            if d < dist:
                best, dist = i, d
        return best

    def mousePressEvent(self, e) -> None:
        if e.button() != Qt.LeftButton:
            return
        i = self._move_near(e.position())
        if i is not None:
            self.moveClicked.emit(i)
        else:
            self.timeClicked.emit(self._t_at(e.position().x()))

    def mouseMoveEvent(self, e) -> None:
        if e.buttons() & Qt.LeftButton:
            self.timeClicked.emit(self._t_at(e.position().x()))
            return
        i = self._move_near(e.position())
        if i is None:                                       # also the symbol above a dot
            i = next((k for k, r in self.label_rects() if r.contains(e.position())), None)
        if i is not None and self.names is not None:
            QToolTip.showText(e.globalPosition().toPoint(), tooltip_html(self.review.moves[i], self.names), self)
        else:
            QToolTip.hideText()

    def leaveEvent(self, e) -> None:
        QToolTip.hideText()


class ReviewPanel(QWidget):
    """Coach column for review mode. Logic stays in the owner, wired through the callbacks."""
    moveSelected = Signal(int)

    def __init__(self, names: Callable[[str], str], on_analyze, on_take_over, on_exit=None, on_cancel=None,
                 on_show_me=None, parent=None):
        super().__init__(parent)
        self.names = names
        self.review: Optional[MatchReview] = None
        self.selected_move_index: Optional[int] = None
        self.diverged = False
        self.graph = WinRateGraph()
        self.graph.names = names
        self.graph.moveClicked.connect(self.select)

        head = QHBoxLayout()
        title = QLabel("Coach review")
        cls(title, "h2")
        head.addWidget(title)
        head.addStretch(1)
        self.exit_btn = cls(QPushButton("✕  Exit review"), "ghost")
        self.exit_btn.setToolTip("Stop reviewing and go back to the match result")
        if on_exit:
            self.exit_btn.clicked.connect(on_exit)
        head.addWidget(self.exit_btn)

        self.accuracy = QLabel("—")
        self.accuracy.setStyleSheet("font-size:30px; font-weight:800;")
        acc_cap = QLabel("accuracy")
        cls(acc_cap, "sub")
        self.summary = QLabel("Not analyzed yet")
        self.summary.setTextFormat(Qt.RichText)
        self.summary.setWordWrap(True)
        self.summary.setStyleSheet("font-size:12px;")
        acc_box = QVBoxLayout()
        acc_box.setSpacing(0)
        acc_box.addWidget(self.accuracy)
        acc_box.addWidget(acc_cap)
        top = QHBoxLayout()
        top.addLayout(acc_box)
        top.addSpacing(12)
        top.addWidget(self.summary, 1)

        moves_cap = QLabel("YOUR PLAYS")
        cls(moves_cap, "section")
        self.list = QListWidget()
        self.list.setStyleSheet("font-size:12px;")
        self.list.setMinimumHeight(90)
        self.list.currentRowChanged.connect(lambda i: i >= 0 and self.select(i, from_list=True))

        self.detail = QLabel()
        self.detail.setTextFormat(Qt.RichText)
        self.detail.setWordWrap(True)
        self.detail.setAlignment(Qt.AlignTop | Qt.AlignLeft)
        self.detail.setStyleSheet(f"font-size:12px; background:{C['panel']}; border:1px solid {C['line']};"
                                  "border-radius:8px; padding:8px;")
        self.detail.setMinimumHeight(96)
        self.detail.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)

        self.take_btn = cls(QPushButton("▶  Take over"), "primary")
        self.take_btn.setToolTip("Play on from the moment shown on the board (3-2-1 countdown)")
        self.take_btn.setEnabled(False)
        self.take_btn.clicked.connect(on_take_over)
        self.show_me_btn = cls(QPushButton("👁  Show me (S)"), "ghost")
        self.show_me_btn.setToolTip("Simulate and watch how the recommended move plays out for 8 seconds (S)")
        self.show_me_btn.setEnabled(False)
        if on_show_me:
            self.show_me_btn.clicked.connect(on_show_me)
        self.analyze_btn = cls(QPushButton("Analyze (R)"), "ghost")
        self.analyze_btn.clicked.connect(on_analyze)
        row = QHBoxLayout()
        row.addWidget(self.take_btn, 1)
        row.addWidget(self.show_me_btn)
        row.addWidget(self.analyze_btn)

        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress_label = QLabel("Analyzing…")
        cls(self.progress_label, "sub")
        self.cancel_btn = cls(QPushButton("■  Stop"), "danger")
        self.cancel_btn.setToolTip("Stop the analysis (Esc)")
        if on_cancel:
            self.cancel_btn.clicked.connect(on_cancel)
        self.busy_box = QWidget()
        bl = QVBoxLayout(self.busy_box)
        bl.setContentsMargins(0, 0, 0, 0)
        bl.setSpacing(4)
        brow = QHBoxLayout()
        brow.addWidget(self.progress_label, 1)
        brow.addWidget(self.cancel_btn)
        bl.addLayout(brow)
        bl.addWidget(self.progress)
        self.busy_box.setVisible(False)

        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(8)
        lay.addLayout(head)
        lay.addLayout(top)
        lay.addWidget(moves_cap)
        lay.addWidget(self.list, 1)
        lay.addWidget(self.detail)
        lay.addLayout(row)
        lay.addWidget(self.busy_box)
        self._placeholder()

    # ------------------------------------------------------------------
    def _placeholder(self) -> None:
        self.detail.setText(f"<span style='color:{C['muted']}'>Click a dot on the graph or a play in the list to "
                            "see what the coach suggests. Drag the timeline under the board to jump to any "
                            "moment.</span>")

    def set_busy(self, busy: bool) -> None:
        """Analysis running: show progress + Stop, lock everything else in the panel."""
        self.busy_box.setVisible(busy)
        if busy:
            self.progress.setMaximum(1)
            self.progress.setValue(0)
        for w in (self.list, self.take_btn, self.show_me_btn, self.analyze_btn, self.exit_btn):
            w.setEnabled(not busy)

    def set_review(self, review: Optional[MatchReview], end_ms: int) -> None:
        self.review = review
        self.selected_move_index = None
        self.diverged = False
        self.graph.set_review(review, end_ms)
        self.list.blockSignals(True)
        self.list.clear()
        self.show_me_btn.setEnabled(False)
        self.show_me_btn.setText("👁  Show me (S)")
        self.show_me_btn.setStyleSheet("")
        self._placeholder()
        if review is None:
            self.accuracy.setText("—")
            self.summary.setText(f"<span style='color:{C['muted']}'>Not analyzed yet. Press Analyze to grade "
                                 "every play.</span>")
            self.analyze_btn.setText("Analyze (R)")
            self.list.blockSignals(False)
            return
        self.accuracy.setText(f"{review.accuracy:.0f}%")
        self.summary.setText(grade_chips_html(review))
        self.analyze_btn.setText("Re-analyze (R)")
        for mv in review.moves:
            it = QListWidgetItem(f"{mmss(mv.t_ms)}   {self.names(mv.actual.card_id)}   {mv.symbol or '·'} "
                                 f"{mv.label_text}   −{mv.loss * 100:.0f}%")
            it.setForeground(LABEL_COLOR[mv.label])
            it.setToolTip(tooltip_html(mv, self.names))
            self.list.addItem(it)
        self.list.blockSignals(False)

    def mark_diverged(self) -> None:
        """You played a card on a reviewed timeline: grades no longer describe this match."""
        if self.review is not None and not self.diverged:
            self.diverged = True
            self.summary.setText(self.summary.text() + f"<br><span style='color:{C['warn']}'>Diverged from the "
                                 "reviewed match · Re-analyze after it ends</span>")

    def select(self, i: int, from_list: bool = False) -> None:
        if not self.review or not (0 <= i < len(self.review.moves)):
            return
        self.selected_move_index = i
        mv = self.review.moves[i]
        self.graph.selected = i
        self.graph.update()
        if not from_list:
            self.list.blockSignals(True)
            self.list.setCurrentRow(i)
            self.list.blockSignals(False)
        self.detail.setText(describe(mv, self.names))
        self.show_me_btn.setEnabled(True)
        rec_name = "Wait" if mv.best.is_wait else self.names(mv.best.card_id)
        self.show_me_btn.setToolTip(f"Simulate 8s ahead with recommended play: {rec_name} (S)")
        self.moveSelected.emit(i)

    def clear_selection(self) -> None:
        self.selected_move_index = None
        self.show_me_btn.setEnabled(False)
        self.graph.selected = None
        self.graph.update()
        self.list.blockSignals(True)
        self.list.clearSelection()
        self.list.setCurrentRow(-1)
        self.list.blockSignals(False)
        self._placeholder()

    def selected_move(self) -> Optional[MoveReview]:
        if self.review and self.selected_move_index is not None and 0 <= self.selected_move_index < len(self.review.moves):
            return self.review.moves[self.selected_move_index]
        return None


def where(x: float, y: float) -> str:
    """Plain-language spot on the board (your side at the bottom, like the real game)."""
    lane = "left lane" if x < 7 else ("right lane" if x > 11 else "centre")
    if y >= 17:
        depth = "deep on their side" if y >= 24 else "on their side, past the river"
    elif y >= 15:
        depth = "on the river"
    elif y >= 13:
        depth = "at the bridge"
    elif y >= 9:
        depth = "mid-way on your side"
    elif y >= 5:
        depth = "by your Princess Towers"
    else:
        depth = "at the back, by your King Tower"
    return f"{lane}, {depth}"


def _shift(a, b) -> str:
    """How your spot differs from the recommended one (same card)."""
    parts = []
    if abs(a.y - b.y) >= 1.5:
        parts.append(f"{abs(a.y - b.y):.0f} tiles too far {'forward' if a.y > b.y else 'back'}")
    if (a.x < 9) != (b.x < 9) and abs(a.x - b.x) >= 3:
        parts.append("in the wrong lane")
    elif abs(a.x - b.x) >= 1.5:
        parts.append(f"{abs(a.x - b.x):.0f} tiles too far {'left' if a.x < b.x else 'right'}")
    return " and ".join(parts) or "slightly off the best tile"


def explain(mv: MoveReview, names: Callable[[str], str]) -> tuple:
    """(mistake, alternative) in plain English for one graded play."""
    a, b = mv.actual, mv.best
    card = names(a.card_id)
    spot = f"{card} at ({a.x:.1f}, {a.y:.1f}), {where(a.x, a.y)}"
    swing = f"win chance {mv.v_before:.0%} → {mv.v_actual:.0%}"
    if mv.loss <= 0.01:
        mistake = (f"None. {spot} was {'the best play found' if mv.label in ('best', 'brilliant') else 'as good as the best option'}"
                   f" ({swing}).")
    elif b.is_wait:
        mistake = (f"Played too early: {spot} spent elixir when holding it was worth more ({swing}; waiting "
                   f"{mv.v_wait:.0%}). It left you short for their next push and gave nothing back.")
    elif b.card_id == a.card_id:
        mistake = (f"Right card, wrong spot: {spot} is {_shift(a, b)}, costing {mv.loss:.0%} win chance ({swing}).")
    else:
        mistake = (f"Wrong card for this moment: {spot} cost {mv.loss:.0%} win chance ({swing}); "
                   f"{names(b.card_id)} handles this situation better.")
    if mv.loss > 0.01 and not b.is_wait and mv.v_wait > mv.v_actual + 0.03:
        mistake += f" Even waiting would have been better ({mv.v_wait:.0%})."

    others = [(c, v) for c, v in mv.alternatives if not c.is_wait and c != b and c != a]
    if b.is_wait:
        alt = f"Wait: keep your elixir and cycle, then answer their next play (win chance {mv.v_wait:.0%})."
        cards = [(c, v) for c, v in mv.alternatives if not c.is_wait]
        if cards:
            c, v = cards[0]
            alt += f" If you must play: {names(c.card_id)} at ({c.x:.1f}, {c.y:.1f}), {where(c.x, c.y)} ({v:.0%})."
    elif mv.loss <= 0.01:
        alt = "Your play was already the best option."
        if others:
            c, v = others[0]
            alt += f" Also fine: {names(c.card_id)} at ({c.x:.1f}, {c.y:.1f}), {where(c.x, c.y)} ({v:.0%})."
    else:
        alt = (f"{names(b.card_id)} at ({b.x:.1f}, {b.y:.1f}), {where(b.x, b.y)} "
               f"(win chance {mv.v_best:.0%}, +{mv.loss:.0%} over your play).")
        if others:
            c, v = others[0]
            alt += f" Next best: {names(c.card_id)} at ({c.x:.1f}, {c.y:.1f}) ({v:.0%})."
    return mistake, alt


def tooltip_html(mv: MoveReview, names: Callable[[str], str]) -> str:
    """Hover text for a graded play (graph dot / list row): Mistake + Alternative."""
    mistake, alt = explain(mv, names)
    col = LABEL_COLOR[mv.label].name()
    return (f"<div style='max-width:340px'><b>{mmss(mv.t_ms)} · {names(mv.actual.card_id)}</b> "
            f"<span style='color:{col}'>{mv.symbol} {mv.label_text}</span> (−{mv.loss:.0%})<br>"
            f"<b>Mistake:</b> {mistake}<br><b style='color:#5ad15a'>Alternative:</b> {alt}<br>"
            f"<span style='color:#8b93a1'>{mv.n_candidates} options simulated 8 s ahead</span></div>")


def describe(mv: MoveReview, names: Callable[[str], str]) -> str:
    """Rich-text explanation of one graded play: win chance before/after, the recommendation, and why."""
    a = mv.actual
    mistake, alt = explain(mv, names)
    lines = [f"<b>{mmss(mv.t_ms)} · {names(a.card_id)} at ({a.x:.1f}, {a.y:.1f})</b> — "
             f"<span style='color:{LABEL_COLOR[mv.label].name()}'>{mv.label_text} {mv.symbol}</span>",
             f"<b>Mistake:</b> {mistake}",
             f"<b style='color:#5ad15a'>Alternative:</b> {alt}"]
    if not mv.best.is_wait and mv.loss > 0.01:
        lines.append(f"<span style='color:{C['muted']}'>On the board: yours in red, recommended in green</span>")
    lines.append(f"<span style='color:{C['muted']}'>{mv.n_candidates} options simulated 8 s ahead</span>")
    return "<br>".join(lines)
