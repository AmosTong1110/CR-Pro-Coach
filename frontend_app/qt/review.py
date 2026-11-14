"""Post-match coach panel: win-rate graph with move markers, move list, take over."""
from __future__ import annotations

from typing import Callable, List, Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from backend_engine.brain.coach_analyzer import MatchReview, MoveReview

LABEL_COLOR = {
    "brilliant": QColor("#1bc7d6"), "best": QColor("#5ad15a"), "good": QColor("#9ccc65"),
    "inaccuracy": QColor("#f2c94c"), "mistake": QColor("#f2994a"), "blunder": QColor("#eb5757"),
}


def mmss(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60}:{s % 60:02d}"


class WinRateGraph(QWidget):
    moveClicked = Signal(int)   # index into review.moves

    def __init__(self, parent=None):
        super().__init__(parent)
        self.review: Optional[MatchReview] = None
        self.end_ms = 300_000
        self.selected: Optional[int] = None
        self.setMinimumHeight(150)
        self.setMouseTracking(True)

    def set_review(self, review: Optional[MatchReview], end_ms: int) -> None:
        self.review, self.end_ms, self.selected = review, max(end_ms, 1000), None
        self.update()

    def _plot_rect(self) -> QRectF:
        return QRectF(34, 10, self.width() - 44, self.height() - 30)

    def _pt(self, t: int, v: float) -> QPointF:
        r = self._plot_rect()
        return QPointF(r.left() + r.width() * t / self.end_ms, r.top() + r.height() * (1 - v))

    def _curve_at(self, t: int) -> float:
        pts = self.review.curve
        best = min(pts, key=lambda p: abs(p[0] - t))
        return best[1]

    def paintEvent(self, e) -> None:
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#1b1f24"))
        r = self._plot_rect()
        f = QFont()
        f.setPixelSize(10)
        p.setFont(f)
        # phase bands: x2 from 2:00, overtime from 3:00, x3 from 4:00
        for t0, label in ((120_000, "x2"), (180_000, "OT"), (240_000, "x3")):
            if t0 < self.end_ms:
                x = self._pt(t0, 0).x()
                p.setPen(QPen(QColor(255, 255, 255, 40), 1, Qt.DashLine))
                p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
                p.setPen(QColor(200, 150, 255))
                p.drawText(QPointF(x + 3, r.top() + 10), label)
        p.setPen(QColor(150, 150, 150))
        for v, txt in ((1.0, "100%"), (0.5, "50%"), (0.0, "0%")):
            p.drawText(QRectF(0, self._pt(0, v).y() - 6, 30, 12), Qt.AlignRight, txt)
        p.setPen(QPen(QColor(255, 255, 255, 70), 1, Qt.DashLine))
        p.drawLine(self._pt(0, 0.5), self._pt(self.end_ms, 0.5))
        for t in range(0, self.end_ms + 1, 30_000):
            p.setPen(QColor(150, 150, 150))
            p.drawText(QRectF(self._pt(t, 0).x() - 20, r.bottom() + 3, 40, 14), Qt.AlignCenter, mmss(t))
        if not self.review or not self.review.curve:
            p.setPen(QColor(180, 180, 180))
            p.drawText(r, Qt.AlignCenter, "Win-rate graph appears after match ends or clicking 'Analyze'")
            p.end()
            return
        pts = [self._pt(t, v) for t, v in self.review.curve]
        mid = self._pt(0, 0.5).y()
        area = QPainterPath(QPointF(pts[0].x(), mid))
        for q in pts:
            area.lineTo(q)
        area.lineTo(QPointF(pts[-1].x(), mid))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(47, 125, 225, 60))
        p.drawPath(area)
        line = QPainterPath(pts[0])
        for q in pts[1:]:
            line.lineTo(q)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor("#7fb2ff"), 2))
        p.drawPath(line)
        f.setBold(True)
        f.setPixelSize(11)
        p.setFont(f)
        for i, mv in enumerate(self.review.moves):
            c = self._pt(mv.t_ms, self._curve_at(mv.t_ms))
            col = LABEL_COLOR[mv.label]
            big = mv.label in ("brilliant", "inaccuracy", "mistake", "blunder")
            rad = 6 if big else 3.5
            if i == self.selected:
                p.setPen(QPen(QColor("white"), 2))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(c, rad + 4, rad + 4)
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawEllipse(c, rad, rad)
            if mv.symbol and big:
                p.setPen(col)
                p.drawText(QRectF(c.x() - 15, c.y() - 22, 30, 14), Qt.AlignCenter, mv.symbol)
        p.end()

    def mousePressEvent(self, e) -> None:
        if not self.review:
            return
        pos = e.position()
        best, dist = None, 14.0
        for i, mv in enumerate(self.review.moves):
            c = self._pt(mv.t_ms, self._curve_at(mv.t_ms))
            d = ((c.x() - pos.x()) ** 2 + (c.y() - pos.y()) ** 2) ** 0.5
            if d < dist:
                best, dist = i, d
        if best is not None:
            self.moveClicked.emit(best)


class ReviewPanel(QWidget):
    """Bottom panel. Callbacks into the main window keep this widget free of game logic."""
    moveSelected = Signal(int)

    def __init__(self, names: Callable[[str], str], on_analyze, on_take_over, parent=None):
        super().__init__(parent)
        self.names = names
        self.review: Optional[MatchReview] = None
        self.graph = WinRateGraph()
        self.graph.moveClicked.connect(self.select)
        self.list = QListWidget()
        self.list.setStyleSheet("font-size:12px;")
        self.list.currentRowChanged.connect(lambda i: i >= 0 and self.select(i, from_list=True))
        self.summary = QLabel("Post-match Coach Review")
        self.summary.setStyleSheet("font-weight:bold;")
        self.detail = QLabel()
        self.detail.setWordWrap(True)
        self.detail.setStyleSheet("font-size:12px;")
        self.analyze_btn = QPushButton("Analyze (R)")
        self.analyze_btn.clicked.connect(on_analyze)
        self.take_btn = QPushButton("▶ Take Over from Here")
        self.take_btn.setEnabled(False)
        self.take_btn.clicked.connect(on_take_over)
        self.progress = QProgressBar()
        self.progress.setVisible(False)

        side = QVBoxLayout()
        side.addWidget(self.summary)
        side.addWidget(self.list, 1)
        side.addWidget(self.detail)
        row = QHBoxLayout()
        row.addWidget(self.analyze_btn)
        row.addWidget(self.take_btn)
        side.addLayout(row)
        side.addWidget(self.progress)
        lay = QHBoxLayout(self)
        lay.addWidget(self.graph, 3)
        sw = QWidget()
        sw.setLayout(side)
        lay.addWidget(sw, 2)

    def set_review(self, review: Optional[MatchReview], end_ms: int) -> None:
        self.review = review
        self.graph.set_review(review, end_ms)
        self.list.blockSignals(True)
        self.list.clear()
        self.take_btn.setEnabled(False)
        self.detail.setText("")
        if review is None:
            self.summary.setText("Post-match Coach Review")
            self.list.blockSignals(False)
            return
        c = review.counts()
        self.summary.setText(
            f"Accuracy {review.accuracy:.0f}%  ·  !! {c['brilliant']}  ★ {c['best']}  "
            f"?! {c['inaccuracy']}  ? {c['mistake']}  ?? {c['blunder']}")
        for mv in review.moves:
            it = QListWidgetItem(f"{mmss(mv.t_ms)}  {self.names(mv.actual.card_id)}  {mv.symbol or '·'}  "
                                 f"{mv.label_text}  −{mv.loss * 100:.0f}%")
            it.setForeground(LABEL_COLOR[mv.label])
            self.list.addItem(it)
        self.list.blockSignals(False)

    def select(self, i: int, from_list: bool = False) -> None:
        if not self.review or not (0 <= i < len(self.review.moves)):
            return
        mv = self.review.moves[i]
        self.graph.selected = i
        self.graph.update()
        if not from_list:
            self.list.blockSignals(True)
            self.list.setCurrentRow(i)
            self.list.blockSignals(False)
        self.take_btn.setEnabled(True)
        self.detail.setText(describe(mv, self.names))
        self.moveSelected.emit(i)


def describe(mv: MoveReview, names: Callable[[str], str]) -> str:
    a = mv.actual
    lines = [f"<b>{mmss(mv.t_ms)} Played {names(a.card_id)} @ ({a.x:.1f}, {a.y:.1f})</b> — "
             f"<span style='color:{LABEL_COLOR[mv.label].name()}'>{mv.label_text} {mv.symbol}</span>",
             f"Win prob: Before play {mv.v_before:.0%} → Actual {mv.v_actual:.0%} · Best {mv.v_best:.0%}"]
    if mv.best.is_wait:
        lines.append("<span style='color:#5ad15a'>Best: Wait — do not play here (preserve elixir/cycle)</span>")
    elif mv.loss > 0.01:
        b = mv.best
        lines.append(f"<span style='color:#5ad15a'>Recommended: {names(b.card_id)} @ ({b.x:.1f}, {b.y:.1f})</span> "
                     f"(green ghost) · Your placement in red")
    if mv.loss > 0.01 and not mv.best.is_wait and mv.v_wait > mv.v_actual + 0.03:
        lines.append(f"Even waiting would be better ({mv.v_wait:.0%})")
    lines.append(f"<span style='color:#888'>Evaluated {mv.n_candidates} candidates rolled out 8s</span>")
    return "<br>".join(lines)
