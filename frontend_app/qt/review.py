"""赛后教练复盘面板 (Post-match Coach Panel)

包含：
1. 胜率走势折线图 (WinRateGraph)：显示随对局时间波动的胜率、关键阶段分界线 (双倍/加时/三倍) 及各手出牌评级标记 (!!, ★, ?!, ?? 等)。
2. 复盘列表与控制面板 (ReviewPanel)：展示综合对局准确率 (Accuracy %)、各评级统计、详细落子损失分析，以及 "从此处接管 (Take Over from Here)" 分支对战功能。
"""
from __future__ import annotations

from typing import Callable, List, Optional

from PySide6.QtCore import QPointF, QRectF, Qt, Signal
from PySide6.QtGui import QColor, QFont, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import (
    QHBoxLayout, QLabel, QListWidget, QListWidgetItem, QProgressBar, QPushButton, QVBoxLayout, QWidget,
)

from backend_engine.brain.coach_analyzer import MatchReview, MoveReview

# 评级颜色映射 (与主流棋类复盘软件配色规范一致)
LABEL_COLOR = {
    "brilliant": QColor("#1bc7d6"),  # 妙手/神来之笔：青色
    "best": QColor("#5ad15a"),       # 最佳手：绿色
    "good": QColor("#9ccc65"),       # 好棋：淡绿
    "inaccuracy": QColor("#f2c94c"), # 不精准：黄色
    "mistake": QColor("#f2994a"),    # 失误：橙色
    "blunder": QColor("#eb5757"),    # 严重失误/漏着：红色
}


def mmss(ms: int) -> str:
    """毫秒时间戳转换为 分:秒 格式字符串"""
    s = ms // 1000
    return f"{s // 60}:{s % 60:02d}"


class WinRateGraph(QWidget):
    """胜率走势图控件 (Win-rate Graph)
    
    绘制一条 0% ~ 100% 的胜率平滑走势曲线。
    横轴为比赛时间 (ms)，并在 2:00 (双倍圣水)、3:00 (加时赛)、4:00 (三倍圣水) 处标记虚线阶段指示。
    每个玩家落子事件按评级绘制不同大小和颜色的圆点，并支持鼠标点击命中检测，触发 moveClicked 信号。
    """
    moveClicked = Signal(int)   # 点击某一手棋时发射该手在 review.moves 中的索引

    def __init__(self, parent=None):
        super().__init__(parent)
        self.review: Optional[MatchReview] = None
        self.end_ms = 300_000
        self.selected: Optional[int] = None
        self.setMinimumHeight(150)
        self.setMouseTracking(True)

    def set_review(self, review: Optional[MatchReview], end_ms: int) -> None:
        """设置复盘数据并刷新图表"""
        self.review, self.end_ms, self.selected = review, max(end_ms, 1000), None
        self.update()

    def _plot_rect(self) -> QRectF:
        """图表主体绘制区域 (留出左侧百分比刻度与底部时间轴边距)"""
        return QRectF(34, 10, self.width() - 44, self.height() - 30)

    def _pt(self, t: int, v: float) -> QPointF:
        """将 (时间 t ms, 胜率 v [0, 1]) 映射到图表像素坐标"""
        r = self._plot_rect()
        return QPointF(r.left() + r.width() * t / self.end_ms, r.top() + r.height() * (1 - v))

    def _curve_at(self, t: int) -> float:
        """在离散采样曲线中查找距离时间 t 最近的点之胜率估值"""
        pts = self.review.curve
        best = min(pts, key=lambda p: abs(p[0] - t))
        return best[1]

    def paintEvent(self, e) -> None:
        """胜率图绘制流程：阶段背景虚线、刻度、中线、填充多边形、走势折线、落子评级标记"""
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor("#1b1f24"))
        r = self._plot_rect()
        f = QFont()
        f.setPixelSize(10)
        p.setFont(f)
        # 1. 绘制对局阶段分界线：2:00 进入双倍，3:00 进入加时，4:00 进入三倍
        for t0, label in ((120_000, "x2"), (180_000, "OT"), (240_000, "x3")):
            if t0 < self.end_ms:
                x = self._pt(t0, 0).x()
                p.setPen(QPen(QColor(255, 255, 255, 40), 1, Qt.DashLine))
                p.drawLine(QPointF(x, r.top()), QPointF(x, r.bottom()))
                p.setPen(QColor(200, 150, 255))
                p.drawText(QPointF(x + 3, r.top() + 10), label)
        # 2. 纵轴百分比刻度 (0%, 50%, 100%)
        p.setPen(QColor(150, 150, 150))
        for v, txt in ((1.0, "100%"), (0.5, "50%"), (0.0, "0%")):
            p.drawText(QRectF(0, self._pt(0, v).y() - 6, 30, 12), Qt.AlignRight, txt)
        # 50% 平衡基准虚线
        p.setPen(QPen(QColor(255, 255, 255, 70), 1, Qt.DashLine))
        p.drawLine(self._pt(0, 0.5), self._pt(self.end_ms, 0.5))
        # 3. 横轴时间刻度 (每 30 秒标刻度)
        for t in range(0, self.end_ms + 1, 30_000):
            p.setPen(QColor(150, 150, 150))
            p.drawText(QRectF(self._pt(t, 0).x() - 20, r.bottom() + 3, 40, 14), Qt.AlignCenter, mmss(t))
        if not self.review or not self.review.curve:
            p.setPen(QColor(180, 180, 180))
            p.drawText(r, Qt.AlignCenter, "Win-rate graph appears after match ends or clicking 'Analyze'")
            p.end()
            return
        # 4. 填充胜率曲线与 50% 轴之间的半透明渐变面积
        pts = [self._pt(t, v) for t, v in self.review.curve]
        mid = self._pt(0, 0.5).y()
        area = QPainterPath(QPointF(pts[0].x(), mid))
        for q in pts:
            area.lineTo(q)
        area.lineTo(QPointF(pts[-1].x(), mid))
        p.setPen(Qt.NoPen)
        p.setBrush(QColor(47, 125, 225, 60))
        p.drawPath(area)
        # 5. 绘制平滑胜率轮廓折线
        line = QPainterPath(pts[0])
        for q in pts[1:]:
            line.lineTo(q)
        p.setBrush(Qt.NoBrush)
        p.setPen(QPen(QColor("#7fb2ff"), 2))
        p.drawPath(line)
        # 6. 在曲线上标出出牌位置的点与符号 (如 !!、?、?? 等)
        f.setBold(True)
        f.setPixelSize(11)
        p.setFont(f)
        for i, mv in enumerate(self.review.moves):
            c = self._pt(mv.t_ms, self._curve_at(mv.t_ms))
            col = LABEL_COLOR[mv.label]
            big = mv.label in ("brilliant", "inaccuracy", "mistake", "blunder")
            rad = 6 if big else 3.5
            if i == self.selected:
                # 选中的落子绘制白色外发光环
                p.setPen(QPen(QColor("white"), 2))
                p.setBrush(Qt.NoBrush)
                p.drawEllipse(c, rad + 4, rad + 4)
            p.setPen(Qt.NoPen)
            p.setBrush(col)
            p.drawEllipse(c, rad, rad)
            if mv.symbol and big:
                # 绘制评级符号文本 (如 !! 或 ??)
                p.setPen(col)
                p.drawText(QRectF(c.x() - 15, c.y() - 22, 30, 14), Qt.AlignCenter, mv.symbol)
        p.end()

    def mousePressEvent(self, e) -> None:
        """处理鼠标点击：若点击位置靠近某个落子圆点 (半径 14px 阈值)，选中该手并触发 moveClicked"""
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
    """赛后教练复盘底部面板 (Review Panel)
    
    左侧为胜率走势图 (WinRateGraph)，右侧包含：
    - 统计大纲 (Accuracy %, 各评级计数)
    - 详细出牌列表 (带有时间、卡牌名称、损失百分比)
    - 详细文字解释 (比较实际落子与 AI 推荐落子)
    - "Analyze (R)" 重新分析按钮与进度条
    - "Take Over from Here" (从此处接管分支模拟)
    通过回调函数解耦，保持界面控件无硬编码游戏模拟逻辑。
    """
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
        """加载复盘报告并填充列表项与总结数据"""
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
        """选中某一步棋：高亮走势图上的点，选中列表中对应项，并展示详细文字评估"""
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
    """生成富文本详细教练点评 (Rich-text Coach Review Explanation)
    
    分析包含：
    - 落子时间、卡牌与坐标
    - 评级标签与胜率变化 (落子前胜率 → 实际胜率 · 最佳可能胜率)
    - 若最佳手为等待 (Wait)，解释为何不应下牌 (避免亏费/破轮换)
    - 推荐落子与红绿虚影提示 (Your placement in red, recommended in green)
    - 是否即使单纯空过/等待收益也比该手更高
    - MCTS 搜索候选分支数与推演深度
    """
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
