"""
CR Bot — 1v1 practice arena (PySide6).

    python -m frontend_app.qt.app

Controls: 1-4 select a card, click the board to deploy, right-click / Esc to cancel,
Space start/pause, N new match, E guess the AI's elixir (quiz mode),
U / F rewind / forward 10 s, double-click a play-log line to take over from just before it.
"""
from __future__ import annotations

import sys
from typing import List, Optional

from PySide6.QtCore import QRectF, Qt, QTimer
from PySide6.QtGui import QColor, QFont, QKeySequence, QPainter, QShortcut
from PySide6.QtWidgets import (
    QApplication, QButtonGroup, QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFormLayout,
    QGroupBox, QHBoxLayout, QInputDialog, QLabel, QListWidget, QListWidgetItem, QMainWindow, QMessageBox, QDockWidget,
    QPushButton, QRadioButton, QSizePolicy, QSlider, QVBoxLayout, QWidget,
)

from backend_engine.contracts import LatencyClass
from backend_engine.simulator.geometry import TICK_MS
from frontend_app.qt.board import BoardWidget
from frontend_app.qt.review import ReviewPanel, mmss as _mmss
from frontend_app.qt.session import AI, HUMAN, PERSONAS, GameSession, LogEntry

REASONS = {
    "not_enough_elixir": "圣水不足 Not enough elixir",
    "enemy_half": "不能下在对方半场 Can't deploy on enemy side",
    "river": "不能下在河里 Can't deploy in the river",
    "blocked_by_structure": "被建筑挡住 Blocked by a building",
    "out_of_bounds": "超出场地 Out of bounds",
    "not_in_hand": "这张牌不在手里 Card not in hand",
    "match_over": "对局已结束 Match is over",
}
LATENCY_CN = {
    LatencyClass.SURPRISE: "突袭", LatencyClass.ANTICIPATED: "预判",
    LatencyClass.ROUTINE: "常规", LatencyClass.HOVER_REFLEX: "悬停秒解",
}
SPEEDS = [("1x", 1), ("2x", 2), ("4x", 4), ("8x", 8)]
COUNTDOWN_MS = 3000


def mmss(ms: int) -> str:
    s = ms // 1000
    return f"{s // 60}:{s % 60:02d}"


# ---------------------------------------------------------------------------
class ElixirBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0.0
        self.setMinimumHeight(26)
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
    def __init__(self, on_select, parent=None):
        super().__init__(parent)
        self.on_select = on_select
        self.group = QButtonGroup(self)
        self.group.setExclusive(False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 0)
        row = QHBoxLayout()
        self.next_label = QLabel()
        self.next_label.setAlignment(Qt.AlignCenter)
        self.next_label.setFixedWidth(64)
        self.next_label.setStyleSheet("color:#aaa; font-size:11px;")
        row.addWidget(self.next_label)
        self.buttons: List[QPushButton] = []
        for i in range(4):
            b = QPushButton()
            b.setCheckable(True)
            b.setMinimumHeight(58)
            b.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            b.clicked.connect(lambda _=False, i=i: self.on_select(i))
            self.group.addButton(b, i)
            self.buttons.append(b)
            row.addWidget(b)
        lay.addLayout(row)
        self.elixir = ElixirBar()
        lay.addWidget(self.elixir)

    def refresh(self, hand: List[str], next_card: str, names, costs, elixir: float,
                selected: Optional[str], evo_ready: set) -> None:
        for b, c in zip(self.buttons, hand):
            evo = " ★觉醒" if c in evo_ready else ""
            b.setText(f"{names[c]}{evo}\n{costs[c]} 圣水")
            b.setChecked(c == selected)
            afford = costs[c] <= elixir + 1e-9
            b.setStyleSheet(
                "QPushButton{font-size:13px; border-radius:6px; padding:4px;"
                f"background:{'#2b4f86' if afford else '#3a3a3a'}; color:{'white' if afford else '#999'};"
                "border:2px solid #1b2f50;}"
                "QPushButton:checked{border:3px solid #ffd400;}")
        self.next_label.setText(f"下一张 Next\n{names[next_card]}")
        self.elixir.set_value(elixir)


# ---------------------------------------------------------------------------
class DeckDialog(QDialog):
    def __init__(self, session: GameSession, parent=None):
        super().__init__(parent)
        self.setWindowTitle("卡组构建 Deck Builder")
        self.session = session
        db = session.db
        lay = QHBoxLayout(self)
        self.lists, self.evos = [], []
        for side, title, deck, evo in ((HUMAN, "你的卡组 Your deck", session.cfg.player_deck, session.cfg.player_evo),
                                       (AI, "AI 卡组 AI deck", session.cfg.ai_deck, session.cfg.ai_evo)):
            box = QGroupBox(title)
            v = QVBoxLayout(box)
            lw = QListWidget()
            for cid, c in sorted(db.cards.items(), key=lambda kv: (kv[1]["elixir"], kv[0])):
                it = QListWidgetItem(f"{c['name']}  ({c['elixir']})")
                it.setData(Qt.UserRole, cid)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if cid in deck else Qt.Unchecked)
                lw.addItem(it)
            v.addWidget(lw)
            evo_box = QComboBox()
            evo_box.addItem("无觉醒 No evolution", None)
            for cid, c in db.cards.items():
                if c.get("evolution"):
                    evo_box.addItem(f"觉醒 {c['name']}", cid)
                    if cid in evo:
                        evo_box.setCurrentIndex(evo_box.count() - 1)
            v.addWidget(evo_box)
            self.lists.append(lw)
            self.evos.append(evo_box)
            lay.addWidget(box)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _chosen(self, i: int):
        lw = self.lists[i]
        return tuple(lw.item(j).data(Qt.UserRole) for j in range(lw.count()) if lw.item(j).checkState() == Qt.Checked)

    def _accept(self) -> None:
        for i in (0, 1):
            deck = self._chosen(i)
            if len(deck) != 8:
                QMessageBox.warning(self, "卡组", f"每个卡组需要 8 张牌（现在 {len(deck)} 张）\nEach deck needs exactly 8 cards.")
                return
            evo = self.evos[i].currentData()
            if evo and evo not in deck:
                QMessageBox.warning(self, "卡组", "觉醒卡必须在卡组里 Evolution card must be in the deck.")
                return
        self.accept()

    def result_config(self) -> dict:
        evo = lambda i: (self.evos[i].currentData(),) if self.evos[i].currentData() else ()  # noqa: E731
        return dict(player_deck=self._chosen(0), ai_deck=self._chosen(1), player_evo=evo(0), ai_evo=evo(1))


# ---------------------------------------------------------------------------
class CounterPanel(QGroupBox):
    """Opponent elixir + cycle counter, with a quiz mode that hides it so you practise counting."""

    def __init__(self, session: GameSession, on_guess, parent=None):
        super().__init__("对手算水 Elixir counter", parent)
        self.session = session
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        self.mode = QComboBox()
        self.mode.addItem("显示 Show", "show")
        self.mode.addItem("训练 Quiz（隐藏，按 E 估算）", "quiz")
        self.mode.currentIndexChanged.connect(lambda _: self.refresh())
        top.addWidget(self.mode, 1)
        self.guess_btn = QPushButton("估算 Guess (E)")
        self.guess_btn.clicked.connect(on_guess)
        top.addWidget(self.guess_btn)
        lay.addLayout(top)
        self.value = QLabel()
        self.value.setStyleSheet("font-size:22px; font-weight:bold; color:#d43cf0;")
        self.adv = QLabel()
        self.adv.setStyleSheet("font-size:13px; font-weight:bold;")
        row = QHBoxLayout()
        row.addWidget(self.value)
        row.addStretch(1)
        row.addWidget(self.adv)
        lay.addLayout(row)
        self.cycle = QLabel()
        self.cycle.setWordWrap(True)
        self.cycle.setStyleSheet("font-size:12px;")
        self.cycle.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)  # never squeezed; the log shrinks
        lay.addWidget(self.cycle)
        self.stats = QLabel()
        self.stats.setStyleSheet("font-size:11px; color:#888;")
        lay.addWidget(self.stats)

    @property
    def quiz_mode(self) -> bool:
        return self.mode.currentData() == "quiz"

    def refresh(self) -> None:
        s = self.session
        names = {cid: c["name"] for cid, c in s.db.cards.items()}
        n, mae, within = s.quiz_stats()
        self.stats.setText(f"训练记录 {n} 次 · 平均误差 {mae:.2f} · 0.5 内 {within:.0%}" if n else "训练记录：还没有")
        if self.quiz_mode:
            self.value.setText("对手 ≈ ???")
            self.adv.setText("")
            self.cycle.setText("自己数：对手出过什么、哪张牌下一张回手？")
            return
        e = s.opp_estimate()
        self.value.setText(f"对手 ≈ {e.point:.1f}")
        adv = s.elixir_advantage()
        if abs(adv) < 0.5:
            self.adv.setText("持平 even")
            self.adv.setStyleSheet("font-size:13px; font-weight:bold; color:#bbb;")
        else:
            self.adv.setText(f"你{'领先' if adv > 0 else '落后'} {adv:+.1f}")
            self.adv.setStyleSheet(f"font-size:13px; font-weight:bold; color:{'#4cd964' if adv > 0 else '#ff6b5e'};")
        q = [names[c] if c else "?" for c in e.queue]
        hand = [names[c] for c in e.known_in_hand] + ["?"] * e.unknown_hand_slots
        lines = [f"手牌 In hand: {' · '.join(hand)}",
                 f"回手顺序 Returns next: ▶ {' → '.join(q)}"]
        evo = [f"{names[c]} {k}/{s.db.cards[c]['evolution']['cycles_required']}"
               for c, k in e.evo_progress.items()]
        if evo:
            lines.append("觉醒充能(若为觉醒卡): " + ", ".join(evo))
        if e.leaked > 0.05:
            lines.append(f"对手已溢出 {e.leaked:.1f} 圣水")
        g = s.opp_deck_guess()
        if g is not None and g.card_probs:
            pretty = lambda c: names.get(c, c.replace("_", " ").title())  # noqa: E731
            lines.append("可能还带 Likely: " + " · ".join(f"{pretty(c)} {p:.0%}" for c, p in g.top(4)))
            if g.likely_decks and g.likely_decks[0][1] >= 0.3 and len(g.revealed) >= 2 and g.matching_weight >= 5:
                deck, p = g.likely_decks[0]
                lines.append(f"最像的卡组 ({p:.0%}): " + ", ".join(pretty(c) for c in deck))
        self.cycle.setText("\n".join(lines))


# ---------------------------------------------------------------------------
class MainWindow(QMainWindow):
    def __init__(self, session: Optional[GameSession] = None):
        super().__init__()
        self.setWindowTitle("CR Bot — 1v1 特训场")
        self.session = session or GameSession()
        self.running = False
        self.speed = 1
        self.selected: Optional[str] = None
        self.countdown_ms = 0

        self.board = BoardWidget(self.session)
        self.board.tileClicked.connect(self.on_tile)
        self.board.cancelled.connect(lambda: self.select(None))
        self.hand = HandBar(self.on_hand_slot)

        tl = QHBoxLayout()
        self.rew_btn = QPushButton("⏪ 10s (U)")
        self.rew_btn.clicked.connect(lambda: self.jump(self.session.rewind(10), "后退 10 秒"))
        self.fwd_btn = QPushButton("10s ⏩ (F)")
        self.fwd_btn.clicked.connect(lambda: self.jump(self.session.forward(10), "前进 10 秒"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setSingleStep(1000)
        self.slider.setPageStep(10_000)
        self.slider.sliderReleased.connect(lambda: self.jump(self.session.seek(self.slider.value()), "跳转"))
        self.slider_label = QLabel()
        self.slider_label.setFixedWidth(40)
        for w in (self.rew_btn, self.slider, self.slider_label, self.fwd_btn):
            tl.addWidget(w, 1 if w is self.slider else 0)

        left = QVBoxLayout()
        left.addWidget(self.board, 1)
        left.addLayout(tl)
        left.addWidget(self.hand)

        right = QVBoxLayout()
        self.clock = QLabel()
        self.clock.setStyleSheet("font-size:34px; font-weight:bold;")
        self.phase = QLabel()
        self.phase.setStyleSheet("font-size:14px; color:#d43cf0; font-weight:bold;")
        self.crowns = QLabel()
        self.crowns.setStyleSheet("font-size:18px;")
        for w in (self.clock, self.phase, self.crowns):
            right.addWidget(w)

        controls = QGroupBox("对局 Match")
        form = QFormLayout(controls)
        self.start_btn = QPushButton("开始 Start (Space)")
        self.start_btn.clicked.connect(self.toggle_run)
        new_btn = QPushButton("新对局 New match (N)")
        new_btn.clicked.connect(self.new_match)
        deck_btn = QPushButton("卡组 Deck builder…")
        deck_btn.clicked.connect(self.edit_decks)
        self.speed_box = QComboBox()
        for label, v in SPEEDS:
            self.speed_box.addItem(label, v)
        self.speed_box.currentIndexChanged.connect(lambda _: setattr(self, "speed", self.speed_box.currentData()))
        self.persona_box = QComboBox()
        for key, p in PERSONAS.items():
            self.persona_box.addItem(p["label"], key)
        self.persona_box.setCurrentIndex(list(PERSONAS).index(self.session.cfg.persona))
        self.persona_box.currentIndexChanged.connect(lambda _: self.restart(persona=self.persona_box.currentData()))
        lv = QHBoxLayout()
        self.lv11, self.lv16 = QRadioButton("Level 11 锦标赛"), QRadioButton("Level 16 满级")
        (self.lv16 if self.session.cfg.level == 16 else self.lv11).setChecked(True)
        self.lv11.toggled.connect(lambda on: on and self.restart(level=11))
        self.lv16.toggled.connect(lambda on: on and self.restart(level=16))
        lv.addWidget(self.lv11)
        lv.addWidget(self.lv16)
        self.vision = QCheckBox("标出 AI 还没看到的单位 (黄色虚线)")
        self.vision.setChecked(True)
        self.vision.toggled.connect(self._toggle_vision)
        form.addRow(self.start_btn)
        form.addRow(new_btn)
        form.addRow(deck_btn)
        review_btn = QPushButton("复盘 Review (R)")
        review_btn.clicked.connect(self.run_review)
        form.addRow(review_btn)
        form.addRow("速度 Speed", self.speed_box)
        form.addRow("AI 人格", self.persona_box)
        form.addRow(lv)
        form.addRow(self.vision)
        right.addWidget(controls)

        self.counter = CounterPanel(self.session, self.quiz_guess)
        right.addWidget(self.counter)

        right.addWidget(QLabel("出牌记录 Play log · 双击 = 从这步之前接管重打"))
        self.log_list = QListWidget()
        self.log_list.setStyleSheet("font-size:12px;")
        self.log_list.itemDoubleClicked.connect(self.take_over)
        right.addWidget(self.log_list, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#ffb347;")
        right.addWidget(self.status)
        note = QLabel("AI 当前是规则脚本基线（RL 模型训练后替换）。\nCard data: game files as of 2023-10.")
        note.setStyleSheet("color:#888; font-size:10px;")
        right.addWidget(note)

        root = QWidget()
        h = QHBoxLayout(root)
        h.addLayout(left, 3)
        rw = QWidget()
        rw.setLayout(right)
        rw.setFixedWidth(330)
        h.addWidget(rw)
        self.setCentralWidget(root)
        self.resize(900, 900)

        names = lambda c: self.session.db.cards.get(c, {}).get("name", c)  # noqa: E731
        self.review_panel = ReviewPanel(names, self.run_review, self.take_over_review)
        self.review_panel.moveSelected.connect(self.show_review_move)
        self.review_dock = QDockWidget("复盘 Post-match coach · 胜率曲线 Win-rate graph", self)
        self.review_dock.setWidget(self.review_panel)
        self.review_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.review_dock)
        self.review_dock.hide()
        self.analyzing = False

        for i in range(4):
            QShortcut(QKeySequence(str(i + 1)), self, activated=lambda i=i: self.on_hand_slot(i))
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_run)
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=lambda: self.select(None))
        QShortcut(QKeySequence("N"), self, activated=self.new_match)
        QShortcut(QKeySequence("E"), self, activated=self.quiz_guess)
        QShortcut(QKeySequence("U"), self, activated=self.rew_btn.click)
        QShortcut(QKeySequence("F"), self, activated=self.fwd_btn.click)
        QShortcut(QKeySequence("R"), self, activated=self.run_review)

        self.timer = QTimer(self)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start()
        self.refresh()
        self.flash("按 Space 开始 · Press Space to start")

    # -- loop --------------------------------------------------------------
    def on_timer(self) -> None:
        if self.countdown_ms > 0:  # real-time 3-2-1 after a jump; the match clock is frozen
            self.countdown_ms -= TICK_MS
            if self.countdown_ms <= 0:
                self.board.overlay_text = None
                self.board.ghosts, self.board.banner = [], None
                self.running = True
                self.start_btn.setText("暂停 Pause (Space)")
            else:
                self.board.overlay_text = str(-(-self.countdown_ms // 1000))
            self.board.update()
            return
        if not self.running or self.session.result is not None:
            return
        self.board.ghosts, self.board.banner = [], None
        self.add_log(self.session.tick(self.speed))
        if self.session.result is not None:
            self.running = False
            self.start_btn.setText("开始 Start (Space)")
            self.refresh()
            self.run_review()  # auto post-match coach
            return
        self.board.refresh_legal()
        self.refresh()

    # -- post-match coach -----------------------------------------------------
    def run_review(self) -> None:
        if self.analyzing:
            return
        if not any(e.side == HUMAN for e in self.session.log):
            self.review_dock.show()
            self.flash("还没有你的出牌可以分析 · Play some cards first")
            return
        self.analyzing = True
        self.running = False
        self._stop_countdown()
        self.start_btn.setText("继续 Resume (Space)")
        self.review_dock.show()
        panel = self.review_panel
        panel.progress.setVisible(True)
        panel.analyze_btn.setEnabled(False)
        self.centralWidget().setEnabled(False)
        self.flash("分析中… 每一步推演几十种打法 · Analysing")

        def progress(done: int, total: int) -> None:
            panel.progress.setMaximum(max(total, 1))
            panel.progress.setValue(done)
            QApplication.processEvents()

        try:
            review = self.session.analyze(progress=progress)
        finally:
            self.centralWidget().setEnabled(True)
            panel.progress.setVisible(False)
            panel.analyze_btn.setEnabled(True)
            self.analyzing = False
        panel.set_review(review, max(self.session.match.t_ms, review.curve[-1][0] if review.curve else 0))
        self.flash(f"复盘完成：准确度 {review.accuracy:.0f} · 点曲线上的标记或列表查看每一步")

    def show_review_move(self, i: int) -> None:
        review = self.review_panel.review
        mv = review.moves[i]
        if self.session.restore_before_event(mv.event_id) is None:
            self.flash("这一步已不在当前时间线上，请重新分析 · Move no longer on this timeline")
            return
        self._stop_countdown()
        self.running = False
        self.start_btn.setText("继续 Resume (Space)")
        self.log_list.clear()
        self.add_log([])
        self.select(None)
        a, b = mv.actual, mv.best
        ghosts = [(a.card_id, a.x, a.y, "actual")]
        if not b.is_wait and (b.card_id, b.x, b.y) != (a.card_id, a.x, a.y):
            ghosts.append((b.card_id, b.x, b.y, "best"))
        self.board.ghosts = ghosts
        tip = "最佳：等待" if b.is_wait else ("这步就是最佳" if mv.loss <= 0.01 else "红=你的落点 · 绿=推荐")
        self.board.banner = f"{_mmss(mv.t_ms)}  {mv.label_text} {mv.symbol}  ·  {tip}"
        self.refresh()

    def take_over_review(self) -> None:
        self.jump(self.session.match.t_ms, "从此处接管")

    def toggle_run(self) -> None:
        if self.countdown_ms > 0:  # Space during the countdown: stay paused instead
            self._stop_countdown()
            self.running = False
            self.start_btn.setText("继续 Resume (Space)")
            return
        if self.session.result is not None:
            return
        self.running = not self.running
        self.start_btn.setText("暂停 Pause (Space)" if self.running else "继续 Resume (Space)")
        self.flash("")

    # -- match setup ---------------------------------------------------------
    def restart(self, **changes) -> None:
        self.session.reconfigure(**changes)
        self._reset_view()
        self.flash("已开新局（设置已更改）· New match with new settings")

    def new_match(self) -> None:
        self.session.new_match()
        self._reset_view()
        self.flash("新对局 · Press Space to start")

    # -- timeline -----------------------------------------------------------
    def jump(self, restored: Optional[int], what: str) -> None:
        """After a rewind/forward/take-over: rebuild the view, then count 3-2-1 and resume."""
        if restored is None:
            self.flash("没有可跳转的时间点 · Nothing to jump to")
            self.refresh()
            return
        self.running = False
        self.log_list.clear()
        self.add_log([])
        self.select(None)
        self.countdown_ms = COUNTDOWN_MS
        self.board.overlay_text = "3"
        self.flash(f"{what} → {mmss(restored)} · 3 秒后继续（Space 保持暂停）")
        self.refresh()

    def take_over(self, item: QListWidgetItem) -> None:
        i = self.log_list.row(item)
        e = self.session.log[i]
        who = "你" if e.side == HUMAN else "AI"
        name = self.session.db.cards[e.card_id]["name"]
        self.jump(self.session.take_over_before(i), f"接管：{who} 出 {name} 之前")

    def _stop_countdown(self) -> None:
        self.countdown_ms = 0
        self.board.overlay_text = None
        self.board.update()

    def _reset_view(self) -> None:
        self._stop_countdown()
        self.board.ghosts, self.board.banner = [], None
        self.review_panel.set_review(None, 0)
        self.running = False
        self.start_btn.setText("开始 Start (Space)")
        self.log_list.clear()
        self.select(None)
        self.refresh()

    def edit_decks(self) -> None:
        dlg = DeckDialog(self.session, self)
        if dlg.exec():
            self.restart(**dlg.result_config())

    def quiz_guess(self, guess: Optional[float] = None) -> None:
        """Pause, ask for your estimate of the AI's elixir, then score it."""
        was_running = self.running
        self.running = False
        if guess is None or isinstance(guess, bool):
            guess, ok = QInputDialog.getDouble(self, "算水训练 Elixir quiz",
                                               "对手现在有多少圣水？ How much elixir does the AI have?",
                                               5.0, 0.0, 10.0, 1)
            if not ok:
                self.running = was_running
                return
        r = self.session.quiz_guess(guess)
        verdict = "完美" if r.error <= 0.3 else "接近" if r.error <= 1.0 else "差距较大"
        self.flash(f"真实 {r.truth:.1f} · 你猜 {r.guess:.1f} · 误差 {r.error:.1f}（{verdict}）")
        self.running = was_running
        self.refresh()

    def _toggle_vision(self, on: bool) -> None:
        self.board.show_ai_vision = on
        self.board.update()

    # -- input ---------------------------------------------------------------
    def on_hand_slot(self, i: int) -> None:
        hand = self.session.match.sides[HUMAN].hand
        if i < len(hand):
            self.select(None if self.selected == hand[i] else hand[i])

    def select(self, card: Optional[str]) -> None:
        self.selected = card
        self.board.set_selected(card)
        self.refresh()

    def on_tile(self, x: float, y: float) -> None:
        if not self.selected:
            return
        if self.countdown_ms > 0:
            self.flash("倒计时中，可以先瞄准，倒数结束再下 · Wait for the countdown")
            return
        res = self.session.play(self.selected, x, y)
        if res.ok and self.review_panel.review is not None:
            self.review_panel.summary.setText(self.review_panel.summary.text().split("  ·  已偏离")[0]
                                              + "  ·  已偏离原对局，按 R 重新分析")
        if res.ok:
            self.flash("")
            self.select(None)
            self.add_log([])
        else:
            self.flash(REASONS.get(res.reason, res.reason))

    # -- view ----------------------------------------------------------------
    def flash(self, msg: str) -> None:
        self.status.setText(msg)

    def add_log(self, _new: List[LogEntry]) -> None:
        shown = self.log_list.count()
        names = {cid: c["name"] for cid, c in self.session.db.cards.items()}
        for e in self.session.log[shown:]:
            who = "你" if e.side == HUMAN else "AI"
            evo = " ★觉醒" if e.evolved else ""
            text = f"{mmss(e.t_ms)}  {who}  {names[e.card_id]}{evo}  ({e.x:.1f}, {e.y:.1f})"
            if e.ai_latency:
                cls, ms = e.ai_latency
                text += f"\n        → AI {ms / 1000:.2f}s 后察觉（{LATENCY_CN.get(cls, cls.value)}）"
            it = QListWidgetItem(text)
            it.setForeground(QColor("#7fb2ff") if e.side == HUMAN else QColor("#ff8a80"))
            self.log_list.addItem(it)
        self.log_list.scrollToBottom()

    def refresh(self) -> None:
        s = self.session
        m = s.match
        st = m.sides[HUMAN]
        clock, phase = s.clock()
        self.clock.setText(clock)
        self.phase.setText(phase)
        self.crowns.setText(f"👑 你 {st.crowns}  :  {m.sides[AI].crowns} AI")
        names = {cid: c["name"] for cid, c in s.db.cards.items()}
        costs = {cid: c["elixir"] for cid, c in s.db.cards.items()}
        evo_ready = {c for c in st.evo_cards
                     if st.evo_counter[c] >= s.db.cards[c]["evolution"]["cycles_required"]}
        self.hand.refresh(st.hand, st.queue[0], names, costs, st.elixir, self.selected, evo_ready)
        self.counter.refresh()
        if not self.slider.isSliderDown():
            self.slider.blockSignals(True)
            self.slider.setRange(0, max(s.max_known_ms(), 1000))
            self.slider.setValue(m.t_ms)
            self.slider.blockSignals(False)
        self.slider_label.setText(mmss(m.t_ms))
        self.rew_btn.setEnabled(s.can_rewind())
        self.fwd_btn.setEnabled(s.can_forward())
        self.board.update()


def main() -> None:
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    w = MainWindow()
    w.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
