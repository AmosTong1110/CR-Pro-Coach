"""
皇室战争 AI 陪练教练 — 1v1 实战对战竞技场 (PySide6 Qt 桌面客户端)

启动方式：
    python -m frontend_app.qt.app

快捷键与操作说明：
- 键盘 1-4：选择对应手牌槽位的卡牌
- 鼠标左键点击棋盘：部署卡牌到指定位置
- 鼠标右键 / Esc：取消选中当前卡牌
- 空格键 (Space)：启动 / 暂停对局模拟
- N 键：重置并开启新对局
- E 键：圣水盲测问答 (Quiz 模式，猜对方此刻剩余圣水)
- U / F 键：后退 / 快进 10 秒 (时间轴回溯)
- R 键：开启赛后教练复盘与走势分析
- 双击右侧出牌记录 (Play log)：从该手出牌前的时间点直接接管 (Take over) 对局
"""
from __future__ import annotations

from typing import List, Optional

from PySide6.QtCore import QRectF, Qt, QTimer, Signal
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

# 下牌违规原因映射字典 (保持英文用于测试匹配)
REASONS = {
    "not_enough_elixir": "Not enough elixir",
    "enemy_half": "Can't deploy on enemy side",
    "river": "Can't deploy in the river",
    "blocked_by_structure": "Blocked by a building",
    "out_of_bounds": "Out of bounds",
    "not_in_hand": "Card not in hand",
    "match_over": "Match is over",
}
# AI 反应延迟分类名称映射
LATENCY_NAMES = {
    LatencyClass.SURPRISE: "Surprise", LatencyClass.ANTICIPATED: "Anticipated",
    LatencyClass.ROUTINE: "Routine", LatencyClass.HOVER_REFLEX: "Hover reflex",
}
# 模拟运行倍速列表
SPEEDS = [("1x", 1), ("2x", 2), ("4x", 4), ("8x", 8)]
# 时间跳转后的缓冲倒计时 (毫秒)
COUNTDOWN_MS = 3000


def mmss(ms: int) -> str:
    """毫秒时间转换为分:秒格式"""
    s = ms // 1000
    return f"{s // 60}:{s % 60:02d}"


# ---------------------------------------------------------------------------
class ElixirBar(QWidget):
    """圣水槽控件 (Elixir Bar)
    
    采用皇室战争官方风格的 10 格紫色彩条展示当前圣水存量：
    - 整数格以亮粉紫色 (#d43cf0) 填满
    - 充能中的小数部分以深紫色 (#8e3aa6) 平滑过渡填充
    - 左侧显示当前圣水向下取整数值
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.value = 0.0
        self.setMinimumHeight(26)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_value(self, v: float) -> None:
        """更新圣水值并重绘"""
        self.value = v
        self.update()

    def paintEvent(self, e) -> None:
        """绘制 10 格独立圆角分段与圣水数值"""
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        label_w = 34
        seg = (w - label_w - 4) / 10
        # 逐格绘制 10 格圣水槽底色与充能填充
        for i in range(10):
            r = QRectF(label_w + i * seg + 1, 3, seg - 2, h - 6)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#3a1f45"))
            p.drawRoundedRect(r, 3, 3)
            fill = max(0.0, min(1.0, self.value - i))
            if fill > 0:
                p.setBrush(QColor("#d43cf0") if fill >= 1 else QColor("#8e3aa6"))
                p.drawRoundedRect(QRectF(r.x(), r.y(), r.width() * fill, r.height()), 3, 3)
        # 绘制左侧当前整数圣水读数
        f = QFont()
        f.setPixelSize(17)
        f.setBold(True)
        p.setFont(f)
        p.setPen(QColor("#e8a7ff"))
        p.drawText(QRectF(0, 0, label_w, h), Qt.AlignCenter, str(int(self.value + 1e-6)))
        p.end()


class HandBar(QWidget):
    """手牌栏与圣水槽组合控件 (Hand Bar)
    
    包含：
    1. 左侧 "Next" 卡牌预览槽位
    2. 4 个可选择手牌槽位 (支持键盘 1-4 快捷切换)，支持觉醒标识 (★Evo) 与费用判断变灰
    3. 下方紧贴的 10 格圣水槽 (ElixirBar)
    """
    def __init__(self, on_select, parent=None):
        super().__init__(parent)
        self.on_select = on_select
        self.group = QButtonGroup(self)
        self.group.setExclusive(False)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 4, 0, 0)
        row = QHBoxLayout()
        # 下一张进牌槽
        self.next_label = QLabel()
        self.next_label.setAlignment(Qt.AlignCenter)
        self.next_label.setFixedWidth(64)
        self.next_label.setStyleSheet("color:#aaa; font-size:11px;")
        row.addWidget(self.next_label)
        # 4 个手牌按钮
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
        # 下方圣水槽
        self.elixir = ElixirBar()
        lay.addWidget(self.elixir)

    def refresh(self, hand: List[str], next_card: str, names, costs, elixir: float,
                selected: Optional[str], evo_ready: set) -> None:
        """刷新手牌按钮状态、觉醒标识、圣水足够时的背景高亮色及选中框线"""
        for b, c in zip(self.buttons, hand):
            evo = " ★Evo" if c in evo_ready else ""
            b.setText(f"{names[c]}{evo}\n{costs[c]} Elixir")
            b.setChecked(c == selected)
            # 圣水充足时为蓝底亮字，费用不足时为暗灰底灰字
            afford = costs[c] <= elixir + 1e-9
            b.setStyleSheet(
                "QPushButton{font-size:13px; border-radius:6px; padding:4px;"
                f"background:{'#2b4f86' if afford else '#3a3a3a'}; color:{'white' if afford else '#999'};"
                "border:2px solid #1b2f50;}"
                "QPushButton:checked{border:3px solid #ffd400;}")
        self.next_label.setText(f"Next\n{names[next_card]}")
        self.elixir.set_value(elixir)


# ---------------------------------------------------------------------------
class DeckDialog(QDialog):
    """卡组构建对话框 (Deck Builder Dialog)
    
    允许配置玩家与 AI 的 8 张手牌卡组及 1 张觉醒核心卡。
    校验规则：
    1. 每个卡组必须严格包含 8 张卡牌。
    2. 若选择了觉醒卡牌，该卡牌必须在卡组的 8 张牌内。
    """
    def __init__(self, session: GameSession, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Deck Builder")
        self.session = session
        db = session.db
        lay = QHBoxLayout(self)
        self.lists, self.evos = [], []
        # 双列布局：左侧玩家卡组，右侧 AI 卡组
        for side, title, deck, evo in ((HUMAN, "Your Deck", session.cfg.player_deck, session.cfg.player_evo),
                                       (AI, "AI Deck", session.cfg.ai_deck, session.cfg.ai_evo)):
            box = QGroupBox(title)
            v = QVBoxLayout(box)
            lw = QListWidget()
            # 遍历卡牌数据库，按圣水费用排序填充复选框列表
            for cid, c in sorted(db.cards.items(), key=lambda kv: (kv[1]["elixir"], kv[0])):
                it = QListWidgetItem(f"{c['name']}  ({c['elixir']})")
                it.setData(Qt.UserRole, cid)
                it.setFlags(it.flags() | Qt.ItemIsUserCheckable)
                it.setCheckState(Qt.Checked if cid in deck else Qt.Unchecked)
                lw.addItem(it)
            v.addWidget(lw)
            # 觉醒卡下拉选择框
            evo_box = QComboBox()
            evo_box.addItem("No evolution", None)
            for cid, c in db.cards.items():
                if db.has_variant(cid, "evo"):
                    evo_box.addItem(f"Evo {c['name']}", cid)
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
        """获取索引为 i 的卡组中被选中的卡牌 ID 元组"""
        lw = self.lists[i]
        return tuple(lw.item(j).data(Qt.UserRole) for j in range(lw.count()) if lw.item(j).checkState() == Qt.Checked)

    def _accept(self) -> None:
        """提交前校验：严格 8 张牌，且觉醒卡必须包含在卡组中"""
        for i in (0, 1):
            deck = self._chosen(i)
            if len(deck) != 8:
                QMessageBox.warning(self, "Deck", f"Each deck needs exactly 8 cards (currently {len(deck)}).\nEach deck needs exactly 8 cards.")
                return
            evo = self.evos[i].currentData()
            if evo and evo not in deck:
                QMessageBox.warning(self, "Deck", "Evolution card must be in the deck.")
                return
        self.accept()

    def result_config(self) -> dict:
        """打包导出卡组配置字典"""
        evo = lambda i: (self.evos[i].currentData(),) if self.evos[i].currentData() else ()  # noqa: E731
        return dict(player_deck=self._chosen(0), ai_deck=self._chosen(1), player_evo=evo(0), ai_evo=evo(1))


# ---------------------------------------------------------------------------
class CounterPanel(QGroupBox):
    """对手圣水与卡牌轮换追踪面板 (Opponent Elixir & Cycle Counter)
    
    支持两种模式：
    1. Show (公开显示)：实时展示对手当前推测圣水、我方圣水差 (Advantage/Deficit)、对手手牌/出牌队列，以及贝叶斯卡组推断。
    2. Quiz (圣水盲测练习)：隐藏对手圣水数值，玩家按 E 键盲猜对方圣水，系统记录平均绝对误差 (MAE) 与得分统计。
    """

    def __init__(self, session: GameSession, on_guess, parent=None):
        super().__init__("Elixir Counter", parent)
        self.session = session
        lay = QVBoxLayout(self)
        top = QHBoxLayout()
        # 模式切换下拉框 (Show vs Quiz)
        self.mode = QComboBox()
        self.mode.addItem("Show", "show")
        self.mode.addItem("Quiz (hidden, press E to guess)", "quiz")
        self.mode.currentIndexChanged.connect(lambda _: self.refresh())
        top.addWidget(self.mode, 1)
        self.guess_btn = QPushButton("Guess (E)")
        self.guess_btn.clicked.connect(on_guess)
        top.addWidget(self.guess_btn)
        lay.addLayout(top)
        # 圣水数值与圣水差指示标签
        self.value = QLabel()
        self.value.setStyleSheet("font-size:22px; font-weight:bold; color:#d43cf0;")
        self.adv = QLabel()
        self.adv.setStyleSheet("font-size:13px; font-weight:bold;")
        row = QHBoxLayout()
        row.addWidget(self.value)
        row.addStretch(1)
        row.addWidget(self.adv)
        lay.addLayout(row)
        # 轮换追踪 (手牌推测与返回序列)
        self.cycle = QLabel()
        self.cycle.setWordWrap(True)
        self.cycle.setStyleSheet("font-size:12px;")
        self.cycle.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Minimum)
        lay.addWidget(self.cycle)
        # 问答测验统计数据
        self.stats = QLabel()
        self.stats.setStyleSheet("font-size:11px; color:#888;")
        lay.addWidget(self.stats)

    @property
    def quiz_mode(self) -> bool:
        """当前是否处于盲测练习模式"""
        return self.mode.currentData() == "quiz"

    def refresh(self) -> None:
        """刷新对手感知数据：计算圣水差、轮换队列、漏水统计以及贝叶斯推断卡组"""
        s = self.session
        names = {cid: c["name"] for cid, c in s.db.cards.items()}
        n, mae, within = s.quiz_stats()
        self.stats.setText(f"Quiz stats: {n} tries · MAE {mae:.2f} · Within 0.5: {within:.0%}" if n else "Quiz stats: No attempts yet")
        if self.quiz_mode:
            # 测验模式下隐藏对手实时圣水
            self.value.setText("AI ≈ ???")
            self.adv.setText("")
            self.cycle.setText("Track yourself: what did AI play, and what card returns next?")
            return
        e = s.opp_estimate()
        self.value.setText(f"AI ≈ {e.point:.1f}")
        adv = s.elixir_advantage()
        if abs(adv) < 0.5:
            self.adv.setText("Even")
            self.adv.setStyleSheet("font-size:13px; font-weight:bold; color:#bbb;")
        else:
            self.adv.setText(f"{'Advantage' if adv > 0 else 'Deficit'} {adv:+.1f}")
            self.adv.setStyleSheet(f"font-size:13px; font-weight:bold; color:{'#4cd964' if adv > 0 else '#ff6b5e'};")
        # 显示轮换队列与已知手牌
        q = [names[c] if c else "?" for c in e.queue]
        hand = [names[c] for c in e.known_in_hand] + ["?"] * e.unknown_hand_slots
        lines = [f"In hand: {' · '.join(hand)}",
                 f"Returns next: ▶ {' → '.join(q)}"]
        # 觉醒轮换层数计数
        evo = [f"{names[c]} {k}/{s.db.evo_cycles(c)}"
               for c, k in e.evo_progress.items()]
        if evo:
            lines.append("Evo cycles: " + ", ".join(evo))
        # 圣水溢出 (Leaked Elixir) 提示
        if e.leaked > 0.05:
            lines.append(f"AI leaked {e.leaked:.1f} elixir")
        # 贝叶斯卡组推断结果
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
    """皇室战争 1v1 对战与复盘教练主窗口 (Main Application Window)
    
    模块组合：
    - 竞技场对战棋盘 (BoardWidget)
    - 时间轴跳转与快进/快退控制栏 (Timeline Scrubbing & 10s Rewind/Forward)
    - 4 槽位手牌与圣水条 (HandBar & ElixirBar)
    - 比赛状态区 (对局时钟、阶段倍速、皇冠比分)
    - 对局配置面板 (启动/暂停、新建比赛、卡组编辑器、AI 角色性格、卡等调节、迷雾盲区高亮)
    - 对手感知与盲测答题面板 (CounterPanel)
    - 出牌历史日志 (Play log)，支持双击任意出牌瞬间接管对局
    - 赛后教练复盘底部悬浮坞 (ReviewPanel QDockWidget)
    """
    # Inside the CRbot shell this window is the Fight page; the shell listens to these signals to save matches.
    matchEnded = Signal()       # the match reached a result
    reviewReady = Signal()      # a coach review finished
    aboutToReset = Signal()     # the current match is about to be replaced (new match, new settings, reopen)

    def __init__(self, session: Optional[GameSession] = None):
        super().__init__()
        self.auto_review = True
        self.setWindowTitle("CR Bot — 1v1 Sparring Arena")
        self.session = session or GameSession()
        self.running = False
        self.speed = 1
        self.selected: Optional[str] = None
        self.countdown_ms = 0

        # 主棋盘交互控件
        self.board = BoardWidget(self.session)
        self.board.tileClicked.connect(self.on_tile)
        self.board.cancelled.connect(lambda: self.select(None))
        self.hand = HandBar(self.on_hand_slot)

        # 时间轴与快进快退控制栏
        tl = QHBoxLayout()
        self.rew_btn = QPushButton("⏪ 10s (U)")
        self.rew_btn.clicked.connect(lambda: self.jump(self.session.rewind(10), "Rewind 10s"))
        self.fwd_btn = QPushButton("10s ⏩ (F)")
        self.fwd_btn.clicked.connect(lambda: self.jump(self.session.forward(10), "Forward 10s"))
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setSingleStep(1000)
        self.slider.setPageStep(10_000)
        self.slider.sliderReleased.connect(lambda: self.jump(self.session.seek(self.slider.value()), "Seek"))
        self.slider_label = QLabel()
        self.slider_label.setFixedWidth(40)
        for w in (self.rew_btn, self.slider, self.slider_label, self.fwd_btn):
            tl.addWidget(w, 1 if w is self.slider else 0)

        # 左侧竖向布局：竞技场棋盘 + 时间轴进度条 + 底部手牌与圣水
        left = QVBoxLayout()
        left.addWidget(self.board, 1)
        left.addLayout(tl)
        left.addWidget(self.hand)

        # 右侧侧边栏：时钟、控制按钮、圣水轮换感知、日志
        right = QVBoxLayout()
        self.clock = QLabel()
        self.clock.setStyleSheet("font-size:34px; font-weight:bold;")
        self.phase = QLabel()
        self.phase.setStyleSheet("font-size:14px; color:#d43cf0; font-weight:bold;")
        self.crowns = QLabel()
        self.crowns.setStyleSheet("font-size:18px;")
        for w in (self.clock, self.phase, self.crowns):
            right.addWidget(w)

        controls = QGroupBox("Match")
        form = QFormLayout(controls)
        self.start_btn = QPushButton("Start (Space)")
        self.start_btn.clicked.connect(self.toggle_run)
        new_btn = QPushButton("New Match (N)")
        new_btn.clicked.connect(self.new_match)
        deck_btn = QPushButton("Deck Builder…")
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
        self.lv11, self.lv16 = QRadioButton("Level 11 (Tournament)"), QRadioButton("Level 16 (Max)")
        (self.lv16 if self.session.cfg.level == 16 else self.lv11).setChecked(True)
        self.lv11.toggled.connect(lambda on: on and self.restart(level=11))
        self.lv16.toggled.connect(lambda on: on and self.restart(level=16))
        lv.addWidget(self.lv11)
        lv.addWidget(self.lv16)
        self.vision = QCheckBox("Highlight units unseen by AI (yellow dashed)")
        self.vision.setChecked(True)
        self.vision.toggled.connect(self._toggle_vision)
        form.addRow(self.start_btn)
        form.addRow(new_btn)
        form.addRow(deck_btn)
        review_btn = QPushButton("Review (R)")
        review_btn.clicked.connect(self.run_review)
        form.addRow(review_btn)
        form.addRow("Speed", self.speed_box)
        form.addRow("AI Persona", self.persona_box)
        form.addRow(lv)
        form.addRow(self.vision)
        right.addWidget(controls)

        self.counter = CounterPanel(self.session, self.quiz_guess)
        right.addWidget(self.counter)

        right.addWidget(QLabel("Play log · Double-click to take over from before that move"))
        self.log_list = QListWidget()
        self.log_list.setStyleSheet("font-size:12px;")
        self.log_list.itemDoubleClicked.connect(self.take_over)
        right.addWidget(self.log_list, 1)
        self.status = QLabel()
        self.status.setWordWrap(True)
        self.status.setStyleSheet("color:#ffb347;")
        right.addWidget(self.status)
        note = QLabel("AI currently runs scripted baseline heuristics (RL model replaces this).\nCard data: game files as of 2023-10.")
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

        # 赛后教练复盘底部停靠窗口 (Post-match Coach Review Dock)
        names = lambda c: self.session.db.cards.get(c, {}).get("name", c)  # noqa: E731
        self.review_panel = ReviewPanel(names, self.run_review, self.take_over_review)
        self.review_panel.moveSelected.connect(self.show_review_move)
        self.review_dock = QDockWidget("Post-match Coach Review · Win-rate Graph", self)
        self.review_dock.setWidget(self.review_panel)
        self.review_dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        self.addDockWidget(Qt.BottomDockWidgetArea, self.review_dock)
        self.review_dock.hide()
        self.analyzing = False

        # 注册全局快捷键
        for i in range(4):
            QShortcut(QKeySequence(str(i + 1)), self, activated=lambda i=i: self.on_hand_slot(i))
        QShortcut(QKeySequence(Qt.Key_Space), self, activated=self.toggle_run)
        QShortcut(QKeySequence(Qt.Key_Escape), self, activated=lambda: self.select(None))
        QShortcut(QKeySequence("N"), self, activated=self.new_match)
        QShortcut(QKeySequence("E"), self, activated=self.quiz_guess)
        QShortcut(QKeySequence("U"), self, activated=self.rew_btn.click)
        QShortcut(QKeySequence("F"), self, activated=self.fwd_btn.click)
        QShortcut(QKeySequence("R"), self, activated=self.run_review)

        # 核心 50ms 定时器驱动模拟循环
        self.timer = QTimer(self)
        self.timer.setInterval(TICK_MS)
        self.timer.timeout.connect(self.on_timer)
        self.timer.start()
        self.refresh()
        self.flash("Press Space to start")

    # -- 模拟主循环 (Simulation Loop) ---------------------------------------
    def on_timer(self) -> None:
        """主定时器回调 (每 50ms 触发一次)
        
        处理：
        1. 时间回溯后的真实时间 3-2-1 缓冲倒计时 (此时游戏逻辑时钟冻结)。
        2. 推进物理模拟帧 (调用 session.tick(speed))。
        3. 对局结束检测：自动暂停并弹出赛后复盘分析。
        4. 刷新合法下牌格子与界面状态。
        """
        if self.countdown_ms > 0:  # 倒计时阶段：时钟停顿，屏幕显示 3, 2, 1
            self.countdown_ms -= TICK_MS
            if self.countdown_ms <= 0:
                self.board.overlay_text = None
                self.board.ghosts, self.board.banner = [], None
                self.running = True
                self.start_btn.setText("Pause (Space)")
            else:
                self.board.overlay_text = str(-(-self.countdown_ms // 1000))
            self.board.update()
            return
        if not self.running or self.session.result is not None:
            return
        self.board.ghosts, self.board.banner = [], None
        # 执行物理 tick 并记录新产生的落子事件
        self.add_log(self.session.tick(self.speed))
        if self.session.result is not None:
            # 比赛结束，停止运行并自动调起教练复盘
            self.running = False
            self.start_btn.setText("Start (Space)")
            self.refresh()
            self.matchEnded.emit()
            if self.auto_review:
                self.run_review()
            return
        self.board.refresh_legal()
        self.refresh()

    # -- 赛后教练复盘 (Post-match Coach) -------------------------------------
    def run_review(self) -> None:
        """运行赛后教练复盘分析
        
        对对局中玩家的所有出牌进行 8 秒 MCTS 深度推演评估，
        计算理论最优解、落子损失，划分棋类等级并绘制胜率波动图。
        """
        if self.analyzing:
            return
        if not any(e.side == HUMAN for e in self.session.log):
            self.review_dock.show()
            self.flash("Play some cards first before analyzing")
            return
        self.analyzing = True
        self.running = False
        self._stop_countdown()
        self.start_btn.setText("Resume (Space)")
        self.review_dock.show()
        panel = self.review_panel
        panel.progress.setVisible(True)
        panel.analyze_btn.setEnabled(False)
        self.centralWidget().setEnabled(False)
        self.flash("Analyzing… simulating candidate moves 8s ahead")

        def progress(done: int, total: int) -> None:
            """进度条更新回调"""
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
        self.flash(f"Review complete: Accuracy {review.accuracy:.0f}% · Click markers or list items to inspect moves")
        self.reviewReady.emit()

    def show_review_move(self, i: int) -> None:
        """在棋盘上展示第 i 手出牌的红绿虚影对比 (Red = Actual, Green = Best)"""
        review = self.review_panel.review
        mv = review.moves[i]
        if self.session.restore_before_event(mv.event_id) is None:
            self.flash("Move is no longer on this timeline; please re-analyze")
            return
        self._stop_countdown()
        self.running = False
        self.start_btn.setText("Resume (Space)")
        self.log_list.clear()
        self.add_log([])
        self.select(None)
        a, b = mv.actual, mv.best
        # 红虚影：玩家实际操作
        ghosts = [(a.card_id, a.x, a.y, "actual")]
        # 绿虚影：引擎推荐最佳操作
        if not b.is_wait and (b.card_id, b.x, b.y) != (a.card_id, a.x, a.y):
            ghosts.append((b.card_id, b.x, b.y, "best"))
        self.board.ghosts = ghosts
        tip = "Best: Wait" if b.is_wait else ("Already best move" if mv.loss <= 0.01 else "Red = Your play · Green = Recommended")
        self.board.banner = f"{_mmss(mv.t_ms)}  {mv.label_text} {mv.symbol}  ·  {tip}"
        self.refresh()

    def take_over_review(self) -> None:
        """从复盘面板当前选中的步骤接管比赛继续对战"""
        self.jump(self.session.match.t_ms, "Take over from here")

    def toggle_run(self) -> None:
        """空格键切换暂停与继续"""
        if self.countdown_ms > 0:
            self._stop_countdown()
            self.running = False
            self.start_btn.setText("Resume (Space)")
            return
        if self.session.result is not None:
            return
        self.running = not self.running
        self.start_btn.setText("Pause (Space)" if self.running else "Resume (Space)")
        self.flash("")

    # -- 对局配置与管理 (Match Setup) ---------------------------------------
    def restart(self, **changes) -> None:
        """应用新参数重开对局"""
        self.aboutToReset.emit()
        self.session.reconfigure(**changes)
        self._reset_view()
        self.flash("New match started with updated settings")

    def new_match(self) -> None:
        """重置并开启全新对局"""
        self.aboutToReset.emit()
        self.session.new_match()
        self._reset_view()
        self.flash("New match · Press Space to start")

    # -- 时间轴与接管 (Timeline & Take Over) ---------------------------------
    def jump(self, restored: Optional[int], what: str) -> None:
        """执行时间轴跳转 (快进/快退/接管)：
        重置界面、弹出 3 秒缓冲倒计时，并在倒计时结束后无缝恢复对局。
        """
        if restored is None:
            self.flash("Nothing to jump to")
            self.refresh()
            return
        self.running = False
        self.log_list.clear()
        self.add_log([])
        self.select(None)
        self.countdown_ms = COUNTDOWN_MS
        self.board.overlay_text = "3"
        self.flash(f"{what} → {mmss(restored)} · Countdown 3s (Space to stay paused)")
        self.refresh()

    def take_over(self, item: QListWidgetItem) -> None:
        """双击日志列表项：从该手下牌前的时间节点分支接管游戏"""
        i = self.log_list.row(item)
        e = self.session.log[i]
        who = "You" if e.side == HUMAN else "AI"
        name = self.session.db.cards[e.card_id]["name"]
        self.jump(self.session.take_over_before(i), f"Take over: Before {who} plays {name}")

    def _stop_countdown(self) -> None:
        """清除倒计时状态"""
        self.countdown_ms = 0
        self.board.overlay_text = None
        self.board.update()

    def _reset_view(self) -> None:
        """重置界面到对局初始状态"""
        self._stop_countdown()
        self.board.ghosts, self.board.banner = [], None
        self.review_panel.set_review(None, 0)
        self.running = False
        self.start_btn.setText("Start (Space)")
        self.log_list.clear()
        self.select(None)
        self.refresh()

    def set_session(self, session: GameSession) -> None:
        """Show another session (e.g. a match reopened from Analysis), with its review if it has one."""
        self.aboutToReset.emit()
        self.session = session
        self.board.session = session
        self.counter.session = session
        self._reset_view()
        self.add_log([])
        self.persona_box.blockSignals(True)
        self.persona_box.setCurrentIndex(list(PERSONAS).index(session.cfg.persona))
        self.persona_box.blockSignals(False)
        for rb, on in ((self.lv11, session.cfg.level != 16), (self.lv16, session.cfg.level == 16)):
            rb.blockSignals(True)
            rb.setChecked(on)
            rb.blockSignals(False)
        if session.review is not None:
            end = max(session.match.t_ms, session.review.curve[-1][0] if session.review.curve else 0)
            self.review_panel.set_review(session.review, end)
            self.review_dock.show()
        self.refresh()

    def edit_decks(self) -> None:
        """打开卡组配置弹窗"""
        dlg = DeckDialog(self.session, self)
        if dlg.exec():
            self.restart(**dlg.result_config())

    def quiz_guess(self, guess: Optional[float] = None) -> None:
        """圣水盲测问答：暂停游戏，弹出输入框让玩家推测 AI 圣水，并根据绝对误差打分"""
        was_running = self.running
        self.running = False
        if guess is None or isinstance(guess, bool):
            guess, ok = QInputDialog.getDouble(self, "Elixir Quiz",
                                               "How much elixir does the AI have right now?",
                                               5.0, 0.0, 10.0, 1)
            if not ok:
                self.running = was_running
                return
        r = self.session.quiz_guess(guess)
        verdict = "Perfect" if r.error <= 0.3 else "Close" if r.error <= 1.0 else "Large gap"
        self.flash(f"Actual {r.truth:.1f} · Your guess {r.guess:.1f} · Error {r.error:.1f} ({verdict})")
        self.running = was_running
        self.refresh()

    def _toggle_vision(self, on: bool) -> None:
        """切换是否用黄色虚线标出 AI 处于迷雾盲区尚未看见的单位"""
        self.board.show_ai_vision = on
        self.board.update()

    # -- 玩家输入处理 (User Input Handling) ----------------------------------
    def on_hand_slot(self, i: int) -> None:
        """手牌快捷键 1-4 槽位选择处理"""
        hand = self.session.match.sides[HUMAN].hand
        if i < len(hand):
            self.select(None if self.selected == hand[i] else hand[i])

    def select(self, card: Optional[str]) -> None:
        """设置当前选中的卡牌"""
        self.selected = card
        self.board.set_selected(card)
        self.refresh()

    def on_tile(self, x: float, y: float) -> None:
        """玩家点击棋盘执行下牌操作"""
        if not self.selected:
            return
        if self.countdown_ms > 0:
            self.flash("Countdown in progress: wait for countdown to end before dropping card")
            return
        res = self.session.play(self.selected, x, y)
        if res.ok and self.review_panel.review is not None:
            self.review_panel.summary.setText(self.review_panel.summary.text().split("  ·  Diverged")[0]
                                              + "  ·  Diverged from original match, press R to re-analyze")
        if res.ok:
            self.flash("")
            self.select(None)
            self.add_log([])
        else:
            self.flash(REASONS.get(res.reason, res.reason))

    # -- 视图同步与日志 (View Sync & Logging) ---------------------------------
    def flash(self, msg: str) -> None:
        """更新底部黄色状态栏提示信息"""
        self.status.setText(msg)

    def add_log(self, _new: List[LogEntry]) -> None:
        """向日志列表追加新落子记录，包含出牌方、觉醒标记、坐标以及 AI 的认知延迟"""
        shown = self.log_list.count()
        names = {cid: c["name"] for cid, c in self.session.db.cards.items()}
        for e in self.session.log[shown:]:
            who = "You" if e.side == HUMAN else "AI"
            evo = " ★Evo" if e.evolved else ""
            text = f"{mmss(e.t_ms)}  {who}  {names[e.card_id]}{evo}  ({e.x:.1f}, {e.y:.1f})"
            if e.ai_latency:
                cls, ms = e.ai_latency
                text += f"\n        → AI noticed after {ms / 1000:.2f}s ({LATENCY_NAMES.get(cls, cls.value)})"
            it = QListWidgetItem(text)
            it.setForeground(QColor("#7fb2ff") if e.side == HUMAN else QColor("#ff8a80"))
            self.log_list.addItem(it)
        self.log_list.scrollToBottom()

    def refresh(self) -> None:
        """全面刷新主窗口视图：时钟、皇冠、手牌栏、圣水条、对手感知、时间轴滑块及棋盘"""
        s = self.session
        m = s.match
        st = m.sides[HUMAN]
        clock, phase = s.clock()
        self.clock.setText(clock)
        self.phase.setText(phase)
        self.crowns.setText(f"👑 You {st.crowns}  :  {m.sides[AI].crowns} AI")
        names = {cid: c["name"] for cid, c in s.db.cards.items()}
        costs = {cid: c["elixir"] for cid, c in s.db.cards.items()}
        evo_ready = {c for c in st.evo_cards
                     if st.evo_counter[c] >= s.db.evo_cycles(c)}
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
    """Entry point: the CRbot shell (Fight / Analysis / Decks / Account) with this arena as the Fight page."""
    from frontend_app.qt.shell import main as shell_main
    shell_main()


if __name__ == "__main__":
    main()
