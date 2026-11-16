"""
游戏会话核心控制器 (GameSession Controller)
作为 1v1 对战界面背后的纯业务逻辑中枢 (完全独立于 Qt 图形库)。

所有 UI 界面、录像回放、算水训练、时间线快照跳转、接管重打 (Take Over)、
以及赛后复盘分析 (Post-Match Coach) 均通过该控制器统一调度驱动。
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from datetime import datetime
from typing import Dict, List, Optional, Tuple

from backend_engine.brain.coach_analyzer import Candidate, CoachAnalyzer, MatchReview
from backend_engine.brain.human_delay_buffer import HumanDelayBuffer, ReactionProfile
from backend_engine.brain.scripted_bot import ScriptedBot
from backend_engine.contracts import ElixirEstimate, LatencyClass
from backend_engine.meta.archetypes import archetype_name
from backend_engine.perception.bayesian_deck_guesser import DeckGuess, default_guesser
from backend_engine.perception.opp_elixir_count import OppElixirTracker
from backend_engine.simulator import geometry as G
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import ENGINE_VERSION, Match, MatchConfig, MatchResult, PlayResult, Unit
from backend_engine.simulator.match_runner import DEFAULT_DECK, MatchRunner
from backend_engine.storage.match_store import (
    MatchMeta, MatchRecord, actions_from_events, new_match_id, outcome_of,
)

HUMAN, AI = 0, 1

# 四种预设的陪练 AI 人格打法风格 (在强化学习策略网络导入前作为高水平基准):
#   - surgeon (完美运营大师): 稳健控水，精细解场防反
#   - aggressor (极致换路莽夫): 压迫感拉满，高频双路穿插进攻
#   - spell_cycler (自闭法术斩杀流): 龟缩防守，靠火箭/火球磨血斩杀
#   - adaptive_counter (自适应克制者): 见招拆招，追求费用差优势
PERSONAS: Dict[str, dict] = {
    "surgeon":    {"label": "The Surgeon (Control/Macro)",       "aggression": 0.3, "profile": "surgeon"},
    "aggressor":  {"label": "The Aggressor (Dual-Lane Pressure)", "aggression": 0.95, "profile": "aggressor"},
    "spell_cycler": {"label": "Spell-Cycler (Turtle/Chip)",       "aggression": 0.2, "profile": "spell_cycler"},
    "adaptive_counter": {"label": "Adaptive Counter",             "aggression": 0.5, "profile": "adaptive_counter"},
}


@dataclass(frozen=True)
class SessionConfig:
    """对局会话初始参数配置"""
    player_deck: Tuple[str, ...] = DEFAULT_DECK    # 玩家携带的 8 张卡牌
    ai_deck: Tuple[str, ...] = DEFAULT_DECK        # AI 携带的 8 张卡牌
    level: int = 11                                # 比赛等级 (11 锦标赛或 16 满级)
    persona: str = "surgeon"                       # AI 陪练人格
    player_evo: Tuple[str, ...] = ()               # 玩家觉醒卡槽
    ai_evo: Tuple[str, ...] = ()                   # AI 觉醒卡槽
    seed: Optional[int] = None                     # 确定性对局随机种子


@dataclass(frozen=True)
class LogEntry:
    """对局出牌事件记录条目"""
    t_ms: int
    side: int
    card_id: str
    x: float
    y: float
    evolved: bool
    ai_latency: Optional[Tuple[LatencyClass, int]] = None  # AI 经过多少毫秒才看清该出牌
    event_id: int = 0
    ability: bool = False                                   # champion ability use (card_id = the champion)


# 快照采样频率：每 1000 毫秒 (1秒) 记录一个全量时间线检查点，用于平滑拖拽跳转与快退
CHECKPOINT_EVERY_MS = 1000


@dataclass
class Checkpoint:
    """时间线全量状态快照 (包含战斗物理引擎、AI 眼睛感知、手指动作及双方算水器状态)"""
    t_ms: int
    match: dict
    ai_buffer: dict
    delivered: int
    ai_trackers: Dict[int, dict]
    opp_tracker: dict


@dataclass(frozen=True)
class QuizResult:
    """玩家单次算水练习测试结果"""
    t_ms: int
    guess: float                       # 玩家猜测的对手圣水数
    truth: float                       # 此时对手真实的圣水数
    error: float                       # 绝对误差


class GameSession:
    """游戏会话核心驱动器"""

    def __init__(self, db: Optional[CardDB] = None, cfg: SessionConfig = SessionConfig()):
        self.db = db or CardDB.load()
        self.cfg = cfg
        self.log: List[LogEntry] = []
        self.quiz: List[QuizResult] = []   # 跨局保留的玩家算水历史测试记录
        self.new_match()

    # -- 对局生命周期管理 ----------------------------------------------------
    def new_match(self, cfg: Optional[SessionConfig] = None) -> None:
        """重置并开启一局全新的比赛"""
        if cfg is not None:
            self.cfg = cfg
        c = self.cfg
        seed = c.seed if c.seed is not None else random.randrange(1 << 30)
        self.seed = seed
        persona = PERSONAS[c.persona]
        try:
            profile = ReactionProfile.load(persona=persona["profile"])
        except Exception:
            profile = ReactionProfile.default()
        # 初始化物理引擎与对战双方
        self.match = Match(MatchConfig((c.player_deck, c.ai_deck), (c.level, c.level),
                                       (c.player_evo, c.ai_evo), seed=seed), self.db)
        self.ai_buffer = HumanDelayBuffer(profile, side=AI, seed=seed + 1)
        self.runner = MatchRunner(self.match, {HUMAN: None, AI: ScriptedBot(self.db, AI, persona["aggression"])},
                                  {AI: self.ai_buffer})
        self.log = []
        self._logged = 0
        evo_capable = self.db.evo_capable()
        # 玩家视角面向 AI 的算水与卡序追踪器
        self.opp_tracker = OppElixirTracker(self.db.rules, observer_side=HUMAN, evo_cards=evo_capable)
        self.timeline: List[Checkpoint] = []       # 当前时间线上的历史快照栈 (从旧到新)
        self.redo: List[Checkpoint] = []           # 快退后保存在前方的未来快照 (在你重新下牌改变历史前有效)
        self.pre_play: Dict[int, Checkpoint] = {}  # 玩家每次下牌前一瞬间的精准状态快照映射表
        self.review: Optional[MatchReview] = None
        self.match_id: Optional[str] = None        # assigned on first save
        self.loaded_from: Optional[str] = None     # set when reopened from the match store
        self.started_at = datetime.now()
        self._checkpoint()

    # -- 时间线控制：撤销、快进与接管重打 ------------------------------------
    def _capture(self) -> Checkpoint:
        """捕获当前时刻的全量状态快照"""
        return Checkpoint(
            t_ms=self.match.t_ms,
            match=self.match.snapshot(),
            ai_buffer=self.ai_buffer.snapshot(),
            delivered=self.runner._delivered,
            ai_trackers={s: t.snapshot() for s, t in self.runner.trackers.items()},
            opp_tracker=self.opp_tracker.snapshot(),
        )

    def _restore(self, cp: Checkpoint) -> None:
        """完全还原快照状态"""
        self.match.restore(cp.match)
        self.ai_buffer.restore(cp.ai_buffer)
        self.runner._delivered = cp.delivered
        for s, snap in cp.ai_trackers.items():
            self.runner.trackers[s].restore(snap)
        self.opp_tracker.restore(cp.opp_tracker)
        self.log, self._logged = [], 0
        self._collect_log()

    def _checkpoint(self) -> None:
        """若跨越到了新的时间步，记录一个新的时间线检查点"""
        t = self.match.t_ms
        if self.timeline and self.timeline[-1].t_ms >= t:
            return
        self.timeline.append(self._capture())
        while self.redo and self.redo[0].t_ms <= t:
            self.redo.pop(0)

    def can_rewind(self) -> bool:
        """是否可以回退时间线"""
        return bool(self.timeline) and (self.timeline[0].t_ms < self.match.t_ms or len(self.timeline) > 1)

    def can_forward(self) -> bool:
        """是否可以快进重做时间线"""
        return bool(self.redo)

    def max_known_ms(self) -> int:
        """获取当前已探索的最远时间线毫秒数"""
        return self.redo[-1].t_ms if self.redo else self.match.t_ms

    def seek(self, t_ms: int) -> Optional[int]:
        """跳转到指定仿真毫秒时刻 t_ms (自动在 timeline 与 redo 之间穿梭)"""
        now = self.match.t_ms
        if t_ms == now and self.match.result is None:
            return now
        # 向后回溯
        if t_ms < now or (t_ms == now and self.match.result is not None):
            idx = max((i for i, cp in enumerate(self.timeline) if cp.t_ms <= t_ms), default=None)
            if idx is None:
                return None
            if self.timeline[-1].t_ms < now:
                self.redo.insert(0, self._capture())
            self.redo = self.timeline[idx + 1:] + self.redo
            self.timeline = self.timeline[:idx + 1]
            self._restore(self.timeline[-1])
            return self.timeline[-1].t_ms
        # 向前快进
        ahead = [i for i, cp in enumerate(self.redo) if cp.t_ms <= t_ms]
        if not ahead:
            return None
        k = ahead[-1]
        self.timeline.extend(self.redo[:k + 1])
        self.redo = self.redo[k + 1:]
        self._restore(self.timeline[-1])
        return self.timeline[-1].t_ms

    def rewind(self, seconds: float = 10) -> Optional[int]:
        """后退指定秒数 (默认 10 秒)"""
        return self.seek(max(0, self.match.t_ms - int(seconds * 1000)))

    def forward(self, seconds: float = 10) -> Optional[int]:
        """前进指定秒数 (默认 10 秒)"""
        if not self.redo:
            return None
        target = self.match.t_ms + int(seconds * 1000)
        return self.seek(target) if self.redo[0].t_ms <= target else self.seek(self.redo[0].t_ms)

    def take_over_before(self, log_index: int) -> Optional[int]:
        """从出牌记录的第 log_index 步之前一瞬间接管游戏，重打这波对拼"""
        e = self.log[log_index]
        if e.event_id in self.pre_play:
            return self.restore_before_event(e.event_id)
        return self.seek(max(0, e.t_ms - 1))

    def restore_before_event(self, event_id: int) -> Optional[int]:
        """精准回溯到特定下牌事件发生前的瞬间"""
        cp = self.pre_play.get(event_id)
        if cp is None or not any(ev.event_id == event_id for ev in self.match.events):
            return None
        self.seek(cp.t_ms)
        if self.match.t_ms != cp.t_ms or any(ev.event_id == event_id for ev in self.match.events):
            self._restore(cp)
        return self.match.t_ms

    # -- 赛后复盘分析调度 ----------------------------------------------------
    def fork(self, cp: Checkpoint) -> Tuple[Match, MatchRunner]:
        """克隆生成独立的平行推演副本 (专用于分析评估，完全不影响正在进行的实时对局)"""
        m = Match(self.match.cfg, self.db)
        m.restore(cp.match)
        buf = HumanDelayBuffer(self.ai_buffer.profile, side=AI, seed=0)
        buf.restore(cp.ai_buffer)
        r = MatchRunner(m, {HUMAN: None, AI: ScriptedBot(self.db, AI, PERSONAS[self.cfg.persona]["aggression"])},
                        {AI: buf})
        r._delivered = cp.delivered
        for s, snap in cp.ai_trackers.items():
            r.trackers[s].restore(snap)
        return m, r

    def analyze(self, progress=None, horizon_ms: int = 8000) -> MatchReview:
        """启动赛后复盘教练：计算胜率曲线并深度复盘玩家本局所有出牌"""
        analyzer = CoachAnalyzer(self.fork, horizon_ms=horizon_ms)
        plays = []
        for ev in self.match.events:
            if ev.side == HUMAN and ev.event_id in self.pre_play and not ev.is_ability:
                plays.append((self.pre_play[ev.event_id], Candidate(ev.card_id, ev.tile.x, ev.tile.y), ev.event_id))
        timeline = [cp for cp in self.timeline if cp.t_ms <= self.match.t_ms]
        self.review = analyzer.review_match(timeline, plays, final_match=self.match, progress=progress)
        return self.review

    # -- 算水练习与猜牌接口 ------------------------------------------------
    def opp_estimate(self) -> ElixirEstimate:
        """获取对手当前圣水与牌序估计"""
        return self.opp_tracker.estimate(self.match.t_ms)

    def opp_deck_guess(self) -> Optional[DeckGuess]:
        """获取贝叶斯推算出的对手可能卡组"""
        g = default_guesser()
        return g.guess(self.opp_estimate().known_cards) if g else None

    def elixir_advantage(self) -> float:
        """计算我方圣水优势差额 (我方当前圣水 - 对手估计圣水)"""
        return self.match.sides[HUMAN].elixir - self.opp_estimate().point

    def quiz_guess(self, guess: float) -> QuizResult:
        """提交一次算水猜测并评判误差"""
        truth = self.match.sides[AI].elixir
        r = QuizResult(self.match.t_ms, guess, truth, abs(guess - truth))
        self.quiz.append(r)
        return r

    def quiz_stats(self) -> Tuple[int, float, float]:
        """统计算水训练战绩：(总测试次数, 平均绝对误差, 误差在0.5水以内的占比)"""
        if not self.quiz:
            return 0, 0.0, 0.0
        n = len(self.quiz)
        return n, sum(q.error for q in self.quiz) / n, sum(q.error <= 0.5 for q in self.quiz) / n

    def reconfigure(self, **changes) -> None:
        """动态修改配置并重启比赛 (如切换 16 级满级或更换卡组)"""
        self.new_match(replace(self.cfg, **changes))

    # -- 时间推进 ----------------------------------------------------------
    def tick(self, n: int = 1) -> List[LogEntry]:
        """向前步进 n 帧并收集新产生的出牌日志"""
        for _ in range(n):
            if self.match.result is not None:
                break
            if self.match.t_ms % CHECKPOINT_EVERY_MS == 0:
                self._checkpoint()
            self.runner.tick()
        if self.match.result is not None:
            self._checkpoint()
        return self._collect_log()

    def _collect_log(self) -> List[LogEntry]:
        """同步并封装新增的出牌日志"""
        new = []
        for ev in self.match.events[self._logged:]:
            self.opp_tracker.observe(ev)
            lat = self.ai_buffer.latency_of(ev.event_id) if ev.side == HUMAN else None
            new.append(LogEntry(ev.t_ms, ev.side, ev.card_id, ev.tile.x, ev.tile.y, ev.is_evolved, lat, ev.event_id,
                                ev.is_ability))
        self._logged = len(self.match.events)
        self.log.extend(new)
        return new

    # -- 玩家下牌动作 ------------------------------------------------------
    def play(self, card_id: str, x: float, y: float) -> PlayResult:
        """玩家在特定网格下牌"""
        if self.match.t_ms % CHECKPOINT_EVERY_MS == 0:
            self._checkpoint()
        before = self._capture()
        res = self.runner.human_play(card_id, x, y, side=HUMAN)
        if res.ok:
            if self.loaded_from is not None:  # playing on a reopened match: a new branch, saved separately
                self.match_id, self.loaded_from, self.started_at = None, None, datetime.now()
            self.redo.clear()  # 改变了历史走向，旧的未来废弃
            self.pre_play[res.event.event_id] = before
            if not self.timeline or self.timeline[-1].t_ms < before.t_ms:
                self.timeline.append(before)
        self.runner.deliver_events()
        self._collect_log()
        return res

    def ability_cards(self) -> List[str]:
        """Your champions whose ability can be used right now (on the field, ready, enough elixir)."""
        m = self.match
        cards = dict.fromkeys(u.card_id for u in m.units.values() if u.side == HUMAN)
        return [c for c in cards if m.check_ability(HUMAN, c) is None]

    def use_ability(self, card_id: str) -> PlayResult:
        """Player uses a champion ability (1 elixir, single use). Mirrors `play`: checkpoint, new branch, log."""
        if self.match.t_ms % CHECKPOINT_EVERY_MS == 0:
            self._checkpoint()
        before = self._capture()
        res = self.runner.human_ability(card_id, side=HUMAN)
        if res.ok:
            if self.loaded_from is not None:
                self.match_id, self.loaded_from, self.started_at = None, None, datetime.now()
            self.redo.clear()
            self.pre_play[res.event.event_id] = before
            if not self.timeline or self.timeline[-1].t_ms < before.t_ms:
                self.timeline.append(before)
        self.runner.deliver_events()
        self._collect_log()
        return res

    def legal_tiles(self, card_id: str) -> List[Tuple[float, float]]:
        """获取当前卡牌可以合法放置的所有网格中心点坐标 (忽略圣水限制，用于鼠标悬停预览)"""
        st = self.match.sides[HUMAN]
        if card_id not in st.hand:
            return []
        saved = st.elixir
        st.elixir = 10.0
        try:
            return [(x + 0.5, y + 0.5) for y in range(int(G.H)) for x in range(int(G.W))
                    if self.match.check_play(HUMAN, card_id, x + 0.5, y + 0.5) is None]
        finally:
            st.elixir = saved

    # -- 视图查询属性 ------------------------------------------------------
    @property
    def result(self) -> Optional[MatchResult]:
        return self.match.result

    def ai_has_seen(self, u: Unit) -> bool:
        """AI 是否已经在屏幕上观察到了该单位 (用于黄色虚线战争迷雾高亮)"""
        return u.side == AI or u.source_event_id is None or self.ai_buffer.is_visible(u.source_event_id)

    def clock(self) -> Tuple[str, str]:
        """获取比赛倒计时与阶段文本 ('2:41', 'x2 Elixir' / 'Overtime x3' / '')"""
        m = self.match
        left = (m.regulation_ms if m.t_ms < m.regulation_ms else m.end_ms) - m.t_ms
        s = max(0, (left + 999) // 1000)
        mult = m.multiplier()
        if m.t_ms >= m.regulation_ms:
            phase = f"Overtime x{mult}"
        else:
            phase = f"x{mult} Elixir" if mult > 1 else ""
        return f"{s // 60}:{s % 60:02d}", phase

    def elixir_cost(self, card_id: str) -> int:
        """查询某张卡牌的圣水费用"""
        return self.db.cards[card_id]["elixir"]

    # -- saving and reopening (match store) -------------------------------------
    def has_human_plays(self) -> bool:
        """True if you played at least one card on the furthest known timeline."""
        return any(e.side == HUMAN for e in self._furthest_match().events)

    def _furthest_match(self) -> Match:
        """The match at the furthest point reached on this branch (after a rewind, that is the redo end)."""
        if not self.redo:
            return self.match
        m = Match(self.match.cfg, self.db)
        m.restore(self.redo[-1].match)
        return m

    def review_is_current(self) -> bool:
        """True if the stored review describes plays that are all still on the furthest timeline."""
        if self.review is None:
            return False
        mine = {(e.event_id, e.t_ms, e.card_id) for e in self._furthest_match().events if e.side == HUMAN}
        return all((mv.event_id, mv.t_ms, mv.actual.card_id) in mine for mv in self.review.moves)

    def config_dict(self) -> dict:
        c = self.cfg
        return {"player_deck": list(c.player_deck), "ai_deck": list(c.ai_deck), "level": c.level,
                "persona": c.persona, "player_evo": list(c.player_evo), "ai_evo": list(c.ai_evo),
                "seed": self.seed}

    def to_record(self, mode: str = "practice") -> Tuple[MatchMeta, MatchRecord]:
        """Summary + rebuildable record of this match (furthest point of the current branch)."""
        if self.match_id is None:
            self.match_id = new_match_id(mode, self.seed, self.started_at)
        m = self._furthest_match()
        c = self.cfg
        res = m.result
        record = MatchRecord(self.config_dict(), actions_from_events(m.events), m.t_ms, m.state_hash())
        meta = MatchMeta(
            match_id=self.match_id, mode=mode, created=self.started_at.isoformat(timespec="seconds"),
            duration_ms=m.t_ms, outcome=outcome_of(res.winner if res else None, res is not None, HUMAN),
            crowns=(m.sides[HUMAN].crowns, m.sides[AI].crowns), reason=res.reason if res else None,
            player_deck=tuple(c.player_deck), opp_deck=tuple(c.ai_deck),
            player_evo=tuple(c.player_evo), opp_evo=tuple(c.ai_evo), level=c.level, persona=c.persona,
            player_archetype=archetype_name(c.player_deck, c.player_evo),
            opp_archetype=archetype_name(c.ai_deck, c.ai_evo),
            n_plays=sum(e.side == HUMAN and not e.is_ability for e in m.events),
            data_version=str(self.db.raw.get("data_version", "")), engine_version=ENGINE_VERSION,
        )
        return meta, record

    @classmethod
    def rebuild(cls, record: MatchRecord, db: Optional[CardDB] = None, progress=None) -> "GameSession":
        """Replay a saved match tick-for-tick (same seed, your plays at the same ticks; the AI re-decides
        identically). Raises ReplayMismatch if the result differs, e.g. after a card-data change."""
        c = record.config
        cfg = SessionConfig(tuple(c["player_deck"]), tuple(c["ai_deck"]), int(c["level"]), c["persona"],
                            tuple(c.get("player_evo", ())), tuple(c.get("ai_evo", ())), int(c["seed"]))
        s = cls(db, cfg)
        mine = sorted(record.human_actions(HUMAN), key=lambda a: (a.t_ms, a.event_id))
        total = max(record.final_t_ms, 1)
        last_report = -10_000

        def advance_to(t: int) -> None:
            nonlocal last_report
            while s.match.t_ms < t and s.match.result is None:
                s.tick(1)
                if progress and s.match.t_ms - last_report >= 5000:
                    last_report = s.match.t_ms
                    progress(s.match.t_ms, total)

        for a in mine:
            advance_to(a.t_ms)
            if s.match.t_ms != a.t_ms:
                raise ReplayMismatch(f"match ended before your play at {a.t_ms} ms")
            res = s.use_ability(a.card_id) if a.ability else s.play(a.card_id, a.x, a.y)
            if not res.ok:
                raise ReplayMismatch(f"{'ability of' if a.ability else 'play'} {a.card_id} at {a.t_ms} ms rejected: "
                                     f"{res.reason}")
        advance_to(record.final_t_ms)
        if s.match.t_ms != record.final_t_ms or s.match.state_hash() != record.final_hash:
            raise ReplayMismatch("rebuilt match differs from the saved one")
        return s

    def mark_loaded(self, meta: MatchMeta, review: Optional[MatchReview]) -> None:
        """After rebuild(): remember where it came from; a new play on it saves as a new match."""
        self.match_id = meta.match_id
        self.loaded_from = meta.match_id
        self.review = review
        try:
            self.started_at = datetime.fromisoformat(meta.created)
        except ValueError:
            pass


class ReplayMismatch(Exception):
    pass
