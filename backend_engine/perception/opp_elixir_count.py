"""
对手算水与卡牌流转追踪器 (Opponent Elixir Counter & Cycle Tracker)
严格遵守人类比赛视角的公平性 (Human-Legal: 仅使用屏幕上公开可见的下牌事件)。

核心工作原理:
1. 确定性算水: 双方初始圣水相同、回水速率完全一致、且对手每一次出牌均在竞技场可见。
   因此只要将对手的历史出牌消耗与阶段回水曲线对齐，即可推算出对手当前的实时圣水值，包括 10 水满费时的漏水量。
   `slack` 参数用于在有噪音的场景（如仅靠屏幕录像 CV 识别、可能存在数帧延迟）下提供置信上下界区间。
2. 皇室战争 4 张回手规则 (Four-card cycle rule):
   打出的卡牌进入 4 张深度的流转队列，后续必须再打出 4 张牌，该卡才会重新回到可用手牌中。
   因此：对手最近打出的 4 张牌当前【绝对不在手中】(概率 0%)；
   而先前已露面且在 4 轮前打出的卡牌，当前【必然已经在手里】(概率 100%)。
3. 容错与乱序重排:
   由于网络抖动或人类感知延迟，事件可能稍微乱序到达。系统使用二分插入有序队列，
   并在需要查询时通过脏标记 (_dirty) 高效增量重建。
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from backend_engine.contracts import ElixirEstimate, PublicDeployEvent

EPS = 1e-6


@dataclass(frozen=True)
class ElixirSchedule:
    """圣水恢复规则与阶段时刻表"""
    start: float                       # 初始开局圣水 (通常为 5.0 或 7.0)
    max: float                         # 圣水上限 (通常为 10.0)
    ms_per_elixir: Dict[int, float]    # 每恢复 1 滴圣水所需的毫秒数 {1: 2800.0, 2: 1400.0, 3: 900.0}
    phases: Tuple[Tuple[int, int], ...] # 圣水倍率阶梯 (起始毫秒, 倍率)

    @classmethod
    def from_rules(cls, rules: dict) -> "ElixirSchedule":
        """从 rules 配置字典解析圣水恢复排程"""
        phases = tuple(sorted((int(p["from_s"] * 1000), int(p["multiplier"])) for p in rules["elixir_phases"]))
        return cls(float(rules["start_elixir"]), float(rules["max_elixir"]),
                   {int(k): float(v) for k, v in rules["ms_per_elixir"].items()}, phases)

    def multiplier(self, t_ms: int) -> int:
        """根据当前比赛时刻返回当前的充能倍率 (1x, 2x, 3x)"""
        m = 1
        for start, mult in self.phases:
            if t_ms >= start:
                m = mult
        return m

    def advance(self, value: float, t0: int, t1: int) -> Tuple[float, float]:
        """
        向前跨越时间段 [t0, t1]，计算圣水充能增长与满 10 费溢出漏水量
        返回: (增长后的圣水存量, 溢出漏水量)
        """
        leaked = 0.0
        t = t0
        # 将时间区间切分为跨越不同回水倍率阶段的关键时间点
        bounds = [s for s, _ in self.phases if t0 < s < t1] + [t1]
        for b in bounds:
            rate = 1.0 / self.ms_per_elixir[self.multiplier(t)]
            value += rate * (b - t)
            if value > self.max:
                leaked += value - self.max
                value = self.max
            t = b
        return value, leaked


class OppElixirTracker:
    """对手圣水与卡序追踪器核心类"""

    def __init__(self, rules: dict, observer_side: int, deck_size: int = 8, hand_size: int = 4,
                 evo_cards: Iterable[str] = (), slack: float = 0.0):
        self.schedule = ElixirSchedule.from_rules(rules)
        self.observer_side = observer_side # 观察者阵营 (通常为人类玩家 side 0)
        self.deck_size = deck_size         # 卡组总容量 (8 张)
        self.hand_size = hand_size         # 手牌槽位数 (4 张)
        self.evo_capable = set(evo_cards)  # 具备觉醒进化特性的卡牌候选集合
        self.slack = slack                 # 容差松弛度 (用于区间边界估算)
        self._events: List[Tuple[int, int, PublicDeployEvent]] = [] # 历史公开下牌事件有序队列
        self._dirty = True                 # 缓存脏标记
        self._state: Tuple[float, int, float, int] = (self.schedule.start, 0, 0.0, 0) # (圣水, 上次时间, 漏水, 纠错次数)

    # -- 事件输入 ----------------------------------------------------------
    def observe(self, ev: PublicDeployEvent) -> None:
        """接收并记录单次公开下牌事件 (忽略我方出牌，只关注对手)"""
        if ev.side == self.observer_side:
            return
        key = (ev.t_ms, ev.event_id, ev)
        if any(e[1] == ev.event_id for e in self._events):
            return
        # 二分法有序插入，确保时间序列严格单调递增
        bisect.insort(self._events, key, key=lambda k: (k[0], k[1]))
        self._dirty = True

    def observe_all(self, events: Sequence[PublicDeployEvent]) -> None:
        """批量接收公开事件"""
        for ev in events:
            self.observe(ev)

    # -- 状态回放与概率模型重建 --------------------------------------------
    def _rebuild(self) -> None:
        """根据历史所有出牌事件完整重新模拟回水与扣费过程"""
        value, t, leaked, resyncs = self.schedule.start, 0, 0.0, 0
        for t_ev, _, ev in self._events:
            value, lk = self.schedule.advance(value, t, t_ev)
            leaked += lk
            t = t_ev
            if value + EPS < ev.elixir_cost:
                # 纠错重同步：对手打出的费用超出了推算结余，说明我们漏算了某些事件，将圣水校正为该卡费用
                resyncs += 1
                value = float(ev.elixir_cost)
            value -= ev.elixir_cost
        self._state = (value, t, leaked, resyncs)
        self._dirty = False

    def _card_plays(self) -> List[PublicDeployEvent]:
        """过滤出真正的手牌打出事件 (排除英雄技能触发)"""
        return [ev for _, _, ev in self._events if not ev.is_ability]

    def estimate(self, t_ms: int) -> ElixirEstimate:
        """
        获取指定仿真毫秒时刻 t_ms 对手的完整状态推算评估
        包含：点估计圣水、上下界区间、已明牌列表、4张回手排队队列、在手概率字典、觉醒充能状态
        """
        if self._dirty:
            self._rebuild()
        value, t_last, leaked, resyncs = self._state
        # 补齐从上次事件时刻推进到当前查询时刻 t_ms 的圣水自然增长
        if t_ms > t_last:
            value, lk = self.schedule.advance(value, t_last, t_ms)
            leaked += lk
        plays = [ev.card_id for ev in self._card_plays()]
        revealed: List[str] = []
        for c in plays:
            if c not in revealed:
                revealed.append(c)
        n = len(plays)
        # 最近打出的 4 张牌必定不在手牌中 (正在卡组队列中等待回手)
        recent = plays[-self.hand_size:]
        queue: Tuple[Optional[str], ...] = tuple([None] * (self.hand_size - len(recent)) + recent)
        # 已经露面且不在最近 4 张队列中的牌，必定在当前手里
        known_in_hand = tuple(c for c in revealed if c not in recent)
        unknown_cards = self.deck_size - len(revealed)
        unknown_slots = self.hand_size - len(known_in_hand)
        probs: Dict[str, float] = {c: 1.0 for c in known_in_hand}
        probs.update({c: 0.0 for c in recent})

        # 统计觉醒进化充能进度
        evo: Dict[str, int] = {}
        for ev in self._card_plays():
            if ev.is_evolved:
                self.evo_capable.add(ev.card_id)
                evo[ev.card_id] = 0
            elif ev.card_id in self.evo_capable:
                evo[ev.card_id] = evo.get(ev.card_id, 0) + 1

        return ElixirEstimate(
            t_ms=t_ms,
            point=value,
            low=max(0.0, value - self.slack),
            high=min(self.schedule.max, value + self.slack),
            known_cards=tuple(revealed),
            cycle_order=tuple(plays),
            in_hand_prob=probs,
            multiplier=self.schedule.multiplier(t_ms),
            queue=queue,
            known_in_hand=known_in_hand,
            unknown_hand_slots=max(0, min(unknown_slots, unknown_cards)),
            leaked=leaked,
            evo_progress=evo,
            resyncs=resyncs,
        )

    # -- 状态快照与回溯 ----------------------------------------------------
    def snapshot(self) -> dict:
        """生成追踪器状态快照"""
        return {"events": list(self._events), "evo": set(self.evo_capable)}

    def restore(self, snap: dict) -> None:
        """从快照还原追踪器状态"""
        self._events = list(snap["events"])
        self.evo_capable = set(snap["evo"])
        self._dirty = True
