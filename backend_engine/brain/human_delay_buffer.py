"""
人类生理反应延迟与操作物理缓冲层 (Human Delay & Physical Reflex Buffer)
位于模拟器与 AI 大脑之间，模拟真实职业高玩的人类生理极限，防止 AI 拥有 0 延迟的非人类外挂作弊能力。

三层人类生理反应架构 (Three-Layer Architecture):
  * 第 1 层 感知闸门 (Perception Gate):
    对手在竞技场上的出牌动作，不会被 AI 立即获悉，必须经过符合生理学正态分布的采样延迟
    （突发意外约 450ms / 预判推算约 250ms / 常规沉底约 350ms）后，才对策略网络可见。
  * 第 2 层 悬停秒解 (Hover Reflex):
    模拟职业选手将克制卡（如滚木）预先悬停在公主塔前的习惯。一旦触发条件卡牌（如飞桶）落地，
    直接绕过策略网络，在 120~200ms 的极致肌肉反射下瞬间秒解。
  * 第 3 层 手指动作限制器 (Finger & APM Limiter):
    模拟单手单指操作手感：连续决策之间受 APM 物理间隙 (180~300ms) 限制；
    连招出牌 (Combo) 按 combo_gap_ms (约 120ms) 依序下牌，无法同时在一瞬间点击多个位置。

确定性与快照保证:
  内部全基于仿真毫秒 (`t_ms`) 与独立的种子随机数 (`_rng`)，支持完美 snapshot() / restore()，
  便于回放复盘、分支试错及蒙特卡洛多路径搜索。
"""
from __future__ import annotations

import copy
import heapq
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from backend_engine.contracts import (
    ActionIntent,
    ActionKind,
    HoverPlan,
    LatencyClass,
    PublicDeployEvent,
    ScheduledAction,
    TilePos,
)

# 默认的人类生理参数配置文件路径
DEFAULT_PROFILE_PATH = Path(__file__).resolve().parents[2] / "configs" / "human_reaction_profile.yaml"


# ---------------------------------------------------------------------------
# 人类反应分布参数类 (Profile & Distribution)
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LatencyDist:
    """带截断的正态延迟分布 (Truncated Normal Distribution)"""
    mean: float                        # 均值 (毫秒)
    std: float                         # 标准差 (毫秒)
    min: float                         # 生理硬下限截断 (毫秒)
    max: float                         # 生理硬上限截断 (毫秒)

    def __post_init__(self) -> None:
        if self.min > self.max:
            raise ValueError(f"LatencyDist min {self.min} > max {self.max}")
        if self.std < 0:
            raise ValueError("LatencyDist std must be >= 0")

    def sample(self, rng: random.Random) -> int:
        """从截断正态分布中采样一个具体毫秒数 (拒绝采样，避免概率在边界堆积)"""
        clamp = lambda v: min(max(v, self.min), self.max)  # noqa: E731
        if self.std == 0 or self.min == self.max:
            return int(round(clamp(self.mean)))
        for _ in range(64):
            v = rng.gauss(self.mean, self.std)
            if self.min <= v <= self.max:
                return int(round(v))
        return int(round(clamp(self.mean)))

    @classmethod
    def from_dict(cls, d: Mapping[str, float]) -> "LatencyDist":
        return cls(float(d["mean"]), float(d["std"]), float(d["min"]), float(d["max"]))


@dataclass(frozen=True)
class ReactionProfile:
    """完整的人类反应与操作参数配置集"""
    surprise: LatencyDist              # 突发意外攻击反应分布 (如桥头突然下精锐/电磁炮)
    anticipated: LatencyDist           # 预判中攻击反应分布 (算准对手圣水与卡序后的防守)
    routine: LatencyDist               # 非紧急常规出牌感知分布 (如对手后排沉底分路)
    hover_reflex: LatencyDist          # 预悬停手指极速秒解分布 (120~200ms)
    apm_gap: LatencyDist               # 连续两次独立操作之间的手指移动时间间隙
    combo_gap_ms: int = 120            # 一口气连下多张牌 (组合技) 时各张牌的间隔时间
    anticipation_threshold: float = 0.35 # 敌方出牌在手概率超过该阈值时，自动转为预判情境
    max_hold_ms: int = 1500            # 当圣水不足卡费时，已排期动作在手里的最长保留等待时间
    max_hover_ms: int = 12000          # 悬停等待触发的最长失效时间 (避免无限悬停)

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "ReactionProfile":
        p = d["perception"]
        return cls(
            surprise=LatencyDist.from_dict(p["surprise"]),
            anticipated=LatencyDist.from_dict(p["anticipated"]),
            routine=LatencyDist.from_dict(p["routine"]),
            hover_reflex=LatencyDist.from_dict(d["hover_reflex"]),
            apm_gap=LatencyDist.from_dict(d["apm_gap"]),
            combo_gap_ms=int(d.get("combo_gap_ms", 120)),
            anticipation_threshold=float(d.get("anticipation_threshold", 0.35)),
            max_hold_ms=int(d.get("max_hold_ms", 1500)),
            max_hover_ms=int(d.get("max_hover_ms", 12000)),
        )

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PROFILE_PATH, persona: Optional[str] = None) -> "ReactionProfile":
        """从 YAML 配置文件中读取特定打法风格 (Persona) 的人类生理参数"""
        import yaml  # 仅在从外部磁盘加载时按需导入

        with open(path, "r", encoding="utf-8") as f:
            raw = yaml.safe_load(f)
        merged = copy.deepcopy(raw["default"])
        if persona is not None:
            personas = raw.get("personas", {}) or {}
            if persona not in personas:
                raise KeyError(f"unknown persona '{persona}', have {sorted(personas)}")
            _deep_merge(merged, personas[persona] or {})
        return cls.from_dict(merged)

    @classmethod
    def default(cls) -> "ReactionProfile":
        """内置默认参数 (与 human_reaction_profile.yaml 保持一致)，适用于单测或无 yaml 环境"""
        return cls(
            surprise=LatencyDist(500, 40, 420, 580),
            anticipated=LatencyDist(320, 40, 240, 420),
            routine=LatencyDist(650, 120, 400, 1000),
            hover_reflex=LatencyDist(160, 20, 120, 200),
            apm_gap=LatencyDist(240, 30, 180, 300),
        )


def _deep_merge(base: Dict[str, Any], over: Mapping[str, Any]) -> None:
    """递归合并字典结构"""
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)


# ---------------------------------------------------------------------------
# 人类反应延迟缓冲区类 (HumanDelayBuffer)
# ---------------------------------------------------------------------------
LegalityCheck = Callable[[ActionIntent], bool]


class HumanDelayBuffer:
    """人类生理延迟核心缓冲队列管理器"""

    def __init__(self, profile: Optional[ReactionProfile] = None, side: int = 1, seed: int = 0):
        self.profile = profile or ReactionProfile.default()
        self.side = side               # 本缓冲层所服务的阵营 (通常为 AI side 1)
        self.seed = seed               # 随机种子
        self.reset()

    # -- 生命周期与重置 ----------------------------------------------------
    def reset(self) -> None:
        """重置所有内部队列与状态"""
        self._rng = random.Random(self.seed)
        self._expectations: Dict[str, float] = {}
        self._hover: Optional[HoverPlan] = None
        self._hover_expires_ms: int = 0
        self._perceive_heap: List[Tuple[int, int, PublicDeployEvent]] = []  # 优先队列：(可见时刻ms, event_id, 事件)
        self._visible_ids: set[int] = set()                                 # 已经对 AI 可见的事件 ID 集合
        self._latency_of: Dict[int, Tuple[LatencyClass, int]] = {}          # event_id -> (分类, 延迟ms)
        self._pending: List[ScheduledAction] = []                           # 等待物理执行的排期动作队列
        self._finger_free_at_ms: int = 0                                    # 手指恢复空闲的毫秒时刻
        self._seq: int = 0
        self._combo_seq: int = 0
        self.dropped: List[ScheduledAction] = []                            # 因超时未出而被废弃的动作

    def snapshot(self) -> Dict[str, Any]:
        """保存缓冲层完整状态快照"""
        state = {k: copy.deepcopy(v) for k, v in self.__dict__.items() if k not in ("_rng", "profile")}
        state["_rng_state"] = self._rng.getstate()
        return state

    def restore(self, snap: Mapping[str, Any]) -> None:
        """从快照还原缓冲层状态"""
        snap = dict(snap)
        rng_state = snap.pop("_rng_state")
        for k, v in snap.items():
            setattr(self, k, copy.deepcopy(v))
        self._rng = random.Random()
        self._rng.setstate(rng_state)

    # -- 感知层输入接口 ----------------------------------------------------
    def set_expectations(self, card_probs: Mapping[str, float]) -> None:
        """接收对手算水与猜牌器输出的在手概率，用于动态切换突袭/预判反应速度"""
        self._expectations = dict(card_probs)

    def arm_hover(self, plan: HoverPlan, now_ms: int) -> bool:
        """设置预悬停方案 (手指悬停等待)。若手指目前正在点击其他卡牌则设置失败"""
        if now_ms < self._finger_free_at_ms:
            return False
        if plan.target_mode == "fixed" and plan.tile is None:
            raise ValueError("fixed hover needs a tile")
        if plan.target_mode not in ("fixed", "event"):
            raise ValueError(f"bad target_mode {plan.target_mode}")
        self._hover = plan
        self._hover_expires_ms = plan.expires_at_ms if plan.expires_at_ms is not None else now_ms + self.profile.max_hover_ms
        return True

    def cancel_hover(self) -> None:
        """取消当前预悬停动作"""
        self._hover = None

    @property
    def hover(self) -> Optional[HoverPlan]:
        return self._hover

    def observe(self, event: PublicDeployEvent) -> LatencyClass:
        """
        接收模拟器中的一次公开下牌事件
        判断该事件是突发意外、预判攻击还是常规防守，采样得出需要多少毫秒的延迟后 AI 才能看清
        """
        if event.side == self.side:  # 己方出牌瞬间自知，无延迟
            self._mark_visible(event.event_id)
            self._latency_of[event.event_id] = (LatencyClass.OWN_INITIATIVE, 0)
            return LatencyClass.OWN_INITIATIVE

        # 检查是否命中了预悬停秒解触发条件
        reflex = self._try_hover_reflex(event)
        if reflex:
            cls, dist = LatencyClass.HOVER_REFLEX, self.profile.anticipated
        elif self._expectations.get(event.card_id, 0.0) >= self.profile.anticipation_threshold:
            cls, dist = LatencyClass.ANTICIPATED, self.profile.anticipated
        elif event.is_threat:
            cls, dist = LatencyClass.SURPRISE, self.profile.surprise
        else:
            cls, dist = LatencyClass.ROUTINE, self.profile.routine

        delay = dist.sample(self._rng)
        # 将事件推入小顶堆，到达 (t_ms + delay) 时才对策略网络真正解冻放行
        heapq.heappush(self._perceive_heap, (event.t_ms + delay, event.event_id, event))
        self._latency_of[event.event_id] = (cls, delay)
        return cls

    def _try_hover_reflex(self, ev: PublicDeployEvent) -> bool:
        """尝试触发预悬停秒解 (极速反射，120~200ms 内排期自动下牌)"""
        h = self._hover
        if h is None:
            return False
        if ev.t_ms > self._hover_expires_ms:
            self._hover = None
            return False
        if ev.card_id not in h.trigger_cards:
            return False
        if h.trigger_lane is not None and ev.tile.lane != h.trigger_lane:
            return False
        tile = h.tile if h.target_mode == "fixed" else ev.tile.offset(*h.offset)
        intent = ActionIntent(ActionKind.DEPLOY, h.card_id, tile, h.elixir_cost, source="hover")
        execute_at = ev.t_ms + self.profile.hover_reflex.sample(self._rng)
        self._schedule(intent, ev.t_ms, execute_at, LatencyClass.HOVER_REFLEX, trigger_event_id=ev.event_id)
        self._finger_free_at_ms = max(self._finger_free_at_ms, execute_at + self.profile.apm_gap.sample(self._rng))
        self._hover = None
        return True

    # -- 策略网络读取接口 --------------------------------------------------
    def pop_visible_events(self, now_ms: int) -> List[PublicDeployEvent]:
        """弹出所有在当前时刻 now_ms 已经度过生理延迟、可被策略网络看到的事件"""
        out: List[PublicDeployEvent] = []
        while self._perceive_heap and self._perceive_heap[0][0] <= now_ms:
            _, eid, ev = heapq.heappop(self._perceive_heap)
            self._mark_visible(eid)
            out.append(ev)
        return out

    def is_visible(self, event_id: int) -> bool:
        """检查特定事件 ID 是否已被 AI 察觉 (战争迷雾判定依据)"""
        return event_id in self._visible_ids

    def latency_of(self, event_id: int) -> Optional[Tuple[LatencyClass, int]]:
        """查询某次事件被 AI 察觉时的反应模式与延迟数值 (用于 UI 显示，如 'AI 在 470ms 后察觉')"""
        return self._latency_of.get(event_id)

    def _mark_visible(self, event_id: int) -> None:
        self._visible_ids.add(event_id)

    # -- 策略网络动作提交接口 ----------------------------------------------
    def submit(self, intents: Sequence[ActionIntent] | ActionIntent, now_ms: int) -> List[ScheduledAction]:
        """
        AI 决策提交动作意图
        按人类手指移动时间 (APM) 及组合连招间隔 (combo_gap_ms) 转换为具备生效时间的排期动作
        """
        if isinstance(intents, ActionIntent):
            intents = [intents]
        real = [i for i in intents if i.kind != ActionKind.NOOP]
        if not real:
            return []
        self._hover = None
        combo_id = None
        if len(real) > 1:
            self._combo_seq += 1
            combo_id = self._combo_seq
        t = max(now_ms, self._finger_free_at_ms)
        out = []
        for idx, intent in enumerate(real):
            if idx:
                t += self.profile.combo_gap_ms
            out.append(self._schedule(intent, now_ms, t, LatencyClass.OWN_INITIATIVE, combo_id=combo_id))
        self._finger_free_at_ms = t + self.profile.apm_gap.sample(self._rng)
        return out

    def _schedule(self, intent: ActionIntent, created: int, execute_at: int, cls: LatencyClass,
                  combo_id: Optional[int] = None, trigger_event_id: Optional[int] = None) -> ScheduledAction:
        """生成内部排期动作对象"""
        self._seq += 1
        a = ScheduledAction(self._seq, intent, created, execute_at, cls, combo_id, trigger_event_id)
        self._pending.append(a)
        return a

    # -- 模拟器执行交付接口 ------------------------------------------------
    def release_due(self, now_ms: int, elixir: Optional[float] = None,
                    is_legal: Optional[LegalityCheck] = None) -> List[ScheduledAction]:
        """
        交付所有到达执行时刻 execute_at_ms 的动作给模拟器执行
        若提供 elixir 参数，会累计扣除预算并阻塞费用不足的连招后续卡牌。
        超过 max_hold_ms 仍卡费无法打出的动作将被自动丢弃并记录到 dropped 中。
        """
        self._pending.sort(key=lambda a: (a.execute_at_ms, a.seq))
        released, keep = [], []
        blocked_combos: set[int] = set()
        budget = elixir
        for a in self._pending:
            if a.execute_at_ms > now_ms:
                keep.append(a)
                continue
            ok = a.combo_id is None or a.combo_id not in blocked_combos
            ok = ok and (budget is None or a.intent.elixir_cost <= budget + 1e-9)
            ok = ok and (is_legal is None or is_legal(a.intent))
            if ok:
                released.append(a)
                if budget is not None:
                    budget -= a.intent.elixir_cost
                continue
            if a.combo_id is not None:
                blocked_combos.add(a.combo_id)
            if now_ms - a.execute_at_ms >= self.profile.max_hold_ms:
                self.dropped.append(a)
            else:
                keep.append(a)
        self._pending = keep
        return released

    def pending(self) -> List[ScheduledAction]:
        """获取当前正在等待倒计时生效的所有动作"""
        return sorted(self._pending, key=lambda a: (a.execute_at_ms, a.seq))

    def finger_free_at_ms(self) -> int:
        """获取手指恢复空闲的最早时间时刻"""
        return self._finger_free_at_ms

    def next_wakeup_ms(self) -> Optional[int]:
        """获取缓冲区内最早需要唤醒推进的时间时刻 (支持事件驱动的快速仿真)"""
        times = [a.execute_at_ms for a in self._pending]
        if self._perceive_heap:
            times.append(self._perceive_heap[0][0])
        return min(times) if times else None
        if event.side == self.side:  # own plays are known instantly
            self._mark_visible(event.event_id)
            self._latency_of[event.event_id] = (LatencyClass.OWN_INITIATIVE, 0)
            return LatencyClass.OWN_INITIATIVE

        reflex = self._try_hover_reflex(event)
        if reflex:
            cls, dist = LatencyClass.HOVER_REFLEX, self.profile.anticipated
        elif self._expectations.get(event.card_id, 0.0) >= self.profile.anticipation_threshold:
            cls, dist = LatencyClass.ANTICIPATED, self.profile.anticipated
        elif event.is_threat:
            cls, dist = LatencyClass.SURPRISE, self.profile.surprise
        else:
            cls, dist = LatencyClass.ROUTINE, self.profile.routine

        delay = dist.sample(self._rng)
        heapq.heappush(self._perceive_heap, (event.t_ms + delay, event.event_id, event))
        self._latency_of[event.event_id] = (cls, delay)
        return cls

    def _try_hover_reflex(self, ev: PublicDeployEvent) -> bool:
        h = self._hover
        if h is None:
            return False
        if ev.t_ms > self._hover_expires_ms:
            self._hover = None
            return False
        if ev.card_id not in h.trigger_cards:
            return False
        if h.trigger_lane is not None and ev.tile.lane != h.trigger_lane:
            return False
        tile = h.tile if h.target_mode == "fixed" else ev.tile.offset(*h.offset)
        intent = ActionIntent(ActionKind.DEPLOY, h.card_id, tile, h.elixir_cost, source="hover")
        execute_at = ev.t_ms + self.profile.hover_reflex.sample(self._rng)
        self._schedule(intent, ev.t_ms, execute_at, LatencyClass.HOVER_REFLEX, trigger_event_id=ev.event_id)
        self._finger_free_at_ms = max(self._finger_free_at_ms, execute_at + self.profile.apm_gap.sample(self._rng))
        self._hover = None
        return True

    # -- outputs to the policy ---------------------------------------------
    def pop_visible_events(self, now_ms: int) -> List[PublicDeployEvent]:
        out: List[PublicDeployEvent] = []
        while self._perceive_heap and self._perceive_heap[0][0] <= now_ms:
            _, eid, ev = heapq.heappop(self._perceive_heap)
            self._mark_visible(eid)
            out.append(ev)
        return out

    def is_visible(self, event_id: int) -> bool:
        return event_id in self._visible_ids

    def latency_of(self, event_id: int) -> Optional[Tuple[LatencyClass, int]]:
        """(class, sampled delay ms) — used by the coach UI to show 'AI reacted in 470 ms'."""
        return self._latency_of.get(event_id)

    def _mark_visible(self, event_id: int) -> None:
        self._visible_ids.add(event_id)

    # -- inputs from the policy --------------------------------------------
    def submit(self, intents: Sequence[ActionIntent] | ActionIntent, now_ms: int) -> List[ScheduledAction]:
        """Policy decision at now_ms. A sequence of >1 intents is one combo (double/triple drop).
        Using the finger cancels any active hover."""
        if isinstance(intents, ActionIntent):
            intents = [intents]
        real = [i for i in intents if i.kind != ActionKind.NOOP]
        if not real:
            return []
        self._hover = None
        combo_id = None
        if len(real) > 1:
            self._combo_seq += 1
            combo_id = self._combo_seq
        t = max(now_ms, self._finger_free_at_ms)
        out = []
        for idx, intent in enumerate(real):
            if idx:
                t += self.profile.combo_gap_ms
            out.append(self._schedule(intent, now_ms, t, LatencyClass.OWN_INITIATIVE, combo_id=combo_id))
        self._finger_free_at_ms = t + self.profile.apm_gap.sample(self._rng)
        return out

    def _schedule(self, intent: ActionIntent, created: int, execute_at: int, cls: LatencyClass,
                  combo_id: Optional[int] = None, trigger_event_id: Optional[int] = None) -> ScheduledAction:
        self._seq += 1
        a = ScheduledAction(self._seq, intent, created, execute_at, cls, combo_id, trigger_event_id)
        self._pending.append(a)
        return a

    # -- outputs to the simulator ------------------------------------------
    def release_due(self, now_ms: int, elixir: Optional[float] = None,
                    is_legal: Optional[LegalityCheck] = None) -> List[ScheduledAction]:
        """Actions whose time has come. If `elixir` is given, costs are checked cumulatively
        (so a 4+3 combo with 5 elixir releases the 4 and holds the 3). Held actions are
        dropped after profile.max_hold_ms; later members of a held combo wait with it."""
        self._pending.sort(key=lambda a: (a.execute_at_ms, a.seq))
        released, keep = [], []
        blocked_combos: set[int] = set()
        budget = elixir
        for a in self._pending:
            if a.execute_at_ms > now_ms:
                keep.append(a)
                continue
            ok = a.combo_id is None or a.combo_id not in blocked_combos
            ok = ok and (budget is None or a.intent.elixir_cost <= budget + 1e-9)
            ok = ok and (is_legal is None or is_legal(a.intent))
            if ok:
                released.append(a)
                if budget is not None:
                    budget -= a.intent.elixir_cost
                continue
            if a.combo_id is not None:
                blocked_combos.add(a.combo_id)
            if now_ms - a.execute_at_ms >= self.profile.max_hold_ms:
                self.dropped.append(a)
            else:
                keep.append(a)
        self._pending = keep
        return released

    def pending(self) -> List[ScheduledAction]:
        return sorted(self._pending, key=lambda a: (a.execute_at_ms, a.seq))

    def finger_free_at_ms(self) -> int:
        return self._finger_free_at_ms

    def next_wakeup_ms(self) -> Optional[int]:
        """Earliest time anything in the buffer changes (for event-driven stepping)."""
        times = [a.execute_at_ms for a in self._pending]
        if self._perceive_heap:
            times.append(self._perceive_heap[0][0])
        return min(times) if times else None
