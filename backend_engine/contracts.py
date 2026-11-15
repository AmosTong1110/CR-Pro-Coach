"""
数据契约层 (Data Contracts Layer)
定义模拟器 (Simulator)、感知层 (Perception) 与 AI 大脑 (Brain) 之间共享的核心数据契约。

设计规范 (Conventions - 贯穿整个 backend_engine):
  * 时间均为整数【仿真毫秒】(`t_ms`)，绝非系统物理时间。
    无论是暂停、2倍速播放、撤销接管、MCTS 快速推演，还是云端 1000x 强化学习，均通过推进该毫秒值驱动。
  * 坐标系统：竞技场为网格系统，x 范围 [0, 18)，y 范围 [0, 32)。
    y=0 位于 side 0（玩家）国王塔底部，y=31.99 位于 side 1（AI）国王塔顶部。
  * side 0 = 底部阵营（UI 中的人类玩家）；side 1 = 顶部阵营（AI 对手）。
  * 此处所有数据结构均为不可变 dataclass，可无缝序列化为 JSON，支持录像回放、UI 通信及状态快照。

单个对手出牌事件的数据流向图 (以 AI 视角为例):

    simulator (模拟器)  --PublicDeployEvent-->  OppElixirTracker (算水与推牌序)  --ElixirEstimate--+
            |                                                                              |
            +--PublicDeployEvent--> HumanDelayBuffer.observe() (人类生理延迟缓冲) <-- set_expectations()
                                          |   (感知延迟 / 悬停秒解)
                                          v
                           pop_visible_events()  -> observation builder -> policy (策略网络)
                                                                             |
                           HumanDelayBuffer.submit(ActionIntent...)  <-------+
                                          |   (手指移动物理延迟 / APM 动作上限)
                                          v
                           release_due() -> ScheduledAction -> simulator.apply() (生效执行)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Optional, Tuple

# 竞技场标准网格尺寸：宽 18 格，高 32 格
ARENA_W_TILES = 18
ARENA_H_TILES = 32


@dataclass(frozen=True)
class TilePos:
    """网格坐标对象 (浮点数，支持半格/微格精度计算)"""
    x: float
    y: float

    @property
    def lane(self) -> str:
        """根据 x 坐标判断所在分路：左路 (left) 或 右路 (right)"""
        return "left" if self.x < ARENA_W_TILES / 2 else "right"

    def offset(self, dx: float, dy: float) -> "TilePos":
        """在竞技场合法网格边界内进行相对坐标偏移"""
        return TilePos(
            min(max(self.x + dx, 0.0), ARENA_W_TILES - 0.5),
            min(max(self.y + dy, 0.0), ARENA_H_TILES - 0.5),
        )


VARIANTS = ("base", "evo", "hero")


@dataclass(frozen=True)
class CardRef:
    """A card plus its variant. String form: "knight" or "knight#evo" (logs, URLs, datasets)."""
    card_id: str
    variant: str = "base"

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}")

    def __str__(self) -> str:
        return self.card_id if self.variant == "base" else f"{self.card_id}#{self.variant}"

    @classmethod
    def parse(cls, text: str) -> "CardRef":
        cid, _, variant = text.strip().partition("#")
        return cls(cid, variant or "base")


# --------------------------------------------------------------------------
# 模拟器 -> 感知层/大脑：对局公开下牌事件
# 严格遵守人类玩家在屏幕上看到的合法信息，绝对不泄露对手暗置手牌（防作弊）
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PublicDeployEvent:
    """单次下牌公开事件 (无论人类还是 AI 出牌均生成此事件)"""
    event_id: int            # 每局自增 ID
    t_ms: int                # 该卡牌在竞技场画面出现的仿真毫秒时刻
    side: int                # 出牌所属方 (0=人类玩家, 1=AI)
    card_id: str             # 卡牌英文唯一标识符 (对应 configs 中的 key)
    elixir_cost: int         # 实际消耗圣水 (含英雄技能费用)
    tile: TilePos            # 下牌落点坐标
    is_evolved: bool = False # 是否为觉醒形态出牌
    is_ability: bool = False # 是否为英雄/冠军技能触发按钮（而非传统手牌）
    is_threat: bool = False  # 是否为威胁攻击单位（落点靠近河道或直接空降半场）
    variant: str = "base"    # "base" | "evo" | "hero" (kept in sync with is_evolved)

    def __post_init__(self) -> None:
        if self.variant not in VARIANTS:
            raise ValueError(f"unknown variant {self.variant!r}")
        if self.is_evolved and self.variant == "base":
            object.__setattr__(self, "variant", "evo")
        elif self.variant == "evo" and not self.is_evolved:
            object.__setattr__(self, "is_evolved", True)

    @property
    def card_ref(self) -> "CardRef":
        return CardRef(self.card_id, self.variant)


# --------------------------------------------------------------------------
# opp_elixir_count.py -> 大脑/UI
# 圣水推算与卡序估计数据契约
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ElixirEstimate:
    """对手圣水与卡牌流转状态的概率推算结果"""
    t_ms: int
    point: float                       # 圣水点估计值 (最佳期望预测)
    low: float                         # 圣水理论下界 (除非漏测事件，否则不可能低于此值)
    high: float                        # 圣水理论上界 (最大为 10，除非刚刚消耗)
    known_cards: Tuple[str, ...]       # 迄今为止已明牌的卡牌列表 (最多 8 张)
    cycle_order: Tuple[str, ...]       # 历史出牌序列，按时间从旧到新
    in_hand_prob: Dict[str, float] = field(default_factory=dict)  # 卡牌 -> 当前在手牌中的后验概率
    multiplier: int = 1                # 当前圣水充能倍率 (1x 常规, 2x 双倍, 3x 三倍)
    # 牌序流转：队列中等待回手的 4 张牌，索引 0 为最先打出（下一张将回手），None 表示未知卡
    queue: Tuple[Optional[str], ...] = ()
    known_in_hand: Tuple[str, ...] = ()   # 当前百分之百在手牌中的已知明牌
    unknown_hand_slots: int = 4           # 手牌中尚未揭晓的暗牌槽位数
    leaked: float = 0.0                   # 对手在 10 水满费状态下浪费/溢出的圣水总量
    evo_progress: Dict[str, int] = field(default_factory=dict)  # 觉醒充能计数：卡牌 -> 距离下次觉醒已打出的次数
    resyncs: int = 0                      # 状态重同步次数 (当实际出牌与推算矛盾时的纠错计数)

    @property
    def next_return(self) -> Optional[str]:
        """对手打出下一张牌时，立即回手补充进手牌的卡牌"""
        return self.queue[0] if self.queue else None

    def p_affordable(self, cost: int) -> float:
        """计算对手当前能支付该费用的概率 P(elixir >= cost)，假定圣水在 [low, high] 上均匀分布"""
        if cost <= self.low:
            return 1.0
        if cost > self.high:
            return 0.0
        span = max(self.high - self.low, 1e-6)
        return (self.high - cost) / span

    def playable_probs(self, costs: Dict[str, int]) -> Dict[str, float]:
        """计算每张卡牌【当前既在手里且圣水足够打出】的综合概率，用于人类反应缓冲区的预判加权"""
        return {c: p * self.p_affordable(costs.get(c, 10)) for c, p in self.in_hand_prob.items()}


# --------------------------------------------------------------------------
# AI 大脑 <-> 人类反应延迟缓冲 (HumanDelayBuffer) <-> 模拟器执行
# --------------------------------------------------------------------------
class ActionKind(str, Enum):
    """动作意图类型"""
    DEPLOY = "deploy"     # 放置手牌
    ABILITY = "ability"   # 释放英雄技能
    NOOP = "noop"         # 显式空操作 (等待/存费，不提交给物理执行队列)


class LatencyClass(str, Enum):
    """人类生理延迟分类 (反映不同情境下的反应速度)"""
    SURPRISE = "surprise"              # 突发意外威胁 (桥头速攻/隐身偷袭，反应耗时约 450ms)
    ANTICIPATED = "anticipated"        # 预判中攻击 (算水/猜卡已推算出的进攻，反应耗时约 250ms)
    ROUTINE = "routine"                # 常规非紧急出牌 (沉底沉兵，反应耗时约 350ms)
    HOVER_REFLEX = "hover_reflex"      # 预选悬停秒解 (手指已悬停在克制卡上，极致反应 150ms)
    OWN_INITIATIVE = "own_initiative"  # 主动发起进攻 (不受敌方事件触发，自主决策)


@dataclass(frozen=True)
class ActionIntent:
    """动作意图 (大脑或策略网络决定的行动雏形，需通过延迟队列后才能转化为物理动作)"""
    kind: ActionKind
    card_id: Optional[str] = None
    tile: Optional[TilePos] = None
    elixir_cost: int = 0
    source: str = "policy"       # 意图来源："policy"(策略网络) | "mcts"(搜索) | "hover"(秒解) | "script"(脚本)


@dataclass(frozen=True)
class HoverPlan:
    """预悬停秒解方案 (单指悬停预判：预先将卡牌悬停在特定位置等待触发条件)"""
    card_id: str
    elixir_cost: int
    trigger_cards: FrozenSet[str]        # 触发该秒解反应的目标卡牌集合 (如敌方出哥布林桶/野猪)
    target_mode: str = "fixed"           # 落点模式："fixed" 固定坐标；"event" 相对触发事件动态偏移
    tile: Optional[TilePos] = None
    offset: Tuple[float, float] = (0.0, 0.0)
    trigger_lane: Optional[str] = None   # 监听路别："left" 左路 | "right" 右路 | None 全场
    expires_at_ms: Optional[int] = None  # 悬停等待超时时刻 (毫秒)


@dataclass(frozen=True)
class ScheduledAction:
    """已排期的物理动作 (已完成生理延迟缓冲，到达 execute_at_ms 时即可被模拟器执行)"""
    seq: int                             # 动作排期自增序号
    intent: ActionIntent                 # 对应的原始动作意图
    created_at_ms: int                   # 产生意图的毫秒时刻
    execute_at_ms: int                   # 允许物理下牌生效的毫秒时刻
    latency_class: LatencyClass          # 所采用的人类生理延迟分类
    combo_id: Optional[int] = None       # 组合连招 ID (如先下肉盾紧接着跟后排)
    trigger_event_id: Optional[int] = None # 触发本次反应的敌方事件 ID
