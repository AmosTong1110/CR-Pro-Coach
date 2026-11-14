"""
贝叶斯对手卡组推断引擎 (Bayesian Opponent Deck Guesser)
根据对手在比赛中已打出的 1~7 张明牌，精准推断对手其余尚未露面的暗置卡牌及最可能的完整天梯主流卡组。

统计学推断模型 (Probabilistic Model):
  1. 先验分布 (Prior):
     海量真实天梯高分对局卡组库中每个卡组 d 的出现频次作为权重 w_d。
  2. 似然度 (Evidence/Likelihood):
     已展示的卡牌集合 R。若 R ⊆ d 则似然为 1，否则为 0 (皇室战争卡组一旦确定绝不可能临时伪造换牌)。
  3. 后验概率融合 (Posterior Blending):
     - 完整卡组后验：P(d | R) ∝ w_d · 1[R ⊆ d]  ->  P_deck(c) = Σ_{d ∋ c} P(d | R)
     - 单卡共现平滑 (Co-occurrence)：当匹配到的样本极少 (遇到冷门绝活哥或冷门杂技套) 时，
       避免出现过度自信的极值，引入条件共现概率：
       P_co(c) = mean_{r ∈ R} P(c ∈ deck | r ∈ deck)
     - 混合调和：P(c) = λ · P_deck(c) + (1-λ) · P_co(c), 其中 λ = W / (W + kappa), W 为匹配总权重。
  4. 注水归一化算法 (Water-filling):
     将所有未知卡牌的概率加和归一化至剩余暗牌槽位数 (8 - |R|)，且每张卡概率严格约束在 [0, 1] 之间。
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

# 默认的天梯主流卡组数据集路径
DEFAULT_META = Path(__file__).resolve().parents[2] / "configs" / "meta_decks_hf.json"
DECK_SIZE = 8


@dataclass(frozen=True)
class DeckGuess:
    """对手卡组贝叶斯猜牌结果"""
    revealed: Tuple[str, ...]                        # 对手当前已打出的明牌元组
    card_probs: Dict[str, float]                     # 未知卡牌 -> 属于该卡组的后验概率 (按降序排列)
    likely_decks: List[Tuple[Tuple[str, ...], float]] = field(default_factory=list) # 最匹配的完整卡组及对应后验置信度
    matching_weight: float = 0.0                     # 样本置信权重 (天梯库中符合条件的样本总数)

    def top(self, n: int = 4) -> List[Tuple[str, float]]:
        """获取概率最高的前 n 张可能携带的未知卡牌"""
        return list(self.card_probs.items())[:n]

    def in_hand_probs(self, unknown_hand_slots: int) -> Dict[str, float]:
        """
        计算某张未知卡牌【当前正处于对手手中】的概率
        对手手牌中的未知槽位由卡组中未露面的暗牌均匀填充
        """
        unknown = DECK_SIZE - len(self.revealed)
        if unknown <= 0 or unknown_hand_slots <= 0:
            return {c: 0.0 for c in self.card_probs}
        share = unknown_hand_slots / unknown
        return {c: p * share for c, p in self.card_probs.items()}


class BayesianDeckGuesser:
    """贝叶斯卡组推断器核心类"""

    def __init__(self, decks: Iterable[Tuple[Sequence[str], float]], alpha: float = 0.0, kappa: float = 4.0):
        self.decks: List[Tuple[FrozenSet[str], float]] = []
        for cards, w in decks:
            s = frozenset(cards)
            if len(s) == DECK_SIZE:
                self.decks.append((s, float(w) + alpha))
        if not self.decks:
            raise ValueError("deck prior is empty")
        self.kappa = kappa             # 平滑超参数：权重 W 越小，向共现先验退化越多
        self.total = sum(w for _, w in self.decks)
        self.usage: Counter = Counter()
        self.pair: Dict[str, Counter] = defaultdict(Counter)
        # 统计全天梯单卡出场率与双卡两两共现频率
        for d, w in self.decks:
            for c in d:
                self.usage[c] += w
                for o in d:
                    if o != c:
                        self.pair[c][o] += w
        self.cards = set(self.usage)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_META, **kw) -> "BayesianDeckGuesser":
        """从 JSON 格式的天梯卡组库直接加载先验数据"""
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(((d["cards"], d["count"]) for d in data["decks"]), **kw)

    # ------------------------------------------------------------------
    def _p_co(self, c: str, revealed: Sequence[str]) -> float:
        """计算单卡基于已明牌的平均条件共现率 P(c | R)"""
        known = [r for r in revealed if self.usage[r] > 0]
        if not known:
            return self.usage[c] / self.total
        return sum(self.pair[r][c] / self.usage[r] for r in known) / len(known)

    def guess(self, revealed: Iterable[str], exclude: Iterable[str] = (), top_decks: int = 3) -> DeckGuess:
        """
        核心推断方法：
        根据已揭晓的卡牌集合 revealed，推算剩余暗牌与最相似卡组
        """
        R = tuple(dict.fromkeys(revealed))        # 保留出牌顺序且去重
        rset = frozenset(R)
        slots = DECK_SIZE - len(rset)
        banned = set(exclude) | rset
        # 若对手 8 张卡全部已打出，推断完全确定
        if slots <= 0:
            return DeckGuess(R, {}, [(tuple(sorted(rset)), 1.0)], self.total)

        # 1. 过滤所有包含当前明牌集合 R 的天梯卡组候选
        matches = [(d, w) for d, w in self.decks if rset <= d]
        W = sum(w for _, w in matches)
        p_deck: Counter = Counter()
        for d, w in matches:
            for c in d - rset:
                p_deck[c] += w / W
        # 2. 混合系数 lambda：样本越多，越信任完整卡组分布；样本越少，越偏向共现网络
        lam = W / (W + self.kappa) if W > 0 else 0.0

        # 3. 综合概率计算
        probs: Dict[str, float] = {}
        for c in self.cards - banned:
            p = lam * p_deck.get(c, 0.0) + (1 - lam) * self._p_co(c, R)
            if p > 1e-4:
                probs[c] = p
        # 4. 注水法重整化：确保概率总和等于剩余暗牌槽位
        probs = _rescale_to(probs, float(slots))

        likely = sorted(matches, key=lambda m: -m[1])[:top_decks]
        likely_out = [(tuple(sorted(d)), w / W) for d, w in likely] if W else []
        ordered = dict(sorted(probs.items(), key=lambda kv: (-kv[1], kv[0])))
        return DeckGuess(R, ordered, likely_out, W)


COSTS_PATH = Path(__file__).resolve().parents[2] / "configs" / "card_costs_hf.json"
_DEFAULT: Optional[BayesianDeckGuesser] = None
_COSTS: Optional[Dict[str, int]] = None


def default_guesser() -> Optional[BayesianDeckGuesser]:
    """获取单例全局默认猜牌器 (根据 configs/meta_decks_hf.json 构建)"""
    global _DEFAULT
    if _DEFAULT is None and DEFAULT_META.exists():
        _DEFAULT = BayesianDeckGuesser.load()
    return _DEFAULT


def known_costs() -> Dict[str, int]:
    """获取真实游戏中所有卡牌的圣水花费 (包含模拟器暂未完全实现的扩展卡牌)"""
    global _COSTS
    if _COSTS is None:
        _COSTS = json.loads(COSTS_PATH.read_text(encoding="utf-8"))["costs"] if COSTS_PATH.exists() else {}
    return _COSTS


def _rescale_to(probs: Dict[str, float], target: float, iters: int = 20) -> Dict[str, float]:
    """
    注水缩放算法 (Water-filling Rescaling)
    将所有候选卡牌的概率之和缩放至目标槽位数 target，同时保证任何单卡的概率值都不超过 1.0 上限。
    """
    probs = dict(probs)
    for _ in range(iters):
        free = {c: p for c, p in probs.items() if p < 1.0}
        fixed = len(probs) - len(free)
        s = sum(free.values())
        if s <= 0:
            break
        k = (target - fixed) / s
        if abs(k - 1) < 1e-9:
            break
        for c in free:
            probs[c] = min(1.0, free[c] * k)
    return probs
