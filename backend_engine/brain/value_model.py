"""
胜率价值模型 (Value Model)

计算局势估值函数：
    V(state) = P(side wins | state) ∈ [0.0, 1.0]

用于驱动赛后胜率走势折线图 (Win-rate graph) 以及评估每步出牌优劣 (MCTS 终局奖励评估)。

算法原理：
- 基于手工提炼的高层态势特征，通过逻辑回归 (Logistic Regression) 拟合自对弈数据。
- 所有特征均定义为“我方指标 减去 对方指标”的对称差分：
    f_i(s, side=1) = -f_i(s, side=0)
- 由此保证数学性质完全对称：
    V(side=1) = 1.0 - V(side=0)
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence

from backend_engine.simulator import geometry as G
from backend_engine.simulator.engine import Match

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "value_model.json"

# 特征工程维度列表 (10 个物理态势特征)
FEATURES = [
    "crowns",            # 皇冠净胜差 (我方皇冠 - 敌方皇冠)
    "crowns_late",       # 皇冠净胜差 × 残局时间加权 (越到后期破塔权重越高)
    "princess_hp",       # 两座公主塔剩余血量百分比之和差值 (-2.0 ~ +2.0)
    "king_hp",           # 国王塔剩余血量百分比差值 (-1.0 ~ +1.0)
    "min_tower",         # 最低血量存活防御塔的血量比例差 (官方破血决胜 tiebreaker 核心依据)
    "min_tower_late",    # 最低血量防御塔差值 × 残局时间加权
    "elixir",            # 双方圣水存量差值 / 10.0
    "board",             # 场上存活部队与建筑的圣水残存价值差 / 10.0
    "pressure",          # 压入对手半场的部队圣水残存价值差 / 10.0 (攻防压迫感)
    "princess_late",     # 公主塔血量差值 × 残局时间加权
]


def features(m: Match, side: int = 0) -> List[float]:
    """提取当前对局状态关于指定阵营 side 的 10 维对称差分特征向量"""
    me, op = side, 1 - side
    # 残局加权因子：比赛越接近尾声，late 越趋近 1.0
    late = 1.0 - max(0, m.end_ms - m.t_ms) / m.end_ms
    cr = m.sides[me].crowns - m.sides[op].crowns

    def towers(s):
        out = {}
        for role in ("princess_left", "princess_right", "king"):
            t = m.tower(s, role)
            out[role] = t.hp / t.max_hp if t is not None else 0.0
        return out

    tm, to = towers(me), towers(op)
    princess = (tm["princess_left"] + tm["princess_right"]) - (to["princess_left"] + to["princess_right"])
    king = tm["king"] - to["king"]
    standing = lambda d: min([v for v in d.values() if v > 0] or [0.0])  # noqa: E731
    min_tower = standing(tm) - standing(to)

    # 场上活体单位圣水残值折算：单位圣水 = 单卡费用 / 产出单位数 * (当前血量 / 最大血量)
    board = [0.0, 0.0]
    pressure = [0.0, 0.0]
    for u in m.units.values():
        if u.kind == "tower" or u.hp <= 0:
            continue
        card = m.db.cards.get(u.card_id) or m.db.tokens.get(u.card_id, {})   # tokens are worth their parent card
        value = card.get("elixir", 0) / max(1, card.get("count", 1)) * (u.hp / u.max_hp)
        k = 0 if u.side == me else 1
        board[k] += value
        # 单位跨过河道进入对方半场算作前线压迫值 (pressure)
        if G.region(u.y) == G.home_region(1 - u.side):
            pressure[k] += value
    return [
        cr, cr * late, princess, king, min_tower, min_tower * late,
        (m.sides[me].elixir - m.sides[op].elixir) / 10.0,
        (board[0] - board[1]) / 10.0,
        (pressure[0] - pressure[1]) / 10.0,
        princess * late,
    ]


@dataclass
class ValueModel:
    """逻辑回归局势价值评估模型"""
    weights: List[float]
    meta: dict

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> "ValueModel":
        """从 JSON 配置文件加载训练好的权重参数"""
        p = Path(path)
        if not p.exists():
            return cls.fallback()
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("features") != FEATURES:
            return cls.fallback()
        return cls(d["weights"], d.get("meta", {}))

    @classmethod
    def fallback(cls) -> "ValueModel":
        """内置默认手工专家先验权重 (供未运行自对弈训练时使用)"""
        return cls([1.5, 3.0, 1.2, 2.0, 1.0, 3.0, 0.8, 0.6, 0.8, 1.5], {"trained": False})

    def logit(self, f: Sequence[float]) -> float:
        """计算特征线性加权和 z = W · X"""
        return sum(w * x for w, x in zip(self.weights, f))

    def value(self, m: Match, side: int = 0) -> float:
        """评估指定阵营获胜概率 V(m, side) ∈ [0.0, 1.0]
        
        若对局已终局：
        - 平局输出 0.5
        - side 胜输出 1.0，side 负输出 0.0
        进行中的对局：
        - 计算 Sigmoid 函数: σ(z) = 1 / (1 + e^(-z))，并限制 z 在 [-30, 30] 防止浮点溢出
        """
        if m.result is not None:
            if m.result.winner is None:
                return 0.5
            return 1.0 if m.result.winner == side else 0.0
        z = max(-30.0, min(30.0, self.logit(features(m, side))))
        return 1.0 / (1.0 + math.exp(-z))


_DEFAULT: Optional[ValueModel] = None


def default_value_model() -> ValueModel:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = ValueModel.load()
    return _DEFAULT
