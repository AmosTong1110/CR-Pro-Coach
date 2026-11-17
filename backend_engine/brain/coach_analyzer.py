"""
赛后复盘与出牌评级分析引擎 (Post-Match Coach & Blunder Analysis Engine)
生成整场对局的胜率起伏曲线，并像国际象棋引擎 (如 Chess.com Review) 一样对玩家的每一步出牌进行严格打分：
  * !! Brilliant (妙手) / ★ Best (最佳) / Good (好棋)
  * ?! Inaccuracy (缓手) / ? Mistake (失误) / ?? Blunder (大错/漏杀)
同时计算出引擎推演出的【最佳推荐下法】(Ghost Tile) 与落点。

出牌评级推演工作流 (Move Grading Workflow):
  1. 状态还原：回溯到出牌前一瞬间的精确快照检查点 (Checkpoint)。
  2. 生成候选打法 (Candidate Generation)：
     - 手里所有费用足够的卡牌 x 战术合理落点 (双路分推位、桥头突进、防守黄金拉扯位、法术砸群伤/斩杀位)
     - 特别包含 "Wait" (等待/存费/不过牌)。
     - 玩家实际下出的真实打法总是被囊括在候选集合中。
  3. 多分支前向推演 (MCTS Horizon Rollout)：
     - 为每个候选动作克隆分支副本，下出该卡牌；
     - 让对手 AI 应对推演 `horizon_ms` (默认 8000 毫秒 / 8秒)；
     - 推演结束时调用胜率价值模型 (ValueModel) 给出当前局面的获胜期望值 V ∈ [0, 1]。
  4. 胜率差损失计算：loss = V(best) - V(actual)
     - 根据胜率落差确定等级 (Blunder / Mistake / Inaccuracy / Best / Brilliant)。
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

from backend_engine.brain.value_model import ValueModel, default_value_model
from backend_engine.simulator import geometry as G
from backend_engine.simulator.engine import Match

# 状态分支回调函数类型：checkpoint -> (match, runner) 克隆副本
Fork = Callable[[Any], Tuple[Match, Any]]

# 胜率落差阈值 (以胜率百分点为单位)
INACCURACY = 0.05    # 胜率损失 >= 5% 判定为缓手
MISTAKE = 0.10       # 胜率损失 >= 10% 判定为失误
BLUNDER = 0.20       # 胜率损失 >= 20% 判定为严重失误/大错
BEST_EPS = 0.01      # 胜率与理论最优差距 <= 1% 判定为最佳
BRILLIANT_GAIN = 0.12 # 相比中位数预期逆风翻盘提升 >= 12% 判定为妙手

# 评级符号与文本标签映射表
LABELS = {
    "brilliant":  ("!!", "Brilliant"),
    "best":       ("★", "Best"),
    "good":       ("", "Good"),
    "inaccuracy": ("?!", "Inaccuracy"),
    "mistake":    ("?", "Mistake"),
    "blunder":    ("??", "Blunder"),
}

# 常见核心防御战术点位 (半格精度: 桥头黄金防守位、主塔前居中拉扯位、沉底分路位)
TROOP_SPOTS = [
    (3.5, 13.5), (14.5, 13.5), # 左右桥头近端
    (3.5, 9.5),  (14.5, 9.5),  # 左右公主塔正前方防守位
    (3.5, 4.5),  (14.5, 4.5),  # 左右沉底防守位
    (9.5, 9.5),                # 场地正中央经典拉扯黄金位 (拉野猪/大皮卡进双塔射程)
    (8.5, 5.5)                 # 国王塔前防守位
]


@dataclass(frozen=True)
class Candidate:
    """单种候选打法 (包含卡牌与目标网格坐标；若 card_id 为 None 则表示等待/存费)"""
    card_id: Optional[str]           # None 表示此时选择保留圣水、不出牌
    x: float = 0.0
    y: float = 0.0

    @property
    def is_wait(self) -> bool:
        """是否为空操作等待"""
        return self.card_id is None


# ---------------------------------------------------------------------------
# 战术失误根因分类字典 (Diagnostic Tags Metadata)
# ---------------------------------------------------------------------------
DIAGNOSTIC_META = {
    "overcommitted": {
        "name": "Overcommitted",
        "tag": "⚡ Overcommitted",
        "color": "#e74c3c",     # Red
        "badge_bg": "#481b1b",
        "icon": "⚡",
        "description": "Spent elixir when waiting or preserving defense was critical.",
    },
    "bad_placement": {
        "name": "Bad Placement",
        "tag": "🎯 Bad Placement",
        "color": "#e67e22",     # Orange
        "badge_bg": "#462912",
        "icon": "🎯",
        "description": "Right card deployed at suboptimal tile or wrong lane.",
    },
    "ignored_threat": {
        "name": "Ignored Threat",
        "tag": "⚠️ Ignored Threat",
        "color": "#d35400",     # Dark amber
        "badge_bg": "#422110",
        "icon": "⚠️",
        "description": "Attacked while high-priority enemy troop required immediate defense.",
    },
    "negative_trade": {
        "name": "Negative Trade",
        "tag": "📉 Negative Trade",
        "color": "#9b59b6",     # Purple
        "badge_bg": "#371842",
        "icon": "📉",
        "description": "Overspent elixir compared to optimal cheaper answer.",
    },
    "missed_cycle": {
        "name": "Missed Cycle",
        "tag": "🔄 Missed Cycle",
        "color": "#1abc9c",     # Teal
        "badge_bg": "#123b34",
        "icon": "🔄",
        "description": "Failed to cycle cheap cards to reach necessary response.",
    },
    "wrong_response": {
        "name": "Wrong Answer",
        "tag": "❌ Wrong Answer",
        "color": "#3498db",     # Blue
        "badge_bg": "#162e45",
        "icon": "❌",
        "description": "Suboptimal counter card for the current enemy board.",
    },
}


def diagnose_move(
    mv: Any,
    db: Optional[Any] = None,
    match_before: Optional[Any] = None,
    side: int = 0,
) -> Optional[str]:
    """
    根据局面与出牌落差，精准诊断失误产生的战术根因。
    返回: 'overcommitted' | 'bad_placement' | 'ignored_threat' | 'negative_trade' | 'missed_cycle' | 'wrong_response' | None
    """
    stored = getattr(mv, "diagnostic", None)
    if stored:
        return stored

    loss = getattr(mv, "loss", 0.0)
    label = getattr(mv, "label", "good")
    if loss <= 0.01 or label in ("best", "brilliant", "good"):
        return None

    actual = getattr(mv, "actual", None)
    best = getattr(mv, "best", None)
    if actual is None or best is None:
        return None

    if actual.is_wait:
        return "overcommitted"

    # 1. 牌选对但位置明显放偏 (Bad Placement)
    if not best.is_wait and best.card_id == actual.card_id:
        return "bad_placement"

    costs: dict = {}
    if db is not None:
        if hasattr(db, "elixir_costs"):
            costs = db.elixir_costs()
        elif hasattr(db, "cards"):
            costs = {cid: c.get("elixir", 3) for cid, c in db.cards.items()}
        elif isinstance(db, dict):
            costs = {cid: (c.get("elixir", 3) if isinstance(c, dict) else 3) for cid, c in db.items()}

    actual_cost = costs.get(actual.card_id, 3) if actual.card_id else 0
    best_cost = costs.get(best.card_id, 0) if best.card_id else 0

    # 2. 漏防严重威胁：敌方单位已兵临城下 (y <= 16.5)，玩家却在异路或敌方半场强攻 (y >= 16.0)
    if match_before is not None and hasattr(match_before, "units"):
        enemy_threats = [
            u for u in match_before.units.values()
            if getattr(u, "side", 0) != side and getattr(u, "hp", 0) > 0
            and getattr(u, "kind", "") == "troop" and getattr(u, "y", 32) <= 16.5
        ]
        if enemy_threats and actual.y >= 16.0:
            return "ignored_threat"

    v_wait = getattr(mv, "v_wait", 0.0)
    v_actual = getattr(mv, "v_actual", 0.0)

    # 3. 圣水透支：最优打法是等待存费 (best.is_wait)
    if best.is_wait:
        return "overcommitted"

    # 4. 严重亏费 (Negative Elixir Trade)：下出 >=5 费高费大牌解场 (如火箭/大电) 造成严重圣水赤字
    if best.card_id and (actual_cost >= 5 and actual_cost - best_cost >= 2):
        return "negative_trade"

    # 5. 错失低费过牌 (Missed Cycle)：最佳打法是过 1-2 费杂毛/过牌卡，玩家却下了 3-4 费中费牌
    if best.card_id and best_cost <= 2 and actual_cost <= 4:
        return "missed_cycle"

    # 6. 其他亏费解场 (差额 >= 2)
    if best.card_id and (actual_cost - best_cost >= 2):
        return "negative_trade"

    # 7. 等待明显强于出重费牌
    if v_wait > v_actual + 0.04 and actual_cost >= 4:
        return "overcommitted"

    # 8. 其余卡牌属性不对位导致的解卡失误 (Wrong Answer)
    return "wrong_response"


@dataclass(frozen=True)
class MoveReview:
    """单步出牌的深度赛后复盘评级报告"""
    event_id: int                    # 对应的出牌事件 ID
    t_ms: int                        # 出牌时刻 (仿真毫秒)
    actual: Candidate                # 玩家实际下出的牌与位置
    v_before: float                  # 出牌瞬间之前的我方胜率
    v_actual: float                  # 玩家该打法在 8 秒后的局面期望胜率
    v_best: float                    # 引擎推演出的理论最优解期望胜率
    v_wait: float                    # 若此时选择什么都不做 (纯等待) 的期望胜率
    best: Candidate                  # 引擎推荐的最优打法 (可为最佳下牌或建议存费)
    loss: float                      # 胜率损失 (v_best - v_actual)
    label: str                       # 评级等级："brilliant" | "best" | "good" | "inaccuracy" | "mistake" | "blunder"
    n_candidates: int                # 本次横向比对推演的候选打法总数量
    alternatives: Tuple[Tuple[Candidate, float], ...] = ()   # 评分最高的前几种备选方案及其胜率
    diagnostic: Optional[str] = None # 战术根因诊断："overcommitted" | "bad_placement" | "ignored_threat" | "negative_trade" | "missed_cycle" | "wrong_response"

    @property
    def symbol(self) -> str:
        """评级国际象棋符号 (如 !!, ★, ?, ??)"""
        return LABELS[self.label][0]

    @property
    def label_text(self) -> str:
        """评级英文全称文本"""
        return LABELS[self.label][1]


@dataclass
class MatchReview:
    """整场比赛完整复盘报告"""
    curve: List[Tuple[int, float]]           # 随时间推移的我方胜率曲线 [(时刻ms, 胜率), ...]
    moves: List[MoveReview]                  # 针对人类玩家每一次出牌的评级列表
    final: Optional[float] = None            # 比赛终局状态胜率 (1.0 胜, 0.0 负, 0.5 平)
    model_meta: dict = field(default_factory=dict) # 胜率估值模型的元数据信息

    @property
    def accuracy(self) -> float:
        """
        类似 Chess.com 准确度评分 (0 ~ 100 分)
        对玩家每一步的胜率落差加权求均值：零失误为 100 分，单步落差 >= 30% 计为 0 分
        """
        if not self.moves:
            return 0.0
        return 100.0 * statistics.mean(max(0.0, 1.0 - m.loss / 0.3) for m in self.moves)

    def counts(self) -> dict:
        """统计各类评级的出牌总数 (如产生了多少次 Blunder，多少次 Brilliant)"""
        out = {k: 0 for k in LABELS}
        for m in self.moves:
            out[m.label] += 1
        return out

    def diagnostics_breakdown(self, db: Optional[Any] = None) -> Dict[str, int]:
        """统计整局各类失误成因的出现次数"""
        out: Dict[str, int] = {}
        for m in self.moves:
            d = m.diagnostic or diagnose_move(m, db=db)
            if d:
                out[d] = out.get(d, 0) + 1
        return out

    def primary_leak_summary(self, db: Optional[Any] = None) -> Optional[str]:
        """生成本局最主要失误漏点与建议的精简摘要文本"""
        breakdown = self.diagnostics_breakdown(db)
        if not breakdown:
            return None
        total = sum(breakdown.values())
        top_key, count = max(breakdown.items(), key=lambda kv: kv[1])
        pct = count / total
        meta = DIAGNOSTIC_META.get(top_key, {})
        name = meta.get("name", top_key)
        desc = meta.get("description", "")
        return f"Primary leak: {pct:.0%} of mistakes were '{name}' ({count}/{total}) — {desc}"


def classify(loss: float, v_actual: float, others: Sequence[float], v_wait: float) -> str:
    """
    根据胜率损失将出牌打上对应的标签等级
    """
    if loss >= BLUNDER:
        return "blunder"
    if loss >= MISTAKE:
        return "mistake"
    if loss >= INACCURACY:
        return "inaccuracy"
    if loss <= BEST_EPS:
        median = statistics.median(others) if others else v_actual
        # 妙手判断：该打法大幅领先其他所有走法的中位数，且显著优于消极等待
        if v_actual - median >= BRILLIANT_GAIN and v_actual - v_wait >= BRILLIANT_GAIN * 0.75:
            return "brilliant"
        return "best"
    return "good"


class CoachAnalyzer:
    """复盘教练分析器核心类"""

    def __init__(self, fork: Fork, model: Optional[ValueModel] = None, side: int = 0, horizon_ms: int = 8000):
        self.fork = fork               # 状态分支克隆方法
        self.model = model or default_value_model() # 胜率价值评估器
        self.side = side               # 被分析学员阵营 (0=人类玩家)
        self.horizon_ms = horizon_ms   # 前向仿真模拟的时间深度 (默认 8000ms = 8 秒)

    # -- 候选方案枚举 ------------------------------------------------------
    def candidates(self, m: Match) -> List[Candidate]:
        """
        根据当前场上残局局势，启发式智能生成所有合理的候选操作方案
        """
        side, st = self.side, m.sides[self.side]
        mine = lambda x, y: G.to_side(side, x, y)  # noqa: E731
        enemies = [u for u in m.units.values() if u.side != side and u.hp > 0]
        troops = sorted((u for u in enemies if u.kind != "tower"), key=lambda u: mine(u.x, u.y)[1])
        threats = [u for u in troops if mine(u.x, u.y)[1] < G.RIVER_Y1 + 2]
        towers = [u for u in enemies if u.kind == "tower"]

        # 候选中必然包含 "Wait" (保留手牌与圣水)
        out: List[Candidate] = [Candidate(None)]
        seen = set()
        for card in st.hand:
            c = m.db.cards[card]
            # 圣水不够的牌无法作为候选
            if c["elixir"] > st.elixir + 1e-9:
                continue
            spots: List[Tuple[float, float]] = []
            if c["type"] == "spell":
                rolling = "rolling_range_tiles" in c.get("spell", {})
                for u in troops[:4] + towers:
                    tx, ty = mine(u.x, u.y)
                    spots.append((tx, ty - 1.5) if rolling else (tx, ty))
            else:
                spots += TROOP_SPOTS
                # 针对靠近河道的敌方进攻威胁，生成贴脸下兵与近身拉扯点位
                for u in threats[:2]:
                    tx, ty = mine(u.x, u.y)
                    spots += [(tx, min(ty - 2.5, G.RIVER_Y0 - 0.5)), (tx, min(ty - 0.5, G.RIVER_Y0 - 0.5))]
            for sx, sy in spots:
                ax, ay = G.snap(*mine(sx, max(0.5, sy)))
                key = (card, ax, ay)
                if key in seen or m.check_play(side, card, ax, ay) is not None:
                    continue
                seen.add(key)
                out.append(Candidate(card, ax, ay))
        return out

    # -- 分支推演评估 ------------------------------------------------------
    def rollout(self, checkpoint: Any, cand: Candidate) -> Optional[float]:
        """
        单分支向前推演：
        在克隆出的平行宇宙中下出 cand，并让 AI 真实应对 horizon_ms，最终计算局势胜率
        """
        m, runner = self.fork(checkpoint)
        if not cand.is_wait:
            if not m.play(self.side, cand.card_id, cand.x, cand.y).ok:
                return None
            runner.deliver_events()
        # 持续步进仿真 8 秒
        for _ in range(self.horizon_ms // G.TICK_MS):
            if m.result is not None:
                break
            runner.tick()
        return self.model.value(m, self.side)

    def review_move(self, checkpoint: Any, actual: Candidate, event_id: int) -> Optional[MoveReview]:
        """
        评析玩家在 checkpoint 时刻打出的单步操作 actual
        """
        m0, _ = self.fork(checkpoint)
        v_before = self.model.value(m0, self.side)
        cands = self.candidates(m0)
        if actual not in cands:
            cands.append(actual)
        scored: List[Tuple[Candidate, float]] = []
        for c in cands:
            v = self.rollout(checkpoint, c)
            if v is not None:
                scored.append((c, v))
        v_actual = next((v for c, v in scored if c == actual), None)
        if v_actual is None:
            return None
        # 按终态胜率降序排序，推选理论最优解
        scored.sort(key=lambda cv: (-cv[1], cv[0].is_wait))
        best, v_best = scored[0]
        v_wait = next((v for c, v in scored if c.is_wait), v_before)
        loss = max(0.0, v_best - v_actual)
        others = [v for c, v in scored if c != actual]
        lbl = classify(loss, v_actual, others, v_wait)
        diag = diagnose_move(
            MoveReview(event_id, m0.t_ms, actual, v_before, v_actual, v_best, v_wait, best, loss, lbl, len(scored)),
            db=getattr(m0, "db", None),
            match_before=m0,
            side=self.side,
        )
        return MoveReview(event_id, m0.t_ms, actual, v_before, v_actual, v_best, v_wait, best, loss,
                          lbl, len(scored), tuple(scored[:5]), diagnostic=diag)

    def value_at(self, checkpoint: Any) -> float:
        """评估特定历史检查点时刻的局势静态胜率"""
        m, _ = self.fork(checkpoint)
        return self.model.value(m, self.side)

    def review_match(self, checkpoints: Sequence[Any], plays: Sequence[Tuple[Any, Candidate, int]],
                     final_match: Optional[Match] = None,
                     progress: Optional[Callable[[int, int], None]] = None) -> MatchReview:
        """
        整场比赛全量复盘：
        生成胜率起伏折线并为玩家的所有出牌进行逐一评审打分
        """
        total = len(checkpoints) + len(plays)
        curve = []
        # 1. 抽取整场时间线上的胜率曲线
        for i, cp in enumerate(checkpoints):
            curve.append((cp.t_ms, self.value_at(cp)))
            if progress and i % 20 == 0:
                progress(i, total)
        final = None
        if final_match is not None:
            final = self.model.value(final_match, self.side)
            if not curve or curve[-1][0] < final_match.t_ms:
                curve.append((final_match.t_ms, final))
        # 2. 逐步评审玩家的每一次出牌
        moves = []
        for j, (cp, actual, eid) in enumerate(plays):
            r = self.review_move(cp, actual, eid)
            if r is not None:
                moves.append(r)
            if progress:
                progress(len(checkpoints) + j + 1, total)
        return MatchReview(curve, moves, final, dict(self.model.meta))
