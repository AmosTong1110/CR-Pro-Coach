"""
比赛运行调度器 (Match Runner)

驱动包含人类反应延迟模拟 (HumanDelayBuffer) 的双智能体完整对战。

每帧时钟推进管线 (Per-tick, t = match.t_ms):
  1. 将新产生的公共部署事件 (PublicDeployEvent) 分发至各阵营的延迟缓冲区 (进入感知门队列 / 触发悬停微反应)。
  2. 针对手指空闲 (finger idle) 的 AI 智能体，提取迷雾观察视界 (fog-of-war observation)，生成对手圣水推断与卡组分布，并调用智能体生成出牌意图 (ActionIntent)。
  3. 延迟缓冲区释放到期的动作 -> 经费用与合法性校验后执行 match.play()。
  4. 分发上述下牌产生的事件 (确保对手在当前 tick 立即开始计算反应延迟时钟)。
  5. 物理引擎步进 match.step()。

若某一阵营 agent=None，则表示由外部驱动 (如 Qt 界面中的人类玩家)，在各 tick 之间通过调用 runner.human_play(card, x, y) 下牌。

命令行无头对战测试：
    python -m backend_engine.simulator.match_runner --seed 3 --level 16
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

from backend_engine.brain.human_delay_buffer import HumanDelayBuffer, ReactionProfile
from backend_engine.contracts import ActionIntent, PublicDeployEvent
from backend_engine.perception.bayesian_deck_guesser import DeckGuess, default_guesser, known_costs
from backend_engine.perception.opp_elixir_count import OppElixirTracker
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig, PlayResult


class Agent(Protocol):
    """智能体协议接口：接收迷雾观察与感知事件，输出下牌意图列表"""
    def decide(self, obs: dict, visible_events: Sequence[PublicDeployEvent], now_ms: int,
               buffer: HumanDelayBuffer) -> List[ActionIntent]: ...


@dataclass
class RunnerStats:
    """运行统计指标：记录各阵营成功出牌数与被拒绝的非法操作"""
    plays: Dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})
    rejected: Dict[int, List[str]] = field(default_factory=lambda: {0: [], 1: []})


class MatchRunner:
    """对战运行调度器
    
    统一协调底层物理对局 (Match)、感知追踪系统 (OppElixirTracker + BayesianDeckGuesser)、
    生理延迟缓冲区 (HumanDelayBuffer) 与高层决策决策智能体 (Agent)。
    """
    def __init__(self, match: Match, agents: Dict[int, Optional[Agent]],
                 buffers: Optional[Dict[int, HumanDelayBuffer]] = None):
        self.match = match
        self.agents = agents
        # 为每个 AI 阵营绑定一个具备独立随机种子的拟人延迟缓冲区
        self.buffers = buffers or {
            s: HumanDelayBuffer(ReactionProfile.default(), side=s, seed=match.cfg.seed * 2 + s)
            for s, a in agents.items() if a is not None
        }
        self._delivered = 0
        self.stats = RunnerStats()
        db = match.db
        self._costs = {**known_costs(), **db.elixir_costs()}
        self.guesser = default_guesser()
        self._guess_cache: Dict[Tuple[int, Tuple[str, ...]], DeckGuess] = {}
        self._hand_cache: Dict[tuple, Dict[str, float]] = {}
        evo_capable = db.evo_capable()
        # 每个 AI 阵营独立实例化一个对手圣水计数器，仅在“观察”到事件后才更新 (保证认知符合生理延迟)
        self.trackers = {s: OppElixirTracker(db.rules, observer_side=s, evo_cards=evo_capable)
                         for s in self.buffers}

    def _guess(self, side: int, known: Tuple[str, ...]) -> Optional[DeckGuess]:
        """带缓存的贝叶斯卡组推断：仅在对手亮出新卡时重新计算后验概率"""
        if self.guesser is None:
            return None
        key = (side, tuple(sorted(known)))
        if key not in self._guess_cache:
            self._guess_cache[key] = self.guesser.guess(known)
        return self._guess_cache[key]

    def deliver_events(self) -> None:
        """立即将新发生的 PublicDeployEvents 投递至所有阵营的缓冲区 (幂等)"""
        self._deliver_events()

    def _deliver_events(self) -> None:
        """事件分发：遍历自上次分发以来产生的新事件并推送到缓冲区"""
        new = self.match.events[self._delivered:]
        self._delivered = len(self.match.events)
        for ev in new:
            for buf in self.buffers.values():
                buf.observe(ev)

    def human_play(self, card_id: str, x: float, y: float, side: int = 0) -> PlayResult:
        """人类玩家下牌接口 (无生理延迟注入，直接下发引擎)"""
        res = self.match.play(side, card_id, x, y)
        if res.ok:
            self.stats.plays[side] += 1
        return res

    def tick(self) -> None:
        """执行单一 50ms 模拟 Tick 推进管线"""
        m = self.match
        if m.result is not None:
            return
        t = m.t_ms
        # 1. 投递上一帧遗留事件
        self._deliver_events()

        # 2. 各 AI 阵营进行认知更新与决策判断
        for side, agent in self.agents.items():
            if agent is None:
                continue
            buf = self.buffers[side]
            # 提取此刻由于反应延迟结束而真正“被大脑意识到”的可见事件
            visible = buf.pop_visible_events(t)
            tracker = self.trackers[side]
            tracker.observe_all(visible)
            est = tracker.estimate(t)
            # 贝叶斯卡组推断融入未知手牌概率
            guess = self._guess(side, est.known_cards)
            if guess is not None:
                key = (side, est.cycle_order, est.unknown_hand_slots)
                merged = self._hand_cache.get(key)
                if merged is None:
                    extra = dict(list(guess.in_hand_probs(est.unknown_hand_slots).items())[:12])
                    merged = self._hand_cache[key] = {**extra, **est.in_hand_prob}
                est = replace(est, in_hand_prob=merged)
            # 将对手可打卡牌先验告知缓冲区 (供预期型反应使用)
            buf.set_expectations(est.playable_probs(self._costs))

            # 当智能体手指未被占用且无排队待发动作时，调起决策
            if buf.finger_free_at_ms() <= t and not buf.pending():
                obs = m.observation(side, buf.is_visible)
                obs["opp_estimate"] = est
                obs["opp_deck_guess"] = guess
                intents = agent.decide(obs, visible, t, buf)
                if intents:
                    buf.submit(intents, t)

        # 3. 释放到达下牌时刻的动作意图并下发引擎
        for side, buf in self.buffers.items():
            legal = lambda i, s=side: m.check_play(s, i.card_id, i.tile.x, i.tile.y) is None  # noqa: E731
            for a in buf.release_due(t, elixir=m.sides[side].elixir, is_legal=legal):
                res = m.play(side, a.intent.card_id, a.intent.tile.x, a.intent.tile.y)
                if res.ok:
                    self.stats.plays[side] += 1
                else:
                    self.stats.rejected[side].append(res.reason)

        # 清空因卡手/费不足被丢弃的意图
        for buf in self.buffers.values():
            buf.dropped.clear()

        # 4. 再次投递由本 tick 动作产生的新事件，确保对手认知时钟立即起算
        self._deliver_events()

        # 5. 步进物理、伤害结算与网格重构
        m.step()

    def run(self, max_ms: Optional[int] = None) -> None:
        """连续运行直到比赛分出胜负或达到指定最大时长"""
        stop = max_ms if max_ms is not None else self.match.end_ms + 1
        while self.match.result is None and self.match.t_ms < stop:
            self.tick()


DEFAULT_DECK = ("hog_rider", "musketeer", "knight", "cannon", "fireball", "the_log", "skeletons", "archers")


def main() -> None:
    from backend_engine.brain.scripted_bot import ScriptedBot

    ap = argparse.ArgumentParser(description="Headless bot-vs-bot match")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--level", type=int, default=11, choices=[11, 12, 13, 14, 15, 16])
    args = ap.parse_args()

    db = CardDB.load()
    cfg = MatchConfig((DEFAULT_DECK, DEFAULT_DECK), (args.level, args.level), (("knight",), ()), seed=args.seed)
    m = Match(cfg, db)
    runner = MatchRunner(m, {0: ScriptedBot(db, 0, aggression=0.7), 1: ScriptedBot(db, 1, aggression=0.4)})
    t0 = time.perf_counter()
    runner.run()
    wall = time.perf_counter() - t0
    ticks = m.t_ms // 50

    r = m.result
    print(f"result      : winner={r.winner} crowns={r.crowns} reason={r.reason} at {r.t_ms / 1000:.1f}s")
    for s in (0, 1):
        towers = {u.tower_role: round(u.hp) for u in m.units.values() if u.kind == "tower" and u.side == s}
        st = m.sides[s]
        print(f"side {s}      : plays={runner.stats.plays[s]} leaked={st.leaked_elixir:.1f} towers={towers}")
    print(f"actions     : {len(m.action_log)}  state_hash={m.state_hash()}")
    print(f"performance : {ticks} ticks in {wall:.2f}s = {ticks / wall:,.0f} ticks/s "
          f"({ticks * 50 / 1000 / wall:,.0f}x real time)")


if __name__ == "__main__":
    main()
