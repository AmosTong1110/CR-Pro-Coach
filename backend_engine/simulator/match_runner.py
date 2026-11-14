"""
Drives a Match with agents that see the game through HumanDelayBuffer.

Per tick (t = match.t_ms):
  1. deliver new PublicDeployEvents to every buffer (perception queue / hover reflex)
  2. each agent whose finger is idle gets a fog-of-war observation and may submit intents
  3. each buffer releases due actions -> match.play()
  4. deliver the events those plays produced (so the opponent's clock starts this tick)
  5. match.step()

A side with agent=None is driven externally (the human in the UI): call
runner.human_play(card, x, y) between ticks.

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
    def decide(self, obs: dict, visible_events: Sequence[PublicDeployEvent], now_ms: int,
               buffer: HumanDelayBuffer) -> List[ActionIntent]: ...


@dataclass
class RunnerStats:
    plays: Dict[int, int] = field(default_factory=lambda: {0: 0, 1: 0})
    rejected: Dict[int, List[str]] = field(default_factory=lambda: {0: [], 1: []})


class MatchRunner:
    def __init__(self, match: Match, agents: Dict[int, Optional[Agent]],
                 buffers: Optional[Dict[int, HumanDelayBuffer]] = None):
        self.match = match
        self.agents = agents
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
        evo_capable = [c for c, d in db.cards.items() if d.get("evolution")]
        # each AI counts the opponent's elixir from plays it has *perceived* (after its reaction delay)
        self.trackers = {s: OppElixirTracker(db.rules, observer_side=s, evo_cards=evo_capable)
                         for s in self.buffers}

    def _guess(self, side: int, known: Tuple[str, ...]) -> Optional[DeckGuess]:
        if self.guesser is None:
            return None
        key = (side, tuple(sorted(known)))
        if key not in self._guess_cache:  # only recomputed when a new card is revealed
            self._guess_cache[key] = self.guesser.guess(known)
        return self._guess_cache[key]

    def deliver_events(self) -> None:
        """Push new PublicDeployEvents into every buffer now (idempotent)."""
        self._deliver_events()

    def _deliver_events(self) -> None:
        new = self.match.events[self._delivered:]
        self._delivered = len(self.match.events)
        for ev in new:
            for buf in self.buffers.values():
                buf.observe(ev)

    def human_play(self, card_id: str, x: float, y: float, side: int = 0) -> PlayResult:
        res = self.match.play(side, card_id, x, y)
        if res.ok:
            self.stats.plays[side] += 1
        return res

    def tick(self) -> None:
        m = self.match
        if m.result is not None:
            return
        t = m.t_ms
        self._deliver_events()
        for side, agent in self.agents.items():
            if agent is None:
                continue
            buf = self.buffers[side]
            visible = buf.pop_visible_events(t)
            tracker = self.trackers[side]
            tracker.observe_all(visible)
            est = tracker.estimate(t)
            guess = self._guess(side, est.known_cards)
            if guess is not None:  # unrevealed cards: P(in deck) from real ladder decks
                key = (side, est.cycle_order, est.unknown_hand_slots)
                merged = self._hand_cache.get(key)
                if merged is None:  # only changes when a card is played
                    extra = dict(list(guess.in_hand_probs(est.unknown_hand_slots).items())[:12])
                    merged = self._hand_cache[key] = {**extra, **est.in_hand_prob}
                est = replace(est, in_hand_prob=merged)
            buf.set_expectations(est.playable_probs(self._costs))
            if buf.finger_free_at_ms() <= t and not buf.pending():
                obs = m.observation(side, buf.is_visible)
                obs["opp_estimate"] = est
                obs["opp_deck_guess"] = guess
                intents = agent.decide(obs, visible, t, buf)
                if intents:
                    buf.submit(intents, t)
        for side, buf in self.buffers.items():
            legal = lambda i, s=side: m.check_play(s, i.card_id, i.tile.x, i.tile.y) is None  # noqa: E731
            for a in buf.release_due(t, elixir=m.sides[side].elixir, is_legal=legal):
                res = m.play(side, a.intent.card_id, a.intent.tile.x, a.intent.tile.y)
                if res.ok:
                    self.stats.plays[side] += 1
                else:
                    self.stats.rejected[side].append(res.reason)
        for buf in self.buffers.values():
            buf.dropped.clear()
        self._deliver_events()
        m.step()

    def run(self, max_ms: Optional[int] = None) -> None:
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
