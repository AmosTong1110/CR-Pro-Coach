"""
Post-match coach: win-rate curve + grade every play (Brilliant !! / Best / Good /
Inaccuracy ?! / Mistake ? / Blunder ??) + the move the engine would have made instead.

How a play is graded
  1. Restore the exact state just before the play (checkpoint captured at play time).
  2. Generate candidates: every affordable card in hand x sensible tiles (lanes, bridge,
     defensive spots, on/behind threats, spells on enemy clusters and towers) + "wait".
     The actual play is always included.
  3. For each candidate: fork the state, make that play, let the AI respond for
     `horizon_ms` (you stay idle in all branches, so the comparison is fair), then score
     the result with the win-probability model.
  4. loss = V(best) - V(actual) in win-probability points -> label.

Everything runs on forks; the live match is never touched. Deterministic: same match,
same review.

Known limits (v1): the timing of a play is graded only against "wait", not against
playing the same card 1 s earlier or later; the AI responds with the scripted bot.
"""
from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple

from backend_engine.brain.value_model import ValueModel, default_value_model
from backend_engine.simulator import geometry as G
from backend_engine.simulator.engine import Match

Fork = Callable[[Any], Tuple[Match, Any]]  # checkpoint -> (match, runner) copies

# thresholds in win-probability points
INACCURACY, MISTAKE, BLUNDER = 0.05, 0.10, 0.20
BEST_EPS = 0.01
BRILLIANT_GAIN = 0.12

LABELS = {
    "brilliant":  ("!!", "妙手 Brilliant"),
    "best":       ("★", "最佳 Best"),
    "good":       ("", "好棋 Good"),
    "inaccuracy": ("?!", "缓手 Inaccuracy"),
    "mistake":    ("?", "失误 Mistake"),
    "blunder":    ("??", "大错 Blunder"),
}

TROOP_SPOTS = [(3.5, 13.5), (14.5, 13.5), (3.5, 9.5), (14.5, 9.5), (3.5, 4.5), (14.5, 4.5), (9.5, 9.5), (8.5, 5.5)]


@dataclass(frozen=True)
class Candidate:
    card_id: Optional[str]           # None = wait / don't play
    x: float = 0.0
    y: float = 0.0

    @property
    def is_wait(self) -> bool:
        return self.card_id is None


@dataclass(frozen=True)
class MoveReview:
    event_id: int
    t_ms: int
    actual: Candidate
    v_before: float
    v_actual: float
    v_best: float
    v_wait: float
    best: Candidate
    loss: float
    label: str
    n_candidates: int
    alternatives: Tuple[Tuple[Candidate, float], ...] = ()   # top few, best first

    @property
    def symbol(self) -> str:
        return LABELS[self.label][0]

    @property
    def label_text(self) -> str:
        return LABELS[self.label][1]


@dataclass
class MatchReview:
    curve: List[Tuple[int, float]]           # (t_ms, P(you win))
    moves: List[MoveReview]
    final: Optional[float] = None
    model_meta: dict = field(default_factory=dict)

    @property
    def accuracy(self) -> float:
        """0-100, chess.com-style: average of per-move scores (no loss = 100, >= 0.3 loss = 0)."""
        if not self.moves:
            return 0.0
        return 100.0 * statistics.mean(max(0.0, 1.0 - m.loss / 0.3) for m in self.moves)

    def counts(self) -> dict:
        out = {k: 0 for k in LABELS}
        for m in self.moves:
            out[m.label] += 1
        return out


def classify(loss: float, v_actual: float, others: Sequence[float], v_wait: float) -> str:
    if loss >= BLUNDER:
        return "blunder"
    if loss >= MISTAKE:
        return "mistake"
    if loss >= INACCURACY:
        return "inaccuracy"
    if loss <= BEST_EPS:
        median = statistics.median(others) if others else v_actual
        if v_actual - median >= BRILLIANT_GAIN and v_actual - v_wait >= BRILLIANT_GAIN * 0.75:
            return "brilliant"
        return "best"
    return "good"


class CoachAnalyzer:
    def __init__(self, fork: Fork, model: Optional[ValueModel] = None, side: int = 0, horizon_ms: int = 8000):
        self.fork = fork
        self.model = model or default_value_model()
        self.side = side
        self.horizon_ms = horizon_ms

    # -- candidates --------------------------------------------------------
    def candidates(self, m: Match) -> List[Candidate]:
        side, st = self.side, m.sides[self.side]
        mine = lambda x, y: G.to_side(side, x, y)  # noqa: E731  (self-inverse)
        enemies = [u for u in m.units.values() if u.side != side and u.hp > 0]
        troops = sorted((u for u in enemies if u.kind != "tower"), key=lambda u: mine(u.x, u.y)[1])
        threats = [u for u in troops if mine(u.x, u.y)[1] < G.RIVER_Y1 + 2]
        towers = [u for u in enemies if u.kind == "tower"]

        out: List[Candidate] = [Candidate(None)]
        seen = set()
        for card in st.hand:
            c = m.db.cards[card]
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

    # -- evaluation --------------------------------------------------------
    def rollout(self, checkpoint: Any, cand: Candidate) -> Optional[float]:
        m, runner = self.fork(checkpoint)
        if not cand.is_wait:
            if not m.play(self.side, cand.card_id, cand.x, cand.y).ok:
                return None
            runner.deliver_events()
        for _ in range(self.horizon_ms // G.TICK_MS):
            if m.result is not None:
                break
            runner.tick()
        return self.model.value(m, self.side)

    def review_move(self, checkpoint: Any, actual: Candidate, event_id: int) -> Optional[MoveReview]:
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
        scored.sort(key=lambda cv: (-cv[1], cv[0].is_wait))
        best, v_best = scored[0]
        v_wait = next((v for c, v in scored if c.is_wait), v_before)
        loss = max(0.0, v_best - v_actual)
        others = [v for c, v in scored if c != actual]
        return MoveReview(event_id, m0.t_ms, actual, v_before, v_actual, v_best, v_wait, best, loss,
                          classify(loss, v_actual, others, v_wait), len(scored), tuple(scored[:5]))

    def value_at(self, checkpoint: Any) -> float:
        m, _ = self.fork(checkpoint)
        return self.model.value(m, self.side)

    def review_match(self, checkpoints: Sequence[Any], plays: Sequence[Tuple[Any, Candidate, int]],
                     final_match: Optional[Match] = None,
                     progress: Optional[Callable[[int, int], None]] = None) -> MatchReview:
        """checkpoints: timeline checkpoints (each has .t_ms); plays: (pre-play checkpoint, actual, event_id)."""
        total = len(checkpoints) + len(plays)
        curve = []
        for i, cp in enumerate(checkpoints):
            curve.append((cp.t_ms, self.value_at(cp)))
            if progress and i % 20 == 0:
                progress(i, total)
        final = None
        if final_match is not None:
            final = self.model.value(final_match, self.side)
            if not curve or curve[-1][0] < final_match.t_ms:
                curve.append((final_match.t_ms, final))
        moves = []
        for j, (cp, actual, eid) in enumerate(plays):
            r = self.review_move(cp, actual, eid)
            if r is not None:
                moves.append(r)
            if progress:
                progress(len(checkpoints) + j + 1, total)
        return MatchReview(curve, moves, final, dict(self.model.meta))
