"""
Win-probability model V(state) = P(side wins | state), used for the post-match
win-rate graph and to grade moves.

A logistic regression on hand-built features, trained on self-play games inside our own
simulator (tools/train_value_model.py), so its probabilities are calibrated for this
engine. It is a placeholder for a learned value network (or the HF winner-predictor
once its inputs are mapped), but it is honest: weights come from data, not guesses.

Features are always "me minus opponent", so V(side=1) = 1 - V(side=0) by construction
(training uses both perspectives).
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

FEATURES = [
    "crowns",            # crown difference
    "crowns_late",       # x (1 - time left)
    "princess_hp",       # sum of princess-tower HP fractions (0..2) difference
    "king_hp",           # king-tower HP fraction difference
    "min_tower",         # weakest standing tower fraction difference (tiebreaker)
    "min_tower_late",
    "elixir",            # elixir difference / 10
    "board",             # elixir value of living troops/buildings (x HP fraction) / 10
    "pressure",          # same, only units in the opponent's half / 10
    "princess_late",
]


def features(m: Match, side: int = 0) -> List[float]:
    me, op = side, 1 - side
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

    board = [0.0, 0.0]
    pressure = [0.0, 0.0]
    for u in m.units.values():
        if u.kind == "tower" or u.hp <= 0:
            continue
        card = m.db.cards.get(u.card_id, {})
        value = card.get("elixir", 0) / max(1, card.get("count", 1)) * (u.hp / u.max_hp)
        k = 0 if u.side == me else 1
        board[k] += value
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
    weights: List[float]
    meta: dict

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> "ValueModel":
        p = Path(path)
        if not p.exists():
            return cls.fallback()
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("features") != FEATURES:
            return cls.fallback()
        return cls(d["weights"], d.get("meta", {}))

    @classmethod
    def fallback(cls) -> "ValueModel":
        """Hand-set weights used only until tools/train_value_model.py has been run."""
        return cls([1.5, 3.0, 1.2, 2.0, 1.0, 3.0, 0.8, 0.6, 0.8, 1.5], {"trained": False})

    def logit(self, f: Sequence[float]) -> float:
        return sum(w * x for w, x in zip(self.weights, f))

    def value(self, m: Match, side: int = 0) -> float:
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
