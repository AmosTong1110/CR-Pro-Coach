"""
Opponent elixir counter + card-cycle tracker (human-legal: public plays only).

Elixir is fully determined by public information: both players start at the same value,
regenerate on the same schedule, and every spend is visible. So the tracker replays
the opponent's plays against the regen schedule and gets the exact value, including
leaked elixir at 10. `slack` widens low/high for noisy inputs (screen-reading mode B,
where play times are only known to within a few frames).

Cycle rule: a played card goes to the back of a 4-card queue and returns to hand after
4 more plays. So the last 4 plays are certainly NOT in hand, and any revealed card
played 4+ plays ago (and not since) certainly IS in hand.

Events may arrive out of order (the AI perceives plays after different delays), so the
model is rebuilt from the sorted event list whenever a new event lands.
"""
from __future__ import annotations

import bisect
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from backend_engine.contracts import ElixirEstimate, PublicDeployEvent

EPS = 1e-6


@dataclass(frozen=True)
class ElixirSchedule:
    start: float
    max: float
    ms_per_elixir: Dict[int, float]
    phases: Tuple[Tuple[int, int], ...]   # (from_ms, multiplier), sorted

    @classmethod
    def from_rules(cls, rules: dict) -> "ElixirSchedule":
        phases = tuple(sorted((int(p["from_s"] * 1000), int(p["multiplier"])) for p in rules["elixir_phases"]))
        return cls(float(rules["start_elixir"]), float(rules["max_elixir"]),
                   {int(k): float(v) for k, v in rules["ms_per_elixir"].items()}, phases)

    def multiplier(self, t_ms: int) -> int:
        m = 1
        for start, mult in self.phases:
            if t_ms >= start:
                m = mult
        return m

    def advance(self, value: float, t0: int, t1: int) -> Tuple[float, float]:
        """Regenerate from t0 to t1. Returns (value, leaked)."""
        leaked = 0.0
        t = t0
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
    def __init__(self, rules: dict, observer_side: int, deck_size: int = 8, hand_size: int = 4,
                 evo_cards: Iterable[str] = (), slack: float = 0.0):
        self.schedule = ElixirSchedule.from_rules(rules)
        self.observer_side = observer_side
        self.deck_size = deck_size
        self.hand_size = hand_size
        self.evo_capable = set(evo_cards)   # cards that *could* be an evolution slot
        self.slack = slack
        self._events: List[Tuple[int, int, PublicDeployEvent]] = []
        self._dirty = True
        self._state: Tuple[float, int, float, int] = (self.schedule.start, 0, 0.0, 0)

    # -- input -------------------------------------------------------------
    def observe(self, ev: PublicDeployEvent) -> None:
        if ev.side == self.observer_side:
            return
        key = (ev.t_ms, ev.event_id, ev)
        if any(e[1] == ev.event_id for e in self._events):
            return
        bisect.insort(self._events, key, key=lambda k: (k[0], k[1]))
        self._dirty = True

    def observe_all(self, events: Sequence[PublicDeployEvent]) -> None:
        for ev in events:
            self.observe(ev)

    # -- model -------------------------------------------------------------
    def _rebuild(self) -> None:
        value, t, leaked, resyncs = self.schedule.start, 0, 0.0, 0
        for t_ev, _, ev in self._events:
            value, lk = self.schedule.advance(value, t, t_ev)
            leaked += lk
            t = t_ev
            if value + EPS < ev.elixir_cost:
                resyncs += 1           # they paid more than we thought they had: we missed regen/ an event
                value = float(ev.elixir_cost)
            value -= ev.elixir_cost
        self._state = (value, t, leaked, resyncs)
        self._dirty = False

    def _card_plays(self) -> List[PublicDeployEvent]:
        return [ev for _, _, ev in self._events if not ev.is_ability]

    def estimate(self, t_ms: int) -> ElixirEstimate:
        if self._dirty:
            self._rebuild()
        value, t_last, leaked, resyncs = self._state
        if t_ms > t_last:
            value, lk = self.schedule.advance(value, t_last, t_ms)
            leaked += lk
        plays = [ev.card_id for ev in self._card_plays()]
        revealed: List[str] = []
        for c in plays:
            if c not in revealed:
                revealed.append(c)
        n = len(plays)
        recent = plays[-self.hand_size:]
        queue: Tuple[Optional[str], ...] = tuple([None] * (self.hand_size - len(recent)) + recent)
        known_in_hand = tuple(c for c in revealed if c not in recent)
        unknown_cards = self.deck_size - len(revealed)
        unknown_slots = self.hand_size - len(known_in_hand)
        probs: Dict[str, float] = {c: 1.0 for c in known_in_hand}
        probs.update({c: 0.0 for c in recent})

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

    # -- snapshots ---------------------------------------------------------
    def snapshot(self) -> dict:
        return {"events": list(self._events), "evo": set(self.evo_capable)}

    def restore(self, snap: dict) -> None:
        self._events = list(snap["events"])
        self.evo_capable = set(snap["evo"])
        self._dirty = True
