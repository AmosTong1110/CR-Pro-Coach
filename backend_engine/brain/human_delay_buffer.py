"""
HumanDelayBuffer: three-layer human reaction model between the simulator and the AI.

Layer 1  Perception gate   opponent plays become visible to the policy only after a
                           sampled latency (surprise / anticipated / routine).
Layer 2  Hover reflex      a pre-hovered card fires 120-200 ms after a matching trigger,
                           bypassing the policy (how pros insta-log a goblin barrel).
Layer 3  Finger limiter    separate decisions are spaced by an APM gap (180-300 ms);
                           cards submitted together as a combo go out `combo_gap_ms`
                           apart (default 120 ms: one card can be selected at a time).

Everything runs on simulation milliseconds and a private seeded RNG, so a match
replays bit-for-bit from (seed, action log) and the state can be snapshotted for
undo / "Take Over From Here" / MCTS.

Fog-of-war contract for the observation builder: an opponent unit may only be
included in the policy's observation if `buffer.is_visible(unit.source_event_id)`.
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

DEFAULT_PROFILE_PATH = Path(__file__).resolve().parents[2] / "configs" / "human_reaction_profile.yaml"


# ---------------------------------------------------------------------------
# Profile
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class LatencyDist:
    mean: float
    std: float
    min: float
    max: float

    def __post_init__(self) -> None:
        if self.min > self.max:
            raise ValueError(f"LatencyDist min {self.min} > max {self.max}")
        if self.std < 0:
            raise ValueError("LatencyDist std must be >= 0")

    def sample(self, rng: random.Random) -> int:
        """Truncated normal (rejection sampling, so no probability piles up on the bounds)."""
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
    surprise: LatencyDist
    anticipated: LatencyDist
    routine: LatencyDist
    hover_reflex: LatencyDist
    apm_gap: LatencyDist
    combo_gap_ms: int = 120
    anticipation_threshold: float = 0.35
    max_hold_ms: int = 1500
    max_hover_ms: int = 12000

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
        import yaml  # local import: only needed when loading from disk

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
        """Built-in defaults (mirror configs/human_reaction_profile.yaml) for tests / no-yaml envs."""
        return cls(
            surprise=LatencyDist(500, 40, 420, 580),
            anticipated=LatencyDist(320, 40, 240, 420),
            routine=LatencyDist(650, 120, 400, 1000),
            hover_reflex=LatencyDist(160, 20, 120, 200),
            apm_gap=LatencyDist(240, 30, 180, 300),
        )


def _deep_merge(base: Dict[str, Any], over: Mapping[str, Any]) -> None:
    for k, v in over.items():
        if isinstance(v, Mapping) and isinstance(base.get(k), dict):
            _deep_merge(base[k], v)
        else:
            base[k] = copy.deepcopy(v)


# ---------------------------------------------------------------------------
# Buffer
# ---------------------------------------------------------------------------
LegalityCheck = Callable[[ActionIntent], bool]


class HumanDelayBuffer:
    def __init__(self, profile: Optional[ReactionProfile] = None, side: int = 1, seed: int = 0):
        self.profile = profile or ReactionProfile.default()
        self.side = side
        self.seed = seed
        self.reset()

    # -- lifecycle ----------------------------------------------------------
    def reset(self) -> None:
        self._rng = random.Random(self.seed)
        self._expectations: Dict[str, float] = {}
        self._hover: Optional[HoverPlan] = None
        self._hover_expires_ms: int = 0
        self._perceive_heap: List[Tuple[int, int, PublicDeployEvent]] = []  # (visible_at, event_id, ev)
        self._visible_ids: set[int] = set()
        self._latency_of: Dict[int, Tuple[LatencyClass, int]] = {}         # event_id -> (class, delay)
        self._pending: List[ScheduledAction] = []
        self._finger_free_at_ms: int = 0
        self._seq: int = 0
        self._combo_seq: int = 0
        self.dropped: List[ScheduledAction] = []

    def snapshot(self) -> Dict[str, Any]:
        state = {k: copy.deepcopy(v) for k, v in self.__dict__.items() if k not in ("_rng", "profile")}
        state["_rng_state"] = self._rng.getstate()
        return state

    def restore(self, snap: Mapping[str, Any]) -> None:
        snap = dict(snap)
        rng_state = snap.pop("_rng_state")
        for k, v in snap.items():
            setattr(self, k, copy.deepcopy(v))
        self._rng = random.Random()
        self._rng.setstate(rng_state)

    # -- inputs from perception --------------------------------------------
    def set_expectations(self, card_probs: Mapping[str, float]) -> None:
        """card -> P(opponent can play it now); typically ElixirEstimate.playable_probs()."""
        self._expectations = dict(card_probs)

    def arm_hover(self, plan: HoverPlan, now_ms: int) -> bool:
        """Hold a card over the arena. Fails if the finger is still busy with a previous drop."""
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
        self._hover = None

    @property
    def hover(self) -> Optional[HoverPlan]:
        return self._hover

    def observe(self, event: PublicDeployEvent) -> LatencyClass:
        """Feed every PublicDeployEvent here the moment it happens in the sim."""
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
