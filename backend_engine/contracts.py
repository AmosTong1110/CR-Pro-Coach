"""
Shared data contracts between the simulator, perception and brain layers.

Conventions (apply everywhere in backend_engine):
  * Time is integer **simulation milliseconds** (`t_ms`), never wall-clock.
    Pause / 2x speed / undo / MCTS rollouts / cloud training at 1000x all just
    advance this number differently.
  * Positions are tiles: x in [0, 18), y in [0, 32), y=0 at side 0's king tower.
  * side 0 = bottom (human player in the UI), side 1 = top (AI).
  * Everything here is a plain dataclass so it can be JSON-serialised for the
    replay log, the UI bridge and snapshots.

Data flow for one opponent card play (AI's point of view):

    simulator  --PublicDeployEvent-->  OppElixirTracker  --ElixirEstimate--+
        |                                                                  |
        +--PublicDeployEvent--> HumanDelayBuffer.observe()   <-- set_expectations()
                                      |   (perception delay / hover reflex)
                                      v
                       pop_visible_events()  -> observation builder -> policy
                                                                         |
                       HumanDelayBuffer.submit(ActionIntent...)  <-------+
                                      |   (finger / APM limiter)
                                      v
                       release_due() -> ScheduledAction -> simulator.apply()
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, FrozenSet, Optional, Tuple

ARENA_W_TILES = 18
ARENA_H_TILES = 32


@dataclass(frozen=True)
class TilePos:
    x: float
    y: float

    @property
    def lane(self) -> str:
        return "left" if self.x < ARENA_W_TILES / 2 else "right"

    def offset(self, dx: float, dy: float) -> "TilePos":
        return TilePos(
            min(max(self.x + dx, 0.0), ARENA_W_TILES - 0.5),
            min(max(self.y + dy, 0.0), ARENA_H_TILES - 0.5),
        )


# --------------------------------------------------------------------------
# Simulator -> everyone: the only thing either side may learn about the
# opponent's plays. It contains exactly what a human sees on screen.
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class PublicDeployEvent:
    event_id: int            # monotonically increasing per match
    t_ms: int                # sim time the card appeared on screen
    side: int                # who played it
    card_id: str             # key into cards_stats_v2026.json["cards"]
    elixir_cost: int         # cost actually paid (incl. ability costs)
    tile: TilePos
    is_evolved: bool = False
    is_ability: bool = False # champion / hero ability button, not a card
    is_threat: bool = False  # set by the sim: lands on/heads into the viewer's half


# --------------------------------------------------------------------------
# opp_elixir_count.py -> brain
# --------------------------------------------------------------------------
@dataclass(frozen=True)
class ElixirEstimate:
    t_ms: int
    point: float                       # best guess of opponent elixir
    low: float                         # hard lower bound (can only be wrong if we missed a play)
    high: float                        # hard upper bound (10 unless they just spent)
    known_cards: Tuple[str, ...]       # cards revealed so far (<= 8)
    cycle_order: Tuple[str, ...]       # plays, oldest first
    in_hand_prob: Dict[str, float] = field(default_factory=dict)  # card -> P(currently in hand)
    multiplier: int = 1                # 1x / 2x / 3x at t_ms
    # card cycle: the 4 cards waiting to come back, oldest (= next to return) first; None = unknown
    queue: Tuple[Optional[str], ...] = ()
    known_in_hand: Tuple[str, ...] = ()   # revealed cards certainly in hand now
    unknown_hand_slots: int = 4           # hand slots holding cards we have not seen yet
    leaked: float = 0.0                   # elixir the opponent wasted at 10
    evo_progress: Dict[str, int] = field(default_factory=dict)  # card -> plays since its last evolved play
    resyncs: int = 0                      # times a play proved the model wrong (missed event / bad data)

    @property
    def next_return(self) -> Optional[str]:
        """The card that comes back to the opponent's hand on their next play."""
        return self.queue[0] if self.queue else None

    def p_affordable(self, cost: int) -> float:
        """P(elixir >= cost), treating elixir as uniform on [low, high]."""
        if cost <= self.low:
            return 1.0
        if cost > self.high:
            return 0.0
        span = max(self.high - self.low, 1e-6)
        return (self.high - cost) / span

    def playable_probs(self, costs: Dict[str, int]) -> Dict[str, float]:
        """card -> P(opponent can play it right now). Feed to HumanDelayBuffer.set_expectations()."""
        return {c: p * self.p_affordable(costs.get(c, 10)) for c, p in self.in_hand_prob.items()}


# --------------------------------------------------------------------------
# brain <-> HumanDelayBuffer <-> simulator
# --------------------------------------------------------------------------
class ActionKind(str, Enum):
    DEPLOY = "deploy"
    ABILITY = "ability"   # champion / hero button
    NOOP = "noop"         # explicit "wait" decision; never scheduled


class LatencyClass(str, Enum):
    SURPRISE = "surprise"        # unpredicted threat
    ANTICIPATED = "anticipated"  # predicted by elixir tracker / deck guesser
    ROUTINE = "routine"          # non-threatening play (back of their side)
    HOVER_REFLEX = "hover_reflex"
    OWN_INITIATIVE = "own_initiative"  # AI acting without a triggering event


@dataclass(frozen=True)
class ActionIntent:
    kind: ActionKind
    card_id: Optional[str] = None
    tile: Optional[TilePos] = None
    elixir_cost: int = 0
    source: str = "policy"       # "policy" | "mcts" | "hover" | "script"


@dataclass(frozen=True)
class HoverPlan:
    """A card held over the arena waiting for a trigger (one finger => one plan)."""
    card_id: str
    elixir_cost: int
    trigger_cards: FrozenSet[str]
    target_mode: str = "fixed"           # "fixed" -> use `tile`; "event" -> trigger tile + offset
    tile: Optional[TilePos] = None
    offset: Tuple[float, float] = (0.0, 0.0)
    trigger_lane: Optional[str] = None   # "left" | "right" | None (any)
    expires_at_ms: Optional[int] = None


@dataclass(frozen=True)
class ScheduledAction:
    seq: int
    intent: ActionIntent
    created_at_ms: int
    execute_at_ms: int
    latency_class: LatencyClass
    combo_id: Optional[int] = None
    trigger_event_id: Optional[int] = None
