"""State records of the match engine (units, spells). Shared by engine.py and the mechanics package.

All of them are plain dataclasses so Match.snapshot() can deep-copy them.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set, Tuple

TROOP_SIGHT = 5.5           # default sight radius (tiles)


@dataclass
class Unit:
    """A live entity on the field: troop, building or tower."""
    uid: int
    side: int                      # 0 = bottom (player), 1 = top (AI)
    card_id: str
    kind: str                      # troop | building | tower
    layer: str                     # ground | air
    x: float
    y: float
    radius: float                  # collision radius (tiles)
    hp: float
    max_hp: float
    damage: int                    # damage per hit
    hit_speed_ms: int
    first_hit_ms: int              # delay before the first hit after locking on
    range: float                   # tiles, edge to edge
    targets: Tuple[str, ...]       # ("ground", "air", "buildings")
    min_range: float = 0.0         # minimum range blind spot (tiles, edge to edge, e.g. Mortar)
    splash: float = 0.0            # splash radius (read by the `splash` mechanic)
    speed: float = 0.0             # tiles / s
    sight: float = TROOP_SIGHT
    deploy_ms: int = 0             # remaining deploy time (no move / attack)
    cooldown_ms: int = 0           # time until the next hit
    target_uid: Optional[int] = None
    decay_per_tick: float = 0.0    # hp lost per tick (read by the `decay` mechanic)
    tower_role: Optional[str] = None   # king | princess_left | princess_right
    active: bool = True            # king tower starts inactive
    source_event_id: Optional[int] = None
    evolved: bool = False
    mass: float = 0.0              # push weight; 0 = immovable
    variant: str = "base"          # base | evo | hero
    mechanics: Tuple[str, ...] = ()            # registered mechanic kinds (see mechanics.REGISTRY)
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)   # kind -> params
    status: Dict[str, int] = field(default_factory=dict)              # status tag -> remaining ms
    mstate: Dict[str, Any] = field(default_factory=dict)              # per-unit mechanic state
    lane: str = "left"                                                # "left" | "right" initial assigned lane

    @property
    def building_only(self) -> bool:
        return self.targets == ("buildings",)

    @property
    def is_structure(self) -> bool:
        return self.kind in ("tower", "building")

    @property
    def river_jump(self) -> bool:
        return "river_jump" in self.mechanics


@dataclass
class Spell:
    """Thrown spell (Fireball, ...): flies from the king tower and lands after impact_in_ms."""
    uid: int
    side: int
    card_id: str
    x: float
    y: float
    radius: float
    damage: int
    tower_damage: int
    impact_in_ms: int
    source_event_id: int
    kind: str = "area_spell"
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class RollingSpell:
    """Rolling spell (The Log): moves forward and hits each ground unit on its path once."""
    uid: int
    side: int
    card_id: str
    x: float
    y: float                       # current front
    direction: int                 # +1 up, -1 down
    remaining: float               # tiles left to roll
    width: float
    speed: float                   # tiles / s
    damage: int
    tower_damage: int
    source_event_id: int
    hit: Set[int] = field(default_factory=set)
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class AreaEffect:
    """Lingering area (Tornado): for `duration_ms` it pulls enemy troops to its centre and damages everything
    in range every `hit_every_ms`."""
    uid: int
    side: int
    card_id: str
    x: float
    y: float
    radius: float
    damage: float
    tower_damage: float
    duration_ms: int
    hit_every_ms: int
    pull_speed: float              # tiles / s
    source_event_id: int
    kind: str = "pull_spell"
    elapsed_ms: int = 0
    next_hit_ms: int = 0
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)


@dataclass
class Projectile:
    """Homing shot (Musketeer, Archers, Cannon, towers ...): flies to its target, damage lands on arrival."""
    uid: int
    side: int
    source_uid: int
    card_id: str
    target_uid: int
    x: float
    y: float
    speed: float                   # tiles / s
    damage: float
    splash: float = 0.0            # > 0: area damage around the target on arrival
    targets: Tuple[str, ...] = ("ground",)
    params: Dict[str, Dict[str, Any]] = field(default_factory=dict)   # payload kinds -> params (on_projectile_hit)
