"""Mechanic / SpellKind base classes and registries.

Hook call points inside Match.step() (order is fixed and uid-sorted, so the engine stays deterministic):
    on_spawn(m, u)                      after a card's unit is created
    on_tick(m, u, dt) -> bool           after deploy time, before targeting; True = unit is done this tick
    on_attack(m, u, target) -> bool     when a hit lands; True = handled (otherwise single-target damage)
    on_damaged(m, u, amount, src)       returns the damage actually taken
    on_death(m, u)                      before a dead unit is removed
    waypoint(m, u, target)              movement override; None = default pathing
    river_passable                      may cross / be pushed over the river
"""
from __future__ import annotations

from functools import lru_cache
from typing import Any, Dict, Optional, Tuple

REGISTRY: Dict[str, "Mechanic"] = {}
SPELLS: Dict[str, "SpellKind"] = {}
HOOK_NAMES = ("on_spawn", "on_tick", "on_attack", "on_damaged", "on_death", "waypoint")
HOOKS: Dict[str, set] = {h: set() for h in HOOK_NAMES}

# Tags in card data that describe a card but need no code (behaviour comes from stats / targeting).
# `knockback` and `apply_status` are simulated by the spell kinds, which read them from the spell's params.
DESCRIPTIVE = frozenset({"building_targeting", "swarm", "ground_only", "reduced_tower_damage",
                         "pulls_building_targeters", "air_unit", "knockback", "apply_status",
                         "spawn_on_end"})
# Known game mechanics that are not simulated yet (listed so data can mention them).
PLANNED: frozenset = frozenset({
    # evolutions whose special behaviour is not simulated yet (data: cycles + stats are in the card)
    "evo_power_shot", "evo_heal_pulse", "evo_barrage", "evo_poison_darts", "evo_sparks", "evo_ice_blast",
    "evo_sniper", "evo_souldiers", "evo_general_gerry", "evo_multiply", "evo_pulse", "evo_tornado", "evo_double_zap",
    # heroes whose ability is not simulated yet
    "hero_rowdy_reroll", "hero_dismount",
})


class Mechanic:
    kind = ""
    river_passable = False

    def on_spawn(self, m, u) -> None:
        pass

    def on_tick(self, m, u, dt: int) -> bool:
        return False

    def on_attack(self, m, u, target) -> bool:
        return False

    def on_damaged(self, m, u, amount: float, source) -> float:
        return amount

    def on_death(self, m, u) -> None:
        pass

    def waypoint(self, m, u, target) -> Optional[Tuple[float, float]]:
        return None


class SpellKind:
    """How a spell card is cast (creates an entity) and how that entity advances."""
    kind = ""

    def cast(self, m, side: int, card_id: str, x: float, y: float, eid: int, stats, params) -> None:
        raise NotImplementedError


def register(kind: str):
    def deco(cls):
        inst = cls()
        inst.kind = kind
        REGISTRY[kind] = inst
        for h in HOOK_NAMES:
            if getattr(cls, h) is not getattr(Mechanic, h):
                HOOKS[h].add(kind)
        hooks_for.cache_clear()
        return cls
    return deco


def register_spell(kind: str):
    def deco(cls):
        inst = cls()
        inst.kind = kind
        SPELLS[kind] = inst
        return cls
    return deco


@lru_cache(maxsize=None)
def hooks_for(mechanics: Tuple[str, ...], hook: str) -> Tuple[Mechanic, ...]:
    """Mechanic singletons of a unit that implement `hook`, in the unit's order (cached per tuple)."""
    impl = HOOKS[hook]
    return tuple(REGISTRY[k] for k in mechanics if k in impl)


def known_kinds() -> frozenset:
    return frozenset(REGISTRY) | frozenset(SPELLS) | DESCRIPTIVE | PLANNED


def params_of(u: Any, kind: str) -> Dict[str, Any]:
    return u.params.get(kind, {})
