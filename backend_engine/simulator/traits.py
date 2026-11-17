"""Trait registry: the reusable gameplay components a card is made of.

A card's behaviour comes from its data, never from its name:
  * stats / attack / movement / spell blocks (hp, damage, range, speed, splash, radius, ...)
  * `mechanics: [{kind, params}]` - each kind is a component in `simulator.mechanics` (REGISTRY / SPELLS) or a
    descriptive tag (DESCRIPTIVE) that needs no code.

`card_traits(card)` names the components a card uses in one vocabulary (TRAITS), whether they come from an
explicit mechanic or from stats (a building with a lifetime decays, an attack with a splash radius splashes).
The observation encoder, the catalog sync tool and the docs use it; the engine keeps running the mechanics.

Catalog flags (cards_stats_v2026.json), both optional:
  * `"enabled": false`  - the card is in the catalog (identity, cost, maybe stats) but not playable yet.
  * `"needs_hook": true` - the card uses a mechanic the engine does not simulate; it stays out of play until a
    component is written for it. `unsupported_kinds()` lists what is missing.
Only playable cards (`is_playable`) enter `CardDB.cards`; the rest live in `CardDB.catalog`.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, FrozenSet, List, Mapping

from backend_engine.simulator.mechanics import DESCRIPTIVE, PLANNED, REGISTRY, SPELLS


def _kinds(c: Mapping[str, Any]) -> set:
    return {m["kind"] if isinstance(m, dict) else m for m in c.get("mechanics") or []}


def _atk(c: Mapping[str, Any]) -> Mapping[str, Any]:
    return c.get("attack") or {}


def _stats(c: Mapping[str, Any]) -> Mapping[str, Any]:
    return c.get("base_stats") or c.get("stats_l11") or {}


def _spell(c: Mapping[str, Any]) -> Mapping[str, Any]:
    return c.get("spell") or {}


def _knocks_or_stuns(c: Mapping[str, Any]) -> bool:
    k = _kinds(c)
    if k & {"knockback", "apply_status", "root_spell", "pull_spell"} or _spell(c).get("knockback"):
        return True
    for m in c.get("mechanics") or []:
        if isinstance(m, dict) and any(t in str(m.get("params", {})) for t in ("stun", "freeze", "knockback")):
            return True
    return False


# trait -> (what it means, predicate over a normalised card record)
TRAITS: Dict[str, tuple] = {
    "building_only": ("targets buildings / towers only",
                      lambda c: list(_atk(c).get("targets") or []) == ["buildings"] or "building_targeting" in _kinds(c)),
    "heavy_knockback_immune": ("not pushed by knockback (except The Log)",
                               lambda c: "heavy_knockback_immune" in _kinds(c)),
    "shield_hp": ("has a shield that absorbs damage first",
                  lambda c: "shield" in _kinds(c) or _stats(c).get("shield_hitpoints", 0) > 0),
    "death_spawn": ("spawns units when it dies", lambda c: "death_spawn" in _kinds(c)),
    "death_damage": ("deals damage when it dies", lambda c: "death_damage" in _kinds(c)),
    "lifetime_decay": ("loses hp over a fixed lifetime",
                       lambda c: "decay" in _kinds(c) or (c.get("type") == "building" and bool(c.get("lifetime_s")))),
    "dot_spell": ("spell that damages in pulses over time", lambda c: "dot_spell" in _kinds(c)),
    "spawn_wave": ("spawns units periodically", lambda c: "periodic_spawn" in _kinds(c)),
    "charge_jump": ("charges or jumps the river", lambda c: bool(_kinds(c) & {"charge", "river_jump"})),
    "splash": ("area damage on each hit",
               lambda c: _atk(c).get("splash_radius_tiles", 0) > 0 or "splash" in _kinds(c)),
    "projectile": ("ranged attack with a travelling projectile", lambda c: bool(_atk(c).get("projectile_speed_tiles_s"))),
    "air": ("flying unit", lambda c: (c.get("movement") or {}).get("layer") == "air" or "air_unit" in _kinds(c)),
    "targets_air": ("can hit air units",
                    lambda c: "air" in (_atk(c).get("targets") or []) or bool(_spell(c).get("hits_air"))
                    or (c.get("type") == "spell" and "ground_only" not in _kinds(c) and "hits" not in _spell(c))),
    "stun_or_knockback": ("stuns, freezes, roots, pulls or knocks back", _knocks_or_stuns),
    "kamikaze": ("dies on its first hit", lambda c: "suicide" in _kinds(c)),
    "burrow": ("travels underground", lambda c: bool(_kinds(c) & {"burrow", "hide_underground"})),
    "stealth": ("turns invisible", lambda c: "invisible_when_idle" in _kinds(c)),
    "ramp_damage": ("damage or hit speed ramps up on one target", lambda c: bool(_kinds(c) & {"ramp_damage", "ramp_hit_speed"})),
    "ability": ("champion / hero ability", lambda c: "champion_ability" in _kinds(c) or c.get("type") == "champion"),
    "area_spell": ("instant area spell", lambda c: c.get("type") == "spell" and not (_kinds(c) & {"dot_spell", "rolling_spell"})
                   and "rolling_range_tiles" not in _spell(c)),
    "rolling_spell": ("rolls forward along the ground", lambda c: "rolling_range_tiles" in _spell(c) or "rolling_spell" in _kinds(c)),
    "has_evolution": ("has an evolution variant", lambda c: bool((c.get("variants") or {}).get("evo"))),
    "has_hero": ("has a hero variant", lambda c: bool((c.get("variants") or {}).get("hero"))),
}


def card_traits(card: Mapping[str, Any]) -> FrozenSet[str]:
    """The trait names (TRAITS keys) that apply to a card record."""
    return frozenset(name for name, (_, pred) in TRAITS.items() if pred(card))


def supported_kinds() -> FrozenSet[str]:
    """Mechanic kinds the engine can run today (components + spell kinds + descriptive tags)."""
    return frozenset(REGISTRY) | frozenset(SPELLS) | DESCRIPTIVE


def unsupported_kinds(card: Mapping[str, Any]) -> List[str]:
    """Base-card mechanic kinds with no engine component yet (unknown to the engine, or PLANNED)."""
    ok = supported_kinds()
    return sorted(k for k in _kinds(card) if k not in ok or k in PLANNED)


def is_playable(card: Mapping[str, Any]) -> bool:
    """In the playable pool: enabled, not waiting for a hook, and every base mechanic is simulated."""
    return bool(card.get("enabled", True)) and not card.get("needs_hook", False) and not unsupported_kinds(card)


def quarantine_reason(card: Mapping[str, Any]) -> str:
    """Why a catalog card is not playable ('' if it is)."""
    if card.get("needs_hook", False):
        return "needs_hook"
    if not card.get("enabled", True):
        return "disabled"
    miss = unsupported_kinds(card)
    return f"unsupported mechanics: {', '.join(miss)}" if miss else ""


TraitPredicate = Callable[[Mapping[str, Any]], bool]
