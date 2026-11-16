"""Champion abilities (single use, 1 elixir; Aug 2026 patch: no cooldown): Golden Knight's Dashing Dash and Mighty
Miner's Explosive Escape.

An ability is triggered by the player through `Match.use_ability(side, card_id)`, which spends the elixir, records a
`PublicDeployEvent(is_ability=True)` and calls `start()` of the card's ability. The unit's `champion_ability`
mechanic then drives it tick by tick (`on_tick`). Everything is uid-sorted, so the engine stays deterministic.

Per-unit state lives in `u.mstate["ab"]`: {"used": bool, "phase": None | "dash"} plus the ability's own fields.
"""
from __future__ import annotations

from typing import Dict, Optional

from backend_engine.simulator import geometry as G
from backend_engine.simulator.mechanics import status as S
from backend_engine.simulator.mechanics.base import Mechanic, register, register_spell
from backend_engine.simulator.mechanics.core import AreaSpell

ABILITY_SUFFIX = "#ability"   # action-log marker: `golden_knight#ability` (cf. `card#evo` in the card refs)
DASH_SPEED = 6.0          # tiles / s while dashing (assumed: the wiki lists no dash speed)
DASH_REACH = 0.3          # edge distance at which a dash connects (tiles)
DASH_MAX_MS = 4000        # a dash chain never lasts longer than this


def _foes(m, u):
    return [e for e in sorted(m.units.values(), key=lambda e: e.uid) if e.side != u.side and e.hp > 0]


class Dash:
    """Golden Knight: chain of dashes onto the nearest enemy troop / building in sight that was not hit yet (at most
    `max_dashes`, each within `max_dash_tiles`). He cannot be hurt while dashing. Towers are not dash targets
    (assumed: the wiki says a tower hit used to end the dash and the king tower can't be woken by it)."""

    @staticmethod
    def start(m, u, p) -> None:
        u.mstate["ab"].update(phase="dash", hit=[], dashes=0, target=None, ms=0)
        u.target_uid = None
        Dash._next(m, u, p)

    @staticmethod
    def _next(m, u, p) -> None:
        ab = u.mstate["ab"]
        if ab["dashes"] >= int(p.get("max_dashes", 10)):
            ab["target"] = None
            return
        reach = float(p.get("max_dash_tiles", 5.5))
        pool = [e for e in _foes(m, u) if e.kind != "tower" and e.uid not in ab["hit"] and m._can_target(u, e)
                and S.targetable(e) and m._edge_dist(u, e) <= reach]
        ab["target"] = min(pool, key=lambda e: (m._edge_dist(u, e), e.uid)).uid if pool else None

    @staticmethod
    def tick(m, u, p, dt: int) -> bool:
        ab = u.mstate["ab"]
        ab["ms"] += dt
        t = m.units.get(ab["target"]) if ab["target"] is not None else None
        if t is None or t.hp <= 0:
            if ab["target"] is not None:                       # the victim died on its own: pick the next one
                Dash._next(m, u, p)
                t = m.units.get(ab["target"]) if ab["target"] is not None else None
            if t is None or ab["ms"] >= DASH_MAX_MS:
                ab["phase"] = None
                return False
        d = G.dist(u.x, u.y, t.x, t.y)
        step = float(p.get("dash_speed_tiles_s", DASH_SPEED)) * dt / 1000.0
        if m._edge_dist(u, t) > DASH_REACH:
            f = min(1.0, step / d) if d > 1e-9 else 1.0
            u.x = min(max(u.x + (t.x - u.x) * f, 0.0), G.W - 0.01)       # a dash may cross the river
            u.y = min(max(u.y + (t.y - u.y) * f, 0.0), G.H - 0.01)
        if m._edge_dist(u, t) <= DASH_REACH:
            m._deal(t, float(p.get("damage", u.damage)), u)
            ab["hit"].append(t.uid)
            ab["dashes"] += 1
            Dash._next(m, u, p)
        if ab["target"] is None or ab["ms"] >= DASH_MAX_MS:
            ab["phase"] = None
            u.cooldown_ms = max(u.cooldown_ms, u.first_hit_ms)
        return True


class Escape:
    """Mighty Miner: digs under to the horizontally opposite spot (x mirrored) and is gone for `dig_ms`; a bomb stays
    where he was and explodes `bomb_ms` later (damage + knockback in `radius_tiles`; towers take it too: assumed)."""

    @staticmethod
    def start(m, u, p) -> None:
        from backend_engine.simulator.entities import Spell
        dmg = float(p.get("damage", 0))
        m.spells.append(Spell(m._uid(), u.side, u.card_id, u.x, u.y, float(p.get("radius_tiles", 2.5)), dmg, dmg,
                              int(p.get("bomb_ms", 1000)), u.source_event_id or 0, "escape_bomb",
                              {"knockback": {"distance_tiles": float(p.get("knockback_tiles", 1.0))}}))
        u.x = min(max(G.W - u.x, 0.0), G.W - 0.01)
        u.target_uid = None
        u.deploy_ms = int(p.get("dig_ms", 933))               # underground: can't act, can't be targeted
        u.cooldown_ms = max(u.cooldown_ms, u.first_hit_ms)
        m.apply_status(u, S.INVISIBLE, u.deploy_ms)

    @staticmethod
    def tick(m, u, p, dt: int) -> bool:
        return False


class Rescue:
    """Little Prince, Royal Rescue: after `delay_ms` Guardienne charges in `dash_tiles` in front of him: ground troops
    around the landing point take `damage` and are knocked away (up to 2 tiles, less the farther from the centre), and
    she stays on the field as a unit (token `token`). Landing spot, radius and knockback shape are assumed."""

    @staticmethod
    def start(m, u, p) -> None:
        u.mstate["ab"].update(phase="rescue", ms=0)

    @staticmethod
    def tick(m, u, p, dt: int) -> bool:
        from backend_engine.simulator.mechanics.core import knockback, land
        ab = u.mstate["ab"]
        ab["ms"] += dt
        if ab["ms"] < int(p.get("delay_ms", 1000)):
            return False
        ab["phase"] = None
        x, y = land(u.x, min(max(u.y + G.forward(u.side) * float(p.get("dash_tiles", 4.0)), 0.0), G.H - 0.01))
        radius = float(p.get("radius_tiles", 2.0))
        for e in _foes(m, u):
            d = G.dist(e.x, e.y, x, y)
            if e.is_structure or e.layer != "ground" or d > radius + e.radius:
                continue
            m._deal(e, float(p.get("damage", 0)), u)
            if e.hp > 0:
                dx, dy = (e.x - x, e.y - y) if d > 1e-9 else (0.0, float(G.forward(u.side)))
                knockback(m, e, dx, dy, float(p.get("knockback_tiles", 2.0)) * max(0.0, 1.0 - d / (radius + e.radius)))
        m._spawn_card_units(u.side, p["token"], x, y, u.source_event_id or 0, False, deploy_ms=int(p.get("deploy_ms", 300)))
        return False


ABILITIES: Dict[str, type] = {"dash": Dash, "explosive_escape": Escape, "royal_rescue": Rescue}


@register("champion_ability")
class ChampionAbility(Mechanic):
    def on_spawn(self, m, u) -> None:
        u.mstate["ab"] = {"used": False, "phase": None}

    def on_tick(self, m, u, dt: int) -> bool:
        ab = u.mstate.get("ab")
        if ab and ab["phase"]:
            p = u.params["champion_ability"]
            return ABILITIES[p["name"]].tick(m, u, p, dt)
        return False

    def on_damaged(self, m, u, amount: float, source) -> float:
        ab = u.mstate.get("ab")
        return 0.0 if ab and ab["phase"] == "dash" else amount


@register_spell("escape_bomb")
class EscapeBomb(AreaSpell):
    """The bomb the Mighty Miner leaves behind: an area hit with knockback (cast only through the ability)."""


def ability_unit(m, side: int, card_id: str) -> Optional[object]:
    """The champion unit of `side` that can use its ability now (oldest first), or None."""
    for u in sorted(m.units.values(), key=lambda e: e.uid):
        ab = u.mstate.get("ab")
        if (u.side == side and u.card_id == card_id and u.hp > 0 and ab is not None and not ab["used"]
                and u.deploy_ms <= 0 and ab["phase"] is None and (not u.status or S.can_attack(u))):
            return u
    return None


def start_ability(m, u) -> None:
    p = u.params["champion_ability"]
    u.mstate["ab"]["used"] = True
    ABILITIES[p["name"]].start(m, u, p)
