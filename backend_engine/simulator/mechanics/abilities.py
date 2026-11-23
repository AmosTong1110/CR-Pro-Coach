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
from backend_engine.simulator.entities import Projectile
from backend_engine.simulator.mechanics import status as S
from backend_engine.simulator.mechanics.base import Mechanic, SpellKind, register, register_spell
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


class _Delayed:
    """Base for abilities that act once after `delay_ms` (the wiki's 1 s cast delay)."""
    phase = ""

    @classmethod
    def start(cls, m, u, p) -> None:
        u.mstate["ab"].update(phase=cls.phase, ms=0)

    @classmethod
    def tick(cls, m, u, p, dt: int) -> bool:
        ab = u.mstate["ab"]
        ab["ms"] += dt
        if ab["ms"] >= int(p.get("delay_ms", 1000)):
            ab["phase"] = None
            cls.fire(m, u, p)
        return False

    @staticmethod
    def fire(m, u, p) -> None:
        raise NotImplementedError


class Taunt(_Delayed):
    """Hero Knight, Triumphant Taunt (2 elixir): after the delay he gains a `shield` for `duration_ms` and every enemy
    troop, building and tower within `taunt_tiles` locks onto him until he is defeated."""
    phase = "taunt"

    @staticmethod
    def fire(m, u, p) -> None:
        hp = float(p.get("shield", 0))
        u.params["shield"] = {"hp": hp}
        u.mstate["shield"], u.mstate["shield_until"] = hp, m.t_ms + int(p.get("duration_ms", 5000))
        u.mechanics = tuple(dict.fromkeys(u.mechanics + ("shield",)))
        for e in _foes(m, u):
            if G.dist(e.x, e.y, u.x, u.y) <= float(p.get("taunt_tiles", 7.5)):
                e.mstate["taunt"] = u.uid


class BreakfastBoost(_Delayed):
    """Hero Mini P.E.K.K.A., Breakfast Boost (1 elixir): heals `heal_pct` of his hitpoints and gains `levels` levels
    (stats from the hero's own level table, which runs up to level 21). The pancake meter is not simulated: the
    ability always gives the 0-bar bonus (+1 level)."""
    phase = "boost"

    @staticmethod
    def fire(m, u, p) -> None:
        lvl = m.sides[u.side].level + int(p.get("levels", 1))
        if lvl in m.db.levels:
            st = m.db.stats(u.card_id, lvl, "hero").stats
        else:                                                      # above 16 only the hero's own table has numbers
            st = (m.db.variant(u.card_id, "hero") or {}).get("level_overrides", {}).get(str(lvl))
        if st:
            new_max = float(st["hitpoints"])
            u.hp = min(new_max, u.hp + (new_max - u.max_hp))       # keeps the missing hitpoints
            u.max_hp, u.damage = new_max, int(st["damage"])
        u.hp = min(u.max_hp, u.hp + u.max_hp * float(p.get("heal_pct", 30)) / 100.0)


class Snowstorm(_Delayed):
    """Hero Ice Golem, Snowstorm (2 elixir): a blizzard of `radius_tiles` where he stands: `pulses` pulses, each damages
    every enemy in it and slows troops by 30% for `slow_ms`."""
    phase = "snow"

    @staticmethod
    def fire(m, u, p) -> None:
        from backend_engine.simulator.entities import AreaEffect
        every, n = int(p.get("pulse_every_ms", 500)), int(p.get("pulses", 3))
        dmg = float(p.get("damage", 0))
        m.effects.append(AreaEffect(m._uid(), u.side, u.card_id, u.x, u.y, float(p.get("radius_tiles", 4.0)), dmg, dmg,
                                    every * n, every, 0.0, u.source_event_id or 0, kind="snowstorm", next_hit_ms=every,
                                    params={"state": {"slow_ms": int(p.get("slow_ms", 2000))}}))


@register_spell("snowstorm")
class SnowstormEffect(SpellKind):
    def cast(self, m, side, card_id, x, y, eid, stats, params) -> None:
        raise NotImplementedError("only started by the Hero Ice Golem ability")

    def advance(self, m, ef, dt: int) -> None:
        ef.elapsed_ms += dt
        while ef.next_hit_ms <= ef.elapsed_ms and ef.next_hit_ms <= ef.duration_ms:
            ef.next_hit_ms += ef.hit_every_ms
            for e in _foes_of(m, ef.side):
                if G.dist(e.x, e.y, ef.x, ef.y) <= ef.radius + e.radius:
                    m._deal(e, ef.damage, ef)
                    if e.hp > 0 and not e.is_structure:
                        m.apply_status(e, S.SLOW30, int(ef.params["state"]["slow_ms"]))


class TrustyTurret(_Delayed):
    """Hero Musketeer, Trusty Turret (3 elixir): `forward_tiles` ahead of her a short-lived turret building appears
    (`token`, 10 s) and hits enemies around it for `spawn_damage`."""
    phase = "turret"

    @staticmethod
    def fire(m, u, p) -> None:
        from backend_engine.simulator.mechanics.core import land
        x, y = land(u.x, min(max(u.y + G.forward(u.side) * float(p.get("forward_tiles", 3.0)), 0.0), G.H - 0.01))
        for e in _foes(m, u):
            if G.dist(e.x, e.y, x, y) <= float(p.get("spawn_radius_tiles", 1.5)) + e.radius:
                m._deal(e, float(p.get("spawn_damage", 0)), u)
        m._spawn_card_units(u.side, p["token"], x, y, u.source_event_id or 0, False, deploy_ms=int(p.get("deploy_ms", 1000)))


class WhirlPower:
    """Hero Valkyrie, Whirl Power (3 elixir): for `duration_ms` she spins, hitting every enemy troop and building within
    `radius_tiles` each `hit_every_ms` (towers take `tower_damage_pct`%) and takes `damage_reduction_pct`% less damage.
    The initial 5.5-tile dash is not simulated."""

    @staticmethod
    def start(m, u, p) -> None:
        u.mstate["ab"].update(phase="whirl", ms=0, next=int(p.get("hit_every_ms", 250)),
                              dr=float(p.get("damage_reduction_pct", 15)))

    @staticmethod
    def tick(m, u, p, dt: int) -> bool:
        ab = u.mstate["ab"]
        ab["ms"] += dt
        while ab["next"] <= ab["ms"] and ab["next"] <= int(p.get("duration_ms", 3500)):
            ab["next"] += int(p.get("hit_every_ms", 250))
            for e in _foes(m, u):
                if (e.is_structure or e.layer == "ground") and G.dist(e.x, e.y, u.x, u.y) <= float(
                        p.get("radius_tiles", 2.5)) + e.radius:
                    pct = float(p.get("tower_damage_pct", 50)) / 100.0 if e.kind == "tower" else 1.0
                    m._deal(e, float(p.get("damage", 0)) * pct, u)
        if ab["ms"] >= int(p.get("duration_ms", 3500)):
            ab["phase"] = None
            ab.pop("dr", None)
        return False


def _foes_of(m, side):
    return [e for e in sorted(m.units.values(), key=lambda e: e.uid) if e.side != side and e.hp > 0]


class PensiveProtection:
    """Monk, Pensive Protection (1 elixir): after a `delay_ms` cast (0.7 s since 2026-10-06) he takes `dr` percent
    less damage for `duration_ms` and reflects projectiles: a projectile hit on him also hits its shooter (the
    reflected shot's own flight is not simulated). He keeps fighting while it is active."""

    @staticmethod
    def start(m, u, p) -> None:
        u.mstate["ab"].update(phase="protect", ms=0)

    @staticmethod
    def tick(m, u, p, dt: int) -> bool:
        ab = u.mstate["ab"]
        ab["ms"] += dt
        delay, dur = int(p.get("delay_ms", 700)), int(p.get("duration_ms", 4000))
        if ab["ms"] >= delay + dur:
            ab.update(phase=None, dr=0, reflect=False)
        elif ab["ms"] >= delay:
            ab.update(dr=float(p.get("dr", 65)), reflect=bool(p.get("reflect", True)))
        return False


def _spawn(m, side: int, token: str, x: float, y: float, eid: int, deploy_ms: int = 0) -> list:
    before = set(m.units)
    m._spawn_card_units(side, token, x, y, eid, False, deploy_ms=deploy_ms)
    return [m.units[k] for k in sorted(set(m.units) - before)]


class CoffinCadets(_Delayed):
    """Hero Balloon, Coffin Cadets (2 elixir): a Skeletrooper (`token`) flies at the closest enemy ground unit (troop,
    building or tower) within `range_tiles` of the Balloon, lands after `delay_ms` for `landing_damage` (towers
    `landing_crown_tower_damage`) and then fights. Nothing in range: the ability is spent for nothing."""
    phase = "cadets"

    @staticmethod
    def fire(m, u, p) -> None:
        pool = [e for e in _foes(m, u) if (e.is_structure or e.layer == "ground") and S.targetable(e)
                and m._edge_dist(u, e) <= float(p.get("range_tiles", 6.5))]
        if not pool:
            return
        t = min(pool, key=lambda e: (m._edge_dist(u, e), e.uid))
        m._deal(t, float(p.get("landing_crown_tower_damage", 0)) if t.kind == "tower" else float(p.get("landing_damage", 0)), u)
        dx, dy = u.x - t.x, u.y - t.y
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        off = t.radius + 0.5
        _spawn(m, u.side, p["token"], t.x + dx / n * off, t.y + dy / n * off, u.source_event_id or 0)


class SurgingStrikes:
    """Hero Electro Wizard, Surging Strikes (2 elixir): two beams lock onto the two closest enemies in range and land
    `hits` zaps, one every `hit_every_ms`, each `damage_pct`% of his zap and a `stun_ms` stun (so the targets stay
    stunned while connected). Numbers from RoyaleAPI's estimate (5 hits, ~3.25 s); the wiki page is unfinished."""

    @staticmethod
    def start(m, u, p) -> None:
        u.mstate["ab"].update(phase="surge", ms=0, done=0)

    @staticmethod
    def tick(m, u, p, dt: int) -> bool:
        ab = u.mstate["ab"]
        ab["ms"] += dt
        every, hits = int(p.get("hit_every_ms", 650)), int(p.get("hits", 5))
        while ab["done"] < hits and ab["ms"] >= every * ab["done"]:
            ab["done"] += 1
            pool = [e for e in _foes(m, u) if m._can_target(u, e) and S.targetable(e) and m._in_range(u, e)]
            pool.sort(key=lambda e: (m._edge_dist(u, e), e.uid))
            for e in pool[:2]:
                m._deal(e, float(u.damage) * float(p.get("damage_pct", 127)) / 100.0, u)
                if e.hp > 0:
                    m.apply_status(e, S.STUN, int(p.get("stun_ms", 650)))
        if ab["done"] >= hits:
            ab["phase"] = None
        return False


class TripleThreat:
    """Hero Magic Archer, Triple Threat (2 elixir): leaves a decoy (`token`, `decoy_hitpoints`) where he stands, darts
    `teleport_tiles` back (away from the nearest enemy) and his next attack is three arrows (`triple_damage` each,
    `triple_range_tiles` long, 10 degrees apart)."""

    @staticmethod
    def start(m, u, p) -> None:
        for d in _spawn(m, u.side, p["token"], u.x, u.y, u.source_event_id or 0):
            d.hp = d.max_hp = float(p.get("decoy_hitpoints", d.max_hp))
        foes = [e for e in _foes(m, u) if e.kind != "tower"]
        if foes:
            e = min(foes, key=lambda e: (G.dist(e.x, e.y, u.x, u.y), e.uid))
            dx, dy = u.x - e.x, u.y - e.y
        else:
            dx, dy = 0.0, -float(G.forward(u.side))
        n = (dx * dx + dy * dy) ** 0.5 or 1.0
        step = float(p.get("teleport_tiles", 3.5))
        m._move_clamped(u, u.x + dx / n * step, u.y + dy / n * step)
        u.mstate["triple"] = {"angles": [-10, 0, 10], "length_tiles": float(p.get("triple_range_tiles", 13.5)),
                              "damage": float(p.get("triple_damage", 0))}
        u.target_uid = None

    @staticmethod
    def tick(m, u, p, dt: int) -> bool:
        return False


class WoundingWarp:
    """Hero Mega Minion, Wounding Warp (2 elixir): warps to the lowest-hitpoint enemy troop or building anywhere, hits
    it for `warp_damage`, fights it until it falls or `stay_ms` pass, then warps back to where he started."""

    @staticmethod
    def start(m, u, p) -> None:
        pool = [e for e in _foes(m, u) if e.kind != "tower" and m._can_target(u, e) and S.targetable(e)]
        if not pool:
            u.mstate["ab"]["phase"] = None
            return
        t = min(pool, key=lambda e: (e.hp, e.uid))
        u.mstate["ab"].update(phase="warp", ms=0, home=(u.x, u.y), target=t.uid)
        u.x, u.y = t.x, min(max(t.y - G.forward(u.side) * (t.radius + u.radius), 0.0), G.H - 0.01)
        m._deal(t, float(p.get("warp_damage", 0)), u)
        u.target_uid = t.uid
        u.cooldown_ms = u.hit_speed_ms

    @staticmethod
    def tick(m, u, p, dt: int) -> bool:
        ab = u.mstate["ab"]
        ab["ms"] += dt
        t = m.units.get(ab["target"])
        if t is None or t.hp <= 0 or ab["ms"] >= int(p.get("stay_ms", 3500)):
            u.x, u.y = ab["home"]
            ab["phase"] = None
            u.target_uid = None
        return False


ABILITIES: Dict[str, type] = {"dash": Dash, "explosive_escape": Escape, "royal_rescue": Rescue,
                              "triumphant_taunt": Taunt, "breakfast_boost": BreakfastBoost, "snowstorm": Snowstorm,
                              "trusty_turret": TrustyTurret, "whirl_power": WhirlPower,
                              "pensive_protection": PensiveProtection, "coffin_cadets": CoffinCadets,
                              "surging_strikes": SurgingStrikes, "triple_threat": TripleThreat,
                              "wounding_warp": WoundingWarp}


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
        if ab and ab["phase"] == "dash":
            return 0.0
        if ab and ab.get("reflect") and isinstance(source, Projectile):     # Monk: the shot goes back
            shooter = m.units.get(source.source_uid)
            if shooter is not None and shooter.hp > 0:
                m._deal(shooter, source.damage, u)
        if ab and ab.get("dr"):
            return amount * (1.0 - ab["dr"] / 100.0)
        return amount


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
