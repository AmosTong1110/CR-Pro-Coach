"""Evolutions / hero helpers of the batch-4 cards (the 8 season meta decks). Sources: Clash Royale Wiki variant pages
(fetched 2026-11-23) with Supercell's 2026 balance notes; see tools/card_specs/variants_batch4.json for numbers and
`assumed` notes. The hero abilities themselves live in mechanics/abilities.py.

Evolved Royal Giant: `evo_recoil_blast`. Evolved Battle Ram: `evo_ram_trample` + `evo_ram_bounce` (and it no longer
breaks: the variant removes `break_on_hit`); its Evolved Barbarians: `boost_on_attack`. Evolved Goblin Cage:
`evo_cage_trap`. Evolved Goblin Drill: `evo_resurface`. Evolved Lumberjack's ghost: `ghost`. Evolved Princess:
`evo_frost_shots` (+ the `frost_zone` she leaves). Hero Magic Archer's decoy: `timed_life`.
Every loop is uid-sorted (deterministic); none of these kinds is used by an older card.
"""
from __future__ import annotations

from backend_engine.simulator import geometry as G
from backend_engine.simulator.entities import AreaEffect
from backend_engine.simulator.mechanics import status as S
from backend_engine.simulator.mechanics.base import Mechanic, SpellKind, register, register_spell
from backend_engine.simulator.mechanics.batch4 import _foes, _push_from, _stat


# ---------------------------------------------------------------- Evolved Royal Giant
@register("evo_recoil_blast")
class EvoRecoilBlast(Mechanic):
    """Every shot also hits ground enemies within `radius_tiles` of him for `recoil_damage` and knocks ground troops
    back `knockback_tiles`."""

    def after_attack(self, m, u, target) -> None:
        p = u.params.get("evo_recoil_blast", {})
        dmg = _stat(m, u, p.get("damage_stat", "recoil_damage"))
        for e in _foes(m, u.side):
            if (e.is_structure or e.layer == "ground") and G.dist(e.x, e.y, u.x, u.y) <= float(p.get("radius_tiles", 2.5)) + e.radius:
                m._deal(e, dmg, u)
                if not e.is_structure:
                    _push_from(m, e, u.x, u.y, u.side, float(p.get("knockback_tiles", 1.0)))


# ---------------------------------------------------------------- Evolved Battle Ram
@register("evo_ram_trample")
class EvoRamTrample(Mechanic):
    """While charging it runs over enemy ground troops in its way: each takes `charge_pushback_damage` once per charge
    and is knocked aside `knockback_tiles` (2 since 2026-09-08)."""

    def on_tick(self, m, u, dt: int) -> bool:
        if not u.mstate.get("charged"):
            u.mstate["trampled"] = []
            return False
        p = u.params.get("evo_ram_trample", {})
        hit = u.mstate.setdefault("trampled", [])
        dmg = _stat(m, u, p.get("damage_stat", "charge_pushback_damage"))
        for e in _foes(m, u.side):
            if e.is_structure or e.layer != "ground" or e.uid in hit:
                continue
            if G.dist(e.x, e.y, u.x, u.y) <= u.radius + e.radius + 0.1:
                hit.append(e.uid)
                m._deal(e, dmg, u)
                _push_from(m, e, u.x, u.y, u.side, float(p.get("knockback_tiles", 2.0)))
        return False


@register("evo_ram_bounce")
class EvoRamBounce(Mechanic):
    """After each hit on a building it bounces back `bounce_tiles` and comes again, until it is destroyed."""

    def after_attack(self, m, u, target) -> None:
        _push_from(m, u, target.x, target.y, 1 - u.side, float(u.params.get("evo_ram_bounce", {}).get("bounce_tiles", 2.0)),
                   ignores_heavy=True)
        u.cooldown_ms = max(u.cooldown_ms, u.first_hit_ms)


@register("boost_on_attack")
class BoostOnAttack(Mechanic):
    """Evolved Barbarians: each attack gives them RAGE (+30% speed and hit speed) for `ms`."""

    def after_attack(self, m, u, target) -> None:
        m.apply_status(u, S.RAGE, int(u.params.get("boost_on_attack", {}).get("ms", 3000)))


# ---------------------------------------------------------------- Evolved Goblin Cage
@register("evo_cage_trap")
class EvoCageTrap(Mechanic):
    """The nearest enemy ground troop within `radius_tiles` is pulled into the cage: held in it, stunned and not
    targetable, it takes `cage_damage` every `hit_every_ms`. One prisoner at a time; the next one `cooldown_ms` after
    the last one dies. The prisoner is freed when the cage breaks (its statuses run out)."""

    def on_tick(self, m, u, dt: int) -> bool:
        p = u.params.get("evo_cage_trap", {})
        st = u.mstate.setdefault("cage", {"prisoner": None, "next": 0, "free_at": 0})
        pr = m.units.get(st["prisoner"]) if st["prisoner"] is not None else None
        if pr is None or pr.hp <= 0:
            if st["prisoner"] is not None:
                st.update(prisoner=None, free_at=m.t_ms + int(p.get("cooldown_ms", 300)))
            pr = None
            if m.t_ms >= st["free_at"]:
                pool = [e for e in _foes(m, u.side) if e.kind == "troop" and e.layer == "ground"
                        and (not e.status or S.targetable(e)) and m._edge_dist(u, e) <= float(p.get("radius_tiles", 3.0))]
                if pool:
                    pr = min(pool, key=lambda e: (m._edge_dist(u, e), e.uid))
                    st.update(prisoner=pr.uid, next=m.t_ms + int(p.get("hit_every_ms", 1000)))
        if pr is not None:
            pr.x, pr.y = u.x, u.y
            m.apply_status(pr, S.STUN, 2 * dt)
            m.apply_status(pr, S.INVISIBLE, 2 * dt)
            if m.t_ms >= st["next"]:
                st["next"] += int(p.get("hit_every_ms", 1000))
                m._deal(pr, _stat(m, u, p.get("damage_stat", "cage_damage")), u)
        return False


# ---------------------------------------------------------------- Evolved Goblin Drill
@register("evo_resurface")
class EvoResurface(Mechanic):
    """At each fraction in `at_hp` (66%, 33%) of its hitpoints it leaves `goblins` `token`s behind and digs back up
    (untouchable for `dig_ms`): next to a Crown Tower it moves 90 degrees around that tower, elsewhere it stays put.
    No damage when it resurfaces (removed in Season 64)."""

    def on_tick(self, m, u, dt: int) -> bool:
        p = u.params.get("evo_resurface", {})
        if u.mstate.get("dig_until", -1) > m.t_ms:
            return False
        marks = list(p.get("at_hp", [0.66, 0.33]))
        i = u.mstate.get("surfaced", 0)
        if i >= len(marks) or u.hp > u.max_hp * float(marks[i]):
            return False
        u.mstate["surfaced"] = i + 1
        m._spawn_card_units(u.side, p.get("token", "drill_goblin"), u.x, u.y, u.source_event_id or 0, False,
                            count=int(p.get("goblins", 1)), deploy_ms=500)
        towers = [e for e in _foes(m, u.side) if e.kind == "tower" and m._edge_dist(u, e) <= float(p.get("tower_tiles", 2.0))]
        if towers:
            t = min(towers, key=lambda e: (m._edge_dist(u, e), e.uid))
            dx, dy = u.x - t.x, u.y - t.y
            for nx, ny in ((t.x - dy, t.y + dx), (t.x + dy, t.y - dx)):      # +90, then -90 degrees
                if 0.5 <= nx <= G.W - 0.5 and 0.5 <= ny <= G.H - 0.5 and G.region(ny) != "river":
                    u.x, u.y = nx, ny
                    break
        dig = int(p.get("dig_ms", 1000))
        u.mstate["dig_until"] = m.t_ms + dig
        m.apply_status(u, S.INVISIBLE, dig)
        return False

    def on_damaged(self, m, u, amount: float, source) -> float:
        return 0.0 if u.mstate.get("dig_until", -1) > m.t_ms else amount


# ---------------------------------------------------------------- Evolved Lumberjack's ghost
@register("ghost")
class Ghost(Mechanic):
    """Can't be targeted or hurt (spells still push, stun and slow it). Fights while the Rage lasts and fades
    `outside_ms` after it leaves the Rage area. `x`, `y`, `r`, `until` are set by the Lumberjack's `rage_on_death`."""

    def on_tick(self, m, u, dt: int) -> bool:
        p = u.params.get("ghost", {})
        m.apply_status(u, S.INVISIBLE, 2 * dt)
        inside = G.dist(u.x, u.y, float(p.get("x", u.x)), float(p.get("y", u.y))) <= float(p.get("r", 3.0))
        u.mstate["outside_ms"] = 0 if inside else u.mstate.get("outside_ms", 0) + dt
        if m.t_ms >= int(p.get("until", m.t_ms + 1)) or u.mstate["outside_ms"] > int(p.get("outside_ms", 1000)):
            u.hp = 0
            return True
        return False

    def on_damaged(self, m, u, amount: float, source) -> float:
        return 0.0


# ---------------------------------------------------------------- Evolved Princess
@register("evo_frost_shots")
class EvoFrostShots(Mechanic):
    """Her first shot and then every `every`-th one (every 2nd since 2026-08-04) slow enemy troops within
    `radius_tiles` of the impact by 30% for `slow_ms` (5.5 s since 2026-08-04). When she dies she leaves a slowing
    area of the same radius for `zone_ms`. List it before the projectile so the flag rides on the shot."""

    def on_attack(self, m, u, target) -> bool:
        p = u.params["evo_frost_shots"]
        n = u.mstate.get("shots", 0)
        p["armed"] = n == 0 or n % int(p.get("every", 2)) == 0
        u.mstate["shots"] = n + 1
        return False

    def on_projectile_hit(self, m, proj, target) -> None:
        p = proj.params.get("evo_frost_shots", {})
        if not p.get("armed") or target is None:
            return
        for e in _foes(m, proj.side):
            if not e.is_structure and G.dist(e.x, e.y, target.x, target.y) <= float(p.get("radius_tiles", 3.0)) + e.radius:
                m.apply_status(e, S.SLOW30, int(p.get("slow_ms", 5500)))

    def on_death(self, m, u) -> None:
        p = u.params.get("evo_frost_shots", {})
        m.effects.append(AreaEffect(m._uid(), u.side, u.card_id, u.x, u.y, float(p.get("radius_tiles", 3.0)), 0.0, 0.0,
                                    int(p.get("zone_ms", 5500)), G.TICK_MS, 0.0, u.source_event_id or 0,
                                    kind="frost_zone"))


@register_spell("frost_zone")
class FrostZone(SpellKind):
    """The Evolved Princess' parting area: enemy troops inside are slowed by 30% (refreshed every tick)."""

    def cast(self, m, side, card_id, x, y, eid, stats, params) -> None:   # pragma: no cover - never a card
        raise NotImplementedError

    def advance(self, m, ef: AreaEffect, dt: int) -> None:
        ef.elapsed_ms += dt
        for e in _foes(m, ef.side):
            if not e.is_structure and G.dist(e.x, e.y, ef.x, ef.y) <= ef.radius + e.radius:
                m.apply_status(e, S.SLOW30, 2 * dt)


# ---------------------------------------------------------------- Hero Magic Archer's decoy
@register("timed_life")
class TimedLife(Mechanic):
    """The unit is gone after `ms` (a troop with a lifetime: Hero Magic Archer's decoy)."""

    def on_tick(self, m, u, dt: int) -> bool:
        u.mstate["life_ms"] = u.mstate.get("life_ms", 0) + dt
        if u.mstate["life_ms"] >= int(u.params.get("timed_life", {}).get("ms", 7000)):
            u.mechanics = tuple(k for k in u.mechanics if k != "timed_life")
            u.hp = 0
            return True
        return False
