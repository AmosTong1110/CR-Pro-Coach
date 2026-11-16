"""Batch-1 mechanics: kamikaze hits (Ice / Fire / Electro Spirit), death bursts (Ice Golem), chain hits.

Status payloads are `{"tag": "stun" | "freeze" | "slow", "ms": int}` inside the mechanic's own params and are
applied through `Match.apply_status`, so the engine's single status check (`mechanics.status`) sees them.
"""
from __future__ import annotations

from typing import List

from backend_engine.simulator import geometry as G
from backend_engine.simulator.entities import AreaEffect
from backend_engine.simulator.mechanics import status as S
from backend_engine.simulator.mechanics.base import Mechanic, SpellKind, register, register_spell
from backend_engine.simulator.mechanics.core import knockback


def _hittable(m, u, e) -> bool:
    """Enemy `u` may hit: alive, allowed by its target flags, and not invisible."""
    return e.side != u.side and e.hp > 0 and m._can_target(u, e) and (not e.status or S.targetable(e))


def apply_payload(m, e, payload) -> None:
    """Apply a status payload to a unit that survived the hit."""
    if payload and e.hp > 0:
        m.apply_status(e, payload["tag"], int(payload["ms"]))


def chain_targets(m, u, first, count: int, radius: float) -> List:
    """`first` plus the nearest not-yet-hit enemy within `radius` of the previous one, `count` units in total.

    Deterministic: nearest first, ties broken by uid.
    """
    hit, seen, cur = [first], {first.uid}, first
    while len(hit) < count:
        near = [e for e in m.units.values()
                if e.uid not in seen and _hittable(m, u, e) and G.dist(e.x, e.y, cur.x, cur.y) - e.radius <= radius]
        if not near:
            break
        cur = min(near, key=lambda e: (G.dist(e.x, e.y, cur.x, cur.y), e.uid))
        hit.append(cur)
        seen.add(cur.uid)
    return hit


@register("suicide")
class Suicide(Mechanic):
    """Kamikaze: the first hit lands on arrival and the unit dies with it.

    params: `status` payload for everything hit; `chain` {count, radius_tiles} jumps on to more enemies
    (Electro Spirit). Otherwise the hit is an area hit of `u.splash` around the target, or a single hit.
    """

    def on_attack(self, m, u, target) -> bool:
        p = u.params.get("suicide", {})
        chain = p.get("chain")
        if chain:
            victims = chain_targets(m, u, target, int(chain["count"]), float(chain["radius_tiles"]))
        elif u.splash > 0:      # area damage also reaches invisible units (only *targeting* is blocked)
            victims = [e for e in sorted(m.units.values(), key=lambda e: e.uid)
                       if e.side != u.side and e.hp > 0 and m._can_target(u, e)
                       and G.dist(e.x, e.y, target.x, target.y) <= u.splash + e.radius]
        else:
            victims = [target]
        for e in victims:
            m._deal(e, u.damage, u)
            apply_payload(m, e, p.get("status"))
        u.hp = 0
        return True


@register("death_damage")
class DeathDamage(Mechanic):
    """Burst when the unit dies (Ice Golem). `damage` is filled in at spawn from the card's scaled `death_damage`.

    params: radius_tiles, damage, optional `status` payload. Hits every enemy unit and building in range.
    """

    def on_death(self, m, u) -> None:
        p = u.params.get("death_damage", {})
        dmg, radius = float(p.get("damage", 0)), float(p.get("radius_tiles", 0))
        for e in sorted(m.units.values(), key=lambda e: e.uid):
            if e.side != u.side and e.hp > 0 and G.dist(e.x, e.y, u.x, u.y) <= radius + e.radius:
                m._deal(e, dmg, u)
                apply_payload(m, e, p.get("status"))


@register("invisible_when_idle")
class InvisibleWhenIdle(Mechanic):
    """Royal Ghost: turns invisible after `delay_ms` without fighting, and shows again as soon as it has a target in
    range or attacks. Invisible units can't be targeted (see `status.targetable`) but area damage still hits them."""

    def on_tick(self, m, u, dt: int) -> bool:
        t = m.units.get(u.target_uid) if u.target_uid is not None else None
        if t is not None and t.hp > 0 and m._in_range(u, t):        # engaged: visible
            u.mstate["idle_ms"] = 0
            u.status.pop(S.INVISIBLE, None)
            return False
        idle = u.mstate.get("idle_ms", 0) + dt
        u.mstate["idle_ms"] = idle
        if idle >= int(u.params.get("invisible_when_idle", {}).get("delay_ms", 1800)):
            m.apply_status(u, S.INVISIBLE, 2 * dt)                  # refreshed every tick while idle
        return False

    def on_attack(self, m, u, target) -> bool:
        u.mstate["idle_ms"] = 0
        u.status.pop(S.INVISIBLE, None)
        return False


@register_spell("pull_spell")
class PullSpell(SpellKind):
    """Tornado: appears at once, then for `duration_ms` drags enemy troops to its centre (structures stay) and damages
    everything in range every `hit_every_ms` (towers take `crown_tower_damage`)."""

    def cast(self, m, side, card_id, x, y, eid, stats, params) -> None:
        sp = stats.spell
        dmg = stats.stats.get("damage", 0)
        m.effects.append(AreaEffect(m._uid(), side, card_id, x, y, sp["radius_tiles"], dmg,
                                    stats.stats.get("crown_tower_damage", dmg), int(sp["duration_ms"]),
                                    int(sp["hit_every_ms"]), float(sp["pull_speed_tiles_s"]), eid,
                                    next_hit_ms=int(sp["hit_every_ms"]), params=dict(params)))

    def advance(self, m, ef: AreaEffect, dt: int) -> None:
        ef.elapsed_ms += dt
        for e in sorted(m.units.values(), key=lambda e: e.uid):
            if e.side == ef.side or e.hp <= 0 or e.is_structure:
                continue
            d = G.dist(e.x, e.y, ef.x, ef.y)
            if 1e-9 < d <= ef.radius + e.radius:
                knockback(m, e, ef.x - e.x, ef.y - e.y, min(d, ef.pull_speed * dt / 1000.0))
        while ef.next_hit_ms <= ef.elapsed_ms and ef.next_hit_ms <= ef.duration_ms:
            ef.next_hit_ms += ef.hit_every_ms
            for e in sorted(m.units.values(), key=lambda e: e.uid):
                if e.side != ef.side and e.hp > 0 and G.dist(e.x, e.y, ef.x, ef.y) <= ef.radius + e.radius:
                    m._deal(e, ef.tower_damage if e.kind == "tower" else ef.damage, ef)
