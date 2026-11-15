"""Built-in mechanics: the behaviour the engine had before the framework, moved here unchanged.

Any change here changes state hashes; tests/test_golden_hashes.py guards it.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from backend_engine.simulator import geometry as G
from backend_engine.simulator.entities import Projectile, RollingSpell, Spell
from backend_engine.simulator.mechanics.base import Mechanic, SpellKind, register, register_spell


@register("river_jump")
class RiverJump(Mechanic):
    """Hog Rider style: walks straight at the target and may cross / be pushed over the river."""
    river_passable = True

    def waypoint(self, m, u, target) -> Optional[Tuple[float, float]]:
        return target.x, target.y


@register("splash")
class Splash(Mechanic):
    """Each hit damages every valid enemy within `u.splash` of the target."""

    def on_attack(self, m, u, target) -> bool:
        for e in sorted(m.units.values(), key=lambda e: e.uid):
            if e.side != u.side and e.hp > 0 and m._can_target(u, e) \
                    and G.dist(e.x, e.y, target.x, target.y) <= u.splash + e.radius:
                m._deal(e, u.damage, u)
        return True


@register("decay")
class Decay(Mechanic):
    """Buildings lose hp every tick over their lifetime."""

    def on_tick(self, m, u, dt: int) -> bool:
        if u.decay_per_tick:
            u.hp -= u.decay_per_tick
            return u.hp <= 0
        return False


@register_spell("area_spell")
class AreaSpell(SpellKind):
    """Thrown from the king tower; lands after travel time and damages everything in the radius."""

    def cast(self, m, side, card_id, x, y, eid, stats, params) -> None:
        sp = stats.spell
        dmg = stats.stats.get("damage", 0)
        tdmg = stats.stats.get("crown_tower_damage", dmg)
        kx, ky, _ = G.tower_pos(side, "king")
        travel = G.dist(kx, ky, x, y) / sp.get("travel_speed_tiles_s", 1e9) * 1000
        m.spells.append(Spell(m._uid(), side, card_id, x, y, sp["radius_tiles"], dmg, tdmg,
                              int(math.ceil(travel / G.TICK_MS) * G.TICK_MS), eid, "area_spell", dict(params)))

    def impact(self, m, sp: Spell) -> None:
        for e in sorted(m.units.values(), key=lambda e: e.uid):
            if e.side != sp.side and e.hp > 0 and G.dist(e.x, e.y, sp.x, sp.y) <= sp.radius + e.radius:
                m._deal(e, sp.tower_damage if e.kind == "tower" else sp.damage, sp)
                kb = sp.params.get("knockback")
                if kb and e.hp > 0:
                    dx, dy = e.x - sp.x, e.y - sp.y
                    if abs(dx) + abs(dy) < 1e-9:            # dead centre: push away from the caster
                        dx, dy = 0.0, float(G.forward(sp.side))
                    knockback(m, e, dx, dy, float(kb.get("distance_tiles", 0)))


@register_spell("rolling_spell")
class Rolling(SpellKind):
    """Rolls forward from the drop point; hits each ground unit on its path once."""

    def cast(self, m, side, card_id, x, y, eid, stats, params) -> None:
        sp = stats.spell
        dmg = stats.stats.get("damage", 0)
        tdmg = stats.stats.get("crown_tower_damage", dmg)
        m.rolling.append(RollingSpell(m._uid(), side, card_id, x, y, G.forward(side),
                                      sp["rolling_range_tiles"], sp["width_tiles"], sp["travel_speed_tiles_s"],
                                      dmg, tdmg, eid, params=dict(params)))

    def advance(self, m, rs: RollingSpell, dt: int) -> None:
        adv = min(rs.speed * dt / 1000.0, rs.remaining)
        y0, y1 = rs.y, rs.y + rs.direction * adv
        lo, hi = min(y0, y1), max(y0, y1)
        for e in sorted(m.units.values(), key=lambda e: e.uid):
            if e.side == rs.side or e.hp <= 0 or e.layer != "ground" or e.uid in rs.hit:
                continue
            if abs(e.x - rs.x) <= rs.width / 2 + e.radius and lo - e.radius <= e.y <= hi + e.radius:
                rs.hit.add(e.uid)
                m._deal(e, rs.tower_damage if e.kind == "tower" else rs.damage, rs)
                kb = rs.params.get("knockback")
                if kb and e.hp > 0:
                    knockback(m, e, 0.0, float(rs.direction), float(kb.get("distance_tiles", 0)))
        rs.y, rs.remaining = y1, rs.remaining - adv


# -- engine version 2: first new mechanics ------------------------------------------------------------
@register("projectile")
class ProjectileAttack(Mechanic):
    """Ranged hit: launch a homing projectile; damage (and splash) lands when it arrives (Match.projectiles)."""

    def on_attack(self, m, u, target) -> bool:
        speed = float(u.params.get("projectile", {}).get("speed_tiles_s") or 0)
        if speed <= 0:
            return False
        m.projectiles.append(Projectile(m._uid(), u.side, u.uid, u.card_id, target.uid, u.x, u.y, speed,
                                        u.damage, u.splash, u.targets))
        return True


def advance_projectile(m, p: Projectile, dt: int) -> bool:
    """Move one tick; returns True when the projectile is finished (landed or target gone)."""
    t = m.units.get(p.target_uid)
    if t is None or t.hp <= 0:
        return True
    step = p.speed * dt / 1000.0
    d = G.dist(p.x, p.y, t.x, t.y)
    if d > step:
        p.x += (t.x - p.x) * step / d
        p.y += (t.y - p.y) * step / d
        return False
    if p.splash > 0:
        for e in sorted(m.units.values(), key=lambda e: e.uid):
            if e.side != p.side and e.hp > 0 and _can_hit(p.targets, e) \
                    and G.dist(e.x, e.y, t.x, t.y) <= p.splash + e.radius:
                m._deal(e, p.damage, p)
    else:
        m._deal(t, p.damage, p)
    return True


def _can_hit(targets, e) -> bool:
    if e.is_structure:
        return "buildings" in targets or "ground" in targets
    return e.layer in targets


def knockback(m, u, dx: float, dy: float, distance: float) -> None:
    """Push a troop by `distance` tiles along (dx, dy). Structures never move; the river stays blocked."""
    if u.is_structure or u.mass == 0 or distance <= 0:
        return
    n = math.hypot(dx, dy)
    if n < 1e-9:
        return
    m._move_clamped(u, u.x + dx / n * distance, u.y + dy / n * distance)


@register("damage_reduction_while_not_attacking")
class DamageReductionIdle(Mechanic):
    """Evo Knight: takes `damage_reduction` less damage unless it is hitting a target in range."""

    def on_damaged(self, m, u, amount: float, source) -> float:
        t = m.units.get(u.target_uid) if u.target_uid is not None else None
        attacking = u.deploy_ms <= 0 and t is not None and t.hp > 0 and m._in_range(u, t)
        if attacking:
            return amount
        red = float(u.params.get("damage_reduction_while_not_attacking", {}).get("damage_reduction", 0.0))
        return amount * (1.0 - red)
