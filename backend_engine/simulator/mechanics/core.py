"""Built-in mechanics: the behaviour the engine had before the framework, moved here unchanged.

Any change here changes state hashes; tests/test_golden_hashes.py guards it.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from backend_engine.simulator import geometry as G
from backend_engine.simulator.entities import Projectile, RollingSpell, Spell
from backend_engine.simulator.mechanics.base import REGISTRY, Mechanic, SpellKind, hooks_for, register, register_spell


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
        # params {"center": "self"}: the area is around the attacker (Valkyrie), not around the target
        c = u if u.params.get("splash", {}).get("center") == "self" else target
        for e in sorted(m.units.values(), key=lambda e: e.uid):
            if e.side != u.side and e.hp > 0 and m._can_target(u, e) \
                    and G.dist(e.x, e.y, c.x, c.y) <= u.splash + e.radius:
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
                              int(math.ceil(travel / G.TICK_MS) * G.TICK_MS), eid, self.kind, dict(params)))

    def impact(self, m, sp: Spell) -> None:
        dz = sp.params.get("evo_double_zap")
        if dz:                                  # Evo Zap: a second, larger hit after `delay_ms`
            more = {k: v for k, v in sp.params.items() if k != "evo_double_zap"}
            m.spells.append(Spell(m._uid(), sp.side, sp.card_id, sp.x, sp.y,
                                  sp.radius + float(dz.get("radius_growth_tiles", 0.5)), sp.damage, sp.tower_damage,
                                  int(dz.get("delay_ms", 500)), sp.source_event_id, sp.kind, more))
        for e in sorted(m.units.values(), key=lambda e: e.uid):
            if e.side != sp.side and e.hp > 0 and G.dist(e.x, e.y, sp.x, sp.y) <= sp.radius + e.radius:
                m._deal(e, sp.tower_damage if e.kind == "tower" else sp.damage, sp)
                st = sp.params.get("apply_status")      # Zap: stun
                if st and e.hp > 0:
                    m.apply_status(e, st["tag"], int(st["ms"]))
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

    def finish(self, m, rs: RollingSpell) -> None:
        """The roll is over: spawn the card's token there (Barbarian Barrel)."""
        sp = rs.params.get("spawn_on_end")
        if sp:
            x, y = land(rs.x, rs.y)
            m._spawn_card_units(rs.side, sp["token"], x, y, rs.source_event_id, False, deploy_ms=int(sp["deploy_ms"]))


def land(x: float, y: float) -> Tuple[float, float]:
    """A spawn point that is not in the river (a roll can end there): move it to the nearer bank."""
    if G.region(y) == "river" and not G.on_bridge(x):
        y = G.RIVER_Y0 - 0.5 if y - G.RIVER_Y0 < G.RIVER_Y1 - y else G.RIVER_Y1 + 0.5
    return x, y


# -- engine version 2: first new mechanics ------------------------------------------------------------
@register("projectile")
class ProjectileAttack(Mechanic):
    """Ranged hit: launch a homing projectile; damage (and splash) lands when it arrives (Match.projectiles)."""

    def on_attack(self, m, u, target) -> bool:
        speed = float(u.params.get("projectile", {}).get("speed_tiles_s") or 0)
        if speed <= 0:
            return False
        payload = {mech.kind: dict(u.params.get(mech.kind, {})) for mech in hooks_for(u.mechanics, "on_projectile_hit")}
        m.projectiles.append(Projectile(m._uid(), u.side, u.uid, u.card_id, target.uid, u.x, u.y, speed,
                                        u.damage, u.splash, u.targets, params=payload))
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
    for kind in p.params:                       # payload effects on landing (Evo Firecracker sparks)
        REGISTRY[kind].on_projectile_hit(m, p, t)
    return True


def _can_hit(targets, e) -> bool:
    if e.is_structure:
        return "buildings" in targets or "ground" in targets
    return e.layer in targets


def knockback(m, u, dx: float, dy: float, distance: float) -> None:
    """Push a troop by `distance` tiles along (dx, dy). Structures never move; the river stays blocked."""
    if u.is_structure or u.mass == 0 or distance <= 0:
        return
    n = math.sqrt(dx * dx + dy * dy)
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
