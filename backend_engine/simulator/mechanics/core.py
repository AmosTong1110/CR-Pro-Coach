"""Built-in mechanics: the behaviour the engine had before the framework, moved here unchanged.

Any change here changes state hashes; tests/test_golden_hashes.py guards it.
"""
from __future__ import annotations

import math
from typing import Optional, Tuple

from backend_engine.simulator import geometry as G
from backend_engine.simulator.entities import RollingSpell, Spell
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
        rs.y, rs.remaining = y1, rs.remaining - adv
