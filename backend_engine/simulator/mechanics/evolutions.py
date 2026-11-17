"""Evolutions, batch 2 (2026-11-16): Firecracker, Skeletons, Zap, Valkyrie, Tesla (Evo Knight is in core.py).

Numbers come from the card data (`variants.evo.level_overrides`, read from the saved wiki pages by
tools/add_variants.py; ability columns via the spec's `stats_extra`) and from each mechanic's params in
tools/card_specs/variants.json. Values the wiki doesn't give are listed in that spec's `assumed`.

* evo_sparks     (Firecracker) her rocket breaks into 5 shrapnel that fly on; a big spark (radius 2.5, 3 s) is left
                 where it lands and a small one (radius 1.2, 2.5 s) where each shrapnel stops. Every 0.25 s each enemy
                 in any spark takes `spark_damage` once (sparks don't stack; towers take the crown-tower value) and
                 is slowed 15%.
* evo_multiply   (Skeletons) after every attack a new Evolved Skeleton spawns next to the attacker while fewer than
                 8 of that play are alive.
* evo_double_zap (Zap) a second hit with the same damage and stun, 0.5 tiles wider (core.AreaSpell.impact).
* evo_tornado    (Valkyrie) every attack makes a 0.5 s tornado around her (radius 5): pulls enemy troops in and
                 deals `tornado_damage` once (towers: the crown-tower value).
* evo_pulse      (Tesla) an electric pulse (radius 6, `pulse_damage`, 0.5 s stun) each time it comes up from
                 underground, including the first time after it deploys, and when it dies (destroyed or expired).
"""
from __future__ import annotations

import math

from backend_engine.simulator import geometry as G
from backend_engine.simulator.entities import AreaEffect
from backend_engine.simulator.mechanics import status as S
from backend_engine.simulator.mechanics.base import Mechanic, SpellKind, register, register_spell


def _enemies_in(m, side: int, x: float, y: float, radius: float, structures: bool = True):
    for e in sorted(m.units.values(), key=lambda e: e.uid):
        if e.side != side and e.hp > 0 and (structures or not e.is_structure) \
                and G.dist(e.x, e.y, x, y) <= radius + e.radius:
            yield e


# ---------------------------------------------------------------- Firecracker
@register("evo_sparks")
class EvoSparks(Mechanic):
    """Rides on the Firecracker's projectile (Projectile.params) and lays the sparks where it lands."""

    def on_projectile_hit(self, m, p, target) -> None:
        sp = p.params["evo_sparks"]
        shooter = m.units.get(p.source_uid)
        sx, sy = (shooter.x, shooter.y) if shooter is not None else (p.x, p.y - G.forward(p.side))
        tx, ty = target.x, target.y
        dx, dy = tx - sx, ty - sy
        d0 = math.sqrt(dx * dx + dy * dy)
        if d0 < 1e-9:
            dx, dy, d0 = 0.0, float(G.forward(p.side)), 1.0
        base = math.atan2(dy, dx)
        travel = max(float(sp["shrapnel_range_tiles"]) - d0, 0.0)
        n = int(sp["small_count"])
        fan = math.radians(float(sp["fan_degrees"]))
        circles = [[tx, ty, float(sp["big_radius_tiles"]), int(sp["big_duration_ms"])]]
        for i in range(n):
            a = base + (fan * (i / (n - 1) - 0.5) if n > 1 else 0.0)
            cx = min(max(tx + math.cos(a) * travel, 0.0), G.W - 0.01)
            cy = min(max(ty + math.sin(a) * travel, 0.0), G.H - 0.01)
            circles.append([cx, cy, float(sp["small_radius_tiles"]), int(sp["small_duration_ms"])])
        every = int(sp["hit_every_ms"])
        m.effects.append(AreaEffect(m._uid(), p.side, p.card_id, tx, ty, float(sp["big_radius_tiles"]),
                                    float(sp.get("damage", 0)), float(sp.get("tower_damage", 0)),
                                    max(c[3] for c in circles), every, 0.0,
                                    (shooter.source_event_id or 0) if shooter is not None else 0,
                                    kind="spark_field", next_hit_ms=every,
                                    params={"circles": circles, "slow": sp.get("slow", S.SLOW15),
                                            "slow_ms": int(sp.get("slow_ms", every + 100))}))


@register_spell("spark_field")
class SparkField(SpellKind):
    """AreaEffect of Evo Firecracker sparks: union of circles, one hit per enemy per pulse (no stacking)."""

    def advance(self, m, ef: AreaEffect, dt: int) -> None:
        ef.elapsed_ms += dt
        while ef.next_hit_ms <= ef.elapsed_ms and ef.next_hit_ms <= ef.duration_ms:
            t = ef.next_hit_ms
            ef.next_hit_ms += ef.hit_every_ms
            live = [c for c in ef.params["circles"] if t <= c[3]]
            for e in sorted(m.units.values(), key=lambda e: e.uid):
                if e.side == ef.side or e.hp <= 0:
                    continue
                if any(G.dist(e.x, e.y, cx, cy) <= r + e.radius for cx, cy, r, _ in live):
                    m._deal(e, ef.tower_damage if e.kind == "tower" else ef.damage, ef)
                    if e.hp > 0 and not e.is_structure:
                        m.apply_status(e, ef.params["slow"], int(ef.params["slow_ms"]))


# ---------------------------------------------------------------- Skeletons
@register("evo_multiply")
class EvoMultiply(Mechanic):
    def after_attack(self, m, u, target) -> None:
        p = u.params["evo_multiply"]
        alive = sum(1 for e in m.units.values() if e.hp > 0 and e.side == u.side and e.card_id == u.card_id
                    and e.source_event_id == u.source_event_id)
        if alive >= int(p["max_alive"]):
            return
        back = -0.5 * G.forward(u.side)                   # just behind the attacker
        m._spawn_card_units(u.side, u.card_id, u.x, min(max(u.y + back, 0.0), G.H - 0.01), u.source_event_id,
                            True, deploy_ms=int(p.get("spawn_deploy_ms", 0)), variant="evo", count=1)


# ---------------------------------------------------------------- Zap (handled by core.AreaSpell.impact)
@register("evo_double_zap")
class EvoDoubleZap(Mechanic):
    """Marker: the spell's params carry it; core.AreaSpell.impact casts the second, wider hit."""


# ---------------------------------------------------------------- Valkyrie
@register("evo_tornado")
class EvoTornado(Mechanic):
    def after_attack(self, m, u, target) -> None:
        p = u.params["evo_tornado"]
        m.effects.append(AreaEffect(m._uid(), u.side, u.card_id, u.x, u.y, float(p["radius_tiles"]),
                                    float(p.get("damage", 0)), float(p.get("tower_damage", 0)),
                                    int(p["duration_ms"]), int(p["duration_ms"]) + 1, float(p["pull_speed_tiles_s"]),
                                    u.source_event_id or 0, kind="pull_spell", next_hit_ms=0))


# ---------------------------------------------------------------- Tesla
def _pulse(m, u) -> None:
    p = u.params["evo_pulse"]
    for e in list(_enemies_in(m, u.side, u.x, u.y, float(p["radius_tiles"]))):
        if e.kind == "tower":
            continue                                       # no crown-tower damage on the wiki page
        m._deal(e, float(p.get("damage", 0)), u)
        if e.hp > 0:
            m.apply_status(e, S.STUN, int(p["stun_ms"]))


@register("evo_pulse")
class EvoPulse(Mechanic):
    def on_tick(self, m, u, dt: int) -> bool:
        hidden = bool(u.mstate.get("hidden", False))
        was = u.mstate.get("pulse_was_hidden", True)       # a fresh Tesla counts as coming up
        if was and not hidden:
            _pulse(m, u)
        u.mstate["pulse_was_hidden"] = hidden
        return False

    def on_death(self, m, u) -> None:
        _pulse(m, u)


# ---------------------------------------------------------------- Mortar
@register("evo_goblin_mortar")
class EvoGoblinMortar(Mechanic):
    """Evolved Mortar: launches a Goblin at the target upon projectile impact."""

    def on_projectile_hit(self, m, p, target) -> None:
        token_id = p.params.get("evo_goblin_mortar", {}).get("token", "goblin")
        shooter = m.units.get(p.source_uid)
        eid = (shooter.source_event_id or 0) if shooter is not None else 0
        tx = target.x if target is not None else p.x
        ty = target.y if target is not None else p.y
        try:
            m._spawn_card_units(p.side, token_id, tx, ty, eid, False, deploy_ms=500)
        except (KeyError, ValueError):
            pass


