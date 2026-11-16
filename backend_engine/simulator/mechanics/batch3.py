"""Batch-3 mechanics (the five modern cards): Rune Giant's enchantment (`enchant` + the `enchanted` tag it puts on
allies) and Little Prince's attack ramp (`ramp_hit_speed`). Poison is a `dot_spell` with `hits_air`."""
from __future__ import annotations

from backend_engine.simulator import geometry as G
from backend_engine.simulator.mechanics.base import Mechanic, register


def _set_enchanted(a, params) -> None:
    a.params["enchanted"] = params
    a.mechanics = tuple(dict.fromkeys(a.mechanics + ("enchanted",)))


def _clear_enchanted(a) -> None:
    a.params.pop("enchanted", None)
    a.mechanics = tuple(k for k in a.mechanics if k != "enchanted")


@register("enchant")
class Enchant(Mechanic):
    """Rune Giant: the `limit` allied troops closest to her (within `range_tiles`) deal `bonus` extra damage on every
    `every`-th attack. Enchantments are handed out as soon as a slot is free (the wiki's throw delay is not
    simulated) and last `linger_ms` after she dies."""

    def on_spawn(self, m, u) -> None:
        u.mstate["ench"] = []

    def on_tick(self, m, u, dt: int) -> bool:
        p = u.params.get("enchant", {})
        live = [uid for uid in u.mstate["ench"] if uid in m.units and m.units[uid].hp > 0
                and "enchanted" in m.units[uid].mechanics]
        free = int(p.get("limit", 2)) - len(live)
        if free > 0:
            pool = [e for e in sorted(m.units.values(), key=lambda e: e.uid)
                    if e.side == u.side and e.uid != u.uid and e.kind == "troop" and e.hp > 0
                    and e.uid not in live and "enchanted" not in e.mechanics and e.deploy_ms <= 0
                    and G.dist(e.x, e.y, u.x, u.y) <= float(p.get("range_tiles", 8.5))]
            pool.sort(key=lambda e: (G.dist(e.x, e.y, u.x, u.y), e.uid))
            for e in pool[:free]:
                _set_enchanted(e, {"every": int(p.get("every", 3)), "bonus": float(p.get("bonus", 0)), "count": 0,
                                   "expires_ms": None})
                live.append(e.uid)
        u.mstate["ench"] = live
        return False

    def on_death(self, m, u) -> None:
        linger = int(u.params.get("enchant", {}).get("linger_ms", 5000))
        for uid in u.mstate.get("ench", []):
            a = m.units.get(uid)
            if a is not None and "enchanted" in a.params:
                a.params["enchanted"]["expires_ms"] = m.t_ms + linger


@register("enchanted")
class Enchanted(Mechanic):
    """An ally under Rune Giant's enchantment: every `every`-th attack adds `bonus` damage on its target."""

    def on_tick(self, m, u, dt: int) -> bool:
        exp = u.params.get("enchanted", {}).get("expires_ms")
        if exp is not None and m.t_ms >= exp:
            _clear_enchanted(u)
        return False

    def on_attack(self, m, u, target) -> bool:
        p = u.params.get("enchanted")
        if p is None:
            return False
        p["count"] += 1
        if p["count"] % int(p["every"]) == 0:
            m._deal(target, float(p["bonus"]), u)
        return False


@register("ramp_hit_speed")
class RampHitSpeed(Mechanic):
    """Little Prince: attacks get faster the longer he keeps hitting the same target: stage n (0-based) starts after
    n * `hits_per_stage` hits, with hit speeds `hit_speeds_ms`. A new target starts over."""

    def on_attack(self, m, u, target) -> bool:
        p = u.params.get("ramp_hit_speed", {})
        if u.mstate.get("ramp_target") != target.uid:
            u.mstate.update(ramp_target=target.uid, ramp_hits=0)
        speeds = p.get("hit_speeds_ms") or [u.hit_speed_ms]
        stage = min(len(speeds) - 1, u.mstate["ramp_hits"] // int(p.get("hits_per_stage", 2)))
        u.hit_speed_ms = int(speeds[stage])                 # the engine arms the next cooldown with this value
        u.mstate["ramp_hits"] += 1
        return False
