"""Batch-2 mechanics (engine v6): charge + shield (Dark Prince), burrow + reduced tower damage (Miner),
damage ramp (Mighty Miner), recoil (Firecracker), hide underground (Tesla), form change (Goblin Demolisher) and the
spell kinds `dot_spell` (Earthquake), `top_n_spell` (Lightning), `root_spell` (Vines), `delayed_drop` (Royal Delivery).

Numbers come from the card data (`params`, filled at spawn from the scaled stats); the engine stays deterministic:
every loop is uid-sorted.
"""
from __future__ import annotations

import math

from backend_engine.simulator import geometry as G
from backend_engine.simulator.entities import AreaEffect, Spell
from backend_engine.simulator.mechanics import status as S
from backend_engine.simulator.mechanics.base import Mechanic, SpellKind, register, register_spell
from backend_engine.simulator.mechanics.core import AreaSpell


def _enemies(m, side):
    return [e for e in sorted(m.units.values(), key=lambda e: e.uid) if e.side != side and e.hp > 0]


def fill_spawn_params(params: dict, stats) -> None:
    """Complete mechanic params that depend on the card's scaled stats (called when a unit spawns)."""
    st = stats.stats
    if "charge" in params:
        params["charge"]["damage"] = st.get(params["charge"].get("damage_stat", "damage"), 0)
    if "shield" in params:
        params["shield"]["hp"] = st.get("shield_hitpoints", 0)
    if "tower_damage" in params:
        params["tower_damage"]["damage"] = st.get("crown_tower_damage", 0)
    if "ramp_damage" in params:
        params["ramp_damage"]["damages"] = [st[k] for k in params["ramp_damage"]["stats"]]
    for kind, p in params.items():                  # evolutions: "<x>_stat": stat name -> "<x>": scaled value
        if kind.startswith("evo_"):
            for k in [k for k in p if k.endswith("_stat")]:
                p[k[: -len("_stat")]] = st.get(p[k], 0)
    if "champion_ability" in params:
        ca = params["champion_ability"]
        ca["damage"] = st.get(ca.get("damage_stat", "damage"), 0)
        for k in [k for k in ca if k.endswith("_stat") and k != "damage_stat"]:    # e.g. shield_stat -> shield
            ca[k[: -len("_stat")]] = st.get(ca[k], 0)
    if "enchant" in params:
        params["enchant"]["bonus"] = st.get(params["enchant"].get("bonus_stat", "bonus_damage"), 0)
    if "form_change" in params:
        fc = params["form_change"]
        fc["death_damage"] = st.get("death_damage", 0)
        fc["hp_loss"] = st.get(fc.get("hp_loss_stat", ""), 0)


# ---------------------------------------------------------------- Dark Prince
@register("charge")
class Charge(Mechanic):
    """Runs `charge_tiles` in a straight line without stopping to become charged: faster, and the next hit does the
    charge damage. A pause before it is charged resets the run; once charged it stays charged until it hits (the
    first-hit wind-up at the target does not cost the charge). Splash is around the target (u.splash)."""

    def on_spawn(self, m, u) -> None:
        u.mstate.update(base_speed=u.speed, run=0.0, last=(u.x, u.y), charged=False)

    def on_tick(self, m, u, dt: int) -> bool:
        p = u.params.get("charge", {})
        lx, ly = u.mstate["last"]
        moved = math.hypot(u.x - lx, u.y - ly)
        u.mstate["last"] = (u.x, u.y)
        u.mstate["run"] = u.mstate["run"] + moved if moved > 1e-6 else 0.0
        charged = u.mstate["charged"] or u.mstate["run"] >= float(p.get("charge_tiles", 3.0))
        u.mstate["charged"] = charged
        u.speed = float(p.get("speed_tiles_s", u.mstate["base_speed"])) if charged else u.mstate["base_speed"]
        return False

    def on_attack(self, m, u, target) -> bool:
        p = u.params.get("charge", {})
        dmg = float(p.get("damage", u.damage)) if u.mstate.get("charged") else u.damage
        for e in _enemies(m, u.side):
            if m._can_target(u, e) and G.dist(e.x, e.y, target.x, target.y) <= u.splash + e.radius:
                m._deal(e, dmg, u)
        u.mstate.update(run=0.0, charged=False)
        u.speed = u.mstate["base_speed"]
        return True


# ---------------------------------------------------------------- shields (Dark Prince, Royal Recruit)
@register("shield")
class Shield(Mechanic):
    """A shield in front of the hitpoints: damage empties it first, the rest carries over to the hitpoints."""

    def on_spawn(self, m, u) -> None:
        u.mstate["shield"] = float(u.params.get("shield", {}).get("hp", 0))

    def on_tick(self, m, u, dt: int) -> bool:
        until = u.mstate.get("shield_until")              # a temporary shield (Hero Knight's taunt)
        if until is not None and m.t_ms >= until:
            u.mstate["shield"] = 0.0
            del u.mstate["shield_until"]
        return False

    def on_damaged(self, m, u, amount: float, source) -> float:
        sh = u.mstate.get("shield", 0.0)
        if sh <= 0:
            return amount
        absorbed = min(sh, amount)
        u.mstate["shield"] = sh - absorbed
        return amount - absorbed


# ---------------------------------------------------------------- Miner
@register("burrow")
class Burrow(Mechanic):
    """Digs in: can be placed anywhere (see Match.check_play) and can't be targeted until it has surfaced."""

    def on_spawn(self, m, u) -> None:
        if u.deploy_ms > 0:
            m.apply_status(u, S.INVISIBLE, u.deploy_ms)


@register("tower_damage")
class TowerDamage(Mechanic):
    """Hits towers for the card's `crown_tower_damage` (Miner), everything else for `damage`."""

    def on_attack(self, m, u, target) -> bool:
        dmg = u.params.get("tower_damage", {}).get("damage")
        if target.kind == "tower" and dmg:
            m._deal(target, float(dmg), u)
            return True
        return False


# ---------------------------------------------------------------- Mighty Miner
@register("ramp_damage")
class RampDamage(Mechanic):
    """Damage grows in stages while it keeps hitting the same target: stage n after n * `stage_ms` of hitting."""

    def on_attack(self, m, u, target) -> bool:
        p = u.params.get("ramp_damage", {})
        if u.mstate.get("ramp_target") != target.uid:
            u.mstate.update(ramp_target=target.uid, ramp_ms=0)
        else:
            u.mstate["ramp_ms"] += u.hit_speed_ms
        damages = p.get("damages") or [u.damage]
        stage = min(len(damages) - 1, u.mstate["ramp_ms"] // int(p.get("stage_ms", 2000)))
        m._deal(target, float(damages[stage]), u)
        return True


# ---------------------------------------------------------------- Firecracker
@register("recoil")
class Recoil(Mechanic):
    """After every shot the unit is pushed `tiles` straight away from its target (the shot is still fired)."""

    def on_attack(self, m, u, target) -> bool:
        dx, dy = u.x - target.x, u.y - target.y
        n = math.hypot(dx, dy)
        if n > 1e-9:
            d = float(u.params.get("recoil", {}).get("tiles", 1.0))
            m._move_clamped(u, u.x + dx / n * d, u.y + dy / n * d)
        return False


# ---------------------------------------------------------------- Tesla
@register("hide_underground")
class HideUnderground(Mechanic):
    """Underground (untargetable, takes no damage) until an enemy troop is within range; it surfaces, shoots, and
    hides again when nothing is in range."""

    def on_tick(self, m, u, dt: int) -> bool:
        near = any(not e.is_structure and m._can_target(u, e) and m._in_range(u, e) for e in _enemies(m, u.side))
        hidden = u.mstate.get("hidden", False)
        if near and hidden:
            u.mstate["hidden"] = False
            u.status.pop(S.INVISIBLE, None)
            u.cooldown_ms = max(u.cooldown_ms, u.first_hit_ms)      # pop-up wind-up
        elif not near:
            u.mstate["hidden"] = True
            m.apply_status(u, S.INVISIBLE, 2 * dt)
        return False

    def on_damaged(self, m, u, amount: float, source) -> float:
        return 0.0 if u.mstate.get("hidden") else amount


# ---------------------------------------------------------------- Goblin Demolisher
@register("form_change")
class FormChange(Mechanic):
    """At `at_hp_fraction` of its hitpoints the unit turns into its kamikaze form: faster, melee, only hunts
    buildings, loses `hp_loss` hitpoints per second and explodes (`death_damage`, `death_radius_tiles`) when it
    reaches a building or runs out of hitpoints."""

    def on_tick(self, m, u, dt: int) -> bool:
        p = u.params.get("form_change", {})
        if not u.mstate.get("changed") and u.hp <= u.max_hp * float(p.get("at_hp_fraction", 0.5)):
            u.mstate["changed"] = True
            u.speed = float(p["speed_tiles_s"])
            u.range = float(p["range_tiles"])
            u.targets = tuple(p["targets"])
            u.splash, u.damage, u.target_uid = 0.0, 0, None
            u.decay_per_tick = float(p.get("hp_loss", 0)) * dt / 1000.0
            u.params["death_damage"] = {"damage": float(p.get("death_damage", 0)),
                                        "radius_tiles": float(p["death_radius_tiles"])}
            u.params["suicide"] = {}
            u.mechanics = tuple(dict.fromkeys(u.mechanics + ("decay", "death_damage", "suicide")))
        return False


# ---------------------------------------------------------------- spells
def _pulse(m, ef: AreaEffect, e) -> float:
    return ef.tower_damage if e.kind == "tower" else ef.damage


@register_spell("dot_spell")
class DotSpell(SpellKind):
    """Earthquake: for `duration_ms` ground troops in the area are slowed, and every `hit_every_ms` troops take
    `damage`, towers `crown_tower_damage` and other buildings `building_damage`."""

    def cast(self, m, side, card_id, x, y, eid, stats, params) -> None:
        sp, st = stats.spell, stats.stats
        dmg = st.get("damage", 0)
        m.effects.append(AreaEffect(m._uid(), side, card_id, x, y, sp["radius_tiles"], dmg,
                                    st.get("crown_tower_damage", dmg), int(sp["duration_ms"]), int(sp["hit_every_ms"]),
                                    0.0, eid, kind="dot_spell", next_hit_ms=int(sp["hit_every_ms"]),
                                    params={**params, "state": {"building": st.get("building_damage", dmg),
                                                                "slow_pct": sp.get("slow_pct", 0),
                                                                "air": bool(sp.get("hits_air", False))}}))

    def advance(self, m, ef: AreaEffect, dt: int) -> None:
        ef.elapsed_ms += dt
        stt = ef.params["state"]
        inside = [e for e in _enemies(m, ef.side)
                  if (e.is_structure or e.layer == "ground" or stt["air"])
                  and G.dist(e.x, e.y, ef.x, ef.y) <= ef.radius + e.radius]
        tag = {50: S.SLOW50, 15: S.SLOW15}.get(int(stt["slow_pct"]))
        if tag:
            for e in inside:
                if not e.is_structure:
                    m.apply_status(e, tag, 2 * dt)
        while ef.next_hit_ms <= ef.elapsed_ms and ef.next_hit_ms <= ef.duration_ms:
            ef.next_hit_ms += ef.hit_every_ms
            for e in inside:
                if e.hp > 0:
                    m._deal(e, ef.tower_damage if e.kind == "tower" else
                            stt["building"] if e.kind == "building" else ef.damage, ef)


@register_spell("top_n_spell")
class TopNSpell(AreaSpell):
    """Lightning: strikes the `n` enemies with the most hitpoints inside the radius (towers and buildings included),
    each for the full damage (towers `crown_tower_damage`), and stuns the survivors."""

    def impact(self, m, sp: Spell) -> None:
        p = sp.params.get("top_n_spell", {})
        pool = [e for e in _enemies(m, sp.side) if G.dist(e.x, e.y, sp.x, sp.y) <= sp.radius + e.radius]
        pool.sort(key=lambda e: (-e.hp, e.uid))
        for e in pool[: int(p.get("n", 3))]:
            m._deal(e, sp.tower_damage if e.kind == "tower" else sp.damage, sp)
            if e.hp > 0 and p.get("stun_ms") and e.kind != "tower":
                m.apply_status(e, S.STUN, int(p["stun_ms"]))


@register_spell("root_spell")
class RootSpell(SpellKind):
    """Vines: after `delay_ms` it grabs the `targets` enemies with the most hitpoints in the radius, roots them for
    `root_ms` and damages them every `hit_every_ms` (2 pulses over the root time)."""

    def cast(self, m, side, card_id, x, y, eid, stats, params) -> None:
        sp, st = stats.spell, stats.stats
        dmg = st.get("damage", 0)
        total = int(sp["delay_ms"]) + int(sp["root_ms"])
        m.effects.append(AreaEffect(m._uid(), side, card_id, x, y, sp["radius_tiles"], dmg,
                                    st.get("crown_tower_damage", dmg), total, int(sp["hit_every_ms"]), 0.0, eid,
                                    kind="root_spell", next_hit_ms=int(sp["delay_ms"]) + int(sp["hit_every_ms"]),
                                    params={**params, "state": {"delay": int(sp["delay_ms"]), "root": int(sp["root_ms"]),
                                                                "n": int(sp.get("targets", 3)), "sel": None}}))

    def advance(self, m, ef: AreaEffect, dt: int) -> None:
        ef.elapsed_ms += dt
        stt = ef.params["state"]
        if stt["sel"] is None and ef.elapsed_ms >= stt["delay"]:
            pool = [e for e in _enemies(m, ef.side) if G.dist(e.x, e.y, ef.x, ef.y) <= ef.radius + e.radius]
            pool.sort(key=lambda e: (-e.hp, e.uid))
            stt["sel"] = [e.uid for e in pool[: stt["n"]]]
            for uid in stt["sel"]:
                e = m.units[uid]
                if not e.is_structure:
                    m.apply_status(e, S.ROOT, stt["root"])
        while stt["sel"] is not None and ef.next_hit_ms <= ef.elapsed_ms and ef.next_hit_ms <= ef.duration_ms:
            ef.next_hit_ms += ef.hit_every_ms
            for uid in stt["sel"]:
                e = m.units.get(uid)
                if e is not None and e.hp > 0:
                    m._deal(e, _pulse(m, ef, e), ef)


@register_spell("delayed_drop")
class DelayedDrop(AreaSpell):
    """Royal Delivery: lands `delay_ms` after it is played, damages the area, then the recruit steps out."""

    def cast(self, m, side, card_id, x, y, eid, stats, params) -> None:
        sp = stats.spell
        dmg = stats.stats.get("damage", 0)
        tdmg = stats.stats.get("crown_tower_damage", dmg)
        m.spells.append(Spell(m._uid(), side, card_id, x, y, sp["radius_tiles"], dmg, tdmg,
                              int(math.ceil(sp["delay_ms"] / G.TICK_MS) * G.TICK_MS), eid, "delayed_drop", dict(params)))

    def impact(self, m, sp: Spell) -> None:
        super().impact(m, sp)
        p = sp.params.get("delayed_drop", {})
        if p.get("token"):
            m._spawn_card_units(sp.side, p["token"], sp.x, sp.y, sp.source_event_id, False,
                                deploy_ms=int(p.get("deploy_ms", 250)))
