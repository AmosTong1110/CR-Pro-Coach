"""Batch-4 mechanics (Phase 2: the 25 cards that complete the 8 meta decks of configs/meta_decks_top200.json).

Unit mechanics: delayed death bombs (Balloon, Giant Skeleton), break on hit (Battle Ram), automatic dash (Bandit),
hook (Fisherman), rebirth + hatch (Phoenix), split attack + spawn zap (Electro Wizard), 3-hit combo (Monk), heal
burst (Heal Spirit), rage on death (Lumberjack), piercing shot (Magic Archer), emerge damage (Goblin Drill).
Spell kinds: `death_bomb` (the bombs above), `barrel_drop` (Goblin Barrel), `volley_spell` (Arrows), and the
lingering areas `heal_zone` / `rage_zone`.

Numbers come from the card data: the Clash Royale Wiki tables (fetched 2026-11-23 through the MediaWiki API),
corrected with the official 2026 balance notes, plus RoyaleAPI's decoded game data for fields the wiki does not
list (mass, collision radius, sight, knockback distances). Every loop is uid-sorted, so the engine stays
deterministic; none of these kinds is used by a card that existed before this batch, so older golden hashes hold.
"""
from __future__ import annotations

import math

from backend_engine.simulator import geometry as G
from backend_engine.simulator.entities import AreaEffect, Spell
from backend_engine.simulator.mechanics import status as S
from backend_engine.simulator.mechanics.base import Mechanic, SpellKind, register, register_spell
from backend_engine.simulator.mechanics.core import AreaSpell, knockback, land


def _stat(m, u, name: str, default: float = 0.0) -> float:
    """A scaled stat of the unit's card at its side's level (tokens included)."""
    return float(m.db.stats(u.card_id, m.sides[u.side].level, u.variant).stats.get(name, default))


def _foes(m, side):
    return [e for e in sorted(m.units.values(), key=lambda e: e.uid) if e.side != side and e.hp > 0]


def _allies(m, side):
    return [e for e in sorted(m.units.values(), key=lambda e: e.uid) if e.side == side and e.hp > 0]


def _push_from(m, e, cx: float, cy: float, side: int, tiles: float, ignores_heavy: bool = False) -> None:
    """Knock `e` away from (cx, cy); dead centre pushes it the way the attacker's side faces."""
    if tiles <= 0 or e.hp <= 0:
        return
    dx, dy = e.x - cx, e.y - cy
    if abs(dx) + abs(dy) < 1e-9:
        dx, dy = 0.0, float(G.forward(side))
    knockback(m, e, dx, dy, tiles, ignores_heavy=ignores_heavy)


def _area_hit(m, side: int, x: float, y: float, radius: float, damage: float, tower_damage: float, source,
              knock: float = 0.0, ground_only: bool = False) -> None:
    for e in _foes(m, side):
        if ground_only and not e.is_structure and e.layer != "ground":
            continue
        if G.dist(e.x, e.y, x, y) <= radius + e.radius:
            m._deal(e, tower_damage if e.kind == "tower" else damage, source)
            if knock and e.hp > 0 and not e.is_structure:
                _push_from(m, e, x, y, side, knock)


# ---------------------------------------------------------------- Balloon, Giant Skeleton
@register("death_bomb")
class DeathBomb(Mechanic):
    """Drops a bomb where it dies; it explodes `delay_ms` later for `death_damage` in `radius_tiles` (towers take
    `tower_pct` percent of it) and knocks troops back `knockback_tiles`."""

    def on_spawn(self, m, u) -> None:
        u.params["death_bomb"]["damage"] = _stat(m, u, u.params["death_bomb"].get("damage_stat", "death_damage"))

    def on_death(self, m, u) -> None:
        p = u.params.get("death_bomb", {})
        dmg = float(p.get("damage", 0))
        extra = {"knockback": {"distance_tiles": float(p["knockback_tiles"])}} if p.get("knockback_tiles") else {}
        delay = int(math.ceil(int(p.get("delay_ms", 3000)) / G.TICK_MS) * G.TICK_MS)
        m.spells.append(Spell(m._uid(), u.side, u.card_id, u.x, u.y, float(p.get("radius_tiles", 3.0)), int(dmg),
                              int(dmg * float(p.get("tower_pct", 100)) / 100.0), delay, u.source_event_id or 0,
                              "death_bomb", extra))


@register_spell("death_bomb")
class DeathBombSpell(AreaSpell):
    """The bomb itself: an area hit (with the knockback in its params) when the fuse runs out. Never cast as a card."""


# ---------------------------------------------------------------- Battle Ram
@register("break_on_hit")
class BreakOnHit(Mechanic):
    """Kamikaze after its hit lands (whoever resolved it, e.g. `charge`): the unit breaks; its `death_spawn` follows."""

    def after_attack(self, m, u, target) -> None:
        u.hp = 0


# ---------------------------------------------------------------- Bandit
@register("dash_attack")
class DashAttack(Mechanic):
    """When its target is `min_tiles`..`max_tiles` away (edge to edge) it stops, winds up for `windup_ms`, then dashes
    at `dash_speed_tiles_s` (over the river too) and hits for `dash_damage`. Immune to damage while dashing. A stun,
    freeze or root during the wind-up cancels it."""

    def on_tick(self, m, u, dt: int) -> bool:
        p = u.params.get("dash_attack", {})
        st = u.mstate.setdefault("dash", {"phase": None, "ms": 0, "target": None})
        if st["phase"] is not None and u.status and not S.can_move(u):
            st.update(phase=None, ms=0, target=None)
            return False
        if st["phase"] is None:
            t = m.units.get(u.target_uid) if u.target_uid is not None else None
            if t is None or t.hp <= 0 or (u.status and not S.can_move(u)):
                return False
            d = m._edge_dist(u, t)
            if float(p.get("min_tiles", 3.5)) <= d <= float(p.get("max_tiles", 6.0)):
                st.update(phase="windup", ms=0, target=t.uid)
                return True
            return False
        t = m.units.get(st["target"])
        if t is None or t.hp <= 0 or (t.status and not S.targetable(t)):
            st.update(phase=None, ms=0, target=None)
            return False
        if st["phase"] == "windup":
            st["ms"] += dt
            if st["ms"] >= int(p.get("windup_ms", 800)):
                st["phase"] = "dash"
            return True
        step = float(p.get("dash_speed_tiles_s", 8.33)) * dt / 1000.0
        d = G.dist(u.x, u.y, t.x, t.y)
        if m._edge_dist(u, t) > 0.3 and d > 1e-9:
            f = min(1.0, step / d)
            u.x = min(max(u.x + (t.x - u.x) * f, 0.0), G.W - 0.01)
            u.y = min(max(u.y + (t.y - u.y) * f, 0.0), G.H - 0.01)
        if m._edge_dist(u, t) <= 0.3:
            m._deal(t, float(p.get("damage") or _stat(m, u, p.get("damage_stat", "dash_damage"))), u)
            st.update(phase=None, ms=0, target=None)
            u.target_uid = t.uid
            u.cooldown_ms = u.hit_speed_ms
        return True

    def on_damaged(self, m, u, amount: float, source) -> float:
        st = u.mstate.get("dash")
        return 0.0 if st and st["phase"] == "dash" else amount


# ---------------------------------------------------------------- Fisherman
@register("hook")
class Hook(Mechanic):
    """When a ground target is `min_tiles`..`max_tiles` away he stops and throws his hook (`windup_ms`): a troop is
    dragged to him at `pull_speed_tiles_s`, or, if the target is a building / tower, he reels himself to it at
    `self_speed_tiles_s`. Heavy troops are pulled too. (The slow was removed on 2026-04-06.) A stun cancels it."""

    def on_tick(self, m, u, dt: int) -> bool:
        p = u.params.get("hook", {})
        st = u.mstate.setdefault("hook", {"phase": None, "ms": 0, "target": None})
        if st["phase"] is not None and u.status and not S.can_attack(u):
            st.update(phase=None, ms=0, target=None)
            return False
        if st["phase"] is None:
            t = m.units.get(u.target_uid) if u.target_uid is not None else None
            if t is None or t.hp <= 0 or (u.status and not S.can_attack(u)):
                return False
            if t.layer != "ground" and not t.is_structure:
                return False
            d = m._edge_dist(u, t)
            if float(p.get("min_tiles", 3.5)) <= d <= float(p.get("max_tiles", 7.0)):
                st.update(phase="windup", ms=0, target=t.uid)
                return True
            return False
        t = m.units.get(st["target"])
        if t is None or t.hp <= 0:
            st.update(phase=None, ms=0, target=None)
            return False
        if st["phase"] == "windup":
            st["ms"] += dt
            if st["ms"] >= int(p.get("windup_ms", 1300)):
                st["phase"] = "self" if t.is_structure else "pull"
            return True
        mover, anchor = (u, t) if st["phase"] == "self" else (t, u)
        speed = float(p.get("self_speed_tiles_s" if st["phase"] == "self" else "pull_speed_tiles_s", 7.5))
        d = G.dist(mover.x, mover.y, anchor.x, anchor.y)
        gap = m._edge_dist(u, t) - u.range
        if gap > 0 and d > 1e-9:
            f = min(1.0, min(speed * dt / 1000.0, gap) / d)
            nx, ny = mover.x + (anchor.x - mover.x) * f, mover.y + (anchor.y - mover.y) * f
            mover.x, mover.y = min(max(nx, 0.0), G.W - 0.01), min(max(ny, 0.0), G.H - 0.01)
        if m._edge_dist(u, t) - u.range <= 1e-6:
            if st["phase"] == "pull" and t.layer == "ground":
                t.x, t.y = land(t.x, t.y)
            st.update(phase=None, ms=0, target=None)
            u.target_uid = t.uid
            u.cooldown_ms = max(u.cooldown_ms, u.first_hit_ms)
        return True


# ---------------------------------------------------------------- Phoenix
@register("rebirth")
class Rebirth(Mechanic):
    """On death: explodes (`death_damage` in `radius_tiles`, towers `tower_pct` %, knockback `knockback_tiles`) and
    leaves an egg (`egg` token) on the ground below it."""

    def on_spawn(self, m, u) -> None:
        u.params["rebirth"]["damage"] = _stat(m, u, "death_damage")

    def on_death(self, m, u) -> None:
        p = u.params.get("rebirth", {})
        dmg = float(p.get("damage", 0))
        _area_hit(m, u.side, u.x, u.y, float(p.get("radius_tiles", 1.5)), dmg,
                  dmg * float(p.get("tower_pct", 100)) / 100.0, u, knock=float(p.get("knockback_tiles", 0)))
        x, y = land(u.x, u.y)
        m._spawn_card_units(u.side, p.get("egg", "phoenix_egg"), x, y, u.source_event_id or 0, False, deploy_ms=0)


@register("hatch")
class Hatch(Mechanic):
    """Phoenix egg: if it survives `delay_ms`, it hatches into `token` (which has no egg of its own) and is gone."""

    def on_tick(self, m, u, dt: int) -> bool:
        p = u.params.get("hatch", {})
        u.mstate["hatch_ms"] = u.mstate.get("hatch_ms", 0) + dt
        if u.mstate["hatch_ms"] >= int(p.get("delay_ms", 4300)):
            m._spawn_card_units(u.side, p["token"], u.x, u.y, u.source_event_id or 0, False,
                                deploy_ms=int(p.get("deploy_ms", 733)))
            u.mechanics = tuple(k for k in u.mechanics if k != "hatch")
            u.hp = 0
            return True
        return False


# ---------------------------------------------------------------- Electro Wizard
@register("spawn_zap")
class SpawnZap(Mechanic):
    """Lands with a Zap: `zap_damage` to every enemy within `radius_tiles` (towers `tower_pct` %) and a stun."""

    def on_spawn(self, m, u) -> None:
        p = u.params.get("spawn_zap", {})
        dmg = _stat(m, u, p.get("damage_stat", "zap_damage"))
        for e in _foes(m, u.side):
            if G.dist(e.x, e.y, u.x, u.y) <= float(p.get("radius_tiles", 2.5)) + e.radius:
                m._deal(e, dmg * float(p.get("tower_pct", 100)) / 100.0 if e.kind == "tower" else dmg, u)
                if e.hp > 0 and p.get("stun_ms"):
                    m.apply_status(e, S.STUN, int(p["stun_ms"]))


@register("split_attack")
class SplitAttack(Mechanic):
    """Each attack is `targets` zaps of `damage`: one on the target, the rest on the next closest enemies in range;
    with nobody else in range every zap hits the target. Everything hit is stunned for `stun_ms` (towers too)."""

    def on_attack(self, m, u, target) -> bool:
        p = u.params.get("split_attack", {})
        n = int(p.get("targets", 2))
        others = [e for e in _foes(m, u.side) if e.uid != target.uid and m._can_target(u, e)
                  and (not e.status or S.targetable(e)) and m._in_range(u, e)]
        others.sort(key=lambda e: (m._edge_dist(u, e), e.uid))
        victims = [target] + others[: n - 1]
        hits = {e.uid: 1 for e in victims}
        hits[target.uid] += n - len(victims)                   # leftover zaps go to the main target
        for e in victims:
            m._deal(e, float(u.damage) * hits[e.uid], u)
            if e.hp > 0 and p.get("stun_ms"):
                m.apply_status(e, S.STUN, int(p["stun_ms"]))
        return True


# ---------------------------------------------------------------- Monk
@register("combo_hit")
class ComboHit(Mechanic):
    """Every `every`-th hit on any target deals `combo_damage` and knocks the target back `knockback_tiles`, heavy
    units included."""

    def on_attack(self, m, u, target) -> bool:
        p = u.params.get("combo_hit", {})
        u.mstate["combo"] = u.mstate.get("combo", 0) + 1
        if u.mstate["combo"] % int(p.get("every", 3)):
            return False
        m._deal(target, _stat(m, u, p.get("damage_stat", "combo_damage")), u)
        if target.hp > 0 and not target.is_structure:
            _push_from(m, target, u.x, u.y, u.side, float(p.get("knockback_tiles", 1.8)), ignores_heavy=True)
        return True


# ---------------------------------------------------------------- Heal Spirit
@register("heal_burst")
class HealBurst(Mechanic):
    """When it lands (list it before `suicide`) it leaves a heal area: `pulses` pulses, `pulse_ms` apart, each heals
    allied troops within `radius_tiles` by `heal` (the card's `heal_per_pulse`)."""

    def on_attack(self, m, u, target) -> bool:
        p = u.params.get("heal_burst", {})
        heal = _stat(m, u, p.get("heal_stat", "heal_per_pulse"))
        every, n = int(p.get("pulse_ms", 250)), int(p.get("pulses", 4))
        m.effects.append(AreaEffect(m._uid(), u.side, u.card_id, target.x, target.y, float(p.get("radius_tiles", 2.5)),
                                    heal, 0.0, every * (n - 1) + G.TICK_MS, every, 0.0, u.source_event_id or 0,
                                    kind="heal_zone", next_hit_ms=0))
        return False


@register_spell("heal_zone")
class HealZone(SpellKind):
    """Lingering heal area (created by `heal_burst`): heals allied troops in range every `hit_every_ms`."""

    def cast(self, m, side, card_id, x, y, eid, stats, params) -> None:   # pragma: no cover - never a card
        raise NotImplementedError

    def advance(self, m, ef: AreaEffect, dt: int) -> None:
        while ef.next_hit_ms <= ef.elapsed_ms and ef.next_hit_ms < ef.duration_ms:
            ef.next_hit_ms += ef.hit_every_ms
            for a in _allies(m, ef.side):
                if a.kind == "troop" and G.dist(a.x, a.y, ef.x, ef.y) <= ef.radius + a.radius:
                    a.hp = min(a.max_hp, a.hp + ef.damage)
        ef.elapsed_ms += dt


# ---------------------------------------------------------------- Lumberjack
@register("rage_on_death")
class RageOnDeath(Mechanic):
    """Drops a Rage where he dies: `rage_damage` to enemies in `radius_tiles` (towers `rage_tower_damage`), then for
    `duration_ms` allied troops and buildings inside move and attack faster (status RAGE)."""

    def on_death(self, m, u) -> None:
        p = u.params.get("rage_on_death", {})
        r = float(p.get("radius_tiles", 3.0))
        _area_hit(m, u.side, u.x, u.y, r, _stat(m, u, "rage_damage"), _stat(m, u, "rage_tower_damage"), u)
        m.effects.append(AreaEffect(m._uid(), u.side, u.card_id, u.x, u.y, r, 0.0, 0.0,
                                    int(p.get("duration_ms", 5500)), G.TICK_MS, 0.0, u.source_event_id or 0,
                                    kind="rage_zone"))


@register_spell("rage_zone")
class RageZone(SpellKind):
    """Lingering Rage area: every tick, allied troops and buildings inside get RAGE for the next two ticks."""

    def cast(self, m, side, card_id, x, y, eid, stats, params) -> None:   # pragma: no cover - never a card
        raise NotImplementedError

    def advance(self, m, ef: AreaEffect, dt: int) -> None:
        ef.elapsed_ms += dt
        for a in _allies(m, ef.side):
            if a.kind != "tower" and G.dist(a.x, a.y, ef.x, ef.y) <= ef.radius + a.radius:
                m.apply_status(a, S.RAGE, 2 * dt)


# ---------------------------------------------------------------- Magic Archer
@register("pierce_shot")
class PierceShot(Mechanic):
    """The arrow flies `length_tiles` from the archer through the target and hits every enemy it can target whose
    body touches the `width_tiles` wide line (instant: its 1000-speed travel time, ~0.6 s, is not simulated)."""

    def on_attack(self, m, u, target) -> bool:
        p = u.params.get("pierce_shot", {})
        dx, dy = target.x - u.x, target.y - u.y
        n = math.sqrt(dx * dx + dy * dy)
        if n < 1e-9:
            return False
        ux, uy, length, half = dx / n, dy / n, float(p.get("length_tiles", 11.0)), float(p.get("width_tiles", 0.5)) / 2
        for e in _foes(m, u.side):
            if not m._can_target(u, e):
                continue
            t = (e.x - u.x) * ux + (e.y - u.y) * uy
            if -e.radius <= t <= length + e.radius and abs((e.x - u.x) * uy - (e.y - u.y) * ux) <= half + e.radius:
                m._deal(e, float(u.damage), u)
        return True


# ---------------------------------------------------------------- Goblin Drill
@register("emerge_damage")
class EmergeDamage(Mechanic):
    """When it surfaces (first tick after its deploy) it hits ground enemies within `radius_tiles` for `spawn_damage`
    (towers `spawn_tower_damage`) and knocks troops back `knockback_tiles`."""

    def on_tick(self, m, u, dt: int) -> bool:
        if not u.mstate.get("emerged"):
            u.mstate["emerged"] = True
            p = u.params.get("emerge_damage", {})
            _area_hit(m, u.side, u.x, u.y, float(p.get("radius_tiles", 2.0)), _stat(m, u, "spawn_damage"),
                      _stat(m, u, "spawn_tower_damage"), u, knock=float(p.get("knockback_tiles", 1.0)),
                      ground_only=True)
        return False


# ---------------------------------------------------------------- spells
@register_spell("barrel_drop")
class BarrelDrop(AreaSpell):
    """Goblin Barrel: thrown from the king tower (travel time); on landing `count` `token`s step out (no damage)."""

    def impact(self, m, sp: Spell) -> None:
        p = sp.params.get("barrel_drop", {})
        x, y = land(sp.x, sp.y)
        m._spawn_card_units(sp.side, p["token"], x, y, sp.source_event_id, False, count=int(p.get("count", 3)),
                            deploy_ms=int(p.get("deploy_ms", 1100)))


@register_spell("volley_spell")
class VolleySpell(AreaSpell):
    """Arrows: lands after its travel time, then `waves` volleys `wave_ms` apart, each an area hit of the card's
    damage (towers `crown_tower_damage`)."""

    def impact(self, m, sp: Spell) -> None:
        super().impact(m, sp)
        p = sp.params.get("volley_spell", {})
        left = int(p.get("left", int(p.get("waves", 3)) - 1))
        if left > 0:
            more = {**sp.params, "volley_spell": {**p, "left": left - 1}}
            m.spells.append(Spell(m._uid(), sp.side, sp.card_id, sp.x, sp.y, sp.radius, sp.damage, sp.tower_damage,
                                  int(p.get("wave_ms", 200)), sp.source_event_id, sp.kind, more))
