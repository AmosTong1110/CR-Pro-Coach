"""Render-side animation tracker: turns successive match states into short visual effects and smooth motion.

It only *reads* the match (units, projectiles, spells) and keeps its own snapshot, so the engine, determinism and
golden hashes are untouched. Wall-clock time is passed in by the caller (tests use a fake clock).

Effects found by diffing two consecutive states:
  land      a new unit appears (card landing: dust ring)
  death     a unit disappears (fades out, a puff rises)
  collapse  a tower disappears (big burst, rubble stays)
  hit       a non-projectile attacker reset its cooldown while its target lost hp (melee spark)
  impact    a projectile disappears (small flash; `burst` too if it had splash)
  burst     an area spell disappears (expanding ring)
Units that lost hp also get a short white "hurt" flash. Time jumps (seek, rewind, a new match) reset the tracker
without producing effects.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

DURATION = {"land": 0.35, "death": 0.45, "collapse": 0.8, "hit": 0.18, "impact": 0.2, "burst": 0.45}
HURT_S = 0.15
JUMP_MS = 1000          # a bigger step than this between two frames is a seek, not play
MAX_FX = 160


@dataclass
class Fx:
    kind: str
    x: float
    y: float
    start: float
    side: int = 0
    size: float = 0.5           # tiles (unit radius, splash or spell radius)
    card_id: str = ""
    air: bool = False

    @property
    def duration(self) -> float:
        return DURATION[self.kind]

    def age(self, now: float) -> float:
        """0 at the start, 1 when finished."""
        return max(0.0, min(1.0, (now - self.start) / self.duration))


@dataclass
class _U:
    x: float
    y: float
    side: int
    card_id: str
    kind: str
    layer: str
    radius: float
    hp: float
    cooldown_ms: int
    target_uid: Optional[int]
    ranged_proj: bool


def _snap_units(match) -> Dict[int, _U]:
    return {uid: _U(u.x, u.y, u.side, u.card_id, u.kind, u.layer, u.radius, u.hp, u.cooldown_ms, u.target_uid,
                    "projectile" in u.mechanics)
            for uid, u in match.units.items()}


class FxTracker:
    def __init__(self) -> None:
        self.fx: List[Fx] = []
        self.hurt: Dict[int, float] = {}
        self._match = None
        self._t: Optional[int] = None
        self._units: Dict[int, _U] = {}
        self._proj: Dict[int, Tuple[float, float, int, float]] = {}
        self._spells: Dict[int, Tuple[float, float, int, float, str]] = {}
        self._prev_pos: Dict[int, Tuple[float, float]] = {}
        self._prev_proj: Dict[int, Tuple[float, float]] = {}
        self.changed_at = 0.0
        self.step_s = 0.05

    # -- state intake --------------------------------------------------------------------------------------------
    def reset(self, match=None) -> None:
        self.fx.clear()
        self.hurt.clear()
        self._match = match
        self._t = match.t_ms if match is not None else None
        self._units = _snap_units(match) if match is not None else {}
        self._proj = self._snap_proj(match) if match is not None else {}
        self._spells = self._snap_spells(match) if match is not None else {}
        self._prev_pos = {uid: (u.x, u.y) for uid, u in self._units.items()}
        self._prev_proj = {}

    @staticmethod
    def _snap_proj(match):
        return {p.uid: (p.x, p.y, p.side, p.splash) for p in match.projectiles}

    @staticmethod
    def _snap_spells(match):
        return {s.uid: (s.x, s.y, s.side, s.radius, s.card_id) for s in match.spells}

    def observe(self, match, now: float) -> None:
        """Call once per painted frame; does work only when the match clock moved."""
        t = match.t_ms
        if match is not self._match or self._t is None or t < self._t or t - self._t > JUMP_MS:
            self.reset(match)
            self.changed_at = now
            return
        if t == self._t:
            return
        units = _snap_units(match)
        old = self._units
        for uid, u in units.items():
            o = old.get(uid)
            if o is None:
                if u.kind != "tower":
                    self._add(Fx("land", u.x, u.y, now, u.side, u.radius, u.card_id, u.layer == "air"))
                continue
            if u.hp < o.hp - 1e-6:
                self.hurt[uid] = now
            if (u.cooldown_ms > o.cooldown_ms and not u.ranged_proj and u.kind != "tower"
                    and u.target_uid is not None and u.target_uid in units and u.target_uid in old
                    and units[u.target_uid].hp < old[u.target_uid].hp - 1e-6):
                tg = units[u.target_uid]
                d = ((u.x - tg.x) ** 2 + (u.y - tg.y) ** 2) ** 0.5 or 1.0
                k = min(0.5, tg.radius * 0.8 / d)       # on the target's edge, facing the attacker
                self._add(Fx("hit", tg.x + (u.x - tg.x) * k, tg.y + (u.y - tg.y) * k, now, u.side, 0.5))
        for uid, o in old.items():
            if uid not in units:
                kind = "collapse" if o.kind == "tower" else "death"
                self._add(Fx(kind, o.x, o.y, now, o.side, o.radius, o.card_id, o.layer == "air"))
                self.hurt.pop(uid, None)
        proj = self._snap_proj(match)
        for uid, (x, y, side, splash) in self._proj.items():
            if uid not in proj:
                self._add(Fx("impact", x, y, now, side, 0.35))
                if splash > 0:
                    self._add(Fx("burst", x, y, now, side, splash))
        spells = self._snap_spells(match)
        for uid, (x, y, side, radius, card_id) in self._spells.items():
            if uid not in spells:
                self._add(Fx("burst", x, y, now, side, radius, card_id))
        self._prev_pos = {uid: (o.x, o.y) for uid, o in old.items()}
        self._prev_proj = {uid: (v[0], v[1]) for uid, v in self._proj.items()}
        step = now - self.changed_at
        self.step_s = min(0.25, max(0.016, step)) if 0 < step < 1.0 else 0.05
        self.changed_at = now
        self._units, self._proj, self._spells, self._t = units, proj, spells, t
        self.prune(now)

    def _add(self, fx: Fx) -> None:
        self.fx.append(fx)
        if len(self.fx) > MAX_FX:
            del self.fx[: len(self.fx) - MAX_FX]

    def prune(self, now: float) -> None:
        self.fx = [f for f in self.fx if now - f.start < f.duration]
        self.hurt = {uid: t for uid, t in self.hurt.items() if now - t < HURT_S}

    # -- render queries ------------------------------------------------------------------------------------------
    def blend(self, now: float) -> float:
        """0 right after the match stepped, 1 one step later: how far to move from the previous position."""
        return max(0.0, min(1.0, (now - self.changed_at) / self.step_s))

    def pos(self, uid: int, x: float, y: float, now: float, projectile: bool = False) -> Tuple[float, float]:
        """Smoothed position between the previous and the current simulation step."""
        prev = (self._prev_proj if projectile else self._prev_pos).get(uid)
        if prev is None:
            return x, y
        a = self.blend(now)
        px, py = prev
        if abs(px - x) + abs(py - y) > 3.0:          # teleport / jump (e.g. a dash): no smear
            return x, y
        return px + (x - px) * a, py + (y - py) * a

    def hurt_level(self, uid: int, now: float) -> float:
        t = self.hurt.get(uid)
        return 0.0 if t is None else max(0.0, 1.0 - (now - t) / HURT_S)

    def animating(self, now: float) -> bool:
        """True while something on screen still changes without a new simulation step."""
        return bool(self.fx) or bool(self.hurt) or self.blend(now) < 1.0
