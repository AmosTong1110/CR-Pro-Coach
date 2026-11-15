"""Timed status effects on units: one place for the engine to ask "can this unit move / attack / be targeted?".

Effects are tags in `Unit.status` (tag -> remaining ms). Nothing in the current 9-card set applies them; batch 1
cards (Zap, Ice Spirit, Freeze, Royal Ghost, ...) will.
"""
from __future__ import annotations

STUN, FREEZE, SLOW, ROOT, INVISIBLE = "stun", "freeze", "slow", "root", "invisible"
SLOW_FACTOR = 0.65      # unverified placeholder; per-card params override it when cards arrive


def apply(u, tag: str, ms: int) -> None:
    """Add or refresh a status (keeps the longer remaining time). Stun/freeze reset the attack wind-up."""
    u.status[tag] = max(int(ms), u.status.get(tag, 0))
    if tag in (STUN, FREEZE):
        u.cooldown_ms = max(u.cooldown_ms, u.first_hit_ms)
        u.target_uid = None if tag == STUN else u.target_uid


def tick(u, dt: int) -> None:
    for tag in sorted(u.status):
        left = u.status[tag] - dt
        if left <= 0:
            del u.status[tag]
        else:
            u.status[tag] = left


def can_move(u) -> bool:
    return not (STUN in u.status or FREEZE in u.status or ROOT in u.status)


def can_attack(u) -> bool:
    return not (STUN in u.status or FREEZE in u.status)


def targetable(u) -> bool:
    return INVISIBLE not in u.status


def speed_factor(u) -> float:
    return SLOW_FACTOR if SLOW in u.status else 1.0
