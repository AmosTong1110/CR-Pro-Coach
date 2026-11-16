"""Timed status effects on units: one place for the engine to ask "can this unit move / attack / be targeted?".

Effects are tags in `Unit.status` (tag -> remaining ms). Appliers: Zap (area spell `apply_status`), Ice Spirit
(`suicide` + freeze), Electro Spirit (`suicide` chain + stun), Ice Golem (`death_damage` + slow). Royal Ghost's
invisibility is still to come.
"""
from __future__ import annotations

STUN, FREEZE, SLOW, ROOT, INVISIBLE = "stun", "freeze", "slow", "root", "invisible"
SLOW50 = "slow50"      # Earthquake: speed and hit speed -50%
SLOW30 = "slow30"      # Hero Ice Golem blizzard: speed and hit speed -30%
SLOW15 = "slow15"      # Poison: speed and hit speed -15% (slows never stack: the strongest one counts)
SLOW_FACTOR = 0.65      # game buff IceWizardSlowDown: speed and hit speed -35% (2023-10 data); duration is per card
SLOW_PCT = 65           # same factor in integer percent, for attack-timer arithmetic


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


def _slow_pct(u) -> int:
    """Percent of normal speed left under the strongest slow on the unit (100 = not slowed)."""
    pct = 100
    if SLOW50 in u.status:
        pct = 50
    if SLOW in u.status:
        pct = min(pct, SLOW_PCT)
    if SLOW30 in u.status:
        pct = min(pct, 70)
    if SLOW15 in u.status:
        pct = min(pct, 85)
    return pct


def speed_factor(u) -> float:
    return _slow_pct(u) / 100.0


def cooldown_dt(u, dt: int) -> int:
    """How much the attack timer advances this tick: slowed units attack slower (integer ms)."""
    return dt * _slow_pct(u) // 100
