"""
Extracts playable training drills from reviewed match moments and blunders.
Allows players to instantly practice correcting tactical mistakes identified by the coach.
"""
from __future__ import annotations

from typing import Optional

from backend_engine.brain.coach_analyzer import MoveReview
from backend_engine.drills.drill_spec import DrillSpec, PreplacedUnit
from backend_engine.simulator.engine import Match


def mmss(t_ms: int) -> str:
    s = int(t_ms / 1000 + 0.5)
    return f"{s // 60}:{s % 60:02d}"


def drill_from_move_review(move: MoveReview, match_before: Match,
                           drill_id: Optional[str] = None) -> DrillSpec:
    """Generate a playable DrillSpec from a match checkpoint and its move review."""
    side_0 = match_before.sides[0]
    hand = tuple(side_0.hand)
    # Ensure exactly 8 unique cards for the match deck
    remaining = [c for c in side_0.queue if c not in hand]
    if len(hand) + len(remaining) < 8:
        for c in ("knight", "archers", "fireball", "the_log", "skeletons", "ice_spirit", "musketeer", "cannon"):
            if c not in hand and c not in remaining:
                remaining.append(c)
            if len(hand) + len(remaining) >= 8:
                break
    queue = tuple(remaining[:4])
    deck = hand + queue
    elixir = float(side_0.elixir)

    # Collect on-field units
    threats = []
    friendlies = []
    for u in sorted(match_before.units.values(), key=lambda u: u.uid):
        if u.kind == "tower" or u.hp <= 0:
            continue
        ratio = max(0.01, min(1.0, u.hp / max(u.max_hp, 1.0)))
        pu = PreplacedUnit(u.card_id, u.side, u.x, u.y, hp_ratio=ratio, variant=u.variant)
        if u.side == 1:
            threats.append(pu)
        elif u.side == 0 and u.kind == "building":
            friendlies.append(pu)

    diag_label = (move.diagnostic or "Tactical Mistake").replace("_", " ").title()
    title = f"Moment {mmss(move.t_ms)}: {diag_label}"
    rec_text = "Wait and hold elixir" if move.best.is_wait else f"Play {move.best.card_id} at ({move.best.x:.1f}, {move.best.y:.1f})"
    description = (f"Review showed a {move.label} at {mmss(move.t_ms)}. "
                   f"Recommended: {rec_text}. "
                   f"Practice defending this wave cleanly!")

    advice = f"Best alternative: {rec_text}."

    d_id = drill_id or f"drill_moment_{move.event_id}_{move.t_ms}"

    return DrillSpec(
        id=d_id,
        title=title,
        category="custom_moment",
        difficulty=2 if move.label in ("mistake", "inaccuracy") else 3,
        description=description,
        player_deck=deck,
        player_hand=hand,
        player_queue=queue,
        player_elixir=elixir,
        enemy_threats=tuple(threats),
        friendly_units=tuple(friendlies),
        target_duration_ms=12_000,
        max_tower_damage_allowed=0.0,
        custom_advice=advice,
        level=match_before.sides[0].level,
    )
