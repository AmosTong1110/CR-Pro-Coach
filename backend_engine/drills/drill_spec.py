"""
Drill specification and result data structures.
Defines tactical training scenarios, objectives, and grading criteria.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class PreplacedUnit:
    """An entity pre-placed on the field when a drill begins."""
    card_id: str
    side: int                  # 0 = player (defending), 1 = opponent (attacking threat)
    x: float
    y: float
    hp_ratio: float = 1.0      # 1.0 = full HP, 0.5 = half HP, etc.
    variant: str = "base"


@dataclass(frozen=True)
class DrillSpec:
    """Complete specification of a micro-challenge tactical drill."""
    id: str
    title: str
    category: str              # "pull_and_kite", "siege_defense", "tank_melt", "swarm_control", "custom_moment"
    difficulty: int            # 1 (Beginner), 2 (Intermediate), 3 (Expert)
    description: str
    player_deck: Tuple[str, ...]
    player_hand: Tuple[str, ...]
    player_queue: Tuple[str, ...]
    player_elixir: float
    enemy_threats: Tuple[PreplacedUnit, ...]
    friendly_units: Tuple[PreplacedUnit, ...] = ()
    target_duration_ms: int = 15_000
    max_tower_damage_allowed: float = 0.0
    custom_advice: Optional[str] = None
    level: int = 11


@dataclass(frozen=True)
class DrillResult:
    """The outcome and coaching grade of an executed drill."""
    solved: bool
    stars: int                 # 0 (Failed), 1 (Passed), 2 (Good), 3 (Perfect ⭐⭐⭐)
    tower_damage_taken: float
    elixir_spent: float
    duration_ms: int
    feedback: str
