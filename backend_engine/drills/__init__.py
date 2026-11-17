"""
Tactical Drills & Micro-Challenge Sandbox.
Provides curated scenarios, moment extraction from match reviews, and live defense grading.
"""
from backend_engine.drills.drill_spec import DrillResult, DrillSpec, PreplacedUnit
from backend_engine.drills.curated_drills import find_drill, get_curated_drills
from backend_engine.drills.moment_extractor import drill_from_move_review
from backend_engine.drills.runner import DrillRunner

__all__ = [
    "DrillSpec",
    "DrillResult",
    "PreplacedUnit",
    "DrillRunner",
    "get_curated_drills",
    "find_drill",
    "drill_from_move_review",
]
