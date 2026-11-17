"""
Curated catalog of built-in tactical micro-defense and positioning drills.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from backend_engine.drills.drill_spec import DrillSpec, PreplacedUnit
from backend_engine.simulator.geometry import BRIDGE_X


def get_curated_drills() -> List[DrillSpec]:
    """Returns the standard catalog of built-in tactical drills."""
    return [
        DrillSpec(
            id="drill_pekka_kite",
            title="P.E.K.K.A Kite Master",
            category="pull_and_kite",
            difficulty=2,
            description="An oncoming P.E.K.K.A is crossing the left bridge. Kite her into the centre using Ice Golem and Skeletons so both Princess Towers finish her off with zero tower damage taken!",
            player_deck=("ice_golem", "skeletons", "cannon", "musketeer", "fireball", "the_log", "ice_spirit", "hog_rider"),
            player_hand=("ice_golem", "skeletons", "cannon", "musketeer"),
            player_queue=("fireball", "the_log", "ice_spirit", "hog_rider"),
            player_elixir=6.0,
            enemy_threats=(PreplacedUnit("pekka", 1, BRIDGE_X[0], 16.5, hp_ratio=0.65),),
            target_duration_ms=15_000,
            max_tower_damage_allowed=0.0,
            custom_advice="Drop Ice Golem at (6.0, 13.5) and Skeletons at (7.5, 11.5) to pull P.E.K.K.A into the crossfire.",
        ),
        DrillSpec(
            id="drill_hog_pull",
            title="Hog 2.6 Golden Cannon Pull",
            category="pull_and_kite",
            difficulty=1,
            description="Hog Rider is rushing down the right bridge! Place your Cannon at the central golden tile (9.0, 9.0) supported by Ice Spirit to completely stop him before he touches your tower.",
            player_deck=("cannon", "ice_spirit", "skeletons", "musketeer", "fireball", "the_log", "ice_golem", "hog_rider"),
            player_hand=("cannon", "ice_spirit", "skeletons", "musketeer"),
            player_queue=("fireball", "the_log", "ice_golem", "hog_rider"),
            player_elixir=4.0,
            enemy_threats=(PreplacedUnit("hog_rider", 1, BRIDGE_X[1], 17.5),),
            target_duration_ms=10_000,
            max_tower_damage_allowed=0.0,
            custom_advice="Place Cannon 3 tiles from the river in the center. An Ice Spirit freeze buys extra tower shots.",
        ),
        DrillSpec(
            id="drill_xbow_lock",
            title="Bridge X-Bow Lock Defense",
            category="siege_defense",
            difficulty=3,
            description="Your offensive X-Bow is locked onto the enemy Princess Tower! Enemy Knight and Archers are crossing the bridge to take it down. Protect your X-Bow with Tesla and cheap cycle cards until the tower falls.",
            player_deck=("x_bow", "tesla", "archers", "knight", "the_log", "fireball", "skeletons", "ice_spirit"),
            player_hand=("tesla", "knight", "the_log", "skeletons"),
            player_queue=("archers", "fireball", "ice_spirit", "x_bow"),
            player_elixir=6.0,
            friendly_units=(PreplacedUnit("x_bow", 0, BRIDGE_X[0], 13.5),),
            enemy_threats=(PreplacedUnit("knight", 1, BRIDGE_X[0], 18.0), PreplacedUnit("archers", 1, BRIDGE_X[0], 19.5)),
            target_duration_ms=15_000,
            max_tower_damage_allowed=0.0,
            custom_advice="Place Knight in front of the X-Bow to absorb hits, and Tesla beside it to shred incoming troops.",
        ),
        DrillSpec(
            id="drill_inferno_melt",
            title="Inferno Tower Giant Melt",
            category="tank_melt",
            difficulty=2,
            description="A heavy 4000+ HP Giant is marching down the right bridge. Drop Inferno Tower in the center and distract incoming support to let the beam ramp to Stage 3 maximum melt damage!",
            player_deck=("inferno_tower", "goblins", "knight", "dart_goblin", "rocket", "the_log", "ice_spirit", "skeleton_army"),
            player_hand=("inferno_tower", "goblins", "knight", "ice_spirit"),
            player_queue=("dart_goblin", "rocket", "the_log", "skeleton_army"),
            player_elixir=6.0,
            enemy_threats=(PreplacedUnit("giant", 1, BRIDGE_X[1], 17.5),),
            target_duration_ms=12_000,
            max_tower_damage_allowed=0.0,
            custom_advice="Place Inferno Tower 4 tiles from the river. Pull the Giant into the middle so the beam stays uninterrupted.",
        ),
        DrillSpec(
            id="drill_witch_surround",
            title="Valkyrie Witch Surround",
            category="swarm_control",
            difficulty=1,
            description="An enemy Witch is advancing and summoning Skeletons. Drop Valkyrie directly on top of her to 360° spin-shred the Witch and all summoned skeletons in one sweep.",
            player_deck=("valkyrie", "baby_dragon", "musketeer", "skeletons", "fireball", "the_log", "zap", "hog_rider"),
            player_hand=("valkyrie", "zap", "skeletons", "baby_dragon"),
            player_queue=("musketeer", "fireball", "the_log", "hog_rider"),
            player_elixir=4.0,
            enemy_threats=(PreplacedUnit("witch", 1, BRIDGE_X[0], 16.5),),
            target_duration_ms=10_000,
            max_tower_damage_allowed=0.0,
            custom_advice="Wait until Witch crosses the bridge into your territory, then drop Valkyrie right on her head.",
        ),
    ]


def find_drill(drill_id: str) -> Optional[DrillSpec]:
    """Look up a curated drill by its unique id."""
    for d in get_curated_drills():
        if d.id == drill_id:
            return d
    return None
