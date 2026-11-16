"""Archetype names, CRbot's own rules (plan §7):

    [Evo tag][win condition] [key building]          e.g. "EvoRoyalHogs GobHut", "Golem"
    [Evo tag][win condition] [avg elixir]            for cycle decks (avg <= 3.0), e.g. "Hog 2.6"

The first win condition found in WIN_CONDITIONS (priority order) names the deck. Decks with no win
condition are named after their most expensive card ("PEKKA Control" style fallback).
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, Iterable, Optional, Sequence

COSTS_PATH = Path(__file__).resolve().parents[2] / "configs" / "card_costs_hf.json"
CYCLE_MAX_AVG = 3.0

# Priority order: the first match names the deck.
WIN_CONDITIONS = (
    ("golem", "Golem"), ("lava_hound", "Lava"), ("elixir_golem", "EGolem"), ("electro_giant", "EGiant"),
    ("goblin_giant", "GobGiant"), ("x_bow", "XBow"), ("mortar", "Mortar"), ("three_musketeers", "3M"),
    ("graveyard", "GY"), ("royal_giant", "RG"), ("giant", "Giant"), ("hog_rider", "Hog"),
    ("royal_hogs", "RoyalHogs"), ("ram_rider", "RamRider"), ("balloon", "Loon"), ("battle_ram", "Ram"),
    ("goblin_drill", "Drill"), ("goblin_barrel", "Bait"), ("skeleton_barrel", "SkellyBarrel"),
    ("wall_breakers", "WB"), ("miner", "Miner"), ("mega_knight", "MK"), ("pekka", "PEKKA"),
    ("royal_recruits", "Recruits"), ("sparky", "Sparky"), ("goblin_machine", "GobMachine"),
)
KEY_BUILDINGS = (
    ("goblin_hut", "GobHut"), ("furnace", "Furnace"), ("barbarian_hut", "BarbHut"),
    ("elixir_collector", "Pump"), ("inferno_tower", "Inferno"), ("tesla", "Tesla"),
    ("bomb_tower", "BombTower"), ("goblin_cage", "Cage"), ("tombstone", "Tombstone"), ("cannon", "Cannon"),
)
_WC = dict(WIN_CONDITIONS)


@lru_cache(maxsize=1)
def default_costs() -> Dict[str, int]:
    try:
        return dict(json.loads(COSTS_PATH.read_text(encoding="utf-8"))["costs"])
    except (OSError, ValueError, KeyError):
        return {}


def pretty(card_id: str) -> str:
    return card_id.replace("_", " ").title()


def avg_elixir(deck: Sequence[str], costs: Optional[Dict[str, int]] = None) -> Optional[float]:
    costs = default_costs() if costs is None else costs
    vals = [costs[c] for c in deck if c in costs]
    return sum(vals) / len(vals) if len(vals) == len(deck) and deck else None


def fmt_avg(avg: float) -> str:
    """Community convention: truncate, so 2.625 reads '2.6'."""
    return f"{int(avg * 10 + 1e-9) / 10:.1f}"


def archetype_name(deck: Sequence[str], evo: Iterable[str] = (), costs: Optional[Dict[str, int]] = None) -> str:
    deck = list(deck)
    if not deck:
        return ""
    costs = default_costs() if costs is None else costs
    evo, hero = {e for e in evo if "#" not in e}, {e.split("#")[0] for e in evo if e.endswith("#hero")}
    avg = avg_elixir(deck, costs)
    wc = next((c for c, _ in WIN_CONDITIONS if c in deck), None)
    if wc is None:
        top = max(deck, key=lambda c: (costs.get(c, 0), c))
        return f"{pretty(top).replace(' ', '')} Control"
    name = ("Evo" if wc in evo else "Hero" if wc in hero else "") + _WC[wc]
    if avg is not None and avg <= CYCLE_MAX_AVG:
        return f"{name} {fmt_avg(avg)}"
    building = next((short for c, short in KEY_BUILDINGS if c in deck and c != wc), None)
    return f"{name} {building}" if building else name
