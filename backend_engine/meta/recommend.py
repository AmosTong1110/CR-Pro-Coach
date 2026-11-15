"""Recommended decks: the most played decks in the replay data, marked by whether CRbot can simulate them.

Ranking today is play count in `configs/meta_decks_hf.json`. Once the value net exists, "decks that beat X"
will be added here.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Collection, Dict, List, Optional, Tuple

from backend_engine.meta.archetypes import archetype_name, avg_elixir, default_costs

META_PATH = Path(__file__).resolve().parents[2] / "configs" / "meta_decks_hf.json"


@dataclass(frozen=True)
class DeckRec:
    cards: Tuple[str, ...]
    count: int                      # times seen in the data
    share: float                    # count / all deck observations
    archetype: str
    avg_elixir: Optional[float]
    missing: Tuple[str, ...]        # cards CRbot can't simulate yet

    @property
    def playable(self) -> bool:
        return not self.missing


@lru_cache(maxsize=1)
def _meta(path: str = str(META_PATH)) -> dict:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"decks": [], "deck_observations": 0}


def meta_info() -> Dict[str, object]:
    m = _meta()
    return {k: m.get(k) for k in ("source", "as_of", "battles", "deck_observations", "unique_decks")}


def recommend(simulated: Collection[str], limit: int = 30, playable_only: bool = False,
              costs: Optional[Dict[str, int]] = None) -> List[DeckRec]:
    """Top decks by play count. `simulated` = card ids present in the CardDB."""
    m = _meta()
    total = max(int(m.get("deck_observations") or 0), 1)
    costs = default_costs() if costs is None else costs
    sim = set(simulated)
    out: List[DeckRec] = []
    for d in sorted(m.get("decks", []), key=lambda d: -d["count"]):
        cards = tuple(d["cards"])
        rec = DeckRec(cards, int(d["count"]), d["count"] / total, archetype_name(cards, costs=costs),
                      avg_elixir(cards, costs), tuple(c for c in cards if c not in sim))
        if playable_only and not rec.playable:
            continue
        out.append(rec)
        if len(out) >= limit:
            break
    return out


def coverage(simulated: Collection[str]) -> Tuple[int, int]:
    """(decks fully playable, unique decks) in the data."""
    sim = set(simulated)
    decks = _meta().get("decks", [])
    return sum(all(c in sim for c in d["cards"]) for d in decks), len(decks)
