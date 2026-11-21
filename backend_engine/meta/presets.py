"""Preset decks for the Deck Builder: ready-made 8-card decks (configs/preset_decks.json).

Every preset uses only cards CRbot simulates, so each one is playable for you and for the AI. Validation lives in
tests/test_presets.py; `playable(db_cards)` filters again at runtime in case the card data changes.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Collection, Iterable, List, Optional, Sequence, Tuple

PRESETS_PATH = Path(__file__).resolve().parents[2] / "configs" / "preset_decks.json"
# The scripted AI attacks with building-targeting win conditions and siege weapons.
AI_PUSH_CARDS = ("hog_rider", "rune_giant", "minion_giant", "x_bow", "mortar", "giant", "pekka")


@dataclass(frozen=True)
class PresetDeck:
    name: str
    cards: Tuple[str, ...]
    evo: Tuple[str, ...]
    style: str
    tip: str

    @property
    def ai_pushes(self) -> bool:
        """True if the scripted AI can attack with this deck (it needs a building-targeting win condition)."""
        return any(c in self.cards for c in AI_PUSH_CARDS)


@lru_cache(maxsize=4)
def load_presets(path: str = str(PRESETS_PATH)) -> Tuple[PresetDeck, ...]:
    try:
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return ()
    return tuple(PresetDeck(d["name"], tuple(d["cards"]), tuple(d.get("evo", ())), d.get("style", ""),
                            d.get("tip", "")) for d in raw.get("decks", []))


def playable(simulated: Collection[str], presets: Optional[Iterable[PresetDeck]] = None) -> List[PresetDeck]:
    """Presets whose 8 cards are all in the card database."""
    sim = set(simulated)
    return [p for p in (load_presets() if presets is None else presets) if all(c in sim for c in p.cards)]


def select_presets(presets: Sequence[PresetDeck], names: Sequence[str]) -> List[PresetDeck]:
    """`presets` restricted to `names` (in that order); every preset when `names` is empty. Used by the PPO
    trainer to make a deck specialist. Unknown or unplayable names raise ValueError listing the valid ones."""
    if not names:
        return list(presets)
    by_name = {p.name: p for p in presets}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f"{missing} are not playable presets; choose from {sorted(by_name)}")
    return [by_name[n] for n in names]


def find_preset(cards: Sequence[str], evo: Sequence[str] = (),
                presets: Optional[Iterable[PresetDeck]] = None) -> Optional[PresetDeck]:
    """The preset with these cards (order does not matter): an exact evolution match first, else the same 8
    cards with another evolution choice."""
    key, ekey = frozenset(cards), frozenset(evo)
    same_cards = [p for p in (load_presets() if presets is None else presets) if frozenset(p.cards) == key]
    return next((p for p in same_cards if frozenset(p.evo) == ekey), same_cards[0] if same_cards else None)
