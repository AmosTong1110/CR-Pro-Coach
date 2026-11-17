"""Unknown-card quarantine for external match data (replays, battle logs).

Supercell ships new cards before the card data knows them. Tools that read real matches must not crash on them
(KeyError deep inside the engine) or silently train on them. `CardQuarantine.check(cards)` resolves every card
name / id against the catalog (ids, internal names, display names, RoyaleAPI / HF aliases) and says whether the
match can be used:

  * a card the catalog does not know at all  -> skip the match ("unknown card"), always
  * a catalog card that isn't playable (`enabled: false`, `needs_hook`, unsupported mechanic)
                                             -> skip only in strict mode (tools can proxy or ignore them otherwise)

It counts what it saw, and `summary()` gives one clean log line at the end instead of one warning per match.
"""
from __future__ import annotations

import collections
import re
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple

from backend_engine.simulator.card_db import CardDB

_FORM = re.compile(r"[-_](ev\d+|evo|hero)$")


def _slug(name: str) -> str:
    s = (name or "").strip().lower().replace(".", "").replace("'", "")
    s = re.sub(r"[\s\-]+", "_", s)
    return {"log": "the_log"}.get(s, s)


def alias_index(db: CardDB) -> Dict[str, str]:
    """Every known spelling (slugged) -> catalog card id (playable or not). Tokens are included."""
    idx: Dict[str, str] = {}
    for cid, c in [*db.catalog.items(), *db.tokens.items()]:
        a = c.get("aliases") or {}
        for v in (cid, c.get("internal_name"), c.get("name"), a.get("display"), a.get("hf"), a.get("royaleapi")):
            if v:
                idx.setdefault(_slug(v), cid)
    return idx


def resolve(db: CardDB, raw: str, index: Optional[Dict[str, str]] = None) -> Optional[str]:
    """Catalog id for a card name / id / alias ('Mini P.E.K.K.A', 'mini-pekka', 'knight-ev1'), else None."""
    index = index if index is not None else alias_index(db)
    key = _slug(_FORM.sub("", (raw or "").strip().lower()))
    return index.get(key)


@dataclass
class CardQuarantine:
    db: CardDB
    strict: bool = False
    unknown: Dict[str, int] = field(default_factory=collections.Counter)        # raw name -> matches
    not_playable: Dict[str, int] = field(default_factory=collections.Counter)   # catalog id -> matches
    checked: int = 0
    skipped: int = 0

    def __post_init__(self):
        self._index = alias_index(self.db)

    def resolve(self, raw: str) -> Optional[str]:
        return resolve(self.db, raw, self._index)

    def check(self, cards: Iterable[str]) -> Optional[str]:
        """None if the match can be used, else the reason it is skipped (counts are updated either way)."""
        self.checked += 1
        unknown: List[str] = []
        blocked: List[str] = []
        for raw in dict.fromkeys(c for c in cards if c):
            cid = self.resolve(raw)
            if cid is None:
                unknown.append(raw)
            elif cid not in self.db.cards and cid not in self.db.tokens:
                blocked.append(cid)
        for c in unknown:
            self.unknown[c] += 1
        for c in blocked:
            self.not_playable[c] += 1
        if unknown:
            self.skipped += 1
            return f"unknown card(s): {', '.join(unknown)}"
        if blocked and self.strict:
            self.skipped += 1
            return f"card(s) not playable yet: {', '.join(blocked)}"
        return None

    def top(self, counter: Dict[str, int], n: int = 8) -> List[Tuple[str, int]]:
        return sorted(counter.items(), key=lambda kv: (-kv[1], kv[0]))[:n]

    def summary(self) -> str:
        """One line for the log ('' when nothing was quarantined or seen)."""
        if not self.unknown and not self.not_playable:
            return ""
        parts = [f"card quarantine: {self.skipped} of {self.checked} matches skipped"]
        if self.unknown:
            parts.append("unknown to cards_stats.json: " + ", ".join(f"{c} ({n})" for c, n in self.top(self.unknown))
                         + " -> run tools/sync_card_catalog.py")
        if self.not_playable:
            parts.append(("skipped" if self.strict else "seen") + " not playable yet: "
                         + ", ".join(f"{c} ({n})" for c, n in self.top(self.not_playable)))
        return "; ".join(parts)

    def as_dict(self) -> dict:
        return {"strict": self.strict, "checked": self.checked, "skipped": self.skipped,
                "unknown": dict(self.unknown), "not_playable": dict(self.not_playable)}
