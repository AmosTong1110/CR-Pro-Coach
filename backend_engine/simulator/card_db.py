"""
Card database loader with exact official level scaling.

    db = CardDB.load()
    db.stats("knight", level=16).hitpoints   # 2822 (exact, not 1766 * 1.1**5 = 2844)

How the game scales stats (verified 100% exact against all 4,945 per-level values in
RoyaleAPI's decoded game data):

    stat(L) = floor(base_stat * M[i] / 100)
    i       = L - 1 - rarity_offset[rarity]      # L = normalized card level (11 = tournament)
    M       = level_scaling.multiplier_pct       # [100, 110, 121, 133, 146, 160, 176, ...]

`base_stat` is the stat at the rarity's first level (what the game files store).
So exactness depends only on having *current* base stats; the table itself is fixed.

Resolution order for a numeric field at level L:
  1. cards[id].level_overrides[str(L)][field]          exact value typed in by hand
  2. base_stats[field] with the official table           exact
  3. stats_l11[field] * M[i_L] / M[i_11] (floored)       approximation, for cards without base_stats
crown_tower_damage is derived from damage and crown_tower_damage_pct when not given explicitly.
"""
from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "cards_stats_v2026.json"
CARD_TYPES = {"troop", "building", "spell", "champion", "hero"}
SCHEMA_VERSION = 3


class CardDataError(ValueError):
    pass


@dataclass(frozen=True)
class ResolvedCardStats:
    card_id: str
    level: int
    type: str
    elixir: int
    stats: Dict[str, int]                 # level-scaled numbers (hitpoints, damage, ...)
    attack: Dict[str, Any] = field(default_factory=dict)
    movement: Dict[str, Any] = field(default_factory=dict)
    spell: Dict[str, Any] = field(default_factory=dict)
    mechanics: Tuple[str, ...] = ()
    exact: bool = False                   # True if every scaled number came from base_stats/overrides
    verified: bool = False                # True only if base data is confirmed current

    @property
    def hitpoints(self) -> Optional[int]:
        return self.stats.get("hitpoints")

    @property
    def damage(self) -> Optional[int]:
        return self.stats.get("damage")

    @property
    def dps(self) -> Optional[float]:
        hs = self.attack.get("hit_speed_s")
        if self.damage is None or not hs:
            return None
        return self.damage / hs


class CardDB:
    def __init__(self, raw: Dict[str, Any]):
        self.raw = raw
        self._validate()
        sc = raw["level_scaling"]
        self.base_level: int = sc["base_level"]
        self.levels: List[int] = list(sc["supported_levels"])
        self.multiplier: List[int] = list(sc["multiplier_pct"])
        self.rarity_offset: Dict[str, Optional[int]] = dict(sc["rarity_offset"])
        self.scaled_fields = set(sc["scaled_fields"])
        self.cards: Dict[str, Dict[str, Any]] = raw["cards"]
        self.towers: Dict[str, Dict[str, Any]] = raw.get("towers", {})
        self.rules: Dict[str, Any] = raw["rules"]

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> "CardDB":
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    # ------------------------------------------------------------------
    def _validate(self) -> None:
        r = self.raw
        for key in ("schema_version", "level_scaling", "rules", "cards"):
            if key not in r:
                raise CardDataError(f"missing top-level key '{key}'")
        if r["schema_version"] != SCHEMA_VERSION:
            raise CardDataError(f"unsupported schema_version {r['schema_version']} (want {SCHEMA_VERSION})")
        offsets = r["level_scaling"].get("rarity_offset", {})
        for cid, c in r["cards"].items():
            for key in ("name", "type", "rarity", "elixir"):
                if key not in c:
                    raise CardDataError(f"card '{cid}' missing '{key}'")
            if "base_stats" not in c and "stats_l11" not in c:
                raise CardDataError(f"card '{cid}' needs base_stats or stats_l11")
            if c["type"] not in CARD_TYPES:
                raise CardDataError(f"card '{cid}' has unknown type '{c['type']}'")
            if c["rarity"] not in offsets:
                raise CardDataError(f"card '{cid}' has unknown rarity '{c['rarity']}'")
            if not 0 <= c["elixir"] <= 10:
                raise CardDataError(f"card '{cid}' elixir out of range")
            stats = c.get("base_stats", c.get("stats_l11", {}))
            if c["type"] in ("troop", "building", "champion", "hero") and "hitpoints" not in stats:
                raise CardDataError(f"card '{cid}' ({c['type']}) needs hitpoints")

    def _index(self, rarity: str, level: int) -> int:
        off = self.rarity_offset.get(rarity)
        if off is None:
            raise CardDataError(f"rarity '{rarity}' has no known level offset yet")
        i = level - 1 - off
        if not 0 <= i < len(self.multiplier):
            raise CardDataError(f"level {level} is not reachable for rarity '{rarity}'")
        return i

    def _scale(self, entry: Dict[str, Any], level: int) -> Tuple[Dict[str, int], bool]:
        if level not in self.levels:
            raise CardDataError(f"level {level} not in supported_levels {self.levels}")
        rarity = entry.get("rarity", "common")
        overrides = entry.get("level_overrides", {}).get(str(level), {})
        m = self.multiplier[self._index(rarity, level)]
        if "base_stats" in entry:
            src, exact = entry["base_stats"], True
            scale = lambda v: int(math.floor(v * m / 100 + 1e-9))  # noqa: E731
        else:
            src, exact = entry["stats_l11"], False
            m11 = self.multiplier[self._index(rarity, self.base_level)]
            scale = lambda v: int(math.floor(v * m / m11 + 1e-6))  # noqa: E731
        out: Dict[str, int] = {}
        for k, v in src.items():
            if k in overrides:
                out[k] = int(overrides[k])
            elif k in self.scaled_fields:
                out[k] = scale(v)
            else:
                out[k] = v
        for k, v in overrides.items():
            out.setdefault(k, int(v))
        pct = entry.get("crown_tower_damage_pct")
        if pct is not None and "crown_tower_damage" not in out and "damage" in out:
            # game files store a negative modifier (e.g. -70 => 30% damage to crown towers)
            out["crown_tower_damage"] = int(math.floor(out["damage"] * (100 + pct) / 100 + 0.5))
        return out, exact or bool(overrides) and set(overrides) >= set(src)

    def stats(self, card_id: str, level: int = 11) -> ResolvedCardStats:
        if card_id not in self.cards:
            raise KeyError(f"unknown card '{card_id}'")
        c = self.cards[card_id]
        stats, exact = self._scale(c, level)
        return ResolvedCardStats(
            card_id=card_id,
            level=level,
            type=c["type"],
            elixir=c["elixir"],
            stats=stats,
            attack=dict(c.get("attack", {})),
            movement=dict(c.get("movement", {})),
            spell=dict(c.get("spell", {})),
            mechanics=tuple(c.get("mechanics", [])),
            exact=exact,
            verified=bool(c.get("verified", False)),
        )

    def tower_stats(self, tower_id: str, level: int = 11) -> Dict[str, int]:
        return self._scale(self.towers[tower_id], level)[0]

    def elixir_costs(self) -> Dict[str, int]:
        return {cid: c["elixir"] for cid, c in self.cards.items()}

    def unverified(self) -> List[str]:
        return [cid for cid, c in self.cards.items() if not c.get("verified", False)]
