"""
Import exact card stats from a RoyaleAPI-format `cards_stats.json` into
configs/cards_stats_v2026.json.

Every card in our config names its record in the source via `source_ref`:
    "source_ref": {"section": "troop", "name": "Knight"}
The importer writes `base_stats` (stats at the rarity's first level, as stored in the
game files) plus timing/range/speed fields; card_db.py turns base_stats into exact
values for any level.

    # dry run: show what would change at level 11 and 16
    python -m backend_engine.simulator.sync_meta --source path/to/cards_stats.json
    # write it
    python -m backend_engine.simulator.sync_meta --source path/to/cards_stats.json --write --as-of 2023-10-18

Source freshness matters: RoyaleAPI/cr-api-data was last updated 2023-10-18, so its numbers
predate later balance changes. The importer stamps `data_as_of` on every card so stale
data is visible; set `verified: true` by hand only after checking the current in-game card.
"""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

from backend_engine.simulator.card_db import DEFAULT_PATH, CardDB

MS = 1000.0
SPEED_UNIT = 60.0  # game speed 60 == 1 tile/s (Medium)


class SourceIndex:
    def __init__(self, raw: Dict[str, Any]):
        self.by = {sec: {r["name"]: r for r in rows if r.get("name")} for sec, rows in raw.items()
                   if isinstance(rows, list)}

    def get(self, section: str, name: Optional[str]) -> Optional[Dict[str, Any]]:
        return self.by.get(section, {}).get(name) if name else None


def _nz(*vals):
    for v in vals:
        if v:
            return v
    return None


def build_card(card: Dict[str, Any], src: SourceIndex, as_of: str) -> Dict[str, Any]:
    ref = card["source_ref"]
    rec = src.get(ref["section"], ref["name"])
    if rec is None:
        raise KeyError(f"{ref} not found in source")
    out = copy.deepcopy(card)
    out["rarity"] = rec["rarity"].lower()
    if rec.get("mana_cost"):
        out["elixir"] = rec["mana_cost"]
    base: Dict[str, int] = {}

    if ref["section"] in ("troop", "building"):
        body = src.get("characters", rec.get("summon_character")) or src.get("building", rec.get("summon_character")) or rec
        proj = src.get("projectile", _nz(body.get("projectile"), rec.get("projectile")))
        base["hitpoints"] = body["hitpoints"]
        dmg = _nz(body.get("damage"), proj and proj.get("damage"))
        if dmg:
            base["damage"] = dmg
        atk = out.setdefault("attack", {})
        atk["hit_speed_s"] = body["hit_speed"] / MS
        atk["first_hit_s"] = (body.get("load_time") or 0) / MS
        atk["range_tiles"] = body["range"] / MS
        if body.get("sight_range"):
            atk["sight_tiles"] = body["sight_range"] / MS
        splash = _nz(body.get("area_damage_radius"), proj and proj.get("radius"))
        atk["splash_radius_tiles"] = (splash or 0) / MS
        if proj and proj.get("speed"):
            atk["projectile_speed_tiles_s"] = proj["speed"] / SPEED_UNIT
        out["deploy_time_s"] = (body.get("deploy_time") or 1000) / MS
        out["collision_radius_tiles"] = (body.get("collision_radius") or 500) / MS
        out["mass"] = body.get("mass") or 0   # push weight; 0 = immovable (buildings)
        if ref["section"] == "troop":
            out["count"] = max(1, rec.get("summon_number") or 1)
            out.setdefault("movement", {})["speed_tiles_s"] = body["speed"] / SPEED_UNIT
        else:
            out["lifetime_s"] = (body.get("life_time") or 0) / MS
    elif ref["section"] in ("spell", "projectile"):
        # projectile-type spells (Fireball, The Log) live directly in the projectile table
        proj = rec if ref["section"] == "projectile" else src.get("projectile", rec.get("projectile"))
        if proj is None:
            raise KeyError(f"{ref}: spell needs a projectile record")
        base["damage"] = proj["damage"]
        if proj.get("crown_tower_damage_percent") is not None:
            out["crown_tower_damage_pct"] = proj["crown_tower_damage_percent"]
        sp = out.setdefault("spell", {})
        sp["travel_speed_tiles_s"] = proj["speed"] / SPEED_UNIT
        if proj.get("projectile_range"):
            sp["rolling_range_tiles"] = proj["projectile_range"] / MS
            sp["width_tiles"] = 2 * proj["projectile_radius"] / MS
        elif proj.get("radius"):
            sp["radius_tiles"] = proj["radius"] / MS
    else:
        raise ValueError(f"unsupported section {ref['section']}")

    out["base_stats"] = base
    out.pop("stats_l11", None)
    out["data_source"] = f"royaleapi-cr-api-data:{ref['section']}/{ref['name']}"
    out["data_as_of"] = as_of
    out["verified"] = False
    return out


def sync(config_path: Path, source_path: Path, as_of: str, write: bool) -> Dict[str, Any]:
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    src = SourceIndex(json.loads(source_path.read_text(encoding="utf-8")))
    before = CardDB(cfg) if cfg.get("schema_version") == 3 else None
    new = copy.deepcopy(cfg)
    for cid, card in cfg["cards"].items():
        if "source_ref" in card:
            new["cards"][cid] = build_card(card, src, as_of)
    after = CardDB(new)

    print(f"{'card':<12} {'field':<20} {'L11 old':>8} {'L11 new':>8} {'L16 old':>8} {'L16 new':>8}")
    for cid in new["cards"]:
        a11, a16 = after.stats(cid, 11).stats, after.stats(cid, 16).stats
        b11 = before.stats(cid, 11).stats if before and cid in before.cards else {}
        b16 = before.stats(cid, 16).stats if before and cid in before.cards else {}
        for f in sorted(set(a11) | set(b11)):
            mark = "" if (b11.get(f), b16.get(f)) == (a11.get(f), a16.get(f)) else "  *"
            print(f"{cid:<12} {f:<20} {str(b11.get(f, '-')):>8} {str(a11.get(f, '-')):>8} "
                  f"{str(b16.get(f, '-')):>8} {str(a16.get(f, '-')):>8}{mark}")
    if write:
        config_path.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[+] wrote {config_path}")
    else:
        print("[i] dry run; pass --write to save")
    return new


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", required=True, type=Path, help="RoyaleAPI-format cards_stats.json")
    ap.add_argument("--config", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--as-of", required=True, help="date the source data reflects, e.g. 2023-10-18")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    sync(a.config, a.source, a.as_of, a.write)


if __name__ == "__main__":
    main()
