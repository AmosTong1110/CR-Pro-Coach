"""Migrate configs/cards_stats_v2026.json from schema 3 to schema 4 (plan §4).

Per card:
  * evolution {cycles_required, mechanics[str], params} -> variants.evo {cycles_required, mechanics[{kind, params}],
    stat_mods, sc_key};  hero -> variants.hero
  * mechanics ["tag", ...] -> [{"kind": "tag", "params": {}}]   ("decaying_building" -> "decay")
  * internal_name (game sc_key), game_id (numeric, used by copyDeck links)
  * aliases {display, hf, royaleapi, detector[]}  (detector names from configs/detector_class_map.json)

    python tools/migrate_cards_v4.py --cards path/to/cr-api-data/cards.json [--write]

`cards.json` is RoyaleAPI's public card list. Without it, internal_name/game_id/royaleapi stay empty.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "cards_stats_v2026.json"
CLASS_MAP = ROOT / "configs" / "detector_class_map.json"
RENAME = {"decaying_building": "decay"}


def _mech(tags, params: Optional[Dict[str, Any]] = None):
    out = []
    for t in tags or []:
        if isinstance(t, dict):
            out.append({"kind": RENAME.get(t["kind"], t["kind"]), "params": dict(t.get("params", {}))})
        else:
            out.append({"kind": RENAME.get(t, t), "params": dict(params or {})})
    return out


def migrate_card(cid: str, c: Dict[str, Any], api: Dict[str, dict], detector: Dict[str, list]) -> Dict[str, Any]:
    c = copy.deepcopy(c)
    by_name = {v["name"]: v for v in api.values()}
    rec = api.get(cid.replace("_", "-")) or by_name.get(c["name"])
    evo = c.pop("evolution", None)
    hero = c.pop("hero", None)
    c["mechanics"] = _mech(c.get("mechanics"))
    c["variants"] = {
        "evo": None if not evo else {
            "cycles_required": int(evo["cycles_required"]),
            "mechanics": _mech(evo.get("mechanics"), evo.get("params")),
            "stat_mods": dict(evo.get("stat_mods", {})),
            "sc_key": (rec or {}).get("evolved_spells_sc_key") or None,
        },
        "hero": hero,
    }
    c["internal_name"] = (rec or {}).get("sc_key") or c.get("internal_name") or ""
    c["game_id"] = (rec or {}).get("id") or c.get("game_id")
    c["aliases"] = {
        "display": c["name"],
        "hf": cid,                                    # HF replay data uses our snake_case ids
        "royaleapi": (rec or {}).get("key") or "",
        "detector": detector.get(cid, []),
    }
    # stable, readable key order: identity first
    first = ["name", "internal_name", "game_id", "aliases", "type", "rarity", "elixir"]
    return {**{k: c[k] for k in first if k in c}, **{k: v for k, v in c.items() if k not in first}}


def migrate(raw: Dict[str, Any], api: Dict[str, dict]) -> Dict[str, Any]:
    if raw.get("schema_version") == 4:
        return raw
    detector: Dict[str, list] = {}
    if CLASS_MAP.exists():
        for cl in json.loads(CLASS_MAP.read_text())["classes"]:
            if cl.get("card"):
                detector.setdefault(cl["card"], []).append(cl["name"])
    out = copy.deepcopy(raw)
    out["schema_version"] = 4
    out["cards"] = {cid: migrate_card(cid, c, api, detector) for cid, c in raw["cards"].items()}
    out.setdefault("_doc", []).append(
        "Schema 4: identity (internal_name = game sc_key, game_id, aliases) and variants {evo, hero} live inside the "
        "base card; mechanics are [{kind, params}] resolved by backend_engine/simulator/mechanics.")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cards", type=Path, help="RoyaleAPI cards.json")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    api = {r["key"]: r for r in json.loads(a.cards.read_text())} if a.cards else {}
    new = migrate(json.loads(CFG.read_text()), api)
    for cid, c in new["cards"].items():
        miss = [k for k in ("internal_name", "game_id") if not c.get(k)]
        print(f"{cid:<12} {c['internal_name'] or '?':<12} {c.get('game_id') or '?':<9} "
              f"detector={c['aliases']['detector']} evo={'yes' if c['variants']['evo'] else 'no'}"
              + (f"  MISSING {miss}" if miss else ""))
    if a.write:
        CFG.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n")
        print(f"wrote {CFG}")


if __name__ == "__main__":
    sys.exit(main())
