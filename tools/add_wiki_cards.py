"""Add cards to configs/cards_stats_v2026.json from saved Clash Royale Wiki pages plus a spec file.

    python tools/add_wiki_cards.py --spec tools/card_specs/batch2.json --html-dir data/raw_sources/cards \
        --as-of 2026-11-16 [--write] [--replace]

Numbers (L11-L16 hitpoints / damage / ...) and attributes (cost, timings, range, speed) come from the wiki page;
identity (game id, rarity, cost) from data/raw_sources/official_cards.json; the mechanics, the targets and the
fields the wiki doesn't list (marked `assumed` in the spec) from the spec. New cards are `verified: false`.
The saved pages are untrusted files: they are only parsed, never executed.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend_engine.simulator.card_db import CardDB  # noqa: E402
from backend_engine.simulator.sync_meta import WikiPage, attribute_fields, load_wiki_dir  # noqa: E402

CFG = ROOT / "configs" / "cards_stats_v2026.json"
OFFICIAL = ROOT / "data" / "raw_sources" / "official_cards.json"
CLASS_MAP = ROOT / "configs" / "detector_class_map.json"
LEVELS = (11, 12, 13, 14, 15, 16)


def _set(d: Dict[str, Any], path: str, value: Any) -> None:
    *parents, leaf = path.split(".")
    for k in parents:
        d = d.setdefault(k, {})
    d[leaf] = value


def _overrides(pg: WikiPage, stats: Dict[str, str], totals: List[str]) -> Dict[str, Dict[str, int]]:
    out: Dict[str, Dict[str, int]] = {}
    for L in LEVELS:
        row = {}
        for key, col in stats.items():
            v = pg.totals.get(L, {}).get(col) if key in totals else pg.levels.get(L, {}).get(col)
            if v is None:
                raise KeyError(f"{pg.title}: no value for column '{col}' at level {L}")
            row[key] = int(v)
        out[str(L)] = row
    return out


def _build(cid: str, sp: Dict[str, Any], pg: WikiPage, official: Dict[str, dict], detector: Dict[str, list],
           as_of: str, token: bool = False) -> Dict[str, Any]:
    sp = {"type": "troop", "layer": "ground", **sp}
    attrs = pg.tables[sp.get("table", 0)] if pg.tables else {}
    stats = {**sp["stats"], **sp.get("stats_extra", {})}
    ov = _overrides(pg, stats, sp.get("totals", []))
    card: Dict[str, Any] = {"type": sp["type"], "verified": False, "level_overrides": ov}
    stub = {"type": card["type"], "spell": {"radius_tiles": 0, "width_tiles": 0} if sp["type"] == "spell" else {}}
    if sp["type"] == "spell":
        stub["spell"] = {"radius_tiles": 0, "duration_ms": 0}
    fields = attribute_fields(pg, stub, attrs)
    card["attack"] = {"targets": sp.get("attack", {}).get("targets", ["ground"])}
    if sp["type"] != "spell":
        card["movement"] = {"layer": sp.get("layer", "ground")}
    for path, v in fields.items():
        if path == "elixir":
            continue
        if path == "count":
            v = int(v)
        _set(card, path, v)
    if sp["type"] == "spell":
        card.pop("attack")
        card.setdefault("spell", {}).update(sp.get("spell", {}))
    else:
        card["attack"].update({k: v for k, v in sp.get("attack", {}).items() if k != "targets"})
        card["attack"].setdefault("splash_radius_tiles", 0.0)
        card["attack"].setdefault("sight_tiles", max(5.5, float(card["attack"].get("range_tiles", 0))))
        if sp["type"] == "troop":
            card.setdefault("count", 1)
    for path, v in sp.get("attr_override", {}).items():
        _set(card, path, v)
    card.update({k: v for k, v in sp.get("extra", {}).items() if k != "name"})
    card["mechanics"] = copy.deepcopy(sp["mechanics"])
    card["variants"] = {"evo": None, "hero": None}
    card["base_stats"] = {k: ov["11"][k] for k in stats}        # reference only: every level is overridden
    card["data_source"] = f"wiki-fandom-html:{pg.title}" + (f"#table{sp['table']}" if token else "")
    card["data_as_of"] = as_of
    card["wiki_as_of"] = as_of
    card["assumed"] = sp.get("assumed", [])
    if token:
        card.update(name=sp["extra"]["name"], rarity=sp["rarity"], elixir=sp["elixir"])
        return card
    meta = official[pg.title.lower()]
    card["name"], card["rarity"] = meta["name"], meta["rarity"].lower()
    card["elixir"] = int(sp.get("attr_override", {}).get("elixir") or fields.get("elixir") or meta["elixirCost"])
    card["internal_name"], card["game_id"] = sp.get("internal_name", ""), meta["id"]
    card["aliases"] = {"display": meta["name"], "hf": cid, "royaleapi": cid.replace("_", "-"),
                       "detector": detector.get(cid, [])}
    first = ["name", "internal_name", "game_id", "aliases", "type", "rarity", "elixir"]
    return {**{k: card[k] for k in first}, **{k: v for k, v in card.items() if k not in first}}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--html-dir", required=True, type=Path)
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--config", type=Path, default=CFG)
    ap.add_argument("--replace", action="store_true")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    spec = {k: v for k, v in json.loads(a.spec.read_text(encoding="utf-8")).items() if not k.startswith("_")}
    token_specs = spec.pop("tokens", {})
    pages = {p.title.lower(): p for p in load_wiki_dir(a.html_dir) if p.variant == "base"}
    official = {i["name"].lower(): i for i in json.loads(OFFICIAL.read_text(encoding="utf-8"))["items"]}
    detector: Dict[str, list] = {}
    for cl in json.loads(CLASS_MAP.read_text(encoding="utf-8"))["classes"]:
        if cl.get("card"):
            detector.setdefault(cl["card"], []).append(cl["name"])

    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    new = copy.deepcopy(cfg)
    for cid, sp in spec.items():
        if cid in cfg["cards"] and not a.replace:
            print(f"skip   {cid} (already in config)")
            continue
        new["cards"][cid] = _build(cid, sp, pages[sp["wiki"].lower()], official, detector, a.as_of)
        print(f"add    {cid}")
    for tid, ts in token_specs.items():
        if tid in cfg.get("tokens", {}) and not a.replace:
            continue
        new.setdefault("tokens", {})[tid] = _build(tid, ts, pages[ts["wiki"].lower()], official, detector, a.as_of, True)
        print(f"add    token {tid}")
    db = CardDB(new)
    for cid in [*spec, *token_specs]:
        if cid in db.cards or cid in db.tokens:
            print(f"{cid:<18}{db.stats(cid, 11).elixir:>2}  L11 {db.stats(cid, 11).stats}")
    if a.write:
        a.config.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[+] wrote {a.config}")
    else:
        print("[i] dry run; pass --write to save")
    return 0


if __name__ == "__main__":
    sys.exit(main())
