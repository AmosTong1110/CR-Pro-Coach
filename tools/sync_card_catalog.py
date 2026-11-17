"""Official patch & new-card sync: keep configs/cards_stats_v2026.json in step with the live game.

Reads one or more card lists and updates the catalog:
  * balance patches  - new cost / rarity, new Level 11 (tournament standard) stats of known cards. Stats land in
                       `level_overrides["11"]` (exact values win over the scaling formula), stamped
                       `stats_patched_as_of`. Other levels are only touched if the source gives them.
  * new cards        - added to the catalog with identity, cost, rarity and whatever stats / mechanics the source
                       has. A new card is **auto-enabled** only when it is complete (stats for its type) and every
                       mechanic it names is one the engine already simulates (backend_engine/simulator/traits.py).
                       A card naming a novel mechanic gets `"enabled": false, "needs_hook": true`; a card with
                       identity only (e.g. the official API, which has no stats) gets `"enabled": false`. Either
                       way it is in the catalog, so replays / battle logs with it are recognised, and it never
                       enters a playable pool, a deck or the encoder's hand until it is complete. Such cards carry
                       `catalog_status` (incomplete / needs_hook / complete); a later sync that brings the missing
                       stats re-decides them. Cards disabled by hand (no catalog_status) are never switched on.
  * Tower Troops     - official `supportItems` (Tower Princess, Cannoneer, Dagger Duchess, Royal Chef, ...) go to
                       the `towers` section the same way (disabled until they have stats).
  * conflicts        - a source card whose game_id belongs to a differently named catalog card is reported, never
                       applied (that is how Tombstone's wrong game_id was found).

Sources (auto-detected; several may be given, later ones win):
  * official API     {"items": [...], "supportItems": [...]}  (api.clashroyale.com /v1/cards, also
                     data/raw_sources/official_cards.json)
  * RoyaleAPI        [{"key", "name", "id", "elixir", "type", "rarity"}, ...]  (cr-api-data json/cards.json)
  * CRbot snapshot   {"format": "crbot_catalog_snapshot_v1", "as_of": "...", "cards": {id_or_name: {"name",
                     "game_id", "elixir", "type", "rarity", "stats": {"11": {...}}, "attack", "movement",
                     "spell", "mechanics", "count", "deploy_time_s", ...}}, "towers": {...}}
                     (hand-written or converted from patch notes: the way to feed exact Level 11 numbers)

    .venv/bin/python tools/sync_card_catalog.py --snapshot data/raw_sources/official_cards.json      # dry run
    .venv/bin/python tools/sync_card_catalog.py --snapshot data/raw_sources/official_cards.json --write
    .venv/bin/python tools/sync_card_catalog.py --snapshot patch_2026_12.json --as-of 2026-12-01 --write
    .venv/bin/python tools/sync_card_catalog.py --fetch royaleapi              # cr-api-data cards.json
    CR_API_TOKEN=... .venv/bin/python tools/sync_card_catalog.py --fetch official --write

Dry run by default: prints the report, writes nothing. --write rewrites the card file in its own format
(json indent 2) and validates it with CardDB first. After a write: run the tests; if gameplay numbers of playable
cards changed, re-record the goldens (tools/record_golden.py --write) and bump engine.ENGINE_VERSION.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import sys
import urllib.request
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend_engine.simulator.card_db import CARD_TYPES, CardDB  # noqa: E402
from backend_engine.simulator.quarantine import _slug, alias_index  # noqa: E402
from backend_engine.simulator.traits import unsupported_kinds  # noqa: E402

CFG = ROOT / "configs" / "cards_stats_v2026.json"
HF_COSTS = ROOT / "configs" / "card_costs_hf.json"
ROYALEAPI_URL = "https://royaleapi.github.io/cr-api-data/json/cards.json"
OFFICIAL_URL = "https://api.clashroyale.com/v1/cards"
SNAPSHOT_FORMAT = "crbot_catalog_snapshot_v1"
TYPE_BY_PREFIX = {26: "troop", 27: "building", 28: "spell", 159: "tower"}
BODY_KEYS = ("attack", "movement", "spell", "mechanics", "count", "deploy_time_s", "collision_radius_tiles", "mass",
             "lifetime_s", "footprint_tiles", "crown_tower_damage_pct", "variants")


@dataclass
class Incoming:
    """One card (or Tower Troop) from a source, in the catalog's vocabulary."""
    name: str
    kind: str                                   # troop / building / spell / tower
    game_id: Optional[int] = None
    elixir: Optional[int] = None
    rarity: Optional[str] = None
    key: Optional[str] = None                   # RoyaleAPI key (aliases.royaleapi)
    stats: Dict[str, Dict[str, int]] = field(default_factory=dict)   # level -> {field: value}
    body: Dict[str, Any] = field(default_factory=dict)
    source: str = ""


@dataclass
class Report:
    added: List[Tuple[str, str]] = field(default_factory=list)       # (id, status)
    updated: List[Tuple[str, str]] = field(default_factory=list)     # (id, what changed)
    conflicts: List[str] = field(default_factory=list)
    towers: List[Tuple[str, str]] = field(default_factory=list)
    unchanged: int = 0

    def lines(self) -> List[str]:
        out = [f"added {len(self.added)}, updated {len(self.updated)}, unchanged {self.unchanged}, "
               f"conflicts {len(self.conflicts)}, tower troops {len(self.towers)}"]
        out += [f"  + {cid:<22} {status}" for cid, status in self.added]
        out += [f"  ~ {cid:<22} {what}" for cid, what in self.updated]
        out += [f"  T {tid:<22} {status}" for tid, status in self.towers]
        out += [f"  ! {c}" for c in self.conflicts]
        return out


# -- parsing ----------------------------------------------------------------------------------------------------
def _kind_from_id(gid: Optional[int], fallback: str = "troop") -> str:
    return TYPE_BY_PREFIX.get((gid or 0) // 1_000_000, fallback)


def parse_source(data: Any, label: str = "") -> List[Incoming]:
    """Any supported card list -> Incoming records."""
    out: List[Incoming] = []
    if isinstance(data, dict) and data.get("format") == SNAPSHOT_FORMAT:
        for section, kind_default in (("cards", None), ("towers", "tower")):
            for key, c in (data.get(section) or {}).items():
                gid = c.get("game_id")
                kind = kind_default or c.get("type") or _kind_from_id(gid)
                stats = {str(k): dict(v) for k, v in (c.get("stats") or {}).items()}
                out.append(Incoming(c.get("name") or key, kind, gid, c.get("elixir"), (c.get("rarity") or "").lower()
                                    or None, c.get("key"), stats, {k: c[k] for k in BODY_KEYS if k in c},
                                    label or "snapshot"))
        return out
    if isinstance(data, dict) and "items" in data:                       # official API
        for it in data["items"]:
            out.append(Incoming(it["name"], _kind_from_id(it.get("id")), it.get("id"), it.get("elixirCost"),
                                (it.get("rarity") or "").lower() or None, source=label or "official"))
        for it in data.get("supportItems") or []:
            out.append(Incoming(it["name"], "tower", it.get("id"), None, (it.get("rarity") or "").lower() or None,
                                source=label or "official"))
        return out
    if isinstance(data, list):                                           # RoyaleAPI cards.json
        for it in data:
            if not isinstance(it, dict) or "name" not in it:
                continue
            kind = (it.get("type") or "").lower() or _kind_from_id(it.get("id"))
            out.append(Incoming(it["name"], kind if kind in CARD_TYPES else _kind_from_id(it.get("id")),
                                it.get("id"), it.get("elixir"), (it.get("rarity") or "").lower() or None,
                                it.get("key"), source=label or "royaleapi"))
        return out
    raise ValueError(f"{label}: unknown card list format")


# -- completeness ------------------------------------------------------------------------------------------------
def missing_for_play(card: Dict[str, Any]) -> List[str]:
    """What a card still lacks before the engine can play it (by type)."""
    stats = card.get("base_stats") or card.get("stats_l11") or {}
    miss = []
    kind = card.get("type")
    if kind in ("troop", "building"):
        if "hitpoints" not in stats:
            miss.append("hitpoints")
        if kind == "troop" and not (card.get("movement") or {}).get("speed_tiles_s"):
            miss.append("movement")
        spawns = any(m.get("kind") in ("periodic_spawn", "death_spawn") for m in card.get("mechanics") or [])
        if not card.get("attack") and not spawns:
            miss.append("attack")
        if card.get("attack") and "damage" not in stats:
            miss.append("damage")
    elif kind == "spell":
        if "damage" not in stats:
            miss.append("damage")
        sp = card.get("spell") or {}
        if not (sp.get("radius_tiles") or sp.get("rolling_range_tiles")):
            miss.append("spell radius")
    return miss


def decide_status(card: Dict[str, Any]) -> str:
    """Set enabled / needs_hook on a new card; returns a short status for the report."""
    novel = unsupported_kinds(card)
    if novel:
        card["enabled"], card["needs_hook"], card["catalog_status"] = False, True, "needs_hook"
        return f"needs_hook (novel mechanics: {', '.join(novel)})"
    miss = missing_for_play(card)
    card.pop("needs_hook", None)
    if miss:
        card["enabled"], card["catalog_status"] = False, "incomplete"
        return f"disabled (missing: {', '.join(miss)})"
    card["enabled"], card["catalog_status"] = True, "complete"
    return "enabled (standard traits only)"


# -- planning ----------------------------------------------------------------------------------------------------
def _internal_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9]", "", name.title())


def _apply_stats(card: Dict[str, Any], stats: Dict[str, Dict[str, int]], db: Optional[CardDB], cid: str,
                 as_of: str) -> List[str]:
    """Write per-level stats as level_overrides where they differ; returns 'field Lx a->b' notes."""
    notes = []
    for level, values in sorted(stats.items()):
        cur: Dict[str, int] = {}
        if db is not None and cid in db.catalog:
            try:
                cur = db.stats(cid, int(level)).stats
            except Exception:                              # quarantined card / unreachable level: compare raw
                cur = dict((card.get("level_overrides") or {}).get(level) or {})
        for k, v in values.items():
            if cur.get(k) != int(v):
                card.setdefault("level_overrides", {}).setdefault(level, {})[k] = int(v)
                notes.append(f"{k} L{level} {cur.get(k)}->{int(v)}")
    if notes:
        card["stats_patched_as_of"] = as_of
    return notes


def plan_sync(raw: Dict[str, Any], incoming: List[Incoming], as_of: str,
              hf_costs: Optional[Dict[str, int]] = None) -> Tuple[Dict[str, Any], Report]:
    """Pure: (new card-file dict, report). Nothing is read or written here."""
    new = copy.deepcopy(raw)
    db = CardDB(copy.deepcopy(raw))
    index = alias_index(db)
    by_gid = {c.get("game_id"): cid for cid, c in new["cards"].items() if c.get("game_id")}
    towers = new.setdefault("towers", {})
    tower_gid = {t.get("game_id"): tid for tid, t in towers.items() if t.get("game_id")}
    rep = Report()
    hf_costs = hf_costs or {}

    for inc in incoming:
        slug = _slug(inc.name)
        if inc.kind == "tower":
            tid = tower_gid.get(inc.game_id) or ("princess_tower" if slug == "tower_princess" else slug)
            t = towers.get(tid)
            if t is None:
                t = towers[tid] = {"name": inc.name, "rarity": inc.rarity or "common", "enabled": False,
                                   "data_as_of": as_of, "data_source": inc.source}
                status = "added (disabled until it has stats_l11 + attack)"
            else:
                status = "known"
            if inc.game_id and t.get("game_id") != inc.game_id:
                t["game_id"] = inc.game_id
                status += f", game_id {inc.game_id}"
            t.setdefault("name", inc.name)
            if inc.stats.get("11"):
                t["stats_l11"] = {**t.get("stats_l11", {}), **inc.stats["11"]}
            for k in ("attack", "footprint_tiles"):
                if k in inc.body:
                    t[k] = inc.body[k]
            if t.get("enabled") is False and t.get("stats_l11", {}).get("hitpoints") and t.get("attack"):
                t["enabled"] = True
                status += ", enabled"
            rep.towers.append((tid, status))
            continue

        cid = by_gid.get(inc.game_id) if inc.game_id else None
        named = index.get(slug) or index.get(_slug(inc.key or ""))
        if cid and named and named != cid:
            rep.conflicts.append(f"{inc.name} (game_id {inc.game_id}) is '{cid}' by id but '{named}' by name: "
                                 f"skipped, fix the game_id in the card file")
            continue
        cid = cid or named
        if cid is not None and cid in new["cards"]:
            card = new["cards"][cid]
            what = []
            if inc.elixir is not None and card.get("elixir") != inc.elixir:
                what.append(f"elixir {card.get('elixir')}->{inc.elixir}")
                card["elixir"] = inc.elixir
            if inc.rarity and card.get("rarity") != inc.rarity:
                what.append(f"rarity {card.get('rarity')}->{inc.rarity}")
                card["rarity"] = inc.rarity
            if inc.game_id and card.get("game_id") != inc.game_id:
                what.append(f"game_id {card.get('game_id')}->{inc.game_id}")
                card["game_id"] = inc.game_id
            # a catalog card the sync created and that isn't playable yet: complete it and decide again
            # (cards disabled by hand carry no catalog_status and are never switched on here)
            pending = cid not in db.cards and bool(card.get("catalog_status"))
            stats = inc.stats
            if pending and stats.get("11") and not (card.get("base_stats") or card.get("stats_l11")):
                card["stats_l11"] = dict(stats["11"])
                what.append("stats_l11")
                stats = {k: v for k, v in stats.items() if k != "11"}
            what += _apply_stats(card, stats, db, cid, as_of)
            for k, v in inc.body.items():
                if (k != "mechanics" or pending) and card.get(k) != v:
                    card[k] = v
                    what.append(k)
            if pending and what:
                before = card.get("catalog_status")
                status = decide_status(card)
                if card["catalog_status"] != before:
                    what.append(f"-> {status}")
            if what:
                rep.updated.append((cid, ", ".join(what)))
            else:
                rep.unchanged += 1
            continue

        # a new card
        cid = slug
        if cid in new.get("tokens", {}):
            cid = f"{cid}_card"
        elixir = inc.elixir if inc.elixir is not None else hf_costs.get(cid)
        card: Dict[str, Any] = {
            "name": inc.name, "internal_name": _internal_name(inc.name), "game_id": inc.game_id,
            "aliases": {"display": inc.name, "hf": cid, "royaleapi": inc.key or cid.replace("_", "-"),
                        "detector": []},
            "type": inc.kind if inc.kind in CARD_TYPES else "troop",
            "rarity": inc.rarity or "common", "elixir": int(elixir) if elixir is not None else 0,
            "verified": False, "data_as_of": as_of, "data_source": inc.source,
            "mechanics": list(inc.body.get("mechanics") or []), "variants": {"evo": None, "hero": None},
            "level_overrides": {},
        }
        if elixir is None:
            card["assumed"] = ["elixir (no cost in the source)"]
        for k, v in inc.body.items():
            if k != "mechanics":
                card[k] = v
        if inc.stats.get("11"):
            card["stats_l11"] = dict(inc.stats["11"])
            for level, values in inc.stats.items():
                if level != "11":
                    card["level_overrides"][level] = dict(values)
        rep.added.append((cid, decide_status(card)))
        new["cards"][cid] = card
        index[slug] = cid
        if inc.game_id:
            by_gid[inc.game_id] = cid
    return new, rep


# -- IO ----------------------------------------------------------------------------------------------------------
def fetch(which: str) -> Any:
    """Download a card list. No personal data is sent; the official API needs CR_API_TOKEN (developer key)."""
    headers = {"User-Agent": "crbot-catalog-sync", "Accept": "application/json"}
    if which == "official":
        token = os.environ.get("CR_API_TOKEN")
        if not token:
            raise SystemExit("--fetch official needs CR_API_TOKEN (developer.clashroyale.com key for this IP)")
        headers["Authorization"] = f"Bearer {token}"
        url = OFFICIAL_URL
    else:
        url = ROYALEAPI_URL
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.loads(r.read().decode("utf-8"))


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--snapshot", action="append", default=[], type=Path, help="local card list (repeatable)")
    ap.add_argument("--fetch", choices=("royaleapi", "official"), action="append", default=[])
    ap.add_argument("--config", type=Path, default=CFG)
    ap.add_argument("--as-of", default=str(date.today()))
    ap.add_argument("--write", action="store_true", help="write the card file (default: dry run)")
    args = ap.parse_args(argv)
    if not args.snapshot and not args.fetch:
        ap.error("give --snapshot PATH and/or --fetch royaleapi|official")
    incoming: List[Incoming] = []
    for p in args.snapshot:
        incoming += parse_source(json.loads(p.read_text(encoding="utf-8")), p.name)
    for which in args.fetch:
        incoming += parse_source(fetch(which), which)
    raw = json.loads(args.config.read_text(encoding="utf-8"))
    hf = json.loads(HF_COSTS.read_text(encoding="utf-8")).get("costs", {}) if HF_COSTS.exists() else {}
    new, rep = plan_sync(raw, incoming, args.as_of, hf)
    print("\n".join(rep.lines()))
    db = CardDB(copy.deepcopy(new))                        # validates; raises before anything is written
    print(f"catalog {len(db.catalog)} cards: {len(db.cards)} playable, {len(db.quarantined())} quarantined")
    if args.write:
        args.config.write_text(json.dumps(new, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"wrote {args.config}")
    else:
        print("dry run: nothing written (use --write)")
    return 1 if rep.conflicts else 0


if __name__ == "__main__":
    raise SystemExit(main())
