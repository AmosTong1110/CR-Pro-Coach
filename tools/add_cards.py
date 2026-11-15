"""Add cards to configs/cards_stats_v2026.json from a spec file plus RoyaleAPI-format source data.

    python tools/add_cards.py --spec tools/card_specs/batch1.json \
        --source cards_stats.json --cards cards.json --as-of 2023-10-18 [--write] [--replace]

Stats come from `sync_meta.build_card` (exact base stats + timings), identity (internal_name, game_id, aliases) from
RoyaleAPI's `cards.json`, the elixir cost from configs/card_costs_hf.json when present (current costs), and the
mechanics from the spec. Existing cards are left alone unless --replace is given. New cards are `verified: false`.
The source files are untrusted downloads: they are only parsed as JSON, never executed.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend_engine.simulator.card_db import CardDB  # noqa: E402
from backend_engine.simulator.sync_meta import SourceIndex, build_card  # noqa: E402

CFG = ROOT / "configs" / "cards_stats_v2026.json"
CLASS_MAP = ROOT / "configs" / "detector_class_map.json"
HF_COSTS = ROOT / "configs" / "card_costs_hf.json"
MS = 1000.0


def _fill(params: Any, values: Dict[str, Any]) -> Any:
    """Replace '$name' strings by values from the source record (recursively)."""
    if isinstance(params, dict):
        return {k: _fill(v, values) for k, v in params.items()}
    if isinstance(params, str) and params.startswith("$"):
        if params[1:] not in values or values[params[1:]] in (None, 0):
            raise KeyError(f"source has no value for {params}")
        return values[params[1:]]
    return params


def _source_values(src: SourceIndex, ref: Dict[str, str], rec: Dict[str, Any]) -> Dict[str, Any]:
    """The fields the spec may reference, read from the card's character / projectile / spell record."""
    body = src.get("characters", rec.get("summon_character")) or {}
    proj = src.get("projectile", body.get("projectile") or rec.get("projectile")) or {}
    spell_like = rec if ref["section"] in ("projectile", "spell") else {}
    buff_time = proj.get("buff_time") or spell_like.get("buff_time")
    return {
        "buff_time": buff_time,
        "pushback": (spell_like.get("pushback") or 0) / MS or None,
        "death_radius": (body.get("death_damage_radius") or 0) / MS or None,
        "chain_count": proj.get("chained_hit_count"),
        "chain_radius": (proj.get("chained_hit_radius") or 0) / MS or None,
        "idle_ms": body.get("buff_when_not_attacking_time"),               # Royal Ghost: idle time before invisible
        "spawn_deploy": spell_like.get("spawn_character_deploy_time"),     # Barbarian Barrel: token deploy time
    }


def _targets(body: Dict[str, Any]) -> list:
    if body.get("target_only_buildings"):
        return ["buildings"]
    return [t for t, on in (("ground", body.get("attacks_ground")), ("air", body.get("attacks_air"))) if on]


def make_card(cid: str, spec: Dict[str, Any], src: SourceIndex, api: Dict[str, dict], detector: Dict[str, list],
              hf_costs: Dict[str, int], as_of: str) -> Dict[str, Any]:
    ref = spec["source_ref"]
    rec = src.get(ref["section"], ref["name"])
    if rec is None:
        raise KeyError(f"{cid}: {ref} not found in source")
    meta = api[spec["api_key"]]
    values = _source_values(src, ref, rec)
    is_spell = ref["section"] in ("projectile", "spell")
    if spec.get("area_buff"):
        return _area_buff_card(cid, spec, rec, meta, src, detector, hf_costs, values, as_of)
    base: Dict[str, Any] = {
        "name": meta["name"], "type": "spell" if is_spell else "troop", "rarity": meta["rarity"].lower(),
        "elixir": hf_costs.get(cid, meta["elixir"]), "verified": False, "level_overrides": {},
        "mechanics": [{"kind": m["kind"], "params": _fill(m.get("params", {}), values)} for m in spec["mechanics"]],
        "source_ref": ref, "variants": {"evo": None, "hero": None},
    }
    if not is_spell:
        body = src.get("characters", rec.get("summon_character")) or rec
        base["movement"] = {"layer": "ground"}
        base["attack"] = {"targets": _targets(body)}
    card = build_card(base, src, as_of)
    if any(m["kind"] == "suicide" for m in card["mechanics"]):
        card["attack"].pop("projectile_speed_tiles_s", None)   # the hit is the kamikaze, not a flying shot
    return _with_identity(card, cid, meta, detector)


def _with_identity(card: Dict[str, Any], cid: str, meta: Dict[str, Any], detector: Dict[str, list]) -> Dict[str, Any]:
    card["internal_name"] = meta.get("sc_key") or ""
    card["game_id"] = meta["id"]
    card["aliases"] = {"display": meta["name"], "hf": cid, "royaleapi": meta["key"], "detector": detector.get(cid, [])}
    first = ["name", "internal_name", "game_id", "aliases", "type", "rarity", "elixir"]
    return {**{k: card[k] for k in first}, **{k: v for k, v in card.items() if k not in first}}


def _area_buff_card(cid, spec, rec, meta, src, detector, hf_costs, values, as_of) -> Dict[str, Any]:
    """A lingering area spell whose damage lives in a buff record (Tornado): damage per hit = dps * frequency,
    duration = the area's life, radius from the spell record. No `source_ref`, so sync_meta leaves it alone."""
    buff = src.get("character_buff", rec["buff"])
    if buff is None or not buff.get("damage_per_second") or not buff.get("hit_frequency"):
        raise KeyError(f"{cid}: buff {rec.get('buff')} has no damage over time")
    card: Dict[str, Any] = {
        "name": meta["name"], "type": "spell", "rarity": meta["rarity"].lower(),
        "elixir": hf_costs.get(cid, meta["elixir"]), "verified": False, "level_overrides": {},
        "mechanics": [{"kind": m["kind"], "params": _fill(m.get("params", {}), values)} for m in spec["mechanics"]],
        "variants": {"evo": None, "hero": None},
        "spell": {"radius_tiles": rec["radius"] / MS, "duration_ms": rec["life_duration"],
                  "hit_every_ms": buff["hit_frequency"], **spec.get("spell", {})},
        "base_stats": {"damage": round(buff["damage_per_second"] * buff["hit_frequency"] / MS)},
        "data_source": f"royaleapi-cr-api-data:spell/{rec['name']}+character_buff/{buff['name']}",
        "data_as_of": as_of,
    }
    if buff.get("crown_tower_damage_percent"):
        card["crown_tower_damage_pct"] = buff["crown_tower_damage_percent"]
    return _with_identity(card, cid, meta, detector)


def make_token(tid: str, spec: Dict[str, Any], src: SourceIndex, values: Dict[str, Any], as_of: str) -> Dict[str, Any]:
    """A unit that a card spawns. Its rarity is the parent card's (level scaling) and its `elixir` is the parent's
    cost (what the value model counts it as worth). Tokens are never in a deck."""
    ref = spec["source_ref"]
    rec = src.get(ref["section"], ref["name"])
    body = src.get("characters", rec.get("summon_character")) or rec
    base: Dict[str, Any] = {
        "name": body.get("name") or tid, "type": "troop", "rarity": spec["rarity"], "elixir": spec["elixir"],
        "verified": False, "level_overrides": {}, "source_ref": ref, "variants": {"evo": None, "hero": None},
        "mechanics": [{"kind": m["kind"], "params": _fill(m.get("params", {}), values)} for m in spec["mechanics"]],
        "movement": {"layer": "ground"}, "attack": {"targets": _targets(body)},
    }
    card = build_card(base, src, as_of)
    card.update(rarity=spec["rarity"], elixir=spec["elixir"], count=int(spec.get("count", 1)))
    card.pop("source_ref")
    return card


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--source", required=True, type=Path, help="RoyaleAPI cards_stats.json")
    ap.add_argument("--cards", required=True, type=Path, help="RoyaleAPI cards.json")
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--config", type=Path, default=CFG)
    ap.add_argument("--replace", action="store_true", help="overwrite cards that already exist")
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()

    spec = {k: v for k, v in json.loads(a.spec.read_text(encoding="utf-8")).items() if not k.startswith("_")}
    token_specs = spec.pop("tokens", {})
    src = SourceIndex(json.loads(a.source.read_text(encoding="utf-8")))
    api = {r["key"]: r for r in json.loads(a.cards.read_text(encoding="utf-8"))}
    detector: Dict[str, list] = {}
    for cl in json.loads(CLASS_MAP.read_text(encoding="utf-8"))["classes"]:
        if cl.get("card"):
            detector.setdefault(cl["card"], []).append(cl["name"])
    hf_costs = json.loads(HF_COSTS.read_text(encoding="utf-8"))["costs"]

    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    new = copy.deepcopy(cfg)
    for cid, sp in spec.items():
        if cid in cfg["cards"] and not a.replace:
            print(f"skip   {cid} (already in config; --replace to overwrite)")
            continue
        new["cards"][cid] = make_card(cid, sp, src, api, detector, hf_costs, a.as_of)
        print(f"{'replace' if cid in cfg['cards'] else 'add':<7}{cid}")
    for tid, ts in token_specs.items():
        if tid in cfg.get("tokens", {}) and not a.replace:
            print(f"skip   token {tid}")
            continue
        new.setdefault("tokens", {})[tid] = make_token(tid, ts, src, {}, a.as_of)
        print(f"add    token {tid}")
    db = CardDB(new)                                  # validates schema, rarities, mechanics, token references
    print(f"\n{'card':<15}{'type':<8}{'cost':>4}  L11 stats / L16 stats")
    for cid in [*spec, *token_specs]:
        if cid in db.cards or cid in db.tokens:
            s11, s16 = db.stats(cid, 11), db.stats(cid, 16)
            print(f"{cid:<15}{s11.type:<8}{s11.elixir:>4}  {s11.stats} / {s16.stats}")
    if a.write:
        a.config.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[+] wrote {a.config}")
    else:
        print("[i] dry run; pass --write to save")
    return 0


if __name__ == "__main__":
    sys.exit(main())
