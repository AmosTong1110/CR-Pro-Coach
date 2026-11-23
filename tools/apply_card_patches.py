"""Apply a reviewed patch file of card / tower corrections to configs/cards_stats_v2026.json.

    python tools/apply_card_patches.py --patch tools/card_specs/audit_2026_11.json [--write]

Each operation names its source (a saved wiki page or an official balance note). Operations:
  {"op": "wiki_levels", "card"|"tower": id, "page": "<wiki title>", "stats": {stat: column}}
        level 11-16 numbers straight from a saved page in data/raw_sources/cards or /towers (parsed, never executed)
  {"op": "official_l11", "card": id, "stat": s, "l11": v [, "variant": "evo"|"hero"]}
        a newer official level-11 value (balance notes) the wiki table does not have yet: every level is scaled by
        v / current L11 (rounded) and the stat is listed in `newer_than_wiki`, so verify_wiki reports it as "newer"
  {"op": "set", "card": id, "path": "attack.hit_speed_s", "value": x}
  {"op": "mech_param", "card": id, "kind": k, "param": p, "value": x [, "variant": v]}   (dotted p for nested params)
Every touched record gets an `audit` list with the operation's `source`. Without --write nothing is saved.
"""
from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from backend_engine.simulator.card_db import CardDB  # noqa: E402
from backend_engine.simulator.sync_meta import load_wiki_dir  # noqa: E402

CFG = ROOT / "configs" / "cards_stats_v2026.json"
PAGES = (ROOT / "data" / "raw_sources" / "cards", ROOT / "data" / "raw_sources" / "towers")
LEVELS = ("11", "12", "13", "14", "15", "16")


def _record(cfg: Dict[str, Any], op: Dict[str, Any]) -> Dict[str, Any]:
    if "tower" in op:
        return cfg["towers"][op["tower"]]
    cid = op["card"]
    rec = cfg["cards"].get(cid) or cfg["tokens"].get(cid)
    if rec is None:
        raise KeyError(f"unknown card or token '{cid}'")
    if op.get("variant"):
        rec = rec["variants"][op["variant"]]
        if rec is None:
            raise KeyError(f"{cid} has no {op['variant']} variant")
    return rec


def _set_path(d: Dict[str, Any], path: str, value: Any) -> None:
    *parents, leaf = path.split(".")
    for k in parents:
        d = d.setdefault(k, {})
    d[leaf] = value


def apply(cfg: Dict[str, Any], ops: List[Dict[str, Any]], pages: Dict[str, Any]) -> List[str]:
    log: List[str] = []
    for op in ops:
        rec = _record(cfg, op)
        name = op.get("card") or op.get("tower")
        kind = op["op"]
        if kind == "wiki_levels":
            pg = pages[op["page"].lower()]
            ov = rec.setdefault("level_overrides", {})
            for L in LEVELS:
                row = ov.setdefault(L, {})
                for stat, col in op["stats"].items():
                    v = pg.levels.get(int(L), {}).get(col)
                    if v is None:
                        raise KeyError(f"{op['page']}: no '{col}' at level {L}")
                    row[stat] = int(v)
            rec.setdefault("wiki_columns", {}).update(op["stats"])
            for stat in op["stats"]:
                (rec.get("newer_than_wiki") or {}).pop(stat, None)
            log.append(f"{name}: wiki L11-16 {sorted(op['stats'])} <- {op['page']}  (L11 {ov['11']})")
        elif kind == "official_l11":
            ov = rec.get("level_overrides") or {}
            cur = (ov.get("11") or {}).get(op["stat"])
            if cur is None:
                raise KeyError(f"{name}: no level-11 '{op['stat']}' to scale")
            for L in LEVELS:
                if op["stat"] in ov.get(L, {}):
                    ov[L][op["stat"]] = int(round(ov[L][op["stat"]] * float(op["l11"]) / float(cur)))
            rec.setdefault("newer_than_wiki", {})[op["stat"]] = op["source"]
            log.append(f"{name}{'#' + op['variant'] if op.get('variant') else ''}: {op['stat']} L11 {cur} -> {op['l11']}")
        elif kind == "set":
            old = copy.deepcopy(rec)
            for k in op["path"].split("."):
                old = old.get(k) if isinstance(old, dict) else None
            _set_path(rec, op["path"], op["value"])
            log.append(f"{name}: {op['path']} {old} -> {op['value']}")
        elif kind == "mech_param":
            mech = next((m for m in rec.get("mechanics", []) if m["kind"] == op["kind"]), None)
            if mech is None:
                raise KeyError(f"{name}: no mechanic '{op['kind']}'")
            *parents, leaf = op["param"].split(".")
            d = mech.setdefault("params", {})
            for k in parents:
                d = d.setdefault(k, {})
            old = d.get(leaf)
            d[leaf] = op["value"]
            log.append(f"{name}: {op['kind']}.{op['param']} {old} -> {op['value']}")
        else:
            raise ValueError(f"unknown op '{kind}'")
        audit = rec.setdefault("audit", [])
        if op["source"] not in audit:
            audit.append(op["source"])
    return log


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--patch", required=True, type=Path)
    ap.add_argument("--config", type=Path, default=CFG)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    ops = [o for o in json.loads(a.patch.read_text(encoding="utf-8"))["ops"]]
    pages = {p.title.lower(): p for d in PAGES if d.is_dir() for p in load_wiki_dir(d)}
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    new = copy.deepcopy(cfg)
    for line in apply(new, ops, pages):
        print(line)
    CardDB(new)                                       # validates
    if a.write:
        a.config.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[+] wrote {a.config}")
    else:
        print("[i] dry run; pass --write to save")
    return 0


if __name__ == "__main__":
    sys.exit(main())
