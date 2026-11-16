"""Add evolution / hero variants to configs/cards_stats_v2026.json from saved Clash Royale Wiki pages plus a spec.

    python tools/add_variants.py --spec tools/card_specs/variants.json --html-dir data/raw_sources/cards \
        --as-of 2026-11-16 [--write]

Per variant: cycles (evolutions), the spec's mechanics (a simulated champion_ability for five heroes, a planned kind
for the rest), and `level_overrides` L11-L21 with the headline stats that differ from the base card plus the ability
numbers. The saved pages are untrusted files: they are only parsed, never executed.
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
from backend_engine.simulator.sync_meta import (VARIANT_LEVELS, load_wiki_dir, page_cycles,  # noqa: E402
                                                variant_overrides)

CFG = ROOT / "configs" / "cards_stats_v2026.json"


def build_variant(cid: str, kind: str, sp: Dict[str, Any], pages: Dict[tuple, Any], card: Dict[str, Any],
                  old: Dict[str, Any], as_of: str, db: CardDB) -> Dict[str, Any]:
    pg, base = pages[(cid, kind)], pages[(cid, "base")]
    v: Dict[str, Any] = {"mechanics": old.get("mechanics", []) if sp.get("keep_mechanics") else copy.deepcopy(sp["mechanics"]),
                         "stat_mods": old.get("stat_mods", {}), "level_overrides": variant_overrides(base, pg, card, sp.get("stats_extra", {}))}
    if kind == "evo":
        v["cycles_required"] = page_cycles(pg)
    for k in ("sc_key",):
        if k in old:
            v[k] = old[k]
    v["data_source"] = f"wiki-fandom-html:{pg.title}"
    v["wiki_as_of"] = as_of
    if sp.get("assumed"):
        v["assumed"] = sp["assumed"]
    return v


def build_token(tid: str, sp: Dict[str, Any], pages: Dict[tuple, Any], as_of: str) -> Dict[str, Any]:
    pg = pages[(sp["from"]["card"], sp["from"]["variant"])]
    ov = {}
    for L in VARIANT_LEVELS:
        if L in pg.levels and all(c in pg.levels[L] for c in sp["stats"].values()):
            ov[str(L)] = {k: int(round(pg.levels[L][c] * sp.get("scale", {}).get(k, 1))) for k, c in sp["stats"].items()}
    first = ov["11"]
    tok = {"name": sp["extra"]["name"], "type": sp["type"], "rarity": sp["rarity"], "elixir": sp["elixir"], "verified": False,
           "level_overrides": ov, "variants": {"evo": None, "hero": None}, "mechanics": [],
           "attack": sp["attack"], "base_stats": dict(first),
           "data_source": f"wiki-fandom-html:{pg.title} (hero turret columns)", "data_as_of": as_of, "wiki_as_of": as_of,
           "assumed": sp.get("assumed", [])}
    tok.update({k: v for k, v in sp["extra"].items() if k != "name"})
    if sp["type"] == "building":
        tok["mass"] = 0
    return tok


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--spec", required=True, type=Path)
    ap.add_argument("--html-dir", required=True, type=Path)
    ap.add_argument("--as-of", required=True)
    ap.add_argument("--config", type=Path, default=CFG)
    ap.add_argument("--write", action="store_true")
    a = ap.parse_args()
    spec = json.loads(a.spec.read_text(encoding="utf-8"))
    pages = {(p.card_id, p.variant): p for p in load_wiki_dir(a.html_dir)}
    cfg = json.loads(a.config.read_text(encoding="utf-8"))
    new = copy.deepcopy(cfg)
    db = CardDB(cfg)
    for kind in ("evo", "hero"):
        for cid, sp in spec[kind].items():
            card = new["cards"][cid]
            card["variants"][kind] = build_variant(cid, kind, sp, pages, db.entry(cid), card["variants"].get(kind) or {},
                                                   a.as_of, db)
            v = card["variants"][kind]
            print(f"{kind:<5}{cid:<18}cycles={v.get('cycles_required', '-'):<3} mech={[m['kind'] for m in v['mechanics']]} "
                  f"overrides@11={v['level_overrides'].get('11', {})}")
    for tid, sp in spec.get("tokens", {}).items():
        new.setdefault("tokens", {})[tid] = build_token(tid, sp, pages, a.as_of)
        print(f"token {tid}")
    CardDB(new)                                             # validates mechanics, tokens, rarities
    if a.write:
        a.config.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[+] wrote {a.config}")
    else:
        print("[i] dry run; pass --write to save")
    return 0


if __name__ == "__main__":
    sys.exit(main())
