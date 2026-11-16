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

Second source (2026-11): `--html-dir data/raw_sources/cards` reads saved Clash Royale Wiki (Fandom) card pages
and checks their level-11 / level-16 numbers against the cards we simulate (read-only, nothing is written):

    python -m backend_engine.simulator.sync_meta --html-dir data/raw_sources/cards
    python -m backend_engine.simulator.sync_meta --html-dir data/raw_sources/cards --dump-json wiki_stats.json
    # store the wiki's exact L11-L16 numbers (hitpoints, damage, death / crown-tower damage) as level_overrides
    python -m backend_engine.simulator.sync_meta --html-dir data/raw_sources/cards --apply-wiki --as-of 2026-11-16 --write

Source freshness matters: RoyaleAPI/cr-api-data was last updated 2023-10-18, so its numbers
predate later balance changes. The importer stamps `data_as_of` on every card so stale
data is visible; set `verified: true` by hand only after checking the current in-game card.
"""
from __future__ import annotations

import argparse
import copy
import json
import re
from dataclasses import dataclass, field
from html.parser import HTMLParser
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend_engine.simulator.card_db import DEFAULT_PATH, SUPPORTED_SCHEMAS, CardDB

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
        if body.get("death_damage"):
            base["death_damage"] = body["death_damage"]
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
        if proj is None and rec.get("radius") and rec.get("damage"):
            proj = rec    # instant area effect with no projectile (Zap): the spell record holds damage and radius
        if proj is None:
            raise KeyError(f"{ref}: spell needs a projectile record")
        base["damage"] = proj["damage"]
        if proj.get("crown_tower_damage_percent") is not None:
            out["crown_tower_damage_pct"] = proj["crown_tower_damage_percent"]
        sp = out.setdefault("spell", {})
        if proj.get("speed"):                 # no speed = lands at once
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
    before = CardDB(cfg) if cfg.get("schema_version") in SUPPORTED_SCHEMAS else None
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


# ----------------------------------------------------------------------------------------------
# Source 2: saved Clash Royale Wiki (Fandom) card pages
# ----------------------------------------------------------------------------------------------
class _TableParser(HTMLParser):
    """Collects the text of every top-level <table> (nested tables are folded into their cell)."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.tables: List[Dict[str, Any]] = []
        self._tab: Optional[Dict[str, Any]] = None
        self._row: Optional[List[str]] = None
        self._cell: Optional[List[str]] = None
        self._depth = 0
        self._skip = 0                      # inside <script>/<style>

    def handle_starttag(self, tag: str, attrs) -> None:
        a = dict(attrs)
        if tag in ("script", "style"):
            self._skip += 1
        elif tag == "table":
            self._depth += 1
            if self._depth == 1:
                self._tab = {"id": a.get("id"), "class": a.get("class") or "", "rows": []}
        elif self._tab is not None and self._depth == 1:
            if tag == "tr":
                self._row = []
            elif tag in ("td", "th") and self._row is not None:
                self._cell = []

    def handle_endtag(self, tag: str) -> None:
        if tag in ("script", "style"):
            self._skip = max(0, self._skip - 1)
        elif tag == "table":
            if self._depth == 1 and self._tab is not None:
                self.tables.append(self._tab)
                self._tab = None
            self._depth = max(0, self._depth - 1)
        elif self._tab is not None and self._depth == 1:
            if tag in ("td", "th") and self._cell is not None and self._row is not None:
                self._row.append(" ".join("".join(self._cell).split()))
                self._cell = None
            elif tag == "tr" and self._row is not None:
                self._tab["rows"].append(self._row)
                self._row = None

    def handle_data(self, data: str) -> None:
        if self._cell is not None and not self._skip:
            self._cell.append(data)


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?")


def wiki_number(text: str) -> Optional[float]:
    """First number in a cell ("1,766" -> 1766, "1.2 sec" -> 1.2); None for empty or non-numeric cells."""
    m = _NUM.search(text or "")
    return float(m.group(0).replace(",", "")) if m else None


@dataclass
class WikiPage:
    title: str                                   # "Evolved Knight"
    card_id: str                                 # snake_case id of the base card ("knight")
    variant: str                                 # base | evo | hero
    attrs: Dict[str, str] = field(default_factory=dict)             # first attributes table (Cost, Hit Speed, ...)
    tables: List[Dict[str, str]] = field(default_factory=list)      # every unit-attributes table in page order
    totals: Dict[int, Dict[str, float]] = field(default_factory=dict)   # level -> {column: N} for cells "64 x5 (320)"
    levels: Dict[int, Dict[str, float]] = field(default_factory=dict)   # level -> {column header: value}
    columns: List[str] = field(default_factory=list)


def slug(title: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", title.replace(".", "").lower()).strip("_")


def classify_title(raw_title: str) -> Tuple[str, str, str]:
    """File title -> (display title, base card id, variant): 'Knight_Hero' / 'Heroic Musketeer' -> hero,
    'Evolved Knight' -> evo."""
    variant, t = "base", raw_title.strip()
    if t.endswith("_Hero"):
        variant, t = "hero", t[: -len("_Hero")]
    elif t.startswith("Heroic "):
        variant, t = "hero", t[len("Heroic "):]
    elif t.startswith("Evolved "):
        variant, t = "evo", t[len("Evolved "):]
    return raw_title.strip(), slug(t), variant


def parse_wiki_html(path: Path) -> WikiPage:
    """Parse one saved wiki page: the `unit-attributes-table` (first one) and `unit-statistics-table`."""
    raw_title = path.name.split(" _ Clash Royale Wiki")[0]
    title, cid, variant = classify_title(raw_title)
    parser = _TableParser()
    parser.feed(path.read_text(encoding="utf-8", errors="replace"))
    page = WikiPage(title, cid, variant)
    for t in parser.tables:
        rows = [r for r in t["rows"] if r]
        if len(rows) < 2:
            continue
        if (t["id"] or "").startswith("unit-attributes-table"):
            page.tables.append(dict(zip(rows[0], rows[1])))
            if not page.attrs:
                page.attrs = page.tables[0]
        elif t["id"] == "unit-statistics-table" and not page.levels:
            page.columns = rows[0][1:]
            for r in rows[1:]:
                lvl = wiki_number(r[0])
                if lvl is None or len(r) < 2:
                    continue
                page.levels[int(lvl)] = {h: v for h, c in zip(rows[0][1:], r[1:])
                                         if (v := wiki_number(c)) is not None}
                for h, c in zip(rows[0][1:], r[1:]):         # "64 x5 (320)": keep the total as well
                    mt = re.search(r"\(\s*(\d[\d,]*)\s*\)\s*$", c)
                    if mt:
                        page.totals.setdefault(int(lvl), {})[h] = float(mt.group(1).replace(",", ""))
    return page


def load_wiki_dir(folder: Path) -> List[WikiPage]:
    return [parse_wiki_html(p) for p in sorted(folder.glob("*.html"))]


# wiki column -> our stat key. Only the headline columns are compared; anything else on the page
# (ability numbers, token stats, spark / pulse damage) is listed in the dump for the card batches that need it.
_COLUMN_TO_STAT = {"hitpoints": "hitpoints", "damage": "damage", "area damage": "damage",
                   "death damage": "death_damage", "crown tower damage": "crown_tower_damage"}
# columns of a spell page that describe the token it spawns (Barbarian Barrel's barbarian, Royal Delivery's recruit)
_SPAWN_COLUMNS = {"Hitpoints", "Damage"}


def _our_stat_for(column: str, card: Dict[str, Any]) -> Optional[str]:
    key = _COLUMN_TO_STAT.get(column.strip().lower())
    if key is None:
        # "Barbarian Barrel Area Damage" / "Royal Delivery Area Damage" = the spell's own damage
        if column.strip().lower().endswith("area damage") and card.get("type") == "spell":
            return "damage"
    return key


def headline_columns(pg: WikiPage, entry: Dict[str, Any]) -> List[Tuple[str, str, bool]]:
    """(wiki column, our stat key, is_token) for the columns we model. Spell pages that also list the unit the
    spell spawns (Barbarian Barrel) have their Hitpoints / Damage columns mapped to that token."""
    spawns = entry.get("type") == "spell" and any(m["kind"] in _SPAWNERS for m in entry.get("mechanics", []))
    out, seen = [], set()
    for col in pg.columns:
        stat = _our_stat_for(col, entry)
        if stat is None:
            continue
        token = spawns and col in _SPAWN_COLUMNS
        if (stat, token) in seen:
            continue
        seen.add((stat, token))
        out.append((col, stat, token))
    return out


_SPAWNERS = ("spawn_on_end", "delayed_drop")       # spell mechanics that put a token unit on the field


def _token_id(entry: Dict[str, Any]) -> Optional[str]:
    return next((m["params"]["token"] for m in entry.get("mechanics", []) if m["kind"] in _SPAWNERS), None)


def verify_wiki(db: CardDB, pages: List[WikiPage], levels: Tuple[int, ...] = (11, 16)) -> List[Dict[str, Any]]:
    """Compare base-card pages with the DB. Returns one row per (card, stat) with wiki/ours per level."""
    rows: List[Dict[str, Any]] = []
    for pg in pages:
        if pg.variant != "base":
            continue
        if pg.card_id not in db.cards:
            rows.append({"card": pg.card_id, "stat": "-", "status": "not simulated yet",
                         "wiki": {L: pg.levels.get(L, {}) for L in levels}})
            continue
        entry = db.entry(pg.card_id)
        cols = headline_columns(pg, entry)
        for col, stat, token in cols:
            holder = _token_id(entry) if token else pg.card_id
            ours = {L: db.stats(holder, L).stats for L in levels}
            wiki = {L: pg.levels.get(L, {}).get(col) for L in levels}
            if all(v is None for v in wiki.values()):
                continue
            mine = {L: ours[L].get(stat) for L in levels}
            tol = 1 if stat == "crown_tower_damage" else 0     # a percentage-derived value may be off by a rounding step
            tot = {L: pg.totals.get(L, {}).get(col) for L in levels}      # "64 x5 (320)": we may hold the total
            ok = all(wiki[L] is None or mine[L] is not None and (abs(mine[L] - wiki[L]) <= tol or mine[L] == tot[L])
                     for L in levels)
            rows.append({"card": pg.card_id + ("/" + holder if token else ""), "stat": stat, "wiki": wiki,
                         "ours": mine, "status": "ok" if ok else "DIFF", "data_as_of": entry.get("data_as_of")})
        if not cols:
            rows.append({"card": pg.card_id, "stat": "-", "status": "no comparable column", "wiki": {}})
    return rows


def apply_wiki(cfg: Dict[str, Any], pages: List[WikiPage], as_of: str,
               levels: Tuple[int, ...] = (11, 12, 13, 14, 15, 16)) -> Dict[str, Any]:
    """Return a copy of `cfg` whose cards carry the wiki's exact per-level numbers as `level_overrides`
    (hitpoints, damage, death damage, crown-tower damage). The wiki's tables are the game's own rounded values,
    which no single multiplier table reproduces, so they are stored as they are. Base stats stay for reference."""
    new = copy.deepcopy(cfg)
    db = CardDB(cfg)
    for pg in pages:
        if pg.variant != "base" or pg.card_id not in db.cards:
            continue
        entry = db.entry(pg.card_id)
        for col, stat, token in headline_columns(pg, entry):
            holder = new["tokens"][_token_id(entry)] if token else new["cards"][pg.card_id]
            for L in levels:
                v = pg.levels.get(L, {}).get(col)
                if v is not None:
                    holder.setdefault("level_overrides", {}).setdefault(str(L), {})[stat] = int(v)
            holder["wiki_as_of"] = as_of
        src = new["cards"][pg.card_id].get("data_source", "")
        if "wiki-fandom-html" not in src:
            new["cards"][pg.card_id]["data_source"] = f"{src} + wiki-fandom-html:{pg.title}".strip()
    return new


def print_verification(rows: List[Dict[str, Any]], pages: List[WikiPage], levels=(11, 16)) -> Dict[str, int]:
    fmt = lambda v: "-" if v is None else (f"{v:g}" if isinstance(v, float) else str(v))   # noqa: E731
    print(f"{'card':<18}{'stat':<20}" + "".join(f"{'wiki L%d' % L:>9}{'ours L%d' % L:>9}" for L in levels) + "  status")
    counts = {"ok": 0, "DIFF": 0, "not simulated yet": 0, "no comparable column": 0}
    for r in rows:
        counts[r["status"]] = counts.get(r["status"], 0) + 1
        if r["status"] in ("ok", "DIFF"):
            cells = "".join(f"{fmt(r['wiki'][L]):>9}{fmt(r['ours'][L]):>9}" for L in levels)
            print(f"{r['card']:<18}{r['stat']:<20}{cells}  {r['status']}")
    print("\n-- on the wiki, not simulated yet (wiki headline numbers, no comparison possible) --")
    for r in rows:
        if r["status"] == "not simulated yet":
            head = {L: {k: v for k, v in r["wiki"][L].items() if k.split()[-1] in ("Hitpoints", "Damage")}
                    for L in levels}
            print(f"{r['card']:<18}" + "  ".join(f"L{L}: {head[L]}" for L in levels))
    variants = [p for p in pages if p.variant != "base"]
    print(f"\n[i] {len(pages)} pages ({len(pages) - len(variants)} base, {len(variants)} evo/hero); "
          f"{counts['ok']} stats match, {counts['DIFF']} differ, {counts['not simulated yet']} cards not simulated yet")
    return counts


# ---- attributes: cost, timings, range, speed, count, targets -------------------------------------------
def attribute_fields(pg: WikiPage, entry: Dict[str, Any], attrs: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """Wiki attribute table -> {dotted config path: value}. Only fields a card of that type has are produced.
    Speeds / projectile speeds are game units / 60 (tiles per second), the same convention as the importer."""
    a, out = (pg.attrs if attrs is None else attrs), {}
    num = lambda k: wiki_number(a.get(k, ""))                        # noqa: E731
    out["elixir"] = num("Cost")
    spell = entry.get("type") == "spell"
    if spell:
        out["spell.radius_tiles"] = num("Radius") if "spell" in entry and "radius_tiles" in entry["spell"] else None
        if "duration_ms" in entry.get("spell", {}) and num("Duration") is not None:
            out["spell.duration_ms"] = round(num("Duration") * 1000)
        if "width_tiles" in entry.get("spell", {}):
            out["spell.width_tiles"] = num("Width")
        if "rolling_range_tiles" in entry.get("spell", {}):
            out["spell.rolling_range_tiles"] = num("Range")
        if "travel_speed_tiles_s" in entry.get("spell", {}) and num("Projectile Speed"):
            out["spell.travel_speed_tiles_s"] = num("Projectile Speed") / SPEED_UNIT
    else:
        out["attack.hit_speed_s"] = num("Hit Speed")
        out["attack.first_hit_s"] = num("First Hit Speed")
        out["attack.range_tiles"] = num("Range")
        out["attack.splash_radius_tiles"] = num("Splash Radius")
        out["deploy_time_s"] = num("Deploy Time")
        if entry.get("type") == "building":
            out["lifetime_s"] = num("Lifetime")
        else:
            out["movement.speed_tiles_s"] = num("Speed") / SPEED_UNIT if num("Speed") is not None else None
            out["count"] = num("Count")
        if num("Projectile Speed"):
            out["attack.projectile_speed_tiles_s"] = num("Projectile Speed") / SPEED_UNIT
    return {k: v for k, v in out.items() if v is not None}


def _get(d: Dict[str, Any], path: str) -> Any:
    for k in path.split("."):
        if not isinstance(d, dict) or k not in d:
            return None
        d = d[k]
    return d


def verify_attributes(db: CardDB, pages: List[WikiPage]) -> List[Dict[str, Any]]:
    rows = []
    for pg in pages:
        if pg.variant != "base" or pg.card_id not in db.cards:
            continue
        entry = db.entry(pg.card_id)
        for path, wiki in attribute_fields(pg, entry).items():
            ours = _get(entry, path)
            if path == "attack.splash_radius_tiles" and not wiki and not ours:
                continue
            ok = ours is not None and abs(float(ours) - float(wiki)) < 1e-6 + 0.011 * abs(float(wiki)) * 0
            rows.append({"card": pg.card_id, "field": path, "wiki": wiki, "ours": ours,
                         "status": "ok" if ok else "DIFF"})
    return rows


def apply_attributes(cfg: Dict[str, Any], pages: List[WikiPage], rows: List[Dict[str, Any]]) -> None:
    """Write every DIFF row's wiki value into `cfg` (in place)."""
    for r in rows:
        if r["status"] != "DIFF":
            continue
        d = cfg["cards"][r["card"]]
        *parents, leaf = r["field"].split(".")
        for k in parents:
            d = d.setdefault(k, {})
        v = r["wiki"]
        d[leaf] = int(v) if leaf in ("elixir", "count", "duration_ms") else v


def print_attribute_rows(rows: List[Dict[str, Any]]) -> int:
    diffs = [r for r in rows if r["status"] == "DIFF"]
    for r in diffs:
        print(f"{r['card']:<18}{r['field']:<34}wiki {r['wiki']:<8g} ours {r['ours']}")
    print(f"[i] {len(rows)} attributes compared, {len(diffs)} differ")
    return len(diffs)


def dump_wiki(pages: List[WikiPage], out: Path) -> None:
    data = {f"{p.card_id}#{p.variant}": {"title": p.title, "attributes": p.attrs,
                                           "levels": {str(L): v for L, v in sorted(p.levels.items())}}
            for p in pages}
    out.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"[+] wrote {out} ({len(data)} pages)")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--source", type=Path, help="RoyaleAPI-format cards_stats.json")
    ap.add_argument("--html-dir", type=Path, help="folder of saved Clash Royale Wiki card pages (verify L11/L16)")
    ap.add_argument("--dump-json", type=Path, help="with --html-dir: write every parsed page to this JSON file")
    ap.add_argument("--config", type=Path, default=DEFAULT_PATH)
    ap.add_argument("--as-of", help="date the source data reflects, e.g. 2023-10-18")
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--attrs", action="store_true", help="with --html-dir: compare cost / timings / range / speed instead")
    ap.add_argument("--apply-wiki", action="store_true",
                    help="with --html-dir, --as-of and --write: store the wiki's L11-L16 numbers as level_overrides")
    a = ap.parse_args()
    if a.html_dir and a.apply_wiki:
        if not (a.as_of and a.write):
            ap.error("--apply-wiki needs --as-of and --write")
        cfg = json.loads(a.config.read_text(encoding="utf-8"))
        pages = load_wiki_dir(a.html_dir)
        new = apply_wiki(cfg, pages, a.as_of)
        apply_attributes(new, pages, verify_attributes(CardDB(new), pages))
        a.config.write_text(json.dumps(new, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        print(f"[+] wrote {a.config}; verify with --html-dir (no --apply-wiki)")
        return
    if a.html_dir and a.attrs:
        cfg = json.loads(a.config.read_text(encoding="utf-8"))
        print_attribute_rows(verify_attributes(CardDB(cfg), load_wiki_dir(a.html_dir)))
        return
    if a.html_dir:
        pages = load_wiki_dir(a.html_dir)
        cfg = json.loads(a.config.read_text(encoding="utf-8"))
        print_verification(verify_wiki(CardDB(cfg), pages), pages)
        if a.dump_json:
            dump_wiki(pages, a.dump_json)
        return
    if not (a.source and a.as_of):
        ap.error("--source and --as-of are required unless --html-dir is given")
    sync(a.config, a.source, a.as_of, a.write)


if __name__ == "__main__":
    main()
