"""sync_meta's saved-wiki-page adapter: table parsing, title classification, L11/L16 verification."""
import json
from pathlib import Path

import pytest

from backend_engine.simulator.card_db import DEFAULT_PATH, CardDB
from backend_engine.simulator.sync_meta import (classify_title, load_wiki_dir, parse_wiki_html, verify_wiki,
                                                wiki_number)

PAGE = """<html><script>var x = "<table id='unit-statistics-table'>";</script><body>
<table id="unit-attributes-table"><tbody><tr><th>Cost</th><th>Hit Speed</th></tr><tr><td>3</td><td>1.2 sec</td></tr></tbody></table>
<table id="unit-statistics-table"><tbody>
<tr><th>Level<br><span><img alt="Level"></span></th><th>Hitpoints</th><th>Damage</th><th>Damage Per Second</th></tr>
<tr><td>11</td><td>{hp11}</td><td>202</td><td>168</td></tr>
<tr><td>16</td><td>2,844</td><td>325</td><td>270</td></tr></tbody></table></body></html>"""
NAME = "{} _ Clash Royale Wiki _ Fandom.html"


def write(folder: Path, title: str, hp11="1,766") -> Path:
    p = folder / NAME.format(title)
    p.write_text(PAGE.format(hp11=hp11), encoding="utf-8")
    return p


@pytest.fixture(scope="module")
def db():
    return CardDB(json.loads(DEFAULT_PATH.read_text(encoding="utf-8")))


def test_wiki_number():
    assert wiki_number("1,766") == 1766 and wiki_number("1.2 sec") == 1.2
    assert wiki_number("Ground") is None and wiki_number("") is None


@pytest.mark.parametrize("raw,card,variant", [
    ("Knight", "knight", "base"), ("Evolved Knight", "knight", "evo"), ("Knight_Hero", "knight", "hero"),
    ("Heroic Mini P.E.K.K.A.", "mini_pekka", "hero"), ("The Log", "the_log", "base"),
    ("Evolved Skeleton Army", "skeleton_army", "evo")])
def test_classify_title(raw, card, variant):
    _, cid, var = classify_title(raw)
    assert (cid, var) == (card, variant)


def test_parse_page(tmp_path):
    pg = parse_wiki_html(write(tmp_path, "Knight"))
    assert (pg.card_id, pg.variant) == ("knight", "base")
    assert pg.attrs == {"Cost": "3", "Hit Speed": "1.2 sec"}
    assert pg.levels[11] == {"Hitpoints": 1766, "Damage": 202, "Damage Per Second": 168}
    assert pg.levels[16]["Hitpoints"] == 2844        # thousands separator handled; <script> text ignored


def test_verify_flags_difference_and_unknown_card(tmp_path, db):
    write(tmp_path, "Knight", hp11="1,800")
    write(tmp_path, "Evolved Knight")                # variants are not compared
    write(tmp_path, "Golem", hp11="5,000")           # not in the DB
    rows = verify_wiki(db, load_wiki_dir(tmp_path))
    by = {(r["card"], r["stat"]): r for r in rows}
    assert by["knight", "hitpoints"]["status"] == "DIFF"     # L11: page says 1800, the DB has 1766
    assert by["knight", "hitpoints"]["wiki"] == {11: 1800, 16: 2844} and by["knight", "hitpoints"]["ours"] == {11: 1766, 16: 2844}
    assert by["knight", "damage"]["status"] == "ok"
    assert by["golem", "-"]["status"] == "not simulated yet"
    assert not any(r["card"] == "knight" and r["status"] == "not simulated yet" for r in rows)


REAL = Path(__file__).resolve().parents[1] / "data" / "raw_sources" / "cards"


@pytest.mark.skipif(not REAL.is_dir(), reason="saved wiki pages not present")
def test_real_pages_cover_the_30_card_batch():
    pages = {p.card_id for p in load_wiki_dir(REAL) if p.variant == "base"}
    batch = ("skeletons barbarian_barrel fireball the_log electro_spirit royal_ghost ice_spirit knight cannon mini_pekka "
             "tornado ice_golem musketeer hog_rider mighty_miner firecracker earthquake lightning tesla golden_knight "
             "vines dart_goblin valkyrie skeleton_army rocket miner royal_delivery goblin_demolisher fire_spirit "
             "dark_prince").split()
    assert not [c for c in batch if c not in pages]
    knight = next(p for p in load_wiki_dir(REAL) if p.card_id == "knight" and p.variant == "base")
    assert knight.levels[11]["Hitpoints"] == 1766 and knight.levels[16]["Damage"] == 325


@pytest.mark.skipif(not REAL.is_dir(), reason="saved wiki pages not present")
def test_config_matches_every_compared_wiki_stat(db):
    """After `--apply-wiki`, every headline stat of every simulated card equals the wiki at L11 and L16."""
    rows = verify_wiki(db, load_wiki_dir(REAL), levels=(11, 12, 13, 14, 15, 16))
    assert [r for r in rows if r["status"] == "DIFF"] == []
    assert sum(r["status"] == "ok" for r in rows) >= 40
