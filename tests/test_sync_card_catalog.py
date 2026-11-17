"""tools/sync_card_catalog.py: balance patches, new cards (auto-enable / needs_hook / identity only), Tower Troops,
conflicts, idempotence. Pure planning only (no network, no writes outside tmp_path)."""
import copy
import json
from pathlib import Path

import pytest

from backend_engine.simulator.card_db import CardDB
from tools.sync_card_catalog import Incoming, main, parse_source, plan_sync

ROOT = Path(__file__).resolve().parents[1]
CFG = ROOT / "configs" / "cards_stats_v2026.json"
OFFICIAL = ROOT / "data" / "raw_sources" / "official_cards.json"


@pytest.fixture(scope="module")
def raw():
    return json.loads(CFG.read_text(encoding="utf-8"))


def _snapshot(**cards):
    return {"format": "crbot_catalog_snapshot_v1", "as_of": "2027-01-01", "cards": cards}


def test_parse_the_three_source_formats():
    official = parse_source({"items": [{"name": "Knight", "id": 26000000, "elixirCost": 3, "rarity": "common"}],
                             "supportItems": [{"name": "Cannoneer", "id": 159000001, "rarity": "epic"}]})
    assert [(i.name, i.kind) for i in official] == [("Knight", "troop"), ("Cannoneer", "tower")]
    royaleapi = parse_source([{"key": "royal-giant", "name": "Royal Giant", "id": 26000024, "elixir": 6,
                               "type": "Troop", "rarity": "Common"}])
    assert royaleapi[0].key == "royal-giant" and royaleapi[0].kind == "troop" and royaleapi[0].rarity == "common"
    snap = parse_source(_snapshot(knight={"name": "Knight", "stats": {"11": {"hitpoints": 1800}}}))
    assert snap[0].stats == {"11": {"hitpoints": 1800}}
    with pytest.raises(ValueError):
        parse_source({"what": 1})


def test_new_cards_are_enabled_only_when_complete_and_standard(raw):
    royal_giant = {"name": "Royal Giant", "game_id": 26000024, "type": "troop", "rarity": "common", "elixir": 6,
                   "stats": {"11": {"hitpoints": 3000, "damage": 254}},
                   "attack": {"targets": ["buildings"], "range_tiles": 5.0, "hit_speed_s": 1.7, "first_hit_s": 0.8,
                              "splash_radius_tiles": 0.0, "sight_tiles": 7.5},
                   "movement": {"layer": "ground", "speed_tiles_s": 1.0}, "mechanics": [{"kind": "building_targeting",
                                                                                         "params": {}}],
                   "count": 1, "deploy_time_s": 1.0, "collision_radius_tiles": 0.75, "mass": 18}
    novel = {"name": "Super Future Card 2027", "game_id": 26000999, "type": "troop", "rarity": "legendary",
             "elixir": 5, "stats": {"11": {"hitpoints": 1500, "damage": 300}}, "attack": royal_giant["attack"],
             "movement": royal_giant["movement"], "mechanics": [{"kind": "time_rewind", "params": {}}]}
    bare = {"name": "Bare Card", "game_id": 28000999, "type": "spell", "rarity": "rare", "elixir": 2}
    new, rep = plan_sync(raw, parse_source(_snapshot(royal_giant=royal_giant, super_future_card_2027=novel,
                                                     bare_card=bare)), "2027-01-01")
    status = {**dict(rep.added), **dict(rep.updated)}
    # Royal Giant is already in the catalog (identity only, from the official list): the stats complete it
    assert "-> enabled" in status["royal_giant"] or status["royal_giant"].startswith("enabled")
    assert status["super_future_card_2027"].startswith("needs_hook") and "time_rewind" in status["super_future_card_2027"]
    assert status["bare_card"].startswith("disabled")
    db = CardDB(new)
    assert "royal_giant" in db.cards and db.stats("royal_giant", 11).hitpoints == 3000
    assert db.quarantined()["super_future_card_2027"] == "needs_hook" and db.quarantined()["bare_card"] == "disabled"


def test_balance_patch_writes_level_11_overrides(raw):
    old = CardDB(copy.deepcopy(raw)).stats("knight", 11).hitpoints
    snap = _snapshot(knight={"name": "Knight", "game_id": 26000000, "elixir": 3, "stats": {"11": {"hitpoints": old + 50}}})
    new, rep = plan_sync(raw, parse_source(snap), "2027-01-01")
    assert rep.updated and rep.updated[0][0] == "knight" and f"hitpoints L11 {old}->{old + 50}" in rep.updated[0][1]
    assert new["cards"]["knight"]["stats_patched_as_of"] == "2027-01-01"
    assert CardDB(new).stats("knight", 11).hitpoints == old + 50
    again, rep2 = plan_sync(new, parse_source(snap), "2027-01-02")     # same patch twice: nothing to do
    assert not rep2.updated and rep2.unchanged == 1


def test_game_id_conflicts_are_reported_not_applied(raw):
    inc = [Incoming("Sir Knight", "troop", game_id=raw["cards"]["knight"]["game_id"], elixir=3)]
    new, rep = plan_sync(raw, inc, "2027-01-01")
    assert rep.conflicts == [] and rep.unchanged == 1           # a name the catalog doesn't know: the id wins
    inc = [Incoming("Musketeer", "troop", game_id=raw["cards"]["knight"]["game_id"], elixir=4)]
    new, rep = plan_sync(raw, inc, "2027-01-01")
    assert len(rep.conflicts) == 1 and new["cards"]["knight"] == raw["cards"]["knight"]


def test_manually_disabled_cards_are_never_switched_on(raw):
    r = copy.deepcopy(raw)
    r["cards"]["knight"]["enabled"] = False                       # a person turned it off: no catalog_status
    snap = _snapshot(knight={"name": "Knight", "game_id": 26000000, "stats": {"11": {"hitpoints": 2000}}})
    new, _ = plan_sync(r, parse_source(snap), "2027-01-01")
    assert new["cards"]["knight"]["enabled"] is False and "knight" not in CardDB(new).cards


def test_tower_troops_stay_disabled_until_they_have_stats(raw):
    inc = parse_source({"format": "crbot_catalog_snapshot_v1", "towers": {
        "cannoneer": {"name": "Cannoneer", "game_id": 159000001, "rarity": "epic",
                      "stats": {"11": {"hitpoints": 2616, "damage": 386}},
                      "attack": {"targets": ["ground", "air"], "range_tiles": 7.5, "hit_speed_s": 2.1,
                                 "projectile_speed_tiles_s": 10.0}},
        "royal_chef": {"name": "Royal Chef", "game_id": 159000004, "rarity": "legendary"}}})
    new, rep = plan_sync(raw, inc, "2027-01-01")
    assert new["towers"]["cannoneer"]["enabled"] is True and new["towers"]["royal_chef"]["enabled"] is False
    assert CardDB(new).tower_stats("cannoneer", 11)["hitpoints"] == 2616


@pytest.mark.skipif(not OFFICIAL.exists(), reason="needs data/raw_sources/official_cards.json")
def test_committed_catalog_is_in_sync_with_the_official_list(raw):
    new, rep = plan_sync(raw, parse_source(json.loads(OFFICIAL.read_text())), "2027-01-01")
    assert not rep.added and not rep.updated and not rep.conflicts
    db = CardDB(raw)
    assert len(db.catalog) >= 123 and len(db.cards) == 47


def test_cli_dry_run_writes_nothing_and_write_validates(tmp_path, raw):
    cfg = tmp_path / "cards.json"
    text = json.dumps(raw, indent=2, ensure_ascii=False)
    cfg.write_text(text, encoding="utf-8")
    snap = tmp_path / "snap.json"
    snap.write_text(json.dumps(_snapshot(new_spell={"name": "New Spell", "game_id": 28000998, "type": "spell",
                                                    "rarity": "epic", "elixir": 3})))
    assert main(["--snapshot", str(snap), "--config", str(cfg)]) == 0
    assert cfg.read_text(encoding="utf-8") == text
    assert main(["--snapshot", str(snap), "--config", str(cfg), "--write", "--as-of", "2027-01-01"]) == 0
    assert "new_spell" in CardDB.load(cfg).catalog
