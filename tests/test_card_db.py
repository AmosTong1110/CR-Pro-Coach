import copy
import json
import math

import pytest

from backend_engine.simulator.card_db import DEFAULT_PATH, CardDataError, CardDB


@pytest.fixture(scope="module")
def raw():
    with open(DEFAULT_PATH, encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def db(raw):
    return CardDB(raw)


def test_loads_template(db):
    assert {"knight", "hog_rider", "fireball", "the_log", "minions"} <= set(db.cards)
    assert db.levels == [11, 12, 13, 14, 15, 16]


# Reference values = the per-level arrays in RoyaleAPI's decoded game data (2023-10-18).
# They pin the *mechanism*: if these break, the scaling code is wrong, not the balance data.
@pytest.mark.parametrize("card,level,field,expected", [
    ("knight", 11, "hitpoints", 1766), ("knight", 16, "hitpoints", 2822),
    ("knight", 11, "damage", 202),     ("knight", 16, "damage", 323),
    ("hog_rider", 11, "hitpoints", 1696), ("hog_rider", 11, "damage", 318),
    ("musketeer", 11, "damage", 218),
    ("fireball", 11, "damage", 689), ("fireball", 11, "crown_tower_damage", 207),
    ("the_log", 11, "damage", 290), ("the_log", 11, "crown_tower_damage", 58),
    ("skeletons", 11, "hitpoints", 81),
])
def test_exact_official_values(db, card, level, field, expected):
    s = db.stats(card, level)
    assert s.stats[field] == expected
    assert s.exact


def test_formula_matches_definition(db, raw):
    m = raw["level_scaling"]["multiplier_pct"]
    base = raw["cards"]["hog_rider"]["base_stats"]["hitpoints"]  # rare: offset 2
    for lvl in range(11, 17):
        assert db.stats("hog_rider", lvl).hitpoints == math.floor(base * m[lvl - 1 - 2] / 100)


def test_rarity_affects_scaling(db):
    # same L11 number scales differently by rarity => naive x1.1^5 is not exact
    assert db.stats("knight", 16).hitpoints != math.floor(1766 * 1.1 ** 5)


def test_override_wins(raw):
    r = copy.deepcopy(raw)
    r["cards"]["knight"]["level_overrides"] = {"16": {"hitpoints": 2900}}
    db = CardDB(r)
    assert db.stats("knight", 16).hitpoints == 2900
    assert db.stats("knight", 16).damage == 323


def test_stats_l11_fallback_is_approximate(raw):
    r = copy.deepcopy(raw)
    c = r["cards"]["knight"]
    c.pop("base_stats")
    c["stats_l11"] = {"hitpoints": 1766, "damage": 202}
    s = CardDB(r).stats("knight", 16)
    assert not s.exact and abs(s.hitpoints - 2822) <= 2


def test_non_scaled_fields_untouched(raw):
    r = copy.deepcopy(raw)
    r["cards"]["knight"]["base_stats"]["count_bonus"] = 3
    assert CardDB(r).stats("knight", 16).stats["count_bonus"] == 3


def test_towers_scale(db, raw):
    assert db.tower_stats("princess_tower", 16)["hitpoints"] > raw["towers"]["princess_tower"]["stats_l11"]["hitpoints"]


def test_unsupported_level(db):
    with pytest.raises(CardDataError):
        db.stats("knight", 17)


def test_hero_rarity_offset_unknown_is_loud(raw):
    r = copy.deepcopy(raw)
    r["cards"]["knight"]["rarity"] = "hero"
    with pytest.raises(CardDataError):
        CardDB(r).stats("knight", 11)


@pytest.mark.parametrize("mutate", [
    lambda r: r.pop("cards"),
    lambda r: r.__setitem__("schema_version", 2),
    lambda r: r["cards"]["knight"].pop("base_stats"),
    lambda r: r["cards"]["knight"].__setitem__("type", "potato"),
    lambda r: r["cards"]["knight"].__setitem__("rarity", "mythic"),
    lambda r: r["cards"]["knight"]["base_stats"].pop("hitpoints"),
    lambda r: r["cards"]["knight"].__setitem__("elixir", 11),
])
def test_validation(raw, mutate):
    r = copy.deepcopy(raw)
    mutate(r)
    with pytest.raises(CardDataError):
        CardDB(r)


def test_data_is_stamped_with_age(raw):
    for cid, c in raw["cards"].items():
        assert c.get("data_as_of"), cid
    assert "knight" in CardDB(raw).unverified()
