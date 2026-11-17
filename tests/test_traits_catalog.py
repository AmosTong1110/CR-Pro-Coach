"""Trait registry + catalog flags (backend_engine/simulator/traits.py, CardDB.catalog / quarantined)."""
import copy

import pytest

from backend_engine.simulator import traits as T
from backend_engine.simulator.card_db import CardDataError, CardDB
from backend_engine.simulator.engine import Match, MatchConfig
from backend_engine.simulator.mechanics.core import knockback
from tests.test_mechanics import play


@pytest.fixture(scope="module")
def raw():
    return copy.deepcopy(CardDB.load().raw)


def test_traits_come_from_data_not_names():
    db = CardDB.load()
    assert {"death_spawn", "spawn_wave", "lifetime_decay"} <= db.traits("tombstone")
    assert {"heavy_knockback_immune"} <= db.traits("pekka") and "heavy_knockback_immune" not in db.traits("knight")
    assert "building_only" in db.traits("hog_rider") and "charge_jump" in db.traits("hog_rider")
    assert "dot_spell" in db.traits("poison") and "shield_hp" in db.traits("dark_prince")
    assert "air" in db.traits("minions") and "targets_air" in db.traits("musketeer")
    assert set(T.TRAITS) >= set().union(*(db.traits(c) for c in db.cards))


def test_heavy_immunity_and_log_exception_are_data(raw):
    db = CardDB.load()
    deck = ("pekka", "knight", "archers", "fireball", "the_log", "skeletons", "cannon", "musketeer")
    m = Match(MatchConfig((deck, deck), shuffle=False, collisions=False), db)
    (pk,) = play(m, 0, "pekka", 9.5, 8.0)
    x, y = pk.x, pk.y
    knockback(m, pk, 0.0, 1.0, 1.0)
    assert (pk.x, pk.y) == (x, y)                                 # trait: immune
    knockback(m, pk, 0.0, 1.0, 1.0, ignores_heavy=True)           # The Log's knockback params say ignores_heavy
    assert pk.y == pytest.approx(y + 1.0)
    kb = [mm for mm in db.cards["the_log"]["mechanics"] if mm["kind"] == "knockback"][0]["params"]
    assert kb["ignores_heavy"] is True


def test_disabled_and_needs_hook_cards_stay_in_the_catalog_only(raw):
    r = copy.deepcopy(raw)
    r["cards"]["knight"]["enabled"] = False
    r["cards"]["future_card"] = {"name": "Future Card", "type": "troop", "rarity": "epic", "elixir": 5,
                                 "needs_hook": True, "mechanics": [{"kind": "time_warp", "params": {}}]}
    db = CardDB(r)                                        # an unknown mechanic is fine on a quarantined card
    assert "knight" in db.catalog and "knight" not in db.cards
    assert "future_card" in db.catalog and "future_card" not in db.cards
    assert db.quarantined() == {"knight": "disabled", "future_card": "needs_hook"}
    with pytest.raises(KeyError):
        db.entry("future_card")                           # nothing that plays can reach it


def test_enabled_card_with_an_unknown_mechanic_is_still_an_error(raw):
    r = copy.deepcopy(raw)
    r["cards"]["knight"]["mechanics"].append({"kind": "time_warp", "params": {}})
    with pytest.raises(CardDataError):
        CardDB(r)


def test_planned_mechanic_keeps_a_card_out_of_play(raw):
    r = copy.deepcopy(raw)
    r["cards"]["knight"]["mechanics"].append({"kind": "evo_sniper", "params": {}})   # known but not simulated
    db = CardDB(r)
    assert "knight" not in db.cards and db.quarantined()["knight"].startswith("unsupported mechanics")
