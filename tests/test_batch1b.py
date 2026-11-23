"""Batch 1, second slice: tokens + spawn-on-end (Barbarian Barrel), invisibility (Royal Ghost), pull area (Tornado)."""
import copy

import pytest

from backend_engine.simulator import geometry as G
from backend_engine.simulator.card_db import CardDataError, CardDB
from backend_engine.simulator.engine import Match, MatchConfig
from backend_engine.simulator.mechanics import status as S
from tests.test_batch1 import BIG, new, run, tank
from tests.test_mechanics import play

DECK = ("barbarian_barrel", "royal_ghost", "tornado", "zap", "valkyrie", "ice_golem", "fire_spirit", "mini_pekka")


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


def mk(db):
    return new(db, a=DECK, b=DECK)         # both sides can field every card under test


# ------------------------------------------------------------------ tokens
def test_tokens_are_separate_from_cards(db):
    assert "barbarian" in db.tokens and "barbarian" not in db.cards
    assert db.entry("barbarian")["name"] == "Barbarian" and db.entry("knight")["name"] == "Knight"
    with pytest.raises(KeyError):
        db.entry("nope")
    with pytest.raises(ValueError):                          # a token can't be put in a deck
        Match(MatchConfig((tuple(DECK[:-1]) + ("barbarian",), DECK)), db)


def test_token_scales_with_the_parent_cards_rarity(db):
    # Wiki values (2026-11-16): the spawned Barbarian is at the same tournament-standard level as the barrel.
    assert db.stats("barbarian", 11).stats == {"hitpoints": 670, "damage": 191}
    assert db.stats("barbarian", 16).stats == {"hitpoints": 1079, "damage": 308}
    raw = copy.deepcopy(db.raw)                      # without the wiki overrides the parent's rarity still decides
    raw["tokens"]["barbarian"].pop("level_overrides")
    assert CardDB(raw).stats("barbarian", 11).hitpoints == 419


def test_card_naming_an_unknown_token_is_rejected(db):
    raw = copy.deepcopy(db.raw)
    raw["tokens"] = {}
    with pytest.raises(CardDataError, match="unknown token"):
        CardDB(raw)


# ------------------------------------------------------------------ Barbarian Barrel
def test_barrel_rolls_hits_and_spawns_a_barbarian_where_it_stops(db):
    m = mk(db)
    k = tank(m, 1, "mini_pekka", 9.5, 10.5)                  # on the barrel's path (it rolls 4.5 tiles from y 8.5)
    play(m, 0, "barbarian_barrel", 9.5, 8.5)
    assert not [u for u in m.units.values() if u.card_id == "barbarian"]       # nothing yet: it is still rolling
    run(m, 1500)
    assert BIG - k.hp == db.stats("barbarian_barrel", 11).stats["damage"] == 215
    (b,) = [u for u in m.units.values() if u.card_id == "barbarian"]
    assert b.side == 0 and b.hp <= b.max_hp == 670 and b.kind == "troop"
    assert b.x == pytest.approx(9.5) and b.y == pytest.approx(13.0, abs=0.6)    # 8.5 + 4.5, then it starts walking
    assert not m.rolling


def test_barbarian_deploys_for_half_a_second(db):
    m = mk(db)
    play(m, 0, "barbarian_barrel", 9.5, 8.5)
    b = None
    for _ in range(100):
        m.step()
        found = [u for u in m.units.values() if u.card_id == "barbarian"]
        if found:
            b = found[0]
            break
    assert b is not None and b.deploy_ms == 450               # data: 500 ms, minus the tick it spawned in
    x0 = b.x, b.y
    run(m, 400)
    assert b.deploy_ms == 50 and (b.x, b.y) == x0            # still deploying: it does not walk yet
    run(m, 100)
    assert b.deploy_ms == 0


def test_barrel_that_ends_in_the_river_spawns_on_a_bank(db):
    m = mk(db)
    play(m, 0, "barbarian_barrel", 9.5, 11.5)                # 11.5 + 4.5 = 16.0, the middle of the river
    run(m, 1500)
    (b,) = [u for u in m.units.values() if u.card_id == "barbarian"]
    assert G.region(b.y) != "river" and b.x == pytest.approx(9.5)


def test_land_moves_river_points_to_the_nearer_bank_but_not_on_bridges():
    from backend_engine.simulator.mechanics.core import land
    assert land(9.5, 15.6) == (9.5, G.RIVER_Y0 - 0.5)
    assert land(9.5, 16.4) == (9.5, G.RIVER_Y1 + 0.5)
    assert land(9.5, 12.0) == (9.5, 12.0) and land(9.5, 20.0) == (9.5, 20.0)
    assert land(3.5, 16.0) == (3.5, 16.0)                    # on a bridge: a valid spot


# ------------------------------------------------------------------ Royal Ghost
def test_ghost_turns_invisible_when_idle_and_enemies_lose_it(db):
    m = mk(db)
    ghost = tank(m, 0, "royal_ghost", 9.5, 10.0)
    knight = tank(m, 1, "mini_pekka", 9.5, 14.0)             # 4 tiles away: sees it, can't reach it
    run(m, 1000)
    assert S.INVISIBLE not in ghost.status and knight.target_uid == ghost.uid
    run(m, 1100)                                              # 2 s idle -> invisible (2026-03-02: 1.8 -> 2 s)
    assert S.INVISIBLE in ghost.status
    assert knight.target_uid != ghost.uid and m._acquire(knight) is not ghost


def test_area_damage_still_hits_an_invisible_ghost(db):
    m = new(db, a=DECK, b=DECK)
    ghost = tank(m, 0, "royal_ghost", 9.5, 10.0)
    pekka = tank(m, 0, "mini_pekka", 9.5, 11.0)              # visible, right next to the ghost
    run(m, 2100)
    assert S.INVISIBLE in ghost.status
    (fire,) = play(m, 1, "fire_spirit", 9.5, 13.0)           # attacks the visible pekka; its area covers the ghost
    run(m, 200)
    assert fire.uid not in m.units
    dmg = db.stats("fire_spirit", 11).stats["damage"]
    assert BIG - pekka.hp == dmg and BIG - ghost.hp == dmg


def test_ghost_shows_itself_when_it_engages(db):
    m = mk(db)
    ghost = tank(m, 0, "royal_ghost", 9.5, 10.0)
    run(m, 2100)
    assert S.INVISIBLE in ghost.status
    k = tank(m, 1, "mini_pekka", 9.5, 11.2)                  # steps right next to it
    run(m, 150)
    assert S.INVISIBLE not in ghost.status and ghost.mstate["idle_ms"] == 0
    run(m, 2000)
    assert k.hp < BIG                                         # it fought, with splash
    assert S.INVISIBLE not in ghost.status


# ------------------------------------------------------------------ Tornado
def test_tornado_pulls_troops_damages_and_ends(db):
    m = mk(db)
    a = tank(m, 1, "mini_pekka", 9.5, 23.5)                  # 4 tiles from the centre (9.5, 19.5)
    b = tank(m, 1, "mini_pekka", 5.5, 19.5)
    far = tank(m, 1, "mini_pekka", 9.5, 31.0)                # outside the 5.5 radius
    own = tank(m, 0, "mini_pekka", 9.5, 15.0)                # own units are never pulled
    far0, own0 = (far.x, far.y), (own.x, own.y)              # (positions snap to tile centres on deploy)
    play(m, 0, "tornado", 9.5, 19.5)
    assert len(m.effects) == 1
    assert any(s["card_id"] == "tornado" for s in m.observation(0)["spells"])
    run(m, 1100)
    st = db.stats("tornado", 11).stats
    assert not m.effects and not any(s["card_id"] == "tornado" for s in m.observation(0)["spells"])
    for u in (a, b):
        assert G.dist(u.x, u.y, 9.5, 19.5) == pytest.approx(4.0 - 3.6 * 1.05, abs=0.05)     # 21 ticks of 0.18 tiles
        assert BIG - u.hp == 2 * st["damage"] == 168                                       # hits at 500 ms and 1000 ms
    assert (far.x, far.y) == far0 and far.hp == BIG
    assert (own.x, own.y) == own0 and own.hp == BIG


def test_tornado_hits_towers_for_less_and_never_moves_them(db):
    m = mk(db)
    tw = m.tower(1, "princess_left")
    x0, y0, hp0 = tw.x, tw.y, tw.hp
    play(m, 0, "tornado", 3.5, 22.0)
    run(m, 1100)
    st = db.stats("tornado", 11).stats
    assert (tw.x, tw.y) == (x0, y0) and hp0 - tw.hp == 2 * st["crown_tower_damage"] == 58


def test_tornado_effect_is_deterministic_across_snapshot(db):
    m = mk(db)
    tank(m, 1, "mini_pekka", 9.5, 23.5)
    play(m, 0, "tornado", 9.5, 19.5)
    run(m, 300)
    h0 = m.state_hash()
    snap = m.snapshot()
    run(m, 1500)
    h1 = m.state_hash()
    m.restore(snap)
    assert m.state_hash() == h0 and len(m.effects) == 1
    run(m, 1500)
    assert m.state_hash() == h1


# ------------------------------------------------------------------ data of the three cards
@pytest.mark.parametrize("cid,expected", [
    ("barbarian_barrel", {"damage": 215}),
    ("royal_ghost", {"hitpoints": 1152, "damage": 263}),     # 2026-10-06 balance notes
    ("tornado", {"damage": 84, "crown_tower_damage": 29}),
])
def test_values_at_level_11(db, cid, expected):
    assert db.stats(cid, 11).stats == expected


def test_card_parameters_come_from_the_source_data(db):
    assert db.cards["tornado"]["spell"]["duration_ms"] == 1050 and db.cards["tornado"]["spell"]["hit_every_ms"] == 500
    assert db.cards["tornado"]["spell"]["radius_tiles"] == 5.5
    ghost = {m["kind"]: m["params"] for m in db.cards["royal_ghost"]["mechanics"]}
    assert ghost["invisible_when_idle"]["delay_ms"] == 2000      # 2026-03-02 balance notes
    barrel = {m["kind"]: m["params"] for m in db.cards["barbarian_barrel"]["mechanics"]}
    assert barrel["spawn_on_end"] == {"token": "barbarian", "deploy_ms": 500}
    assert db.cards["royal_ghost"]["game_id"] == 26000050 and db.cards["tornado"]["game_id"] == 28000012
