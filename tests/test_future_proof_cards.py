"""Proof tests: the card architecture survives new official content without rework.

1. Zero-rework expansion: a new card (Royal Giant, completed through tools/sync_card_catalog.py on top of the
   47 playable cards) and a new Tower Troop (Cannoneer) leave OBS_DIM and the encoder schema exactly as they were;
   the encoder describes them with non-zero physical attributes; the unchanged NeuralBot model plays a match with
   them, the engine simulates it, and the value model scores it. No model, encoder or engine code is touched.
2. Unknown-card resilience: replays / battle logs containing a card the data has never heard of
   ("super_future_card_2027") are quarantined (skipped, counted, summarised) with zero exceptions.
"""
import copy
import json
from pathlib import Path

import numpy as np
import pytest

from backend_engine.brain import obs_encoder as E
from backend_engine.brain.neural_bot import NeuralBot, check_meta, load_neural_bot
from backend_engine.brain.value_model import ValueModel
from backend_engine.perception.opp_elixir_count import OppElixirTracker
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig
from backend_engine.simulator.match_runner import MatchRunner
from backend_engine.simulator.quarantine import CardQuarantine
from tools.sync_card_catalog import parse_source, plan_sync

ROOT = Path(__file__).resolve().parents[1]
FIXTURE = Path(__file__).parent / "fixtures" / "crbot_il_dummy"
FROZEN_OBS_DIM = 1752                    # il_obs_v2: must never change when cards are added
NEW_CARD = "royal_giant"
DECK = ("knight", "musketeer", "cannon", "fireball", "the_log", "skeletons", "ice_spirit", NEW_CARD)

ROYAL_GIANT = {   # a standard building-targeting troop: only traits the engine already simulates
    "name": "Royal Giant", "game_id": 26000024, "type": "troop", "rarity": "common", "elixir": 6,
    "stats": {"11": {"hitpoints": 3013, "damage": 254}},
    "attack": {"targets": ["buildings"], "range_tiles": 5.0, "hit_speed_s": 1.7, "first_hit_s": 0.8,
               "splash_radius_tiles": 0.0, "sight_tiles": 7.5},
    "movement": {"layer": "ground", "speed_tiles_s": 1.0},
    "mechanics": [{"kind": "building_targeting", "params": {}}, {"kind": "heavy_knockback_immune", "params": {}}],
    "count": 1, "deploy_time_s": 1.0, "collision_radius_tiles": 0.75, "mass": 18,
}
CANNONEER = {     # a Tower Troop for the princess slots
    "name": "Cannoneer", "game_id": 159000001, "rarity": "epic",
    "stats": {"11": {"hitpoints": 2616, "damage": 386}},
    "attack": {"targets": ["ground", "air"], "range_tiles": 7.5, "hit_speed_s": 2.1, "projectile_speed_tiles_s": 10.0},
}


@pytest.fixture(scope="module")
def base_db():
    return CardDB.load()


@pytest.fixture(scope="module")
def expanded(base_db):
    """The catalog after a sync that brings Royal Giant's stats and the Cannoneer Tower Troop."""
    snap = {"format": "crbot_catalog_snapshot_v1", "as_of": "2027-01-01", "cards": {NEW_CARD: ROYAL_GIANT},
            "towers": {"cannoneer": CANNONEER}}
    raw, rep = plan_sync(copy.deepcopy(base_db.raw), parse_source(snap), "2027-01-01")
    assert not rep.conflicts
    return CardDB(raw)


# -- 1. zero-rework expansion ------------------------------------------------------------------------------------
def test_new_card_and_tower_troop_join_the_playable_pool(base_db, expanded):
    assert NEW_CARD not in base_db.cards and NEW_CARD in expanded.cards          # it really is new to the pool
    assert len(expanded.cards) == len(base_db.cards) + 1
    assert expanded.towers["cannoneer"]["enabled"] is True
    assert {"building_only", "heavy_knockback_immune"} <= expanded.traits(NEW_CARD)


def test_obs_dim_and_schema_are_unchanged(base_db, expanded):
    assert E.OBS_DIM == FROZEN_OBS_DIM
    before = E.schema()
    m = Match(MatchConfig((DECK, DECK), seed=1, shuffle=False, tower_troops=("cannoneer", "princess_tower")),
              expanded)
    est = OppElixirTracker(expanded.rules, observer_side=0).estimate(0)
    vec = E.encode_observation(m.observation(0), est, None, expanded)
    assert vec.shape == (FROZEN_OBS_DIM,) and E.schema() == before
    check_meta(json.loads((FIXTURE / "crbot_il_v2.json").read_text()), expanded)   # same model still valid


def test_encoder_describes_the_new_card_and_tower_troop_physically(expanded):
    a = dict(zip(E.ATTRIBUTE_KEYS, E.card_attributes(expanded, NEW_CARD)))
    assert a["hitpoints"] == pytest.approx(3013 / 3000) and a["elixir"] == pytest.approx(0.6)
    assert a["building_only"] == 1.0 and a["range"] == pytest.approx(0.5) and a["dps"] > 0 and a["move_speed"] > 0
    t = dict(zip(E.TOWER_ATTRIBUTE_KEYS, E.tower_attributes(expanded, "cannoneer")))
    assert t["hitpoints"] == pytest.approx(2616 / 5000) and t["damage"] == pytest.approx(0.386)
    # in a live observation: the new card in hand, the Cannoneer in the own tower-troop block
    m = Match(MatchConfig((DECK, DECK), seed=1, shuffle=False, tower_troops=("cannoneer", "princess_tower")),
              expanded)
    m.sides[0].hand = [NEW_CARD, "knight", "musketeer", "cannon"]
    vec = E.encode_observation(m.observation(0), OppElixirTracker(expanded.rules, observer_side=0).estimate(0),
                               None, expanded)
    h0 = E.LAYOUT["hand"][0]
    assert np.array_equal(vec[h0 + E.HAND_DYN:h0 + E.HAND_TOK], E.card_attributes(expanded, NEW_CARD))
    t0 = E.LAYOUT["tower_troops"][0] + E.N_TATTR                     # own princess slot (after own king)
    assert np.array_equal(vec[t0:t0 + E.N_TATTR], E.tower_attributes(expanded, "cannoneer"))


def test_neural_bot_and_engine_play_a_match_with_them(expanded):
    bot = load_neural_bot(expanded, 1, FIXTURE, "crbot_il_v2")       # the unchanged model file
    assert isinstance(bot, NeuralBot)
    m = Match(MatchConfig((DECK, DECK), seed=5, shuffle=False, tower_troops=("princess_tower", "cannoneer")),
              expanded)
    st = m.sides[1]
    st.hand = ["knight", "musketeer", "cannon", NEW_CARD]                # the dummy model prefers slot 3
    st.queue = [c for c in DECK if c not in st.hand]
    st.elixir = 10.0
    runner = MatchRunner(m, {0: None, 1: bot})
    seen_new_unit = False
    for _ in range(400):                                                # 20 s of match time
        runner.tick()
        seen_new_unit |= any(u.card_id == NEW_CARD and u.side == 1 for u in m.units.values())
    assert seen_new_unit and not runner.stats.rejected[1] and runner.stats.plays[1] >= 1
    assert {u.card_id for u in m.units.values() if u.side == 1 and u.kind == "tower"} == {"king_tower", "cannoneer"}
    p = ValueModel.load().value(m, 1)                                   # roster-independent features
    assert 0.0 <= p <= 1.0 and p == pytest.approx(1.0 - ValueModel.load().value(m, 0))


# -- 2. unknown-card resilience ----------------------------------------------------------------------------------
FUTURE = "super_future_card_2027"


def test_dataset_builder_quarantines_unknown_cards(base_db, tmp_path):
    from tests.test_il_dataset import HOG, _ev, _payload
    from tools.build_il_dataset import ProxyMap, ShardWriter, Stats, load_card_catalog, parse_payload, replay_battle
    q = CardQuarantine(base_db)
    proxies = ProxyMap(base_db, load_card_catalog())
    w, st = ShardWriter(tmp_path, 100), Stats()
    future = [FUTURE.replace("_", "-")] + HOG[1:]
    events = [_ev("team", 60, "hog-rider", 3.5, 14.5), _ev("opponent", 80, FUTURE.replace("_", "-"), 9.5, 24.5)]
    ok = replay_battle(parse_payload(_payload(opp=future, events=events), "f"), base_db, w, 0, st, proxies, None,
                       quarantine=q)
    assert ok is False and st.rejected["quarantine"] == 1 and w.total_samples == 0
    assert q.unknown == {FUTURE: 1} and FUTURE in q.summary() and "sync_card_catalog" in q.summary()
    ok = replay_battle(parse_payload(_payload(events=events[:1]), "ok"), base_db, w, 1, st, proxies, None,
                       quarantine=q)
    assert ok is True and w.total_samples > 0                          # normal replays still go through


def test_battlelog_analyzer_quarantines_unknown_cards(base_db, tmp_path):
    from tools.analyze_my_battlelog import parse_battlelog
    deck = [{"name": n} for n in ("Hog Rider", "Musketeer", "Cannon", "Fireball", "The Log", "Skeletons",
                                  "Ice Spirit", "Ice Golem")]
    future = [{"name": "Super Future Card 2027"}] + deck[1:]
    battle = lambda opp: {"battleTime": "20270101T000000.000Z", "gameMode": {"name": "Ladder"},  # noqa: E731
                          "team": [{"tag": "#ME", "crowns": 1, "cards": deck}],
                          "opponent": [{"tag": "#OPP", "crowns": 0, "cards": opp}]}
    log = tmp_path / "log.json"
    log.write_text(json.dumps([battle(deck), battle(future), battle(deck)]))
    q = CardQuarantine(base_db)
    matches = parse_battlelog(log, "#ME", base_db, q)
    assert len(matches) == 2 and q.skipped == 1 and "Super Future Card 2027" in q.unknown
    assert all(len(m["opp_deck"]) == 8 for m in matches)


def test_quarantine_strict_mode_and_catalog_only_cards(base_db):
    loose, strict = CardQuarantine(base_db), CardQuarantine(base_db, strict=True)
    deck = ["hog-rider", "Golem", "Mini P.E.K.K.A", "knight-ev1"]       # Golem: catalog only (not playable yet)
    assert "golem" in base_db.catalog and "golem" not in base_db.cards
    assert loose.check(deck) is None and strict.check(deck) is not None
    assert loose.resolve("Mini P.E.K.K.A") == "mini_pekka" and loose.resolve("knight-ev1") == "knight"
    assert loose.check([FUTURE]) and loose.unknown[FUTURE] == 1


def test_encoder_never_raises_on_unknown_ids(base_db):
    m = Match(MatchConfig((DECK[:7] + ("hog_rider",),) * 2, seed=0, shuffle=False), base_db)
    obs = m.observation(0)
    obs["units"].append({**obs["units"][0], "uid": 999, "kind": "troop", "card_id": FUTURE, "y": 12.0})
    obs["spells"].append({"side": 1, "card_id": FUTURE, "x": 9.0, "y": 10.0})
    vec = E.encode_observation(obs, OppElixirTracker(base_db.rules, observer_side=0).estimate(0), None, base_db)
    assert vec.shape == (FROZEN_OBS_DIM,) and np.isfinite(vec).all()
