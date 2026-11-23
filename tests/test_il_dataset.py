"""IL dataset builder (tools/build_il_dataset.py) on synthetic IL_Replay payloads (no network)."""
import json

import numpy as np
import pytest

from backend_engine.brain import obs_encoder as F
from backend_engine.simulator.card_db import CardDB
from tools.build_il_dataset import (
    ProxyMap, ShardWriter, Stats, load_card_catalog, parse_payload, replay_battle, split_card_key,
)

HOG = ["hog-rider", "musketeer", "cannon", "ice-golem", "ice-spirit", "skeletons", "the-log", "fireball"]


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


@pytest.fixture(scope="module")
def proxies(db):
    return ProxyMap(db, load_card_catalog())


def _ev(side, tick, card, x, y, kind="play_card"):
    return {"kind": kind, "side": side, "card_key": card, "replay_tick_20hz": tick,
            "coordinates": {"native_world_units": {"x": int(x * 1000), "y": int(y * 1000)}}}


def _payload(team=HOG, opp=HOG, events=(), crowns=(1, 0)):
    deck = lambda cards: [{"card_key": c, "level": 16} for c in cards]  # noqa: E731
    return {"battle": {"game_mode": "Ranked",
                       "team": {"crowns": crowns[0], "players": [{"deck": deck(team)}]},
                       "opponent": {"crowns": crowns[1], "players": [{"deck": deck(opp)}]}},
            "events": list(events)}


def test_split_card_key():
    assert split_card_key("hog-rider") == ("hog_rider", "base")
    assert split_card_key("firecracker-ev1") == ("firecracker", "evo")
    assert split_card_key("knight-hero") == ("knight", "hero")
    assert split_card_key("the-log") == ("the_log", "base")
    assert split_card_key("mini-pekka") == ("mini_pekka", "base")


def test_proxy_map_keeps_type_layer_and_uniqueness(db, proxies):
    deck = ["hog_rider", "arrows", "giant", "baby_dragon", "bomb_tower", "zap", "knight", "goblin_gang"]
    m = proxies.deck(deck)
    assert m["hog_rider"] == "hog_rider" and m["zap"] == "zap"
    assert len(set(m.values())) == 8
    assert db.cards[m["arrows"]]["type"] == "spell"
    assert db.cards[m["bomb_tower"]]["type"] == "building"
    assert db.cards[m["baby_dragon"]]["movement"]["layer"] == "air"
    assert db.cards[m["giant"]]["type"] == "troop"


def test_parse_payload(db):
    rep = parse_payload(_payload(events=[_ev("team", 40, "hog-rider", 3.5, 14.5),
                                         {"kind": "activate_ability", "side": "opponent", "replay_tick_20hz": 50,
                                          "ability_source_candidates": ["golden-knight"]}]), "r1")
    assert rep.decks[0][0] == "hog_rider" and rep.levels == (16, 16) and rep.crowns == (1, 0)
    assert [(e.side, e.card, e.ability) for e in rep.events] == [(0, "hog_rider", False), (1, "golden_knight", True)]
    assert (rep.events[0].x, rep.events[0].y) == (3.5, 14.5)
    assert parse_payload({"battle": {"team": {"players": []}}}) is None


def _run(db, proxies, payload, tmp_path, **kw):
    w = ShardWriter(tmp_path, shard_size=1000)
    st = Stats()
    ok = replay_battle(parse_payload(payload, "x"), db, w, 7, st, proxies, None, **kw)
    w.close({"stats": st.as_dict()})
    data = np.load(tmp_path / "shard_0000.npz") if ok and w.total_samples else None
    return ok, w, st, data


def test_replay_both_sides_learn_with_legal_labels(db, proxies, tmp_path):
    events = [_ev("team", 60, "hog-rider", 3.5, 14.5), _ev("opponent", 70, "musketeer", 14.5, 21.5),
              _ev("team", 130, "the-log", 14.5, 24.5), _ev("opponent", 200, "fireball", 3.5, 6.5),
              _ev("team", 260, "skeletons", 9.5, 9.5), _ev("opponent", 300, "cannon", 9.5, 22.5)]
    ok, w, st, d = _run(db, proxies, _payload(events=events), tmp_path)
    assert ok and st.learner_sides == 2 and w.total_plays == 6
    assert 0 < w.total_waits <= w.total_plays
    play = d["gate"] == 1
    tm = F.unpack_mask(d["tile_mask"][play])
    tiles = d["tile_index"][play].astype(int)
    assert tm[np.arange(len(tiles)), tiles].all()
    assert d["slot_mask"][play][np.arange(len(tiles)), d["card_slot"][play]].all()
    # each side's labels are in its own frame: the opening Hog Rider and Musketeer both land in the frame's south
    assert set(d["value_target"][d["side"] == 0]) == {1.0} and set(d["value_target"][d["side"] == 1]) == {0.0}
    assert (d["battle"] == 7).all()
    first = {int(s): int(t) for s, t in zip(d["side"][play], tiles)}
    assert first[0] // 18 < 15 and first[1] // 18 < 15
    assert (d["tile_index"][~play] == -1).all() and (d["card_slot"][~play] == -1).all()
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["obs_dim"] == F.OBS_DIM and manifest["schema"] == "il_shards_v2"
    assert manifest["attribute_keys"] == list(F.ATTRIBUTE_KEYS) and "vocab" not in manifest


def test_proxy_side_is_not_a_learner_and_proxy_plays_still_happen(db, proxies, tmp_path):
    opp = ["giant", "freeze", "musketeer", "knight", "skeletons", "zap", "minions", "cannon"]   # 7/8 simulated
    events = [_ev("opponent", 40, "giant", 9.5, 30.5), _ev("team", 80, "hog-rider", 14.5, 14.5),
              _ev("opponent", 120, "freeze", 14.5, 10.5)]
    ok, w, st, d = _run(db, proxies, _payload(opp=opp, events=events), tmp_path)
    assert ok and st.learner_sides == 1
    assert set(d["side"].tolist()) == {0}
    assert st.dropped_plays == 0                               # the proxied Freeze was played


def test_coverage_filter_and_illegal_tile_skip(db, proxies, tmp_path):
    low = ["giant", "freeze", "pekka", "wizard", "executioner", "zappies", "bowler", "skeletons"]   # 3/8
    ok, *_ = _run(db, proxies, _payload(team=low, opp=low), tmp_path / "a")
    assert not ok
    # a troop "played" on the enemy side (impossible in the sim): no sample, but the card is relocated and played
    events = [_ev("team", 60, "hog-rider", 3.5, 24.5), _ev("team", 200, "the-log", 3.5, 20.5)]
    ok, w, st, d = _run(db, proxies, _payload(events=events), tmp_path / "b")
    assert ok and st.skipped_plays.get("illegal_tile") == 1 and st.relocated_plays == 1
    assert w.total_plays == 1


def test_shard_writer_lifecycle(tmp_path):
    w = ShardWriter(tmp_path, shard_size=5)
    for i in range(12):
        w.add_sample(np.zeros(F.OBS_DIM, np.float32), i % 2, 0 if i % 2 else -1, 100 if i % 2 else -1, 1.0)
    assert w.current_shard_idx == 2
    w.close()
    m = json.loads((tmp_path / "manifest.json").read_text())
    assert (m["total_samples"], m["total_plays"], m["num_shards"]) == (12, 6, 3)
    z = np.load(tmp_path / "shard_0000.npz")
    assert z["obs"].shape == (5, F.OBS_DIM) and z["tile_mask"].shape == (5, 72) and z["slot_mask"].shape == (5, 4)


def test_game_mode_filter(db, proxies, tmp_path):
    events = [_ev("team", 60, "hog-rider", 3.5, 14.5)]
    friendly = _payload(events=events)
    friendly["battle"]["game_mode"] = "1v1 Battle"
    ok, _, st, _ = _run(db, proxies, friendly, tmp_path / "a", modes=("Ranked", "Ladder"))
    assert not ok and st.rejected.get("mode") == 1 and st.replays_seen == 1
    ok, *_ = _run(db, proxies, _payload(events=events), tmp_path / "b", modes=("Ranked", "Ladder"))
    assert ok                                                   # a Ranked battle passes
    ok, *_ = _run(db, proxies, friendly, tmp_path / "c")        # no filter: every mode is kept
    assert ok


def test_samples_record_the_learner_deck(db, proxies, tmp_path):
    events = [_ev("team", 60, "hog-rider", 3.5, 14.5), _ev("opponent", 70, "musketeer", 14.5, 21.5)]
    ok, w, st, d = _run(db, proxies, _payload(events=events), tmp_path)
    assert ok and (d["deck"] == 0).all()                       # both sides play the same Hog deck: one entry
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    hog = sorted(split_card_key(c)[0] for c in HOG)
    assert manifest["learner_decks"] == [hog]
    assert w.deck_id(list(reversed(hog))) == 0 and w.deck_id(["knight"] * 8) == 1   # order-free, new deck -> new id
