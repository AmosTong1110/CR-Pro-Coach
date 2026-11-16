"""
Unit tests for Imitation Learning Dataset Builder (tools/build_il_dataset.py)
"""
import json
import numpy as np
import pytest
from pathlib import Path

from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig
from backend_engine.perception.opp_elixir_count import OppElixirTracker
from backend_engine.perception.bayesian_deck_guesser import default_guesser
from tools.build_il_dataset import (
    OBS_DIM,
    NUM_TILES,
    HAND_SLOTS,
    get_card_vocab,
    normalize_card_name,
    check_deck_coverage,
    encode_observation,
    ShardWriter,
    replay_battle,
)


@pytest.fixture
def db():
    return CardDB.load()


@pytest.fixture
def vocab(db):
    return get_card_vocab(db)


def test_card_normalization():
    assert normalize_card_name("Hog-Rider") == "hog_rider"
    assert normalize_card_name("the-log") == "the_log"
    assert normalize_card_name("log") == "the_log"
    assert normalize_card_name("skeletons-ev1") == "skeletons"
    assert normalize_card_name("firecracker_evo") == "firecracker"
    assert normalize_card_name("barb-barrel") == "barbarian_barrel"
    assert normalize_card_name("") == ""


def test_deck_coverage(vocab):
    vocab_set = set(vocab)
    full_deck = ["hog-rider", "musketeer", "cannon", "ice-golem", "ice-spirit", "skeletons", "the-log", "fireball"]
    cov, unsupp = check_deck_coverage(full_deck, vocab_set)
    assert cov == 8
    assert len(unsupp) == 0

    partial_deck = ["hog-rider", "musketeer", "cannon", "mega-knight", "electro-wizard", "goblin-gang", "sparky", "fireball"]
    cov2, unsupp2 = check_deck_coverage(partial_deck, vocab_set)
    assert cov2 == 4
    assert set(unsupp2) == {"mega_knight", "electro_wizard", "goblin_gang", "sparky"}


def test_encode_observation_shape_and_bounds(db, vocab):
    t_deck = ("hog_rider", "musketeer", "cannon", "ice_golem", "ice_spirit", "skeletons", "the_log", "fireball")
    o_deck = ("hog_rider", "musketeer", "cannon", "ice_golem", "ice_spirit", "skeletons", "the_log", "fireball")
    cfg = MatchConfig(decks=(t_deck, o_deck), levels=(11, 11), seed=42)
    match = Match(cfg, db)

    tracker = OppElixirTracker(db.rules, observer_side=0, evo_cards=db.evo_capable())
    guesser = default_guesser()

    obs = encode_observation(match, side=0, tracker=tracker, guesser=guesser, vocab=vocab)
    assert isinstance(obs, np.ndarray)
    assert obs.dtype == np.float32
    assert obs.shape == (OBS_DIM,)
    assert not np.isnan(obs).any()
    assert not np.isinf(obs).any()

    # Step simulation forward to create units and re-verify
    match.play(0, "hog_rider", 3.5, 14.0)
    for _ in range(20):
        match.step()

    obs2 = encode_observation(match, side=0, tracker=tracker, guesser=guesser, vocab=vocab)
    assert obs2.shape == (OBS_DIM,)
    assert not np.isnan(obs2).any()
    assert not np.isinf(obs2).any()


def test_shard_writer_lifecycle(tmp_path, vocab):
    writer = ShardWriter(output_dir=tmp_path, shard_size=5)
    dummy_obs = np.zeros(OBS_DIM, dtype=np.float32)

    for i in range(12):
        writer.add_sample(obs=dummy_obs, gate=1 if i % 2 == 0 else 0, slot=0 if i % 2 == 0 else -1, tile=100, value=1.0)

    # 12 samples with shard_size=5 should create 2 flushed shards + 2 samples in buffer
    assert writer.current_shard_idx == 2
    assert len(writer.obs_buffer) == 2

    manifest_path = writer.close(vocab)
    assert manifest_path.exists()
    assert writer.current_shard_idx == 3  # total 3 shards

    # Verify manifest
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["total_samples"] == 12
    assert manifest["total_plays"] == 6
    assert manifest["total_waits"] == 6
    assert manifest["obs_dim"] == OBS_DIM
    assert manifest["num_shards"] == 3

    # Verify first shard contents
    shard0 = np.load(tmp_path / "shard_0000.npz")
    assert shard0["obs"].shape == (5, OBS_DIM)
    assert shard0["gate"].shape == (5,)
    assert shard0["card_slot"].shape == (5,)
    assert shard0["tile_index"].shape == (5,)
    assert shard0["value_target"].shape == (5,)


def test_replay_battle_synthetic(tmp_path, db, vocab):
    writer = ShardWriter(output_dir=tmp_path, shard_size=100)

    match_meta = {
        "battle_id": "synthetic_01",
        "decks": {
            "team": ["hog-rider", "musketeer", "cannon", "ice-golem", "ice-spirit", "skeletons", "the-log", "fireball"],
            "opponent": ["hog-rider", "musketeer", "cannon", "ice-golem", "ice-spirit", "skeletons", "the-log", "fireball"],
        },
        "crowns": {"team": 1, "opponent": 0},
    }

    events = [
        {"ticks": 20, "sequence": 0, "side": "team", "event_type": "card_play", "card": "hog-rider", "x": 3500, "y": 18000},
        {"ticks": 50, "sequence": 1, "side": "opponent", "event_type": "card_play", "card": "cannon", "x": 8500, "y": 12000},
        {"ticks": 90, "sequence": 2, "side": "team", "event_type": "card_play", "card": "the-log", "x": 3500, "y": 16000},
    ]

    ok = replay_battle(
        match_meta=match_meta,
        events=events,
        db=db,
        vocab=vocab,
        shard_writer=writer,
        min_deck_coverage=8,
    )
    assert ok is True
    writer.close(vocab)

    assert writer.total_plays >= 2
    assert writer.total_battles == 1

    # Load shard and check
    shard_files = list(tmp_path.glob("shard_*.npz"))
    assert len(shard_files) == 1
    data = np.load(shard_files[0])
    assert len(data["obs"]) == writer.total_samples
    assert (data["value_target"] == 1.0).all()  # team won
