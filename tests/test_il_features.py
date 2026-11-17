"""Shared IL helpers (backend_engine/brain/il_features.py): frames, tiles, masks. The observation itself is
tested in tests/test_obs_encoder.py."""
import numpy as np
import pytest

from backend_engine.brain import il_features as F
from backend_engine.perception.bayesian_deck_guesser import default_guesser
from backend_engine.perception.opp_elixir_count import OppElixirTracker
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig

DECK = ("hog_rider", "musketeer", "cannon", "miner", "fireball", "skeletons", "the_log", "knight")


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


def test_tile_index_and_center_round_trip():
    for side in (0, 1):
        for t in (0, 17, 300, 575):
            x, y = F.tile_center(side, t)
            assert F.tile_index(side, x, y) == t
    assert F.tile_index(0, 3.5, 10.5) == 10 * 18 + 3
    assert F.tile_index(1, 3.5, 21.5) == 10 * 18 + 3


def test_tile_mask_matches_engine_rules(db):
    m = Match(MatchConfig((DECK, DECK), seed=1, shuffle=False), db)
    m.play(0, "cannon", 9.5, 10.5)
    m.play(1, "cannon", 9.5, 21.5)
    for _ in range(10):
        m.step()
    m.tower(1, "princess_left").hp = 0             # opens a pocket for side 0 on the left
    m.tower(0, "princess_right").hp = 0            # and for side 1 on its right
    m.step()
    buildings = [c for c, d in db.cards.items() if d["type"] == "building"]
    for side in (0, 1):
        for card in ["hog_rider", "miner", "fireball", *buildings]:            # buildings: footprint rules
            st = m.sides[side]
            st.hand, st.elixir = [card, "skeletons", "the_log", "knight"], 10.0
            mask = F.tile_mask(m.observation(side), card, db)
            truth = [m.check_play(side, card, *F.tile_center(side, t)) is None for t in range(F.NUM_TILES)]
            assert mask.tolist() == truth, (side, card)


def test_slot_mask_and_pack(db):
    obs = {"hand": ["knight", "fireball", "skeletons", "hog_rider"], "elixir": 3.5}
    assert F.slot_mask(obs, db).tolist() == [True, False, True, False]
    mask = np.random.default_rng(0).random(F.NUM_TILES) > 0.5
    packed = F.pack_mask(mask)
    assert packed.shape == (72,) and np.array_equal(F.unpack_mask(packed), mask)


def test_merged_hand_probs_is_idempotent(db):
    tr = OppElixirTracker(db.rules, observer_side=0)
    m = Match(MatchConfig((DECK, DECK), seed=2), db)
    tr.observe(m.play(1, m.sides[1].hand[0], 9.5, 25.5).event)
    est = tr.estimate(1000)
    guess = default_guesser().guess(est.known_cards) if default_guesser() else None
    merged = F.merged_hand_probs(est, guess)
    from dataclasses import replace
    assert F.merged_hand_probs(replace(est, in_hand_prob=merged), guess) == merged
