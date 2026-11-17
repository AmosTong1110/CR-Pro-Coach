"""il_obs_v2 encoder (backend_engine/brain/obs_encoder.py): fixed layout, perspective, attributes, schema."""
import numpy as np
import pytest

from backend_engine.brain import obs_encoder as E
from backend_engine.perception.opp_elixir_count import OppElixirTracker
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig

DECK = ("hog_rider", "musketeer", "cannon", "miner", "fireball", "skeletons", "the_log", "knight")


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


def _encode(m, side, db):
    est = OppElixirTracker(db.rules, observer_side=side).estimate(m.t_ms)
    return E.encode_observation(m.observation(side), est, None, db)


def test_layout_is_a_fixed_constant():
    assert E.OBS_DIM == 1752
    blocks = list(E.LAYOUT.values())
    assert blocks[0][0] == 0 and blocks[-1][1] == E.OBS_DIM
    assert all(a[1] == b[0] for a, b in zip(blocks, blocks[1:]))         # contiguous, no gaps
    assert len(E.ATTRIBUTE_KEYS) == len(set(E.ATTRIBUTE_KEYS)) == E.N_ATTR == 30
    s = E.schema()
    assert s["obs_dim"] == E.OBS_DIM and s["attribute_keys"] == list(E.ATTRIBUTE_KEYS)


def test_encoding_is_finite_and_tokens_are_filled(db):
    m = Match(MatchConfig((DECK, DECK), seed=3, shuffle=False), db)
    assert m.play(0, "musketeer", 3.5, 10.5).ok
    m.sides[0].elixir = 10.0
    assert m.play(0, "fireball", 3.5, 25.5).ok
    for _ in range(10):
        m.step()
    v = _encode(m, 0, db)
    assert v.shape == (E.OBS_DIM,) and v.dtype == np.float32 and np.isfinite(v).all()
    u0 = E.LAYOUT["units"][0]
    assert v[u0] == 1.0 and v[u0 + 1] == 1.0                                # first unit token: present, own
    attrs = v[u0 + E.UNIT_DYN:u0 + E.UNIT_TOK]
    assert np.array_equal(attrs, E.card_attributes(db, "musketeer"))
    s0 = E.LAYOUT["spells"][0]
    assert v[s0] == 1.0 and v[s0 + E.SPELL_DYN + E.ATTRIBUTE_KEYS.index("is_spell")] == 1.0   # Fireball in flight
    h0 = E.LAYOUT["hand"][0]
    for k in range(E.HAND_TOKENS):                                          # every hand token holds a card
        assert v[h0 + k * E.HAND_TOK] == 1.0
    t0 = E.LAYOUT["tower_troops"][0]
    assert np.array_equal(v[t0:t0 + E.N_TATTR], E.tower_attributes(db, "king_tower"))


def test_mirrored_positions_encode_the_same_for_both_sides(db):
    m = Match(MatchConfig((DECK, DECK), seed=0, shuffle=False), db)
    assert np.array_equal(_encode(m, 0, db), _encode(m, 1, db))
    assert m.play(0, "musketeer", 3.5, 10.5).ok
    assert m.play(1, "musketeer", 3.5, 21.5).ok    # same spot in side 1's own frame
    for _ in range(10):
        m.step()
    assert np.allclose(_encode(m, 0, db), _encode(m, 1, db))


def test_attributes_describe_the_card(db):
    a = {c: dict(zip(E.ATTRIBUTE_KEYS, E.card_attributes(db, c))) for c in ("pekka", "minions", "poison", "tombstone",
                                                                          "hog_rider", "dark_prince")}
    assert a["pekka"]["hitpoints"] > 1.0 and a["pekka"]["elixir"] == pytest.approx(0.7)   # smooth, not clipped
    assert a["minions"]["is_air"] == 1.0 and a["minions"]["spawn_count"] == pytest.approx(0.3)
    assert a["poison"]["is_spell"] == 1.0 and a["poison"]["lifetime"] > 0 and a["poison"]["dps"] > 0
    assert a["tombstone"]["is_building"] == 1.0 and a["tombstone"]["death_spawn_count"] > 0
    assert a["tombstone"]["spawn_rate"] > 0
    assert a["hog_rider"]["building_only"] == 1.0 and a["dark_prince"]["shield_hp"] > 0
    evo = dict(zip(E.ATTRIBUTE_KEYS, E.card_attributes(db, "knight", "evo")))
    assert evo["is_evolved"] == 1.0 and dict(zip(E.ATTRIBUTE_KEYS, E.card_attributes(db, "knight")))["is_evolved"] == 0
    assert not E.card_attributes(db, "no_such_card").any()                  # unknown id: zeros, no exception


def test_mirror_spec_is_an_involution():
    perm, xs, tiles = E.mirror_spec()
    assert np.array_equal(perm[perm], np.arange(E.OBS_DIM)) and np.array_equal(tiles[tiles], np.arange(E.NUM_TILES))
    assert tiles[0] == 17 and len(xs) == E.MAX_UNITS + E.MAX_SPELLS


def test_model_meta_check_flags_legacy_and_mismatches():
    assert E.check_model_meta(E.schema()) is None
    legacy = {"obs_version": "il_obs_v1", "obs_dim": 333, "vocab": ["knight"] * 38}
    assert "legacy" in E.check_model_meta(legacy) and "333" in E.check_model_meta(legacy)
    bad = {**E.schema(), "attribute_keys": list(E.ATTRIBUTE_KEYS)[:-1]}
    assert "attribute_keys" in E.check_model_meta(bad)
    assert "obs_dim" in E.check_model_meta({**E.schema(), "obs_dim": 999})
