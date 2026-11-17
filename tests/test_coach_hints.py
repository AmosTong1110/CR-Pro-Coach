"""Mode B -> NeuralBot coach hints (backend_engine/brain/coach_hints.py), with the dummy ONNX model."""
from pathlib import Path

import numpy as np
import pytest

from backend_engine.brain import il_features as F
from backend_engine.brain.coach_hints import (
    CoachHinter, ModeBEncoder, ModeBSnapshot, SeenUnit, hand_hypotheses, snapshot_from_companion,
    snapshot_from_log,
)
from backend_engine.brain.scripted_bot import ScriptedBot
from backend_engine.perception.bayesian_deck_guesser import default_guesser
from backend_engine.perception.opp_elixir_count import OppElixirTracker
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig
from backend_engine.simulator.match_runner import MatchRunner

FIXTURE = Path(__file__).parent / "fixtures" / "crbot_il_dummy"
HOG = ("hog_rider", "musketeer", "cannon", "ice_golem", "ice_spirit", "skeletons", "fireball", "the_log")


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


@pytest.fixture(scope="module")
def hinter(db):
    return CoachHinter.load(FIXTURE, db)


def test_hand_from_cycle():
    hands, nxt = hand_hypotheses(HOG, [])
    assert len(hands) == 70 and nxt == ""
    hands, _ = hand_hypotheses(HOG, ["hog_rider", "cannon"])
    assert len(hands) == 15 and all("hog_rider" not in h and "cannon" not in h for h in hands)
    plays = ["hog_rider", "cannon", "the_log", "skeletons", "ice_spirit"]
    hands, nxt = hand_hypotheses(HOG, plays)
    assert hands == [tuple(sorted(["hog_rider", "musketeer", "ice_golem", "fireball"]))] and nxt == "cannon"
    # a duplicate detection inside the cycle doesn't break it, unknown cards are ignored
    hands, nxt = hand_hypotheses(HOG, plays + ["ice_spirit", "pekka"])
    assert len(hands) == 1 and nxt == "cannon"


def test_mode_b_encoding_matches_the_simulator(db):
    """A snapshot built from what Mode B can know about a simulated match encodes like the engine's own
    observation (the deploying flag excepted: Mode B only estimates it)."""
    m = Match(MatchConfig((HOG, HOG), seed=5), db)
    r = MatchRunner(m, {0: ScriptedBot(db, 0, 0.8), 1: ScriptedBot(db, 1, 0.8)})
    while m.t_ms < 170_000 and (sum(e.side == 0 for e in m.events) < 6 or not any(
            u.kind != "tower" for u in m.units.values())):
        r.tick()
    assert m.result is None and sum(e.side == 0 for e in m.events) >= 4
    towers = {(s, role): (m.tower(s, role).hp / m.tower(s, role).max_hp if m.tower(s, role) else 0.0)
              for s in (0, 1) for role in ("king", "princess_left", "princess_right")}
    units = [SeenUnit(u.card_id, u.side, u.x, u.y, 0, u.hp / u.max_hp) for u in m.units.values()
             if u.kind != "tower" and u.hp > 0]
    snap = ModeBSnapshot(m.t_ms, HOG, list(m.events), units, towers=towers,
                         crowns=(m.sides[0].crowns, m.sides[1].crowns))
    enc = ModeBEncoder(db)
    est_opp, est_me, guess = enc.estimates(snap)
    hands, nxt = hand_hypotheses(HOG, [e.card_id for e in m.events if e.side == 0 and not e.is_ability])
    assert set(hands[0]) == set(m.sides[0].hand) and nxt == m.sides[0].queue[0]
    assert est_me.point == pytest.approx(m.sides[0].elixir, abs=1e-6)
    mode_b = F.encode_observation(enc.observation(snap, hands[0], nxt, est_me.point, {}), est_opp, guess, db,
                                  F.card_vocab(db))

    tr = OppElixirTracker(db.rules, observer_side=0, evo_cards=db.evo_capable())
    tr.observe_all(m.events)
    sim = F.encode_observation(m.observation(0), tr.estimate(m.t_ms), guess, db, F.card_vocab(db))
    v_len = len(F.card_vocab(db))
    opp_end = 25 + v_len
    hand_end = opp_end + 24
    nxt_end = hand_end + 6
    assert np.allclose(mode_b[:opp_end], sim[:opp_end], atol=1e-6)                       # global, towers, opponent view
    blocks = lambda v: sorted(map(tuple, v[opp_end:hand_end].reshape(4, 6)))  # noqa: E731  (hand order may differ)
    assert blocks(mode_b) == blocks(sim) and np.allclose(mode_b[hand_end:nxt_end], sim[hand_end:nxt_end])
    keep = [k for k in range(8) if k != 6]
    assert np.allclose(mode_b[nxt_end:].reshape(30, 8)[:, keep], sim[nxt_end:].reshape(30, 8)[:, keep], atol=1e-6)


def test_hints_from_known_hand(db, hinter):
    """The dummy prefers the last hand slot (hand is sorted: alphabetically last affordable card) and the tile
    (gx 3, gy 20): troops land on the nearest own-half tile, spells on the tile itself."""
    plays = ["fireball", "cannon", "ice_golem", "the_log"]
    events = [_ev(i, 1000 * (i + 1), 0, c, 9.5, 8.5) for i, c in enumerate(plays)]
    snap = ModeBSnapshot(60_000, HOG, events)
    h = hinter.hints(snap)
    assert h.hand == ("hog_rider", "ice_spirit", "musketeer", "skeletons") and h.hand_hypotheses == 1
    assert h.elixir > 4 and h.p_play > 0.5 and not h.wait
    assert [x.card for x in h.hints] == ["skeletons", "musketeer"]
    for x in h.hints:
        assert F.tile_index(0, x.x, x.y) == x.tile == 14 * 18 + 3
        assert x.value_delta == pytest.approx(0.0, abs=1e-6) and 0 < x.prob <= 1
    assert sum(x.prob for x in h.hints) <= 1 + 1e-6


def test_hints_average_unknown_hands_and_low_elixir(db, hinter):
    h = hinter.hints(ModeBSnapshot(1000, HOG, []))
    assert h.hand is None and h.hand_hypotheses == 70 and len(h.hints) == 2
    assert {x.card for x in h.hints} <= set(HOG)
    spent = [_ev(i, 2500 + i, 0, c, 9.5, 8.5) for i, c in enumerate(["hog_rider", "musketeer"])]
    h = hinter.hints(ModeBSnapshot(2600, HOG, spent))                          # ~0 elixir left
    assert h.hints == [] and h.note == "no card affordable" and h.wait


def test_unsimulated_deck_cards_are_never_suggested(db, hinter):
    deck = ("hog_rider", "musketeer", "cannon", "arrows", "ice_spirit", "skeletons", "giant_snowball", "the_log")
    events = [_ev(i, 1000 * (i + 1), 0, c, 9.5, 8.5) for i, c in enumerate(["cannon", "the_log", "hog_rider",
                                                                           "ice_spirit"])]
    h = hinter.hints(ModeBSnapshot(60_000, deck, events))
    assert "arrows" in h.note and all(x.card in db.cards for x in h.hints) and h.hints


def test_snapshot_from_log_and_companion(db, hinter):
    log = [{"t_ms": 2000, "side": 1, "card": "arrows", "cost": 3, "x": 9.0, "y": 8.0},
           {"t_ms": 3000, "side": 1, "card": "knight", "cost": 3, "x": 9.5, "y": 20.5},
           {"t_ms": 9000, "side": 0, "card": "cannon", "cost": 3, "x": 9.5, "y": 8.5}]
    snap = snapshot_from_log(log, 4000, HOG, db=db)
    assert [e.card_id for e in snap.events] == ["arrows", "knight"]
    assert [u.card for u in snap.units] == ["knight"]                           # spells are not units
    assert hinter.hints(snap).hints

    from backend_engine.perception.live.companion import LiveCompanion
    comp = LiveCompanion.create(db=db)
    assert snapshot_from_companion(comp, HOG) is None                           # clock not started
    comp.start_match(wall=0.0)
    for k in range(12):                                                         # a Knight walks down their lane
        comp.process_tiles([("knight", 1, 3.5, 22.0 - 0.2 * k)], wall=1.0 + 0.1 * k)
    snap = snapshot_from_companion(comp, HOG, wall=2.15)
    assert [(u.card, u.side) for u in snap.units] == [("knight", 1)] and len(snap.events) == 1
    h = hinter.hints(snap)
    assert h.t_ms == 2150 and h.hints


def _ev(i, t, side, card, x, y):
    from backend_engine.contracts import PublicDeployEvent, TilePos
    return PublicDeployEvent(i + 1, t, side, card, CardDB.load().cards[card]["elixir"], TilePos(x, y))
