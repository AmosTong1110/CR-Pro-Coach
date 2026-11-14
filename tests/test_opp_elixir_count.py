import pytest

from backend_engine.brain.scripted_bot import ScriptedBot
from backend_engine.contracts import PublicDeployEvent, TilePos
from backend_engine.perception.opp_elixir_count import ElixirSchedule, OppElixirTracker
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig
from backend_engine.simulator.match_runner import DEFAULT_DECK, MatchRunner

OPP = 1


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


@pytest.fixture
def tr(db):
    return OppElixirTracker(db.rules, observer_side=0, evo_cards=["knight"])


_id = [0]


def play(card, t_ms, cost, side=OPP, evolved=False, ability=False):
    _id[0] += 1
    return PublicDeployEvent(_id[0], t_ms, side, card, cost, TilePos(9, 25), is_evolved=evolved, is_ability=ability)


# ---------------------------------------------------------------- elixir
def test_start_and_regen(tr):
    assert tr.estimate(0).point == 5
    assert tr.estimate(2800).point == pytest.approx(6)
    assert tr.estimate(14_000).point == 10          # 5 + 14/2.8 = 10 exactly
    assert tr.estimate(16_800).leaked == pytest.approx(1)


def test_phase_boundary_regen(db):
    s = ElixirSchedule.from_rules(db.rules)
    v, _ = s.advance(0.0, 118_600, 121_400)          # 1.4 s at 1x + 1.4 s at 2x
    assert v == pytest.approx(0.5 + 1.0)


def test_spend(tr):
    tr.observe(play("hog_rider", 2800, 4))
    e = tr.estimate(2800)
    assert e.point == pytest.approx(2) and e.low == e.high == e.point


def test_own_plays_ignored(tr):
    tr.observe(play("knight", 1000, 3, side=0))
    assert tr.estimate(1000).point > 5


def test_out_of_order_events(tr):
    a, b = play("knight", 1000, 3), play("archers", 6000, 3)
    tr.observe(b)
    tr.observe(a)
    tr.observe(a)  # duplicate ignored
    assert tr.estimate(6000).cycle_order == ("knight", "archers")
    assert tr.estimate(6000).point == pytest.approx(5 + 6 / 2.8 - 6)


def test_resync_when_play_proves_model_wrong(tr):
    tr.observe(play("golem", 0, 8))   # impossible with 5: we must have missed something
    e = tr.estimate(0)
    assert e.resyncs == 1 and e.point == 0


def test_ability_costs_elixir_but_not_a_cycle_play(tr):
    tr.observe(play("little_prince", 0, 3))
    tr.observe(play("little_prince", 100, 1, ability=True))
    e = tr.estimate(100)
    assert e.cycle_order == ("little_prince",)
    assert e.point == pytest.approx(5 + 100 / 2800 - 4)


def test_slack_widens_bounds(db):
    t = OppElixirTracker(db.rules, observer_side=0, slack=0.5)
    e = t.estimate(2800)
    assert (e.low, e.point, e.high) == pytest.approx((5.5, 6, 6.5))


# ---------------------------------------------------------------- cycle
def test_cycle_queue_and_hand(tr):
    cards = ["hog_rider", "knight", "cannon", "the_log", "fireball"]
    for i, c in enumerate(cards):
        tr.observe(play(c, i * 10_000, 1))
    e = tr.estimate(60_000)
    assert e.queue == ("knight", "cannon", "the_log", "fireball")
    assert e.next_return == "knight"
    assert e.known_in_hand == ("hog_rider",)
    assert e.in_hand_prob["hog_rider"] == 1.0 and e.in_hand_prob["fireball"] == 0.0
    assert e.unknown_hand_slots == 3


def test_early_queue_has_unknowns(tr):
    tr.observe(play("hog_rider", 0, 4))
    e = tr.estimate(0)
    assert e.queue == (None, None, None, "hog_rider") and e.next_return is None
    assert e.unknown_hand_slots == 4


def test_full_deck_revealed(tr):
    deck = list(DEFAULT_DECK)
    for i, c in enumerate(deck + deck[:2]):
        tr.observe(play(c, i * 5000, 1))
    e = tr.estimate(60_000)
    assert len(e.known_cards) == 8 and e.unknown_hand_slots == 0 and len(e.known_in_hand) == 4


def test_evo_progress(tr):
    for i, evolved in enumerate([False, False, True, False]):
        tr.observe(play("knight", i * 20_000, 3, evolved=evolved))
    assert tr.estimate(90_000).evo_progress == {"knight": 1}


def test_snapshot_restore(tr):
    tr.observe(play("knight", 0, 3))
    snap = tr.snapshot()
    tr.observe(play("hog_rider", 5000, 4))
    tr.restore(snap)
    assert tr.estimate(5000).cycle_order == ("knight",)


# ---------------------------------------------------------------- integration
def test_exact_against_engine_for_whole_match(db):
    m = Match(MatchConfig((DEFAULT_DECK, DEFAULT_DECK), seed=12), db)
    r = MatchRunner(m, {0: ScriptedBot(db, 0, 0.7), 1: ScriptedBot(db, 1, 0.4)})
    t = OppElixirTracker(db.rules, observer_side=0)
    worst = 0.0
    while m.result is None:
        r.tick()
        t.observe_all(m.events)
        worst = max(worst, abs(t.estimate(m.t_ms).point - m.sides[1].elixir))
    assert worst < 1e-9
    assert t.estimate(m.t_ms).leaked == pytest.approx(m.sides[1].leaked_elixir)


def test_ai_tracker_only_learns_after_reaction_delay(db):
    m = Match(MatchConfig((DEFAULT_DECK, DEFAULT_DECK), seed=13, shuffle=False), db)
    r = MatchRunner(m, {0: None, 1: ScriptedBot(db, 1)})
    st = m.sides[0]
    st.elixir = 10
    card = next(c for c in st.hand if db.cards[c]["type"] == "troop")
    r.human_play(card, 9.5, 8.5)
    r.tick()
    assert r.trackers[1].estimate(m.t_ms).cycle_order == ()      # not perceived yet
    for _ in range(25):
        r.tick()
    assert r.trackers[1].estimate(m.t_ms).cycle_order == (card,)
