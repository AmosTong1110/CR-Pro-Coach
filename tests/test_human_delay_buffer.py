import random
import statistics

import pytest

from backend_engine.brain.human_delay_buffer import HumanDelayBuffer, LatencyDist, ReactionProfile
from backend_engine.contracts import (
    ActionIntent,
    ActionKind,
    ElixirEstimate,
    HoverPlan,
    LatencyClass,
    PublicDeployEvent,
    TilePos,
)

AI, HUMAN = 1, 0
_eid = 0


def ev(card, t_ms, x=4.0, y=14.0, side=HUMAN, threat=True, cost=3):
    global _eid
    _eid += 1
    return PublicDeployEvent(_eid, t_ms, side, card, cost, TilePos(x, y), is_threat=threat)


def deploy(card, cost, x=9.0, y=25.0):
    return ActionIntent(ActionKind.DEPLOY, card, TilePos(x, y), cost)


@pytest.fixture
def buf():
    return HumanDelayBuffer(ReactionProfile.default(), side=AI, seed=42)


# ---------------------------------------------------------------- distributions
def test_latency_dist_truncated_and_centered():
    d = LatencyDist(500, 40, 420, 580)
    rng = random.Random(0)
    xs = [d.sample(rng) for _ in range(5000)]
    assert min(xs) >= 420 and max(xs) <= 580
    assert 490 <= statistics.mean(xs) <= 510


def test_zero_std_is_deterministic():
    assert LatencyDist(0, 0, 0, 0).sample(random.Random(1)) == 0


def test_bad_dist_rejected():
    with pytest.raises(ValueError):
        LatencyDist(100, 10, 200, 100)


# ---------------------------------------------------------------- layer 1: perception
def test_surprise_threat_hidden_until_reaction_window(buf):
    e = ev("hog_rider", 10_000)
    assert buf.observe(e) == LatencyClass.SURPRISE
    assert buf.pop_visible_events(10_419) == []
    assert not buf.is_visible(e.event_id)
    assert buf.pop_visible_events(10_580) == [e]
    assert buf.is_visible(e.event_id)
    cls, delay = buf.latency_of(e.event_id)
    assert cls == LatencyClass.SURPRISE and 420 <= delay <= 580


def test_anticipated_play_is_seen_faster(buf):
    buf.set_expectations({"hog_rider": 0.8})
    e = ev("hog_rider", 0)
    assert buf.observe(e) == LatencyClass.ANTICIPATED
    assert buf.pop_visible_events(420) == [e]


def test_routine_back_play(buf):
    e = ev("golem", 0, y=2.0, threat=False, cost=8)
    assert buf.observe(e) == LatencyClass.ROUTINE
    assert buf.pop_visible_events(399) == []
    assert buf.pop_visible_events(1000) == [e]


def test_own_plays_visible_immediately(buf):
    e = ev("knight", 0, side=AI)
    assert buf.observe(e) == LatencyClass.OWN_INITIATIVE
    assert buf.is_visible(e.event_id)


def test_events_pop_in_visibility_order(buf):
    a = ev("golem", 0, threat=False)      # routine, >= 400
    b = ev("hog_rider", 0)                 # surprise, 420-580
    buf.observe(a)
    buf.observe(b)
    got = buf.pop_visible_events(2000)
    assert set(x.event_id for x in got) == {a.event_id, b.event_id}
    delays = [buf.latency_of(x.event_id)[1] for x in got]
    assert delays == sorted(delays)


# ---------------------------------------------------------------- layer 2: hover reflex
def log_hover(**kw):
    base = dict(card_id="the_log", elixir_cost=2, trigger_cards=frozenset({"goblin_barrel"}),
                target_mode="event", offset=(0.0, 1.0))
    base.update(kw)
    return HoverPlan(**base)


def test_hover_reflex_fires_fast_at_event_tile(buf):
    assert buf.arm_hover(log_hover(), now_ms=0)
    e = ev("goblin_barrel", 5_000, x=3.5, y=25.0)
    assert buf.observe(e) == LatencyClass.HOVER_REFLEX
    (a,) = buf.release_due(5_200)
    assert a.intent.card_id == "the_log" and a.intent.source == "hover"
    assert 5_120 <= a.execute_at_ms <= 5_200
    assert a.intent.tile == TilePos(3.5, 26.0)
    assert a.trigger_event_id == e.event_id
    assert buf.hover is None  # consumed


def test_hover_used_once(buf):
    buf.arm_hover(log_hover(), 0)
    buf.observe(ev("goblin_barrel", 1000))
    assert buf.observe(ev("goblin_barrel", 9000)) != LatencyClass.HOVER_REFLEX


def test_hover_lane_and_card_mismatch(buf):
    buf.arm_hover(log_hover(trigger_lane="right"), 0)
    assert buf.observe(ev("goblin_barrel", 100, x=3.0)) != LatencyClass.HOVER_REFLEX  # left lane
    assert buf.observe(ev("hog_rider", 200, x=14.0)) != LatencyClass.HOVER_REFLEX     # wrong card
    assert buf.hover is not None
    assert buf.observe(ev("goblin_barrel", 300, x=14.0)) == LatencyClass.HOVER_REFLEX


def test_hover_expires(buf):
    buf.arm_hover(log_hover(expires_at_ms=1000), 0)
    assert buf.observe(ev("goblin_barrel", 1001)) != LatencyClass.HOVER_REFLEX
    assert buf.hover is None


def test_fixed_hover_needs_tile(buf):
    with pytest.raises(ValueError):
        buf.arm_hover(log_hover(target_mode="fixed"), 0)


def test_princess_anti_steal_fixed_hover(buf):
    plan = log_hover(trigger_cards=frozenset({"princess"}), target_mode="fixed", tile=TilePos(3.5, 20.0))
    buf.arm_hover(plan, 0)
    buf.observe(ev("princess", 2000, x=3.5, y=10.0, threat=False))
    (a,) = buf.release_due(2200)
    assert a.intent.tile == TilePos(3.5, 20.0)


def test_cannot_arm_hover_while_finger_busy(buf):
    buf.submit(deploy("knight", 3), now_ms=0)
    assert not buf.arm_hover(log_hover(), now_ms=100)
    assert buf.arm_hover(log_hover(), now_ms=buf.finger_free_at_ms())


def test_submit_cancels_hover(buf):
    buf.arm_hover(log_hover(), 0)
    buf.submit(deploy("knight", 3), 10)
    assert buf.hover is None


# ---------------------------------------------------------------- layer 3: finger / APM
def test_combo_double_drop_default_human_gap(buf):
    out = buf.submit([deploy("ice_golem", 2), deploy("hog_rider", 4)], now_ms=1000)
    assert [a.execute_at_ms for a in out] == [1000, 1120]
    assert out[0].combo_id == out[1].combo_id is not None
    assert len(buf.release_due(1000)) == 1
    assert len(buf.release_due(1120)) == 1


def test_combo_gap_configurable():
    p = ReactionProfile.default()
    b = HumanDelayBuffer(ReactionProfile(**{**p.__dict__, "combo_gap_ms": 0}), side=AI, seed=0)
    out = b.submit([deploy("a", 1), deploy("b", 1), deploy("c", 1)], 0)
    assert [a.execute_at_ms for a in out] == [0, 0, 0]


def test_yaml_default_combo_gap():
    pytest.importorskip("yaml")
    assert ReactionProfile.load().combo_gap_ms == 120
    assert ReactionProfile.load(persona="superhuman_debug").combo_gap_ms == 0


def test_apm_gap_between_separate_decisions(buf):
    (a,) = buf.submit(deploy("knight", 3), 1000)
    (b,) = buf.submit(deploy("musketeer", 4), 1000)  # decided same instant, separate decision
    assert 180 <= b.execute_at_ms - a.execute_at_ms <= 300


def test_noop_not_scheduled(buf):
    assert buf.submit(ActionIntent(ActionKind.NOOP), 0) == []
    assert buf.finger_free_at_ms() == 0


# ---------------------------------------------------------------- elixir gating
def test_cumulative_elixir_check_holds_rest_of_combo(buf):
    buf.submit([deploy("hog_rider", 4), deploy("knight", 3)], 0)
    got = buf.release_due(0, elixir=5.0)
    assert [a.intent.card_id for a in got] == ["hog_rider"]
    assert [a.intent.card_id for a in buf.pending()] == ["knight"]
    got = buf.release_due(800, elixir=3.2)
    assert [a.intent.card_id for a in got] == ["knight"]


def test_unaffordable_action_dropped_after_hold(buf):
    buf.submit(deploy("golem", 8), 0)
    assert buf.release_due(0, elixir=2) == []
    assert buf.release_due(1499, elixir=2) == []
    assert buf.release_due(1500, elixir=2) == []
    assert buf.pending() == [] and buf.dropped[0].intent.card_id == "golem"


def test_is_legal_callback(buf):
    buf.submit(deploy("knight", 3), 0)
    assert buf.release_due(0, is_legal=lambda i: False) == []
    assert len(buf.release_due(10, is_legal=lambda i: True)) == 1


def test_next_wakeup(buf):
    assert buf.next_wakeup_ms() is None
    buf.observe(ev("hog_rider", 0))
    buf.submit(deploy("knight", 3), 100)
    assert buf.next_wakeup_ms() == 100


# ---------------------------------------------------------------- determinism / snapshots
def _script(b):
    b.observe(ev("hog_rider", 0))
    b.arm_hover(log_hover(), 0)
    b.observe(ev("goblin_barrel", 300))
    b.submit(deploy("knight", 3), 600)
    return [(a.execute_at_ms, a.intent.card_id) for a in b.pending()] + [b.finger_free_at_ms()]


def test_same_seed_same_timeline():
    assert _script(HumanDelayBuffer(seed=7)) == _script(HumanDelayBuffer(seed=7))


def test_snapshot_restore_replays_identically(buf):
    buf.observe(ev("hog_rider", 0))
    snap = buf.snapshot()
    first = _script(buf)
    buf.restore(snap)
    assert _script(buf) == first


def test_reset_clears_everything(buf):
    _script(buf)
    buf.reset()
    assert buf.pending() == [] and buf.next_wakeup_ms() is None and buf.hover is None


# ---------------------------------------------------------------- profile yaml / contracts
def test_yaml_profile_and_personas():
    pytest.importorskip("yaml")
    base = ReactionProfile.load()
    assert base.surprise.min == 420 and base.surprise.max == 580
    surgeon = ReactionProfile.load(persona="surgeon")
    assert surgeon.surprise.max == 540 and surgeon.routine == base.routine
    fast = ReactionProfile.load(persona="superhuman_debug")
    b = HumanDelayBuffer(fast, side=AI)
    e = ev("hog_rider", 0)
    b.observe(e)
    assert b.pop_visible_events(0) == [e]
    with pytest.raises(KeyError):
        ReactionProfile.load(persona="nope")


def test_elixir_estimate_feeds_expectations(buf):
    est = ElixirEstimate(t_ms=0, point=5, low=3, high=7, known_cards=("hog_rider",),
                         cycle_order=(), in_hand_prob={"hog_rider": 1.0, "golem": 1.0})
    probs = est.playable_probs({"hog_rider": 4, "golem": 8})
    assert probs["hog_rider"] == pytest.approx(0.75) and probs["golem"] == 0.0
    buf.set_expectations(probs)
    assert buf.observe(ev("hog_rider", 0)) == LatencyClass.ANTICIPATED
