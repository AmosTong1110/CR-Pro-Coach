import os

import pytest

from backend_engine.brain.coach_analyzer import (
    BLUNDER, INACCURACY, MISTAKE, Candidate, CoachAnalyzer, classify,
)
from backend_engine.brain.scripted_bot import ScriptedBot
from backend_engine.brain.value_model import FEATURES, ValueModel, default_value_model, features
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig
from backend_engine.simulator.match_runner import DEFAULT_DECK
from frontend_app.qt.session import AI, HUMAN, GameSession, SessionConfig


def played_session(seed=31, until_ms=60_000):
    """A session where a scripted bot plays the human side."""
    s = GameSession(cfg=SessionConfig(seed=seed))
    bot = ScriptedBot(s.db, HUMAN, 0.6)
    while s.result is None and s.match.t_ms < until_ms:
        if s.match.t_ms % 1500 == 0:
            for i in bot.decide(s.match.observation(HUMAN), [], s.match.t_ms):
                s.play(i.card_id, i.tile.x, i.tile.y)
        s.tick(1)
    return s


# ---------------------------------------------------------------- value model
def test_value_model_trained_and_symmetric():
    vm = default_value_model()
    assert vm.meta.get("trained") and vm.meta["test_log_loss"] < vm.meta["coin_flip_log_loss"]
    m = Match(MatchConfig((DEFAULT_DECK, DEFAULT_DECK)), CardDB.load())
    assert vm.value(m, 0) == pytest.approx(0.5) and vm.value(m, 1) == pytest.approx(0.5)
    m.tower(1, "princess_left").hp *= 0.3
    assert vm.value(m, 0) > 0.6
    assert vm.value(m, 0) + vm.value(m, 1) == pytest.approx(1.0)
    assert len(features(m)) == len(FEATURES)


def test_value_of_finished_match():
    m = Match(MatchConfig((DEFAULT_DECK, DEFAULT_DECK)), CardDB.load())
    m.tower(1, "king").hp = 0
    m._resolve_deaths()
    m._check_end()
    assert default_value_model().value(m, 0) == 1.0


def test_fallback_when_weights_missing(tmp_path):
    assert ValueModel.load(tmp_path / "nope.json").meta == {"trained": False}


# ---------------------------------------------------------------- grading rules
@pytest.mark.parametrize("loss,label", [
    (BLUNDER, "blunder"), (MISTAKE, "mistake"), (INACCURACY, "inaccuracy"), (0.03, "good"), (0.0, "best"),
])
def test_classify_thresholds(loss, label):
    assert classify(loss, 0.5, [0.5, 0.5], 0.5) == label


def test_brilliant_needs_a_standout_move():
    assert classify(0.0, 0.8, [0.4, 0.45, 0.5], 0.4) == "brilliant"
    assert classify(0.0, 0.8, [0.78, 0.79, 0.8], 0.4) == "best"


# ---------------------------------------------------------------- analyzer
def test_candidates_are_legal_and_include_wait():
    s = played_session(until_ms=20_000)
    an = CoachAnalyzer(s.fork)
    cands = an.candidates(s.match)
    assert cands[0].is_wait
    for c in cands[1:]:
        assert s.match.check_play(HUMAN, c.card_id, c.x, c.y) is None


def test_review_is_deterministic_and_never_touches_live_match():
    s = played_session()
    h = s.match.state_hash()
    r1 = s.analyze()
    assert s.match.state_hash() == h
    r2 = s.analyze()
    assert [(m.event_id, round(m.loss, 9), m.label) for m in r1.moves] == \
           [(m.event_id, round(m.loss, 9), m.label) for m in r2.moves]


def test_review_contents():
    s = played_session()
    r = s.analyze()
    human_plays = [e for e in s.log if e.side == HUMAN]
    assert len(r.moves) == len(human_plays) > 0
    for mv in r.moves:
        assert mv.v_best >= mv.v_actual - 1e-9 and mv.loss >= 0
        assert mv.n_candidates > 3
        assert 0 <= mv.v_before <= 1
    assert r.curve[0] == (0, pytest.approx(0.5))
    assert 0 <= r.accuracy <= 100 and sum(r.counts().values()) == len(r.moves)


def test_best_move_really_is_better():
    """Replaying the recommended move from the same state reaches the predicted value."""
    s = played_session()
    r = s.analyze()
    mv = max(r.moves, key=lambda m: m.loss)
    an = CoachAnalyzer(s.fork)
    cp = s.pre_play[mv.event_id]
    assert an.rollout(cp, mv.best) == pytest.approx(mv.v_best)
    assert an.rollout(cp, mv.actual) == pytest.approx(mv.v_actual)


def test_restore_before_event_and_take_over():
    s = played_session()
    s.analyze()
    mv = s.review.moves[len(s.review.moves) // 2]
    t = s.restore_before_event(mv.event_id)
    assert t == mv.t_ms
    assert all(ev.event_id != mv.event_id for ev in s.match.events)
    assert mv.actual.card_id in s.match.sides[HUMAN].hand
    assert s.can_forward()


# ---------------------------------------------------------------- UI
def test_review_panel_offscreen():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from frontend_app.qt.app import MainWindow

    app = QApplication.instance() or QApplication([])
    w = MainWindow(played_session(until_ms=40_000))
    w.show()
    w.run_review()
    panel = w.review_panel
    assert w.review_dock.isVisible() and panel.review is not None
    assert panel.list.count() == len(panel.review.moves)
    i = max(range(len(panel.review.moves)), key=lambda k: panel.review.moves[k].loss)
    panel.select(i)
    mv = panel.review.moves[i]
    assert w.session.match.t_ms == mv.t_ms and w.board.ghosts[0][3] == "actual"
    assert w.board.banner and panel.take_btn.isEnabled()
    panel.take_btn.click()
    assert w.countdown_ms > 0
    assert not w.grab().isNull()
    w.close()
