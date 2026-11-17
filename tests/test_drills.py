"""
Tests for Drills Sandbox:
- Curated drill specs and catalog
- DrillRunner simulation, live evaluation, star grading
- Moment extractor (generating playable drills from match review blunders)
"""
import os
import pytest

from backend_engine.brain.coach_analyzer import Candidate, MoveReview
from backend_engine.drills import (
    DrillRunner,
    DrillSpec,
    drill_from_move_review,
    find_drill,
    get_curated_drills,
)
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig
from backend_engine.simulator.geometry import BRIDGE_X


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


@pytest.fixture
def qapp():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_curated_drills_validity(db):
    """Every curated drill must use valid cards and define sound objectives."""
    drills = get_curated_drills()
    assert len(drills) >= 5

    ids = [d.id for d in drills]
    assert len(set(ids)) == len(ids)

    for d in drills:
        assert d.id and d.title and d.description
        assert 1 <= d.difficulty <= 3
        # Player deck cards must exist in db
        assert all(c in db.cards for c in d.player_deck), d.id
        assert all(c in d.player_deck for c in d.player_hand), d.id
        # Threat cards must exist in db
        assert all(t.card_id in db.cards for t in d.enemy_threats), d.id
        assert d.target_duration_ms > 0

    assert find_drill("drill_pekka_kite") is not None
    assert find_drill("unknown_id") is None


def test_drill_runner_pekka_kite_success(db):
    """Playing Ice Golem and Skeletons in center successfully kites P.E.K.K.A and earns 3 stars."""
    drill = find_drill("drill_pekka_kite")
    assert drill is not None

    runner = DrillRunner(drill, db)
    assert len(runner.threat_uids) == 1
    assert runner.threats_alive() == 1
    assert runner.tower_damage_taken() == 0.0

    # Play Ice Golem at (6.0, 13.5) and Skeletons at (7.5, 11.5) to pull P.E.K.K.A into crossfire
    res = runner.play("ice_golem", 6.0, 13.5)
    assert res.ok
    assert runner.play("skeletons", 7.5, 11.5).ok
    assert runner.elixir_spent == 3.0

    # Step simulation until P.E.K.K.A is eliminated by princess towers
    for _ in range(250):
        runner.step()
        if runner.is_finished():
            break

    result = runner.evaluate()
    assert result.solved
    assert result.stars == 3
    assert result.tower_damage_taken == 0.0
    assert "Perfect" in result.feedback


def test_drill_runner_failure_when_idle(db):
    """Doing nothing allows P.E.K.K.A to hit the tower, failing the drill (0 stars)."""
    drill = find_drill("drill_pekka_kite")
    runner = DrillRunner(drill, db)

    # Do not play any defense
    for _ in range(200):
        runner.step()
        if runner.is_finished() or runner.tower_damage_taken() > 0:
            break

    # P.E.K.K.A deals heavy damage to left Princess Tower
    assert runner.tower_damage_taken() > 500.0
    result = runner.evaluate()
    assert not result.solved
    assert result.stars == 0
    assert "Failed" in result.feedback


def test_drill_from_move_review(db):
    """Extract a drill from a reviewed match blunder moment."""
    deck = ("hog_rider", "cannon", "ice_spirit", "skeletons", "musketeer", "fireball", "the_log", "ice_golem")
    m = Match(MatchConfig((deck, deck), seed=1, shuffle=False), db)

    # Spawn enemy Hog Rider approaching
    m._spawn_card_units(1, "hog_rider", BRIDGE_X[1], 17.5, eid=100, evolved=False)
    m.sides[0].elixir = 5.0
    m.sides[0].hand = ["cannon", "ice_spirit", "skeletons", "musketeer"]

    # Fake blunder move review
    mv = MoveReview(
        event_id=10,
        t_ms=45_000,
        actual=Candidate("skeletons", 3.5, 5.0),
        v_before=0.50,
        v_actual=0.35,
        v_best=0.55,
        v_wait=0.40,
        best=Candidate("cannon", 9.0, 9.0),
        loss=0.20,
        label="blunder",
        n_candidates=10,
        diagnostic="bad_placement",
    )

    drill = drill_from_move_review(mv, m)
    assert drill.category == "custom_moment"
    assert "Bad Placement" in drill.title
    assert drill.player_elixir == 5.0
    assert "cannon" in drill.player_hand
    assert len(drill.enemy_threats) == 1
    assert drill.enemy_threats[0].card_id == "hog_rider"
    assert "cannon" in drill.custom_advice.lower()

    # Can run this extracted drill immediately!
    runner = DrillRunner(drill, db)
    assert runner.threats_alive() == 1
    assert runner.play("cannon", 9.0, 9.0).ok


def test_game_session_drill_integration(db):
    """GameSession runs drills cleanly, reporting drill mode, clock, and evaluating on finish."""
    from frontend_app.qt.session import GameSession
    session = GameSession(db=db)
    assert not session.is_drill()

    drill = find_drill("drill_hog_pull")
    assert drill is not None

    session.start_drill(drill)
    assert session.is_drill()
    clock, phase = session.clock()
    assert "0:10" in clock
    assert "🎯" in phase
    assert "threat" in phase

    # Defend hog
    res = session.play("cannon", 9.0, 9.0)
    assert res.ok

    # Step session
    for _ in range(150):
        session.tick(1)
        if session.result is not None:
            break

    assert session.result is not None
    meta, record = session.to_record("practice")
    assert meta.mode == "drill"
    assert "Drill: Hog 2.6 Golden Cannon Pull" in meta.opp_archetype


def test_drills_dialog_selection(qapp, db):
    """DrillsDialog displays curated drills and triggers callback on start."""
    from frontend_app.qt.drills_dialog import DrillsDialog
    chosen = []
    dlg = DrillsDialog(on_start=lambda d: chosen.append(d))
    assert dlg.isVisible() is False
    drills = get_curated_drills()
    dlg._start_drill(drills[0])
    assert len(chosen) == 1
    assert chosen[0].id == "drill_pekka_kite"


def test_main_window_drill_flow(qapp, db):
    """MainWindow correctly enters drill mode, ticks, evaluates stars, and retries/exits."""
    from frontend_app.qt.app import ENDED, LIVE, SETUP, MainWindow
    win = MainWindow()
    drill = find_drill("drill_witch_surround")
    assert drill is not None

    # Launch drill
    win.start_drill(drill)
    assert win.mode == LIVE
    assert win.running
    assert win.session.is_drill()
    assert win.retry_drill_btn.isHidden()

    # Simulate defense: drop valkyrie right on the witch
    win.selected = "valkyrie"
    win.on_tile(BRIDGE_X[0], 15.5)

    # Step simulation to completion
    for _ in range(250):
        win.on_timer()
        if win.mode == ENDED:
            break

    assert win.mode == ENDED
    assert not win.retry_drill_btn.isHidden()
    assert "Witch Surround" in win.result_sub.text()

    # Retry drill with R
    win.retry_current_drill()
    assert win.mode == LIVE
    assert win.running

    # Exit drill with Esc
    win.exit_drill()
    assert win.mode == SETUP
    assert not win.session.is_drill()
