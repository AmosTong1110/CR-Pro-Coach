"""Tests for MoveReview mistake diagnostics (Overcommitted, Bad Placement, Negative Trade, etc.)."""
import os
from unittest.mock import MagicMock
import pytest

from backend_engine.brain.coach_analyzer import (
    Candidate,
    DIAGNOSTIC_META,
    MatchReview,
    MoveReview,
    diagnose_move,
)
from backend_engine.simulator.card_db import default_card_db
from backend_engine.storage.match_store import review_from_dict, review_to_dict
from frontend_app.qt.session import GameSession, SessionConfig


@pytest.fixture
def db():
    return default_card_db()


def test_diagnose_clean_play(db):
    # Loss <= 0.01 or good/best/brilliant should have no mistake diagnostic
    mv = MoveReview(
        event_id=1, t_ms=5000,
        actual=Candidate("knight", 8.5, 9.5),
        v_before=0.50, v_actual=0.52, v_best=0.52, v_wait=0.48,
        best=Candidate("knight", 8.5, 9.5),
        loss=0.0, label="best", n_candidates=10,
    )
    assert diagnose_move(mv, db=db) is None


def test_diagnose_bad_placement(db):
    # Same card, but tile shifted significantly
    mv = MoveReview(
        event_id=2, t_ms=12000,
        actual=Candidate("cannon", 9.5, 8.5),
        v_before=0.50, v_actual=0.35, v_best=0.55, v_wait=0.40,
        best=Candidate("cannon", 9.5, 12.5),
        loss=0.20, label="blunder", n_candidates=15,
    )
    assert diagnose_move(mv, db=db) == "bad_placement"


def test_diagnose_overcommitted(db):
    # Best play was to wait
    mv_wait = MoveReview(
        event_id=3, t_ms=25000,
        actual=Candidate("hog_rider", 3.5, 16.5),
        v_before=0.55, v_actual=0.38, v_best=0.55, v_wait=0.55,
        best=Candidate(None),  # wait
        loss=0.17, label="mistake", n_candidates=8,
    )
    assert diagnose_move(mv_wait, db=db) == "overcommitted"


def test_diagnose_ignored_threat(db):
    # Fake a pre-play match where an enemy troop is on our side (y <= 16.5)
    mock_match = MagicMock()
    mock_enemy = MagicMock()
    mock_enemy.side = 1
    mock_enemy.hp = 500
    mock_enemy.kind = "troop"
    mock_enemy.y = 12.0
    mock_match.units = {101: mock_enemy}

    # Player dropped an attack card into enemy half (y >= 16)
    mv = MoveReview(
        event_id=4, t_ms=30000,
        actual=Candidate("hog_rider", 3.5, 17.5),
        v_before=0.45, v_actual=0.25, v_best=0.50, v_wait=0.42,
        best=Candidate("cannon", 9.5, 12.5),
        loss=0.25, label="blunder", n_candidates=12,
    )
    diag = diagnose_move(mv, db=db, match_before=mock_match, side=0)
    assert diag == "ignored_threat"


def test_diagnose_negative_trade(db):
    # Player played rocket (6 elixir) when zap (2 elixir) or the_log (2 elixir) was best
    mv = MoveReview(
        event_id=5, t_ms=40000,
        actual=Candidate("rocket", 9.5, 12.5),
        v_before=0.50, v_actual=0.42, v_best=0.52, v_wait=0.45,
        best=Candidate("the_log", 9.5, 12.5),
        loss=0.10, label="mistake", n_candidates=10,
    )
    assert diagnose_move(mv, db=db) == "negative_trade"


def test_diagnose_missed_cycle(db):
    # Player played heavy card (cost 4) when 1-2 elixir cycle card was optimal
    mv = MoveReview(
        event_id=6, t_ms=50000,
        actual=Candidate("musketeer", 8.5, 6.5),  # 4 elixir
        v_before=0.50, v_actual=0.42, v_best=0.52, v_wait=0.48,
        best=Candidate("ice_spirit", 8.5, 6.5),   # 1 elixir
        loss=0.10, label="mistake", n_candidates=10,
    )
    assert diagnose_move(mv, db=db) == "missed_cycle"


def test_diagnose_wrong_response(db):
    # Same elixir range, but wrong counter card
    mv = MoveReview(
        event_id=7, t_ms=60000,
        actual=Candidate("knight", 8.5, 9.5),    # 3 elixir
        v_before=0.50, v_actual=0.41, v_best=0.53, v_wait=0.44,
        best=Candidate("cannon", 9.5, 12.5),     # 3 elixir
        loss=0.12, label="mistake", n_candidates=10,
    )
    assert diagnose_move(mv, db=db) == "wrong_response"


def test_diagnostics_breakdown_and_primary_leak(db):
    m1 = MoveReview(1, 10000, Candidate("cannon", 9.5, 8.5), 0.5, 0.35, 0.55, 0.40,
                    Candidate("cannon", 9.5, 12.5), 0.20, "blunder", 10, diagnostic="bad_placement")
    m2 = MoveReview(2, 20000, Candidate("hog_rider", 3.5, 16.5), 0.55, 0.38, 0.55, 0.55,
                    Candidate(None), 0.17, "mistake", 8, diagnostic="overcommitted")
    m3 = MoveReview(3, 30000, Candidate("musketeer", 3.5, 16.5), 0.50, 0.35, 0.50, 0.50,
                    Candidate(None), 0.15, "mistake", 8, diagnostic="overcommitted")
    m4 = MoveReview(4, 40000, Candidate("knight", 8.5, 9.5), 0.50, 0.50, 0.50, 0.45,
                    Candidate("knight", 8.5, 9.5), 0.0, "best", 10)

    review = MatchReview([(0, 0.5), (60000, 0.4)], [m1, m2, m3, m4])
    bd = review.diagnostics_breakdown(db)

    assert bd.get("overcommitted") == 2
    assert bd.get("bad_placement") == 1
    assert "best" not in bd

    leak = review.primary_leak_summary(db)
    assert leak is not None
    assert "Overcommitted" in leak
    assert "67%" in leak  # 2 out of 3 mistakes


def test_serialization_roundtrip():
    mv = MoveReview(
        event_id=1, t_ms=15000,
        actual=Candidate("cannon", 9.5, 8.5),
        v_before=0.50, v_actual=0.35, v_best=0.55, v_wait=0.40,
        best=Candidate("cannon", 9.5, 12.5),
        loss=0.20, label="blunder", n_candidates=10,
        diagnostic="bad_placement",
    )
    review = MatchReview([(0, 0.5), (30000, 0.4)], [mv])
    d = review_to_dict(review)

    assert d["moves"][0]["diagnostic"] == "bad_placement"

    r2 = review_from_dict(d)
    assert r2.moves[0].diagnostic == "bad_placement"


# ---------------------------------------------------------------------------
# Qt UI Tests
# ---------------------------------------------------------------------------

@pytest.fixture
def qapp():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_ui_review_panel_diagnostic_rendering(qapp):
    from frontend_app.qt.review import ReviewPanel, describe, tooltip_html

    names = lambda cid: cid.replace("_", " ").title() if cid else "Wait"
    mv = MoveReview(
        event_id=1, t_ms=20000,
        actual=Candidate("hog_rider", 3.5, 16.5),
        v_before=0.55, v_actual=0.38, v_best=0.55, v_wait=0.55,
        best=Candidate(None),
        loss=0.17, label="mistake", n_candidates=8,
        diagnostic="overcommitted",
    )
    desc = describe(mv, names)
    assert "Overcommitted" in desc
    assert "Spent elixir" in desc

    tip = tooltip_html(mv, names)
    assert "Overcommitted" in tip

    panel = ReviewPanel(names, lambda: None, lambda: None)
    review = MatchReview([(0, 0.5), (30000, 0.4)], [mv])
    panel.set_review(review, 30000)

    # Check that item in list contains the diagnostic name
    assert panel.list.count() == 1
    item_text = panel.list.item(0).text()
    assert "[Overcommitted]" in item_text
    assert "Overcommitted" in panel.summary.text()

    panel.close()
    qapp.processEvents()


def test_ui_analysis_page_diagnostic_rendering(qapp, tmp_path):
    from frontend_app.qt.shell import CRbotWindow

    session = GameSession(cfg=SessionConfig(seed=42))
    win = CRbotWindow(session, data_dir=tmp_path)
    win.show()
    win.show_page("analysis")

    analysis_page = win.analysis

    mv = MoveReview(
        event_id=1, t_ms=20000,
        actual=Candidate("cannon", 9.5, 8.5),
        v_before=0.50, v_actual=0.35, v_best=0.55, v_wait=0.40,
        best=Candidate("cannon", 9.5, 12.5),
        loss=0.20, label="blunder", n_candidates=10,
        diagnostic="bad_placement",
    )
    review = MatchReview([(0, 0.5), (30000, 0.4)], [mv])

    analysis_page.review = review
    analysis_page._populate_moments()

    assert analysis_page.moments_list.count() == 1
    assert "[Bad Placement]" in analysis_page.moments_list.item(0).text()
    assert not analysis_page.leak_banner.isHidden()
    assert "Bad Placement" in analysis_page.leak_banner.text()

    win.close()
    qapp.processEvents()
