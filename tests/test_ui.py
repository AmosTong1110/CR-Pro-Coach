import os

import pytest

from backend_engine.contracts import LatencyClass
from frontend_app.qt.session import AI, HUMAN, PERSONAS, GameSession, SessionConfig


@pytest.fixture
def s():
    return GameSession(cfg=SessionConfig(seed=3))


def force_hand(s, card):
    st = s.match.sides[HUMAN]
    if card not in st.hand:
        st.queue.remove(card)
        st.queue.insert(0, st.hand[0])
        st.hand[0] = card
    st.elixir = 10


# ---------------------------------------------------------------- session (no Qt)
def test_play_logs_ai_reaction_time(s):
    force_hand(s, "hog_rider")
    assert s.play("hog_rider", 3.5, 14.5).ok
    e = s.log[-1]
    assert e.side == HUMAN and e.card_id == "hog_rider"
    cls, ms = e.ai_latency
    assert cls == LatencyClass.SURPRISE and 420 <= ms <= 580


def test_ai_vision_flag(s):
    force_hand(s, "knight")
    s.play("knight", 3.5, 10.5)
    u = next(u for u in s.match.units.values() if u.card_id == "knight" and u.side == HUMAN)
    assert not s.ai_has_seen(u)
    s.tick(20)  # 1 s
    assert s.ai_has_seen(u)


def test_illegal_play_reason(s):
    force_hand(s, "knight")
    assert s.play("knight", 9.5, 25.5).reason == "enemy_half"


def test_legal_tiles(s):
    force_hand(s, "knight")
    s.match.sides[HUMAN].elixir = 0  # area still shown while saving elixir
    tiles = set(s.legal_tiles("knight"))
    assert (9.5, 8.5) in tiles and (9.5, 25.5) not in tiles and (9.5, 15.5) not in tiles
    assert s.match.sides[HUMAN].elixir == 0
    force_hand(s, "fireball")
    assert len(s.legal_tiles("fireball")) == 18 * 32


def test_ai_plays_during_match(s):
    s.tick(20 * 40)
    assert any(e.side == AI for e in s.log)


@pytest.mark.parametrize("persona", list(PERSONAS))
def test_personas_and_levels(persona):
    g = GameSession(cfg=SessionConfig(seed=1, persona=persona, level=16))
    assert g.match.tower(HUMAN, "king").max_hp > GameSession(cfg=SessionConfig(seed=1)).match.tower(HUMAN, "king").max_hp
    g.tick(100)


def test_reconfigure_restarts(s):
    s.tick(100)
    s.reconfigure(level=16)
    assert s.match.t_ms == 0 and s.cfg.level == 16 and s.log == []


def test_clock_phases(s):
    assert s.clock() == ("3:00", "")
    s.match.t_ms = 125_000
    assert s.clock()[1] == "x2 圣水"
    s.match.t_ms = 250_000
    assert s.clock()[0] == "0:50" and "x3" in s.clock()[1]


def test_counter_matches_ai_truth(s):
    s.tick(20 * 30)
    assert any(e.side == AI for e in s.log)
    assert s.opp_estimate().point == pytest.approx(s.match.sides[AI].elixir)
    assert s.elixir_advantage() == pytest.approx(s.match.sides[HUMAN].elixir - s.match.sides[AI].elixir)


def test_quiz_scoring(s):
    truth = s.match.sides[AI].elixir
    r = s.quiz_guess(truth + 1.0)
    assert r.error == pytest.approx(1.0)
    s.quiz_guess(truth)
    n, mae, within = s.quiz_stats()
    assert n == 2 and mae == pytest.approx(0.5) and within == 0.5
    s.new_match()
    assert s.quiz_stats()[0] == 2  # training record survives a new match


# ---------------------------------------------------------------- Qt smoke test
def test_main_window_offscreen():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from frontend_app.qt.app import MainWindow

    app = QApplication.instance() or QApplication([])
    w = MainWindow(GameSession(cfg=SessionConfig(seed=4)))
    w.show()
    st = w.session.match.sides[HUMAN]
    slot = next(i for i, c in enumerate(st.hand) if w.session.db.cards[c]["type"] != "spell")
    card = st.hand[slot]
    w.on_hand_slot(slot)
    assert w.selected == card and w.board.legal
    w.on_tile(9.5, 25.5)                     # illegal: stays selected, shows reason
    assert w.selected == card and w.status.text()
    st.elixir = 10
    w.on_tile(9.5, 8.5)
    assert w.selected is None and w.log_list.count() == 1
    w.toggle_run()
    w.speed = 8
    for _ in range(20):
        w.on_timer()
    assert w.session.match.t_ms == 20 * 8 * 50
    assert not w.grab().isNull()
    w.close()
