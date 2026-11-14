import os

import pytest

from frontend_app.qt.session import AI, HUMAN, GameSession, SessionConfig


def session(seed=5):
    return GameSession(cfg=SessionConfig(seed=seed))


def force_hand(s, card):
    st = s.match.sides[HUMAN]
    if card not in st.hand:
        st.queue.remove(card)
        st.queue.insert(0, st.hand[0])
        st.hand[0] = card
    st.elixir = 10


def test_rewind_then_replay_is_identical():
    s = session()
    s.tick(20 * 30)
    h, n_log = s.match.state_hash(), len(s.log)
    assert s.rewind(10) == 20_000
    assert s.match.t_ms == 20_000 and all(e.t_ms < 20_000 for e in s.log)
    s.tick(20 * 10)
    assert s.match.state_hash() == h and len(s.log) == n_log


def test_forward_restores_abandoned_future():
    s = session()
    s.tick(20 * 30)
    h = s.match.state_hash()
    s.rewind(10)
    assert s.can_forward()
    assert s.forward(10) == 30_000 and s.match.state_hash() == h
    assert not s.can_forward()


def test_forward_in_steps():
    s = session()
    s.tick(20 * 40)
    s.seek(5_000)
    assert s.forward(10) == 15_000
    assert s.forward(10) == 25_000
    assert s.max_known_ms() == 40_000


def test_new_play_discards_redo_branch():
    s = session()
    s.tick(20 * 30)
    s.rewind(10)
    force_hand(s, "knight")
    assert s.play("knight", 9.5, 8.5).ok
    assert not s.can_forward() and s.forward(10) is None


def test_rewind_restores_ai_eyes_and_counters():
    s = session(7)
    s.tick(20 * 10)
    force_hand(s, "hog_rider")
    s.play("hog_rider", 3.5, 14.5)
    ev = s.match.events[-1]
    s.tick(20 * 5)
    assert s.ai_buffer.is_visible(ev.event_id)
    s.seek(10_000)
    assert ev not in s.match.events                     # the play is undone...
    assert not s.ai_buffer.is_visible(ev.event_id)      # ...and the AI "forgets" it saw it
    assert "hog_rider" not in s.runner.trackers[AI].estimate(s.match.t_ms).cycle_order
    assert "hog_rider" in s.match.sides[HUMAN].hand and s.match.sides[HUMAN].plays == 0


def test_take_over_before_a_play():
    s = session(8)
    s.tick(20 * 12)
    force_hand(s, "hog_rider")
    s.play("hog_rider", 3.5, 14.5)
    i = len(s.log) - 1
    t_play = s.log[i].t_ms
    s.tick(20 * 20)
    ev_id = s.log[i].event_id
    t = s.take_over_before(i)
    assert t == t_play and s.match.t_ms == t                      # the exact moment, not a second early
    assert all(ev.event_id != ev_id for ev in s.match.events)    # with the play undone
    assert all(e.t_ms <= t for e in s.log)
    assert "hog_rider" in s.match.sides[HUMAN].hand or "hog_rider" in s.match.sides[HUMAN].queue


def test_rewind_from_result_screen():
    s = session()
    s.tick(20 * 400)
    assert s.result is not None
    end = s.result.t_ms
    assert s.rewind(10) is not None
    assert s.result is None and s.match.t_ms <= end - 9_000
    s.tick(20 * 30)
    assert s.result is not None and s.result.t_ms == end  # same future if you don't intervene


def test_cannot_rewind_before_start():
    s = session()
    assert s.seek(0) == 0
    assert s.rewind(10) == 0


def test_new_match_clears_timeline():
    s = session()
    s.tick(20 * 20)
    s.new_match()
    assert [cp.t_ms for cp in s.timeline] == [0] and s.redo == []


def test_window_rewind_countdown_and_take_over():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from frontend_app.qt.app import COUNTDOWN_MS, MainWindow

    app = QApplication.instance() or QApplication([])
    w = MainWindow(session(9))
    w.show()
    w.session.tick(20 * 30)
    w.add_log([])
    w.refresh()
    assert w.rew_btn.isEnabled() and not w.fwd_btn.isEnabled()
    w.rew_btn.click()
    assert w.session.match.t_ms == 20_000 and w.board.overlay_text == "3" and not w.running
    assert w.fwd_btn.isEnabled() and w.slider.maximum() >= 30_000
    # plays are blocked during the countdown
    st = w.session.match.sides[HUMAN]
    slot = next(i for i, c in enumerate(st.hand) if w.session.db.cards[c]["type"] != "spell")
    w.on_hand_slot(slot)
    w.on_tile(9.5, 8.5)
    assert "倒计时" in w.status.text()
    for _ in range(COUNTDOWN_MS // 50):
        w.on_timer()
    assert w.running and w.board.overlay_text is None and w.session.match.t_ms == 20_000
    w.on_timer()
    assert w.session.match.t_ms > 20_000
    # take over from the first log line
    if w.log_list.count():
        w.take_over(w.log_list.item(0))
        assert w.countdown_ms == COUNTDOWN_MS and w.log_list.count() == len(w.session.log)
    w.close()
