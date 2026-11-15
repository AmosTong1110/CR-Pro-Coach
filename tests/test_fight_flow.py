"""Fight page states: setup → live → ended → review, and what each one shows."""
import os

import pytest

from frontend_app.qt.session import AI, HUMAN, GameSession, SessionConfig


@pytest.fixture
def win():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from frontend_app.qt.app import MainWindow

    app = QApplication.instance() or QApplication([])
    w = MainWindow(GameSession(cfg=SessionConfig(seed=11)))
    w.auto_review = False
    w.show()
    yield w
    w.close()
    app.processEvents()


def play_one(w):
    st = w.session.match.sides[HUMAN]
    st.elixir = 10
    slot = next(i for i, c in enumerate(st.hand) if w.session.db.cards[c]["type"] != "spell")
    w.on_hand_slot(slot)
    w.on_tile(9.5, 8.5)
    assert w.log_list.count() == 1


def finish(w):
    w.speed = 400                              # 20 s of match time per timer tick
    for _ in range(40):
        w.on_timer()
        if w.session.result is not None:
            break
    assert w.session.result is not None


def test_setup_shows_settings_only_before_start(win):
    w = win
    assert w.mode == "setup"
    assert w.setup_box.isVisible() and w.persona_box.isVisible() and w.start_btn.isVisible()
    assert not w.ended_box.isVisible() and not w.review_panel.isVisible()
    assert not hasattr(w, "speed_box")                                    # no speed control on the Fight page
    w.run_review()
    assert w.mode == "setup"                                              # review only after a match
    w.persona_box.setCurrentIndex(w.persona_box.findData("aggressor"))   # settings work before the start
    assert w.session.cfg.persona == "aggressor" and w.mode == "setup"


def test_start_hides_every_setup_button(win):
    w = win
    w.start_match()
    assert w.mode == "live" and w.running
    for widget in (w.setup_box, w.start_btn, w.persona_box, w.deck_btn, w.lv11, w.ended_box, w.review_btn,
                   w.new_btn, w.review_panel, w.rew_btn, w.slider):
        assert not widget.isVisible()
    assert w.hand.isVisible() and w.counter.isVisible() and w.log_list.isVisible()
    assert not w.counter.guess_btn.isVisible()                            # only in quiz mode
    w.start_match()                                                       # Space no longer pauses
    assert w.running


def test_end_review_cancel_exit(win):
    from frontend_app.qt.app import COUNTDOWN_MS
    w = win
    w.start_match()
    play_one(w)
    finish(w)
    assert w.mode == "ended" and not w.running
    assert w.review_btn.isVisible() and w.review_btn.isEnabled() and w.new_btn.isVisible()
    assert not w.persona_box.isVisible()

    # Stop during analysis: the review screen stays, without grades
    real = w.session.analyze

    def stopped(progress=None, **_):
        w.cancel_analysis()
        progress(1, 10)
    w.session.analyze = stopped
    w.run_review()
    assert w.mode == "review" and not w.analyzing and w.review_panel.review is None
    assert "stopped" in w.status.text().lower()
    assert w.review_strip.isVisible() and w.slider.isVisible()
    w.session.analyze = real

    # timeline slider jumps straight to that moment (no countdown, nothing runs)
    w.slider.setValue(20_000)
    assert w.session.match.t_ms == 20_000 and w.countdown_ms == 0 and not w.running
    assert w.review_panel.graph.cursor_ms == 20_000
    w.step(-10)
    assert w.session.match.t_ms == 10_000 and w.countdown_ms == 0
    w.review_panel.graph.timeClicked.emit(30_000)                          # click on the graph
    assert w.session.match.t_ms == 30_000

    # Exit review → back to the final position and the result screen
    w.exit_review()
    assert w.mode == "ended" and w.session.result is not None

    # Take over from a reviewed moment → live, after a 3-2-1
    w.run_review()
    assert w.mode == "review" and w.review_panel.review is not None
    w.slider.setValue(40_000)
    assert w.review_panel.take_btn.isEnabled()
    w.review_panel.take_btn.click()
    assert w.mode == "live" and w.countdown_ms == COUNTDOWN_MS and w.session.match.t_ms == 40_000

    w.new_match()
    assert w.mode == "setup" and w.log_list.count() == 0


def test_leaving_the_page_pauses_and_resumes_with_countdown(win):
    from frontend_app.qt.app import COUNTDOWN_MS
    w = win
    w.start_match()
    w.pause_for_away()
    assert not w.running and w.away_paused
    w.resume_after_away()
    assert w.countdown_ms == COUNTDOWN_MS and not w.away_paused
    for _ in range(COUNTDOWN_MS // 50):
        w.on_timer()
    assert w.running


def test_quiz_mode_shows_guess_button(win):
    w = win
    w.counter_mode.setCurrentIndex(w.counter_mode.findData("quiz"))
    w.start_match()
    assert w.counter.quiz_mode and w.counter.guess_btn.isVisible()
    w.quiz_guess(3.0)
    assert w.session.quiz and "you guessed 3.0" in w.status.text()


def test_deck_builder_presets(win):
    from backend_engine.meta.presets import load_presets
    from frontend_app.qt.app import DeckDialog
    w = win
    d = DeckDialog(w.session, w, AI)
    assert d.list.count() == 30
    assert d.p_name.text() == "CRbot Starter"                             # the current AI deck is a preset
    row = next(i for i, p in enumerate(d.presets) if not p.ai_pushes)
    d.list.setCurrentRow(row)
    assert d.p_note.isVisibleTo(d)                                        # warns: the AI won't attack with it
    d.set_side(HUMAN)
    hog = next(i for i, p in enumerate(d.presets) if p.name == "Hog 2.6 Cycle")
    d.list.setCurrentRow(hog)
    cfg = d.result_config()
    presets = load_presets()
    assert cfg["player_deck"] == next(p for p in presets if p.name == "Hog 2.6 Cycle").cards
    assert cfg["ai_deck"] == d.presets[row].cards
    w.restart(**cfg)
    assert w.session.cfg.player_deck == cfg["player_deck"] and w.mode == "setup"
    assert "Hog 2.6 Cycle" in w.deck_title[HUMAN].text()


def test_stop_button_ends_match_instantly(win):
    w = win
    assert not w.stop_btn.isVisible()
    w.start_match()
    play_one(w)
    w.stop_btn.click()
    assert w.mode == "ended" and not w.running and w.session.result is None
    assert not w.stop_btn.isVisible() and w.review_btn.isEnabled() and "stopped" in w.status.text().lower()
    w.on_timer()
    assert w.session.match.t_ms < 1000                    # nothing runs after Stop
