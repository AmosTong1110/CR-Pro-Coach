"""Review mode: video-like playback, graph grade markers, Mistake / Alternative explanations."""
import os

import pytest

from backend_engine.brain.coach_analyzer import Candidate, MatchReview, MoveReview
from backend_engine.brain.scripted_bot import ScriptedBot
from backend_engine.contracts import ActionKind
from frontend_app.qt.session import HUMAN, GameSession, SessionConfig

NAMES = {"knight": "Knight", "cannon": "Cannon", "ice_golem": "Ice Golem", "hog_rider": "Hog Rider",
         "the_log": "The Log", "fireball": "Fireball"}
names = lambda c: NAMES.get(c, c)  # noqa: E731


def _played_session(seed=4, until_ms=None):
    """A session where a scripted 'human' played through GameSession.play, like the UI does."""
    s = GameSession(cfg=SessionConfig(seed=seed))
    bot = ScriptedBot(s.db, HUMAN, 0.7)
    while s.result is None and (until_ms is None or s.match.t_ms < until_ms):
        if s.match.t_ms % 500 == 0:
            obs = s.match.observation(HUMAN)
            obs["opp_estimate"], obs["opp_deck_guess"] = s.opp_estimate(), None
            for it in bot.decide(obs, [], s.match.t_ms):
                if it.kind == ActionKind.DEPLOY:
                    s.play(it.card_id, it.tile.x, it.tile.y)
                else:
                    s.use_ability(it.card_id)
        s.tick(1)
    return s


def test_replay_reproduces_the_match_exactly():
    s = _played_session(until_ms=90_000)
    final, end = s.match.state_hash(), s.match.t_ms
    events = [(e.t_ms, e.side, e.card_id) for e in s.match.events]
    s.seek(5_000)
    acts = s.review_actions()
    assert acts and all(t >= 0 for t in acts)
    t = s.match.t_ms
    while t < end:
        t = s.replay_to(t + 150, acts)
    assert t == end and s.match.state_hash() == final
    assert [(e.t_ms, e.side, e.card_id) for e in s.match.events] == events
    s.seek(30_000)                                    # the timeline still works afterwards
    assert s.match.t_ms == 30_000 and s.can_forward()
    assert s.replay_to(10**9, acts) == end            # never past what was played


# -- explanations -------------------------------------------------------------------------------------------
def _mv(label, actual, best, loss, v_before=0.5, v_wait=0.4, alternatives=()):
    v_actual = 0.5 - loss / 2
    return MoveReview(1, 70_000, actual, v_before, v_actual, v_actual + loss, v_wait, best, loss, label, 32,
                      tuple(alternatives))


def test_explain_cases():
    from frontend_app.qt.review import explain, tooltip_html, where
    a = Candidate("cannon", 9.5, 12.5)
    m, alt = explain(_mv("mistake", a, Candidate("cannon", 9.5, 9.5), 0.12), names)
    assert m.startswith("Right card, wrong spot") and "3 tiles too far forward" in m
    assert "Cannon at (9.5, 9.5)" in alt and "centre, mid-way on your side" in alt
    m, alt = explain(_mv("blunder", Candidate("hog_rider", 3.5, 14.5), Candidate(None), 0.25, v_wait=0.6,
                         alternatives=((Candidate(None), 0.6), (Candidate("the_log", 3.5, 18.0), 0.52))), names)
    assert m.startswith("Played too early") and alt.startswith("Wait") and "If you must play: The Log" in alt
    m, alt = explain(_mv("inaccuracy", Candidate("knight", 3.5, 8.5), Candidate("ice_golem", 14.5, 13.5), 0.06,
                         v_wait=0.6), names)
    assert m.startswith("Wrong card") and "Ice Golem handles" in m and "Even waiting" in m
    assert alt.startswith("Ice Golem at (14.5, 13.5), right lane, at the bridge")
    m, alt = explain(_mv("best", a, a, 0.0, alternatives=((a, 0.5), (Candidate("knight", 9.5, 10.5), 0.49))), names)
    assert m.startswith("None.") and "already the best" in alt and "Also fine: Knight" in alt
    html = tooltip_html(_mv("inaccuracy", a, Candidate("cannon", 8.5, 9.5), 0.05), names)
    assert "Mistake:" in html and "Alternative:" in html and "?!" in html
    assert where(9.0, 2.0) == "centre, at the back, by your King Tower"


# -- graph markers ------------------------------------------------------------------------------------------
@pytest.fixture
def qapp():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def test_every_grade_symbol_gets_a_visible_label(qapp):
    """Clustered ?! / ? / ?? plays and a curve near 100% used to lose their symbols (dropped when closer than
    16 px to the previous one, or drawn above the widget)."""
    from frontend_app.qt.review import WinRateGraph
    labels = ["inaccuracy", "mistake", "inaccuracy", "blunder", "good", "inaccuracy", "brilliant", "inaccuracy"]
    moves = [_mv(lb, Candidate("knight", 9.5, 9.5), Candidate("cannon", 9.5, 9.5), 0.05) for lb in labels]
    moves = [MoveReview(i + 1, 200_000 + i * 900, *[getattr(mv, f) for f in
             ("actual", "v_before", "v_actual", "v_best", "v_wait", "best", "loss", "label", "n_candidates")])
             for i, mv in enumerate(moves)]
    review = MatchReview([(t, 0.99) for t in range(0, 240_001, 1000)], moves)
    g = WinRateGraph()
    g.resize(900, 150)
    g.set_review(review, 240_000)
    rects = g.label_rects()
    shown = {i for i, _ in rects}
    assert shown == {i for i, lb in enumerate(labels) if lb != "good"}       # none dropped, ?! included
    bounds = g.rect()
    for _, r in rects:
        assert r.top() >= bounds.top() and r.bottom() <= bounds.bottom()
    assert not any(a.intersects(b) for k, (_, a) in enumerate(rects) for _, b in rects[k + 1:])
    g.grab()                                                                  # paints without errors


# -- the Fight page in review mode --------------------------------------------------------------------------
@pytest.fixture
def win(qapp):
    from frontend_app.qt.app import MainWindow
    w = MainWindow(_played_session(seed=6, until_ms=60_000))
    w.auto_review = False
    w.show()
    yield w
    w.close()
    qapp.processEvents()


def test_play_and_stop_buttons(win):
    from frontend_app.qt.app import TICK_MS
    w = win
    s = w.session
    s.stop_match() if hasattr(s, "stop_match") else None
    moves = []
    for i, e in enumerate(e for e in s.match.events if e.side == HUMAN):
        moves.append(MoveReview(e.event_id, e.t_ms, Candidate(e.card_id, e.tile.x, e.tile.y), 0.5, 0.48, 0.5, 0.45,
                                Candidate(e.card_id, e.tile.x, e.tile.y), 0.02 if i % 2 else 0.06,
                                "good" if i % 2 else "inaccuracy", 30))
    w.session.analyze = lambda progress=None, **_: MatchReview([(0, 0.5), (s.max_known_ms(), 0.5)], moves)
    w._set_mode("ended")
    w.run_review()
    assert w.mode == "review" and w.play_btn.isVisible() and w.speed_btn.isVisible()
    w.slider.setValue(10_000)
    assert s.match.t_ms == 10_000
    w.play_btn.click()
    assert w.playing and "Stop" in w.play_btn.text()
    for _ in range(20):
        w.on_timer()
    assert s.match.t_ms == 10_000 + 20 * TICK_MS and w.slider.value() == s.match.t_ms
    assert w.review_panel.graph.cursor_ms == s.match.t_ms
    w.play_btn.click()                                                      # Stop: stays where it is
    assert not w.playing and "Play" in w.play_btn.text()
    t_stop = s.match.t_ms
    w.on_timer()
    assert s.match.t_ms == t_stop
    w.speed_btn.click()
    assert w.play_speed == 2 and w.speed_btn.text() == "2x"
    w.toggle_playback()
    first = next(mv for mv in moves if mv.t_ms > s.match.t_ms)
    while s.match.t_ms < first.t_ms:
        w.on_timer()
    assert w.board.banner and first.symbol in w.board.banner                 # grade shown while passing it
    w.slider.setValue(5_000)                                                # dragging the timeline stops playback
    assert not w.playing and s.match.t_ms == 5_000
    w.toggle_playback()
    w.play_speed = 4
    for _ in range(2000):
        w.on_timer()
        if not w.playing:
            break
    assert not w.playing and s.match.t_ms == s.max_known_ms()               # stops by itself at the end
    w.toggle_playback()                                                     # at the end: Play starts over
    assert w.playing and s.match.t_ms == 0
    w.exit_review()
    assert not w.playing


def test_fork_session():
    s = _played_session(until_ms=30_000)
    cp = s._capture()
    forked = s.fork_session(cp)
    assert forked.match.t_ms == s.match.t_ms == 30_000
    assert forked.match.state_hash() == s.match.state_hash()
    assert forked.match is not s.match
    assert forked.runner is not s.runner

    # Mutating the fork does not touch the original session
    h0 = s.match.state_hash()
    forked.tick(10)
    assert forked.match.t_ms == 30_000 + 10 * 50
    assert s.match.t_ms == 30_000
    assert s.match.state_hash() == h0


def test_show_me_animated_rollout(win):
    from frontend_app.qt.app import TICK_MS
    w = win
    s = w.session
    first_human_ev = next(e for e in s.match.events if e.side == HUMAN)
    mv_blunder = MoveReview(
        first_human_ev.event_id, first_human_ev.t_ms,
        Candidate(first_human_ev.card_id, first_human_ev.tile.x, first_human_ev.tile.y),
        0.5, 0.25, 0.70, 0.55,
        Candidate("knight", 8.5, 6.0), 0.45, "blunder", 30
    )
    mv_wait = MoveReview(
        first_human_ev.event_id, first_human_ev.t_ms,
        Candidate(first_human_ev.card_id, first_human_ev.tile.x, first_human_ev.tile.y),
        0.5, 0.25, 0.60, 0.60,
        Candidate(None), 0.35, "blunder", 30
    )
    review = MatchReview([(0, 0.5), (s.max_known_ms(), 0.5)], [mv_blunder, mv_wait])
    w.session.analyze = lambda **_: review
    w._set_mode("ended")
    w.run_review()
    assert w.mode == "review"

    # Select the blunder move
    w.review_panel.select(0)
    assert w.review_panel.show_me_btn.isEnabled()
    assert "Show me" in w.review_panel.show_me_btn.text()
    assert not w.showing_me

    # 1. Start Show Me
    origin = w.session
    origin_t = origin.match.t_ms
    w.review_panel.show_me_btn.click()
    assert w.showing_me
    assert "Stop" in w.review_panel.show_me_btn.text()
    assert w._show_me_origin_session is origin
    assert w.session is not origin
    assert "Show me: Knight" in w.board.banner
    # The actual play is drawn as a red ghost during rollout
    assert any(kind == "actual" for _, _, _, kind in w.board.ghosts)

    # 2. Advance ticks during Show Me
    start_ms = w._show_me_start_ms
    for _ in range(10):
        w.on_timer()
    assert w.showing_me
    assert w.session.match.t_ms == start_ms + 10 * TICK_MS
    assert origin.match.t_ms == origin_t  # Origin untouched

    # 3. Simulate until 8.0s rollout ends
    while w.session.match.t_ms < w._show_me_end_ms:
        w.on_timer()
    w.on_timer()
    assert "complete" in w.board.banner
    assert "Exit preview" in w.review_panel.show_me_btn.text()

    # 4. Revert / Stop Show Me back to review
    w.review_panel.show_me_btn.click()
    assert not w.showing_me
    assert w.session is origin
    assert w.board.session is origin
    assert "Show me" in w.review_panel.show_me_btn.text()

    # 5. Show Me on a "Wait" recommendation
    w.review_panel.select(1)
    w.toggle_show_me()
    assert w.showing_me
    assert "Wait" in w.board.banner
    w.on_timer()
    w.stop_show_me()
    assert not w.showing_me

    # 6. Take over directly from Show Me
    w.review_panel.select(0)
    w.toggle_show_me()
    assert w.showing_me
    forked_branch = w.session
    w.review_panel.take_btn.click()
    assert not w.showing_me
    assert w.session is forked_branch
    assert w.mode == "live"

