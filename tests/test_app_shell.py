"""CRbot app shell: match store, rebuild, archetype names, deck recommendations, profile, Qt shell."""
import os
from datetime import datetime

import pytest

from backend_engine.brain.coach_analyzer import Candidate, MatchReview, MoveReview
from backend_engine.meta.archetypes import archetype_name, fmt_avg
from backend_engine.meta.recommend import coverage, recommend
from backend_engine.storage.match_store import (
    MatchStore, MatchStoreError, new_match_id, review_from_dict, review_to_dict,
)
from backend_engine.storage.profile import Profile, load_profile, save_profile
from frontend_app.qt.session import HUMAN, GameSession, ReplayMismatch, SessionConfig

HOG26 = ["cannon", "fireball", "hog_rider", "ice_golem", "ice_spirit", "musketeer", "skeletons", "the_log"]


def force_hand(s, card):
    """Test-only cheat (edits the hand): never use it on a match that will be rebuilt."""
    st = s.match.sides[HUMAN]
    if card not in st.hand:
        st.queue.remove(card)
        st.queue.insert(0, st.hand[0])
        st.hand[0] = card
    st.elixir = 10


def play_at(s, t_ms, x, y):
    """Legal play: from t_ms, wait until a troop in hand is affordable, then drop it at (x, y)."""
    while s.match.t_ms < t_ms:
        s.tick(1)
    while True:
        st = s.match.sides[HUMAN]
        card = next((c for c in st.hand if s.db.cards[c]["type"] != "spell" and s.elixir_cost(c) <= st.elixir), None)
        if card is not None:
            res = s.play(card, x, y)
            assert res.ok, res.reason
            return card
        s.tick(1)


@pytest.fixture
def played():
    """A session with plays, a rewind and a different branch (the hard case for rebuilding)."""
    s = GameSession(cfg=SessionConfig(seed=11))
    play_at(s, 2000, 8.5, 10.5)
    play_at(s, 8000, 3.5, 13.5)
    s.tick(20 * 10)
    s.rewind(10)
    play_at(s, s.match.t_ms + 500, 14.5, 8.5)
    s.tick(20 * 12)
    return s


def fake_review(s) -> MatchReview:
    mine = [e for e in s.match.events if e.side == HUMAN]
    moves = [MoveReview(e.event_id, e.t_ms, Candidate(e.card_id, e.tile.x, e.tile.y), 0.5, 0.45, 0.6, 0.5,
                        Candidate(None), 0.15, "mistake", 12, ((Candidate("knight", 9.5, 8.5), 0.6),))
             for e in mine]
    return MatchReview([(0, 0.5), (s.match.t_ms, 0.4)], moves, None, {"kind": "test"})


# ---------------------------------------------------------------- archetypes / decks
def test_archetype_names():
    assert archetype_name(HOG26) == "Hog 2.6"
    assert fmt_avg(2.625) == "2.6"
    hut = ["royal_hogs", "goblin_hut", "fireball", "zap", "valkyrie", "musketeer", "skeletons", "giant_snowball"]
    assert archetype_name(hut, evo=["royal_hogs"]).startswith("EvoRoyalHogs")
    assert archetype_name(["golem", "night_witch", "baby_dragon", "lumberjack", "tornado", "lightning",
                           "barbarian_barrel", "elixir_collector"]) == "Golem Pump"
    assert archetype_name(["knight", "musketeer", "archers", "skeletons", "fireball", "the_log", "minions",
                           "cannon"]).endswith("Control")
    assert archetype_name([]) == ""


def test_recommend_marks_playable_decks():
    recs = recommend({"hog_rider", "musketeer", "cannon", "fireball", "skeletons", "the_log"}, limit=5)
    assert recs[0].archetype == "Hog 2.6" and set(recs[0].missing) == {"ice_golem", "ice_spirit"}
    assert recs[0].share > 0.05 and not recs[0].playable
    every = {c for r in recommend(set(), limit=2000) for c in r.cards}
    full = recommend(every, limit=3, playable_only=True)
    assert len(full) == 3 and all(r.playable for r in full)
    ok, total = coverage(every)
    assert ok == total > 900


# ---------------------------------------------------------------- store
def test_match_id_format():
    mid = new_match_id("practice", 3, datetime(2026, 11, 15, 9, 5, 7))
    assert mid.startswith("20261115-090507_practice_") and len(mid.split("_")[-1]) == 8
    assert new_match_id("drill", 1) != new_match_id("drill", 1)
    with pytest.raises(ValueError):
        new_match_id("ranked")


def test_store_round_trip(tmp_path, played):
    store = MatchStore(tmp_path)
    meta, rec = played.to_record()
    n_mine = sum(e.side == HUMAN for e in played.match.events)
    assert n_mine >= 2 and meta.n_plays == n_mine and meta.outcome is None and meta.player_archetype == "Hog 3.0"
    store.save(meta, rec)
    assert [m.match_id for m in store.list()] == [meta.match_id]
    assert store.load_review(meta.match_id) is None and not store.load_meta(meta.match_id).analyzed
    review = fake_review(played)
    m2 = store.save_review(meta.match_id, review)
    assert m2.analyzed and m2.counts["mistake"] == n_mine
    back = store.load_review(meta.match_id)
    assert review_to_dict(back) == review_to_dict(review)
    assert back.moves[0].best.is_wait and back.accuracy == pytest.approx(review.accuracy)
    # saving again without a review drops the stale one
    store.save(meta, rec, None)
    assert store.load_review(meta.match_id) is None
    store.delete(meta.match_id)
    assert store.list() == []


def test_store_rejects_bad_ids_and_skips_broken(tmp_path):
    store = MatchStore(tmp_path)
    for bad in ("", "../x", ".hidden", "a/b"):
        with pytest.raises(MatchStoreError):
            store.load_meta(bad)
    (tmp_path / "broken").mkdir()
    (tmp_path / "broken" / "meta.json").write_text("{not json")
    assert store.list() == []


def test_review_dict_round_trip(played):
    r = fake_review(played)
    assert review_to_dict(review_from_dict(review_to_dict(r))) == review_to_dict(r)


# ---------------------------------------------------------------- rebuild
def test_rebuild_reproduces_branched_match(played):
    meta, rec = played.to_record()
    assert any(a.side == 1 for a in rec.actions)       # the AI played too
    s = GameSession.rebuild(rec)
    assert s.match.t_ms == rec.final_t_ms and s.match.state_hash() == rec.final_hash
    assert [(e.t_ms, e.card_id) for e in s.match.events] == [(a.t_ms, a.card_id) for a in rec.actions]
    # every play of yours has its exact pre-play state again, so review ghosts / take over work
    mine = [e.event_id for e in s.match.events if e.side == HUMAN]
    assert all(eid in s.pre_play for eid in mine)


def test_rebuild_after_rewind_saves_furthest_point(played):
    end = played.match.t_ms
    played.rewind(10)
    meta, rec = played.to_record()
    assert rec.final_t_ms == end and meta.duration_ms == end


def test_rebuild_detects_mismatch(played):
    _, rec = played.to_record()
    rec.final_hash = "0" * 16
    with pytest.raises(ReplayMismatch):
        GameSession.rebuild(rec)


def test_reopened_match_branches_to_new_id(played):
    meta, rec = played.to_record()
    s = GameSession.rebuild(rec)
    s.mark_loaded(meta, fake_review(played))
    assert s.match_id == meta.match_id and s.review_is_current()
    assert s.to_record()[0].match_id == meta.match_id     # unchanged: saving overwrites the same match
    first = next(e for e in s.log if e.side == HUMAN)
    s.take_over_before(s.log.index(first))
    play_at(s, s.match.t_ms, 9.5, 6.5)
    assert s.match_id is None and not s.review_is_current()
    assert s.to_record()[0].match_id != meta.match_id


# ---------------------------------------------------------------- profile
def test_profile_round_trip(tmp_path):
    assert load_profile(tmp_path) == Profile()
    p = Profile(display_name="me", player_tag="#2pq8 ", default_level=16, default_speed=4, auto_review=False)
    save_profile(p, tmp_path)
    q = load_profile(tmp_path)
    assert q.player_tag == "2PQ8" and q.default_level == 16 and q.default_speed == 4 and not q.auto_review
    assert Profile.clean_tag("#ABC") == ""                 # A, B are not tag letters
    (tmp_path / "profile.json").write_text('{"default_level": 12, "unknown": 1}')
    assert load_profile(tmp_path).default_level == 11


# ---------------------------------------------------------------- Qt shell
@pytest.fixture
def shell(tmp_path):
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from frontend_app.qt.shell import CRbotWindow

    app = QApplication.instance() or QApplication([])
    w = CRbotWindow(GameSession(cfg=SessionConfig(seed=5)), data_dir=tmp_path)
    w.show()
    yield w
    w.close()
    app.processEvents()


def test_shell_pages_and_pause(shell):
    assert shell.current_page() == "fight"
    shell.fight.toggle_run()
    assert shell.fight.running
    shell.show_page("decks")
    assert not shell.fight.running and shell.current_page() == "decks"
    assert all(not sc.isEnabled() for sc in shell._fight_shortcuts)
    shell.show_page("fight")
    assert all(sc.isEnabled() for sc in shell._fight_shortcuts)
    for key in ("analysis", "account", "fight"):
        shell.show_page(key)
        assert not shell.grab().isNull()


def test_shell_saves_and_reopens_match(shell):
    f = shell.fight
    assert shell.save_current() is None                   # nothing played: nothing saved
    play_at(f.session, 1000, 8.5, 10.5)
    f.session.tick(20 * 5)
    f.new_match()                                         # aboutToReset -> saved
    metas = shell.store.list()
    assert len(metas) == 1 and metas[0].n_plays == 1
    shell.show_page("analysis")
    assert shell.analysis.list.count() == 1 and shell.analysis.open_btn.isVisible()
    shell.store.save_review(metas[0].match_id, fake_review(GameSession.rebuild(shell.store.load_record(metas[0].match_id))))
    assert shell.open_match(metas[0].match_id)
    assert shell.current_page() == "fight" and f.session.loaded_from == metas[0].match_id
    assert f.review_panel.review is not None and f.log_list.count() == len(f.session.log)
    f.review_panel.select(0)                               # ghosts for the first graded move
    assert f.board.ghosts


def test_shell_decks_and_account(shell, tmp_path):
    d = shell.decks
    assert d.rows[0][0] == "CRbot starter" and d.use_btn.isEnabled()
    d.table.selectRow(1)
    assert not d.use_btn.isEnabled()                      # real meta decks need cards we don't simulate yet
    d.table.selectRow(0)
    d.use(as_ai=True)
    assert shell.current_page() == "fight" and shell.fight.session.cfg.ai_deck == tuple(d.rows[0][1])
    a = shell.account
    a.tag.setText("#2PQ8")
    a.level.setCurrentIndex(a.level.findData(16))
    a.autoreview.setChecked(False)
    a.save()
    p = load_profile(tmp_path)
    assert p.player_tag == "2PQ8" and p.default_level == 16 and not shell.fight.auto_review


def test_old_engine_match_is_summary_only(shell):
    from backend_engine.simulator.engine import ENGINE_VERSION
    f = shell.fight
    play_at(f.session, 1000, 8.5, 10.5)
    mid = shell.save_current()
    meta = shell.store.load_meta(mid)
    assert meta.engine_version == ENGINE_VERSION
    meta.engine_version = ENGINE_VERSION - 1
    shell.store.save(meta, shell.store.load_record(mid))
    shell.show_page("analysis")
    assert not shell.analysis.open_btn.isEnabled() and "older engine" in shell.analysis.grades.text()
