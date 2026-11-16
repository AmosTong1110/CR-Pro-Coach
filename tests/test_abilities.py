"""Champion abilities (engine v6): Golden Knight's Dashing Dash and Mighty Miner's Explosive Escape, the
`use_ability` action (event, action log, replay), the scripted bot, the session and the saved-match record."""
import pytest

from backend_engine.simulator import geometry as G
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig
from backend_engine.simulator.golden import B2_A, B2_B
from backend_engine.simulator.mechanics import status as S
from backend_engine.simulator.mechanics.abilities import ABILITY_SUFFIX
from tests.test_batch1 import BIG, new, run, tank
from tests.test_mechanics import play


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


# the champion sits in the opening hand, so matches can be replayed without touching the hand
GKD = ("golden_knight", "miner", "tesla", "firecracker", "goblin_demolisher", "mighty_miner", "dark_prince",
       "royal_delivery")


def mk(db):
    return new(db, a=B2_A, b=B2_B)


def foe(m, card, x, y):
    """An enemy that stays put and does no damage, so only the ability changes its hp."""
    u = tank(m, 1, card, x, y)
    u.damage = 0
    u.decay_per_tick = 0.0
    return u


# ------------------------------------------------------------------ the action
def test_use_ability_rules_event_and_log(db):
    m = mk(db)
    assert m.check_ability(0, "knight") == "no_ability" and m.check_ability(0, "golden_knight") == "ability_not_ready"
    (gk,) = play(m, 0, "golden_knight", 9.5, 12.0)
    m.sides[0].elixir = 0.5
    assert m.use_ability(0, "golden_knight").reason == "not_enough_elixir"
    m.sides[0].elixir = 5.0
    res = m.use_ability(0, "golden_knight")
    assert res.ok and m.sides[0].elixir == pytest.approx(4.0)
    ev = res.event
    assert ev.is_ability and ev.card_id == "golden_knight" and ev.elixir_cost == 1 and ev.side == 0
    assert m.events[-1] is ev and m.action_log[-1][2] == "golden_knight" + ABILITY_SUFFIX
    m.sides[0].elixir = 5.0
    assert m.use_ability(0, "golden_knight").reason == "ability_not_ready"      # single use
    assert m.check_ability(1, "golden_knight") == "ability_not_ready"            # not the other side's champion


def test_ability_not_usable_while_deploying(db):
    m = mk(db)
    play(m, 0, "golden_knight", 9.5, 12.0, ready=False)
    m.sides[0].elixir = 5.0
    assert m.check_ability(0, "golden_knight") == "ability_not_ready"


# ------------------------------------------------------------------ Golden Knight
def test_dashing_dash_hits_each_enemy_once_and_makes_him_immune(db):
    m = mk(db)
    (gk,) = play(m, 0, "golden_knight", 9.5, 12.0)
    gk.hp = gk.max_hp = BIG
    foes = [foe(m, "knight", 9.5 + dx, 14.0 + dy) for dx, dy in ((0, 0), (2.0, 0.5), (-2.0, 1.0), (1.0, 3.0))]
    for f in foes:
        f.damage = 50                                       # they would hurt him if he were not immune
    m.sides[0].elixir = 5.0
    assert m.use_ability(0, "golden_knight").ok
    for _ in range(200):
        if gk.mstate["ab"]["phase"] is None:
            break
        assert gk.hp == BIG                                  # immune while dashing
        m.step()
    assert gk.mstate["ab"]["phase"] is None and gk.mstate["ab"]["dashes"] == 4
    dash = db.stats("golden_knight", 11).stats["dash_damage"]
    assert dash == 335 and [BIG - f.hp for f in foes] == [dash] * 4


def test_dash_ignores_towers_and_far_enemies_and_never_repeats_a_target(db):
    m = mk(db)
    (gk,) = play(m, 0, "golden_knight", 9.5, 12.0)
    near = foe(m, "knight", 9.5, 14.0)
    far = foe(m, "valkyrie", 9.5, 22.0)
    m.sides[0].elixir = 5.0
    m.use_ability(0, "golden_knight")
    while gk.mstate["ab"]["phase"] is not None:
        m.step()
    dash = 335
    assert BIG - near.hp == dash and far.hp == BIG
    towers = [t for t in m.units.values() if t.side == 1 and t.kind == "tower"]
    assert all(t.hp == t.max_hp for t in towers)
    assert gk.mstate["ab"]["dashes"] == 1                    # one enemy in reach, hit once


def test_dash_with_no_enemy_still_costs_elixir_and_ends(db):
    m = mk(db)
    play(m, 0, "golden_knight", 9.5, 12.0)
    m.sides[0].elixir = 5.0
    assert m.use_ability(0, "golden_knight").ok and m.sides[0].elixir == pytest.approx(4.0)
    m.step()
    gk = next(u for u in m.units.values() if u.card_id == "golden_knight")
    assert gk.mstate["ab"]["phase"] is None


# ------------------------------------------------------------------ Mighty Miner
def test_explosive_escape_mirrors_the_lane_and_leaves_a_bomb(db):
    m = mk(db)
    (mm,) = play(m, 0, "mighty_miner", 4.5, 20.0)
    target = foe(m, "knight", 4.5, 20.5)
    m.sides[0].elixir = 5.0
    assert m.use_ability(0, "mighty_miner").ok
    assert mm.x == pytest.approx(G.W - 4.5) and mm.y == pytest.approx(20.0)
    assert S.INVISIBLE in mm.status and mm.deploy_ms == 933            # underground: can't act, can't be targeted
    run(m, 900)
    assert target.hp == BIG                                           # the bomb has not gone off yet
    run(m, 200)
    esc = db.stats("mighty_miner", 11).stats["escape_damage"]
    assert esc == 332 and BIG - target.hp == esc
    assert target.y > 20.5                                            # knocked away from the blast
    run(m, 200)
    assert S.INVISIBLE not in mm.status and mm.deploy_ms <= 0          # resurfaced


# ------------------------------------------------------------------ replay, observation, bot
def test_ability_replays_exactly(db):
    cfg = MatchConfig((GKD, B2_B), shuffle=False)
    m = Match(cfg, db)
    assert m.play(0, "golden_knight", 4.0, 12.0).ok
    m.run_until(2000)
    assert m.play(1, "knight", 4.0, 20.0).ok
    m.run_until(6000)
    assert m.use_ability(0, "golden_knight").ok
    m.run_until(10_000)
    r = Match.replay(cfg, m.action_log, db, until_ms=10_000)
    assert r.state_hash() == m.state_hash() and [e.is_ability for e in r.events] == [e.is_ability for e in m.events]


def test_observation_lists_ready_abilities(db):
    m = mk(db)
    play(m, 0, "golden_knight", 9.5, 12.0)
    m.sides[0].elixir = 5.0
    assert m.observation(0)["abilities"] == ["golden_knight"] and m.observation(1)["abilities"] == []
    m.sides[0].elixir = 0.2
    assert m.observation(0)["abilities"] == []


def test_scripted_bot_uses_dash_on_a_group(db):
    from backend_engine.brain.scripted_bot import ScriptedBot
    from backend_engine.contracts import ActionKind
    m = mk(db)
    play(m, 0, "golden_knight", 9.5, 12.0)
    m.sides[0].elixir = 5.0
    bot = ScriptedBot(db, 0, 0.5)
    obs = m.observation(0)             # nothing near: it may play a card, but not use the ability
    assert not [i for i in bot.decide(obs, [], m.t_ms) if i.kind == ActionKind.ABILITY]
    foe(m, "knight", 9.5, 14.0)
    foe(m, "valkyrie", 11.0, 14.5)
    (i,) = bot.decide(m.observation(0), [], m.t_ms)
    assert i.kind == ActionKind.ABILITY and i.card_id == "golden_knight"


def test_ai_executes_ability_intents_through_the_runner(db):
    from backend_engine.brain.scripted_bot import ScriptedBot
    from backend_engine.simulator.match_runner import MatchRunner
    m = new(db, a=B2_B, b=B2_A)                                          # the AI (side 1) has the champion
    r = MatchRunner(m, {0: None, 1: ScriptedBot(db, 1, 0.5)})
    play(m, 1, "golden_knight", 9.5, 20.0)
    for card, x in (("knight", 9.5), ("valkyrie", 8.0)):
        u = play(m, 0, card, x, 17.5)[0]
        u.hp = u.max_hp = BIG
    m.sides[1].elixir = 8.0
    for _ in range(100):
        r.tick()
    assert any(e.is_ability and e.side == 1 for e in m.events)


# ------------------------------------------------------------------ session, saved match, UI
def _session(db):
    """A session whose opening hand holds the Golden Knight (the hand is shuffled by the seed)."""
    from frontend_app.qt.session import GameSession, SessionConfig
    for seed in range(60):
        s = GameSession(db, SessionConfig(player_deck=GKD, ai_deck=B2_B, seed=seed))
        if "golden_knight" in s.match.sides[0].hand:
            return s
    raise AssertionError("no seed puts the Golden Knight in the opening hand")


def test_session_ability_roundtrips_through_a_saved_match(db):
    from backend_engine.storage.match_store import MatchRecord
    s = _session(db)
    assert s.play("golden_knight", 4.0, 12.0).ok
    s.tick(60)
    assert s.ability_cards() == ["golden_knight"]
    res = s.use_ability("golden_knight")
    assert res.ok and s.log[-1].ability and s.log[-1].card_id == "golden_knight"
    assert s.use_ability("golden_knight").reason == "ability_not_ready"
    s.tick(400)
    meta, record = s.to_record()
    assert meta.n_plays == 1                                          # the ability is not a card play
    assert [a.ability for a in record.actions if a.side == 0] == [False, True]
    record2 = MatchRecord.from_dict(record.to_dict())
    from frontend_app.qt.session import GameSession
    rebuilt = GameSession.rebuild(record2, db)
    assert rebuilt.match.state_hash() == s.match.state_hash()


def test_coach_does_not_grade_ability_uses(db):
    s = _session(db)
    s.play("golden_knight", 4.0, 12.0)
    s.tick(60)
    s.use_ability("golden_knight")
    plays = [ev for ev in s.match.events if ev.side == 0 and ev.event_id in s.pre_play and not ev.is_ability]
    assert len(plays) == 1


def test_ui_shows_the_ability_button_only_when_usable(db):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from frontend_app.qt.app import LIVE, MainWindow
    app = QApplication.instance() or QApplication([])
    s = _session(db)
    w = MainWindow(s)
    w._set_mode(LIVE)
    w.countdown_ms = 0
    w.refresh()
    assert w.hand.ability_btn.isHidden()
    s.play("golden_knight", 4.0, 12.0)
    s.tick(60)
    w.refresh()
    assert not w.hand.ability_btn.isHidden() and "Golden Knight" in w.hand.ability_btn.text()
    w.use_ability()
    assert s.log[-1].ability
    w.refresh()
    assert w.hand.ability_btn.isHidden()
    w.add_log([])
    assert "ability" in w.log_list.item(w.log_list.count() - 1).text()
    app.processEvents()
