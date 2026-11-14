import pytest

from backend_engine.brain.human_delay_buffer import HumanDelayBuffer
from backend_engine.brain.scripted_bot import ScriptedBot
from backend_engine.simulator import geometry as G
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig
from backend_engine.simulator.match_runner import DEFAULT_DECK, MatchRunner

DECK = DEFAULT_DECK


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


def new(db, seed=0, levels=(11, 11), evo=((), ()), shuffle=False):
    return Match(MatchConfig((DECK, DECK), levels, evo, seed=seed, shuffle=shuffle), db)


def give(m, side, card, elixir=10.0):
    """Force `card` into hand slot 0 with enough elixir (test helper)."""
    st = m.sides[side]
    if card not in st.hand:
        st.queue.remove(card)
        st.queue.insert(0, st.hand[0])
        st.hand[0] = card
    st.elixir = elixir


def kill_towers(m, side, *roles):
    for r in roles:
        m.tower(side, r).hp = 0
    m._resolve_deaths()


# ---------------------------------------------------------------- setup / elixir / cycle
def test_initial_state(db):
    m = new(db)
    assert m.sides[0].elixir == 5 and len(m.sides[0].hand) == 4 and len(m.sides[0].queue) == 4
    assert sum(1 for u in m.units.values() if u.kind == "tower") == 6
    assert not m.tower(0, "king").active and m.tower(0, "princess_left").active


def test_bad_deck_rejected(db):
    with pytest.raises(ValueError):
        Match(MatchConfig((DECK[:7] + ("knight",), DECK)), db)
    with pytest.raises(ValueError):
        Match(MatchConfig((DECK, DECK), evo_cards=(("hog_rider",), ())), db)  # no evolution data


def test_elixir_regen_phases_and_leak(db):
    m = new(db)
    m.run_until(2800)
    assert m.sides[0].elixir == pytest.approx(6.0)
    m.sides[0].elixir = 0
    m.t_ms = 120_000
    for _ in range(28):
        m.step()
    assert m.multiplier() == 2 and m.sides[0].elixir == pytest.approx(1.0)
    m.sides[0].elixir = 10
    m.step()
    assert m.sides[0].elixir == 10 and m.sides[0].leaked_elixir > 0


def test_card_cycle_returns_after_four_plays(db):
    m = new(db)
    first = m.sides[0].hand[0]
    played = []
    for _ in range(5):
        m.sides[0].elixir = 10
        c = next(c for c in m.sides[0].hand if c not in played and db.cards[c]["type"] != "spell")
        assert m.play(0, c, 9, 5).ok
        played.append(c)
        assert c not in m.sides[0].hand
        if len(played) <= 4:
            assert m.sides[0].queue[-1] == c
        m.step()
    assert played[0] in m.sides[0].hand  # back after 4 later plays
    assert first is not None


def test_evolution_charge(db):
    m = new(db, evo=(("knight",), ()))
    evolved = []
    for _ in range(3):
        give(m, 0, "knight")
        evolved.append(m.play(0, "knight", 9, 5).event.is_evolved)
    assert evolved == [False, False, True]


# ---------------------------------------------------------------- placement
@pytest.mark.parametrize("card,x,y,reason", [
    ("knight", 9, 5, None),
    ("knight", 9, 20, "enemy_half"),
    ("knight", 9, 15.5, "river"),
    ("knight", 3.5, 6.5, "blocked_by_structure"),
    ("fireball", 9, 28, None),
    ("knight", 30, 5, "out_of_bounds"),
])
def test_placement(db, card, x, y, reason):
    m = new(db)
    give(m, 0, card)
    assert m.check_play(0, card, x, y) == reason


def test_not_in_hand_and_not_enough_elixir(db):
    m = new(db)
    missing = m.sides[0].queue[0]
    assert m.check_play(0, missing, 9, 5) == "not_in_hand"
    give(m, 0, "hog_rider", elixir=3.9)
    assert m.check_play(0, "hog_rider", 9, 5) == "not_enough_elixir"


def test_pocket_after_princess_falls(db):
    m = new(db)
    give(m, 0, "knight")
    assert m.check_play(0, "knight", 3.5, 18.5) == "enemy_half"
    kill_towers(m, 1, "princess_left")
    assert m.check_play(0, "knight", 3.5, 18.5) is None
    assert m.check_play(0, "knight", 14.5, 18.5) == "enemy_half"
    assert m.sides[0].crowns == 1 and m.tower(1, "king").active


def test_side1_mirrored_placement(db):
    m = new(db)
    give(m, 1, "knight")
    assert m.check_play(1, "knight", 9, 26) is None
    assert m.check_play(1, "knight", 9, 5) == "enemy_half"


def test_threat_flag(db):
    m = new(db)
    give(m, 0, "hog_rider")
    assert m.play(0, "hog_rider", 3.5, 14.5).event.is_threat
    give(m, 0, "knight")
    assert not m.play(0, "knight", 16.5, 1.5).event.is_threat


# ---------------------------------------------------------------- combat / movement
def test_deploy_time_then_walks_over_bridge(db):
    m = new(db)
    give(m, 0, "knight")
    m.play(0, "knight", 3.5, 10.5)
    k = next(u for u in m.units.values() if u.card_id == "knight")
    m.run_until(950)
    assert (k.x, k.y) == (3.5, 10.5)
    m.run_until(9000)
    assert k.y > G.RIVER_Y1 and abs(k.x - 3.5) < 1.5


def test_hog_reaches_and_hits_tower(db):
    m = new(db)
    give(m, 0, "hog_rider")
    m.play(0, "hog_rider", 3.5, 14.5)
    tower = m.tower(1, "princess_left")
    m.run_until(8000)
    assert tower.hp < tower.max_hp


def test_cannon_pulls_hog(db):
    m = new(db)
    give(m, 0, "hog_rider")
    m.play(0, "hog_rider", 14.5, 14.5)
    give(m, 1, "cannon")
    m.play(1, "cannon", 9.5, 22.5)
    hog = next(u for u in m.units.values() if u.card_id == "hog_rider")
    cannon = next(u for u in m.units.values() if u.card_id == "cannon")
    m.run_until(4000)
    assert hog.target_uid == cannon.uid


def test_tower_shoots_troop_in_range(db):
    m = new(db)
    give(m, 1, "knight")
    m.play(1, "knight", 3.5, 17.5)
    k = next(u for u in m.units.values() if u.card_id == "knight")
    m.run_until(6000)
    assert k.hp < k.max_hp or k.hp <= 0


def test_fireball_tower_damage_and_king_activation(db):
    m = new(db)
    give(m, 0, "fireball")
    king = m.tower(1, "king")
    m.play(0, "fireball", king.x, king.y)
    m.run_until(3000)
    tdmg = db.stats("fireball", 11).stats["crown_tower_damage"]
    assert king.max_hp - king.hp == pytest.approx(tdmg)
    assert king.active


def test_log_hits_ground_not_air(db):
    air_deck = tuple("minions" if c == "skeletons" else c for c in DECK)
    m = Match(MatchConfig((DECK, air_deck), shuffle=False), db)
    give(m, 1, "knight")
    m.play(1, "knight", 9.5, 20.5)
    give(m, 1, "minions")
    m.play(1, "minions", 9.5, 22.5)
    give(m, 0, "the_log")
    m.play(0, "the_log", 9.5, 14.5)
    m.run_until(3000)  # log rolls 3.3 tiles/s; knight is 6 tiles ahead
    knight = next(u for u in m.units.values() if u.card_id == "knight")
    minions = [u for u in m.units.values() if u.card_id == "minions"]
    assert knight.max_hp - knight.hp == pytest.approx(db.stats("the_log", 11).damage)
    assert all(u.hp == u.max_hp for u in minions)


def test_level16_units_stronger(db):
    m = new(db, levels=(16, 11))
    give(m, 0, "knight")
    m.play(0, "knight", 9, 5)
    k = next(u for u in m.units.values() if u.card_id == "knight")
    assert k.max_hp == db.stats("knight", 16).hitpoints > db.stats("knight", 11).hitpoints
    assert m.tower(0, "king").max_hp > m.tower(1, "king").max_hp


# ---------------------------------------------------------------- end conditions
def test_king_destroyed_ends_match(db):
    m = new(db)
    m.tower(1, "king").hp = 0
    m._resolve_deaths()
    m._check_end()
    assert m.result.winner == 0 and m.result.reason == "king_destroyed" and m.result.crowns == (3, 0)


def test_regulation_end_on_crowns(db):
    m = new(db)
    kill_towers(m, 1, "princess_left")
    m.run_until(m.regulation_ms)
    assert m.result.winner == 0 and m.result.reason == "crowns" and m.result.t_ms == 180_000


def test_overtime_then_tiebreaker(db):
    m = new(db)
    m.run_until(m.regulation_ms + 50)
    assert m.result is None and m.multiplier() == 2
    m.tower(0, "princess_right").hp -= 100
    m.run_until(m.end_ms)
    assert m.result.reason == "tiebreaker" and m.result.winner == 1


def test_draw(db):
    m = new(db)
    m.run_until(m.end_ms)
    assert m.result.winner is None and m.result.reason == "draw"


def test_play_after_match_over(db):
    m = new(db)
    m.run_until(m.end_ms)
    assert m.check_play(0, m.sides[0].hand[0], 9, 5) == "match_over"


# ---------------------------------------------------------------- fog of war
def test_observation_hides_unseen_enemy_units(db):
    m = new(db)
    give(m, 0, "knight")
    ev = m.play(0, "knight", 9, 5).event
    hidden = m.observation(1, is_visible=lambda eid: False)
    shown = m.observation(1, is_visible=lambda eid: eid == ev.event_id)
    assert not any(u["card_id"] == "knight" for u in hidden["units"])
    assert any(u["card_id"] == "knight" for u in shown["units"])
    assert sum(1 for u in hidden["units"] if u["kind"] == "tower") == 6
    assert "elixir" in hidden and hidden["hand"] == m.sides[1].hand  # own info only


# ---------------------------------------------------------------- determinism
def bot_match(db, seed):
    m = Match(MatchConfig((DECK, DECK), seed=seed), db)
    MatchRunner(m, {0: ScriptedBot(db, 0, 0.7), 1: ScriptedBot(db, 1, 0.4)}).run()
    return m


def test_bot_match_completes_and_is_deterministic(db):
    a, b = bot_match(db, 5), bot_match(db, 5)
    assert a.result is not None and len(a.action_log) > 20
    assert a.state_hash() == b.state_hash() and a.action_log == b.action_log


def test_replay_reproduces_match(db):
    a = bot_match(db, 6)
    r = Match.replay(a.cfg, a.action_log, db, until_ms=a.end_ms + 1)
    assert r.state_hash() == a.state_hash() and r.result == a.result


def test_snapshot_branching(db):
    m = Match(MatchConfig((DECK, DECK), seed=7), db)
    runner = MatchRunner(m, {0: ScriptedBot(db, 0), 1: ScriptedBot(db, 1)})
    runner.run(max_ms=60_000)
    snap = m.snapshot()
    m.run_until(90_000)
    h1 = m.state_hash()
    m.restore(snap)
    assert m.t_ms == 60_000
    m.run_until(90_000)
    assert m.state_hash() == h1


def test_human_side_with_bot(db):
    m = Match(MatchConfig((DECK, DECK), seed=8), db)
    runner = MatchRunner(m, {0: None, 1: ScriptedBot(db, 1)})
    assert set(runner.buffers) == {1}
    give(m, 0, "hog_rider")
    assert runner.human_play("hog_rider", 3.5, 14.5).ok
    runner.run(max_ms=5000)
    assert runner.buffers[1].is_visible(m.events[0].event_id)
