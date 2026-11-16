"""Evolutions batch 2: Firecracker sparks, Skeletons multiply, double Zap, Valkyrie tornado, Tesla pulse."""
import pytest

from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig
from backend_engine.simulator.mechanics import status as S

DECK = ("firecracker", "skeletons", "zap", "valkyrie", "tesla", "knight", "hog_rider", "minions")
FOE = ("mini_pekka", "knight", "hog_rider", "minions", "skeletons", "zap", "valkyrie", "cannon")


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


def _match(db, evo=(), level=11):
    m = Match(MatchConfig((DECK, FOE), (level, level), (tuple(evo), ()), seed=0, shuffle=False, collisions=False), db)
    for st in m.sides:
        st.hand = list(st.hand) + [c for c in st.queue]          # every card playable for the tests
        st.queue = []
        st.elixir = 10.0
    for u in m.units.values():
        if u.kind == "tower":
            u.damage = 0                                         # towers don't interfere with the counts
    return m


def _play(m, side, card, x, y, evolve=True):
    st = m.sides[side]
    st.elixir = 10.0
    if evolve and card in st.evo_counter:
        st.evo_counter[card] = m.db.evo_cycles(card)
    st.hand = [card] + [c for c in st.hand if c != card]
    st.queue = [c for c in st.queue if c != card] or [st.hand[-1]]
    res = m.play(side, card, x, y)
    assert res.ok, res.reason
    return res.event


def _dummy(m, side, card, x, y, hp=None):
    m._spawn_card_units(side, card, x, y, 999, False, deploy_ms=0, count=1)
    u = max(m.units.values(), key=lambda u: u.uid)
    u.speed, u.damage = 0.0, 0                                   # a punching bag that stays put
    if hp:
        u.hp = u.max_hp = hp
    return u


def _run(m, ms):
    for _ in range(ms // 50):
        m.step()


def test_the_five_evolutions_are_selectable(db):
    assert {"firecracker", "skeletons", "zap", "valkyrie", "tesla"} <= set(db.evo_capable())
    Match(MatchConfig((DECK, FOE), evo_cards=(("firecracker", "tesla"), ())), db)   # accepted as slots


def test_evo_zap_hits_twice_and_wider(db):
    m = _match(db, evo=("zap",))
    near = _dummy(m, 1, "knight", 9.5, 20.5, hp=5000)
    ring = _dummy(m, 1, "knight", 9.5 + 2.5 + 0.3 + near.radius, 20.5, hp=5000)    # outside 2.5, inside 3.0
    ev = _play(m, 0, "zap", 9.5, 20.5)
    assert ev.is_evolved
    dmg = db.stats("zap", 11, "evo").stats["damage"]
    _run(m, 2000)
    assert 5000 - near.hp == pytest.approx(2 * dmg)
    assert 5000 - ring.hp == pytest.approx(dmg)                  # only the second, wider hit
    base = _match(db)
    b = _dummy(base, 1, "knight", 9.5, 20.5, hp=5000)
    _play(base, 0, "zap", 9.5, 20.5, evolve=False)
    _run(base, 2000)
    assert 5000 - b.hp == pytest.approx(dmg)


def test_evo_skeletons_multiply_up_to_eight(db):
    m = _match(db, evo=("skeletons",))
    _dummy(m, 1, "knight", 9.5, 12.0, hp=100_000)
    ev = _play(m, 0, "skeletons", 9.5, 10.5)
    peak = 0
    for _ in range(300):
        m.step()
        mine = [u for u in m.units.values() if u.side == 0 and u.source_event_id == ev.event_id]
        peak = max(peak, len(mine))
        assert len(mine) <= 8 and all(u.evolved and u.variant == "evo" for u in mine)
    assert peak == 8
    base = _match(db)
    _dummy(base, 1, "knight", 9.5, 12.0, hp=100_000)
    ev = _play(base, 0, "skeletons", 9.5, 10.5, evolve=False)
    _run(base, 15_000)
    assert len([u for u in base.units.values() if u.side == 0 and u.source_event_id == ev.event_id]) == 3


def test_evo_valkyrie_tornado_pulls_and_hurts_air(db):
    m = _match(db, evo=("valkyrie",))
    tank = _dummy(m, 1, "knight", 9.5, 11.5, hp=100_000)        # what she swings at
    far = _dummy(m, 1, "knight", 13.0, 10.0, hp=100_000)        # 3.8 tiles away: pulled, not hit by the axe
    air = _dummy(m, 1, "minions", 7.0, 9.0, hp=100_000)
    _play(m, 0, "valkyrie", 9.5, 10.5)
    d0 = ((far.x - 9.5) ** 2 + (far.y - 10.5) ** 2) ** 0.5
    _run(m, 3000)
    t_dmg = db.stats("valkyrie", 11, "evo").stats["tornado_damage"]
    assert t_dmg == 84
    swings = round((100_000 - air.hp) / t_dmg)
    assert swings >= 1 and 100_000 - air.hp == pytest.approx(swings * t_dmg)
    assert 100_000 - far.hp >= swings * t_dmg                     # tornado hits (+ the axe once pulled in)
    assert ((far.x - 9.5) ** 2 + (far.y - 10.5) ** 2) ** 0.5 < d0 - 0.5
    assert tank.hp < 100_000 - swings * t_dmg                       # axe + tornado
    base = _match(db)
    a2 = _dummy(base, 1, "minions", 7.0, 9.0, hp=100_000)
    _dummy(base, 1, "knight", 9.5, 11.5, hp=100_000)
    _play(base, 0, "valkyrie", 9.5, 10.5, evolve=False)
    _run(base, 3000)
    assert a2.hp == 100_000                                          # a plain Valkyrie can't hit air


def test_evo_tesla_pulses_when_it_surfaces_and_when_it_dies(db):
    m = _match(db, evo=("tesla",))
    t = None
    _play(m, 0, "tesla", 9.5, 8.5)
    t = max((u for u in m.units.values() if u.card_id == "tesla"), key=lambda u: u.uid)
    _run(m, 2000)                                                    # deployed, nothing near: hidden
    assert t.mstate.get("hidden")
    pulse = db.stats("tesla", 11, "evo").stats["pulse_damage"]
    assert pulse == 174
    foe = _dummy(m, 1, "knight", 9.5, 12.5, hp=100_000)
    m.step(); m.step()
    assert 100_000 - foe.hp >= pulse and S.STUN in foe.status     # surfaced: pulse + stun
    hp_after = foe.hp
    t.hp = 0                                                         # destroyed: death pulse
    m.step()
    assert hp_after - foe.hp >= pulse


def test_evo_firecracker_sparks(db):
    m = _match(db, evo=("firecracker",))
    tgt = _dummy(m, 1, "knight", 9.5, 15.0, hp=100_000)
    _play(m, 0, "firecracker", 9.5, 9.5)
    for _ in range(200):
        m.step()
        if m.effects:
            break
    assert m.effects and m.effects[0].kind == "spark_field"
    ef = m.effects[0]
    assert len(ef.params["circles"]) == 6 and ef.damage == 48 and ef.tower_damage == 15
    hp0 = tgt.hp
    _run(m, 400)
    assert tgt.hp < hp0 and any(k.startswith("slow") for k in tgt.status)
    # one big-spark field = 12 pulses of 48 on a target that stays inside (sparks don't stack)
    m2 = _match(db)
    from backend_engine.simulator.entities import AreaEffect
    sitter = _dummy(m2, 1, "knight", 9.5, 20.0, hp=100_000)
    m2.effects.append(AreaEffect(1, 0, "firecracker", 9.5, 20.0, 2.5, 48.0, 15.0, 3000, 250, 0.0, 0,
                                 kind="spark_field", next_hit_ms=250,
                                 params={"circles": [[9.5, 20.0, 2.5, 3000], [9.5, 20.5, 1.2, 2500]],
                                         "slow": S.SLOW15, "slow_ms": 350}))
    _run(m2, 3500)
    assert 100_000 - sitter.hp == pytest.approx(12 * 48)


def test_evo_matches_are_deterministic(db):
    from backend_engine.brain.scripted_bot import ScriptedBot
    from backend_engine.simulator.match_runner import MatchRunner
    hashes = []
    for _ in range(2):
        deck = ("hog_rider", "firecracker", "skeletons", "zap", "valkyrie", "tesla", "the_log", "ice_spirit")
        m = Match(MatchConfig((deck, deck), evo_cards=(("firecracker", "tesla"), ("skeletons", "valkyrie")),
                              seed=7), db)
        MatchRunner(m, {0: ScriptedBot(db, 0, 0.7), 1: ScriptedBot(db, 1, 0.7)}).run()
        hashes.append(m.state_hash())
        assert any(e.is_evolved for e in m.events)
    assert hashes[0] == hashes[1]
