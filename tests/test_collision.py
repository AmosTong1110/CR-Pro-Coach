import math
import time

import pytest

from backend_engine.simulator import geometry as G
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig
from backend_engine.simulator.match_runner import DEFAULT_DECK

AIR_DECK = tuple("minions" if c == "skeletons" else c for c in DEFAULT_DECK)


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


def new(db, collisions=True, decks=(DEFAULT_DECK, DEFAULT_DECK)):
    return Match(MatchConfig(decks, shuffle=False, collisions=collisions), db)


def spawn(m, side, card, x, y, ready=True):
    st = m.sides[side]
    if card not in st.hand:
        st.queue.remove(card)
        st.queue.insert(0, st.hand[0])
        st.hand[0] = card
    st.elixir = 10
    assert m.play(side, card, x, y).ok
    units = [u for u in m.units.values() if u.card_id == card and u.side == side]
    for u in units:
        u.speed = 0.0  # keep them still so we only see collision effects
        if ready:
            u.deploy_ms = 0
    return units


def gap(a, b):
    return math.hypot(a.x - b.x, a.y - b.y) - a.radius - b.radius


def test_stacked_units_separate(db):
    m = new(db)
    (a,) = spawn(m, 0, "knight", 9.5, 8.5)
    (b,) = spawn(m, 0, "musketeer", 9.5, 8.5)
    assert gap(a, b) < -0.9
    m.run_until(1000)
    assert gap(a, b) > -0.05


def test_light_unit_moves_more(db):
    m = new(db)
    (k,) = spawn(m, 0, "knight", 9.5, 8.5)       # mass 6
    skels = spawn(m, 0, "skeletons", 9.5, 8.5)   # mass 1 each
    k0 = (k.x, k.y)
    s0 = [(s.x, s.y) for s in skels]
    m.run_until(500)
    knight_moved = math.dist(k0, (k.x, k.y))
    skel_moved = max(math.dist(p, (s.x, s.y)) for p, s in zip(s0, skels))
    assert skel_moved > 2 * knight_moved


def test_air_and_ground_do_not_collide(db):
    m = new(db, decks=(AIR_DECK, AIR_DECK))
    (k,) = spawn(m, 0, "knight", 9.5, 8.5)
    mins = spawn(m, 0, "minions", 9.5, 8.5)
    k0 = (k.x, k.y)
    m.run_until(500)
    assert math.dist(k0, (k.x, k.y)) == pytest.approx(0, abs=1e-9)
    assert len({(round(u.x, 3), round(u.y, 3)) for u in mins}) == 3  # minions still spread among themselves


def test_towers_and_buildings_are_immovable(db):
    m = new(db)
    tower = m.tower(0, "princess_left")
    t0 = (tower.x, tower.y)
    (k,) = spawn(m, 0, "knight", 5.5, 6.5)       # overlapping the tower's edge (radius 1.5 + 0.5)
    m.run_until(1500)
    assert (tower.x, tower.y) == t0
    assert gap(k, tower) > -0.05


def test_not_pushed_into_river(db):
    m = new(db)
    (k,) = spawn(m, 0, "knight", 9.5, 14.5)       # river bank, far from bridges
    for s in spawn(m, 0, "skeletons", 9.5, 14.5):
        pass
    m.run_until(1000)
    assert G.region(k.y) != "river"
    for u in m.units.values():
        if u.kind == "troop" and u.layer == "ground" and not u.river_jump:
            assert G.region(u.y) != "river" or G.on_bridge(u.x)


def test_collisions_can_be_disabled(db):
    m = new(db, collisions=False)
    (a,) = spawn(m, 0, "knight", 9.5, 8.5)
    (b,) = spawn(m, 0, "musketeer", 9.5, 8.5)
    m.run_until(1000)
    assert (a.x, a.y) == (b.x, b.y)


def test_crowd_is_deterministic_and_fast(db):
    def crowd():
        m = new(db)
        for side, ys in ((0, (10.5, 12.5)), (1, (21.5, 19.5))):
            for y in ys:
                for x in (3.5, 6.5, 11.5, 14.5):
                    st = m.sides[side]
                    st.elixir = 10
                    c = next(c for c in st.hand if db.cards[c]["type"] == "troop")
                    m.play(side, c, x, y)
        t = time.perf_counter()
        m.run_until(10_000)
        return m.state_hash(), time.perf_counter() - t

    (h1, dt), (h2, _) = crowd(), crowd()
    assert h1 == h2
    assert dt < 3.0  # 200 ticks with a crowded board
