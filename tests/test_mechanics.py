"""Engine v2 mechanics: projectiles with travel time, knockback (Fireball, Log), Evo Knight damage reduction."""
import pytest

from backend_engine.simulator import geometry as G
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import ENGINE_VERSION, Match, MatchConfig
from backend_engine.simulator.match_runner import DEFAULT_DECK

AIR_DECK = tuple("minions" if c == "skeletons" else c for c in DEFAULT_DECK)


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


def new(db, decks=(DEFAULT_DECK, DEFAULT_DECK), evo=((), ())):
    return Match(MatchConfig(decks, evo_cards=evo, shuffle=False, collisions=False), db)


def play(m, side, card, x, y, ready=True, still=True):
    """Deploy (on a legal tile of that side) then place the new units at (x, y): test setups only."""
    st = m.sides[side]
    if card not in st.hand:
        st.queue.remove(card)
        st.queue.insert(0, st.hand[0])
        st.hand[0] = card
    st.elixir = 10
    before = set(m.units)
    legal = (x, y) if m.check_play(side, card, x, y) is None else (9.5, 8.5 if side == 0 else 23.5)
    assert m.play(side, card, *legal).ok
    units = [m.units[k] for k in sorted(set(m.units) - before)]
    for u in units:
        u.x, u.y = u.x - legal[0] + x, u.y - legal[1] + y
        u.lane = "left" if u.x < G.W / 2 else "right"       # the lane follows the moved unit, as at spawn
        if ready:
            u.deploy_ms = 0
        if still:
            u.speed = 0.0
    return units


def test_engine_version():
    assert ENGINE_VERSION == 7


# ---------------------------------------------------------------- projectiles
def test_musketeer_damage_lands_after_travel_time(db):
    m = new(db)
    (mus,) = play(m, 0, "musketeer", 9.5, 9.5)
    (k,) = play(m, 1, "knight", 9.5, 14.0)          # 4.5 tiles away, inside range 6
    k.hp = k.max_hp = 10_000
    for t in m.units.values():
        if t.kind == "tower":
            t.active = False                         # only the Musketeer shoots
    mus.cooldown_ms = 0
    m.step()                                         # shot fired
    assert len(m.projectiles) == 1 and k.hp == k.max_hp
    p = m.projectiles[0]
    d = G.dist(p.x, p.y, k.x, k.y)
    ticks = 0
    while m.projectiles and m.projectiles[0].uid == p.uid:
        m.step()
        ticks += 1
    step = p.speed * G.TICK_MS / 1000
    assert ticks == pytest.approx(d / step, abs=1.5)  # ~4.5 tiles at 16.7 tiles/s = ~5-6 ticks
    assert k.max_hp - k.hp == mus.damage


def test_projectile_vanishes_when_target_dies(db):
    m = new(db)
    (arc, _) = play(m, 0, "archers", 9.5, 9.5)
    (k,) = play(m, 1, "knight", 9.5, 14.0)
    arc.cooldown_ms = 0
    m.step()
    assert m.projectiles
    k.hp = 0
    m._resolve_deaths()
    m.step()
    assert not m.projectiles


def test_towers_and_cannon_shoot_projectiles(db):
    m = new(db)
    assert all("projectile" in u.mechanics for u in m.units.values() if u.kind == "tower")
    (c,) = play(m, 0, "cannon", 9.5, 8.5)
    assert "projectile" in c.mechanics and "decay" in c.mechanics


def test_projectile_hash_and_snapshot(db):
    m = new(db)
    play(m, 0, "musketeer", 9.5, 9.5)
    (k,) = play(m, 1, "knight", 9.5, 14.0)
    k.hp = k.max_hp = 10_000
    m.step()
    snap, h = m.snapshot(), m.state_hash()
    m.run_until(m.t_ms + 2000)
    h2 = m.state_hash()
    m.restore(snap)
    assert m.state_hash() == h and m.projectiles
    m.run_until(m.t_ms + 2000)
    assert m.state_hash() == h2


# ---------------------------------------------------------------- knockback
def test_fireball_pushes_troops_not_towers(db):
    m = new(db)
    (k,) = play(m, 1, "knight", 9.5, 21.5)
    k.hp = k.max_hp = 10_000
    tower = m.tower(1, "princess_left")
    tx, ty = tower.x, tower.y
    play(m, 0, "fireball", 9.5, 20.5)               # 1 tile below the knight: pushed up (+y)
    m.run_until(m.t_ms + 3000)
    assert k.y == pytest.approx(22.5, abs=1e-6) and k.x == pytest.approx(9.5)
    assert (tower.x, tower.y) == (tx, ty)


def test_log_pushes_ground_forward_not_air(db):
    m = new(db, (DEFAULT_DECK, AIR_DECK))
    (k,) = play(m, 1, "knight", 9.5, 20.5)
    k.hp = k.max_hp = 10_000
    minions = play(m, 1, "minions", 9.5, 22.5)
    before = [(u.x, u.y) for u in minions]
    play(m, 0, "the_log", 9.5, 14.5)
    m.run_until(m.t_ms + 3500)
    assert k.y == pytest.approx(21.2, abs=1e-6)       # 0.7 tiles along the roll direction
    assert [(u.x, u.y) for u in minions] == before


def test_knockback_never_pushes_into_river(db):
    m = new(db)
    (k,) = play(m, 0, "knight", 9.5, G.RIVER_Y0 - 0.3)  # just below the river, own side
    k.hp = k.max_hp = 10_000
    play(m, 1, "fireball", 9.5, G.RIVER_Y0 - 1.3)      # pushes the knight up, towards the water
    m.run_until(m.t_ms + 3000)
    assert G.region(k.y) != "river"


# ---------------------------------------------------------------- Evo Knight
def test_evo_knight_reduction_only_while_not_attacking(db):
    m = new(db, evo=(("knight",), ()))
    m.sides[0].evo_counter["knight"] = 2
    (evo,) = play(m, 0, "knight", 3.5, 8.5)
    (base,) = play(m, 0, "knight", 14.5, 8.5)
    assert evo.variant == "evo" and base.variant == "base"
    for u in (evo, base):
        u.hp = 1000
    m._deal(evo, 100)
    m._deal(base, 100)
    assert evo.hp == pytest.approx(960) and base.hp == pytest.approx(900)
    (enemy,) = play(m, 1, "knight", 3.5, 9.6)        # in melee range: now it is attacking
    evo.target_uid = enemy.uid
    m._deal(evo, 100)
    assert evo.hp == pytest.approx(860)
