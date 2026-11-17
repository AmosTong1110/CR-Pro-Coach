"""
Tests for Phase 3 Core Cards and Mechanics:
- Giant (5 elixir, building targeter, tank)
- P.E.K.K.A (7 elixir, high mass, massive melee damage)
- Baby Dragon (4 elixir, flying air unit, ranged area splash)
- Witch (5 elixir, ranged splash, periodic skeleton summoning)
- Presets and Deck links with Phase 3 cards
"""
import pytest

from backend_engine.meta.archetypes import archetype_name
from backend_engine.meta.deck_link import format_copy_deck_link, parse_deck_link
from backend_engine.meta.presets import load_presets
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig
from backend_engine.simulator.geometry import BRIDGE_X


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


def spawn(m, card_id, side, x, y, evolved=False, variant=None):
    m._spawn_card_units(side, card_id, x, y, eid=m._uid(), evolved=evolved, variant=variant or ("evo" if evolved else "base"))


def test_phase3_cards_exist_and_stats(db):
    """Verify Giant, P.E.K.K.A, Baby Dragon, and Witch stats and attributes."""
    assert "giant" in db.cards
    assert "pekka" in db.cards
    assert "baby_dragon" in db.cards
    assert "witch" in db.cards

    # Giant: Rare, 5 Elixir, 4091 HP, 254 Damage, Building targeter
    g = db.stats("giant", 11)
    assert g.elixir == 5
    assert g.stats["hitpoints"] == 4091
    assert g.stats["damage"] == 254
    assert g.attack["targets"] == ["buildings"]
    assert db.cards["giant"]["mass"] == 6

    # P.E.K.K.A: Epic, 7 Elixir, 3760 HP, 816 Damage (DPS 453), Mass 8
    p = db.stats("pekka", 11)
    assert p.elixir == 7
    assert p.stats["hitpoints"] == 3760
    assert p.stats["damage"] == 816
    assert p.attack["hit_speed_s"] == 1.8
    assert db.cards["pekka"]["mass"] == 8

    # Baby Dragon: Epic, 4 Elixir, 1200 HP, 160 Splash Damage, Air
    bd = db.stats("baby_dragon", 11)
    assert bd.elixir == 4
    assert bd.stats["hitpoints"] == 1200
    assert bd.stats["damage"] == 160
    assert bd.movement["layer"] == "air"
    assert bd.attack["splash_radius_tiles"] == 1.2
    assert set(bd.attack["targets"]) == {"ground", "air"}

    # Witch: Epic, 5 Elixir, 832 HP, 134 Splash Damage, Periodic Spawn
    w = db.stats("witch", 11)
    assert w.elixir == 5
    assert w.stats["hitpoints"] == 832
    assert w.stats["damage"] == 134
    assert w.attack["splash_radius_tiles"] == 1.0
    mechs = [m["kind"] for m in db.cards["witch"]["mechanics"]]
    assert "periodic_spawn" in mechs


def test_giant_building_targeting(db):
    """Giant ignores enemy troops and paths only toward buildings."""
    deck = ("giant", "musketeer", "archers", "skeletons", "fireball", "the_log", "ice_spirit", "zap")
    m = Match(MatchConfig((deck, deck), shuffle=False), db)

    # Spawn Giant at bridge
    spawn(m, "giant", 0, BRIDGE_X[0], 14.0)
    giant = [u for u in m.units.values() if u.card_id == "giant" and u.side == 0][0]
    assert giant.building_only

    # Spawn enemy troop nearby
    spawn(m, "knight", 1, BRIDGE_X[0], 15.5)
    knight = [u for u in m.units.values() if u.card_id == "knight" and u.side == 1][0]

    # Target should be the Princess Tower, not the Knight!
    target = m._acquire(giant)
    assert target is not None
    assert target.is_structure
    assert target.uid != knight.uid


def test_pekka_massive_damage_and_mass(db):
    """P.E.K.K.A deals high single-target melee damage and has heavy push mass."""
    deck = ("pekka", "musketeer", "archers", "skeletons", "fireball", "the_log", "ice_spirit", "zap")
    m = Match(MatchConfig((deck, deck), shuffle=False), db)

    spawn(m, "pekka", 0, 9.0, 9.0)
    pekka = [u for u in m.units.values() if u.card_id == "pekka" and u.side == 0][0]
    assert pekka.mass == 8.0
    assert pekka.damage == 816

    # Spawn a Knight (1766 HP at L11)
    spawn(m, "knight", 1, 9.0, 10.0)
    knight = [u for u in m.units.values() if u.card_id == "knight" and u.side == 1][0]
    initial_hp = knight.hp

    # P.E.K.K.A strikes
    m._attack(pekka, knight)
    assert initial_hp - knight.hp == 816.0


def test_baby_dragon_air_and_splash(db):
    """Baby Dragon flies over ground and deals area splash to multiple targets."""
    deck = ("baby_dragon", "musketeer", "archers", "skeletons", "fireball", "the_log", "ice_spirit", "zap")
    m = Match(MatchConfig((deck, deck), shuffle=False), db)

    spawn(m, "baby_dragon", 0, 9.0, 9.0)
    bd = [u for u in m.units.values() if u.card_id == "baby_dragon" and u.side == 0][0]
    assert bd.layer == "air"
    assert bd.splash == 1.2

    # Spawn two enemy skeletons close together
    spawn(m, "skeletons", 1, 9.0, 11.5)
    skellys = [u for u in m.units.values() if u.card_id == "skeletons" and u.side == 1]
    assert len(skellys) == 3

    # Baby dragon attacks primary target; splash hits adjacent skeletons
    m._attack(bd, skellys[0])
    assert len(m.projectiles) == 1
    proj = m.projectiles[0]
    assert proj.splash == 1.2


def test_witch_periodic_skeleton_spawn(db):
    """Witch periodically spawns a ring of 4 Skeletons around herself."""
    deck = ("witch", "musketeer", "archers", "skeletons", "fireball", "the_log", "ice_spirit", "zap")
    m = Match(MatchConfig((deck, deck), shuffle=False), db)

    spawn(m, "witch", 0, 9.0, 9.0)
    witch = [u for u in m.units.values() if u.card_id == "witch" and u.side == 0][0]
    witch.deploy_ms = 0

    # Before tick, no spawned skeletons on side 0
    skellys_before = [u for u in m.units.values() if u.card_id == "skeletons" and u.side == 0]
    assert len(skellys_before) == 0

    # Step for 1.1s (22 ticks @ 50ms) -> initial delay of 1000ms triggers!
    for _ in range(25):
        m.step()
        if any(u.card_id == "skeletons" and u.side == 0 for u in m.units.values()):
            break

    skellys_after = [u for u in m.units.values() if u.card_id == "skeletons" and u.side == 0]
    assert len(skellys_after) == 4
    for s in skellys_after:
        # Spawned around the witch
        assert abs(s.x - witch.x) <= 2.5
        assert abs(s.y - witch.y) <= 2.5


def test_phase3_presets_and_archetypes(db):
    """Presets using Phase 3 cards are loaded and detectable."""
    presets = load_presets()
    names = {p.name for p in presets}
    assert "Classic Giant Witch" in names
    assert "Classic P.E.K.K.A Control" in names
    assert "Giant Dragon Beatdown" in names

    # Archetype detection
    giant_deck = ("giant", "witch", "musketeer", "mini_pekka", "fireball", "the_log", "skeletons", "zap")
    assert "Giant" in archetype_name(giant_deck)

    pekka_deck = ("pekka", "baby_dragon", "poison", "zap", "dark_prince", "bats", "ice_spirit", "goblins")
    assert "PEKKA" in archetype_name(pekka_deck)

    # Deck link
    link = format_copy_deck_link(giant_deck, db=db)
    parsed = parse_deck_link(link, db=db)
    assert parsed.playable
    assert set(parsed.cards) == set(giant_deck)
