"""
Tests for Phase 2 Core Cards and Evolution Mechanics:
- X-Bow (Siege, 11.5 range, 0.3s rapid fire, decay)
- Mortar & Evolved Mortar (11.5 range, 3.5 blind spot, area projectile, Evo Goblin spawn on hit)
- Inferno Tower (Ramp damage stages 1/2/3, beam retarget on stun, anti-tank melting)
- Goblins (3-count swarm, high melee DPS, speed)
- Archetypes and Presets (Classic 3.0 X-Bow, Classic Mortar Bait, Classic Log Bait)
"""
import pytest

from backend_engine.meta.archetypes import archetype_name
from backend_engine.meta.deck_link import format_copy_deck_link, parse_deck_link
from backend_engine.meta.presets import find_preset, load_presets
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig
from backend_engine.simulator.geometry import BRIDGE_X, RIVER_Y0, RIVER_Y1
from backend_engine.simulator.mechanics import status as S


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


def spawn(m, card_id, side, x, y, evolved=False, variant=None):
    m._spawn_card_units(side, card_id, x, y, eid=m._uid(), evolved=evolved, variant=variant or ("evo" if evolved else "base"))


def test_phase2_cards_exist_and_stats(db):
    """Verify all 4 new cards and the token are registered with valid stats."""
    assert "x_bow" in db.cards
    assert "mortar" in db.cards
    assert "inferno_tower" in db.cards
    assert "goblins" in db.cards
    assert "goblin" in db.tokens

    # X-Bow: Epic, 6 Elixir, 11.5 Range, 0.3s Hit Speed
    xb = db.stats("x_bow", 11)
    assert xb.elixir == 6
    assert xb.type == "building"
    assert xb.attack["range_tiles"] == 11.5
    assert xb.attack["hit_speed_s"] == 0.3
    assert xb.stats["hitpoints"] == 1600
    assert xb.stats["damage"] == 42

    # Mortar: Common, 4 Elixir, 11.5 Range, 3.5 Min Range (blind spot), 2.0 Splash
    mo = db.stats("mortar", 11)
    assert mo.elixir == 4
    assert mo.type == "building"
    assert mo.attack["range_tiles"] == 11.5
    assert mo.attack["min_range_tiles"] == 3.5
    assert mo.attack["splash_radius_tiles"] == 2.0
    assert mo.stats["hitpoints"] == 1472
    assert mo.stats["damage"] == 266

    # Inferno Tower: Rare, 5 Elixir, 6.0 Range, 0.4s Hit Speed, Ramp stages
    it = db.stats("inferno_tower", 11)
    assert it.elixir == 5
    assert it.type == "building"
    assert it.attack["range_tiles"] == 6.0
    assert it.attack["hit_speed_s"] == 0.4
    assert it.stats["hitpoints"] == 1730
    assert it.stats["damage"] == 42
    assert it.stats["damage_stage2"] == 173
    assert it.stats["damage_stage3"] == 868

    # Goblins: Common, 2 Elixir, count 3, 1.5 speed
    gob = db.stats("goblins", 11)
    assert gob.elixir == 2
    assert gob.type == "troop"
    assert db.cards["goblins"]["count"] == 3
    assert gob.stats["hitpoints"] == 202
    assert gob.stats["damage"] == 120


def test_mortar_blind_spot(db):
    """Mortar must not target units within its 3.5 tile minimum range (blind spot)."""
    deck = ("mortar", "knight", "archers", "skeletons", "fireball", "the_log", "ice_spirit", "musketeer")
    m = Match(MatchConfig((deck, deck), shuffle=False), db)

    # Spawn Mortar for side 0 at (BRIDGE_X[0], 12.0)
    mx, my = BRIDGE_X[0], 12.0
    spawn(m, "mortar", 0, mx, my)
    mortar_unit = [u for u in m.units.values() if u.card_id == "mortar" and u.side == 0][0]
    assert mortar_unit.min_range == 3.5
    assert mortar_unit.range == 11.5

    # Spawn enemy inside blind spot: 1.5 tiles away (at mx, my + 1.5)
    spawn(m, "knight", 1, mx, my + 1.5)
    close_enemy = [u for u in m.units.values() if u.card_id == "knight" and u.side == 1][0]
    assert not m._in_range(mortar_unit, close_enemy)

    # Spawn enemy outside blind spot, within 11.5: 6.0 tiles away
    spawn(m, "musketeer", 1, mx, my + 6.0)
    far_enemy = [u for u in m.units.values() if u.card_id == "musketeer" and u.side == 1][0]
    assert m._in_range(mortar_unit, far_enemy)

    # Acquire target should pick the far enemy and ignore the one inside the blind spot
    target = m._acquire(mortar_unit)
    assert target is not None
    assert target.uid == far_enemy.uid


def test_x_bow_siege_lock(db):
    """X-Bow placed at bridge reaches enemy Princess Tower across the river."""
    deck = ("x_bow", "tesla", "archers", "knight", "the_log", "fireball", "skeletons", "ice_spirit")
    m = Match(MatchConfig((deck, deck), shuffle=False), db)

    # Place X-Bow at left bridge (BRIDGE_X[0], 14.5)
    spawn(m, "x_bow", 0, BRIDGE_X[0], 14.5)
    xb = [u for u in m.units.values() if u.card_id == "x_bow" and u.side == 0][0]
    assert xb.range == 11.5

    enemy_tower = [u for u in m.units.values() if u.side == 1 and u.tower_role == "princess_left"][0]
    # Distance from (3.5, 14.5) to (3.5, 25.5) edge to edge
    assert m._in_range(xb, enemy_tower)

    target = m._acquire(xb)
    assert target.uid == enemy_tower.uid


def test_inferno_tower_ramp_and_stun_reset(db):
    """Inferno Tower damage ramps up across stages and resets on stun."""
    deck = ("inferno_tower", "zap", "knight", "archers", "skeletons", "fireball", "the_log", "ice_spirit")
    m = Match(MatchConfig((deck, deck), shuffle=False), db)

    spawn(m, "inferno_tower", 0, 9.0, 9.0)
    it = [u for u in m.units.values() if u.card_id == "inferno_tower"][0]
    it.deploy_ms = 0
    it.cooldown_ms = 0

    # High HP target
    spawn(m, "knight", 1, 9.0, 12.0)
    tank = [u for u in m.units.values() if u.card_id == "knight" and u.side == 1][0]
    tank.max_hp = 10000.0
    tank.hp = 10000.0

    # Initial hit at stage 1
    m._attack(it, tank)
    assert it.mstate.get("ramp_ms") == 0
    hp_after_1 = tank.hp
    # Deals initial stage 1 damage (42)
    assert 10000.0 - hp_after_1 == 42.0

    # Advance ramp time to 2500ms (Stage 2)
    it.mstate["ramp_ms"] = 2500
    it.mstate["ramp_target"] = tank.uid
    m._attack(it, tank)
    hp_after_2 = tank.hp
    # Deals stage 2 damage (173)
    assert hp_after_1 - hp_after_2 == 173.0

    # Advance ramp time to 4500ms (Stage 3)
    it.mstate["ramp_ms"] = 4500
    m._attack(it, tank)
    hp_after_3 = tank.hp
    # Deals stage 3 maximum melt damage (868)
    assert hp_after_2 - hp_after_3 == 868.0

    # Stun Inferno Tower (e.g. Zap)
    m.apply_status(it, S.STUN, 500)
    assert it.target_uid is None

    # Next attack should reset back to stage 1!
    it.status.clear()
    m._attack(it, tank)
    assert it.mstate["ramp_ms"] == 0
    hp_after_reset = tank.hp
    assert hp_after_3 - hp_after_reset == 42.0


def test_evolved_mortar_spawns_goblin_on_hit(db):
    """Evolved Mortar projectile landing spawns a Goblin at the target."""
    deck = ("mortar", "knight", "archers", "skeletons", "fireball", "the_log", "ice_spirit", "musketeer")
    m = Match(MatchConfig((deck, deck), shuffle=False, evo_cards=(("mortar",), ())), db)

    spawn(m, "mortar", 0, BRIDGE_X[0], 12.0, variant="evo", evolved=True)
    mo = [u for u in m.units.values() if u.card_id == "mortar" and u.side == 0][0]
    mo.deploy_ms = 0
    mo.cooldown_ms = 0

    # Target
    spawn(m, "knight", 1, BRIDGE_X[0], 18.0)
    target = [u for u in m.units.values() if u.card_id == "knight" and u.side == 1][0]
    target.max_hp = 5000.0
    target.hp = 5000.0

    # Mortar attacks, firing projectile
    m._attack(mo, target)
    mo.cooldown_ms = mo.hit_speed_ms
    assert len(m.projectiles) == 1
    proj = m.projectiles[0]
    assert "evo_goblin_mortar" in proj.params

    # Advance ticks until projectile hits and Goblin spawns
    goblins_before = [u for u in m.units.values() if u.card_id == "goblin"]
    assert len(goblins_before) == 0

    for _ in range(25):
        m.step()
        if any(u.card_id == "goblin" and u.side == 0 for u in m.units.values()):
            break

    # Goblin token should now be spawned on the field on side 0!
    goblins_after = [u for u in m.units.values() if u.card_id == "goblin" and u.side == 0]
    assert len(goblins_after) == 1
    assert abs(goblins_after[0].x - target.x) < 2.0


def test_goblins_swarm_spawn(db):
    """Goblins card spawns 3 fast melee units."""
    deck = ("goblins", "knight", "archers", "skeletons", "fireball", "the_log", "ice_spirit", "musketeer")
    m = Match(MatchConfig((deck, deck), shuffle=False), db)

    spawn(m, "goblins", 0, 9.0, 9.0)
    gobs = [u for u in m.units.values() if u.card_id == "goblins" and u.side == 0]
    assert len(gobs) == 3
    for g in gobs:
        assert g.speed == 1.5
        assert g.range == 0.5
        assert g.hp == 202


def test_phase2_presets_and_deck_links(db):
    """Presets for X-Bow, Mortar Bait, and Log Bait are valid and playable."""
    presets = load_presets()
    names = {p.name for p in presets}
    assert "Classic 3.0 X-Bow" in names
    assert "Classic Mortar Bait" in names
    assert "Classic Log Bait" in names

    # Archetype detection
    xbow_deck = ("x_bow", "tesla", "archers", "knight", "the_log", "fireball", "skeletons", "ice_spirit")
    assert "XBow" in archetype_name(xbow_deck)

    mortar_deck = ("mortar", "miner", "goblins", "bats", "musketeer", "fireball", "the_log", "ice_spirit")
    assert "Mortar" in archetype_name(mortar_deck)

    bait_deck = ("inferno_tower", "goblins", "knight", "dart_goblin", "rocket", "the_log", "ice_spirit", "skeleton_army")
    # Named after most expensive or key building
    arch = archetype_name(bait_deck)
    assert "Rocket" in arch or "Inferno" in arch

    # Official deck link export & import
    link = format_copy_deck_link(xbow_deck, evo=("tesla",), db=db)
    parsed = parse_deck_link(link, db=db)
    assert parsed.playable
    assert set(parsed.cards) == set(xbow_deck)
    assert "tesla" in parsed.evo
