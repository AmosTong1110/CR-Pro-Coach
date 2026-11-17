"""
Unit tests for physics & game simulation logic audit:
1. Multi-unit spawn formations & king tower lane splitting
2. Cross-lane pathing, bridge waypoints & lane locking
3. Sight range (aggro_range), pull mechanics & building-only targeting
4. Tower hitbox collision & unit crowding/stuck logic
5. Deploy delays, melee/ranged attack loops & retargeting
6. Spell travel time, river obstacles & bridge jumpers
"""
import pytest

from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig
from backend_engine.simulator.geometry import BRIDGE_X, W


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


def test_task1_archers_king_tower_split(db):
    """Deploying Archers at/behind the King Tower (x=9.0) causes one to go left and one to go right."""
    deck = ("archers", "knight", "skeletons", "fireball", "the_log", "ice_spirit", "musketeer", "cannon")
    cfg = MatchConfig((deck, deck), (11, 11), seed=42, shuffle=False)
    m = Match(cfg, db)

    # Deploy archers behind King Tower at x = 9.0, y = 0.5
    res = m.play(0, "archers", 9.0, 0.5)
    assert res.ok

    # Find the two spawned archers
    archers = [u for u in m.units.values() if u.card_id == "archers" and u.side == 0]
    assert len(archers) == 2

    # Check horizontal spawn formation offsets
    x_coords = sorted(a.x for a in archers)
    assert x_coords[0] < 9.0, f"Left archer spawned at {x_coords[0]}, expected < 9.0"
    assert x_coords[1] >= 9.0, f"Right archer spawned at {x_coords[1]}, expected >= 9.0"

    # Step match for 35 ticks (deploy delay is 1000ms = 20 ticks, so 15 ticks of movement)
    for _ in range(35):
        m.step()

    # Verify lane pathing: one moves leftwards toward Left Bridge (x < 9.0), other toward Right Bridge (x >= 9.0)
    left_archer = min(archers, key=lambda a: a.x)
    right_archer = max(archers, key=lambda a: a.x)

    assert left_archer.x < 9.0
    assert right_archer.x >= 9.0

    # Ensure left archer is heading towards left bridge (x = 3.5) and right archer towards right bridge (x = 14.5)
    assert left_archer.lane == "left"
    assert right_archer.lane == "right"
    assert left_archer.x < 8.75  # actively moving leftward towards left bridge (3.5)
    assert right_archer.x > 10.25  # actively moving rightward towards right bridge (14.5)


def test_task2_lane_locking_and_bridge_pathing(db):
    """Ground troop deployed in right lane routes strictly via right bridge and locks on right princess tower."""
    deck = ("knight", "archers", "skeletons", "fireball", "the_log", "ice_spirit", "musketeer", "cannon")
    cfg = MatchConfig((deck, deck), (11, 11), seed=123, shuffle=False)
    m = Match(cfg, db)

    # Spawn Knight in right lane at (14.5, 9.0)
    res = m.play(0, "knight", 14.5, 9.0)
    assert res.ok

    knight = [u for u in m.units.values() if u.card_id == "knight" and u.side == 0][0]
    assert knight.lane == "right"

    # Step unopposed for 400 ticks (20 seconds) - Knight walks across bridge to right tower
    reached_river = False
    reached_tower = False
    enemy_rp_tower = m.tower(1, "princess_right")
    initial_hp = enemy_rp_tower.hp

    for _ in range(400):
        m.step()
        if knight.hp <= 0:
            break

        # Ground troop must NEVER drift into left side of arena
        assert knight.x >= 9.0, f"Knight crossed into left lane at x={knight.x}, y={knight.y}"

        # In river zone, must be on right bridge
        if 15.0 <= knight.y <= 17.0:
            reached_river = True
            assert abs(knight.x - 14.5) <= 1.0, f"Knight off bridge in river: x={knight.x}"

        # Target must be princess_right
        if knight.target_uid is not None:
            tgt = m.units.get(knight.target_uid)
            assert tgt is not None
            assert tgt.tower_role in ("princess_right", "king"), f"Unexpected target role: {tgt.tower_role}"

        if enemy_rp_tower.hp < initial_hp:
            reached_tower = True
            break

    assert reached_river, "Knight should have crossed the river"
    assert reached_tower, "Knight should have reached and damaged the right Princess Tower"


def test_task2_tower_fallback_to_king_not_cross_lane(db):
    """When the assigned lane's Princess Tower is destroyed, units path to King Tower, NOT opposite lane."""
    deck = ("knight", "archers", "skeletons", "fireball", "the_log", "ice_spirit", "musketeer", "cannon")
    cfg = MatchConfig((deck, deck), (11, 11), seed=123, shuffle=False)
    m = Match(cfg, db)

    # Destroy side 1's right princess tower
    enemy_rp = m.tower(1, "princess_right")
    enemy_rp.hp = 0

    # Spawn Knight in right lane
    res = m.play(0, "knight", 14.5, 9.0)
    assert res.ok

    knight = [u for u in m.units.values() if u.card_id == "knight" and u.side == 0][0]
    assert knight.lane == "right"

    # Step past deploy delay
    for _ in range(25):
        m.step()

    # Target should be enemy King Tower, NOT opposite princess tower
    tgt = m.units.get(knight.target_uid)
    assert tgt is not None
    assert tgt.tower_role == "king", f"Expected target to be King tower, but got {tgt.tower_role}"


def test_task3_building_only_ignores_troops(db):
    """Building-only units (Hog Rider) strictly ignore nearby enemy troops and target towers/buildings."""
    deck0 = ("hog_rider", "knight", "archers", "skeletons", "fireball", "the_log", "ice_spirit", "cannon")
    deck1 = ("skeletons", "knight", "archers", "fireball", "the_log", "ice_spirit", "cannon", "musketeer")
    cfg = MatchConfig((deck0, deck1), (11, 11), seed=42, shuffle=False)
    m = Match(cfg, db)

    # Spawn Hog Rider for side 0 in right lane at (14.5, 9.0)
    res0 = m.play(0, "hog_rider", 14.5, 9.0)
    assert res0.ok
    hog = [u for u in m.units.values() if u.card_id == "hog_rider" and u.side == 0][0]
    assert hog.building_only

    # Spawn enemy Skeletons for side 1 in south territory (e.g. at 14.5, 11.0, right in front of Hog)
    # Give side 1 enough elixir and drop skeletons
    m.sides[1].elixir = 10.0
    # Skeletons can be deployed legally in side 1 territory or pocket; or we can inject them directly
    res1 = m.play(1, "skeletons", 14.5, 20.0)
    assert res1.ok
    enemy_skels = [u for u in m.units.values() if u.card_id == "skeletons" and u.side == 1]
    assert len(enemy_skels) > 0
    # Manually place one skeleton right next to hog (e.g. at 14.5, 11.0, distance 2.0 tiles)
    enemy_skels[0].x = 14.5
    enemy_skels[0].y = 11.0

    # Step past deploy delay
    for _ in range(25):
        m.step()

    # Hog must target right princess tower, NEVER the skeleton right next to him
    tgt = m.units.get(hog.target_uid)
    assert tgt is not None
    assert tgt.uid != enemy_skels[0].uid
    assert tgt.tower_role == "princess_right"


def test_task3_sight_range_pull_and_reacquisition(db):
    """Troops only aggro within sight_range and return to lane bridge waypoint when enemy target dies."""
    deck = ("knight", "archers", "skeletons", "fireball", "the_log", "ice_spirit", "musketeer", "cannon")
    cfg = MatchConfig((deck, deck), (11, 11), seed=42, shuffle=False)
    m = Match(cfg, db)

    # Spawn Knight for side 0 at (14.5, 9.0)
    res = m.play(0, "knight", 14.5, 9.0)
    assert res.ok
    knight = [u for u in m.units.values() if u.card_id == "knight" and u.side == 0][0]

    # Spawn enemy skeletons at side 1 backfield
    res_skel = m.play(1, "skeletons", 14.5, 23.0)
    assert res_skel.ok
    enemy_skels = [u for u in m.units.values() if u.card_id == "skeletons" and u.side == 1]
    skel = enemy_skels[0]

    # Position skeleton far away (distance 8.0 tiles > sight 5.5) at (14.5, 17.5)
    skel.x = 14.5
    skel.y = 17.5

    # Step past deploy delay
    for _ in range(25):
        m.step()

    # Knight should NOT aggro onto far skeleton; target should be princess_right
    tgt = m.units.get(knight.target_uid)
    assert tgt is not None
    assert tgt.tower_role == "princess_right"

    # Now move skeleton into sight (e.g. at 11.0, distance ~2.0 tiles from knight)
    skel.x = 14.5
    skel.y = knight.y + 2.0
    m.step()

    # Knight should now aggro onto skeleton
    assert knight.target_uid == skel.uid

    # Kill skeleton
    skel.hp = 0
    m.step()

    # Knight should reacquire right princess tower and resume pathing to bridge
    tgt_after = m.units.get(knight.target_uid)
    assert tgt_after is not None
    assert tgt_after.tower_role == "princess_right"


def test_task4_princess_tower_sliding(db):
    """Troop deployed directly behind Princess Tower smoothly slides around without vibrating in place."""
    deck = ("knight", "archers", "skeletons", "fireball", "the_log", "ice_spirit", "musketeer", "cannon")
    cfg = MatchConfig((deck, deck), (11, 11), seed=42, shuffle=False)
    m = Match(cfg, db)

    # Deploy directly behind right Princess Tower at (14.5, 4.0)
    res = m.play(0, "knight", 14.5, 4.0)
    assert res.ok
    knight = [u for u in m.units.values() if u.card_id == "knight" and u.side == 0][0]

    # Step for 180 ticks (9.0s)
    for _ in range(180):
        m.step()

    # Knight must have smoothly navigated past the tower (tower top is y = 6.5 + 1.5 = 8.0)
    assert knight.y > 8.0, f"Knight got stuck behind tower at y={knight.y}"
    assert knight.hp > 0


def test_task4_king_tower_sliding(db):
    """Troop deployed directly behind King Tower smoothly slides around without vibrating in place."""
    deck = ("knight", "archers", "skeletons", "fireball", "the_log", "ice_spirit", "musketeer", "cannon")
    cfg = MatchConfig((deck, deck), (11, 11), seed=42, shuffle=False)
    m = Match(cfg, db)

    # Deploy directly behind King Tower at (9.0, 0.5)
    res = m.play(0, "knight", 9.0, 0.5)
    assert res.ok
    knight = [u for u in m.units.values() if u.card_id == "knight" and u.side == 0][0]

    # Step for 180 ticks (9.0s)
    for _ in range(180):
        m.step()

    # Knight must have smoothly navigated past King Tower (y > 5.5)
    assert knight.y > 5.5, f"Knight got stuck behind king tower at y={knight.y}"


def test_task4_crowding_near_river(db):
    """Dense unit crowding near river separates units naturally without pushing ground units into river."""
    deck = ("skeletons", "knight", "archers", "fireball", "the_log", "ice_spirit", "musketeer", "cannon")
    cfg = MatchConfig((deck, deck), (11, 11), seed=42, shuffle=False)
    m = Match(cfg, db)

    # Deploy skeletons near river
    res = m.play(0, "skeletons", 10.0, 14.0)
    assert res.ok
    skels = [u for u in m.units.values() if u.card_id == "skeletons" and u.side == 0]

    # Force them to stack on top of each other right at river edge off-bridge
    for s in skels:
        s.x = 10.0
        s.y = 14.8

    # Step match to let collision resolve
    for _ in range(50):
        m.step()

    # None should have been pushed into river (15.0 <= y < 17.0) off-bridge
    for s in skels:
        if 15.0 <= s.y < 17.0:
            assert abs(s.x - 3.5) <= 1.0 or abs(s.x - 14.5) <= 1.0, f"Unit pushed into river at {s.x}, {s.y}"


def test_task5_deploy_delay_behavior(db):
    """Units stand still and do not attack during 1.0s deploy delay, but take damage and tick status."""
    deck = ("musketeer", "knight", "skeletons", "fireball", "the_log", "ice_spirit", "archers", "cannon")
    cfg = MatchConfig((deck, deck), (11, 11), seed=42, shuffle=False)
    m = Match(cfg, db)

    res = m.play(0, "musketeer", 14.5, 9.5)
    assert res.ok
    musk = [u for u in m.units.values() if u.card_id == "musketeer" and u.side == 0][0]

    assert musk.deploy_ms == 1000

    # Step for 10 ticks (500ms)
    for _ in range(10):
        m.step()

    assert musk.deploy_ms == 500
    assert musk.x == 14.5 and musk.y == 9.5  # perfectly still

    # Musketeer takes damage while deploying
    initial_hp = musk.hp
    m._deal(musk, 100)
    assert musk.hp == initial_hp - 100

    # Step remaining 10 ticks to finish deploy delay
    for _ in range(10):
        m.step()

    assert musk.deploy_ms == 0

    # Next steps: unit begins moving
    for _ in range(10):
        m.step()

    assert musk.y > 9.5


def test_task5_mid_swing_instant_retargeting(db):
    """When a target dies mid-swing, attacker immediately retargets without freezing."""
    deck = ("knight", "skeletons", "musketeer", "fireball", "the_log", "ice_spirit", "archers", "cannon")
    cfg = MatchConfig((deck, deck), (11, 11), seed=42, shuffle=False)
    m = Match(cfg, db)

    res0 = m.play(0, "knight", 14.5, 9.5)
    knight = [u for u in m.units.values() if u.card_id == "knight" and u.side == 0][0]
    knight.deploy_ms = 0

    res1 = m.play(1, "skeletons", 14.5, 20.5)
    skels = [u for u in m.units.values() if u.card_id == "skeletons" and u.side == 1]
    skel = skels[0]
    skel.deploy_ms = 0
    skel.x = 14.5
    skel.y = knight.y + 0.8  # within knight melee range 1.2

    # Step 1 tick: Knight locks onto skeleton and starts attack windup
    m.step()
    assert knight.target_uid == skel.uid
    assert knight.cooldown_ms > 0

    # Kill skeleton mid-swing
    skel.hp = 0

    # Step 1 tick
    m.step()

    # Knight must have immediately retargeted without freezing
    assert knight.target_uid != skel.uid
    new_tgt = m.units.get(knight.target_uid)
    assert new_tgt is not None
    assert new_tgt.hp > 0
    assert new_tgt.tower_role == "princess_right"


def test_task5_melee_and_ranged_attack_loops(db):
    """Melee deals direct damage; ranged fires projectile that damages target upon impact."""
    deck = ("musketeer", "knight", "skeletons", "fireball", "the_log", "ice_spirit", "archers", "cannon")
    cfg = MatchConfig((deck, deck), (11, 11), seed=42, shuffle=False)
    m = Match(cfg, db)

    # Spawn Musketeer (ranged)
    res = m.play(0, "musketeer", 14.5, 9.5)
    musk = [u for u in m.units.values() if u.card_id == "musketeer" and u.side == 0][0]
    musk.deploy_ms = 0

    # Spawn enemy Knight at distance 5.0 (within range 6.0)
    res_e = m.play(1, "knight", 14.5, 20.5)
    enemy_knight = [u for u in m.units.values() if u.card_id == "knight" and u.side == 1][0]
    enemy_knight.deploy_ms = 0
    enemy_knight.x = 14.5
    enemy_knight.y = musk.y + 5.0
    initial_hp = enemy_knight.hp

    # Step for 20 ticks (1.0s) to allow first hit projectile to land
    for _ in range(20):
        m.step()

    assert enemy_knight.hp < initial_hp


def test_task6_river_crossing_air_and_jumpers(db):
    """Ground units require bridges to cross river, while air units and Hog Rider jump directly."""
    deck = ("hog_rider", "knight", "minions", "fireball", "the_log", "poison", "zap", "cannon")
    cfg = MatchConfig((deck, deck), (11, 11), seed=42, shuffle=False)
    m = Match(cfg, db)

    # 1. Deploy Hog Rider near center at (9.5, 14.5)
    m.sides[0].elixir = 10.0
    res_hog = m.play(0, "hog_rider", 9.5, 14.5)
    assert res_hog.ok
    hog = [u for u in m.units.values() if u.card_id == "hog_rider" and u.side == 0][0]
    hog.deploy_ms = 0

    # 2. Deploy Knight near center at (9.5, 14.5)
    m.sides[0].elixir = 10.0
    res_k = m.play(0, "knight", 9.5, 14.5)
    assert res_k.ok
    knight = [u for u in m.units.values() if u.card_id == "knight" and u.side == 0][0]
    knight.deploy_ms = 0

    # 3. Deploy Minions (air) at (9.5, 14.5)
    m.sides[0].elixir = 10.0
    res_m = m.play(0, "minions", 9.5, 14.5)
    assert res_m.ok
    minion = [u for u in m.units.values() if u.card_id == "minions" and u.side == 0][0]
    minion.deploy_ms = 0

    # Step match for 50 ticks (2.5s)
    for _ in range(50):
        m.step()

    # Hog Rider jumped across river directly (already in enemy territory y > 17.0)
    assert hog.y > 17.0, f"Hog should have jumped across river, but y={hog.y}"

    # Minions flew straight across river (layer == 'air')
    assert minion.y > 17.0, f"Minions should have flown across river, but y={minion.y}"

    # Ground Knight CANNOT cross river at x ~ 9.5; must still be on south bank (y < 15.0) routing toward bridge
    assert knight.y < 15.0, f"Ground knight illegally crossed river without bridge: y={knight.y}"
    assert knight.x > 11.0, f"Knight should be heading towards right bridge (x=14.5), but x={knight.x}"


def test_task6_spell_travel_and_radial_knockback(db):
    """Area spells apply damage after travel time and knock back units radially away from spell center."""
    deck = ("fireball", "the_log", "knight", "skeletons", "minions", "poison", "zap", "cannon")
    cfg = MatchConfig((deck, deck), (11, 11), seed=42, shuffle=False)
    m = Match(cfg, db)

    # Spawn enemy Knight in north territory
    res_k = m.play(1, "knight", 9.5, 20.5)
    assert res_k.ok
    enemy_knight = [u for u in m.units.values() if u.card_id == "knight" and u.side == 1][0]
    enemy_knight.deploy_ms = 0
    enemy_knight.speed = 0.0  # isolate knockback displacement
    initial_hp = enemy_knight.hp
    initial_y = enemy_knight.y

    # Cast Fireball centered below enemy Knight at (9.5, 19.5)
    res_fb = m.play(0, "fireball", 9.5, 19.5)
    assert res_fb.ok
    assert len(m.spells) == 1
    assert m.spells[0].impact_in_ms > 0

    # Step while projectile is in flight: enemy takes zero damage
    for _ in range(5):
        m.step()
    assert enemy_knight.hp == initial_hp

    # Step until impact
    while len(m.spells) > 0:
        m.step()

    # After impact: damage applied and knocked radially upward (+y)
    assert enemy_knight.hp < initial_hp
    assert enemy_knight.y > initial_y, f"Expected knockback upward, got y={enemy_knight.y}"


def test_task6_the_log_rolling_knockback_direction(db):
    """The Log rolls forward, damaging ground units and pushing them forward along the roll vector."""
    deck = ("the_log", "knight", "skeletons", "minions", "fireball", "poison", "zap", "cannon")
    cfg = MatchConfig((deck, deck), (11, 11), seed=42, shuffle=False)
    m = Match(cfg, db)

    # Spawn enemy Knight at (9.5, 18.5)
    res_k = m.play(1, "knight", 9.5, 18.5)
    assert res_k.ok
    enemy_knight = [u for u in m.units.values() if u.card_id == "knight" and u.side == 1][0]
    enemy_knight.deploy_ms = 0
    enemy_knight.speed = 0.0
    initial_hp = enemy_knight.hp
    initial_y = enemy_knight.y

    # Cast The Log from (9.5, 14.0) rolling north (direction = +1)
    res_log = m.play(0, "the_log", 9.5, 14.0)
    assert res_log.ok

    # Step match until Log hits and advances
    for _ in range(40):
        m.step()

    assert enemy_knight.hp < initial_hp
    assert enemy_knight.y > initial_y, f"Expected forward knockback along roll direction, got y={enemy_knight.y}"
