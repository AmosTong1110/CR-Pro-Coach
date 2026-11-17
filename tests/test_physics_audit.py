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
