"""Batch-1 mechanics and cards: status effects (stun / freeze / slow), kamikaze hits, chain, death burst,
self-centred splash, and the data of the 10 cards added by tools/add_cards.py."""
import math

import pytest

from backend_engine.meta.recommend import recommend
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig
from backend_engine.simulator.golden import HOG_26, SPIRITS
from backend_engine.simulator.match_runner import DEFAULT_DECK
from backend_engine.simulator.mechanics import status as S
from tests.test_mechanics import play

BIG = 100_000
UTIL = ("rocket", "dart_goblin", "skeleton_army", "zap", "valkyrie", "ice_golem", "fire_spirit", "mini_pekka")


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


def new(db, a=SPIRITS, b=DEFAULT_DECK, level=11):
    m = Match(MatchConfig((a, b), (level, level), shuffle=False, collisions=False), db)
    for t in m.units.values():
        if t.kind == "tower":
            t.active = False                  # towers stay silent: only the units under test act
    return m


def tank(m, side, card, x, y):
    """A near-unkillable unit that stays where it is."""
    (u,) = play(m, side, card, x, y)
    u.hp = u.max_hp = BIG
    return u


def run(m, ms):
    m.run_until(m.t_ms + ms)


# ------------------------------------------------------------------ Zap: area spell + stun
def test_zap_stuns_damages_and_resets_attack(db):
    m = new(db)
    knight = tank(m, 1, "knight", 9.5, 10.0)
    victim = tank(m, 0, "mini_pekka", 9.5, 11.0)             # inside the knight's range
    victim.damage = 0                                          # so the knight's hp only changes through the zap
    run(m, 1750)                                               # first hit landed at 700 ms; the next is due at 1900
    hp0 = victim.hp
    assert hp0 < victim.max_hp
    play(m, 0, "zap", 9.5, 10.5)
    m.step()
    assert S.STUN in knight.status
    assert BIG - knight.hp == db.stats("zap", 11).stats["damage"]
    assert knight.cooldown_ms >= knight.first_hit_ms          # the wind-up restarted
    run(m, 600)                                                # past the 1900 ms hit AND past the 500 ms stun
    assert S.STUN not in knight.status and victim.hp == hp0   # the stunned knight skipped that hit
    run(m, 1200)
    assert victim.hp < hp0                                    # and resumes afterwards


def test_zap_does_less_to_towers(db):
    m = new(db)
    tw = m.tower(1, "princess_left")
    hp0 = tw.hp
    play(m, 0, "zap", tw.x, tw.y)
    run(m, 100)
    st = db.stats("zap", 11).stats
    assert hp0 - tw.hp == st["crown_tower_damage"] < st["damage"]
    assert S.STUN in tw.status


# ------------------------------------------------------------------ Freeze / stun / slow in the engine
def test_frozen_unit_neither_moves_nor_attacks(db):
    m = new(db)
    (k,) = play(m, 1, "knight", 9.5, 22.5, still=False)       # a walking knight
    m.apply_status(k, S.FREEZE, 1000)
    y0 = k.y
    run(m, 950)
    assert k.y == y0 and S.FREEZE in k.status
    run(m, 1000)
    assert k.y < y0 and not k.status                           # side 1 walks down the board once thawed


def test_slow_reduces_speed_and_attack_rate(db):
    def walked(slow):
        m = new(db)
        (k,) = play(m, 1, "knight", 9.5, 24.5, still=False)
        x0, y0 = k.x, k.y
        if slow:
            m.apply_status(k, S.SLOW, 5000)
        run(m, 1000)
        return math.hypot(k.x - x0, k.y - y0)                  # path length (it walks diagonally to the bridge)
    assert walked(True) == pytest.approx(walked(False) * S.SLOW_FACTOR, rel=0.02)
    m = new(db)
    (k,) = play(m, 1, "knight", 9.5, 24.0)
    k.cooldown_ms = 1000
    m.apply_status(k, S.SLOW, 5000)
    m.step()
    assert k.cooldown_ms == 1000 - 50 * S.SLOW_PCT // 100      # attack timer runs 35% slower


def test_status_is_in_hash_and_survives_snapshot(db):
    m = new(db)
    k = tank(m, 1, "knight", 9.5, 20.0)
    h0 = m.state_hash()
    m.apply_status(k, S.FREEZE, 2000)
    assert m.state_hash() != h0
    run(m, 500)
    snap = m.snapshot()
    run(m, 3000)
    h1 = m.state_hash()
    m.restore(snap)
    assert S.FREEZE in m.units[k.uid].status
    run(m, 3000)
    assert m.state_hash() == h1


# ------------------------------------------------------------------ Ice / Fire / Electro Spirit: kamikaze
def test_ice_spirit_hits_freezes_and_dies(db):
    m = new(db)
    k = tank(m, 1, "knight", 9.5, 19.5)
    (sp,) = play(m, 0, "ice_spirit", 9.5, 14.5, still=False)
    for _ in range(200):
        m.step()
        if sp.uid not in m.units:
            break
    assert sp.uid not in m.units                               # kamikaze
    assert BIG - k.hp == db.stats("ice_spirit", 11).stats["damage"]
    assert 0 < k.status[S.FREEZE] <= 1200


def test_fire_spirit_area_hit_only_inside_radius(db):
    m = new(db)
    skels = play(m, 1, "skeletons", 9.5, 19.5)
    far = tank(m, 1, "knight", 9.5, 25.0)
    (sp,) = play(m, 0, "fire_spirit", 9.5, 14.5, still=False)
    for _ in range(200):
        m.step()
        if sp.uid not in m.units:
            break
    assert sp.uid not in m.units
    assert all(s.uid not in m.units for s in skels)            # 207 damage kills all three at once
    assert far.hp == far.max_hp


def test_electro_spirit_chains_to_nine_units_and_stuns(db):
    m = new(db)
    line = [tank(m, 1, "knight", 2.5 + i, 19.5) for i in range(13)]
    (sp,) = play(m, 0, "electro_spirit", 4.5, 14.5, still=False)
    for _ in range(200):
        m.step()
        if sp.uid not in m.units:
            break
    assert sp.uid not in m.units
    hit = [u for u in line if u.hp < BIG]
    assert len(hit) == 9 == len([u for u in line if S.STUN in u.status])
    assert {BIG - u.hp for u in hit} == {db.stats("electro_spirit", 11).stats["damage"]}
    assert {u.uid for u in hit} == {u.uid for u in line if S.STUN in u.status}
    assert all(0 < u.status[S.STUN] <= 500 for u in hit)


def test_chain_picks_nearest_first_and_skips_out_of_range(db):
    m = new(db)
    near = tank(m, 1, "knight", 4.5, 19.5)
    nxt = tank(m, 1, "knight", 6.0, 19.5)                      # 1.5 from `near`
    lone = tank(m, 1, "knight", 14.5, 19.5)                    # > 4 tiles from the others: out of the chain
    (sp,) = play(m, 0, "electro_spirit", 4.5, 14.5, still=False)
    for _ in range(200):
        m.step()
        if sp.uid not in m.units:
            break
    assert near.hp < BIG and nxt.hp < BIG and lone.hp == BIG


# ------------------------------------------------------------------ Ice Golem: death burst + slow
def test_ice_golem_death_burst_scales_with_level_and_slows(db):
    for level in (11, 16):
        m = new(db, level=level)
        (g,) = play(m, 0, "ice_golem", 9.5, 14.5)
        assert g.params["death_damage"]["damage"] == db.stats("ice_golem", level).stats["death_damage"]
        near = tank(m, 1, "knight", 9.5, 15.5)
        far = tank(m, 1, "knight", 9.5, 19.5)
        g.hp = 0
        m.step()
        assert BIG - near.hp == g.params["death_damage"]["damage"]
        assert S.SLOW in near.status and far.hp == BIG and not far.status


def test_ice_golem_only_targets_buildings(db):
    assert db.cards["ice_golem"]["attack"]["targets"] == ["buildings"]


# ------------------------------------------------------------------ Valkyrie: splash around herself
def test_valkyrie_splash_is_centred_on_her(db):
    m = new(db)
    valk = tank(m, 0, "valkyrie", 9.5, 13.0)
    target = tank(m, 1, "knight", 9.5, 14.3)
    behind = tank(m, 1, "knight", 9.5, 11.2)                   # 1.8 from her, 3.1 from the target
    beyond = tank(m, 1, "knight", 9.5, 16.5)                   # 2.2 from the target, 3.5 from her
    valk.cooldown_ms = 0
    m.step()
    assert target.hp < BIG and behind.hp < BIG and beyond.hp == BIG
    assert BIG - behind.hp == db.stats("valkyrie", 11).stats["damage"]


# ------------------------------------------------------------------ Skeleton Army, Dart Goblin, Rocket
def test_skeleton_army_spawns_fifteen(db):
    m = new(db)
    units = play(m, 0, "skeleton_army", 9.5, 10.0)
    assert len(units) == 15 and len({(round(u.x, 3), round(u.y, 3)) for u in units}) == 15


def test_dart_goblin_shoots_from_range(db):
    m = new(db, a=UTIL)
    dg = tank(m, 0, "dart_goblin", 9.5, 10.0)
    k = tank(m, 1, "knight", 9.5, 15.0)                        # 5 tiles: inside range 6.5, outside the knight's
    dg.cooldown_ms = 0
    run(m, 1500)
    assert k.hp < BIG and dg.hp == dg.max_hp


def test_rocket_damage_radius_knockback_and_tower_share(db):
    m = new(db, a=UTIL)
    k = tank(m, 1, "knight", 9.5, 20.0)
    y0 = k.y
    play(m, 0, "rocket", 9.5, 20.0)
    run(m, 3500)
    st = db.stats("rocket", 11).stats
    assert BIG - k.hp == st["damage"] == 1484
    assert k.y > y0 + 1.0                                      # pushed 1.8 tiles away from the centre / caster side
    tw = m.tower(1, "princess_left")
    hp0 = tw.hp
    play(m, 0, "rocket", tw.x, tw.y)
    run(m, 5000)                                               # ~24 tiles from the king tower at 5.8 tiles/s
    assert hp0 - tw.hp == st["crown_tower_damage"] == 371


# ------------------------------------------------------------------ data of the 10 added cards
EXPECTED_L11 = {            # exact values from the RoyaleAPI 2023-10 tables (damage_per_level at L11)
    "ice_spirit": {"hitpoints": 230, "damage": 110},
    "fire_spirit": {"hitpoints": 230, "damage": 207},
    "electro_spirit": {"hitpoints": 230, "damage": 99},
    "zap": {"damage": 192, "crown_tower_damage": 58},
    "rocket": {"damage": 1484, "crown_tower_damage": 371},
    "ice_golem": {"hitpoints": 1197, "damage": 84, "death_damage": 84},
}


@pytest.mark.parametrize("cid", sorted(EXPECTED_L11))
def test_new_card_values_at_level_11(db, cid):
    assert db.stats(cid, 11).stats == EXPECTED_L11[cid]


def test_new_cards_have_identity_and_are_marked_unverified(db):
    for cid in ("zap", "ice_spirit", "fire_spirit", "electro_spirit", "ice_golem", "mini_pekka", "valkyrie",
                "skeleton_army", "dart_goblin", "rocket"):
        c = db.cards[cid]
        assert c["game_id"] and c["internal_name"] and not c["verified"] and c["data_as_of"] == "2023-10-18"
        assert db.by_alias(c["aliases"]["royaleapi"], "royaleapi") == cid
    assert db.cards["ice_spirit"]["game_id"] == 26000030
    assert "suicide" in [m["kind"] for m in db.mechanics_for("ice_spirit")]
    assert "projectile" not in [m["kind"] for m in db.mechanics_for("ice_spirit")]   # the hit is the jump


def test_hog_26_is_now_fully_playable(db):
    recs = recommend(set(db.cards), limit=5, playable_only=True)
    assert recs and recs[0].archetype == "Hog 2.6" and set(recs[0].cards) == set(HOG_26)
    Match(MatchConfig((HOG_26, HOG_26)), db)                   # and the engine accepts it
