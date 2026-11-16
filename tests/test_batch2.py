"""Batch 2 (the last 11 cards of the plan's 30): charge, shield, burrow, damage ramp, recoil, hide underground, form
change, and the Earthquake / Lightning / Vines / Royal Delivery spell kinds. Numbers are the wiki's (2026-11-16)."""
import pytest

from backend_engine.meta.recommend import coverage
from backend_engine.simulator import geometry as G
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.golden import B2_A, B2_B
from backend_engine.simulator.mechanics import status as S
from tests.test_batch1 import BIG, new, run, tank
from tests.test_mechanics import play

NEW = ("dark_prince", "golden_knight", "mighty_miner", "miner", "firecracker", "tesla", "earthquake", "lightning",
       "vines", "royal_delivery", "goblin_demolisher")


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


def mk(db):
    return new(db, a=B2_A, b=B2_B)         # side 0 fields the batch-2 units, side 1 the spells / targets


def mk_rev(db):
    return new(db, a=B2_B, b=B2_A)


def hp_deltas(u, steps, m, ms):
    """hp lost by `u` in each step-change over `ms` of play."""
    out, last = [], u.hp
    for _ in range(ms // G.TICK_MS):
        m.step()
        if u.hp != last:
            out.append(last - u.hp)
            last = u.hp
    return out


# ------------------------------------------------------------------ data
def test_all_30_plan_cards_are_simulated(db):
    plan = ("skeletons barbarian_barrel fireball the_log electro_spirit royal_ghost ice_spirit knight cannon mini_pekka "
            "tornado ice_golem musketeer hog_rider mighty_miner firecracker earthquake lightning tesla golden_knight "
            "vines dart_goblin valkyrie skeleton_army rocket miner royal_delivery goblin_demolisher fire_spirit "
            "dark_prince").split()
    assert [c for c in plan if c not in db.cards] == []
    assert set(NEW) <= set(db.cards) and "royal_recruit" in db.tokens and "royal_recruit" not in db.cards


@pytest.mark.parametrize("cid,level,expected", [
    ("dark_prince", 11, {"hitpoints": 1200, "damage": 266, "shield_hitpoints": 256, "charge_damage": 532}),
    ("dark_prince", 16, {"hitpoints": 1933, "damage": 428, "shield_hitpoints": 412, "charge_damage": 857}),
    ("miner", 11, {"hitpoints": 1210, "damage": 194, "crown_tower_damage": 48}),
    ("miner", 16, {"hitpoints": 1949, "damage": 312, "crown_tower_damage": 77}),
    ("mighty_miner", 11, {"hitpoints": 2250, "damage": 40, "damage_stage2": 204, "damage_stage3": 409,
                           "escape_damage": 332}),
    ("lightning", 16, {"damage": 1702, "crown_tower_damage": 461}),
    ("earthquake", 11, {"damage": 84, "crown_tower_damage": 53, "building_damage": 287}),
    ("firecracker", 11, {"hitpoints": 304, "damage": 320}),
    ("goblin_demolisher", 11, {"hitpoints": 1300, "damage": 186, "death_damage": 404, "kamikaze_hp_per_s": 65}),
    ("royal_recruit", 11, {"hitpoints": 547, "damage": 133, "shield_hitpoints": 240}),
])
def test_values(db, cid, level, expected):
    assert db.stats(cid, level).stats == expected


def test_costs_and_champions(db):
    assert {c: db.cards[c]["elixir"] for c in ("lightning", "tesla", "miner", "vines")} == \
        {"lightning": 6, "tesla": 4, "miner": 3, "vines": 3}
    assert db.cards["golden_knight"]["rarity"] == "champion"
    assert db.unsimulated_tags() == {}                             # champion abilities are simulated (test_abilities)


def test_decks_unlocked(db):
    n, total = coverage(db.cards)
    assert n >= 33                         # the 30-card batch makes 33 of the 932 replay decks playable


# ------------------------------------------------------------------ Dark Prince: charge + shield
def test_dark_prince_charges_for_double_damage_then_hits_normally(db):
    m = mk(db)
    dp = play(m, 0, "dark_prince", 3.5, 8.0, still=False)[0]
    target = tank(m, 1, "knight", 3.5, 22.0)
    top = 0.0
    deltas = []
    last = target.hp
    for _ in range(400):
        m.step()
        top = max(top, dp.speed)
        if target.hp != last:
            deltas.append(last - target.hp)
            last = target.hp
        if len(deltas) == 2:
            break
    assert deltas == [532, 266]                               # charge hit, then a normal area hit
    assert top == pytest.approx(2.0)                          # charged speed
    assert dp.speed == pytest.approx(1.0)                     # back to normal after the hit


def test_charge_needs_a_straight_run(db):
    m = mk(db)
    dp = play(m, 0, "dark_prince", 3.5, 8.0, still=False)[0]
    run(m, 500)                                               # < 3 tiles walked
    assert dp.speed == pytest.approx(1.0) and not dp.mstate["charged"]


def test_shield_absorbs_then_hitpoints(db):
    m = mk(db)
    (dp,) = play(m, 0, "dark_prince", 3.5, 8.0)
    m._deal(dp, 100)
    assert dp.hp == dp.max_hp and dp.mstate["shield"] == 156
    m._deal(dp, 300)
    assert dp.mstate["shield"] == 0 and dp.hp == dp.max_hp - 144


# ------------------------------------------------------------------ Miner: burrow + tower damage
def test_miner_can_dig_in_anywhere_but_not_the_river(db):
    m = mk(db)
    m.sides[0].hand[0], m.sides[0].elixir = "miner", 10
    assert m.check_play(0, "miner", 6.5, 24.0) is None          # deep in the enemy half
    assert m.check_play(0, "miner", 9.5, 16.0) == "river"
    m.sides[0].hand[1] = "tesla"
    assert m.check_play(0, "tesla", 6.5, 24.0) == "enemy_half"  # other cards keep the normal rule


def test_miner_is_untargetable_while_digging_and_hits_towers_for_less(db):
    m = mk(db)
    (mi,) = play(m, 0, "miner", 3.5, 21.0, ready=False, still=True)
    assert S.INVISIBLE in mi.status and not S.targetable(mi)
    run(m, 1100)
    assert S.targetable(mi) and mi.deploy_ms <= 0
    foe = tank(m, 1, "knight", 3.5, 21.5)
    tw = m.tower(1, "princess_left")
    mi.x, mi.y = tw.x, tw.y - 2.0                               # next to the tower, off the knight
    foe.x, foe.y = 14.0, 30.0
    d = hp_deltas(tw, 0, m, 3000)
    assert d and set(d) == {48}                                 # crown-tower damage
    mi.x, mi.y, foe.x, foe.y = 9.5, 21.0, 9.5, 21.8
    mi.target_uid = None
    d2 = hp_deltas(foe, 0, m, 3000)
    assert d2 and set(d2) == {194}                              # full damage on a troop


# ------------------------------------------------------------------ Mighty Miner: damage ramp
def test_mighty_miner_damage_ramps_every_two_seconds(db):
    m = mk(db)
    play(m, 0, "mighty_miner", 9.5, 20.0)
    foe = tank(m, 1, "knight", 9.5, 21.0)
    d = hp_deltas(foe, 0, m, 7000)
    assert d[0] == 40 and 204 in d and d[-1] == 409
    assert d == sorted(d)                                       # never drops while on the same target
    assert [d.count(x) for x in (40, 204)] == [5, 5]            # 2 s per stage at 0.4 s per hit


# ------------------------------------------------------------------ Firecracker: recoil
def test_firecracker_recoils_and_lands_all_five_sparks(db):
    m = mk(db)
    (fc,) = play(m, 0, "firecracker", 9.5, 10.0)
    foe = tank(m, 1, "knight", 9.5, 14.0)
    y0 = fc.y
    run(m, 2000)
    assert y0 - fc.y == pytest.approx(1.0, abs=0.01)           # one shot, one tile back
    assert BIG - foe.hp == 320


# ------------------------------------------------------------------ Tesla: hide underground
def test_tesla_hides_until_a_troop_is_in_range(db):
    m = mk(db)
    (te,) = play(m, 0, "tesla", 9.5, 8.5)
    te.decay_per_tick = 0.0
    m.step()
    assert S.INVISIBLE in te.status and not S.targetable(te)
    hp = te.hp
    play(m, 1, "knight", 9.5, 22.0)                                # keep a cannon-fodder knight far away
    m.step()
    m._deal(te, 500)                                               # underground: no damage
    assert te.hp == hp
    foe = tank(m, 1, "valkyrie", 9.5, 12.0)                        # in range (5.5)
    run(m, 2000)
    assert S.INVISIBLE not in te.status and BIG - foe.hp >= 220    # surfaced and shot
    foe.x, foe.y = 9.5, 30.0
    run(m, 200)
    assert S.INVISIBLE in te.status                                # hides again


# ------------------------------------------------------------------ Goblin Demolisher: form change
def test_demolisher_becomes_a_kamikaze_and_explodes_on_the_tower(db):
    m = mk(db)
    (gd,) = play(m, 0, "goblin_demolisher", 3.5, 20.0, still=False)
    gd.hp = gd.max_hp * 0.4
    bystander = tank(m, 1, "knight", 3.5, 24.0)                    # inside the 2.5 radius of the blast
    tw = m.tower(1, "princess_left")
    run(m, 100)
    assert gd.mstate["changed"] and gd.speed == pytest.approx(2.0) and gd.targets == ("buildings",)
    assert "death_damage" in gd.mechanics and "decay" in gd.mechanics
    hp_t = tw.hp
    run(m, 6000)
    assert gd.uid not in m.units                                   # blew up
    assert hp_t - tw.hp == 404 and BIG - bystander.hp == 404


def test_demolisher_burns_out_and_still_bursts(db):
    m = mk(db)
    (gd,) = play(m, 0, "goblin_demolisher", 14.5, 12.0, still=False)
    gd.hp = 100.0                                                  # below half; 65 hp / s: gone in ~1.5 s
    foe = tank(m, 1, "knight", 14.5, 14.0)
    foe.damage = 0                                                 # so only the burst can hurt it
    run(m, 3000)
    assert gd.uid not in m.units and BIG - foe.hp == 404


# ------------------------------------------------------------------ Earthquake
def test_earthquake_three_pulses_troops_buildings_towers_and_slow(db):
    m = mk_rev(db)                                                 # side 0 = the spells
    foe = tank(m, 1, "firecracker", 9.5, 22.0)
    cannon_like = tank(m, 1, "tesla", 11.0, 22.0)
    cannon_like.decay_per_tick = 0.0
    tank(m, 0, "knight", 11.5, 18.5)                          # in the Tesla's range: it stays surfaced
    air = tank(m, 1, "golden_knight", 8.0, 22.0)
    play(m, 0, "earthquake", 9.5, 22.0)
    run(m, 1000)
    assert S.SLOW50 in foe.status and S.speed_factor(foe) == 0.5
    run(m, 2100)
    assert BIG - foe.hp == 3 * 84 and BIG - cannon_like.hp == 3 * 287 and BIG - air.hp == 3 * 84
    run(m, 200)
    assert S.SLOW50 not in foe.status


def test_underground_tesla_is_immune_to_earthquake(db):
    m = mk_rev(db)
    te = tank(m, 1, "tesla", 11.0, 22.0)
    te.decay_per_tick = 0.0
    play(m, 0, "earthquake", 11.0, 22.0)
    run(m, 3200)
    assert te.hp == BIG


def test_earthquake_hits_towers_for_tower_damage(db):
    m = mk_rev(db)
    tw = m.tower(1, "princess_left")
    hp = tw.hp
    play(m, 0, "earthquake", tw.x, tw.y)
    run(m, 3200)
    assert hp - tw.hp == 3 * 53


# ------------------------------------------------------------------ Lightning
def test_lightning_strikes_the_three_highest_hp_and_stuns_them(db):
    m = mk_rev(db)
    foes = [tank(m, 1, c, 9.5 + dx, 22.0) for c, dx in (("firecracker", -2.0), ("golden_knight", 0.0),
                                                        ("mighty_miner", 2.0), ("tesla", 0.5))]
    for f, hp in zip(foes, (5000, 4000, 3000, 2000)):
        f.hp = f.max_hp = hp
        f.decay_per_tick = 0.0
    play(m, 0, "lightning", 9.5, 22.0)
    run(m, 100)
    hit = [f for f in foes if f.max_hp - f.hp == 1057]
    assert [f.max_hp for f in hit] == [5000, 4000, 3000]
    assert foes[3].hp == foes[3].max_hp                            # fourth is spared
    assert all(S.STUN in f.status for f in hit)


def test_lightning_hits_a_tower_for_tower_damage(db):
    m = mk_rev(db)
    tw = m.tower(1, "princess_left")
    hp = tw.hp
    play(m, 0, "lightning", tw.x, tw.y)
    run(m, 100)
    assert hp - tw.hp == 286


# ------------------------------------------------------------------ Vines
def test_vines_roots_the_three_biggest_for_two_seconds_with_two_pulses(db):
    m = mk_rev(db)
    foes = [tank(m, 1, c, 9.5 + dx, 22.0) for c, dx in (("firecracker", -1.0), ("golden_knight", 0.0),
                                                        ("mighty_miner", 1.0), ("tesla", 0.2))]
    for f, hp in zip(foes, (5000, 4000, 3000, 2000)):
        f.hp = f.max_hp = hp
        f.decay_per_tick = 0.0
        f.speed = 1.0
    play(m, 0, "vines", 9.5, 22.0)
    run(m, 850)
    assert not any(S.ROOT in f.status for f in foes)               # still growing (0.9 s)
    run(m, 100)
    assert [S.ROOT in f.status for f in foes] == [True, True, True, False]
    run(m, 1000)
    assert [f.max_hp - f.hp for f in foes] == [153, 153, 153, 0]
    run(m, 1000)
    assert [f.max_hp - f.hp for f in foes] == [306, 306, 306, 0]
    run(m, 200)
    assert not any(S.ROOT in f.status for f in foes)


# ------------------------------------------------------------------ Royal Delivery
def test_royal_delivery_lands_after_three_seconds_and_a_shielded_recruit_steps_out(db):
    m = mk(db)                                                     # side 0 has the card
    foe = tank(m, 1, "knight", 9.5, 22.0)
    play(m, 0, "royal_delivery", 9.5, 22.0)
    run(m, 2900)
    assert foe.hp == BIG and not [u for u in m.units.values() if u.card_id == "royal_recruit"]
    run(m, 200)
    assert BIG - foe.hp == 437
    (r,) = [u for u in m.units.values() if u.card_id == "royal_recruit"]
    assert r.side == 0 and r.max_hp == 547 and r.mstate["shield"] == 240 and r.deploy_ms > 0


# ------------------------------------------------------------------ determinism
def test_batch2_effects_are_deterministic_across_snapshot(db):
    m = mk_rev(db)
    tank(m, 1, "dark_prince", 9.5, 22.0)
    play(m, 0, "vines", 9.5, 22.0)
    play(m, 0, "earthquake", 9.5, 22.0)
    run(m, 400)
    h0, snap = m.state_hash(), m.snapshot()
    run(m, 3000)
    h1 = m.state_hash()
    m.restore(snap)
    assert m.state_hash() == h0
    run(m, 3000)
    assert m.state_hash() == h1
