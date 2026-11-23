"""Evolutions and heroes of the batch-4 cards (wiki variant pages 2026-11-23 + 2026 balance notes):
mechanics/variants4.py and the four hero abilities in mechanics/abilities.py."""
import pytest

from backend_engine.brain.scripted_bot import ScriptedBot
from backend_engine.simulator import geometry as G
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig
from backend_engine.simulator.match_runner import MatchRunner
from backend_engine.simulator.mechanics import status as S
from tests.test_batch1 import BIG, run
from tests.test_mechanics import play

E1 = ("royal_giant", "battle_ram", "goblin_barrel", "goblin_cage", "goblin_drill", "inferno_dragon", "lumberjack", "princess")
H1 = ("balloon", "electro_wizard", "magic_archer", "mega_minion", "knight", "valkyrie", "musketeer", "guards")
FOE = ("knight", "valkyrie", "musketeer", "golem", "guards", "bats", "giant", "pekka")


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


def mk(db, a, evo=(), b=FOE):
    m = Match(MatchConfig((a, b), (11, 11), (tuple(evo), ()), shuffle=False, collisions=False), db)
    for t in m.units.values():
        if t.kind == "tower":
            t.active = False
    return m


def evo_play(m, side, card, x, y, **kw):
    st = m.sides[side]
    if card in st.evo_counter:
        st.evo_counter[card] = m.db.evo_cycles(card)
    units = play(m, side, card, x, y, **kw)
    return units


def dummy(m, card, x, y, side=1):
    m._spawn_card_units(side, card, x, y, 999, False, deploy_ms=0, count=1)
    u = max(m.units.values(), key=lambda u: u.uid)
    u.speed, u.damage, u.hp, u.max_hp = 0.0, 0, BIG, BIG
    return u


def units(m, card, side=0):
    return [u for u in sorted(m.units.values(), key=lambda u: u.uid) if u.card_id == card and u.side == side]


# ------------------------------------------------------------------ data
def test_new_variants_are_selectable(db):
    assert set(E1) <= set(db.evo_capable())
    assert {"balloon", "electro_wizard", "magic_archer", "mega_minion"} <= set(db.hero_capable())
    Match(MatchConfig((E1, FOE), evo_cards=(E1, ())), db)
    Match(MatchConfig((H1, FOE), evo_cards=(("balloon#hero", "electro_wizard#hero", "magic_archer#hero",
                                              "mega_minion#hero"), ())), db)


def test_variant_mechanics_replace_and_remove_base_kinds(db):
    kinds = [m["kind"] for m in db.mechanics_for("battle_ram", "evo")]
    assert "break_on_hit" not in kinds and kinds.index("death_spawn") == 2
    assert next(m for m in db.mechanics_for("battle_ram", "evo") if m["kind"] == "death_spawn")["params"]["token"] == "evo_barbarian"
    assert next(m for m in db.mechanics_for("battle_ram")
                if m["kind"] == "death_spawn")["params"]["token"] == "battle_ram_barbarian"
    assert db.stats("goblin_cage", 11, "evo").stats["cage_damage"] == 366          # 2026-08-04 337 -> 366
    assert db.stats("decoy_goblin", 11).stats["damage"] == 66                       # 2026-08-04 89 -> 66
    assert db.evo_cycles("royal_giant") == 1 and db.evo_cycles("princess") == 2


# ------------------------------------------------------------------ evolutions
def test_evo_royal_giant_shots_blast_and_push_troops_around_him(db):
    m = mk(db, E1, evo=("royal_giant",))
    tw = m.tower(1, "princess_left")
    (rg,) = evo_play(m, 0, "royal_giant", tw.x, tw.y - tw.radius - 5.0)
    assert rg.variant == "evo"
    foe = dummy(m, "knight", rg.x + 1.5, rg.y)
    x0 = foe.x
    while foe.hp == BIG and m.t_ms < 5000:              # the blast goes off when he fires
        m.step()
    assert BIG - foe.hp == 81 and foe.x - x0 == pytest.approx(1.0, abs=1e-6)


def test_evo_battle_ram_tramples_bounces_keeps_hitting_and_drops_evolved_barbarians(db):
    m = mk(db, E1, evo=("battle_ram",))
    (ram,) = evo_play(m, 0, "battle_ram", 3.5, 12.0, still=False)
    foe = dummy(m, "knight", 4.0, 19.5)
    tw = m.tower(1, "princess_left")
    while foe.hp == BIG and m.t_ms < 15_000:
        m.step()
    assert BIG - foe.hp == 212                                   # trampled while charging
    for _ in range(400):
        m.step()
        if tw.max_hp - tw.hp >= 573 + 286:
            break
    assert ram.hp > 0 and tw.max_hp - tw.hp >= 573 + 286          # charged hit, bounce, then normal hits
    ram.hp = 0
    m.step()
    barbs = units(m, "evo_barbarian")
    assert len(barbs) == 2 and barbs[0].max_hp == 737
    b = barbs[0]
    b.deploy_ms, b.cooldown_ms = 0, 0
    b.x, b.y = foe.x, foe.y - 1.0
    run(m, 400)
    assert S.RAGE in b.status


def test_evo_goblin_barrel_sends_decoys_to_the_other_lane(db):
    m = mk(db, E1, evo=("goblin_barrel",))
    tw = m.tower(1, "princess_left")
    st = m.sides[0]
    st.evo_counter["goblin_barrel"] = db.evo_cycles("goblin_barrel")
    st.elixir = 10
    if "goblin_barrel" not in st.hand:
        st.queue.remove("goblin_barrel")
        st.queue.insert(0, st.hand[0])
        st.hand[0] = "goblin_barrel"
    assert m.play(0, "goblin_barrel", tw.x, tw.y).ok
    assert len(m.spells) == 2 and {s.x for s in m.spells} == {tw.x, G.W - tw.x}
    run(m, 6000)
    gobs, decoys = units(m, "barrel_goblin"), units(m, "decoy_goblin")
    assert len(gobs) == 3 and len(decoys) == 3
    assert all(g.x < G.W / 2 for g in gobs) and all(d.x > G.W / 2 for d in decoys) and decoys[0].max_hp == 81


def test_evo_goblin_cage_traps_hurts_and_frees(db):
    m = mk(db, E1, evo=("goblin_cage",))
    (cage,) = evo_play(m, 0, "goblin_cage", 9.5, 8.5)
    foe = dummy(m, "knight", 9.5, 11.0)
    run(m, 100)
    assert (foe.x, foe.y) == (cage.x, cage.y) and S.INVISIBLE in foe.status and S.STUN in foe.status
    run(m, 1100)
    assert BIG - foe.hp == 366
    cage.hp = 0
    run(m, 200)
    assert S.INVISIBLE not in foe.status and S.STUN not in foe.status and len(units(m, "goblin_brawler")) == 1


def test_evo_goblin_drill_resurfaces_around_the_tower_and_leaves_goblins(db):
    m = mk(db, E1, evo=("goblin_drill",))
    tw = m.tower(1, "princess_left")
    (drill,) = evo_play(m, 0, "goblin_drill", tw.x, tw.y - tw.radius - 0.5)
    r0 = G.dist(drill.x, drill.y, tw.x, tw.y)
    pos0 = (drill.x, drill.y)
    drill.hp = drill.max_hp * 0.6
    m.step()
    assert (drill.x, drill.y) != pos0 and G.dist(drill.x, drill.y, tw.x, tw.y) == pytest.approx(r0)
    assert len(units(m, "drill_goblin")) == 1 and S.INVISIBLE in drill.status
    hp = drill.hp
    m._deal(drill, 100)
    assert drill.hp == hp                                        # underground: untouchable
    run(m, 1100)
    n = len(units(m, "drill_goblin"))                            # its regular spawns keep coming meanwhile
    drill.hp = drill.max_hp * 0.3
    m.step()
    assert len(units(m, "drill_goblin")) == n + 1


def test_evo_inferno_dragon_keeps_its_stage_after_a_kill_and_reaches_a_fourth(db):
    m = mk(db, E1, evo=("inferno_dragon",))
    (dr,) = evo_play(m, 0, "inferno_dragon", 9.5, 10.0)
    a = dummy(m, "knight", 9.5, 12.5)
    run(m, 5500)
    assert dr.mstate["ramp_ms"] >= 4000                           # stage 3 on the knight
    a.hp = 1
    run(m, 400)
    assert a.uid not in m.units
    b = dummy(m, "valkyrie", 9.5, 12.5)
    while b.hp == BIG and m.t_ms < 20_000:                        # lock-on (first hit) on the new target
        m.step()
    assert BIG - b.hp == 422                                      # stage 3 kept on the next target
    run(m, 16000)
    hp = b.hp
    run(m, 400)
    assert hp - b.hp == 844                                       # 4th stage after 20 s of beam
    m.apply_status(dr, S.STUN, 300)
    assert dr.mstate["ramp_ms"] == 0                               # a stun resets everything


def test_evo_lumberjack_leaves_an_untouchable_ghost_inside_his_rage(db):
    m = mk(db, E1, evo=("lumberjack",))
    (lj,) = evo_play(m, 0, "lumberjack", 9.5, 10.0)
    foe = dummy(m, "knight", 9.5, 11.5)
    lj.hp = 0
    m.step()
    (gh,) = units(m, "lumberjack_ghost")
    m.step()
    assert S.INVISIBLE in gh.status
    m._deal(gh, 500)
    assert gh.hp == gh.max_hp
    run(m, 1500)
    assert BIG - foe.hp >= 179 + 256                              # rage drop + ghost hits
    run(m, 4500)
    assert not units(m, "lumberjack_ghost")


def test_evo_princess_slows_on_first_and_every_second_shot_and_leaves_a_frost_zone(db):
    m = mk(db, E1, evo=("princess",))
    (pr,) = evo_play(m, 0, "princess", 9.5, 7.0)
    a = dummy(m, "knight", 9.5, 15.0)
    side_foe = dummy(m, "valkyrie", 10.5, 15.5)
    run(m, 1500)
    assert S.SLOW30 in a.status and S.SLOW30 in side_foe.status and S.speed_factor(a) == pytest.approx(0.7)
    assert pr.mstate["shots"] == 1
    pr.hp = 0
    m.step()
    assert any(e.kind == "frost_zone" for e in m.effects)


# ------------------------------------------------------------------ heroes
def hero_match(db, card):
    return mk(db, H1, evo=(card + "#hero",))


def test_hero_balloon_coffin_cadets_lands_a_skeletrooper(db):
    m = hero_match(db, "balloon")
    (bl,) = play(m, 0, "balloon", 9.5, 10.0)
    assert bl.variant == "hero"
    foe = dummy(m, "knight", 9.5, 14.0)
    m.sides[0].elixir = 10
    assert m.use_ability(0, "balloon").ok and m.sides[0].elixir == pytest.approx(8.0)
    run(m, 1100)
    assert BIG - foe.hp == 263 and len(units(m, "skeletrooper")) == 1
    assert m.check_ability(0, "balloon") == "ability_not_ready"            # single use


def test_hero_electro_wizard_surging_strikes_hits_two_targets_five_times(db):
    m = hero_match(db, "electro_wizard")
    (ew,) = play(m, 0, "electro_wizard", 9.5, 8.0)
    ew.damage = int(ew.damage)
    a, b = dummy(m, "knight", 9.0, 12.0), dummy(m, "valkyrie", 10.5, 12.0)
    hp_a, hp_b = a.hp, b.hp
    ew.cooldown_ms = 10_000                                                 # isolate the ability from his normal zaps
    m.sides[0].elixir = 10
    assert m.use_ability(0, "electro_wizard").ok
    m.step()
    assert S.STUN in a.status and S.STUN in b.status
    run(m, 3300)
    per_hit = 115 * 1.27
    assert hp_a - a.hp == pytest.approx(5 * per_hit) and hp_b - b.hp == pytest.approx(5 * per_hit)


def test_hero_magic_archer_triple_threat_decoy_dart_and_three_arrows(db):
    m = hero_match(db, "magic_archer")
    (ma,) = play(m, 0, "magic_archer", 9.5, 8.0)
    foe = dummy(m, "knight", 9.5, 12.0)
    x0, y0 = ma.x, ma.y
    m.sides[0].elixir = 10
    assert m.use_ability(0, "magic_archer").ok
    (decoy,) = units(m, "ma_decoy")
    assert (decoy.x, decoy.y) == (x0, y0) and decoy.max_hp == 271
    assert G.dist(ma.x, ma.y, x0, y0) == pytest.approx(3.5) and ma.y < y0
    ma.cooldown_ms = 0
    while foe.hp == BIG and m.t_ms < 5000:
        m.step()
    assert BIG - foe.hp == 48                                     # one of the three 48-damage arrows hits it
    run(m, 7000)
    assert not units(m, "ma_decoy")


def test_hero_mega_minion_wounding_warp_hits_the_weakest_and_flies_home(db):
    m = hero_match(db, "mega_minion")
    (mm,) = play(m, 0, "mega_minion", 9.5, 6.0)
    strong = dummy(m, "knight", 4.0, 20.0)
    weak = dummy(m, "valkyrie", 14.0, 24.0)
    weak.hp = weak.max_hp = 1000
    home = (mm.x, mm.y)
    m.sides[0].elixir = 10
    assert m.use_ability(0, "mega_minion").ok
    assert 1000 - weak.hp == 399 and G.dist(mm.x, mm.y, weak.x, weak.y) < 2.0 and strong.hp == BIG
    run(m, 3600)
    assert (mm.x, mm.y) == home


# ------------------------------------------------------------------ bots with every new variant
def test_scripted_bots_play_full_matches_with_the_new_evolutions_and_heroes(db):
    decks = [(E1, E1[:2]), (E1, E1[2:4]), (E1, E1[4:6]), (E1, E1[6:]),
             (H1, ("balloon#hero", "electro_wizard#hero")), (H1, ("magic_archer#hero", "mega_minion#hero"))]
    for i, (deck, slots) in enumerate(decks):
        m = Match(MatchConfig((deck, FOE), (11, 11), (slots, ()), seed=20 + i), db)
        MatchRunner(m, {0: ScriptedBot(db, 0, 0.9), 1: ScriptedBot(db, 1, 0.7)}).run()
        assert m.result is not None
