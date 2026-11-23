"""Batch 4 (Phase 2): the 25 cards that complete the 8 season meta decks. Numbers are the Clash Royale Wiki's (fetched
2026-11-23) with Supercell's 2026 balance notes on top; mechanics in backend_engine/simulator/mechanics/batch4.py."""
import json
from pathlib import Path

import pytest

from backend_engine.brain.scripted_bot import ScriptedBot
from backend_engine.simulator import geometry as G
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig
from backend_engine.simulator.entities import Projectile
from backend_engine.simulator.match_runner import MatchRunner
from backend_engine.simulator.mechanics import status as S
from tests.test_batch1 import BIG, new, run, tank
from tests.test_mechanics import play

ROOT = Path(__file__).resolve().parents[1]
NEW = ("inferno_dragon", "goblin_cage", "goblin_drill", "magic_archer", "goblin_barrel", "princess", "goblin_gang",
       "golem", "night_witch", "mega_minion", "lumberjack", "royal_giant", "monk", "fisherman", "phoenix",
       "lava_hound", "balloon", "guards", "skeleton_dragons", "battle_ram", "electro_wizard", "bandit", "arrows",
       "giant_skeleton", "heal_spirit")
TOKENS = ("goblin_brawler", "drill_goblin", "barrel_goblin", "gang_spear_goblin", "golemite", "lava_pup",
          "battle_ram_barbarian", "phoenix_egg", "phoenix_reborn")
D1 = ("balloon", "giant_skeleton", "battle_ram", "bandit", "fisherman", "phoenix", "electro_wizard", "monk")
D2 = ("heal_spirit", "lumberjack", "magic_archer", "goblin_drill", "goblin_barrel", "arrows", "goblin_gang", "golem")
D3 = ("lava_hound", "night_witch", "guards", "inferno_dragon", "knight", "valkyrie", "musketeer", "royal_giant")


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


def dummy(m, card, x, y, side=1):
    u = tank(m, side, card, x, y)
    u.damage = 0
    return u


def units(m, card, side=0):
    return [u for u in sorted(m.units.values(), key=lambda u: u.uid) if u.card_id == card and u.side == side]


def hits(target, m, ms):
    out, last = [], target.hp
    for _ in range(ms // G.TICK_MS):
        m.step()
        if target.hp != last:
            out.append((m.t_ms, last - target.hp))
            last = target.hp
    return out


# ------------------------------------------------------------------ data
def test_all_25_cards_and_their_tokens_are_playable(db):
    assert set(NEW) <= set(db.cards) and set(TOKENS) <= set(db.tokens)
    assert len(db.cards) == 72


def test_the_8_season_meta_decks_are_complete_and_in_the_presets(db):
    decks = json.loads((ROOT / "configs" / "meta_decks_top200.json").read_text())["decks"]
    alias = {"healing_spirit": "heal_spirit"}
    for d in decks:
        assert all(alias.get(c, c) in db.cards for c in d["cards"]), d["name"]
    presets = {tuple(sorted(p["cards"])) for p in json.loads((ROOT / "configs" / "preset_decks.json").read_text())["decks"]}
    for d in decks:
        assert tuple(sorted(alias.get(c, c) for c in d["cards"])) in presets, d["name"]


@pytest.mark.parametrize("cid,expected", [
    ("skeleton_dragons", {"hitpoints": 560, "damage": 151}),                       # 2026-05-04 damage 161 -> 151
    ("arrows", {"damage": 122, "crown_tower_damage": 25}),                         # 2026-06-01 tower 93 -> 75 (3 volleys)
    ("magic_archer", {"hitpoints": 529, "damage": 135}),                           # 2026-08-04 143 -> 135
    ("lava_hound", {"hitpoints": 3581, "damage": 71}),                             # 2026-10-06 53 -> 71
    ("goblin_drill", {"hitpoints": 1313, "spawn_damage": 84, "spawn_tower_damage": 20}),   # 2026-10-06 tower 0 -> 20
    ("goblin_brawler", {"hitpoints": 1121, "damage": 337}),                        # 2026-09-08 1080 -> 1121
    ("phoenix_egg", {"hitpoints": 317}),                                           # 2026-03-02 egg +32%
    ("giant_skeleton", {"hitpoints": 3361, "damage": 276, "death_damage": 688}),   # 2026-05-04 rework
    ("electro_wizard", {"hitpoints": 714, "damage": 115, "zap_damage": 192}),
])
def test_level_11_values_include_the_2026_balance_notes(db, cid, expected):
    assert db.stats(cid, 11).stats == expected


def test_timings_from_the_2026_balance_notes(db):
    assert db.cards["royal_giant"]["attack"]["hit_speed_s"] == 1.7
    assert db.cards["lava_hound"]["attack"]["hit_speed_s"] == 1.5
    assert db.cards["battle_ram"]["attack"]["sight_tiles"] == 6.5 and db.cards["golem"]["attack"]["sight_tiles"] == 7.5
    assert db.tokens["barrel_goblin"]["attack"]["first_hit_s"] == 0.3
    assert db.tokens["battle_ram_barbarian"]["attack"]["hit_speed_s"] == 1.4
    assert db.cards["giant_skeleton"]["collision_radius_tiles"] == 0.75
    monk = next(m for m in db.cards["monk"]["mechanics"] if m["kind"] == "champion_ability")["params"]
    assert monk["delay_ms"] == 700 and monk["uses"] == 1


# ------------------------------------------------------------------ death bombs
def test_balloon_bomb_explodes_three_seconds_after_death(db):
    m = new(db, a=D1, b=D3)
    (b,) = play(m, 0, "balloon", 9.5, 15.0)
    foe = dummy(m, "knight", 10.5, 15.5)
    b.hp = 0
    m.step()
    run(m, 2900)
    assert foe.hp == BIG
    run(m, 150)
    assert BIG - foe.hp == 240


def test_giant_skeleton_bomb_hits_towers_once_and_knocks_troops_back(db):
    m = new(db, a=D1, b=D3)
    tw = m.tower(1, "princess_left")
    (gs,) = play(m, 0, "giant_skeleton", tw.x, tw.y - tw.radius - 0.5)
    foe = dummy(m, "knight", tw.x + 2.0, gs.y)
    x0, hp0 = foe.x, tw.hp
    gs.hp = 0
    run(m, 3200)
    assert hp0 - tw.hp == 688                       # no double damage on Crown Towers since 2026-05-04
    assert BIG - foe.hp == 688 and foe.x - x0 == pytest.approx(1.8, abs=1e-6)


# ------------------------------------------------------------------ Battle Ram
def test_battle_ram_charges_breaks_on_the_tower_and_releases_two_barbarians(db):
    m = new(db, a=D1, b=D3)
    (ram,) = play(m, 0, "battle_ram", 3.5, 12.0, still=False)
    tw = m.tower(1, "princess_left")
    hp0 = tw.hp
    while ram.uid in m.units and m.t_ms < 20_000:
        m.step()
    assert ram.uid not in m.units and hp0 - tw.hp == 573                 # charged hit = Charge Damage
    barbs = units(m, "battle_ram_barbarian")
    assert len(barbs) == 2 and all(b.max_hp == 716 and b.deploy_ms > 0 for b in barbs)


# ------------------------------------------------------------------ Bandit
def test_bandit_dashes_from_mid_range_for_dash_damage_and_is_immune_meanwhile(db):
    m = new(db, a=D1, b=D3)
    (bd,) = play(m, 0, "bandit", 9.5, 10.0, still=False)
    foe = dummy(m, "knight", 9.5, 15.5)
    for _ in range(60):
        m.step()
        if bd.mstate.get("dash", {}).get("phase") == "dash":
            break
    assert bd.mstate["dash"]["phase"] == "dash"
    hp = bd.hp
    m._deal(bd, 500)
    assert bd.hp == hp                                                  # immune while dashing
    d = hits(foe, m, 1500)
    assert d and d[0][1] == 389


# ------------------------------------------------------------------ Fisherman
def test_fisherman_hooks_a_troop_to_himself_without_slowing_it(db):
    m = new(db, a=D1, b=D3)
    (fm,) = play(m, 0, "fisherman", 9.5, 10.0)
    foe = dummy(m, "knight", 9.5, 15.5)
    run(m, 2500)
    assert m._edge_dist(fm, foe) <= fm.range + 1e-6 and S.SLOW not in foe.status


def test_fisherman_reels_himself_to_a_tower(db):
    m = new(db, a=D1, b=D3)
    tw = m.tower(1, "princess_left")
    (fm,) = play(m, 0, "fisherman", tw.x, tw.y - tw.radius - 5.5)
    run(m, 3000)
    assert m._edge_dist(fm, tw) <= fm.range + 1e-6


# ------------------------------------------------------------------ Phoenix
def test_phoenix_explodes_lays_an_egg_and_is_reborn_after_4_3_seconds(db):
    m = new(db, a=D1, b=D3)
    (ph,) = play(m, 0, "phoenix", 9.5, 15.0)
    foe = dummy(m, "knight", 9.5, 16.0)
    ph.hp = 0
    m.step()
    assert BIG - foe.hp == 163 and foe.y > 16.0
    (egg,) = units(m, "phoenix_egg")
    assert egg.max_hp == 317 and egg.layer == "ground"
    run(m, 4250)
    assert not units(m, "phoenix_reborn")
    run(m, 150)
    (re,) = units(m, "phoenix_reborn")
    assert re.max_hp == 1052 and re.damage == 217 and not units(m, "phoenix_egg") and "rebirth" not in re.mechanics


def test_a_destroyed_egg_means_no_rebirth(db):
    m = new(db, a=D1, b=D3)
    (ph,) = play(m, 0, "phoenix", 9.5, 15.0)
    ph.hp = 0
    m.step()
    units(m, "phoenix_egg")[0].hp = 0
    run(m, 5000)
    assert not units(m, "phoenix_reborn")


# ------------------------------------------------------------------ Electro Wizard
def test_electro_wizard_spawn_zap_stuns_and_spares_crown_towers(db):
    m = new(db, a=D1, b=D3)
    foe = dummy(m, "knight", 9.5, 13.5)
    play(m, 0, "electro_wizard", 9.5, 12.0)
    assert BIG - foe.hp == 192 and S.STUN in foe.status
    tw = m.tower(1, "princess_left")
    hp0 = tw.hp
    m._spawn_card_units(0, "electro_wizard", tw.x, tw.y - tw.radius - 1.0, 0, False)
    assert tw.hp == hp0 and S.STUN in tw.status


def test_electro_wizard_splits_on_two_targets_or_doubles_on_one(db):
    m = new(db, a=D1, b=D3)
    play(m, 0, "electro_wizard", 9.5, 9.0)
    a, b = dummy(m, "knight", 9.0, 13.0), dummy(m, "valkyrie", 10.5, 13.0)
    while a.hp == BIG and m.t_ms < 3000:
        m.step()
    assert BIG - a.hp == 115 and BIG - b.hp == 115 and S.STUN in a.status and S.STUN in b.status
    m2 = new(db, a=D1, b=D3)
    play(m2, 0, "electro_wizard", 9.5, 9.0)
    c = dummy(m2, "knight", 9.5, 13.0)
    run(m2, 1500)
    assert BIG - c.hp == 230


# ------------------------------------------------------------------ Monk
def test_monk_third_hit_is_a_combo_that_knocks_back_even_a_golem(db):
    m = new(db, a=D1, b=D2)
    play(m, 0, "monk", 9.5, 8.0)
    golem = dummy(m, "golem", 9.5, 9.5)                                 # far from the river (no push into it)
    y0 = golem.y
    d = [x for _, x in hits(golem, m, 3500)]
    assert d[:3] == [140, 140, 422] and golem.y - y0 == pytest.approx(1.8, abs=1e-6)


def test_monk_ability_cuts_damage_by_65_percent_and_reflects_projectiles(db):
    m = new(db, a=D1, b=D3)
    (mk,) = play(m, 0, "monk", 9.5, 12.0)
    mk.hp = mk.max_hp = BIG
    shooter = dummy(m, "musketeer", 9.5, 18.0)
    m.sides[0].elixir = 10
    assert m.use_ability(0, "monk").ok
    run(m, 750)
    m._deal(mk, 1000)
    assert BIG - mk.hp == pytest.approx(350)
    shot = Projectile(m._uid(), 1, shooter.uid, "musketeer", mk.uid, mk.x, mk.y, 10.0, 100.0)
    m._deal(mk, 100.0, shot)
    assert BIG - shooter.hp == 100


# ------------------------------------------------------------------ Heal Spirit / Lumberjack
def test_heal_spirit_damages_and_heals_allies_in_four_pulses(db):
    m = new(db, a=D2, b=D3)
    ally = tank(m, 0, "golem", 9.5, 13.0)
    ally.damage, ally.hp = 0, BIG - 1000
    foe = dummy(m, "knight", 9.5, 14.5)
    play(m, 0, "heal_spirit", 9.5, 12.5, still=False)
    run(m, 2000)
    assert BIG - foe.hp == 110 and BIG - ally.hp == 1000 - 400 and not units(m, "heal_spirit")


def test_lumberjack_drops_a_rage_that_hurts_foes_and_speeds_allies(db):
    m = new(db, a=D2, b=D3)
    (lj,) = play(m, 0, "lumberjack", 9.5, 12.0)
    ally = tank(m, 0, "golem", 11.0, 12.0)
    foe = dummy(m, "knight", 9.5, 13.5)
    lj.hp = 0
    m.step()
    assert BIG - foe.hp == 179
    m.step()
    assert S.RAGE in ally.status and S.RAGE not in foe.status and S.speed_factor(ally) == pytest.approx(1.3)
    run(m, 5600)
    assert S.RAGE not in ally.status


# ------------------------------------------------------------------ Magic Archer / Arrows / Goblin Barrel
def test_magic_archer_arrow_pierces_everything_on_its_line(db):
    m = new(db, a=D2, b=D3)
    play(m, 0, "magic_archer", 9.5, 7.0)
    front, back = dummy(m, "knight", 9.5, 12.0), dummy(m, "valkyrie", 9.5, 15.0)
    side = dummy(m, "musketeer", 13.0, 13.0)
    run(m, 1500)
    assert BIG - front.hp == 135 and BIG - back.hp == 135 and side.hp == BIG


def test_arrows_land_three_volleys(db):
    m = new(db, a=D2, b=D3)
    tw = m.tower(1, "princess_left")
    foe = dummy(m, "knight", tw.x, tw.y - tw.radius - 1.0)
    hp0 = tw.hp
    m.sides[0].elixir = 10
    st = m.sides[0]
    if "arrows" not in st.hand:
        st.queue.remove("arrows")
        st.queue.insert(0, st.hand[0])
        st.hand[0] = "arrows"
    assert m.play(0, "arrows", tw.x, tw.y - 1.0).ok
    d = hits(foe, m, 4000)
    assert [x for _, x in d] == [122, 122, 122] and d[1][0] - d[0][0] == 200
    assert hp0 - tw.hp == 75


def test_goblin_barrel_drops_three_goblins_and_does_not_wake_the_king(db):
    m = new(db, a=D2, b=D3)
    kt = m.tower(1, "king")
    st = m.sides[0]
    st.elixir = 10
    if "goblin_barrel" not in st.hand:
        st.queue.remove("goblin_barrel")
        st.queue.insert(0, st.hand[0])
        st.hand[0] = "goblin_barrel"
    assert m.play(0, "goblin_barrel", kt.x, kt.y - kt.radius - 1.0).ok
    while not units(m, "barrel_goblin") and m.t_ms < 8000:
        m.step()
    gobs = units(m, "barrel_goblin")
    assert len(gobs) == 3 and all(g.max_hp == 202 and g.damage == 125 for g in gobs)
    assert not kt.active and kt.hp == kt.max_hp and m.t_ms > 1000      # it flew from the own king tower


# ------------------------------------------------------------------ Goblin Drill / Cage / Gang
def test_goblin_drill_surfaces_anywhere_hits_spawns_and_splits(db):
    m = new(db, a=D2, b=D3)
    tw = m.tower(1, "princess_left")
    x, y = tw.x, tw.y - tw.radius - 0.6
    assert m.check_play(0, "goblin_drill", x, y) is None                # on the enemy side
    foe = dummy(m, "knight", x + 1.0, y)
    x0, hp0 = foe.x, tw.hp
    (drill,) = play(m, 0, "goblin_drill", x, y, ready=False)
    run(m, 1100)
    assert BIG - foe.hp == 84 and foe.x - x0 == pytest.approx(1.0, abs=1e-6) and hp0 - tw.hp == 20
    run(m, 900)
    assert len(units(m, "drill_goblin")) == 1
    run(m, 3000)
    assert len(units(m, "drill_goblin")) == 2
    drill.hp = 0
    m.step()
    assert len(units(m, "drill_goblin")) == 4


def test_goblin_cage_releases_the_brawler(db):
    m = new(db, a=("goblin_cage",) + D3[1:], b=D3)
    (cage,) = play(m, 0, "goblin_cage", 9.5, 8.0)
    cage.hp = 0
    m.step()
    (br,) = units(m, "goblin_brawler")
    assert br.max_hp == 1121 and br.deploy_ms == 500


def test_goblin_gang_is_three_goblins_with_three_spear_goblins_behind(db):
    m = new(db, a=D2, b=D3)
    play(m, 0, "goblin_gang", 9.5, 10.0, ready=False, still=False)
    gob, spear = units(m, "goblin_gang"), units(m, "gang_spear_goblin")
    assert len(gob) == 3 and len(spear) == 3
    assert max(s.y for s in spear) < min(g.y for g in gob) and spear[0].deploy_ms == gob[0].deploy_ms + 100
    assert spear[0].targets == ("ground", "air") and gob[0].damage == 125


# ------------------------------------------------------------------ Golem / Lava Hound / Night Witch / Guards / Inferno
def test_golem_death_damage_knockback_and_golemites(db):
    m = new(db, a=D2, b=D3)
    (g,) = play(m, 0, "golem", 9.5, 8.0)
    foe = dummy(m, "knight", 9.5, 9.5)
    y0 = foe.y
    g.hp = 0
    m.step()
    assert BIG - foe.hp == 225 and foe.y - y0 == pytest.approx(1.8, abs=1e-6)
    mites = units(m, "golemite")
    assert len(mites) == 2 and all(u.max_hp == 1039 and u.targets == ("buildings",) for u in mites)


def test_lava_hound_bursts_into_six_pups(db):
    m = new(db, a=D3, b=D2)
    (lh,) = play(m, 0, "lava_hound", 9.5, 12.0)
    assert lh.layer == "air" and lh.targets == ("buildings",)
    lh.hp = 0
    m.step()
    pups = units(m, "lava_pup")
    assert len(pups) == 6 and all(p.layer == "air" and p.max_hp == 217 for p in pups)


def test_night_witch_summons_two_bats_every_five_seconds_and_one_on_death(db):
    m = new(db, a=D3, b=D2)
    (nw,) = play(m, 0, "night_witch", 9.5, 8.0)
    run(m, 1200)
    assert len(units(m, "bats")) == 2
    run(m, 5000)
    assert len(units(m, "bats")) == 4
    nw.hp = 0
    m.step()
    assert len(units(m, "bats")) == 5


def test_guards_shields_soak_damage_first(db):
    m = new(db, a=D3, b=D2)
    gs = play(m, 0, "guards", 9.5, 8.0)
    assert len(gs) == 3
    m._deal(gs[0], 300)
    assert gs[0].hp == 81 - 44


def test_inferno_dragon_damage_ramps_in_three_stages(db):
    m = new(db, a=D3, b=D2)
    play(m, 0, "inferno_dragon", 9.5, 10.0)
    foe = dummy(m, "golem", 9.5, 13.0)
    d = [x for _, x in hits(foe, m, 6500)]
    assert d[0] == 35 and 120 in d and d[-1] == 422


# ------------------------------------------------------------------ the 8 meta decks in bot-vs-bot play
def test_scripted_bots_play_full_matches_with_every_meta_deck(db):
    presets = [p for p in json.loads((ROOT / "configs" / "preset_decks.json").read_text())["decks"]
               if p.get("tip", "").startswith("Season meta") or p["name"] == "Hog 2.6 Cycle"]
    assert len(presets) == 8
    for i, a in enumerate(presets):
        b = presets[(i + 3) % len(presets)]
        m = Match(MatchConfig((tuple(a["cards"]), tuple(b["cards"])), (11, 11),
                              evo_cards=(tuple(a["evo"]), tuple(b["evo"])), seed=i), db)
        MatchRunner(m, {0: ScriptedBot(db, 0, 0.8), 1: ScriptedBot(db, 1, 0.8)}).run()
        assert m.result is not None, a["name"]
        played = {r[2] for r in m.action_log}
        assert played & set(NEW) or a["name"] == "Hog 2.6 Cycle", a["name"]
