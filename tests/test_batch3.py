"""Batch 3 (the five modern meta cards): Bats, Minion Giant, Rune Giant's enchantment, Poison, Little Prince (attack ramp
+ Royal Rescue). Numbers are the wiki's (2026-11-16)."""
import pytest

from backend_engine.simulator import geometry as G
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.mechanics import status as S
from tests.test_batch1 import BIG, new, run, tank
from tests.test_mechanics import play

M3 = ("rune_giant", "knight", "bats", "minion_giant", "poison", "little_prince", "musketeer", "valkyrie")
NEW = ("bats", "minion_giant", "rune_giant", "poison", "little_prince")


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


def mk(db):
    return new(db, a=M3, b=M3)


def dummy(m, card, x, y, side=1):
    """A unit that stays put and does no damage."""
    u = tank(m, side, card, x, y)
    u.damage = 0
    return u


def hits(target, m, ms):
    """(time, damage) of every hp change of `target` over `ms`."""
    out, last = [], target.hp
    for _ in range(ms // G.TICK_MS):
        m.step()
        if target.hp != last:
            out.append((m.t_ms, last - target.hp))
            last = target.hp
    return out


# ------------------------------------------------------------------ data
def test_the_plan_and_modern_cards_are_all_simulated(db):
    assert set(NEW) <= set(db.cards) and "guardienne" in db.tokens and "guardienne" not in db.cards
    assert all(k.startswith(("evo_", "hero_")) for ks in db.unsimulated_tags().values() for k in ks)   # base cards: all simulated


@pytest.mark.parametrize("cid,level,expected", [
    ("bats", 11, {"hitpoints": 81, "damage": 81}),
    ("minion_giant", 11, {"hitpoints": 1817, "damage": 153}),                 # 2026-10-19
    ("rune_giant", 11, {"hitpoints": 2816, "damage": 153, "bonus_damage": 220}),   # 2026-06-01 / 08-04
    ("rune_giant", 16, {"hitpoints": 4535, "damage": 246, "bonus_damage": 354}),
    ("poison", 11, {"damage": 92, "building_damage": 92, "crown_tower_damage": 21}),   # 2026-06-01 184 -> 168 (8 pulses)
    ("poison", 16, {"damage": 148, "building_damage": 148, "crown_tower_damage": 33}),
    ("little_prince", 11, {"hitpoints": 698, "damage": 104, "rescue_damage": 320}),   # 2026-08-26 256 -> 320
    ("little_prince", 16, {"hitpoints": 1124, "damage": 167, "rescue_damage": 515}),
    ("guardienne", 11, {"hitpoints": 1600, "damage": 232}),                   # 2026-08-04 217 -> 232
    ("guardienne", 16, {"hitpoints": 2577, "damage": 373}),
])
def test_values(db, cid, level, expected):
    assert db.stats(cid, level).stats == expected


def test_all_wiki_base_pages_are_simulated_and_match(db):
    from pathlib import Path
    from backend_engine.simulator.sync_meta import load_wiki_dir, verify_wiki
    real = Path(__file__).resolve().parents[1] / "data" / "raw_sources" / "cards"
    if not real.is_dir():
        pytest.skip("saved wiki pages not present")
    rows = verify_wiki(db, load_wiki_dir(real), levels=(11, 12, 13, 14, 15, 16))
    # "newer": the card holds a newer official value (Supercell balance notes) than the wiki table, on purpose
    assert [r for r in rows if r["status"] not in ("ok", "newer")] == []


def test_the_8_meta_decks_that_need_only_these_cards_are_playable(db):
    from backend_engine.meta.recommend import coverage
    assert coverage(db.cards)[0] >= 36                    # Phase 1 target: 36 of 932 replay decks (plan section 10)


# ------------------------------------------------------------------ Bats / Minion Giant
def test_bats_are_five_fast_flyers_that_ground_only_troops_cannot_hit(db):
    m = mk(db)
    bats = play(m, 0, "bats", 9.5, 12.0)
    assert len(bats) == 5 and all(b.layer == "air" and b.max_hp == 81 for b in bats)
    knight = dummy(m, "knight", 9.5, 12.8)
    knight.damage = 202
    run(m, 3000)
    assert [b for b in bats if b.hp < b.max_hp] == []                  # none was hurt by the ground-only knight


def test_minion_giant_flies_and_only_attacks_buildings_from_range(db):
    m = mk(db)
    (mg,) = play(m, 0, "minion_giant", 3.5, 22.0)
    assert mg.layer == "air" and mg.targets == ("buildings",) and mg.range == 4.0
    troop = dummy(m, "knight", 3.5, 23.0)
    tw = m.tower(1, "princess_left")
    mg.speed = 0.0
    mg.x, mg.y = tw.x, tw.y - 4.0 - tw.radius - mg.radius + 0.1        # just inside range of the tower
    d = hits(tw, m, 4000)
    assert d and {x for _, x in d} == {153} and troop.hp == BIG        # tower yes, the troop never
    assert 1500 <= d[1][0] - d[0][0] <= 1900                           # one hit every 1.7 s


# ------------------------------------------------------------------ Rune Giant
def test_rune_giant_enchants_the_two_nearest_allies_for_a_bonus_every_third_hit(db):
    m = mk(db)
    (rg,) = play(m, 0, "rune_giant", 9.5, 10.0)
    near1 = play(m, 0, "knight", 8.5, 11.0)[0]
    near2 = play(m, 0, "knight", 13.0, 9.0)[0]                    # enchanted too, but far from the target below
    far = play(m, 0, "knight", 9.5, 0.2)[0]                    # 9.8 tiles away: out of range
    for k in (near1, near2, far):
        k.hp = k.max_hp = BIG
    run(m, 100)
    assert "enchanted" in near1.mechanics and "enchanted" in near2.mechanics and "enchanted" not in far.mechanics
    foe = dummy(m, "valkyrie", 8.5, 12.0)
    near1.x, near1.y, near1.speed = 8.5, 11.5, 0.0
    d = [x for _, x in hits(foe, m, 6000)]
    assert d[:4] == [202, 202, 422, 202]                                # 202 normal, 202 + 220 on every third attack
    assert db.stats("rune_giant", 11).stats["bonus_damage"] == 220


def test_enchantment_outlives_the_rune_giant_by_half_a_second(db):          # 2026-10-19: 5 s -> 0.5 s
    m = mk(db)
    (rg,) = play(m, 0, "rune_giant", 9.5, 10.0)
    ally = play(m, 0, "knight", 9.5, 11.5)[0]
    ally.hp = ally.max_hp = BIG
    run(m, 100)
    assert "enchanted" in ally.mechanics
    rg.hp = 0
    run(m, 300)
    assert "enchanted" in ally.mechanics
    run(m, 400)
    assert "enchanted" not in ally.mechanics and "enchanted" not in ally.params


# ------------------------------------------------------------------ Poison
def test_poison_eight_pulses_slow_air_ground_and_towers(db):
    m = mk(db)
    ground = dummy(m, "knight", 9.5, 22.0)
    air = play(m, 1, "bats", 10.5, 22.0)[0]
    air.hp = air.max_hp = BIG
    air.damage = 0
    tw = m.tower(1, "princess_left")
    hp_t = tw.hp
    play(m, 0, "poison", 9.5, 22.0)
    run(m, 1100)
    assert S.SLOW15 in ground.status and S.speed_factor(ground) == pytest.approx(0.85)
    run(m, 7000)
    assert BIG - ground.hp == 8 * 92 and BIG - air.hp == 8 * 92
    run(m, 300)
    assert S.SLOW15 not in ground.status
    m2 = mk(db)
    tw2 = m2.tower(1, "princess_left")
    hp2 = tw2.hp
    play(m2, 0, "poison", tw2.x, tw2.y)
    run(m2, 8200)
    assert hp2 - tw2.hp == 8 * 21


def test_slows_do_not_stack_the_strongest_wins(db):
    m = mk(db)
    u = dummy(m, "knight", 9.5, 22.0)
    m.apply_status(u, S.SLOW15, 1000)
    assert S.speed_factor(u) == pytest.approx(0.85) and S.cooldown_dt(u, 100) == 85
    m.apply_status(u, S.SLOW, 1000)
    assert S.speed_factor(u) == pytest.approx(0.65)
    m.apply_status(u, S.SLOW50, 1000)
    assert S.speed_factor(u) == pytest.approx(0.5) and S.cooldown_dt(u, 100) == 50


# ------------------------------------------------------------------ Little Prince
def test_little_prince_attacks_speed_up_on_the_same_target(db):
    m = mk(db)
    play(m, 0, "little_prince", 9.5, 10.0)
    foe = dummy(m, "valkyrie", 9.5, 14.0)
    ts = [t for t, _ in hits(foe, m, 8000)]
    gaps = [b - a for a, b in zip(ts, ts[1:])]
    assert gaps[0] == pytest.approx(1200, abs=50) and gaps[1] == pytest.approx(1200, abs=50)
    assert gaps[2] == pytest.approx(600, abs=50) and gaps[4] == pytest.approx(400, abs=50)
    assert min(gaps) >= 350


def test_royal_rescue_costs_three_and_calls_guardienne_after_a_second(db):
    m = mk(db)
    (lp,) = play(m, 0, "little_prince", 9.5, 6.0)
    foe = dummy(m, "valkyrie", 9.5, 10.6)
    m.sides[0].elixir = 2.9
    assert m.check_ability(0, "little_prince") == "not_enough_elixir"
    m.sides[0].elixir = 5.0
    res = m.use_ability(0, "little_prince")
    assert res.ok and res.event.elixir_cost == 3 and m.sides[0].elixir == pytest.approx(2.0)
    run(m, 900)
    assert not [u for u in m.units.values() if u.card_id == "guardienne"]
    run(m, 200)
    (g,) = [u for u in m.units.values() if u.card_id == "guardienne"]
    assert g.side == 0 and g.max_hp == 1600 and g.y == pytest.approx(10.0, abs=0.7)    # 4 tiles ahead of the prince
    assert foe.max_hp - foe.hp >= 256 and foe.y > 11.0                    # damaged and pushed away
    m.sides[0].elixir = 5.0
    assert m.use_ability(0, "little_prince").reason == "ability_not_ready"


def test_bot_uses_royal_rescue_with_an_enemy_close(db):
    from backend_engine.brain.scripted_bot import ScriptedBot
    from backend_engine.contracts import ActionKind
    m = mk(db)
    play(m, 0, "little_prince", 9.5, 10.0)
    m.sides[0].elixir = 6.0
    bot = ScriptedBot(db, 0, 0.5)
    assert not [i for i in bot.decide(m.observation(0), [], m.t_ms) if i.kind == ActionKind.ABILITY]
    dummy(m, "valkyrie", 9.5, 12.5)
    (i,) = bot.decide(m.observation(0), [], m.t_ms)
    assert i.kind == ActionKind.ABILITY and i.card_id == "little_prince" and i.elixir_cost == 3


def test_ui_button_shows_the_ability_cost(db):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from frontend_app.qt.app import LIVE, MainWindow
    from frontend_app.qt.session import GameSession, SessionConfig
    app = QApplication.instance() or QApplication([])
    for seed in range(60):
        s = GameSession(db, SessionConfig(player_deck=M3, ai_deck=M3, seed=seed))
        if "little_prince" in s.match.sides[0].hand:
            break
    w = MainWindow(s)
    w._set_mode(LIVE)
    w.countdown_ms = 0
    assert s.play("little_prince", 4.0, 12.0).ok
    s.tick(60)
    s.match.sides[0].elixir = 6.0
    w.refresh()
    assert not w.hand.ability_btn.isHidden() and "· 3" in w.hand.ability_btn.text()
    app.processEvents()
