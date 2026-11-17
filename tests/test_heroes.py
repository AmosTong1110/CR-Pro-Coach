"""Evolution / hero data (wiki 2026-11-16) and the five simulated hero abilities: Knight's taunt, Mini P.E.K.K.A.'s
boost, Ice Golem's snowstorm, Musketeer's turret, Valkyrie's whirl. Slots are written `card#hero` (CardRef form)."""
import pytest

from backend_engine.simulator import geometry as G
from backend_engine.simulator.card_db import CardDB, split_slot
from backend_engine.simulator.engine import Match, MatchConfig
from backend_engine.simulator.mechanics import status as S
from tests.test_batch1 import BIG, run
from tests.test_mechanics import play

H = ("knight", "mini_pekka", "ice_golem", "musketeer", "valkyrie", "skeletons", "cannon", "fireball")
E = ("hog_rider", "knight", "tesla", "valkyrie", "skeletons", "cannon", "fireball", "musketeer")
HEROES = ("knight#hero", "mini_pekka#hero", "ice_golem#hero", "musketeer#hero", "valkyrie#hero")


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


def mk(db, hero=HEROES, level=11):
    m = Match(MatchConfig((H, E), (level, level), evo_cards=(hero, ()), shuffle=False, collisions=False), db)
    for t in m.units.values():
        if t.kind == "tower":
            t.active = False
    return m


def dummy(m, card, x, y, side=1, hp=BIG):
    u = play(m, side, card, x, y)[0]
    u.hp = u.max_hp = hp
    u.damage = 0
    u.decay_per_tick = 0.0
    return u


def hero_unit(m, card, x, y):
    (u,) = play(m, 0, card, x, y)
    assert u.variant == "hero"
    return u


def use(m, card, elixir=9.0):
    m.sides[0].elixir = elixir
    res = m.use_ability(0, card)
    assert res.ok, res.reason
    return res


# ------------------------------------------------------------------ data + verification
def test_slots_parse_and_validate(db):
    assert split_slot("knight") == ("knight", "evo") and split_slot("knight#hero") == ("knight", "hero")
    assert set(db.hero_capable()) == {"knight", "mini_pekka", "ice_golem", "musketeer", "valkyrie"}
    assert set(db.evo_capable()) == {"knight", "firecracker", "skeletons", "zap", "valkyrie", "tesla", "mortar"}   # the rest are data only
    for bad in ("archers", "archers#hero", "barbarian_barrel#hero", "skeletons#hero", "fireball#hero"):
        deck = (*H[:7], "archers") if bad.startswith("archers") else H if bad.split("#")[0] in H else (*H[:7], bad.split("#")[0])
        with pytest.raises(ValueError, match="cannot be an evolution / hero slot"):
            Match(MatchConfig((deck, E), evo_cards=((bad,), ())), db)


def test_every_evo_and_hero_page_is_in_the_data_and_matches(db):
    from pathlib import Path
    from backend_engine.simulator.sync_meta import load_wiki_dir, verify_variants
    real = Path(__file__).resolve().parents[1] / "data" / "raw_sources" / "cards"
    if not real.is_dir():
        pytest.skip("saved wiki pages not present")
    rows = verify_variants(db, load_wiki_dir(real))
    assert len(rows) == 21 and all(r["status"].startswith("ok") for r in rows)
    assert all(r["cycles_db"] == r["cycles_wiki"] == 2 for r in rows if r["variant"] == "evo")


@pytest.mark.parametrize("cid,level,expected", [
    ("knight", 11, {"hitpoints": 1766, "damage": 202, "shield_hitpoints": 512}),
    ("knight", 16, {"hitpoints": 2844, "damage": 325, "shield_hitpoints": 825}),
    ("mini_pekka", 11, {"hitpoints": 1433, "damage": 755}),
    ("ice_golem", 11, {"hitpoints": 1315, "damage": 84, "death_damage": 84, "blizzard_damage": 69}),
    ("valkyrie", 16, {"hitpoints": 3071, "damage": 428, "ability_damage": 156}),
    ("musketeer", 16, {"hitpoints": 1161, "damage": 349, "turret_spawn_damage": 329}),
])
def test_hero_stats(db, cid, level, expected):
    st = db.stats(cid, level, "hero").stats
    assert {k: st[k] for k in expected} == expected
    assert db.stats(cid, level).stats.get("shield_hitpoints") is None      # the base card is untouched


def test_evolution_data_without_simulation(db):
    assert db.stats("bats", 11, "evo").stats["hitpoints"] == 122 and db.stats("bats", 11).stats["hitpoints"] == 81
    assert db.evo_cycles("bats") == 2 and not db.variant_simulated("bats", "evo")
    assert db.unsimulated_tags()["bats"] == ["evo_heal_pulse"] and "knight" not in db.unsimulated_tags()


# ------------------------------------------------------------------ hero slot
def test_a_hero_card_is_played_as_the_hero_and_uses_its_stats(db):
    m = mk(db)
    mp = hero_unit(m, "mini_pekka", 9.5, 8.5)
    assert mp.max_hp == 1433
    m.sides[0].hand[0] = "mini_pekka"
    ev = m.events[-1]
    assert ev.variant == "hero" and not ev.is_evolved
    plain = Match(MatchConfig((H, E), shuffle=False), db)                  # same card without the slot: base stats
    (b,) = play(plain, 0, "mini_pekka", 9.5, 8.5)
    assert b.variant == "base" and b.max_hp == 1390


def test_a_non_hero_copy_has_no_ability(db):
    m = mk(db, hero=())
    play(m, 0, "knight", 9.5, 8.5)
    m.sides[0].elixir = 9
    assert m.check_ability(0, "knight") == "ability_not_ready"


# ------------------------------------------------------------------ Hero Knight: Triumphant Taunt
def test_triumphant_taunt_shield_and_forced_targets(db):
    m = mk(db)
    k = hero_unit(m, "knight", 9.5, 10.0)
    k.hp = k.max_hp = BIG
    trooper = play(m, 1, "valkyrie", 9.5, 15.0)[0]                      # would normally ignore him: another target near
    trooper.speed, trooper.damage = 0.0, 0
    bait = dummy(m, "skeletons", 6.0, 15.5)                             # a nearer ally of the trooper's... not hit
    tesla = play(m, 1, "tesla", 13.0, 14.0)[0]
    tesla.damage = 0
    far = play(m, 1, "musketeer", 9.5, 30.0)[0]                          # 20 tiles away: not taunted
    res = use(m, "knight")
    assert res.event.elixir_cost == 2
    run(m, 900)
    assert "taunt" not in trooper.mstate and k.mstate.get("shield", 0) == 0
    run(m, 200)
    assert k.mstate["shield"] == 512 and "shield" in k.mechanics
    assert trooper.mstate["taunt"] == k.uid and tesla.mstate["taunt"] == k.uid and "taunt" not in far.mstate
    run(m, 100)
    assert trooper.target_uid == k.uid and tesla.target_uid == k.uid       # locked on, even the building
    m._deal(k, 300)
    assert k.mstate["shield"] == 212 and k.hp == BIG                      # the shield takes it
    run(m, 5000)
    assert k.mstate["shield"] == 0                                        # the shield is gone after 5 s
    assert trooper.mstate["taunt"] == k.uid                               # ... but they still hunt him
    k.hp = 0
    run(m, 100)
    assert "taunt" not in trooper.mstate


# ------------------------------------------------------------------ Hero Mini P.E.K.K.A.: Breakfast Boost
def test_breakfast_boost_levels_up_and_heals(db):
    m = mk(db)
    mp = hero_unit(m, "mini_pekka", 9.5, 10.0)
    mp.hp = mp.max_hp * 0.4
    old = mp.hp
    res = use(m, "mini_pekka")
    assert res.event.elixir_cost == 1
    run(m, 1100)
    st = db.stats("mini_pekka", 12, "hero").stats
    assert mp.max_hp == st["hitpoints"] and mp.damage == st["damage"]
    assert mp.hp == pytest.approx(min(mp.max_hp, old + (st["hitpoints"] - 1433) + 0.30 * st["hitpoints"]))
    assert m.check_ability(0, "mini_pekka") == "ability_not_ready"


def test_breakfast_boost_at_level_16_uses_the_hero_level_17_table(db):
    m = mk(db, level=16)
    mp = hero_unit(m, "mini_pekka", 9.5, 10.0)
    use(m, "mini_pekka")
    run(m, 1100)
    assert mp.max_hp == db.variant("mini_pekka", "hero")["level_overrides"]["17"]["hitpoints"] > 2308


# ------------------------------------------------------------------ Hero Ice Golem: Snowstorm
def test_snowstorm_three_pulses_damage_and_slow(db):
    m = mk(db)
    g = hero_unit(m, "ice_golem", 9.5, 12.0)
    near = dummy(m, "valkyrie", 9.5, 15.0)
    air = dummy(m, "valkyrie", 12.0, 12.0)
    far = dummy(m, "valkyrie", 9.5, 22.0)
    use(m, "ice_golem")
    run(m, 1600)                                                          # 1 s delay, then the first pulse after 0.5 s
    assert BIG - near.hp == 69 and S.SLOW30 in near.status and S.speed_factor(near) == pytest.approx(0.7)
    run(m, 1000)
    assert BIG - near.hp == 3 * 69 and BIG - air.hp == 3 * 69 and far.hp == BIG
    run(m, 2500)
    assert S.SLOW30 not in near.status                                    # slow lasts 2 s after the last pulse


# ------------------------------------------------------------------ Hero Musketeer: Trusty Turret
def test_trusty_turret_appears_ahead_hits_and_expires(db):
    m = mk(db)
    mu = hero_unit(m, "musketeer", 9.5, 8.0)
    mu.damage = 0                                                         # only the turret hurts the dummy
    near = dummy(m, "knight", 9.5, 11.5)
    use(m, "musketeer")
    assert m.sides[0].elixir == pytest.approx(6.0)
    run(m, 950)
    assert not [u for u in m.units.values() if u.card_id == "trusty_turret"]
    run(m, 150)
    (tu,) = [u for u in m.units.values() if u.card_id == "trusty_turret"]
    assert tu.kind == "building" and tu.y == pytest.approx(11.0, abs=0.7) and tu.side == 0
    assert BIG - near.hp == 204                                           # spawn damage
    assert tu.max_hp == round(153.6 * 10) and tu.range == 4.0
    run(m, 3000)
    assert BIG - near.hp > 204 and (BIG - near.hp - 204) % 140 == 0         # then 140 per hit
    run(m, 9000)
    assert not [u for u in m.units.values() if u.card_id == "trusty_turret"]   # 10 s lifetime


# ------------------------------------------------------------------ Hero Valkyrie: Whirl Power
def test_whirl_power_spins_for_three_and_a_half_seconds_and_cuts_damage(db):
    m = mk(db)
    v = hero_unit(m, "valkyrie", 9.5, 10.0)
    v.hp = v.max_hp = BIG
    foe = dummy(m, "knight", 9.5, 11.5)
    foe.damage = 200
    far = dummy(m, "knight", 9.5, 20.0)
    use(m, "valkyrie")
    assert v.mstate["ab"]["dr"] == 15
    foe.speed = 0.0
    run(m, 3600)
    spin = 14                                                              # one hit every 0.25 s for 3.5 s
    lost = BIG - foe.hp
    assert lost >= spin * 97 and far.hp == BIG
    assert "dr" not in v.mstate["ab"] and v.mstate["ab"]["phase"] is None  # the spin is over: no reduction any more


def test_whirl_power_damage_reduction_applies_while_spinning(db):
    m = mk(db)
    v = hero_unit(m, "valkyrie", 9.5, 10.0)
    v.hp = v.max_hp = BIG
    use(m, "valkyrie")
    before = v.hp
    m._deal(v, 1000)
    assert before - v.hp == pytest.approx(850)


# ------------------------------------------------------------------ plumbing: bot, session, record, UI
def test_bot_uses_a_hero_ability_at_the_right_moment(db):
    from backend_engine.brain.scripted_bot import ScriptedBot
    from backend_engine.contracts import ActionKind
    m = mk(db)
    hero_unit(m, "valkyrie", 9.5, 10.0)
    m.sides[0].elixir = 6.0
    bot = ScriptedBot(db, 0, 0.5)
    assert not [i for i in bot.decide(m.observation(0), [], m.t_ms) if i.kind == ActionKind.ABILITY]
    dummy(m, "knight", 9.5, 11.2)
    dummy(m, "skeletons", 10.5, 11.0)
    (i,) = bot.decide(m.observation(0), [], m.t_ms)
    assert i.kind == ActionKind.ABILITY and i.card_id == "valkyrie" and i.elixir_cost == 3


def test_session_saves_and_rebuilds_a_hero_match(db):
    from backend_engine.storage.match_store import MatchRecord
    from frontend_app.qt.session import GameSession, SessionConfig
    for seed in range(80):
        s = GameSession(db, SessionConfig(player_deck=H, ai_deck=E, player_evo=("knight#hero",), seed=seed))
        if "knight" in s.match.sides[0].hand:
            break
    assert s.play("knight", 9.5, 10.0).ok and s.match.events[-1].variant == "hero"
    s.tick(60)
    assert s.ability_cards() == ["knight"] and s.ability_cost("knight") == 2
    assert s.use_ability("knight").ok
    s.tick(100)
    meta, record = s.to_record()
    assert meta.player_evo == ("knight#hero",) and meta.n_plays == 1
    rebuilt = GameSession.rebuild(MatchRecord.from_dict(record.to_dict()), db)
    assert rebuilt.match.state_hash() == s.match.state_hash()


def test_deck_chip_and_dialog_show_heroes(db):
    import os
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from frontend_app.qt.app import CustomDeckDialog
    from frontend_app.qt.widgets import DeckView
    app = QApplication.instance() or QApplication([])
    view = DeckView()
    view.set_deck(H, ("knight#hero",), lambda c: db.cards[c]["name"], lambda c: db.cards[c]["elixir"])
    assert "Hero" in view.chips[0].text() and "Hero" not in view.chips[1].text()
    dlg = CustomDeckDialog(db, H, ("valkyrie#hero",), "Test")
    items = [dlg.evo_box.itemText(i) for i in range(dlg.evo_box.count())]
    assert "Hero Valkyrie" in items and "Evo Knight" in items and "Evo Archers" not in items
    deck, evo = dlg.result_deck()
    assert evo == ("valkyrie#hero",)
    app.processEvents()
