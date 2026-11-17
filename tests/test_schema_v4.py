"""Card schema v4, CardRef / event variants, mechanics registry and status effects."""
import copy
import json

import pytest

from backend_engine.contracts import CardRef, PublicDeployEvent, TilePos
from backend_engine.simulator.card_db import DEFAULT_PATH, CardDataError, CardDB, normalise_card
from backend_engine.simulator.engine import Match, MatchConfig
from backend_engine.simulator.match_runner import DEFAULT_DECK
from backend_engine.simulator.mechanics import REGISTRY, SPELLS, Mechanic, hooks_for, register
from backend_engine.simulator.mechanics import status as S


@pytest.fixture(scope="module")
def raw():
    return json.loads(DEFAULT_PATH.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def db(raw):
    return CardDB(raw)


def match(db):
    return Match(MatchConfig((DEFAULT_DECK, DEFAULT_DECK), shuffle=False), db)


def play(m, side, card, x, y):
    st = m.sides[side]
    if card not in st.hand:
        st.queue.remove(card)
        st.queue.insert(0, st.hand[0])
        st.hand[0] = card
    st.elixir = 10
    assert m.play(side, card, x, y).ok
    return [u for u in m.units.values() if u.card_id == card and u.side == side]


# ---------------------------------------------------------------- data
def test_file_is_schema_4_with_identity(raw, db):
    assert raw["schema_version"] == 4
    for cid, c in raw["cards"].items():
        assert c["internal_name"] and isinstance(c["game_id"], int), cid
        assert c["aliases"]["hf"] == cid and c["aliases"]["display"] == c["name"]
        assert "evolution" not in c and set(c["variants"]) == {"evo", "hero"}
    ids = [c["game_id"] for c in raw["cards"].values()]
    assert len(set(ids)) == len(ids)
    det = [a for c in raw["cards"].values() for a in c["aliases"]["detector"]]
    assert len(set(det)) == len(det)
    assert db.cards["the_log"]["internal_name"] == "Log" and db.cards["knight"]["game_id"] == 26000000


def test_v3_file_still_loads_identically(raw, db):
    v3 = copy.deepcopy(raw)
    v3["schema_version"] = 3
    # v3 mechanics are bare tags without params, so a card that must name its token (Barbarian Barrel) can't be v3
    v3["cards"] = {cid: c for cid, c in v3["cards"].items() if all(m["kind"] != "spawn_on_end" for m in c["mechanics"])}
    v3.pop("tokens", None)
    for c in v3["cards"].values():
        evo = c.pop("variants")["evo"]
        c["evolution"] = evo and {"cycles_required": evo["cycles_required"],
                                  "mechanics": [m["kind"] for m in evo["mechanics"]],
                                  "params": evo["mechanics"][0]["params"] if evo["mechanics"] else {}}
        c["mechanics"] = [m["kind"] for m in c["mechanics"]]
    old = CardDB(v3)
    assert old.evo_cycles("knight") == db.evo_cycles("knight") == 2
    assert old.mechanics_for("knight", "evo") == db.mechanics_for("knight", "evo")
    assert normalise_card(db.cards["knight"]) == db.cards["knight"]          # idempotent


def test_variant_helpers(db):
    assert db.has_variant("knight", "evo") and not db.has_variant("hog_rider", "evo")
    assert set(db.evo_capable()) == {"knight", "firecracker", "skeletons", "zap", "valkyrie", "tesla", "mortar"}
    with pytest.raises(KeyError):
        db.evo_cycles("hog_rider")
    assert db.stats("knight", 11, "evo").hitpoints == db.stats("knight", 11).hitpoints   # no stat_mods yet


def test_stat_mods_scale_variant(raw):
    r = copy.deepcopy(raw)
    r["cards"]["knight"]["variants"]["evo"]["stat_mods"] = {"hitpoints": 110}
    d = CardDB(r)
    assert d.stats("knight", 11, "evo").hitpoints == int(d.stats("knight", 11).hitpoints * 110 / 100)


def test_derived_mechanics(db):
    kinds = lambda cid, v="base": [m["kind"] for m in db.mechanics_for(cid, v)]  # noqa: E731
    assert "decay" in kinds("cannon") and kinds("cannon").count("decay") == 1
    assert "river_jump" in kinds("hog_rider")
    assert db.spell_kind("fireball") == "area_spell" and db.spell_kind("the_log") == "rolling_spell"
    assert "damage_reduction_while_not_attacking" in kinds("knight", "evo")
    assert "damage_reduction_while_not_attacking" not in kinds("knight")


def test_aliases(db):
    assert db.by_alias("knight-evolution") == "knight"
    assert db.by_alias("Log", "internal_name") == "the_log"
    assert db.by_alias("hog-rider", "royaleapi") == "hog_rider"
    assert db.by_alias("The Log") == "the_log" and db.by_alias("nope") is None


def test_unknown_mechanic_and_variant_rejected(raw):
    r = copy.deepcopy(raw)
    r["cards"]["knight"]["mechanics"] = [{"kind": "teleport", "params": {}}]
    with pytest.raises(CardDataError):
        CardDB(r)
    r = copy.deepcopy(raw)
    r["cards"]["knight"]["variants"]["mega"] = None
    with pytest.raises(CardDataError):
        CardDB(r)


# ---------------------------------------------------------------- CardRef / events
def test_card_ref():
    assert str(CardRef("knight", "evo")) == "knight#evo" and str(CardRef("knight")) == "knight"
    assert CardRef.parse("knight#evo") == CardRef("knight", "evo") and CardRef.parse("fireball").variant == "base"
    with pytest.raises(ValueError):
        CardRef("knight", "mega")


def test_event_variant_and_is_evolved_stay_in_sync():
    a = PublicDeployEvent(1, 0, 0, "knight", 3, TilePos(9, 5), is_evolved=True)
    b = PublicDeployEvent(1, 0, 0, "knight", 3, TilePos(9, 5), variant="evo")
    assert a.variant == "evo" and b.is_evolved and a == b and str(a.card_ref) == "knight#evo"


def test_engine_marks_evolved_units(db):
    m = Match(MatchConfig((DEFAULT_DECK, DEFAULT_DECK), evo_cards=(("knight",), ()), shuffle=False), db)
    m.sides[0].evo_counter["knight"] = 2
    (k,) = play(m, 0, "knight", 9.5, 5.5)
    assert k.variant == "evo" and m.events[-1].variant == "evo"


# ---------------------------------------------------------------- framework
def test_units_carry_registered_mechanics_only(db):
    m = match(db)
    (hog,) = play(m, 0, "hog_rider", 3.5, 13.5)
    (cannon,) = play(m, 0, "cannon", 9.5, 8.5)
    assert hog.mechanics == ("river_jump",) and hog.river_jump
    assert cannon.mechanics == ("decay", "projectile")
    assert all(k in REGISTRY for k in hog.mechanics + cannon.mechanics)
    assert set(SPELLS) >= {"area_spell", "rolling_spell"}


def test_custom_mechanic_hooks_run(db):
    calls = []

    @register("_probe")
    class Probe(Mechanic):
        def on_spawn(self, m, u):
            calls.append("spawn")

        def on_damaged(self, m, u, amount, source):
            calls.append("damaged")
            return amount / 2

        def on_death(self, m, u):
            calls.append("death")

    try:
        m = match(db)
        (k,) = play(m, 0, "knight", 9.5, 5.5)
        k.mechanics = ("_probe",)
        hooks_for(k.mechanics, "on_spawn")[0].on_spawn(m, k)
        hp = k.hp
        m._deal(k, 100)
        assert k.hp == hp - 50
        k.hp = 0
        m._resolve_deaths()
        assert calls == ["spawn", "damaged", "death"]
    finally:
        REGISTRY.pop("_probe")
        hooks_for.cache_clear()


def test_status_stun_blocks_move_and_attack(db):
    m = match(db)
    (k,) = play(m, 0, "knight", 3.5, 13.5)
    m.run_until(1000)                       # deployed
    x, y = k.x, k.y
    m.apply_status(k, S.STUN, 500)
    m.run_until(1400)
    assert (k.x, k.y) == (x, y) and S.STUN in k.status
    m.run_until(1600)
    assert not k.status
    m.run_until(2500)
    assert (k.x, k.y) != (x, y)


def test_status_invisible_not_targeted_and_slow(db):
    m = match(db)
    (k,) = play(m, 1, "knight", 4.5, 18.5)
    (e,) = play(m, 0, "knight", 4.5, 13.5)
    assert m._acquire(e).uid == k.uid
    e.target_uid = None
    m.apply_status(k, S.INVISIBLE, 5000)
    assert m._acquire(e).uid != k.uid
    s = match(db)
    (a,) = play(s, 0, "knight", 7.5, 9.5)
    b = play(s, 0, "knight", 10.5, 9.5)[-1]
    s.run_until(1000)
    s.apply_status(a, S.SLOW, 10_000)
    ya, yb = a.y, b.y
    s.run_until(3000)
    assert (a.y - ya) == pytest.approx((b.y - yb) * S.SLOW_FACTOR, rel=0.05)
