"""Deck Builder presets: ready-made decks, all playable with the cards CRbot simulates."""
import pytest

from backend_engine.meta.presets import find_preset, load_presets, playable, select_presets
from backend_engine.simulator.card_db import CardDB, split_slot
from backend_engine.simulator.match_runner import DEFAULT_DECK


def test_presets_validity():
    db = CardDB.load()
    ps = load_presets()
    n = len(ps)
    assert n >= 30
    assert len({p.name for p in ps}) == n                                 # unique names
    assert len({frozenset(p.cards) for p in ps}) == n                     # no duplicate decks
    for p in ps:
        assert len(p.cards) == 8 and len(set(p.cards)) == 8, p.name
        assert all(c in db.cards for c in p.cards), p.name               # every card is simulated
        for slot in p.evo:                                                # "card" = evolution, "card#hero" = hero
            c, var = split_slot(slot)
            assert c in p.cards and db.variant_simulated(c, var), (p.name, slot)
        assert p.style and p.tip, p.name
    assert len(playable(db.cards)) == n
    assert sum(p.ai_pushes for p in ps) >= 20                             # most give the AI a win condition


def test_find_preset():
    starter = find_preset(DEFAULT_DECK)
    assert starter is not None and starter.name == "CRbot Starter"        # the default config has a name
    hog26 = next(p for p in load_presets() if p.name == "Hog 2.6 Cycle")
    assert find_preset(tuple(reversed(hog26.cards))) == hog26             # order does not matter
    assert find_preset(hog26.cards, ("knight",)) == hog26                 # other evo: same cards still match
    assert find_preset(("knight",) * 8) is None
    assert playable({"knight"}) == []


def test_select_presets_makes_a_deck_specialist():
    pool = playable(CardDB.load().cards, load_presets())
    assert select_presets(pool, ()) == pool                      # no names: every playable preset
    hog = select_presets(pool, ("Hog 2.6 Cycle",))
    assert [p.name for p in hog] == ["Hog 2.6 Cycle"] and "hog_rider" in hog[0].cards
    with pytest.raises(ValueError, match="not playable presets"):
        select_presets(pool, ("Hog 2.6 Cycle", "No Such Deck"))
