"""Deck Builder presets: ready-made decks, all playable with the cards CRbot simulates."""
from backend_engine.meta.presets import find_preset, load_presets, playable
from backend_engine.simulator.card_db import CardDB
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
        assert all(e in p.cards and db.has_variant(e, "evo") for e in p.evo), p.name
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
