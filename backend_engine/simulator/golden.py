"""Golden determinism scenarios: fixed setups whose `state_hash` sequence pins down engine behaviour.

`tools/record_golden.py` records them into `tests/golden/engine_hashes.json`; `tests/test_golden_hashes.py`
replays them. A refactor must leave every hash unchanged; an intentional gameplay change re-records them
(`python tools/record_golden.py --write`) and the diff is reviewed.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from backend_engine.brain.scripted_bot import ScriptedBot
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig
from backend_engine.simulator.match_runner import DEFAULT_DECK, MatchRunner

SAMPLE_EVERY_MS = 5000
AIR_DECK = tuple("minions" if c == "skeletons" else c for c in DEFAULT_DECK)


def _give(m: Match, side: int, card: str, elixir: float = 10.0) -> None:
    st = m.sides[side]
    if card not in st.hand:
        st.queue.remove(card)
        st.queue.insert(0, st.hand[0])
        st.hand[0] = card
    st.elixir = elixir


def _play(m: Match, side: int, card: str, x: float, y: float) -> None:
    _give(m, side, card)
    res = m.play(side, card, x, y)
    if not res.ok:
        raise RuntimeError(f"golden setup: {card} rejected ({res.reason})")


def _sampled(m: Match, until_ms: int, step: Optional[Callable[[], None]] = None) -> List[str]:
    """Run to until_ms (or the result), sampling the hash every SAMPLE_EVERY_MS and at the end."""
    out = []
    step = step or m.step
    while m.t_ms < until_ms and m.result is None:
        step()
        if m.t_ms % SAMPLE_EVERY_MS == 0:
            out.append(m.state_hash())
    out.append(m.state_hash())
    return out


def bot_match(db: CardDB, seed: int, level: int, evo: Tuple[Tuple[str, ...], Tuple[str, ...]] = ((), ())) -> List[str]:
    m = Match(MatchConfig((DEFAULT_DECK, DEFAULT_DECK), (level, level), evo, seed=seed), db)
    r = MatchRunner(m, {0: ScriptedBot(db, 0, 0.7), 1: ScriptedBot(db, 1, 0.4)})
    return _sampled(m, m.end_ms + 50, r.tick)


def log_vs_swarm(db: CardDB) -> List[str]:
    m = Match(MatchConfig((DEFAULT_DECK, AIR_DECK), shuffle=False), db)
    _play(m, 1, "knight", 9.5, 20.5)
    _play(m, 1, "minions", 9.5, 22.5)
    _play(m, 1, "archers", 4.5, 21.5)
    _play(m, 0, "skeletons", 9.5, 8.5)
    m.run_until(2500)
    _play(m, 0, "the_log", 9.5, 14.5)
    return _sampled(m, 20_000)


def fireball_king(db: CardDB) -> List[str]:
    m = Match(MatchConfig((DEFAULT_DECK, DEFAULT_DECK), shuffle=False), db)
    king = m.tower(1, "king")
    _play(m, 1, "musketeer", 9.5, 25.0)
    _play(m, 0, "fireball", king.x, king.y)
    return _sampled(m, 15_000)


def cannon_pulls_hog(db: CardDB) -> List[str]:
    m = Match(MatchConfig((DEFAULT_DECK, DEFAULT_DECK), shuffle=False), db)
    _play(m, 0, "hog_rider", 14.5, 14.5)
    _play(m, 1, "cannon", 9.5, 22.5)
    _play(m, 1, "musketeer", 12.5, 24.5)
    return _sampled(m, 30_000)


def hog_river_jump(db: CardDB) -> List[str]:
    m = Match(MatchConfig((DEFAULT_DECK, DEFAULT_DECK), shuffle=False), db)
    _play(m, 0, "hog_rider", 3.5, 13.5)
    _play(m, 0, "knight", 4.5, 11.5)
    _play(m, 1, "skeletons", 3.5, 20.5)
    return _sampled(m, 25_000)


def crowd(db: CardDB) -> List[str]:
    m = Match(MatchConfig((DEFAULT_DECK, AIR_DECK), shuffle=False), db)
    for side, ys in ((0, (10.5, 12.5)), (1, (21.5, 19.5))):
        for y in ys:
            for x in (3.5, 6.5, 11.5, 14.5):
                st = m.sides[side]
                st.elixir = 10
                c = next(c for c in st.hand if db.cards[c]["type"] == "troop")
                m.play(side, c, x, y)
    return _sampled(m, 40_000)


def evo_knight_tank(db: CardDB) -> List[str]:
    m = Match(MatchConfig((DEFAULT_DECK, DEFAULT_DECK), evo_cards=(("knight",), ()), shuffle=False), db)
    m.sides[0].evo_counter["knight"] = 99          # next knight is evolved
    _play(m, 0, "knight", 3.5, 13.5)
    _play(m, 1, "knight", 3.5, 21.0)
    _play(m, 1, "archers", 6.5, 21.5)
    return _sampled(m, 30_000)


def scenarios(db: CardDB) -> Dict[str, Callable[[], List[str]]]:
    out: Dict[str, Callable[[], List[str]]] = {}
    for seed in (1, 2, 3, 4):
        for level in (11, 16):
            out[f"bots_s{seed}_l{level}"] = lambda s=seed, lv=level: bot_match(db, s, lv)
    out["bots_evo_knight"] = lambda: bot_match(db, 9, 11, (("knight",), ("knight",)))
    for fn in (log_vs_swarm, fireball_king, cannon_pulls_hog, hog_river_jump, crowd, evo_knight_tank):
        out[fn.__name__] = lambda fn=fn: fn(db)
    return out
