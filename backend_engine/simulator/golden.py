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


HOG_26 = ("cannon", "fireball", "hog_rider", "ice_golem", "ice_spirit", "musketeer", "skeletons", "the_log")
SPIRITS = ("zap", "ice_spirit", "fire_spirit", "electro_spirit", "ice_golem", "valkyrie", "skeleton_army", "mini_pekka")
STEALTH = ("barbarian_barrel", "royal_ghost", "tornado", "zap", "valkyrie", "ice_golem", "fire_spirit", "mini_pekka")


def bot_match(db: CardDB, seed: int, level: int, evo: Tuple[Tuple[str, ...], Tuple[str, ...]] = ((), ()),
              deck: Tuple[str, ...] = DEFAULT_DECK) -> List[str]:
    m = Match(MatchConfig((deck, deck), (level, level), evo, seed=seed), db)
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


# -- batch 1: status effects, kamikaze hits, death bursts, self-centred splash -------------------------------
def zap_stun(db: CardDB) -> List[str]:
    m = Match(MatchConfig((SPIRITS, DEFAULT_DECK), shuffle=False), db)
    _play(m, 1, "knight", 9.5, 19.5)
    _play(m, 1, "archers", 6.5, 20.5)
    _play(m, 0, "mini_pekka", 9.5, 14.5)
    m.run_until(3000)
    _play(m, 0, "zap", 9.0, 18.0)
    return _sampled(m, 15_000)


def ice_spirit_freeze(db: CardDB) -> List[str]:
    m = Match(MatchConfig((SPIRITS, DEFAULT_DECK), shuffle=False), db)
    _play(m, 1, "knight", 9.5, 19.5)
    _play(m, 0, "ice_spirit", 9.5, 14.5)
    return _sampled(m, 12_000)


def electro_chain(db: CardDB) -> List[str]:
    m = Match(MatchConfig((SPIRITS, DEFAULT_DECK), shuffle=False), db)
    _play(m, 1, "skeletons", 9.5, 19.5)
    _play(m, 1, "knight", 12.5, 19.5)
    _play(m, 0, "electro_spirit", 9.5, 14.5)
    return _sampled(m, 12_000)


def ice_golem_burst(db: CardDB) -> List[str]:
    m = Match(MatchConfig((SPIRITS, DEFAULT_DECK), shuffle=False), db)
    _play(m, 0, "ice_golem", 9.5, 14.5)
    _play(m, 1, "knight", 9.5, 20.5)
    _play(m, 1, "archers", 6.5, 21.5)
    return _sampled(m, 40_000)


def valkyrie_swarm(db: CardDB) -> List[str]:
    m = Match(MatchConfig((SPIRITS, DEFAULT_DECK), shuffle=False), db)
    _play(m, 0, "valkyrie", 9.5, 14.5)
    _play(m, 1, "skeletons", 9.5, 19.5)
    _play(m, 1, "archers", 6.5, 20.5)
    return _sampled(m, 25_000)


def skeleton_army_fireball(db: CardDB) -> List[str]:
    m = Match(MatchConfig((SPIRITS, DEFAULT_DECK), shuffle=False), db)
    _play(m, 0, "skeleton_army", 9.5, 14.5)
    _play(m, 1, "knight", 9.5, 22.5)
    m.run_until(6000)
    _play(m, 1, "fireball", 9.5, 17.0)
    return _sampled(m, 20_000)


def barrel_spawn(db: CardDB) -> List[str]:
    m = Match(MatchConfig((STEALTH, DEFAULT_DECK), shuffle=False), db)
    _play(m, 1, "knight", 9.5, 20.5)
    _play(m, 1, "archers", 6.5, 21.5)
    _play(m, 0, "barbarian_barrel", 9.5, 10.5)
    return _sampled(m, 20_000)


def ghost_stealth(db: CardDB) -> List[str]:
    m = Match(MatchConfig((STEALTH, DEFAULT_DECK), shuffle=False), db)
    _play(m, 1, "knight", 9.5, 20.5)
    _play(m, 1, "archers", 6.5, 21.5)
    _play(m, 0, "royal_ghost", 9.5, 10.5)
    return _sampled(m, 30_000)


def tornado_pull(db: CardDB) -> List[str]:
    m = Match(MatchConfig((STEALTH, DEFAULT_DECK), shuffle=False), db)
    _play(m, 1, "knight", 9.5, 19.5)
    _play(m, 1, "archers", 6.5, 20.5)
    _play(m, 1, "musketeer", 12.5, 21.5)
    m.run_until(2000)
    _play(m, 0, "tornado", 9.5, 20.0)
    return _sampled(m, 15_000)


B2_A = ("dark_prince", "miner", "tesla", "firecracker", "goblin_demolisher", "mighty_miner", "golden_knight",
        "royal_delivery")
B2_B = ("earthquake", "lightning", "vines", "knight", "archers", "musketeer", "valkyrie", "hog_rider")


def dark_prince_charge(db: CardDB) -> List[str]:
    m = Match(MatchConfig((B2_A, B2_B), shuffle=False), db)
    _play(m, 1, "knight", 3.5, 21.0)
    _play(m, 1, "archers", 3.5, 23.5)
    _play(m, 0, "dark_prince", 3.5, 10.0)
    return _sampled(m, 14_000)


def spells_batch2(db: CardDB) -> List[str]:
    m = Match(MatchConfig((B2_B, B2_A), shuffle=False), db)      # side 0 casts, side 1 fields the targets
    for card, x, y in (("goblin_demolisher", 9.5, 21.0), ("dark_prince", 8.5, 22.5), ("golden_knight", 11.0, 22.5),
                       ("tesla", 9.5, 24.0), ("mighty_miner", 10.5, 20.5)):
        _play(m, 1, card, x, y)
    for t, card in ((1000, "lightning"), (3500, "vines"), (7000, "earthquake")):
        m.run_until(t)
        _play(m, 0, card, 9.5, 22.0)
    return _sampled(m, 14_000)


def tesla_demolisher_miner(db: CardDB) -> List[str]:
    m = Match(MatchConfig((B2_A, B2_A), shuffle=False), db)
    _play(m, 0, "tesla", 9.5, 12.0)
    _play(m, 1, "goblin_demolisher", 3.5, 22.0)
    _play(m, 0, "miner", 11.5, 27.5)
    m.run_until(8000)
    _play(m, 1, "dark_prince", 9.5, 22.0)
    return _sampled(m, 24_000)


def delivery_ramp_firecracker(db: CardDB) -> List[str]:
    m = Match(MatchConfig((B2_A, B2_A), shuffle=False), db)
    _play(m, 1, "golden_knight", 9.5, 21.0)
    _play(m, 0, "royal_delivery", 9.5, 21.0)
    _play(m, 0, "firecracker", 9.5, 10.0)
    _play(m, 0, "mighty_miner", 3.5, 12.0)
    return _sampled(m, 16_000)


CHAMPIONS = ("golden_knight", "mighty_miner", "hog_rider", "knight", "musketeer", "fireball", "the_log", "skeletons")


def champion_abilities(db: CardDB) -> List[str]:
    m = Match(MatchConfig((B2_A, B2_B), shuffle=False), db)
    for card, x, y in (("knight", 9.5, 21.0), ("valkyrie", 11.0, 22.0), ("musketeer", 7.5, 22.0)):
        _play(m, 1, card, x, y)
    _play(m, 0, "golden_knight", 9.5, 14.0)
    _play(m, 0, "mighty_miner", 4.5, 12.0)
    m.run_until(3000)
    m.sides[0].elixir = 6.0
    m.use_ability(0, "golden_knight")
    m.run_until(5000)
    m.use_ability(0, "mighty_miner")
    return _sampled(m, 14_000)


MODERN = ("rune_giant", "little_prince", "bats", "minion_giant", "poison", "knight", "musketeer", "valkyrie")


def modern_meta(db: CardDB) -> List[str]:
    m = Match(MatchConfig((MODERN, MODERN), shuffle=False), db)
    for card, x, y in (("minion_giant", 3.5, 22.0), ("valkyrie", 9.5, 21.0), ("bats", 11.0, 22.0)):
        _play(m, 1, card, x, y)
    _play(m, 0, "rune_giant", 9.5, 12.0)
    _play(m, 0, "knight", 8.0, 12.0)
    _play(m, 0, "little_prince", 11.0, 10.0)
    m.run_until(3000)
    _play(m, 0, "poison", 9.5, 22.0)
    m.run_until(5000)
    m.sides[0].elixir = 6.0
    m.use_ability(0, "little_prince")
    return _sampled(m, 16_000)


def scenarios(db: CardDB) -> Dict[str, Callable[[], List[str]]]:
    out: Dict[str, Callable[[], List[str]]] = {}
    for seed in (1, 2, 3, 4):
        for level in (11, 16):
            out[f"bots_s{seed}_l{level}"] = lambda s=seed, lv=level: bot_match(db, s, lv)
    out["bots_evo_knight"] = lambda: bot_match(db, 9, 11, (("knight",), ("knight",)))
    for fn in (log_vs_swarm, fireball_king, cannon_pulls_hog, hog_river_jump, crowd, evo_knight_tank,
               zap_stun, ice_spirit_freeze, electro_chain, ice_golem_burst, valkyrie_swarm, skeleton_army_fireball,
               barrel_spawn, ghost_stealth, tornado_pull, dark_prince_charge, spells_batch2, tesla_demolisher_miner,
               delivery_ramp_firecracker, champion_abilities, modern_meta):
        out[fn.__name__] = lambda fn=fn: fn(db)
    out["bots_hog26_s5"] = lambda: bot_match(db, 5, 11, deck=HOG_26)
    out["bots_stealth_s6"] = lambda: bot_match(db, 6, 11, deck=STEALTH)
    out["bots_modern_s8"] = lambda: bot_match(db, 8, 11, deck=MODERN)
    out["bots_champions_s7"] = lambda: bot_match(db, 7, 11, deck=CHAMPIONS)
    return out
