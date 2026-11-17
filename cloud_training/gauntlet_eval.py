"""Gauntlet: NeuralBot (crbot_il_v2.onnx) vs ScriptedBot, sweeping the gate threshold and decision cadence.

Runs locally on the Mac (CPU, project venv), not on Colab. Every configuration plays the **same** games:
deck pairs drawn deterministically from the preset decks (NeuralBot: any simulated preset; ScriptedBot: presets
it can attack with, `PresetDeck.ai_pushes`), each pair with several seeds and NeuralBot once as side 0 and once
as side 1, so the configurations are compared on identical matchups. Both agents act through their own
HumanDelayBuffer (MatchRunner's default reaction profile).

Reported per configuration: W / L / D, score ((W + D/2) / N) with a Wilson 95% interval, crowns for / against,
plays per match (NeuralBot and ScriptedBot), rejected NeuralBot plays, NeuralBot elixir leaked, model latency.
The best configuration is the one with the highest score (ties: fewer leaked elixir). `--write-meta` stores
its `gate_threshold` / `decide_every_ms` in the model's .json (the original is kept as .json.bak), which is
where NeuralBot reads them from.

    .venv/bin/python cloud_training/gauntlet_eval.py                                  # default sweep
    .venv/bin/python cloud_training/gauntlet_eval.py --thresholds 0.3:0.7:0.05 --cadences 250,500,1000 \\
        --pairs 8 --seeds 2 --workers 3 --out gauntlet.json --write-meta
    .venv/bin/python cloud_training/gauntlet_eval.py --model-dir tests/fixtures/crbot_il_dummy --pairs 1 \\
        --seeds 1 --max-ms 20000                                                       # smoke test

Time: a full match with both bots takes ~1-2 s on one M1 core, so N configs x pairs x seeds x 2 matches; the
default sweep (9 thresholds x 2 cadences x 6 pairs x 2 seeds x 2 sides = 432 matches) is ~5 min on 3 workers.
Each worker holds one engine + one onnxruntime session (~150-250 MB), so keep --workers <= 3 on an 8 GB Mac.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend_engine.brain.neural_bot import MODEL_NAME, load_neural_bot, model_dir, model_files  # noqa: E402
from backend_engine.brain.scripted_bot import ScriptedBot  # noqa: E402
from backend_engine.meta.presets import PresetDeck, load_presets, playable  # noqa: E402
from backend_engine.simulator.card_db import CardDB  # noqa: E402
from backend_engine.simulator.engine import Match, MatchConfig  # noqa: E402
from backend_engine.simulator.match_runner import MatchRunner  # noqa: E402


def wilson_score_interval(k: float, n: int, confidence: float = 0.95) -> Tuple[float, float, float]:
    """Wilson score interval for a binomial proportion (k successes of n; draws may count as 0.5).
    Returns (rate %, lower %, upper %)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    z = 1.95996 if confidence == 0.95 else _z(confidence)
    p = k / n
    denom = 1 + z ** 2 / n
    center = (p + z ** 2 / (2 * n)) / denom
    spread = z * math.sqrt(p * (1 - p) / n + z ** 2 / (4 * n ** 2)) / denom
    return round(p * 100, 2), round(max(0.0, center - spread) * 100, 2), round(min(1.0, center + spread) * 100, 2)


def _z(confidence: float) -> float:
    from statistics import NormalDist
    return NormalDist().inv_cdf(0.5 + confidence / 2)


# -- one match -----------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Game:
    neural_deck: Tuple[str, ...]
    neural_evo: Tuple[str, ...]
    scripted_deck: Tuple[str, ...]
    scripted_evo: Tuple[str, ...]
    seed: int
    neural_side: int
    label: str = ""


@dataclass
class GameResult:
    outcome: str                  # "win" | "loss" | "draw" (NeuralBot's view)
    crowns_for: int
    crowns_against: int
    neural_plays: int
    scripted_plays: int
    neural_rejected: int
    neural_leaked: float
    model_calls: int
    model_ms: float
    t_ms: int


_DB: Optional[CardDB] = None


def _db() -> CardDB:
    global _DB
    if _DB is None:
        _DB = CardDB.load()
    return _DB


def play_game(game: Game, model_folder: str, gate_threshold: float, decide_every_ms: int,
              aggression: float = 0.5, max_ms: Optional[int] = None, model: str = MODEL_NAME) -> GameResult:
    db = _db()
    n, s = game.neural_side, 1 - game.neural_side
    decks = [None, None]
    evos = [None, None]
    decks[n], decks[s] = game.neural_deck, game.scripted_deck
    evos[n], evos[s] = game.neural_evo, game.scripted_evo
    m = Match(MatchConfig((decks[0], decks[1]), (11, 11), (evos[0], evos[1]), seed=game.seed), db)
    bot = load_neural_bot(db, n, Path(model_folder), model, decide_every_ms=decide_every_ms,
                          gate_threshold=gate_threshold)
    calls = [0, 0.0]
    evaluate = bot.evaluate

    def timed(obs):
        t0 = time.perf_counter()
        out = evaluate(obs)
        calls[0] += 1
        calls[1] += time.perf_counter() - t0
        return out
    bot.evaluate = timed
    runner = MatchRunner(m, {n: bot, s: ScriptedBot(db, s, aggression)})
    runner.run(max_ms)
    res = m.result
    cf, ca = m.sides[n].crowns, m.sides[s].crowns
    if res is not None and res.winner is not None:
        outcome = "win" if res.winner == n else "loss"
    elif res is not None:
        outcome = "draw"
    else:                                          # --max-ms cut the match short: judge by crowns
        outcome = "win" if cf > ca else ("loss" if ca > cf else "draw")
    return GameResult(outcome, cf, ca, runner.stats.plays[n], runner.stats.plays[s],
                      len(runner.stats.rejected[n]), round(m.sides[n].leaked_elixir, 2), calls[0],
                      round(calls[1] * 1000, 1), m.t_ms)


# -- schedule ------------------------------------------------------------------------------------------------
def make_games(db: CardDB, pairs: int, seeds: int, base_seed: int = 0) -> List[Game]:
    """Deterministic deck pairs x seeds x both sides (identical for every configuration)."""
    pool = playable(db.cards, load_presets())
    pushers = [p for p in pool if p.ai_pushes]
    if not pool or not pushers:
        raise SystemExit("no playable preset decks")
    rng = random.Random(base_seed)
    chosen: List[Tuple[PresetDeck, PresetDeck]] = []
    for _ in range(pairs):
        chosen.append((rng.choice(pool), rng.choice(pushers)))
    games = []
    for i, (nd, sd) in enumerate(chosen):
        for k in range(seeds):
            seed = base_seed * 1000 + i * 37 + k
            for side in (1, 0):
                games.append(Game(nd.cards, nd.evo, sd.cards, sd.evo, seed, side, f"{nd.name} vs {sd.name}"))
    return games


def parse_range(spec: str) -> List[float]:
    """'0.3:0.7:0.05' (inclusive) or '0.4,0.5'."""
    if ":" in spec:
        a, b, step = (float(x) for x in spec.split(":"))
        n = int(round((b - a) / step))
        return [round(a + i * step, 4) for i in range(n + 1)]
    return [float(x) for x in spec.split(",") if x]


@dataclass
class ConfigResult:
    gate_threshold: float
    decide_every_ms: int
    n: int = 0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    crowns_for: int = 0
    crowns_against: int = 0
    neural_plays: int = 0
    scripted_plays: int = 0
    neural_rejected: int = 0
    neural_leaked: float = 0.0
    model_calls: int = 0
    model_ms: float = 0.0
    by_side: Dict[str, List[int]] = field(default_factory=lambda: {"0": [0, 0, 0], "1": [0, 0, 0]})

    def add(self, g: Game, r: GameResult) -> None:
        self.n += 1
        k = {"win": 0, "loss": 1, "draw": 2}[r.outcome]
        self.wins += k == 0
        self.losses += k == 1
        self.draws += k == 2
        self.by_side[str(g.neural_side)][k] += 1
        self.crowns_for += r.crowns_for
        self.crowns_against += r.crowns_against
        self.neural_plays += r.neural_plays
        self.scripted_plays += r.scripted_plays
        self.neural_rejected += r.neural_rejected
        self.neural_leaked += r.neural_leaked
        self.model_calls += r.model_calls
        self.model_ms += r.model_ms

    def summary(self) -> dict:
        n = max(self.n, 1)
        score, lo, hi = wilson_score_interval(self.wins + 0.5 * self.draws, self.n)
        return {"gate_threshold": self.gate_threshold, "decide_every_ms": self.decide_every_ms, "matches": self.n,
                "wins": self.wins, "losses": self.losses, "draws": self.draws, "score_pct": score,
                "ci95": [lo, hi], "crowns_for": round(self.crowns_for / n, 2),
                "crowns_against": round(self.crowns_against / n, 2),
                "neural_plays": round(self.neural_plays / n, 1), "scripted_plays": round(self.scripted_plays / n, 1),
                "neural_rejected": self.neural_rejected, "neural_leaked": round(self.neural_leaked / n, 2),
                "ms_per_call": round(self.model_ms / max(self.model_calls, 1), 2),
                "by_side": self.by_side}


def _run(args) -> Tuple[int, Game, GameResult]:
    idx, game, folder, thr, every, aggression, max_ms, model = args
    return idx, game, play_game(game, folder, thr, every, aggression, max_ms, model)


def run_sweep(folder: Path, thresholds: Sequence[float], cadences: Sequence[int], games: Sequence[Game],
              workers: int = 1, aggression: float = 0.5, max_ms: Optional[int] = None,
              progress: bool = True, model: str = MODEL_NAME) -> List[ConfigResult]:
    configs = [ConfigResult(t, c) for c in cadences for t in thresholds]
    jobs = [(i, g, str(folder), cfg.gate_threshold, cfg.decide_every_ms, aggression, max_ms, model)
            for i, cfg in enumerate(configs) for g in games]
    t0 = time.time()
    done = 0

    def collect(results):
        nonlocal done
        for i, g, r in results:
            configs[i].add(g, r)
            done += 1
            if progress and (done % 25 == 0 or done == len(jobs)):
                print(f"  {done}/{len(jobs)} matches ({time.time() - t0:.0f} s)", flush=True)
    if workers <= 1:
        collect(_run(j) for j in jobs)
    else:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            collect(ex.map(_run, jobs, chunksize=4))
    return configs


def best_config(configs: Sequence[ConfigResult]) -> dict:
    rows = [c.summary() for c in configs]
    return max(rows, key=lambda r: (r["score_pct"], -r["neural_leaked"], -abs(r["gate_threshold"] - 0.5)))


def print_table(configs: Sequence[ConfigResult]) -> None:
    print(f"{'thr':>5} {'every':>5} | {'W':>3} {'L':>3} {'D':>3} | {'score':>6} {'95% CI':>15} | "
          f"{'cr+':>4} {'cr-':>4} | {'plays N/S':>11} | {'leak':>5} {'rej':>4} | {'ms':>5}")
    for c in configs:
        s = c.summary()
        print(f"{s['gate_threshold']:5.2f} {s['decide_every_ms']:5d} | {s['wins']:3d} {s['losses']:3d} {s['draws']:3d} | "
              f"{s['score_pct']:5.1f}% [{s['ci95'][0]:5.1f}, {s['ci95'][1]:5.1f}] | {s['crowns_for']:4.2f} "
              f"{s['crowns_against']:4.2f} | {s['neural_plays']:5.1f}/{s['scripted_plays']:<5.1f} | "
              f"{s['neural_leaked']:5.2f} {s['neural_rejected']:4d} | {s['ms_per_call']:5.2f}")


def write_meta(folder: Path, best: dict, model: str = MODEL_NAME) -> Path:
    _, meta_path = model_files(folder, model)
    shutil.copy(meta_path, meta_path.with_suffix(".json.bak"))
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    meta["gate_threshold"] = best["gate_threshold"]
    meta["decide_every_ms"] = best["decide_every_ms"]
    meta["gauntlet"] = {k: best[k] for k in ("matches", "wins", "losses", "draws", "score_pct", "ci95")}
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    return meta_path


def main(argv: Optional[Sequence[str]] = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=MODEL_NAME, help="model name: crbot_il_v2 (default) or crbot_ppo_v2")
    ap.add_argument("--model-dir", default="", help="folder with <model>.onnx + .json (default pretrained_weights/<model>)")
    ap.add_argument("--thresholds", default="0.3:0.7:0.05")
    ap.add_argument("--cadences", default="250,500", help="decision cadences in ms (multiples of 50)")
    ap.add_argument("--pairs", type=int, default=6, help="deck pairs")
    ap.add_argument("--seeds", type=int, default=2, help="seeds per deck pair (each played from both sides)")
    ap.add_argument("--base-seed", type=int, default=0)
    ap.add_argument("--aggression", type=float, default=0.5, help="ScriptedBot aggression")
    ap.add_argument("--workers", type=int, default=min(3, os.cpu_count() or 1))
    ap.add_argument("--max-ms", type=int, default=0, help="cut matches short (smoke tests); 0 = full match")
    ap.add_argument("--out", default="", help="write the full report as JSON")
    ap.add_argument("--write-meta", action="store_true", help="store the best config in the model's .json")
    args = ap.parse_args(argv)

    folder = Path(args.model_dir) if args.model_dir else model_dir(args.model)
    load_neural_bot(_db(), 1, folder, args.model)   # fail fast (ModelUnavailable) before the sweep
    games = make_games(_db(), args.pairs, args.seeds, args.base_seed)
    thresholds = parse_range(args.thresholds)
    cadences = [int(c) for c in args.cadences.split(",") if c]
    print(f"model {folder} | {len(thresholds)} thresholds x {len(cadences)} cadences x {len(games)} matches "
          f"= {len(thresholds) * len(cadences) * len(games)} matches on {args.workers} worker(s)")
    configs = run_sweep(folder, thresholds, cadences, games, args.workers, args.aggression, args.max_ms or None,
                        model=args.model)
    print_table(configs)
    best = best_config(configs)
    print(f"best: gate_threshold {best['gate_threshold']}, decide_every_ms {best['decide_every_ms']} -> score "
          f"{best['score_pct']}% [{best['ci95'][0]}, {best['ci95'][1]}] over {best['matches']} matches")
    report = {"model": args.model, "model_dir": str(folder), "games": [asdict(g) for g in games], "aggression": args.aggression,
              "max_ms": args.max_ms or None, "configs": [c.summary() for c in configs], "best": best}
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1), encoding="utf-8")
        print("report:", args.out)
    if args.write_meta:
        print("updated", write_meta(folder, best, args.model))
    return report


if __name__ == "__main__":
    main()
