"""
Train the win-probability model on bot-vs-bot games in our simulator.

    python tools/train_value_model.py --games 200

Each game: random seed, random aggression for both scripted bots, random level (11/16),
occasionally varied decks. A state is sampled every 3 s; label = did that side win.
Both perspectives are used (features flip sign, label flips), so the model is symmetric.
Draws are dropped. 20% of games are held out to report accuracy and log loss.
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend_engine.brain.scripted_bot import ScriptedBot  # noqa: E402
from backend_engine.brain.value_model import DEFAULT_PATH, FEATURES, features  # noqa: E402
from backend_engine.simulator.card_db import CardDB  # noqa: E402
from backend_engine.simulator.engine import Match, MatchConfig  # noqa: E402
from backend_engine.simulator.match_runner import DEFAULT_DECK, MatchRunner  # noqa: E402

SAMPLE_EVERY_MS = 3000


def play_game(db, rng):
    decks = []
    for _ in range(2):
        d = list(DEFAULT_DECK)
        if rng.random() < 0.4:
            d[d.index("skeletons")] = "minions"
        decks.append(tuple(d))
    lvl = rng.choice([11, 16])
    m = Match(MatchConfig(tuple(decks), (lvl, lvl), seed=rng.randrange(1 << 30)), db)
    r = MatchRunner(m, {0: ScriptedBot(db, 0, rng.uniform(0.1, 1.0)), 1: ScriptedBot(db, 1, rng.uniform(0.1, 1.0))})
    states = []
    while m.result is None:
        if m.t_ms % SAMPLE_EVERY_MS == 0:
            states.append(features(m, 0))
        r.tick()
    return states, m.result


def fit(X, y, l2=1e-3, iters=3000, lr=0.5):
    w = np.zeros(X.shape[1])
    for _ in range(iters):
        p = 1 / (1 + np.exp(-(X @ w)))
        grad = X.T @ (p - y) / len(y) + l2 * w
        w -= lr * grad
    return w


def evaluate(X, y, w):
    p = np.clip(1 / (1 + np.exp(-(X @ w))), 1e-6, 1 - 1e-6)
    return float(np.mean((p > 0.5) == (y > 0.5))), float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=200)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=DEFAULT_PATH)
    a = ap.parse_args()

    db = CardDB.load()
    rng = random.Random(a.seed)
    games, draws = [], 0
    t0 = time.perf_counter()
    for i in range(a.games):
        states, res = play_game(db, rng)
        if res.winner is None:
            draws += 1
            continue
        games.append((states, res.winner))
        if (i + 1) % 25 == 0:
            print(f"  {i + 1}/{a.games} games ({time.perf_counter() - t0:.0f}s)", flush=True)

    def to_xy(gs):
        X, y = [], []
        for states, winner in gs:
            for f in states:
                X.append(f)
                y.append(1.0 if winner == 0 else 0.0)
                X.append([-v for v in f])
                y.append(0.0 if winner == 0 else 1.0)
        return np.array(X), np.array(y)

    rng.shuffle(games)
    cut = int(len(games) * 0.8)
    Xtr, ytr = to_xy(games[:cut])
    Xte, yte = to_xy(games[cut:])
    w = fit(Xtr, ytr)
    acc, ll = evaluate(Xte, yte, w)
    base_ll = math.log(2)
    late = Xte[:, FEATURES.index("crowns_late")] != 0  # not a great slice; report overall
    meta = {"trained": True, "games": a.games, "decisive_games": len(games), "draws": draws,
            "train_states": int(len(ytr)), "test_states": int(len(yte)),
            "test_accuracy": round(acc, 4), "test_log_loss": round(ll, 4), "coin_flip_log_loss": round(base_ll, 4),
            "sample_every_ms": SAMPLE_EVERY_MS, "opponents": "ScriptedBot vs ScriptedBot"}
    a.out.write_text(json.dumps({"features": FEATURES, "weights": [round(float(v), 5) for v in w], "meta": meta},
                                indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, indent=1))
    print("weights:", dict(zip(FEATURES, [round(float(v), 3) for v in w])))
    _ = late


if __name__ == "__main__":
    main()
