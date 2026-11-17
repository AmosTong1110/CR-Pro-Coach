"""Gauntlet sweep (cloud_training/gauntlet_eval.py) smoke test with the dummy ONNX model."""
import json
import shutil
from pathlib import Path

import pytest

from backend_engine.brain.neural_bot import load_neural_bot
from backend_engine.simulator.card_db import CardDB
from cloud_training.gauntlet_eval import main, make_games, parse_range, wilson_score_interval

FIXTURE = Path(__file__).parent / "fixtures" / "crbot_il_dummy"


def test_helpers():
    assert parse_range("0.3:0.7:0.1") == [0.3, 0.4, 0.5, 0.6, 0.7]
    assert parse_range("0.4,0.6") == [0.4, 0.6]
    assert wilson_score_interval(0, 0) == (0.0, 0.0, 0.0)
    rate, lo, hi = wilson_score_interval(64, 150)
    assert rate == pytest.approx(42.67, abs=0.01) and lo < rate < hi
    games = make_games(CardDB.load(), pairs=2, seeds=2)
    assert len(games) == 8 and {g.neural_side for g in games} == {0, 1}
    assert games == make_games(CardDB.load(), pairs=2, seeds=2)          # same schedule for every config


def test_sweep_and_write_meta(tmp_path):
    model = tmp_path / "model"
    shutil.copytree(FIXTURE, model)
    report = main(["--model-dir", str(model), "--thresholds", "0.4,0.6", "--cadences", "250,500",
                   "--pairs", "1", "--seeds", "1", "--max-ms", "15000", "--workers", "1",
                   "--out", str(tmp_path / "r.json"), "--write-meta"])
    assert len(report["configs"]) == 4
    for c in report["configs"]:
        assert c["matches"] == 2 and c["wins"] + c["losses"] + c["draws"] == 2
        assert c["neural_plays"] > 0 and c["neural_rejected"] == 0
    assert json.loads((tmp_path / "r.json").read_text())["best"] == report["best"]
    meta = json.loads((model / "crbot_il_v2.json").read_text())
    assert (meta["gate_threshold"], meta["decide_every_ms"]) == (report["best"]["gate_threshold"],
                                                                 report["best"]["decide_every_ms"])
    assert (model / "crbot_il_v2.json.bak").exists()
    bot = load_neural_bot(CardDB.load(), 1, model)
    assert bot.decide_every_ms == meta["decide_every_ms"] and bot.gate_threshold == meta["gate_threshold"]


def test_cadence_must_be_tick_multiple():
    with pytest.raises(ValueError):
        load_neural_bot(CardDB.load(), 1, FIXTURE, decide_every_ms=120)
