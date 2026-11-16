"""PPO self-play pipeline: bundle packer, PPO model loading in the app, and (with torch) a 2-update smoke run."""
import json
import shutil
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
IL = ROOT / "pretrained_weights" / "crbot_il_v1" / "crbot_il_v1.onnx"
FIXTURE = Path(__file__).parent / "fixtures" / "crbot_il_dummy"


def test_neural_bot_loads_a_ppo_model_by_name(tmp_path, monkeypatch):
    from backend_engine.brain.neural_bot import PPO_MODEL_NAME, NeuralBot, load_neural_bot, make_agent, model_available
    from backend_engine.brain.scripted_bot import ScriptedBot
    from backend_engine.simulator.card_db import CardDB
    folder = tmp_path / PPO_MODEL_NAME
    folder.mkdir()
    shutil.copy(FIXTURE / "crbot_il_v1.onnx", folder / f"{PPO_MODEL_NAME}.onnx")
    meta = json.loads((FIXTURE / "crbot_il_v1.json").read_text())
    meta.update(name=PPO_MODEL_NAME, decide_every_ms=250)
    (folder / f"{PPO_MODEL_NAME}.json").write_text(json.dumps(meta))
    db = CardDB.load()
    bot = load_neural_bot(db, 1, folder, PPO_MODEL_NAME)
    assert isinstance(bot, NeuralBot) and bot.decide_every_ms == 250
    monkeypatch.setenv("CRBOT_CRBOT_PPO_V1_DIR", str(folder))
    assert model_available(name=PPO_MODEL_NAME)
    agent, note = make_agent(db, 1, "neural", model=PPO_MODEL_NAME)
    assert isinstance(agent, NeuralBot) and note is None
    monkeypatch.setenv("CRBOT_CRBOT_PPO_V1_DIR", str(tmp_path / "nothing"))
    agent, note = make_agent(db, 1, "neural", model=PPO_MODEL_NAME)
    assert isinstance(agent, ScriptedBot) and "crbot_ppo_v1" in note


@pytest.mark.skipif(not IL.exists(), reason="needs pretrained_weights/crbot_il_v1 (git-ignored weights)")
def test_bundle_contents(tmp_path):
    from tools.pack_ppo_bundle import MODEL_FILES, build, tracked_files
    files = tracked_files() + list(MODEL_FILES)
    out = tmp_path / "b.zip"
    m = build(out, files)
    names = set(zipfile.ZipFile(out).namelist())
    for f in ("backend_engine/brain/ppo_trainer.py", "backend_engine/brain/il_model.py",
              "backend_engine/brain/il_features.py", "backend_engine/simulator/engine.py",
              "configs/cards_stats_v2026.json", "configs/preset_decks.json", *MODEL_FILES, "BUNDLE_MANIFEST.json"):
        assert f in names, f
    assert not any(n.startswith(("data/", "user_data/", "recordings/", "tests/")) or "__pycache__" in n for n in names)
    assert not any(n.endswith((".bak", ".pt")) for n in names)
    assert len(m["files"]) == len(names) - 1 and all(len(e["sha256"]) == 64 for e in m["files"])


@pytest.mark.skipif(not IL.exists(), reason="needs pretrained_weights/crbot_il_v1 (git-ignored weights)")
def test_ppo_smoke_two_updates(tmp_path):
    torch = pytest.importorskip("torch")
    pytest.importorskip("onnx")
    pytest.importorskip("onnxruntime")
    from backend_engine.brain.il_model import check_against_onnx, load_onnx_weights
    from backend_engine.brain.neural_bot import PPO_MODEL_NAME, load_neural_bot
    from backend_engine.brain.ppo_trainer import PPOConfig, PPOTrainer
    from backend_engine.simulator.card_db import CardDB

    model = load_onnx_weights(IL)                                  # rebuilt weights == the ONNX graph
    assert check_against_onnx(model, IL) < 1e-4
    cfg = PPOConfig(n_envs=2, updates=2, max_match_ms=12_000, minibatch=32, epochs=2, ckpt_every=1,
                    snapshot_every=1, score_window=1)
    tr = PPOTrainer(cfg, IL, tmp_path / "ckpt", tmp_path / "out", log=lambda *_: None)
    before = {k: v.clone() for k, v in tr.model.state_dict().items()}
    tr.train()
    h = tr.history
    assert len(h) == 2 and all(r["steps"] > 0 for r in h)
    for r in h:
        for k in ("pg", "vf", "ent", "kl_ref", "approx_kl"):
            assert r[k] == r[k] and abs(r[k]) < 1e3, (k, r[k])        # finite
        assert r["kl_ref"] >= 0
    assert any(not torch.equal(before[k], v) for k, v in tr.model.state_dict().items())   # it learned something
    for p in tr.ref.parameters():
        assert not p.requires_grad
    out = tmp_path / "out"
    assert (out / f"{PPO_MODEL_NAME}.onnx").exists() and (tmp_path / "ckpt" / "ppo_state.pt").exists()
    meta = json.loads((out / f"{PPO_MODEL_NAME}.json").read_text())
    assert meta["init"] == "crbot_il_v1" and meta["onnx_check_max_abs_diff"] < 1e-3
    bot = load_neural_bot(CardDB.load(), 0, out, PPO_MODEL_NAME)
    assert bot.gate_threshold == 0.45 and bot.decide_every_ms == 500
    again = PPOTrainer(cfg, IL, tmp_path / "ckpt", tmp_path / "out", log=lambda *_: None)    # resume
    assert again.update == 2 and len(again.snapshots) == 2
