"""NeuralBot with the dummy ONNX fixture (tests/fixtures/crbot_il_dummy, see tools/make_dummy_il_model.py)."""
import json
import shutil
from pathlib import Path

import pytest

from backend_engine.brain import il_features as F
from backend_engine.brain.neural_bot import (
    ModelUnavailable, NeuralBot, load_neural_bot, make_agent, model_available,
)
from backend_engine.brain.scripted_bot import ScriptedBot
from backend_engine.contracts import ActionKind
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig

FIXTURE = Path(__file__).parent / "fixtures" / "crbot_il_dummy"
DECK = ("knight", "musketeer", "cannon", "hog_rider", "fireball", "skeletons", "the_log", "ice_spirit")


@pytest.fixture(scope="module")
def db():
    return CardDB.load()


def _match(db, hand, elixir, side=1):
    m = Match(MatchConfig((DECK, DECK), seed=0, shuffle=False), db)
    st = m.sides[side]
    st.hand, st.queue = list(hand), [c for c in DECK if c not in hand]
    st.elixir = elixir
    return m


def test_load_and_fallbacks(db, tmp_path):
    bot = load_neural_bot(db, 1, FIXTURE)
    assert isinstance(bot, NeuralBot) and bot.gate_threshold == 0.5
    assert not model_available(tmp_path)
    with pytest.raises(ModelUnavailable):
        load_neural_bot(db, 1, tmp_path)
    agent, note = make_agent(db, 1, "neural", 0.5, tmp_path)
    assert isinstance(agent, ScriptedBot) and "unavailable" in note
    bad = tmp_path / "bad"
    shutil.copytree(FIXTURE, bad)
    meta = json.loads((bad / "crbot_il_v1.json").read_text())
    meta["vocab"] = meta["vocab"][:-1]
    (bad / "crbot_il_v1.json").write_text(json.dumps(meta))
    with pytest.raises(ModelUnavailable, match="vocab"):
        load_neural_bot(db, 1, bad)


@pytest.mark.parametrize("side", [0, 1])
def test_masked_choice_and_frame(db, side):
    """The dummy prefers slot 3 and tile (gx 3, gy 20) in the bot's own frame (enemy side): a troop must land on
    the nearest legal tile (gy 14, just before the river), a spell may go to the favourite tile itself."""
    bot = load_neural_bot(db, side, FIXTURE)
    m = _match(db, ["skeletons", "the_log", "cannon", "knight"], 5.0, side)
    (intent,) = bot.decide(m.observation(side), [], 1000)
    assert intent.kind == ActionKind.DEPLOY and intent.card_id == "knight"
    assert F.tile_index(side, intent.tile.x, intent.tile.y) == 14 * 18 + 3
    assert m.check_play(side, intent.card_id, intent.tile.x, intent.tile.y) is None
    m = _match(db, ["skeletons", "knight", "cannon", "the_log"], 5.0, side)
    (intent,) = bot.decide(m.observation(side), [], 1000)
    assert intent.card_id == "the_log" and F.tile_index(side, intent.tile.x, intent.tile.y) == 20 * 18 + 3


def test_gate_slots_and_cadence(db):
    bot = load_neural_bot(db, 1, FIXTURE)
    m = _match(db, ["musketeer", "hog_rider", "fireball", "skeletons"], 2.0)        # gate 10*0.2-3 < 0
    assert bot.decide(m.observation(1), [], 1000) == []
    assert bot.last["p_play"] < 0.5 and bot.last["card"] == "skeletons"            # only affordable slot
    m = _match(db, ["musketeer", "hog_rider", "fireball", "knight"], 3.5)
    (intent,) = bot.decide(m.observation(1), [], 1000)
    assert intent.card_id == "knight"                                              # slot 3 preferred + affordable
    assert bot.decide(m.observation(1), [], 1050) == []                            # between decision points
    m = _match(db, ["musketeer", "hog_rider", "fireball", "knight"], 0.5)
    assert bot.decide(m.observation(1), [], 1000) == []                            # nothing affordable


def test_session_neural_persona_plays_and_is_deterministic(db, monkeypatch):
    from frontend_app.qt.session import AI, PERSONAS, GameSession, SessionConfig, persona_label
    monkeypatch.setenv("CRBOT_IL_MODEL_DIR", str(FIXTURE))
    assert "neural" in PERSONAS and persona_label("neural") == "Neural (IL v1)"
    hashes = []
    for _ in range(2):
        s = GameSession(db, SessionConfig(persona="neural", seed=11))
        assert isinstance(s.runner.agents[AI], NeuralBot) and s.ai_note is None
        for _ in range(400):                                                      # 20 s
            s.runner.tick()
        assert s.runner.stats.plays[AI] >= 2 and not s.runner.stats.rejected[AI]
        hashes.append(s.match.state_hash())
    assert hashes[0] == hashes[1]


def test_session_neural_persona_falls_back_without_model(db, monkeypatch, tmp_path):
    from frontend_app.qt.session import AI, GameSession, SessionConfig, persona_label
    monkeypatch.setenv("CRBOT_IL_MODEL_DIR", str(tmp_path))
    assert "not installed" in persona_label("neural")
    s = GameSession(db, SessionConfig(persona="neural", seed=3))
    assert isinstance(s.runner.agents[AI], ScriptedBot) and "unavailable" in s.ai_note
