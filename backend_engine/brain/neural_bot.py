"""NeuralBot: the imitation-learning policy (crbot_il_v1.onnx) as a MatchRunner agent.

The model is trained by cloud_training/train_imitation_colab.ipynb on shards from tools/build_il_dataset.py.
Both use backend_engine/brain/il_features.py, so the observation here is built by exactly the same code as the
training data: the fog-of-war `observation()` MatchRunner passes in (only events the HumanDelayBuffer has let
the bot "see"), plus the bot's own opponent-elixir estimate and deck guess. Actions still go through the
HumanDelayBuffer like every other agent, so reaction time, APM and combo limits apply.

Decision: every DECIDE_EVERY_MS of match time (fixed boundaries, so the bot has no hidden state and forked
rollouts / match rebuilds replay identically), if a card is affordable: play when sigmoid(gate) >= threshold,
take the best affordable slot, then the best legal tile for that card. Champion / hero abilities use the
scripted rules (the model has no ability head). Inference runs on one CPU thread (~1-2 ms per call).

Files: pretrained_weights/crbot_il_v1/crbot_il_v1.onnx + crbot_il_v1.json (git-ignored; download from the
notebook). Override the folder with CRBOT_IL_MODEL_DIR. `load_neural_bot` raises ModelUnavailable when the files
are missing or were trained for another observation version / card vocabulary; `make_agent` then falls back.
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend_engine.brain import il_features as F
from backend_engine.brain.human_delay_buffer import HumanDelayBuffer
from backend_engine.brain.scripted_bot import ScriptedBot
from backend_engine.contracts import ActionIntent, ActionKind, PublicDeployEvent, TilePos
from backend_engine.perception.opp_elixir_count import OppElixirTracker
from backend_engine.simulator.card_db import CardDB

ROOT = Path(__file__).resolve().parents[2]
MODEL_NAME = "crbot_il_v1"
DEFAULT_MODEL_DIR = ROOT / "pretrained_weights" / MODEL_NAME
DECIDE_EVERY_MS = 500
# fallback when the model .json has no gate_threshold: the crbot_il_v1 gauntlet optimum (2026-11-16: 0.45 at 500 ms,
# 23-1 vs ScriptedBot; every setting from 0.30 to 0.70 scored within the same 95% interval)
DEFAULT_GATE_THRESHOLD = 0.45
OUTPUTS = ("gate_logit", "slot_logits", "tile_logits", "value")


class ModelUnavailable(RuntimeError):
    """The IL model can't be used (missing files, no onnxruntime, or trained for different features)."""


def model_dir() -> Path:
    return Path(os.environ.get("CRBOT_IL_MODEL_DIR") or DEFAULT_MODEL_DIR)


def model_files(folder: Optional[Path] = None) -> Tuple[Path, Path]:
    folder = Path(folder) if folder is not None else model_dir()
    return folder / f"{MODEL_NAME}.onnx", folder / f"{MODEL_NAME}.json"


def model_available(folder: Optional[Path] = None) -> bool:
    onnx, meta = model_files(folder)
    return onnx.exists() and meta.exists()


_SESSIONS: Dict[Tuple[str, float], object] = {}


def _session(path: Path):
    """One onnxruntime session per model file (cached: new matches and analysis forks reuse it)."""
    key = (str(path), path.stat().st_mtime)
    if key not in _SESSIONS:
        try:
            import onnxruntime as ort
        except ImportError as e:                           # pragma: no cover - onnxruntime is a core dependency
            raise ModelUnavailable(f"onnxruntime is not installed: {e}") from e
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = 1                      # deterministic + light on the M1
        opts.inter_op_num_threads = 1
        _SESSIONS[key] = ort.InferenceSession(str(path), opts, providers=["CPUExecutionProvider"])
    return _SESSIONS[key]


def check_meta(meta: dict, db: CardDB) -> None:
    vocab = F.card_vocab(db)
    if meta.get("obs_version") != F.OBS_VERSION:
        raise ModelUnavailable(f"model features {meta.get('obs_version')!r}, app uses {F.OBS_VERSION!r}")
    if list(meta.get("vocab") or []) != vocab:
        raise ModelUnavailable("model was trained on a different card pool (vocab mismatch): rebuild the shards "
                               "and retrain")
    if int(meta.get("obs_dim", -1)) != F.obs_dim(vocab):
        raise ModelUnavailable(f"model obs_dim {meta.get('obs_dim')} != {F.obs_dim(vocab)}")


class NeuralBot:
    """Agent protocol of MatchRunner: decide(obs, visible_events, now_ms, buffer) -> [ActionIntent]."""

    def __init__(self, db: CardDB, side: int, session, meta: dict, decide_every_ms: Optional[int] = None,
                 gate_threshold: Optional[float] = None):
        self.db = db
        self.side = side
        self.session = session
        self.meta = meta
        self.vocab = F.card_vocab(db)
        every = int(decide_every_ms if decide_every_ms is not None else meta.get("decide_every_ms", DECIDE_EVERY_MS))
        if every <= 0 or every % 50:
            raise ValueError(f"decide_every_ms must be a positive multiple of the 50 ms tick, got {every}")
        self.decide_every_ms = every
        self.gate_threshold = float(gate_threshold if gate_threshold is not None
                                    else meta.get("gate_threshold", DEFAULT_GATE_THRESHOLD))
        self._input = session.get_inputs()[0].name
        self._rules = ScriptedBot(db, side)                # abilities only
        self.last: Optional[dict] = None                   # last model call (for debugging / UI)

    # ------------------------------------------------------------------
    def _estimate(self, obs: dict):
        est = obs.get("opp_estimate")
        if est is None:                                    # outside MatchRunner: nothing seen yet
            est = OppElixirTracker(self.db.rules, observer_side=self.side).estimate(obs["t_ms"])
        return est, obs.get("opp_deck_guess")

    def evaluate(self, obs: dict) -> dict:
        """Run the model on one observation: gate probability, value and the masked best action."""
        est, guess = self._estimate(obs)
        vec = F.encode_observation(obs, est, guess, self.db, self.vocab)
        gate, slots, tiles, value = self.session.run(list(OUTPUTS), {self._input: vec[None, :]})
        smask = F.slot_mask(obs, self.db)
        out = {"t_ms": obs["t_ms"], "p_play": float(1.0 / (1.0 + np.exp(-float(np.ravel(gate)[0])))),
               "value": float(np.ravel(value)[0]), "card": None, "tile": None}
        if smask.any():
            slot = int(np.argmax(np.where(smask, slots[0], -np.inf)))
            card = obs["hand"][slot]
            tmask = F.tile_mask(obs, card, self.db)
            if tmask.any():
                tile = int(np.argmax(np.where(tmask, tiles[0, slot], -np.inf)))
                out.update(slot=slot, card=card, tile=tile, xy=F.tile_center(self.side, tile))
        self.last = out
        return out

    def decide(self, obs: dict, visible_events: Sequence[PublicDeployEvent], now_ms: int,
               buffer: Optional[HumanDelayBuffer] = None) -> List[ActionIntent]:
        ab = self._rules._ability(obs)
        if ab:
            return [ab]
        if now_ms % self.decide_every_ms:
            return []
        if not F.slot_mask(obs, self.db).any():
            return []
        out = self.evaluate(obs)
        if out["card"] is None or out["p_play"] < self.gate_threshold:
            return []
        x, y = out["xy"]
        return [ActionIntent(ActionKind.DEPLOY, out["card"], TilePos(x, y), self.db.cards[out["card"]]["elixir"],
                             source="policy")]


def load_neural_bot(db: CardDB, side: int, folder: Optional[Path] = None, **kw) -> NeuralBot:
    onnx_path, meta_path = model_files(folder)
    if not (onnx_path.exists() and meta_path.exists()):
        raise ModelUnavailable(f"no {MODEL_NAME} model in {onnx_path.parent}")
    meta = json.loads(meta_path.read_text(encoding="utf-8"))
    check_meta(meta, db)
    try:
        session = _session(onnx_path)
    except ModelUnavailable:
        raise
    except Exception as e:                                 # corrupt / incompatible file
        raise ModelUnavailable(f"could not load {onnx_path.name}: {e}") from e
    names = {o.name for o in session.get_outputs()}
    if not set(OUTPUTS) <= names:
        raise ModelUnavailable(f"{onnx_path.name} lacks outputs {sorted(set(OUTPUTS) - names)}")
    return NeuralBot(db, side, session, meta, **kw)


def make_agent(db: CardDB, side: int, kind: str = "scripted", aggression: float = 0.5,
               folder: Optional[Path] = None):
    """Agent for a persona: (agent, note). kind "neural" falls back to ScriptedBot (note says why)."""
    if kind == "neural":
        try:
            return load_neural_bot(db, side, folder), None
        except ModelUnavailable as e:
            return ScriptedBot(db, side, aggression), f"Neural model unavailable ({e}); using the rule-based AI."
    return ScriptedBot(db, side, aggression), None
