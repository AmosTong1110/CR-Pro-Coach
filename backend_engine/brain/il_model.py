"""The crbot policy / value network in PyTorch (same architecture as cloud_training/train_imitation_colab.ipynb).

Used by the PPO trainer (backend_engine/brain/ppo_trainer.py) and anything else that needs gradients. The app
itself never imports this module: NeuralBot runs the exported ONNX file with onnxruntime. Needs `torch`.

* `CRbotIL`: token transformer over 1 global token, 5 hand tokens and up to 30 unit tokens built from the
  il_features observation (OBS_VERSION il_obs_v1); heads gate [B], slot [B,4], tile [B,4,576], value [B] (sigmoid).
* `load_onnx_weights(path)`: rebuild the PyTorch weights from an exported ONNX file (no pickle). The exporter
  keeps the parameter names of biases / norms / heads, stores Linear weights as transposed MatMul constants
  named after their module path (/blocks.0/qkv/MatMul) and folds the token-type embedding into four Add
  constants; all are mapped back, and the result is checked against onnxruntime (`verify=True`).
* `export_onnx(model, path)`: the same inputs / outputs NeuralBot expects.
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import torch
import torch.nn as nn

from backend_engine.brain import il_features as Fe

NUM_TILES = Fe.NUM_TILES
GW, GH = Fe.GRID_W, Fe.GRID_H
HF, NH = Fe.HAND_FEAT_DIM, Fe.HAND_SLOTS + 1
UF, NU = Fe.UNIT_FEAT_DIM, Fe.MAX_VISIBLE_UNITS
OUTPUTS = ("gate_logit", "slot_logits", "tile_logits", "value")
ARCH = dict(d=256, h=8, layers=4, ff=512)          # crbot_il_v1


def layout(vocab_size: int) -> Tuple[int, int, int, int]:
    """(G_END, H0, U0, OBS_DIM) for a card vocabulary of this size."""
    g_end = Fe.N_GLOBAL + Fe.N_TOWER + 4 + vocab_size + 1
    h0 = g_end
    u0 = h0 + NH * HF
    return g_end, h0, u0, u0 + NU * UF


class Block(nn.Module):
    def __init__(self, d, h, ff, p):
        super().__init__()
        self.h, self.dk = h, d // h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(p)

    def forward(self, x, key_bias):
        B, S, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, S, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dk) + key_bias, dim=-1)
        y = (self.drop(att) @ v).transpose(1, 2).reshape(B, S, D)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.ff(self.ln2(x)))


class CRbotIL(nn.Module):
    def __init__(self, vocab_size: int, d=256, h=8, layers=4, ff=512, p=0.0):
        super().__init__()
        self.V = vocab_size
        self.G_END, self.H0, self.U0, self.OBS_DIM = layout(vocab_size)
        self.glob = nn.Sequential(nn.Linear(self.G_END, d), nn.GELU(), nn.Linear(d, d))
        self.card_emb = nn.Embedding(vocab_size + 1, d)
        self.hand_in = nn.Linear(HF, d)
        self.unit_in = nn.Sequential(nn.Linear(UF, d), nn.GELU(), nn.Linear(d, d))
        self.type_emb = nn.Parameter(torch.zeros(1 + 1 + NH + 1, d))
        nn.init.normal_(self.type_emb, std=0.02)
        self.cls = nn.Parameter(torch.zeros(1, 1, d))
        self.blocks = nn.ModuleList(Block(d, h, ff, p) for _ in range(layers))
        self.ln = nn.LayerNorm(d)
        self.gate_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.value_head = nn.Sequential(nn.Linear(d, d), nn.GELU(), nn.Linear(d, 1))
        self.slot_head = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, 1))
        self.tile_query = nn.Sequential(nn.Linear(2 * d, d), nn.GELU(), nn.Linear(d, d))
        gy, gx = torch.meshgrid(torch.arange(GH), torch.arange(GW), indexing="ij")
        pos = torch.stack([gx.flatten() / (GW - 1), gy.flatten() / (GH - 1)], 1).float()
        freqs = torch.tensor([1.0, 2.0, 4.0, 8.0]) * math.pi
        four = torch.cat([torch.sin(pos[:, :, None] * freqs), torch.cos(pos[:, :, None] * freqs)], -1).flatten(1)
        self.register_buffer("tile_feats", torch.cat([pos, four], 1))
        self.tile_pos = nn.Linear(self.tile_feats.shape[1], d)
        self.tile_free = nn.Parameter(torch.zeros(NUM_TILES, d))
        self.tile_bias = nn.Parameter(torch.zeros(NUM_TILES))

    def forward(self, obs):
        B = obs.shape[0]
        g = self.glob(obs[:, :self.G_END])[:, None]
        hand = obs[:, self.H0:self.U0].reshape(B, NH, HF)
        idx = torch.clamp(torch.round(hand[:, :, 0] * (self.V + 1)), 0, self.V).long()
        h = self.hand_in(hand) + self.card_emb(idx)
        units = obs[:, self.U0:].reshape(B, NU, UF)
        u = self.unit_in(units)
        te = self.type_emb
        x = torch.cat([self.cls.expand(B, 1, -1) + te[0], g + te[1], h + te[2:2 + NH], u + te[2 + NH]], 1)
        pad = torch.cat([torch.zeros(B, 2 + NH, device=obs.device), (units[:, :, 0] <= 0).float()], 1)
        key_bias = (pad * -1e4)[:, None, None, :]
        for blk in self.blocks:
            x = blk(x, key_bias)
        x = self.ln(x)
        c, hs = x[:, 0], x[:, 2:6]
        ch = torch.cat([c[:, None].expand(B, 4, -1), hs], -1)
        gate = self.gate_head(c)[:, 0]
        slot = self.slot_head(ch)[:, :, 0]
        q = self.tile_query(ch)
        tiles = self.tile_pos(self.tile_feats) + self.tile_free
        tile = q @ tiles.t() / math.sqrt(q.shape[-1]) + self.tile_bias
        value = torch.sigmoid(self.value_head(c)[:, 0])
        return gate, slot, tile, value


# -- ONNX <-> PyTorch ------------------------------------------------------------------------------------------
def _scope_to_param(node_name: str) -> str:
    """Exporter scope -> module path: each child scope repeats its parent's local name
    ('/blocks.0/ff/ff.0/MatMul' -> 'blocks.0.ff.0', '/unit_in/unit_in.0/MatMul' -> 'unit_in.0')."""
    parts = [p for p in node_name.split("/") if p][:-1]
    path = parts[0]
    for p in parts[1:]:
        head, _, last = path.rpartition(".") if "." in path and not path.split(".")[-1].isdigit() else ("", "", path)
        if p.startswith(last + "."):
            path = (head + "." if head else "") + p
        else:
            path = path + "." + p
    return path


def state_dict_from_onnx(path: Path | str) -> Tuple[Dict[str, torch.Tensor], int]:
    """(state_dict, vocab_size) rebuilt from an ONNX file exported by the IL notebook / export_onnx."""
    import onnx
    from onnx import numpy_helper
    m = onnx.load(str(path))
    inits = {t.name: numpy_helper.to_array(t) for t in m.graph.initializer}
    sd: Dict[str, torch.Tensor] = {}
    for name, arr in inits.items():
        if not name.startswith("onnx::"):
            sd[name] = torch.from_numpy(np.array(arr))
    type_rows: Dict[int, np.ndarray] = {}
    for node in m.graph.node:
        for inp in node.input:
            if not inp.startswith("onnx::") or inp not in inits:
                continue
            arr = inits[inp]
            if node.op_type == "MatMul":                    # /blocks.0/ff/ff.0/MatMul -> blocks.0.ff.0.weight
                sd[_scope_to_param(node.name) + ".weight"] = torch.from_numpy(np.array(arr.T))
            elif node.op_type == "Add" and node.name.startswith("/Add_"):
                type_rows[int(node.name.split("_")[1])] = arr  # /Add_1..4: type_emb[0], [1], [2:7], [7]
    if sorted(type_rows) != [1, 2, 3, 4]:
        raise ValueError(f"{path}: unexpected type-embedding constants {sorted(type_rows)}")
    sd["type_emb"] = torch.from_numpy(np.concatenate([type_rows[1][None], type_rows[2][None], type_rows[3],
                                                      type_rows[4][None]], 0).astype(np.float32))
    vocab_size = int(sd["card_emb.weight"].shape[0]) - 1
    return sd, vocab_size


def load_onnx_weights(path: Path | str, verify: bool = True, atol: float = 1e-4) -> CRbotIL:
    sd, vocab = state_dict_from_onnx(path)
    d = int(sd["cls"].shape[-1])
    layers = len({k.split(".")[1] for k in sd if k.startswith("blocks.")})
    ff = int(sd["blocks.0.ff.0.bias"].shape[0])
    model = CRbotIL(vocab, d=d, h=ARCH["h"], layers=layers, ff=ff)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise ValueError(f"{path}: weights don't match the architecture (missing {missing}, unexpected {unexpected})")
    model.eval()
    if verify:
        check_against_onnx(model, path, atol)
    return model


def _probe(obs_dim: int, n: int = 16, seed: int = 0) -> np.ndarray:
    """Plausible random observations (present units, hand cards) to compare two implementations."""
    rng = np.random.default_rng(seed)
    x = rng.random((n, obs_dim)).astype(np.float32)
    return x


def check_against_onnx(model: CRbotIL, path: Path | str, atol: float = 1e-4) -> float:
    import onnxruntime as ort
    x = _probe(model.OBS_DIM)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ref = sess.run(list(OUTPUTS), {sess.get_inputs()[0].name: x})
    with torch.no_grad():
        out = [t.numpy() for t in model(torch.from_numpy(x))]
    diff = max(float(np.abs(a - b).max()) for a, b in zip(out, ref))
    if diff > atol:
        raise ValueError(f"{path}: rebuilt PyTorch model differs from ONNX by {diff:.2e}")
    return diff


def export_onnx(model: CRbotIL, path: Path | str, verify: bool = True) -> float:
    """Export with the inputs / outputs NeuralBot expects; returns the max |torch - onnxruntime|."""
    model = model.eval()
    sample = torch.from_numpy(_probe(model.OBS_DIM, 2))
    kw = dict(input_names=["obs"], output_names=list(OUTPUTS),
              dynamic_axes={n: {0: "batch"} for n in ("obs", *OUTPUTS)}, opset_version=17)
    with torch.no_grad():
        try:
            torch.onnx.export(model, (sample,), str(path), dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(model, (sample,), str(path), **kw)
    return check_against_onnx(model, path) if verify else 0.0
