"""The crbot policy / value network in PyTorch, for il_obs_v2 (backend_engine/brain/obs_encoder.py).

Used by cloud_training/train_imitation_colab.ipynb (training), the PPO trainer and anything that needs gradients.
The app never imports this module: NeuralBot runs the exported ONNX file with onnxruntime. Needs `torch`.

`CRbotIL` is a token transformer whose size does not depend on the card pool: every card-like token (opponent
hand candidates, own hand, units, spells) is projected from the same fixed attribute vector by one shared
`attr_in` layer, plus a small per-group projection of its dynamic features (position, hp, affordability, ...).
There is no card-id embedding, so a new card is just a new attribute vector.

    tokens: cls, global (global + towers + tower troops + opponent summary), 8 opponent cards, 5 hand
            (4 slots + next), 30 units, 4 spells  ->  49 tokens, padded ones masked out of attention
    heads:  gate [B] (play logit), slot [B,4], tile [B,4,576], value [B] (sigmoid)

Weights travel as `<name>.onnx` (inference) + `<name>.npz` (plain arrays, no pickle: `save_weights` /
`load_weights`, checked against the ONNX file by `check_against_onnx`).
"""
from __future__ import annotations

import math
from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn as nn

from backend_engine.brain import obs_encoder as E

NUM_TILES = E.NUM_TILES
GW, GH = E.GRID_W, E.GRID_H
OUTPUTS = ("gate_logit", "slot_logits", "tile_logits", "value")
ARCH = dict(d=256, h=8, layers=4, ff=512)
G_END = E.LAYOUT["opp_summary"][1]                     # global token input: everything before opp_cards
GROUPS = (("opp", "opp_cards", E.N_OPP_CARDS, E.OPP_TOK, E.OPP_DYN),
          ("hand", "hand", E.HAND_TOKENS, E.HAND_TOK, E.HAND_DYN),
          ("unit", "units", E.MAX_UNITS, E.UNIT_TOK, E.UNIT_DYN),
          ("spell", "spells", E.MAX_SPELLS, E.SPELL_TOK, E.SPELL_DYN))
HAND_POS = 2 + E.N_OPP_CARDS                           # index of hand slot 0 in the token sequence


class Block(nn.Module):
    def __init__(self, d, h, ff, p):
        super().__init__()
        self.h, self.dk = h, d // h
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, ff), nn.GELU(), nn.Linear(ff, d))
        self.drop = nn.Dropout(p)

    def forward(self, x, key_bias):                    # key_bias [B, 1, 1, S]: 0 or -1e4
        B, S, D = x.shape
        q, k, v = self.qkv(self.ln1(x)).reshape(B, S, 3, self.h, self.dk).permute(2, 0, 3, 1, 4)
        att = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dk) + key_bias, dim=-1)
        y = (self.drop(att) @ v).transpose(1, 2).reshape(B, S, D)
        x = x + self.drop(self.proj(y))
        return x + self.drop(self.ff(self.ln2(x)))


class CRbotIL(nn.Module):
    def __init__(self, d=256, h=8, layers=4, ff=512, p=0.0):
        super().__init__()
        self.OBS_DIM = E.OBS_DIM
        self.glob = nn.Sequential(nn.Linear(G_END, d), nn.GELU(), nn.Linear(d, d))
        self.attr_in = nn.Sequential(nn.Linear(E.N_ATTR, d), nn.GELU(), nn.Linear(d, d))   # shared by all cards
        self.dyn_in = nn.ModuleDict({g: nn.Linear(dyn, d) for g, _, _, _, dyn in GROUPS})
        # type rows: cls, global, opp, hand slot 0..4, unit, spell
        self.type_emb = nn.Parameter(torch.zeros(2 + 1 + E.HAND_TOKENS + 2, d))
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

    def _group(self, obs, name, block, n, tok, dyn):
        a, b = E.LAYOUT[block]
        x = obs[:, a:b].reshape(obs.shape[0], n, tok)
        return self.dyn_in[name](x[:, :, :dyn]) + self.attr_in(x[:, :, dyn:]), x[:, :, 0]

    def forward(self, obs):
        B = obs.shape[0]
        te = self.type_emb
        g = self.glob(obs[:, :G_END])[:, None] + te[1]
        opp, opp_p = self._group(obs, *GROUPS[0])
        hand, _ = self._group(obs, *GROUPS[1])
        unit, unit_p = self._group(obs, *GROUPS[2])
        spell, spell_p = self._group(obs, *GROUPS[3])
        hand_te = te[3:3 + E.HAND_TOKENS]
        x = torch.cat([self.cls.expand(B, 1, -1) + te[0], g, opp + te[2], hand + hand_te,
                       unit + te[3 + E.HAND_TOKENS], spell + te[4 + E.HAND_TOKENS]], 1)
        always = torch.zeros(B, 2, device=obs.device)
        pad = torch.cat([always, (opp_p <= 0).float(), torch.zeros(B, E.HAND_TOKENS, device=obs.device),
                         (unit_p <= 0).float(), (spell_p <= 0).float()], 1)
        key_bias = (pad * -1e4)[:, None, None, :]
        for blk in self.blocks:
            x = blk(x, key_bias)
        x = self.ln(x)
        c, hs = x[:, 0], x[:, HAND_POS:HAND_POS + 4]
        ch = torch.cat([c[:, None].expand(B, 4, -1), hs], -1)
        gate = self.gate_head(c)[:, 0]
        slot = self.slot_head(ch)[:, :, 0]
        q = self.tile_query(ch)
        tiles = self.tile_pos(self.tile_feats) + self.tile_free
        tile = q @ tiles.t() / math.sqrt(q.shape[-1]) + self.tile_bias
        value = torch.sigmoid(self.value_head(c)[:, 0])
        return gate, slot, tile, value


def arch_of(model: CRbotIL) -> Dict[str, int]:
    return {"d": int(model.cls.shape[-1]), "h": int(model.blocks[0].h), "layers": len(model.blocks),
            "ff": int(model.blocks[0].ff[0].out_features)}


# -- training augmentation (the notebook and the PPO trainer use this, so it always matches the layout) ----------
_MIRROR = E.mirror_spec()


def augment(obs, slot, tile, smask, tmask, mirror: bool = True, hand_perm: bool = True):
    """Random left-right mirror (towers swap, x -> 1 - x, tiles mirrored) and hand-slot permutation."""
    B, dev = obs.shape[0], obs.device
    if mirror:
        perm, xs, tmir = (torch.as_tensor(a, device=dev) for a in _MIRROR)
        flip = torch.rand(B, device=dev) < 0.5
        o = obs[:, perm].clone()
        present = o[:, xs - 2] > 0
        o[:, xs] = torch.where(present, 1.0 - o[:, xs], o[:, xs])
        obs = torch.where(flip[:, None], o, obs)
        tile = torch.where(flip & (tile >= 0), tmir[tile.clamp(min=0)], tile)
        tmask = torch.where(flip[:, None], tmask[:, tmir], tmask)
    if hand_perm:
        h0, tok = E.LAYOUT["hand"][0], E.HAND_TOK
        perm = torch.argsort(torch.rand(B, 4, device=dev), dim=1)          # new slot j shows old slot perm[j]
        hand = obs[:, h0:h0 + 4 * tok].reshape(B, 4, tok)
        hand = torch.gather(hand, 1, perm[:, :, None].expand(B, 4, tok))
        obs = obs.clone()
        obs[:, h0:h0 + 4 * tok] = hand.reshape(B, -1)
        smask = torch.gather(smask, 1, perm)
        inv = torch.argsort(perm, dim=1)
        slot = torch.where(slot >= 0, torch.gather(inv, 1, slot.clamp(min=0)[:, None])[:, 0], slot)
    return obs, slot, tile, smask, tmask


# -- weights: .npz (no pickle) <-> PyTorch, ONNX export / check -------------------------------------------------
def save_weights(model: CRbotIL, path: Path | str) -> None:
    arrays = {k: v.detach().cpu().numpy() for k, v in model.state_dict().items()}
    arrays["__arch__"] = np.array([arch_of(model)[k] for k in ("d", "h", "layers", "ff")], dtype=np.int64)
    np.savez(str(path), **arrays)


def load_weights(path: Path | str) -> CRbotIL:
    with np.load(str(path), allow_pickle=False) as z:
        d, h, layers, ff = (int(v) for v in z["__arch__"])
        sd = {k: torch.from_numpy(np.array(z[k])) for k in z.files if k != "__arch__"}
    model = CRbotIL(d=d, h=h, layers=layers, ff=ff)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        raise ValueError(f"{path}: weights don't match the architecture (missing {missing}, unexpected {unexpected})")
    return model.eval()


def _probe(n: int = 16, seed: int = 0) -> np.ndarray:
    """Random observations with some present tokens, to compare two implementations."""
    rng = np.random.default_rng(seed)
    return rng.random((n, E.OBS_DIM)).astype(np.float32)


def check_against_onnx(model: CRbotIL, path: Path | str, atol: float = 1e-4) -> float:
    import onnxruntime as ort
    x = _probe()
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    ref = sess.run(list(OUTPUTS), {sess.get_inputs()[0].name: x})
    with torch.no_grad():
        out = [t.numpy() for t in model.eval()(torch.from_numpy(x))]
    diff = max(float(np.abs(a - b).max()) for a, b in zip(out, ref))
    if diff > atol:
        raise ValueError(f"{path}: PyTorch model differs from ONNX by {diff:.2e}")
    return diff


def export_onnx(model: CRbotIL, path: Path | str, verify: bool = True) -> float:
    """Export with the inputs / outputs NeuralBot expects; returns the max |torch - onnxruntime|."""
    model = model.eval()
    sample = torch.from_numpy(_probe(2))
    kw = dict(input_names=["obs"], output_names=list(OUTPUTS),
              dynamic_axes={n: {0: "batch"} for n in ("obs", *OUTPUTS)}, opset_version=17)
    with torch.no_grad():
        try:
            torch.onnx.export(model, (sample,), str(path), dynamo=False, **kw)
        except TypeError:
            torch.onnx.export(model, (sample,), str(path), **kw)
    return check_against_onnx(model, path) if verify else 0.0


def model_meta(name: str, **extra) -> dict:
    """The .json NeuralBot reads: obs schema (version, dim, attribute keys, layout) + outputs + extras."""
    return {"name": name, **E.schema(),
            "outputs": {"gate_logit": "[B] play logit", "slot_logits": "[B,4]",
                        "tile_logits": "[B,4,576] gy*18+gx in the acting side's frame", "value": "[B] win probability"},
            **extra}
