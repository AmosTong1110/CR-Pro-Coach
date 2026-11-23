"""Roster-independent IL helpers shared by the observation encoder (obs_encoder.py), NeuralBot, the PPO trainer
and the dataset builder: acting-side frames, tile indices, legal-slot / legal-tile masks, mask packing.

Perspective: everything is in the acting side's own frame ("my side at the bottom", attack towards +y), using
the engine convention `geometry.to_side` (mirror y only). Side 0 is unchanged; for side 1, y -> 32 - y.

Action space: gate (0 wait / 1 play), card slot 0..3 (masked by `slot_mask`), tile 0..575 = gy * 18 + gx in the
acting side's frame (masked by `tile_mask` for the chosen card).

The observation itself is built by obs_encoder.encode_observation (il_obs_v2). The old card-ID encoder of this
module (il_obs_v1: 333 / 341 floats, one in-hand probability per card of the pool) was removed on 2026-11-17;
models trained on it (crbot_il_v1, crbot_ppo_v1) are reported as legacy by NeuralBot.
"""
from __future__ import annotations

import math
from typing import Dict, Mapping, Tuple

import numpy as np

from backend_engine.contracts import ElixirEstimate
from backend_engine.simulator import geometry as G
from backend_engine.simulator.card_db import CardDB

HAND_SLOTS = 4
GRID_W, GRID_H = int(G.W), int(G.H)
NUM_TILES = GRID_W * GRID_H                      # 576
TOWER_ROLES = ("king", "princess_left", "princess_right")
MAX_TIME_MS = 300_000.0
TOP_GUESS_CARDS = 12                             # same merge as MatchRunner (guesser cards added to in_hand_prob)


def frame_xy(side: int, x: float, y: float) -> Tuple[float, float]:
    """World -> acting side's frame (the mapping is its own inverse)."""
    return G.to_side(side, x, y)


def tile_index(side: int, x: float, y: float) -> int:
    """World position -> tile index in the acting side's frame."""
    fx, fy = frame_xy(side, x, y)
    gx = min(max(int(math.floor(fx)), 0), GRID_W - 1)
    gy = min(max(int(math.floor(fy)), 0), GRID_H - 1)
    return gy * GRID_W + gx


def tile_center(side: int, tile: int) -> Tuple[float, float]:
    """Tile index in the acting side's frame -> world coordinates of the tile centre."""
    gy, gx = divmod(int(tile), GRID_W)
    return frame_xy(side, gx + 0.5, gy + 0.5)


def merged_hand_probs(est: ElixirEstimate, guess) -> Dict[str, float]:
    """The tracker's in-hand probabilities plus the deck guesser's top cards for unknown slots: exactly what
    MatchRunner hands to agents (idempotent, so an already merged estimate gives the same result)."""
    if guess is None:
        return dict(est.in_hand_prob)
    extra = dict(list(guess.in_hand_probs(est.unknown_hand_slots).items())[:TOP_GUESS_CARDS])
    return {**extra, **est.in_hand_prob}


def slot_mask(obs: Mapping, db: CardDB) -> np.ndarray:
    """bool[4]: hand slot holds a card the side can afford now."""
    m = np.zeros(HAND_SLOTS, dtype=bool)
    for k, c in enumerate(obs["hand"][:HAND_SLOTS]):
        m[k] = c in db.cards and db.cards[c]["elixir"] <= obs["elixir"] + 1e-9
    return m


# tile centres of the acting side's frame, as world coordinates (per side)
_GX, _GY = np.meshgrid(np.arange(GRID_W) + 0.5, np.arange(GRID_H) + 0.5)
_FX, _FY = _GX.ravel(), _GY.ravel()                      # index = gy * 18 + gx


def tile_mask(obs: Mapping, card_id: str, db: CardDB) -> np.ndarray:
    """bool[576] in the acting side's frame: tiles where `card_id` may be deployed now (same rules as
    Match.check_play: spells anywhere, burrowers anywhere on the ground, otherwise own half / pocket, never the
    river, never on a structure; buildings: own half only, their footprint must stay inside the arena, off the
    river and off other structures). Elixir and hand are not checked here (see slot_mask)."""
    side = int(obs["side"])
    card = db.cards[card_id]
    wx = _FX
    wy = _FY if side == 0 else G.H - _FY
    if card["type"] == "spell":
        return np.ones(NUM_TILES, dtype=bool)
    building = card["type"] == "building"
    if building and "footprint_tiles" in card:
        rad = card["footprint_tiles"][0] / 2.0
    else:
        rad = float(card.get("collision_radius_tiles", 0.5))
    burrow = any(m["kind"] == "burrow" for m in card["mechanics"])
    blocked = np.zeros(NUM_TILES, dtype=bool)
    for u in obs["units"]:
        if u["kind"] in ("tower", "building") and u["hp"] > 0:
            # burrowers (Miner, Goblin Drill) only may not surface inside a structure (Match.check_play)
            r = (rad + u["radius"] - 1e-4) if building and not burrow else u["radius"]
            blocked |= (np.abs(wx - u["x"]) < r) & (np.abs(wy - u["y"]) < r)
    river = (wy >= G.RIVER_Y0) & (wy < G.RIVER_Y1)
    if burrow:
        on_bridge = np.zeros(NUM_TILES, dtype=bool)
        for b in G.BRIDGE_X:
            on_bridge |= np.abs(wx - b) <= G.BRIDGE_HALF_W
        return ~(river & ~on_bridge) & ~blocked
    home = (wy < G.RIVER_Y0) if side == 0 else (wy >= G.RIVER_Y1)
    if building:                                     # geometry.deploy_check's building rules
        inside = (wx - rad >= 0.0) & (wx + rad <= G.W) & (wy - rad >= 0.0) & (wy + rad <= G.H)
        off_river = (wy + rad <= G.RIVER_Y0) if side == 0 else (wy - rad >= G.RIVER_Y1)
        return home & inside & off_river & ~river & ~blocked
    alive = {role: False for role in ("princess_left", "princess_right")}
    for u in obs["units"]:
        if u["kind"] == "tower" and u["side"] != side and u["hp"] > 0 and u["tower_role"] in alive:
            alive[u["tower_role"]] = True
    depth = (wy - G.RIVER_Y1) if side == 0 else (G.RIVER_Y0 - wy)
    left = wx < G.W / 2
    lane_open = np.where(left, not alive["princess_left"], not alive["princess_right"])
    pocket = (depth >= 0) & (depth < G.POCKET_DEPTH) & lane_open
    return (home | pocket) & ~river & ~blocked


def pack_mask(mask: np.ndarray) -> np.ndarray:
    """bool[576] -> uint8[72] (shards store tile masks packed)."""
    return np.packbits(mask.astype(bool))


def unpack_mask(packed: np.ndarray) -> np.ndarray:
    return np.unpackbits(packed, axis=-1)[..., :NUM_TILES].astype(bool)
