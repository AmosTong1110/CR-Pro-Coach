"""Imitation-learning features shared by the dataset builder (tools/build_il_dataset.py) and NeuralBot.

Everything here is computed from what a human player can see: `Match.observation(side, is_visible)` (fog of war
through the reaction-delay buffer), the side's own `OppElixirTracker` estimate and the Bayesian deck guess. No
hidden opponent state (hand, queue, real elixir) is ever read.

Perspective: every vector is in the acting side's own frame ("my side at the bottom", attack towards +y), using
the engine convention `geometry.to_side` (mirror y only). Side 0 is unchanged; for side 1, y -> 32 - y. The same
replay therefore gives training samples for both players, and NeuralBot can play either side.

Observation layout (OBS_DIM = 333 float32), version OBS_VERSION:
    [0:8]     global     t, time left, elixir multiplier, overtime, own crowns, opp crowns, crown diff, own elixir
    [8:20]    towers     own king / left / right, then opponent's: (hp fraction, active)
    [20:63]   opponent   elixir point / low / high, known cards, 38 in-hand probabilities (vocab order), deck confidence
    [63:93]   hand       4 hand slots + next card: (vocab index, cost, affordable, evo slot, evo ready, type)
    [93:333]  units      30 nearest-to-river non-tower units: (present, own +1 / enemy -1, x, y, hp, cost,
                         deploying, air)

Action space: gate (0 wait / 1 play), card slot 0..3 (masked by `slot_mask`), tile 0..575 = gy * 18 + gx in the
acting side's frame (masked by `tile_mask` for the chosen card).
"""
from __future__ import annotations

import math
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from backend_engine.contracts import ElixirEstimate
from backend_engine.simulator import geometry as G
from backend_engine.simulator.card_db import CardDB

OBS_VERSION = "il_obs_v1"
MAX_VISIBLE_UNITS = 30
UNIT_FEAT_DIM = 8
HAND_SLOTS = 4
HAND_FEAT_DIM = 6
GRID_W, GRID_H = int(G.W), int(G.H)
NUM_TILES = GRID_W * GRID_H                      # 576
N_GLOBAL, N_TOWER = 8, 12
N_HAND = (HAND_SLOTS + 1) * HAND_FEAT_DIM        # 30
N_UNITS = MAX_VISIBLE_UNITS * UNIT_FEAT_DIM      # 240
TOWER_ROLES = ("king", "princess_left", "princess_right")
MAX_TIME_MS = 300_000.0
TOP_GUESS_CARDS = 12                             # same merge as MatchRunner (guesser cards added to in_hand_prob)


def card_vocab(db: CardDB) -> List[str]:
    """Sorted ids of the simulated cards (the order of the in-hand probabilities and the card index feature)."""
    return sorted(db.cards)


def obs_dim(vocab: Sequence[str]) -> int:
    return N_GLOBAL + N_TOWER + (4 + len(vocab) + 1) + N_HAND + N_UNITS


OBS_DIM = obs_dim(card_vocab(CardDB.load()))     # 333 with the 38-card pool


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


def _cost(db: CardDB, card_id: str) -> float:
    try:
        return float(db.entry(card_id).get("elixir", 3))
    except KeyError:
        return 3.0


def encode_observation(obs: Mapping, est: ElixirEstimate, guess, db: CardDB, vocab: Sequence[str]) -> np.ndarray:
    """Build the OBS_DIM float32 vector for `obs["side"]` (see the module docstring for the layout)."""
    side = int(obs["side"])
    t = float(obs["t_ms"])
    own_crowns, opp_crowns = obs["crowns"]
    elixir = float(obs["elixir"])
    out = np.zeros(obs_dim(vocab), dtype=np.float32)

    # global
    out[0:8] = (min(t / MAX_TIME_MS, 1.0), min(max(obs["time_left_ms"], 0) / MAX_TIME_MS, 1.0),
                obs["multiplier"] / 3.0, 1.0 if obs["overtime"] else 0.0, own_crowns / 3.0, opp_crowns / 3.0,
                (own_crowns - opp_crowns) / 3.0, min(max(elixir, 0.0), 10.0) / 10.0)

    # towers (own first)
    towers = {(u["side"], u["tower_role"]): u for u in obs["units"] if u["kind"] == "tower" and u["hp"] > 0}
    i = N_GLOBAL
    for s in (side, 1 - side):
        for role in TOWER_ROLES:
            u = towers.get((s, role))
            if u is not None:
                out[i], out[i + 1] = u["hp"] / max(u["max_hp"], 1.0), 1.0 if u["active"] else 0.0
            i += 2

    # opponent perception (human-legal tracker + deck guess)
    probs = merged_hand_probs(est, guess)
    out[i:i + 4] = (est.point / 10.0, est.low / 10.0, est.high / 10.0, len(est.known_cards) / 8.0)
    i += 4
    out[i:i + len(vocab)] = [min(float(probs.get(c, 0.0)), 1.0) for c in vocab]
    i += len(vocab)
    if guess is not None and getattr(guess, "likely_decks", None):
        out[i] = float(guess.likely_decks[0][1])
    i += 1

    # own hand + next card
    vidx = {c: k for k, c in enumerate(vocab)}
    evo_counter = obs.get("evo_counter") or {}
    cards = list(obs["hand"][:HAND_SLOTS]) + [""] * (HAND_SLOTS - len(obs["hand"][:HAND_SLOTS]))
    cards.append(obs.get("next_card") or "")
    for c in cards:
        if c and c in db.cards:
            meta = db.cards[c]
            cost = float(meta["elixir"])
            is_evo = c in evo_counter
            ready = is_evo and evo_counter[c] >= db.evo_cycles(c)
            kind = meta.get("type", "troop")
            out[i:i + HAND_FEAT_DIM] = ((vidx.get(c, -1) + 1.0) / (len(vocab) + 1.0), cost / 10.0,
                                        1.0 if elixir + 1e-9 >= cost else 0.0, 1.0 if is_evo else 0.0,
                                        1.0 if ready else 0.0,
                                        0.0 if kind == "troop" else (0.5 if kind == "building" else 1.0))
        i += HAND_FEAT_DIM

    # visible units (no towers), nearest to the river first
    units = [(u, *frame_xy(side, u["x"], u["y"])) for u in obs["units"] if u["kind"] != "tower" and u["hp"] > 0]
    # nearest to the river first; ties broken the same way from either side (own first, then frame x / y)
    units.sort(key=lambda r: (abs(r[2] - G.H / 2), r[0]["side"] != side, r[1], r[2], r[0]["uid"]))
    for u, fx, fy in units[:MAX_VISIBLE_UNITS]:
        out[i:i + UNIT_FEAT_DIM] = (1.0, 1.0 if u["side"] == side else -1.0, min(max(fx / G.W, 0.0), 1.0),
                                    min(max(fy / G.H, 0.0), 1.0), min(max(u["hp"] / max(u["max_hp"], 1.0), 0.0), 1.0),
                                    _cost(db, u["card_id"]) / 10.0, 1.0 if u["deploying"] else 0.0,
                                    1.0 if u["layer"] == "air" else 0.0)
        i += UNIT_FEAT_DIM
    return out


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
    river, never on a structure). Elixir and hand are not checked here (see slot_mask)."""
    side = int(obs["side"])
    card = db.cards[card_id]
    wx = _FX
    wy = _FY if side == 0 else G.H - _FY
    if card["type"] == "spell":
        return np.ones(NUM_TILES, dtype=bool)
    blocked = np.zeros(NUM_TILES, dtype=bool)
    for u in obs["units"]:
        if u["kind"] in ("tower", "building") and u["hp"] > 0:
            r = u["radius"]
            blocked |= (np.abs(wx - u["x"]) < r) & (np.abs(wy - u["y"]) < r)
    river = (wy >= G.RIVER_Y0) & (wy < G.RIVER_Y1)
    if any(m["kind"] == "burrow" for m in card["mechanics"]):
        on_bridge = np.zeros(NUM_TILES, dtype=bool)
        for b in G.BRIDGE_X:
            on_bridge |= np.abs(wx - b) <= G.BRIDGE_HALF_W
        return ~(river & ~on_bridge) & ~blocked
    home = (wy < G.RIVER_Y0) if side == 0 else (wy >= G.RIVER_Y1)
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
