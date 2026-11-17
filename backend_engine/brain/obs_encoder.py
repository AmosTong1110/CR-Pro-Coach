"""Fixed-dimension observation encoder (OBS_VERSION il_obs_v2): roster-size independent.

Why: il_obs_v1 (il_features.encode_observation) had one in-hand probability per card of the pool and a card-index
feature per hand slot, so its size (333 floats with 38 cards, 341 with 46, ...) and meaning changed whenever a card
was added. Every new card broke the trained model, the shards and the tests.

il_obs_v2 never names a card. Every card a side holds, the opponent might hold, every unit and every spell on the
board is described by the same fixed-length **physical attribute vector** (ATTRIBUTE_KEYS) computed from the card
data at tournament level 11: cost, hp, dps, range, speed, targeting, splash, lifetime, spell radius, shield,
spawns, ... An evolution or hero is the same card with different stats and the `is_evolved` / `is_hero` flag, not
a new id. Towers get their own attribute block (TOWER_ATTRIBUTE_KEYS), so Tower Troops (Cannoneer, Dagger
Duchess, ...) also fit without a new dimension.

    OBS_DIM is a constant (1752). Adding card #47 or #127 changes values, never the layout.

Scaling is smooth division (no clipping), so a future card with 9,000 hp or a 9-elixir cost is still encoded
faithfully (values above 1 are fine for the network).

Layout (acting side's frame, my side at the bottom; see LAYOUT for exact offsets):
    global          8   t, time left, multiplier, overtime, own / opp crowns, crown diff, own elixir
    towers         12   own then opponent king / left / right: (hp fraction, active)
    tower_troops   28   own then opponent king / princess: TOWER_ATTRIBUTE_KEYS (7)
    opp_summary     5   opponent elixir point / low / high, known cards / 8, deck-guess confidence
    opp_cards     248   8 most likely opponent hand cards: (probability, attributes)       sorted by -p, id
    hand          175   4 slots + next card: (present, affordable, evo slot, evo ready, elixir gap, attributes)
    units        1140   30 units nearest the river: (present, own +1 / enemy -1, x, y, hp fraction, deploying,
                        air now, evolved, attributes)
    spells        136   4 active spells / areas: (present, own +1 / enemy -1, x, y, attributes)

Actions are unchanged and roster independent: gate (wait / play), hand slot 0..3, tile 0..575 (il_features).
"""
from __future__ import annotations

from typing import Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from backend_engine.brain.il_features import (  # noqa: F401  (re-exported: same masks / frames as v1)
    GRID_H, GRID_W, HAND_SLOTS, MAX_TIME_MS, NUM_TILES, TOWER_ROLES, frame_xy, merged_hand_probs, pack_mask,
    slot_mask, tile_center, tile_index, tile_mask, unpack_mask,
)
from backend_engine.contracts import ElixirEstimate
from backend_engine.simulator import geometry as G
from backend_engine.simulator.card_db import CardDataError, CardDB
from backend_engine.simulator.traits import card_traits

OBS_VERSION = "il_obs_v2"
LEGACY_OBS_VERSIONS = ("il_obs_v1",)          # card-id based (333 / 341 floats): needs retraining
ATTR_LEVEL = 11                               # tournament standard

ATTRIBUTE_KEYS: Tuple[str, ...] = (
    "elixir",             # cost / 10
    "hitpoints",          # hp of one unit / 3000
    "dps",                # damage per second / 300 (dot spells: per pulse rate)
    "damage",             # damage per hit or per cast / 1000
    "hit_speed",          # seconds between hits
    "range",              # attack range in tiles / 10
    "move_speed",         # game speed units (60 = 1 tile/s) / 100
    "is_air",
    "targets_air",
    "targets_ground",
    "building_only",
    "splash_radius",      # tiles / 5
    "is_building",
    "lifetime",           # building lifetime or spell duration, seconds / 60
    "is_spell",
    "spell_radius",       # tiles / 5 (rolling spells: half their width)
    "spell_reach",        # rolling distance in tiles / 10
    "stun_or_knockback",
    "shield_hp",          # / 1000
    "spawn_count",        # units deployed by one play / 10
    "death_spawn_count",  # units spawned on death / at the end / 5
    "spawn_rate",         # periodic spawns: units per second
    "death_damage",       # / 1000
    "tower_damage",       # damage per hit or cast on a crown tower / 1000
    "deploy_time",        # seconds
    "size",               # collision radius (buildings: half the footprint), tiles
    "kamikaze",
    "has_ability",
    "is_evolved",
    "is_hero",
)
TOWER_ATTRIBUTE_KEYS: Tuple[str, ...] = (
    "hitpoints",          # / 5000
    "dps",                # / 300
    "damage",             # / 1000
    "hit_speed",          # seconds
    "range",              # tiles / 10
    "splash_radius",      # tiles / 5
    "targets_air",
)
N_ATTR, N_TATTR = len(ATTRIBUTE_KEYS), len(TOWER_ATTRIBUTE_KEYS)

N_OPP_CARDS = 8
MAX_UNITS = 30
MAX_SPELLS = 4
HAND_TOKENS = HAND_SLOTS + 1
OPP_DYN, HAND_DYN, UNIT_DYN, SPELL_DYN = 1, 5, 8, 4
OPP_TOK, HAND_TOK, UNIT_TOK, SPELL_TOK = (OPP_DYN + N_ATTR, HAND_DYN + N_ATTR, UNIT_DYN + N_ATTR,
                                          SPELL_DYN + N_ATTR)
TOWER_KINDS = ("king", "princess")

_SIZES = (("global", 8), ("towers", 12), ("tower_troops", 2 * len(TOWER_KINDS) * N_TATTR), ("opp_summary", 5),
          ("opp_cards", N_OPP_CARDS * OPP_TOK), ("hand", HAND_TOKENS * HAND_TOK), ("units", MAX_UNITS * UNIT_TOK),
          ("spells", MAX_SPELLS * SPELL_TOK))
LAYOUT: Dict[str, Tuple[int, int]] = {}
_o = 0
for _name, _n in _SIZES:
    LAYOUT[_name] = (_o, _o + _n)
    _o += _n
OBS_DIM = _o                                  # 1752, whatever the size of the card pool
del _o, _name, _n


def schema() -> dict:
    """What a model trained on these features must record in its .json (checked by NeuralBot on load)."""
    return {"obs_version": OBS_VERSION, "obs_dim": OBS_DIM, "attribute_keys": list(ATTRIBUTE_KEYS),
            "tower_attribute_keys": list(TOWER_ATTRIBUTE_KEYS), "attr_level": ATTR_LEVEL,
            "layout": {k: list(v) for k, v in LAYOUT.items()}}


# -- attribute vectors -------------------------------------------------------------------------------------------
def _record(db: CardDB, card_id: str) -> Optional[dict]:
    return db.catalog.get(card_id) or db.tokens.get(card_id)


def _stats(db: CardDB, card_id: str, variant: str) -> Tuple[Dict[str, float], dict, dict, dict]:
    """(scaled stats, attack, movement, spell) at ATTR_LEVEL; raw base numbers if the card can't be scaled."""
    rec = _record(db, card_id)
    try:
        s = db.stats(card_id, ATTR_LEVEL, variant if variant in ("evo", "hero") else "base")
        return dict(s.stats), s.attack, s.movement, s.spell
    except (CardDataError, KeyError, ValueError):
        raw = dict(rec.get("base_stats") or rec.get("stats_l11") or {})
        return raw, dict(rec.get("attack") or {}), dict(rec.get("movement") or {}), dict(rec.get("spell") or {})


def _params(rec: Mapping, kind: str) -> Optional[dict]:
    for m in rec.get("mechanics") or []:
        if isinstance(m, dict) and m.get("kind") == kind:
            return m.get("params") or {}
    return None


def compute_attributes(db: CardDB, card_id: str, variant: str = "base") -> np.ndarray:
    """ATTRIBUTE_KEYS vector of a card or token (zeros for an id the data doesn't know)."""
    rec = _record(db, card_id)
    out = np.zeros(N_ATTR, dtype=np.float32)
    if rec is None:
        return out
    stats, atk, mov, sp = _stats(db, card_id, variant)
    tr = card_traits(rec)
    kind = rec.get("type", "troop")
    is_spell = kind == "spell"
    hp, dmg = float(stats.get("hitpoints", 0)), float(stats.get("damage", 0))
    hs = float(atk.get("hit_speed_s") or 0.0)
    if is_spell and sp.get("hit_every_ms"):
        dps = dmg * 1000.0 / float(sp["hit_every_ms"])
    else:
        dps = dmg / hs if hs > 0 and atk else 0.0
    targets = list(atk.get("targets") or [])
    count = 0 if is_spell else int(rec.get("count", 1) or 1)
    death = sum(int((_params(rec, k) or {}).get("count", 1)) for k in ("death_spawn", "spawn_on_end")
                if _params(rec, k) is not None)
    wave = _params(rec, "periodic_spawn")
    rate = float(wave.get("count", 1)) * 1000.0 / float(wave.get("interval_ms", 1000)) if wave else 0.0
    if kind == "building" and rec.get("footprint_tiles"):
        size = rec["footprint_tiles"][0] / 2.0
    else:
        size = 0.0 if is_spell else float(rec.get("collision_radius_tiles", 0.5))
    radius = float(sp.get("radius_tiles", 0.0)) or float(sp.get("width_tiles", 0.0)) / 2.0
    vals = {
        "elixir": float(rec.get("elixir", 0)) / 10.0,
        "hitpoints": hp / 3000.0,
        "dps": dps / 300.0,
        "damage": dmg / 1000.0,
        "hit_speed": hs,
        "range": float(atk.get("range_tiles", 0.0)) / 10.0,
        "move_speed": float(mov.get("speed_tiles_s", 0.0)) * 60.0 / 100.0,
        "is_air": float("air" in tr),
        "targets_air": float("targets_air" in tr),
        "targets_ground": float(is_spell or "ground" in targets),
        "building_only": float("building_only" in tr),
        "splash_radius": float(atk.get("splash_radius_tiles", 0.0)) / 5.0,
        "is_building": float(kind == "building"),
        "lifetime": max(float(rec.get("lifetime_s") or 0.0), float(sp.get("duration_ms", 0.0)) / 1000.0) / 60.0,
        "is_spell": float(is_spell),
        "spell_radius": radius / 5.0,
        "spell_reach": float(sp.get("rolling_range_tiles", 0.0)) / 10.0,
        "stun_or_knockback": float("stun_or_knockback" in tr),
        "shield_hp": float(stats.get("shield_hitpoints", 0)) / 1000.0,
        "spawn_count": count / 10.0,
        "death_spawn_count": death / 5.0,
        "spawn_rate": rate,
        "death_damage": float(stats.get("death_damage", 0)) / 1000.0,
        "tower_damage": float(stats.get("crown_tower_damage", dmg)) / 1000.0,
        "deploy_time": 0.0 if is_spell else float(rec.get("deploy_time_s", 1.0)),
        "size": size,
        "kamikaze": float("kamikaze" in tr),
        "has_ability": float("ability" in tr or (variant == "hero" and bool((rec.get("variants") or {}).get("hero")))),
        "is_evolved": float(variant == "evo"),
        "is_hero": float(variant == "hero"),
    }
    out[:] = [vals[k] for k in ATTRIBUTE_KEYS]
    return out


def card_attributes(db: CardDB, card_id: str, variant: str = "base") -> np.ndarray:
    """Cached compute_attributes (per CardDB instance; read-only array)."""
    cache = db.__dict__.setdefault("_obs_attr_cache", {})
    key = (card_id, variant or "base")
    v = cache.get(key)
    if v is None:
        v = compute_attributes(db, card_id, variant or "base")
        v.setflags(write=False)
        cache[key] = v
    return v


def tower_attributes(db: CardDB, tower_id: str) -> np.ndarray:
    """TOWER_ATTRIBUTE_KEYS vector of a tower (or Tower Troop) entry in the card data."""
    cache = db.__dict__.setdefault("_obs_tower_cache", {})
    if tower_id in cache:
        return cache[tower_id]
    out = np.zeros(N_TATTR, dtype=np.float32)
    rec = db.towers.get(tower_id)
    if rec is not None:
        try:
            st = db.tower_stats(tower_id, ATTR_LEVEL)
        except (CardDataError, KeyError, ValueError):
            st = dict(rec.get("base_stats") or rec.get("stats_l11") or {})
        atk = rec.get("attack") or {}
        hs = float(atk.get("hit_speed_s") or 0.0)
        dmg = float(st.get("damage", 0))
        vals = {"hitpoints": float(st.get("hitpoints", 0)) / 5000.0, "dps": (dmg / hs if hs else 0.0) / 300.0,
                "damage": dmg / 1000.0, "hit_speed": hs, "range": float(atk.get("range_tiles", 0.0)) / 10.0,
                "splash_radius": float(atk.get("splash_radius_tiles", 0.0)) / 5.0,
                "targets_air": float("air" in (atk.get("targets") or []))}
        out[:] = [vals[k] for k in TOWER_ATTRIBUTE_KEYS]
    out.setflags(write=False)
    cache[tower_id] = out
    return out


# -- observation -------------------------------------------------------------------------------------------------
def _hand_variant(card: str, obs: Mapping, db: CardDB) -> Tuple[str, bool, bool]:
    """(variant the card would be played as now, is an evo slot, evo ready)."""
    evo_counter = obs.get("evo_counter") or {}
    if card in (obs.get("hero_cards") or ()):
        return "hero", False, False
    if card in evo_counter:
        try:
            ready = evo_counter[card] >= db.evo_cycles(card)
        except KeyError:
            ready = False
        return ("evo" if ready else "base"), True, ready
    return "base", False, False


def encode_observation(obs: Mapping, est: ElixirEstimate, guess, db: CardDB) -> np.ndarray:
    """OBS_DIM float32 vector for `obs["side"]` (layout in the module docstring and LAYOUT)."""
    side = int(obs["side"])
    t = float(obs["t_ms"])
    own_crowns, opp_crowns = obs["crowns"]
    elixir = float(obs["elixir"])
    out = np.zeros(OBS_DIM, dtype=np.float32)

    # global
    out[0:8] = (min(t / MAX_TIME_MS, 1.0), min(max(obs["time_left_ms"], 0) / MAX_TIME_MS, 1.0),
                obs["multiplier"] / 3.0, 1.0 if obs["overtime"] else 0.0, own_crowns / 3.0, opp_crowns / 3.0,
                (own_crowns - opp_crowns) / 3.0, max(elixir, 0.0) / 10.0)

    # tower state + tower troop attributes (own first)
    towers = {(u["side"], u["tower_role"]): u for u in obs["units"] if u["kind"] == "tower" and u["hp"] > 0}
    i = LAYOUT["towers"][0]
    for s in (side, 1 - side):
        for role in TOWER_ROLES:
            u = towers.get((s, role))
            if u is not None:
                out[i], out[i + 1] = u["hp"] / max(u["max_hp"], 1.0), 1.0 if u["active"] else 0.0
            i += 2
    i = LAYOUT["tower_troops"][0]
    for s in (side, 1 - side):
        for kind in TOWER_KINDS:
            roles = ("king",) if kind == "king" else ("princess_left", "princess_right")
            tid = next((towers[(s, r)]["card_id"] for r in roles if (s, r) in towers), None)
            if tid is None:
                tid = "king_tower" if kind == "king" else "princess_tower"
            out[i:i + N_TATTR] = tower_attributes(db, tid)
            i += N_TATTR

    # opponent perception (human-legal tracker + deck guess)
    i = LAYOUT["opp_summary"][0]
    out[i:i + 4] = (est.point / 10.0, est.low / 10.0, est.high / 10.0, len(est.known_cards) / 8.0)
    if guess is not None and getattr(guess, "likely_decks", None):
        out[i + 4] = float(guess.likely_decks[0][1])
    probs = merged_hand_probs(est, guess)
    top = sorted(((min(float(p), 1.0), c) for c, p in probs.items() if p > 0 and _record(db, c) is not None),
                 key=lambda r: (-r[0], r[1]))[:N_OPP_CARDS]
    i = LAYOUT["opp_cards"][0]
    for p, c in top:
        out[i] = p
        out[i + OPP_DYN:i + OPP_TOK] = card_attributes(db, c)
        i += OPP_TOK

    # own hand + next card
    hand = list(obs["hand"][:HAND_SLOTS]) + [""] * (HAND_SLOTS - len(obs["hand"][:HAND_SLOTS]))
    hand.append(obs.get("next_card") or "")
    i = LAYOUT["hand"][0]
    for c in hand:
        rec = _record(db, c) if c else None
        if rec is not None:
            variant, evo_slot, ready = _hand_variant(c, obs, db)
            cost = float(rec.get("elixir", 0))
            out[i:i + HAND_DYN] = (1.0, 1.0 if elixir + 1e-9 >= cost else 0.0, float(evo_slot), float(ready),
                                   (elixir - cost) / 10.0)
            out[i + HAND_DYN:i + HAND_TOK] = card_attributes(db, c, variant)
        i += HAND_TOK

    # units (no towers), nearest to the river first; ties broken the same way from either side
    units = [(u, *frame_xy(side, u["x"], u["y"])) for u in obs["units"] if u["kind"] != "tower" and u["hp"] > 0]
    units.sort(key=lambda r: (abs(r[2] - G.H / 2), r[0]["side"] != side, r[1], r[2], r[0]["uid"]))
    i = LAYOUT["units"][0]
    for u, fx, fy in units[:MAX_UNITS]:
        variant = u.get("variant") or "base"
        out[i:i + UNIT_DYN] = (1.0, 1.0 if u["side"] == side else -1.0, fx / G.W, fy / G.H,
                               max(u["hp"] / max(u["max_hp"], 1.0), 0.0), 1.0 if u["deploying"] else 0.0,
                               1.0 if u["layer"] == "air" else 0.0, 1.0 if variant == "evo" else 0.0)
        out[i + UNIT_DYN:i + UNIT_TOK] = card_attributes(db, u["card_id"], variant)
        i += UNIT_TOK

    # active spells / lingering areas (no uid in the observation: order by distance to the river, own first)
    spells = [(sp, *frame_xy(side, sp["x"], sp["y"])) for sp in obs.get("spells") or []]
    spells.sort(key=lambda r: (abs(r[2] - G.H / 2), r[0]["side"] != side, r[1], r[2], r[0]["card_id"]))
    i = LAYOUT["spells"][0]
    for sp, fx, fy in spells[:MAX_SPELLS]:
        out[i:i + SPELL_DYN] = (1.0, 1.0 if sp["side"] == side else -1.0, fx / G.W, fy / G.H)
        out[i + SPELL_DYN:i + SPELL_TOK] = card_attributes(db, sp["card_id"], sp.get("variant") or "base")
        i += SPELL_TOK
    return out


# -- left / right mirror (training augmentation) -----------------------------------------------------------------
def mirror_spec() -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(feature permutation, x-feature indices, tile permutation) for the left-right mirror of the arena:
    left / right princess tower slots swap, every present token's x becomes 1 - x, tile gx -> 17 - gx."""
    perm = np.arange(OBS_DIM)
    t0 = LAYOUT["towers"][0]
    for s in (0, 6):                                   # own / opponent block: (king, left, right) x (hp, active)
        for k in (0, 1):
            a, b = t0 + s + 2 + k, t0 + s + 4 + k
            perm[a], perm[b] = b, a
    xs = [LAYOUT["units"][0] + j * UNIT_TOK + 2 for j in range(MAX_UNITS)]
    xs += [LAYOUT["spells"][0] + j * SPELL_TOK + 2 for j in range(MAX_SPELLS)]
    tiles = np.arange(NUM_TILES).reshape(GRID_H, GRID_W)[:, ::-1].reshape(-1)
    return perm, np.array(xs), tiles.copy()


def check_model_meta(meta: Mapping) -> Optional[str]:
    """None if a model's .json matches this encoder, else the reason it can't be used."""
    ver = meta.get("obs_version")
    if ver in LEGACY_OBS_VERSIONS:
        return (f"legacy card-ID model ({ver}, {meta.get('obs_dim')}-dim, tied to a {len(meta.get('vocab') or [])}"
                f"-card pool): retrain it on {OBS_VERSION} shards")
    if ver != OBS_VERSION:
        return f"model features {ver!r}, app uses {OBS_VERSION!r}"
    if int(meta.get("obs_dim", -1)) != OBS_DIM:
        return f"model obs_dim {meta.get('obs_dim')} != {OBS_DIM}"
    if list(meta.get("attribute_keys") or []) != list(ATTRIBUTE_KEYS):
        return "model attribute_keys differ from this encoder's"
    if list(meta.get("tower_attribute_keys") or []) != list(TOWER_ATTRIBUTE_KEYS):
        return "model tower_attribute_keys differ from this encoder's"
    return None


def card_ids_unknown(db: CardDB, ids: Sequence[str]) -> List[str]:
    """Ids that are not playable cards (quarantine helper for replays / battlelogs)."""
    return [c for c in ids if c not in db.cards]
