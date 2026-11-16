"""
Imitation Learning Dataset Builder (tools/build_il_dataset.py)

Replays high-tier battle event streams (from Hugging Face Cochon123/clash-royale-replays
or VanguardX101/IL_Replay or local JSONL files) through CRbot's deterministic physics
simulator (Engine v6).

Generates human-legal (observation, action) training tuples with:
  1. Card Pool Guardrail: Strict or configurable deck coverage filter against the 38 supported cards.
  2. Human-Legal Observation Parity: Only visible units, own hand/elixir, OppElixirTracker,
     and BayesianDeckGuesser outputs (strictly no hidden ground truth opponent state).
  3. Balanced Negative/Wait Sampling: Emits gate=0 (wait) samples between plays at valid decision points.
  4. Memory-Safe Streaming Export: Streams samples directly into chunked .npz shards
     (e.g., data/il_shards/shard_0000.npz) on memory-constrained machines (8GB M1).
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import math
import os
import re
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Set, Tuple

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from backend_engine.contracts import TilePos
from backend_engine.perception.bayesian_deck_guesser import BayesianDeckGuesser, default_guesser
from backend_engine.perception.opp_elixir_count import OppElixirTracker
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig
import backend_engine.simulator.geometry as G

logger = logging.getLogger("build_il_dataset")

# Fixed Observation Tensor Specs
MAX_VISIBLE_UNITS = 30
UNIT_FEAT_DIM = 8
HAND_SLOTS = 4
GRID_W, GRID_H = 18, 32
NUM_TILES = GRID_W * GRID_H  # 576


def get_card_vocab(db: CardDB) -> List[str]:
    """Alphabetically sorted list of all 38 simulated cards."""
    return sorted(list(db.cards))


def normalize_card_name(raw: str) -> str:
    """Normalize raw card string from RoyaleAPI/HF datasets to CardDB id."""
    if not raw:
        return ""
    slug = raw.strip().lower().replace("-", "_")
    # Strip common evolution and hero prefixes/suffixes
    slug = re.sub(r"_(ev1|ev2|evo|evolution|hero)$", "", slug)
    slug = re.sub(r"^(evo|hero)_", "", slug)
    # Common alias corrections
    aliases = {
        "log": "the_log",
        "barb_barrel": "barbarian_barrel",
        "barb_hut": "barbarian_hut",
    }
    return aliases.get(slug, slug)


def check_deck_coverage(deck: Iterable[str], vocab_set: Set[str]) -> Tuple[int, List[str]]:
    """Return count of supported cards and list of unsupported cards."""
    normalized = [normalize_card_name(c) for c in deck]
    unsupported = [c for c in normalized if c not in vocab_set]
    return len(normalized) - len(unsupported), unsupported


def encode_observation(
    match: Match,
    side: int,
    tracker: OppElixirTracker,
    guesser: Optional[BayesianDeckGuesser],
    vocab: List[str],
) -> np.ndarray:
    """Construct a strictly human-legal 1D float32 observation vector."""
    opp_side = 1 - side
    st = match.sides[side]
    t_ms = match.t_ms
    est = tracker.estimate(t_ms)

    # 1. Global & Phase Features (8 floats)
    t_norm = min(t_ms / 300000.0, 1.0)
    reg_ms = match.regulation_ms
    time_left = max((reg_ms if t_ms < reg_ms else match.end_ms) - t_ms, 0)
    time_left_norm = min(time_left / 300000.0, 1.0)
    multiplier_norm = match.multiplier(t_ms) / 3.0
    is_overtime = 1.0 if t_ms >= reg_ms else 0.0
    own_crowns = st.crowns / 3.0
    opp_crowns = match.sides[opp_side].crowns / 3.0
    crown_diff = (st.crowns - match.sides[opp_side].crowns) / 3.0
    own_elixir_norm = min(max(st.elixir, 0.0), 10.0) / 10.0

    global_feats = [
        t_norm,
        time_left_norm,
        multiplier_norm,
        is_overtime,
        own_crowns,
        opp_crowns,
        crown_diff,
        own_elixir_norm,
    ]

    # 2. Tower Features (6 own + 6 opp = 12 floats)
    tower_feats = []
    for s in (side, opp_side):
        for role in ("king", "princess_left", "princess_right"):
            tow = match.tower(s, role)
            if tow is not None and tow.hp > 0:
                hp_frac = tow.hp / max(tow.max_hp, 1)
                is_active = 1.0 if tow.active else 0.0
            else:
                hp_frac = 0.0
                is_active = 0.0
            tower_feats.extend([hp_frac, is_active])

    # 3. Opponent Perception (Human-Legal Tracker & Guesser) (4 + 38 + 1 = 43 floats)
    opp_elixir_point = est.point / 10.0
    opp_elixir_low = est.low / 10.0
    opp_elixir_high = est.high / 10.0
    opp_known_count = len(est.known_cards) / 8.0
    opp_perception = [opp_elixir_point, opp_elixir_low, opp_elixir_high, opp_known_count]

    # In-hand probability distribution over the 38 cards
    in_hand_probs = [float(est.in_hand_prob.get(card, 0.0)) for card in vocab]
    opp_perception.extend(in_hand_probs)

    # Bayesian Deck Guesser confidence
    guess_conf = 0.0
    if guesser is not None and est.known_cards:
        guess = guesser.guess(est.known_cards)
        if guess and guess.likely_decks:
            guess_conf = float(guess.likely_decks[0][1])
    opp_perception.append(guess_conf)

    # 4. Own Hand & Queue (5 cards x 6 features = 30 floats)
    # 4 hand cards + 1 next card from queue
    hand_and_next = list(st.hand[:HAND_SLOTS])
    while len(hand_and_next) < HAND_SLOTS:
        hand_and_next.append("")
    next_card = st.queue[0] if st.queue else ""
    hand_and_next.append(next_card)

    vocab_map = {c: i for i, c in enumerate(vocab)}
    hand_feats = []
    for slot_card in hand_and_next:
        if slot_card and slot_card in match.db.cards:
            c_meta = match.db.cards[slot_card]
            c_idx_norm = (vocab_map.get(slot_card, 0) + 1.0) / (len(vocab) + 1.0)
            cost_norm = c_meta.get("elixir", 3) / 10.0
            can_play = 1.0 if st.elixir >= c_meta.get("elixir", 3) else 0.0
            is_evo_cap = 1.0 if match.db.has_variant(slot_card, "evo") else 0.0
            is_evo_ready = 1.0 if (is_evo_cap and st.evo_counter.get(slot_card, 0) >= match.db.evo_cycles(slot_card)) else 0.0
            ctype = c_meta.get("type", "troop")
            type_val = 0.0 if ctype == "troop" else (0.5 if ctype == "building" else 1.0)
            hand_feats.extend([c_idx_norm, cost_norm, can_play, is_evo_cap, is_evo_ready, type_val])
        else:
            hand_feats.extend([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

    # 5. Visible Board Units (MAX_VISIBLE_UNITS x UNIT_FEAT_DIM = 240 floats)
    # Collect all living non-tower units
    active_units = [u for u in match.units.values() if u.hp > 0 and u.kind != "tower"]
    # Sort by proximity to South player for consistent representation
    active_units.sort(key=lambda u: (abs(u.y - 16.0), u.uid))

    unit_feats = []
    for i in range(MAX_VISIBLE_UNITS):
        if i < len(active_units):
            u = active_units[i]
            is_present = 1.0
            is_own = 1.0 if u.side == side else -1.0
            x_norm = min(max(u.x / G.W, 0.0), 1.0)
            y_norm = min(max(u.y / G.H, 0.0), 1.0)
            hp_norm = min(max(u.hp / max(u.max_hp, 1), 0.0), 1.0)
            cost_norm = match.db.cards.get(u.card_id, {}).get("elixir", 3) / 10.0
            is_deploying = 1.0 if u.deploy_ms > 0 else 0.0
            is_air = 1.0 if u.layer == "air" else 0.0
            unit_feats.extend([is_present, is_own, x_norm, y_norm, hp_norm, cost_norm, is_deploying, is_air])
        else:
            unit_feats.extend([0.0] * UNIT_FEAT_DIM)

    # Concatenate all features
    full_vector = np.array(
        global_feats + tower_feats + opp_perception + hand_feats + unit_feats,
        dtype=np.float32,
    )
    return full_vector


OBS_DIM = 8 + 12 + 43 + 30 + (MAX_VISIBLE_UNITS * UNIT_FEAT_DIM)  # 333 floats


class ShardWriter:
    """Memory-safe chunked streaming exporter to .npz shards."""

    def __init__(self, output_dir: Path | str, shard_size: int = 5000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.current_shard_idx = 0

        self.obs_buffer: List[np.ndarray] = []
        self.gate_buffer: List[int] = []
        self.slot_buffer: List[int] = []
        self.tile_buffer: List[int] = []
        self.value_buffer: List[float] = []

        self.total_samples = 0
        self.total_plays = 0
        self.total_waits = 0
        self.total_battles = 0

    def add_sample(self, obs: np.ndarray, gate: int, slot: int, tile: int, value: float) -> None:
        """Add a single observation-action pair to memory buffer."""
        self.obs_buffer.append(obs)
        self.gate_buffer.append(gate)
        self.slot_buffer.append(slot)
        self.tile_buffer.append(tile)
        self.value_buffer.append(value)

        self.total_samples += 1
        if gate == 1:
            self.total_plays += 1
        else:
            self.total_waits += 1

        if len(self.obs_buffer) >= self.shard_size:
            self.flush()

    def flush(self) -> Optional[Path]:
        """Flush buffered samples to disk and free memory."""
        if not self.obs_buffer:
            return None

        shard_path = self.output_dir / f"shard_{self.current_shard_idx:04d}.npz"
        np.savez_compressed(
            shard_path,
            obs=np.array(self.obs_buffer, dtype=np.float32),
            gate=np.array(self.gate_buffer, dtype=np.int64),
            card_slot=np.array(self.slot_buffer, dtype=np.int64),
            tile_index=np.array(self.tile_buffer, dtype=np.int64),
            value_target=np.array(self.value_buffer, dtype=np.float32),
        )

        logger.info(f"Saved {shard_path.name} with {len(self.obs_buffer)} samples")
        self.current_shard_idx += 1
        self.obs_buffer.clear()
        self.gate_buffer.clear()
        self.slot_buffer.clear()
        self.tile_buffer.clear()
        self.value_buffer.clear()
        gc.collect()
        return shard_path

    def close(self, vocab: List[str]) -> Path:
        """Flush any remaining samples and write dataset manifest."""
        self.flush()
        manifest_path = self.output_dir / "manifest.json"
        manifest = {
            "total_samples": self.total_samples,
            "total_plays": self.total_plays,
            "total_waits": self.total_waits,
            "total_battles": self.total_battles,
            "num_shards": self.current_shard_idx,
            "shard_size": self.shard_size,
            "obs_dim": OBS_DIM,
            "num_tiles": NUM_TILES,
            "num_card_slots": HAND_SLOTS,
            "vocab": vocab,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info(f"Dataset complete: {self.total_samples} samples across {self.current_shard_idx} shards")
        return manifest_path


def replay_battle(
    match_meta: dict,
    events: List[dict],
    db: CardDB,
    vocab: List[str],
    shard_writer: ShardWriter,
    wait_ratio: float = 1.0,
    min_deck_coverage: int = 8,
) -> bool:
    """Replay a single battle event stream through Engine v6 and emit training samples."""
    vocab_set = set(vocab)
    team_deck = [normalize_card_name(c) for c in match_meta.get("decks", {}).get("team", [])]
    opp_deck = [normalize_card_name(c) for c in match_meta.get("decks", {}).get("opponent", [])]

    # Guardrail check
    t_cov, t_unsupp = check_deck_coverage(team_deck, vocab_set)
    o_cov, o_unsupp = check_deck_coverage(opp_deck, vocab_set)

    if t_cov < min_deck_coverage or o_cov < min_deck_coverage:
        logger.debug(f"Skipping match {match_meta.get('battle_id')}: coverage team {t_cov}/8, opp {o_cov}/8")
        return False

    # Filter out empty or unsupported cards for simulation initialization
    t_deck_sim = [c for c in team_deck if c in vocab_set]
    o_deck_sim = [c for c in opp_deck if c in vocab_set]
    if len(t_deck_sim) < 8 or len(o_deck_sim) < 8:
        return False

    # Determine ground-truth match outcome for Team (side 0)
    crowns = match_meta.get("crowns", {})
    t_crowns = crowns.get("team", 0)
    o_crowns = crowns.get("opponent", 0)
    if t_crowns > o_crowns:
        val_target = 1.0
    elif t_crowns < o_crowns:
        val_target = 0.0
    else:
        val_target = 0.5

    # Identify evolved cards
    evo_capable = db.evo_capable()
    team_evo = tuple(c for c in t_deck_sim if c in evo_capable)
    opp_evo = tuple(c for c in o_deck_sim if c in evo_capable)

    cfg = MatchConfig(
        decks=(tuple(t_deck_sim), tuple(o_deck_sim)),
        levels=(11, 11),
        evo_cards=(team_evo, opp_evo),
        seed=0,
        shuffle=False,
    )
    match = Match(cfg, db)

    # Initialize human-legal perception
    tracker = OppElixirTracker(db.rules, observer_side=0, evo_cards=evo_capable)
    guesser = default_guesser()

    # Track samples for this match to ensure balanced wait vs play
    play_samples: List[Tuple[np.ndarray, int, int, int, float]] = []
    wait_candidates: List[Tuple[np.ndarray, int, int, int, float]] = []

    last_wait_t_ms = 0
    wait_sample_interval_ms = 1000  # candidate wait check every 1 second

    # Sort events by tick
    sorted_events = sorted(events, key=lambda e: (e.get("ticks", 0), e.get("sequence", 0)))

    for ev in sorted_events:
        t_target_ms = ev.get("ticks", 0) * 50

        # Advance physics simulation up to event timestamp
        while match.t_ms < t_target_ms and match.result is None:
            # Check for negative/wait candidate emission for Team (side 0)
            if (match.t_ms - last_wait_t_ms) >= wait_sample_interval_ms:
                last_wait_t_ms = match.t_ms
                st = match.sides[0]
                # Only sample wait if team has actionable elixir (>= 3.5)
                # and at least one card in hand is playable
                has_playable = any(
                    st.elixir >= db.cards.get(c, {}).get("elixir", 99) for c in st.hand if c in db.cards
                )
                if has_playable and st.elixir >= 3.5:
                    obs = encode_observation(match, side=0, tracker=tracker, guesser=guesser, vocab=vocab)
                    wait_candidates.append((obs, 0, -1, -1, val_target))

            match.step()

        if match.result is not None:
            break

        side = 0 if ev.get("side") == "team" else 1
        ev_type = ev.get("event_type", "card_play")

        # Handle champion/hero ability
        if ev_type == "ability_activation":
            card = normalize_card_name(ev.get("ability_card") or "")
            if card and card in db.cards:
                match.use_ability(side, card)
            continue

        # Handle card play
        card = normalize_card_name(ev.get("card") or "")
        if not card or card not in db.cards:
            continue

        raw_x = ev.get("x")
        raw_y = ev.get("y")
        if raw_x is None or raw_y is None:
            continue

        # Convert RoyaleAPI millitiles to arena tiles
        x = float(raw_x) / 1000.0
        y = 32.0 - (float(raw_y) / 1000.0)
        x = min(max(x, 0.5), G.W - 0.5)
        y = min(max(y, 0.5), G.H - 0.5)

        st = match.sides[side]

        # Reconstruct hand cycle: if card is in queue rather than hand, swap into hand
        if card not in st.hand and card in st.queue:
            q_idx = st.queue.index(card)
            st.hand[0], st.queue[q_idx] = st.queue[q_idx], st.hand[0]

        # Ensure sufficient elixir for replay continuity
        cost = db.cards[card]["elixir"]
        if st.elixir < cost:
            st.elixir = float(cost)

        # If team played, extract observation and record positive sample
        if side == 0 and card in st.hand:
            slot_idx = st.hand.index(card)
            grid_x = min(max(int(x), 0), GRID_W - 1)
            grid_y = min(max(int(y), 0), GRID_H - 1)
            tile_idx = grid_y * GRID_W + grid_x

            # Encode observation right before the card play executes
            obs = encode_observation(match, side=0, tracker=tracker, guesser=guesser, vocab=vocab)
            play_samples.append((obs, 1, slot_idx, tile_idx, val_target))

        # Execute play in engine
        res = match.play(side, card, x, y)

        # Update perception tracker for side 0
        if side == 1:
            deploy_ev = match.events[-1] if match.events else None
            if deploy_ev is not None:
                tracker.observe(deploy_ev)

    # Subsample wait candidates to match requested wait_ratio
    num_waits = int(round(len(play_samples) * wait_ratio))
    if wait_candidates and num_waits > 0:
        if len(wait_candidates) > num_waits:
            indices = np.linspace(0, len(wait_candidates) - 1, num_waits, dtype=int)
            selected_waits = [wait_candidates[i] for i in indices]
        else:
            selected_waits = wait_candidates
    else:
        selected_waits = []

    # Emit all samples for this match to shard writer
    for s in play_samples:
        shard_writer.add_sample(*s)
    for s in selected_waits:
        shard_writer.add_sample(*s)

    shard_writer.total_battles += 1
    return True


def stream_hf_corpus(
    matches_url: str,
    events_url: str,
    db: CardDB,
    vocab: List[str],
    shard_writer: ShardWriter,
    max_matches: int = 500,
    min_deck_coverage: int = 8,
) -> None:
    """Stream replay matches and events from remote Hugging Face JSONL files line by line."""
    import io

    logger.info(f"Opening matches stream: {matches_url}")
    req_m = urllib.request.Request(matches_url, headers={"User-Agent": "Mozilla/5.0"})
    matches_by_id: Dict[str, dict] = {}

    with urllib.request.urlopen(req_m) as resp_m:
        reader = io.TextIOWrapper(resp_m, encoding="utf-8")
        for line in reader:
            if not line.strip():
                continue
            m = json.loads(line)
            bid = m.get("battle_id")
            if bid:
                matches_by_id[bid] = m
                if len(matches_by_id) >= max_matches:
                    break

    logger.info(f"Loaded {len(matches_by_id)} match headers. Streaming events...")

    req_e = urllib.request.Request(events_url, headers={"User-Agent": "Mozilla/5.0"})
    events_by_id: Dict[str, List[dict]] = {bid: [] for bid in matches_by_id}

    with urllib.request.urlopen(req_e) as resp_e:
        reader = io.TextIOWrapper(resp_e, encoding="utf-8")
        for line in reader:
            if not line.strip():
                continue
            ev = json.loads(line)
            bid = ev.get("battle_id")
            if bid in events_by_id:
                events_by_id[bid].append(ev)

    logger.info(f"Events loaded for target matches. Replaying through Engine v6...")
    for bid, m_meta in matches_by_id.items():
        evs = events_by_id.get(bid, [])
        if evs:
            replay_battle(
                match_meta=m_meta,
                events=evs,
                db=db,
                vocab=vocab,
                shard_writer=shard_writer,
                min_deck_coverage=min_deck_coverage,
            )


def main():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    parser = argparse.ArgumentParser(description="CRbot Imitation Learning Dataset Builder")
    parser.add_argument("--output-dir", type=str, default="data/il_shards", help="Target directory for shards")
    parser.add_argument("--shard-size", type=int, default=5000, help="Samples per shard")
    parser.add_argument("--max-matches", type=int, default=500, help="Max matches to process")
    parser.add_argument("--min-coverage", type=int, default=8, help="Min supported cards in deck (default 8/8)")
    parser.add_argument("--wait-ratio", type=float, default=1.0, help="Wait-to-play ratio")
    parser.add_argument(
        "--matches-file",
        type=str,
        default="",
        help="Local matches.jsonl file (or leave empty to stream from Hugging Face)",
    )
    parser.add_argument(
        "--events-file",
        type=str,
        default="",
        help="Local events.jsonl file (or leave empty to stream from Hugging Face)",
    )
    args = parser.parse_args()

    db = CardDB.load()
    vocab = get_card_vocab(db)
    logger.info(f"Loaded CardDB: {len(vocab)} cards in vocabulary")

    shard_writer = ShardWriter(output_dir=args.output_dir, shard_size=args.shard_size)

    if args.matches_file and args.events_file:
        logger.info(f"Reading from local files: {args.matches_file}, {args.events_file}")
        # Local file reading implementation
        matches_by_id = {}
        with open(args.matches_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                m = json.loads(line)
                bid = m.get("battle_id")
                if bid:
                    matches_by_id[bid] = m
                    if len(matches_by_id) >= args.max_matches:
                        break

        events_by_id = {bid: [] for bid in matches_by_id}
        with open(args.events_file, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                ev = json.loads(line)
                bid = ev.get("battle_id")
                if bid in events_by_id:
                    events_by_id[bid].append(ev)

        for bid, m_meta in matches_by_id.items():
            evs = events_by_id.get(bid, [])
            if evs:
                replay_battle(
                    match_meta=m_meta,
                    events=evs,
                    db=db,
                    vocab=vocab,
                    shard_writer=shard_writer,
                    min_deck_coverage=args.min_coverage,
                )
    else:
        # Default Hugging Face corpus URLs
        matches_url = "https://huggingface.co/datasets/Cochon123/clash-royale-replays/resolve/main/cleaned-legacy/matches.jsonl"
        events_url = "https://huggingface.co/datasets/Cochon123/clash-royale-replays/resolve/main/cleaned-legacy/events.jsonl"
        stream_hf_corpus(
            matches_url=matches_url,
            events_url=events_url,
            db=db,
            vocab=vocab,
            shard_writer=shard_writer,
            max_matches=args.max_matches,
            min_deck_coverage=args.min_coverage,
        )

    manifest_path = shard_writer.close(vocab)
    logger.info(f"Done! Manifest saved to: {manifest_path}")


if __name__ == "__main__":
    main()
