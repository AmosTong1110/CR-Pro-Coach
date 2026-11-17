"""Imitation-learning dataset builder: real replays -> (observation, action) shards.

Source: Hugging Face `VanguardX101/IL_Replay` (252,238 anonymized RoyaleAPI replays, 52 parquet parts of ~5,000;
the `replays/part-*.parquet` table holds the decks, crowns and full event stream in `payload_json`). Parts are
downloaded one at a time into a git-ignored cache (`data/il_raw/`), checked against the dataset's SHA-256
manifest, read in small record batches and (unless --keep-raw) deleted after use, so RAM and disk stay small.

Unknown-card quarantine (backend_engine/simulator/quarantine.py): a replay with a card that
configs/cards_stats_v2026.json does not know at all (a newly released card) is skipped and counted, never
guessed; the build ends with one summary warning (run tools/sync_card_catalog.py to add the card). Catalog cards
that are not playable yet (enabled: false / needs_hook) are proxied on the non-learner side as below, or skipped
too with --strict-quarantine.

Per replay, both decks are mapped to the simulated card pool. A side is a **learner** when its deck has at least
--min-coverage simulated cards (default 8 = all). The other side may contain unsimulated cards: each one is
replaced by a **proxy** (simulated card of the same type and air/ground layer, nearest elixir cost), so the
board still shows a unit of about the right cost where the real card was played. A replay is used when at
least one side is a learner and the other side has >= --min-opp-coverage simulated cards.

The replay is run through the engine (seed 0, deck order = order of first play, the replay's card levels and
simulated evolution / hero slots). For every learner side:
  * play sample  just before each of its real plays: observation, slot of the card, tile in the side's own frame,
                 slot / tile masks. Skipped (but still played, to keep the board going) when the card is a proxy,
                 the side is > --max-elixir-deficit short of the cost in the sim, or the real tile is illegal in
                 the sim (sim divergence, e.g. a tower that fell for real is still up).
  * wait sample  at 1 s decision points where a card is affordable and the side doesn't play in the next 0.5 s;
                 subsampled to --wait-ratio x the number of play samples of that side.
  * value target final result from that side's view (1 win, 0 loss, 0.5 draw).
Observations come from backend_engine/brain/obs_encoder.py (il_obs_v2, the same code NeuralBot uses): own hand /
elixir, visible units and spells, the side's own OppElixirTracker + deck guess, every card described by its
physical attributes (no card ids). No hidden opponent state. Shards stay valid when cards are added.

Shards: data/il_shards/shard_NNNN.npz with obs f32[N,1752], gate i8, card_slot i8 (-1 for waits),
tile_index i16 (-1), value_target f32, slot_mask bool[N,4], tile_mask u8[N,72] (packed bits of the played
card's legal tiles), battle u32, side i8. manifest.json has the counts, filter statistics and the obs schema
(obs_version, obs_dim, attribute_keys, layout).

    .venv/bin/python tools/build_il_dataset.py --max-samples 100000            # streams parts until 100k
    .venv/bin/python tools/build_il_dataset.py --parts 0-3 --keep-raw
    .venv/bin/python tools/build_il_dataset.py --survey --parts 0               # coverage / yield only
"""
from __future__ import annotations

import argparse
import bisect
import collections
import gc
import hashlib
import json
import logging
import re
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np

from backend_engine.brain import obs_encoder as F
from backend_engine.perception.bayesian_deck_guesser import default_guesser
from backend_engine.perception.opp_elixir_count import OppElixirTracker
from backend_engine.simulator import geometry as G
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig
from backend_engine.simulator.quarantine import CardQuarantine

ROOT = Path(__file__).resolve().parents[1]
HF_REPO = "VanguardX101/IL_Replay"
HF_BASE = f"https://huggingface.co/datasets/{HF_REPO}/resolve/main"
N_PARTS = 52
SCHEMA = "il_shards_v2"
WAIT_EVERY_MS = 1000
WAIT_QUIET_MS = 500
logger = logging.getLogger("build_il_dataset")

# unsimulated flying troops (proxied by a flying simulated troop)
AIR_TROOPS = {"baby_dragon", "balloon", "electro_dragon", "flying_machine", "inferno_dragon", "lava_hound",
              "mega_minion", "minion_horde", "phoenix", "skeleton_dragons", "night_witch_bats", "lava_pups"}


# -- card names ------------------------------------------------------------------------------------------------
_FORM = re.compile(r"-(ev\d+|hero)$")


def split_card_key(key: str) -> Tuple[str, str]:
    """RoyaleAPI card key -> (snake_case id, form): 'firecracker-ev1' -> ('firecracker', 'evo')."""
    key = (key or "").strip().lower()
    m = _FORM.search(key)
    form = "base" if not m else ("hero" if m.group(1) == "hero" else "evo")
    base = _FORM.sub("", key).replace("-", "_").replace(" ", "_").replace(".", "")
    return {"log": "the_log"}.get(base, base), form


def normalize_card_name(raw: str) -> str:
    return split_card_key(raw)[0]


def _slug(name: str) -> str:
    return name.lower().replace(".", "").replace("-", " ").strip().replace(" ", "_")


def load_card_catalog() -> Dict[str, Tuple[str, int]]:
    """id -> (type, cost) for every card in the game (official list: id prefix 26 troop, 27 building, 28 spell;
    costs from the official list, else configs/card_costs_hf.json)."""
    costs = json.loads((ROOT / "configs" / "card_costs_hf.json").read_text())["costs"]
    out: Dict[str, Tuple[str, int]] = {}
    path = ROOT / "data" / "raw_sources" / "official_cards.json"
    if path.exists():
        for it in json.loads(path.read_text())["items"]:
            kind = {26: "troop", 27: "building", 28: "spell"}.get(it["id"] // 1_000_000, "troop")
            cid = _slug(it["name"])
            out[cid] = (kind, int(it.get("elixirCost") or costs.get(cid, 4)))
    for cid, c in costs.items():
        out.setdefault(cid, ("troop", int(c)))
    return out


class ProxyMap:
    """Unsimulated card -> simulated stand-in of the same type / layer and the nearest cost (deterministic)."""

    def __init__(self, db: CardDB, catalog: Dict[str, Tuple[str, int]]):
        self.db, self.catalog = db, catalog
        self.pool = sorted(db.cards)

    def _layer(self, cid: str) -> str:
        if cid in self.db.cards:
            return (self.db.cards[cid].get("movement") or {}).get("layer", "ground")
        return "air" if cid in AIR_TROOPS else "ground"

    def candidates(self, cid: str) -> List[str]:
        kind, cost = self.catalog.get(cid, ("troop", 4))
        layer = self._layer(cid)
        same = [c for c in self.pool if self.db.cards[c]["type"] == kind]
        if kind == "troop":
            same = [c for c in same if self._layer(c) == layer] or same
        return sorted(same, key=lambda c: (abs(self.db.cards[c]["elixir"] - cost), c))

    def deck(self, cards: Sequence[str]) -> Dict[str, str]:
        """Map a deck's unsimulated cards to unused simulated ones: {original: simulated}."""
        used = {c for c in cards if c in self.db.cards}
        out = {}
        for c in cards:
            if c in self.db.cards:
                out[c] = c
                continue
            pick = next((p for p in self.candidates(c) if p not in used), None) \
                or next(p for p in self.pool if p not in used)
            used.add(pick)
            out[c] = pick
        return out


# -- replay model ------------------------------------------------------------------------------------------------
@dataclass
class ReplayEvent:
    tick: int
    side: int                 # 0 team (south), 1 opponent
    card: str                 # original snake_case id
    ability: bool
    x: float = 0.0
    y: float = 0.0


@dataclass
class Replay:
    rid: str
    decks: Tuple[List[str], List[str]]
    forms: Tuple[Dict[str, str], Dict[str, str]]
    levels: Tuple[int, int]
    crowns: Tuple[int, int]
    events: List[ReplayEvent]
    mode: str = ""


def parse_payload(payload: dict, rid: str = "") -> Optional[Replay]:
    """IL_Replay `payload_json` -> Replay (None if not a 1v1 battle with two 8-card decks)."""
    b = payload.get("battle") or {}
    decks, forms, levels, crowns = [], [], [], []
    for key in ("team", "opponent"):
        players = (b.get(key) or {}).get("players") or []
        if len(players) != 1 or len(players[0].get("deck") or []) != 8:
            return None
        ids, fm, lv = [], {}, []
        for c in players[0]["deck"]:
            cid, form = split_card_key(c.get("card_key", ""))
            ids.append(cid)
            fm[cid] = form
            if c.get("level"):
                lv.append(int(c["level"]))
        if len(set(ids)) != 8:
            return None
        decks.append(ids)
        forms.append(fm)
        levels.append(collections.Counter(lv).most_common(1)[0][0] if lv else 11)
        crowns.append(int((b.get(key) or {}).get("crowns") or 0))
    events = []
    for ev in payload.get("events") or []:
        side = {"team": 0, "opponent": 1}.get(ev.get("side"))
        tick = ev.get("replay_tick_20hz")
        if side is None or tick is None:
            continue
        if ev.get("kind") == "activate_ability":
            cands = ev.get("ability_source_candidates") or []
            if len(cands) == 1:
                events.append(ReplayEvent(int(tick), side, split_card_key(cands[0])[0], True))
            continue
        if ev.get("kind") != "play_card":
            continue
        nw = (ev.get("coordinates") or {}).get("native_world_units") or {}
        if nw.get("x") is None or nw.get("y") is None:
            continue
        events.append(ReplayEvent(int(tick), side, split_card_key(ev.get("card_key", ""))[0], False,
                                  nw["x"] / 1000.0, nw["y"] / 1000.0))
    events.sort(key=lambda e: e.tick)       # stable: keeps the source order inside a tick
    return Replay(rid, (decks[0], decks[1]), (forms[0], forms[1]), (levels[0], levels[1]),
                  (crowns[0], crowns[1]), events, b.get("game_mode", ""))


# -- samples -----------------------------------------------------------------------------------------------------
@dataclass
class Stats:
    replays_seen: int = 0
    replays_used: int = 0
    learner_sides: int = 0
    rejected: Dict[str, int] = field(default_factory=collections.Counter)
    skipped_plays: Dict[str, int] = field(default_factory=collections.Counter)
    hand_repairs: int = 0
    relocated_plays: int = 0
    dropped_plays: int = 0

    def as_dict(self) -> dict:
        return {k: (dict(v) if isinstance(v, collections.Counter) else v) for k, v in self.__dict__.items()}


class ShardWriter:
    """Buffers samples and writes compressed .npz shards of `shard_size` samples."""

    def __init__(self, output_dir: Path | str, shard_size: int = 5000):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.shard_size = shard_size
        self.current_shard_idx = 0
        self.buf: Dict[str, list] = collections.defaultdict(list)
        self.total_samples = self.total_plays = self.total_waits = self.total_battles = 0

    def add_sample(self, obs: np.ndarray, gate: int, slot: int, tile: int, value: float,
                   slot_mask: Optional[np.ndarray] = None, tile_mask: Optional[np.ndarray] = None,
                   battle: int = 0, side: int = 0) -> None:
        self.buf["obs"].append(obs)
        self.buf["gate"].append(gate)
        self.buf["card_slot"].append(slot)
        self.buf["tile_index"].append(tile)
        self.buf["value_target"].append(value)
        self.buf["slot_mask"].append(np.ones(F.HAND_SLOTS, bool) if slot_mask is None else slot_mask)
        self.buf["tile_mask"].append(F.pack_mask(np.zeros(F.NUM_TILES, bool) if tile_mask is None else tile_mask))
        self.buf["battle"].append(battle)
        self.buf["side"].append(side)
        self.total_samples += 1
        if gate == 1:
            self.total_plays += 1
        else:
            self.total_waits += 1
        if len(self.buf["obs"]) >= self.shard_size:
            self.flush()

    def flush(self) -> Optional[Path]:
        if not self.buf["obs"]:
            return None
        path = self.output_dir / f"shard_{self.current_shard_idx:04d}.npz"
        b = self.buf
        np.savez_compressed(
            path,
            obs=np.asarray(b["obs"], dtype=np.float32),
            gate=np.asarray(b["gate"], dtype=np.int8),
            card_slot=np.asarray(b["card_slot"], dtype=np.int8),
            tile_index=np.asarray(b["tile_index"], dtype=np.int16),
            value_target=np.asarray(b["value_target"], dtype=np.float32),
            slot_mask=np.asarray(b["slot_mask"], dtype=bool),
            tile_mask=np.asarray(b["tile_mask"], dtype=np.uint8),
            battle=np.asarray(b["battle"], dtype=np.uint32),
            side=np.asarray(b["side"], dtype=np.int8),
        )
        logger.info("saved %s (%d samples)", path.name, len(b["obs"]))
        self.current_shard_idx += 1
        self.buf = collections.defaultdict(list)
        gc.collect()
        return path

    def close(self, extra: Optional[dict] = None) -> Path:
        self.flush()
        manifest = {
            "schema": SCHEMA,
            **F.schema(),
            "total_samples": self.total_samples,
            "total_plays": self.total_plays,
            "total_waits": self.total_waits,
            "total_battles": self.total_battles,
            "num_shards": self.current_shard_idx,
            "shard_size": self.shard_size,
            "num_tiles": F.NUM_TILES,
            "num_card_slots": F.HAND_SLOTS,
            **(extra or {}),
        }
        path = self.output_dir / "manifest.json"
        path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return path


class _SidePerception:
    """One learner side's human-legal view: its OppElixirTracker + cached deck guesses."""

    def __init__(self, db: CardDB, side: int, guesser):
        self.side = side
        self.tracker = OppElixirTracker(db.rules, observer_side=side, evo_cards=db.evo_capable())
        self.guesser = guesser
        self._cache: Dict[Tuple[str, ...], object] = {}

    def view(self, t_ms: int):
        est = self.tracker.estimate(t_ms)
        if self.guesser is None:
            return est, None
        key = tuple(sorted(est.known_cards))
        if key not in self._cache:
            self._cache[key] = self.guesser.guess(est.known_cards)
        return est, self._cache[key]


def _deck_order(deck: Sequence[str], events: Sequence[ReplayEvent], side: int) -> List[str]:
    """Order of first play (cards never played go last): with an unshuffled deck this makes the first plays
    always come from the hand, so few hand repairs are needed."""
    seen = list(dict.fromkeys(e.card for e in events if e.side == side and not e.ability and e.card in deck))
    return seen + [c for c in deck if c not in seen]


def _repair_hand(st, card: str, upcoming: Sequence[str]) -> None:
    """The real player had `card` in hand but the sim's reconstructed cycle has it in the queue: swap it with
    the hand card that is needed latest (or never) in the side's upcoming plays."""
    q = st.queue.index(card)

    def next_use(c: str) -> int:
        return upcoming.index(c) if c in upcoming else 1 << 30
    slot = max(range(len(st.hand)), key=lambda k: (next_use(st.hand[k]), k))
    st.hand[slot], st.queue[q] = st.queue[q], st.hand[slot]


def replay_battle(rep: Replay, db: CardDB, writer: ShardWriter, battle_idx: int,
                  stats: Stats, proxies: ProxyMap, guesser=None, min_coverage: int = 8,
                  min_opp_coverage: int = 4, wait_ratio: float = 1.0, max_elixir_deficit: float = 1.0,
                  quarantine: Optional[CardQuarantine] = None) -> bool:
    """Replay one battle and write its samples. Returns False if the replay was filtered out."""
    stats.replays_seen += 1
    if quarantine is not None:
        reason = quarantine.check([*rep.decks[0], *rep.decks[1]])
        if reason:
            stats.rejected["quarantine"] += 1
            logger.debug("replay %s skipped: %s", rep.rid, reason)
            return False
    cov = [sum(c in db.cards for c in d) for d in rep.decks]
    learners = [s for s in (0, 1) if cov[s] >= min_coverage and cov[1 - s] >= min_opp_coverage]
    if not learners:
        stats.rejected["coverage"] += 1
        return False

    maps = [proxies.deck(d) for d in rep.decks]                  # original -> simulated, per side
    sim_decks = [tuple(_deck_order([maps[s][c] for c in rep.decks[s]],
                                   [ReplayEvent(e.tick, e.side, maps[s].get(e.card, ""), e.ability)
                                    for e in rep.events if e.side == s], s)) for s in (0, 1)]
    slots = []
    for s in (0, 1):
        sl = []
        for c in rep.decks[s]:
            form = rep.forms[s].get(c, "base")
            if c in db.cards and form in ("evo", "hero") and db.variant_simulated(c, form):
                sl.append(c if form == "evo" else f"{c}#hero")
        slots.append(tuple(sl))
    levels = tuple(min(max(int(lv), 1), 16) for lv in rep.levels)
    try:
        match = Match(MatchConfig((sim_decks[0], sim_decks[1]), levels, (slots[0], slots[1]), seed=0,
                                  shuffle=False), db)
    except ValueError as e:
        stats.rejected["config"] += 1
        logger.debug("replay %s: %s", rep.rid, e)
        return False

    stats.replays_used += 1
    stats.learner_sides += len(learners)
    persp = {s: _SidePerception(db, s, guesser) for s in learners}
    value = {0: 0.5 if rep.crowns[0] == rep.crowns[1] else float(rep.crowns[0] > rep.crowns[1])}
    value[1] = 1.0 - value[0]
    plays: Dict[int, list] = {s: [] for s in learners}
    waits: Dict[int, list] = {s: [] for s in learners}
    play_ticks = {s: [e.tick * G.TICK_MS for e in rep.events if e.side == s and not e.ability] for s in (0, 1)}
    upcoming = {s: [maps[s].get(e.card, "") for e in rep.events if e.side == s and not e.ability] for s in (0, 1)}
    played = {0: 0, 1: 0}
    next_wait = WAIT_EVERY_MS

    def sample(s: int):
        obs = match.observation(s)
        est, guess = persp[s].view(match.t_ms)
        return obs, F.encode_observation(obs, est, guess, db)

    for ev in rep.events:
        t_target = ev.tick * G.TICK_MS
        while match.t_ms < t_target and match.result is None:
            if match.t_ms >= next_wait:
                next_wait += WAIT_EVERY_MS
                for s in learners:
                    k = bisect.bisect_left(play_ticks[s], match.t_ms)
                    if k < len(play_ticks[s]) and play_ticks[s][k] - match.t_ms < WAIT_QUIET_MS:
                        continue
                    st = match.sides[s]
                    if any(db.cards[c]["elixir"] <= st.elixir + 1e-9 for c in st.hand):
                        obs, vec = sample(s)
                        waits[s].append((vec, F.slot_mask(obs, db)))
            match.step()
        if match.result is not None:
            break
        s = ev.side
        card = maps[s].get(ev.card)
        if ev.ability:
            if card:
                match.use_ability(s, card)
            continue
        played[s] += 1                                   # index into upcoming[s] (all of the side's plays)
        if card is None:                                 # card outside the listed deck: skip the event
            stats.dropped_plays += 1
            continue
        st = match.sides[s]
        if card not in st.hand:
            if card not in st.queue:
                stats.dropped_plays += 1
                continue
            _repair_hand(st, card, upcoming[s][played[s]:])
            stats.hand_repairs += 1
        cost = float(db.cards[card]["elixir"])
        deficit = cost - st.elixir

        # positive sample (before the play changes anything)
        if s in persp:
            reason = None
            if card != ev.card:
                reason = "proxy_card"
            elif deficit > max_elixir_deficit:
                reason = "elixir_deficit"
            if deficit > 0:
                st.elixir = cost                         # small timing / leak differences: top up
            obs, vec = sample(s)
            tile = F.tile_index(s, ev.x, ev.y)
            tmask = F.tile_mask(obs, card, db)
            if reason is None and not tmask[tile]:
                reason = "illegal_tile"
            if reason is None:
                plays[s].append((vec, obs["hand"].index(card), tile, F.slot_mask(obs, db), tmask))
            else:
                stats.skipped_plays[reason] += 1
        elif deficit > 0:
            st.elixir = cost

        res = match.play(s, card, ev.x, ev.y)
        if not res.ok and res.reason in ("enemy_half", "river", "blocked_by_structure", "out_of_bounds"):
            obs = match.observation(s)
            legal = np.flatnonzero(F.tile_mask(obs, card, db))
            if legal.size:
                want = F.tile_index(s, ev.x, ev.y)
                wy, wx = divmod(want, F.GRID_W)
                ly, lx = np.divmod(legal, F.GRID_W)
                best = int(legal[np.argmin((lx - wx) ** 2 + (ly - wy) ** 2)])
                res = match.play(s, card, *F.tile_center(s, best))
                stats.relocated_plays += int(res.ok)
        if not res.ok:
            stats.dropped_plays += 1
            continue
        for o, p in persp.items():
            if o != s:
                p.tracker.observe(res.event)

    for s in learners:
        n_wait = int(round(len(plays[s]) * wait_ratio))
        chosen = waits[s]
        if len(chosen) > n_wait:
            chosen = [chosen[i] for i in np.linspace(0, len(chosen) - 1, n_wait).round().astype(int)]
        for vec, slot, tile, smask, tmask in plays[s]:
            writer.add_sample(vec, 1, slot, tile, value[s], smask, tmask, battle_idx, s)
        for vec, smask in chosen:
            writer.add_sample(vec, 0, -1, -1, value[s], smask, None, battle_idx, s)
    writer.total_battles += 1
    return True


# -- IL_Replay streaming -----------------------------------------------------------------------------------------
def _fetch(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_suffix(dest.suffix + ".part")
    req = urllib.request.Request(url, headers={"User-Agent": "crbot-il-builder"})
    with urllib.request.urlopen(req, timeout=120) as r, open(tmp, "wb") as f:
        shutil.copyfileobj(r, f, 1 << 20)
    tmp.replace(dest)


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def dataset_manifest(cache: Path) -> dict:
    path = cache / "manifest.json"
    if not path.exists():
        _fetch(f"{HF_BASE}/manifest.json", path)
    return json.loads(path.read_text())


def part_path(cache: Path, part: int) -> Path:
    """Download (if needed) and verify one replays part."""
    rel = f"replays/part-{part:06d}.parquet"
    dest = cache / rel
    want = {f["path"]: f["sha256"] for f in dataset_manifest(cache)["files"]}.get(rel)
    if not dest.exists():
        logger.info("downloading %s", rel)
        _fetch(f"{HF_BASE}/{rel}", dest)
    if want and _sha256(dest) != want:
        dest.unlink()
        raise RuntimeError(f"{rel}: SHA-256 mismatch (deleted, re-run to download again)")
    return dest


def iter_replays(path: Path, batch_rows: int = 200) -> Iterator[Replay]:
    import pyarrow.parquet as pq
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_rows, columns=["replay_tag", "payload_json"]):
        tags, payloads = batch.column(0).to_pylist(), batch.column(1).to_pylist()
        for tag, raw in zip(tags, payloads):
            rep = parse_payload(json.loads(raw), tag)
            if rep is not None:
                yield rep
        del tags, payloads, batch


def parse_parts(spec: str) -> List[int]:
    out: List[int] = []
    for chunk in spec.split(","):
        a, _, b = chunk.partition("-")
        out.extend(range(int(a), int(b or a) + 1))
    return [p for p in out if 0 <= p < N_PARTS]


def survey(paths: Iterable[Path], db: CardDB) -> dict:
    """Coverage / yield statistics without simulating anything."""
    n, cov_hist = 0, collections.Counter()
    use = collections.Counter()
    plays_by_rule = collections.Counter()
    quarantine = CardQuarantine(db)
    for path in paths:
        for rep in iter_replays(path):
            n += 1
            quarantine.check([*rep.decks[0], *rep.decks[1]])
            cov = [sum(c in db.cards for c in d) for d in rep.decks]
            cov_hist.update(cov)
            sides_plays = [sum(1 for e in rep.events if e.side == s and not e.ability) for s in (0, 1)]
            for mc in (8, 7):
                for mo in (8, 6, 4, 0):
                    ls = [s for s in (0, 1) if cov[s] >= mc and cov[1 - s] >= mo]
                    if ls:
                        use[f"learner>={mc} opp>={mo}"] += 1
                        plays_by_rule[f"learner>={mc} opp>={mo}"] += sum(sides_plays[s] for s in ls)
    return {"replays": n, "deck_coverage_hist": dict(sorted(cov_hist.items())),
            "replays_used": dict(use), "learner_plays": dict(plays_by_rule), "quarantine": quarantine.as_dict()}


def main(argv: Optional[Sequence[str]] = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--output-dir", default=str(ROOT / "data" / "il_shards"))
    ap.add_argument("--cache-dir", default=str(ROOT / "data" / "il_raw"))
    ap.add_argument("--parts", default=f"0-{N_PARTS - 1}", help="replay parts to read, e.g. 0-3 or 0,5,7")
    ap.add_argument("--max-samples", type=int, default=100_000, help="stop after this many samples")
    ap.add_argument("--max-replays", type=int, default=0, help="stop after this many replays read (0 = no limit)")
    ap.add_argument("--shard-size", type=int, default=5000)
    ap.add_argument("--min-coverage", type=int, default=8, help="simulated cards a learner deck needs (8 = all)")
    ap.add_argument("--min-opp-coverage", type=int, default=4, help="simulated cards the other deck needs")
    ap.add_argument("--wait-ratio", type=float, default=1.0)
    ap.add_argument("--max-elixir-deficit", type=float, default=1.0)
    ap.add_argument("--keep-raw", action="store_true", help="keep downloaded parquet parts in the cache")
    ap.add_argument("--survey", action="store_true", help="only report coverage / yield of the parts")
    ap.add_argument("--overwrite", action="store_true", help="replace existing shards in --output-dir")
    ap.add_argument("--strict-quarantine", action="store_true",
                    help="also skip replays with catalog cards that are not playable yet (default: proxy them)")
    args = ap.parse_args(argv)

    db = CardDB.load()
    cache = Path(args.cache_dir)
    parts = parse_parts(args.parts)
    if args.survey:
        paths = [part_path(cache, p) for p in parts]
        print(json.dumps(survey(paths, db), indent=2))
        return

    out = Path(args.output_dir)
    if out.exists() and any(out.glob("shard_*.npz")):
        if not args.overwrite:
            raise SystemExit(f"{out} already has shards (use --overwrite)")
        for p in out.glob("shard_*.npz"):
            p.unlink()
    writer = ShardWriter(out, args.shard_size)
    stats = Stats()
    proxies = ProxyMap(db, load_card_catalog())
    guesser = default_guesser()
    quarantine = CardQuarantine(db, strict=args.strict_quarantine)
    t0 = time.time()
    done_parts: List[int] = []
    battle_idx = 0
    stop = False
    for part in parts:
        path = part_path(cache, part)
        for rep in iter_replays(path):
            replay_battle(rep, db, writer, battle_idx, stats, proxies, guesser, args.min_coverage,
                          args.min_opp_coverage, args.wait_ratio, args.max_elixir_deficit, quarantine)
            battle_idx += 1
            if stats.replays_seen % 500 == 0:
                logger.info("part %d: %d replays read, %d used, %d samples (%.0f s)", part, stats.replays_seen,
                            stats.replays_used, writer.total_samples, time.time() - t0)
            if writer.total_samples >= args.max_samples or (args.max_replays and stats.replays_seen >= args.max_replays):
                stop = True
                break
        done_parts.append(part)
        if not args.keep_raw:
            path.unlink(missing_ok=True)
        if stop:
            break
    extra = {
        "source": {"dataset": HF_REPO, "parts": done_parts, "last_part_complete": not stop},
        "filters": {"min_coverage": args.min_coverage, "min_opp_coverage": args.min_opp_coverage,
                    "wait_ratio": args.wait_ratio, "max_elixir_deficit": args.max_elixir_deficit},
        "stats": stats.as_dict(),
        "quarantine": quarantine.as_dict(),
        "build_seconds": round(time.time() - t0, 1),
        "card_pool": sorted(db.cards),          # informational: the encoder does not depend on it
    }
    path = writer.close(extra)
    if quarantine.summary():
        logger.warning(quarantine.summary())
    logger.info("done: %d samples (%d plays, %d waits) from %d battles -> %s", writer.total_samples,
                writer.total_plays, writer.total_waits, writer.total_battles, path)


if __name__ == "__main__":
    main()
