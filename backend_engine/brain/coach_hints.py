"""Mode B coach hints: what crbot_il_v1 would play in a real match (live or replayed from a play log).

Mode B (read-only screen reading) knows: the plays of both sides (PublicDeployEvent, from the PlayTracker or a
saved log), the units on screen right now (confirmed tracks: card, side, x, y), the match clock, and therefore
both elixir counts (the same OppElixirTracker the simulator uses). It does not see your hand, unit / tower hp or
crowns. This module turns that into the simulator's `observation()` dict and encodes it with the very same
`il_features.encode_observation` used for training and by NeuralBot, then asks the model.

Your hand comes from the card cycle: with your 8-card deck known, after >= 4 plays your hand is exactly
deck - last 4 plays (and the next card is the 4th-latest play). Before that every possible hand is scored in one
batched model call and the results are averaged (the hand is uniformly unknown). Unknowns get neutral defaults:
all towers up at full hp unless `towers` says otherwise, units at full hp, crowns 0-0 unless given.

Output (CoachHints): the model's play probability now, its win-probability estimate, and the top cards (default 2)
with the best legal tile and a value delta = value(after the play) - value(now). "After" is the same snapshot
with the elixir spent, the card's unit placed (deploying) and the hand cycled, scored by the value head: a quick
what-if from one network call, not a simulation (spells add no unit, so their delta is mostly the elixir).

Coordinates: Mode B tiles are your frame (you at the bottom, y = 0), which is engine side 0.
"""
from __future__ import annotations

import itertools
import math
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from backend_engine.brain import il_features as F
from backend_engine.brain.neural_bot import NeuralBot, load_neural_bot
from backend_engine.contracts import ElixirEstimate, PublicDeployEvent, TilePos
from backend_engine.perception.bayesian_deck_guesser import default_guesser, known_costs
from backend_engine.perception.opp_elixir_count import OppElixirTracker
from backend_engine.simulator import geometry as G
from backend_engine.simulator.card_db import CardDB

YOU, OPP = 0, 1
UNIT_FRESH_MS = 300           # live tracks seen within this long count as on screen
DEPLOY_MS = 1000              # a unit first seen less than this ago is still deploying
TOWER_ROLES = ("king", "princess_left", "princess_right")
_AIR = {"baby_dragon", "balloon", "electro_dragon", "flying_machine", "inferno_dragon", "lava_hound",
        "mega_minion", "minion_horde", "phoenix", "skeleton_dragons", "lava_pups"}


@dataclass(frozen=True)
class SeenUnit:
    card: str
    side: int
    x: float
    y: float
    first_ms: int
    hp_frac: float = 1.0


@dataclass
class ModeBSnapshot:
    """Everything Mode B knows at time t_ms (your frame: you are side 0 at the bottom)."""
    t_ms: int
    my_deck: Tuple[str, ...]
    events: List[PublicDeployEvent]                         # both sides, any order
    units: List[SeenUnit] = field(default_factory=list)
    my_evo: Tuple[str, ...] = ()
    towers: Dict[Tuple[int, str], float] = field(default_factory=dict)   # (side, role) -> hp fraction (0 = down)
    crowns: Tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class PlayHint:
    card: str
    prob: float                   # model's probability of this card among your affordable cards
    tile: int                     # tile index in your frame (gy * 18 + gx)
    x: float
    y: float
    value_after: float
    value_delta: float            # value_after - value_now


@dataclass
class CoachHints:
    t_ms: int
    p_play: float                 # model's probability that you should play a card now (vs wait)
    value_now: float              # model's win probability for you
    elixir: float
    hand: Optional[Tuple[str, ...]]   # None while the hand is still uncertain
    hand_hypotheses: int
    hints: List[PlayHint]
    note: str = ""

    @property
    def wait(self) -> bool:
        return self.p_play < 0.5


# -- snapshots ---------------------------------------------------------------------------------------------
def snapshot_from_companion(comp, my_deck: Sequence[str], wall: Optional[float] = None,
                            my_evo: Sequence[str] = ()) -> Optional[ModeBSnapshot]:
    """Live: the companion's play log + its confirmed tracks seen in the last UNIT_FRESH_MS."""
    t = comp.clock.now_ms(wall)
    if t is None:
        return None
    tr = comp.tracker
    units = [SeenUnit(k.card, k.side, k.x, k.y, k.first_ms) for k in tr.tracks
             if k.hits >= tr.confirm_hits and k.side in (YOU, OPP) and t - k.last_ms <= UNIT_FRESH_MS]
    return ModeBSnapshot(t, tuple(my_deck), list(comp.events), units, tuple(my_evo))


def snapshot_from_log(plays: Iterable, t_ms: int, my_deck: Sequence[str], my_evo: Sequence[str] = (),
                      unit_ttl_ms: int = 4000, db: Optional[CardDB] = None) -> ModeBSnapshot:
    """Review: a saved play log (LiveCompanion.save_log entries or PublicDeployEvents) up to t_ms. Units are
    approximated as the non-spell plays of the last `unit_ttl_ms`, standing where they were dropped."""
    db = db or CardDB.load()
    costs = known_costs()
    events: List[PublicDeployEvent] = []
    for i, p in enumerate(plays):
        if isinstance(p, PublicDeployEvent):
            ev = p
        else:
            card = p["card"]
            ev = PublicDeployEvent(i + 1, int(p["t_ms"]), int(p["side"]), card,
                                   int(p.get("cost", db.cards.get(card, {}).get("elixir", costs.get(card, 0)))),
                                   TilePos(float(p["x"]), float(p["y"])))
        if ev.t_ms <= t_ms:
            events.append(ev)
    units = [SeenUnit(e.card_id, e.side, e.tile.x, e.tile.y, e.t_ms) for e in events
             if not e.is_ability and t_ms - e.t_ms <= unit_ttl_ms and _card_type(db, e.card_id) != "spell"]
    return ModeBSnapshot(t_ms, tuple(my_deck), events, units, tuple(my_evo))


# -- your hand from the cycle --------------------------------------------------------------------------------
def hand_hypotheses(deck: Sequence[str], my_plays: Sequence[str]) -> Tuple[List[Tuple[str, ...]], str]:
    """Possible hands (sorted card tuples) and the next card ('' if unknown) from your 8-card deck and the
    order of your plays. The queue holds the last 4 plays, so after 4 plays the hand is fixed."""
    deck = list(dict.fromkeys(deck))
    plays = [c for c in my_plays if c in deck]
    queue: List[str] = []
    for c in plays:                                         # cycle: a played card goes to the back of the queue
        if c in queue:
            queue.remove(c)                                 # (a detector miss made it look early: keep it unique)
        queue.append(c)
    queue = queue[-4:]
    rest = [c for c in deck if c not in queue]
    if len(queue) == 4 or len(rest) <= 4:
        return [tuple(sorted(rest[:4]))], (queue[0] if len(queue) == 4 else "")
    return [tuple(sorted(h)) for h in itertools.combinations(sorted(rest), 4)], ""


def _evo_counter(my_evo: Sequence[str], my_plays: Sequence[str], db: CardDB) -> Dict[str, int]:
    counter = {c: 0 for c in my_evo if db.has_variant(c, "evo")}
    for c in my_plays:
        if c in counter:
            counter[c] = 0 if counter[c] >= db.evo_cycles(c) else counter[c] + 1
    return counter


def _card_type(db: CardDB, card: str) -> str:
    try:
        return db.entry(card)["type"]
    except KeyError:                                        # a real card the simulator doesn't have
        from backend_engine.perception.live.companion import KNOWN_SPELLS
        return "spell" if card in KNOWN_SPELLS else "troop"


# -- observation ---------------------------------------------------------------------------------------------
class ModeBEncoder:
    """Mode B snapshot -> simulator-style observation dict -> il_features vector (side 0 = you)."""

    def __init__(self, db: Optional[CardDB] = None, guesser="default"):
        self.db = db or CardDB.load()
        self.guesser = default_guesser() if guesser == "default" else guesser
        self.vocab = F.card_vocab(self.db)
        r = self.db.rules
        self._phases = sorted((int(p["from_s"] * 1000), int(p["multiplier"])) for p in r["elixir_phases"])
        self._reg = int(r["regulation_s"] * 1000)
        self._end = self._reg + int(r["overtime_s"] * 1000)

    def multiplier(self, t: int) -> int:
        m = 1
        for start, mult in self._phases:
            if t >= start:
                m = mult
        return m

    def estimates(self, snap: ModeBSnapshot) -> Tuple[ElixirEstimate, ElixirEstimate, object]:
        """(opponent estimate, your own elixir estimate, deck guess) from the play log, like LiveCompanion."""
        opp = OppElixirTracker(self.db.rules, observer_side=YOU, evo_cards=self.db.evo_capable())
        me = OppElixirTracker(self.db.rules, observer_side=OPP, evo_cards=self.db.evo_capable())
        for ev in sorted(snap.events, key=lambda e: (e.t_ms, e.event_id)):
            opp.observe(ev)
            me.observe(ev)
        est_opp, est_me = opp.estimate(snap.t_ms), me.estimate(snap.t_ms)
        guess = self.guesser.guess(est_opp.known_cards) if self.guesser else None
        return est_opp, est_me, guess

    def _units(self, snap: ModeBSnapshot) -> List[dict]:
        out = []
        downed = {s: [r for r in TOWER_ROLES[1:] if snap.towers.get((s, r), 1.0) <= 0] for s in (YOU, OPP)}
        uid = 1
        for s in (YOU, OPP):
            for role in TOWER_ROLES:
                hp = float(snap.towers.get((s, role), 1.0))
                if hp <= 0:
                    continue
                x, y, rad = G.tower_pos(s, role)
                active = role != "king" or bool(downed[s]) or hp < 1.0
                out.append({"uid": uid, "side": s, "card_id": "king_tower" if role == "king" else "princess_tower",
                            "kind": "tower", "layer": "ground", "x": x, "y": y, "hp": 100.0 * hp, "max_hp": 100.0,
                            "deploying": False, "tower_role": role, "active": active, "targets": (), "radius": rad})
                uid += 1
        for u in snap.units:
            if u.side not in (YOU, OPP) or not G.in_bounds(u.x, u.y):
                continue
            entry = self.db.cards.get(u.card) or self.db.tokens.get(u.card) or {}
            kind = entry.get("type", "troop")
            if kind == "spell":
                continue
            layer = (entry.get("movement") or {}).get("layer") or ("air" if u.card in _AIR else "ground")
            out.append({"uid": uid, "side": u.side, "card_id": u.card, "kind": kind, "layer": layer,
                        "x": u.x, "y": u.y, "hp": 100.0 * u.hp_frac, "max_hp": 100.0,
                        "deploying": snap.t_ms - u.first_ms < DEPLOY_MS, "tower_role": None, "active": True,
                        "targets": (), "radius": float(entry.get("collision_radius_tiles", 0.5))})
            uid += 1
        return out

    def observation(self, snap: ModeBSnapshot, hand: Sequence[str], next_card: str, elixir: float,
                    evo_counter: Mapping[str, int]) -> dict:
        t = snap.t_ms
        return {"side": YOU, "t_ms": t, "multiplier": self.multiplier(t), "overtime": t >= self._reg,
                "time_left_ms": (self._reg if t < self._reg else self._end) - t, "elixir": elixir,
                "hand": list(hand), "next_card": next_card, "evo_counter": dict(evo_counter), "level": 11,
                "crowns": tuple(snap.crowns), "units": self._units(snap), "spells": [], "abilities": [],
                "ability_info": {}}


# -- hints ---------------------------------------------------------------------------------------------------
def _sigmoid(x: float) -> float:
    return 1.0 / (1.0 + math.exp(-x))


def _softmax(x: np.ndarray, mask: np.ndarray) -> np.ndarray:
    z = np.where(mask, x, -np.inf)
    if not mask.any():
        return np.zeros_like(x, dtype=np.float64)
    z = z - z[mask].max()
    e = np.where(mask, np.exp(z), 0.0)
    return e / e.sum()


def _after_play(obs: dict, card: str, xy: Tuple[float, float], db: CardDB) -> dict:
    """The same observation right after you play `card` at `xy` (elixir spent, unit placed, hand cycled)."""
    new = dict(obs)
    meta = db.cards[card]
    new["elixir"] = max(obs["elixir"] - meta["elixir"], 0.0)
    hand = list(obs["hand"])
    hand[hand.index(card)] = obs.get("next_card") or ""
    new["hand"] = hand
    new["next_card"] = ""                                    # the next card after that is not known from here
    units = list(obs["units"])
    if meta["type"] != "spell":
        layer = (meta.get("movement") or {}).get("layer", "ground")
        units.append({"uid": 10_000, "side": YOU, "card_id": card, "kind": meta["type"], "layer": layer,
                      "x": xy[0], "y": xy[1], "hp": 100.0, "max_hp": 100.0, "deploying": True, "tower_role": None,
                      "active": True, "targets": (), "radius": float(meta.get("collision_radius_tiles", 0.5))})
    new["units"] = units
    return new


class CoachHinter:
    """crbot_il_v1 as a coach for Mode B. `hints(snapshot)` -> CoachHints (None-free: falls back to notes)."""

    def __init__(self, bot: NeuralBot, encoder: Optional[ModeBEncoder] = None):
        self.bot = bot
        self.db = bot.db
        self.encoder = encoder or ModeBEncoder(self.db)
        self._input = bot.session.get_inputs()[0].name

    @classmethod
    def load(cls, folder: Optional[Path] = None, db: Optional[CardDB] = None) -> "CoachHinter":
        """Raises neural_bot.ModelUnavailable when there is no usable model."""
        db = db or CardDB.load()
        return cls(load_neural_bot(db, YOU, folder), ModeBEncoder(db))

    def _run(self, vecs: np.ndarray):
        return self.bot.session.run(["gate_logit", "slot_logits", "tile_logits", "value"], {self._input: vecs})

    def hints(self, snap: ModeBSnapshot, top_k: int = 2) -> CoachHints:
        db, enc = self.db, self.encoder
        deck = [c for c in snap.my_deck]
        unknown = [c for c in deck if c not in db.cards]
        mine = [e for e in sorted(snap.events, key=lambda e: (e.t_ms, e.event_id)) if e.side == YOU and not e.is_ability]
        my_plays = [e.card_id for e in mine]
        est_opp, est_me, guess = enc.estimates(snap)
        elixir = float(est_me.point)
        hands, next_card = hand_hypotheses(deck, my_plays)
        evo = _evo_counter(snap.my_evo, my_plays, db)
        note = ""
        if len(set(deck)) != 8:
            note = "your deck must list 8 different cards"
        elif unknown:
            note = f"not simulated, never suggested: {', '.join(unknown)}"
        obs_list = [enc.observation(snap, [c if c in db.cards else "" for c in h], next_card, elixir, evo)
                    for h in hands]
        vecs = np.stack([F.encode_observation(o, est_opp, guess, db, self.bot.vocab) for o in obs_list])
        gate, slots, tiles, value = self._run(vecs)
        p_play = float(np.mean([_sigmoid(float(g)) for g in np.ravel(gate)]))
        value_now = float(np.mean(np.ravel(value)))

        card_prob: Dict[str, float] = {}
        tile_logp: Dict[str, List[np.ndarray]] = {}
        for i, o in enumerate(obs_list):
            smask = F.slot_mask(o, db)
            probs = _softmax(slots[i].astype(np.float64), smask)
            for k, c in enumerate(o["hand"]):
                if not smask[k]:
                    continue
                card_prob[c] = card_prob.get(c, 0.0) + probs[k] / len(obs_list)
                row = tiles[i, k].astype(np.float64)
                tile_logp.setdefault(c, []).append(row - (row.max() + np.log(np.exp(row - row.max()).sum())))
        ranked = sorted(card_prob.items(), key=lambda kv: (-kv[1], kv[0]))[:top_k]

        hints: List[PlayHint] = []
        if ranked:
            picks = []
            after_obs = []
            for card, p in ranked:
                mask = F.tile_mask(obs_list[0], card, db)
                if not mask.any():
                    continue
                score = np.mean(tile_logp[card], axis=0)
                tile = int(np.argmax(np.where(mask, score, -np.inf)))
                xy = F.tile_center(YOU, tile)
                picks.append((card, p, tile, xy))
                after_obs.extend(_after_play(o, card, xy, db) for o in obs_list if card in o["hand"])
            if picks:
                after_vecs = np.stack([F.encode_observation(o, est_opp, guess, db, self.bot.vocab) for o in after_obs])
                after_val = np.ravel(self._run(after_vecs)[3])
                j = 0
                for card, p, tile, xy in picks:
                    n = sum(1 for o in obs_list if card in o["hand"])
                    v = float(np.mean(after_val[j:j + n]))
                    j += n
                    hints.append(PlayHint(card, round(float(p), 4), tile, xy[0], xy[1], round(v, 4),
                                          round(v - value_now, 4)))
        elif not note:
            note = "no card affordable"
        return CoachHints(snap.t_ms, round(p_play, 4), round(value_now, 4), round(elixir, 2),
                          hands[0] if len(hands) == 1 else None, len(hands), hints, note)
