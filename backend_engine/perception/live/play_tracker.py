"""
Turns noisy per-frame detections into card plays (PublicDeployEvent).

The detector sees *units*, not plays. A play is the first appearance of a new unit
(or group of units: Skeletons = 3, Archers = 2) that is confirmed over a few frames.

Rules
  * Tracks: each detection is matched to the nearest live track of the same card+side
    within `match_radius` tiles; unmatched detections start new tracks.
  * A new track becomes "real" after `confirm_hits` frames (kills one-frame false positives).
  * Group: a confirmed track joins an existing play of the same card+side if that play
    started within `group_ms` and `group_radius`, and the play still has room
    (units_per_play from the card's `count`). Otherwise it is a new play -> event.
  * Re-identification: a unit that disappears briefly (occluded by a spell, the detector
    blinked) and comes back near where it vanished revives its old track instead of
    being counted as a second play.
  * Cycle rule (the game's own rule, the strongest filter): a card can only be played again
    after `hand_size` other plays. A 'new' unit of a card that can't be back in hand yet is
    a duplicate of the previous play (lost track in a crowd). Conversely, if the card *is*
    back in hand and a unit reappears after a gap >= reborn_gap_ms, it is a fresh copy
    (e.g. a new Cannon on the tile where the old one just expired).
  * The event time is when the unit was *first* seen, not when it was confirmed.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

from backend_engine.contracts import PublicDeployEvent, TilePos

TileDetection = Tuple[str, Optional[int], float, float]  # card_id, side (None = unknown), x, y (tiles)


@dataclass
class _Track:
    tid: int
    card: str
    side: int
    x: float
    y: float
    first_ms: int
    last_ms: int
    hits: int = 1
    play: Optional[int] = None


@dataclass
class _Play:
    event: PublicDeployEvent
    units: int = 1


@dataclass
class PlayTracker:
    costs: Dict[str, int]
    units_per_play: Dict[str, int] = field(default_factory=dict)
    match_radius: float = 1.75
    confirm_hits: int = 2
    expire_ms: int = 1200
    group_ms: int = 900
    group_radius: float = 3.0
    reid_ms: int = 600
    reid_radius: float = 2.5
    hand_size: int = 4
    reborn_gap_ms: int = 800
    lifetimes_ms: Dict[str, int] = field(default_factory=dict)   # buildings: card -> lifetime
    lifetime_margin_ms: int = 200
    # units that other cards also produce: card -> parent cards (Skeletons <- Graveyard, Witch, ...)
    spawned_by: Dict[str, List[str]] = field(default_factory=dict)
    spawn_radius: float = 7.0
    spawn_window_ms: int = 15_000
    spells: frozenset = frozenset()           # card ids that are spells (side can't come from position)
    river_y0: float = 15.0
    river_y1: float = 17.0

    def __post_init__(self) -> None:
        self.tracks: List[_Track] = []
        self.lost: List[_Track] = []
        self.plays: List[_Play] = []
        self._tid = 0
        self._eid = 1
        self.suppressed = 0   # duplicates rejected by the cycle rule (diagnostics)
        self.spawns = 0       # units attributed to a spawner card instead of counted as plays

    def update(self, t_ms: int, detections: Sequence[TileDetection]) -> List[PublicDeployEvent]:
        # 1) match detections to live tracks (greedy by distance, deterministic)
        pairs = []
        for di, (card, side, x, y) in enumerate(detections):
            for tr in self.tracks:
                if tr.card == card and (side is None or tr.side == side):
                    if self._reborn(tr, t_ms):
                        continue   # long gap + card back in hand: a fresh copy, not the old unit
                    d = math.hypot(tr.x - x, tr.y - y)
                    if d <= self.match_radius:
                        pairs.append((d, di, tr.tid))
        pairs.sort()
        used_d, used_t = set(), set()
        by_id = {tr.tid: tr for tr in self.tracks}
        for _, di, tid in pairs:
            if di in used_d or tid in used_t:
                continue
            used_d.add(di)
            used_t.add(tid)
            tr = by_id[tid]
            _, _, tr.x, tr.y = detections[di]
            tr.last_ms, tr.hits = t_ms, tr.hits + 1

        # 2) unmatched detections: revive a recently lost track or start a new one
        for di, (card, side, x, y) in enumerate(detections):
            if di in used_d:
                continue
            if side is None:
                side = self._side_from_position(card, y)
                if side is None:
                    continue   # first seen on the river: wait until it is clearly on one half
            revived = self._reidentify(card, side, x, y, t_ms)
            if revived is not None:
                revived.x, revived.y, revived.last_ms = x, y, t_ms
                revived.hits += 1
                self.tracks.append(revived)
                continue
            self._tid += 1
            self.tracks.append(_Track(self._tid, card, side, x, y, t_ms, t_ms))

        # 3) expire stale tracks
        alive = []
        for tr in self.tracks:
            if t_ms - tr.last_ms > self.expire_ms:
                if tr.play is not None:
                    self.lost.append(tr)
            else:
                alive.append(tr)
        self.tracks = alive
        self.lost = [tr for tr in self.lost if t_ms - tr.last_ms <= self.reid_ms]

        # 3b) a decaying building still visible after its lifetime must be a new copy on the
        #     same tile (players re-place a Cannon the moment the old one expires)
        for tr in self.tracks:
            life = self.lifetimes_ms.get(tr.card)
            if (life and tr.play is not None and tr.last_ms == t_ms
                    and t_ms - tr.first_ms >= life + self.lifetime_margin_ms
                    and self._cycle_impossible(tr) is None):
                tr.first_ms, tr.play = t_ms - self.lifetime_margin_ms, None

        # 4) confirmed new tracks -> plays
        new_events = []
        for tr in sorted(self.tracks, key=lambda t: (t.first_ms, t.tid)):
            if tr.play is not None or tr.hits < self.confirm_hits:
                continue
            play = self._group_with(tr)
            if play is not None:
                play.units += 1
                tr.play = self.plays.index(play)
                continue
            parent = self._spawn_parent(tr)
            if parent is not None:  # e.g. a skeleton next to a live Witch / fresh Graveyard
                tr.play = parent
                self.spawns += 1
                continue
            dup = self._cycle_impossible(tr)
            if dup is not None:   # can't be a new play: the card is still cycling -> a duplicate unit
                tr.play = dup
                self.suppressed += 1
                continue
            ev = PublicDeployEvent(self._eid, tr.first_ms, tr.side, tr.card, self.costs.get(tr.card, 0),
                                   TilePos(round(tr.x, 2), round(tr.y, 2)),
                                   is_threat=self._is_threat(tr))
            self._eid += 1
            self.plays.append(_Play(ev))
            tr.play = len(self.plays) - 1
            new_events.append(ev)
        return new_events

    def _side_from_position(self, card: str, y: float) -> Optional[int]:
        """Owner of a unit whose colour we couldn't read: troops and buildings can only be
        deployed on their owner's half, so where a unit first appears tells whose it is.
        Spells can land anywhere; heuristic: a spell is aimed at the enemy, so one landing on
        your half was most likely the opponent's (and vice versa). Less certain than for troops."""
        if card in self.spells:
            return 1 if y < (self.river_y0 + self.river_y1) / 2 else 0
        if y >= self.river_y1:
            return 1
        if y < self.river_y0:
            return 0
        return None

    def _spawn_parent(self, tr: _Track) -> Optional[int]:
        parents = set(self.spawned_by.get(tr.card, ()))
        if not parents:
            return None
        for other in self.tracks:   # a live spawner nearby (Witch, Tombstone, Goblin Hut, ...)
            if (other.card in parents and other.side == tr.side and other.play is not None
                    and math.hypot(other.x - tr.x, other.y - tr.y) <= self.spawn_radius):
                return other.play
        for i in range(len(self.plays) - 1, -1, -1):   # a recent spawner play (Graveyard, Barrel, ...)
            ev = self.plays[i].event
            if tr.first_ms - ev.t_ms > self.spawn_window_ms:
                break
            if (ev.card_id in parents and ev.side == tr.side
                    and math.hypot(ev.tile.x - tr.x, ev.tile.y - tr.y) <= self.spawn_radius):
                return i
        return None

    def _reborn(self, tr: _Track, t_ms: int) -> bool:
        return t_ms - tr.last_ms >= self.reborn_gap_ms and self._cycle_impossible(tr) is None

    def _cycle_impossible(self, tr: _Track) -> Optional[int]:
        """Clash Royale cycle rule: after playing a card, `hand_size` other cards must be played
        before it is back in hand. If that hasn't happened, this 'new' unit belongs to the
        previous play of the card (e.g. a skeleton whose track was lost in a crowd).
        Returns that play's index, or None if a new play is possible."""
        since = 0
        for i in range(len(self.plays) - 1, -1, -1):
            ev = self.plays[i].event
            if ev.side != tr.side:
                continue
            if ev.card_id == tr.card:
                return i if since < self.hand_size else None
            since += 1
        return None

    def _reidentify(self, card: str, side: int, x: float, y: float, t_ms: int) -> Optional[_Track]:
        best, best_d = None, self.reid_radius
        for tr in self.lost:
            if tr.card == card and tr.side == side and t_ms - tr.last_ms <= self.reid_ms \
                    and not self._reborn(tr, t_ms):
                d = math.hypot(tr.x - x, tr.y - y)
                if d <= best_d:
                    best, best_d = tr, d
        if best is not None:
            self.lost.remove(best)
        return best

    def _group_with(self, tr: _Track) -> Optional[_Play]:
        cap = self.units_per_play.get(tr.card, 1)
        for play in reversed(self.plays):
            ev = play.event
            if tr.first_ms - ev.t_ms > self.group_ms:
                break
            if (ev.card_id == tr.card and ev.side == tr.side and play.units < cap
                    and math.hypot(ev.tile.x - tr.x, ev.tile.y - tr.y) <= self.group_radius):
                return play
        return None

    @staticmethod
    def _is_threat(tr: _Track) -> bool:
        # threat to the *other* side: deployed near the river or across it
        return tr.y < 19.0 if tr.side == 1 else tr.y > 13.0

    @property
    def events(self) -> List[PublicDeployEvent]:
        return [p.event for p in self.plays]
