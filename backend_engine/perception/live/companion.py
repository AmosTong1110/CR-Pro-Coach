"""
LiveCompanion: the Qt-free brain of the Mode B overlay.

    comp = LiveCompanion.create(detector=YoloOnnxDetector(...), calibration=ArenaCalibration.load())
    comp.start_match()                # press when the battle starts (the clock drives x2 / x3)
    state = comp.process_frame(frame, wall_time)

Everything after detection is the same code the simulator uses (OppElixirTracker,
BayesianDeckGuesser), so what you practise against in Mode A is what you see here.
"""
from __future__ import annotations

import json
import random
import time
from collections import deque
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Deque, Dict, List, Optional, Sequence, Tuple

import numpy as np

from backend_engine.contracts import ElixirEstimate, PublicDeployEvent, TilePos
from backend_engine.perception.bayesian_deck_guesser import DeckGuess, default_guesser, known_costs
from backend_engine.perception.live.calibration import ArenaCalibration
from backend_engine.perception.live.detector import Detector
from backend_engine.perception.live.play_tracker import PlayTracker, TileDetection
from backend_engine.perception.opp_elixir_count import OppElixirTracker
from backend_engine.simulator.card_db import CardDB

YOU, OPP = 0, 1


class LiveClock:
    """Match clock from wall time. Start it when the battle starts; nudge to resync."""

    def __init__(self) -> None:
        self.start_wall: Optional[float] = None
        self.offset_ms = 0

    @property
    def running(self) -> bool:
        return self.start_wall is not None

    def start(self, wall: Optional[float] = None) -> None:
        self.start_wall = time.monotonic() if wall is None else wall
        self.offset_ms = 0

    def stop(self) -> None:
        self.start_wall = None

    def nudge(self, ms: int) -> None:
        self.offset_ms += ms

    def now_ms(self, wall: Optional[float] = None) -> Optional[int]:
        if self.start_wall is None:
            return None
        wall = time.monotonic() if wall is None else wall
        return max(0, int(round((wall - self.start_wall) * 1000)) + self.offset_ms)


@dataclass
class CompanionState:
    t_ms: Optional[int]
    clock_text: str
    phase_text: str
    opp: Optional[ElixirEstimate]
    me: Optional[ElixirEstimate]
    guess: Optional[DeckGuess]
    recent_plays: List[PublicDeployEvent]
    fps: float
    last_detections: int
    warnings: List[str] = field(default_factory=list)

    @property
    def advantage(self) -> Optional[float]:
        if self.opp is None or self.me is None:
            return None
        return self.me.point - self.opp.point


class LiveCompanion:
    def __init__(self, rules: dict, costs: Dict[str, int], units_per_play: Dict[str, int],
                 calibration: Optional[ArenaCalibration] = None, detector: Optional[Detector] = None,
                 guesser=None, min_conf: float = 0.0):
        self.rules = rules
        self.costs = costs
        self.units_per_play = units_per_play
        self.calibration = calibration
        self.detector = detector
        self.guesser = guesser
        self.min_conf = min_conf
        self.clock = LiveClock()
        self._frame_times: Deque[float] = deque(maxlen=30)
        self._last_dets = 0
        self.reset()

    @classmethod
    def create(cls, detector: Optional[Detector] = None, calibration: Optional[ArenaCalibration] = None,
               db: Optional[CardDB] = None) -> "LiveCompanion":
        db = db or CardDB.load()
        costs = {**known_costs(), **db.elixir_costs()}
        units = {cid: int(c.get("count", 1)) for cid, c in db.cards.items()}
        return cls(db.rules, costs, units, calibration, detector, default_guesser())

    def reset(self) -> None:
        self.tracker = PlayTracker(self.costs, self.units_per_play)
        self.opp_tracker = OppElixirTracker(self.rules, observer_side=YOU)   # counts the opponent
        self.me_tracker = OppElixirTracker(self.rules, observer_side=OPP)    # counts you
        self.events: List[PublicDeployEvent] = []

    # -- control -----------------------------------------------------------
    def start_match(self, wall: Optional[float] = None) -> None:
        self.reset()
        self.clock.start(wall)

    def stop_match(self) -> None:
        self.clock.stop()

    # -- input -------------------------------------------------------------
    def process_frame(self, frame: np.ndarray, wall: Optional[float] = None) -> List[PublicDeployEvent]:
        if self.detector is None or self.calibration is None:
            return []
        h, w = frame.shape[:2]
        tiles: List[TileDetection] = []
        for d in self.detector.detect(frame):
            if d.side is None or d.conf < self.min_conf:
                continue
            x, y = self.calibration.px_to_tile(*d.foot, frame_w=w, frame_h=h)
            if ArenaCalibration.in_arena(x, y):
                tiles.append((d.card_id, d.side, x, y))
        return self.process_tiles(tiles, wall)

    def process_tiles(self, detections: Sequence[TileDetection], wall: Optional[float] = None) -> List[PublicDeployEvent]:
        wall = time.monotonic() if wall is None else wall
        self._frame_times.append(wall)
        self._last_dets = len(detections)
        t = self.clock.now_ms(wall)
        if t is None:
            return []
        new = self.tracker.update(t, detections)
        for ev in new:
            self._add(ev)
        return new

    def manual_play(self, card_id: str, side: int = OPP, wall: Optional[float] = None,
                    x: float = 9.0, y: float = 24.0) -> Optional[PublicDeployEvent]:
        """Log a play by hand (detector missed it, or replaying a recording)."""
        t = self.clock.now_ms(wall)
        if t is None:
            return None
        ev = PublicDeployEvent(10_000 + len(self.events), t, side, card_id, self.costs.get(card_id, 0), TilePos(x, y))
        self._add(ev)
        return ev

    def _add(self, ev: PublicDeployEvent) -> None:
        self.events.append(ev)
        self.opp_tracker.observe(ev)
        self.me_tracker.observe(ev)

    # -- output ------------------------------------------------------------
    def state(self, wall: Optional[float] = None) -> CompanionState:
        t = self.clock.now_ms(wall)
        fps = 0.0
        if len(self._frame_times) >= 2 and self._frame_times[-1] > self._frame_times[0]:
            fps = (len(self._frame_times) - 1) / (self._frame_times[-1] - self._frame_times[0])
        warnings = []
        if self.detector is None:
            warnings.append("No detector model loaded")
        if self.calibration is None or not self.calibration.calibrated:
            warnings.append("Arena not calibrated")
        if t is None:
            return CompanionState(None, "--:--", "Press Start when the battle begins", None, None, None,
                                  self.events[-6:], fps, self._last_dets, warnings)
        opp = self.opp_tracker.estimate(t)
        me = self.me_tracker.estimate(t)
        if opp.resyncs:
            warnings.append(f"{opp.resyncs} impossible play(s) seen: a detection was missed or wrong")
        reg = int(self.rules["regulation_s"] * 1000)
        end = reg + int(self.rules["overtime_s"] * 1000)
        left = (reg if t < reg else end) - t
        s = max(0, (left + 999) // 1000)
        phase = ("Overtime " if t >= reg else "") + f"x{opp.multiplier} elixir"
        guess = self.guesser.guess(opp.known_cards) if self.guesser else None
        return CompanionState(t, f"{s // 60}:{s % 60:02d}", phase, opp, me, guess,
                              self.events[-6:], fps, self._last_dets, warnings)

    def save_log(self, path: Path | str) -> None:
        """Play log of the match (for post-match review)."""
        data = [{"t_ms": e.t_ms, "side": e.side, "card": e.card_id, "cost": e.elixir_cost,
                 "x": e.tile.x, "y": e.tile.y} for e in self.events]
        Path(path).write_text(json.dumps({"plays": data}, indent=1) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
class SimDemoFeed:
    """Demo / test input without a phone: plays a bot-vs-bot match in the simulator and
    emits what a *noisy* detector would see (position jitter, missed frames, false positives)."""

    def __init__(self, seed: int = 0, fps: float = 10.0, noise_tiles: float = 0.3,
                 miss_rate: float = 0.1, false_positive_rate: float = 0.02, db: Optional[CardDB] = None):
        from backend_engine.brain.scripted_bot import ScriptedBot
        from backend_engine.simulator.engine import Match, MatchConfig
        from backend_engine.simulator.match_runner import DEFAULT_DECK, MatchRunner

        self.db = db or CardDB.load()
        self.match = Match(MatchConfig((DEFAULT_DECK, DEFAULT_DECK), seed=seed), self.db)
        self.runner = MatchRunner(self.match, {0: ScriptedBot(self.db, 0, 0.6), 1: ScriptedBot(self.db, 1, 0.5)})
        self.rng = random.Random(seed + 99)
        self.step_ms = int(1000 / fps)
        self.noise, self.miss, self.fp = noise_tiles, miss_rate, false_positive_rate
        self.cards = list(self.db.cards)

    def next(self) -> Optional[Tuple[List[TileDetection], float]]:
        m = self.match
        if m.result is not None:
            return None
        for _ in range(self.step_ms // 50):
            self.runner.tick()
        dets: List[TileDetection] = []
        g = self.rng.gauss
        for u in sorted(m.units.values(), key=lambda u: u.uid):
            if u.kind == "tower" or self.rng.random() < self.miss:
                continue
            dets.append((u.card_id, u.side, u.x + g(0, self.noise), u.y + g(0, self.noise)))
        for sp in m.spells:
            if self.rng.random() >= self.miss:
                dets.append((sp.card_id, sp.side, sp.x + g(0, self.noise), sp.y + g(0, self.noise)))
        for rs in m.rolling:
            if self.rng.random() >= self.miss:
                dets.append((rs.card_id, rs.side, rs.x + g(0, self.noise), rs.y + g(0, self.noise)))
        if self.rng.random() < self.fp:  # a one-frame hallucination
            dets.append((self.rng.choice(self.cards), self.rng.choice((0, 1)),
                         self.rng.uniform(0, 18), self.rng.uniform(0, 32)))
        return dets, m.t_ms / 1000.0
