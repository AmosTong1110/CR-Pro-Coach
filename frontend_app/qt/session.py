"""
GameSession: Qt-free controller behind the 1v1 board UI (side 0 = you, side 1 = AI).

Everything the window needs goes through here, so it can be unit-tested headlessly
and reused by later features (scenario drill, take-over-from-here).
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field, replace
from typing import Dict, List, Optional, Tuple

from backend_engine.brain.coach_analyzer import Candidate, CoachAnalyzer, MatchReview
from backend_engine.brain.human_delay_buffer import HumanDelayBuffer, ReactionProfile
from backend_engine.brain.scripted_bot import ScriptedBot
from backend_engine.contracts import ElixirEstimate, LatencyClass
from backend_engine.perception.bayesian_deck_guesser import DeckGuess, default_guesser
from backend_engine.perception.opp_elixir_count import OppElixirTracker
from backend_engine.simulator import geometry as G
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig, MatchResult, PlayResult, Unit
from backend_engine.simulator.match_runner import DEFAULT_DECK, MatchRunner

HUMAN, AI = 0, 1

# Scripted stand-ins for the four coach personas until the trained policy exists.
PERSONAS: Dict[str, dict] = {
    "surgeon":    {"label": "The Surgeon (Control/Macro)",       "aggression": 0.3, "profile": "surgeon"},
    "aggressor":  {"label": "The Aggressor (Dual-Lane Pressure)", "aggression": 0.95, "profile": "aggressor"},
    "spell_cycler": {"label": "Spell-Cycler (Turtle/Chip)",       "aggression": 0.2, "profile": "spell_cycler"},
    "adaptive_counter": {"label": "Adaptive Counter",             "aggression": 0.5, "profile": "adaptive_counter"},
}


@dataclass(frozen=True)
class SessionConfig:
    player_deck: Tuple[str, ...] = DEFAULT_DECK
    ai_deck: Tuple[str, ...] = DEFAULT_DECK
    level: int = 11
    persona: str = "surgeon"
    player_evo: Tuple[str, ...] = ()
    ai_evo: Tuple[str, ...] = ()
    seed: Optional[int] = None


@dataclass(frozen=True)
class LogEntry:
    t_ms: int
    side: int
    card_id: str
    x: float
    y: float
    evolved: bool
    ai_latency: Optional[Tuple[LatencyClass, int]] = None  # how long the AI took to "see" your play
    event_id: int = 0


CHECKPOINT_EVERY_MS = 1000


@dataclass
class Checkpoint:
    """Everything needed to resume a match exactly (engine + AI's eyes, fingers and counters)."""
    t_ms: int
    match: dict
    ai_buffer: dict
    delivered: int
    ai_trackers: Dict[int, dict]
    opp_tracker: dict


@dataclass(frozen=True)
class QuizResult:
    t_ms: int
    guess: float
    truth: float
    error: float


class GameSession:
    def __init__(self, db: Optional[CardDB] = None, cfg: SessionConfig = SessionConfig()):
        self.db = db or CardDB.load()
        self.cfg = cfg
        self.log: List[LogEntry] = []
        self.quiz: List[QuizResult] = []   # kept across matches: it's your training record
        self.new_match()

    # -- lifecycle ---------------------------------------------------------
    def new_match(self, cfg: Optional[SessionConfig] = None) -> None:
        if cfg is not None:
            self.cfg = cfg
        c = self.cfg
        seed = c.seed if c.seed is not None else random.randrange(1 << 30)
        self.seed = seed
        persona = PERSONAS[c.persona]
        try:
            profile = ReactionProfile.load(persona=persona["profile"])
        except Exception:  # pyyaml missing or profile broken: still playable
            profile = ReactionProfile.default()
        self.match = Match(MatchConfig((c.player_deck, c.ai_deck), (c.level, c.level),
                                       (c.player_evo, c.ai_evo), seed=seed), self.db)
        self.ai_buffer = HumanDelayBuffer(profile, side=AI, seed=seed + 1)
        self.runner = MatchRunner(self.match, {HUMAN: None, AI: ScriptedBot(self.db, AI, persona["aggression"])},
                                  {AI: self.ai_buffer})
        self.log = []
        self._logged = 0
        evo_capable = [cid for cid, d in self.db.cards.items() if d.get("evolution")]
        # your counter for the AI: sees AI plays the moment they appear on screen
        self.opp_tracker = OppElixirTracker(self.db.rules, observer_side=HUMAN, evo_cards=evo_capable)
        self.timeline: List[Checkpoint] = []   # past checkpoints of the current branch, oldest first
        self.redo: List[Checkpoint] = []       # checkpoints ahead of us after a rewind (valid until you play)
        self.pre_play: Dict[int, Checkpoint] = {}  # event_id -> state just before your play
        self.review: Optional[MatchReview] = None
        self._checkpoint()

    # -- timeline: undo / forward / take over --------------------------------
    def _capture(self) -> Checkpoint:
        return Checkpoint(
            t_ms=self.match.t_ms,
            match=self.match.snapshot(),
            ai_buffer=self.ai_buffer.snapshot(),
            delivered=self.runner._delivered,
            ai_trackers={s: t.snapshot() for s, t in self.runner.trackers.items()},
            opp_tracker=self.opp_tracker.snapshot(),
        )

    def _restore(self, cp: Checkpoint) -> None:
        self.match.restore(cp.match)
        self.ai_buffer.restore(cp.ai_buffer)
        self.runner._delivered = cp.delivered
        for s, snap in cp.ai_trackers.items():
            self.runner.trackers[s].restore(snap)
        self.opp_tracker.restore(cp.opp_tracker)
        # the play log is derived from match.events (the AI reaction times live in the buffer snapshot)
        self.log, self._logged = [], 0
        self._collect_log()

    def _checkpoint(self) -> None:
        t = self.match.t_ms
        if self.timeline and self.timeline[-1].t_ms >= t:
            return
        self.timeline.append(self._capture())
        while self.redo and self.redo[0].t_ms <= t:  # replaying the same future: those are regenerated
            self.redo.pop(0)

    def can_rewind(self) -> bool:
        return bool(self.timeline) and (self.timeline[0].t_ms < self.match.t_ms or len(self.timeline) > 1)

    def can_forward(self) -> bool:
        return bool(self.redo)

    def max_known_ms(self) -> int:
        return self.redo[-1].t_ms if self.redo else self.match.t_ms

    def seek(self, t_ms: int) -> Optional[int]:
        """Jump to the latest checkpoint at or before t_ms (rewind), or forward along the
        abandoned branch. Returns the time actually restored, or None if impossible."""
        now = self.match.t_ms
        if t_ms == now and self.match.result is None:
            return now
        if t_ms < now or (t_ms == now and self.match.result is not None):
            idx = max((i for i, cp in enumerate(self.timeline) if cp.t_ms <= t_ms), default=None)
            if idx is None:
                return None
            if self.timeline[-1].t_ms < now:            # remember where we were, for forward
                self.redo.insert(0, self._capture())
            self.redo = self.timeline[idx + 1:] + self.redo
            self.timeline = self.timeline[:idx + 1]
            self._restore(self.timeline[-1])
            return self.timeline[-1].t_ms
        ahead = [i for i, cp in enumerate(self.redo) if cp.t_ms <= t_ms]
        if not ahead:
            return None
        k = ahead[-1]
        self.timeline.extend(self.redo[:k + 1])
        self.redo = self.redo[k + 1:]
        self._restore(self.timeline[-1])
        return self.timeline[-1].t_ms

    def rewind(self, seconds: float = 10) -> Optional[int]:
        return self.seek(max(0, self.match.t_ms - int(seconds * 1000)))

    def forward(self, seconds: float = 10) -> Optional[int]:
        if not self.redo:
            return None
        target = self.match.t_ms + int(seconds * 1000)
        return self.seek(target) if self.redo[0].t_ms <= target else self.seek(self.redo[0].t_ms)

    def take_over_before(self, log_index: int) -> Optional[int]:
        """Go back to just before log entry `log_index` was played, so you can replay that moment."""
        e = self.log[log_index]
        if e.event_id in self.pre_play:
            return self.restore_before_event(e.event_id)
        return self.seek(max(0, e.t_ms - 1))

    def restore_before_event(self, event_id: int) -> Optional[int]:
        """Exact state just before one of your plays (works for plays on the current branch)."""
        cp = self.pre_play.get(event_id)
        if cp is None or not any(ev.event_id == event_id for ev in self.match.events):
            return None
        self.seek(cp.t_ms)
        if self.match.t_ms != cp.t_ms or any(ev.event_id == event_id for ev in self.match.events):
            self._restore(cp)  # same-tick combos: the timeline holds the state before the first card
        return self.match.t_ms

    # -- post-match coach ----------------------------------------------------
    def fork(self, cp: Checkpoint) -> Tuple[Match, MatchRunner]:
        """Independent copy of the game at a checkpoint (for analysis; never touches the live match)."""
        m = Match(self.match.cfg, self.db)
        m.restore(cp.match)
        buf = HumanDelayBuffer(self.ai_buffer.profile, side=AI, seed=0)
        buf.restore(cp.ai_buffer)
        r = MatchRunner(m, {HUMAN: None, AI: ScriptedBot(self.db, AI, PERSONAS[self.cfg.persona]["aggression"])},
                        {AI: buf})
        r._delivered = cp.delivered
        for s, snap in cp.ai_trackers.items():
            r.trackers[s].restore(snap)
        return m, r

    def analyze(self, progress=None, horizon_ms: int = 8000) -> MatchReview:
        """Win-rate curve + grade each of your plays on the current branch."""
        analyzer = CoachAnalyzer(self.fork, horizon_ms=horizon_ms)
        plays = []
        for ev in self.match.events:
            if ev.side == HUMAN and ev.event_id in self.pre_play:
                plays.append((self.pre_play[ev.event_id], Candidate(ev.card_id, ev.tile.x, ev.tile.y), ev.event_id))
        timeline = [cp for cp in self.timeline if cp.t_ms <= self.match.t_ms]
        self.review = analyzer.review_match(timeline, plays, final_match=self.match, progress=progress)
        return self.review

    # -- elixir counter training -------------------------------------------
    def opp_estimate(self) -> ElixirEstimate:
        return self.opp_tracker.estimate(self.match.t_ms)

    def opp_deck_guess(self) -> Optional[DeckGuess]:
        g = default_guesser()
        return g.guess(self.opp_estimate().known_cards) if g else None

    def elixir_advantage(self) -> float:
        """Your elixir minus the AI's (counted from public plays)."""
        return self.match.sides[HUMAN].elixir - self.opp_estimate().point

    def quiz_guess(self, guess: float) -> QuizResult:
        truth = self.match.sides[AI].elixir
        r = QuizResult(self.match.t_ms, guess, truth, abs(guess - truth))
        self.quiz.append(r)
        return r

    def quiz_stats(self) -> Tuple[int, float, float]:
        """(attempts, mean abs error, share within 0.5 elixir)."""
        if not self.quiz:
            return 0, 0.0, 0.0
        n = len(self.quiz)
        return n, sum(q.error for q in self.quiz) / n, sum(q.error <= 0.5 for q in self.quiz) / n

    def reconfigure(self, **changes) -> None:
        self.new_match(replace(self.cfg, **changes))

    # -- time --------------------------------------------------------------
    def tick(self, n: int = 1) -> List[LogEntry]:
        for _ in range(n):
            if self.match.result is not None:
                break
            if self.match.t_ms % CHECKPOINT_EVERY_MS == 0:
                self._checkpoint()
            self.runner.tick()
        if self.match.result is not None:
            self._checkpoint()  # so "rewind" from the result screen works
        return self._collect_log()

    def _collect_log(self) -> List[LogEntry]:
        new = []
        for ev in self.match.events[self._logged:]:
            self.opp_tracker.observe(ev)
            lat =self.ai_buffer.latency_of(ev.event_id) if ev.side == HUMAN else None
            new.append(LogEntry(ev.t_ms, ev.side, ev.card_id, ev.tile.x, ev.tile.y, ev.is_evolved, lat, ev.event_id))
        self._logged = len(self.match.events)
        self.log.extend(new)
        return new

    # -- player input ------------------------------------------------------
    def play(self, card_id: str, x: float, y: float) -> PlayResult:
        if self.match.t_ms % CHECKPOINT_EVERY_MS == 0:
            self._checkpoint()  # this second's checkpoint must be the state *before* your play
        before = self._capture()
        res = self.runner.human_play(card_id, x, y, side=HUMAN)
        if res.ok:
            self.redo.clear()  # you changed history: the old future no longer exists
            self.pre_play[res.event.event_id] = before   # exact state for review / take over
            if not self.timeline or self.timeline[-1].t_ms < before.t_ms:
                self.timeline.append(before)
        self.runner.deliver_events()  # AI's reaction clock starts at the moment of the play
        self._collect_log()
        return res

    def legal_tiles(self, card_id: str) -> List[Tuple[float, float]]:
        """Tile centres where the card could be dropped right now, ignoring elixir."""
        st = self.match.sides[HUMAN]
        if card_id not in st.hand:
            return []
        saved = st.elixir
        st.elixir = 10.0  # show the area even while saving up elixir
        try:
            return [(x + 0.5, y + 0.5) for y in range(int(G.H)) for x in range(int(G.W))
                    if self.match.check_play(HUMAN, card_id, x + 0.5, y + 0.5) is None]
        finally:
            st.elixir = saved

    # -- queries for the view ---------------------------------------------
    @property
    def result(self) -> Optional[MatchResult]:
        return self.match.result

    def ai_has_seen(self, u: Unit) -> bool:
        return u.side == AI or u.source_event_id is None or self.ai_buffer.is_visible(u.source_event_id)

    def clock(self) -> Tuple[str, str]:
        """('2:41', 'x2 Elixir' / 'Overtime x3' / '')."""
        m = self.match
        left = (m.regulation_ms if m.t_ms < m.regulation_ms else m.end_ms) - m.t_ms
        s = max(0, (left + 999) // 1000)
        mult = m.multiplier()
        if m.t_ms >= m.regulation_ms:
            phase = f"Overtime x{mult}"
        else:
            phase = f"x{mult} Elixir" if mult > 1 else ""
        return f"{s // 60}:{s % 60:02d}", phase

    def elixir_cost(self, card_id: str) -> int:
        return self.db.cards[card_id]["elixir"]
