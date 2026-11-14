"""
Bayesian deck guesser: from the 1-7 cards an opponent has revealed, estimate which
other cards they carry.

Model
  Prior     every deck d observed in real ladder battles, weight = times seen (+ alpha).
  Evidence  revealed set R. Likelihood is 1 if R ⊆ d else 0 (cards can't be faked).
  Posterior P(d | R) ∝ w_d · 1[R ⊆ d]   ->   P_deck(c) = Σ_{d ∋ c} P(d | R)

  With few matching decks (off-meta / rare decks) that is overconfident or empty, so it
  is blended with a card co-occurrence model:
      P_co(c) = mean_{r ∈ R} P(c ∈ deck | r ∈ deck)   (overall usage rate when R is empty)
      P(c)    = λ·P_deck(c) + (1-λ)·P_co(c),   λ = W / (W + kappa),  W = matching weight
  Finally rescaled so the unknown cards sum to the 8-|R| slots left (each P capped at 1).

Input/Output are plain card ids, so it works the same for the simulator and for real
games read from the screen (mode B).
"""
from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, FrozenSet, Iterable, List, Optional, Sequence, Tuple

DEFAULT_META = Path(__file__).resolve().parents[2] / "configs" / "meta_decks_hf.json"
DECK_SIZE = 8


@dataclass(frozen=True)
class DeckGuess:
    revealed: Tuple[str, ...]
    card_probs: Dict[str, float]                     # unrevealed card -> P(in deck), sorted high->low
    likely_decks: List[Tuple[Tuple[str, ...], float]] = field(default_factory=list)  # (deck, posterior)
    matching_weight: float = 0.0                     # how much real-deck evidence backs the guess

    def top(self, n: int = 4) -> List[Tuple[str, float]]:
        return list(self.card_probs.items())[:n]

    def in_hand_probs(self, unknown_hand_slots: int) -> Dict[str, float]:
        """P(card currently in hand) for unrevealed cards: unknown slots are filled uniformly
        from the unrevealed part of the deck."""
        unknown = DECK_SIZE - len(self.revealed)
        if unknown <= 0 or unknown_hand_slots <= 0:
            return {c: 0.0 for c in self.card_probs}
        share = unknown_hand_slots / unknown
        return {c: p * share for c, p in self.card_probs.items()}


class BayesianDeckGuesser:
    def __init__(self, decks: Iterable[Tuple[Sequence[str], float]], alpha: float = 0.0, kappa: float = 4.0):
        self.decks: List[Tuple[FrozenSet[str], float]] = []
        for cards, w in decks:
            s = frozenset(cards)
            if len(s) == DECK_SIZE:
                self.decks.append((s, float(w) + alpha))
        if not self.decks:
            raise ValueError("deck prior is empty")
        self.kappa = kappa
        self.total = sum(w for _, w in self.decks)
        self.usage: Counter = Counter()
        self.pair: Dict[str, Counter] = defaultdict(Counter)
        for d, w in self.decks:
            for c in d:
                self.usage[c] += w
                for o in d:
                    if o != c:
                        self.pair[c][o] += w
        self.cards = set(self.usage)

    @classmethod
    def load(cls, path: Path | str = DEFAULT_META, **kw) -> "BayesianDeckGuesser":
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls(((d["cards"], d["count"]) for d in data["decks"]), **kw)

    # ------------------------------------------------------------------
    def _p_co(self, c: str, revealed: Sequence[str]) -> float:
        known = [r for r in revealed if self.usage[r] > 0]
        if not known:
            return self.usage[c] / self.total
        return sum(self.pair[r][c] / self.usage[r] for r in known) / len(known)

    def guess(self, revealed: Iterable[str], exclude: Iterable[str] = (), top_decks: int = 3) -> DeckGuess:
        R = tuple(dict.fromkeys(revealed))        # keep order, drop duplicates
        rset = frozenset(R)
        slots = DECK_SIZE - len(rset)
        banned = set(exclude) | rset
        if slots <= 0:
            return DeckGuess(R, {}, [(tuple(sorted(rset)), 1.0)], self.total)

        matches = [(d, w) for d, w in self.decks if rset <= d]
        W = sum(w for _, w in matches)
        p_deck: Counter = Counter()
        for d, w in matches:
            for c in d - rset:
                p_deck[c] += w / W
        lam = W / (W + self.kappa) if W > 0 else 0.0

        probs: Dict[str, float] = {}
        for c in self.cards - banned:
            p = lam * p_deck.get(c, 0.0) + (1 - lam) * self._p_co(c, R)
            if p > 1e-4:
                probs[c] = p
        probs = _rescale_to(probs, float(slots))

        likely = sorted(matches, key=lambda m: -m[1])[:top_decks]
        likely_out = [(tuple(sorted(d)), w / W) for d, w in likely] if W else []
        ordered = dict(sorted(probs.items(), key=lambda kv: (-kv[1], kv[0])))
        return DeckGuess(R, ordered, likely_out, W)


COSTS_PATH = Path(__file__).resolve().parents[2] / "configs" / "card_costs_hf.json"
_DEFAULT: Optional[BayesianDeckGuesser] = None
_COSTS: Optional[Dict[str, int]] = None


def default_guesser() -> Optional[BayesianDeckGuesser]:
    """Shared instance built from configs/meta_decks_hf.json (None if the file is missing)."""
    global _DEFAULT
    if _DEFAULT is None and DEFAULT_META.exists():
        _DEFAULT = BayesianDeckGuesser.load()
    return _DEFAULT


def known_costs() -> Dict[str, int]:
    """Elixir cost of every card in the real game (for cards the simulator doesn't model yet)."""
    global _COSTS
    if _COSTS is None:
        _COSTS = json.loads(COSTS_PATH.read_text(encoding="utf-8"))["costs"] if COSTS_PATH.exists() else {}
    return _COSTS


def _rescale_to(probs: Dict[str, float], target: float, iters: int = 20) -> Dict[str, float]:
    """Scale so the sum equals target, with every value capped at 1 (water-filling)."""
    probs = dict(probs)
    for _ in range(iters):
        free = {c: p for c, p in probs.items() if p < 1.0}
        fixed = len(probs) - len(free)
        s = sum(free.values())
        if s <= 0:
            break
        k = (target - fixed) / s
        if abs(k - 1) < 1e-9:
            break
        for c in free:
            probs[c] = min(1.0, free[c] * k)
    return probs
