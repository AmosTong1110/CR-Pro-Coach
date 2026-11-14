"""
Opponent Elixir & Cycle Estimator (Strictly Human-Legal / Non-Cheating).
Ported and modularized from ClashAI's public-events accounting pipeline.
Estimates opponent's current elixir based solely on match clock and visible plays.
"""
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

START_ELIXIR = 5.0
MAX_ELIXIR = 10.0

# Regeneration rates per second (Tick = 0.05s)
REGEN_SINGLE = 1.0 / 2.80   # ~0.357 elixir/sec
REGEN_DOUBLE = 1.0 / 1.40   # ~0.714 elixir/sec
REGEN_TRIPLE = 1.0 / 0.93   # ~1.075 elixir/sec


@dataclass
class OpponentElixirTracker:
    current_elixir: float = START_ELIXIR
    last_update_time_s: float = 0.0
    deck_cards_seen: List[str] = field(default_factory=list)
    cycle_history: List[str] = field(default_factory=list)

    def _get_regen_rate(self, match_time_s: float) -> float:
        if match_time_s >= 240.0:
            return REGEN_TRIPLE
        elif match_time_s >= 120.0:
            return REGEN_DOUBLE
        return REGEN_SINGLE

    def update_to_time(self, current_time_s: float):
        """Accumulate regenerated elixir between ticks."""
        if current_time_s <= self.last_update_time_s:
            return
            
        dt = current_time_s - self.last_update_time_s
        rate = self._get_regen_rate(current_time_s)
        self.current_elixir = min(MAX_ELIXIR, self.current_elixir + rate * dt)
        self.last_update_time_s = current_time_s

    def register_opponent_play(self, card_name: str, card_cost: int, play_time_s: float):
        """Account for opponent card deployment seen on screen."""
        self.update_to_time(play_time_s)
        
        # If the opponent played a card costing more than our estimate,
        # it proves our estimate was too low. Re-base to the card cost.
        if self.current_elixir < card_cost:
            self.current_elixir = float(card_cost)
            
        self.current_elixir = max(0.0, self.current_elixir - card_cost)
        
        # Card cycle tracking
        if card_name not in self.deck_cards_seen:
            self.deck_cards_seen.append(card_name)
        self.cycle_history.append(card_name)

    def is_card_in_opponent_hand(self, card_name: str) -> bool:
        """
        Clash Royale 4-card cycle rule:
        Once played, a card takes 4 subsequent plays to return to hand.
        """
        if card_name not in self.deck_cards_seen:
            return True  # Unseen card could still be in opening hand
            
        # Check last 4 plays
        recent_4 = self.cycle_history[-4:] if len(self.cycle_history) >= 4 else self.cycle_history
        return card_name not in recent_4

    def get_state(self) -> dict:
        return {
            "estimated_elixir": round(self.current_elixir, 2),
            "cards_seen": list(self.deck_cards_seen),
            "known_deck_size": len(self.deck_cards_seen),
            "last_played": self.cycle_history[-1] if self.cycle_history else None
        }
