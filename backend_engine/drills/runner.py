"""
Drill execution runner and live grading evaluator.
Manages isolated micro-scenario simulations, objective tracking, and scoring.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Tuple

from backend_engine.drills.drill_spec import DrillResult, DrillSpec
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.engine import Match, MatchConfig, PlayResult


class DrillRunner:
    """Manages an active drill simulation session and computes real-time performance grades."""

    def __init__(self, drill: DrillSpec, db: Optional[CardDB] = None):
        self.drill = drill
        self.db = db or CardDB.load()
        self.elixir_spent = 0.0

        # Construct isolated match
        # AI side receives empty deck cycle so it won't play unexpected cards outside the drill
        dummy_ai_deck = ("knight", "archers", "skeletons", "fireball", "the_log", "ice_spirit", "musketeer", "zap")
        cfg = MatchConfig(
            decks=(drill.player_deck, dummy_ai_deck),
            levels=(drill.level, drill.level),
            seed=42,
            shuffle=False,
        )
        self.match = Match(cfg, self.db)

        # Override player hand and queue
        st0 = self.match.sides[0]
        st0.hand = list(drill.player_hand)
        st0.queue = list(drill.player_queue)
        st0.elixir = float(drill.player_elixir)

        # Zero out AI elixir so only preplaced threats act
        st1 = self.match.sides[1]
        st1.elixir = 0.0

        # Spawn friendly pre-placed units
        for f in drill.friendly_units:
            self.match._spawn_card_units(0, f.card_id, f.x, f.y, eid=self.match._uid(),
                                         evolved=False, variant=f.variant, deploy_ms=0)

        # Spawn oncoming enemy threats
        self.threat_uids: List[int] = []
        for t in drill.enemy_threats:
            eid = self.match._uid()
            self.match._spawn_card_units(1, t.card_id, t.x, t.y, eid=eid,
                                         evolved=False, variant=t.variant, deploy_ms=0)
            # Find the spawned units
            for u in self.match.units.values():
                if u.source_event_id == eid:
                    if t.hp_ratio < 1.0:
                        u.hp = max(1.0, u.max_hp * t.hp_ratio)
                    self.threat_uids.append(u.uid)

        # Record baseline tower hitpoints
        self.initial_tower_hp: Dict[str, float] = {}
        for role in ("king", "princess_left", "princess_right"):
            tow = self.match.tower(0, role)
            if tow:
                self.initial_tower_hp[role] = tow.hp

    def play(self, card_id: str, x: float, y: float) -> PlayResult:
        """Play a defending card, tracking elixir expenditure."""
        cost = self.db.cards.get(card_id, {}).get("elixir", 0)
        res = self.match.play(0, card_id, x, y)
        if res.ok:
            self.elixir_spent += cost
        return res

    def step(self) -> None:
        """Advance simulation by one 50ms tick."""
        # Keep AI elixir locked at 0 so no external troops interfere with the drill
        self.match.sides[1].elixir = 0.0
        self.match.step()

    def threats_alive(self) -> int:
        """Count how many initial oncoming threats are still alive."""
        count = 0
        for uid in self.threat_uids:
            u = self.match.units.get(uid)
            if u is not None and u.hp > 0:
                count += 1
        return count

    def tower_damage_taken(self) -> float:
        """Total damage taken by the player's Princess Towers during this drill."""
        dmg = 0.0
        for role in ("princess_left", "princess_right"):
            tow = self.match.tower(0, role)
            initial = self.initial_tower_hp.get(role, 0.0)
            if tow:
                dmg += max(0.0, initial - tow.hp)
            else:
                dmg += initial
        return dmg

    def is_finished(self) -> bool:
        """True if the drill has reached a terminal outcome."""
        # 1. All threats neutralized
        if self.threats_alive() == 0:
            return True
        # 2. Time ran out
        if self.match.t_ms >= self.drill.target_duration_ms:
            return True
        # 3. Both princess towers lost
        alive_towers = sum(1 for role in ("princess_left", "princess_right")
                           if self.match.tower(0, role) and self.match.tower(0, role).hp > 0)
        if alive_towers == 0:
            return True
        return False

    def evaluate(self) -> DrillResult:
        """Produce the final evaluation and stars for this drill attempt."""
        cleared = (self.threats_alive() == 0)
        dmg = self.tower_damage_taken()
        t_ms = self.match.t_ms

        if cleared and dmg <= self.drill.max_tower_damage_allowed + 1e-6:
            stars = 3
            solved = True
            feedback = "⭐⭐⭐ Perfect Defense! Threat neutralized with zero tower damage taken."
        elif cleared and dmg < 400.0:
            stars = 2
            solved = True
            feedback = f"⭐⭐ Good Defense! Stopped the push with {int(dmg)} chip damage."
        elif cleared:
            stars = 1
            solved = True
            feedback = f"⭐ Solved! Threat cleared, but tower took {int(dmg)} damage."
        else:
            stars = 0
            solved = False
            feedback = f"❌ Defense Failed. Incoming wave damaged your tower by {int(dmg)}."

        return DrillResult(
            solved=solved,
            stars=stars,
            tower_damage_taken=dmg,
            elixir_spent=self.elixir_spent,
            duration_ms=t_ms,
            feedback=feedback,
        )
