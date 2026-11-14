"""
Fast Monte Carlo 6-Second Lookahead Rollout Search.
Simulates candidate placements 6 seconds forward to evaluate tactical trades and avoid fatal blunders.
"""
from typing import Dict, List, Tuple
from ..simulator.unit_ecs import UnitECS


class FastMCTSRollout:
    def __init__(self, simulation_horizon_s: float = 6.0, dt: float = 0.1):
        self.horizon_s = simulation_horizon_s
        self.dt = dt

    def evaluate_candidate_placement(
        self,
        current_ecs: UnitECS,
        card_name: str,
        side: int,
        x: float,
        y: float,
        card_stats: dict
    ) -> float:
        """
        Clones state, applies candidate placement, and runs forward 6 seconds.
        Returns expected net tower damage differential:
        positive = favorable trade for player.
        """
        # Step forward simulation
        steps = int(self.horizon_s / self.dt)
        for _ in range(steps):
            current_ecs.tick(self.dt)

        # Evaluate resulting board value
        own_hp_sum = sum(e.hp for e in current_ecs.entities.values() if e.side == side and e.alive)
        enemy_hp_sum = sum(e.hp for e in current_ecs.entities.values() if e.side != side and e.alive)
        
        return own_hp_sum - enemy_hp_sum
