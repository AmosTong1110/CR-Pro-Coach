"""
Rule-based baseline bot. It only reads the fog-of-war observation and visible events,
and acts through HumanDelayBuffer like any future neural policy would.

Reasoning is done in "my perspective" coordinates (as if I were side 0, attacking +y)
and converted with geometry.to_side() at the end.
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from backend_engine.brain.human_delay_buffer import HumanDelayBuffer
from backend_engine.contracts import ActionIntent, ActionKind, PublicDeployEvent, TilePos
from backend_engine.simulator import geometry as G
from backend_engine.simulator.card_db import CardDB


class ScriptedBot:
    def __init__(self, db: CardDB, side: int, aggression: float = 0.5):
        self.db = db
        self.side = side
        self.aggression = aggression  # 0 = only defend/leak-protect, 1 = push whenever affordable

    # ------------------------------------------------------------------
    def _mine(self, x: float, y: float):
        return G.to_side(self.side, x, y)  # mirror is its own inverse

    def _intent(self, card: str, x: float, y: float) -> ActionIntent:
        ax, ay = G.to_side(self.side, x, y)
        return ActionIntent(ActionKind.DEPLOY, card, TilePos(ax, ay), self.db.cards[card]["elixir"], source="script")

    def _card(self, c: str) -> dict:
        return self.db.cards[c]

    def decide(self, obs: dict, visible_events: Sequence[PublicDeployEvent], now_ms: int,
               buffer: Optional[HumanDelayBuffer] = None) -> List[ActionIntent]:
        elixir, hand = obs["elixir"], obs["hand"]
        affordable = [c for c in hand if self._card(c)["elixir"] <= elixir + 1e-9]
        if not affordable:
            return []

        enemy = [u for u in obs["units"] if u["side"] != self.side]
        towers = [u for u in enemy if u["kind"] == "tower"]
        troops = [u for u in enemy if u["kind"] == "troop"]
        # threats: enemy troops in my half or about to cross (my-perspective y < 19)
        threats = [u for u in troops if self._mine(u["x"], u["y"])[1] < G.RIVER_Y1 + 2]

        # 1) finisher: spell a tower that is in kill range
        for c in affordable:
            card = self._card(c)
            if card["type"] != "spell":
                continue
            tdmg = self.db.stats(c, obs["level"]).stats.get("crown_tower_damage", 0)
            for t in towers:
                if 0 < t["hp"] <= tdmg and "radius_tiles" in card.get("spell", {}):
                    mx, my = self._mine(t["x"], t["y"])
                    return [self._intent(c, mx, my)]

        # 2) defend the most advanced threat
        if threats:
            t = min(threats, key=lambda u: self._mine(u["x"], u["y"])[1])
            tx, ty = self._mine(t["x"], t["y"])
            pick = self._defender(affordable, t, threats)
            if pick:
                card = self._card(pick)
                if card["type"] == "building":
                    return [self._intent(pick, 9.0, 9.0)]  # centre: pulls building-targeters
                if card["type"] == "spell":
                    if "rolling_range_tiles" in card.get("spell", {}):
                        return [self._intent(pick, tx, max(ty - 2.0, 0.5))]
                    return [self._intent(pick, tx, ty)]
                return [self._intent(pick, tx, max(min(ty - 3.0, G.RIVER_Y0 - 0.5), 0.5))]

        wincons = [c for c in affordable if self._card(c).get("attack", {}).get("targets") == ["buildings"]]

        # 2b) punish: opponent is low, or we lead by 2+ elixir -> hit the lane opposite their last troop
        est = obs.get("opp_estimate")
        if est is not None and wincons:
            low_opp = est.point <= 1.5 + 2.5 * self.aggression
            lead = elixir - est.point >= 2.0 and self.aggression >= 0.8
            if low_opp or lead:
                committed = [u for u in troops if self._mine(u["x"], u["y"])[1] > G.RIVER_Y1]
                if committed:
                    last = max(committed, key=lambda u: u["uid"])
                    lane_x = G.BRIDGE_X[1] if G.lane_of(last["x"]) == "left" else G.BRIDGE_X[0]
                else:
                    lane_x = min(G.BRIDGE_X, key=lambda bx: self._lane_tower_hp(towers, bx))
                return [self._intent(wincons[0], lane_x, G.RIVER_Y0 - 0.5)]

        # 3) push: win-condition at the bridge when we have elixir to support it
        if wincons and elixir >= self._card(wincons[0])["elixir"] + 4 * (1 - self.aggression):
            lane_x = min(G.BRIDGE_X, key=lambda bx: self._lane_tower_hp(towers, bx))
            return [self._intent(wincons[0], lane_x, G.RIVER_Y0 - 0.5)]

        # 4) don't leak: play the cheapest non-spell at the back
        if elixir >= 9.5:
            troops_in_hand = [c for c in affordable if self._card(c)["type"] == "troop"]
            if troops_in_hand:
                c = min(troops_in_hand, key=lambda c: self._card(c)["elixir"])
                return [self._intent(c, 9.0 + (2.0 if now_ms // 10_000 % 2 else -2.0), 1.5)]
        return []

    def _lane_tower_hp(self, towers: List[dict], bridge_x: float) -> float:
        lane = G.lane_of(bridge_x)
        hp = [t["hp"] for t in towers if t["tower_role"] != "king" and G.lane_of(t["x"]) == lane]
        return hp[0] if hp else 0.0

    def _defender(self, affordable: List[str], threat: dict, threats: List[dict]) -> Optional[str]:
        air = threat["layer"] == "air"
        swarm = sum(1 for u in threats if abs(u["x"] - threat["x"]) < 2.5 and abs(u["y"] - threat["y"]) < 2.5) >= 2
        scored = []
        for c in affordable:
            card = self._card(c)
            ctype, sp = card["type"], card.get("spell", {})
            targets = card.get("attack", {}).get("targets", [])
            if ctype == "spell":
                if swarm and not (air and "rolling_range_tiles" in sp):
                    scored.append((0, card["elixir"], c))
                continue
            if targets == ["buildings"]:
                continue
            if air and "air" not in targets:
                continue
            if ctype == "building" and "buildings" in threat["targets"]:
                scored.append((1, card["elixir"], c))
            elif ctype == "troop":
                scored.append((2, card["elixir"], c))
            elif ctype == "building":
                scored.append((3, card["elixir"], c))
        return min(scored)[2] if scored else None
