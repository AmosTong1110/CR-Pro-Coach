"""
基于启发式规则的高性能基准陪练机器人 (Rule-Based Baseline Bot)
严格遵守不作弊原则：仅读取带有战争迷雾的环境观察 (obs) 与可见事件，
并通过人类反应延迟缓冲区 (HumanDelayBuffer) 发出动作，模拟真实人类操作。

内部空间推理基于“己方第一视角”坐标系 (视自身为 side 0，向 +y 方向推进攻击)，
并在最后决策输出时通过 geometry.to_side() 无缝翻转映射为全局坐标。
"""
from __future__ import annotations

from typing import List, Optional, Sequence

from backend_engine.brain.human_delay_buffer import HumanDelayBuffer
from backend_engine.contracts import ActionIntent, ActionKind, PublicDeployEvent, TilePos
from backend_engine.simulator import geometry as G
from backend_engine.simulator.card_db import CardDB


class ScriptedBot:
    """启发式策略机器人核心类"""

    def __init__(self, db: CardDB, side: int, aggression: float = 0.5):
        self.db = db
        self.side = side               # AI 所属阵营 (通常为 side 1)
        self.aggression = aggression  # 进攻激进系数：0.0 为稳健防守自闭运营，1.0 为疯狂换路强压

    # ------------------------------------------------------------------
    def _mine(self, x: float, y: float):
        """将竞技场全局坐标转换为己方视角的相对坐标"""
        return G.to_side(self.side, x, y)

    def _intent(self, card: str, x: float, y: float) -> ActionIntent:
        """从己方视角坐标生成标准的下牌动作意图"""
        ax, ay = G.to_side(self.side, x, y)
        return ActionIntent(ActionKind.DEPLOY, card, TilePos(ax, ay), self.db.cards[card]["elixir"], source="script")

    def _ability(self, obs: dict) -> Optional[ActionIntent]:
        """Champion ability: Dashing Dash when 2+ enemy troops / buildings are within its reach; Explosive Escape when
        the Miner is hurt (below 60%) with an enemy on top of it. Single use, so the bot just waits for the moment."""
        for c in obs.get("abilities", ()):
            ab = next((m["params"] for m in self._card(c)["mechanics"] if m["kind"] == "champion_ability"), {})
            name, cost = ab.get("name"), int(ab.get("elixir", 1))
            for u in obs["units"]:
                if u["side"] != self.side or u["card_id"] != c or u["deploying"]:
                    continue
                reach = {"dash": 5.0, "royal_rescue": 4.0}.get(name, 2.0)
                near = [e for e in obs["units"] if e["side"] != self.side and e["kind"] != "tower"
                        and G.dist(e["x"], e["y"], u["x"], u["y"]) <= reach]
                if ((name == "dash" and len(near) >= 2) or (name == "royal_rescue" and near)
                        or (name == "explosive_escape" and near and u["hp"] < 0.6 * u["max_hp"])):
                    return ActionIntent(ActionKind.ABILITY, c, None, cost, source="script")
        return None

    def _card(self, c: str) -> dict:
        return self.db.cards[c]

    def decide(self, obs: dict, visible_events: Sequence[PublicDeployEvent], now_ms: int,
               buffer: Optional[HumanDelayBuffer] = None) -> List[ActionIntent]:
        """
        核心分层决策逻辑：
        优先级 1: 法术斩杀残血公主塔
        优先级 2: 紧急解场（拉扯黄金位、对空解场、法术清群怪）
        优先级 2b: 算水抓空挡换路惩罚 (抓对手沉底或亏费时反向速推)
        优先级 3: 费用充沛时桥头进攻核心推进
        优先级 4: 圣水满费防漏水沉底过牌
        """
        elixir, hand = obs["elixir"], obs["hand"]
        ab = self._ability(obs)
        if ab:
            return [ab]
        affordable = [c for c in hand if self._card(c)["elixir"] <= elixir + 1e-9]
        if not affordable:
            return []

        enemy = [u for u in obs["units"] if u["side"] != self.side]
        towers = [u for u in enemy if u["kind"] == "tower"]
        troops = [u for u in enemy if u["kind"] == "troop"]
        # 威胁单位：已经进入我方半场或正在过桥逼近的敌军
        threats = [u for u in troops if self._mine(u["x"], u["y"])[1] < G.RIVER_Y1 + 2]

        # 1) 法术斩杀终结 (Finisher): 敌方防御塔血量低于法术斩杀线时直接砸塔
        for c in affordable:
            card = self._card(c)
            if card["type"] != "spell":
                continue
            tdmg = self.db.stats(c, obs["level"]).stats.get("crown_tower_damage", 0)
            for t in towers:
                if 0 < t["hp"] <= tdmg and "radius_tiles" in card.get("spell", {}):
                    mx, my = self._mine(t["x"], t["y"])
                    return [self._intent(c, mx, my)]

        # 2) 紧急防守 (Defend the most advanced threat): 针对压境最深的前线威胁进行精准解场
        if threats:
            t = min(threats, key=lambda u: self._mine(u["x"], u["y"])[1])
            tx, ty = self._mine(t["x"], t["y"])
            pick = self._defender(affordable, t, threats)
            if pick:
                card = self._card(pick)
                # 防御建筑 (如加农炮/地狱塔) 下在场地中央黄金拉扯位 (9.0, 9.0)
                if card["type"] == "building":
                    return [self._intent(pick, 9.0, 9.0)]
                # 法术解场：预判落点或滚木从身前滚过
                if card["type"] == "spell":
                    if "rolling_range_tiles" in card.get("spell", {}):
                        return [self._intent(pick, tx, max(ty - 2.0, 0.5))]
                    return [self._intent(pick, tx, ty)]
                # 军队防守：放置在敌军身前安全拉扯距离
                return [self._intent(pick, tx, max(min(ty - 3.0, G.RIVER_Y0 - 0.5), 0.5))]

        # 寻找进攻核心卡牌 (只攻击建筑的攻城核心，如野猪骑士、攻城槌)
        wincons = [c for c in affordable if self._card(c).get("attack", {}).get("targets") == ["buildings"]]

        # 2b) 抓空挡突袭换路 (Punish / Lane Switch):
        # 推算出对手圣水空虚 (<=1.5水) 或我方圣水领先 >= 2 水且处于激进模式时，在对手沉底相反路桥头速攻
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

        # 3) 桥头推进进攻 (Push): 圣水充裕、足够跟上后续法术与支援时桥头施放进攻核心
        if wincons and elixir >= self._card(wincons[0])["elixir"] + 4 * (1 - self.aggression):
            lane_x = min(G.BRIDGE_X, key=lambda bx: self._lane_tower_hp(towers, bx))
            return [self._intent(wincons[0], lane_x, G.RIVER_Y0 - 0.5)]

        # 4) 满费防漏水过牌 (Leak Protection): 圣水逼近 9.5~10 费时，沉底最便宜的轻量单位过牌
        if elixir >= 9.5:
            troops_in_hand = [c for c in affordable if self._card(c)["type"] == "troop"]
            if troops_in_hand:
                c = min(troops_in_hand, key=lambda c: self._card(c)["elixir"])
                return [self._intent(c, 9.0 + (2.0 if now_ms // 10_000 % 2 else -2.0), 1.5)]
        return []

    def _lane_tower_hp(self, towers: List[dict], bridge_x: float) -> float:
        """获取指定桥梁对应分路敌方公主塔的剩余血量"""
        lane = G.lane_of(bridge_x)
        hp = [t["hp"] for t in towers if t["tower_role"] != "king" and G.lane_of(t["x"]) == lane]
        return hp[0] if hp else 0.0

    def _defender(self, affordable: List[str], threat: dict, threats: List[dict]) -> Optional[str]:
        """解场卡牌智能挑选器：按对空克制、人海针对、建筑拉扯进行打分"""
        air = threat["layer"] == "air"
        swarm = sum(1 for u in threats if abs(u["x"] - threat["x"]) < 2.5 and abs(u["y"] - threat["y"]) < 2.5) >= 2
        scored = []
        for c in affordable:
            card = self._card(c)
            ctype, sp = card["type"], card.get("spell", {})
            targets = card.get("attack", {}).get("targets", [])
            # 人海怪优先用法术清场
            if ctype == "spell":
                if swarm and not (air and "rolling_range_tiles" in sp):
                    scored.append((0, card["elixir"], c))
                continue
            # 纯打建筑的核心卡不用于解场
            if targets == ["buildings"]:
                continue
            # 敌方为空中单位但该卡无法对空时跳过
            if air and "air" not in targets:
                continue
            # 针对建筑特攻单位 (野猪等)，建筑拉扯为第一优先级
            if ctype == "building" and "buildings" in threat["targets"]:
                scored.append((1, card["elixir"], c))
            elif ctype == "troop":
                scored.append((2, card["elixir"], c))
            elif ctype == "building":
                scored.append((3, card["elixir"], c))
        return min(scored)[2] if scored else None
