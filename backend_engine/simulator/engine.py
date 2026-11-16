"""
确定性皇室战争战斗物理引擎 (Deterministic Clash Royale Match Engine)

核心保证 (Guarantees):
  * 固定时间步长 (TICK_MS = 50ms，每秒 20 帧)。所有内部时间均为整数仿真毫秒 (t_ms)。
  * 强确定性：相同配置 (MatchConfig) + 相同动作序列 (action_log) -> 无论在何种机器上运行，state_hash() 绝对严格一致。
  * 精确快照与分支回溯：snapshot() / restore() 完整克隆所有实体对象及底层 RNG 随机数生成器状态，
    完美支持对局即时回退、接管重打、MCTS 蒙特卡洛多分支前向推演。
  * 战争迷雾与合法信息边界：唯一向外界感知识别层公布的只有 PublicDeployEvent (屏幕可见的下牌事件)。
    observation() 函数根据人类视线自动屏蔽尚未察觉的敌方单位。
"""
from __future__ import annotations

import copy
import hashlib
import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from backend_engine.contracts import PublicDeployEvent, TilePos
from backend_engine.simulator import geometry as G
from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.entities import AreaEffect, Projectile, RollingSpell, Spell, Unit  # noqa: F401  (re-exported)
from backend_engine.simulator.mechanics import REGISTRY, SPELLS, hooks_for
from backend_engine.simulator.mechanics import status as S
from backend_engine.simulator.mechanics.batch2 import fill_spawn_params
from backend_engine.simulator.mechanics.core import advance_projectile
from backend_engine.simulator.rng import Pcg32
from backend_engine.simulator.tables import MAX_SPAWN, SPAWN_RING, UNIT_DIRS

# Bumped whenever gameplay changes on purpose (golden hashes re-recorded). Saved matches made with another
# version can't be rebuilt, only shown from their saved summary / review.
#   1: initial engine   2: projectiles with travel time, knockback (Fireball, Log), Evo Knight damage reduction
#   3: portable arithmetic (PCG32 RNG, constant trig tables, sqrt instead of hypot) for the C++ core
#   4: card numbers from the Clash Royale Wiki (2026-11-16): exact L11-L16 hitpoints / damage / tower damage
#   5: wiki timings (hit speed, first hit, range) of the simulated cards
#   6: batch 2: charge, shield, burrow, ramp damage, recoil, hide underground, form change, Earthquake / Lightning /
#      Vines / Royal Delivery spell kinds, 50% slow
ENGINE_VERSION = 6

# 战斗感知与物理常量
from backend_engine.simulator.entities import TROOP_SIGHT  # noqa: E402,F401  军队通用视野警戒半径 (格子)
UNIT_RADIUS = 0.5           # 单位默认碰撞半径 (半格)
THREAT_DEPTH = 4.0          # 距河道 4 格以内的下兵直接判定为前线进攻威胁 (is_threat)
COLLISION_STIFFNESS = 0.5   # 碰撞重叠软缓解系数 (每帧消除 50% 重叠量，贴近官方手感)
MAX_PUSH_PER_TICK = 0.25    # 单帧最大位移推挤上限 (防止卡飞或穿模)


@dataclass
class SideState:
    """单边阵营经济与手牌流转状态"""
    level: int                     # 该阵营所采用的卡牌与塔身等级 (11 锦标赛或 16 满级)
    hand: List[str]                # 当前手里的 4 张可选卡牌
    queue: List[str]               # 卡组流转队列 (打出一张后，从 queue 头部补充下一张进手牌)
    elixir: float                  # 当前圣水存量 (浮点数累加，最大 10)
    evo_cards: Set[str] = field(default_factory=set)    # 携带了觉醒形态的卡牌集合
    evo_counter: Dict[str, int] = field(default_factory=dict) # 每张觉醒卡当前的过牌充能计数
    crowns: int = 0                # 当前获得的皇冠数 (0 ~ 3)
    leaked_elixir: float = 0.0     # 圣水满 10 后由于未及时出牌而白白浪费的漏水量
    plays: int = 0                 # 本局总出牌次数


@dataclass(frozen=True)
class MatchConfig:
    """对局初始参数配置"""
    decks: Tuple[Tuple[str, ...], Tuple[str, ...]] # 双方各自 8 张卡组
    levels: Tuple[int, int] = (11, 11)             # 双方等级 (0号位, 1号位)
    evo_cards: Tuple[Tuple[str, ...], Tuple[str, ...]] = ((), ()) # 双方选定的觉醒卡
    seed: int = 0                                  # 确定性随机数种子
    shuffle: bool = True                           # 开局是否随机洗切初始手牌顺序
    collisions: bool = True                        # 是否开启单位软碰撞推挤物理模拟


@dataclass(frozen=True)
class PlayResult:
    """下牌操作尝试的返回结果"""
    ok: bool                                       # 下牌是否成功
    reason: Optional[str] = None                   # 若失败，返回对应的错误原因标识
    event: Optional[PublicDeployEvent] = None      # 若成功，返回生成的公开出牌事件


@dataclass(frozen=True)
class MatchResult:
    """比赛终局胜负裁定结果"""
    winner: Optional[int]          # 胜者索引：0 或 1；None 表示最终平局
    crowns: Tuple[int, int]        # 双方最终比分皇冠数 (玩家皇冠, AI皇冠)
    reason: str                    # 获胜原因：king_destroyed (破三冠) | crowns (冠数领先) | tiebreaker (塔血判定) | draw (平局)
    t_ms: int                      # 比赛终局时刻 (毫秒)


ActionRecord = Tuple[int, int, str, float, float]  # 动作日志元组 (时间ms, 阵营, 卡牌ID, 落点x, 落点y)


# ---------------------------------------------------------------------------
# 战斗对局主引擎 (Match Engine)
# ---------------------------------------------------------------------------
class Match:
    """
    对局物理与逻辑引擎
    维护整个 18x32 竞技场的所有单位碰撞、攻击索敌、圣水流转与终局胜负。
    """

    def __init__(self, cfg: MatchConfig, db: Optional[CardDB] = None):
        self.db = db or CardDB.load()
        self.cfg = cfg
        r = self.db.rules
        # 圣水阶段配置：如从 2:00 开始双倍 (2x)，从 4:00 开始三倍 (3x)
        self._phases = sorted((int(p["from_s"] * 1000), int(p["multiplier"])) for p in r["elixir_phases"])
        self._ms_per_elixir = {int(k): float(v) for k, v in r["ms_per_elixir"].items()}
        self._max_elixir = float(r["max_elixir"])
        self.regulation_ms = int(r["regulation_s"] * 1000) # 常规比赛时长 (如 3 分钟 = 180,000ms)
        self.end_ms = self.regulation_ms + int(r["overtime_s"] * 1000) # 加上加时赛后的总时长上限 (5 分钟)

        self.rng = Pcg32(cfg.seed)
        self.t_ms = 0
        self.units: Dict[int, Unit] = {}
        self.spells: List[Spell] = []
        self.rolling: List[RollingSpell] = []
        self.projectiles: List[Projectile] = []
        self.effects: List[AreaEffect] = []
        self.tower_uid: Dict[Tuple[int, str], int] = {}
        self.next_uid = 1
        self.next_event_id = 1
        self.events: List[PublicDeployEvent] = []
        self.action_log: List[ActionRecord] = []
        self.result: Optional[MatchResult] = None
        self.status_used = False   # set once any status effect is applied (keeps the hot path cheap)
        self.sides = [self._make_side(s, float(r["start_elixir"])) for s in (0, 1)]
        self._spawn_towers()

    # -- 初始化辅助方法 --------------------------------------------------------
    def _make_side(self, side: int, start_elixir: float) -> SideState:
        """构建单边阵营的初始牌库与手牌"""
        deck = list(self.cfg.decks[side])
        n = self.db.rules["deck_size"]
        if len(deck) != n or len(set(deck)) != n:
            raise ValueError(f"side {side}: deck must have {n} unique cards")
        for c in deck:
            if c not in self.db.cards:
                raise ValueError(f"side {side}: unknown card '{c}'")
        if self.cfg.shuffle:
            self.rng.shuffle(deck)
        hs = self.db.rules["hand_size"]
        evo = set(self.cfg.evo_cards[side])
        for c in evo:
            if c not in deck or not self.db.has_variant(c, "evo"):
                raise ValueError(f"side {side}: '{c}' cannot be an evolution slot")
        return SideState(self.cfg.levels[side], deck[:hs], deck[hs:], start_elixir,
                         evo_cards=evo, evo_counter={c: 0 for c in evo})

    def _uid(self) -> int:
        """生成全局唯一的实体自增 ID"""
        self.next_uid += 1
        return self.next_uid - 1

    def _spawn_towers(self) -> None:
        """在竞技场固定点位生成双方的国王塔与左右公主塔"""
        for side in (0, 1):
            for role in ("king", "princess_left", "princess_right"):
                tid = "king_tower" if role == "king" else "princess_tower"
                st = self.db.tower_stats(tid, self.sides[side].level)
                atk = self.db.towers[tid]["attack"]
                x, y, rad = G.tower_pos(side, role)
                proj = atk.get("projectile_speed_tiles_s")
                u = Unit(self._uid(), side, tid, "tower", "ground", x, y, rad, st["hitpoints"], st["hitpoints"],
                         st["damage"], int(atk["hit_speed_s"] * 1000), int(atk["hit_speed_s"] * 1000),
                         atk["range_tiles"], tuple(atk["targets"]), tower_role=role, active=(role != "king"),
                         mechanics=("projectile",) if proj else (),
                         params={"projectile": {"speed_tiles_s": proj}} if proj else {})
                self.units[u.uid] = u
                self.tower_uid[(side, role)] = u.uid

    # -- 状态查询接口 ----------------------------------------------------------
    def tower(self, side: int, role: str) -> Optional[Unit]:
        """获取指定阵营特定角色的存活防御塔对象 (若已破塔则返回 None)"""
        u = self.units.get(self.tower_uid[(side, role)])
        return u if u is not None and u.hp > 0 else None

    def multiplier(self, t_ms: Optional[int] = None) -> int:
        """获取指定时刻的圣水产出倍率 (1x 常规, 2x 双倍, 3x 三倍)"""
        t = self.t_ms if t_ms is None else t_ms
        m = 1
        for start, mult in self._phases:
            if t >= start:
                m = mult
        return m

    def _princess_alive(self, side: int) -> Dict[str, bool]:
        """获取某一方左右公主塔的存活布尔字典"""
        return {"left": self.tower(side, "princess_left") is not None,
                "right": self.tower(side, "princess_right") is not None}

    def check_play(self, side: int, card_id: str, x: float, y: float) -> Optional[str]:
        """只读检测：验证在当前局势下出牌是否合法。合法返回 None，非法返回错误码"""
        if self.result is not None:
            return "match_over"
        st = self.sides[side]
        if card_id not in st.hand:
            return "not_in_hand"
        if st.elixir + 1e-9 < self.db.cards[card_id]["elixir"]:
            return "not_enough_elixir"
        if not G.in_bounds(x, y):
            return "out_of_bounds"
        sx, sy = G.snap(x, y)
        blockers = [(u.x, u.y, u.radius) for u in self.units.values() if u.is_structure and u.hp > 0]
        if any(m["kind"] == "burrow" for m in self.db.cards[card_id]["mechanics"]):
            # burrowing units (Miner) may be dug in anywhere on the ground: not in the river, not on a structure
            if G.region(sy) == "river" and not G.on_bridge(sx):
                return "river"
            if any(abs(sx - bx) < br and abs(sy - by) < br for bx, by, br in blockers):
                return "blocked_by_structure"
            return None
        return G.deploy_check(side, self.db.cards[card_id]["type"], sx, sy, self._princess_alive(1 - side), blockers)

    # -- 出牌动作执行 ----------------------------------------------------------
    def play(self, side: int, card_id: str, x: float, y: float) -> PlayResult:
        """执行下牌动作：扣除圣水、推进手牌循环、检测觉醒触发、生成场上实体"""
        reason = self.check_play(side, card_id, x, y)
        if reason:
            return PlayResult(False, reason)
        x, y = G.snap(x, y)
        st = self.sides[side]
        card = self.db.cards[card_id]
        cost = card["elixir"]
        st.elixir -= cost
        st.plays += 1
        # 手牌循环：打出的卡牌进入 queue 队尾，从 queue 队头补充新卡进手牌
        idx = st.hand.index(card_id)
        st.queue.append(card_id)
        st.hand[idx] = st.queue.pop(0)

        # 觉醒充能机制：达到指定出牌轮次后触发觉醒进化形态
        evolved = False
        if card_id in st.evo_cards:
            need = self.db.evo_cycles(card_id)
            if st.evo_counter[card_id] >= need:
                evolved, st.evo_counter[card_id] = True, 0
            else:
                st.evo_counter[card_id] += 1

        ev = PublicDeployEvent(self.next_event_id, self.t_ms, side, card_id, cost, TilePos(x, y),
                               is_evolved=evolved, is_threat=self._is_threat(side, card["type"], y),
                               variant="evo" if evolved else "base")
        self.next_event_id += 1
        self.events.append(ev)
        self.action_log.append((self.t_ms, side, card_id, x, y))

        if card["type"] == "spell":
            self._cast_spell(side, card_id, x, y, ev.event_id)
        else:
            self._spawn_card_units(side, card_id, x, y, ev.event_id, evolved)
        return PlayResult(True, event=ev)

    @staticmethod
    def _is_threat(side: int, card_type: str, y: float) -> bool:
        """判定该下牌是否构成立即威胁 (靠近河道或直接在敌方半场施放)"""
        enemy_home = G.home_region(1 - side)
        if card_type == "spell":
            return G.region(y) == enemy_home
        depth_to_river = (G.RIVER_Y0 - y) if side == 0 else (y - G.RIVER_Y1)
        return depth_to_river < THREAT_DEPTH or G.region(y) == enemy_home

    def _spawn_card_units(self, side: int, card_id: str, x: float, y: float, eid: int, evolved: bool,
                          deploy_ms: Optional[int] = None) -> None:
        """在指定网格生成军队或建筑实体 (支持多单位阵型散开生成，如骷髅军团/三枪)；
        card_id 也可以是 token (由卡牌生成的单位)，deploy_ms 覆盖其部署时间"""
        card = self.db.entry(card_id)
        variant = "evo" if evolved else "base"
        s = self.db.stats(card_id, self.sides[side].level, variant)
        atk, mov = s.attack, s.movement
        mechs = [m for m in self.db.mechanics_for(card_id, variant) if m["kind"] in REGISTRY]
        kinds = tuple(m["kind"] for m in mechs)
        kind = "building" if s.type == "building" else "troop"
        n = int(card.get("count", 1))
        if kind == "building" and "footprint_tiles" in card:
            radius = card["footprint_tiles"][0] / 2
        else:
            radius = float(card.get("collision_radius_tiles", UNIT_RADIUS))
        decay = 0.0
        if kind == "building" and card.get("lifetime_s"):
            decay = s.hitpoints / (card["lifetime_s"] * 1000 / G.TICK_MS)
        params = {m["kind"]: dict(m["params"]) for m in mechs}
        if "death_damage" in params:       # the burst scales with level like any other damage stat
            params["death_damage"]["damage"] = s.stats.get("death_damage", 0)
        fill_spawn_params(params, s)
        for i in range(n):
            if n > MAX_SPAWN:
                raise ValueError(f"{card_id}: count {n} > MAX_SPAWN {MAX_SPAWN}")
            ox, oy = SPAWN_RING[n][i] if n > 1 else (0.0, 0.0)
            u = Unit(
                self._uid(), side, card_id, kind, mov.get("layer", "ground"),
                min(max(x + ox, 0.0), G.W - 0.01), min(max(y + oy, 0.0), G.H - 0.01), radius,
                s.hitpoints, s.hitpoints, s.damage or 0,
                int(atk.get("hit_speed_s", 1.0) * 1000), int(atk.get("first_hit_s", 0.5) * 1000),
                float(atk.get("range_tiles", 1.0)), tuple(atk.get("targets", ["ground"])),
                splash=float(atk.get("splash_radius_tiles", 0.0)),
                speed=float(mov.get("speed_tiles_s", 0.0)),
                sight=float(atk.get("sight_tiles", TROOP_SIGHT)),
                deploy_ms=int(card.get("deploy_time_s", 1.0) * 1000) if deploy_ms is None else deploy_ms,
                cooldown_ms=int(atk.get("first_hit_s", 0.5) * 1000),
                decay_per_tick=decay, source_event_id=eid, evolved=evolved,
                mass=0.0 if kind == "building" else float(card.get("mass") or 1),
                variant=variant, mechanics=kinds, params=copy.deepcopy(params),
            )
            self.units[u.uid] = u
            for mech in hooks_for(u.mechanics, "on_spawn"):
                mech.on_spawn(self, u)

    def _cast_spell(self, side: int, card_id: str, x: float, y: float, eid: int) -> None:
        """释放法术：交给对应的法术类型组件 (area_spell 弹道落地 / rolling_spell 滚木)"""
        kind = self.db.spell_kind(card_id)
        params = {m["kind"]: dict(m["params"]) for m in self.db.mechanics_for(card_id)}
        SPELLS[kind].cast(self, side, card_id, x, y, eid, self.db.stats(card_id, self.sides[side].level), params)

    # -- 核心仿真步进循环 (Simulation Step) ------------------------------------
    def step(self) -> None:
        """向前推进一个固定时间步长 (50ms)，执行物理移动、索敌、攻击与碰撞"""
        if self.result is not None:
            return
        dt = G.TICK_MS
        # 1. 圣水自然充能与漏水计算
        gain = dt / self._ms_per_elixir[self.multiplier()]
        for st in self.sides:
            st.elixir += gain
            if st.elixir > self._max_elixir:
                st.leaked_elixir += st.elixir - self._max_elixir
                st.elixir = self._max_elixir

        # 2. 投掷法术飞行与落地爆炸伤害结算
        for sp in list(self.spells):
            sp.impact_in_ms -= dt
            if sp.impact_in_ms <= 0:
                SPELLS[sp.kind].impact(self, sp)
                self.spells.remove(sp)

        # 3. 滚动法术 (滚木) 推进与沿途贯穿伤害
        for rs in list(self.rolling):
            SPELLS["rolling_spell"].advance(self, rs, dt)
            if rs.remaining <= 0:
                self.rolling.remove(rs)
                SPELLS["rolling_spell"].finish(self, rs)

        # 3a. 持续区域效果 (龙卷风)：拉拽敌方单位并按间隔造成伤害
        for ef in list(self.effects):
            SPELLS[ef.kind].advance(self, ef, dt)
            if ef.elapsed_ms >= ef.duration_ms:
                self.effects.remove(ef)

        # 3b. 弹道飞行物 (火枪手、弓箭手、加农炮、防御塔…) 追踪目标，抵达时结算伤害
        if self.projectiles:
            self.projectiles = [p for p in self.projectiles if not advance_projectile(self, p, dt)]

        # 4. 场上全部单位的战斗行为逻辑 (按 uid 稳定排序以保证绝对确定性)
        for uid in sorted(self.units):
            u = self.units[uid]
            if u.hp <= 0:
                continue
            if u.status:
                S.tick(u, dt)
            if u.deploy_ms > 0:
                u.deploy_ms -= dt
                continue
            # 机制组件的逐帧钩子 (如建筑自然衰减)；返回 True 表示该单位本帧结束
            if u.mechanics and self._tick_hooks(u, dt):
                continue
            if u.kind == "tower" and not u.active:
                continue
            if u.status and not S.can_attack(u) and not S.can_move(u):
                continue
            u.cooldown_ms = max(0, u.cooldown_ms - (S.cooldown_dt(u, dt) if u.status else dt))
            # 索敌锁定目标
            tgt = self._acquire(u)
            if tgt is None:
                continue
            # 若在射程内，且攻速冷却完毕，发起普通攻击
            if self._in_range(u, tgt):
                if u.cooldown_ms == 0 and (not u.status or S.can_attack(u)):
                    self._attack(u, tgt)
                    u.cooldown_ms = u.hit_speed_ms
            elif u.kind == "troop" and (not u.status or S.can_move(u)):
                u.cooldown_ms = max(u.cooldown_ms, u.first_hit_ms)
                self._move(u, tgt, dt)

        # 5. 单位之间软圆形碰撞挤压与分流
        if self.cfg.collisions:
            self._resolve_collisions()
        # 6. 清理阵亡单位并更新比分皇冠
        self._resolve_deaths()
        self.t_ms += dt
        # 7. 终局胜负检测
        self._check_end()

    def run_until(self, t_ms: int) -> None:
        """连续运行仿真步进，直到达到目标毫秒时刻或比赛决出胜负"""
        while self.t_ms < t_ms and self.result is None:
            self.step()

    # -- 战斗细节运算辅助方法 --------------------------------------------------
    @staticmethod
    def _edge_dist(a: Unit, b: Unit) -> float:
        """计算两单位圆边缘之间的最短距离 (边缘距离 = 中心距离 - 两者半径)"""
        return max(0.0, G.dist(a.x, a.y, b.x, b.y) - a.radius - b.radius)

    def _in_range(self, u: Unit, t: Unit) -> bool:
        """检测目标是否在攻击者的射程之内"""
        return self._edge_dist(u, t) <= u.range + 1e-9

    @staticmethod
    def _can_target(u: Unit, e: Unit) -> bool:
        """根据防空/对地属性检测是否能够攻击该目标类型"""
        if e.is_structure:
            return "buildings" in u.targets or "ground" in u.targets
        return e.layer in u.targets

    def _acquire(self, u: Unit) -> Optional[Unit]:
        """
        AI 索敌目标锁定算法
        若已有存活目标且仍处于攻击范围内，保持攻击该目标（不频繁转火）。
        否则按照优先顺序（视野内最近敌人、或全图最近建筑）进行搜索。
        """
        cur = self.units.get(u.target_uid) if u.target_uid is not None else None
        if cur is not None and cur.hp > 0 and self._in_range(u, cur) and (not cur.status or S.targetable(cur)):
            return cur
        enemies = [e for e in self.units.values() if e.side != u.side and e.hp > 0 and self._can_target(u, e)]
        if self.status_used:
            enemies = [e for e in enemies if not e.status or S.targetable(e)]
        if u.is_structure:
            pool = [e for e in enemies if self._in_range(u, e)]
        elif u.building_only:
            pool = [e for e in enemies if e.is_structure]
        else:
            pool = [e for e in enemies if self._edge_dist(u, e) <= u.sight] or [e for e in enemies if e.is_structure]
        if not pool:
            u.target_uid = None
            return None
        # 选择边缘距离最近的有效目标
        best = min(pool, key=lambda e: (self._edge_dist(u, e), e.uid))
        u.target_uid = best.uid
        return best

    def _deal(self, t: Unit, amount: float, source=None) -> None:
        """造成伤害 (经过 on_damaged 钩子，如护盾/减伤) 并处理国王塔受击激活"""
        if t.mechanics:
            for mech in hooks_for(t.mechanics, "on_damaged"):
                amount = mech.on_damaged(self, t, amount, source)
        t.hp -= amount
        if t.kind == "tower" and t.tower_role == "king" and not t.active:
            t.active = True

    def _attack(self, u: Unit, t: Unit) -> None:
        """执行单次攻击：机制组件 (溅射、弹道…) 可接管，否则单体伤害"""
        if u.mechanics:
            for mech in hooks_for(u.mechanics, "on_attack"):
                if mech.on_attack(self, u, t):
                    return
        self._deal(t, u.damage, u)

    def _move(self, u: Unit, t: Unit, dt: int) -> None:
        """推进单位位移：空中单位直线飞向目标；地面单位依桥梁拐点智能寻路"""
        wp = None
        if u.mechanics:
            for mech in hooks_for(u.mechanics, "waypoint"):
                wp = mech.waypoint(self, u, t)
                if wp is not None:
                    break
        if u.layer == "air":
            wx, wy = t.x, t.y
        elif wp is not None:
            wx, wy = wp
        else:
            wx, wy = G.ground_waypoint(u.x, u.y, t.x, t.y)
        step = u.speed * dt / 1000.0
        if u.status:
            step *= S.speed_factor(u)
        d = G.dist(u.x, u.y, wx, wy)
        if (wx, wy) == (t.x, t.y):  # 避免直接走进目标核心内部
            step = min(step, max(0.0, d - u.radius - t.radius - u.range + 0.01))
        if d <= 1e-9:
            return
        f = min(1.0, step / d)
        u.x = min(max(u.x + (wx - u.x) * f, 0.0), G.W - 0.01)
        u.y = min(max(u.y + (wy - u.y) * f, 0.0), G.H - 0.01)

    def _resolve_collisions(self) -> None:
        """
        软圆形推挤碰撞系统 (Soft Circle Collision & Push Physics)
        同一层级 (同为地面或同为空中) 且体积发生重叠的单位相互推开。
        推力按质量比例 (mass) 分配：重质量单位 (如骑士/皮卡) 几乎不退，轻量单位 (如小骷髅) 被推开。
        建筑与防御塔质量为 0 (完全静止不可推动)。
        地面单位不可被挤入河道中。采用 x 轴一维扫掠修剪算法 (Sweep-and-prune) 实现 O(n log n) 高效碰撞。
        """
        bodies = sorted((u for u in self.units.values() if u.hp > 0), key=lambda u: (u.x, u.uid))
        if len(bodies) < 2:
            return
        max_r = max(u.radius for u in bodies)
        push: Dict[int, List[float]] = {}
        for i, a in enumerate(bodies):
            for b in bodies[i + 1:]:
                # 一维 x 坐标修剪，快速跳过不可能发生碰撞的单位对
                if b.x - a.x >= a.radius + max_r:
                    break
                if a.layer != b.layer or (a.mass == 0 and b.mass == 0):
                    continue
                dx, dy = b.x - a.x, b.y - a.y
                r = a.radius + b.radius
                d2 = dx * dx + dy * dy
                if d2 >= r * r:
                    continue
                d = math.sqrt(d2)
                if d < 1e-6:  # 若完全重合，基于 uid 生成伪随机方向分散
                    k = ((min(a.uid, b.uid) * 2654435761) & 0xFFFFFFFF) >> 26   # 0..63, golden-ratio hash
                    (nx, ny), d = UNIT_DIRS[k], 0.0
                else:
                    nx, ny = dx / d, dy / d
                overlap = (r - d) * COLLISION_STIFFNESS
                # 按两者的物理质量比分配位移排斥量
                ma, mb = a.mass, b.mass
                if ma == 0:
                    sa, sb = 0.0, 1.0
                elif mb == 0:
                    sa, sb = 1.0, 0.0
                else:
                    sa, sb = mb / (ma + mb), ma / (ma + mb)
                pa = push.setdefault(a.uid, [0.0, 0.0])
                pb = push.setdefault(b.uid, [0.0, 0.0])
                pa[0] -= nx * overlap * sa
                pa[1] -= ny * overlap * sa
                pb[0] += nx * overlap * sb
                pb[1] += ny * overlap * sb
        # 应用推挤位移，并限制最大位移与河道防落水边界限制
        for uid in sorted(push):
            u = self.units[uid]
            px, py = push[uid]
            n = math.sqrt(px * px + py * py)
            if n > MAX_PUSH_PER_TICK:
                px, py = px * MAX_PUSH_PER_TICK / n, py * MAX_PUSH_PER_TICK / n
            self._move_clamped(u, u.x + px, u.y + py)

    def _move_clamped(self, u: Unit, nx: float, ny: float) -> None:
        """Move a unit (pushes, knockback) inside the arena; ground units can't be pushed into the river."""
        ox, oy = u.x, u.y
        nx_, ny_ = min(max(nx, 0.0), G.W - 0.01), min(max(ny, 0.0), G.H - 0.01)
        # 地面单位若不是跳河兵种，严禁被推挤进河水区域 (桥梁除外)
        if u.layer == "ground" and not self._river_passable(u) and G.region(ny_) == "river" and not G.on_bridge(nx_):
            if G.region(oy) == "river":       # 原本在桥上被挤：限制在桥面范围内滑动
                bx = min(G.BRIDGE_X, key=lambda b: abs(b - ox))
                nx_ = min(max(nx_, bx - G.BRIDGE_HALF_W), bx + G.BRIDGE_HALF_W)
            else:                              # 原本在岸上：禁止落入河水
                ny_ = oy
        u.x, u.y = nx_, ny_

    def apply_status(self, u: Unit, tag: str, ms: int) -> None:
        """Mechanics apply status effects through here (stun, slow, root, invisible, freeze)."""
        self.status_used = True
        S.apply(u, tag, ms)

    def _tick_hooks(self, u: Unit, dt: int) -> bool:
        for mech in hooks_for(u.mechanics, "on_tick"):
            if mech.on_tick(self, u, dt):
                return True
        return False

    @staticmethod
    def _river_passable(u: Unit) -> bool:
        return any(REGISTRY[k].river_passable for k in u.mechanics)

    def _resolve_deaths(self) -> None:
        """结算生命值 <= 0 的阵亡单位并更新比分 (拔公主塔得 1 冠，拆国王塔直接 3 冠)"""
        for uid in sorted(self.units):
            u = self.units[uid]
            if u.hp > 0:
                continue
            if u.mechanics:
                for mech in hooks_for(u.mechanics, "on_death"):
                    mech.on_death(self, u)
            if u.kind == "tower":
                enemy = self.sides[1 - u.side]
                if u.tower_role == "king":
                    enemy.crowns = 3
                else:
                    enemy.crowns += 1
                    king = self.tower(u.side, "king")
                    if king is not None:
                        king.active = True # 爆了公主塔后，我方国王塔立即激活拔枪
            del self.units[uid]

    def _check_end(self) -> None:
        """判定比赛结束条件：三冠绝杀、常规时间分出胜负、加时赛塔血判定"""
        c = (self.sides[0].crowns, self.sides[1].crowns)
        # 1. 任一方破三冠，立即绝杀结束
        if 3 in c:
            self.result = MatchResult(c.index(3), c, "king_destroyed", self.t_ms)
        # 2. 常规时间走完且比分不同
        elif self.t_ms >= self.regulation_ms and c[0] != c[1]:
            self.result = MatchResult(0 if c[0] > c[1] else 1, c, "crowns", self.t_ms)
        # 3. 加时赛完全耗尽：比拼双方最低防御塔血量 (Tiebreaker 塔血判定)
        elif self.t_ms >= self.end_ms:
            lows = [min(u.hp for u in self.units.values() if u.kind == "tower" and u.side == s) for s in (0, 1)]
            if abs(lows[0] - lows[1]) < 1e-9:
                self.result = MatchResult(None, c, "draw", self.t_ms)
            else:
                self.result = MatchResult(0 if lows[0] > lows[1] else 1, c, "tiebreaker", self.t_ms)

    # -- 战争迷雾视角状态构建 (Observation) -------------------------------------
    def observation(self, side: int, is_visible: Optional[Callable[[int], bool]] = None) -> dict:
        """构建面向特定阵营 (side) 的环境观察字典 (遵守战争迷雾，不泄露未察觉的敌方动作)"""
        def seen(eid: Optional[int], owner: int) -> bool:
            return owner == side or eid is None or is_visible is None or is_visible(eid)

        st, opp = self.sides[side], self.sides[1 - side]
        units = [
            {"uid": u.uid, "side": u.side, "card_id": u.card_id, "kind": u.kind, "layer": u.layer,
             "x": u.x, "y": u.y, "hp": u.hp, "max_hp": u.max_hp, "deploying": u.deploy_ms > 0,
             "tower_role": u.tower_role, "active": u.active, "targets": u.targets}
            for u in sorted(self.units.values(), key=lambda u: u.uid) if seen(u.source_event_id, u.side)
        ]
        spells = [{"side": s.side, "card_id": s.card_id, "x": s.x, "y": s.y}
                  for s in [*self.spells, *self.effects] if seen(s.source_event_id, s.side)]
        return {
            "side": side,
            "t_ms": self.t_ms,
            "multiplier": self.multiplier(),
            "overtime": self.t_ms >= self.regulation_ms,
            "time_left_ms": (self.regulation_ms if self.t_ms < self.regulation_ms else self.end_ms) - self.t_ms,
            "elixir": st.elixir,
            "hand": list(st.hand),
            "next_card": st.queue[0],
            "evo_counter": dict(st.evo_counter),
            "level": st.level,
            "crowns": (st.crowns, opp.crowns),
            "units": units,
            "spells": spells,
        }

    # -- 快照保存、回溯与确定性哈希 (Snapshots & Determinism) -------------------
    _NO_COPY = ("db", "cfg", "rng")

    def snapshot(self) -> dict:
        """生成当前战斗状态的深拷贝快照 (包含 RNG 种子状态)，支持即时回溯"""
        snap = {k: copy.deepcopy(v) for k, v in self.__dict__.items() if k not in self._NO_COPY}
        snap["_rng_state"] = self.rng.getstate()
        return snap

    def restore(self, snap: dict) -> None:
        """从先前保存的快照中完全还原整个战斗引擎状态"""
        snap = dict(snap)
        self.rng = Pcg32()
        self.rng.setstate(snap.pop("_rng_state"))
        for k, v in snap.items():
            setattr(self, k, copy.deepcopy(v))

    def state_hash(self) -> str:
        """计算当前战斗状态的 SHA-256 截断哈希值，用于确定性回放一致性校验"""
        r = lambda v: round(v, 6)  # noqa: E731
        units = tuple((u.uid, u.side, u.card_id, r(u.x), r(u.y), r(u.hp), u.cooldown_ms, u.deploy_ms,
                       u.target_uid, u.active) for u in sorted(self.units.values(), key=lambda u: u.uid))
        sides = tuple((r(s.elixir), tuple(s.hand), tuple(s.queue), s.crowns) for s in self.sides)
        spells = tuple((s.uid, r(s.x), r(s.y), s.impact_in_ms) for s in self.spells)
        rolling = tuple((s.uid, r(s.y), r(s.remaining), tuple(sorted(s.hit))) for s in self.rolling)
        projectiles = tuple((p.uid, p.target_uid, r(p.x), r(p.y)) for p in self.projectiles)
        blob = repr((self.t_ms, sides, units, spells, rolling, projectiles, self.result))
        if self.effects:                   # lingering areas join the hash only while one exists
            blob += repr(tuple((e.uid, r(e.x), r(e.y), e.elapsed_ms) for e in self.effects))
        if self.status_used:               # statuses join the hash only once one exists: older hashes stay valid
            blob += repr(tuple((u.uid, tuple(sorted(u.status.items())))
                               for u in sorted(self.units.values(), key=lambda u: u.uid) if u.status))
        return hashlib.sha256(blob.encode()).hexdigest()[:16]

    @classmethod
    def replay(cls, cfg: MatchConfig, log: Sequence[ActionRecord], db: Optional[CardDB] = None,
               until_ms: Optional[int] = None) -> "Match":
        """根据完整的动作日志列表重放整场比赛"""
        m = cls(cfg, db)
        i = 0
        log = list(log)
        stop = until_ms if until_ms is not None else (log[-1][0] + G.TICK_MS if log else 0)
        while m.result is None and m.t_ms < stop:
            while i < len(log) and log[i][0] == m.t_ms:
                _, side, card, x, y = log[i]
                res = m.play(side, card, x, y)
                if not res.ok:
                    raise RuntimeError(f"replay diverged at {log[i]}: {res.reason}")
                i += 1
            m.step()
        return m
