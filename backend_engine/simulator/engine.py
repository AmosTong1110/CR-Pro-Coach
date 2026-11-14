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
import random
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Set, Tuple

from backend_engine.contracts import PublicDeployEvent, TilePos
from backend_engine.simulator import geometry as G
from backend_engine.simulator.card_db import CardDB

# 战斗感知与物理常量
TROOP_SIGHT = 5.5           # 军队通用视野警戒半径 (格子)
UNIT_RADIUS = 0.5           # 单位默认碰撞半径 (半格)
THREAT_DEPTH = 4.0          # 距河道 4 格以内的下兵直接判定为前线进攻威胁 (is_threat)
COLLISION_STIFFNESS = 0.5   # 碰撞重叠软缓解系数 (每帧消除 50% 重叠量，贴近官方手感)
MAX_PUSH_PER_TICK = 0.25    # 单帧最大位移推挤上限 (防止卡飞或穿模)


# ---------------------------------------------------------------------------
# 实体状态数据类 (State Records)
# ---------------------------------------------------------------------------
@dataclass
class Unit:
    """场上存活实体对象 (包括军队 Troop、防御建筑 Building、皇室防御塔 Tower)"""
    uid: int                       # 实体唯一数字 ID
    side: int                      # 归属方 (0=玩家, 1=AI)
    card_id: str                   # 对应卡牌 ID
    kind: str                      # 实体大类：troop | building | tower
    layer: str                     # 空间层级：ground (地面) | air (空中)
    x: float                       # 当前横坐标 (网格)
    y: float                       # 当前纵坐标 (网格)
    radius: float                  # 碰撞体积半径
    hp: float                      # 当前生命值
    max_hp: float                  # 生命值上限
    damage: int                    # 单次普通攻击伤害
    hit_speed_ms: int              # 攻击速度/攻速周期 (毫秒)
    first_hit_ms: int              # 接触目标后首次出刀延迟 (毫秒)
    range: float                   # 攻击射程 (格子)
    targets: Tuple[str, ...]       # 攻击目标喜好 ("ground", "air", "buildings")
    splash: float = 0.0            # 范围伤害半径 (0.0 为单体攻击)
    speed: float = 0.0             # 移动速度 (格 / 秒)
    sight: float = TROOP_SIGHT     # 索敌警戒半径 (格)
    river_jump: bool = False       # 是否具备跳河特性 (如野猪骑士、野蛮人攻城槌)
    deploy_ms: int = 0             # 部署读条剩余时间 (毫秒，读条期间不可移动和攻击)
    cooldown_ms: int = 0           # 距离下次攻击的冷却倒计时 (毫秒)
    target_uid: Optional[int] = None # 当前锁定的目标实体 ID
    decay_per_tick: float = 0.0    # 建筑每帧自然掉血量 (根据生命周期换算)
    tower_role: Optional[str] = None # 防御塔角色 ("king", "princess_left", "princess_right")
    active: bool = True            # 是否处于激活开炮状态 (国王塔初始为 False，挨打或破塔后激活)
    source_event_id: Optional[int] = None # 生成该单位的原始下牌事件 ID
    evolved: bool = False          # 是否为觉醒形态单位
    mass: float = 0.0              # 物理推挤质量 (0 = 无法被推动的静止建筑/防御塔，数值越大越不易被推开)

    @property
    def building_only(self) -> bool:
        """是否为只打建筑的推进单位 (如野猪骑士、巨人)"""
        return self.targets == ("buildings",)

    @property
    def is_structure(self) -> bool:
        """是否为静止结构体 (防御塔或召唤建筑)"""
        return self.kind in ("tower", "building")


@dataclass
class Spell:
    """范围投掷型法术实体 (如火球、火箭、小电等，具有飞行弹道时间)"""
    uid: int
    side: int
    card_id: str
    x: float                       # 目标落点 x
    y: float                       # 目标落点 y
    radius: float                  # 爆炸作用半径
    damage: int                    # 对常规单位伤害
    tower_damage: int              # 对皇室防御塔伤害 (通常有大幅衰减)
    impact_in_ms: int              # 距离着地爆炸的剩余飞行毫秒数
    source_event_id: int


@dataclass
class RollingSpell:
    """线性滚动型法术实体 (如滚木，沿直线向前推移并对路径上的地面目标造成贯穿伤害)"""
    uid: int
    side: int
    card_id: str
    x: float                       # 滚动起始中轴 x
    y: float                       # 当前滚动前锋 y
    direction: int                 # 推进方向 (+1 为向上滚，-1 为向下滚)
    remaining: float               # 剩余滚动距离 (格)
    width: float                   # 滚木横向碾压宽度
    speed: float                   # 滚动速度 (格 / 秒)
    damage: int                    # 常规伤害
    tower_damage: int              # 塔伤
    source_event_id: int
    hit: Set[int] = field(default_factory=set) # 已被击中过的实体 ID 集合 (避免单次滚动多重伤害)


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

        self.rng = random.Random(cfg.seed)
        self.t_ms = 0
        self.units: Dict[int, Unit] = {}
        self.spells: List[Spell] = []
        self.rolling: List[RollingSpell] = []
        self.tower_uid: Dict[Tuple[int, str], int] = {}
        self.next_uid = 1
        self.next_event_id = 1
        self.events: List[PublicDeployEvent] = []
        self.action_log: List[ActionRecord] = []
        self.result: Optional[MatchResult] = None
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
            if c not in deck or not self.db.cards[c].get("evolution"):
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
                u = Unit(self._uid(), side, tid, "tower", "ground", x, y, rad, st["hitpoints"], st["hitpoints"],
                         st["damage"], int(atk["hit_speed_s"] * 1000), int(atk["hit_speed_s"] * 1000),
                         atk["range_tiles"], tuple(atk["targets"]), tower_role=role, active=(role != "king"))
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
            need = card["evolution"]["cycles_required"]
            if st.evo_counter[card_id] >= need:
                evolved, st.evo_counter[card_id] = True, 0
            else:
                st.evo_counter[card_id] += 1

        ev = PublicDeployEvent(self.next_event_id, self.t_ms, side, card_id, cost, TilePos(x, y),
                               is_evolved=evolved, is_threat=self._is_threat(side, card["type"], y))
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

    def _spawn_card_units(self, side: int, card_id: str, x: float, y: float, eid: int, evolved: bool) -> None:
        """在指定网格生成军队或建筑实体 (支持多单位阵型散开生成，如骷髅军团/三枪)"""
        card = self.db.cards[card_id]
        s = self.db.stats(card_id, self.sides[side].level)
        atk, mov = s.attack, s.movement
        kind = "building" if s.type == "building" else "troop"
        n = int(card.get("count", 1))
        if kind == "building" and "footprint_tiles" in card:
            radius = card["footprint_tiles"][0] / 2
        else:
            radius = float(card.get("collision_radius_tiles", UNIT_RADIUS))
        decay = 0.0
        if kind == "building" and card.get("lifetime_s"):
            decay = s.hitpoints / (card["lifetime_s"] * 1000 / G.TICK_MS)
        for i in range(n):
            if n == 1:
                ox = oy = 0.0
            else:
                a = 2 * math.pi * i / n + math.pi / 2
                ox, oy = 0.7 * math.cos(a), 0.7 * math.sin(a)
            u = Unit(
                self._uid(), side, card_id, kind, mov.get("layer", "ground"),
                min(max(x + ox, 0.0), G.W - 0.01), min(max(y + oy, 0.0), G.H - 0.01), radius,
                s.hitpoints, s.hitpoints, s.damage or 0,
                int(atk.get("hit_speed_s", 1.0) * 1000), int(atk.get("first_hit_s", 0.5) * 1000),
                float(atk.get("range_tiles", 1.0)), tuple(atk.get("targets", ["ground"])),
                splash=float(atk.get("splash_radius_tiles", 0.0)),
                speed=float(mov.get("speed_tiles_s", 0.0)),
                sight=float(atk.get("sight_tiles", TROOP_SIGHT)),
                river_jump="river_jump" in s.mechanics,
                deploy_ms=int(card.get("deploy_time_s", 1.0) * 1000),
                cooldown_ms=int(atk.get("first_hit_s", 0.5) * 1000),
                decay_per_tick=decay, source_event_id=eid, evolved=evolved,
                mass=0.0 if kind == "building" else float(card.get("mass") or 1),
            )
            self.units[u.uid] = u

    def _cast_spell(self, side: int, card_id: str, x: float, y: float, eid: int) -> None:
        """释放法术：计算从国王塔发起到落点的弹道飞行耗时或创建滚木实体"""
        s = self.db.stats(card_id, self.sides[side].level)
        sp = s.spell
        dmg, tdmg = s.stats.get("damage", 0), s.stats.get("crown_tower_damage", s.stats.get("damage", 0))
        if "rolling_range_tiles" in sp:
            self.rolling.append(RollingSpell(self._uid(), side, card_id, x, y, G.forward(side),
                                             sp["rolling_range_tiles"], sp["width_tiles"],
                                             sp["travel_speed_tiles_s"], dmg, tdmg, eid))
        else:
            kx, ky, _ = G.tower_pos(side, "king")
            travel = G.dist(kx, ky, x, y) / sp.get("travel_speed_tiles_s", 1e9) * 1000
            self.spells.append(Spell(self._uid(), side, card_id, x, y, sp["radius_tiles"], dmg, tdmg,
                                     int(math.ceil(travel / G.TICK_MS) * G.TICK_MS), eid))

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
                self._spell_impact(sp)
                self.spells.remove(sp)

        # 3. 滚动法术 (滚木) 推进与沿途贯穿伤害
        for rs in list(self.rolling):
            self._roll(rs, dt)
            if rs.remaining <= 0:
                self.rolling.remove(rs)

        # 4. 场上全部单位的战斗行为逻辑 (按 uid 稳定排序以保证绝对确定性)
        for uid in sorted(self.units):
            u = self.units[uid]
            if u.hp <= 0:
                continue
            if u.deploy_ms > 0:
                u.deploy_ms -= dt
                continue
            if u.decay_per_tick:
                u.hp -= u.decay_per_tick
                if u.hp <= 0:
                    continue
            if u.kind == "tower" and not u.active:
                continue
            u.cooldown_ms = max(0, u.cooldown_ms - dt)
            # 索敌锁定目标
            tgt = self._acquire(u)
            if tgt is None:
                continue
            # 若在射程内，且攻速冷却完毕，发起普通攻击
            if self._in_range(u, tgt):
                if u.cooldown_ms == 0:
                    self._attack(u, tgt)
                    u.cooldown_ms = u.hit_speed_ms
            elif u.kind == "troop":
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
        if cur is not None and cur.hp > 0 and self._in_range(u, cur):
            return cur
        enemies = [e for e in self.units.values() if e.side != u.side and e.hp > 0 and self._can_target(u, e)]
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

    def _deal(self, t: Unit, amount: float) -> None:
        """造成伤害并处理国王塔受击激活"""
        t.hp -= amount
        if t.kind == "tower" and t.tower_role == "king" and not t.active:
            t.active = True

    def _attack(self, u: Unit, t: Unit) -> None:
        """执行单次攻击 (包含溅射 AoE 范围伤害处理)"""
        if u.splash <= 0:
            self._deal(t, u.damage)
            return
        for e in sorted(self.units.values(), key=lambda e: e.uid):
            if e.side != u.side and e.hp > 0 and self._can_target(u, e) \
                    and G.dist(e.x, e.y, t.x, t.y) <= u.splash + e.radius:
                self._deal(e, u.damage)

    def _move(self, u: Unit, t: Unit, dt: int) -> None:
        """推进单位位移：空中单位直线飞向目标；地面单位依桥梁拐点智能寻路"""
        if u.layer == "air" or u.river_jump:
            wx, wy = t.x, t.y
        else:
            wx, wy = G.ground_waypoint(u.x, u.y, t.x, t.y)
        step = u.speed * dt / 1000.0
        d = G.dist(u.x, u.y, wx, wy)
        if (wx, wy) == (t.x, t.y):  # 避免直接走进目标核心内部
            step = min(step, max(0.0, d - u.radius - t.radius - u.range + 0.01))
        if d <= 1e-9:
            return
        f = min(1.0, step / d)
        u.x = min(max(u.x + (wx - u.x) * f, 0.0), G.W - 0.01)
        u.y = min(max(u.y + (wy - u.y) * f, 0.0), G.H - 0.01)

    def _spell_impact(self, sp: Spell) -> None:
        """范围法术落地引爆：对半径内所有存活敌方实体造成伤害"""
        for e in sorted(self.units.values(), key=lambda e: e.uid):
            if e.side != sp.side and e.hp > 0 and G.dist(e.x, e.y, sp.x, sp.y) <= sp.radius + e.radius:
                self._deal(e, sp.tower_damage if e.kind == "tower" else sp.damage)

    def _roll(self, rs: RollingSpell, dt: int) -> None:
        """滚木前进物理推算：沿线横向矩形检测碰撞，避免多次重复伤害"""
        adv = min(rs.speed * dt / 1000.0, rs.remaining)
        y0, y1 = rs.y, rs.y + rs.direction * adv
        lo, hi = min(y0, y1), max(y0, y1)
        for e in sorted(self.units.values(), key=lambda e: e.uid):
            if e.side == rs.side or e.hp <= 0 or e.layer != "ground" or e.uid in rs.hit:
                continue
            if abs(e.x - rs.x) <= rs.width / 2 + e.radius and lo - e.radius <= e.y <= hi + e.radius:
                rs.hit.add(e.uid)
                self._deal(e, rs.tower_damage if e.kind == "tower" else rs.damage)
        rs.y, rs.remaining = y1, rs.remaining - adv

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
                    ang = ((min(a.uid, b.uid) * 0.6180339887) % 1.0) * 2 * math.pi
                    nx, ny, d = math.cos(ang), math.sin(ang), 0.0
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
            n = math.hypot(px, py)
            if n > MAX_PUSH_PER_TICK:
                px, py = px * MAX_PUSH_PER_TICK / n, py * MAX_PUSH_PER_TICK / n
            ox, oy = u.x, u.y
            nx_, ny_ = min(max(ox + px, 0.0), G.W - 0.01), min(max(oy + py, 0.0), G.H - 0.01)
            # 地面单位若不是跳河兵种，严禁被推挤进河水区域 (桥梁除外)
            if u.layer == "ground" and not u.river_jump and G.region(ny_) == "river" and not G.on_bridge(nx_):
                if G.region(oy) == "river":       # 原本在桥上被挤：限制在桥面范围内滑动
                    bx = min(G.BRIDGE_X, key=lambda b: abs(b - ox))
                    nx_ = min(max(nx_, bx - G.BRIDGE_HALF_W), bx + G.BRIDGE_HALF_W)
                else:                              # 原本在岸上：禁止落入河水
                    ny_ = oy
            u.x, u.y = nx_, ny_

    def _resolve_deaths(self) -> None:
        """结算生命值 <= 0 的阵亡单位并更新比分 (拔公主塔得 1 冠，拆国王塔直接 3 冠)"""
        for uid in sorted(self.units):
            u = self.units[uid]
            if u.hp > 0:
                continue
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
                  for s in self.spells if seen(s.source_event_id, s.side)]
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
        self.rng = random.Random()
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
        blob = repr((self.t_ms, sides, units, spells, rolling, self.result))
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
