"""
Deterministic Clash-Royale-like match engine (Milestone 2).

Guarantees
  * Fixed tick (geometry.TICK_MS). All time is integer sim-ms.
  * Same (MatchConfig, action_log) -> identical state_hash() on the same machine.
  * snapshot()/restore() are exact, including the RNG, so undo / take-over / MCTS can branch.
  * The only information about a play that leaves the engine is PublicDeployEvent.
    observation(side, is_visible) applies fog of war for the opponent's units.

Not modelled yet (tracked for later milestones): projectiles
(hits are instant), knockback, stun/slow/freeze, death spawns, champion/hero abilities,
evolution effects (evolution charge is tracked, effects are not).
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

TROOP_SIGHT = 5.5
UNIT_RADIUS = 0.5
THREAT_DEPTH = 4.0          # a troop deployed within this many tiles of the river counts as a threat
COLLISION_STIFFNESS = 0.5   # fraction of the overlap resolved per tick (soft, like the game)
MAX_PUSH_PER_TICK = 0.25    # tiles


# ---------------------------------------------------------------------------
# State records
# ---------------------------------------------------------------------------
@dataclass
class Unit:
    uid: int
    side: int
    card_id: str
    kind: str                      # troop | building | tower
    layer: str                     # ground | air
    x: float
    y: float
    radius: float
    hp: float
    max_hp: float
    damage: int
    hit_speed_ms: int
    first_hit_ms: int
    range: float
    targets: Tuple[str, ...]
    splash: float = 0.0
    speed: float = 0.0             # tiles / s
    sight: float = TROOP_SIGHT
    river_jump: bool = False
    deploy_ms: int = 0             # remaining deploy time
    cooldown_ms: int = 0
    target_uid: Optional[int] = None
    decay_per_tick: float = 0.0
    tower_role: Optional[str] = None
    active: bool = True            # king tower starts inactive
    source_event_id: Optional[int] = None
    evolved: bool = False
    mass: float = 0.0              # push weight; 0 = immovable (towers, buildings)

    @property
    def building_only(self) -> bool:
        return self.targets == ("buildings",)

    @property
    def is_structure(self) -> bool:
        return self.kind in ("tower", "building")


@dataclass
class Spell:
    uid: int
    side: int
    card_id: str
    x: float
    y: float
    radius: float
    damage: int
    tower_damage: int
    impact_in_ms: int
    source_event_id: int


@dataclass
class RollingSpell:
    uid: int
    side: int
    card_id: str
    x: float
    y: float
    direction: int
    remaining: float
    width: float
    speed: float
    damage: int
    tower_damage: int
    source_event_id: int
    hit: Set[int] = field(default_factory=set)


@dataclass
class SideState:
    level: int
    hand: List[str]
    queue: List[str]
    elixir: float
    evo_cards: Set[str] = field(default_factory=set)
    evo_counter: Dict[str, int] = field(default_factory=dict)
    crowns: int = 0
    leaked_elixir: float = 0.0
    plays: int = 0


@dataclass(frozen=True)
class MatchConfig:
    decks: Tuple[Tuple[str, ...], Tuple[str, ...]]
    levels: Tuple[int, int] = (11, 11)
    evo_cards: Tuple[Tuple[str, ...], Tuple[str, ...]] = ((), ())
    seed: int = 0
    shuffle: bool = True
    collisions: bool = True


@dataclass(frozen=True)
class PlayResult:
    ok: bool
    reason: Optional[str] = None
    event: Optional[PublicDeployEvent] = None


@dataclass(frozen=True)
class MatchResult:
    winner: Optional[int]          # None = draw
    crowns: Tuple[int, int]
    reason: str                    # king_destroyed | crowns | tiebreaker | draw
    t_ms: int


ActionRecord = Tuple[int, int, str, float, float]  # (t_ms, side, card_id, x, y)


# ---------------------------------------------------------------------------
# Engine
# ---------------------------------------------------------------------------
class Match:
    def __init__(self, cfg: MatchConfig, db: Optional[CardDB] = None):
        self.db = db or CardDB.load()
        self.cfg = cfg
        r = self.db.rules
        self._phases = sorted((int(p["from_s"] * 1000), int(p["multiplier"])) for p in r["elixir_phases"])
        self._ms_per_elixir = {int(k): float(v) for k, v in r["ms_per_elixir"].items()}
        self._max_elixir = float(r["max_elixir"])
        self.regulation_ms = int(r["regulation_s"] * 1000)
        self.end_ms = self.regulation_ms + int(r["overtime_s"] * 1000)

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

    # -- setup -------------------------------------------------------------
    def _make_side(self, side: int, start_elixir: float) -> SideState:
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
        self.next_uid += 1
        return self.next_uid - 1

    def _spawn_towers(self) -> None:
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

    # -- queries -----------------------------------------------------------
    def tower(self, side: int, role: str) -> Optional[Unit]:
        u = self.units.get(self.tower_uid[(side, role)])
        return u if u is not None and u.hp > 0 else None

    def multiplier(self, t_ms: Optional[int] = None) -> int:
        t = self.t_ms if t_ms is None else t_ms
        m = 1
        for start, mult in self._phases:
            if t >= start:
                m = mult
        return m

    def _princess_alive(self, side: int) -> Dict[str, bool]:
        return {"left": self.tower(side, "princess_left") is not None,
                "right": self.tower(side, "princess_right") is not None}

    def check_play(self, side: int, card_id: str, x: float, y: float) -> Optional[str]:
        """None if the play is legal right now, else a reason. Does not mutate."""
        if self.result is not None:
            return "match_over"
        st = self.sides[side]
        if card_id not in st.hand:
            return "not_in_hand"
        if st.elixir + 1e-9 < self.db.cards[card_id]["elixir"]:
            return "not_enough_elixir"
        if not G.in_bounds(x, y):  # check before snap(), which clamps
            return "out_of_bounds"
        sx, sy = G.snap(x, y)
        blockers = [(u.x, u.y, u.radius) for u in self.units.values() if u.is_structure and u.hp > 0]
        return G.deploy_check(side, self.db.cards[card_id]["type"], sx, sy, self._princess_alive(1 - side), blockers)

    # -- actions -----------------------------------------------------------
    def play(self, side: int, card_id: str, x: float, y: float) -> PlayResult:
        reason = self.check_play(side, card_id, x, y)
        if reason:
            return PlayResult(False, reason)
        x, y = G.snap(x, y)
        st = self.sides[side]
        card = self.db.cards[card_id]
        cost = card["elixir"]
        st.elixir -= cost
        st.plays += 1
        idx = st.hand.index(card_id)
        st.queue.append(card_id)
        st.hand[idx] = st.queue.pop(0)

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
        enemy_home = G.home_region(1 - side)
        if card_type == "spell":
            return G.region(y) == enemy_home
        depth_to_river = (G.RIVER_Y0 - y) if side == 0 else (y - G.RIVER_Y1)
        return depth_to_river < THREAT_DEPTH or G.region(y) == enemy_home

    def _spawn_card_units(self, side: int, card_id: str, x: float, y: float, eid: int, evolved: bool) -> None:
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

    # -- simulation --------------------------------------------------------
    def step(self) -> None:
        if self.result is not None:
            return
        dt = G.TICK_MS
        gain = dt / self._ms_per_elixir[self.multiplier()]
        for st in self.sides:
            st.elixir += gain
            if st.elixir > self._max_elixir:
                st.leaked_elixir += st.elixir - self._max_elixir
                st.elixir = self._max_elixir

        for sp in list(self.spells):
            sp.impact_in_ms -= dt
            if sp.impact_in_ms <= 0:
                self._spell_impact(sp)
                self.spells.remove(sp)
        for rs in list(self.rolling):
            self._roll(rs, dt)
            if rs.remaining <= 0:
                self.rolling.remove(rs)

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
            tgt = self._acquire(u)
            if tgt is None:
                continue
            if self._in_range(u, tgt):
                if u.cooldown_ms == 0:
                    self._attack(u, tgt)
                    u.cooldown_ms = u.hit_speed_ms
            elif u.kind == "troop":
                u.cooldown_ms = max(u.cooldown_ms, u.first_hit_ms)
                self._move(u, tgt, dt)

        if self.cfg.collisions:
            self._resolve_collisions()
        self._resolve_deaths()
        self.t_ms += dt
        self._check_end()

    def run_until(self, t_ms: int) -> None:
        while self.t_ms < t_ms and self.result is None:
            self.step()

    # -- combat helpers ----------------------------------------------------
    @staticmethod
    def _edge_dist(a: Unit, b: Unit) -> float:
        return max(0.0, G.dist(a.x, a.y, b.x, b.y) - a.radius - b.radius)

    def _in_range(self, u: Unit, t: Unit) -> bool:
        return self._edge_dist(u, t) <= u.range + 1e-9

    @staticmethod
    def _can_target(u: Unit, e: Unit) -> bool:
        if e.is_structure:
            return "buildings" in u.targets or "ground" in u.targets
        return e.layer in u.targets

    def _acquire(self, u: Unit) -> Optional[Unit]:
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
        best = min(pool, key=lambda e: (self._edge_dist(u, e), e.uid))
        u.target_uid = best.uid
        return best

    def _deal(self, t: Unit, amount: float) -> None:
        t.hp -= amount
        if t.kind == "tower" and t.tower_role == "king" and not t.active:
            t.active = True

    def _attack(self, u: Unit, t: Unit) -> None:
        if u.splash <= 0:
            self._deal(t, u.damage)
            return
        for e in sorted(self.units.values(), key=lambda e: e.uid):
            if e.side != u.side and e.hp > 0 and self._can_target(u, e) \
                    and G.dist(e.x, e.y, t.x, t.y) <= u.splash + e.radius:
                self._deal(e, u.damage)

    def _move(self, u: Unit, t: Unit, dt: int) -> None:
        if u.layer == "air" or u.river_jump:
            wx, wy = t.x, t.y
        else:
            wx, wy = G.ground_waypoint(u.x, u.y, t.x, t.y)
        step = u.speed * dt / 1000.0
        d = G.dist(u.x, u.y, wx, wy)
        if (wx, wy) == (t.x, t.y):  # don't walk into the target's centre
            step = min(step, max(0.0, d - u.radius - t.radius - u.range + 0.01))
        if d <= 1e-9:
            return
        f = min(1.0, step / d)
        u.x = min(max(u.x + (wx - u.x) * f, 0.0), G.W - 0.01)
        u.y = min(max(u.y + (wy - u.y) * f, 0.0), G.H - 0.01)

    def _spell_impact(self, sp: Spell) -> None:
        for e in sorted(self.units.values(), key=lambda e: e.uid):
            if e.side != sp.side and e.hp > 0 and G.dist(e.x, e.y, sp.x, sp.y) <= sp.radius + e.radius:
                self._deal(e, sp.tower_damage if e.kind == "tower" else sp.damage)

    def _roll(self, rs: RollingSpell, dt: int) -> None:
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
        """Soft circle collision. Same-layer bodies that overlap are pushed apart; the lighter
        one moves more (game-data mass). Towers/buildings never move. Ground units are not
        pushed into the river (except jumpers). Sweep-and-prune on x keeps it ~O(n log n)."""
        bodies = sorted((u for u in self.units.values() if u.hp > 0), key=lambda u: (u.x, u.uid))
        if len(bodies) < 2:
            return
        max_r = max(u.radius for u in bodies)
        push: Dict[int, List[float]] = {}
        for i, a in enumerate(bodies):
            for b in bodies[i + 1:]:
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
                if d < 1e-6:  # exactly stacked: split along a direction fixed by the uids
                    ang = ((min(a.uid, b.uid) * 0.6180339887) % 1.0) * 2 * math.pi
                    nx, ny, d = math.cos(ang), math.sin(ang), 0.0
                else:
                    nx, ny = dx / d, dy / d
                overlap = (r - d) * COLLISION_STIFFNESS
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
        for uid in sorted(push):
            u = self.units[uid]
            px, py = push[uid]
            n = math.hypot(px, py)
            if n > MAX_PUSH_PER_TICK:
                px, py = px * MAX_PUSH_PER_TICK / n, py * MAX_PUSH_PER_TICK / n
            ox, oy = u.x, u.y
            nx_, ny_ = min(max(ox + px, 0.0), G.W - 0.01), min(max(oy + py, 0.0), G.H - 0.01)
            if u.layer == "ground" and not u.river_jump and G.region(ny_) == "river" and not G.on_bridge(nx_):
                if G.region(oy) == "river":       # was on the bridge: slide along it, don't fall off
                    bx = min(G.BRIDGE_X, key=lambda b: abs(b - ox))
                    nx_ = min(max(nx_, bx - G.BRIDGE_HALF_W), bx + G.BRIDGE_HALF_W)
                else:                              # was on a bank: stay on it
                    ny_ = oy
            u.x, u.y = nx_, ny_

    def _resolve_deaths(self) -> None:
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
                        king.active = True
            del self.units[uid]

    def _check_end(self) -> None:
        c = (self.sides[0].crowns, self.sides[1].crowns)
        if 3 in c:
            self.result = MatchResult(c.index(3), c, "king_destroyed", self.t_ms)
        elif self.t_ms >= self.regulation_ms and c[0] != c[1]:
            self.result = MatchResult(0 if c[0] > c[1] else 1, c, "crowns", self.t_ms)
        elif self.t_ms >= self.end_ms:
            lows = [min(u.hp for u in self.units.values() if u.kind == "tower" and u.side == s) for s in (0, 1)]
            if abs(lows[0] - lows[1]) < 1e-9:
                self.result = MatchResult(None, c, "draw", self.t_ms)
            else:
                self.result = MatchResult(0 if lows[0] > lows[1] else 1, c, "tiebreaker", self.t_ms)

    # -- observation (fog of war) -----------------------------------------
    def observation(self, side: int, is_visible: Optional[Callable[[int], bool]] = None) -> dict:
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

    # -- snapshots / determinism ------------------------------------------
    _NO_COPY = ("db", "cfg", "rng")

    def snapshot(self) -> dict:
        snap = {k: copy.deepcopy(v) for k, v in self.__dict__.items() if k not in self._NO_COPY}
        snap["_rng_state"] = self.rng.getstate()
        return snap

    def restore(self, snap: dict) -> None:
        snap = dict(snap)
        self.rng = random.Random()
        self.rng.setstate(snap.pop("_rng_state"))
        for k, v in snap.items():
            setattr(self, k, copy.deepcopy(v))

    def state_hash(self) -> str:
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
