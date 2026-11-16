"""
卡牌数据库加载器与官方等级数值缩放引擎 (Card Database & Official Level Scaling)

功能概述:
    db = CardDB.load()
    db.stats("knight", level=16).hitpoints   # 2822 (与官方游戏数据 100% 精确吻合，非浮点近似 1766 * 1.1**5 = 2844)

游戏数值缩放原理 (与 RoyaleAPI 解码的全部 4,945 条每级官方数值 100% 完全一致):
    stat(L) = floor(base_stat * M[i] / 100)
    i       = L - 1 - rarity_offset[rarity]      # L = 标准化卡牌等级 (11 = 锦标赛标准等级)
    M       = level_scaling.multiplier_pct       # [100, 110, 121, 133, 146, 160, 176, ...] 百分比倍率表

等级 L 属性数值的解析优先级:
  1. cards[id].level_overrides[str(L)][field]          手动填写的官方特定等级覆盖值
  2. base_stats[field] + 官方标准乘数表公式              100% 精确计算
  3. stats_l11[field] * M[i_L] / M[i_11] (向下取整)      对于缺失 1 级基础数值卡牌的近似推算
"""
from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from backend_engine.simulator.mechanics import SPELLS, known_kinds

# 默认读取的卡牌配置 JSON 路径
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "cards_stats_v2026.json"
# 支持的卡牌类别枚举
CARD_TYPES = {"troop", "building", "spell", "champion", "hero"}
SCHEMA_VERSION = 4
SUPPORTED_SCHEMAS = (3, 4)          # schema 3 files are upgraded in memory (normalise_card)
VARIANTS = ("base", "evo", "hero")
_RENAME = {"decaying_building": "decay"}


def _mech_list(tags, params: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    out = []
    for t in tags or []:
        if isinstance(t, dict):
            out.append({"kind": _RENAME.get(t["kind"], t["kind"]), "params": dict(t.get("params", {}))})
        else:
            out.append({"kind": _RENAME.get(t, t), "params": dict(params or {})})
    return out


def normalise_card(c: Dict[str, Any]) -> Dict[str, Any]:
    """Schema 3 card -> schema 4 shape (idempotent): mechanics as [{kind, params}], variants {evo, hero}."""
    c = copy.deepcopy(c)
    c["mechanics"] = _mech_list(c.get("mechanics"))
    if "variants" not in c:
        evo = c.pop("evolution", None)
        hero = c.pop("hero", None)
        c["variants"] = {
            "evo": None if not evo else {
                "cycles_required": int(evo["cycles_required"]),
                "mechanics": _mech_list(evo.get("mechanics"), evo.get("params")),
                "stat_mods": dict(evo.get("stat_mods", {})),
            },
            "hero": hero,
        }
    else:
        for v in c["variants"].values():
            if v:
                v["mechanics"] = _mech_list(v.get("mechanics"))
                v.setdefault("stat_mods", {})
    c.setdefault("aliases", {})
    return c


class CardDataError(ValueError):
    """卡牌配置文件校验异常"""
    pass


@dataclass(frozen=True)
class ResolvedCardStats:
    """已解析并按指定等级缩放的单卡全部数值对象"""
    card_id: str                          # 卡牌 ID
    level: int                            # 所属等级 (如 11 级锦标赛, 16 级满级)
    type: str                             # 类别 ("troop", "building", "spell")
    elixir: int                           # 圣水花费
    stats: Dict[str, int]                 # 等级缩放后的数值表 (hitpoints 生命值, damage 伤害等)
    attack: Dict[str, Any] = field(default_factory=dict)     # 攻击机制 (射程、攻击间隔、首刀延迟、目标类型等)
    movement: Dict[str, Any] = field(default_factory=dict)   # 移动机制 (移动速度、空中/地面、是否跳河等)
    spell: Dict[str, Any] = field(default_factory=dict)      # 法术属性 (作用半径、滚木宽度、击退等)
    mechanics: Tuple[str, ...] = ()       # 特殊机制标签 (如冲锋、觉醒、隐形、减速等)
    exact: bool = False                   # 若所有缩放数值均直接来自 base_stats 官方公式或 overrides 则为 True
    verified: bool = False                # 基础数值是否经过官方 API 交叉核验

    @property
    def hitpoints(self) -> Optional[int]:
        """生命值上限"""
        return self.stats.get("hitpoints")

    @property
    def damage(self) -> Optional[int]:
        """单次攻击伤害"""
        return self.stats.get("damage")

    @property
    def dps(self) -> Optional[float]:
        """每秒伤害输出 (DPS = damage / hit_speed)"""
        hs = self.attack.get("hit_speed_s")
        if self.damage is None or not hs:
            return None
        return self.damage / hs


class CardDB:
    """卡牌数据库主类：负责管理所有卡牌属性、防御塔属性及等级缩放运算"""

    def __init__(self, raw: Dict[str, Any]):
        if raw.get("schema_version") in SUPPORTED_SCHEMAS and isinstance(raw.get("cards"), dict):
            raw = {**raw, "cards": {cid: normalise_card(c) for cid, c in raw["cards"].items()}}
        if isinstance(raw.get("tokens"), dict):
            raw = {**raw, "tokens": {tid: normalise_card(c) for tid, c in raw["tokens"].items()}}
        self.raw = raw
        self._validate()
        sc = raw["level_scaling"]
        self.base_level: int = sc["base_level"]
        self.levels: List[int] = list(sc["supported_levels"])
        self.multiplier: List[int] = list(sc["multiplier_pct"])
        self.rarity_offset: Dict[str, Optional[int]] = dict(sc["rarity_offset"])
        self.scaled_fields = set(sc["scaled_fields"])
        self.cards: Dict[str, Dict[str, Any]] = raw["cards"]
        # tokens: units a card spawns (Barbarian from the Barbarian Barrel). They are never in a deck, so they live
        # apart from `cards`; `entry()` finds either. A token's `rarity` is its parent card's (for level scaling).
        self.tokens: Dict[str, Dict[str, Any]] = raw.get("tokens", {})
        self.towers: Dict[str, Dict[str, Any]] = raw.get("towers", {})
        self.rules: Dict[str, Any] = raw["rules"]

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> "CardDB":
        """从 JSON 配置文件加载卡牌数据库"""
        with open(path, "r", encoding="utf-8") as f:
            return cls(json.load(f))

    # ------------------------------------------------------------------
    def _validate(self) -> None:
        """数据模式与完整性校验"""
        r = self.raw
        for key in ("schema_version", "level_scaling", "rules", "cards"):
            if key not in r:
                raise CardDataError(f"missing top-level key '{key}'")
        if r["schema_version"] not in SUPPORTED_SCHEMAS:
            raise CardDataError(f"unsupported schema_version {r['schema_version']} (want {SCHEMA_VERSION})")
        known = known_kinds()
        offsets = r["level_scaling"].get("rarity_offset", {})
        clash = set(r["cards"]) & set(r.get("tokens", {}))
        if clash:
            raise CardDataError(f"token ids clash with card ids: {sorted(clash)}")
        for cid, c in [*r["cards"].items(), *r.get("tokens", {}).items()]:
            for key in ("name", "type", "rarity", "elixir"):
                if key not in c:
                    raise CardDataError(f"card '{cid}' missing '{key}'")
            if "base_stats" not in c and "stats_l11" not in c:
                raise CardDataError(f"card '{cid}' needs base_stats or stats_l11")
            if c["type"] not in CARD_TYPES:
                raise CardDataError(f"card '{cid}' has unknown type '{c['type']}'")
            if c["rarity"] not in offsets:
                raise CardDataError(f"card '{cid}' has unknown rarity '{c['rarity']}'")
            if not 0 <= c["elixir"] <= 10:
                raise CardDataError(f"card '{cid}' elixir out of range")
            stats = c.get("base_stats", c.get("stats_l11", {}))
            if c["type"] in ("troop", "building", "champion", "hero") and "hitpoints" not in stats:
                raise CardDataError(f"card '{cid}' ({c['type']}) needs hitpoints")
            lists = [c["mechanics"]] + [v["mechanics"] for v in c["variants"].values() if v]
            for m in (m for lst in lists for m in lst):
                if m["kind"] not in known:
                    raise CardDataError(f"card '{cid}' has unknown mechanic '{m['kind']}'")
            for v in c["variants"]:
                if v not in VARIANTS[1:]:
                    raise CardDataError(f"card '{cid}' has unknown variant '{v}'")
            for m in c["mechanics"]:
                tok = m["params"].get("token") if m["kind"] == "spawn_on_end" else None
                if m["kind"] == "spawn_on_end" and tok not in r.get("tokens", {}):
                    raise CardDataError(f"card '{cid}' spawns unknown token '{tok}'")

    def _index(self, rarity: str, level: int) -> int:
        """根据卡牌稀有度 (普通/稀有/史诗/传奇/英雄) 换算出在乘数表中的索引位置"""
        off = self.rarity_offset.get(rarity)
        if off is None:
            raise CardDataError(f"rarity '{rarity}' has no known level offset yet")
        i = level - 1 - off
        if not 0 <= i < len(self.multiplier):
            raise CardDataError(f"level {level} is not reachable for rarity '{rarity}'")
        return i

    def _scale(self, entry: Dict[str, Any], level: int) -> Tuple[Dict[str, int], bool]:
        """将某张卡牌或防御塔的属性根据指定等级进行官方乘数放大计算"""
        if level not in self.levels:
            raise CardDataError(f"level {level} not in supported_levels {self.levels}")
        rarity = entry.get("rarity", "common")
        overrides = entry.get("level_overrides", {}).get(str(level), {})
        m = self.multiplier[self._index(rarity, level)]
        if "base_stats" in entry:
            src, exact = entry["base_stats"], True
            scale = lambda v: int(math.floor(v * m / 100 + 1e-9))  # noqa: E731
        else:
            src, exact = entry["stats_l11"], False
            m11 = self.multiplier[self._index(rarity, self.base_level)]
            scale = lambda v: int(math.floor(v * m / m11 + 1e-6))  # noqa: E731
        out: Dict[str, int] = {}
        for k, v in src.items():
            if k in overrides:
                out[k] = int(overrides[k])
            elif k in self.scaled_fields:
                out[k] = scale(v)
            else:
                out[k] = v
        for k, v in overrides.items():
            out.setdefault(k, int(v))
        # 针对皇室皇冠防御塔的法术伤害衰减换算 (如法术对塔伤害降低 70%)
        pct = entry.get("crown_tower_damage_pct")
        if pct is not None and "crown_tower_damage" not in out and "damage" in out:
            # 游戏文件中存储的是衰减百分比 (如 -70 表示只造成原本 30% 伤害)
            out["crown_tower_damage"] = int(math.floor(out["damage"] * (100 + pct) / 100 + 0.5))
        return out, exact or bool(overrides) and set(overrides) >= set(src)

    def entry(self, card_id: str) -> Dict[str, Any]:
        """The data record of a card or a token (KeyError if neither)."""
        c = self.cards.get(card_id) or self.tokens.get(card_id)
        if c is None:
            raise KeyError(f"unknown card '{card_id}'")
        return c

    def stats(self, card_id: str, level: int = 11, variant: str = "base") -> ResolvedCardStats:
        """获取指定卡牌在特定等级 (默认 11 级) 下的完整属性结构体 (variant 的 stat_mods 为百分比)"""
        c = self.entry(card_id)
        stats, exact = self._scale(c, level)
        v = self.variant(card_id, variant) if variant != "base" else None
        for k, pct in ((v or {}).get("stat_mods") or {}).items():
            if k in stats:
                stats[k] = int(math.floor(stats[k] * pct / 100 + 1e-9))
        for k, val in (((v or {}).get("level_overrides") or {}).get(str(level)) or {}).items():
            stats[k] = int(val)               # exact per-level numbers of the variant (wiki), win over stat_mods
        return ResolvedCardStats(
            card_id=card_id,
            level=level,
            type=c["type"],
            elixir=c["elixir"],
            stats=stats,
            attack=dict(c.get("attack", {})),
            movement=dict(c.get("movement", {})),
            spell=dict(c.get("spell", {})),
            mechanics=tuple(m["kind"] for m in c.get("mechanics", [])),
            exact=exact,
            verified=bool(c.get("verified", False)),
        )

    def tower_stats(self, tower_id: str, level: int = 11) -> Dict[str, int]:
        """获取指定防御塔在特定等级下的属性数值表"""
        return self._scale(self.towers[tower_id], level)[0]

    def elixir_costs(self) -> Dict[str, int]:
        """获取全部卡牌的圣水花费字典映射"""
        return {cid: c["elixir"] for cid, c in self.cards.items()}

    # -- schema 4: variants, mechanics, aliases ---------------------------
    def variant(self, card_id: str, variant: str) -> Optional[Dict[str, Any]]:
        """The variant block ("evo" / "hero") of a card, or None if the card has no such variant."""
        if variant == "base":
            return {}
        return self.entry(card_id)["variants"].get(variant)

    def has_variant(self, card_id: str, variant: str) -> bool:
        return card_id in self.cards and bool(self.variant(card_id, variant))

    def evo_cycles(self, card_id: str) -> int:
        v = self.variant(card_id, "evo")
        if not v:
            raise KeyError(f"'{card_id}' has no evolution")
        return int(v["cycles_required"])

    def variant_simulated(self, card_id: str, variant: str) -> bool:
        """The card has this variant and every mechanic of it is simulated (planned-only variants can't be picked)."""
        v = self.variant(card_id, variant) if card_id in self.cards else None
        if not v:
            return False
        from backend_engine.simulator.mechanics import PLANNED
        return not any(m["kind"] in PLANNED for m in v.get("mechanics", []))

    def evo_capable(self) -> List[str]:
        return [cid for cid in self.cards if self.variant_simulated(cid, "evo")]

    def hero_capable(self) -> List[str]:
        return [cid for cid in self.cards if self.variant_simulated(cid, "hero")]

    def mechanics_for(self, card_id: str, variant: str = "base") -> List[Dict[str, Any]]:
        """Explicit mechanics (base + variant) plus ones derived from stats, deduplicated by kind:
        `splash` (attack splash radius > 0), `decay` (building lifetime), spell kind (area / rolling)."""
        c = self.entry(card_id)
        lst = list(c["mechanics"])
        if variant != "base":
            lst += list((self.variant(card_id, variant) or {}).get("mechanics", []))
        derived = []
        if c.get("attack", {}).get("projectile_speed_tiles_s"):
            derived.append("projectile")          # before splash: the projectile carries the splash
        if c.get("attack", {}).get("splash_radius_tiles", 0) > 0:
            derived.append("splash")
        if c["type"] == "building" and c.get("lifetime_s"):
            derived.append("decay")
        if c["type"] == "spell":
            derived.append(self.spell_kind(card_id))
        out, seen = [], set()
        dparams = {"projectile": {"speed_tiles_s": c.get("attack", {}).get("projectile_speed_tiles_s")}}
        for m in lst + [{"kind": k, "params": dparams.get(k, {})} for k in derived]:
            if m["kind"] not in seen:
                seen.add(m["kind"])
                out.append(m)
        return out

    def spell_kind(self, card_id: str) -> str:
        sp = self.cards[card_id].get("spell", {})
        explicit = [m["kind"] for m in self.cards[card_id]["mechanics"] if m["kind"] in SPELLS]
        if explicit:
            return explicit[0]
        return "rolling_spell" if "rolling_range_tiles" in sp else "area_spell"

    def by_alias(self, name: str, source: Optional[str] = None) -> Optional[str]:
        """Card id for an id, display name, internal name or alias (optionally from one source:
        hf / royaleapi / detector / display / internal_name)."""
        key = name.strip().lower()
        for cid, c in self.cards.items():
            a = c.get("aliases", {})
            cands = {
                "id": [cid], "internal_name": [c.get("internal_name") or ""], "display": [a.get("display") or c["name"]],
                "hf": [a.get("hf") or ""], "royaleapi": [a.get("royaleapi") or ""], "detector": list(a.get("detector", [])),
            }
            pool = cands.get(source, []) if source else [v for vs in cands.values() for v in vs]
            if any(key == v.lower() for v in pool if v):
                return cid
        return None

    def unsimulated_tags(self) -> Dict[str, List[str]]:
        """card id -> mechanic kinds in the data that the engine does not simulate yet."""
        from backend_engine.simulator.mechanics import PLANNED
        out: Dict[str, List[str]] = {}
        for cid, c in self.cards.items():
            lists = [c["mechanics"]] + [v["mechanics"] for v in c["variants"].values() if v]
            miss = sorted({m["kind"] for lst in lists for m in lst if m["kind"] in PLANNED})
            if miss:
                out[cid] = miss
        return out

    def unverified(self) -> List[str]:
        """返回尚未进行官方完全核验的卡牌列表"""
        return [cid for cid, c in self.cards.items() if not c.get("verified", False)]


def split_slot(slot: str) -> tuple:
    """Evolution / hero slot entry -> (card id, variant): "knight" = evolution, "knight#hero" = hero (CardRef form)."""
    cid, _, var = slot.partition("#")
    return cid, (var or "evo")
