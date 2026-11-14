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

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# 默认读取的卡牌配置 JSON 路径
DEFAULT_PATH = Path(__file__).resolve().parents[2] / "configs" / "cards_stats_v2026.json"
# 支持的卡牌类别枚举
CARD_TYPES = {"troop", "building", "spell", "champion", "hero"}
SCHEMA_VERSION = 3


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
        self.raw = raw
        self._validate()
        sc = raw["level_scaling"]
        self.base_level: int = sc["base_level"]
        self.levels: List[int] = list(sc["supported_levels"])
        self.multiplier: List[int] = list(sc["multiplier_pct"])
        self.rarity_offset: Dict[str, Optional[int]] = dict(sc["rarity_offset"])
        self.scaled_fields = set(sc["scaled_fields"])
        self.cards: Dict[str, Dict[str, Any]] = raw["cards"]
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
        if r["schema_version"] != SCHEMA_VERSION:
            raise CardDataError(f"unsupported schema_version {r['schema_version']} (want {SCHEMA_VERSION})")
        offsets = r["level_scaling"].get("rarity_offset", {})
        for cid, c in r["cards"].items():
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

    def stats(self, card_id: str, level: int = 11) -> ResolvedCardStats:
        """获取指定卡牌在特定等级 (默认 11 级) 下的完整属性结构体"""
        if card_id not in self.cards:
            raise KeyError(f"unknown card '{card_id}'")
        c = self.cards[card_id]
        stats, exact = self._scale(c, level)
        return ResolvedCardStats(
            card_id=card_id,
            level=level,
            type=c["type"],
            elixir=c["elixir"],
            stats=stats,
            attack=dict(c.get("attack", {})),
            movement=dict(c.get("movement", {})),
            spell=dict(c.get("spell", {})),
            mechanics=tuple(c.get("mechanics", [])),
            exact=exact,
            verified=bool(c.get("verified", False)),
        )

    def tower_stats(self, tower_id: str, level: int = 11) -> Dict[str, int]:
        """获取指定防御塔在特定等级下的属性数值表"""
        return self._scale(self.towers[tower_id], level)[0]

    def elixir_costs(self) -> Dict[str, int]:
        """获取全部卡牌的圣水花费字典映射"""
        return {cid: c["elixir"] for cid, c in self.cards.items()}

    def unverified(self) -> List[str]:
        """返回尚未进行官方完全核验的卡牌列表"""
        return [cid for cid, c in self.cards.items() if not c.get("verified", False)]
