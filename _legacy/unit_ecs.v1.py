"""
Entity Component System (ECS) for Clash Royale Simulator.
Models unit combat, targeting, movement, and specific card mechanics:
- Rune Giant's enhancement buff throwing to allies
- Little Prince's Royal Rescue Guardienne summon
- Inferno Dragon laser beam ramp
- Electro Wizard stun / ramp reset
- Goblin Cage death release of Brawler
"""
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import math


@dataclass
class Entity:
    uid: int
    name: str
    side: int  # 0 = own, 1 = enemy
    x: float   # normalized [0, 1]
    y: float   # normalized [0, 1]
    hp: float
    max_hp: float
    is_building: bool = False
    is_flying: bool = False
    targets_buildings_only: bool = False
    range_tiles: float = 1.0
    damage: float = 100.0
    hit_speed_s: float = 1.0
    move_speed_tiles_s: float = 1.2
    
    # State tracking
    attack_cooldown_s: float = 0.0
    current_target_uid: Optional[int] = None
    alive: bool = True
    
    # Specialized mechanics
    beam_ramp_stage: int = 0
    beam_target_uid: Optional[int] = None
    beam_time_s: float = 0.0
    
    buff_count: int = 0  # Attack count for 3rd attack multiplier (Rune Giant)
    is_buffed_by_rune_giant: bool = False
    
    stun_timer_s: float = 0.0


class UnitECS:
    """Manages active battlefield entities and ticks their combat logic."""

    def __init__(self):
        self.entities: Dict[int, Entity] = {}
        self.next_uid: int = 1
        self.time_s: float = 0.0

    def spawn(self, name: str, side: int, x: float, y: float, stats: dict) -> Entity:
        uid = self.next_uid
        self.next_uid += 1
        
        hp = stats.get("hitpoints_l16", stats.get("hitpoints_l11", 1000.0))
        ent = Entity(
            uid=uid,
            name=name,
            side=side,
            x=x,
            y=y,
            hp=hp,
            max_hp=hp,
            is_building=(stats.get("kind") == "building"),
            is_flying=(stats.get("movement") == "air"),
            targets_buildings_only=(stats.get("targets") == "buildings"),
            range_tiles=float(stats.get("range", 1.0)),
            damage=float(stats.get("damage_l16", stats.get("damage_l11", 100.0))),
            hit_speed_s=float(stats.get("hit_speed", 1.0)),
            move_speed_tiles_s=1.0 if stats.get("speed") == "medium" else 1.5
        )
        self.entities[uid] = ent
        
        # Immediate on-spawn triggers
        if name == "rune_giant":
            self._apply_rune_giant_buff(ent)
            
        return ent

    def trigger_little_prince_ability(self, lp_uid: int) -> Optional[Entity]:
        """Activate Royal Rescue: summons Guardienne in front of Little Prince."""
        lp = self.entities.get(lp_uid)
        if not lp or not lp.alive or lp.name != "little_prince":
            return None
        
        # Guardienne spawns 1 tile in front (toward enemy river)
        dy = -0.05 if lp.side == 0 else 0.05
        guardienne_stats = {
            "hitpoints_l16": 2576,
            "damage_l16": 373,
            "hit_speed": 1.2,
            "speed": "medium",
            "range": 1.2,
            "kind": "troop",
            "movement": "ground",
            "targets": "ground"
        }
        g = self.spawn("guardienne", lp.side, lp.x, lp.y + dy, guardienne_stats)
        
        # Knock back close enemy ground units
        for target in self.entities.values():
            if target.side != lp.side and target.alive and not target.is_flying and not target.is_building:
                dist = math.hypot((target.x - g.x) * 18.0, (target.y - g.y) * 32.0)
                if dist <= 3.0:
                    push_dy = -0.1 if lp.side == 0 else 0.1
                    target.y = max(0.05, min(0.95, target.y + push_dy))
                    target.stun_timer_s = 0.5
        return g

    def _apply_rune_giant_buff(self, rune_giant: Entity):
        """Buffs up to 2 closest friendly allied troops with 3rd-attack damage multiplier."""
        allies = [
            e for e in self.entities.values()
            if e.side == rune_giant.side and e.alive and e.uid != rune_giant.uid and not e.is_building
        ]
        allies.sort(key=lambda e: math.hypot((e.x - rune_giant.x) * 18.0, (e.y - rune_giant.y) * 32.0))
        for ally in allies[:2]:
            ally.is_buffed_by_rune_giant = True

    def tick(self, dt: float):
        """Simulate one engine step (typically dt = 0.05s = 1 tick at 20Hz)."""
        self.time_s += dt
        
        for ent in list(self.entities.values()):
            if not ent.alive:
                continue
                
            # Handle stun
            if ent.stun_timer_s > 0:
                ent.stun_timer_s = max(0.0, ent.stun_timer_s - dt)
                continue
                
            # Cooldowns
            if ent.attack_cooldown_s > 0:
                ent.attack_cooldown_s = max(0.0, ent.attack_cooldown_s - dt)
                
            # Inferno Dragon beam logic
            if ent.name == "inferno_dragon":
                self._tick_inferno_dragon(ent, dt)
            else:
                self._tick_standard_combat(ent, dt)

        # Cleanup dead entities
        for uid, ent in list(self.entities.items()):
            if ent.hp <= 0 and ent.alive:
                ent.alive = False
                self._handle_death_triggers(ent)
                del self.entities[uid]

    def _tick_inferno_dragon(self, idrag: Entity, dt: float):
        """Inferno Dragon ramps through 3 stages; resets if target changes or is stunned."""
        target = self._find_nearest_target(idrag)
        if not target:
            idrag.beam_target_uid = None
            idrag.beam_time_s = 0.0
            idrag.beam_ramp_stage = 0
            return
            
        if idrag.beam_target_uid != target.uid:
            idrag.beam_target_uid = target.uid
            idrag.beam_time_s = 0.0
            idrag.beam_ramp_stage = 0

        idrag.beam_time_s += dt
        if idrag.beam_time_s >= 4.0:
            idrag.beam_ramp_stage = 2
            dps = 1127.0
        elif idrag.beam_time_s >= 2.0:
            idrag.beam_ramp_stage = 1
            dps = 225.0
        else:
            idrag.beam_ramp_stage = 0
            dps = 56.0
            
        target.hp -= dps * dt

    def _tick_standard_combat(self, ent: Entity, dt: float):
        target = self._find_nearest_target(ent)
        if not target:
            # Move towards enemy crown side
            if not ent.is_building:
                target_y = 0.1 if ent.side == 0 else 0.9
                dy = (target_y - ent.y)
                step = (ent.move_speed_tiles_s / 32.0) * dt
                ent.y += math.copysign(min(abs(dy), step), dy)
            return

        dist_tiles = math.hypot((target.x - ent.x) * 18.0, (target.y - ent.y) * 32.0)
        if dist_tiles <= ent.range_tiles:
            # In range, attack
            if ent.attack_cooldown_s <= 0:
                ent.attack_cooldown_s = ent.hit_speed_s
                dmg = ent.damage
                
                # Rune Giant 3rd-attack enhancement
                if ent.is_buffed_by_rune_giant:
                    ent.buff_count += 1
                    if ent.buff_count % 3 == 0:
                        dmg *= 1.75
                        
                target.hp -= dmg
                
                # Electro Wizard stun trigger
                if ent.name == "electro_wizard":
                    target.stun_timer_s = 0.5
                    target.beam_ramp_stage = 0
                    target.beam_time_s = 0.0
        else:
            # Move towards target
            if not ent.is_building:
                dx = (target.x - ent.x)
                dy = (target.y - ent.y)
                angle = math.atan2(dy * 32.0, dx * 18.0)
                step_x = (math.cos(angle) * ent.move_speed_tiles_s * dt) / 18.0
                step_y = (math.sin(angle) * ent.move_speed_tiles_s * dt) / 32.0
                ent.x += step_x
                ent.y += step_y

    def _find_nearest_target(self, ent: Entity) -> Optional[Entity]:
        valid = []
        for other in self.entities.values():
            if other.side != ent.side and other.alive:
                if ent.targets_buildings_only and not other.is_building:
                    continue
                if other.is_flying and ent.range_tiles < 2.0:
                    continue  # Melee ground cannot hit flying
                dist = math.hypot((other.x - ent.x) * 18.0, (other.y - ent.y) * 32.0)
                valid.append((dist, other))
        if not valid:
            return None
        valid.sort(key=lambda t: t[0])
        return valid[0][1]

    def _handle_death_triggers(self, ent: Entity):
        # Goblin Cage drops Goblin Brawler
        if ent.name == "goblin_cage":
            brawler_stats = {
                "hitpoints_l16": 1578,
                "damage_l16": 507,
                "hit_speed": 1.1,
                "speed": "very_fast",
                "range": 0.8,
                "kind": "troop",
                "movement": "ground",
                "targets": "ground"
            }
            self.spawn("goblin_brawler", ent.side, ent.x, ent.y, brawler_stats)
            
        # Goblin Demolisher death bomb
        elif ent.name == "goblin_demolisher":
            bomb_dmg = 676.0
            for other in self.entities.values():
                if other.side != ent.side and other.alive and not other.is_flying:
                    dist = math.hypot((other.x - ent.x) * 18.0, (other.y - ent.y) * 32.0)
                    if dist <= 2.0:
                        other.hp -= bomb_dmg
