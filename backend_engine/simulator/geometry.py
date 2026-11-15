"""
竞技场几何与空间坐标系统 (Arena Geometry & Spatial Math)
以格子 (Tile) 为单位定义皇室战争竞技场 (18 x 32 网格)。
Side 0 = 底部己方半场，Side 1 = 顶部敌方半场 (镜像映射 y -> 32 - y)。

纵向区域划分 (Rows):
  * y in [0, 15)  side-0 半场（南半场 / 玩家半场）
  * y in [15, 17) 河道区域（仅桥梁可通行，或由跳河/飞行单位跨越）
  * y in [17, 32) side-1 半场（北半场 / AI 半场）
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

# 模拟器基础时间步长：每帧 50 毫秒 (20 tick/秒)
TICK_MS = 50
# 竞技场宽度 18 格，长度 32 格
W, H = 18.0, 32.0
# 河道南北边界 y 坐标 (宽 2 格)
RIVER_Y0, RIVER_Y1 = 15.0, 17.0
# 左右两座桥梁中心 x 坐标 (左桥 3.5, 右桥 14.5)
BRIDGE_X = (3.5, 14.5)
# 桥梁宽度为 2 格 (半宽为 1.0)
BRIDGE_HALF_W = 1.0
# 当敌方对应路公主塔被摧毁后，允许在过河该路下牌的口袋深度 (6 格)
POCKET_DEPTH = 6.0

# 皇家防御塔基础布局：角色 -> (x, y, 碰撞与占地半径) (以 side 0 视角为基准，side 1 镜像翻转)
TOWER_LAYOUT: Dict[str, Tuple[float, float, float]] = {
    "princess_left": (3.5, 6.5, 1.5),   # 左路公主塔
    "princess_right": (14.5, 6.5, 1.5), # 右路公主塔
    "king": (9.0, 3.0, 2.0),            # 国王塔 (居中)
}


def mirror_y(y: float) -> float:
    """沿竞技场中线上下镜像翻转 y 坐标 (y -> 32 - y)"""
    return H - y


def to_side(side: int, x: float, y: float) -> Tuple[float, float]:
    """将基于 side-0 (底部玩家视角) 的局部坐标转换为竞技场绝对坐标"""
    return (x, y) if side == 0 else (x, mirror_y(y))


def tower_pos(side: int, role: str) -> Tuple[float, float, float]:
    """获取指定阵营特定防御塔的绝对坐标与半径 (x, y, radius)"""
    x, y, r = TOWER_LAYOUT[role]
    return (x, y if side == 0 else mirror_y(y), r)


def lane_of(x: float) -> str:
    """根据横坐标 x 判断所在路别：左路 ("left") 或 右路 ("right")"""
    return "left" if x < W / 2 else "right"


def region(y: float) -> str:
    """判断纵坐标 y 所处的竞技场地理区域："south" (南), "river" (河道), "north" (北)"""
    if y < RIVER_Y0:
        return "south"
    if y < RIVER_Y1:
        return "river"
    return "north"


def home_region(side: int) -> str:
    """获取阵营所属的自家半场区域标识 (side 0 -> "south", side 1 -> "north")"""
    return "south" if side == 0 else "north"


def forward(side: int) -> int:
    """获取阵营向前进攻推进的 y 轴方向向量 (+1 表示向 y 增大方向进攻，-1 表示向 y 减小方向进攻)"""
    return 1 if side == 0 else -1


def snap(x: float, y: float) -> Tuple[float, float]:
    """将任意浮点坐标吸附到所在格子的几何中心点 (如 3.2, 5.8 -> 3.5, 5.5)"""
    tx = min(max(math.floor(x), 0), int(W) - 1) + 0.5
    ty = min(max(math.floor(y), 0), int(H) - 1) + 0.5
    return tx, ty


def in_bounds(x: float, y: float) -> bool:
    """检测坐标是否在竞技场有效边界之内 [0, 18) x [0, 32)"""
    return 0.0 <= x < W and 0.0 <= y < H


def dist(ax: float, ay: float, bx: float, by: float) -> float:
    """计算两点间的欧几里得距离 (Euclidean Distance)"""
    dx, dy = ax - bx, ay - by
    return math.sqrt(dx * dx + dy * dy)   # sqrt is correctly rounded (IEEE 754); hypot is not portable


def on_bridge(x: float) -> bool:
    """检测横坐标 x 是否处于左桥或右桥的可通行跨度范围内"""
    return any(abs(x - b) <= BRIDGE_HALF_W for b in BRIDGE_X)


def deploy_check(
    side: int,
    card_type: str,
    x: float,
    y: float,
    enemy_princess_alive: Dict[str, bool],
    blockers: Iterable[Tuple[float, float, float]] = (),
) -> Optional[str]:
    """
    下牌物理合法性校验 (Deploy Validity Check)
    返回 None 表示下牌合法；否则返回对应的错误原因标识码 (例如 "enemy_half", "river" 等)。
    
    参数说明:
      side: 下牌阵营 (0 或 1)
      card_type: 卡牌大类 ("spell", "troop", "building")
      x, y: 已吸附到中心点的目标坐标
      enemy_princess_alive: 敌方左右公主塔存活状态字典 {"left": bool, "right": bool}
      blockers: 场上现存防御塔或建筑的占地范围集合 [(x, y, radius), ...]
    """
    if not in_bounds(x, y):
        return "out_of_bounds"
    # 法术类卡牌（火球、箭雨、小电等）全图任意位置可下
    if card_type == "spell":
        return None
    # 军队与建筑绝对不能直接放置在河道中央
    if region(y) == "river":
        return "river"
    # 检查是否与场上已有防御塔或建筑模型重叠冲突
    for bx, by, br in blockers:
        if abs(x - bx) < br and abs(y - by) < br:
            return "blocked_by_structure"
    # 自家半场内下牌完全合法
    if region(y) == home_region(side):
        return None
    # 破塔口袋区逻辑 (Pocket Deploy): 
    # 若敌方某一路公主塔已被拔掉，允许在靠近河道的该路敌方半场前沿 (深度 < POCKET_DEPTH) 下兵偷袭
    depth = (y - RIVER_Y1) if side == 0 else (RIVER_Y0 - y)
    if 0 <= depth < POCKET_DEPTH and not enemy_princess_alive[lane_of(x)]:
        return None
    return "enemy_half"


def ground_waypoint(x: float, y: float, tx: float, ty: float) -> Tuple[float, float]:
    """
    地面常规单位寻路路径点计算 (Ground Waypoint Navigation)
    当目标位于河道另一侧时，地面单位必须优先导向最近的桥梁中轴线。
    
    参数:
      x, y: 单位当前坐标
      tx, ty: 攻击目标终点坐标
    返回:
      当前帧应该朝向移动的下一个拐点目标 (nx, ny)
    """
    r, tr = region(y), region(ty)
    # 如果处于同一半场，或者目标就在河道上，直接直线走向目标
    if r == tr or tr == "river":
        return tx, ty
    # 选择总曼哈顿距离最近的一座桥梁作为过河桥头堡
    bx = min(BRIDGE_X, key=lambda b: abs(x - b) + abs(tx - b))
    # 若当前单位正处于河道内
    if r == "river":
        if abs(x - bx) > BRIDGE_HALF_W:
            return bx, y  # 横向滑回桥面，防止跌落河水
        return (bx, RIVER_Y1) if tr == "north" else (bx, RIVER_Y0 - 1e-3)
    # 处于南半场，先走向桥南侧入口
    if r == "south":
        return bx, RIVER_Y0
    # 处于北半场，先走向桥北侧入口
    return bx, RIVER_Y1 - 1e-3
