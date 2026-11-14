"""
Arena geometry in tile units (18 x 32). Side 0 = bottom, side 1 = top (mirror y -> 32 - y).

Rows:  y in [0, 15)  side-0 half   ("south")
       y in [15, 17) river          (crossable only on bridges, or by jumpers / air)
       y in [17, 32) side-1 half   ("north")
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, Optional, Tuple

TICK_MS = 50
W, H = 18.0, 32.0
RIVER_Y0, RIVER_Y1 = 15.0, 17.0
BRIDGE_X = (3.5, 14.5)
BRIDGE_HALF_W = 1.0
POCKET_DEPTH = 6.0  # tiles past the river you may deploy into once that lane's princess tower falls

# role -> (x, y, radius) for side 0; side 1 is mirrored.
TOWER_LAYOUT: Dict[str, Tuple[float, float, float]] = {
    "princess_left": (3.5, 6.5, 1.5),
    "princess_right": (14.5, 6.5, 1.5),
    "king": (9.0, 3.0, 2.0),
}


def mirror_y(y: float) -> float:
    return H - y


def to_side(side: int, x: float, y: float) -> Tuple[float, float]:
    """Convert side-0-perspective coordinates into absolute arena coordinates for `side`."""
    return (x, y) if side == 0 else (x, mirror_y(y))


def tower_pos(side: int, role: str) -> Tuple[float, float, float]:
    x, y, r = TOWER_LAYOUT[role]
    return (x, y if side == 0 else mirror_y(y), r)


def lane_of(x: float) -> str:
    return "left" if x < W / 2 else "right"


def region(y: float) -> str:
    if y < RIVER_Y0:
        return "south"
    if y < RIVER_Y1:
        return "river"
    return "north"


def home_region(side: int) -> str:
    return "south" if side == 0 else "north"


def forward(side: int) -> int:
    """+1 if this side attacks toward larger y."""
    return 1 if side == 0 else -1


def snap(x: float, y: float) -> Tuple[float, float]:
    """Snap to the centre of the tile containing (x, y)."""
    tx = min(max(math.floor(x), 0), int(W) - 1) + 0.5
    ty = min(max(math.floor(y), 0), int(H) - 1) + 0.5
    return tx, ty


def in_bounds(x: float, y: float) -> bool:
    return 0.0 <= x < W and 0.0 <= y < H


def dist(ax: float, ay: float, bx: float, by: float) -> float:
    return math.hypot(ax - bx, ay - by)


def on_bridge(x: float) -> bool:
    return any(abs(x - b) <= BRIDGE_HALF_W for b in BRIDGE_X)


def deploy_check(
    side: int,
    card_type: str,
    x: float,
    y: float,
    enemy_princess_alive: Dict[str, bool],
    blockers: Iterable[Tuple[float, float, float]] = (),
) -> Optional[str]:
    """None if legal, else a reason string. (x, y) must already be snapped.
    blockers: (x, y, radius) footprints of standing towers/buildings."""
    if not in_bounds(x, y):
        return "out_of_bounds"
    if card_type == "spell":
        return None
    if region(y) == "river":
        return "river"
    for bx, by, br in blockers:
        if abs(x - bx) < br and abs(y - by) < br:
            return "blocked_by_structure"
    if region(y) == home_region(side):
        return None
    # pocket: enemy half, same lane as a fallen enemy princess tower, close to the river
    depth = (y - RIVER_Y1) if side == 0 else (RIVER_Y0 - y)
    if 0 <= depth < POCKET_DEPTH and not enemy_princess_alive[lane_of(x)]:
        return None
    return "enemy_half"


def ground_waypoint(x: float, y: float, tx: float, ty: float) -> Tuple[float, float]:
    """Next point a ground (non-jumping) unit should walk to on its way to (tx, ty)."""
    r, tr = region(y), region(ty)
    if r == tr or tr == "river":
        return tx, ty
    bx = min(BRIDGE_X, key=lambda b: abs(x - b) + abs(tx - b))
    if r == "river":
        if abs(x - bx) > BRIDGE_HALF_W:
            return bx, y
        return (bx, RIVER_Y1) if tr == "north" else (bx, RIVER_Y0 - 1e-3)
    if r == "south":
        return bx, RIVER_Y0
    return bx, RIVER_Y1 - 1e-3
