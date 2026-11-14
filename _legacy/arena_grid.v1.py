"""
Arena Grid & Collision System for Clash Royale 18x32 Board.
Quantized into 36 columns x 64 rows (2,304 half-tile cells).
Handles non-rectangular field boundaries, river bridges, and valid deployment rules.
"""
import math
from typing import Tuple, List

TILES_X = 18.0
TILES_Y = 32.0
GRID_X = 36
GRID_Y = 64
N_CELLS = GRID_X * GRID_Y  # 2,304

# River spans y = 15.5 to 16.5 tiles (half-cell indices ~31 to 32)
RIVER_TILES_Y0 = 15.0
RIVER_TILES_Y1 = 17.0

# Two bridges on the left and right lanes
BRIDGE_LEFT_TILES_X = (3.0, 4.5)
BRIDGE_RIGHT_TILES_X = (13.5, 15.0)

# Unplaceable decorative ledges near river banks (outermost columns)
LEDGE_Y0_NORM = 14.0 / 32.0
LEDGE_Y1_NORM = 18.0 / 32.0

# King Tower platforms (cannot place directly on king platform unless activated)
KING_STRIP_X0_NORM = 1.0 / 3.0
KING_STRIP_X1_NORM = 2.0 / 3.0
KING_Y0_NORM = 1.0 / 32.0
KING_Y1_NORM = 4.0 / 32.0


class ArenaGrid:
    """Provides coordinate transforms and tile legality checks."""

    @staticmethod
    def cell_to_norm_xy(cell_id: int) -> Tuple[float, float]:
        """Convert cell index [0, 2303] to normalized board coordinates [0, 1]x[0, 1]."""
        gx = cell_id % GRID_X
        gy = cell_id // GRID_X
        nx = (gx + 0.5) / float(GRID_X)
        ny = (gy + 0.5) / float(GRID_Y)
        return nx, ny

    @staticmethod
    def norm_xy_to_cell(nx: float, ny: float) -> int:
        """Convert normalized board coordinates [0, 1] to closest cell index [0, 2303]."""
        gx = max(0, min(GRID_X - 1, int(nx * GRID_X)))
        gy = max(0, min(GRID_Y - 1, int(ny * GRID_Y)))
        return gy * GRID_X + gx

    @staticmethod
    def norm_to_tiles(nx: float, ny: float) -> Tuple[float, float]:
        """Normalized [0, 1] to tile units [0, 18] x [0, 32]."""
        return nx * TILES_X, ny * TILES_Y

    @staticmethod
    def is_ledge_blocked(nx: float, ny: float) -> bool:
        """True if coordinate is on the decorative riverbank ledges (outer columns)."""
        is_outer_x = (nx < 1.0 / TILES_X) or (nx > (TILES_X - 1.0) / TILES_X)
        if not is_outer_x:
            return False
        return LEDGE_Y0_NORM <= ny <= LEDGE_Y1_NORM

    @staticmethod
    def is_valid_placement(cell_id: int, my_side: int = 0, enemy_towers_standing: int = 2) -> bool:
        """
        Check if a player can legally drop a troop/building at cell_id.
        Standard rules:
        - Only on player's half of the arena (ny >= 0.5 for side 0, bottom)
        - Unless enemy princess tower is down (pocket opens)
        - Cannot place on river, king platform, or border ledges
        """
        nx, ny = ArenaGrid.cell_to_norm_xy(cell_id)

        # Ledge check
        if ArenaGrid.is_ledge_blocked(nx, ny):
            return False

        # King platform check (back 3 rows, center 6 tiles)
        in_center_strip = KING_STRIP_X0_NORM < nx < KING_STRIP_X1_NORM
        if in_center_strip:
            if KING_Y0_NORM <= ny <= KING_Y1_NORM:  # Enemy king platform
                return False
            if (1.0 - KING_Y1_NORM) <= ny <= (1.0 - KING_Y0_NORM):  # Own king platform
                return False

        # Own half rule (side 0 is bottom: y from 0.5 to 1.0)
        if my_side == 0:
            if ny < 0.5 and enemy_towers_standing == 2:
                return False
        else:
            if ny > 0.5 and enemy_towers_standing == 2:
                return False

        return True

    @staticmethod
    def legal_placement_mask(my_side: int = 0, enemy_towers_standing: int = 2) -> List[bool]:
        """Returns boolean mask of length 2,304 for PPO Action Masking."""
        return [
            ArenaGrid.is_valid_placement(cid, my_side, enemy_towers_standing)
            for cid in range(N_CELLS)
        ]
