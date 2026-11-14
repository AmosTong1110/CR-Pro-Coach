"""
Maps pixels of the (mirrored) phone screen to arena tiles.

The arena is 18 tiles wide and 32 tall. You are always at the bottom in the real game,
which matches the simulator's side 0. Calibration = two clicks on a screenshot:
the top-left corner of the arena (tile x=0, y=32) and the bottom-right corner (x=18, y=0).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Optional, Tuple

DEFAULT_PATH = Path(__file__).resolve().parents[3] / "configs" / "mode_b_calibration.json"
W_TILES, H_TILES = 18.0, 32.0

# Rough arena box as fractions of a portrait phone frame (~9:19.5). Calibrate for real use.
DEFAULT_FRACTIONS = (0.0, 0.115, 1.0, 0.79)   # left, top, right, bottom


@dataclass
class ArenaCalibration:
    frame_w: int
    frame_h: int
    left: float
    top: float
    right: float
    bottom: float
    screen_region: Optional[Tuple[int, int, int, int]] = None   # x, y, w, h of the mirror window
    calibrated: bool = False

    @classmethod
    def default_for(cls, frame_w: int, frame_h: int) -> "ArenaCalibration":
        l, t, r, b = DEFAULT_FRACTIONS
        return cls(frame_w, frame_h, l * frame_w, t * frame_h, r * frame_w, b * frame_h)

    @classmethod
    def from_corners(cls, frame_w: int, frame_h: int, top_left: Tuple[float, float],
                     bottom_right: Tuple[float, float], screen_region=None) -> "ArenaCalibration":
        (l, t), (r, b) = top_left, bottom_right
        if r - l < 10 or b - t < 10:
            raise ValueError("calibration corners are too close together")
        return cls(frame_w, frame_h, l, t, r, b, tuple(screen_region) if screen_region else None, True)

    # -- mapping -----------------------------------------------------------
    def _scaled(self, frame_w: int, frame_h: int) -> Tuple[float, float, float, float]:
        sx, sy = frame_w / self.frame_w, frame_h / self.frame_h
        return self.left * sx, self.top * sy, self.right * sx, self.bottom * sy

    def px_to_tile(self, px: float, py: float, frame_w: Optional[int] = None,
                   frame_h: Optional[int] = None) -> Tuple[float, float]:
        l, t, r, b = self._scaled(frame_w or self.frame_w, frame_h or self.frame_h)
        return (px - l) / (r - l) * W_TILES, (b - py) / (b - t) * H_TILES

    def tile_to_px(self, x: float, y: float, frame_w: Optional[int] = None,
                   frame_h: Optional[int] = None) -> Tuple[float, float]:
        l, t, r, b = self._scaled(frame_w or self.frame_w, frame_h or self.frame_h)
        return l + x / W_TILES * (r - l), b - y / H_TILES * (b - t)

    @staticmethod
    def in_arena(x: float, y: float) -> bool:
        return 0.0 <= x < W_TILES and 0.0 <= y < H_TILES

    # -- persistence -------------------------------------------------------
    def save(self, path: Path | str = DEFAULT_PATH) -> None:
        Path(path).write_text(json.dumps(asdict(self), indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str = DEFAULT_PATH) -> Optional["ArenaCalibration"]:
        p = Path(path)
        if not p.exists():
            return None
        d = json.loads(p.read_text(encoding="utf-8"))
        if d.get("screen_region"):
            d["screen_region"] = tuple(d["screen_region"])
        return cls(**d)
