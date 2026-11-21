"""Pre-rendered sprite art (CC0 3D models rendered by tools/art/build_sprites.py) for the battle board.

`SpriteBank` lazily loads the sheets listed in assets/sprites/manifest.json and hands out frames scaled to the
board's tile size (cached, so a frame is scaled once per size). `pose()` picks a unit's animation from engine
state only (moving -> walk, locked on -> attack synced to its hit cooldown, else idle); it never changes the match.
When the assets are missing the board falls back to CRbot's vector tokens.
"""
from __future__ import annotations

import json
import os
from collections import OrderedDict
from pathlib import Path
from typing import Dict, Optional, Tuple

from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPixmap

ASSETS = Path(__file__).resolve().parent / "assets" / "sprites"
IMPACT_AT = 0.45          # fraction of the attack animation where the blow lands (KayKit chop / shoot)


def pose(u, step: Tuple[float, float], target_xy: Optional[Tuple[float, float]], t_ms: int,
         counts: Dict[str, int]) -> Tuple[str, str, int]:
    """(facing, action, frame index) for a unit. `step` = its movement over the last simulation step (tiles)."""
    dx, dy = step
    moving = abs(dx) + abs(dy) > 0.004
    if moving:
        facing = "up" if dy > 0 else "down"
    elif target_xy is not None:
        facing = "up" if target_xy[1] > u.y else "down"
    else:
        facing = "up" if u.side == 0 else "down"
    if u.deploy_ms > 0:
        return facing, "idle", 0
    if moving:
        n = counts.get("walk", 1)
        return facing, "walk", (t_ms // 90 + u.uid) % n
    if u.target_uid is not None and target_xy is not None and u.hit_speed_ms > 0:
        n = counts.get("attack", 1)
        phase = 1.0 - max(0.0, min(1.0, u.cooldown_ms / u.hit_speed_ms))   # 0 right after a hit
        return facing, "attack", int((phase + IMPACT_AT) * n) % n
    n = counts.get("idle", 1)
    return facing, "idle", (t_ms // 200 + u.uid) % n


class SpriteBank:
    _shared: Optional["SpriteBank"] = None

    def __init__(self, root: Path = ASSETS, cache_limit: int = 900):
        self.root = Path(root)
        self.manifest: dict = {}
        path = self.root / "manifest.json"
        if path.exists():
            try:
                self.manifest = json.loads(path.read_text())
            except (OSError, ValueError):
                self.manifest = {}
        self.ok = bool(self.manifest.get("arena")) and bool(self.manifest.get("towers"))
        self._sheets: Dict[str, QImage] = {}
        self._cache: "OrderedDict[tuple, QPixmap]" = OrderedDict()
        self._limit = cache_limit

    @classmethod
    def shared(cls) -> "SpriteBank":
        if cls._shared is None:
            cls._shared = SpriteBank()
        return cls._shared

    # -- metadata ---------------------------------------------------------------------------------------------
    @property
    def px_per_tile(self) -> float:
        return float(self.manifest.get("px_per_tile", 64))

    @property
    def tilt(self) -> float:
        return float(self.manifest.get("arena", {}).get("tilt", self.manifest.get("tilt", 35.0)))

    @property
    def margin(self) -> Tuple[float, float]:
        mx, my = self.manifest.get("arena", {}).get("margin", [0.0, 0.0])
        return float(mx), float(my)

    def character(self, card_id: str) -> Optional[str]:
        name = self.manifest.get("cards", {}).get(card_id)
        return name if name in self.manifest.get("characters", {}) else None

    def card_scale(self, card_id: str) -> float:
        return float(self.manifest.get("card_scale", {}).get(card_id, 1.0))

    def counts(self, name: str) -> Dict[str, int]:
        acts = self.manifest["characters"][name]["actions"]
        return {k.split("/", 1)[1]: len(v) for k, v in acts.items() if k.startswith("up/")}

    def feet(self, name: str) -> Tuple[float, float]:
        fx, fy = self.manifest["characters"][name]["feet"]
        return float(fx), float(fy)

    # -- pixels -----------------------------------------------------------------------------------------------
    def _image(self, fn: str) -> QImage:
        img = self._sheets.get(fn)
        if img is None:
            img = self._sheets[fn] = QImage(str(self.root / fn))
        return img

    def _remember(self, key, pm: QPixmap) -> QPixmap:
        self._cache[key] = pm
        if len(self._cache) > self._limit:
            self._cache.popitem(last=False)
        return pm

    def _scaled(self, key, img: QImage, scale: float, dpr: float) -> QPixmap:
        pm = self._cache.get(key)
        if pm is not None:
            self._cache.move_to_end(key)
            return pm
        w, h = max(1, round(img.width() * scale * dpr)), max(1, round(img.height() * scale * dpr))
        pm = QPixmap.fromImage(img.scaled(w, h, Qt.IgnoreAspectRatio, Qt.SmoothTransformation))
        pm.setDevicePixelRatio(dpr)
        return self._remember(key, pm)

    def frame(self, name: str, side: str, facing: str, action: str, index: int, ts: float,
              dpr: float, extra: float = 1.0) -> Optional[QPixmap]:
        """One character frame scaled so a tile is `ts` logical px (times `extra` for small units)."""
        ch = self.manifest["characters"].get(name)
        cells = ch["actions"].get(f"{facing}/{action}") if ch else None
        if not cells:
            return None
        cell = cells[index % len(cells)]
        scale = ts / self.px_per_tile * extra
        key = ("f", name, side, cell, round(scale * dpr, 3))
        pm = self._cache.get(key)
        if pm is not None:
            self._cache.move_to_end(key)
            return pm
        sheet = self._image(ch["sheets"][side])
        c, cols = ch["cell"], ch["cols"]
        img = sheet.copy((cell % cols) * c, (cell // cols) * c, c, c)
        return self._scaled(key, img, scale, dpr)

    def tower(self, kind: str, side: str, ts: float, dpr: float) -> Tuple[Optional[QPixmap], Tuple[float, float]]:
        meta = self.manifest.get("towers", {}).get(f"{kind}_{side}")
        if not meta:
            return None, (0.5, 0.5)
        scale = ts / self.px_per_tile
        pm = self._scaled(("t", kind, side, round(scale * dpr, 3)), self._image(meta["file"]), scale, dpr)
        gx, gy = meta["ground_center"]
        return pm, (float(gx), float(gy))

    def arena(self, width: float, height: float, dpr: float) -> Optional[QPixmap]:
        meta = self.manifest.get("arena")
        if not meta:
            return None
        img = self._image(meta["file"])
        key = ("a", round(width * dpr), round(height * dpr))
        pm = self._cache.get(key)
        if pm is None:
            pm = QPixmap.fromImage(img.scaled(round(width * dpr), round(height * dpr), Qt.IgnoreAspectRatio,
                                              Qt.SmoothTransformation))
            pm.setDevicePixelRatio(dpr)
            self._remember(key, pm)
        return pm


def sprites_wanted() -> bool:
    """Sprites unless CRBOT_ART=vector (or the assets are missing)."""
    return os.environ.get("CRBOT_ART", "sprites").lower() != "vector"
