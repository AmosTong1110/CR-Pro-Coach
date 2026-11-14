"""
Frame sources (read-only). Frames are numpy uint8 arrays, H x W x 3, RGB.

QtScreenSource   grabs a rectangle of the Mac screen (e.g. the QuickTime / scrcpy mirror).
                 macOS asks once for Screen Recording permission for your terminal / Python.
FolderSource     replays PNG/JPG frames from a folder (recorded matches, tests). File names
                 must sort in time order; timestamps come from `fps`.
"""
from __future__ import annotations

import time
from pathlib import Path
from typing import List, Optional, Protocol, Tuple

import numpy as np

Frame = np.ndarray


class FrameSource(Protocol):
    def grab(self) -> Optional[Tuple[Frame, float]]:
        """(frame, timestamp in seconds) or None when no frame is available / finished."""
        ...


class QtScreenSource:
    def __init__(self, region: Tuple[int, int, int, int]):
        self.region = tuple(int(v) for v in region)

    def grab(self) -> Optional[Tuple[Frame, float]]:
        from PySide6.QtGui import QGuiApplication, QImage

        screen = QGuiApplication.primaryScreen()
        if screen is None:
            return None
        x, y, w, h = self.region
        pix = screen.grabWindow(0, x, y, w, h)
        if pix.isNull():
            return None
        img = pix.toImage().convertToFormat(QImage.Format_RGB888)
        arr = np.frombuffer(img.constBits(), dtype=np.uint8, count=img.sizeInBytes())
        arr = arr.reshape(img.height(), img.bytesPerLine())[:, : img.width() * 3]
        return arr.reshape(img.height(), img.width(), 3).copy(), time.monotonic()


class FolderSource:
    EXTS = {".png", ".jpg", ".jpeg"}

    def __init__(self, folder: Path | str, fps: float = 10.0):
        self.files: List[Path] = sorted(p for p in Path(folder).iterdir() if p.suffix.lower() in self.EXTS)
        self.fps = fps
        self.i = 0

    def grab(self) -> Optional[Tuple[Frame, float]]:
        if self.i >= len(self.files):
            return None
        from PIL import Image

        f = self.files[self.i]
        t = self.i / self.fps
        self.i += 1
        with Image.open(f) as im:
            return np.asarray(im.convert("RGB")), t
