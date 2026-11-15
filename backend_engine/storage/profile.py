"""Local player profile and app settings (`<data_dir>/profile.json`).

Everything here stays on this Mac. The optional player tag is never sent anywhere by CRbot.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Optional

from backend_engine.storage import default_data_dir

LEVELS = (11, 16)


@dataclass
class Profile:
    display_name: str = ""
    player_tag: str = ""            # optional, local only
    default_level: int = 11
    default_persona: str = "surgeon"
    default_speed: int = 1
    autosave_matches: bool = True
    auto_review: bool = True        # analyze automatically when a match ends
    detector_model: str = ""        # Mode B YOLO ONNX path ("" = auto-detect in pretrained_weights/)
    calibration_file: str = ""      # Mode B screen calibration ("" = none yet)

    @staticmethod
    def clean_tag(tag: str) -> str:
        """Upper-case, strip spaces and a leading '#'. Returns '' for invalid tags."""
        t = tag.strip().upper().lstrip("#")
        ok = t and len(t) <= 14 and all(c in "0289PYLQGRJCUV" for c in t)
        return t if ok else ""

    def validate(self) -> None:
        if self.default_level not in LEVELS:
            self.default_level = 11
        if self.default_speed not in (1, 2, 4, 8):
            self.default_speed = 1
        self.player_tag = self.clean_tag(self.player_tag) if self.player_tag else ""


def profile_path(data_dir: Optional[Path] = None) -> Path:
    return (data_dir or default_data_dir()) / "profile.json"


def load_profile(data_dir: Optional[Path] = None) -> Profile:
    p = profile_path(data_dir)
    try:
        raw = json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return Profile()
    known = {f.name for f in fields(Profile)}
    try:
        prof = Profile(**{k: v for k, v in raw.items() if k in known})
    except TypeError:
        return Profile()
    prof.validate()
    return prof


def save_profile(prof: Profile, data_dir: Optional[Path] = None) -> Path:
    prof.validate()
    p = profile_path(data_dir)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(asdict(prof), indent=1), encoding="utf-8")
    os.replace(tmp, p)
    return p
