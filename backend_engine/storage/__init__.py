"""Local, on-disk app data: saved matches and the user's profile (never leaves this Mac)."""
from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def default_data_dir() -> Path:
    """Folder for personal app data. Override with the CRBOT_DATA_DIR environment variable."""
    env = os.environ.get("CRBOT_DATA_DIR")
    return Path(env) if env else REPO_ROOT / "user_data"
