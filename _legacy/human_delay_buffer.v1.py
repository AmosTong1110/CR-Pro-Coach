"""
Human Reaction Latency Buffer.
Emulates realistic human reflex delay after perceiving critical threats (Win-Conditions).
Prevents 0ms superhuman robotic reflexes by inserting calibrated stochastic delays.
"""
import time
import random
from pathlib import Path
from typing import Optional, Tuple

CONFIGS_DIR = Path(__file__).resolve().parents[2] / "configs"
PROFILE_PATH = CONFIGS_DIR / "human_reaction_profile.yaml"


class HumanDelayBuffer:
    def __init__(self):
        # Calibrated default human reaction distributions (ms)
        self.profile = {
            "reaction_latency": {
                "sudden_win_con_ms": {"mean": 450, "std": 85, "min": 280, "max": 850},
                "hover_premonition_ms": {"mean": 160, "std": 40, "min": 100, "max": 260},
                "standard_defense_ms": {"mean": 520, "std": 90, "min": 350, "max": 950}
            }
        }
        self.pending_actions = []  # list of (execution_timestamp, action_payload)
        self._load_profile()

    def _load_profile(self):
        if not PROFILE_PATH.exists():
            return
        try:
            import yaml
            with open(PROFILE_PATH, "r", encoding="utf-8") as f:
                loaded = yaml.safe_load(f)
                if loaded:
                    self.profile = loaded
        except ImportError:
            # Safe fallback if pyyaml is not yet installed in active python env
            pass

    def sample_delay_seconds(self, is_sudden_win_con: bool = False, is_premonition: bool = False) -> float:
        """Draws human delay from calibrated latency distributions."""
        lat = self.profile.get("reaction_latency", {})
        if is_premonition:
            cfg = lat.get("hover_premonition_ms", {"mean": 160, "std": 40})
        elif is_sudden_win_con:
            cfg = lat.get("sudden_win_con_ms", {"mean": 450, "std": 85})
        else:
            cfg = lat.get("standard_defense_ms", {"mean": 520, "std": 90})

        ms = random.gauss(cfg["mean"], cfg["std"])
        ms = max(cfg.get("min", 100), min(cfg.get("max", 1200), ms))
        return ms / 1000.0

    def queue_action(self, action: dict, is_sudden_threat: bool = False):
        delay = self.sample_delay_seconds(is_sudden_win_con=is_sudden_threat)
        execute_at = time.time() + delay
        self.pending_actions.append((execute_at, action))

    def poll_ready_action(self) -> Optional[dict]:
        """Returns the oldest action that has satisfied its human reaction latency delay."""
        now = time.time()
        for i, (execute_at, action) in enumerate(self.pending_actions):
            if now >= execute_at:
                return self.pending_actions.pop(i)[1]
        return None
