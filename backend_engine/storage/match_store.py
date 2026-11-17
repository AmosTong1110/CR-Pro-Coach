"""Match store: every match is saved under `<data_dir>/matches/<id>/` and can be reopened without re-analysis.

Layout per match:
    meta.json     summary for the Analysis list (decks, archetypes, result, accuracy, ...)
    actions.json  everything needed to rebuild the match deterministically (config, seed, every play,
                  final tick and state hash for verification)
    review.json   the coach review (win-rate curve and graded moves), if the match was analyzed

Match ids look like `YYYYMMDD-HHMMSS_<mode>_<hash8>` with mode in MODES. The store is Qt-free.
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
from dataclasses import asdict, dataclass, field, fields
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from backend_engine.brain.coach_analyzer import Candidate, MatchReview, MoveReview
from backend_engine.storage import default_data_dir

MODES = ("practice", "drill")
FORMAT_VERSION = 1


class MatchStoreError(Exception):
    pass


def new_match_id(mode: str, salt: Any = "", now: Optional[datetime] = None) -> str:
    """`YYYYMMDD-HHMMSS_<mode>_<hash8>`; the hash keeps ids unique when two matches start in the same second."""
    if mode not in MODES:
        raise ValueError(f"unknown mode {mode!r}")
    now = now or datetime.now()
    h = hashlib.sha256(f"{now.isoformat()}|{mode}|{salt}|{os.urandom(8).hex()}".encode()).hexdigest()[:8]
    return f"{now:%Y%m%d-%H%M%S}_{mode}_{h}"


@dataclass(frozen=True)
class SavedAction:
    """One play on the final timeline of a match (both sides)."""
    t_ms: int
    side: int
    card_id: str
    x: float
    y: float
    evolved: bool = False
    event_id: int = 0
    variant: str = "base"
    ability: bool = False                 # a champion ability use (card_id = the champion), not a card play


@dataclass
class MatchMeta:
    """Summary row shown in the Analysis list."""
    match_id: str
    mode: str
    created: str                          # local ISO time the match started
    duration_ms: int
    outcome: Optional[str]                # "win" | "loss" | "draw" | None (unfinished)
    crowns: Tuple[int, int]               # (you, opponent)
    reason: Optional[str]                 # engine result reason, if finished
    player_deck: Tuple[str, ...]
    opp_deck: Tuple[str, ...]
    player_evo: Tuple[str, ...] = ()
    opp_evo: Tuple[str, ...] = ()
    level: int = 11
    persona: Optional[str] = None
    player_archetype: str = ""
    opp_archetype: str = ""
    n_plays: int = 0                      # your plays on the final timeline
    accuracy: Optional[float] = None      # None until analyzed
    counts: Dict[str, int] = field(default_factory=dict)
    data_version: str = ""                # card data the match was simulated with
    engine_version: int = 1               # engine.ENGINE_VERSION at save time (files without it: 1)
    format: int = FORMAT_VERSION

    @property
    def analyzed(self) -> bool:
        return self.accuracy is not None

    @property
    def title(self) -> str:
        a, b = self.player_archetype or "Your deck", self.opp_archetype or "Opponent"
        return f"{a} vs {b}"

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "MatchMeta":
        known = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        for k in ("crowns", "player_deck", "opp_deck", "player_evo", "opp_evo"):
            if k in d and d[k] is not None:
                d[k] = tuple(d[k])
        return cls(**d)


@dataclass
class MatchRecord:
    """Contents of actions.json: enough to rebuild the match tick-for-tick."""
    config: Dict[str, Any]                # SessionConfig fields (decks, evo, level, persona, seed)
    actions: List[SavedAction]
    final_t_ms: int
    final_hash: str

    def human_actions(self, side: int = 0) -> List[SavedAction]:
        return [a for a in self.actions if a.side == side]

    def to_dict(self) -> dict:
        return {"format": FORMAT_VERSION, "config": self.config, "final_t_ms": self.final_t_ms,
                "final_hash": self.final_hash, "actions": [asdict(a) for a in self.actions]}

    @classmethod
    def from_dict(cls, d: dict) -> "MatchRecord":
        return cls(dict(d["config"]), [SavedAction(**a) for a in d["actions"]],
                   int(d["final_t_ms"]), str(d["final_hash"]))


# -- review (de)serialisation ------------------------------------------------
def _cand(c: Candidate) -> dict:
    return {"card_id": c.card_id, "x": c.x, "y": c.y}


def _uncand(d: dict) -> Candidate:
    return Candidate(d.get("card_id"), float(d.get("x", 0.0)), float(d.get("y", 0.0)))


def review_to_dict(r: MatchReview) -> dict:
    moves = []
    for m in r.moves:
        moves.append({
            "event_id": m.event_id, "t_ms": m.t_ms, "actual": _cand(m.actual), "best": _cand(m.best),
            "v_before": m.v_before, "v_actual": m.v_actual, "v_best": m.v_best, "v_wait": m.v_wait,
            "loss": m.loss, "label": m.label, "n_candidates": m.n_candidates,
            "alternatives": [[_cand(c), v] for c, v in m.alternatives],
            "diagnostic": getattr(m, "diagnostic", None),
        })
    return {"format": FORMAT_VERSION, "curve": [[t, v] for t, v in r.curve], "moves": moves,
            "final": r.final, "model_meta": r.model_meta}


def review_from_dict(d: dict) -> MatchReview:
    moves = [MoveReview(
        event_id=int(m["event_id"]), t_ms=int(m["t_ms"]), actual=_uncand(m["actual"]),
        v_before=m["v_before"], v_actual=m["v_actual"], v_best=m["v_best"], v_wait=m["v_wait"],
        best=_uncand(m["best"]), loss=m["loss"], label=m["label"], n_candidates=int(m["n_candidates"]),
        alternatives=tuple((_uncand(c), v) for c, v in m.get("alternatives", ())),
        diagnostic=m.get("diagnostic"),
    ) for m in d["moves"]]
    return MatchReview([(int(t), float(v)) for t, v in d["curve"]], moves, d.get("final"),
                       dict(d.get("model_meta", {})))


# -- the store -----------------------------------------------------------------
def _write_json(path: Path, obj: Any) -> None:
    """Atomic write: a crash never leaves a half-written file."""
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=1, ensure_ascii=False)
    os.replace(tmp, path)


def _read_json(path: Path) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


class MatchStore:
    def __init__(self, root: Optional[Path | str] = None):
        self.root = Path(root) if root is not None else default_data_dir() / "matches"

    def _dir(self, match_id: str) -> Path:
        if not match_id or "/" in match_id or "\\" in match_id or match_id.startswith("."):
            raise MatchStoreError(f"bad match id {match_id!r}")
        return self.root / match_id

    def exists(self, match_id: str) -> bool:
        return (self._dir(match_id) / "meta.json").exists()

    def save(self, meta: MatchMeta, record: MatchRecord, review: Optional[MatchReview] = None) -> Path:
        """Create or overwrite a match. A stale review.json is removed when no review is given."""
        d = self._dir(meta.match_id)
        d.mkdir(parents=True, exist_ok=True)
        if review is not None:
            _apply_review(meta, review)
            _write_json(d / "review.json", review_to_dict(review))
        elif (d / "review.json").exists():
            (d / "review.json").unlink()
        _write_json(d / "actions.json", record.to_dict())
        _write_json(d / "meta.json", meta.to_dict())   # written last: its presence marks a complete save
        return d

    def save_review(self, match_id: str, review: MatchReview) -> MatchMeta:
        """Attach (or replace) the review of an existing match and update its summary."""
        meta = self.load_meta(match_id)
        _apply_review(meta, review)
        d = self._dir(match_id)
        _write_json(d / "review.json", review_to_dict(review))
        _write_json(d / "meta.json", meta.to_dict())
        return meta

    def load_meta(self, match_id: str) -> MatchMeta:
        try:
            return MatchMeta.from_dict(_read_json(self._dir(match_id) / "meta.json"))
        except (OSError, ValueError, TypeError, KeyError) as e:
            raise MatchStoreError(f"cannot read match {match_id}: {e}") from e

    def load_record(self, match_id: str) -> MatchRecord:
        try:
            return MatchRecord.from_dict(_read_json(self._dir(match_id) / "actions.json"))
        except (OSError, ValueError, TypeError, KeyError) as e:
            raise MatchStoreError(f"cannot read actions of {match_id}: {e}") from e

    def load_review(self, match_id: str) -> Optional[MatchReview]:
        p = self._dir(match_id) / "review.json"
        if not p.exists():
            return None
        try:
            return review_from_dict(_read_json(p))
        except (OSError, ValueError, TypeError, KeyError) as e:
            raise MatchStoreError(f"cannot read review of {match_id}: {e}") from e

    def list(self, mode: Optional[str] = None) -> List[MatchMeta]:
        """All complete matches, newest first. Unreadable folders are skipped."""
        if not self.root.exists():
            return []
        out = []
        for d in self.root.iterdir():
            if not (d / "meta.json").is_file():
                continue
            try:
                m = self.load_meta(d.name)
            except MatchStoreError:
                continue
            if mode is None or m.mode == mode:
                out.append(m)
        out.sort(key=lambda m: (m.created, m.match_id), reverse=True)
        return out

    def delete(self, match_id: str) -> None:
        d = self._dir(match_id)
        if d.exists():
            shutil.rmtree(d)


def _apply_review(meta: MatchMeta, review: MatchReview) -> None:
    meta.accuracy = round(review.accuracy, 1) if review.moves else None
    meta.counts = review.counts()


def outcome_of(winner: Optional[int], finished: bool, side: int = 0) -> Optional[str]:
    if not finished:
        return None
    if winner is None:
        return "draw"
    return "win" if winner == side else "loss"


def actions_from_events(events: Sequence[Any]) -> List[SavedAction]:
    """PublicDeployEvent list -> SavedAction list."""
    return [SavedAction(e.t_ms, e.side, e.card_id, float(e.tile.x), float(e.tile.y), bool(e.is_evolved), e.event_id,
                        getattr(e, "variant", "base"), bool(getattr(e, "is_ability", False))) for e in events]
