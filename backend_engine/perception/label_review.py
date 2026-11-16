"""
Review pseudo labels from tools/eval_mode_b_recordings.py into verified ground truth (Qt-free core).

Folder layout (written by the eval tool's --export):
    images/<stem>.jpg          568x896 arena crops, stem = <recording slug>_t<ms, 7 digits>
    labels/<stem>.txt          pseudo labels, YOLO format (class cx cy w h, normalised)
    review.jsonl               detections the eval dropped (card in neither deck): shown as suggestions
    eval_report.json           per recording: decks, missed cards, ...
This module adds:
    reviewed/<stem>.txt        the human-checked labels (the pseudo label file is never changed)
    review_state.json          {stem: "done" | "skipped"}

`export()` writes a YOLO training set from the reviewed frames only, with one whole match held out for
validation (frames of the same match are near-duplicates, so a random split would overstate accuracy).
"""
from __future__ import annotations

import json
import re
import shutil
from collections import defaultdict
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Set

CROP_W, CROP_H = 568, 896
TOWERS = {"king-tower", "queen-tower", "cannoneer-tower", "dagger-duchess-tower", "royal-chef-tower"}
# model classes that are spells / spell effects: the detector misses these most on real matches
SPELLS = {"arrows", "zap", "fireball", "freeze", "poison", "lightning", "earthquake", "graveyard", "tornado",
          "the-log", "rocket", "rage", "clone", "goblin-curse", "giant-snowball", "barbarian-barrel",
          "royal-delivery", "vines", "void", "goblin-barrel"}
_STEM = re.compile(r"^(?P<match>.+)_t(?P<ms>\d{7})$")


def slug(recording_name: str) -> str:
    """Same slug the eval tool gives the recording's frames."""
    return re.sub(r"[^0-9A-Za-z]+", "_", Path(recording_name).stem).strip("_")


@dataclass(frozen=True)
class Box:
    cls: int
    x0: float
    y0: float
    x1: float
    y1: float
    conf: Optional[float] = None

    def clamped(self) -> "Box":
        x0, x1 = sorted((min(max(self.x0, 0.0), CROP_W), min(max(self.x1, 0.0), CROP_W)))
        y0, y1 = sorted((min(max(self.y0, 0.0), CROP_H), min(max(self.y1, 0.0), CROP_H)))
        return replace(self, x0=x0, y0=y0, x1=x1, y1=y1)

    @property
    def valid(self) -> bool:
        return self.x1 - self.x0 >= 2 and self.y1 - self.y0 >= 2

    def yolo(self) -> str:
        b = self.clamped()
        return (f"{b.cls} {(b.x0 + b.x1) / 2 / CROP_W:.6f} {(b.y0 + b.y1) / 2 / CROP_H:.6f} "
                f"{(b.x1 - b.x0) / CROP_W:.6f} {(b.y1 - b.y0) / CROP_H:.6f}")

    @classmethod
    def from_yolo(cls, line: str) -> "Box":
        c, cx, cy, w, h = line.split()[:5]
        cx, cy, w, h = float(cx) * CROP_W, float(cy) * CROP_H, float(w) * CROP_W, float(h) * CROP_H
        return cls(int(c), cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2)


def read_boxes(path: Path) -> List[Box]:
    if not path.exists():
        return []
    return [Box.from_yolo(l) for l in path.read_text(encoding="utf-8").splitlines() if l.strip()]


def write_boxes(path: Path, boxes: Iterable[Box]) -> None:
    lines = [b.yolo() for b in boxes if b.clamped().valid]
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    tmp.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    tmp.replace(path)


class LabelSet:
    def __init__(self, root: Path | str, names: Sequence[str], class_cards: Dict[int, Optional[str]],
                 spawned_by: Optional[Dict[str, List[str]]] = None):
        self.root = Path(root)
        self.names = list(names)
        self.class_cards = class_cards
        self.spawned_by = spawned_by or {}
        self.stems = sorted(p.stem for p in (self.root / "images").glob("*.jpg"))
        self._state_path = self.root / "review_state.json"
        self.state: Dict[str, str] = (json.loads(self._state_path.read_text(encoding="utf-8"))
                                      if self._state_path.exists() else {})
        self.dropped: Dict[str, List[Box]] = defaultdict(list)
        rj = self.root / "review.jsonl"
        if rj.exists():
            for line in rj.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    d = json.loads(line)
                    self.dropped[Path(d["image"]).stem].append(Box(int(d["class"]), *d["box"], conf=d.get("conf")))
        self.decks: Dict[str, Set[str]] = {}
        rp = self.root / "eval_report.json"
        if rp.exists():
            for row in json.loads(rp.read_text(encoding="utf-8")):
                self.decks[slug(row["recording"])] = {c for cards in row["decks"].values() for c in cards}

    @classmethod
    def open(cls, root: Path | str, model_dir: Path | str, class_map: Path | str) -> "LabelSet":
        names = json.loads((Path(model_dir) / "names.json").read_text(encoding="utf-8"))
        cm = json.loads(Path(class_map).read_text(encoding="utf-8"))
        cards = {int(c["index"]): c.get("card") for c in cm["classes"]}
        return cls(root, names, cards, cm.get("spawned_by", {}))

    # -- frame facts -----------------------------------------------------------------
    @staticmethod
    def match_of(stem: str) -> str:
        m = _STEM.match(stem)
        return m.group("match") if m else stem

    @staticmethod
    def time_ms(stem: str) -> int:
        m = _STEM.match(stem)
        return int(m.group("ms")) if m else 0

    @property
    def matches(self) -> List[str]:
        return sorted({self.match_of(s) for s in self.stems})

    def image_path(self, stem: str) -> Path:
        return self.root / "images" / f"{stem}.jpg"

    def is_reviewed(self, stem: str) -> bool:
        return self.state.get(stem) == "done"

    def boxes(self, stem: str) -> List[Box]:
        """The reviewed labels if the frame was reviewed, else the pseudo labels."""
        reviewed = self.root / "reviewed" / f"{stem}.txt"
        return read_boxes(reviewed if reviewed.exists() else self.root / "labels" / f"{stem}.txt")

    def suggestions(self, stem: str) -> List[Box]:
        return list(self.dropped.get(stem, ()))

    def in_deck(self, stem: str, cls: int) -> bool:
        card = self.class_cards.get(cls)
        deck = self.decks.get(self.match_of(stem), set())
        return bool(card) and (card in deck or bool(set(self.spawned_by.get(card, ())) & deck))

    def class_choices(self, stem: str) -> List[int]:
        """All classes, this match's deck units first, then towers, then the rest (alphabetical in each)."""
        order = lambda c: (0 if self.in_deck(stem, c) else 1 if self.names[c] in TOWERS else 2, self.names[c])  # noqa: E731
        return sorted(range(len(self.names)), key=order)

    # -- review actions ----------------------------------------------------------------
    def save(self, stem: str, boxes: Iterable[Box]) -> None:
        write_boxes(self.root / "reviewed" / f"{stem}.txt", boxes)
        self._set(stem, "done")

    def skip(self, stem: str) -> None:
        """Bad frame (menu, transition, unreadable): never exported."""
        self._set(stem, "skipped")

    def _set(self, stem: str, value: str) -> None:
        self.state[stem] = value
        tmp = self._state_path.with_suffix(".tmp")
        tmp.write_text(json.dumps(self.state, indent=1, sort_keys=True) + "\n", encoding="utf-8")
        tmp.replace(self._state_path)

    def progress(self) -> Dict[str, int]:
        done = sum(v == "done" for v in self.state.values())
        return {"done": done, "skipped": sum(v == "skipped" for v in self.state.values()), "total": len(self.stems)}

    # -- what to review first ------------------------------------------------------------
    def priority(self, stem: str) -> float:
        """Frames where the model is most likely wrong or incomplete: dropped detections (confusion,
        often a rare card), spells, and busy frames (more units = more missed boxes)."""
        units = [b for b in self.boxes(stem) if self.names[b.cls] not in TOWERS]
        spells = sum(self.names[b.cls] in SPELLS for b in units)
        return 3.0 * len(self.dropped.get(stem, ())) + 2.0 * spells + len(units)

    def queue(self, stride_ms: int = 2000, include_reviewed: bool = False) -> List[str]:
        """Review order: one frame per `stride_ms` (neighbours are near-duplicates), best first within each
        match, matches interleaved so every match gets reviewed evenly."""
        per: Dict[str, List[str]] = defaultdict(list)
        for s in self.stems:
            if self.time_ms(s) % stride_ms:
                continue
            if not include_reviewed and s in self.state:
                continue
            per[self.match_of(s)].append(s)
        for m in per:
            per[m].sort(key=lambda s: (-self.priority(s), s))
        out, k = [], 0
        while any(k < len(v) for v in per.values()):
            out += [per[m][k] for m in sorted(per) if k < len(per[m])]
            k += 1
        return out

    # -- training set -------------------------------------------------------------------
    def export(self, out: Path | str, val_match: Optional[str] = None) -> Dict[str, int]:
        """YOLO dataset from reviewed frames: train = every match but `val_match`, val = that match
        (default: the last match in name order)."""
        out = Path(out)
        val_match = val_match or (self.matches[-1] if self.matches else None)
        counts = {"train": 0, "val": 0, "boxes": 0}
        for split in ("train", "val"):
            for sub in ("images", "labels"):
                d = out / sub / split
                if d.exists():
                    shutil.rmtree(d)
                d.mkdir(parents=True)
        for stem in self.stems:
            if not self.is_reviewed(stem):
                continue
            split = "val" if self.match_of(stem) == val_match else "train"
            boxes = self.boxes(stem)
            shutil.copy2(self.image_path(stem), out / "images" / split / f"{stem}.jpg")
            write_boxes(out / "labels" / split / f"{stem}.txt", boxes)
            counts[split] += 1
            counts["boxes"] += len(boxes)
        (out / "data.yaml").write_text(
            f"# reviewed Mode B frames; val = whole match {val_match}\npath: .\n"
            "train: images/train\nval: images/val\nnames:\n"
            + "".join(f"  {i}: {n}\n" for i, n in enumerate(self.names)), encoding="utf-8")
        return counts
