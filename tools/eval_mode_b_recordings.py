"""
Evaluate the Mode B detector on real screen recordings, using the official battlelog as (weak) ground truth,
and export the frames as a pseudo-labelled fine-tuning set.

    python tools/eval_mode_b_recordings.py recordings/ data/raw_sources/my_battlelog.json \
        [--fps 2] [--conf 0.4] [--export data/finetune/ipad_20261116] [--max-seconds 30]

What the battlelog can and cannot tell us
* It has both decks, crowns and final tower hitpoints, but no play-by-play timeline and no boxes.
* So each match is checked against its own 16 cards: a detection of a card in neither deck (or a unit no deck
  card spawns) is certainly wrong; one of a deck card is "consistent" (likely right, not proven right).
* Card recall = deck cards the model can see (has a class for) that were detected in >= --min-frames frames.
* Calibration check: the queen / king tower detections must sit where the training crop has the towers.

Each recording is matched to a battle by its end time (file name start time + duration ~ battleTime, UTC).
Two layouts are compared: the phone default box (what an uncalibrated iPad user gets) and the layout from the
frame's aspect ratio (ArenaCalibration.default_for, iPad preset for tablets).

The export writes the 568x896 arena crops (what the detector sees) as JPEGs, YOLO label files holding only the
deck-consistent detections (pseudo labels, review before training), review.jsonl with every rejected
detection, and data.yaml with the model's permanent class order. Battle frames only (towers visible).
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from collections import Counter, defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend_engine.perception.live.calibration import DEFAULT_FRACTIONS, ArenaCalibration  # noqa: E402
from backend_engine.perception.live.detector import letterbox, yolo_postprocess  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
MODEL = ROOT / "pretrained_weights" / "sudhishneel_v8" / "v8_best.onnx"
CLASS_MAP = ROOT / "configs" / "detector_class_map.json"
CROP_W, CROP_H = 568, 896
# where the dataset's 568x896 crop has the towers (dataset README)
TOWER_POINTS = {"queen-tower": [(116, 200), (461, 200), (116, 689), (461, 689)],
                "king-tower": [(284, 108), (284, 790)]}
VIDEO_EXT = {".mov", ".mp4", ".m4v"}


# ------------------------------------------------------------------ matching recordings to battles
def card_id(name: str) -> str:
    """Battlelog display name -> snake_case card id ("Mini P.E.K.K.A" -> mini_pekka)."""
    s = name.lower().replace(".", "").replace("'", "")
    return re.sub(r"[^a-z0-9]+", "_", s).strip("_")


def recording_start(path: Path) -> Optional[datetime]:
    """iOS names recordings 'ScreenRecording_MM-DD-YYYY HH-MM-SS_1' (local time)."""
    m = re.search(r"(\d\d)-(\d\d)-(\d{4}) (\d\d)-(\d\d)-(\d\d)", path.stem)
    if not m:
        return None
    mo, d, y, hh, mi, ss = map(int, m.groups())
    return datetime(y, mo, d, hh, mi, ss)


def battle_time(b: dict) -> datetime:
    return datetime.strptime(b["battleTime"], "%Y%m%dT%H%M%S.%fZ").replace(tzinfo=timezone.utc)


def match_recordings(recs: List[Tuple[Path, float]], battles: List[dict],
                     tol_s: float = 90.0) -> Dict[Path, Optional[dict]]:
    """Pair each (recording, duration) with the battle that ended when the recording ended.
    The local-time offset is unknown: try every quarter hour and keep the one that pairs the most."""
    ends = {p: recording_start(p) + timedelta(seconds=dur) for p, dur in recs if recording_start(p)}
    best: Tuple[int, float, Dict[Path, Optional[dict]]] = (-1, 0.0, {})
    for q in range(-14 * 4, 14 * 4 + 1):
        off = timedelta(minutes=15 * q)
        pairs, err = {}, 0.0
        for p, end in ends.items():
            end_utc = (end - off).replace(tzinfo=timezone.utc)
            b = min(battles, key=lambda x: abs((battle_time(x) - end_utc).total_seconds()))
            e = abs((battle_time(b) - end_utc).total_seconds())
            if e <= tol_s:
                pairs[p], err = b, err + e
        if (len(pairs), -err) > (best[0], -best[1]):
            best = (len(pairs), err, pairs)
    return {p: best[2].get(p) for p, _ in recs}


def probe(path: Path) -> Tuple[int, int, float]:
    out = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                          "stream=width,height:format=duration", "-of", "json", str(path)],
                         capture_output=True, text=True, check=True).stdout
    j = json.loads(out)
    s = j["streams"][0]
    return int(s["width"]), int(s["height"]), float(j["format"]["duration"])


def frames(path: Path, w: int, h: int, fps: float, max_s: Optional[float]):
    cmd = ["ffmpeg", "-v", "error", "-hwaccel", "videotoolbox", "-i", str(path), "-an"]
    if max_s:
        cmd += ["-t", str(max_s)]
    cmd += ["-vf", f"fps={fps}", "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE)
    n, k = w * h * 3, 0
    try:
        while True:
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            yield k / fps, np.frombuffer(buf, np.uint8).reshape(h, w, 3)
            k += 1
    finally:
        proc.stdout.close()
        proc.wait()


def crop(frame: np.ndarray, cal: ArenaCalibration) -> np.ndarray:
    from PIL import Image
    h, w = frame.shape[:2]
    l, t, r, b = cal.crop_box(w, h)
    l, t, r, b = max(0, round(l)), max(0, round(t)), min(w, round(r)), min(h, round(b))
    return np.asarray(Image.fromarray(frame[t:b, l:r]).resize((CROP_W, CROP_H), Image.BILINEAR))


# ------------------------------------------------------------------ per-match bookkeeping
class MatchStats:
    def __init__(self, allowed: Set[str], visible: Set[str]):
        self.allowed, self.visible = allowed, visible
        self.frames = self.battle_frames = 0
        self.dets = self.consistent = 0
        self.card_frames: Counter = Counter()
        self.wrong: Counter = Counter()
        self.tower_err: List[float] = []
        self.ms: List[float] = []

    def add(self, preds, names, cmap, units_ok) -> List[Tuple[int, float, np.ndarray, bool]]:
        self.frames += 1
        towers = [(names[c], box) for c, conf, box in preds if names[c] in TOWER_POINTS]
        if not towers:
            return []
        self.battle_frames += 1
        for n, box in towers:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            self.tower_err.append(min(np.hypot(cx - x, cy - y) for x, y in TOWER_POINTS[n]))
        seen, out = set(), []
        for c, conf, box in preds:
            card = cmap.get(c)
            if card is None:
                continue
            ok = units_ok(card)
            self.dets += 1
            self.consistent += ok
            if ok:
                seen.add(card)
            else:
                self.wrong[card] += 1
            out.append((c, conf, box, ok))
        self.card_frames.update(seen)
        return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("recordings", type=Path)
    ap.add_argument("battlelog", type=Path)
    ap.add_argument("--model", type=Path, default=MODEL)
    ap.add_argument("--fps", type=float, default=2.0)
    ap.add_argument("--conf", type=float, default=0.4)
    ap.add_argument("--min-frames", type=int, default=2, help="frames a card needs to count as found")
    ap.add_argument("--max-seconds", type=float, default=None, help="only the first N seconds (quick test)")
    ap.add_argument("--export", type=Path, default=None, help="write the fine-tuning set here")
    ap.add_argument("--report", type=Path, default=None, help="write the metrics as JSON here")
    a = ap.parse_args()

    import onnxruntime as ort
    from PIL import Image

    providers = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    sess = ort.InferenceSession(str(a.model), providers=providers)
    inp = sess.get_inputs()[0].name
    names = json.loads((a.model.parent / "names.json").read_text())
    cm = json.loads(CLASS_MAP.read_text())
    imgsz = int(cm.get("input_size", 800))
    cmap = {int(c["index"]): c.get("card") for c in cm["classes"]}
    spawned_by = cm.get("spawned_by", {})
    model_cards = {c for c in cmap.values() if c}

    battles = json.loads(a.battlelog.read_text())
    recs = sorted(p for p in a.recordings.iterdir() if p.suffix.lower() in VIDEO_EXT)
    info = {p: probe(p) for p in recs}
    pairs = match_recordings([(p, info[p][2]) for p in recs], battles)

    if a.export:
        (a.export / "images").mkdir(parents=True, exist_ok=True)
        (a.export / "labels").mkdir(parents=True, exist_ok=True)
        review = (a.export / "review.jsonl").open("w", encoding="utf-8")

    report = []
    for rec in recs:
        w, h, dur = info[rec]
        b = pairs[rec]
        print(f"\n== {rec.name}: {w}x{h} (aspect {w / h:.3f}), {dur:.0f} s")
        if b is None:
            print("   no battle in the battlelog ends with this recording: skipped")
            continue
        team, opp = b["team"][0], b["opponent"][0]
        decks = {0: {card_id(c["name"]) for c in team["cards"]}, 1: {card_id(c["name"]) for c in opp["cards"]}}
        allowed = decks[0] | decks[1]

        def units_ok(card: str, allowed=allowed) -> bool:
            return card in allowed or bool(set(spawned_by.get(card, ())) & allowed)

        print(f"   battle {b['battleTime']} {b['gameMode']['name']}: {team['crowns']}-{opp['crowns']}"
              f"   unknown to the model: {sorted(allowed - model_cards) or 'none'}")
        layouts = {"phone_default": ArenaCalibration(w, h, *(f * s for f, s in zip(DEFAULT_FRACTIONS, (w, h, w, h)))),
                   "auto_layout": ArenaCalibration.default_for(w, h)}
        stats = {k: MatchStats(allowed, allowed & model_cards) for k in layouts}
        slug = re.sub(r"[^0-9A-Za-z]+", "_", rec.stem).strip("_")
        for t, frame in frames(rec, w, h, a.fps, a.max_seconds):
            for k, cal in layouts.items():
                view = crop(frame, cal)
                img, scale, pad = letterbox(view, imgsz)
                x = img.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
                preds = yolo_postprocess(sess.run(None, {inp: x})[0], scale, pad, a.conf)
                kept = stats[k].add(preds, names, cmap, units_ok)
                if a.export and k == "auto_layout":
                    if not any(names[c] in TOWER_POINTS for c, _, _ in preds):
                        continue
                    stem = f"{slug}_t{int(t * 1000):07d}"
                    Image.fromarray(view).save(a.export / "images" / f"{stem}.jpg", quality=92)
                    lines = []
                    for c, conf, bx, ok in kept:
                        x0, y0, x1, y1 = (float(v) for v in bx)
                        if ok:
                            lines.append(f"{c} {(x0 + x1) / 2 / CROP_W:.6f} {(y0 + y1) / 2 / CROP_H:.6f} "
                                         f"{(x1 - x0) / CROP_W:.6f} {(y1 - y0) / CROP_H:.6f}")
                        else:
                            review.write(json.dumps({"image": f"images/{stem}.jpg", "class": c, "name": names[c],
                                                     "conf": round(conf, 3), "box": [round(float(v), 1) for v in bx],
                                                     "why": "card in neither deck"}) + "\n")
                    # towers are labelled too, so the model keeps seeing them
                    for c, conf, bx in preds:
                        if names[c] in TOWER_POINTS:
                            x0, y0, x1, y1 = (float(v) for v in bx)
                            lines.append(f"{c} {(x0 + x1) / 2 / CROP_W:.6f} {(y0 + y1) / 2 / CROP_H:.6f} "
                                         f"{(x1 - x0) / CROP_W:.6f} {(y1 - y0) / CROP_H:.6f}")
                    (a.export / "labels" / f"{stem}.txt").write_text("\n".join(lines) + ("\n" if lines else ""))

        row = {"recording": rec.name, "size": [w, h], "battle": b["battleTime"], "mode": b["gameMode"]["name"],
               "crowns": [team["crowns"], opp["crowns"]], "decks": {k: sorted(v) for k, v in decks.items()},
               "layouts": {}}
        for k, s in stats.items():
            found = {c for c in s.visible if s.card_frames[c] >= a.min_frames}
            m = {"frames": s.frames, "battle_frames": s.battle_frames, "detections": s.dets,
                 "deck_consistent": round(s.consistent / max(1, s.dets), 3),
                 "card_recall": round(len(found) / max(1, len(s.visible)), 3),
                 "found": sorted(found), "missed": sorted(s.visible - found),
                 "tower_err_px": round(float(np.median(s.tower_err)), 1) if s.tower_err else None,
                 "wrong_cards": dict(s.wrong.most_common(6))}
            row["layouts"][k] = m
            print(f"   [{k:13s}] battle frames {s.battle_frames}/{s.frames}  dets {s.dets}  "
                  f"deck-consistent {m['deck_consistent']:.1%}  card recall {len(found)}/{len(s.visible)}  "
                  f"tower err {m['tower_err_px']} px")
            print(f"   {'':15s} missed {m['missed']}  wrong {m['wrong_cards']}")
        report.append(row)

    if a.export:
        review.close()
        (a.export / "data.yaml").write_text("path: .\ntrain: images\nval: images\nnames:\n" +
                                            "".join(f"  {i}: {n}\n" for i, n in enumerate(names)))
        (a.export / "README.txt").write_text(
            "Pseudo labels from SudhishNeel v8 on the user's own iPad recordings, filtered to each match's 16\n"
            "battlelog cards. NOT verified: review/correct every label before training. review.jsonl lists the\n"
            "detections that were dropped (card in neither deck). Private gameplay: do not publish.\n")
    if a.report:
        a.report.parent.mkdir(parents=True, exist_ok=True)
        a.report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    for k in ("phone_default", "auto_layout"):
        rows = [r["layouts"][k] for r in report]
        if rows:
            d = sum(r["detections"] for r in rows)
            c = sum(r["detections"] * r["deck_consistent"] for r in rows)
            f = sum(len(r["found"]) for r in rows)
            v = sum(len(r["found"]) + len(r["missed"]) for r in rows)
            print(f"\nTOTAL [{k}] deck-consistent {c / max(1, d):.1%} of {d} detections, card recall {f}/{v}")


if __name__ == "__main__":
    main()
