"""Label review core (backend_engine/perception/label_review.py) on a tiny fake export."""
import json

import pytest
from PIL import Image

from backend_engine.perception.label_review import Box, LabelSet, read_boxes

NAMES = ["king-tower", "queen-tower", "hog-rider", "zap", "skeleton", "knight"]
CARDS = {0: None, 1: None, 2: "hog_rider", 3: "zap", 4: "skeletons", 5: "knight"}


def make_set(tmp_path):
    root = tmp_path / "set"
    (root / "images").mkdir(parents=True)
    (root / "labels").mkdir()
    for m in ("Rec_A_1", "Rec_B_1"):
        for ms in (0, 500, 2000, 4000):
            stem = f"{m}_t{ms:07d}"
            Image.new("RGB", (568, 896)).save(root / "images" / f"{stem}.jpg")
            (root / "labels" / f"{stem}.txt").write_text("1 0.2 0.2 0.1 0.1\n2 0.5 0.5 0.1 0.1\n")
    # B at 4 s: busy frame with a spell -> reviewed first in its match
    (root / "labels" / "Rec_B_1_t0004000.txt").write_text(
        "1 0.2 0.2 0.1 0.1\n3 0.5 0.5 0.2 0.2\n4 0.3 0.6 0.05 0.05\n4 0.35 0.6 0.05 0.05\n")
    (root / "review.jsonl").write_text(json.dumps(
        {"image": "images/Rec_A_1_t0002000.jpg", "class": 5, "conf": 0.5, "box": [10, 20, 60, 90]}) + "\n")
    (root / "eval_report.json").write_text(json.dumps([
        {"recording": "Rec A_1.mov", "decks": {"0": ["hog_rider"], "1": ["zap"]}},
        {"recording": "Rec B_1.MP4", "decks": {"0": ["graveyard"], "1": ["knight"]}}]))
    return LabelSet(root, NAMES, CARDS, {"skeletons": ["graveyard"]})


def test_box_yolo_round_trip_and_clamp():
    b = Box(2, 100, 200, 160, 290)
    back = Box.from_yolo(b.yolo())
    assert (back.cls, back.x0, back.y0, back.x1, back.y1) == pytest.approx((2, 100, 200, 160, 290), abs=1e-3)
    c = Box(1, -20, 880, 30, 950).clamped()
    assert (c.x0, c.y1) == (0.0, 896.0) and c.valid
    assert not Box(1, 5, 5, 6, 6).valid


def test_queue_stride_priority_and_interleave(tmp_path):
    ls = make_set(tmp_path)
    q = ls.queue()
    assert all(ls.time_ms(s) % 2000 == 0 for s in q)       # 0.5 s neighbours left out
    assert [ls.match_of(s) for s in q[:2]] == ["Rec_A_1", "Rec_B_1"]   # matches interleaved
    assert q[0] == "Rec_A_1_t0002000"                       # dropped detection first
    assert q[1] == "Rec_B_1_t0004000"                       # spell + busy first
    assert len(q) == 6


def test_deck_classes_first_and_suggestions(tmp_path):
    ls = make_set(tmp_path)
    stem = "Rec_B_1_t0004000"
    choices = ls.class_choices(stem)
    assert set(choices[:2]) == {4, 5}                        # skeleton (graveyard spawns it) + knight
    assert set(choices[2:4]) == {0, 1}                       # then towers
    assert ls.suggestions("Rec_A_1_t0002000")[0].cls == 5


def test_save_skip_persist_and_export_holds_out_last_match(tmp_path):
    ls = make_set(tmp_path)
    ls.save("Rec_A_1_t0000000", [Box(2, 10, 10, 50, 60), Box(5, 0, 0, 1, 1)])   # tiny box dropped
    ls.save("Rec_B_1_t0002000", [])
    ls.skip("Rec_A_1_t0002000")
    assert len(read_boxes(tmp_path / "set" / "labels" / "Rec_A_1_t0000000.txt")) == 2   # pseudo kept
    again = LabelSet(ls.root, NAMES, CARDS, ls.spawned_by)   # state survives a reload
    assert again.progress() == {"done": 2, "skipped": 1, "total": 8}
    assert [b.cls for b in again.boxes("Rec_A_1_t0000000")] == [2]
    assert "Rec_A_1_t0002000" not in again.queue()
    n = again.export(tmp_path / "out")
    assert (n["train"], n["val"], n["boxes"]) == (1, 1, 1)
    assert (tmp_path / "out" / "labels" / "val" / "Rec_B_1_t0002000.txt").exists()
    assert "val: images/val" in (tmp_path / "out" / "data.yaml").read_text()
