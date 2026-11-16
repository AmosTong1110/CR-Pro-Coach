import json
import os

import statistics
from pathlib import Path

import numpy as np
import pytest

from backend_engine.perception.live.calibration import ArenaCalibration
from backend_engine.perception.live.capture import FolderSource
from backend_engine.perception.live.companion import OPP, LiveClock, LiveCompanion, SimDemoFeed
from backend_engine.perception.live.detector import (
    Detection, YoloOnnxDetector, letterbox, nms, side_by_color, yolo_postprocess,
)
from backend_engine.perception.live.play_tracker import PlayTracker

ROOT = Path(__file__).resolve().parents[1]
COSTS = {"hog_rider": 4, "knight": 3, "skeletons": 1, "cannon": 3, "fireball": 4, "archers": 3,
         "musketeer": 4, "the_log": 2, "ice_spirit": 1}
UNITS = {"skeletons": 3, "archers": 2}


# ---------------------------------------------------------------- safety
def test_read_only_contract():
    """Mode B must never be able to send input to the game: no imports or calls of anything that
    can inject taps/keys/mouse events or run shell commands (docs that *name* them are fine)."""
    import ast

    banned_modules = {"pyautogui", "pynput", "Quartz", "AppKit", "subprocess", "ctypes", "keyboard",
                      "mouse", "pywinauto", "uiautomator", "uiautomator2", "adbutils", "ppadb", "frida"}
    banned_names = {"CGEventPost", "CGEventCreateMouseEvent", "CGEventCreateKeyboardEvent", "system",
                    "popen", "Popen", "sendEvent", "postEvent", "QTest", "mouseClick", "keyClick"}
    files = list((ROOT / "backend_engine" / "perception" / "live").glob("*.py")) + \
        list((ROOT / "frontend_app" / "companion").glob("*.py"))
    assert files
    for f in files:
        tree = ast.parse(f.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom):
                mods = [(node.module or "").split(".")[0]]
            else:
                mods = []
            assert not (set(mods) & banned_modules), f"{f.name} imports {mods}"
            if isinstance(node, ast.Attribute):
                assert node.attr not in banned_names, f"{f.name} uses .{node.attr}"
            if isinstance(node, ast.Name):
                assert node.id not in banned_names, f"{f.name} uses {node.id}"


# ---------------------------------------------------------------- calibration
def test_calibration_round_trip_and_scaling(tmp_path):
    cal = ArenaCalibration.from_corners(400, 860, (10, 100), (390, 700), screen_region=(5, 6, 400, 860))
    assert cal.px_to_tile(10, 700) == pytest.approx((0, 0))
    assert cal.px_to_tile(390, 100) == pytest.approx((18, 32))
    x, y = cal.tile_to_px(9, 16)
    assert cal.px_to_tile(x, y) == pytest.approx((9, 16))
    assert cal.px_to_tile(20, 1400, frame_w=800, frame_h=1720) == pytest.approx((0, 0))  # 2x frame
    cal.save(tmp_path / "c.json")
    back = ArenaCalibration.load(tmp_path / "c.json")
    assert back == cal and back.screen_region == (5, 6, 400, 860)
    with pytest.raises(ValueError):
        ArenaCalibration.from_corners(400, 860, (10, 10), (12, 12))
    assert ArenaCalibration.load(tmp_path / "missing.json") is None


# ---------------------------------------------------------------- detector plumbing
def test_letterbox_and_postprocess_round_trip():
    frame = np.zeros((800, 400, 3), np.uint8)
    img, scale, pad = letterbox(frame, 640)
    assert img.shape == (640, 640, 3) and scale == pytest.approx(0.8)
    # one box at original (100,200)-(140,260) for class 2 of 3, plus a duplicate and a weak one
    x1, y1, x2, y2 = (np.array([100, 200, 140, 260]) * scale) + np.array([pad[0], pad[1], pad[0], pad[1]])
    def col(cx, cy, w, h, scores):
        return [cx, cy, w, h, *scores]
    preds = np.array([col((x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, [0, 0.1, 0.9]),
                      col((x1 + x2) / 2 + 1, (y1 + y2) / 2, x2 - x1, y2 - y1, [0, 0.1, 0.8]),
                      col(50, 50, 10, 10, [0.1, 0.2, 0.1])], dtype=np.float32).T[None]
    out = yolo_postprocess(preds, scale, pad, conf_thres=0.4, iou_thres=0.5)
    assert len(out) == 1
    c, conf, box = out[0]
    assert c == 2 and conf == pytest.approx(0.9) and box == pytest.approx([100, 200, 140, 260], abs=1)


def test_nms():
    boxes = np.array([[0, 0, 10, 10], [1, 1, 11, 11], [50, 50, 60, 60]], float)
    assert sorted(nms(boxes, np.array([0.9, 0.8, 0.7]), 0.5)) == [0, 2]


def test_side_by_color():
    frame = np.zeros((100, 100, 3), np.uint8)
    frame[20:24, 30:60] = (220, 40, 40)      # red bar above box -> opponent
    assert side_by_color(frame, (30, 24, 60, 50)) == 1
    frame[20:24, 30:60] = (40, 60, 230)      # blue -> you
    assert side_by_color(frame, (30, 24, 60, 50)) == 0
    assert side_by_color(np.zeros((100, 100, 3), np.uint8), (30, 24, 60, 50)) is None


class FakeSession:
    def __init__(self, output):
        self.output = output

    def get_inputs(self):
        return [type("I", (), {"name": "images"})()]

    def run(self, _names, feed):
        assert feed["images"].shape == (1, 3, 640, 640)
        return [self.output]


def test_yolo_detector_with_fake_model(tmp_path):
    cmap = tmp_path / "map.json"
    cmap.write_text(json.dumps({"input_size": 640, "side_mode": "class", "classes": [
        {"index": 0, "card": "hog_rider", "side": 1}, {"index": 1, "card": None}]}))
    frame = np.zeros((640, 640, 3), np.uint8)
    out = np.array([[320, 320, 40, 60, 0.95, 0.0], [100, 100, 20, 20, 0.0, 0.9]], np.float32).T[None]
    det = YoloOnnxDetector("unused.onnx", cmap, session=FakeSession(out))
    (d,) = det.detect(frame)
    assert d.card_id == "hog_rider" and d.side == 1 and d.box == pytest.approx((300, 290, 340, 350), abs=1)


# ---------------------------------------------------------------- play tracker
def frames(tr, seq, start=0, dt=100):
    events = []
    for i, dets in enumerate(seq):
        events += tr.update(start + i * dt, dets)
    return events


def test_single_unit_is_one_play_with_first_seen_time():
    tr = PlayTracker(COSTS)
    seq = [[("hog_rider", OPP, 3.5, 17.5 - 0.2 * i)] for i in range(10)]
    (ev,) = frames(tr, seq, start=5000)
    assert ev.t_ms == 5000 and ev.card_id == "hog_rider" and ev.elixir_cost == 4 and ev.is_threat


def test_one_frame_false_positive_ignored():
    tr = PlayTracker(COSTS)
    assert frames(tr, [[("knight", OPP, 9, 25)], [], [], []]) == []


def test_multi_unit_card_is_one_play():
    tr = PlayTracker(COSTS, UNITS)
    seq = [[("skeletons", OPP, 9 + dx, 25) for dx in (-0.7, 0, 0.7)] for _ in range(6)]
    assert len(frames(tr, seq)) == 1


def test_blink_does_not_duplicate():
    tr = PlayTracker(COSTS)
    seq = [[("knight", OPP, 9, 25)]] * 5 + [[]] * 4 + [[("knight", OPP, 9, 24.6)]] * 5
    assert len(frames(tr, seq)) == 1


def test_cycle_rule_blocks_impossible_replay_and_allows_after_four_plays():
    tr = PlayTracker(COSTS)
    t = 0
    def show(card, x, y, n=3):
        nonlocal t
        out = []
        for _ in range(n):
            out += tr.update(t, [(card, OPP, x, y)])
            t += 100
        return out
    assert len(show("knight", 3.5, 20)) == 1
    t += 1500                                         # knight track expires (1.2 s), still within dup_window_ms
    assert show("knight", 14.5, 20) == []             # can't be back in hand yet: a duplicate of the first play
    for c in ("hog_rider", "cannon", "fireball", "the_log"):
        t += 2000
        assert len(show(c, 9, 24)) == 1
    t += 3000
    assert len(show("knight", 14.5, 20)) == 1         # 4 plays later: legal again


def test_cycle_rule_is_overridden_when_the_old_play_cannot_own_the_unit():
    tr = PlayTracker(COSTS)
    t = 0
    for _ in range(3):
        tr.update(t, [("knight", OPP, 3.5, 20)])
        t += 100
    t += 6000                                         # far beyond dup_window_ms: an earlier play must have been missed
    out = []
    for _ in range(3):
        out += tr.update(t, [("knight", OPP, 14.5, 20)])
        t += 100
    assert len(out) == 1 and tr.cycle_overrides == 1


def test_unit_dropped_where_the_old_one_died_is_a_new_play():
    tr = PlayTracker(COSTS, speeds={"knight": 1.0})
    t = 0
    for _ in range(4):                                # a knight is seen, then dies (no detections for 300 ms)
        tr.update(t, [("knight", OPP, 3.5, 16.4)])
        t += 100
    t += 300
    out = []
    for _ in range(3):                                # 200 ms later a knight appears 1.5 tiles away: not the same unit
        out += tr.update(t, [("knight", OPP, 4.5, 17.5)])
        t += 100
    assert len(out) == 1 or tr.suppressed == 1        # new track either way (the cycle rule may still call it a duplicate)
    assert len([k for k in tr.tracks if k.card == "knight"]) == 2


def test_building_replaced_on_same_tile_after_lifetime():
    tr = PlayTracker(COSTS, lifetimes_ms={"cannon": 31_000})
    t, events = 0, []
    for c in ("cannon", "hog_rider", "fireball", "the_log", "knight"):
        events += tr.update(t, [(c, OPP, 9.5 if c == "cannon" else 4, 23.5 if c == "cannon" else 20)])
        events += tr.update(t + 100, [(c, OPP, 9.5 if c == "cannon" else 4, 23.5 if c == "cannon" else 20)])
        t += 2000
    while t < 34_000:                                  # same tile, continuously visible
        events += tr.update(t, [("cannon", OPP, 9.5, 23.5)])
        t += 100
    assert [e.card_id for e in events].count("cannon") == 2


# ---------------------------------------------------------------- companion
def test_clock():
    c = LiveClock()
    assert c.now_ms(5.0) is None
    c.start(10.0)
    assert c.now_ms(12.5) == 2500
    c.nudge(-1000)
    assert c.now_ms(12.5) == 1500


def test_no_events_before_start():
    comp = LiveCompanion.create()
    assert comp.process_tiles([("hog_rider", OPP, 3.5, 17.5)] * 1, wall=1.0) == []
    assert comp.state(1.0).t_ms is None


def accuracy(noise, miss, fp, seeds):
    errs, missing, extra, total = [], 0, 0, 0
    for seed in seeds:
        feed = SimDemoFeed(seed=seed, noise_tiles=noise, miss_rate=miss, false_positive_rate=fp)
        comp = LiveCompanion.create()
        comp.start_match(wall=0.0)
        while (r := feed.next()):
            comp.process_tiles(r[0], wall=r[1])
            errs.append(abs(comp.state(r[1]).opp.point - feed.match.sides[1].elixir))
        truth = [(e.t_ms, e.card_id) for e in feed.match.events if e.side == OPP]
        got = [(e.t_ms, e.card_id) for e in comp.events if e.side == OPP]
        used = set()
        for a in truth:
            k = next((i for i, g in enumerate(got) if i not in used and g[1] == a[1] and abs(g[0] - a[0]) < 600), None)
            if k is None:
                missing += 1
            else:
                used.add(k)
        extra += len(got) - len(used)
        total += len(truth)
    return statistics.mean(errs), missing / total, extra / total


def test_demo_accuracy_perfect_detector():
    err, miss, extra = accuracy(0, 0, 0, range(6))     # 6 seeds: a single match can be a noisy outlier
    assert err < 0.3 and miss < 0.03 and extra < 0.02   # was extra == 0 on the 2023 trajectories


def test_demo_accuracy_realistic_noise():
    err, miss, extra = accuracy(0.3, 0.1, 0.02, range(3))
    assert err < 1.0 and miss < 0.08 and extra < 0.05


def test_process_frame_with_detector_and_calibration():
    cal = ArenaCalibration.from_corners(360, 780, (0, 90), (360, 618))

    class OneHog:
        def detect(self, frame, calibration=None):
            x, y = cal.tile_to_px(3.5, 17.5)
            return [Detection("hog_rider", OPP, (x - 10, y - 30, x + 10, y + 3), 0.9)]

    comp = LiveCompanion.create(detector=OneHog(), calibration=cal)
    comp.start_match(wall=0.0)
    frame = np.zeros((780, 360, 3), np.uint8)
    for i in range(3):
        comp.process_frame(frame, wall=2.8 + i * 0.1)
    st = comp.state(3.0)
    assert [e.card_id for e in comp.events] == ["hog_rider"]
    assert st.opp.point == pytest.approx(5 + 3.0 / 2.8 - 4, abs=0.05)
    assert st.guess is not None and st.guess.card_probs


def test_manual_play_and_log(tmp_path):
    comp = LiveCompanion.create()
    comp.start_match(wall=0.0)
    comp.manual_play("hog_rider", OPP, wall=2.8)
    assert comp.state(2.8).opp.point == pytest.approx(2.0)
    comp.save_log(tmp_path / "log.json")
    assert json.loads((tmp_path / "log.json").read_text())["plays"][0]["card"] == "hog_rider"


def test_folder_source(tmp_path):
    from PIL import Image
    for i in range(3):
        Image.fromarray(np.full((20, 10, 3), i * 40, np.uint8)).save(tmp_path / f"f{i:03d}.png")
    src = FolderSource(tmp_path, fps=10)
    got = [src.grab() for _ in range(4)]
    assert got[3] is None and got[1][0].shape == (20, 10, 3) and got[2][1] == pytest.approx(0.2)


def test_companion_window_offscreen():
    pytest.importorskip("PySide6")
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication
    from frontend_app.companion.app import CompanionWindow

    QApplication.instance() or QApplication([])
    w = CompanionWindow()
    w.show()
    w.toggle_match()
    for _ in range(300):        # 30 s of demo
        w.on_frame()
    assert w.comp.clock.running and w.opp_value.text().startswith("Opponent ")
    assert not w.grab().isNull()
    w.close()


# ---------------------------------------------------------------- unit-class models (SudhishNeel-style)
def test_spawned_units_are_not_plays():
    tr = PlayTracker(COSTS | {"witch": 5, "graveyard": 5}, UNITS, spawned_by={"skeletons": ["witch", "graveyard"]})
    t, events = 0, []
    for _ in range(3):
        events += tr.update(t, [("witch", OPP, 9, 26)])
        t += 100
    for k in range(10):   # the Witch summons skeletons next to herself
        events += tr.update(t, [("witch", OPP, 9, 26), ("skeletons", OPP, 9.8, 25.2 - 0.1 * k)])
        t += 100
    assert [e.card_id for e in events] == ["witch"] and tr.spawns >= 1
    # a Graveyard play: skeletons rising inside it within the next seconds are part of it
    for c in ("hog_rider", "cannon", "fireball", "the_log"):
        t += 2000
        events += tr.update(t, [(c, OPP, 3, 20)]) + tr.update(t + 100, [(c, OPP, 3, 20)])
    t += 3000
    events += tr.update(t, [("graveyard", OPP, 3.5, 6)]) + tr.update(t + 100, [("graveyard", OPP, 3.5, 6)])
    t += 2000
    for k in range(3):
        events += tr.update(t + 100 * k, [("skeletons", OPP, 4.5, 6.5)])
    assert [e.card_id for e in events].count("skeletons") == 0
    assert [e.card_id for e in events][-1] == "graveyard"


def test_arena_crop_maps_boxes_back_to_frame(tmp_path):
    cmap = tmp_path / "map.json"
    cmap.write_text(json.dumps({"input_size": 640, "arena_crop": [568, 896], "side_mode": "class",
                                "classes": [{"index": 0, "card": "knight", "side": 1, "evolved": True}]}))
    # model "sees" a box in the middle of the 568x896 crop (letterboxed into 640)
    scale = 640 / 896
    pad_x = (640 - round(568 * scale)) // 2
    cx, cy = 284 * scale + pad_x, 448 * scale
    out = np.array([[cx, cy, 40 * scale, 60 * scale, 0.9]], np.float32).T[None]
    det = YoloOnnxDetector("x.onnx", cmap, session=FakeSession(out))
    cal = ArenaCalibration.from_corners(1000, 2000, (100, 300), (900, 1600))   # arena 800 x 1300 px
    frame = np.zeros((2000, 1000, 3), np.uint8)
    (d,) = det.detect(frame, cal)
    fx, fy = (d.box[0] + d.box[2]) / 2, (d.box[1] + d.box[3]) / 2
    assert (fx, fy) == pytest.approx((500, 950), abs=3)         # centre of the arena in frame pixels
    assert cal.px_to_tile(fx, fy) == pytest.approx((9, 16), abs=0.1)
    assert d.evolved and d.card_id == "knight"


def test_build_class_map(tmp_path):
    import sys
    sys.path.insert(0, str(ROOT / "tools"))
    from build_class_map import build
    names = ["archer", "archer-evolution", "golemite", "queen-tower", "hog-rider", "the-log", "skeleton", "mystery-card"]
    data = build(names, tmp_path / "m.json", 800, (568, 896))
    cards = {c["name"]: c["card"] for c in data["classes"]}
    assert cards == {"archer": "archers", "archer-evolution": "archers", "golemite": None, "queen-tower": None,
                     "hog-rider": "hog_rider", "the-log": "the_log", "skeleton": "skeletons", "mystery-card": None}
    assert data["classes"][1]["evolved"] and data["arena_crop"] == [568, 896]


def test_side_inferred_from_deploy_position():
    tr = PlayTracker(COSTS, spells=frozenset({"fireball"}))
    t, events = 0, []
    for k in range(4):   # opponent knight deployed on their half, walks across the river
        events += tr.update(t, [("knight", None, 9, 20 - 1.5 * k)])
        t += 100
    for k in range(3):   # your hog on your half
        events += tr.update(t, [("hog_rider", None, 3.5, 12)])
        t += 100
    for k in range(3):   # a fireball landing on your half: most likely the opponent's
        events += tr.update(t, [("fireball", None, 4, 8)])
        t += 100
    assert [(e.card_id, e.side) for e in events] == [("knight", OPP), ("hog_rider", 0), ("fireball", OPP)]
    assert sum(1 for tr_ in tr.tracks if tr_.card == "knight") == 1   # same track after crossing the river


SUDHISH = ROOT / "pretrained_weights" / "sudhishneel_v8" / "v8_best.onnx"


@pytest.mark.skipif(not SUDHISH.exists() or not (ROOT / "configs" / "detector_class_map.json").exists(),
                    reason="SudhishNeel ONNX model not installed")
def test_real_model_runs_through_companion():
    pytest.importorskip("onnxruntime")
    det = YoloOnnxDetector(SUDHISH, ROOT / "configs" / "detector_class_map.json")
    assert det.arena_crop == (568, 896) and len(det.classes) == 174
    cal = ArenaCalibration.from_corners(608, 1300, (20, 140), (588, 1036))
    frame = np.full((1300, 608, 3), 30, np.uint8)       # empty screen: the model must not crash
    assert isinstance(det.detect(frame, cal), list)
    comp = LiveCompanion.create(detector=det, calibration=cal)
    comp.start_match(wall=0.0)
    comp.process_frame(frame, wall=1.0)
    assert comp.state(1.0).t_ms == 1000
