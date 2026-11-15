"""
Unit/spell detectors. Output is pixel boxes; the companion converts them to tiles.

YoloOnnxDetector works with any YOLOv8 model exported to ONNX
(`yolo export model=best.pt format=onnx`), e.g. a KataCR detector or your own.
It needs a class map (configs/detector_class_map.json, see the .example file):

    {"input_size": 640,
     "side_mode": "class" | "color",          # side encoded in the class, or read from bar colour
     "classes": [{"index": 0, "card": "hog_rider", "side": 1},
                 {"index": 1, "card": "hog_rider", "side": 0},
                 {"index": 7, "card": "fireball",  "side": null}, ...
                 {"index": 9, "card": null}]}  # null card = ignore (towers, effects, ...)

With side_mode "color" the owner is read from the health bar above the box
(red = opponent, blue = you).
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Protocol, Sequence, Tuple

import numpy as np

WEIGHTS_DIR = Path(__file__).resolve().parents[3] / "pretrained_weights"
CLASS_MAP_PATH = Path(__file__).resolve().parents[3] / "configs" / "detector_class_map.json"


@dataclass(frozen=True)
class Detection:
    card_id: str
    side: Optional[int]          # 0 = you (bottom), 1 = opponent, None = unknown
    box: Tuple[float, float, float, float]   # x1, y1, x2, y2 in frame pixels
    conf: float
    evolved: bool = False

    @property
    def foot(self) -> Tuple[float, float]:
        """Ground contact point: bottom-centre of the box (better than the centre for tall sprites)."""
        x1, y1, x2, y2 = self.box
        return (x1 + x2) / 2, y2 - 0.15 * (y2 - y1)


class Detector(Protocol):
    def detect(self, frame: np.ndarray, calibration=None) -> List[Detection]: ...


# ---------------------------------------------------------------------------
def letterbox(frame: np.ndarray, size: int) -> Tuple[np.ndarray, float, Tuple[float, float]]:
    """Resize keeping aspect ratio and pad to size x size. Returns (img, scale, (pad_x, pad_y))."""
    from PIL import Image

    h, w = frame.shape[:2]
    scale = min(size / w, size / h)
    nw, nh = max(1, round(w * scale)), max(1, round(h * scale))
    img = np.asarray(Image.fromarray(frame).resize((nw, nh), Image.BILINEAR))
    out = np.full((size, size, 3), 114, dtype=np.uint8)
    px, py = (size - nw) // 2, (size - nh) // 2
    out[py:py + nh, px:px + nw] = img
    return out, scale, (float(px), float(py))


def nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> List[int]:
    order = scores.argsort()[::-1]
    keep = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        xx1 = np.maximum(boxes[i, 0], boxes[rest, 0])
        yy1 = np.maximum(boxes[i, 1], boxes[rest, 1])
        xx2 = np.minimum(boxes[i, 2], boxes[rest, 2])
        yy2 = np.minimum(boxes[i, 3], boxes[rest, 3])
        inter = np.clip(xx2 - xx1, 0, None) * np.clip(yy2 - yy1, 0, None)
        area = lambda b: (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])  # noqa: E731
        ious = inter / (area(boxes[i]) + area(boxes[rest]) - inter + 1e-9)
        order = rest[ious <= iou]
    return keep


def yolo_postprocess(output: np.ndarray, scale: float, pad: Tuple[float, float],
                     conf_thres: float = 0.4, iou_thres: float = 0.5) -> List[Tuple[int, float, np.ndarray]]:
    """YOLOv8 head (1, 4+nc, N) -> [(class, conf, xyxy in original pixels)], class-wise NMS."""
    pred = output[0].T if output.ndim == 3 else output.T   # (N, 4+nc)
    scores_all = pred[:, 4:]
    cls = scores_all.argmax(1)
    conf = scores_all[np.arange(len(cls)), cls]
    m = conf >= conf_thres
    pred, cls, conf = pred[m], cls[m], conf[m]
    if not len(pred):
        return []
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad[0]) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad[1]) / scale
    out = []
    for c in np.unique(cls):
        idx = np.where(cls == c)[0]
        for k in nms(boxes[idx], conf[idx], iou_thres):
            out.append((int(c), float(conf[idx[k]]), boxes[idx[k]]))
    return out


def side_by_color(frame: np.ndarray, box: Sequence[float]) -> Optional[int]:
    """Look at a strip just above the box (health bar / name tag): red-ish = opponent, blue-ish = you."""
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    bh = max(4, (y2 - y1) // 4)
    ys, ye = max(0, y1 - bh), max(0, min(h, y1 + bh // 2))
    xs, xe = max(0, x1), min(w, x2)
    if ye <= ys or xe <= xs:
        return None
    patch = frame[ys:ye, xs:xe].astype(np.int32)
    r, g, b = patch[..., 0], patch[..., 1], patch[..., 2]
    red = np.sum((r > 150) & (r > b + 60) & (r > g + 40))
    blue = np.sum((b > 150) & (b > r + 60))
    if max(red, blue) < 6:
        return None
    return 1 if red > blue else 0


class YoloOnnxDetector:
    def __init__(self, model_path: Path | str, class_map_path: Path | str = CLASS_MAP_PATH,
                 conf_thres: float = 0.4, iou_thres: float = 0.5, session=None):
        cm = json.loads(Path(class_map_path).read_text(encoding="utf-8"))
        self.size = int(cm.get("input_size", 640))
        self.side_mode = cm.get("side_mode", "class")
        self.classes: Dict[int, Tuple[Optional[str], Optional[int], bool]] = {
            int(c["index"]): (c.get("card"), c.get("side"), bool(c.get("evolved"))) for c in cm["classes"]}
        # models trained on a fixed arena crop (e.g. 568x896): crop the calibrated arena first
        self.arena_crop: Optional[Tuple[int, int]] = tuple(cm["arena_crop"]) if cm.get("arena_crop") else None
        self.units_per_play: Dict[str, int] = cm.get("units_per_play", {})
        self.spawned_by: Dict[str, List[str]] = cm.get("spawned_by", {})
        self.conf_thres, self.iou_thres = conf_thres, iou_thres
        if session is None:
            import onnxruntime as ort  # pip install onnxruntime

            providers = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider")
                         if p in ort.get_available_providers()]
            session = ort.InferenceSession(str(model_path), providers=providers)
        self.session = session
        self.input_name = session.get_inputs()[0].name

    def _arena_view(self, frame: np.ndarray, calibration) -> Tuple[np.ndarray, Tuple[float, float, float, float]]:
        """Crop the calibrated arena and resize to the model's crop size.
        Returns (image, (offset_x, offset_y, scale_x, scale_y)) to map boxes back to frame pixels."""
        from PIL import Image

        h, w = frame.shape[:2]
        l, t, r, b = calibration._scaled(w, h)
        l, t = max(0, int(round(l))), max(0, int(round(t)))
        r, b = min(w, int(round(r))), min(h, int(round(b)))
        cw, ch = self.arena_crop
        img = np.asarray(Image.fromarray(frame[t:b, l:r]).resize((cw, ch), Image.BILINEAR))
        return img, (l, t, (r - l) / cw, (b - t) / ch)

    def detect(self, frame: np.ndarray, calibration=None) -> List[Detection]:
        view, (ox, oy, sx, sy) = frame, (0.0, 0.0, 1.0, 1.0)
        if self.arena_crop and calibration is not None:
            view, (ox, oy, sx, sy) = self._arena_view(frame, calibration)
        img, scale, pad = letterbox(view, self.size)
        x = img.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        out = self.session.run(None, {self.input_name: x})[0]
        dets = []
        for c, conf, box in yolo_postprocess(out, scale, pad, self.conf_thres, self.iou_thres):
            card, side, evolved = self.classes.get(c, (None, None, False))
            if card is None:
                continue
            fbox = (ox + box[0] * sx, oy + box[1] * sy, ox + box[2] * sx, oy + box[3] * sy)
            # side_mode "position": leave side unknown; the play tracker assigns it from where the
            # unit is first deployed (more reliable than colour when no health bar is shown)
            if side is None and self.side_mode == "color":
                side = side_by_color(frame, fbox)
            dets.append(Detection(card, side, tuple(float(v) for v in fbox), conf, evolved))
        return dets
