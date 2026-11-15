"""
Measure an ONNX detector on labelled 568x896 arena frames (e.g. SudhishNeel synthetic_arena rows).

    python tools/eval_detector.py pretrained_weights/sudhishneel_v8/v8_best.onnx frames_dir/

frames_dir/labels.json: [{"file": ..., "bbox": [[x0,y0,x1,y1], ...], "category": [class_id, ...]}, ...]
Class ids are the dataset's permanent ids, which match the model's class order.
Reports precision / recall at IoU >= 0.5 (class must match), and inference speed on this Mac.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend_engine.perception.live.detector import letterbox, yolo_postprocess  # noqa: E402


def iou(a, b) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("model", type=Path)
    ap.add_argument("frames", type=Path)
    ap.add_argument("--imgsz", type=int, default=800)
    ap.add_argument("--conf", type=float, default=0.4)
    a = ap.parse_args()

    import onnxruntime as ort
    from PIL import Image

    providers = [p for p in ("CoreMLExecutionProvider", "CPUExecutionProvider") if p in ort.get_available_providers()]
    sess = ort.InferenceSession(str(a.model), providers=providers)
    name = sess.get_inputs()[0].name
    labels = json.loads((a.frames / "labels.json").read_text())
    names_path = a.model.parent / "names.json"
    names = json.loads(names_path.read_text()) if names_path.exists() else None
    nc = len(names) if names else None

    tp = fp = fn = skipped = 0
    times = []
    for item in labels:
        frame = np.asarray(Image.open(a.frames / Path(item["file"]).name).convert("RGB"))
        img, scale, pad = letterbox(frame, a.imgsz)
        x = img.astype(np.float32).transpose(2, 0, 1)[None] / 255.0
        t0 = time.perf_counter()
        out = sess.run(None, {name: x})[0]
        times.append(time.perf_counter() - t0)
        preds = yolo_postprocess(out, scale, pad, conf_thres=a.conf)
        gt = [(c, b) for c, b in zip(item["category"], item["bbox"]) if nc is None or c < nc]
        skipped += len(item["category"]) - len(gt)   # classes this model version doesn't have
        used = set()
        for c, _, box in sorted(preds, key=lambda p: -p[1]):
            hit = next((i for i, (gc, gb) in enumerate(gt) if i not in used and gc == c and iou(box, gb) >= 0.5), None)
            if hit is None:
                fp += 1
            else:
                used.add(hit)
                tp += 1
        fn += len(gt) - len(used)

    p = tp / max(1, tp + fp)
    r = tp / max(1, tp + fn)
    warm = times[1:] or times
    print(f"frames {len(labels)} | boxes scored {tp + fn} (+{skipped} of classes the model lacks)")
    print(f"precision {p:.1%}  recall {r:.1%}  (IoU>=0.5, class must match, conf>={a.conf})")
    print(f"speed {1000 * sum(warm) / len(warm):.0f} ms/frame on {sess.get_providers()[0]}")


if __name__ == "__main__":
    main()
