"""
Export an Ultralytics YOLOv8 `best.pt` to ONNX and write its class names, for the Mode B companion.

Run it with a Python that has `ultralytics` installed (keep it out of the app's environment):

    uv venv .venv-export --python 3.12 && uv pip install --python .venv-export/bin/python ultralytics onnx onnxslim
    .venv-export/bin/python tools/export_detector.py pretrained_weights/sudhishneel_v8/v8_best.pt --imgsz 800
    python tools/build_class_map.py pretrained_weights/sudhishneel_v8/names.json

Only export weights you trust: a .pt file is a pickle and loading it can run code. Scan it first
(the companion's docs show how: list the pickle's GLOBAL references without loading it).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("weights", type=Path)
    ap.add_argument("--imgsz", type=int, default=800)
    a = ap.parse_args()

    from ultralytics import YOLO

    model = YOLO(str(a.weights))
    names = [model.names[i] for i in range(len(model.names))]
    out = Path(model.export(format="onnx", imgsz=a.imgsz, dynamic=False, simplify=True, opset=17))
    target = a.weights.with_suffix(".onnx")
    if out.resolve() != target.resolve():
        out.replace(target)
    (a.weights.parent / "names.json").write_text(json.dumps(names, indent=0) + "\n", encoding="utf-8")
    print(f"exported {target} ({target.stat().st_size / 1e6:.1f} MB), {len(names)} classes")


if __name__ == "__main__":
    main()
