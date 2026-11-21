"""Pack what the Colab PPO notebook needs into dist/crbot_ppo_bundle.zip.

    .venv/bin/python tools/pack_ppo_bundle.py            # -> dist/crbot_ppo_bundle.zip (~11 MB, mostly the IL weights)
    .venv/bin/python tools/pack_ppo_bundle.py --code-only   # -> dist/crbot_code_bundle.zip (for the IL notebook)
    .venv/bin/python tools/pack_ppo_bundle.py --il-model crbot_il_v2_hog26   # PPO starting from a deck fine-tune

Contents: the git-tracked files of backend_engine/ and configs/ (the engine, perception, NeuralBot features,
il_model / ppo_trainer, card data, preset decks, reaction profiles), pretrained_weights/<il-model>/ (.onnx +
.npz + .json, the PPO starting point; crbot_il_v2 by default) and BUNDLE_MANIFEST.json (git commit, file list with SHA-256, Python packages the
notebook installs). Untracked files, user data, datasets, recordings and other weights are never included.
Upload the zip to Google Drive as MyDrive/CRbot_Training/crbot_ppo_bundle.zip, then run
cloud_training/train_ppo_colab.ipynb.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import zipfile
from datetime import datetime, timezone
from pathlib import Path
from typing import List

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IL_MODEL = "crbot_il_v2"


def model_files(name: str = DEFAULT_IL_MODEL) -> tuple:
    """pretrained_weights/<name>/<name>.onnx + .npz + .json (the PPO notebook's PPO_INIT)."""
    return tuple(f"pretrained_weights/{name}/{name}.{ext}" for ext in ("onnx", "npz", "json"))


MODEL_FILES = model_files()
TRACKED_DIRS = ("backend_engine", "configs")
PACKAGES = ["torch", "numpy", "pyyaml", "onnx", "onnxruntime", "onnxscript"]


def tracked_files() -> List[str]:
    out = subprocess.run(["git", "ls-files", *TRACKED_DIRS], cwd=ROOT, capture_output=True, text=True, check=True)
    files = [f for f in out.stdout.splitlines() if f and "__pycache__" not in f]
    missing = [f for f in files if not (ROOT / f).exists()]
    if missing:
        raise SystemExit(f"tracked files missing on disk (commit or restore them first): {missing[:5]}")
    return files


def build(out: Path, files: List[str]) -> dict:
    out.parent.mkdir(parents=True, exist_ok=True)
    commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "status", "--porcelain", *TRACKED_DIRS], cwd=ROOT, capture_output=True,
                                text=True).stdout.strip())
    entries = []
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=6) as z:
        for f in files:
            data = (ROOT / f).read_bytes()
            z.writestr(f, data)
            entries.append({"path": f, "bytes": len(data), "sha256": hashlib.sha256(data).hexdigest()})
        manifest = {"bundle": "crbot_ppo_bundle", "created": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                    "git_commit": commit, "uncommitted_changes": dirty, "packages": PACKAGES,
                    "entry": "backend_engine.brain.ppo_trainer", "files": entries}
        z.writestr("BUNDLE_MANIFEST.json", json.dumps(manifest, indent=1))
    return manifest


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--out", default=None, help="default dist/crbot_ppo_bundle.zip (dist/crbot_code_bundle.zip "
                    "with --code-only)")
    ap.add_argument("--code-only", action="store_true",
                    help="no weights: the code the IL notebook imports (model, encoder, augmentation)")
    ap.add_argument("--il-model", default=DEFAULT_IL_MODEL,
                    help="IL model to start PPO from (pretrained_weights/<name>/), e.g. crbot_il_v2_hog26")
    args = ap.parse_args(argv)
    files = tracked_files()
    if not args.code_only:
        name = args.il_model
        for f in model_files(name):
            if not (ROOT / f).exists():
                raise SystemExit(f"missing {f}: put {name}.onnx + .npz + .json (from the IL notebook) in "
                                 f"pretrained_weights/{name}/ first")
        files += list(model_files(name))
    out = Path(args.out or ROOT / "dist" / ("crbot_code_bundle.zip" if args.code_only else "crbot_ppo_bundle.zip"))
    m = build(out, files)
    total = sum(e["bytes"] for e in m["files"])
    print(f"wrote {out} ({out.stat().st_size / 1e6:.1f} MB zipped, {total / 1e6:.1f} MB raw, {len(files)} files, "
          f"commit {m['git_commit']}{' + uncommitted changes' if m['uncommitted_changes'] else ''})")
    print(f"upload it to Google Drive as MyDrive/CRbot_Training/{out.name}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
