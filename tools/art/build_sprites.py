"""Build CRbot's sprite assets: download CC0 model packs, render them in Blender, pack sprite sheets + manifest.

    .venv/bin/python tools/art/build_sprites.py            # everything (download if missing, render, pack)
    .venv/bin/python tools/art/build_sprites.py --only knight archer
    BLENDER=/path/to/Blender .venv/bin/python tools/art/build_sprites.py

Raw models go to art_src/ (git-ignored). Output: frontend_app/qt/assets/sprites/ (committed, so the app runs
without Blender): arena.png, tower_*.png, castle_*.png, one sheet per character and side, manifest.json.
Only CC0 packs are accepted (the pack's LICENSE.txt must say Creative Commons Zero); see CREDITS.md.
"""
from __future__ import annotations

import argparse
import io
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC = ROOT / "art_src"
OUT = ROOT / "frontend_app" / "qt" / "assets" / "sprites"
BLENDER = os.environ.get("BLENDER", "/Applications/Blender.app/Contents/MacOS/Blender")
RENDER = Path(__file__).with_name("blender_render.py")

PACKS = {   # folder -> GitHub repo of the official KayKit organisation (CC0, Kay Lousberg)
    "adventurers": "KayKit-Game-Assets/KayKit-Character-Pack-Adventures-1.0",
    "skeletons": "KayKit-Game-Assets/KayKit-Character-Pack-Skeletons-1.0",
    "medieval": "KayKit-Game-Assets/KayKit-Medieval-Hexagon-Pack-1.0",
}
ADV = "adventurers/addons/kaykit_character_pack_adventures"
SKL = "skeletons/addons/kaykit_character_pack_skeletons"
MED = "medieval/addons/kaykit_medieval_hexagon_pack/Assets/gltf"

SIDES = {"blue": [0.12, 0.38, 1.0], "red": [1.0, 0.16, 0.12]}
ACTIONS = {"idle": ["Idle", 4], "walk": ["Walking_A", 8], "death": ["Death_A", 6]}
CLOTH = ["Cape", "Cloak", "Hood", "Hat"]

CHARACTERS = {
    "knight": {"model": f"{ADV}/Characters/gltf/Knight.glb", "height": 2.0,
               "keep": ["Knight_ArmLeft", "Knight_ArmRight", "Knight_Body", "Knight_Cape", "Knight_Head",
                        "Knight_Helmet", "Knight_LegLeft", "Knight_LegRight", "1H_Sword", "Round_Shield"],
               "attack": ["1H_Melee_Attack_Chop", 8]},
    "archer": {"model": f"{ADV}/Characters/gltf/Rogue.glb", "height": 1.9,
               "keep": ["Rogue_ArmLeft", "Rogue_ArmRight", "Rogue_Body", "Rogue_Cape", "Rogue_Head", "Rogue_LegLeft",
                        "Rogue_LegRight", "2H_Crossbow"],
               "attack": ["2H_Ranged_Shoot", 8]},
    "skeleton": {"model": f"{SKL}/Characters/gltf/Skeleton_Minion.glb", "height": 1.9,
                 "attach": [[f"{SKL}/Assets/gltf/Skeleton_Blade.gltf", "handslot.r"]],
                 "attack": ["1H_Melee_Attack_Chop", 8]},
}
# card id -> character sprite (cards not listed keep CRbot's vector tokens)
CARDS = {"knight": "knight", "archers": "archer", "skeletons": "skeleton", "skeleton_army": "skeleton"}
# per-card display scale (swarm units are smaller)
CARD_SCALE = {"skeletons": 0.8, "skeleton_army": 0.75, "archers": 0.95}
TOWERS = {"tower": ("building_tower_A_{c}.gltf", 3.0, 0.55), "castle": ("building_castle_{c}.gltf", 4.2, 0.6)}


def fetch() -> None:
    """Download each pack once (GitHub zip), extract inside art_src/<pack>/ and check its licence is CC0."""
    for name, repo in PACKS.items():
        dest = SRC / name
        if dest.exists():
            continue
        url = f"https://codeload.github.com/{repo}/zip/refs/heads/main"
        print("download", url)
        data = urllib.request.urlopen(url, timeout=120).read()
        with zipfile.ZipFile(io.BytesIO(data)) as zf:
            dest.mkdir(parents=True)
            for info in zf.infolist():
                parts = Path(info.filename).parts[1:]            # drop the "<repo>-main/" folder
                if not parts or info.is_dir():
                    continue
                target = (dest / Path(*parts)).resolve()
                if dest.resolve() not in target.parents:          # no path traversal
                    raise RuntimeError(f"unsafe path in zip: {info.filename}")
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_bytes(zf.read(info))
    for name in PACKS:
        lic = list((SRC / name).rglob("LICENSE.txt"))
        text = lic[0].read_text(errors="ignore") if lic else ""
        if "Creative Commons Zero" not in text:
            raise RuntimeError(f"{name}: licence is not CC0 ({lic[:1]}); refusing to use it")


def blender(job: dict, tmp: Path) -> None:
    path = tmp / f"job_{job.get('name', job['mode'])}_{len(list(tmp.iterdir()))}.json"
    path.write_text(json.dumps(job))
    cmd = [BLENDER, "-b", "--factory-startup", "--disable-autoexec", "--python", str(RENDER), "--", str(path)]
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0 or "Traceback" in res.stderr + res.stdout:
        print(res.stdout[-3000:], res.stderr[-3000:])
        raise RuntimeError(f"blender failed for {path.name}")


def to_jpeg(png: Path, quality: int = 88) -> str:
    """The arena has no transparency: store it as JPEG (~5x smaller than PNG). Returns the new file name."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QGuiApplication, QImage
    _app = QGuiApplication.instance() or QGuiApplication([])
    jpg = png.with_suffix(".jpg")
    QImage(str(png)).convertToFormat(QImage.Format_RGB32).save(str(jpg), "JPG", quality)
    png.unlink()
    return jpg.name


def pack(frames_dir: Path, name: str) -> dict:
    """Frames -> one sheet per side (grid of equal cells) + the cell indices per action."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtGui import QGuiApplication, QImage, QPainter
    _app = QGuiApplication.instance() or QGuiApplication([])
    meta = json.loads((frames_dir / f"{name}.json").read_text())
    cell, cols = meta["cell"], 8
    entry = {"cell": cell, "cols": cols, "feet": meta["feet"], "sheets": {}, "actions": {}}
    for side in SIDES:
        keys = [k for k in meta["frames"] if k.startswith(side + "/")]
        files = [f for k in keys for f in meta["frames"][k]]
        rows = (len(files) + cols - 1) // cols
        sheet = QImage(cols * cell, rows * cell, QImage.Format_ARGB32)
        sheet.fill(0)
        p = QPainter(sheet)
        idx = 0
        for k in keys:
            cells = []
            for f in meta["frames"][k]:
                p.drawImage((idx % cols) * cell, (idx // cols) * cell, QImage(str(frames_dir / f)))
                cells.append(idx)
                idx += 1
            entry["actions"][k.split("/", 1)[1]] = cells          # same layout for both sides
        p.end()
        fn = f"{name}_{side}.png"
        sheet.save(str(OUT / fn), "PNG", 9)
        entry["sheets"][side] = fn
    return entry


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*", help="characters to (re)build; default all + arena + towers")
    args = ap.parse_args(argv)
    fetch()
    OUT.mkdir(parents=True, exist_ok=True)
    man_path = OUT / "manifest.json"
    manifest = json.loads(man_path.read_text()) if man_path.exists() else {"characters": {}}
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        if not args.only:
            blender({"mode": "arena", "root": str(SRC / MED), "out": str(OUT / "arena.png")}, tmp)
            manifest["arena"] = {"file": to_jpeg(OUT / "arena.png"), **json.loads((OUT / "arena.json").read_text())}
            (OUT / "arena.json").unlink()
            manifest["towers"] = {}
            for kind, (fn, foot, squash) in TOWERS.items():
                for side, c in (("blue", "blue"), ("red", "red")):
                    out = OUT / f"{kind}_{side}.png"
                    blender({"mode": "static", "model": str(SRC / MED / "buildings" / c / fn.format(c=c)),
                             "out": str(out), "footprint": foot, "squash": squash}, tmp)
                    meta = json.loads(out.with_suffix(".json").read_text())
                    out.with_suffix(".json").unlink()
                    manifest["towers"][f"{kind}_{side}"] = {"file": out.name, **meta}
        for name in args.only or CHARACTERS:
            spec = CHARACTERS[name]
            frames = tmp / name
            job = {"mode": "char", "name": name, "model": str(SRC / spec["model"]), "out_dir": str(frames),
                   "keep": spec.get("keep"), "height": spec["height"], "frame_tiles": 3.0,
                   "attach": [[str(SRC / a), b] for a, b in spec.get("attach", [])], "tint_parts": CLOTH,
                   "sides": SIDES, "actions": {**ACTIONS, "attack": spec["attack"]}}
            print("render", name)
            blender(job, tmp)
            manifest["characters"][name] = pack(frames, name)
    manifest.update({"version": 1, "px_per_tile": 64, "tilt": 35.0, "cards": CARDS, "card_scale": CARD_SCALE,
                     "credits": "KayKit by Kay Lousberg (CC0) - see CREDITS.md"})
    man_path.write_text(json.dumps(manifest, indent=1))
    print("wrote", man_path, sum(f.stat().st_size for f in OUT.iterdir()) // 1024, "KB")


if __name__ == "__main__":
    sys.exit(main())
