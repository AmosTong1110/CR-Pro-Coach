"""
Pretrained Weights Downloader for CR-Pro-Coach.
Downloads ready-to-use models from Hugging Face & GitHub:
- Cochon123/clash-royale-policy-bc-v4
- Cochon123/clash-royale-action-clock-v1
- Cochon123/clash-royale-winner-predictor
- wty-yy/KataCR (YOLOv8 ONNX)
- Pbatch/ClashRoyaleBuildABot (ONNX Vision Engine)
"""
import os
import sys
import urllib.request
from pathlib import Path

DEST_DIR = Path(__file__).resolve().parent

MODELS = {
    "action_clock_v1.pt": {
        "repo": "Cochon123/clash-royale-action-clock-v1",
        "file": "action_clock_v1.pt",
        "fallback_url": "https://huggingface.co/Cochon123/clash-royale-action-clock-v1/resolve/main/action_clock_v1.pt",
        "size_hint": "~1.5 MB"
    },
    "policy_bc_v4.pt": {
        "repo": "Cochon123/clash-royale-policy-bc-v4",
        "file": "best_model.pt",
        "fallback_url": "https://huggingface.co/Cochon123/clash-royale-policy-bc-v4/resolve/main/best_model.pt",
        "size_hint": "2.27 MB"
    },
    "winner_predictor.pt": {
        "repo": "Cochon123/clash-royale-winner-predictor",
        "file": "winner_predictor.pt",
        "fallback_url": "https://huggingface.co/Cochon123/clash-royale-winner-predictor/resolve/main/winner_predictor.pt",
        "size_hint": "~3.8 MB"
    },
    "katacr_yolov8_detector.onnx": {
        "repo": "wty-yy/KataCR",
        "file": "weights/yolov8_clash_royale.onnx",
        "fallback_url": "https://github.com/wty-yy/KataCR/raw/main/weights/yolov8_clash_royale.onnx",
        "size_hint": "~12 MB"
    }
}


def download_file(url: str, dest_path: Path, desc: str):
    print(f"[*] Downloading {desc} to {dest_path.name}...")
    try:
        def progress(count, block_size, total_size):
            pct = int(count * block_size * 100 / max(total_size, 1))
            sys.stdout.write(f"\r    Progress: {min(pct, 100)}%")
            sys.stdout.flush()

        urllib.request.urlretrieve(url, str(dest_path), reporthook=progress)
        print(f"\n[+] Successfully saved {dest_path.name} ({dest_path.stat().st_size} bytes)")
    except Exception as e:
        print(f"\n[-] Direct download failed: {e}")
        print(f"[!] You can manually download from {url} or clone via HuggingFace CLI.")


def main():
    print("=" * 60)
    print("CR-Pro-Coach Pretrained Models Downloader")
    print("=" * 60)
    for filename, info in MODELS.items():
        target = DEST_DIR / filename
        if target.exists() and target.stat().st_size > 1024:
            print(f"[✓] {filename} already exists ({target.stat().st_size} bytes). Skipping.")
            continue
        download_file(info["fallback_url"], target, f"{filename} ({info['size_hint']})")


if __name__ == "__main__":
    main()
