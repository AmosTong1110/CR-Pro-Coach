"""
Build the deck prior for the Bayesian deck guesser from real ladder replays.

Source: Hugging Face dataset Cochon123/clash-royale-replays (MIT),
        cleaned-legacy/matches.jsonl (December 2025, top-ladder battles).
Only the 8-card decks are kept. Player tags and names are dropped.

    python tools/build_meta_decks.py path/to/matches.jsonl
"""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "configs" / "meta_decks_hf.json"


def slug_to_id(slug: str) -> str:
    return slug.strip().lower().replace("-", "_")


def build(src: Path, out: Path, source_name: str, as_of: str) -> dict:
    decks: Counter = Counter()
    battles = 0
    for line in src.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        rec = json.loads(line)
        battles += 1
        for side in ("team", "opponent"):
            cards = rec.get("decks", {}).get(side) or []
            ids = tuple(sorted({slug_to_id(c) for c in cards}))
            if len(ids) == 8:
                decks[ids] += 1
    card_counts: Counter = Counter()
    for d, n in decks.items():
        for c in d:
            card_counts[c] += n
    data = {
        "source": source_name,
        "as_of": as_of,
        "battles": battles,
        "deck_observations": sum(decks.values()),
        "unique_decks": len(decks),
        "card_usage": dict(card_counts.most_common()),
        "decks": [{"cards": list(d), "count": n} for d, n in decks.most_common()],
    }
    out.write_text(json.dumps(data, indent=1) + "\n", encoding="utf-8")
    return data


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("matches_jsonl", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--source", default="huggingface:Cochon123/clash-royale-replays/cleaned-legacy/matches.jsonl")
    ap.add_argument("--as-of", default="2025-12")
    a = ap.parse_args()
    d = build(a.matches_jsonl, a.out, a.source, a.as_of)
    print(f"{d['battles']} battles -> {d['deck_observations']} decks ({d['unique_decks']} unique), "
          f"{len(d['card_usage'])} cards -> {a.out}")
    print("top cards:", list(d["card_usage"].items())[:12])


if __name__ == "__main__":
    main()
