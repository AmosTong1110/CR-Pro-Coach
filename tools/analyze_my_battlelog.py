"""
Personal Battlelog Matchup Analyzer for #VLYCGJGQR

Parses player battlelogs (official RoyaleAPI / Supercell format),
computes win rates by opponent archetype, identifies tough matchups,
and registers missing tough opponent decks (compatible with our card pool)
into configs/preset_decks.json for targeted bot sparring.

Usage:
    python tools/analyze_my_battlelog.py [--input my_battlelog.json] [--tag #VLYCGJGQR]
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

# Ensure repository root is in sys.path
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from backend_engine.meta.archetypes import archetype_name
from backend_engine.simulator.card_db import CardDB

DEFAULT_TAG = "#VLYCGJGQR"
DEFAULT_BATTLELOG = REPO_ROOT / "my_battlelog.json"
PRESET_DECKS_PATH = REPO_ROOT / "configs" / "preset_decks.json"


def normalize_card_id(raw_name: str, db: CardDB) -> Optional[str]:
    """Resolve an official card name or id to our internal simulator card_id."""
    if not raw_name:
        return None
    # 1. CardDB alias lookup
    cid = db.by_alias(raw_name)
    if cid:
        return cid
    # 2. Slug / formatting fallback
    slug = raw_name.strip().lower().replace(" ", "_").replace(".", "").replace("-", "_")
    if slug in db.cards:
        return slug
    return None


def extract_deck(player_data: Dict[str, Any], db: CardDB) -> Tuple[List[str], List[str]]:
    """Extract (resolved_cards, missing_cards) from a battlelog player entry."""
    cards_list = player_data.get("cards", [])
    resolved = []
    missing = []
    for c in cards_list:
        raw_name = c.get("name", "")
        cid = normalize_card_id(raw_name, db)
        if cid:
            resolved.append(cid)
        else:
            missing.append(raw_name)
    return resolved, missing


def parse_battlelog(
    path: Path,
    player_tag: str,
    db: CardDB,
) -> List[Dict[str, Any]]:
    """Parse battlelog JSON and return a list of parsed match records."""
    if not path.exists():
        return []

    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "battles" in data:
        battles = data["battles"]
    elif isinstance(data, list):
        battles = data
    else:
        return []

    norm_tag = player_tag.lstrip("#").upper()
    parsed_matches = []

    for b in battles:
        team = b.get("team", [])
        opponents = b.get("opponent", [])
        if not team or not opponents:
            continue

        # Find user's side
        user_entry = None
        opp_entry = None

        for p in team:
            if p.get("tag", "").lstrip("#").upper() == norm_tag:
                user_entry = p
                opp_entry = opponents[0]
                break

        if user_entry is None:
            # Check if user was recorded on the opponent side
            for p in opponents:
                if p.get("tag", "").lstrip("#").upper() == norm_tag:
                    user_entry = p
                    opp_entry = team[0]
                    break

        # Fallback to team[0] vs opponent[0] if single 1v1 battle
        if user_entry is None and len(team) == 1 and len(opponents) == 1:
            user_entry = team[0]
            opp_entry = opponents[0]

        if not user_entry or not opp_entry:
            continue

        user_crowns = user_entry.get("crowns", 0)
        opp_crowns = opp_entry.get("crowns", 0)

        if user_crowns > opp_crowns:
            result = "WIN"
        elif user_crowns < opp_crowns:
            result = "LOSS"
        else:
            result = "DRAW"

        user_deck, _ = extract_deck(user_entry, db)
        opp_deck, opp_missing = extract_deck(opp_entry, db)

        opp_arch = archetype_name(opp_deck) if opp_deck else "Unknown"
        user_arch = archetype_name(user_deck) if user_deck else "Unknown"

        parsed_matches.append({
            "battle_time": b.get("battleTime", ""),
            "game_mode": b.get("gameMode", {}).get("name", "1v1"),
            "result": result,
            "user_crowns": user_crowns,
            "opp_crowns": opp_crowns,
            "user_deck": user_deck,
            "user_archetype": user_arch,
            "opp_deck": opp_deck,
            "opp_missing": opp_missing,
            "opp_archetype": opp_arch,
            "opp_name": opp_entry.get("name", "Opponent"),
            "opp_tag": opp_entry.get("tag", ""),
        })

    return parsed_matches


def summarize_matchups(matches: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Aggregate stats by opponent archetype."""
    stats: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "matches": 0, "wins": 0, "losses": 0, "draws": 0,
        "crowns_for": 0, "crowns_against": 0, "sample_decks": []
    })

    for m in matches:
        arch = m["opp_archetype"]
        s = stats[arch]
        s["matches"] += 1
        if m["result"] == "WIN":
            s["wins"] += 1
        elif m["result"] == "LOSS":
            s["losses"] += 1
        else:
            s["draws"] += 1
        s["crowns_for"] += m["user_crowns"]
        s["crowns_against"] += m["opp_crowns"]
        if len(m["opp_deck"]) == 8 and not m["opp_missing"]:
            if m["opp_deck"] not in s["sample_decks"]:
                s["sample_decks"].append(m["opp_deck"])

    # Calculate win rates
    for arch, s in stats.items():
        s["win_rate"] = (s["wins"] / s["matches"]) * 100.0 if s["matches"] > 0 else 0.0

    return dict(stats)


def register_tough_matchups(
    stats: Dict[str, Dict[str, Any]],
    presets_path: Path,
    db: CardDB,
    win_rate_threshold: float = 50.0,
) -> List[Dict[str, Any]]:
    """Register tough opponent decks (win_rate < threshold) into preset_decks.json."""
    if not presets_path.exists():
        presets_data = {"about": "Ready-made decks", "decks": []}
    else:
        presets_data = json.loads(presets_path.read_text(encoding="utf-8"))

    existing_decks = {tuple(sorted(d["cards"])) for d in presets_data.get("decks", [])}
    new_presets = []

    for arch, s in stats.items():
        # Tough matchup: win rate < threshold and at least 1 loss
        if s["win_rate"] < win_rate_threshold and s["losses"] > 0:
            for deck in s["sample_decks"]:
                sorted_deck = sorted(deck)
                # Verify all cards are supported in simulator
                if len(sorted_deck) == 8 and all(c in db.cards for c in sorted_deck):
                    if tuple(sorted_deck) not in existing_decks:
                        entry = {
                            "name": f"Sparring: {arch}",
                            "style": "Tough Matchup",
                            "evo": [],
                            "cards": sorted_deck,
                            "tip": f"Targeted sparring deck from battlelog ({arch}, {s['win_rate']:.0f}% win rate)."
                        }
                        presets_data["decks"].append(entry)
                        existing_decks.add(tuple(sorted_deck))
                        new_presets.append(entry)

    if new_presets:
        presets_path.write_text(json.dumps(presets_data, indent=2) + "\n", encoding="utf-8")

    return new_presets


def generate_sample_battlelog(out_path: Path, player_tag: str = DEFAULT_TAG) -> None:
    """Generate a realistic battlelog sample for testing/demonstration."""
    sample = [
        {
            "battleTime": "20261117T080000.000Z",
            "gameMode": {"name": "Ladder 1v1"},
            "team": [{
                "tag": player_tag,
                "name": "ProPlayer",
                "crowns": 1,
                "cards": [
                    {"name": "Hog Rider"}, {"name": "Musketeer"}, {"name": "Knight"}, {"name": "Cannon"},
                    {"name": "Fireball"}, {"name": "The Log"}, {"name": "Skeletons"}, {"name": "Ice Spirit"}
                ]
            }],
            "opponent": [{
                "tag": "#OPP111",
                "name": "PekkaMaster",
                "crowns": 2,
                "cards": [
                    {"name": "P.E.K.K.A"}, {"name": "Dark Prince"}, {"name": "Zap"}, {"name": "Fireball"},
                    {"name": "Minions"}, {"name": "Archers"}, {"name": "Knight"}, {"name": "Cannon"}
                ]
            }]
        },
        {
            "battleTime": "20261117T081500.000Z",
            "gameMode": {"name": "Ladder 1v1"},
            "team": [{
                "tag": player_tag,
                "name": "ProPlayer",
                "crowns": 2,
                "cards": [
                    {"name": "Hog Rider"}, {"name": "Musketeer"}, {"name": "Knight"}, {"name": "Cannon"},
                    {"name": "Fireball"}, {"name": "The Log"}, {"name": "Skeletons"}, {"name": "Ice Spirit"}
                ]
            }],
            "opponent": [{
                "tag": "#OPP222",
                "name": "GiantBeatdown",
                "crowns": 0,
                "cards": [
                    {"name": "Giant"}, {"name": "Musketeer"}, {"name": "Mini P.E.K.K.A"}, {"name": "Fireball"},
                    {"name": "Zap"}, {"name": "Minions"}, {"name": "Archers"}, {"name": "Knight"}
                ]
            }]
        },
        {
            "battleTime": "20261117T083000.000Z",
            "gameMode": {"name": "Ladder 1v1"},
            "team": [{
                "tag": player_tag,
                "name": "ProPlayer",
                "crowns": 0,
                "cards": [
                    {"name": "Hog Rider"}, {"name": "Musketeer"}, {"name": "Knight"}, {"name": "Cannon"},
                    {"name": "Fireball"}, {"name": "The Log"}, {"name": "Skeletons"}, {"name": "Ice Spirit"}
                ]
            }],
            "opponent": [{
                "tag": "#OPP333",
                "name": "SiegeGod",
                "crowns": 1,
                "cards": [
                    {"name": "X-Bow"}, {"name": "Tesla"}, {"name": "Archers"}, {"name": "Knight"},
                    {"name": "Fireball"}, {"name": "The Log"}, {"name": "Skeletons"}, {"name": "Ice Spirit"}
                ]
            }]
        }
    ]
    out_path.write_text(json.dumps(sample, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze personal battlelog and register tough matchups for sparring.")
    parser.add_argument("--input", "-i", type=Path, default=DEFAULT_BATTLELOG, help="Path to battlelog JSON")
    parser.add_argument("--tag", "-t", type=str, default=DEFAULT_TAG, help="Target player tag (e.g. #VLYCGJGQR)")
    parser.add_argument("--presets", "-p", type=Path, default=PRESET_DECKS_PATH, help="Path to preset_decks.json")
    parser.add_argument("--generate-sample", action="store_true", help="Generate sample battlelog if input is missing")
    args = parser.parse_args()

    if not args.input.exists():
        print(f"Notice: Battlelog file '{args.input}' not found. Generating sample battlelog for {args.tag}...")
        generate_sample_battlelog(args.input, args.tag)

    db = CardDB.load()
    matches = parse_battlelog(args.input, args.tag, db)
    if not matches:
        print(f"No matches found for player {args.tag} in {args.input}.")
        return

    stats = summarize_matchups(matches)
    new_registered = register_tough_matchups(stats, args.presets, db)

    total_matches = len(matches)
    total_wins = sum(1 for m in matches if m["result"] == "WIN")
    total_losses = sum(1 for m in matches if m["result"] == "LOSS")
    total_draws = sum(1 for m in matches if m["result"] == "DRAW")
    win_rate = (total_wins / total_matches) * 100.0 if total_matches > 0 else 0.0

    print("=" * 72)
    print(f"📊 PERSONAL BATTLELOG ANALYSIS FOR {args.tag}")
    print("=" * 72)
    print(f"Overall Record: {total_wins}W - {total_losses}L - {total_draws}D  |  Win Rate: {win_rate:.1f}%  |  Matches: {total_matches}")
    print("-" * 72)
    print(f"{'Opponent Archetype':<26} {'Matches':<9} {'Record (W-L-D)':<16} {'Win Rate':<10} {'Status'}")
    print("-" * 72)

    # Sort archetypes by win rate ascending (toughest first)
    for arch, s in sorted(stats.items(), key=lambda x: (x[1]["win_rate"], -x[1]["matches"])):
        rec_str = f"{s['wins']}-{s['losses']}-{s['draws']}"
        status = "⚠️ TOUGH" if s["win_rate"] < 50.0 else "✅ FAVORABLE"
        print(f"{arch:<26} {s['matches']:<9} {rec_str:<16} {s['win_rate']:>5.1f}%    {status}")

    print("-" * 72)
    if new_registered:
        print(f"🎯 Registered {len(new_registered)} tough matchup deck(s) into {args.presets.name} for sparring:")
        for reg in new_registered:
            cards_str = ", ".join(reg["cards"])
            print(f"   • {reg['name']} -> [{cards_str}]")
    else:
        print("🎯 No new tough sparring decks needed (already registered or non-standard card pool).")
    print("=" * 72)


if __name__ == "__main__":
    main()
