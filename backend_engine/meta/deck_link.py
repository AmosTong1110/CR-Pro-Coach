"""Deck link import & export (clashroyale://copyDeck, link.clashroyale.com, RoyaleAPI).

Supports:
1. Official deep links: clashroyale://copyDeck?deck=26000000;26000021;...&slots=1;0;...
2. Official web links: https://link.clashroyale.com/deck/en?deck=...&slots=...
3. RoyaleAPI links: https://royaleapi.com/decks/stats/hog-rider,musketeer,...
4. Plain card lists: "hog_rider, musketeer, cannon, fireball, the_log, skeletons, ice_spirit, ice_golem"
"""
from __future__ import annotations

import re
import urllib.parse
from dataclasses import dataclass
from typing import Collection, Dict, List, Optional, Sequence, Tuple

from backend_engine.meta.archetypes import archetype_name, avg_elixir
from backend_engine.simulator.card_db import CardDB, default_card_db


@dataclass(frozen=True)
class ParsedDeck:
    """Result of parsing a deck link or text representation."""
    cards: Tuple[str, ...]          # Resolved simulated card ids in the deck (length <= 8)
    raw_tokens: Tuple[str, ...]     # Raw tokens or game IDs parsed from the input
    evo: Tuple[str, ...]            # Cards placed in evolution / hero slots
    missing: Tuple[str, ...]        # Tokens that could not be mapped to simulated cards
    archetype: str                  # Detected archetype (e.g. "Hog 2.6")
    avg_elixir: Optional[float]     # Average elixir cost of resolved cards
    playable: bool                  # True if exactly 8 cards and 0 missing
    warning: Optional[str] = None   # Warning message, if any


def parse_deck_link(link_or_text: str, db: Optional[CardDB] = None) -> ParsedDeck:
    """Parse a Clash Royale deck link, RoyaleAPI URL, or card list into a ParsedDeck."""
    db = db or default_card_db()
    text = link_or_text.strip()
    raw_tokens: List[str] = []
    evo_indices: List[int] = []

    # 1. Official clashroyale:// or link.clashroyale.com link
    if "deck=" in text or "copydeck" in text.lower():
        # Clean up any wrapping scheme (e.g. link.clashroyale.com/en/?clashroyale://copyDeck?...)
        qs = text
        if "?" in text:
            qs = text.split("?", 1)[1]
            if "?" in qs:  # nested query
                qs = qs.split("?", 1)[1]
        params = urllib.parse.parse_qs(qs)
        deck_vals = params.get("deck", [""])[0]
        # Semicolon or comma separated
        raw_tokens = [tok.strip() for tok in re.split(r"[;,]", deck_vals) if tok.strip()]
        
        # Check slots param (e.g. slots=1;0;0;0;0;0;0;0 or slots=1;2;0;0;0;0;0;0)
        slots_vals = params.get("slots", [""])[0]
        if slots_vals:
            slot_list = [s.strip() for s in re.split(r"[;,]", slots_vals) if s.strip()]
            for idx, s in enumerate(slot_list):
                if s in ("1", "2") or (s.isdigit() and int(s) > 0):
                    evo_indices.append(idx)

    # 2. RoyaleAPI URL (e.g. https://royaleapi.com/decks/stats/hog-rider,musketeer,...)
    elif "royaleapi.com" in text:
        # Extract path component with card slugs
        path = urllib.parse.urlparse(text).path
        slug_match = re.search(r"/(?:stats|deck-builder)/([a-zA-Z0-9\-_,]+)", path)
        if slug_match:
            raw_tokens = [tok.strip() for tok in slug_match.group(1).split(",") if tok.strip()]
        else:
            raw_tokens = [tok.strip() for tok in re.split(r"[/,]", path) if tok.strip()]

    # 3. Plain text / comma / semicolon / newline separated list
    else:
        raw_tokens = [tok.strip() for tok in re.split(r"[;,\n\r]+", text) if tok.strip()]

    resolved_cards: List[str] = []
    missing_tokens: List[str] = []
    resolved_evo: List[str] = []
    warnings: List[str] = []

    for i, tok in enumerate(raw_tokens):
        cid = _resolve_token(tok, db)
        if cid is not None:
            if cid not in resolved_cards:
                resolved_cards.append(cid)
                if i in evo_indices:
                    if db.variant_simulated(cid, "evo") or db.variant_simulated(cid, "hero"):
                        resolved_evo.append(cid)
                    else:
                        warnings.append(f"Evolution for {db.cards[cid]['name']} is not simulated yet (using base card).")
        else:
            missing_tokens.append(tok)

    # Cap to standard deck size (8)
    cards_tuple = tuple(resolved_cards[:8])
    evo_tuple = tuple(resolved_evo[:2])
    arch = archetype_name(cards_tuple, evo=evo_tuple) if cards_tuple else ""
    avg = avg_elixir(cards_tuple) if cards_tuple else None
    playable = len(cards_tuple) == 8 and len(missing_tokens) == 0

    if len(cards_tuple) < 8 and not missing_tokens:
        warnings.append(f"Deck has only {len(cards_tuple)} cards (8 required).")
    elif missing_tokens:
        warnings.append(f"{len(missing_tokens)} card(s) could not be resolved or are not simulated yet.")

    warn_str = " · ".join(warnings) if warnings else None
    return ParsedDeck(
        cards=cards_tuple,
        raw_tokens=tuple(raw_tokens),
        evo=evo_tuple,
        missing=tuple(missing_tokens),
        archetype=arch,
        avg_elixir=avg,
        playable=playable,
        warning=warn_str,
    )


def _resolve_token(token: str, db: CardDB) -> Optional[str]:
    """Resolve a single token (game_id, alias, slug, or name) to a simulated card ID."""
    tok = token.strip()
    if not tok:
        return None

    # Check numeric game_id
    if tok.isdigit():
        cid = db.by_game_id(tok)
        if cid:
            return cid

    # Direct match in db.cards
    if tok in db.cards:
        return tok

    # Normalized snake_case (e.g. hog-rider -> hog_rider)
    snake = tok.replace("-", "_").lower()
    if snake in db.cards:
        return snake

    # Alias / name match
    return db.by_alias(tok)


def format_copy_deck_link(cards: Sequence[str], evo: Sequence[str] = (),
                          db: Optional[CardDB] = None, web: bool = False) -> str:
    """Generate a clashroyale://copyDeck or link.clashroyale.com URL for an 8-card deck."""
    db = db or default_card_db()
    game_ids = []
    slots = []

    evo_set = {e.split("#")[0] for e in evo}
    evo_counter = 1

    for c in cards:
        info = db.cards.get(c)
        if not info:
            raise ValueError(f"Unknown card {c!r}")
        gid = info.get("game_id")
        if not gid:
            raise ValueError(f"Card {c!r} has no official game_id")
        game_ids.append(str(gid))
        
        # Determine slot
        if c in evo_set and evo_counter <= 2:
            slots.append(str(evo_counter))
            evo_counter += 1
        else:
            slots.append("0")

    deck_str = ";".join(game_ids)
    slots_str = ";".join(slots)
    query = f"deck={deck_str}&slots={slots_str}"

    if web:
        return f"https://link.clashroyale.com/deck/en?{query}"
    return f"clashroyale://copyDeck?{query}"


def format_royaleapi_url(cards: Sequence[str], db: Optional[CardDB] = None) -> str:
    """Generate a RoyaleAPI deck stats link."""
    db = db or default_card_db()
    slugs = []
    for c in cards:
        info = db.cards.get(c, {})
        slug = (info.get("aliases", {}).get("royaleapi") or
                c.replace("_", "-"))
        slugs.append(slug)
    return f"https://royaleapi.com/decks/stats/{','.join(slugs)}"
