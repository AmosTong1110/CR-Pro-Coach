"""Deck meta: archetype names, deck recommendations, presets, and deck link import/export."""

from backend_engine.meta.archetypes import archetype_name, avg_elixir
from backend_engine.meta.deck_link import (
    ParsedDeck,
    format_copy_deck_link,
    format_royaleapi_url,
    parse_deck_link,
)
from backend_engine.meta.presets import AI_PUSH_CARDS, PresetDeck, find_preset, load_presets, playable
from backend_engine.meta.recommend import coverage, meta_info, recommend

__all__ = [
    "archetype_name",
    "avg_elixir",
    "load_presets",
    "playable",
    "find_preset",
    "PresetDeck",
    "AI_PUSH_CARDS",
    "recommend",
    "coverage",
    "meta_info",
    "ParsedDeck",
    "parse_deck_link",
    "format_copy_deck_link",
    "format_royaleapi_url",
]
