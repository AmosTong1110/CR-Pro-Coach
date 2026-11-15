# CRbot — Session Handoff (2026-11-15)

Read this first in every new session. Roadmap: `docs/CRBOT_UPGRADE_PLAN.md`.
State at handoff: **313 tests passing** (all green) (`QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q`), engine **v2**, all work merged into `main`.
Environment: project venv `.venv` (Python 3.14; numpy, pyyaml, PySide6-Essentials, onnxruntime, Pillow, pytest).
Homebrew's `python3` has no packages, so always use `.venv/bin/python`. Run the app: `.venv/bin/python -m frontend_app.qt.app`.

## 1. Completed work

### Simulator (Mode A)
| file | what it does |
|---|---|
| `backend_engine/contracts.py` | Shared dataclasses: `PublicDeployEvent`, `ElixirEstimate`, `ActionIntent`, `HoverPlan`, `ScheduledAction`, `TilePos`. All time is integer **sim ms**. |
| `backend_engine/simulator/geometry.py` | 18×32 tile arena, river/bridges, tower layout, `deploy_check` (own half, pocket after a princess falls, spells anywhere), bridge pathing. |
| `backend_engine/simulator/engine.py` | Deterministic 50 ms tick engine: elixir phases (x1/x2/OT x2→x3), 4+4 card cycle, evo charge counter, towers + king activation on any damage, targeting, attacks, splash, spells (travel + area), rolling Log, building decay, **mass-based soft collision** (river-safe), win/crowns/tiebreaker, `snapshot/restore`, `state_hash`, `Match.replay`, fog-of-war `observation()`. |
| `backend_engine/simulator/card_db.py` | **Schema v4** loader (still reads v3 in memory via `normalise_card`). **Exact official scaling**: `stat(L)=floor(base·M[L-1-rarity_offset]/100)` (verified on 4,945 values). Helpers: `variant`, `has_variant`, `evo_cycles`, `evo_capable`, `mechanics_for(cid, variant)` (explicit + derived `projectile`/`splash`/`decay`/spell kind), `spell_kind`, `by_alias`, `unsimulated_tags`; `stats(cid, level, variant)` applies `stat_mods`. **Tokens** (2026-11-15): a top-level `tokens` section holds units that a card spawns (Barbarian from the Barbarian Barrel); `CardDB.tokens`, `entry(id)` finds a card or token, and a token's `rarity` is its parent card's (level scaling) and its `elixir` the parent's cost (value model). Tokens are never in `db.cards`, so decks, the guesser and the coach can't see them. |
| `backend_engine/simulator/sync_meta.py` | Imports RoyaleAPI-format game data into `configs/cards_stats_v2026.json` (base stats, timings, ranges, speed, mass). |
| `backend_engine/simulator/mechanics/` | **Mechanics framework** (plan §5). **Batch 1 (2026-11-15)**: `effects.py` adds `suicide` (kamikaze hit, optional `status` payload and `chain` {count, radius_tiles}; Ice/Fire/Electro Spirit) and `death_damage` (burst on death, scaled `death_damage` stat, optional status; Ice Golem). `splash` takes `{"center": "self"}` (Valkyrie); area spells apply an `apply_status` payload (Zap). Slow now also slows attack speed (`status.cooldown_dt`); `state_hash` includes statuses only once one exists, so older hashes are unchanged. `base.py`: `Mechanic` hooks (`on_spawn`, `on_tick`, `on_attack`, `on_damaged`, `on_death`, `waypoint`, `river_passable`), `REGISTRY` + `@register`, `SpellKind` + `SPELLS`, `DESCRIPTIVE` tags. `core.py`: `river_jump`, `splash`, `decay`, `area_spell`, `rolling_spell` (ported, identical hashes) + engine v2: `projectile` (homing, travel time; Musketeer/Archers/Minions/Cannon/towers), knockback (Fireball 1.0, Log 0.7 tiles from game `pushback`), `damage_reduction_while_not_attacking` (Evo Knight 60%). `status.py`: stun/freeze/slow/root/invisible via `Match.apply_status` (appliers now: Zap, Ice Spirit, Electro Spirit, Ice Golem; nothing applies root/invisible yet). **Slice 2**: `spawn_on_end` (rolling spell spawns a token where it stops, moved off the river; Barbarian Barrel), `invisible_when_idle` (Royal Ghost: invisible after 1.8 s without a target in range; enemies lose their lock-on, area damage still hits it), and the `pull_spell` kind with the `AreaEffect` entity in `Match.effects` (Tornado: pulls troops to the centre, damage every 500 ms for 1.05 s). |
| `backend_engine/simulator/entities.py` | `Unit` (+ `variant`, `mechanics`, `params`, `status`, `mstate`), `Spell`, `RollingSpell`, `Projectile` (engine re-exports them). |
| `backend_engine/simulator/golden.py` + `tests/golden/engine_hashes.json` | 15 determinism scenarios (bot matches L11/L16, Evo Knight, Log vs swarm, Fireball king, Cannon pulls Hog, river jump, crowd). `python tools/record_golden.py` checks + times them; `--write` re-records after an **intentional** gameplay change (then bump `engine.ENGINE_VERSION`). |
| `backend_engine/simulator/match_runner.py` | Drives matches; AI perceives through HumanDelayBuffer + OppElixirTracker + deck guesser; human side via `human_play`. CLI demo. |
| `configs/cards_stats_v2026.json` | Schema 4. **22 cards**: knight, hog_rider, musketeer, cannon, fireball, the_log, skeletons, archers, minions + batch 1 so far: zap (extra, not in the 30), ice_spirit, fire_spirit, electro_spirit, ice_golem, mini_pekka, valkyrie, skeleton_army, dart_goblin, rocket, barbarian_barrel, royal_ghost, tornado (+ token `barbarian`); + towers + rules. Each card: `internal_name` (game sc_key, e.g. Log), `game_id` (e.g. 26000000, for copyDeck links), `aliases {display, hf, royaleapi, detector[]}`, `variants {evo, hero}`, `mechanics [{kind, params}]`. Balance data **as of 2023-10** (new cards are `verified: false`; costs come from `card_costs_hf.json`). Migrated by `tools/migrate_cards_v4.py`; new cards are added by `tools/add_cards.py --spec tools/card_specs/batch1.json --source cards_stats.json --cards cards.json --as-of 2023-10-18 [--write]` (source files: `royaleapi.github.io/cr-api-data/json/cards_stats.json` and `cards.json`; the spec lists mechanics, and `$buff_time/$pushback/$death_radius` are read from the source record). **Hog 2.6, the most common meta deck, is playable since batch 1.** |

### Brain / perception
| file | what it does |
|---|---|
| `brain/human_delay_buffer.py` | 3-layer human limits: perception delay (surprise 420–580 / anticipated / routine), hover reflex 120–200, APM gap 180–300, combo gap **120 ms**. Snapshot-able. |
| `brain/scripted_bot.py` | Rule-based baseline AI (all 4 personas use it with different aggression). |
| `brain/value_model.py` + `configs/value_model.json` | Logistic win-prob model, retrained on **engine v2**: 400 bot-vs-bot games, 78.2% held-out accuracy, log loss 0.448 (v1 model on the same games: 77.9% / 0.453). `tools/train_value_model.py` records `engine_version` and scores the previous model on the same test games. |
| `brain/coach_analyzer.py` | Post-match review: ~30–40 candidates per play, 8 s rollouts, grades, accuracy, curve. |
| `perception/opp_elixir_count.py` | Exact opponent elixir + cycle/hand inference + evo progress (error 1e-13 vs engine). |
| `perception/bayesian_deck_guesser.py` | Deck posterior from `configs/meta_decks_hf.json` (3,184 real decks, HF Dec 2025). |
| `perception/live/*` | **Mode B** read-only: `capture.py` (Qt screen / frame folder), `calibration.py`, `detector.py` (YOLOv8 ONNX, 568×896 arena crop), `play_tracker.py` (cycle rule, building lifetime, spawn attribution, side from deploy position), `companion.py` (+ `SimDemoFeed`). |

### UI (PySide6)
| file | what it does |
|---|---|
| `frontend_app/qt/session.py` | Qt-free controller: checkpoints every 1 s + before each play, rewind/forward/seek/take-over, quiz, `fork()`, `analyze()`. |
| `frontend_app/qt/app.py`, `board.py`, `review.py` | **Fight page, redesigned 2026-11-15 (UI pass)** as four states (`MainWindow.mode`): **setup** (decks via Deck Builder, AI persona, level 11/16, elixir-counter Show/Quiz, vision outline → *Start match*), **live** (board, hand, clock, counter, play log only: no buttons, **no pause, no speed control**; leaving the page pauses, coming back resumes after a 3-2-1), **ended** (result card + *Review (R)* + *New match (N)*; Review exists only here), **review** (coach column on the right; win-rate graph + timeline slider + ⏪/⏩ under the board; slider / graph click jumps without countdown; *Take over from here* → live after 3-2-1; *Exit review* → ended; analysis has a progress bar with **Stop** / Esc, raises `AnalysisCancelled` from the progress callback). Deck Builder (`DeckDialog`) = 30 presets with You / AI toggle + preview + *Build your own…* (`CustomDeckDialog`, the old tick-8 list). U/F and log double-click still take over during live play (hotkeys only). Speed comes from the Account default. **UI is English-only.** |
| `frontend_app/qt/theme.py`, `widgets.py`, `icons/` | Shared palette `C`, stylesheet `STYLE` (QSS classes: `primary`, `big`, `ghost`, `danger`, `seg`, `card`, `h1/h2/sub/section/...`), `dark_palette()`, `cls()`; `DeckView` (8 card chips), `StatTile`, `card_frame`. shell.py re-exports `C/STYLE/dark_palette`. |
| `backend_engine/meta/presets.py` + `configs/preset_decks.json` | 30 ready-made decks (name, 8 cards, evo, style, tip) using only simulated cards; `find_preset` (exact evo, else same cards), `playable`, `ai_pushes` (the scripted AI only attacks with Hog Rider, so 22 presets are Hog decks; the dialog warns when a non-Hog deck is given to the AI). |
| `frontend_app/qt/shell.py` | **CRbot app shell** (2026-11-15): sidebar Fight / Analysis / Decks / Account, own charcoal + teal/violet palette (`dark_palette()`, `STYLE`). Fight = `MainWindow` embedded as a page (its hotkeys are disabled on other pages; leaving Fight pauses the match). Auto-saves on match end, review done, new match/settings change, reopen, app close (only if you played a card). Analysis (redesigned 2026-11-15) = 4 stat tiles (matches, win rate, W·L·D, avg accuracy), match list with outcome stripes, scrollable detail card (preset/archetype names, deck chips, win-rate graph + *Jump to* slider, *Open review at m:ss* → `open_match(id, at_ms=…)`, Analyze, Delete). The sidebar is locked while the coach analyzes. Window size fits the screen (`_fit_size`). Decks = top meta decks with play rate and "Needs …" (starter deck always playable), Fight with / against. Account = local profile, defaults, Mode B paths, data freshness. |
| `backend_engine/storage/` | `match_store.py`: `user_data/matches/<YYYYMMDD-HHMMSS_mode_hash8>/{meta,actions,review}.json`, atomic writes, review (de)serialisation. `profile.py`: `user_data/profile.json` (tag stays local). `CRBOT_DATA_DIR` overrides the folder; `user_data/` is git-ignored. |
| `backend_engine/meta/` | `archetypes.py` (plan §7 naming: "Hog 2.6", "EvoRoyalHogs GobHut"), `recommend.py` (top decks by play rate, missing cards, coverage). |
| `frontend_app/companion/app.py` | Always-on-top Mode B overlay (demo / screen / frames), calibration dialog, model loader. |

### Data, models, tools
* `pretrained_weights/sudhishneel_v8/v8_best.onnx` + `names.json` (174 classes; exported via Colab; .pt SHA-256 `af0046fc…391cfa`, pickle scanned clean). `configs/detector_class_map.json` (154 mapped, 20 ignored).
* `configs/card_costs_hf.json` (124 cards, current costs).
* Tools: `build_meta_decks.py`, `train_value_model.py`, `build_class_map.py`, `export_detector.py`, `eval_detector.py`. Notebook: `cloud_training/export_detector_colab.ipynb`.
* Superseded skeleton files are in `_legacy/` (do not import).

### Key architectural decisions
* Sim time (int ms) everywhere; fixed tick; deterministic; snapshot = rewind/MCTS/take-over basis.
* AI sees only human-visible info (fog of war); every AI action goes through HumanDelayBuffer.
* Card data is data-driven; exact scaling via base stats + official multiplier table.
* Mode B is strictly read-only (enforced by `test_read_only_contract`).
* UI layer is thin; logic lives in Qt-free classes (`GameSession`, `LiveCompanion`, `MatchStore`).
* Saved matches are rebuilt, not snapshotted: `GameSession.rebuild(record)` replays your plays at the same ticks
  with the same seed (the AI re-decides identically) and checks the final `state_hash`. A full 3:00 match rebuilds
  in ~0.3 s. If card data changes, rebuild raises `ReplayMismatch`; the saved summary/review still show.
  A new play on a reopened match starts a new match id (the original is kept).

## 2. Progress vs `CRBOT_UPGRADE_PLAN.md`

| section | % | remaining subtasks |
|---|---|---|
| App shell (CRbot sidebar: Fight / Analysis / Recommended deck / Account) | 92% | Mode B recordings into the store (`mode="recording"`); drills; review still runs on the UI thread (now with Stop + live progress; a worker thread would keep the board animating) |
| UI layout (existing arena/review screens) | 88% | *Show me* playback (animate the recommended line); user feedback on the new layout; Decks page could list the 30 presets too |
| Card schema v4 (internal names, aliases, variants) | 95% | hero variants (data + offset) when the first hero card arrives |
| Simulation core | 95% | charge (Dark Prince), burrow (Miner, Mighty Miner), root (Vines), top-N targeting (Lightning), damage-over-time + building multiplier (Earthquake), hide-underground (Tesla), form change + death bomb (Goblin Demolisher), champion abilities (Golden Knight, Mighty Miner), spawn-after-delay spell (Royal Delivery), death spawn (no remaining batch card needs it) |
| Mechanics framework + 30-card batch | 63% | 19 of the plan's 30 cards simulated (7 old + 12 new; Zap is an extra, so 22 cards in total); 11 left: mighty_miner, firecracker, earthquake, lightning, tesla, golden_knight, vines, miner, royal_delivery, goblin_demolisher, dark_prince |
| Current card stats (Nov 2026, L16) | 10% | find a fresh data source; golden-value tests |
| Deck link import + archetype names | 40% | naming rules done (`meta/archetypes.py`); `copyDeck` parser |
| IL dataset + policy/value | 5% | event-stream sample builder; Colab training; ONNX `NeuralBot` |
| Self-play PPO + personas | 0% | league training; persona embedding + reward shaping |
| Coach v2 (value-net evaluator, tiers, timing ±1 s) | 40% | swap evaluator; Great/Excellent tiers; weighted accuracy; timing variants |
| Elixir counter / deck guesser / delay system | 92% | minor |
| Mode B companion | 60% | test on real recorded matches; RoyaleAPI timeline comparison; detector fine-tuning |
| **Overall** | **~55%** | |

## 3. Open issues & constraints

**Open issues / risks**
1. No machine-readable **current (Nov 2026) card stats** source; RoyaleAPI repo data is from 2023-10.
2. Only 22 cards simulated (19 of the plan's 30) → at 30 cards only 13 HF battles are fully re-simulatable, so IL must train on event streams.
3. Mode B never tested on a real match; SudhishNeel v8 on rare-card synthetic frames: 51% found, 72% right card; 14 ms/frame (CoreML). Spell-side heuristic can be wrong.
4. Coach grades depend on a scripted-bot win model (baseline quality only).
5. Local network is slow (~0.27 MB/s): do heavy installs/exports on Colab.
6. Value model, personas and the AI are not yet neural.

7. Code hygiene (fixed 2026-11-15): the docstring commit `b3043fd` had appended duplicate copies of methods to
   `coach_analyzer.py`, `human_delay_buffer.py` and `session.py` (the later copy silently won). Trimmed; behaviour
   unchanged (identical ASTs). Watch for this if docstrings are regenerated.
8. With 22 cards only Hog 2.6 is fully playable (2 of 932 decks, 220 of 3,184 deck listings): Tornado/Ghost/Barrel did not unlock anything alone, because coverage is cumulative. 13 decks are 1 card short and 37 are 2 short; the cards missing from those near-complete decks, by battles, are x_bow (73), tesla (61), executioner (43), mortar (43), goblins (42), skeleton_barrel (40), firecracker (16): worth adding next (X-Bow/Mortar/Goblins need no new mechanic). The Decks page shows what each deck needs.
9. Engine v2 (2026-11-15) changed gameplay (projectiles, knockback, Evo Knight). Matches saved with v1 show their
   summary/review in Analysis but can't be reopened. The value model has been retrained on v2.
10. Git (2026-11-15): `main` and `feat/app-shell` were rewritten and force-pushed to remove `v8_best.onnx`,
   `v8_best.pt` (licence "personal use only") and a `user_data/` match from all history. Clones made before that
   must re-clone or `git reset --hard origin/<branch>`. Weights (`pretrained_weights/**/*.{onnx,pt,pth}`) and
   `user_data/` are git-ignored; the local files are kept on disk only.
11. **Uncommitted work from another session** (found 2026-11-15, kept, not committed): `play_tracker.py` (cycle-rule override + gap-aware speed gate), `human_delay_buffer.py`, portable-arithmetic engine v3 (`engine.py`/`geometry.py`/`core.py`/`rng.py`/`tables.py`, `tests/test_portable.py`), `value_model.json`, re-recorded goldens. The 2 Mode B test failures were fixed 2026-11-15: the speed gate was a no-op (`gate_base` = `match_radius` = 1.75), so a unit dropped where the old one had just died inherited its track; it now applies `gate_base` 1.0 + the card's own speed (`PlayTracker.speeds`, from the card DB) only after a sighting gap of 250 ms or more (tuned on 6 seeds: perfect feed err 0.24 / miss 0.016, noisy feed 0.54 / 0.030 / extra 0.026). The cycle-rule test now covers both sides of `dup_window_ms`. Plays that die within one frame stay unobservable by design.
12. Batch-1 assumptions to verify in game (not in the 2023-10 data): Ice Golem death slow lasts 2 s; Electro Spirit chain count 9 means 9 units in total (first included); Zap / Ice Spirit / Electro Spirit durations (0.5 s / 1.2 s / 0.5 s) come from the data. Kamikaze hits land on arrival (no jump flight time). Tornado: duration 1.05 s, damage every 0.5 s and radius 5.5 tiles come from the data, but the pull speed (3.6 tiles/s, from attract_percentage 360) is assumed. Royal Ghost's 1.8 s idle time comes from the data (`buff_when_not_attacking_time`). The Barbarian Barrel does full damage to towers (the data has no tower reduction for it).

## 4. Recommended next steps
1. **Next cards** (see open issue 8 for the data-driven order): plain-data cards first (goblins, x_bow, mortar, firecracker), then skeleton_barrel (needs a `death_spawn` mechanic on top of the new tokens), then the mechanics: charge (Dark Prince), burrow (Miner), hide-underground (Tesla), root (Vines), top-N targeting (Lightning), DoT + building multiplier (Earthquake). Add specs to `tools/card_specs/` and run `tools/add_cards.py`; every mechanic gets a unit test in the style of `tests/test_batch1*.py` plus a golden scenario. **Get current (Nov 2026) stats before trusting any of it** (open issue 1).
2. **Deck link import**: parse `clashroyale://copyDeck?deck=<ids>` using `game_id` and `CardDB.by_alias`.
3. **Validate Mode B on one real recorded match** (record to `recordings/`, extract frames with ffmpeg, calibrate, compare with the RoyaleAPI timeline).
4. **Shell polish**: coach review in a background worker (Stop already works); save Mode B sessions as `recording` matches; get the user's feedback on the 2026-11-15 UI redesign (screens checked offscreen at 1200×860 and 1000×700).
