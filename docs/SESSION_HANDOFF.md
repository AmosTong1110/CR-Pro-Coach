# CRbot — Session Handoff (2026-11-15)

Read this first in every new session. Roadmap: `docs/CRBOT_UPGRADE_PLAN.md`.
State at handoff: **241 tests passing** (`QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q`), engine **v2**, branch `feat/app-shell` (not merged into `main`).
Environment: project venv `.venv` (Python 3.14; numpy, pyyaml, PySide6-Essentials, onnxruntime, Pillow, pytest).
Homebrew's `python3` has no packages, so always use `.venv/bin/python`. Run the app: `.venv/bin/python -m frontend_app.qt.app`.

## 1. Completed work

### Simulator (Mode A)
| file | what it does |
|---|---|
| `backend_engine/contracts.py` | Shared dataclasses: `PublicDeployEvent`, `ElixirEstimate`, `ActionIntent`, `HoverPlan`, `ScheduledAction`, `TilePos`. All time is integer **sim ms**. |
| `backend_engine/simulator/geometry.py` | 18×32 tile arena, river/bridges, tower layout, `deploy_check` (own half, pocket after a princess falls, spells anywhere), bridge pathing. |
| `backend_engine/simulator/engine.py` | Deterministic 50 ms tick engine: elixir phases (x1/x2/OT x2→x3), 4+4 card cycle, evo charge counter, towers + king activation on any damage, targeting, attacks, splash, spells (travel + area), rolling Log, building decay, **mass-based soft collision** (river-safe), win/crowns/tiebreaker, `snapshot/restore`, `state_hash`, `Match.replay`, fog-of-war `observation()`. |
| `backend_engine/simulator/card_db.py` | **Schema v4** loader (still reads v3 in memory via `normalise_card`). **Exact official scaling**: `stat(L)=floor(base·M[L-1-rarity_offset]/100)` (verified on 4,945 values). Helpers: `variant`, `has_variant`, `evo_cycles`, `evo_capable`, `mechanics_for(cid, variant)` (explicit + derived `projectile`/`splash`/`decay`/spell kind), `spell_kind`, `by_alias`, `unsimulated_tags`; `stats(cid, level, variant)` applies `stat_mods`. |
| `backend_engine/simulator/sync_meta.py` | Imports RoyaleAPI-format game data into `configs/cards_stats_v2026.json` (base stats, timings, ranges, speed, mass). |
| `backend_engine/simulator/mechanics/` | **Mechanics framework** (plan §5). `base.py`: `Mechanic` hooks (`on_spawn`, `on_tick`, `on_attack`, `on_damaged`, `on_death`, `waypoint`, `river_passable`), `REGISTRY` + `@register`, `SpellKind` + `SPELLS`, `DESCRIPTIVE` tags. `core.py`: `river_jump`, `splash`, `decay`, `area_spell`, `rolling_spell` (ported, identical hashes) + engine v2: `projectile` (homing, travel time; Musketeer/Archers/Minions/Cannon/towers), knockback (Fireball 1.0, Log 0.7 tiles from game `pushback`), `damage_reduction_while_not_attacking` (Evo Knight 60%). `status.py`: stun/freeze/slow/root/invisible via `Match.apply_status` (nothing applies them yet). |
| `backend_engine/simulator/entities.py` | `Unit` (+ `variant`, `mechanics`, `params`, `status`, `mstate`), `Spell`, `RollingSpell`, `Projectile` (engine re-exports them). |
| `backend_engine/simulator/golden.py` + `tests/golden/engine_hashes.json` | 15 determinism scenarios (bot matches L11/L16, Evo Knight, Log vs swarm, Fireball king, Cannon pulls Hog, river jump, crowd). `python tools/record_golden.py` checks + times them; `--write` re-records after an **intentional** gameplay change (then bump `engine.ENGINE_VERSION`). |
| `backend_engine/simulator/match_runner.py` | Drives matches; AI perceives through HumanDelayBuffer + OppElixirTracker + deck guesser; human side via `human_play`. CLI demo. |
| `configs/cards_stats_v2026.json` | Schema 4. 9 cards (knight, hog_rider, musketeer, cannon, fireball, the_log, skeletons, archers, minions) + towers + rules. Each card: `internal_name` (game sc_key, e.g. Log), `game_id` (e.g. 26000000, for copyDeck links), `aliases {display, hf, royaleapi, detector[]}`, `variants {evo, hero}`, `mechanics [{kind, params}]`. Balance data **as of 2023-10**. Migrated by `tools/migrate_cards_v4.py`. |

### Brain / perception
| file | what it does |
|---|---|
| `brain/human_delay_buffer.py` | 3-layer human limits: perception delay (surprise 420–580 / anticipated / routine), hover reflex 120–200, APM gap 180–300, combo gap **120 ms**. Snapshot-able. |
| `brain/scripted_bot.py` | Rule-based baseline AI (all 4 personas use it with different aggression). |
| `brain/value_model.py` + `configs/value_model.json` | Logistic win-prob model trained on 200 bot-vs-bot games (73% held-out acc). |
| `brain/coach_analyzer.py` | Post-match review: ~30–40 candidates per play, 8 s rollouts, grades, accuracy, curve. |
| `perception/opp_elixir_count.py` | Exact opponent elixir + cycle/hand inference + evo progress (error 1e-13 vs engine). |
| `perception/bayesian_deck_guesser.py` | Deck posterior from `configs/meta_decks_hf.json` (3,184 real decks, HF Dec 2025). |
| `perception/live/*` | **Mode B** read-only: `capture.py` (Qt screen / frame folder), `calibration.py`, `detector.py` (YOLOv8 ONNX, 568×896 arena crop), `play_tracker.py` (cycle rule, building lifetime, spawn attribution, side from deploy position), `companion.py` (+ `SimDemoFeed`). |

### UI (PySide6)
| file | what it does |
|---|---|
| `frontend_app/qt/session.py` | Qt-free controller: checkpoints every 1 s + before each play, rewind/forward/seek/take-over, quiz, `fork()`, `analyze()`. |
| `frontend_app/qt/app.py`, `board.py`, `review.py` | Practice arena, elixir-counter panel + quiz, timeline slider, 3-2-1 countdown, review dock (graph, marks, ghosts, take over). **User translated UI to English-only.** |
| `frontend_app/qt/shell.py` | **CRbot app shell** (2026-11-15): sidebar Fight / Analysis / Decks / Account, own charcoal + teal/violet palette (`dark_palette()`, `STYLE`). Fight = `MainWindow` embedded as a page (its hotkeys are disabled on other pages; leaving Fight pauses the match). Auto-saves on match end, review done, new match/settings change, reopen, app close (only if you played a card). Analysis = saved-match list, stats, win-rate preview, *Open review in Fight* (rebuilds the match: ghosts + take over work), Analyze, Delete. Decks = top meta decks with play rate and "Needs …" (starter deck always playable), Fight with / against. Account = local profile, defaults, Mode B paths, data freshness. |
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
| App shell (CRbot sidebar: Fight / Analysis / Recommended deck / Account) | 85% | Mode B recordings into the store (`mode="recording"`); drills; restyle Fight page widgets (hand bar, buttons still old inline colours); review runs synchronously (UI blocks ~5 s) → move to a worker |
| UI layout (existing arena/review screens) | 70% | move into shell; *Show me* playback; visual identity (own palette) |
| Card schema v4 (internal names, aliases, variants) | 95% | hero variants (data + offset) when the first hero card arrives |
| Simulation core | 90% | stun/slow appliers, death spawn/damage, charge, burrow (with batch 1) |
| Mechanics framework + 30-card batch | 30% | framework + projectiles/knockback/Evo Knight done; 21 new cards (§1 of plan) |
| Current card stats (Nov 2026, L16) | 10% | find a fresh data source; golden-value tests |
| Deck link import + archetype names | 40% | naming rules done (`meta/archetypes.py`); `copyDeck` parser |
| IL dataset + policy/value | 5% | event-stream sample builder; Colab training; ONNX `NeuralBot` |
| Self-play PPO + personas | 0% | league training; persona embedding + reward shaping |
| Coach v2 (value-net evaluator, tiers, timing ±1 s) | 40% | swap evaluator; Great/Excellent tiers; weighted accuracy; timing variants |
| Elixir counter / deck guesser / delay system | 92% | minor |
| Mode B companion | 60% | test on real recorded matches; RoyaleAPI timeline comparison; detector fine-tuning |
| **Overall** | **~48%** | |

## 3. Open issues & constraints

**Open issues / risks**
1. No machine-readable **current (Nov 2026) card stats** source; RoyaleAPI repo data is from 2023-10.
2. Only 9 cards simulated → at 30 cards only 13 HF battles are fully re-simulatable, so IL must train on event streams.
3. Mode B never tested on a real match; SudhishNeel v8 on rare-card synthetic frames: 51% found, 72% right card; 14 ms/frame (CoreML). Spell-side heuristic can be wrong.
4. Coach grades depend on a scripted-bot win model (baseline quality only).
5. Local network is slow (~0.27 MB/s): do heavy installs/exports on Colab.
6. Value model, personas and the AI are not yet neural.

7. Code hygiene (fixed 2026-11-15): the docstring commit `b3043fd` had appended duplicate copies of methods to
   `coach_analyzer.py`, `human_delay_buffer.py` and `session.py` (the later copy silently won). Trimmed; behaviour
   unchanged (identical ASTs). Watch for this if docstrings are regenerated.
8. Only 0 of 932 real meta decks are playable with 9 cards; the Decks page shows what each one still needs.
9. Engine v2 (2026-11-15) changed gameplay (projectiles, knockback, Evo Knight). Matches saved with v1 show their
   summary/review in Analysis but can't be reopened. `configs/value_model.json` was trained on v1 bot games: retrain
   (`tools/train_value_model.py`) before trusting coach grades.
10. Git (2026-11-15): `main` and `feat/app-shell` were rewritten and force-pushed to remove `v8_best.onnx`,
   `v8_best.pt` (licence "personal use only") and a `user_data/` match from all history. Clones made before that
   must re-clone or `git reset --hard origin/<branch>`. Weights (`pretrained_weights/**/*.{onnx,pt,pth}`) and
   `user_data/` are git-ignored; the local files are kept on disk only.

## 4. Recommended next steps
1. **Retrain the value model on engine v2** (coach grades currently use a v1-trained model), then merge `feat/app-shell` into `main`.
2. **Batch-1 mechanics + first new cards** (plan Week 2): stun/slow appliers (Zap/Ice Spirit/Electro Spirit), death spawn,
   multi-spawn, burrow (Miner), invisibility (Royal Ghost); add cards with `tools/migrate_cards_v4.py` + `sync_meta`.
3. **Deck link import**: parse `clashroyale://copyDeck?deck=<ids>` using `game_id` and `CardDB.by_alias`.
4. **Validate Mode B on one real recorded match** (record to `recordings/`, extract frames with ffmpeg, calibrate, compare with the RoyaleAPI timeline).
5. **Shell polish**: coach review in a background worker with cancel; restyle Fight page controls; save Mode B sessions as `recording` matches.
