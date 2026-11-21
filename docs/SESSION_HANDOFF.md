# CRbot — Session Handoff (2026-11-21)

Short on purpose: read this at session start. Details by topic: `docs/HANDOFF_ARCHIVE.md` (history to 2026-11-17,
sections: 1 completed work per file, 2 progress, 3 open issues 1–27, 4 next steps). Roadmap: `docs/CRBOT_UPGRADE_PLAN.md`
(§6 training, §7 coach, §10 card phases). Grep those files for the section you need; don't read them whole.

## State
* CRbot = local Clash Royale simulator + human-limited AI opponent + post-match coach (single mode; screen capture
  "Mode B" removed 2026-11-17). Not a bot for the real game.
* Engine **v7** (deterministic 50 ms tick), observation **il_obs_v2** (OBS_DIM 1752, attribute-based, card-pool
  independent). Cards: 123 in the catalog, **47 playable**, 76 quarantined. 7 of 14 evolutions, 5 heroes simulated.
* Tests: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q` (578 pass, 1 skip expected; torch tests skip).
* Env: `.venv` (Python 3.14). Homebrew `python3` has no packages. App: `.venv/bin/python -m frontend_app.qt.app`.
* Overall vs the 4-week plan: ~70%. Weakest: PPO 45%, Coach v2 65%, current card stats 78%.

## Constraints (keep)
M1 8 GB budget (no torch at runtime; heavy work on Colab) · AI sees only human-visible info and acts through
HumanDelayBuffer · determinism (re-record goldens + bump `ENGINE_VERSION` only for intentional gameplay changes) ·
English-only UI · snake_case card ids · scan downloaded model files · user data stays local (`user_data/`).
No automation of the real game or play against real players on the live ladder (Supercell ToS).

## Done 2026-11-21 (IL dataset details: HANDOFF_ARCHIVE §5)
* **IL v2 shards** (`data/il_shards*/` git-ignored; v1 shards, the 400k set and the old PPO bundle deleted).
  **`data/il_shards_full/` = the training set (done 14:59, checked)**: 1,551,234 samples (1:1 play/wait) from
  18,570 battles, all 52 parts, modes Ranked/Ladder/Grand Challenge/Classic Challenge, 311 shards, 243 MB, 63 min.
  Rejected: coverage 208,772, mode 24,896. **Hog 2.6 = deck id 2: 671,836 samples / 7,741 battles (43% of all).**
  Colab RAM need ≈ 5.6 GB (float16 loader).
* Notebooks are name-aware (defaults unchanged). IL: `IL_NAME`, `IL_INIT` (.npz to start from), `IL_DECK` (8 card
  ids: keep only learners with that deck), `IL_LR`. PPO: `PPO_NAME` (own checkpoint folder unless crbot_ppo_v2),
  `PPO_INIT`, `PPO_DECKS`, `PPO_HOURS`. `pack_ppo_bundle.py --il-model <name>`.
* `build_il_dataset.py --modes` (game-mode filter, rejected count `mode`; test in `tests/test_il_dataset.py`).
* **Deck specialist + daily budget** in `brain/ppo_trainer.py`: `PPOConfig.learner_decks` (preset names, e.g.
  `("Hog 2.6 Cycle",)`; only the learner's deck, opponents keep all presets), `time_budget_h` (clean stop with
  checkpoint + export, next run resumes), `export_name` (e.g. `crbot_ppo_hog26` → NeuralBot loads it from
  `pretrained_weights/<name>/`). Helper `meta.presets.select_presets` (tested in `tests/test_presets.py`).

## Next steps
1. **Done 2026-11-21: crbot_il_v2 installed** in `pretrained_weights/crbot_il_v2/` (git-ignored; scanned: SHA-256 =
   meta, opset 17, standard ops only, no external data, npz plain arrays). Val (epoch 16/20, plateau from ~12):
   slot top1 61.7% (baseline 35.4%), tile top3 49.3% (14.8%), gate acc 73.4% (F1 0.758), value Brier 0.237 (weak).
   Gauntlet (432 matches, 2 workers; 3 workers got SIGKILLed on 8 GB) wrote `gate_threshold 0.5`,
   `decide_every_ms 250` (24/24 vs ScriptedBot; most configs ≥ 95%: ScriptedBot is too weak to separate them).
   `.json.bak` = original meta. Neural persona now uses NeuralBot (~4-5 ms/decision).
2. Hog 2.6 specialist: (a) IL fine-tune: rerun the IL notebook with `IL_NAME=crbot_il_v2_hog26`,
   `IL_INIT=<Drive>/models/crbot_il_v2/crbot_il_v2.npz`, `IL_DECK=cannon,fireball,hog_rider,ice_golem,ice_spirit,
   musketeer,skeletons,the_log`, `IL_LR=1e-4`, `IL_EPOCHS=5`; install it in `pretrained_weights/crbot_il_v2_hog26/`.
   (b) `pack_ppo_bundle.py --il-model crbot_il_v2_hog26`; PPO notebook with `PPO_NAME=crbot_ppo_hog26`,
   `PPO_INIT=crbot_il_v2_hog26`, `PPO_DECKS=Hog 2.6 Cycle`, `PPO_HOURS=12`, `PPO_UPDATES=100000` (so the time budget decides); rerun daily (resumes). No app persona yet.
2b. **Visual upgrade (user priority)**. Phase 1 (2026-11-21): original QPainter art (`art.py`, `fx.py` animations,
   `widgets.CardButton` card faces; no placement-zone overlay, `BoardWidget.show_zone=False`). **Phase 2 started
   2026-11-21: CC0 3D sprites** (user approved after a trial): `tools/art/build_sprites.py` downloads KayKit packs
   (Kay Lousberg, CC0, licence checked) to git-ignored `art_src/`, renders them with Blender 5.3 (`brew install --cask
   blender`; `tools/art/blender_render.py`, headless, autoexec off) and packs `frontend_app/qt/assets/sprites/`
   (committed, 4.7 MB: arena.jpg, towers/castles, knight/archer/skeleton sheets per side, manifest.json).
   `sprites.py` (SpriteBank + `pose()` walk/attack/idle from engine state, attack synced to cooldown) and board
   sprite mode: 35° tilted mapping (`BoardWidget.cos`, margins; clicks inverted), ground circles flattened.
   `CRBOT_ART=vector` or missing assets -> old vector board. Unmapped cards still draw vector tokens.
   **Next**: models for the other 43 cards (`CHARACTERS`/`CARDS` in build_sprites.py; fliers, giants, Hog Rider,
   buildings need more CC0 packs, e.g. Quaternius / other KayKit), 8 facings, sprite effects, sound (CC0).
3. Retrain the value model on engine v7 (`tools/train_value_model.py`, Colab or a quiet machine).
4. Then: replay-agreement benchmark (re-simulate real battles, compare outcomes), Phase 2 cards (plan §10), Elo ladder.

## Open issues (top)
* Fight page: the elixir counter's "Ahead +x" label is clipped at 1200 px (pre-existing, not the board).
* No machine-readable current (Nov 2026) card stats; many mass / radius / timing values are `assumed` per card.
* IL v2 value head ≈ coin flip (Brier 0.237); coach still grades with the logistic value model (trained on engine v6).
* Replay data (HF IL_Replay) has no licence: personal research only.
