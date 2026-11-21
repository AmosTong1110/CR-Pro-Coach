# CRbot — Session Handoff (2026-11-21)

Short on purpose: read this at session start. Details by topic: `docs/HANDOFF_ARCHIVE.md` (history to 2026-11-17,
sections: 1 completed work per file, 2 progress, 3 open issues 1–27, 4 next steps). Roadmap: `docs/CRBOT_UPGRADE_PLAN.md`
(§6 training, §7 coach, §10 card phases). Grep those files for the section you need; don't read them whole.

## State
* CRbot = local Clash Royale simulator + human-limited AI opponent + post-match coach (single mode; screen capture
  "Mode B" removed 2026-11-17). Not a bot for the real game.
* Engine **v7** (deterministic 50 ms tick), observation **il_obs_v2** (OBS_DIM 1752, attribute-based, card-pool
  independent). Cards: 123 in the catalog, **47 playable**, 76 quarantined. 7 of 14 evolutions, 5 heroes simulated.
* Tests: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q` (546 pass, 1 skip expected; torch tests skip).
* Env: `.venv` (Python 3.14). Homebrew `python3` has no packages. App: `.venv/bin/python -m frontend_app.qt.app`.
* Overall vs the 4-week plan: ~70%. Weakest: PPO 45%, Coach v2 65%, current card stats 78%.

## Constraints (keep)
M1 8 GB budget (no torch at runtime; heavy work on Colab) · AI sees only human-visible info and acts through
HumanDelayBuffer · determinism (re-record goldens + bump `ENGINE_VERSION` only for intentional gameplay changes) ·
English-only UI · snake_case card ids · scan downloaded model files · user data stays local (`user_data/`).
No automation of the real game or play against real players on the live ladder (Supercell ToS).

## Done 2026-11-21
* **IL v2 shards**: v1 shards moved to `data/il_shards_v1/` (all `data/il_shards*/` are git-ignored).
  `data/il_shards/`: 400,044 samples, 4,926 battles (parts 0–13, all modes; fallback set).
  **`data/il_shards_full/` = the training set (done 14:59, checked)**: 1,551,234 samples (1:1 play/wait) from
  18,570 battles, all 52 parts, modes Ranked/Ladder/Grand Challenge/Classic Challenge, 311 shards, 243 MB, 63 min.
  Rejected: coverage 208,772, mode 24,896. **Hog 2.6 = deck id 2: 671,836 samples / 7,741 battles (43% of all).**
  Not uploaded yet. Colab RAM need ≈ 5.6 GB (float16 loader).
* Shards now store `deck` (index into manifest `learner_decks`, sorted 8 card ids) per sample.
* Notebooks are name-aware (defaults unchanged). IL: `IL_NAME`, `IL_INIT` (.npz to start from), `IL_DECK` (8 card
  ids: keep only learners with that deck), `IL_LR`. PPO: `PPO_NAME` (own checkpoint folder unless crbot_ppo_v2),
  `PPO_INIT`, `PPO_DECKS`, `PPO_HOURS`. `pack_ppo_bundle.py --il-model <name>`.
* Dataset facts (VanguardX101/IL_Replay, "Firstlight anonymized"): 252,238 battles, 17.8M actions; trophies and dates
  removed; part 0 modes: 89% Ranked, 7% friendly-type ("1v1 Battle", "Normal Battle"), rest Ladder / challenges.
  Only ~7% of battles are usable with 47 cards; full 123-card coverage would allow ~30M samples.
* `build_il_dataset.py --modes` (game-mode filter, rejected count `mode`; test in `tests/test_il_dataset.py`).
* IL notebook cell 3 loads obs as preallocated **float16** (~3.5 KB/sample; ~1.3M samples ≈ 4.6 GB; RAM check
  before loading); `get_batch` casts obs back to float32. Verified locally on 12k real samples (labels identical,
  max obs error 8e-4). For ~1.3M samples consider `IL_EPOCHS=10` (T4 time not measured yet).
* **Deck specialist + daily budget** in `brain/ppo_trainer.py`: `PPOConfig.learner_decks` (preset names, e.g.
  `("Hog 2.6 Cycle",)`; only the learner's deck, opponents keep all presets), `time_budget_h` (clean stop with
  checkpoint + export, next run resumes), `export_name` (e.g. `crbot_ppo_hog26` → NeuralBot loads it from
  `pretrained_weights/<name>/`). Helper `meta.presets.select_presets` (tested in `tests/test_presets.py`).

## Next steps
1. (User, in progress) Empty Drive `datasets/il_shards/` (old v1 files break the run), upload the contents of
   `data/il_shards_full/` there and `dist/crbot_code_bundle.zip` (commit a2826a3) →
   `MyDrive/CRbot_Training/`; run `cloud_training/train_imitation_colab.ipynb` (T4, `IL_EPOCHS=10`); scan the downloaded .onnx (opset, no custom
   domains / external data, SHA-256 = meta) and install `crbot_il_v2.onnx/.npz/.json`
   in `pretrained_weights/crbot_il_v2/`; calibrate: `.venv/bin/python cloud_training/gauntlet_eval.py --out gauntlet.json --write-meta`.
2. Hog 2.6 specialist: (a) IL fine-tune: rerun the IL notebook with `IL_NAME=crbot_il_v2_hog26`,
   `IL_INIT=<Drive>/models/crbot_il_v2/crbot_il_v2.npz`, `IL_DECK=cannon,fireball,hog_rider,ice_golem,ice_spirit,
   musketeer,skeletons,the_log`, `IL_LR=1e-4`, `IL_EPOCHS=5`; install it in `pretrained_weights/crbot_il_v2_hog26/`.
   (b) `pack_ppo_bundle.py --il-model crbot_il_v2_hog26`; PPO notebook with `PPO_NAME=crbot_ppo_hog26`,
   `PPO_INIT=crbot_il_v2_hog26`, `PPO_DECKS=Hog 2.6 Cycle`, `PPO_HOURS=12`, `PPO_UPDATES=100000` (so the time budget decides); rerun daily (resumes). No app persona yet.
3. Retrain the value model on engine v7 (`tools/train_value_model.py`, Colab or a quiet machine).
4. Then: replay-agreement benchmark (re-simulate real battles, compare outcomes), Phase 2 cards (plan §10), Elo ladder.

## Open issues (top)
* No machine-readable current (Nov 2026) card stats; many mass / radius / timing values are `assumed` per card.
* IL v1 value head ≈ coin flip; coach still grades with the logistic value model (trained on engine v6).
* Neural persona falls back to ScriptedBot until crbot_il_v2 exists.
* Replay data (HF IL_Replay) has no licence: personal research only.
