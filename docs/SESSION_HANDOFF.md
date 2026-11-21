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
  `data/il_shards/`: 400,044 samples from 4,926 battles (parts 0–13, all modes, 18 min, 63 MB).
  **Full build** (started 13:45, ~70 min): `--output-dir data/il_shards_full --max-samples 100000000 --modes
  "Ranked,Ladder,Grand Challenge,Classic Challenge"` → expect ~1.3M samples from ~17k battles. Use it if its manifest
  is sane (il_obs_v2, obs_dim 1752, `filters.modes` set); upload it to Drive as `datasets/il_shards/`.
* Dataset facts (VanguardX101/IL_Replay, "Firstlight anonymized"): 252,238 battles, 17.8M actions; trophies and dates
  removed; part 0 modes: 89% Ranked, 7% friendly-type ("1v1 Battle", "Normal Battle"), rest Ladder / challenges.
  Only ~7% of battles are usable with 47 cards; full 123-card coverage would allow ~30M samples.
* `build_il_dataset.py --modes` (game-mode filter, rejected count `mode`; test in `tests/test_il_dataset.py`).
* IL notebook cell 3 loads obs as preallocated **float16** (~3.5 KB/sample; ~1.3M samples ≈ 4.6 GB; RAM check
  before loading); `get_batch` casts obs back to float32. Verified locally on 12k real samples (labels identical,
  max obs error 8e-4). For ~1.3M samples consider `IL_EPOCHS=10` (T4 time not measured yet).
* `dist/crbot_code_bundle.zip` packed (commit b254b4e; repack after committing today's changes).
* **Deck specialist + daily budget** in `brain/ppo_trainer.py`: `PPOConfig.learner_decks` (preset names, e.g.
  `("Hog 2.6 Cycle",)`; only the learner's deck, opponents keep all presets), `time_budget_h` (clean stop with
  checkpoint + export, next run resumes), `export_name` (e.g. `crbot_ppo_hog26` → NeuralBot loads it from
  `pretrained_weights/<name>/`). Helper `meta.presets.select_presets` (tested in `tests/test_presets.py`).
* Session start trimmed: this file replaces the 54 KB handoff (now the archive).

## Next steps
1. Commit, repack `tools/pack_ppo_bundle.py --code-only` (the notebook change is in the notebook, the bundle holds
   the model code). Upload `data/il_shards_full/` → `MyDrive/CRbot_Training/datasets/il_shards/` and `dist/crbot_code_bundle.zip` →
   `MyDrive/CRbot_Training/`; run `cloud_training/train_imitation_colab.ipynb` (T4); install `crbot_il_v2.onnx/.npz/.json`
   in `pretrained_weights/crbot_il_v2/`; calibrate: `.venv/bin/python cloud_training/gauntlet_eval.py --out gauntlet.json --write-meta`.
2. Hog 2.6 specialist PPO: `tools/pack_ppo_bundle.py`, then `train_ppo_colab.ipynb` with
   `PPOConfig(learner_decks=("Hog 2.6 Cycle",), time_budget_h=12, export_name="crbot_ppo_hog26")` (the notebook
   cell still needs these fields set); one run per day resumes from Drive checkpoints.
3. Retrain the value model on engine v7 (`tools/train_value_model.py`, Colab or a quiet machine).
4. Then: replay-agreement benchmark (re-simulate real battles, compare outcomes), Phase 2 cards (plan §10), Elo ladder.

## Open issues (top)
* No machine-readable current (Nov 2026) card stats; many mass / radius / timing values are `assumed` per card.
* IL v1 value head ≈ coin flip; coach still grades with the logistic value model (trained on engine v6).
* Neural persona falls back to ScriptedBot until crbot_il_v2 exists.
* Replay data (HF IL_Replay) has no licence: personal research only.
