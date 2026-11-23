# CRbot — Session Handoff (2026-11-23)

Short on purpose: read this at session start. **Plan: `docs/ROADMAP.md`** (status by module, milestones MS1–MS6,
pending owner decisions; visual bilingual copy: https://claude.ai/artifact/C9gEHUS6fuqVdJjvQdXqAv). Specs:
`docs/CRBOT_UPGRADE_PLAN.md` (§4 card schema, §6 training, §7 coach, §10 card phases). History: `docs/HANDOFF_ARCHIVE.md`.
Grep the long files for the section you need; don't read them whole.

## State
* CRbot = local Clash Royale simulator + human-limited AI opponent + post-match coach (single mode; screen capture
  "Mode B" removed 2026-11-17). Not a bot for the real game.
* Engine **v8** (deterministic 50 ms tick), observation **il_obs_v2** (OBS_DIM 1752, attribute-based, card-pool
  independent). Cards: 123 in the catalog, **72 playable** (all 8 season meta decks), 51 quarantined. 7 of 14
  evolutions, 5 heroes simulated. Models: `crbot_il_v2` (Neural persona; trained on v7, still valid: card-pool free).
* Tests: `QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest -q` (588 pass, 1 skip expected; torch tests skip).
* Env: `.venv` (Python 3.14). Homebrew `python3` has no packages. App: `.venv/bin/python -m frontend_app.qt.app`.
* Overall ~75%. Weakest: PPO 45%, Coach v2 65%.

## Constraints (keep)
M1 8 GB budget (no torch at runtime; heavy work on Colab) · AI sees only human-visible info and acts through
HumanDelayBuffer · determinism (re-record goldens + bump `ENGINE_VERSION` only for intentional gameplay changes) ·
English-only UI · snake_case card ids · scan downloaded model files · user data stays local (`user_data/`).
No automation of the real game or play against real players on the live ladder (Supercell ToS).
Board art (user decision 2026-11-21): **official Supercell Fan Kit only** (fankit.supercell.com; downloading it means
accepting Supercell's Fan Content Policy: ask the user), under the Fan Content Policy: non-commercial (remove it if CRbot ever charges), "unofficial / not endorsed"
disclaimer with policy link, kit files stay local + git-ignored. Never extract assets from the game client / APK.

## Done 2026-11-23 (details: HANDOFF_ARCHIVE §7)
* `docs/ROADMAP.md` (one-page plan) + bilingual artifact; `CRBOT_UPGRADE_PLAN.md` trimmed to specs; mobile plan on hold.
* **MS4 done**: 25 cards + 9 tokens for the 8 season meta decks, from wiki pages fetched through the MediaWiki API
  (fandom blocks plain fetches; the API works), 2026 balance notes on top; 7 new "Season meta" presets; ScriptedBot
  plays Goblin Barrel / Drill as win conditions.
* **Audit** of all older cards and the towers (`tools/apply_card_patches.py` + `tools/card_specs/audit_2026_11.json`):
  the 9 hand-entered cards were wrong (e.g. Goblins x4 not x3, Giant / P.E.K.K.A. speed, Baby Dragon splash, Witch
  range, Tombstone spawns); ~40 balance-note fixes; Crown Tower L12-16. Engine v8, goldens re-recorded.
* `verify_wiki` status `newer` = official value newer than the wiki table on purpose (`newer_than_wiki`).
* Value model retrained on v8 (2000 bot games, 45 presets) and installed: log loss 0.442 vs 0.443 for the old one.

## Next steps (details: ROADMAP)
1. **MS1 Hog 2.6 specialist** (Colab, the user runs it): IL fine-tune `IL_NAME=crbot_il_v2_hog26`,
   `IL_INIT=<Drive>/models/crbot_il_v2/crbot_il_v2.npz`, `IL_DECK=cannon,fireball,hog_rider,ice_golem,ice_spirit,
   musketeer,skeletons,the_log`, `IL_LR=1e-4`, `IL_EPOCHS=5` → `pretrained_weights/crbot_il_v2_hog26/`; then
   `pack_ppo_bundle.py --il-model crbot_il_v2_hog26`, PPO `PPO_NAME=crbot_ppo_hog26`, `PPO_INIT=crbot_il_v2_hog26`,
   `PPO_DECKS=Hog 2.6 Cycle`, `PPO_HOURS=12`, `PPO_UPDATES=100000`, rerun daily. Re-run the gauntlet on v8.
2. **MS3 board art**: needs the user's OK to download the Fan Kit (accepting the Fan Content Policy), then a loader
   for official card / character art for the 72 cards + disclaimer. Own art was rejected 2026-11-21.
3. **MS2 coach**: value net on simulator states. Then MS5 validation, MS6 personas, evos / heroes of new cards.

## Open issues (top)
* Pocket depth after a Princess Tower falls is unverified (engine: 6 tiles past the river); ask the user.
* IL v2 value head ≈ coin flip (Brier 0.237); the coach grades with the logistic value model.
* Replay data (HF IL_Replay) has no licence: personal research only.
