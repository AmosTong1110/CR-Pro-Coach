# CRbot roadmap

Updated 2026-11-23. The one-page plan for the whole project. Current session state: `SESSION_HANDOFF.md`.
Specs (card schema, mechanics, observation, coach grades, card phases): `CRBOT_UPGRADE_PLAN.md`.
Phone port (on hold): `MOBILE_PORT_PLAN.md`. History: `HANDOFF_ARCHIVE.md`.

## 1. What CRbot is

A local Clash Royale practice tool for one player: a deterministic simulator, an AI sparring partner that
plays under human limits, and a post-match coach that grades every play.

It is not a bot for the real game. It never automates the live client, never plays real players, and never
reads the screen (Mode B was dropped on 2026-11-17).

Rules that stay fixed: M1 8 GB budget (no torch at runtime) · the AI sees only what a human sees and acts
through the HumanDelayBuffer · deterministic engine (goldens + `ENGINE_VERSION`) · English-only UI ·
snake_case card ids · user data stays in `user_data/` · board art = official Supercell Fan Kit only,
non-commercial, with the "unofficial / not endorsed" notice.

## 2. Status by module (2026-11-23)

| module | done | what is left |
|---|---|---|
| Simulator engine (v8, 50 ms tick) | 100% | only new mechanics for new cards |
| Cards | 72 of 123 playable | all 8 season meta decks complete (MS4 done 2026-11-23); 15 evolutions + 9 heroes simulated, 7 evolutions data-only |
| Current card stats (Nov 2026) | 92% | audited 2026-11-23 against the wiki + the 2026 balance notes; open: pocket depth, skeleton spawn spread, a few `assumed` fields |
| App shell + Fight / Analysis / Decks / Account | 95% | board art (MS3) |
| Elixir counter / deck guesser / delay buffer | 92% | minor |
| IL policy (`crbot_il_v2`, in the app) | 90% | Hog 2.6 fine-tune (MS1) |
| Self-play PPO + personas | 45% | first real PPO run (MS1), personas (MS6) |
| Coach v2 | 65% | value net, timing ±1 s, review ≤ 2 s (MS2) |
| Validation (replay agreement, Elo) | 0% | MS5 |
| **Overall** | **~75%** | |

## 3. Milestones, in order

Exit criteria marked *(proposed)* are a first suggestion. Change them if they don't fit.

### MS1. Strong sparring AI: Hog 2.6 specialist (next)
1. IL fine-tune on Hog 2.6 games only: `IL_NAME=crbot_il_v2_hog26`, `IL_INIT=crbot_il_v2.npz`,
   `IL_DECK=<8 Hog 2.6 ids>`, `IL_LR=1e-4`, `IL_EPOCHS=5` (Colab).
2. PPO from that model: `PPO_NAME=crbot_ppo_hog26`, `PPO_DECKS=Hog 2.6 Cycle`, `PPO_HOURS=12`. Run it each day;
   each run resumes from the last one.
3. Stronger gauntlet: ScriptedBot is too weak to tell configs apart (most score ≥ 95%), so add IL v2 as the
   benchmark opponent.
4. App persona "Neural (Hog 2.6)".
* Exit *(proposed)*: the PPO model beats `crbot_il_v2` in ≥ 60% of 200 gauntlet matches, still human-limited.

### MS2. Coach v2 complete
1. A better win-probability model. 2026-11-23: the logistic model, retrained on engine v8 with all 45 presets,
   ties the old one (log loss 0.442 vs 0.443, accuracy 77.6% vs 77.8% on the same 400 test games) and is installed; the logistic form has hit its ceiling. The bigger step is a value net trained on simulator states (the PPO critic, or a model
   fitted to self-play states); the IL v2 value head is too weak (Brier 0.237).
2. Timing analysis (±1 s, spec in `CRBOT_UPGRADE_PLAN.md` §7).
3. Full review ≤ 2 s on the M1.
* Exit *(proposed)*: the new value model beats the logistic one on log loss on the same held-out games, and the coach uses it.

### MS3. Official look (waiting on the user)
The user downloads the Clash Royale Fan Kit into `fankit/` (git-ignored). Then: inspect it, write a loader that
draws official card and character images on the board, and add the disclaimer + policy link. The kit has only
static art, with no animation frames.

### MS4. Cards Phase 2: all 8 meta decks playable (done 2026-11-23)
25 cards + 9 tokens from wiki pages fetched through the MediaWiki API, the 2026 balance notes on top, RoyaleAPI
game data for mass / radius / sight (`tools/card_specs/batch4.json`, mechanics in `mechanics/batch4.py`). The 7
new decks are presets ("Season meta"). Same day: audit of every older card (`tools/card_specs/audit_2026_11.json`,
`tools/apply_card_patches.py`): 9 hand-entered cards rebuilt from the wiki, ~40 balance-note fixes, Crown Tower
L12-16. Also 2026-11-23: 8 evolutions + 4 heroes of the new cards (`tools/card_specs/variants_batch4.json`,
`mechanics/variants4.py`, hero abilities in `mechanics/abilities.py`). Left: the remaining 51 catalog cards and 7
data-only evolutions; Hero Electro Wizard's numbers are estimates (unfinished wiki page).

### MS5. Validation
1. Replay agreement: re-simulate real battles whose cards are all playable and compare tower damage and results.
2. Elo ladder for all personas and models, so each new model gets a number.

### MS6. Personas
One network + persona embedding + reward shaping (Surgeon, Aggressor, Spell-Cycler, Adaptive Counter;
`CRBOT_UPGRADE_PLAN.md` §6.4). Needs MS1 working first.

### Later / on hold
* MCTS "Hard" mode (analysis + short search inside the AI's reaction window).
* Phone app (`MOBILE_PORT_PLAN.md`): on hold until the owner decides on a public release (see §5).

## 4. How the AI is trained

```
HF replays (18,570 battles)          simulator (engine v8)
        │ build_il_dataset.py               │
        ▼                                   ▼
 IL shards (1.55 M samples) ──► IL (Colab) ──► PPO self-play (Colab, daily) ──► gauntlet calibration
                                 crbot_il_v2     league: ScriptedBot / IL / snapshots     gate + cadence
                                       │                │                                   │
                                       └────────────────┴──► ONNX + .json in pretrained_weights/ ──► NeuralBot in the app
```

* Observation `il_obs_v2` (1,752 numbers, card-attribute based, so balance changes need only fine-tuning).
* Action: wait, or (hand slot, tile), masked to legal plays. Human delay buffer on in training and in play.
* PPO reward: ±1 at the end, tower-HP / crown shaping, a small leak penalty, KL to the frozen IL model.

## 5. Decisions the owner still has to make
1. Phone app and store release: the Fan Kit rules allow non-commercial use only. Free, or not at all?
2. Exit criteria for MS1 and MS2 (the proposed numbers above).
3. After MS1: specialise in more decks (one model per deck) or train one general model?
4. Pocket depth: kept at 6 tiles past the river (user decision 2026-11-23; no source gives the number).
5. Board art: the user agreed (2026-11-23) to the Fan Kit, but fankit.supercell.com is a browser portal (Frontify);
   the user downloads it into `fankit/`.
