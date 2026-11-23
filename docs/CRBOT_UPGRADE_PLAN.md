# CRbot — design reference

Specs that the code follows. The plan (what to build next, in which order) is `ROADMAP.md`; the current state
is `SESSION_HANDOFF.md`. Sections 0–3 (the original 4-week plan, the 2025 coverage table, the app-shell sketch,
the week-by-week list) were removed on 2026-11-23: done or replaced by `ROADMAP.md` (see git history).
Section numbers below are kept so older references still match.

Ground rule (from the Counterpush analysis): take only common product patterns (deterministic sim +
rollout-based review, chess-style grades, accuracy score, archetype names). Never their code, look,
endpoints or data; CRbot analyses locally.

## 4. Card schema v4

```json
"elite_barbarians": {
  "name": "Elite Barbarians",
  "internal_name": "AngryBarbarians",
  "aliases": {"hf": "elite-barbarians", "detector": ["elite-barbarian", "elite-barbarian-evolution"],
              "royaleapi": "elite-barbarians"},
  "type": "troop", "rarity": "common", "elixir": 6,
  "base_stats": {"hitpoints": 0, "damage": 0},
  "count": 2,
  "attack": {...}, "movement": {...},
  "mechanics": [{"kind": "melee"}],
  "variants": {
    "evo":  {"cycles_required": 1, "mechanics": [{"kind": "...", "params": {}}], "stat_mods": {}},
    "hero": null
  },
  "data_as_of": "2026-11", "verified": false
}
```

* References are `CardRef(card_id, variant)`; string form `elite_barbarians#evo` (for logs and URLs).
* `PublicDeployEvent` gains `variant: "base"|"evo"|"hero"` (replaces `is_evolved`, kept as a property).
* `sync_meta.py` gets a source adapter per data provider; stats stay `base_stats` with the exact
  multiplier table, so "Nov 2026 L16" means only fresher base numbers. **Open issue: a current stats
  source.** RoyaleAPI's repo is from 2023; we need per-card current values (your browser copy of
  RoyaleAPI card pages, or another provider).

## 5. Mechanics framework

```python
class Mechanic(Protocol):
    def on_spawn(self, ctx, unit): ...
    def on_tick(self, ctx, unit, dt): ...
    def on_attack(self, ctx, unit, target) -> AttackResult: ...  # projectile / splash / chain
    def on_damaged(self, ctx, unit, amount, source): ...         # shields, resets
    def on_death(self, ctx, unit): ...                            # death spawn / damage
REGISTRY = {"projectile": Projectile, "splash": Splash, "stun": Stun, "knockback": Knockback, ...}
```

* Status effects (stun, slow, root, invisible) are timed tags on the unit, checked in one place by the
  engine (`can_move`, `can_attack`, `targetable`).
* Order inside a tick stays fixed and uid-sorted, so the engine remains deterministic.
* Tests: one unit test per mechanic, one golden scenario per card ("Log knocks back Skeleton Army,
  kills all"), plus the existing replay and snapshot determinism tests.

## 6. Training pipeline

### 6.1 Observation and action (human-legal only)
* Observation: own hand, next card, own elixir; visible units (card, side, x, y, hp fraction, variant);
  opponent estimate from OppElixirTracker (point/low/high, queue, known hand, evo progress); deck-guess
  top-k; time and phase; tower HPs. Card features are **parametric** (cost, hp, dps, range, speed, flags)
  rather than ids, so balance patches only need fine-tuning.
* Action: `wait` or (hand slot, tile). Tiles: 18×15 own half plus spell tiles; masking from
  `check_play`. Factorised heads: P(wait / card) then P(tile | card).

### 6.2 Imitation learning
* Data: all HF battles (event streams). Each play becomes a sample; "wait" samples are drawn between plays.
* Model: small transformer over the unit set (≈2–4M params) with policy and value heads.
  Value target = final result. Colab T4 ≈ 1–2 h.
* Output: `pretrained_weights/crbot_il_v2/` (`.onnx` + `.npz` + `.json`, target < 20 MB). Current model and
  metrics: `SESSION_HANDOFF.md`.

### 6.3 Self-play PPO
* Init from IL. Simulator in subprocess workers (Kaggle/Colab CPUs; the sim runs ~13k ticks/s per core).
* League: latest plus frozen snapshots plus the scripted bot; 15-checkpoint pool, sampled by win rate.
* The HumanDelayBuffer stays on during training, so the agent learns under human reaction limits.

### 6.4 Personas (one net + persona embedding + reward shaping)
| persona | shaping |
|---|---|
| Surgeon | +elixir-trade value, −leak, −risky pushes |
| Aggressor | +damage dealt within 10 s after an opponent spend ≥ 2, +opposite-lane pressure |
| Spell-Cycler | +spell tower damage, +cycle speed, bonus for 2x/3x spell finishes |
| Adaptive Counter | trained against the player's own decks (from Analysis history) |

### 6.5 MCTS
Use it for analysis only, plus a "Hard" mode: short-horizon search (policy-guided, value-net leaf
evaluation) during big fights. It runs during the AI's own reaction window, so it never gives the AI
superhuman reaction speed.

## 7. Coach v2

### Grades (CRbot's own tiers; loss = win-prob points vs best)
| tier | rule |
|---|---|
| **!! Brilliant** | best move, and it beats the median alternative by ≥ 12 pts and waiting by ≥ 9 pts |
| **! Great** | best move, and the 2nd-best *different card* is ≥ 8 pts worse (the only good move) |
| **★ Best** | loss ≤ 1 pt |
| **Excellent** | 1–3 pts |
| Good (unmarked) | 3–5 pts |
| **?! Inaccuracy** | 5–10 pts |
| **? Mistake** | 10–20 pts |
| **?? Blunder** | > 20 pts |

### Accuracy (0–100)
`score_i = 100·max(0, 1 − loss_i/25)`; match accuracy = weighted mean with
`w_i = 0.5 + min(1, spread_i/10)` (spread = std of candidate values), so critical moments count more.

### Timing analysis (±1 s)
For each play, also evaluate the same card and tile at t−1 s (restore the 1 s checkpoint if it was
already in hand and affordable) and at t+0.5 s / t+1 s (from the pre-play state, wait, then play).
If a timing variant beats the actual play by ≥ 3 pts, report "too early/late by X s".

### Archetype names (own rules)
`[Evo/Hero tag][win condition] [key building or spell] [avg elixir if cycle]`, built from a win-condition
table (hog_rider→Hog, royal_giant→RG, golem→Golem, lava_hound→Lava, balloon→Loon, x_bow→XBow,
graveyard→GY, goblin_barrel→Bait, miner→Miner, rune_giant→RuneGiant, minion_giant→MinionGiant,
little_prince→LP…), e.g. "EvoRoyalHogs GobHut", "Hog 2.6", "Double Giant Miner Poison".

## 8. M1 budget (8 GB)

| part | target |
|---|---|
| policy/value net (ONNX) | < 2 ms per call, < 100 MB RAM (NeuralBot measures ~4–5 ms per decision) |
| review: ~30 plays × ~30 candidates | value net with 2 s rollouts instead of 8 s, 4 worker processes → ≤ 2 s |
| app total | < 1.2 GB RAM, no torch at runtime |

## 9. Risks / open questions
1. **Current (Nov 2026) card stats**: no fresh machine-readable source; many values are still `assumed`.
2. **Simulator fidelity**: the AI is only as good as the simulator matches the real game; per-card golden
   tests against real values are required (and the replay-agreement benchmark, `ROADMAP.md` MS5).
3. **Heroes/Evos**: mechanics are new and under-documented.
4. **Data licence**: HF replays have no clear licence: personal research only, no redistribution.
5. **IL from event streams** has no unit-state history; its value head is weak (Brier 0.237) until a value
   model is trained on simulator states.

## 10. Phased card expansion to 8 complete meta decks

Target decks = the 8 archetypes in `configs/meta_decks_top200.json` (season 2026 top-200), 51 distinct base cards.
Phase 1 (38 cards) finished 2026-11-16. **Phase 2 finished 2026-11-23: all 8 decks are playable (72 cards).**
Specs: `tools/card_specs/batch4.json`; mechanics: `backend_engine/simulator/mechanics/batch4.py`; tests:
`tests/test_batch4.py`. The `healing_spirit` id of the top-200 file is the game's `heal_spirit`.

Mechanics as built (2026-11-23):
1. **Building-only targeting** (existing): royal_giant, balloon, lava_hound, battle_ram, golem (+ golemites).
2. **Death effects**: `death_bomb` (balloon, giant_skeleton: bomb 3 s later), `death_damage` + knockback (golem),
   `death_spawn` (golem, lava_hound, battle_ram, goblin_cage, goblin_drill, night_witch).
3. **Spawners**: `periodic_spawn` (goblin_drill, night_witch), `barrel_drop` spell (goblin_barrel), card
   `companions` (goblin_gang's Spear Goblins).
4. `ramp_damage` (inferno_dragon), `pierce_shot` (magic_archer), `hook` (fisherman, no slow since 2026-04-06).
5. `charge` + `break_on_hit` (battle_ram), `dash_attack` (bandit).
6. `shield` (guards), `heal_burst` + `heal_zone` (heal_spirit), `rage_on_death` + `rage_zone` (lumberjack, +30%),
   `spawn_zap` + `split_attack` (electro_wizard), `emerge_damage` (goblin_drill).
7. `rebirth` + `hatch` (phoenix), `combo_hit` + ability `pensive_protection` (monk), `volley_spell` (arrows).

### Evolutions and heroes
Simulated: 7 of 14 evolutions (Knight, Firecracker, Skeletons, Zap, Valkyrie, Tesla, Mortar) and 5 heroes (Knight,
Mini P.E.K.K.A., Ice Golem, Musketeer, Valkyrie). Data only, cheapest first: bats (heal pulse), ice spirit, archers,
then the ones that add a new entity (cannon barrage, dart goblin poison, royal ghost soldiers, skeleton army General
Gerry); Barbarian Barrel / Dark Prince heroes (HANDOFF_ARCHIVE issue 18).
* Pages still to save for Phase 2 evos: battle_ram, goblin_barrel, goblin_cage, goblin_drill, inferno_dragon,
  lumberjack, princess, royal_giant. Hero pages: electro_wizard, balloon, magic_archer, mega_minion.

### Stats source
`sync_meta --html-dir data/raw_sources/cards` parses the saved Fandom pages (current numbers, L1–L16) and checks
L11 / L16 against the DB (read-only); `--apply-wiki` stores the numbers as `level_overrides`.
* Fetching pages: clashroyale.fandom.com refuses plain page requests (403) but its MediaWiki API works:
  `api.php?action=parse&page=<Title>&prop=text|revid&format=json&formatversion=2`; save the `text` as
  `<Title> _ Clash Royale Wiki _ Fandom.html` (Crown Tower pages go to `data/raw_sources/towers/`).
* Wiki tables can lag the official balance notes (supercell.com blog). Newer official values go into a patch file
  applied by `tools/apply_card_patches.py` (each line names its source); the card lists the stat in
  `newer_than_wiki`, and `verify_wiki` reports it as `newer` instead of `DIFF`.
* Fields the wiki does not list (mass, collision radius, sight, knockback, spawn delays): RoyaleAPI cr-api-data
  `cards_stats.json` (decoded game files, 2023-10: reliable for mechanics, stale for numbers).
