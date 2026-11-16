# CRbot — 4-week upgrade plan

Status at start (2026-11-15): ~40% overall, 187 tests green. Infrastructure is mostly done;
the remaining work is **card/mechanic coverage**, the **trained AI brain**, and **validation on real games**.

## 0. Ground rules from the Counterpush analysis

What we take: product patterns that are common practice (deterministic sim + rollout-based review,
chess-style move grades, an accuracy score, phase-split timeline, variant flags on cards, archetype names).

What we don't take: their code, art, UI look, endpoints or data. CRbot gets its own visual identity
(see §2) and never calls their API. Several observations in the analysis are guesses from DevTools
(for example "server-side inference" or what `?knows=cut,partial` means). They don't change our design:
CRbot does its analysis **locally** and doesn't need a polling queue.

## 1. The data fact that shapes the plan

Coverage of real battles (HF replays, 1,592 battles / 3,184 decks, Dec 2025) by number of modelled cards,
choosing cards greedily to complete the most decks:

| cards | decks fully playable | battles with *both* decks playable |
|---|---|---|
| 9 (now) | 0% | 0 |
| 30 | 11% | 13 |
| 50 | 29% | 123 |
| 60 | 43% | 278 |

Consequences:
* **Imitation learning (IL) can't depend on re-simulating real battles** at 30 cards (13 battles).
  IL is trained on the **event streams** (who played what, where, when; elixir/cycle reconstructed by our
  tracker), which works for all 1,592 battles (and the 65k-replay snapshot later).
* **Self-play PPO** runs only inside the simulator, so it covers only modelled cards. Card coverage
  directly limits how good and how varied the sparring AI can be.
* Card batch 1 (data-driven, greedy order): skeletons, barbarian_barrel, fireball, the_log, electro_spirit,
  royal_ghost, ice_spirit, knight, cannon, mini_pekka, tornado, ice_golem, musketeer, hog_rider,
  mighty_miner, firecracker, earthquake, lightning, tesla, golden_knight, vines, dart_goblin, valkyrie,
  skeleton_army, rocket, miner, royal_delivery, goblin_demolisher, fire_spirit, dark_prince.
  This set makes Hog 2.6 (the most common deck, 218/3184) and several cycle decks fully playable.
* **Modern (2026) Meta deck expansion**: Double Giant Miner-Poison (`rune_giant`, `minion_giant`, `little_prince` [champion],
  `miner`, `poison`, `barbarian_barrel`, `bats`#evo, `skeletons`#evo). Adding `rune_giant`, `minion_giant`, `little_prince`
  (+ `guardienne` token), `bats`, and `poison` gives CRbot immediate coverage of the top-tier 2026 beatdown/control archetype
  alongside classic Hog 2.6.

## 2. App shell: CRbot (own look)

```
┌────────────┬─────────────────────────────────────────────┐
│ CRbot      │                                             │
│            │   page content                              │
│ ⚔ Fight    │   Fight      = arena + hand + timeline        │
│ 📈 Analysis │   Analysis   = match list → review page      │
│ 🃏 Decks    │   Decks      = recommended decks + builder    │
│            │                                             │
│            │                                             │
│ 👤 Account  │   Account    = local profile & settings      │
└────────────┴─────────────────────────────────────────────┘
```

* **Fight**: today's practice arena (persona, level 11/16, speed, undo/forward, elixir counter).
* **Analysis**: list of saved matches (CRbot practice and Mode B recordings), each with archetype names,
  result, accuracy; opens the review page (graph, grades, ghosts, *Show me* playback, *Take over*).
* **Recommended deck**: top decks from the meta prior (ranked by play rate in the data), filtered to cards
  CRbot can simulate, plus "decks that beat X" once the value net exists; one click → Fight with that deck.
* **Account** (bottom-left): local profile only: optional player tag (stays on this Mac), default level,
  calibration, model paths, data freshness ("card stats as of …").
* Visual identity: our own palette (charcoal + teal/violet accents), our own iconography; no Supercell art.
* Match IDs: `YYYYMMDD-HHMMSS_<mode>_<hash8>` with mode ∈ {practice, recording, drill}; saved under
  `matches/<id>/` as `meta.json`, `actions.json`, `review.json` (fast to reopen, no re-analysis).

## 3. Week-by-week

### Week 1: schema v4, mechanics framework, app shell
1. **Card schema v4** (§4): canonical game `internal_name`, alias table (display, HF slug, detector class,
   RoyaleAPI name), variants (`base`/`evo`/`hero`) inside the base card. Keep our snake_case ids as primary
   keys, so all 187 tests stay green.
2. **Mechanics framework** (§5): component registry plus hooks. Port the existing behaviour (rolling log,
   spells, buildings, river jump) into components first, with tests proving identical state hashes.
3. **App shell** with the four sections; Fight = current window; Analysis = match store + existing review.
4. Golden-value tests per card (L11 and L16 numbers) once current stats are available.

### Week 2: 30 cards + 2026 meta deck, deck links, archetypes, IL dataset
1. Mechanics needed by batch 1 + modern meta expansion:
   - Projectiles with travel time, splash, stun/reset, slow/freeze, knockback and pull (tornado), death spawn and death damage,
     multi-unit spawns (skeleton army), burrow/tunnel (miner, mighty miner), invisibility (royal ghost), damage-over-time and
     building multiplier (earthquake, vines root, poison lingering AoE + slow), charge and shield (dark prince), champion abilities
     (golden knight dash, mighty miner lane swap, little prince attack ramp + Royal Rescue Guardienne charge knockback),
     form change (goblin demolisher), recoil (firecracker), chain (electro spirit).
   - Modern meta unit behaviors: flying ranged tank (minion_giant, range 4.0, knockback immune), ally enhancement aura
     (rune_giant, every 3rd attack damage boost), flying melee swarm + lifesteal overheal (bats, evo bats), evo skeletons multiplying on hit.
2. **Deck link import**: parse `clashroyale://copyDeck?deck=<ids>`, mapping numeric card ids to ours.
3. **Archetype naming** (§7).
4. **IL dataset builder**: replay events → (observation, next action) samples using our OppElixirTracker,
   cycle tracker and deck guesser, so every sample sees only what a human could see.

### Week 3: brains (IL → PPO), personas
1. Train **IL policy + value** on Colab/Kaggle (§6), export ONNX, plug in as `NeuralBot` behind the same
   `Agent` interface and the existing HumanDelayBuffer.
2. **Self-play PPO**: league of checkpoints inside the simulator (30-card pool), CPU-parallel envs.
3. **Personas** as one network with a persona embedding plus reward shaping (§6.4), not 4 networks.

### Week 4: Coach v2, M1 performance, Mode B validation
1. Replace the scripted-bot win model with the **value net**; grading tiers, accuracy, timing (§7).
2. *Show me*: animate the recommended branch (fork + rollout already exist).
3. Performance budget (§8): full review in ≤ 2 s locally.
4. Mode B: run recorded matches, compare with RoyaleAPI replay timelines, start detector fine-tuning set.

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
* Output: `pretrained_weights/crbot_il_v1.onnx` (target < 20 MB).

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
| policy/value net (ONNX, CoreML) | < 2 ms per call, < 100 MB RAM |
| Mode B detector | 14 ms/frame (measured) |
| review: ~30 plays × ~30 candidates | value net with 2 s rollouts instead of 8 s, 4 worker processes → ≤ 2 s |
| app total | < 1.2 GB RAM, no torch at runtime |

## 9. Risks / open questions
1. **Current (Nov 2026) card stats**: no fresh machine-readable source yet. This blocks "verified" L16 numbers.
2. **Simulator fidelity**: the AI is only as good as the simulator matches the real game; per-card golden
   tests against real values are required.
3. **Heroes/Evos**: mechanics are new and under-documented; implement only the ones in the 30-card batch first.
4. **Data licence**: HF replays MIT (personal research); SudhishNeel detector "other". Fine for personal
   training, not for redistribution.
5. **IL from event streams** has no unit-state history; the value head is weaker until self-play adds sim states.
