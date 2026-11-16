# CR-Pro-Coach 🏆
**Professional AI Sparring Partner & Post-Match Review Coach for Clash Royale**

Engineered for 13,050+ elite ladder gameplay. Integrates open-source model weights, a lightweight high-speed deterministic battle simulator, human physiological reaction delay buffers, and a non-invasive post-match review engine.

---

## Architecture & Directory Structure

```text
CR-Pro-Coach/
│
├── 📂 configs/                       # [1. Dynamic Version & Meta Configuration]
│   ├── cards_stats_v2026.json        # Full card base stats + Level 11 / Level 16 scaling curves
│   ├── mechanics_tags.json           # Atomic card mechanic tags (Evolutions, Champions, Charge, etc.)
│   ├── meta_decks_top200.json        # Current season Top 200 ladder decks (used for Bayesian deck guessing)
│   └── human_reaction_profile.yaml   # Human physiological reaction limits (surprise 450ms, hover 150ms)
│
├── 📂 pretrained_weights/            # [2. Model Weights & Downloader]
│   ├── download_weights.py           # Automated model weight downloader script
│   ├── policy_bc_v4.pt               # Downloaded from Cochon123/clash-royale-policy-bc-v4 (~2.3MB)
│   ├── action_clock_v1.pt            # Downloaded from Cochon123/clash-royale-action-clock-v1 (human delay timing)
│   ├── winner_predictor.pt           # Downloaded from Cochon123/clash-royale-winner-predictor (win-rate curve engine)
│   ├── katacr_yolov8_detector.onnx   # Read-only game screen detector (for live screen capture review)
│   └── pro_selfplay_ppo_latest.onnx  # Self-play PPO sparring policy model
│
├── 📂 backend_engine/                # [3. Core Python Engine & Simulator (<500MB RAM footprint)]
│   ├── 📂 simulator/                 # High-speed deterministic game logic engine
│   │   ├── arena_grid.py             # 18x32 arena grid and pathfinding collision logic (36x64 half-tile precision)
│   │   ├── card_db.py                # Card database with level scaling, spells, and evolution mechanics
│   │   ├── engine.py                 # Core deterministic simulation loop (unit physics, projectile tracking, towers)
│   │   ├── geometry.py               # Coordinate system, half-tile math, and bounding boxes
│   │   └── sync_meta.py              # Script to sync balance updates and meta changes from RoyaleAPI
│   │
│   ├── 📂 perception/                # Human-perspective perception layer (fair play, no cheats)
│   │   ├── opp_elixir_count.py       # Real-time opponent elixir counter and cycle tracker based on play events
│   │   ├── bayesian_deck_guesser.py  # Bayesian posterior probability inference for unknown enemy cards
│   │   └── screen_reader_cv.py       # Read-only screen capture + ONNX detector (strictly non-intrusive)
│   │
│   ├── 📂 brain/                     # Decision-making & Post-Match Coaching Core
│   │   ├── human_delay_buffer.py     # 0.5s human reaction latency buffer queue integrated with action_clock_v1
│   │   ├── ppo_agent.py              # Policy network inference + action masking (blocks illegal spots/low elixir)
│   │   ├── coach_analyzer.py         # Post-match coach: win-rate delta, blunder detection, and candidate rollouts
│   │   └── scripted_bot.py           # Scripted baseline personas (The Surgeon, The Aggressor, Spell-Cycler)
│   │
│   └── server_bridge.py              # Local WebSocket / JSON-RPC server (bridges with UI clients)
│
├── 📂 cloud_training/                # [4. Cloud Training & Evaluation Notebooks (Google Colab / Cloud)]
│   ├── train_imitation_bc.ipynb      # Stage 1: Behavioral Cloning (BC) from top player replays
│   ├── train_selfplay_ppo.ipynb      # Stage 2: Self-play Reinforcement Learning (PPO)
│   └── gauntlet_eval.py              # Gauntlet tournament evaluation for policy iterations
│
└── 📂 frontend_app/                  # [5. Frontend Client Applications]
    └── 📂 qt/                        # Native PySide6 Desktop Application
        ├── app.py                    # Main sparring window, timeline scrubber, and controls
        ├── board.py                  # 18x32 arena canvas, ghost tile visualizations, and range rings
        ├── review.py                 # Win-rate graph, move-by-move evaluation panel, and take-over engine
        └── session.py                # Game session orchestrator, rewind/fast-forward, and state history
```

---

## Quickstart

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Model Assets
```bash
python pretrained_weights/download_weights.py
```

### 3. Launch the Desktop Coach App
```bash
python -m frontend_app.qt.app
```

### 4. Run the Full Test Suite
```bash
pytest
```

---

## Key Features

- **Accurate Arena Simulation**: 18x32 arena grid, half-tile pathfinding, projectile collisions, unit mass pushing, and tower defense logic.
- **Post-Match Chess-Style Analysis**: Evaluates every move with win-probability swing scores (`!! Brilliant`, `★ Best`, `Good`, `?! Inaccuracy`, `? Mistake`, `?? Blunder`).
- **Interactive Takeover**: Scrub the timeline back to any critical blunder and take control to practice better responses.
- **Opponent Tracking & Quiz**: Track opponent elixir count, rotation cycle, and Bayesian deck archetypes, or enable Quiz Mode to train your mental tracking skills.
