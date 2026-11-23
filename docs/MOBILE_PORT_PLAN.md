# CRbot mobile port plan (iPhone + Android)

Status: **on hold** (2026-11-23). Accepted 2026-11-15 with a public App Store / Play Store target, but the board-art
decision of 2026-11-21 (official Fan Kit, non-commercial only) means the release model must be decided again first
(§9). Overall plan: `ROADMAP.md`. Mode B parts were removed (Mode B was dropped on 2026-11-17).

## 1. Goal and targets

CRbot on iPhone and Android that opens fast and runs smoothly, with the same simulator, AI and coach as the Mac app.

| target | value |
|---|---|
| download size | < 30 MB |
| cold start to the Fight screen | < 1 s on a 2021+ phone |
| arena rendering | 60 fps, simulation never drops a 50 ms tick |
| full coach review (~30 plays) | < 2 s on the phone |
| memory | < 250 MB during a match, < 400 MB during a review |
| battery | a 3-minute practice match costs < 1% |

## 2. Decisions

| question | decision | why |
|---|---|---|
| Keep Python on phones? | **No.** Python stays for training, data tools and the Mac dev app. | PySide6 has no iOS support and Android is experimental; every Python-on-phone option bundles an interpreter (seconds of startup, tens of MB). |
| Core language | **C++20** core library (`crbot_core`) | Runs natively on iOS, Android, macOS; ONNX Runtime has a C++ API; Python can call it through pybind11, so training uses the *same* engine. |
| Phone UI | **Flutter** (Dart), calling the core through `dart:ffi` | One UI codebase for both phones; `CustomPainter` handles the 18×32 arena at 60 fps; small plain-C API boundary. |
| C#? | Not used. | Only worth it with Unity, which is heavy for this app. |
| AI inference | ONNX Runtime Mobile (Android, iOS) or Core ML (iOS) | Same `.onnx` models we train on Colab. |
| When | **After the release decision** and a stable engine (Phase 2 cards). Phase 0 can start any time. | Porting a moving engine doubles every change. |

Alternatives considered: Rust core (equally good technically; C++ chosen for pybind11 and ONNX Runtime integration),
SwiftUI + Kotlin (best native feel, but two UIs to maintain), Kotlin Multiplatform (weaker iOS UI story).

## 3. Architecture

```
            Mac / Colab (Python)                         Phone
 ┌─────────────────────────────────────┐   ┌───────────────────────────────────┐
 │ training, data tools, golden tests  │   │ Flutter UI: Fight / Analysis /    │
 │ Mac dev app (PySide6)               │   │ Decks / Account                   │
 │        │ pybind11                   │   │        │ dart:ffi (C API)         │
 └────────┼────────────────────────────┘   └────────┼──────────────────────────┘
          ▼                                         ▼
   ┌──────────────────────────────────────────────────────────────────┐
   │ crbot_core (C++20)                                                │
   │ engine · mechanics · card DB + scaling · human delay buffer ·     │
   │ opponent elixir tracker · deck guesser · scripted bot ·           │
   │ value model · coach analyzer · match store (JSON)                 │
   │ ONNX Runtime (policy/value nets)                                  │
   └──────────────────────────────────────────────────────────────────┘
   shared data: configs/*.json (cards, rules, meta decks, value model), match JSON formats
```

The C API is small and coarse-grained (create session, play, tick N, get render state as a flat struct,
snapshot/restore, analyze, save/load match), so the per-frame FFI cost is negligible.

## 4. What moves to C++ (≈3,000 lines of Python today)

| Python module | lines | notes |
|---|---|---|
| `simulator/engine.py`, `entities.py`, `geometry.py` | ~930 | hot path; deterministic |
| `simulator/mechanics/*` | ~310 | registry becomes a static table of function pointers |
| `simulator/card_db.py` | ~320 | exact integer scaling formula |
| `brain/human_delay_buffer.py`, `scripted_bot.py` | ~510 | the AI's human limits must stay identical |
| `perception/opp_elixir_count.py`, `bayesian_deck_guesser.py` | ~355 | |
| `brain/value_model.py`, `coach_analyzer.py` | ~410 | review gets fast enough to run on-device |
| `storage/match_store.py` (formats only) | — | same JSON files, so matches move between Mac and phone |

Stays in Python: `sync_meta`, `migrate_cards_v4`, training (IL, PPO), dataset builders, `record_golden`, the Mac shell.

## 5. Phases

### Phase 0: make the Python engine port-friendly (next engine version, in Python, any time)
Bit-exact parity between two languages needs every source of platform-dependent arithmetic removed first:
1. **Own RNG.** Replace Python's `random.Random` (Mersenne Twister with Python-specific seeding, `shuffle` via
   `getrandbits`, `gauss` via `cos`/`log`) with a small project RNG (e.g. PCG32) with integer-only `shuffle` and
   a table-based or integer Box–Muller for the delay buffer. Used in `engine.py` and `human_delay_buffer.py`.
2. **No transcendental functions in the tick.** Replace `math.cos/sin` (spawn ring offsets, collision tie-break
   angle) with precomputed constant tables; replace `math.hypot` with `sqrt(dx*dx + dy*dy)` (`sqrt` is exact in
   IEEE 754, `hypot` is not guaranteed identical across libms).
3. Re-record golden hashes on purpose, bump `ENGINE_VERSION`, retrain the value model.
4. Freeze the JSON formats (cards v4, match store) with a JSON-schema file each.

### Phase 1: `crbot_core` engine with parity
1. CMake project `core/`, C++20, compiled with `-ffp-contract=off` and no fast-math (no fused multiply-add
   differences), integer milliseconds as today.
2. Port geometry, entities, mechanics, engine, card DB.
3. **Parity test:** pybind11 module `crbot_core`; a pytest runs every golden scenario through both engines and
   compares `state_hash` sequences. The 15 existing golden scenarios plus bot matches over 1,000 seeds must match.
4. Then Python switches to the C++ engine (the Python engine is kept only as the reference for parity tests),
   which also speeds up training and coach rollouts on the Mac.

### Phase 2: brain and coach in C++
Human delay buffer, opponent elixir tracker, deck guesser, scripted bot, value model, coach analyzer.
Parity: identical delay samples, estimates and review grades on recorded sessions.

### Phase 3: Flutter app
1. Screens: Fight (arena painter, hand, elixir bar, timeline, review sheet), Analysis, Decks, Account.
   One design shared with the Mac app (CRbot palette; Fan Kit art only if the release stays non-commercial).
2. The sim runs on a background isolate/thread; the UI reads a render snapshot each frame.
3. Match store in the app's private folder; optional export/import of match JSON (Mac ↔ phone).

### Phase 4: neural AI on device
Policy/value ONNX models from the IL/PPO work (target < 20 MB, < 2 ms per call) through ONNX Runtime Mobile /
Core ML. MCTS "Hard" mode only if it fits the frame budget.

## 7. Launch blockers (public release)

These must be solved before a store release; none of them blocks Phases 0–3.

1. **Supercell IP**: the Fan Content Policy allows Fan Kit art only in non-commercial fan content; the app name and
   listing must not imply endorsement; keep a visible "unofficial / not endorsed" notice with the policy link. A paid
   app or paid features would mean removing all Supercell art.
2. **Current card stats**: a stats source we are allowed to use and can refresh each balance patch.
3. **Replay data licence**: the HF replays have no licence; models trained on them may not be fit to ship.
4. **Privacy**: no account needed; player tag stays on device; a privacy policy and the store privacy labels
   ("data not collected") are still required.

## 8. Risks and concerns

1. **Parity bugs** between Python and C++ engines: mitigated by Phase 0 and hash-sequence tests on many seeds;
   any mismatch pinpoints the first differing tick.
2. **Two engines during the transition**: keep the Python engine frozen once the C++ engine passes parity;
   new mechanics are written in C++ only after that point.
3. **IP, licences and stats**: see §7 (launch blockers).
4. **Rounding parity**: Python `round()` rounds halves to even; C++ must use `std::nearbyint` (default mode), not
   `std::round`. The value model's `exp` is not correctly rounded across platforms: compare coach values with a
   tolerance, not bit-exactly.
5. **Effort**: Phase 0 is small; Phase 1–2 is the bulk (~3k lines plus tests); Phase 3 is a new UI codebase.

## 9. Open questions for the owner
1. Public release: free and non-commercial (keeps Fan Kit art), paid (own art only), or no store release at all?
2. Phone-only app, or keep the Mac app too? (The plan keeps both, sharing `crbot_core`.)
