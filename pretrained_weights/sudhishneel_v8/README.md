---
license: other
tags: [clash-royale, yolov8, object-detection, image-classification, dinov2]
---
# Clash Royale arena detector

Models that see the Clash Royale arena for a Hog 2.6 playing bot. Fan-made research, not affiliated with
Supercell. Training data: [SudhishNeel/clash-royale-detection](https://huggingface.co/datasets/SudhishNeel/clash-royale-detection).
Last updated 2026-10-15.

**Every card in the game as of 9 October 2026, including the brand-new ones.** The label set has 208
classes: every troop, building, spell, evolution, hero and champion. It already includes the latest
releases:

* **Evo Electro Giant** (`electro-giant-evolution`)
* **Hero Electro Wizard** (`electro-wizard-hero`)
* **Hero Ice Wizard** (`ice-wizard-hero`), labelled within a day of release
* **Hero Musketeer**, **Hero Giant**, **Hero Ice Golem**, **Hero Magic Archer**, **Hero Berserker**
* **Ronin** and **Minion Giant**

Heroes are classes in their own right, so the bot knows a Hero Musketeer from a plain Musketeer.

Built on real gameplay, not guesses:

* **Hand-verified labels:** 18,995 YouTube tracks graded one by one, plus 8,505 near-identical tracks
  auto-filled from those answers.
* **Real footage for most of the game:** 173 of the 208 classes already have labels from real matches.
* **30,000 real-arena synthetic frames** for the rarer cards and spells: real arenas, real unit cut-outs and
  see-through spell overlays, over a third with overlapping fights (the dataset's `synthetic_arena` config).
* **Strict filtering:** a label only reaches training if it survives a filter the author audits by hand.
* **Next:** a detector trained on all 208 classes.

| Folder | Model | Classes |
|---|---|---|
| `v5/` ... `v9/` | YOLOv8n detectors | v9: 187 |
| `v10/` | YOLOv8s detector | 187 |
| `crop_classifier/` | DINOv2-S unit crop classifier (second stage) | 170 + `drop` |

## Detectors (`v5/` - `v10/`)

YOLOv8 on the 568x896 arena crop at imgsz 800, with `best.pt` in each folder. The class order is the
dataset's `classes.txt`, which only appends new classes, so older ids stay valid.

| Version | Phone-frame mAP50 |
|---|---|
| v8 | 0.839 |
| v9 | 0.825 |
| v10 | 0.792 |

`v8/best_185.pt` is v8 widened to the 185-class list, with identical predictions.

v10 diagnosis at conf 0.25: 92% recall. Of its errors, 43% are missed units, 48% are false positives and
9% are the wrong class. The weak cards had almost no real training boxes; the labelling work below fixes
that.

**Next:** a 208-class detector (every card, evolution and hero as of 8 Oct 2026, heroes as `<card>-hero`
classes). It will be trained on the dataset's trusted video labels: about 29.5k trusted unit tracks across 65
YouTube videos, with untrusted tracks painted out.

## Unit crop classifier (`crop_classifier/`)

A second-stage classifier: given a unit cut out of its detector box, it names the card, or `drop` when it is
not a unit.

* **Model:** DINOv2 ViT-S/14, last 4 blocks fine-tuned, plus a linear head; 112x112 input.
* **Training:** class-balanced sampling, label smoothing 0.1, 12 epochs on CPU.
* **Data:** the author's hand grading of YouTube gameplay tracks plus near-identical auto-fills. The
  published version (2026-10-09) is trained on 24,421 labelled tracks; the accuracy figures below were
  measured on the first version (11.4k labels).

**Accuracy (2026-10-09, first version):**

| Test | Overall | Confidence ≥ 0.7 |
|---|---|---|
| Whole videos it never saw (held out) | 75.2% | 93.5% (about half of tracks) |
| Frozen DINOv2 kNN + linear probe on the same videos | 59.3% | |
| Later grades from the author, out of sample | 82.8% | 96.1% |

On those out-of-sample grades:

* Where the classifier agrees with an agent label: 94.0%.
* Confidence ≥ 0.9: 96.2%.
* Spells, when the detector's spell call is in the top 3: 88.8%.

Below about 0.5 confidence it is often wrong, so use the confidence. `predict.py` shows the input format and
`classes.json` is the label order.

**How it is used:** it decides which video tracks are trusted enough to train the detector on (see the
dataset card). It also queues its confident calls for a quick yes/no check by the author, and every answer
goes into the next retrain.
