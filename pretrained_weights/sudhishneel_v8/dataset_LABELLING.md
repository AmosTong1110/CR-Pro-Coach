# Labelling conventions (class design)

Applies to the detector classes and to the published dataset (`SudhishNeel/clash-royale-detection`). Last updated 2026-10-08.

## 1. A class is what appears on the field, not what is in the card deck

One box = one object you can see in the arena: a troop, a building, a spell effect, a tower. A card that spawns several
units therefore has **no class of its own**; each spawned unit uses the unit's class.

| Card (what the player plays) | Classes on the field |
|---|---|
| Skeleton Army | `skeleton` (one box per skeleton) |
| Goblin Gang | `goblin`, `spear-goblin` (spear goblins carry a striped spear case on their back) |
| Minion Horde | `minion` (one box per minion) |
| Three Musketeers | `musketeer` |
| Barbarians, Archers, Guards, Royal Hogs ... | `barbarian`, `archer`, `guard`, `royal-hog` |
| Mirror (spell, no unit) | none |
| Skeleton Army **evolution** | the evolved skeleton unit is `skeleton-evolution`; General Gerry (purple cape, skull banner spear, wooden shield) is `skeleton-army-evolution` |
| Minion Horde **evolution** (Apr 2026) | `minion-evolution` (one box per evolved minion) |

Reason: a box can only carry one class, and the same minion cannot be both "minion" and "minion horde". Per-unit classes
keep labels consistent and let the detector count units.

## 2. Evolutions, heroes, champions

* **Evolution**: class `<card>-evolution`. On the field the unit's level badge has a **diamond**.
* **Hero** (since 2026-10-08): class `<card>-hero` (`knight-hero`, `giant-hero`, `mini-pekka-hero`, `musketeer-hero`,
  `wizard-hero`, `ice-golem-hero`, `goblin-hero`, `mega-minion-hero`, `barbarian-barrel-hero`, `magic-archer-hero`,
  `balloon-hero`, `dark-prince-hero`, `bowler-hero`, `tombstone-hero`, `valkyrie-hero`, `berserker-hero`,
  `ice-wizard-hero`, `electro-wizard-hero`). The level badge shows a **crown**. A hero card is played as the hero every
  time (unlike evolutions, which apply on selected plays in the cycle), so every unit of that card in a hero deck is the hero.
  Known looks: Hero Giant has golden hair; Hero Ice Golem has separate gold arm pieces (labelled as part of the same unit).
* **Champion**: own class (`golden-knight`, `skeleton-king`, `archer-queen`, `mighty-miner`, `monk`, `little-prince`,
  `goblinstein`, `boss-bandit`, `terry`, ...).
* **Normal card**: square level badge.

*Data note:* labels made before 2026-10-08 used the base card's class for heroes (e.g. Hero Wizard = `wizard`). Those units
are being relabelled to the hero classes; frames not yet relabelled may still carry the old base-card label.

## 3. Towers, tower troops, UI

`king-tower`, `queen-tower`, `cannoneer-tower`, `dagger-duchess-tower`, `royal-chef` (tower troops) and a few UI classes
(`clock`, `elixir`, ...) are labelled. Card-name text, deploy clocks, see-through placement previews, damage numbers, emotes,
streamer overlays and death/hit effects are **not** labelled (they are dropped / painted out in the training copy).

## 4. Class ids are permanent

`classes.txt` order never changes; new classes are only appended. The v1 release has 187 classes. Classes appended since:
`electro-giant-evolution`, the 18 `<card>-hero` classes, `royal-chef`, `terry` (208 in total). They enter the published
files with the next data release.

## 5. How labels are produced

Detector, then tracking, then per-track naming by agents (every card in the video's decks is looked up online), then human
grading of the unsure tracks. Tracks that stay unsure are excluded from training.
