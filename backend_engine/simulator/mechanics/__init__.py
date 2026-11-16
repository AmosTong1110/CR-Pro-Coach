"""Card mechanics as components (plan §5).

* Unit mechanics (`REGISTRY`): stateless singletons with hooks the engine calls at fixed points of a tick.
  Per-unit data lives on the unit (`params[kind]`, `mstate`), so snapshots stay plain dataclasses.
* Spell kinds (`SPELLS`): how a spell card is cast and how its entity advances / lands.
* Status effects (`status`): timed tags (stun, slow, root, invisible) checked by the engine in one place.

Importing this package registers the built-in mechanics.
"""
from backend_engine.simulator.mechanics.base import (  # noqa: F401
    DESCRIPTIVE, HOOKS, PLANNED, REGISTRY, SPELLS, Mechanic, SpellKind, hooks_for, known_kinds, register,
    register_spell,
)
from backend_engine.simulator.mechanics import core, effects  # noqa: F401,E402  (registers built-ins)
