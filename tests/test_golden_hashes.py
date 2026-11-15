"""Engine behaviour is pinned by recorded state hashes (see backend_engine/simulator/golden.py).

If this fails after a refactor, behaviour changed. If the change is intentional, re-record with
`python tools/record_golden.py --write` and review the diff.
"""
import json
from pathlib import Path

import pytest

from backend_engine.simulator.card_db import CardDB
from backend_engine.simulator.golden import scenarios

GOLDEN = json.loads((Path(__file__).parent / "golden" / "engine_hashes.json").read_text())
_DB = CardDB.load()
_SCEN = scenarios(_DB)


def test_every_scenario_is_recorded():
    assert set(GOLDEN) == set(_SCEN)


@pytest.mark.parametrize("name", sorted(GOLDEN))
def test_golden_hashes(name):
    assert _SCEN[name]() == GOLDEN[name]
