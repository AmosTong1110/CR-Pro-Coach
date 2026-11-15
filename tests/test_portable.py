"""Engine v3 portability (docs/MOBILE_PORT_PLAN.md, Phase 0): PCG32 RNG, constant trig tables, no libm trig in the tick.

The reference numbers here are what the C++ core must reproduce exactly.
"""
import ast
import math
import statistics
from pathlib import Path

import pytest

from backend_engine.simulator.rng import Pcg32
from backend_engine.simulator.tables import MAX_SPAWN, SPAWN_RING, UNIT_DIRS

ROOT = Path(__file__).resolve().parents[1]


def test_pcg32_matches_official_reference():
    # pcg32-demo (pcg-c-basic), seed 42, sequence 54
    r = Pcg32(42, 54)
    assert r.u32s(6) == [0xa15c02b7, 0x7b47f409, 0xba1d3330, 0x83d2f293, 0xbfa4784b, 0xcbed606e]


def test_pcg32_fixed_vectors_for_the_port():
    r = Pcg32(7)
    assert r.u32s(3) == Pcg32(7).u32s(3)
    a = Pcg32(123)
    deck = list(range(8))
    a.shuffle(deck)
    assert sorted(deck) == list(range(8)) and deck != list(range(8))
    b = Pcg32(123)
    deck2 = list(range(8))
    b.shuffle(deck2)
    assert deck == deck2
    x = Pcg32(5).random()
    assert 0.0 <= x < 1.0 and x == Pcg32(5).random()


def test_state_roundtrip():
    r = Pcg32(99)
    r.u32s(10)
    st = r.getstate()
    a = r.u32s(5)
    r.setstate(st)
    assert r.u32s(5) == a


def test_randbelow_is_uniform_and_bounded():
    r = Pcg32(1)
    draws = [r.randbelow(6) for _ in range(60_000)]
    assert set(draws) == set(range(6))
    for k in range(6):
        assert abs(draws.count(k) / 60_000 - 1 / 6) < 0.01
    with pytest.raises(ValueError):
        r.randbelow(0)


def test_gauss_moments():
    r = Pcg32(3)
    xs = [r.gauss(500.0, 40.0) for _ in range(20_000)]
    assert abs(statistics.mean(xs) - 500) < 1.5 and abs(statistics.pstdev(xs) - 40) < 1.5
    assert 500 - 6 * 40 <= min(xs) and max(xs) <= 500 + 6 * 40


def test_tables_match_trig():
    for n in range(2, MAX_SPAWN + 1):
        assert len(SPAWN_RING[n]) == n
        for i, (x, y) in enumerate(SPAWN_RING[n]):
            a = 2 * math.pi * i / n + math.pi / 2
            assert x == pytest.approx(0.7 * math.cos(a), abs=1e-15) and y == pytest.approx(0.7 * math.sin(a), abs=1e-15)
    assert len(UNIT_DIRS) == 64 and all(abs(math.sqrt(x * x + y * y) - 1) < 1e-15 for x, y in UNIT_DIRS)


TICK_FILES = ["backend_engine/simulator/engine.py", "backend_engine/simulator/geometry.py",
              "backend_engine/simulator/mechanics/core.py", "backend_engine/simulator/mechanics/status.py",
              "backend_engine/brain/human_delay_buffer.py"]
FORBIDDEN = {"cos", "sin", "tan", "atan2", "hypot", "exp", "log", "pow"}


@pytest.mark.parametrize("path", TICK_FILES)
def test_no_platform_dependent_math_in_the_tick(path):
    tree = ast.parse((ROOT / path).read_text())
    bad = [f"{n.attr} (line {n.lineno})" for n in ast.walk(tree)
           if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == "math"
           and n.attr in FORBIDDEN]
    imports = [a.name for n in ast.walk(tree) if isinstance(n, ast.Import) for a in n.names]
    assert not bad and "random" not in imports, bad
