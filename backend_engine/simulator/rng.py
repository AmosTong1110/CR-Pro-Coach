"""Portable deterministic RNG (PCG32, XSH-RR) for the simulator and the AI's human-delay sampling.

Replaces Python's `random.Random` so the future C++ core (docs/MOBILE_PORT_PLAN.md, Phase 0) can reproduce
every draw bit for bit:
  * integers only, mod 2**64, in the generator;
  * `random()` builds a 53-bit float exactly like CPython's genrand_res53 (exact in IEEE 754);
  * `randbelow` / `shuffle` use integer rejection sampling, no floats;
  * `gauss` is Irwin–Hall (sum of 12 uniforms − 6): additions only, no cos/log/sqrt, so no libm differences.
    Its tails stop at ±6σ, which is fine for the clipped latency distributions it feeds.
"""
from __future__ import annotations

from typing import List, MutableSequence, Tuple

_MASK64 = (1 << 64) - 1
_MULT = 6364136223846793005
_DEFAULT_SEQ = 54            # PCG reference stream id


class Pcg32:
    __slots__ = ("_state", "_inc")

    def __init__(self, seed: int = 0, seq: int = _DEFAULT_SEQ):
        self.seed(seed, seq)

    def seed(self, seed: int, seq: int = _DEFAULT_SEQ) -> None:
        """pcg32_srandom_r: same sequence for the same (seed, seq) in any language."""
        self._state = 0
        self._inc = ((seq << 1) | 1) & _MASK64
        self.next_u32()
        self._state = (self._state + (seed & _MASK64)) & _MASK64
        self.next_u32()

    def next_u32(self) -> int:
        old = self._state
        self._state = (old * _MULT + self._inc) & _MASK64
        xorshifted = (((old >> 18) ^ old) >> 27) & 0xFFFFFFFF
        rot = old >> 59
        return ((xorshifted >> rot) | (xorshifted << ((-rot) & 31))) & 0xFFFFFFFF

    def random(self) -> float:
        """Uniform in [0, 1) with 53 random bits."""
        a = self.next_u32() >> 5
        b = self.next_u32() >> 6
        return (a * 67108864.0 + b) * (1.0 / 9007199254740992.0)

    def randbelow(self, n: int) -> int:
        """Uniform integer in [0, n), 0 < n <= 2**32, without modulo bias."""
        if not 0 < n <= 1 << 32:
            raise ValueError(f"randbelow: n={n} out of range")
        limit = (1 << 32) - ((1 << 32) % n)
        while True:
            x = self.next_u32()
            if x < limit:
                return x % n

    def shuffle(self, x: MutableSequence) -> None:
        """Fisher–Yates, from the end."""
        for i in range(len(x) - 1, 0, -1):
            j = self.randbelow(i + 1)
            x[i], x[j] = x[j], x[i]

    def gauss(self, mean: float, std: float) -> float:
        s = 0.0
        for _ in range(12):
            s += self.random()
        return mean + std * (s - 6.0)

    def getstate(self) -> Tuple[int, int]:
        return self._state, self._inc

    def setstate(self, st: Tuple[int, int]) -> None:
        self._state, self._inc = int(st[0]) & _MASK64, int(st[1]) & _MASK64

    def u32s(self, n: int) -> List[int]:
        """Convenience for cross-language tests."""
        return [self.next_u32() for _ in range(n)]
