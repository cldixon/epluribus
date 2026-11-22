"""Small, dependency-light statistics used by the aggregation layer."""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Interval:
    """A point estimate with a 95% interval and the sample size behind it.

    Attributes:
        value: The estimate: a share, mean or median.
        lo: Lower end of the 95% interval.
        hi: Upper end of the 95% interval.
        n: Number of answers the estimate is based on.
    """

    value: float
    lo: float
    hi: float
    n: int

    def __repr__(self) -> str:
        return f"{self.value:.3f} [{self.lo:.3f}, {self.hi:.3f}] (n={self.n})"

    def __float__(self) -> float:
        return float(self.value)

    def __format__(self, spec: str) -> str:
        if not spec:
            return repr(self)
        return f"{format(self.value, spec)} [{format(self.lo, spec)}, {format(self.hi, spec)}]"

    @property
    def width(self) -> float:
        """Width of the interval, `hi - lo`."""
        return self.hi - self.lo


def wilson(k: int, n: int, z: float = 1.96) -> Interval:
    """Share `k / n` with a Wilson score interval, which behaves well at small `n`."""
    if n == 0:
        return Interval(math.nan, math.nan, math.nan, 0)
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return Interval(p, max(0.0, c - h), min(1.0, c + h), n)


def newcombe_diff(k1: int, n1: int, k2: int, n2: int, z: float = 1.96) -> Interval:
    """Difference of two shares, `k1/n1 - k2/n2`, with Newcombe's hybrid score interval."""
    a, b = wilson(k1, n1, z), wilson(k2, n2, z)
    if a.n == 0 or b.n == 0:
        return Interval(math.nan, math.nan, math.nan, min(a.n, b.n))
    d = a.value - b.value
    lo = d - math.sqrt((a.value - a.lo) ** 2 + (b.hi - b.value) ** 2)
    hi = d + math.sqrt((a.hi - a.value) ** 2 + (b.value - b.lo) ** 2)
    return Interval(d, max(-1.0, lo), min(1.0, hi), min(n1, n2))


Stat = Callable[..., object]


def bootstrap(
    values: Sequence[float] | np.ndarray,
    stat: Stat = np.mean,
    reps: int = 2000,
    seed: int = 0,
) -> Interval:
    """Percentile bootstrap interval for a statistic such as `np.mean` or `np.median`.

    The statistic must accept an `axis` keyword, as NumPy reductions do.
    """
    x = np.asarray(values, dtype=float)
    n = len(x)
    if n == 0:
        return Interval(math.nan, math.nan, math.nan, 0)
    point = float(stat(x))  # ty: ignore[invalid-argument-type]
    if n == 1:
        return Interval(point, point, point, 1)
    rng = np.random.default_rng(seed)
    boots = np.asarray(stat(x[rng.integers(0, n, size=(reps, n))], axis=1))
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return Interval(point, float(lo), float(hi), n)


def bootstrap_diff(
    a: Sequence[float] | np.ndarray,
    b: Sequence[float] | np.ndarray,
    stat: Stat = np.mean,
    reps: int = 2000,
    seed: int = 0,
) -> Interval:
    """Difference `stat(a) - stat(b)` with a percentile bootstrap interval."""
    x, y = np.asarray(a, dtype=float), np.asarray(b, dtype=float)
    n = min(len(x), len(y))
    if n == 0:
        return Interval(math.nan, math.nan, math.nan, 0)
    point = float(stat(x)) - float(stat(y))  # ty: ignore[invalid-argument-type]
    rng = np.random.default_rng(seed)
    bx = np.asarray(stat(x[rng.integers(0, len(x), size=(reps, len(x)))], axis=1))
    by = np.asarray(stat(y[rng.integers(0, len(y), size=(reps, len(y)))], axis=1))
    lo, hi = np.percentile(bx - by, [2.5, 97.5])
    return Interval(point, float(lo), float(hi), n)


def entropy(probs: Sequence[float]) -> float:
    """Shannon entropy in bits. 0 means unanimous; `log2(k)` means uniform over k options."""
    p = np.asarray([q for q in probs if q > 0], dtype=float)
    return float(-(p * np.log2(p)).sum()) if len(p) else 0.0
