from __future__ import annotations

import math

import numpy as np

from epluribus.stats import Interval, bootstrap, bootstrap_diff, entropy, newcombe_diff, wilson


def test_wilson():
    iv = wilson(25, 40)
    assert iv.value == 0.625
    assert 0.46 < iv.lo < 0.48
    assert 0.75 < iv.hi < 0.77
    assert iv.n == 40
    assert math.isnan(wilson(0, 0).value)
    edge = wilson(0, 10)
    assert edge.lo == 0.0
    assert edge.hi > 0


def test_bootstrap():
    iv = bootstrap([1, 2, 3, 4, 5], np.mean)
    assert iv.value == 3
    assert iv.lo < 3 < iv.hi
    assert bootstrap([7], np.mean) == Interval(7, 7, 7, 1)
    assert bootstrap([], np.mean).n == 0
    assert bootstrap([1, 2, 3, 100], np.median).value == 2.5


def test_differences():
    d = newcombe_diff(30, 40, 10, 40)
    assert d.value == 0.5
    assert 0 < d.lo < 0.5 < d.hi <= 1
    assert math.isnan(newcombe_diff(1, 0, 1, 2).value)
    b = bootstrap_diff([5, 6, 7], [1, 2, 3])
    assert b.value == 4
    assert b.lo > 0
    assert bootstrap_diff([], [1]).n == 0


def test_entropy_and_format():
    assert entropy([1.0, 0.0]) == 0.0
    assert entropy([0.5, 0.5]) == 1.0
    assert entropy([]) == 0.0
    iv = Interval(0.5, 0.25, 0.75, 10)
    assert f"{iv:.1f}" == "0.5 [0.2, 0.8]"
    assert f"{iv}" == repr(iv) == "0.500 [0.250, 0.750] (n=10)"
    assert float(iv) == 0.5
    assert iv.width == 0.5
