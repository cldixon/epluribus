from __future__ import annotations

import numpy as np
import pytest

import epluribus as ep
from epluribus.records import Record, ResponseMeta

from .conftest import Everything, Vote


@pytest.fixture(scope="module")
def rs() -> ep.ResultSet:
    task = ep.Task("Question about $x", Everything)
    models = [ep.FakeClient("m1", pricing=(1.0, 2.0)), ep.FakeClient("m2", error_rate=0.1)]
    return ep.run(task, models, n=15, grid={"x": [1, 2]}, max_retries=0)


def test_aggregations(rs):
    f = rs["cat"].freq()
    assert set(f) == {"a", "b", "c"}
    assert abs(sum(e.value for e in f.values()) - 1) < 1e-9
    assert 1 <= rs["likert"].mean().value <= 5
    assert rs["marked"].mean().value <= 2  # text ordinal levels use ranks 0..2
    assert rs["flag"].rate().n == rs["flag"].n
    assert rs["multi"].array().shape == (60, 2)
    assert rs["multi"].mean_selected().value <= 2
    assert rs["cat"].mode() in {"a", "b", "c"}
    assert 0 <= rs["cat"].entropy() <= np.log2(3)
    assert set(rs["num"].quantiles((0.5,))) == {0.5}
    assert rs["num"].std() >= 0
    assert len(rs["text"].sample(3)) == 3
    assert rs["level"].kind == ep.Kind.ORDINAL


def test_kind_errors(rs):
    with pytest.raises(ep.KindError, match="categorical"):
        rs["cat"].mean()
    with pytest.raises(ep.KindError):
        rs["text"].freq()
    with pytest.raises(ep.KindError):
        rs["num"].rate()
    with pytest.raises(ep.KindError):
        rs["text"].diff(rs["text"])
    with pytest.raises(KeyError, match="fields"):
        rs["nope"]


def test_arrays_mark_missing(rs):
    arrays = rs.arrays()
    missing = [not r.ok for r in rs]
    assert all(arrays["cat"][missing] == -1)
    assert np.isnan(arrays["likert"][missing]).all()
    assert arrays["text"].dtype == object
    assert arrays["flag"].min() >= -1


def test_group_filter_and_diff(rs):
    g = rs.group_by("model")
    assert set(g.groups) == {"m1", "m2"}
    assert len(g) == 2
    assert sum(g.size().values()) == len(rs)
    assert set(g["cat"].freq()) == {"m1", "m2"}
    assert set(g["likert"].mean()) == {"m1", "m2"}
    for method in ("counts", "mode", "entropy", "summary", "values", "sample"):
        assert len(getattr(g["cat"], method)()) == 2
    assert len(g["flag"].rate()) == 2
    assert len(g["num"].median()) == 2
    assert len(g["num"].std()) == 2
    assert len(g["num"].quantiles()) == 2
    assert len(g["multi"].mean_selected()) == 2
    assert sum(g["cat"].n.values()) + sum(g["cat"].missing.values()) == len(rs)
    assert set(g.usage()) == {"m1", "m2"}
    assert len(g.get("m1")) == 30
    assert dict(iter(g)).keys() == g.groups.keys()

    two = rs.group_by("model", "x")
    assert ("m1", 1) in two.groups
    assert "m1" in repr(two["cat"].freq())

    d = g["cat"].diff("m1", "m2")
    assert set(d) == {"a", "b", "c"}
    assert isinstance(g["likert"].diff("m1", "m2"), ep.Interval)
    assert isinstance(g["flag"].diff("m1", "m2"), ep.Interval)
    assert set(g["multi"].diff("m1", "m2")) == {"x", "y"}

    only = rs.filter(model="m1", x=1)
    assert len(only) == 15
    assert len(rs.filter(lambda r: r.coords["x"] == 2)) == 30
    with pytest.raises(KeyError, match="no record has"):
        rs.group_by("modle")
    with pytest.raises(KeyError):
        rs.filter(modle="m1")
    with pytest.raises(ValueError, match="at least one"):
        rs.group_by()
    assert rs.group_by("usage.input_tokens")  # meta paths work too


def test_usage_and_cost(rs):
    u = rs.usage()
    assert u["records"] == 60
    assert u["errors"] == len(rs.errors)
    assert u["input_tokens"] > 0
    m1 = rs.filter(model="m1").usage()
    assert m1["cost_usd"] > 0
    assert len(rs.ok) == 60 - u["errors"]


def test_display(rs):
    text = repr(rs)
    assert "ResultSet Everything" in text
    assert "categorical" in text
    assert "<table>" in rs._repr_html_()
    freq = rs["cat"].freq()
    assert "95% interval" in repr(freq)
    assert "<table>" in freq._repr_html_()
    assert freq.table() == repr(freq)
    grouped = rs.group_by("model")["likert"].mean()
    assert "95% interval" in repr(grouped)
    assert "m1" in grouped._repr_html_()
    assert "approve" not in repr(rs.usage())
    assert "errors" in repr(rs.usage())
    assert repr(ep.Shares()) == "Shares({})"
    assert "<ResultSet" in repr(ep.ResultSet(Vote, []))
    assert "ResultSet Vote" in ep.ResultSet(Vote, [])._repr_html_()
    assert "Grouped" in repr(rs.group_by("model"))
    assert "GroupedField" in repr(rs.group_by("model")["cat"])
    assert "FieldView" in repr(rs["cat"])


def test_pandas_exports(rs):
    df = rs.to_pandas()
    assert len(df) == 60
    assert {"model", "x", "cat", "meta.provider", "usage.input_tokens", "error_kind"} <= set(
        df.columns
    )
    freq = rs["cat"].freq().to_pandas()
    assert list(freq.columns) == ["level", "share", "lo", "hi", "n"]
    by = rs.group_by("model")["likert"].mean().to_pandas()
    assert list(by.columns) == ["model", "value", "value_lo", "value_hi", "value_n"]
    shares = rs.group_by("model", "x")["cat"].freq().to_pandas()
    assert {"model", "x", "a", "a_lo"} <= set(shares.columns)
    assert len(rs.summary().to_pandas()) == len(rs.summary())
    assert len(rs.group_by("model")["cat"].mode().to_pandas()) == 2


def test_record_get_and_flat():
    out = Vote(choice="approve", confidence=3, reason="r")
    meta = ResponseMeta(provider="p", model="m", raw={"seed_sent": False})
    r = Record(out, meta, {"model": "label", "reason": "coord"}, 0, "prompt")
    assert r.get("choice") == "approve"
    assert r.get("model") == "label"
    assert r.get("reason") == "r"  # answer fields come first
    assert r.get("coords.reason") == "coord"
    assert r.get("output.reason") == "r"
    assert r.get("meta.model") == "m"
    assert r.get("raw.seed_sent") is False
    assert r.get("usage.input_tokens") is None
    for bad in ("nope", "output.nope", "raw.nope", "coords.nope"):
        with pytest.raises(KeyError):
            r.get(bad)
    row = r.flat()
    assert row["reason"] == "coord"
    assert row["output.reason"] == "r"
    failed = Record(None, meta, {}, 0, "p", error="x")
    with pytest.raises(KeyError):
        failed.get("output.choice")


def test_combine_results():
    t = ep.Task("Q", Vote)
    a = ep.run(t, ep.FakeClient("a"), n=3)
    b = ep.run(t, ep.FakeClient("b"), n=2)
    assert len(a + b) == 5
    with pytest.raises(TypeError):
        a + ep.ResultSet(Everything, [])


def test_meta_roundtrip():
    meta = ResponseMeta(provider="p", model="m", raw={"k": 1})
    back = ResponseMeta.from_dict({**meta.to_dict(), "unknown": 1})
    assert back == meta
