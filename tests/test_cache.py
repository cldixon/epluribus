from __future__ import annotations

from pydantic import BaseModel

import epluribus as ep

from .conftest import Vote


def test_disk_cache_survives_restarts(tmp_path, vote_task):
    path = tmp_path / "sub" / "cache.sqlite"
    grid = {"role": ["a", "b"]}
    cache = ep.DiskCache(path)
    first = ep.run(vote_task, ep.FakeClient(), n=5, grid=grid, cache=cache)
    assert len(cache) == 10
    cache.close()

    reopened = ep.DiskCache(path)
    again = ep.run(vote_task, ep.FakeClient(), n=5, grid=grid, cache=reopened)
    assert again.usage()["cache_hits"] == 10
    assert [r.output for r in again] == [r.output for r in first]
    r0, a0 = first.records[0], again.records[0]
    assert a0.meta.created_at == r0.meta.created_at
    assert a0.meta.usage == r0.meta.usage
    assert a0.raw_text == r0.raw_text
    assert "10 entries" in repr(reopened)
    assert isinstance(reopened, ep.Cache)
    reopened.clear()
    assert len(reopened) == 0
    reopened.close()


def test_failures_are_not_cached(tmp_path, vote_task):
    cache = ep.DiskCache(tmp_path / "c.sqlite")
    rs = ep.run(
        vote_task, ep.FakeClient(error_rate=1.0), n=2, grid={"role": ["a"]}, cache=cache,
        max_retries=0,
    )  # fmt: skip
    assert len(rs.errors) == 2
    assert len(cache) == 0
    cache.close()


def test_stale_entries_are_misses(tmp_path):
    class Other(BaseModel):
        x: int

    cache = ep.DiskCache(tmp_path / "c.sqlite")
    gen = ep.run(ep.Task("Q", Vote), ep.FakeClient(), n=1).records[0]
    cache.set("k", ep.Generation(gen.output, gen.meta))  # ty: ignore[invalid-argument-type]
    assert cache.get("k", Vote) is not None
    assert cache.get("k", Other) is None
    assert cache.get("missing", Vote) is None
    assert repr(ep.MemoryCache()) == "MemoryCache(0 entries)"
    cache.close()
