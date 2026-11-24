from __future__ import annotations

import asyncio
import time
from typing import Any

import pytest
from pydantic import BaseModel

import epluribus as ep
from epluribus.clients.base import Generation
from epluribus.errors import InvalidOutputError, ProviderError, RefusalError
from epluribus.records import ResponseMeta

from .conftest import Everything, Vote

ROLES = {"role": ["a retiree", "a cyclist"]}


def test_run_counts_and_coords(vote_task):
    a, b = ep.FakeClient("a"), ep.FakeClient("b")
    rs = ep.run(vote_task, [a, b], n=5, grid=ROLES)
    assert len(rs) == 20
    assert {r.coords["model"] for r in rs} == {"a", "b"}
    assert {r.coords["role"] for r in rs} == {"a retiree", "a cyclist"}
    assert all(r.ok for r in rs)
    assert rs.records[0].prompt == "You are a retiree. How do you vote?"


def test_every_kind_runs():
    rs = ep.run(ep.Task("Q", Everything), ep.FakeClient(), n=10)
    assert len(rs.ok) == 10
    assert set(rs.summary()) == set(ep.schema_specs(Everything)) - {"text"}


def test_n_means_per_model_and_mix_counts_as_one(vote_task):
    a, b, c = ep.FakeClient("a"), ep.FakeClient("b"), ep.FakeClient("c")
    calls = ep.plan(vote_task, [a, ep.Mix({b: 1, c: 1})], n=10, grid=ROLES)
    assert len(calls) == 2 * (10 + 10)


def test_mix_weights_and_stability(vote_task):
    a, b = ep.FakeClient("a"), ep.FakeClient("b")
    mix = ep.Mix({a: 0.8, b: 0.2})
    small = ep.plan(vote_task, mix, n=50, grid=ROLES)
    big = ep.plan(vote_task, mix, n=200, grid=ROLES)
    share_a = sum(c.client is a for c in big) / len(big)
    assert 0.7 < share_a < 0.9
    # Raising n keeps every earlier draw, so the cache still hits for them.
    keyed_big = {(c.coords["role"], c.index): (c.client.label, c.seed) for c in big}
    for c in small:
        assert keyed_big[(c.coords["role"], c.index)] == (c.client.label, c.seed)
    assert "a: 0.80" in repr(mix)


def test_model_argument_errors(vote_task):
    a = ep.FakeClient("a")
    with pytest.raises(TypeError, match=r"ep\.Mix"):
        ep.plan(vote_task, {a: 1.0}, n=1, grid=ROLES)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="label"):
        ep.plan(vote_task, [a, ep.FakeClient("a")], n=1, grid=ROLES)
    with pytest.raises(ValueError, match="empty"):
        ep.plan(vote_task, [], n=1, grid=ROLES)
    with pytest.raises(TypeError, match="clients"):
        ep.plan(vote_task, ["a"], n=1, grid=ROLES)  # ty: ignore[invalid-argument-type]
    with pytest.raises(ValueError, match="positive"):
        ep.Mix({a: 0})
    with pytest.raises(ValueError, match="at least one"):
        ep.Mix({})
    with pytest.raises(ValueError, match=">= 0"):
        ep.plan(vote_task, a, n=-1, grid=ROLES)


def test_templates_and_grid_checks():
    t = ep.Task('Reply as JSON like {"a": 1}. Cost: $$5. Topic: ${topic}.', Vote)
    assert t.render({"topic": "x"})[0] == 'Reply as JSON like {"a": 1}. Cost: $5. Topic: x.'
    assert t.placeholders == {"topic"}
    with pytest.raises(ValueError, match="invalid placeholder"):
        ep.Task("costs $5", Vote)
    with pytest.raises(ValueError, match="at least one"):
        ep.Task([], Vote)
    a = ep.FakeClient()
    with pytest.raises(ValueError, match="no values"):
        ep.plan(t, a, n=1)
    with pytest.raises(ValueError, match="reserved"):
        ep.plan(ep.Task("Q", Vote), a, n=1, grid={"model": ["x"]})
    with pytest.raises(ValueError, match="answer fields"):
        ep.plan(ep.Task("$choice", Vote), a, n=1, grid={"choice": ["x"]})
    with pytest.raises(TypeError, match="wrap it in a list"):
        ep.plan(t, a, n=1, grid={"topic": "x"})


def test_grid_labels_and_system_prompt():
    t = ep.Task("Vote as $persona.", Vote, system="Answer as $persona.")
    personas = {"retiree": "a retired teacher", "cyclist": "a daily cyclist"}
    rs = ep.run(t, ep.FakeClient(), n=2, grid={"persona": personas})
    assert {r.coords["persona"] for r in rs} == {"retiree", "cyclist"}
    r = rs.filter(persona="retiree").records[0]
    assert r.prompt == "Vote as a retired teacher."
    assert r.system == "Answer as a retired teacher."


def test_template_variants():
    t = ep.Task(["Do you approve?", "Would you vote yes?"], Vote)
    rs = ep.run(t, ep.FakeClient(), n=4)
    assert len(rs) == 8
    assert {r.coords["variant"] for r in rs} == {0, 1}
    assert "+1 variants" in repr(t)


def test_reserved_answer_field_warns():
    class Odd(BaseModel):
        model: str

    with pytest.warns(UserWarning, match="coords.model"):
        ep.Task("Q", Odd)


def test_cache_extends_and_reuses(vote_task):
    cache = ep.MemoryCache()
    a = ep.FakeClient()
    first = ep.run(vote_task, a, n=8, grid=ROLES, cache=cache, seed=3)
    again = ep.run(vote_task, a, n=8, grid=ROLES, cache=cache, seed=3)
    more = ep.run(vote_task, a, n=10, grid=ROLES, cache=cache, seed=3)
    fresh = ep.run(vote_task, a, n=8, grid=ROLES, cache=cache, seed=4)
    assert again.usage()["api_calls"] == 0
    assert all(r.meta.cache_hit for r in again)
    assert [r.output for r in first] == [r.output for r in again]
    assert more.usage()["api_calls"] == 4
    assert fresh.usage()["cache_hits"] == 0
    assert len(cache) == 16 + 4 + 16
    cache.clear()
    assert len(cache) == 0


def test_cache_separates_fake_labels(vote_task):
    cache = ep.MemoryCache()
    ep.run(vote_task, ep.FakeClient("m", label="x"), n=3, grid=ROLES, cache=cache)
    rs = ep.run(vote_task, ep.FakeClient("m", label="y"), n=3, grid=ROLES, cache=cache)
    assert rs.usage()["cache_hits"] == 0


def test_errors_are_recorded_and_retried(vote_task):
    rs = ep.run(vote_task, ep.FakeClient(error_rate=1.0), n=3, grid=ROLES, backoff=0)
    assert len(rs.errors) == 6
    assert rs["choice"].n == 0
    assert rs["choice"].missing == 6
    assert all(r.meta.attempts == 3 for r in rs)
    assert rs.usage()["error_kinds"] == {"server": 6}
    assert rs.usage()["api_calls"] == 18

    flaky = ep.run(vote_task, ep.FakeClient(error_rate=0.5), n=20, grid=ROLES, backoff=0)
    assert len(flaky.ok) > 30


class Scripted(ep.Client):
    """Raises the queued exceptions in order, then succeeds."""

    provider = "scripted"

    def __init__(self, *errors: Exception, **kw: Any) -> None:
        super().__init__("scripted", **kw)
        self.errors = list(errors)
        self.calls = 0
        self.times: list[float] = []

    async def agenerate(self, prompt, schema, *, system=None, temperature=None, seed=None,
                        max_tokens=1024):  # fmt: skip
        self.calls += 1
        self.times.append(time.monotonic())
        if self.errors:
            raise self.errors.pop(0)
        out = schema(choice="approve", confidence=3, reason="r")
        return Generation(out, ResponseMeta(provider=self.provider, model=self.model))


def test_non_retryable_failures_are_kept():
    task = ep.Task("Q", Vote)
    for err, kind in [
        (RefusalError("no", raw_text="I won't"), "refusal"),
        (InvalidOutputError("bad", raw_text='{"choice": "maybe"}'), "invalid_output"),
        (ProviderError("bad request", kind="bad_request"), "bad_request"),
        (RuntimeError("boom"), "error"),
    ]:
        client = Scripted(err)
        rs = ep.run(task, client, n=1)
        r = rs.records[0]
        assert not r.ok
        assert r.error_kind == kind
        assert client.calls == 1
        assert r.raw_text == getattr(err, "raw_text", None)


def test_retry_after_is_honored():
    client = Scripted(
        ProviderError("slow down", kind="rate_limit", retryable=True, retry_after=0.05),
        TimeoutError("timed out"),
    )
    rs = ep.run(ep.Task("Q", Vote), client, n=1, backoff=0)
    assert rs.records[0].ok
    assert rs.records[0].meta.attempts == 3
    assert client.times[1] - client.times[0] >= 0.03


def test_max_concurrency_per_client():
    class Slow(ep.FakeClient):
        active = 0
        peak = 0

        async def agenerate(self, *args, **kw):
            Slow.active += 1
            Slow.peak = max(Slow.peak, Slow.active)
            await asyncio.sleep(0.01)
            Slow.active -= 1
            return await super().agenerate(*args, **kw)

    ep.run(ep.Task("Q", Vote), Slow(max_concurrency=2), n=10, concurrency=8)
    assert Slow.peak == 2


def test_temperature_check_runs_before_calls():
    class NoTemp(Scripted):
        def check(self, *, temperature=None):
            if temperature is not None:
                raise ValueError("no temperature")

    client = NoTemp()
    with pytest.raises(ValueError, match="no temperature"):
        ep.run(ep.Task("Q", Vote), client, n=3, temperature=0.5)
    assert client.calls == 0


def test_fake_temperature_changes_spread(vote_task):
    a = ep.FakeClient()
    cold = ep.run(vote_task, a, n=200, grid=ROLES, temperature=0.1)
    hot = ep.run(vote_task, a, n=200, grid=ROLES, temperature=5)
    assert cold["choice"].entropy() < hot["choice"].entropy()


def test_run_inside_event_loop(vote_task):
    async def notebook_cell():
        return ep.run(vote_task, ep.FakeClient(), n=3, grid=ROLES)

    assert len(asyncio.run(notebook_cell())) == 6

    async def awaited():
        return await ep.arun(vote_task, ep.FakeClient(), n=3, grid=ROLES)

    assert len(asyncio.run(awaited())) == 6


def test_progress(vote_task, capsys):
    ep.run(vote_task, ep.FakeClient(), n=2, grid=ROLES, progress=True)
    assert "4/4 calls" in capsys.readouterr().err


class _Closable:
    def __init__(self) -> None:
        self.closed = False

    async def close(self) -> None:
        self.closed = True


class LoopLocalClient(ep.FakeClient):
    """Opens one closable resource per event loop, like the provider clients do."""

    def __init__(self, *args: Any, **kw: Any) -> None:
        super().__init__(*args, **kw)
        self.opened: list[_Closable] = []

    def _resource(self) -> _Closable:
        def make() -> _Closable:
            self.opened.append(_Closable())
            return self.opened[-1]

        return self.loop_local(make)

    async def agenerate(self, *args: Any, **kw: Any) -> Generation:
        assert not self._resource().closed
        return await super().agenerate(*args, **kw)


def test_run_closes_loop_local_resources():
    client = LoopLocalClient()
    ep.run(ep.Task("Q", Vote), client, n=3)
    ep.run(ep.Task("Q", Vote), client, n=3)
    assert len(client.opened) == 2
    assert all(r.closed for r in client.opened)


def test_concurrent_runs_share_resources_until_the_last_ends():
    client = LoopLocalClient(latency=0.02)
    task = ep.Task("Q", Vote)

    async def both():
        return await asyncio.gather(ep.arun(task, client, n=5), ep.arun(task, client, n=8, seed=1))

    a, b = asyncio.run(both())
    assert len(a.ok) == 5
    assert len(b.ok) == 8
    assert len(client.opened) == 1
    assert client.opened[0].closed


def test_resources_close_even_when_a_run_raises():
    client = LoopLocalClient()

    class BrokenCacheError(Exception):
        pass

    class BrokenCache:
        def get(self, key: str, schema: type[BaseModel]) -> Generation | None:
            return None

        def set(self, key: str, gen: Generation) -> None:
            raise BrokenCacheError

    with pytest.raises(BrokenCacheError):
        ep.run(ep.Task("Q", Vote), client, n=2, cache=BrokenCache())
    assert len(client.opened) == 1
    assert client.opened[0].closed
