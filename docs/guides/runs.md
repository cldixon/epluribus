# Run at scale

Samples cost money and time. This page covers counting calls before you make them, caching, retries and rate limits.

```python
import epluribus as ep
from typing import Literal
from pydantic import BaseModel
from epluribus.clients.anthropic import AnthropicClient

sonnet = AnthropicClient("claude-sonnet-5-5", effort="low")


class Vote(BaseModel):
    choice: Literal["approve", "reject", "abstain"]


task = ep.Task("You are $role. Do you support the housing plan?", Vote)
grid = {"role": ["a retiree", "a cyclist", "a shop owner"]}
```

## Count calls first

`ep.plan` expands a run without making any call:

```python
calls = ep.plan(task, sonnet, n=50, grid=grid)
len(calls)  # 150
```

Start with a small `n`, such as 5, to check the prompt and the schema. Then scale up: with a cache, the first 5 samples are reused.

## Caching

Pass a cache to make reruns free:

```python
cache = ep.DiskCache("runs/cache.sqlite")

rs = ep.run(task, sonnet, n=25, grid=grid, cache=cache)
rs2 = ep.run(task, sonnet, n=25, grid=grid, cache=cache)  # all 75 from the cache
rs3 = ep.run(task, sonnet, n=40, grid=grid, cache=cache)  # 45 new calls
```

`DiskCache` keeps answers in a SQLite file, so they survive restarts. It stores JSON, not pickled objects. `ep.MemoryCache()` lasts as long as the Python process.

### What the cache key covers

Each sample gets a stable seed from the run's `seed` (default 0), its grid cell, its model and its index. The cache key is that seed plus the exact prompt, system prompt, schema, model, client settings (such as `effort`), `temperature` and `max_tokens`. So:

| You change | Result |
|---|---|
| Raise `n`, add grid values or models | Only the new samples are called |
| The template, schema, temperature or a client setting | Every affected sample is called again |
| `seed` | A fresh set of samples |

Failed calls are never cached, so a rerun retries them.

!!! note "Seeds identify samples; they don't fix answers"
    Most providers ignore seeds, and Anthropic and OpenAI have no seed parameter for these calls. A seed tells the cache which sample is which. It doesn't make the model's answer reproducible. `meta.raw["seed_sent"]` records whether the seed was sent.

## Retries

Rate limits, server errors and connection errors are retried up to `max_retries` times (default 2). The delay starts at `backoff` seconds (default 1) and doubles each try, with jitter. When the provider sends `retry-after`, epluribus waits that long (up to a minute) and holds new calls to that client until then.

```python
rs = ep.run(task, sonnet, n=25, grid=grid, max_retries=4, backoff=2.0)
```

Refusals, invalid answers and request errors are not retried. See [error kinds](results.md#usage-and-errors).

The default provider clients turn off their SDK's own retries, so `meta.attempts` counts every try.

## Concurrency and rate limits

`concurrency` (default 8) limits calls in flight for the whole run. To limit one model, for example a provider with a low rate limit, set `max_concurrency` on its client. The limit holds across every run that uses the client.

```python
slow_tier = AnthropicClient(
    "claude-sonnet-5-5", effort="low", label="sonnet-slow", max_concurrency=2
)
rs = ep.run(task, [sonnet, slow_tier], n=10, grid=grid, concurrency=16)
```

## Progress

`progress=True` prints a progress line to stderr:

```python
rs = ep.run(task, sonnet, n=25, grid=grid, cache=cache, progress=True)
# 75/75 calls (75 cached, 0 failed)
```

## Costs

Give clients `pricing` to get `cost_usd` per record and in `rs.usage()`. Without it, token counts are still recorded.

Thinking and reasoning tokens are billed as output, so they often cost more than the answer itself. `rs.usage()["reasoning_tokens"]` shows how many there were. Lower `effort` to cut them.

## Token budget

`max_tokens` limits each answer, thinking included. When you don't set it, each client picks a default: 16,000 for Anthropic and OpenAI, whose models think first, and 1,024 for others. A lower limit doesn't save money unless answers hit it; it only turns long answers into `truncated` errors.
