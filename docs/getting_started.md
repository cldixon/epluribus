# Getting started

epluribus asks one or more models the same structured question many times. You get
back **every answer with its call metadata**, plus **summaries that fit each answer's
type**: shares for choices, means for scales, quantiles for numbers. Use it for quick
polls, comparing how models lean, and sweeping a question across framings or personas
to see where the answers change.

## Install

epluribus isn't on PyPI yet. Install from a local checkout:

```bash
pip install -e ".[anthropic,pandas]"      # or ".[openai,pandas]"
export ANTHROPIC_API_KEY=...
```

## Set up a model client

```python
import epluribus as ep
from epluribus.clients.anthropic import AnthropicClient

sonnet = AnthropicClient("claude-sonnet-5-5")
haiku = AnthropicClient("claude-haiku-5-5")
```

To track spending, pass `pricing=(input_usd, output_usd)` per million tokens, using
your provider's current prices. Without it, token counts are still recorded and
`cost_usd` stays `None`.

> **Try it offline.** Replace the clients with `ep.FakeClient("model-a")` and every
> example below runs without an API key. The outputs shown in comments are illustrative.

> **In Jupyter**, use `rs = await ep.arun(...)` wherever these examples call `ep.run(...)`.
> `run` starts its own event loop, which notebooks don't allow.

## How it fits together

1. **A schema**, a pydantic model describing the answer. Field types decide how answers
   are summarized, and the same class is sent to the model as the output format.
2. **A `Task`**: a prompt template plus the schema.
3. **`ep.run(task, models, n=...)`** makes the calls and returns a `ResultSet`.
   `rs["field"]` gives you the summaries for one field.

---

## 1. A quick poll

Ask one question 40 times and look at the spread.

```python
from typing import Literal
from pydantic import BaseModel, Field


class Vote(BaseModel):
    """Your vote on the proposal."""

    choice: Literal["approve", "reject", "abstain"]
    confidence: int = Field(ge=1, le=5, description="1 = coin flip, 5 = certain")
    reason: str


vote_task = ep.Task(
    "A city council is voting on converting two downtown parking garages into "
    "affordable housing. You are a council member. How do you vote?",
    Vote,
)

rs = ep.run(vote_task, sonnet, n=40)

rs["choice"].freq()
# {'approve': 0.625 [0.470, 0.758] (n=40),
#  'reject':  0.250 [0.142, 0.403] (n=40),
#  'abstain': 0.125 [0.055, 0.261] (n=40)}

rs["choice"].mode()  # 'approve'
rs["confidence"].mean()  # 3.400 [3.100, 3.700] (n=40)
rs["reason"].sample(3)  # three of the free-text reasons
rs.summary()  # default summary of every non-text field
```

Each share comes with a 95% interval and its sample size. At n=40 the intervals are
wide, and that's useful to know before you read much into a 60/40 split.

---

## 2. Compare models

Pass a list of models to give each one the same question.

```python
rs = ep.run(vote_task, [sonnet, haiku], n=30)  # 30 samples per model, 60 calls

by_model = rs.group_by("model")
by_model["choice"].freq()
# {'claude-sonnet-5-5': {'approve': 0.633 [0.455, 0.781] (n=30), 'reject': 0.267 ..., 'abstain': 0.100 ...},
#  'claude-haiku-5-5':  {'approve': 0.800 [0.627, 0.905] (n=30), 'reject': 0.133 ..., 'abstain': 0.067 ...}}

by_model["choice"].entropy()  # disagreement in bits; 0 means every sample agreed
by_model["confidence"].mean()
```

Any field method works on a group and returns one result per group.

To draw models at random instead, pass weights. Here **`n` is the total**, not per model:

```python
rs = ep.run(vote_task, {sonnet: 0.5, haiku: 0.5}, n=60)  # 60 calls, each model drawn with p=0.5
```

---

## 3. Sweep across roles and situations

Put `{placeholders}` in the template and pass a `grid`. Every combination becomes a
cell, and each cell gets `n` samples.

```python
from typing import Annotated


class Decision(BaseModel):
    """The department's decision."""

    option: Literal["hire_contractor", "train_staff", "delay"]
    urgency: Annotated[Literal["low", "medium", "high"], ep.Ordinal()]
    risks: list[Literal["cost", "quality", "morale", "schedule"]]


decision_task = ep.Task(
    "You are {role} in a mid-sized city's IT department. {situation}\n"
    "The legacy permit system must be replaced this year. What should the department do?",
    Decision,
)

grid = {
    "role": ["the department head", "the budget officer", "a senior engineer"],
    "situation": ["The budget was just cut 10%.", "A security audit flagged the old system."],
}

rs = ep.run(decision_task, sonnet, n=20, grid=grid)  # 6 cells x 20 = 120 calls

rs.group_by("role", "situation")["option"].mode()
# {('the department head', 'The budget was just cut 10%.'): 'train_staff',
#  ('the department head', 'A security audit flagged the old system.'): 'hire_contractor', ...}

rs.group_by("situation")["risks"].freq()  # share of answers flagging each risk
rs.filter(role="the budget officer")["urgency"].freq()
rs.group_by("situation")["urgency"].mean()  # mean rank: low=0, medium=1, high=2
```

Text choices are unordered by default. The `ep.Ordinal()` marker on `urgency` tells
epluribus the levels have an order, which unlocks `mean`, `median` and `quantiles`.

Templates use Python's `str.format`, so write literal braces as `{{` and `}}`.

---

## 4. Numeric estimates

Number fields get means, medians, spread and quantiles, which shows how widely a
model's estimates range.

```python
class TimeEstimate(BaseModel):
    months: float = Field(ge=0, description="Estimated months to complete")
    on_time_probability: float = Field(ge=0, le=1)


estimate_task = ep.Task(
    "A team of 4 engineers is rebuilding a permit-application website with "
    "online payments. Estimate how long it will take.",
    TimeEstimate,
)

rs = ep.run(estimate_task, [sonnet, haiku], n=25)

rs.group_by("model")["months"].median()
rs.group_by("model")["months"].quantiles((0.1, 0.5, 0.9))
rs.group_by("model")["on_time_probability"].mean()
```

A probability the model states is just another answer, not a calibrated probability.
The spread across samples usually tells you more.

---

## 5. Working with results

These use `rs` from the previous example.

**Each record keeps its answer and call metadata**

```python
r = rs.ok.records[0]  # first successful record
r.output  # the validated TimeEstimate
r.coords  # {'model': 'claude-sonnet-5-5'} plus any grid values
r.meta.resolved_model  # the exact model snapshot the provider reported
r.meta.usage  # input, output, reasoning and cached tokens
r.meta.latency_s, r.meta.cost_usd, r.meta.attempts
r.meta.raw  # provider-specific extras
```

**Usage and errors**

```python
rs.usage()
# {'records': 50, 'api_calls': 50, 'cache_hits': 0, 'errors': 1,
#  'input_tokens': 9120, 'output_tokens': 2410, 'cost_usd': 0.051, ...}

rs.errors  # failed calls, kept as records with .error; a failure never stops a run
rs.ok  # only the successful records
```

Failed calls are retried twice by default (`max_retries`). Summaries count failures
as missing: each field view reports `n` and `missing`.

**Caching**

```python
cache = ep.MemoryCache()
rs = ep.run(estimate_task, sonnet, n=25, cache=cache)
rs2 = ep.run(estimate_task, sonnet, n=25, cache=cache)  # all 25 come from the cache
rs3 = ep.run(estimate_task, sonnet, n=40, cache=cache)  # only 15 new calls
```

Each sample gets a stable ID from `seed` (default 0), the grid cell, the model and its
index, and the cache is keyed on that ID plus the exact prompt and settings. Raising `n`
or adding a grid value only makes the new calls. Changing the template, the schema or
`temperature` makes everything new. To draw a fresh set of samples, pass a different `seed`.

Most providers ignore seeds, so a seed identifies a sample for the cache. It doesn't
make a model's answer reproducible.

`MemoryCache` lasts as long as your Python session.

**Export**

```python
df = rs.to_pandas()  # one row per sample: coords, answer fields, meta.*, usage.*
arr = rs.arrays()  # numpy arrays per field: categories as integer codes,
# multi-select as a 0/1 matrix, missing as -1 or NaN
```

---

## Tips

- **Start small.** Run `n=5` first to check the prompt and the schema, then scale up.
- **Not enough variety?** If every sample says the same thing, raise `temperature`
  (some reasoning models don't accept it), or put the variation in the prompt with a
  `grid` of personas or phrasings.
- **Text fields are kept but not summarized.** Use `sample()` to read some, or export
  to pandas.

---

## Field types at a glance

| You write | Treated as | Summaries |
|---|---|---|
| `Literal["a", "b"]`, `Enum` | categorical | `freq`, `counts`, `mode`, `entropy` |
| `Literal[1, 2, 3]`, `int` with both `ge` and `le`, `IntEnum`, `Annotated[..., ep.Ordinal()]` | ordinal | categorical ones, plus `mean`, `median`, `quantiles`, `std` |
| `bool` | boolean | `rate`, `freq`, `counts` |
| `float`, `int` without both bounds | numeric | `mean`, `median`, `std`, `quantiles` |
| `list[Literal[...]]` | multi-select | `freq` (share selecting each option), `counts`, `mean_selected` |
| `str` | text | `sample`, `values` |
| `Optional[X]` | same as `X` | `None` counts as missing |

Every field also has `values()`, `n`, `missing`, `array()` and `summary()`. Using a
summary that doesn't fit, like `mean()` on a categorical field, raises `ep.KindError`
naming the field's type.
