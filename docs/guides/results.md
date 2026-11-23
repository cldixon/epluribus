# Work with results

`ep.run` returns a `ResultSet`: one record per call, failed calls included. This page uses this run:

```python
import epluribus as ep
from typing import Literal
from pydantic import BaseModel, Field
from epluribus.clients.anthropic import AnthropicClient

sonnet = AnthropicClient("claude-sonnet-5-5", effort="low", pricing=(2.0, 10.0))
haiku = AnthropicClient("claude-haiku-5-5", effort="low", pricing=(0.10, 0.50))


class Vote(BaseModel):
    choice: Literal["approve", "reject", "abstain"]
    confidence: int = Field(ge=1, le=5)
    in_favor_of_parking: bool
    concerns: list[Literal["cost", "traffic", "businesses", "safety"]]
    reason: str


task = ep.Task("You are $role. The council votes on converting a garage into housing.", Vote)
rs = ep.run(task, [sonnet, haiku], n=20, grid={"role": ["a retiree", "a cyclist"]})
```

## Summaries of one field

`rs["field"]` is a `FieldView`. Its methods depend on the field's [kind](../reference/field-types.md):

```python
rs["choice"].freq()  # categorical: shares with intervals
rs["choice"].counts()
rs["confidence"].mean()  # ordinal: mean, median, quantiles, std
rs["in_favor_of_parking"].rate()  # boolean: share of True
rs["concerns"].freq()  # multi-select: share selecting each option
rs["concerns"].mean_selected()  # options picked per answer
rs["reason"].sample(3)  # text: read some answers
```

A summary that doesn't fit the field raises `ep.KindError`:

```python
try:
    rs["choice"].mean()
except ep.KindError as e:
    print(e)  # mean() needs a field of kind boolean, numeric, ordinal; 'choice' is categorical
```

Each field view also has `values()`, `n`, `missing`, `array()` and `summary()`. Failed calls and `None` answers count as missing.

## Groups and filters

Group and filter on grid values, `model`, answer fields, or metadata:

```python
rs.group_by("model", "role")["choice"].freq()
rs.group_by("choice")["confidence"].mean()  # confidence by the answer chosen
rs.filter(role="a cyclist", model="claude-haiku-5-5")["choice"].freq()
rs.filter(lambda r: r.meta.latency_s is not None and r.meta.latency_s < 10)
rs.group_by("role")["choice"].diff("a retiree", "a cyclist")
```

A name that no record has raises `KeyError`, so a typo can't silently put everything in one group.

If an answer field has the same name as a coord or a metadata field, the answer wins. Use a prefix to choose: `"coords.model"`, `"output.model"`, `"meta.model"`.

## Display

Results are dicts with a table display. In a terminal they print as aligned text; in a notebook they render as HTML. Indexing works as usual:

```python
shares = rs["choice"].freq()
shares["approve"].value, shares["approve"].lo, shares["approve"].hi, shares["approve"].n
f"{shares['approve']:.2f}"  # '0.62 [0.40, 0.80]'
```

Each summary value with an interval is an `ep.Interval`: `value`, `lo`, `hi` and `n`.

## Records and metadata

Each record keeps its answer and everything known about the call:

```python
r = rs.ok.records[0]  # first successful record
r.output  # the validated Vote
r.coords  # {'role': 'a retiree', 'model': 'claude-sonnet-5-5'}
r.prompt, r.system  # what was sent
r.meta.resolved_model  # the model snapshot the provider reported
r.meta.usage  # input, output, reasoning and cached tokens
r.meta.latency_s, r.meta.cost_usd, r.meta.attempts
r.meta.request_id  # quote this to the provider when reporting a problem
r.meta.raw  # provider-specific extras
```

## Usage and errors

```python
rs.usage()
```

```text
                     value
-------------------  ---------------
records              80
api_calls            81
cache_hits           0
errors               1
error_kinds          {'refusal': 1}
input_tokens         9120
output_tokens        2410
...
```

`api_calls` counts every try, retries included. Token and cost totals count only calls made in this run, not cache hits.

```python
rs.errors  # failed records, each with .error, .error_kind and .raw_text
rs.ok  # only the successful records
```

Error kinds:

| Kind | Meaning | Retried |
|---|---|---|
| `rate_limit` | The provider asked us to slow down (HTTP 429). | Yes, after `retry-after` |
| `server`, `connection`, `timeout` | The provider failed or could not be reached. | Yes |
| `auth`, `not_found`, `bad_request` | The request is wrong: key, model name or parameter. | No |
| `refusal` | The model declined to answer. | No |
| `invalid_output` | The answer doesn't match the schema. `raw_text` has it. | No |
| `truncated` | The answer hit `max_tokens`, thinking included. Raise it, or lower `effort`, and rerun. | No |

Refusals and invalid answers are kept, not retried: retrying only those would bias the results toward answers that pass. Count them; they are part of how the model behaves.

## Export

```python
df = rs.to_pandas()  # one row per record: coords, answers, meta.*, usage.*
rs.group_by("model")["choice"].freq().to_pandas()  # intervals split into _lo, _hi, _n columns
arrays = rs.arrays()  # NumPy arrays per field
```

`arrays()` encodes categories as integer codes, booleans as 0/1, multi-select as a 0/1 matrix, and missing values as -1 or NaN.

Combine results that share a schema with `+`:

```python
more = ep.run(task, sonnet, n=5, grid={"role": ["a shop owner"]})
combined = rs + more
```
