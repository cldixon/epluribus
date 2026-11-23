# Get started

## Install

epluribus needs Python 3.11 or later. Install it with the extra for your provider:

```bash
uv add "epluribus[anthropic]"      # or: pip install "epluribus[anthropic]"
uv add "epluribus[openai]"         # OpenAI models
uv add "epluribus[pandas]"         # DataFrame export
```

Set your provider's API key in the environment:

```bash
export ANTHROPIC_API_KEY="..."
export OPENAI_API_KEY="..."
```

## Make a client

A client is one model. Its label names it in your results.

```python
import epluribus as ep
from epluribus.clients.anthropic import AnthropicClient

sonnet = AnthropicClient("claude-sonnet-5-5", pricing=(2.0, 10.0))
haiku = AnthropicClient("claude-haiku-5-5", pricing=(0.10, 0.50))
```

`pricing` is optional: `(input, output)` in USD per million tokens. With it, each record has a `cost_usd`. Check your provider's current prices.

!!! tip "Try everything offline"
    Use `ep.FakeClient("model-a")` instead of a real client. It makes up valid answers for any schema, with no API key and no cost. Every example in these docs runs with fake clients in the test suite.

## Ask a question

A run needs three things:

1. **A schema.** A pydantic model that describes the answer. Its field types decide how answers are summarized.
2. **A task.** A prompt template plus the schema.
3. **Models and a sample size.** `ep.run(task, models, n=...)` makes the calls and returns a `ResultSet`.

```python
from typing import Literal
from pydantic import BaseModel, Field


class Vote(BaseModel):
    """Your vote on the proposal."""

    choice: Literal["approve", "reject", "abstain"]
    confidence: int = Field(ge=1, le=5, description="1 = coin flip, 5 = certain")
    reason: str


task = ep.Task(
    "A city council is voting on converting two downtown parking garages into "
    "affordable housing. You are a council member. How do you vote?",
    Vote,
)

rs = ep.run(task, sonnet, n=40)
rs
```

Printing a `ResultSet` shows a summary of every field:

```text
<ResultSet Vote: 40 records, 0 errors>
field       kind         n   missing  summary
----------  -----------  --  -------  ---------------------------------------------------
choice      categorical  40  0        approve 0.62, reject 0.25, abstain 0.12
confidence  ordinal      40  0        mean 3.400 [3.100, 3.700]; median 3.000 [3.000, ...
```

Look at one field:

```python
rs["choice"].freq()
```

```text
level    share  95% interval
-------  -----  --------------
approve  0.625  [0.470, 0.758]
reject   0.250  [0.142, 0.403]
abstain  0.125  [0.055, 0.261]
(n)      40
```

At `n=40` the intervals are wide. Know that before you read much into a 60/40 split.

## In a notebook

`ep.run` works in Jupyter as it does in scripts. If you prefer the notebook's own event loop, await the async version:

<!-- skip -->
```python
rs = await ep.arun(task, sonnet, n=40)
```

In a notebook, results display as HTML tables.

## Next steps

- [Polls and model comparisons](guides/polls.md)
- [Sweeps](guides/sweeps.md): roles, framings and phrasings
- [Work with results](guides/results.md)
