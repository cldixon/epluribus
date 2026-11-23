# Polls and model comparisons

This guide uses the `Vote` schema and `task` from [Get started](../get-started.md).

```python
import epluribus as ep
from typing import Literal
from pydantic import BaseModel, Field
from epluribus.clients.anthropic import AnthropicClient

sonnet = AnthropicClient("claude-sonnet-5-5")
haiku = AnthropicClient("claude-haiku-5-5")


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
```

## A quick poll

```python
rs = ep.run(task, sonnet, n=40)

rs["choice"].freq()  # share for each option, with 95% intervals
rs["choice"].mode()  # 'approve'
rs["choice"].entropy()  # disagreement in bits; 0 means every answer agreed
rs["confidence"].mean()  # 3.400 [3.100, 3.700] (n=40)
rs["reason"].sample(3)  # three of the free-text reasons
rs.summary()  # default summary of every non-text field
```

## Compare models

Pass a list of models. **`n` is per model**, so this run makes 60 calls:

```python
rs = ep.run(task, [sonnet, haiku], n=30)

by_model = rs.group_by("model")
by_model["choice"].freq()
```

```text
model              approve               reject                abstain               n
-----------------  --------------------  --------------------  --------------------  --
claude-sonnet-5-5  0.633 [0.455, 0.781]  0.267 [0.142, 0.444]  0.100 [0.035, 0.256]  30
claude-haiku-5-5   0.800 [0.627, 0.905]  0.133 [0.053, 0.297]  0.067 [0.018, 0.213]  30
```

Every field method works on a group and returns one result per group:

```python
by_model["choice"].entropy()
by_model["confidence"].mean()
by_model.size()
```

### Is the difference real?

`diff(a, b)` subtracts group `b` from group `a`, with intervals:

```python
by_model["choice"].diff("claude-sonnet-5-5", "claude-haiku-5-5")  # share difference per option
by_model["confidence"].diff("claude-sonnet-5-5", "claude-haiku-5-5")  # difference in means
```

If an interval includes 0, the data don't show a difference at this sample size.

## A population of models

`ep.Mix` draws each sample's model at random by weight. A mix counts as one entry in `models`, so `n` is the number of draws from it:

```python
population = ep.Mix({sonnet: 0.5, haiku: 0.5})
rs = ep.run(task, population, n=60)  # 60 calls, about half from each model
rs.group_by("model").size()
```

You can put mixes and single models in the same list.

## Use several providers

Clients from different providers work together:

```python
from epluribus.clients.openai import OpenAIClient

gpt = OpenAIClient("gpt-5-mini")
rs = ep.run(task, [sonnet, gpt], n=30)
```

See [Providers](providers.md) for each provider's options.
