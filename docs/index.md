# epluribus

*Out of many, one.* epluribus asks one or more language models the same structured question many times. It returns **every answer with its call metadata**, and **summaries that fit each answer's type**: shares for choices, means for scales, quantiles for numbers. Every summary comes with an interval and its sample size.

```python
import epluribus as ep
from typing import Literal
from pydantic import BaseModel, Field
from epluribus.clients.anthropic import AnthropicClient


class Vote(BaseModel):
    choice: Literal["approve", "reject", "abstain"]
    confidence: int = Field(ge=1, le=5)
    reason: str


task = ep.Task(
    "You are $role. The council votes on converting a garage into housing. How do you vote?", Vote
)

sonnet = AnthropicClient("claude-sonnet-5-5", effort="low")
haiku = AnthropicClient("claude-haiku-5-5", effort="low")

rs = ep.run(task, [sonnet, haiku], n=30, grid={"role": ["a retiree", "a cyclist"]})

rs["choice"].freq()  # shares with 95% intervals
rs.group_by("model", "role")["confidence"].mean()  # one row per group
rs.usage()  # calls, tokens, cost, errors
```

## Uses

- **Quick polls.** How does a model lean on a question, and how firmly?
- **Model comparisons.** Where do two models disagree?
- **Decision-space sweeps.** Vary roles, framings and constraints, and find where the answer flips.
- **Robustness checks.** Ask the same question in several phrasings. Trust only findings that survive.

## Features

- **The schema decides the summaries.** A pydantic model is the answer format sent to the model *and* the rule for how each field is summarized. A summary that doesn't fit a field, such as the mean of a category, raises an error.
- **Intervals by default.** Shares have Wilson intervals. Means and medians have bootstrap intervals. Each result shows its `n`.
- **Every sample is kept.** Each answer keeps its prompt, model snapshot, tokens, latency, cost and position in the sweep.
- **Failures are data.** A failed call becomes a record with an error kind. One failure never stops a run.
- **Cheap reruns.** A cache keyed on each sample makes reruns free. Raising `n` only makes the new calls.
- **Any provider.** Anthropic and OpenAI clients are included, and a fake client runs everything offline. You can mix providers in one run.
- **No logprobs needed.** epluribus only samples, so it works with frontier APIs that don't expose token probabilities.

## Next steps

1. [Get started](get-started.md). Install the package and run your first poll.
2. Read the [guides](guides/polls.md).
3. Learn [how it works and where its limits are](concepts.md).

!!! warning "Simulated answers are not real people"
    A model's answers show the model's tendencies. They don't measure what real voters, customers or committees think. See [limits](concepts.md#limits).
