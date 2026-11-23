# Numeric estimates

Number fields get means, medians, spread and quantiles. Use them to see how widely a model's estimates range.

```python
import epluribus as ep
from pydantic import BaseModel, Field
from epluribus.clients.anthropic import AnthropicClient

sonnet = AnthropicClient("claude-sonnet-5-5", effort="low")
haiku = AnthropicClient("claude-haiku-5-5", effort="low")


class TimeEstimate(BaseModel):
    months: float = Field(ge=0, description="Estimated months to complete")
    on_time_probability: float = Field(ge=0, le=1)


task = ep.Task(
    "A team of 4 engineers is rebuilding a permit-application website with "
    "online payments. Estimate how long it will take.",
    TimeEstimate,
)

rs = ep.run(task, [sonnet, haiku], n=25)

by_model = rs.group_by("model")
by_model["months"].median()
by_model["months"].quantiles((0.1, 0.5, 0.9))
by_model["months"].std()
by_model["on_time_probability"].mean()
```

## Bounds are checked

The constraints in a schema, such as `ge=0` and `le=1`, are checked on every answer. Some providers can't enforce numeric bounds while the model writes, so a model can still answer outside them. Such an answer is kept as a failed record with `error_kind="invalid_output"` and its raw text, so you can count and read it:

```python
rs.usage()["error_kinds"]  # for example {'invalid_output': 1}
[r.raw_text for r in rs.errors]
```

These answers are not retried: retrying only the failures would bias the results toward answers that pass.

## Stated confidence is just an answer

A probability the model states, like `on_time_probability`, is not a calibrated probability. It is one more answer to summarize. The spread of `months` across samples usually tells you more about the model's uncertainty.
