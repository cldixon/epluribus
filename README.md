# epluribus

[![CI](https://github.com/cldixon/epluribus/actions/workflows/ci.yml/badge.svg)](https://github.com/cldixon/epluribus/actions/workflows/ci.yml)
[![PyPI](https://img.shields.io/pypi/v/epluribus)](https://pypi.org/project/epluribus/)
[![Python](https://img.shields.io/pypi/pyversions/epluribus)](https://pypi.org/project/epluribus/)

*Out of many, one.* Sample LLMs many times on a structured question. Get every answer with its call metadata, plus summaries that fit each answer's type, with intervals.

Documentation: <https://cldixon.github.io/epluribus/>

## Install

```sh
uv add "epluribus[anthropic]"     # or epluribus[openai]
```

## Use

```python
from typing import Literal

from pydantic import BaseModel, Field

import epluribus as ep
from epluribus.clients.anthropic import AnthropicClient


class Vote(BaseModel):
    choice: Literal["approve", "reject", "abstain"]  # categorical
    confidence: int = Field(ge=1, le=5)  # ordinal
    reason: str  # text: kept, not summarized


task = ep.Task(
    "You are $role. The council votes on converting a garage into housing. How do you vote?", Vote
)
models = [AnthropicClient("claude-sonnet-5-5"), AnthropicClient("claude-haiku-5-5")]

rs = ep.run(task, models, n=30, grid={"role": ["a retiree", "a cyclist"]})

rs["choice"].freq()  # shares with 95% intervals
rs.group_by("model", "role")["confidence"].mean()  # means with bootstrap intervals
rs.usage()  # calls, tokens, cost, errors
df = rs.to_pandas()  # one row per sample, with metadata
```

Use `ep.FakeClient("model-a")` instead of a real client to try everything offline.

- **The schema decides the summaries.** The pydantic model is the answer format sent to the model and the rule for how each field is summarized.
- **Intervals by default**, always with `n`.
- **Every sample is kept**, with its prompt, model snapshot, tokens, latency and cost.
- **Failures are data.** Refusals and invalid answers are recorded, not retried.
- **Cheap reruns.** A disk cache keyed on each sample; raising `n` only makes the new calls.
- **No logprobs needed**, so it works with frontier APIs.

Simulated answers show a model's tendencies. They are not real voters or customers.

## Develop

```sh
uv sync --all-groups
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest                       # offline; also runs every docs example
uv run pytest -m live               # real APIs; needs keys and costs a few cents
uv run zensical serve               # docs at http://localhost:8000
```

Design notes and the roadmap are in [`design/`](design/).

## License

MIT.
