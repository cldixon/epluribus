"""A small real poll: two Claude models, two roles, 5 samples each (20 calls).

Needs `ANTHROPIC_API_KEY`. Costs a few cents. Answers are cached in
`.epluribus-cache.sqlite`, so running it again is free.

    uv run --extra anthropic examples/live_poll.py
"""

from typing import Literal

from pydantic import BaseModel, Field

import epluribus as ep
from epluribus.clients.anthropic import AnthropicClient


class Vote(BaseModel):
    """Your vote on the proposal."""

    choice: Literal["approve", "reject", "abstain"]
    confidence: int = Field(ge=1, le=5, description="1 = coin flip, 5 = certain")
    reason: str = Field(description="One sentence.")


task = ep.Task(
    "You are $role. The city council is voting on converting two downtown parking "
    "garages into affordable housing. You are a council member. How do you vote?",
    Vote,
)

models = [
    AnthropicClient("claude-sonnet-5-5", effort="low", pricing=(2.0, 10.0)),
    AnthropicClient("claude-haiku-5-5", effort="low", pricing=(0.10, 0.50)),
]


def main() -> None:
    cache = ep.DiskCache(".epluribus-cache.sqlite")
    rs = ep.run(
        task,
        models,
        n=5,
        grid={"role": ["a retired homeowner", "a renter who bikes to work"]},
        cache=cache,
        progress=True,
    )
    print(rs)
    print(rs.group_by("model", "role")["choice"].freq())
    print(rs.usage())
    for r in rs.errors:
        print("error:", r.coords, r.error_kind, r.error)


if __name__ == "__main__":
    main()
