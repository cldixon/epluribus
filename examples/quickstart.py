"""A tour of the API with fake models: one schema, a sweep grid, two models, summaries.

Run it with `uv run examples/quickstart.py`. No API key needed.
"""

from typing import Annotated, Literal

from pydantic import BaseModel, Field

import epluribus as ep


class Opinion(BaseModel):
    """Your answer to the poll."""

    stance: Literal["support", "oppose", "unsure"]  # categorical
    confidence: int = Field(ge=1, le=5, description="1 = guessing, 5 = certain")  # ordinal
    support_level: float = Field(ge=0, le=1)  # numeric
    would_vote: bool  # boolean
    concerns: list[Literal["cost", "traffic", "businesses", "safety"]]  # multi-select
    priority: Annotated[Literal["low", "medium", "high"], ep.Ordinal()]  # ordinal by marker
    second_choice: Literal["support", "oppose"] | None = None  # optional
    reason: str  # text, kept but not summarized


task = ep.Task(
    [
        "You are $persona. $framing\nShould the city ban cars downtown on weekends?",
        "$framing You are $persona. Would you back a weekend car ban downtown?",
    ],
    Opinion,
)

models = [
    ep.FakeClient("model-a", pricing=(3.0, 15.0)),
    ep.FakeClient("model-b", pricing=(1.0, 5.0), error_rate=0.03),
]

grid = {
    "persona": {
        "retiree": "a retiree who walks downtown daily",
        "cyclist": "a bike commuter",
        "owner": "a shop owner on Main Street",
    },
    "framing": ["Air quality is worsening.", "Parking revenue funds schools."],
}


def main() -> None:
    print("Inferred field kinds:")
    for spec in ep.schema_specs(Opinion).values():
        print("  ", spec)

    print("\nCalls planned:", len(ep.plan(task, models, n=25, grid=grid)))
    cache = ep.MemoryCache()
    rs = ep.run(task, models, n=25, grid=grid, cache=cache, seed=7)
    print()
    print(rs)
    print()
    print(rs.usage())

    print("\nStance by model:")
    print(rs.group_by("model")["stance"].freq())

    print("\nWould vote, by persona:")
    print(rs.group_by("persona")["would_vote"].rate())

    print("\nDoes the phrasing matter?")
    print(rs.group_by("variant")["stance"].diff(0, 1))

    print("\nMean confidence by persona and framing:")
    print(rs.group_by("persona", "framing")["confidence"].mean())

    try:
        rs["stance"].mean()
    except ep.KindError as e:
        print("\nKindError:", e)

    rs2 = ep.run(task, models, n=25, grid=grid, cache=cache, seed=7)
    print("\nRerun with the cache:", rs2.usage()["cache_hits"], "cache hits")


if __name__ == "__main__":
    main()
