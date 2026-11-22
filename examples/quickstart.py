"""Quickstart with fake models: one schema, a sweep grid, two models, then aggregate."""

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
    second_choice: Literal["support", "oppose"] | None = None  # optional (may abstain)
    reason: str  # text, not aggregated


task = ep.Task(
    template="You are {persona}. {framing}\nShould the city ban cars downtown on weekends?",
    schema=Opinion,
)

models = [
    ep.FakeClient("model-a", pricing=(3.0, 15.0)),
    ep.FakeClient("model-b", pricing=(1.0, 5.0), error_rate=0.03),
]

grid = {
    "persona": ["a retiree", "a cyclist", "a shop owner"],
    "framing": ["Air quality is worsening.", "Parking revenue funds schools."],
}

if __name__ == "__main__":
    print("Inferred field kinds:")
    for spec in ep.schema_specs(Opinion).values():
        print("  ", spec)

    cache = ep.MemoryCache()
    rs = ep.run(task, models, n=25, grid=grid, cache=cache, seed=7)
    print("\n", rs)
    print("usage:", rs.usage())

    print("\n-- whole sweep --")
    for name, s in rs.summary().items():
        print(f"{name}: {s}")

    print("\n-- stance by model --")
    for model, freq in rs.group_by("model")["stance"].freq().items():
        print(" ", model, freq)

    print("\n-- would_vote by persona --")
    print(rs.group_by("persona")["would_vote"].rate())

    print("\n-- mean confidence by (persona, framing) --")
    for cell, est in rs.group_by("persona", "framing")["confidence"].mean().items():
        print(" ", cell, est)

    print("\n-- filter + arrays --")
    sub = rs.filter(persona="a cyclist", model="model-a")
    arr = sub.arrays(["stance", "confidence", "concerns"])
    print({k: (v.shape, v[:3].tolist()) for k, v in arr.items()})

    print("\n-- kind checks --")
    try:
        rs["stance"].mean()
    except ep.KindError as e:
        print("KindError:", e)

    print("\n-- metadata on one record --")
    r = rs.ok.records[0]
    print(r.meta.provider, r.meta.resolved_model, r.meta.usage, r.meta.cost_usd)
    if rs.errors:
        print("first error:", rs.errors[0].coords, rs.errors[0].error)

    print("\n-- rerun hits the cache --")
    rs2 = ep.run(task, models, n=25, grid=grid, cache=cache, seed=7)
    print(rs2.usage())

    try:
        df = rs.to_pandas()
        print("\nDataFrame:", df.shape, list(df.columns)[:12], "...")
    except ImportError:
        pass
