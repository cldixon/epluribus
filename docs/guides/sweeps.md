# Sweeps

A sweep asks the same question under different conditions: roles, situations, framings. Put placeholders in the template and pass a `grid`. Every combination of grid values is a **cell**, and each cell gets `n` samples per model.

## Placeholders

Write placeholders as `$name` or `${name}`. Braces need no escaping, so you can put JSON examples in a prompt. Write a literal dollar sign as `$$`.

```python
import epluribus as ep
from typing import Annotated, Literal
from pydantic import BaseModel
from epluribus.clients.anthropic import AnthropicClient

sonnet = AnthropicClient("claude-sonnet-5-5", effort="low")


class Decision(BaseModel):
    """The department's decision."""

    option: Literal["hire_contractor", "train_staff", "delay"]
    urgency: Annotated[Literal["low", "medium", "high"], ep.Ordinal()]
    risks: list[Literal["cost", "quality", "morale", "schedule"]]


task = ep.Task(
    "You are $role in a mid-sized city's IT department. $situation\n"
    "The legacy permit system must be replaced this year. What should the department do?",
    Decision,
)

grid = {
    "role": ["the department head", "the budget officer", "a senior engineer"],
    "situation": ["The budget was just cut 10%.", "A security audit flagged the old system."],
}

rs = ep.run(task, sonnet, n=20, grid=grid)  # 3 x 2 cells x 20 = 120 calls
```

Each record keeps its grid values in `coords`, so you can group and filter on them:

```python
rs.group_by("role", "situation")["option"].mode()
rs.group_by("situation")["risks"].freq()  # share of answers flagging each risk
rs.filter(role="the budget officer")["urgency"].freq()
rs.group_by("situation")["urgency"].mean()  # mean rank: low=0, medium=1, high=2
```

Text choices are unordered by default. The `ep.Ordinal()` marker on `urgency` says the levels have an order, which allows `mean`, `median` and `quantiles`.

`ep.run` checks the grid before it makes any call. A placeholder without grid values, a misspelled name, or a grid name that is also an answer field raises an error.

## Short labels for long values

Give a dict instead of a list. The template gets the value; your results show the key:

```python
personas = {
    "retiree": "a retired teacher who has lived downtown for 40 years",
    "cyclist": "a bike courier who rides downtown every day",
    "owner": "the owner of a small restaurant near the garages",
}


class Vote(BaseModel):
    choice: Literal["approve", "reject", "abstain"]


vote_task = ep.Task(
    "You are $persona. The council votes on converting a downtown garage "
    "into housing. How do you vote?",
    Vote,
)
rs = ep.run(vote_task, sonnet, n=10, grid={"persona": personas})
rs.group_by("persona")["choice"].freq()  # rows: retiree, cyclist, owner
```

## Ask in several phrasings

A finding that depends on one phrasing is fragile. Pass a list of templates to ask the question several ways. `n` applies to each phrasing, and each record stores its phrasing's index as `coords["variant"]`:

```python
paraphrased = ep.Task(
    [
        "Should the city replace its permit system this year? You are $role.",
        "As $role, would you push to replace the permit system now?",
        "You are $role. Is replacing the permit system this year the right call?",
    ],
    Decision,
)
rs = ep.run(paraphrased, sonnet, n=10, grid={"role": ["the budget officer", "a senior engineer"]})
rs.group_by("variant")["option"].freq()  # does the answer depend on the phrasing?
rs.group_by("role")["option"].freq()  # pooled over phrasings
```

## System prompts

A task's `system` prompt can use placeholders too:

```python
task = ep.Task(
    "The council votes on converting a downtown garage into housing. How do you vote?",
    Vote,
    system="You are $persona. Answer as that person would.",
)
rs = ep.run(task, sonnet, n=10, grid={"persona": personas})
```

## Getting variety

Each call is an independent sample, so answers vary somewhat. But a model with a strong default can give the same answer nearly every time. Raising `n` won't change that. Changing the question will:

- **Sweep personas.** Ask as different people, with a grid. This is usually the strongest lever.
- **Sweep phrasings.** Pass several templates. A finding that survives every phrasing is worth more than one from a single prompt.
- **Compare models.** Different models have different defaults.
- **Use `temperature`** only with providers that accept it. Current Claude models don't, and many reasoning models don't either.

A narrow spread is a finding too: it says the model has a strong default. `rs["field"].entropy()` measures the spread in bits; 0 means every answer agreed.
