# Field types

epluribus reads each field's type in your schema and decides its **kind**. The kind decides which summaries apply.

| You write | Kind | Summaries |
|---|---|---|
| `Literal["a", "b"]`, `Enum` | categorical | `freq`, `counts`, `mode`, `entropy` |
| `Literal[1, 2, 3]`, `IntEnum`, `int` with both bounds (at most 21 levels), `Annotated[..., ep.Ordinal()]` | ordinal | the categorical ones, plus `mean`, `median`, `quantiles`, `std` |
| `bool` | boolean | `rate`, `freq`, `counts`, `mean` |
| `float`, `int` without both bounds | numeric | `mean`, `median`, `std`, `quantiles`, `mode` |
| `list[Literal[...]]`, `list[Enum]` | multi-select | `freq` (share selecting each option), `counts`, `mean_selected` |
| `str`, and anything else | text | `sample`, `values` |
| `X \| None` | the kind of `X` | `None` counts as missing |

Every field also has `values()`, `n`, `missing`, `array()`, `summary()` and `sample()`. Fields that support a mean also support `diff()`, as do categorical and multi-select fields.

A summary that doesn't fit raises `ep.KindError`, for example `mean()` on a categorical field.

## Ordinal levels

For an ordinal field with text levels, such as `Literal["low", "medium", "high"]` with `ep.Ordinal()`, `mean` and `median` use ranks: 0, 1, 2 in declared order. Give a different order with `ep.Ordinal(levels=("low", "medium", "high"))`.

An `int` with both bounds is ordinal when it has at most 21 levels, as in `Field(ge=1, le=5)`. A wider range, such as `Field(ge=0, le=100)`, is numeric unless you mark it `ep.Ordinal()`.

## Markers

Add a marker with `Annotated` to override the inferred kind:

```python
from typing import Annotated, Literal
from pydantic import BaseModel, Field
import epluribus as ep


class Answer(BaseModel):
    urgency: Annotated[Literal["low", "medium", "high"], ep.Ordinal()]
    priority: Annotated[Literal[1, 2, 3], ep.Categorical()]  # numbers, but unordered
    score: Annotated[int, Field(ge=1, le=5), ep.Numeric()]  # bounded, but continuous
    code: Annotated[Literal["A1", "B2"], ep.Text()]  # never summarized


ep.schema_specs(Answer)
```

`ep.schema_specs(schema)` shows what epluribus inferred for each field.

## Field names

Avoid answer fields named `model` or `variant`: epluribus uses those names for coords. If you need one, `Task` warns you, and `"coords.model"` still reaches the coord.
