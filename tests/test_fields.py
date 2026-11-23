from __future__ import annotations

from enum import Enum
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, Field

import epluribus as ep
from epluribus.fields import Kind

from .conftest import Everything


def test_inference_covers_every_kind():
    kinds = {n: s.kind for n, s in ep.schema_specs(Everything).items()}
    assert kinds == {
        "cat": Kind.CATEGORICAL,
        "ordl": Kind.ORDINAL,
        "likert": Kind.ORDINAL,
        "count": Kind.NUMERIC,
        "num": Kind.NUMERIC,
        "flag": Kind.BOOLEAN,
        "multi": Kind.MULTI_SELECT,
        "marked": Kind.ORDINAL,
        "color": Kind.CATEGORICAL,
        "level": Kind.ORDINAL,
        "maybe": Kind.CATEGORICAL,
        "text": Kind.TEXT,
    }
    specs = ep.schema_specs(Everything)
    assert specs["likert"].levels == (1, 2, 3, 4, 5)
    assert specs["maybe"].optional
    assert specs["count"].integer


class Shade(Enum):
    LIGHT = "light"
    DARK = "dark"


class Edge(BaseModel):
    exclusive: int = Field(gt=0, lt=4)
    wide: int = Field(ge=0, le=100)
    wide_marked: Annotated[int, Field(ge=0, le=100), ep.Ordinal()]
    as_number: Annotated[int, Field(ge=1, le=5), ep.Numeric()]
    as_text: Annotated[Literal["a", "b"], ep.Text()]
    as_cat: Annotated[Literal[1, 2, 3], ep.Categorical()]
    reordered: Annotated[Literal["mid", "lo", "hi"], ep.Ordinal(levels=("lo", "mid", "hi"))]
    shades: list[Shade]
    tags: list[str]


def test_edge_cases():
    s = ep.schema_specs(Edge)
    assert s["exclusive"].levels == (1, 2, 3)
    assert s["wide"].kind == Kind.NUMERIC  # too many levels to be ordinal
    assert s["wide_marked"].kind == Kind.ORDINAL
    assert s["as_number"].kind == Kind.NUMERIC
    assert s["as_text"].kind == Kind.TEXT
    assert s["as_cat"].kind == Kind.CATEGORICAL
    assert s["reordered"].levels == ("lo", "mid", "hi")
    assert s["shades"].kind == Kind.MULTI_SELECT
    assert s["shades"].levels == (Shade.LIGHT, Shade.DARK)
    assert s["tags"].kind == Kind.TEXT
    assert "levels" in repr(s["exclusive"])


def test_ordinal_levels_must_cover_declared_values():
    class Bad(BaseModel):
        x: Annotated[Literal["a", "b", "c"], ep.Ordinal(levels=("a", "b"))]

    with pytest.raises(ValueError, match="missing"):
        ep.schema_specs(Bad)


def test_rank():
    s = ep.schema_specs(Everything)
    assert s["marked"].rank("hi") == 2.0
    assert s["likert"].rank(4) == 4.0
    levels = s["level"].levels
    assert levels is not None
    assert s["level"].rank(levels[1]) == 2.0
