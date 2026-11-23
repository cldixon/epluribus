"""The JSON schemas epluribus sends to providers stay within what their APIs accept.

Real API calls are the final check (tests/live). These catch schema problems offline,
for every kind of field epluribus summarizes.
"""

from __future__ import annotations

from enum import Enum, IntEnum
from typing import Annotated, Any, Literal

import anthropic
import pytest
from openai.lib._pydantic import to_strict_json_schema
from pydantic import BaseModel, Field

import epluribus as ep

from .conftest import Everything, Vote

# Keywords Anthropic structured outputs don't support; the SDK must remove them.
ANTHROPIC_UNSUPPORTED = {
    "minimum",
    "maximum",
    "exclusiveMinimum",
    "exclusiveMaximum",
    "multipleOf",
    "minLength",
    "maxLength",
    "minItems",
    "maxItems",
    "uniqueItems",
}


class Mood(Enum):
    CALM = "calm"
    TENSE = "tense"


class Rank(IntEnum):
    LOW = 1
    HIGH = 2


class Markers(BaseModel):
    """Markers, enums in lists, optional values and descriptions."""

    urgency: Annotated[Literal["low", "medium", "high"], ep.Ordinal()]
    priority: Annotated[Literal[1, 2, 3], ep.Categorical()]
    score: Annotated[int, Field(ge=1, le=5, description="1 = low, 5 = high"), ep.Numeric()]
    code: Annotated[Literal["A1", "B2"], ep.Text()]
    moods: list[Mood]
    rank: Rank
    estimate: float = Field(ge=0, le=1, description="A share")
    second: Literal["x", "y"] | None = None
    tags: list[str]


SCHEMAS = [Everything, Vote, Markers]


def walk(node: Any, path: str = "$"):
    if isinstance(node, dict):
        yield path, node
        for k, v in node.items():
            yield from walk(v, f"{path}.{k}")
    elif isinstance(node, list):
        for i, v in enumerate(node):
            yield from walk(v, f"{path}[{i}]")


def objects(schema: dict) -> list[tuple[str, dict]]:
    return [(p, n) for p, n in walk(schema) if n.get("type") == "object"]


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda s: s.__name__)
def test_anthropic_schema_is_supported(schema: type[BaseModel]) -> None:
    sent = anthropic.transform_schema(schema)
    for path, node in walk(sent):
        bad = ANTHROPIC_UNSUPPORTED & set(node)
        assert not bad, f"{path} uses unsupported {bad}"
    for path, node in objects(sent):
        assert node.get("additionalProperties") is False, path
    # Bounds the API can't enforce stay visible to the model in the description.
    if schema is Markers:
        score = sent["properties"]["score"]["description"]
        assert "1 = low, 5 = high" in score
        assert "maximum: 5" in score


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda s: s.__name__)
def test_openai_schema_is_strict(schema: type[BaseModel]) -> None:
    sent = to_strict_json_schema(schema)
    for path, node in objects(sent):
        assert node.get("additionalProperties") is False, path
        assert set(node.get("required", [])) == set(node.get("properties", {})), path


@pytest.mark.parametrize("schema", SCHEMAS, ids=lambda s: s.__name__)
def test_fake_answers_round_trip(schema: type[BaseModel]) -> None:
    # Every schema the tests use can be answered and summarized end to end.
    rs = ep.run(ep.Task("Q", schema), ep.FakeClient(), n=5)
    assert len(rs.ok) == 5
    assert rs.summary()
