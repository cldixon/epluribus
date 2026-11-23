"""Contract tests against the real provider APIs.

Run with `uv run pytest -m live`. They need API keys, make a few small calls and
cost a few cents. Model names can be changed with environment variables.
"""

from __future__ import annotations

import os
from typing import Literal

import pytest
from pydantic import BaseModel, Field

import epluribus as ep

pytestmark = pytest.mark.live


class Vote(BaseModel):
    """Your vote."""

    choice: Literal["approve", "reject", "abstain"]
    confidence: int = Field(ge=1, le=5)
    reason: str = Field(description="One short sentence.")


TASK = ep.Task("You are $role. Do you support a weekend car ban downtown? Vote.", Vote)
GRID = {"role": ["a retiree", "a cyclist"]}


def check(rs: ep.ResultSet, provider: str) -> None:
    assert len(rs) == 4
    assert rs.ok, [r.error for r in rs.errors]
    r = rs.ok.records[0]
    assert r.meta.provider == provider
    assert r.meta.resolved_model
    assert r.meta.request_id
    assert r.meta.usage.input_tokens
    assert r.meta.usage.output_tokens
    assert r.raw_text
    assert sum(rs["choice"].counts().values()) == rs["choice"].n


needs_anthropic = pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"), reason="needs ANTHROPIC_API_KEY"
)
ANTHROPIC_MODEL = os.environ.get("EPLURIBUS_ANTHROPIC_MODEL", "claude-haiku-5-5")


@needs_anthropic
def test_anthropic():
    from epluribus.clients.anthropic import AnthropicClient

    rs = ep.run(TASK, AnthropicClient(ANTHROPIC_MODEL, effort="low"), n=2, grid=GRID)
    check(rs, "anthropic")
    assert rs.records[0].meta.max_tokens == 16_000


@needs_anthropic
def test_anthropic_every_field_kind():
    """The API accepts the schema for every kind of field epluribus summarizes."""
    from epluribus.clients.anthropic import AnthropicClient

    from ..test_schemas import Markers

    task = ep.Task("Fill in a plausible example answer.", Markers)
    rs = ep.run(task, AnthropicClient(ANTHROPIC_MODEL, effort="low"), n=1)
    assert rs.ok, [(r.error_kind, r.error, r.raw_text) for r in rs.errors]


@needs_anthropic
def test_anthropic_failures_are_classified():
    from epluribus.clients.anthropic import AnthropicClient

    short = ep.run(TASK, AnthropicClient(ANTHROPIC_MODEL), n=1, grid=GRID, max_tokens=16)
    assert {r.error_kind for r in short} == {"truncated"}

    missing = ep.run(TASK, AnthropicClient("claude-nonexistent"), n=1, grid=GRID)
    for r in missing:
        assert r.error_kind == "not_found"
        assert r.meta.attempts == 1


@pytest.mark.skipif(not os.environ.get("OPENAI_API_KEY"), reason="needs OPENAI_API_KEY")
def test_openai():
    from epluribus.clients.openai import OpenAIClient

    model = os.environ.get("EPLURIBUS_OPENAI_MODEL", "gpt-5-mini")
    rs = ep.run(TASK, OpenAIClient(model, effort="low"), n=2, grid=GRID)
    check(rs, "openai")
