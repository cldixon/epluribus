from __future__ import annotations

import json
from collections.abc import Callable
from enum import Enum, IntEnum
from typing import Annotated, Any, Literal

import pytest
from pydantic import BaseModel, Field

import epluribus as ep


class Color(Enum):
    RED = "red"
    BLUE = "blue"


class Level(IntEnum):
    LOW = 1
    HIGH = 2


class Everything(BaseModel):
    """One field of every kind."""

    cat: Literal["a", "b", "c"]
    ordl: Literal[1, 2, 3]
    likert: int = Field(ge=1, le=5)
    count: int
    num: float
    flag: bool
    multi: list[Literal["x", "y"]]
    marked: Annotated[Literal["lo", "mid", "hi"], ep.Ordinal()]
    color: Color
    level: Level
    maybe: Literal["p", "q"] | None = None
    text: str


class Vote(BaseModel):
    """A vote."""

    choice: Literal["approve", "reject", "abstain"]
    confidence: int = Field(ge=1, le=5)
    reason: str


@pytest.fixture
def vote_task() -> ep.Task:
    return ep.Task("You are $role. How do you vote?", Vote)


Handler = Callable[[Any], Any]


class Recorder:
    """A mock HTTP transport that records requests and replays queued responses."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responses: list[tuple[int, dict[str, Any], dict[str, str]]] = []

    def queue(self, body: dict[str, Any], status: int = 200, **headers: str) -> None:
        self.responses.append((status, body, headers))

    def handler(self, request: Any) -> Any:
        import httpx2

        self.requests.append(json.loads(request.content))
        status, body, headers = self.responses.pop(0) if self.responses else (500, {}, {})
        return httpx2.Response(status, json=body, headers=headers)

    def http_client(self, sdk: Any) -> Any:
        import httpx2

        return sdk.DefaultAsyncHttpxClient(transport=httpx2.MockTransport(self.handler))


@pytest.fixture
def recorder() -> Recorder:
    return Recorder()
