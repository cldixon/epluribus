"""The client interface. A client turns a prompt and a schema into a `Generation`."""

from __future__ import annotations

import asyncio
import inspect
import time
import weakref
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, TypeVar

from pydantic import BaseModel, ValidationError

from ..errors import InvalidOutputError
from ..records import ResponseMeta, Usage

T = TypeVar("T")


@dataclass
class Generation:
    """What a client returns for one successful call."""

    output: BaseModel
    meta: ResponseMeta
    raw_text: str | None = None


class Client:
    """Base class for model integrations.

    Subclasses set `provider`, implement `agenerate`, and fill `ResponseMeta` with
    whatever the provider returns. Anything without a normalized slot goes in
    `meta.raw`. To mark a failure as worth retrying, raise
    `epluribus.errors.ProviderError(..., retryable=True)`; other exceptions are
    recorded and not retried.

    Args:
        model: The provider's model name.
        label: How the model appears in results (`coords["model"]`). Defaults to `model`.
        pricing: `(input_usd, output_usd)` per million tokens, used for `cost_usd`.
            Reasoning tokens are counted as output tokens.
        max_concurrency: Most calls in flight to this client at once, across a run.
            `None` means only the run's `concurrency` limit applies.
    """

    provider: str = "base"
    default_max_tokens: int = 1024
    """`max_tokens` for a run that doesn't set it."""

    def __init__(
        self,
        model: str,
        *,
        label: str | None = None,
        pricing: tuple[float, float] | None = None,
        max_concurrency: int | None = None,
    ) -> None:
        self.model = model
        self.label = label or model
        self.pricing = pricing
        self.max_concurrency = max_concurrency
        self._cooldown_until = 0.0
        self._limiters: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, asyncio.Semaphore] = (
            weakref.WeakKeyDictionary()
        )
        self._per_loop: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, Any] = (
            weakref.WeakKeyDictionary()
        )
        self._runs: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, int] = (
            weakref.WeakKeyDictionary()
        )

    async def agenerate(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        system: str | None = None,
        temperature: float | None = None,
        seed: int | None = None,
        max_tokens: int = 1024,
    ) -> Generation:
        """Make one call and return the validated answer with its metadata."""
        raise NotImplementedError

    def check(self, *, temperature: float | None = None) -> None:
        """Raise `ValueError` before a run starts if the settings can't work with this client."""

    def settings(self) -> dict[str, Any]:
        """Client settings that change the answer, such as a reasoning effort level.

        They are part of the cache key, so changing one makes new calls.
        """
        return {}

    def cache_identity(self) -> dict[str, Any]:
        """Everything about the client that belongs in the cache key."""
        return {"provider": self.provider, "model": self.model, **self.settings()}

    def cost(self, usage: Usage) -> float | None:
        """Cost in USD from `pricing`, or `None` if pricing or token counts are missing."""
        if not self.pricing or usage.input_tokens is None or usage.output_tokens is None:
            return None
        pin, pout = self.pricing
        return (usage.input_tokens * pin + usage.output_tokens * pout) / 1_000_000

    # --- rate limiting --------------------------------------------------------

    def limiter(self) -> asyncio.Semaphore | None:
        """This client's concurrency limit for the running event loop."""
        if self.max_concurrency is None:
            return None
        loop = asyncio.get_running_loop()
        if loop not in self._limiters:
            self._limiters[loop] = asyncio.Semaphore(self.max_concurrency)
        return self._limiters[loop]

    def pause(self, seconds: float) -> None:
        """Hold new calls to this client for `seconds` (after a rate-limit response)."""
        self._cooldown_until = max(self._cooldown_until, time.monotonic() + seconds)

    async def wait_ready(self) -> None:
        """Wait out any pause set by `pause`."""
        while (delay := self._cooldown_until - time.monotonic()) > 0:
            await asyncio.sleep(delay)

    # --- helpers for subclasses -----------------------------------------------

    def begin_run(self) -> None:
        """Note that a run using this client started on the running event loop."""
        loop = asyncio.get_running_loop()
        self._runs[loop] = self._runs.get(loop, 0) + 1

    async def end_run(self) -> None:
        """Note that a run ended. After the last one on this loop, close what `loop_local` opened.

        Connections an SDK client keeps open would otherwise be left behind when the
        event loop closes. Clients passed in by the caller are never closed here.
        """
        loop = asyncio.get_running_loop()
        remaining = self._runs.get(loop, 1) - 1
        if remaining > 0:
            self._runs[loop] = remaining
            return
        self._runs.pop(loop, None)
        obj = self._per_loop.pop(loop, None)
        close = getattr(obj, "close", None)
        if callable(close):
            result = close()
            if inspect.isawaitable(result):
                await result

    def loop_local(self, factory: Callable[[], T]) -> T:
        """One object per event loop, for SDK clients whose connections belong to a loop."""
        loop = asyncio.get_running_loop()
        if loop not in self._per_loop:
            self._per_loop[loop] = factory()
        return self._per_loop[loop]

    @staticmethod
    def validate(schema: type[BaseModel], text: str | None) -> BaseModel:
        """Parse the model's JSON text into `schema`, or raise `InvalidOutputError`."""
        if not text:
            raise InvalidOutputError("the model returned no text", raw_text=text)
        try:
            return schema.model_validate_json(text)
        except ValidationError as e:
            msg = f"answer does not match {schema.__name__}: {e.error_count()} error(s); {e}"
            raise InvalidOutputError(msg, raw_text=text) from e

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.label!r})"
