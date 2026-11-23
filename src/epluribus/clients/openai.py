"""OpenAI models through the Responses API.

Install the extra first: `pip install "epluribus[openai]"`.
"""

from __future__ import annotations

import re
import time
from typing import Any

from pydantic import BaseModel

from ..errors import RefusalError, TruncatedError
from ..records import ResponseMeta, Usage
from ._sdk import dump, translate
from .base import Client, Generation

try:
    import openai
    from openai.lib._pydantic import to_strict_json_schema
except ImportError as e:  # pragma: no cover - exercised only without the extra
    msg = 'OpenAIClient needs the openai SDK: pip install "epluribus[openai]"'
    raise ImportError(msg) from e


class OpenAIClient(Client):
    """An OpenAI model.

    Answers use strict structured outputs (`text.format` with a JSON schema), and
    responses are not stored on OpenAI's side (`store=False`) unless you pass
    `extra_params={"store": True}`.

    Reasoning models reject `temperature`; with those, leave it unset and use
    `effort` instead. Reasoning tokens count against `max_tokens`, so the default
    is 16,000. OpenAI's Responses API has no seed parameter.

    Args:
        model: Model name.
        effort: Optional reasoning effort for reasoning models, for example `"low"`.
        sdk_client: An `openai.AsyncOpenAI` to use. By default one is made for each
            event loop, with the SDK's own retries off.
        extra_params: Extra keyword arguments for `responses.create`. They are part
            of the cache key.
        **kw: Passed to `Client` (`label`, `pricing`, `max_concurrency`).
    """

    provider = "openai"
    default_max_tokens = 16_000

    def __init__(
        self,
        model: str,
        *,
        effort: str | None = None,
        sdk_client: Any = None,
        extra_params: dict[str, Any] | None = None,
        **kw: Any,
    ) -> None:
        super().__init__(model, **kw)
        self.effort = effort
        self.extra_params = dict(extra_params or {})
        self._sdk_client = sdk_client

    def settings(self) -> dict[str, Any]:
        return {"effort": self.effort, "extra_params": self.extra_params}

    def _sdk(self) -> Any:
        if self._sdk_client is not None:
            return self._sdk_client
        return self.loop_local(lambda: openai.AsyncOpenAI(max_retries=0))

    def request(
        self,
        prompt: str,
        schema: type[BaseModel],
        *,
        system: str | None,
        temperature: float | None,
        max_tokens: int,
    ) -> dict[str, Any]:
        """The keyword arguments sent to `responses.create`."""
        kwargs: dict[str, Any] = {
            "model": self.model,
            "input": [{"role": "user", "content": prompt}],
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": _schema_name(schema),
                    "schema": to_strict_json_schema(schema),
                    "strict": True,
                }
            },
            "max_output_tokens": max_tokens,
            "store": False,
        }
        if system:
            kwargs["instructions"] = system
        if temperature is not None:
            kwargs["temperature"] = temperature
        if self.effort is not None:
            kwargs["reasoning"] = {"effort": self.effort}
        kwargs.update(self.extra_params)
        return kwargs

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
        kwargs = self.request(
            prompt, schema, system=system, temperature=temperature, max_tokens=max_tokens
        )
        t0 = time.perf_counter()
        try:
            resp = await self._sdk().responses.create(**kwargs)
        except Exception as e:
            raise translate(e, openai) from e
        latency = time.perf_counter() - t0

        text = resp.output_text or None
        refusal = _refusal(resp)
        incomplete = getattr(getattr(resp, "incomplete_details", None), "reason", None)
        if refusal is not None or incomplete == "content_filter":
            raise RefusalError(f"the model declined to answer: {refusal or incomplete}")
        if incomplete == "max_output_tokens":
            msg = (
                f"answer cut off at max_tokens={max_tokens}. Thinking counts against this "
                "limit: raise max_tokens or lower effort."
            )
            raise TruncatedError(msg, raw_text=text)
        output = self.validate(schema, text)

        u = resp.usage
        usage = Usage(
            input_tokens=getattr(u, "input_tokens", None),
            output_tokens=getattr(u, "output_tokens", None),
            reasoning_tokens=getattr(
                getattr(u, "output_tokens_details", None), "reasoning_tokens", None
            ),
            cached_input_tokens=getattr(
                getattr(u, "input_tokens_details", None), "cached_tokens", None
            ),
        )
        meta = ResponseMeta(
            provider=self.provider,
            model=self.model,
            resolved_model=resp.model,
            request_id=getattr(resp, "_request_id", None) or resp.id,
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            usage=usage,
            stop_reason=incomplete or resp.status,
            latency_s=latency,
            cost_usd=self.cost(usage),
            raw={"response_id": resp.id, "usage": dump(u), "seed_sent": False},
        )
        return Generation(output=output, meta=meta, raw_text=text)


def _schema_name(schema: type[BaseModel]) -> str:
    # The API wants ^[a-zA-Z0-9_-]{1,64}$.
    return re.sub(r"[^a-zA-Z0-9_-]", "_", schema.__name__)[:64] or "answer"


def _refusal(resp: Any) -> str | None:
    for item in getattr(resp, "output", None) or []:
        for part in getattr(item, "content", None) or []:
            if getattr(part, "type", None) == "refusal":
                return getattr(part, "refusal", None) or "refused"
    return None
