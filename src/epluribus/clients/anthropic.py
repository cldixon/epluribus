"""Claude models through the Anthropic Messages API.

Install the extra first: `pip install "epluribus[anthropic]"`.
"""

from __future__ import annotations

import time
from typing import Any

from pydantic import BaseModel

from ..errors import RefusalError, TruncatedError
from ..records import ResponseMeta, Usage
from ._sdk import dump, translate
from .base import Client, Generation

try:
    import anthropic
except ImportError as e:  # pragma: no cover - exercised only without the extra
    msg = 'AnthropicClient needs the anthropic SDK: pip install "epluribus[anthropic]"'
    raise ImportError(msg) from e


class AnthropicClient(Client):
    """A Claude model.

    Answers use the API's native structured outputs (`output_config.format`), so the
    model's reply is constrained to the schema's JSON shape. The SDK drops
    constraints the API can't enforce, such as `ge`/`le`, from the schema it sends;
    epluribus still validates them, and an answer that breaks one is recorded as
    `invalid_output` with its raw text.

    Current Claude models don't accept `temperature`, so leave it unset. To vary
    answers, sweep personas or phrasings with a grid, or use several templates.

    Args:
        model: Model ID, for example `"claude-opus-5-5"` or `"claude-haiku-5-5"`.
        effort: Optional effort level (`"low"`, `"medium"`, `"high"`, `"xhigh"`,
            `"max"`). Lower effort is cheaper and faster.
        sdk_client: An `anthropic.AsyncAnthropic` to use. By default one is made
            for each event loop, with the SDK's own retries off: epluribus retries
            and records the attempts itself.
        extra_params: Extra keyword arguments for `messages.create`, for example
            `{"thinking": {"type": "adaptive"}}`. They are part of the cache key.
        **kw: Passed to `Client` (`label`, `pricing`, `max_concurrency`).
    """

    provider = "anthropic"

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

    def check(self, *, temperature: float | None = None) -> None:
        if temperature is not None:
            msg = (
                "The Anthropic API no longer accepts `temperature`. Leave it unset, and "
                "add variety with a grid of personas or phrasings, or several templates."
            )
            raise ValueError(msg)

    def _sdk(self) -> Any:
        if self._sdk_client is not None:
            return self._sdk_client
        return self.loop_local(lambda: anthropic.AsyncAnthropic(max_retries=0))

    def request(
        self, prompt: str, schema: type[BaseModel], *, system: str | None, max_tokens: int
    ) -> dict[str, Any]:
        """The keyword arguments sent to `messages.create`."""
        output_config: dict[str, Any] = {
            "format": {"type": "json_schema", "schema": anthropic.transform_schema(schema)}
        }
        if self.effort is not None:
            output_config["effort"] = self.effort
        kwargs: dict[str, Any] = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
            "output_config": output_config,
        }
        if system:
            kwargs["system"] = system
        extra = dict(self.extra_params)
        if "output_config" in extra:
            output_config.update(extra.pop("output_config"))
        kwargs.update(extra)
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
        self.check(temperature=temperature)
        kwargs = self.request(prompt, schema, system=system, max_tokens=max_tokens)
        t0 = time.perf_counter()
        try:
            resp = await self._sdk().messages.create(**kwargs)
        except Exception as e:
            raise translate(e, anthropic) from e
        latency = time.perf_counter() - t0

        text = "".join(b.text for b in resp.content if b.type == "text") or None
        if resp.stop_reason == "refusal":
            details = dump(getattr(resp, "stop_details", None)) or {}
            category = details.get("category")
            msg = f"the model declined to answer (category: {category})"
            raise RefusalError(msg, raw_text=text)
        if resp.stop_reason == "max_tokens":
            raise TruncatedError(f"answer cut off at max_tokens={max_tokens}", raw_text=text)
        output = self.validate(schema, text)

        u = resp.usage
        details = getattr(u, "output_tokens_details", None)
        usage = Usage(
            input_tokens=u.input_tokens,
            output_tokens=u.output_tokens,
            reasoning_tokens=getattr(details, "thinking_tokens", None),
            cached_input_tokens=getattr(u, "cache_read_input_tokens", None),
        )
        meta = ResponseMeta(
            provider=self.provider,
            model=self.model,
            resolved_model=resp.model,
            request_id=getattr(resp, "_request_id", None) or resp.id,
            temperature=None,
            seed=seed,
            max_tokens=max_tokens,
            usage=usage,
            stop_reason=resp.stop_reason,
            latency_s=latency,
            cost_usd=self.cost(usage),
            raw={"message_id": resp.id, "usage": dump(u), "seed_sent": False},
        )
        return Generation(output=output, meta=meta, raw_text=text)
