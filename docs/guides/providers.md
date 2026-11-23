# Providers

All clients share these options:

| Option | Meaning |
|---|---|
| `model` | The provider's model name. |
| `label` | How the model appears in results. Defaults to `model`. Give two clients of the same model different labels to compare their settings. |
| `pricing` | `(input, output)` USD per million tokens, for `cost_usd`. |
| `max_concurrency` | Most calls in flight to this client, across runs. |

## Anthropic

```python
from epluribus.clients.anthropic import AnthropicClient

sonnet = AnthropicClient("claude-sonnet-5-5")
quick = AnthropicClient("claude-sonnet-5-5", effort="low", label="sonnet-low")
```

- Needs `pip install "epluribus[anthropic]"` and `ANTHROPIC_API_KEY` (or another credential the Anthropic SDK finds).
- Answers use the API's native structured outputs, so the reply has the schema's JSON shape.
- The API can't enforce some constraints, such as `ge` and `le`, while the model writes. epluribus checks them afterwards; see [bounds](numbers.md#bounds-are-checked).
- **`temperature` is not accepted** by current Claude models. `ep.run(..., temperature=...)` raises before any call. Get variety from [personas and phrasings](sweeps.md#getting-variety).
- `effort` (`"low"` to `"max"`) trades depth for cost and speed.
- `extra_params` passes extra arguments to `messages.create`, for example `{"thinking": {"type": "adaptive", "display": "summarized"}}`. They are part of the cache key.
- A refusal is recorded with `error_kind="refusal"`.

## OpenAI

```python
from epluribus.clients.openai import OpenAIClient

gpt = OpenAIClient("gpt-5-mini", effort="low")
```

- Needs `pip install "epluribus[openai]"` and `OPENAI_API_KEY`.
- Uses the Responses API with strict structured outputs.
- Responses are not stored on OpenAI's side (`store=False`). Pass `extra_params={"store": True}` to change that.
- Reasoning models reject `temperature`; use `effort` with them.

## Fake models

`ep.FakeClient` makes up valid answers for any schema, offline and free. Use it for tests, docs and trying out a schema.

```python
import epluribus as ep

a = ep.FakeClient("model-a")
b = ep.FakeClient("model-b", error_rate=0.05)  # 5% of calls fail and are retried
```

Each fake model leans towards some options. The leanings depend on the label, the system prompt and the prompt, so different fakes and different prompts give different distributions. A higher `temperature` flattens them. Fake answers say nothing about real models.

## Write your own client

Subclass `ep.Client` and implement `agenerate`. Return a `Generation` with the validated answer and a `ResponseMeta`:

```python
import time
from pydantic import BaseModel
import epluribus as ep


class EchoClient(ep.Client):
    """Always votes approve. Replace the fixed text with a call to your model."""

    provider = "echo"

    async def agenerate(
        self, prompt, schema, *, system=None, temperature=None, seed=None, max_tokens=1024
    ):
        t0 = time.perf_counter()
        text = '{"choice": "approve"}'  # call your model here
        output = self.validate(schema, text)  # raises InvalidOutputError if it doesn't fit
        meta = ep.ResponseMeta(
            provider=self.provider, model=self.model, seed=seed, latency_s=time.perf_counter() - t0
        )
        return ep.Generation(output=output, meta=meta, raw_text=text)


class Vote(BaseModel):
    choice: str


rs = ep.run(ep.Task("Vote?", Vote), EchoClient("echo-1"), n=3)
```

To control retries, raise `ep.ProviderError(message, kind=..., retryable=True, retry_after=...)`. Other exceptions are recorded and not retried. Override `settings()` to put client options in the cache key, and `check()` to reject unsupported run options before any call.
