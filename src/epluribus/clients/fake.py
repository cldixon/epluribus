"""A schema-aware fake client for tests, docs and demos. No network, no cost."""

from __future__ import annotations

import asyncio
import hashlib
import random
import time
from typing import Any

from pydantic import BaseModel

from ..errors import ProviderError
from ..fields import FieldSpec, Kind, schema_specs, unwrap_optional
from ..records import ResponseMeta, Usage
from .base import Client, Generation


class FakeClient(Client):
    """Produces valid answers for any schema without calling a model.

    Each fake model leans towards some options. The leanings are a stable function
    of the label, the system prompt, the prompt and the option, so different fake
    models and different prompts give different distributions. A higher
    `temperature` flattens the leanings; a lower one sharpens them.

    Args:
        model: Model name. Also the label unless `label` is given.
        error_rate: Share of calls that fail with a retryable `ProviderError`.
        latency: Upper bound on the simulated delay, in seconds.
        **kw: Passed to `Client` (`label`, `pricing`, `max_concurrency`).
    """

    provider = "fake"

    def __init__(
        self,
        model: str = "fake-1",
        *,
        error_rate: float = 0.0,
        latency: float = 0.005,
        **kw: Any,
    ) -> None:
        super().__init__(model, **kw)
        self.error_rate = error_rate
        self.latency = latency
        self._tries: dict[int | None, int] = {}

    def cache_identity(self) -> dict[str, Any]:
        # Fake answers depend on the label, so two labels must not share cache entries.
        return {**super().cache_identity(), "label": self.label}

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
        rng = random.Random(seed)
        t0 = time.perf_counter()
        await asyncio.sleep(self.latency * rng.random())
        # Failures depend on the try number too, so a retry can succeed.
        tries = self._tries[seed] = self._tries.get(seed, 0) + 1
        if random.Random(f"{seed}|{tries}").random() < self.error_rate:
            raise ProviderError("fake provider error (simulated)", kind="server", retryable=True)

        sharpness = 1.0 / max(temperature, 0.05) if temperature is not None else 1.0
        context = f"{self.label}|{system}|{prompt}"
        out = {
            name: self._fake_value(spec, schema, rng, context, sharpness)
            for name, spec in schema_specs(schema).items()
        }
        output = schema(**out)

        usage = Usage(input_tokens=len(prompt) // 4 + 40, output_tokens=20 + rng.randint(0, 40))
        meta = ResponseMeta(
            provider=self.provider,
            model=self.model,
            resolved_model=f"{self.model}-20261101",
            request_id=f"fake_{seed}",
            temperature=temperature,
            seed=seed,
            max_tokens=max_tokens,
            usage=usage,
            stop_reason="end_turn",
            cost_usd=self.cost(usage),
            latency_s=time.perf_counter() - t0,
            raw={"seed_sent": True},
        )
        return Generation(output=output, meta=meta, raw_text=output.model_dump_json())

    def _fake_value(
        self,
        spec: FieldSpec,
        schema: type[BaseModel],
        rng: random.Random,
        context: str,
        sharpness: float,
    ) -> Any:
        def lean(option: object) -> float:
            return (1 + _h(f"{context}|{spec.name}|{option}") % 6) ** sharpness

        if spec.kind in (Kind.CATEGORICAL, Kind.ORDINAL):
            levels = spec.levels or ()
            return rng.choices(levels, weights=[lean(o) for o in levels])[0]
        if spec.kind == Kind.BOOLEAN:
            t, f = lean(True), lean(False)
            return rng.random() < t / (t + f)
        if spec.kind == Kind.MULTI_SELECT:
            return [o for o in spec.levels or () if rng.random() < lean(o) / (lean(o) + lean("-"))]
        if spec.kind == Kind.NUMERIC:
            lo = spec.lo if spec.lo is not None else 0
            hi = spec.hi if spec.hi is not None else (10 if spec.integer else 1)
            mode = lo + (hi - lo) * (1 + _h(f"{context}|{spec.name}") % 6) / 7
            x = rng.triangular(lo, hi, mode)
            return round(x) if spec.integer else round(x, 3)
        ann, optional = unwrap_optional(schema.model_fields[spec.name].annotation)
        if ann is str:
            return f"({self.label}) placeholder answer {rng.randint(1, 999)}"
        if optional:
            return None
        msg = f"FakeClient can't make up a value for field {spec.name!r} of type {ann!r}"
        raise TypeError(msg)


def _h(s: str) -> int:
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)
