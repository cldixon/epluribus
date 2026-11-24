"""`Task`, `Mix`, and the sampler that fans calls out across models and a sweep grid."""

from __future__ import annotations

import asyncio
import contextlib
import hashlib
import itertools
import json
import random
import sys
import warnings
from collections.abc import Coroutine, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from string import Template
from typing import Any

from pydantic import BaseModel, ValidationError

from .cache import Cache
from .clients.base import Client, Generation
from .errors import CallError
from .records import Record, ResponseMeta
from .results import ResultSet

RESERVED_COORDS = ("model", "variant")
"""Coord names epluribus sets itself."""

MAX_RETRY_WAIT = 60.0
"""Longest wait between tries, in seconds, even if the provider asks for more."""

_CACHE_VERSION = 1


# ===========================================================================
# Task
# ===========================================================================


class Task:
    """A prompt template plus the answer schema.

    Templates use `$name` or `${name}` placeholders, filled from the run's grid.
    Write a literal dollar sign as `$$`. Braces need no escaping, so JSON examples
    can go straight into a prompt.

    Args:
        template: The prompt. Pass a list of phrasings to ask the same question
            several ways; each sample then records its phrasing as
            `coords["variant"]` (0, 1, ...), and `n` applies to each phrasing.
        schema: A pydantic model describing the answer.
        system: Optional system prompt. It can use placeholders too.

    Raises:
        ValueError: If a template has an invalid placeholder.
    """

    def __init__(
        self,
        template: str | Sequence[str],
        schema: type[BaseModel],
        system: str | None = None,
    ) -> None:
        templates = (template,) if isinstance(template, str) else tuple(template)
        if not templates:
            raise ValueError("Task needs at least one template")
        self.templates: tuple[str, ...] = templates
        self.schema = schema
        self.system = system
        self._compiled = [_compile(t) for t in templates]
        self._system = _compile(system) if system is not None else None
        clash = sorted(set(schema.model_fields) & set(RESERVED_COORDS))
        if clash:
            warnings.warn(
                f"{schema.__name__} has answer field(s) {clash} with the same name as a "
                f"coord epluribus sets. rs.group_by({clash[0]!r}) will use the answer; "
                f"use 'coords.{clash[0]}' for the coord.",
                stacklevel=2,
            )

    @property
    def placeholders(self) -> set[str]:
        """Names used by the templates and the system prompt."""
        names: set[str] = set()
        for t in [*self._compiled, self._system]:
            if t is not None:
                names.update(t.get_identifiers())
        return names

    def render(self, values: Mapping[str, Any], variant: int = 0) -> tuple[str, str | None]:
        """Fill the placeholders and return `(prompt, system)`.

        Raises:
            KeyError: If a placeholder has no value.
        """
        prompt = self._compiled[variant].substitute(values)
        system = self._system.substitute(values) if self._system is not None else None
        return prompt, system

    def __repr__(self) -> str:
        first = self.templates[0]
        short = first if len(first) <= 60 else first[:57] + "..."
        more = f" (+{len(self.templates) - 1} variants)" if len(self.templates) > 1 else ""
        return f"Task({short!r}{more}, {self.schema.__name__})"


def _compile(text: str) -> Template:
    t = Template(text)
    if not t.is_valid():
        msg = (
            f"invalid placeholder in template {text!r}. Use $name or ${{name}} and "
            "write a literal dollar sign as $$."
        )
        raise ValueError(msg)
    return t


# ===========================================================================
# Model mixes
# ===========================================================================


class Mix:
    """A population of models: each sample's model is drawn at random by weight.

    A mix counts as one entry in `models`, so `n` is the number of samples drawn
    from the mix per grid cell. Each sample's `coords["model"]` is the label of the
    model that was drawn.

    Args:
        weights: `{client: weight}`. Weights need not sum to 1.
        label: Name for the mix, used in sample seeds. Defaults to `"mix"`.
    """

    def __init__(self, weights: Mapping[Client, float], label: str = "mix") -> None:
        if not weights:
            raise ValueError("Mix needs at least one model")
        if any(w <= 0 for w in weights.values()):
            raise ValueError("Mix weights must be positive")
        self.clients = list(weights)
        self.weights = [float(w) for w in weights.values()]
        self.label = label
        _check_labels(self.clients)

    def draw(self, rng: random.Random) -> Client:
        """Pick one model by weight."""
        return rng.choices(self.clients, weights=self.weights)[0]

    def __repr__(self) -> str:
        total = sum(self.weights)
        pairs = zip(self.clients, self.weights, strict=True)
        parts = ", ".join(f"{c.label}: {w / total:.2f}" for c, w in pairs)
        return f"Mix({{{parts}}})"


Models = Client | Mix | Sequence[Client | Mix]
Grid = Mapping[str, Sequence[Any] | Mapping[Any, Any]]


@dataclass(frozen=True)
class PlannedCall:
    """One call a run will make. `ep.plan` returns a list of these.

    Attributes:
        coords: Where the sample sits: grid labels, `variant` if any, and `model`.
        values: The values that fill the template's placeholders.
        variant: Index of the template used.
        client: The model to call.
        index: Sample index within the cell and model (or mix).
        seed: The sample's seed, which identifies it in the cache.
    """

    coords: dict[str, Any]
    values: dict[str, Any]
    variant: int
    client: Client
    index: int
    seed: int


def _entries(models: Models) -> list[Client | Mix]:
    if isinstance(models, (Client, Mix)):
        return [models]
    if isinstance(models, Mapping):
        msg = "pass a weighted mix as ep.Mix({client: weight, ...}) instead of a plain dict"
        raise TypeError(msg)
    entries = list(models)
    if not entries:
        raise ValueError("models is empty")
    for e in entries:
        if not isinstance(e, (Client, Mix)):
            raise TypeError(f"models must be clients or ep.Mix, got {type(e).__name__}")
    clients = [c for e in entries for c in (e.clients if isinstance(e, Mix) else [e])]
    _check_labels(clients)
    return entries


def _check_labels(clients: Sequence[Client]) -> None:
    labels = [c.label for c in clients]
    dupes = sorted({lbl for lbl in labels if labels.count(lbl) > 1})
    if dupes:
        msg = f"two models share the label(s) {dupes}; give each a distinct label=..."
        raise ValueError(msg)


def _cells(grid: Grid | None) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """Every grid combination as `(coords, template values)`."""
    grid = grid or {}
    axes = []
    for name, vals in grid.items():
        if isinstance(vals, Mapping):
            axes.append([(k, v) for k, v in vals.items()])
        elif isinstance(vals, (str, bytes)):
            msg = f"grid[{name!r}] is a string; wrap it in a list: [{vals!r}]"
            raise TypeError(msg)
        else:
            axes.append([(v, v) for v in vals])
    names = list(grid)
    cells = []
    for combo in itertools.product(*axes):
        coords = {n: c for n, (c, _) in zip(names, combo, strict=True)}
        values = {n: v for n, (_, v) in zip(names, combo, strict=True)}
        cells.append((coords, values))
    return cells


def _check_grid(task: Task, grid: Grid | None) -> None:
    names = set(grid or {})
    missing = sorted(task.placeholders - names)
    if missing:
        raise ValueError(f"the template uses {missing} but the grid has no values for them")
    reserved = sorted(names & set(RESERVED_COORDS))
    if reserved:
        raise ValueError(f"grid names {reserved} are reserved for epluribus; rename them")
    clash = sorted(names & set(task.schema.model_fields))
    if clash:
        msg = f"grid names {clash} are also answer fields of {task.schema.__name__}; rename one"
        raise ValueError(msg)


def plan(
    task: Task,
    models: Models,
    n: int = 20,
    *,
    grid: Grid | None = None,
    seed: int = 0,
) -> list[PlannedCall]:
    """Expand a run into its calls without making them. `len(plan(...))` is the call count.

    Each entry of `models` gets `n` samples per grid cell and per template variant.
    """
    if n < 0:
        raise ValueError("n must be >= 0")
    _check_grid(task, grid)
    entries = _entries(models)
    multi = len(task.templates) > 1
    calls = []
    for coords, values in _cells(grid):
        for variant in range(len(task.templates)):
            cell = {**coords, "variant": variant} if multi else dict(coords)
            for entry in entries:
                for i in range(n):
                    if isinstance(entry, Mix):
                        rng = random.Random(_hash(seed, cell, f"mix:{entry.label}", i))
                        client, who = entry.draw(rng), f"mix:{entry.label}"
                    else:
                        client, who = entry, entry.label
                    calls.append(
                        PlannedCall(
                            coords={**cell, "model": client.label},
                            values=values,
                            variant=variant,
                            client=client,
                            index=i,
                            seed=_hash(seed, cell, who, i),
                        )
                    )
    return calls


def _hash(*parts: Any) -> int:
    s = json.dumps(parts, sort_keys=True, default=str)
    return int(hashlib.sha256(s.encode()).hexdigest()[:8], 16)


def cache_key(
    client: Client,
    prompt: str,
    *,
    system: str | None,
    schema: type[BaseModel],
    temperature: float | None,
    max_tokens: int,
    seed: int,
) -> str:
    """The cache key of one call."""
    payload = json.dumps(
        [
            _CACHE_VERSION,
            client.cache_identity(),
            prompt,
            system,
            schema.model_json_schema(),
            temperature,
            max_tokens,
            seed,
        ],
        sort_keys=True,
        default=str,
    )
    return hashlib.sha256(payload.encode()).hexdigest()


# ===========================================================================
# Run
# ===========================================================================


async def arun(
    task: Task,
    models: Models,
    n: int = 20,
    *,
    grid: Grid | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    seed: int = 0,
    concurrency: int = 8,
    max_retries: int = 2,
    backoff: float = 1.0,
    cache: Cache | None = None,
    progress: bool = False,
) -> ResultSet:
    """Sample `task` across models and grid cells. The async version of `run`.

    In a notebook, `rs = await ep.arun(...)` runs on the notebook's own event loop.
    See `run` for the arguments.
    """
    calls = plan(task, models, n, grid=grid, seed=seed)
    for client in {id(c.client): c.client for c in calls}.values():
        client.check(temperature=temperature)
    sem = asyncio.Semaphore(concurrency)
    tracker = _Progress(len(calls)) if progress else None

    async def one(call: PlannedCall) -> Record:
        record = await _sample(
            task,
            call,
            sem=sem,
            temperature=temperature,
            max_tokens=max_tokens,
            max_retries=max_retries,
            backoff=backoff,
            cache=cache,
        )
        if tracker is not None:
            tracker.update(record)
        return record

    clients = list({id(c.client): c.client for c in calls}.values())
    for client in clients:
        client.begin_run()
    try:
        records = await asyncio.gather(*(one(c) for c in calls))
    finally:
        for client in clients:
            await client.end_run()
    if tracker is not None:
        tracker.done()
    return ResultSet(task.schema, list(records))


def run(
    task: Task,
    models: Models,
    n: int = 20,
    *,
    grid: Grid | None = None,
    temperature: float | None = None,
    max_tokens: int | None = None,
    seed: int = 0,
    concurrency: int = 8,
    max_retries: int = 2,
    backoff: float = 1.0,
    cache: Cache | None = None,
    progress: bool = False,
) -> ResultSet:
    """Sample `task` across models and grid cells and return every answer.

    Works in scripts and in notebooks. In a notebook it runs on a helper thread;
    use `await ep.arun(...)` if you prefer the notebook's own event loop.

    Args:
        task: The question and its answer schema.
        models: A client, an `ep.Mix`, or a list of them.
        n: Samples per model (or per mix) for every grid cell and template variant.
        grid: `{name: values}` filling the template's placeholders. Every
            combination is a cell. Give a dict instead of a list to use short
            labels in results for long values: `{"persona": {"retiree": "a ..."}}`.
        temperature: Sent to the provider if set. Some models don't accept it.
        max_tokens: Most tokens per answer, thinking or reasoning included. `None`
            uses each client's `default_max_tokens`: 16,000 for Anthropic and
            OpenAI, whose models think before answering, and 1,024 otherwise.
        seed: Base seed. Each sample's seed comes from it, the cell, the model and
            the sample index, and identifies the sample in the cache. Change it to
            draw a fresh set of samples.
        concurrency: Most calls in flight at once.
        max_retries: Extra tries for failures worth retrying (rate limits, server
            and connection errors). Refusals and invalid answers are kept, not retried.
        backoff: Base delay in seconds between tries, doubled each time, with
            jitter. A provider's `retry-after` takes precedence.
        cache: A `MemoryCache`, `DiskCache` or your own `Cache`.
        progress: Print a progress line to stderr.

    Returns:
        A `ResultSet` with one record per call, failed calls included.
    """
    coro = arun(
        task,
        models,
        n,
        grid=grid,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=seed,
        concurrency=concurrency,
        max_retries=max_retries,
        backoff=backoff,
        cache=cache,
        progress=progress,
    )
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # Already inside an event loop (Jupyter): run on a fresh loop in a helper thread.
    with ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(_run_coro, coro).result()


def _run_coro(coro: Coroutine[Any, Any, ResultSet]) -> ResultSet:
    return asyncio.run(coro)


async def _sample(
    task: Task,
    call: PlannedCall,
    *,
    sem: asyncio.Semaphore,
    temperature: float | None,
    max_tokens: int | None,
    max_retries: int,
    backoff: float,
    cache: Cache | None,
) -> Record:
    client = call.client
    if max_tokens is None:
        max_tokens = client.default_max_tokens
    prompt, system = task.render(call.values, call.variant)
    key = cache_key(
        client,
        prompt,
        system=system,
        schema=task.schema,
        temperature=temperature,
        max_tokens=max_tokens,
        seed=call.seed,
    )

    def record(gen: Generation) -> Record:
        return Record(gen.output, gen.meta, call.coords, call.index, prompt, system, gen.raw_text)

    if cache is not None and (hit := cache.get(key, task.schema)) is not None:
        return record(replace(hit, meta=replace(hit.meta, cache_hit=True)))

    attempt = 0
    while True:
        attempt += 1
        await client.wait_ready()
        limiter = client.limiter() or contextlib.nullcontext()
        try:
            async with sem, limiter:
                gen = await client.agenerate(
                    prompt,
                    task.schema,
                    system=system,
                    temperature=temperature,
                    seed=call.seed,
                    max_tokens=max_tokens,
                )
        except Exception as e:  # every failure becomes a record
            err = e
        else:
            gen.meta.attempts = attempt
            if cache is not None:
                cache.set(key, gen)
            return record(gen)

        retryable = getattr(err, "retryable", False) or isinstance(err, (TimeoutError,))
        if not retryable or attempt > max_retries:
            meta = ResponseMeta(
                provider=client.provider,
                model=client.model,
                temperature=temperature,
                seed=call.seed,
                max_tokens=max_tokens,
                attempts=attempt,
            )
            return Record(
                None,
                meta,
                call.coords,
                call.index,
                prompt,
                system,
                raw_text=getattr(err, "raw_text", None),
                error=f"{type(err).__name__}: {err}",
                error_kind=_error_kind(err),
            )
        wait = getattr(err, "retry_after", None)
        if wait is not None:
            wait = min(wait, MAX_RETRY_WAIT)
            client.pause(wait)
        else:
            wait = min(MAX_RETRY_WAIT, backoff * 2 ** (attempt - 1)) * (0.5 + random.random())
        await asyncio.sleep(wait)


def _error_kind(err: Exception) -> str:
    if isinstance(err, CallError):
        return err.kind
    if isinstance(err, ValidationError):
        return "invalid_output"
    if isinstance(err, TimeoutError):
        return "timeout"
    return "error"


class _Progress:
    def __init__(self, total: int) -> None:
        self.total, self.done_n, self.cached, self.failed = total, 0, 0, 0

    def update(self, r: Record) -> None:
        self.done_n += 1
        self.cached += r.meta.cache_hit
        self.failed += not r.ok
        sys.stderr.write(
            f"\r{self.done_n}/{self.total} calls ({self.cached} cached, {self.failed} failed)"
        )
        sys.stderr.flush()

    def done(self) -> None:
        sys.stderr.write("\n")
        sys.stderr.flush()
