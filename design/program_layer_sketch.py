"""
llm_ppl v2 -- a sampling-only, Pyro-flavoured framework for LLM calls.

Design (no logprobs needed, works with any frontier model):
  * An LLM call is a *simulator with typed outputs*: you can sample it, never score it.
  * Pydantic schemas make outputs computable: Literal -> category, int(ge,le) -> ordinal,
    float -> number, bool -> decision, str -> text (kept, not encoded).
  * Programs are plain Python functions with named sites: sample(), deterministic().
  * Predictive runs a program many times -> arrays per site (schema-driven encoding).
  * Handlers: intervene (force a site's value), condition (rejection on an outcome).
  * Cache + deterministic per-site seeds => reruns are free, and branching from a
    changed site only re-queries what's downstream of it.

Three use cases at the bottom: quick poll, decision-space sweep, branching scenario.
Runs as-is with FakeClient (needs: pydantic, numpy).
"""

from __future__ import annotations

import hashlib
import itertools
import math
import random
from collections import Counter
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable, Literal, get_args, get_origin

import numpy as np
from pydantic import BaseModel, Field


# ===========================================================================
# Schema introspection: what kind of value is each field?
# ===========================================================================

def field_kinds(schema: type[BaseModel]) -> dict[str, tuple[str, Any]]:
    kinds = {}
    for name, f in schema.model_fields.items():
        ann = f.annotation
        lo = hi = None
        for m in f.metadata:                      # Field(ge=..., le=...) constraints
            lo = getattr(m, "ge", lo)
            hi = getattr(m, "le", hi)
        if get_origin(ann) is Literal:
            kinds[name] = ("category", get_args(ann))
        elif ann is bool:
            kinds[name] = ("decision", None)
        elif ann is int:
            kinds[name] = ("ordinal", (lo, hi))
        elif ann is float:
            kinds[name] = ("number", (lo, hi))
        else:
            kinds[name] = ("text", None)
    return kinds


def encode(schema: type[BaseModel], outputs: list[BaseModel]) -> dict[str, np.ndarray]:
    """Structured outputs -> arrays. Categories become integer codes."""
    arrays = {}
    for name, (kind, info) in field_kinds(schema).items():
        vals = [getattr(o, name) for o in outputs]
        if kind == "category":
            arrays[name] = np.array([info.index(v) for v in vals])
        elif kind in ("decision", "ordinal"):
            arrays[name] = np.array(vals, dtype=int)
        elif kind == "number":
            arrays[name] = np.array(vals, dtype=float)
        else:
            arrays[name] = np.array(vals, dtype=object)
    return arrays


# ===========================================================================
# Clients
# ===========================================================================

class FakeClient:
    """Schema-aware fake. Different prompts -> different (but stable) leanings."""

    def __init__(self, name: str):
        self.name = name

    def generate(self, prompt: str, schema: type[BaseModel], temperature: float,
                 seed: int) -> BaseModel:
        rng = random.Random(seed)
        out = {}
        for fname, (kind, info) in field_kinds(schema).items():
            if kind == "category":
                w = [1 + _h(self.name + prompt + o) % 5 for o in info]
                out[fname] = rng.choices(info, weights=w)[0]
            elif kind == "decision":
                out[fname] = rng.random() < 0.3 + 0.4 * (_h(self.name + prompt) % 100) / 100
            elif kind == "ordinal":
                lo, hi = info
                out[fname] = rng.randint(lo or 0, hi or 10)
            elif kind == "number":
                lo, hi = info
                out[fname] = round(rng.uniform(lo or 0, hi or 1), 3)
            else:
                out[fname] = f"[{self.name} text]"
        return schema(**out)

# Real client: wrap the provider SDK's structured-output / tool mode, e.g.
#   resp = anthropic.messages.create(model=..., tools=[schema_tool], tool_choice=...)
#   return schema.model_validate(resp_tool_input)


def _h(s: str) -> int:
    return int(hashlib.md5(s.encode()).hexdigest()[:8], 16)


# ===========================================================================
# LLM "distribution": sample-only
# ===========================================================================

@dataclass
class LLM:
    client: Any
    prompt: str
    schema: type[BaseModel]
    temperature: float = 1.0


CACHE: dict[tuple, BaseModel] = {}
STATS = Counter()


def _call(d: LLM, seed: int) -> BaseModel:
    key = (d.client.name, d.prompt, d.schema.__name__, d.temperature, seed)
    if key in CACHE:
        STATS["cache_hit"] += 1
    else:
        STATS["api_call"] += 1
        CACHE[key] = d.client.generate(d.prompt, d.schema, d.temperature, seed)
    return CACHE[key]


# ===========================================================================
# Samples: a bag of structured outputs + arithmetic on them
# ===========================================================================

@dataclass
class Samples:
    schema: type[BaseModel]
    outputs: list[BaseModel]

    def values(self, f): return [getattr(o, f) for o in self.outputs]
    def tally(self, f): return Counter(self.values(f))
    def mode(self, f): return self.tally(f).most_common(1)[0][0]
    def mean(self, f): return float(np.mean(self.values(f)))
    def median(self, f): return float(np.median(self.values(f)))
    def share(self, f, value): return self.values(f).count(value) / len(self.outputs)
    def arrays(self): return encode(self.schema, self.outputs)

    def freq(self, f) -> dict[Any, tuple[float, float, float]]:
        """Empirical frequency with a 95% Wilson interval: {value: (p, lo, hi)}."""
        n, out = len(self.outputs), {}
        for v, k in self.tally(f).most_common():
            out[v] = _wilson(k, n)
        return out


def _wilson(k, n, z=1.96):
    p = k / n
    d = 1 + z**2 / n
    c = (p + z**2 / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z**2 / (4 * n**2)) / d
    return round(p, 3), round(c - h, 3), round(c + h, 3)


# ===========================================================================
# Runtime: named sites + handler stack
# ===========================================================================

@dataclass
class _Run:
    seed: int
    trace: dict[str, Any] = field(default_factory=dict)
    forced: dict[str, Any] = field(default_factory=dict)


_RUN: _Run | None = None
_FORCED: dict[str, Any] = {}


def sample(name: str, d: LLM, n: int = 1):
    """A named LLM site. n=1 -> one output; n>1 -> Samples (a 'plate')."""
    run = _RUN or _Run(seed=0)
    if name in _FORCED:                                  # intervene
        val = _FORCED[name]
    else:
        seeds = [_h(f"{run.seed}/{name}/{i}") for i in range(n)]   # stable per site
        outs = [_call(d, s) for s in seeds]
        val = outs[0] if n == 1 else Samples(d.schema, outs)
    run.trace[name] = val
    return val


def deterministic(name: str, value):
    if _RUN is not None:
        _RUN.trace[name] = value
    return value


@contextmanager
def intervene(values: dict[str, Any]):
    """Force sites to fixed values (do-operator). Upstream reruns hit the cache."""
    global _FORCED
    old, _FORCED = _FORCED, {**_FORCED, **values}
    try:
        yield
    finally:
        _FORCED = old


def condition(program: Callable, on: Callable[[dict], bool], max_tries: int = 1000):
    """Rejection sampling: only keep runs whose trace satisfies `on`."""
    def wrapped(*args, _seed_iter, **kw):
        for _ in range(max_tries):
            tr = _run_once(program, next(_seed_iter), *args, **kw)
            if on(tr):
                return tr
        raise RuntimeError("condition: no accepted run (event too rare?)")
    wrapped._conditioned = True
    return wrapped


def _run_once(program, seed, *args, **kw) -> dict:
    global _RUN
    prev, _RUN = _RUN, _Run(seed=seed)
    try:
        program(*args, **kw)
        return _RUN.trace
    finally:
        _RUN = prev


# ===========================================================================
# Predictive: run a program many times -> per-site results
# ===========================================================================

@dataclass
class Results:
    traces: list[dict]

    def __getitem__(self, site):            # raw values across runs
        return [t[site] for t in self.traces]

    def freq(self, site, f=None):
        vals = [getattr(v, f) if f else v for v in self[site]]
        n = len(vals)
        return {v: _wilson(k, n) for v, k in Counter(vals).most_common()}

    def arrays(self, site) -> dict[str, np.ndarray]:
        vals = self[site]
        if isinstance(vals[0], BaseModel):
            return encode(type(vals[0]), vals)
        return {"value": np.array(vals)}


class Predictive:
    def __init__(self, program: Callable, n: int, seed: int = 0):
        self.program, self.n, self.seed = program, n, seed

    def __call__(self, *args, **kw) -> Results:
        seeds = iter(range(self.seed * 10_000, self.seed * 10_000 + 10**9))
        if getattr(self.program, "_conditioned", False):
            traces = [self.program(*args, _seed_iter=seeds, **kw) for _ in range(self.n)]
        else:
            traces = [_run_once(self.program, next(seeds), *args, **kw)
                      for _ in range(self.n)]
        return Results(traces)


# ===========================================================================
# Convenience: quick poll and decision-space sweep
# ===========================================================================

def poll(client, prompt: str, schema: type[BaseModel], n: int = 50) -> Samples:
    return sample("poll", LLM(client, prompt, schema), n=n)


def sweep(program: Callable, grid: dict[str, list], n: int, report: Callable[[Results], Any]):
    """Run a program over every combination of arguments; summarize each cell."""
    keys = list(grid)
    table = {}
    for combo in itertools.product(*grid.values()):
        kw = dict(zip(keys, combo))
        table[combo] = report(Predictive(program, n=n)(**kw))
    return table


# ===========================================================================
# Demos
# ===========================================================================

if __name__ == "__main__":
    model = FakeClient("frontier-model")

    # -- 1. Quick poll ------------------------------------------------------
    class Opinion(BaseModel):
        stance: Literal["support", "oppose", "unsure"]
        confidence: int = Field(ge=1, le=5)
        reason: str

    s = poll(model, "Should the city ban cars downtown on weekends?", Opinion, n=60)
    print("== Quick poll ==")
    print("mode:", s.mode("stance"), "| mean confidence:", round(s.mean("confidence"), 2))
    print("freq (p, lo, hi):", s.freq("stance"))
    print("arrays:", {k: v[:8] for k, v in s.arrays().items() if v.dtype != object})

    # -- 2. Decision-space sweep -------------------------------------------
    class Pick(BaseModel):
        option: Literal["A", "B", "C"]

    def decision(framing: str, persona: str):
        prompt = (f"You are {persona}. {framing}\n"
                  "Options: A) expand the bus network, B) cut fares, C) do nothing.")
        votes = sample("votes", LLM(model, prompt, Pick), n=30)
        deterministic("winner", votes.mode("option"))

    table = sweep(decision,
                  grid={"framing": ["Budget is tight.", "Congestion is rising."],
                        "persona": ["a commuter", "a small-business owner"]},
                  n=5,
                  report=lambda r: r.freq("winner"))
    print("\n== Decision sweep: P(winner) per cell over 5 runs x 30 voters ==")
    for cell, dist in table.items():
        print(" ", cell, "->", {k: v[0] for k, v in dist.items()})

    # -- 3. Branching scenario ---------------------------------------------
    class Proposal(BaseModel):
        plan: Literal["move_quota", "split_quota", "keep_quota"]

    class Vote(BaseModel):
        yes: bool

    class Campaign(BaseModel):
        tone: Literal["triumphant", "reassuring", "silent"]

    MEMBERS = ["ideologue", "planner", "security_chief", "agri_minister", "propaganda_chief"]

    def committee(topic: str):
        prop = sample("chair/proposal", LLM(model, f"Chair proposes on: {topic}", Proposal))
        yes = sum(sample(f"vote/{m}", LLM(model, f"You are the {m}. Proposal: {prop.plan}",
                                          Vote)).yes for m in MEMBERS)
        passed = deterministic("passed", yes >= 3)
        if passed:
            sample("campaign", LLM(model, f"Sell '{prop.plan}' to the public.", Campaign))

    topic = "galosh production quota, Minsk vs Kharkov"
    STATS.clear()
    base = Predictive(committee, n=40, seed=1)(topic)
    print("\n== Branching: committee ==")
    print("P(passed):", base.freq("passed"))
    print("proposal mix:", base.freq("chair/proposal", "plan"))
    print("calls:", dict(STATS))

    # Counterfactual: force the chair's proposal; same seeds -> paired comparison
    STATS.clear()
    with intervene({"chair/proposal": Proposal(plan="keep_quota")}):
        cf = Predictive(committee, n=40, seed=1)(topic)
    print("P(passed | do(plan=keep_quota)):", cf.freq("passed"))
    print("calls:", dict(STATS))

    # Rerun the baseline: everything is cached
    STATS.clear()
    Predictive(committee, n=40, seed=1)(topic)
    print("baseline rerun calls:", dict(STATS))

    # Condition: among meetings that passed, which proposals and campaign tones?
    passed_only = Predictive(condition(committee, on=lambda t: t["passed"]), n=20, seed=2)(topic)
    print("given passed -> proposal:", passed_only.freq("chair/proposal", "plan"))
    print("given passed -> tone:", passed_only.freq("campaign", "tone"))
