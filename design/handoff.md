# epluribus: handoff for full implementation

> **Status (2026-11-23, first implementation pass).** Packaging, CI and docs follow the
> faa-drs template (uv, ruff, ty, Zensical). Done from sections 6 and 7: items 1-4 (`n` is
> per model, weighted mixes are `ep.Mix`; `$name` templates; tables and notebook display;
> `Estimate` renamed `Interval`), 6 (template variants, labeled grid values), 7 (`run` works in
> Jupyter), 8 (both clients rewritten against current SDKs and tested over mocked HTTP; Anthropic
> uses native structured outputs), 9 (prefixed lookups and collision checks), plus the disk
> cache, retry-after handling, per-client concurrency limits, error kinds, and group
> comparisons with `diff()`. Not yet done: live verification against real APIs, a LiteLLM
> client, Bayesian summaries, and phases 2-4. File paths in this document refer to the original
> sketch; see the repository for the current layout.

*Out of many, one.* epluribus samples LLMs many times on a structured question and
returns every answer with its call metadata, plus summaries matched to each answer's
type. It borrows its ideas from Pyro/NumPyro (named random choices, running a model
many times, handlers for interventions) but treats an LLM call as a **simulator you
can sample but not score**, so it works with frontier APIs that expose no logprobs.

This document is the hand-off from a design conversation to an implementation
session. It covers what exists, the decisions already made, where the boundaries
are, and where the project could go.

---

## 1. What's in this repo

| Path | What it is | State |
|---|---|---|
| `src/epluribus/fields.py` | Schema introspection: pydantic field -> `FieldSpec` (kind + levels) | Working, tested |
| `src/epluribus/records.py` | `Record`, `ResponseMeta`, `Usage` | Working |
| `src/epluribus/results.py` | `ResultSet`, `FieldView`, `Grouped`, `KindError` | Working, tested |
| `src/epluribus/stats.py` | `Estimate`, Wilson interval, bootstrap, entropy | Working |
| `src/epluribus/sampler.py` | `Task`, `run` / `arun`, `MemoryCache`, sample planning | Working, tested |
| `src/epluribus/clients/fake.py` | `FakeClient`: schema-aware, offline, deterministic | Working |
| `src/epluribus/clients/anthropic.py` | `AnthropicClient` (forced tool call) | **Untested sketch** |
| `src/epluribus/clients/openai.py` | `OpenAIClient` (Responses API `parse`) | **Untested sketch** |
| `docs/getting_started.md` | Getting Started page: 5 starter use cases | Reviewed twice |
| `examples/getting_started.py` | Runs every doc snippet with fake clients | Passes |
| `examples/quickstart.py` | Broader demo of the API | Passes |
| `tests/test_core.py` | 5 tests: inference, aggregation, cache, errors, mixes | Pass |
| `reference/program_layer_sketch.py` | Earlier standalone sketch of the *program layer* (named sites, `Predictive`, `intervene`, `condition`, branching). Not part of the package. | Reference only |

Check it works:

```bash
pip install -e ".[dev,pandas]"
pytest -q                          # 5 passed
python examples/getting_started.py # ends with "ALL SNIPPETS RAN"
```

Requires Python >= 3.10, pydantic v2, numpy. Developed on Python 3.13.

---

## 2. The core idea in one example

```python
import epluribus as ep
from typing import Literal
from pydantic import BaseModel, Field

class Vote(BaseModel):
    choice: Literal["approve", "reject", "abstain"]      # categorical
    confidence: int = Field(ge=1, le=5)                  # ordinal
    reason: str                                          # text, kept not aggregated

task = ep.Task("You are {role}. The council votes on converting a garage "
               "into housing. How do you vote?", Vote)

rs = ep.run(task, [sonnet, haiku], n=30, grid={"role": ["a retiree", "a cyclist"]})

rs["choice"].freq()                              # shares with 95% intervals
rs.group_by("model", "role")["confidence"].mean()
rs.usage()                                       # tokens, cost, cache hits, errors
rs.to_pandas()                                   # one row per sample, with metadata
```

See `docs/getting_started.md` for the full set: quick poll, comparing models,
sweeps over roles and situations, numeric estimates, and working with results.

---

## 3. How the idea developed

1. **Starting point.** The user had used Pyro and wanted "Pyro/NumPyro, but for LLM
   or agent calls": first-class sampling and distributions over model outputs.
   Motivating uses: simulating social-style experiments (elections, negotiations,
   committees), exploring decision spaces, quick polls.
2. **Pyro vocabulary mapped onto LLMs.** A sample site becomes an LLM call; `plate`
   becomes a population or repeated runs; `trace`/`replay` become logging and cached
   reruns; `condition` and `do` (intervene) become "given that X happened" and
   "force X to happen" for counterfactuals.
3. **Logprobs debate, settled.** An early draft leaned on token logprobs for exact
   distributions. The user pushed back: frontier APIs (Anthropic, reasoning models
   generally) don't expose them, and a design that depends on them can't use the
   models that matter. **Decision: the core is sampling-only.** Structured outputs
   make answers computable, and distributions are empirical.
4. **Structured outputs are the foundation.** The pydantic schema is the single
   source of truth: it's sent as the tool / structured-output definition, and its
   field types decide which aggregations apply.
5. **Exploration, not prediction.** Fitting simulated populations to real survey
   data was discussed and judged interesting, but it isn't the goal. The tool is for
   searching a space of possibilities, getting a range of responses, and finding
   tendencies or biases of particular models. Classic agent-based modeling
   (Schelling's segregation model) was the comparison: value comes from insight into
   mechanisms, not from forecasting.
6. **Stay with the core.** Full multi-agent infrastructure (agent memory, message
   buses, environments) was discussed and deliberately set aside. Simple multi-step
   scenarios can be built by chaining calls.
7. **Package built.** Output object, metadata, schema-driven aggregation, sweeps,
   model mixes, caching, Getting Started docs.
8. **Future paths explored:** local models with logprobs (exact distributions, RL),
   RL training with this library as the environment, DSPy interop.
9. **Naming.** Settled on **epluribus** (see section 9).

---

## 4. Decisions and boundaries

**In scope / decided**
- **Sampling-only core.** Never require logprobs. Logprobs are an optional client
  capability for later (local models).
- **Schema as single source of truth** for both the model-facing format and the
  aggregation rules.
- **Typed aggregation.** Each field kind gets only the summaries that make sense;
  the wrong one raises `KindError`.
- **Keep every sample and its metadata.** Normalized metadata in `ResponseMeta`,
  provider extras in `meta.raw`, sweep position in `coords`.
- **Failures are data.** A failed call becomes a `Record` with `.error`; a run never
  crashes because one call failed. Failed calls aren't cached, so reruns retry them.
- **Report uncertainty by default.** Shares come with Wilson intervals, means and
  medians with bootstrap intervals, always with `n`.
- **Model-agnostic.** Works with any provider through the `Client` interface; mixing
  providers in one run is normal.
- **No gradients, ever, in this library.** For RL it is the environment and rollout
  layer; the training framework owns weights.

**Out of scope for now**
- Calibrating simulations to real survey data.
- Agent infrastructure: persistent memory, message passing, environments.
- Gradient-based inference through an LLM (impossible without likelihoods).
- Treating a model's stated confidence as a calibrated probability. It is just
  another answer; the spread across samples is the real distribution.

**Honest limits to keep in the docs**
- Sampling variance isn't population variance. Models collapse toward the same
  answers, defer, and carry biases; simulated "voters" are not real voters.
- Samples cost money, so `n` is small and intervals matter.
- Most providers ignore seeds. In epluribus a seed identifies a sample for the
  cache; it doesn't make the model's answer reproducible.

---

## 5. Architecture and semantics

```
Task(template, schema, system)          Client(s): agenerate(prompt, schema, ...) -> Generation
        \                                  /
         run / arun(task, models, n, grid, temperature, seed, cache, concurrency, max_retries)
                         |
                  ResultSet(schema, [Record])
                         |
     rs["field"] -> FieldView (kind-aware: freq, mean, ...)
     rs.group_by(...)["field"] -> per-group results
     rs.filter / usage / summary / to_pandas / arrays
```

**Field kind inference** (`fields.py`):

| Annotation | Kind |
|---|---|
| `Literal["a","b"]`, `Enum` | categorical |
| `Literal[1,2,3]`, `IntEnum`, `int` with both `ge` and `le`, `Annotated[..., Ordinal()]` | ordinal |
| `bool` | boolean |
| `float`, `int` without both bounds | numeric |
| `list[Literal[...]]` | multi_select |
| anything else (`str`, ...) | text |
| `Optional[X]` | kind of `X`; `None` counts as missing |

Markers `Categorical()`, `Ordinal(levels=...)`, `Numeric()`, `Text()` override inference.

**Sampling plan** (`sampler._plan`):
- One client, or a list -> `n` samples **per model** per grid cell.
- `{client: weight}` -> `n` samples **in total** per cell, models drawn by weight.

**Seeds and cache**:
- Each sample's seed = hash(run `seed`, grid cell, client label, sample index).
- Cache key = provider, model, prompt, system, schema JSON, temperature, max_tokens, sample seed.
- So raising `n` or adding grid values only makes new calls; changing template,
  schema or temperature makes all calls new; a new `seed` draws a fresh set.

**Metadata** (`ResponseMeta`): provider, requested model, resolved model, request id,
temperature, seed, max_tokens, usage (input / output / reasoning / cached tokens),
stop reason, latency, cost (if `pricing` given), timestamp, cache_hit, attempts, raw.

---

## 6. Known issues and API review items

Raised while writing the Getting Started docs. Fix 1-4 early, since they shape every
user-facing call.

1. **`n` means two things**: per model for a list, total for a weighted mix. Options:
   `n_per_model` / `n_total`, or one consistent meaning.
2. **Templates use `str.format`**, so literal braces must be doubled. Consider
   `string.Template` (`$role`) or Jinja2.
3. **Results print as raw dicts of `Estimate`s.** Add a `.table()` method and
   notebook HTML display for `FieldView`, `Grouped` and `ResultSet`.
4. **`ep.Estimate` clashes** with natural schema names. Consider `Interval` or `Stat`.
5. **Seeds can mislead.** Docs already explain; consider renaming the parameter
   (for example `sample_key`) or recording `seed_sent: bool` (clients already put
   it in `meta.raw`).
6. **Getting variety.** Only `temperature` exists, and some reasoning models restrict
   it. Add first-class variety controls (template paraphrases, persona lists).
7. **`run` fails inside Jupyter** (it calls `asyncio.run`). Detect a running loop and
   give a helpful error, or support it.
8. **Real clients are untested.** Verify against the current Anthropic and OpenAI
   SDKs. Forced `tool_choice` can't be combined with Anthropic extended thinking;
   consider the provider's native structured-output mode. OpenAI reasoning models
   reject `temperature`.
9. **`Record.get` precedence**: output fields shadow coords and meta, so a schema
   field named `model` would hide the model label. Decide whether to namespace.
10. **`FakeClient`** ignores `system` and temperature, which is fine for tests but
    worth documenting.

---

## 7. Suggested implementation plan

**Phase 0: make the core real**
- Verify and finish `AnthropicClient` and `OpenAIClient` against current SDK docs;
  consider a LiteLLM client for wide provider coverage.
- Fix API items 1-4 and 7 above.
- Disk-backed cache (SQLite or similar) with the same `get`/`set` interface.
- Per-provider rate limiting and backoff that respects `retry-after`.
- Packaging: CI, type checking, linting, versioning. Keep tests offline using `FakeClient`.

**Phase 1: analysis quality of life**
- Pretty display (item 3). Comparisons between groups with intervals
  (for example the difference in share between two framings).
- Variety controls (item 6): template paraphrase lists, persona lists as a grid axis.
- Optional Bayesian summaries: Dirichlet posterior for category shares (this can use
  NumPyro on the counts).

**Phase 2: the program layer** (see `reference/program_layer_sketch.py`)
- Named sites inside plain Python functions: `sample(name, ...)`, `deterministic(...)`.
- `Predictive(program, n)` returns per-site results across runs.
- Handlers: `intervene` (force a site's value for counterfactuals), `condition`
  (rejection sampling on an outcome), `trace` / `replay`.
- Cache plus per-site seeds make branching cheap: rerunning with one site changed only
  re-queries what comes after it.
- This enables branching scenarios such as the committee example in the sketch.

**Phase 3: local models with logprobs (optional capability)**
- `Client.capabilities` set, for example `{"logprobs", "token_ids", "score", "seed"}`.
- `TokenTrace` on `Generation`/`Record`: prompt ids, generated ids, per-token logprobs,
  optional top-k, and token spans for each schema field.
- `VLLMClient` (and/or transformers, SGLang, llama.cpp).
- `ascore(task, field=...)`: exact distribution over a fixed-choice field by
  teacher-forcing each option, one pass instead of n samples. Expose as
  `rs["field"].exact()` alongside `freq()`.
- Pitfalls: constrained decoding can renormalize logprobs (record which kind); field
  order matters (later fields condition on earlier ones); mapping tokens to JSON
  fields needs character offsets and careful tests.

**Phase 4: RL bridge**
- `ResultSet.to_rl_batch(reward=..., group_by=..., policy=...)` producing prompt ids,
  response ids, old logprobs, rewards and group ids for GRPO-style training.
- Integrate through a training framework's custom rollout or environment hook
  (verl, OpenRLHF, TRL). The framework owns the inference engine and weight sync.
- Pitfalls: inference-engine and trainer logprobs differ slightly (trainers recompute
  or correct); reward hacking against peer models (rotate peers, hold some out,
  check suspicious wins with a judge). Training for persuasion is a sensitive
  capability and normally sits under safety review at a lab.

---

## 8. Possible directions

- **Quick polls and model audits**: how models lean on a question, where they disagree,
  how stable their answers are. The near-term sweet spot.
- **Decision-space exploration**: sweep framings, roles and constraints to find where
  the recommended option flips.
- **Branching scenarios**: multi-step simulations (committees, negotiations,
  a satirical Soviet Politburo meeting on galosh quotas) with counterfactual
  interventions and paired comparisons.
- **Robustness checks**: rerun the same question across paraphrases, models and
  personas; trust only findings that survive.
- **DSPy interop**: DSPy optimizes a program toward its best output; epluribus measures
  the distribution of outputs. Possible links: use DSPy modules as the call layer
  (watch DSPy's own caching, which can make repeated samples identical); use
  epluribus win rates as the metric for DSPy's GEPA optimizer (a prompt-level
  version of the RL idea); use epluribus sweeps to audit optimized DSPy programs.
- **Exact distributions** from local models, compared against sampled ones.
- **RL environments** for multi-agent settings, as above.
- **Calibration** to real survey data, if it ever becomes a goal.

Related work to know about (from memory; check before relying on it): Expected
Parrot's EDSL (LLM surveys across models), DeepMind's Concordia (social simulation),
LLaMPPL / GenLM (probabilistic programming with sequential Monte Carlo over LLMs),
DSPy, Generative Agents (Park et al.), and the "silicon samples" research
(Argyle et al.) on LLMs standing in for survey respondents.

---

## 9. Naming

**epluribus**, from *e pluribus unum*, "out of many, one": many samples in, one
aggregate or distribution out. Unclaimed on PyPI as of 2026-11-22 (`pluribus` itself
is taken). Import as `import epluribus as ep`. Before publishing, check GitHub and
trademarks, and register the name with a real first release.

Rejected: `manyfold` (taken on PyPI), `poll` (taken, and means polling loops),
`surve`/Survé (reads as "serve", which means model serving), `canvass`/`panelists`
(too generic), `polypoll` (fun, but promises only polls; could name the quick-poll
helper instead).

---

## 10. Notes for the implementation session

- Keep tests offline: use `FakeClient`, never call real APIs in the test suite.
- `examples/getting_started.py` must keep running; update it alongside
  `docs/getting_started.md` whenever the API changes.
- Check provider SDK details (structured outputs, usage fields, request ids) against
  current documentation rather than the sketches here.
- Suggested first tasks: (1) settle API items 1-4; (2) make one real client work end to
  end with a small poll; (3) disk cache; (4) pretty display.
