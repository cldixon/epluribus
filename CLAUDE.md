# epluribus

Sample LLMs many times on a structured question; summarize answers by type. `design/handoff.md`
has the design history, decisions and roadmap; `design/program_layer_sketch.py` is a sketch of
the planned program layer (not package code).

## Commands

```bash
uv sync --all-groups
uv run ruff check . && uv run ruff format --check .
uv run ty check
uv run pytest                 # offline; includes tests/test_docs.py, which runs every docs example
uv run pytest -m live         # real APIs (needs keys, costs money); never in normal CI
uv run zensical build --clean --strict
```

## Releasing

Only when the user asks. Releasing = merging a version bump: open a PR that changes `version`
in `pyproject.toml`, wait for CI, then merge it. On push to main, `.github/workflows/release.yml`
sees that tag `vX.Y.Z` doesn't exist, runs CI, builds, publishes to PyPI through trusted
publishing (environment `pypi`), then creates the tag and GitHub release. Any version bump
merged to main ships, so bump `version` only in a release PR. Published versions can't be
replaced: never reuse a version.

## Rules

- The core is sampling-only. Never make a feature depend on logprobs.
- The pydantic schema is the single source of truth for the model-facing format and for
  how fields are summarized.
- A failed call becomes a `Record` with `error` and `error_kind`; one failure never stops a run.
  Refusals and invalid answers are kept, not retried (retrying them biases results).
- Tests stay offline: `FakeClient`, or provider SDKs over a mocked HTTP transport
  (`tests/test_clients.py`).
- Docs examples are tests. When the public API changes, update `docs/` and keep
  `tests/test_docs.py` green. Mark a block that must not run with `<!-- skip -->`.
- Check provider SDK details against current docs, not memory. Current Claude models reject
  `temperature` and forced tool use; the Anthropic client uses `output_config.format`.
- Docs style: short sentences, one idea each, the same term for the same thing.
