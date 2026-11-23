"""Run every Python example in the docs and README, with fake models in place of real ones.

Blocks on a page run in order and share one namespace, as a reader would run them.
A block right after `<!-- skip -->` is not run (for example, top-level `await`).
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pytest

import epluribus as ep
import epluribus.clients.anthropic as anthropic_mod
import epluribus.clients.openai as openai_mod

ROOT = Path(__file__).parent.parent
PAGES = [*sorted((ROOT / "docs").rglob("*.md")), ROOT / "README.md"]
BLOCK = re.compile(r"(<!-- skip -->\s*)?```python\n(.*?)```", re.DOTALL)


def fake_client(model: str, **kw: Any) -> ep.FakeClient:
    keep = {k: v for k, v in kw.items() if k in ("label", "pricing", "max_concurrency")}
    return ep.FakeClient(model, error_rate=0.02, **keep)


@pytest.mark.parametrize("page", PAGES, ids=lambda p: str(p.relative_to(ROOT)))
def test_examples_run(page: Path, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(anthropic_mod, "AnthropicClient", fake_client)
    monkeypatch.setattr(openai_mod, "OpenAIClient", fake_client)
    monkeypatch.chdir(tmp_path)
    namespace: dict[str, Any] = {"__name__": "__docs__"}
    blocks = [code for skip, code in BLOCK.findall(page.read_text()) if not skip]
    for i, code in enumerate(blocks):
        try:
            exec(compile(code, f"{page.name}[{i}]", "exec", dont_inherit=True), namespace)  # noqa: S102
        except Exception as e:
            pytest.fail(f"{page.name} block {i} failed: {type(e).__name__}: {e}\n\n{code}")


def test_some_blocks_are_checked() -> None:
    assert sum(len(BLOCK.findall(p.read_text())) for p in PAGES) > 20
