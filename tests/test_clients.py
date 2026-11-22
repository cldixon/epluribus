"""Provider clients against a mocked HTTP layer: real SDK request building and parsing."""

from __future__ import annotations

import asyncio
import json
from typing import Any

import anthropic
import openai
import pytest

import epluribus as ep
from epluribus.clients._sdk import retry_after
from epluribus.clients.anthropic import AnthropicClient
from epluribus.clients.openai import OpenAIClient
from epluribus.errors import InvalidOutputError, ProviderError, RefusalError, TruncatedError

from .conftest import Recorder, Vote

ANSWER: dict[str, Any] = {"choice": "approve", "confidence": 4, "reason": "housing matters"}


def claude_message(text: str | None = None, stop_reason: str = "end_turn", **extra: Any) -> dict:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude-haiku-5-5",
        "content": [{"type": "text", "text": text if text is not None else json.dumps(ANSWER)}],
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {
            "input_tokens": 50,
            "output_tokens": 20,
            "cache_read_input_tokens": 0,
            "output_tokens_details": {"thinking_tokens": 7},
        },
        **extra,
    }


def openai_response(
    text: str | None = None, *, refusal: str | None = None, incomplete: str | None = None
) -> dict:
    part = (
        {"type": "refusal", "refusal": refusal}
        if refusal
        else {"type": "output_text", "text": text or json.dumps(ANSWER), "annotations": []}
    )
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "status": "incomplete" if incomplete else "completed",
        "incomplete_details": {"reason": incomplete} if incomplete else None,
        "model": "gpt-test-2026",
        "output": [
            {
                "type": "message",
                "id": "m1",
                "status": "completed",
                "role": "assistant",
                "content": [part],
            }
        ],
        "parallel_tool_calls": True,
        "tool_choice": "auto",
        "tools": [],
        "usage": {
            "input_tokens": 40,
            "input_tokens_details": {"cached_tokens": 8},
            "output_tokens": 30,
            "output_tokens_details": {"reasoning_tokens": 12},
            "total_tokens": 70,
        },
    }


def claude(recorder: Recorder, **kw: Any) -> AnthropicClient:
    sdk = anthropic.AsyncAnthropic(
        api_key="test", max_retries=0, http_client=recorder.http_client(anthropic)
    )
    return AnthropicClient("claude-haiku-5-5", sdk_client=sdk, **kw)


def gpt(recorder: Recorder, **kw: Any) -> OpenAIClient:
    sdk = openai.AsyncOpenAI(
        api_key="test", max_retries=0, http_client=recorder.http_client(openai)
    )
    return OpenAIClient("gpt-test", sdk_client=sdk, **kw)


def generate(client: ep.Client, **kw: Any) -> ep.Generation:
    return asyncio.run(client.agenerate("How do you vote?", Vote, **kw))


# --- Anthropic ------------------------------------------------------------------


def test_anthropic_success_and_request_shape(recorder):
    recorder.queue(claude_message(), **{"request-id": "req_123"})
    client = claude(recorder, effort="low", pricing=(1.0, 5.0))
    gen = generate(client, system="Be brief.", seed=9, max_tokens=300)
    assert gen.output == Vote(**ANSWER)
    m = gen.meta
    assert (m.provider, m.resolved_model, m.request_id) == (
        "anthropic",
        "claude-haiku-5-5",
        "req_123",
    )
    assert (m.usage.input_tokens, m.usage.output_tokens, m.usage.reasoning_tokens) == (50, 20, 7)
    assert m.cost_usd == pytest.approx((50 * 1 + 20 * 5) / 1e6)
    assert m.raw["seed_sent"] is False
    assert m.seed == 9

    body = recorder.requests[0]
    assert body["model"] == "claude-haiku-5-5"
    assert body["max_tokens"] == 300
    assert body["system"] == "Be brief."
    assert body["output_config"]["effort"] == "low"
    fmt = body["output_config"]["format"]
    assert fmt["type"] == "json_schema"
    assert fmt["schema"]["properties"]["choice"]["enum"] == ["approve", "reject", "abstain"]
    assert "temperature" not in body
    assert "tools" not in body


def test_anthropic_extra_params_merge(recorder):
    recorder.queue(claude_message())
    client = claude(recorder, extra_params={"output_config": {"effort": "high"}, "metadata": {}})
    generate(client)
    body = recorder.requests[0]
    assert body["output_config"]["effort"] == "high"
    assert "format" in body["output_config"]
    assert client.cache_identity()["extra_params"]["metadata"] == {}


def test_anthropic_rejects_temperature(recorder):
    client = claude(recorder)
    with pytest.raises(ValueError, match="temperature"):
        client.check(temperature=0.7)
    with pytest.raises(ValueError, match="temperature"):
        ep.run(ep.Task("Q", Vote), client, n=1, temperature=0.7)


def test_anthropic_outcomes(recorder):
    client = claude(recorder)
    recorder.queue(claude_message('{"choice": "maybe"}'))
    with pytest.raises(InvalidOutputError) as e:
        generate(client)
    assert e.value.raw_text == '{"choice": "maybe"}'

    recorder.queue(claude_message("", stop_reason="refusal", stop_details={"category": "cyber"}))
    with pytest.raises(RefusalError, match="cyber"):
        generate(client)

    recorder.queue(claude_message('{"choice": "appr', stop_reason="max_tokens"))
    with pytest.raises(TruncatedError) as t:
        generate(client)
    assert t.value.raw_text == '{"choice": "appr'

    # Out-of-range values are checked locally even though the API schema drops ge/le.
    recorder.queue(claude_message(json.dumps({**ANSWER, "confidence": 9})))
    with pytest.raises(InvalidOutputError):
        generate(client)


def test_anthropic_errors_map_to_retry_decisions(recorder):
    client = claude(recorder)
    err = {"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}}
    recorder.queue(err, status=429, **{"retry-after": "3"})
    with pytest.raises(ProviderError) as e:
        generate(client)
    assert (e.value.kind, e.value.retryable, e.value.retry_after) == ("rate_limit", True, 3.0)

    recorder.queue({"type": "error", "error": {"type": "overloaded_error", "message": "x"}}, 529)
    with pytest.raises(ProviderError) as e:
        generate(client)
    assert (e.value.kind, e.value.retryable) == ("server", True)

    recorder.queue(
        {"type": "error", "error": {"type": "invalid_request_error", "message": "x"}}, 400
    )
    with pytest.raises(ProviderError) as e:
        generate(client)
    assert (e.value.kind, e.value.retryable, e.value.status_code) == ("bad_request", False, 400)

    recorder.queue(
        {"type": "error", "error": {"type": "authentication_error", "message": "x"}}, 401
    )
    with pytest.raises(ProviderError) as e:
        generate(client)
    assert e.value.kind == "auth"


def test_anthropic_in_a_run_retries_then_succeeds(recorder):
    client = claude(recorder)
    recorder.queue({"type": "error", "error": {"type": "api_error", "message": "x"}}, 500)
    recorder.queue(claude_message())
    rs = ep.run(ep.Task("Q", Vote), client, n=1, backoff=0)
    assert rs.records[0].ok
    assert rs.records[0].meta.attempts == 2


# --- OpenAI ---------------------------------------------------------------------


def test_openai_success_and_request_shape(recorder):
    recorder.queue(openai_response(), **{"x-request-id": "req_9"})
    client = gpt(recorder, effort="low")
    gen = generate(client, system="Be brief.", temperature=0.5, max_tokens=200)
    assert gen.output == Vote(**ANSWER)
    m = gen.meta
    assert (m.resolved_model, m.request_id, m.stop_reason) == (
        "gpt-test-2026",
        "req_9",
        "completed",
    )
    assert (m.usage.reasoning_tokens, m.usage.cached_input_tokens) == (12, 8)
    body = recorder.requests[0]
    assert body["instructions"] == "Be brief."
    assert body["temperature"] == 0.5
    assert body["reasoning"] == {"effort": "low"}
    assert body["store"] is False
    assert body["max_output_tokens"] == 200
    fmt = body["text"]["format"]
    assert (fmt["type"], fmt["name"], fmt["strict"]) == ("json_schema", "Vote", True)
    assert fmt["schema"]["additionalProperties"] is False


def test_openai_outcomes(recorder):
    client = gpt(recorder)
    recorder.queue(openai_response(refusal="I can't help with that."))
    with pytest.raises(RefusalError):
        generate(client)
    recorder.queue(openai_response('{"choice": "ap', incomplete="max_output_tokens"))
    with pytest.raises(TruncatedError):
        generate(client)
    recorder.queue(openai_response("not json"))
    with pytest.raises(InvalidOutputError):
        generate(client)
    recorder.queue(
        {"error": {"message": "slow", "type": "rate_limit"}}, 429, **{"retry-after-ms": "250"}
    )
    with pytest.raises(ProviderError) as e:
        generate(client)
    assert e.value.retry_after == 0.25


# --- shared ---------------------------------------------------------------------


def test_retry_after_parsing():
    assert retry_after(None) is None
    assert retry_after({"retry-after": "2"}) == 2.0
    assert retry_after({"retry-after-ms": "1500"}) == 1.5
    assert retry_after({"retry-after-ms": "x", "retry-after": "1"}) == 1.0
    assert retry_after({"retry-after": "garbage"}) is None
    assert retry_after({"retry-after": "Wed, 21 Oct 2015 07:28:00 GMT"}) == 0.0


def test_default_sdk_clients_are_per_loop(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")
    monkeypatch.setenv("OPENAI_API_KEY", "test")
    for client in (AnthropicClient("claude-haiku-5-5"), OpenAIClient("gpt-test")):

        async def get(c=client):
            return c._sdk(), c._sdk()

        a1, a2 = asyncio.run(get())
        b1, _ = asyncio.run(get())
        assert a1 is a2
        assert a1 is not b1
        assert a1.max_retries == 0
