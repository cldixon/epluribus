"""Shared helpers for SDK-backed clients (Anthropic and OpenAI use the same error shapes)."""

from __future__ import annotations

import email.utils
import time
from collections.abc import Mapping
from typing import Any

from ..errors import ProviderError

_RETRYABLE_STATUS = {408, 409, 429}


def retry_after(headers: Mapping[str, str] | None) -> float | None:
    """Seconds to wait from `retry-after-ms` or `retry-after` (seconds or an HTTP date)."""
    if not headers:
        return None
    ms = headers.get("retry-after-ms")
    if ms:
        try:
            return max(0.0, float(ms) / 1000)
        except ValueError:
            pass
    value = headers.get("retry-after")
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        when = email.utils.parsedate_to_datetime(value)
    except (TypeError, ValueError):
        return None
    return max(0.0, when.timestamp() - time.time())


def translate(e: Exception, sdk: Any) -> Exception:
    """Map an SDK exception to a `ProviderError`, or return it unchanged if it isn't one."""
    if isinstance(e, sdk.APIStatusError):
        status = e.status_code
        headers = getattr(e.response, "headers", None)
        if status == 429:
            kind = "rate_limit"
        elif status >= 500:
            kind = "server"
        elif status in (401, 403):
            kind = "auth"
        elif status == 404:
            kind = "not_found"
        else:
            kind = "bad_request"
        should = headers.get("x-should-retry") if headers else None
        retryable = status >= 500 or status in _RETRYABLE_STATUS
        if should in ("true", "false"):
            retryable = should == "true"
        return ProviderError(
            f"{type(e).__name__} ({status}): {getattr(e, 'message', e)}",
            kind=kind,
            retryable=retryable,
            retry_after=retry_after(headers),
            status_code=status,
        )
    if isinstance(e, (sdk.APITimeoutError, sdk.APIConnectionError)):
        return ProviderError(f"{type(e).__name__}: {e}", kind="connection", retryable=True)
    return e


def dump(obj: Any) -> Any:
    """`model_dump()` for SDK objects, `None` stays `None`."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    return obj
