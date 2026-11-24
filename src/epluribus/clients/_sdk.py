"""Shared helpers for SDK-backed clients (Anthropic and OpenAI use the same error shapes)."""

from __future__ import annotations

import email.utils
import os
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
        cause = e.__cause__ or e.__context__
        # Name the cause's type but never its text: it can quote headers, API key included.
        why = f" ({type(cause).__name__})" if cause is not None else ""
        if cause is not None and type(cause).__name__ == "LocalProtocolError":
            msg = (
                f"{type(e).__name__}: the request could not be sent{why}. This usually means "
                "an invalid header value, such as an API key with a trailing newline."
            )
            return ProviderError(msg, kind="bad_request", retryable=False)
        return ProviderError(f"{type(e).__name__}: {e}{why}", kind="connection", retryable=True)
    return e


def check_key(name: str) -> None:
    """Raise `ValueError` if the API key in environment variable `name` has stray whitespace."""
    value = os.environ.get(name)
    if value is not None and value != value.strip():
        msg = (
            f"{name} has whitespace before or after the key, often a newline left by "
            "pasting. Save it again without it."
        )
        raise ValueError(msg)


def dump(obj: Any) -> Any:
    """`model_dump()` for SDK objects, `None` stays `None`."""
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump(mode="json", exclude_none=True)
    return obj
