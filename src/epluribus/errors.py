"""Exceptions raised by epluribus and by its clients.

Clients turn provider failures into these classes so that the sampler can decide
whether to retry. Every failed call still becomes a `Record` with `error` and
`error_kind` set; a run never stops because one call failed.
"""

from __future__ import annotations


class EpluribusError(Exception):
    """Base class for all epluribus errors."""


class KindError(EpluribusError, TypeError):
    """An aggregation does not fit the field's kind, for example `mean()` on a categorical."""


class CallError(EpluribusError):
    """A model call failed.

    Attributes:
        kind: A short label stored on the record as `error_kind`, for example
            `"rate_limit"`, `"server"`, `"invalid_output"` or `"refusal"`.
        retryable: Whether the sampler should try the call again.
        retry_after: Seconds the provider asked us to wait, if it said.
        raw_text: The model's text, when there was one. Kept for inspection.
    """

    kind: str = "error"
    retryable: bool = False

    def __init__(
        self,
        message: str,
        *,
        kind: str | None = None,
        retryable: bool | None = None,
        retry_after: float | None = None,
        raw_text: str | None = None,
        status_code: int | None = None,
    ) -> None:
        super().__init__(message)
        if kind is not None:
            self.kind = kind
        if retryable is not None:
            self.retryable = retryable
        self.retry_after = retry_after
        self.raw_text = raw_text
        self.status_code = status_code


class ProviderError(CallError):
    """The provider returned an error or could not be reached."""

    kind = "provider"


class InvalidOutputError(CallError):
    """The model answered, but the answer does not match the schema.

    Not retried by default: the invalid answer is a sample like any other, and
    retrying only the failures would bias the results. `raw_text` holds the answer.
    """

    kind = "invalid_output"


class RefusalError(CallError):
    """The model declined to answer. Not retried, for the same reason as invalid output."""

    kind = "refusal"


class TruncatedError(CallError):
    """The answer hit `max_tokens` before it was complete. Raise `max_tokens` and rerun."""

    kind = "truncated"
