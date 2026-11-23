"""One model call is one `Record`: the structured answer plus everything known about the call."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from datetime import UTC, datetime
from typing import Any

from pydantic import BaseModel


@dataclass
class Usage:
    """Token counts reported by the provider. `None` means the provider did not report it."""

    input_tokens: int | None = None
    output_tokens: int | None = None
    reasoning_tokens: int | None = None
    """Reasoning or thinking tokens, if reported. Usually billed as output tokens."""
    cached_input_tokens: int | None = None
    """Prompt-cache reads, if reported."""


@dataclass
class ResponseMeta:
    """Normalized call metadata. Provider-specific extras go in `raw`."""

    provider: str
    """Provider name, for example `"anthropic"`, `"openai"` or `"fake"`."""
    model: str
    """The model that was requested."""
    resolved_model: str | None = None
    """The model or snapshot the provider says it used."""
    request_id: str | None = None
    temperature: float | None = None
    """Temperature sent to the provider. `None` means the provider default."""
    seed: int | None = None
    """This sample's seed. It identifies the sample for the cache; most providers ignore it."""
    max_tokens: int | None = None
    usage: Usage = field(default_factory=Usage)
    stop_reason: str | None = None
    latency_s: float | None = None
    cost_usd: float | None = None
    """Cost of the call, if the client was given `pricing`."""
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    cache_hit: bool = False
    """True when this record came from the cache instead of a new call."""
    attempts: int = 1
    """Number of tries the call took, including the successful one."""
    raw: dict[str, Any] = field(default_factory=dict)
    """Provider-specific extras, such as the full usage object and whether the seed was sent."""

    def to_dict(self) -> dict[str, Any]:
        """A JSON-ready dict (used by `DiskCache`)."""
        d = asdict(self)
        d["created_at"] = self.created_at.isoformat()
        return d

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ResponseMeta:
        """Rebuild metadata from `to_dict()` output. Unknown keys are ignored."""
        known = {f.name for f in fields(cls)}
        d = {k: v for k, v in d.items() if k in known}
        d["usage"] = Usage(**d.get("usage") or {})
        if isinstance(d.get("created_at"), str):
            d["created_at"] = datetime.fromisoformat(d["created_at"])
        return cls(**d)


_NAMESPACES = ("output", "coords", "meta")


@dataclass
class Record:
    """A single sample.

    Attributes:
        output: The validated schema instance, or `None` if the call failed.
        meta: Call metadata.
        coords: Where the sample sits in the run: `model`, any grid values, and
            `variant` when the task has several templates.
        sample_index: Index of the sample within its cell and model.
        prompt: The prompt that was sent.
        system: The system prompt that was sent, if any.
        raw_text: The model's raw text, when the provider returned one. Kept for
            failed calls too, so you can see what an invalid answer looked like.
        error: Error message, or `None` for a successful call.
        error_kind: Short error label, for example `"rate_limit"` or `"refusal"`.
    """

    output: BaseModel | None
    meta: ResponseMeta
    coords: dict[str, Any]
    sample_index: int
    prompt: str
    system: str | None = None
    raw_text: str | None = None
    error: str | None = None
    error_kind: str | None = None

    @property
    def ok(self) -> bool:
        """True when the call succeeded and produced a valid answer."""
        return self.error is None and self.output is not None

    def get(self, key: str) -> Any:
        """Look up a value by name.

        Plain names are searched in order: answer fields, then coords, then metadata.
        Prefix a name to pick the place explicitly: `"output.model"`,
        `"coords.model"`, `"meta.usage.output_tokens"`. Dotted paths reach into
        metadata, for example `"usage.output_tokens"` or `"raw.seed_sent"`.

        Raises:
            KeyError: If the name is not found.
        """
        head, _, rest = key.partition(".")
        if head in _NAMESPACES and rest:
            return self._get_in(head, rest)
        if self.output is not None and key in type(self.output).model_fields:
            return getattr(self.output, key)
        if key in self.coords:
            return self.coords[key]
        return self._get_in("meta", key)

    def _get_in(self, space: str, key: str) -> Any:
        if space == "output":
            if self.output is None or key not in type(self.output).model_fields:
                raise KeyError(key)
            return getattr(self.output, key)
        if space == "coords":
            return self.coords[key]
        obj: Any = self.meta
        for part in key.split("."):
            if isinstance(obj, dict):
                if part not in obj:
                    raise KeyError(key)
                obj = obj[part]
            elif hasattr(obj, part):
                obj = getattr(obj, part)
            else:
                raise KeyError(key)
        return obj

    def flat(self) -> dict[str, Any]:
        """One row for a DataFrame: coords, answer fields, then `meta.*` and `usage.*` columns.

        If an answer field has the same name as a coord, the answer column is
        named `output.<field>` so that neither is lost.
        """
        row: dict[str, Any] = {**self.coords, "sample_index": self.sample_index}
        if self.output is not None:
            for k, v in self.output.model_dump().items():
                row[f"output.{k}" if k in row else k] = v
        meta = asdict(self.meta)
        usage = meta.pop("usage")
        meta.pop("raw")
        row.update({f"meta.{k}": v for k, v in meta.items()})
        row.update({f"usage.{k}": v for k, v in usage.items()})
        row["error"] = self.error
        row["error_kind"] = self.error_kind
        return row
