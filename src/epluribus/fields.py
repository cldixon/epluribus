"""Schema introspection: decide how each answer field can be summarized.

The pydantic schema is the single source of truth. The same class is sent to the
model as the output format, and its field types decide which summaries apply.

Inference rules (override them with an `Annotated` marker):

| Annotation | Kind |
|---|---|
| `Literal["a", "b"]`, `Enum` | categorical (levels in declared order) |
| `Literal[1, 2, 3]`, `IntEnum` | ordinal |
| `int` with both bounds, at most 21 levels (`Field(ge=1, le=5)`) | ordinal |
| `int` without both bounds, `float` | numeric |
| `bool` | boolean |
| `list[Literal[...]]`, `list[Enum]` | multi_select |
| anything else (`str`, nested models, ...) | text (kept, not summarized) |
| `X | None` | the kind of `X`; `None` counts as missing |
"""

from __future__ import annotations

import math
import types
import typing
from dataclasses import dataclass
from enum import Enum, IntEnum, StrEnum
from typing import Any, Literal, get_args, get_origin

from pydantic import BaseModel
from pydantic.fields import FieldInfo

MAX_INT_LEVELS = 21
"""An `int` field with both bounds is ordinal only if it has at most this many levels."""


class Kind(StrEnum):
    """How a field is summarized."""

    CATEGORICAL = "categorical"
    ORDINAL = "ordinal"
    NUMERIC = "numeric"
    BOOLEAN = "boolean"
    MULTI_SELECT = "multi_select"
    TEXT = "text"


# --- Annotated markers that override inference ------------------------------


@dataclass(frozen=True)
class Categorical:
    """Marker: treat the field as unordered categories."""


@dataclass(frozen=True)
class Ordinal:
    """Marker: treat the field as ordered levels.

    Args:
        levels: The order, lowest first. Defaults to the declared order.
    """

    levels: tuple[Any, ...] | None = None


@dataclass(frozen=True)
class Numeric:
    """Marker: treat the field as a number with no fixed levels."""


@dataclass(frozen=True)
class Text:
    """Marker: keep the field as free text and never summarize it."""


_MARKERS = (Categorical, Ordinal, Numeric, Text)


@dataclass(frozen=True)
class FieldSpec:
    """What epluribus knows about one answer field.

    Attributes:
        name: Field name.
        kind: How the field is summarized.
        levels: Categories or ordinal levels, in order. `None` for numeric and text.
        lo: Lower bound, if declared.
        hi: Upper bound, if declared.
        integer: True for `int` fields.
        optional: True if the field accepts `None`.
        description: The field's description, if any.
    """

    name: str
    kind: Kind
    levels: tuple[Any, ...] | None = None
    lo: float | None = None
    hi: float | None = None
    integer: bool = False
    optional: bool = False
    description: str | None = None

    def __repr__(self) -> str:
        extra = f", levels={self.levels}" if self.levels else ""
        return f"FieldSpec({self.name!r}, {self.kind.value}{extra})"

    def rank(self, value: Any) -> float:
        """Position of an ordinal value: the value itself if numeric, else its index."""
        if isinstance(value, bool):
            return float(value)
        if isinstance(value, (int, float)) and not isinstance(value, Enum):
            return float(value)
        if isinstance(value, IntEnum):
            return float(value.value)
        if self.levels is None:
            raise ValueError(f"{self.name!r} has no levels to rank {value!r} by")
        return float(self.levels.index(value))


def schema_specs(schema: type[BaseModel]) -> dict[str, FieldSpec]:
    """Infer a `FieldSpec` for every field of a pydantic model."""
    return {name: infer_spec(name, f) for name, f in schema.model_fields.items()}


def infer_spec(name: str, f: FieldInfo) -> FieldSpec:  # noqa: PLR0911, PLR0912
    """Infer the `FieldSpec` of one pydantic field."""
    ann, optional = unwrap_optional(f.annotation)
    lo, hi = _bounds(f.metadata)
    marker = next((m for m in f.metadata if isinstance(m, _MARKERS)), None)
    is_int = ann is int
    common: dict[str, Any] = {
        "name": name,
        "optional": optional,
        "description": f.description,
        "lo": lo,
        "hi": hi,
        "integer": is_int,
    }

    if isinstance(marker, Text):
        return FieldSpec(kind=Kind.TEXT, **common)

    if get_origin(ann) is Literal:
        vals = get_args(ann)
        all_int = all(isinstance(v, int) and not isinstance(v, bool) for v in vals)
        if isinstance(marker, Ordinal) or (all_int and not isinstance(marker, Categorical)):
            return FieldSpec(kind=Kind.ORDINAL, levels=_levels(marker, vals), **common)
        return FieldSpec(kind=Kind.CATEGORICAL, levels=vals, **common)

    if isinstance(ann, type) and issubclass(ann, Enum):
        vals = tuple(ann)
        ordinal = issubclass(ann, IntEnum) and not isinstance(marker, Categorical)
        if isinstance(marker, Ordinal) or ordinal:
            return FieldSpec(kind=Kind.ORDINAL, levels=_levels(marker, vals), **common)
        return FieldSpec(kind=Kind.CATEGORICAL, levels=vals, **common)

    if get_origin(ann) in (list, set, frozenset, tuple):
        inner = get_args(ann)[0] if get_args(ann) else None
        if get_origin(inner) is Literal:
            return FieldSpec(kind=Kind.MULTI_SELECT, levels=get_args(inner), **common)
        if isinstance(inner, type) and issubclass(inner, Enum):
            return FieldSpec(kind=Kind.MULTI_SELECT, levels=tuple(inner), **common)
        return FieldSpec(kind=Kind.TEXT, **common)

    if ann is bool:
        return FieldSpec(kind=Kind.BOOLEAN, levels=(False, True), **common)

    if is_int:
        if isinstance(marker, Numeric):
            return FieldSpec(kind=Kind.NUMERIC, **common)
        if isinstance(marker, Ordinal) and marker.levels:
            return FieldSpec(kind=Kind.ORDINAL, levels=tuple(marker.levels), **common)
        first, last = _int_range(f.metadata)
        bounded = first is not None and last is not None
        if bounded and (last - first + 1 <= MAX_INT_LEVELS or isinstance(marker, Ordinal)):
            return FieldSpec(kind=Kind.ORDINAL, levels=tuple(range(first, last + 1)), **common)
        return FieldSpec(kind=Kind.NUMERIC, **common)

    if ann is float:
        return FieldSpec(kind=Kind.NUMERIC, **common)

    return FieldSpec(kind=Kind.TEXT, **common)


# --- helpers ----------------------------------------------------------------


def unwrap_optional(ann: Any) -> tuple[Any, bool]:
    """`X | None` -> `(X, True)`; anything else -> `(ann, False)`."""
    origin = get_origin(ann)
    if origin is typing.Union or origin is types.UnionType:
        args = [a for a in get_args(ann) if a is not type(None)]
        if len(args) == 1:
            return args[0], True
    return ann, False


def _bounds(metadata: list[Any]) -> tuple[float | None, float | None]:
    lo = hi = None
    for m in metadata:
        for attr in ("ge", "gt"):
            if getattr(m, attr, None) is not None:
                lo = getattr(m, attr)
        for attr in ("le", "lt"):
            if getattr(m, attr, None) is not None:
                hi = getattr(m, attr)
    return lo, hi


def _int_range(metadata: list[Any]) -> tuple[int | None, int | None]:
    """Inclusive integer range from ge/gt/le/lt constraints."""
    first = last = None
    for m in metadata:
        if getattr(m, "ge", None) is not None:
            first = math.ceil(m.ge)
        if getattr(m, "gt", None) is not None:
            first = math.floor(m.gt) + 1
        if getattr(m, "le", None) is not None:
            last = math.floor(m.le)
        if getattr(m, "lt", None) is not None:
            last = math.ceil(m.lt) - 1
    return first, last


def _levels(marker: Any, default: tuple[Any, ...]) -> tuple[Any, ...]:
    if isinstance(marker, Ordinal) and marker.levels:
        missing = set(default) - set(marker.levels)
        if missing:
            raise ValueError(f"Ordinal(levels=...) is missing declared values: {sorted(missing)}")
        return tuple(marker.levels)
    return tuple(default)
