"""`ResultSet` holds every record from a run; `FieldView` summarizes one answer field."""

from __future__ import annotations

import random
from collections import Counter
from collections.abc import Callable, Iterator, Sequence
from typing import Any

import numpy as np
from pydantic import BaseModel

from .display import ByGroup, Info, Shares, Summary
from .errors import KindError
from .fields import FieldSpec, Kind, schema_specs
from .records import Record
from .stats import Interval, bootstrap, bootstrap_diff, entropy, newcombe_diff, wilson

_LEVELED = {Kind.CATEGORICAL, Kind.ORDINAL, Kind.BOOLEAN}
_ORDERED = {Kind.ORDINAL, Kind.NUMERIC, Kind.BOOLEAN}


# ===========================================================================
# FieldView
# ===========================================================================


class FieldView:
    """Summaries of one answer field. Which methods work depends on the field's kind.

    A summary that doesn't fit the kind raises `KindError`. Failed calls and
    `None` answers count as missing.
    """

    def __init__(self, spec: FieldSpec, records: list[Record]) -> None:
        self.spec = spec
        self._records = records

    # --- raw access ------------------------------------------------------------

    def values(self, include_missing: bool = False) -> list[Any]:
        """The answers, one per record. Missing ones are left out unless asked for."""
        vals = [getattr(r.output, self.spec.name) if r.ok else None for r in self._records]
        return vals if include_missing else [v for v in vals if v is not None]

    @property
    def kind(self) -> Kind:
        """The field's kind."""
        return self.spec.kind

    @property
    def n(self) -> int:
        """Number of answers present."""
        return len(self.values())

    @property
    def missing(self) -> int:
        """Number of records without an answer: failed calls and `None` answers."""
        return len(self._records) - self.n

    def _need(self, kinds: set[Kind], op: str) -> None:
        if self.spec.kind not in kinds:
            allowed = ", ".join(sorted(k.value for k in kinds))
            msg = f"{op}() needs a field of kind {allowed}; {self.spec.name!r} is {self.spec.kind}"
            raise KindError(msg)

    # --- discrete ----------------------------------------------------------------

    def counts(self) -> dict[Any, int]:
        """Count of each level, including levels nobody picked.

        For multi-select fields: how many answers selected each option.
        """
        if self.spec.kind == Kind.MULTI_SELECT:
            c = Counter(opt for v in self.values() for opt in set(v))
        else:
            self._need(_LEVELED, "counts")
            c = Counter(self.values())
        levels = self.spec.levels or tuple(c)
        return {lvl: c.get(lvl, 0) for lvl in levels}

    def freq(self) -> Shares:
        """Share choosing each level, with 95% Wilson intervals.

        For multi-select fields: share of answers that selected each option, so
        shares can add up to more than 1.
        """
        n = self.n
        return Shares({lvl: wilson(k, n) for lvl, k in self.counts().items()})

    def mode(self) -> Any:
        """The most common answer (the first declared level on a tie)."""
        self._need(_LEVELED | {Kind.NUMERIC}, "mode")
        c = Counter(self.values())
        if not c:
            return None
        top = max(c.values())
        order = self.spec.levels or tuple(c)
        return next(v for v in order if c.get(v) == top)

    def entropy(self) -> float:
        """Disagreement in bits. 0 means every answer agreed."""
        self._need(_LEVELED, "entropy")
        return entropy([e.value for e in self.freq().values()])

    def rate(self) -> Interval:
        """Share of `True` answers (boolean fields)."""
        self._need({Kind.BOOLEAN}, "rate")
        vals = self.values()
        return wilson(sum(vals), len(vals))

    # --- ordered and numeric -----------------------------------------------------

    def numbers(self) -> np.ndarray:
        """Answers as floats. Ordinal levels that aren't numbers become ranks 0..k-1."""
        self._need(_ORDERED, "numbers")
        return np.asarray([self.spec.rank(v) for v in self.values()], dtype=float)

    def mean(self) -> Interval:
        """Mean with a 95% bootstrap interval. Text ordinal levels use ranks 0..k-1."""
        self._need(_ORDERED, "mean")
        return bootstrap(self.numbers(), np.mean)

    def median(self) -> Interval:
        """Median with a 95% bootstrap interval."""
        self._need(_ORDERED, "median")
        return bootstrap(self.numbers(), np.median)

    def std(self) -> float:
        """Sample standard deviation."""
        self._need(_ORDERED, "std")
        x = self.numbers()
        return float(x.std(ddof=1)) if len(x) > 1 else float("nan")

    def quantiles(self, qs: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9)) -> dict[float, float]:
        """Quantiles of the answers, as `{q: value}`."""
        self._need(_ORDERED, "quantiles")
        x = self.numbers()
        return {q: float(np.quantile(x, q)) for q in qs} if len(x) else {}

    # --- multi-select --------------------------------------------------------------

    def mean_selected(self) -> Interval:
        """Average number of options selected per answer."""
        self._need({Kind.MULTI_SELECT}, "mean_selected")
        return bootstrap([len(set(v)) for v in self.values()], np.mean)

    # --- text --------------------------------------------------------------------

    def sample(self, k: int = 5, seed: int = 0) -> list[Any]:
        """Up to `k` answers chosen at random. Works for every kind."""
        vals = self.values()
        return random.Random(seed).sample(vals, min(k, len(vals)))

    # --- comparison ----------------------------------------------------------------

    def diff(self, other: FieldView) -> Shares | Interval:
        """This field minus the same field in another set of results.

        Categorical and multi-select fields: the difference in share for each
        level (Newcombe intervals). Boolean: difference in rate. Ordinal and
        numeric: difference in means (bootstrap interval).
        """
        if self.spec.kind == Kind.BOOLEAN:
            a, b = self.values(), other.values()
            return newcombe_diff(sum(a), len(a), sum(b), len(b))
        if self.spec.kind in (Kind.ORDINAL, Kind.NUMERIC):
            return bootstrap_diff(self.numbers(), other.numbers())
        if self.spec.kind in (Kind.CATEGORICAL, Kind.MULTI_SELECT):
            ca, cb = self.counts(), other.counts()
            return Shares(
                {lvl: newcombe_diff(ca[lvl], self.n, cb.get(lvl, 0), other.n) for lvl in ca}
            )
        raise KindError(f"diff() doesn't apply to text field {self.spec.name!r}")

    # --- encoding ------------------------------------------------------------------

    def array(self) -> np.ndarray:
        """Answers encoded for numeric work, one row per record.

        | Kind | Encoding | Missing |
        |---|---|---|
        | categorical | integer code into `spec.levels` | -1 |
        | boolean | 0 or 1 | -1 |
        | ordinal | the level if numeric, else its rank | NaN |
        | numeric | float | NaN |
        | multi-select | 0/1 matrix, one column per option | all 0 |
        | text | object array | None |
        """
        vals = self.values(include_missing=True)
        k = self.spec.kind
        levels = self.spec.levels or ()
        if k == Kind.CATEGORICAL:
            return np.array([-1 if v is None else levels.index(v) for v in vals])
        if k == Kind.BOOLEAN:
            return np.array([-1 if v is None else int(v) for v in vals])
        if k in (Kind.ORDINAL, Kind.NUMERIC):
            return np.array([np.nan if v is None else self.spec.rank(v) for v in vals])
        if k == Kind.MULTI_SELECT:
            m = np.zeros((len(vals), len(levels)), dtype=int)
            for i, v in enumerate(vals):
                for opt in v or []:
                    m[i, levels.index(opt)] = 1
            return m
        return np.array(vals, dtype=object)

    # --- summary -------------------------------------------------------------------

    def summary(self) -> dict[str, Any]:
        """The default summary for the field's kind."""
        k = self.spec.kind
        s: dict[str, Any] = {"kind": k.value, "n": self.n, "missing": self.missing}
        if self.n == 0:
            return s
        if k == Kind.CATEGORICAL:
            s.update(mode=self.mode(), freq=self.freq(), entropy=round(self.entropy(), 3))
        elif k == Kind.BOOLEAN:
            s.update(rate=self.rate())
        elif k == Kind.ORDINAL:
            numeric_levels = all(
                isinstance(v, (int, float)) and not isinstance(v, bool)
                for v in self.spec.levels or ()
            )
            if numeric_levels:
                s.update(mean=self.mean(), median=self.median())
            else:  # text levels: mean and median are over ranks 0..k-1
                s.update(mean_rank=self.mean(), median_rank=self.median())
            s.update(mode=self.mode(), freq=self.freq())
        elif k == Kind.NUMERIC:
            s.update(
                mean=self.mean(),
                median=self.median(),
                std=round(self.std(), 4),
                quantiles=self.quantiles(),
            )
        elif k == Kind.MULTI_SELECT:
            s.update(freq=self.freq(), mean_selected=self.mean_selected())
        else:
            s.update(examples=self.sample(3))
        return s

    def __repr__(self) -> str:
        return f"<FieldView {self.spec.name!r} {self.spec.kind} n={self.n}>"


# ===========================================================================
# ResultSet
# ===========================================================================


class ResultSet:
    """Every record from a run, with summaries by field.

    `rs["field"]` gives a `FieldView`. Iterating gives the records.
    """

    def __init__(self, schema: type[BaseModel], records: list[Record]) -> None:
        self.schema = schema
        self.records = records
        self.specs: dict[str, FieldSpec] = schema_specs(schema)

    # --- basics ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self.records)

    def __iter__(self) -> Iterator[Record]:
        return iter(self.records)

    def __getitem__(self, field_name: str) -> FieldView:
        return self.field(field_name)

    def __add__(self, other: ResultSet) -> ResultSet:
        if other.schema is not self.schema:
            raise TypeError("can only combine results that share a schema")
        return ResultSet(self.schema, self.records + other.records)

    def field(self, name: str) -> FieldView:
        """The `FieldView` for one answer field."""
        if name not in self.specs:
            msg = f"{name!r} is not a field of {self.schema.__name__}; fields: {list(self.specs)}"
            raise KeyError(msg)
        return FieldView(self.specs[name], self.records)

    @property
    def ok(self) -> ResultSet:
        """Only the successful records."""
        return ResultSet(self.schema, [r for r in self.records if r.ok])

    @property
    def errors(self) -> list[Record]:
        """The failed records."""
        return [r for r in self.records if not r.ok]

    # --- slicing -----------------------------------------------------------------

    def filter(self, pred: Callable[[Record], bool] | None = None, **equals: Any) -> ResultSet:
        """Keep records matching a predicate and/or exact values.

        `rs.filter(lambda r: r.meta.latency_s < 5)` or `rs.filter(model="a", role="b")`.
        Names are looked up like `Record.get`.
        """
        for key in equals:
            self._check_key(key)

        def keep(r: Record) -> bool:
            if pred is not None and not pred(r):
                return False
            return all(_safe_get(r, k) == v for k, v in equals.items())

        return ResultSet(self.schema, [r for r in self.records if keep(r)])

    def group_by(self, *keys: str) -> Grouped:
        """Group on coords, answer fields or metadata, for example `"model"` or `"role"`."""
        if not keys:
            raise ValueError("group_by needs at least one key")
        for key in keys:
            self._check_key(key)
        groups: dict[Any, list[Record]] = {}
        for r in self.records:
            k = tuple(_safe_get(r, key) for key in keys)
            groups.setdefault(k[0] if len(keys) == 1 else k, []).append(r)
        return Grouped(keys, {k: ResultSet(self.schema, v) for k, v in groups.items()})

    def _check_key(self, key: str) -> None:
        if not self.records:
            return
        for r in self.records:
            try:
                r.get(key)
            except (KeyError, AttributeError):
                continue
            return
        names = sorted({k for r in self.records for k in r.coords} | set(self.specs))
        raise KeyError(f"no record has {key!r}; try one of {names} or a meta field")

    # --- summaries ---------------------------------------------------------------

    def summary(self, fields: Sequence[str] | None = None) -> Summary:
        """Default summary of each field. Text fields are left out unless named."""
        names = (
            list(fields) if fields else [n for n, s in self.specs.items() if s.kind != Kind.TEXT]
        )
        return Summary({n: self.field(n).summary() for n in names})

    def usage(self) -> Info:
        """Call counts, errors, tokens, cost and latency.

        Token, cost and latency totals count only calls made in this run, not cache hits.
        """
        live = [r for r in self.records if not r.meta.cache_hit]

        def total(attr: str) -> int | None:
            vals = [getattr(r.meta.usage, attr) for r in live]
            present = [v for v in vals if v is not None]
            return sum(present) if present else None

        costs = [r.meta.cost_usd for r in live if r.meta.cost_usd is not None]
        lat = [r.meta.latency_s for r in live if r.meta.latency_s is not None]
        kinds = Counter(r.error_kind or "error" for r in self.errors)
        return Info(
            records=len(self.records),
            api_calls=sum(r.meta.attempts for r in live),
            cache_hits=len(self.records) - len(live),
            errors=len(self.errors),
            error_kinds=dict(kinds),
            input_tokens=total("input_tokens"),
            output_tokens=total("output_tokens"),
            reasoning_tokens=total("reasoning_tokens"),
            cached_input_tokens=total("cached_input_tokens"),
            cost_usd=round(sum(costs), 6) if costs else None,
            median_latency_s=round(float(np.median(lat)), 3) if lat else None,
        )

    # --- export ------------------------------------------------------------------

    def to_rows(self) -> list[dict[str, Any]]:
        """One dict per record (see `Record.flat`)."""
        return [r.flat() for r in self.records]

    def to_pandas(self) -> Any:
        """One row per record: coords, answer fields, `meta.*`, `usage.*` and errors."""
        import pandas as pd  # noqa: PLC0415

        return pd.DataFrame(self.to_rows())

    def arrays(self, fields: Sequence[str] | None = None) -> dict[str, np.ndarray]:
        """`FieldView.array()` for each field."""
        names = list(fields) if fields else list(self.specs)
        return {n: self.field(n).array() for n in names}

    def __repr__(self) -> str:
        head = (
            f"<ResultSet {self.schema.__name__}: {len(self.records)} records, "
            f"{len(self.errors)} errors>"
        )
        if not self.records:
            return head
        return f"{head}\n{self.summary()!r}"

    def _repr_html_(self) -> str:
        head = (
            f"<p><b>ResultSet {self.schema.__name__}</b>: {len(self.records)} records, "
            f"{len(self.errors)} errors</p>"
        )
        return head + (self.summary()._repr_html_() if self.records else "")


# ===========================================================================
# Grouped
# ===========================================================================


class Grouped:
    """Results split into groups. `grouped["field"]` summarizes a field per group.

    Attributes:
        keys: The names grouped by.
        groups: `{group: ResultSet}`. A group is a value, or a tuple of values
            when grouping by several names.
    """

    def __init__(self, keys: tuple[str, ...], groups: dict[Any, ResultSet]) -> None:
        self.keys = keys
        self.groups = groups

    def __getitem__(self, field_name: str) -> GroupedField:
        return GroupedField(self.keys, {k: g.field(field_name) for k, g in self.groups.items()})

    def __iter__(self) -> Iterator[tuple[Any, ResultSet]]:
        return iter(self.groups.items())

    def __len__(self) -> int:
        return len(self.groups)

    def get(self, group: Any) -> ResultSet:
        """The results of one group."""
        return self.groups[group]

    def map(self, fn: Callable[[ResultSet], Any]) -> ByGroup:
        """Apply `fn` to each group's results."""
        return ByGroup({k: fn(g) for k, g in self.groups.items()}, self.keys)

    def size(self) -> ByGroup:
        """Number of records in each group."""
        return self.map(len)

    def usage(self) -> ByGroup:
        """`ResultSet.usage()` per group."""
        return self.map(lambda g: g.usage())

    def __repr__(self) -> str:
        sizes = {k: len(g) for k, g in self.groups.items()}
        return f"<Grouped by {self.keys}: {sizes}>"


class GroupedField:
    """One field across groups. Each method returns `{group: result}` as a `ByGroup`."""

    def __init__(self, keys: tuple[str, ...], views: dict[Any, FieldView]) -> None:
        self.keys = keys
        self.views = views

    def _each(self, fn: Callable[[FieldView], Any]) -> ByGroup:
        return ByGroup({k: fn(v) for k, v in self.views.items()}, self.keys)

    @property
    def n(self) -> ByGroup:
        """Answers present per group."""
        return self._each(lambda v: v.n)

    @property
    def missing(self) -> ByGroup:
        """Missing answers per group."""
        return self._each(lambda v: v.missing)

    def values(self, include_missing: bool = False) -> ByGroup:
        """See `FieldView.values`."""
        return self._each(lambda v: v.values(include_missing))

    def counts(self) -> ByGroup:
        """See `FieldView.counts`."""
        return self._each(lambda v: v.counts())

    def freq(self) -> ByGroup:
        """See `FieldView.freq`."""
        return self._each(lambda v: v.freq())

    def mode(self) -> ByGroup:
        """See `FieldView.mode`."""
        return self._each(lambda v: v.mode())

    def entropy(self) -> ByGroup:
        """See `FieldView.entropy`."""
        return self._each(lambda v: v.entropy())

    def rate(self) -> ByGroup:
        """See `FieldView.rate`."""
        return self._each(lambda v: v.rate())

    def mean(self) -> ByGroup:
        """See `FieldView.mean`."""
        return self._each(lambda v: v.mean())

    def median(self) -> ByGroup:
        """See `FieldView.median`."""
        return self._each(lambda v: v.median())

    def std(self) -> ByGroup:
        """See `FieldView.std`."""
        return self._each(lambda v: v.std())

    def quantiles(self, qs: Sequence[float] = (0.1, 0.25, 0.5, 0.75, 0.9)) -> ByGroup:
        """See `FieldView.quantiles`."""
        return self._each(lambda v: v.quantiles(qs))

    def mean_selected(self) -> ByGroup:
        """See `FieldView.mean_selected`."""
        return self._each(lambda v: v.mean_selected())

    def sample(self, k: int = 5, seed: int = 0) -> ByGroup:
        """See `FieldView.sample`."""
        return self._each(lambda v: v.sample(k, seed))

    def summary(self) -> ByGroup:
        """See `FieldView.summary`."""
        return self._each(lambda v: v.summary())

    def diff(self, a: Any, b: Any) -> Shares | Interval:
        """Group `a` minus group `b` for this field. See `FieldView.diff`."""
        return self.views[a].diff(self.views[b])

    def __repr__(self) -> str:
        return f"<GroupedField by {self.keys}: {len(self.views)} groups>"


def _safe_get(r: Record, key: str) -> Any:
    try:
        return r.get(key)
    except (AttributeError, KeyError):
        return None
