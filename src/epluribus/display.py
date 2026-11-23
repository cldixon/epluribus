"""Readable tables for results, in the terminal and in notebooks.

The result types here are `dict` subclasses, so indexing works as usual
(`rs["choice"].freq()["approve"].value`). They add a table display, an HTML
display for notebooks, and `to_pandas()`.
"""

from __future__ import annotations

import html
import math
from collections.abc import Sequence
from enum import Enum
from typing import Any

from .stats import Interval

Row = list[Any]


def fmt(v: Any) -> str:  # noqa: PLR0911
    """Format one cell."""
    if isinstance(v, Interval):
        return f"{fmt(v.value)} [{fmt(v.lo)}, {fmt(v.hi)}]"
    if isinstance(v, bool) or v is None:
        return str(v)
    if isinstance(v, Enum):
        return str(v.value)
    if isinstance(v, float):
        if math.isnan(v):
            return "nan"
        if abs(v) >= 1000:
            return f"{v:,.0f}"
        return f"{v:.3f}"
    if isinstance(v, tuple):
        return ", ".join(fmt(x) for x in v)
    if isinstance(v, (list, set, frozenset)):
        return "[" + ", ".join(fmt(x) for x in v) + "]"
    return str(v)


def text_table(headers: Sequence[str], rows: Sequence[Row], title: str | None = None) -> str:
    """A plain-text table with aligned columns."""
    cells = [[fmt(c) for c in r] for r in rows]
    widths = [len(h) for h in headers]
    for r in cells:
        for i, c in enumerate(r):
            widths[i] = max(widths[i], len(c))

    def line(parts: Sequence[str]) -> str:
        return "  ".join(p.ljust(w) for p, w in zip(parts, widths, strict=True)).rstrip()

    out = [title] if title else []
    out += [line(headers), line(["-" * w for w in widths])]
    out += [line(r) for r in cells]
    return "\n".join(out)


def html_table(headers: Sequence[str], rows: Sequence[Row], title: str | None = None) -> str:
    """An HTML table for notebooks."""
    head = "".join(f"<th>{html.escape(h)}</th>" for h in headers)
    body = "".join(
        "<tr>" + "".join(f"<td>{html.escape(fmt(c))}</td>" for c in r) + "</tr>" for r in rows
    )
    cap = f"<caption>{html.escape(title)}</caption>" if title else ""
    return f"<table>{cap}<thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


class _TableDict(dict):
    """Base for dict results with a table display."""

    title: str | None = None

    def rows(self) -> tuple[list[str], list[Row]]:  # pragma: no cover - overridden
        raise NotImplementedError

    def __repr__(self) -> str:
        if not self:
            return f"{type(self).__name__}({{}})"
        return text_table(*self.rows(), title=self.title)

    def _repr_html_(self) -> str:
        return html_table(*self.rows(), title=self.title)

    def table(self) -> str:
        """The table as text."""
        return repr(self)

    def raw(self) -> tuple[list[str], list[Row]]:
        """Headers and rows with values left unformatted (used by `to_pandas`)."""
        return self.rows()

    def to_pandas(self) -> Any:
        """The table as a pandas DataFrame, with intervals split into columns."""
        import pandas as pd  # noqa: PLC0415

        headers, rows = _split_intervals(*self.raw())
        return pd.DataFrame(rows, columns=headers)


def _split_intervals(headers: list[str], rows: list[Row]) -> tuple[list[str], list[Row]]:
    split = [any(isinstance(r[i], Interval) for r in rows) for i in range(len(headers))]
    out_h: list[str] = []
    for h, sp in zip(headers, split, strict=True):
        out_h += [h, f"{h}_lo", f"{h}_hi", f"{h}_n"] if sp else [h]
    out_rows = []
    for r in rows:
        row: Row = []
        for c, sp in zip(r, split, strict=True):
            if not sp:
                row.append(c)
            elif isinstance(c, Interval):
                row += [c.value, c.lo, c.hi, c.n]
            else:
                row += [c, None, None, None]
        out_rows.append(row)
    return out_h, out_rows


class Shares(_TableDict):
    """Share choosing each level, as `{level: Interval}`. Returned by `freq()`."""

    value_header = "share"

    def rows(self) -> tuple[list[str], list[Row]]:
        n = next(iter(self.values())).n if self else 0
        rows: list[Row] = [
            [lvl, iv.value, f"[{fmt(iv.lo)}, {fmt(iv.hi)}]"] for lvl, iv in self.items()
        ]
        return ["level", self.value_header, "95% interval"], [*rows, ["(n)", n, ""]]

    def to_pandas(self) -> Any:
        """Columns `level`, `share`, `lo`, `hi`, `n`."""
        import pandas as pd  # noqa: PLC0415

        return pd.DataFrame(
            [[lvl, iv.value, iv.lo, iv.hi, iv.n] for lvl, iv in self.items()],
            columns=["level", "share", "lo", "hi", "n"],
        )


class Differences(Shares):
    """Difference in share for each level, as `{level: Interval}`. Returned by `diff()`."""

    value_header = "difference"


class Info(_TableDict):
    """Named values, such as `usage()` totals."""

    def rows(self) -> tuple[list[str], list[Row]]:
        return ["", "value"], [[k, v] for k, v in self.items()]


class ByGroup(_TableDict):
    """One result per group, as `{group: result}`. Returned by grouped field methods.

    Attributes:
        group_keys: The names the results were grouped by.
    """

    def __init__(self, data: dict[Any, Any], keys: Sequence[str] = ("group",)) -> None:
        super().__init__(data)
        self.group_keys = tuple(keys)

    def _key_cells(self, k: Any) -> Row:
        if len(self.group_keys) > 1 and isinstance(k, tuple):
            return list(k)
        return [k]

    def raw(self) -> tuple[list[str], list[Row]]:
        return self._rows(raw=True)

    def rows(self) -> tuple[list[str], list[Row]]:
        return self._rows(raw=False)

    def _rows(self, raw: bool) -> tuple[list[str], list[Row]]:
        keys = list(self.group_keys)
        values = list(self.values())
        if values and all(isinstance(v, dict) for v in values):
            cols: list[Any] = []
            for v in values:
                cols += [c for c in v if c not in cols]
            rows = [self._key_cells(k) + [v.get(c) for c in cols] for k, v in self.items()]
            if not raw and all(isinstance(v, Shares) for v in values):
                for r, v in zip(rows, values, strict=True):
                    r.append(next(iter(v.values())).n if v else 0)
                return keys + [fmt(c) for c in cols] + ["n"], rows
            return keys + [fmt(c) for c in cols], rows
        if values and all(isinstance(v, Interval) for v in values):
            if raw:
                return [*keys, "value"], [[*self._key_cells(k), v] for k, v in self.items()]
            return [*keys, "value", "95% interval", "n"], [
                [*self._key_cells(k), v.value, f"[{fmt(v.lo)}, {fmt(v.hi)}]", v.n]
                for k, v in self.items()
            ]
        return [*keys, "value"], [[*self._key_cells(k), v] for k, v in self.items()]


class Summary(_TableDict):
    """Default summary of each field, as `{field: {statistic: value}}`."""

    def rows(self) -> tuple[list[str], list[Row]]:
        rows = [[name, s["kind"], s["n"], s["missing"], _describe(s)] for name, s in self.items()]
        return ["field", "kind", "n", "missing", "summary"], rows


def _describe(s: dict[str, Any]) -> str:
    stats = ("mean", "mean_rank", "median", "rate", "mean_selected")
    parts = [f"{key} {format(s[key], '.3f')}" for key in stats if key in s]
    if "freq" in s:
        top = sorted(s["freq"].items(), key=lambda kv: -kv[1].value)[:4]
        parts.append(", ".join(f"{fmt(lvl)} {iv.value:.2f}" for lvl, iv in top))
    if "examples" in s:
        parts.append(" | ".join(str(x)[:40] for x in s["examples"]))
    return "; ".join(parts)
