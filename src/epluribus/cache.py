"""Caches keyed on everything that defines a call, including the sample's seed.

Same key, same stored answer: rerunning a sweep costs nothing, and raising `n` or
adding grid values only makes the new calls. Failed calls are never cached.
"""

from __future__ import annotations

import json
import sqlite3
import threading
from collections.abc import Generator
from contextlib import closing, contextmanager
from pathlib import Path
from typing import Protocol, runtime_checkable

from pydantic import BaseModel

from .clients.base import Generation
from .records import ResponseMeta


@runtime_checkable
class Cache(Protocol):
    """What `run` needs from a cache. Implement it to plug in your own store."""

    def get(self, key: str, schema: type[BaseModel]) -> Generation | None:
        """The stored generation for `key`, rebuilt as `schema`, or `None`."""
        ...

    def set(self, key: str, gen: Generation) -> None:
        """Store a successful generation."""
        ...


class MemoryCache:
    """A cache that lasts as long as the Python process."""

    def __init__(self) -> None:
        self._d: dict[str, Generation] = {}

    def get(self, key: str, schema: type[BaseModel]) -> Generation | None:
        return self._d.get(key)

    def set(self, key: str, gen: Generation) -> None:
        self._d[key] = gen

    def clear(self) -> None:
        """Remove every entry."""
        self._d.clear()

    def __len__(self) -> int:
        return len(self._d)

    def __repr__(self) -> str:
        return f"MemoryCache({len(self)} entries)"


class DiskCache:
    """A cache in a SQLite file, so answers survive restarts and can be shared.

    Answers are stored as JSON and rebuilt with the run's schema, so the file holds
    no pickled code. A stored answer that no longer validates against the schema is
    treated as a miss.

    Args:
        path: The SQLite file. Parent directories are created if needed.
    """

    def __init__(self, path: str | Path = ".epluribus-cache.sqlite") -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        with self._connect() as db:
            db.execute(
                "CREATE TABLE IF NOT EXISTS generations ("
                "key TEXT PRIMARY KEY, output TEXT NOT NULL, meta TEXT NOT NULL, raw_text TEXT)"
            )

    @contextmanager
    def _connect(self) -> Generator[sqlite3.Connection]:
        # A short-lived connection per operation: nothing to close, safe across threads.
        with self._lock, closing(sqlite3.connect(self.path)) as db, db:
            yield db

    def get(self, key: str, schema: type[BaseModel]) -> Generation | None:
        with self._connect() as db:
            row = db.execute(
                "SELECT output, meta, raw_text FROM generations WHERE key = ?", (key,)
            ).fetchone()
        if row is None:
            return None
        output_json, meta_json, raw_text = row
        try:
            output = schema.model_validate_json(output_json)
        except ValueError:
            return None
        meta = ResponseMeta.from_dict(json.loads(meta_json))
        return Generation(output=output, meta=meta, raw_text=raw_text)

    def set(self, key: str, gen: Generation) -> None:
        meta_json = json.dumps(gen.meta.to_dict(), default=str)
        with self._connect() as db:
            db.execute(
                "INSERT OR REPLACE INTO generations VALUES (?, ?, ?, ?)",
                (key, gen.output.model_dump_json(), meta_json, gen.raw_text),
            )

    def clear(self) -> None:
        """Remove every entry."""
        with self._connect() as db:
            db.execute("DELETE FROM generations")

    def __len__(self) -> int:
        with self._connect() as db:
            return db.execute("SELECT COUNT(*) FROM generations").fetchone()[0]

    def __repr__(self) -> str:
        return f"DiskCache({str(self.path)!r}, {len(self)} entries)"
