"""epluribus: sample LLMs many times on a structured question and summarize the answers by type.

Provider clients load their SDK when imported, so import them from their modules:

    from epluribus.clients.anthropic import AnthropicClient
    from epluribus.clients.openai import OpenAIClient
"""

from importlib.metadata import PackageNotFoundError, version

from .cache import Cache, DiskCache, MemoryCache
from .clients.base import Client, Generation
from .clients.fake import FakeClient
from .display import ByGroup, Info, Shares, Summary
from .errors import (
    CallError,
    EpluribusError,
    InvalidOutputError,
    KindError,
    ProviderError,
    RefusalError,
    TruncatedError,
)
from .fields import Categorical, FieldSpec, Kind, Numeric, Ordinal, Text, schema_specs
from .records import Record, ResponseMeta, Usage
from .results import FieldView, Grouped, GroupedField, ResultSet
from .sampler import Mix, Task, arun, plan, run
from .stats import Interval

try:
    __version__ = version("epluribus")
except PackageNotFoundError:  # pragma: no cover - running from a source tree
    __version__ = "0.0.0"

__all__ = [
    "ByGroup",
    "Cache",
    "CallError",
    "Categorical",
    "Client",
    "DiskCache",
    "EpluribusError",
    "FakeClient",
    "FieldSpec",
    "FieldView",
    "Generation",
    "Grouped",
    "GroupedField",
    "Info",
    "Interval",
    "InvalidOutputError",
    "Kind",
    "KindError",
    "MemoryCache",
    "Mix",
    "Numeric",
    "Ordinal",
    "ProviderError",
    "Record",
    "RefusalError",
    "ResponseMeta",
    "ResultSet",
    "Shares",
    "Summary",
    "Task",
    "Text",
    "TruncatedError",
    "Usage",
    "__version__",
    "arun",
    "plan",
    "run",
    "schema_specs",
]
