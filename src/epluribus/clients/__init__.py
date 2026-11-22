"""Model clients.

`Client` and `FakeClient` load with the package. Provider clients load their SDK
when imported, so import them from their modules:

    from epluribus.clients.anthropic import AnthropicClient
    from epluribus.clients.openai import OpenAIClient
"""

from .base import Client, Generation
from .fake import FakeClient

__all__ = ["Client", "FakeClient", "Generation"]
