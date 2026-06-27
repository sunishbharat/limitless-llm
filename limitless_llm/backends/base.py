from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Protocol, runtime_checkable

from limitless_llm.models.requests import LLMRequest
from limitless_llm.models.responses import LLMChunk, LLMResponse


@runtime_checkable
class LLMBackend(Protocol):
    """Protocol for all LLM backends. Implementations must be async."""

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Send a request and wait for the full response.

        Args:
            request: The LLM request to execute.

        Returns:
            The complete model response with usage stats.

        Raises:
            TPMBudgetExceededError: If rate limits are exhausted after retries.
            ContextLengthExceededError: If the prompt exceeds the context window.
        """
        ...

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        """Send a request and stream tokens as they arrive.

        Args:
            request: The LLM request to execute.

        Yields:
            LLMChunk instances; the final chunk has finish_reason set.

        Raises:
            TPMBudgetExceededError: If rate limits are exhausted after retries.
            ContextLengthExceededError: If the prompt exceeds the context window.
        """
        ...
