from __future__ import annotations

from collections.abc import AsyncIterator

from limitless_llm.core.rate_limiter import TPMRateLimiter
from limitless_llm.core.token_counter import TokenCounter
from limitless_llm.models.requests import LLMRequest
from limitless_llm.models.responses import LLMChunk, LLMResponse, UsageStats


class MockBackend:
    """Deterministic backend for tests. Never makes real API calls.

    Responses are set via the `responses` list; each complete() call pops
    the first item. For stream(), the same response text is yielded word by word.
    """

    def __init__(
        self,
        rate_limiter: TPMRateLimiter,
        responses: list[str] | None = None,
    ) -> None:
        if rate_limiter is None:
            raise ValueError("rate_limiter is required")
        self._rate_limiter = rate_limiter
        self._responses: list[str] = responses or []
        self.calls: list[LLMRequest] = []

    def set_responses(self, responses: list[str]) -> None:
        """Replace the response queue."""
        self._responses = list(responses)

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Return the next queued response.

        Args:
            request: Recorded for inspection in tests.

        Returns:
            LLMResponse with the queued text and approximate token counts.

        Raises:
            IndexError: If the response queue is empty.
        """
        self.calls.append(request)
        content = self._responses.pop(0) if self._responses else ""
        input_tokens = TokenCounter.count(request.system_prompt) + TokenCounter.count(
            request.user_prompt
        )
        output_tokens = TokenCounter.count(content)
        total = input_tokens + output_tokens

        reservation_id = await self._rate_limiter.wait_if_needed(input_tokens + request.max_tokens)
        self._rate_limiter.record(total, reservation_id)

        return LLMResponse(
            content=content,
            usage=UsageStats(
                prompt_tokens=input_tokens,
                completion_tokens=output_tokens,
                total_tokens=total,
            ),
            model=request.model,
        )

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        """Yield the next queued response word by word.

        Args:
            request: Recorded for inspection in tests.

        Yields:
            LLMChunk per word, with finish_reason set on the last chunk.
        """
        self.calls.append(request)
        content = self._responses.pop(0) if self._responses else ""
        words = content.split()
        input_tokens = TokenCounter.count(request.system_prompt) + TokenCounter.count(
            request.user_prompt
        )
        reservation_id = await self._rate_limiter.wait_if_needed(input_tokens + request.max_tokens)

        output_tokens = 0
        for i, word in enumerate(words):
            is_last = i == len(words) - 1
            delta = word + ("" if is_last else " ")
            output_tokens += TokenCounter.count(delta)
            yield LLMChunk(delta=delta, finish_reason="stop" if is_last else None)

        self._rate_limiter.record(input_tokens + output_tokens, reservation_id)
