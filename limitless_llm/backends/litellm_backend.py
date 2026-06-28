from __future__ import annotations

from collections.abc import AsyncIterator
from typing import Any

import litellm
import structlog
import tenacity

from limitless_llm.core.rate_limiter import TPMRateLimiter
from limitless_llm.core.token_counter import TokenCounter, get_context_window
from limitless_llm.exceptions import ContextLengthExceededError, TPMBudgetExceededError
from limitless_llm.models.requests import LLMRequest
from limitless_llm.models.responses import LLMChunk, LLMResponse, UsageStats

litellm.suppress_debug_info = True

log = structlog.get_logger(__name__)

# Default retry-after duration when the header is absent from a 429 response.
_DEFAULT_RETRY_AFTER = 60


def _extract_retry_after(exc: BaseException) -> float:
    try:
        headers = getattr(exc, "response", None) and exc.response.headers  # type: ignore[attr-defined]
        if headers:
            return float(headers.get("retry-after", _DEFAULT_RETRY_AFTER))
    except Exception:
        pass
    return float(_DEFAULT_RETRY_AFTER)


class LiteLLMBackend:
    """LLM backend implemented via LiteLLM - supports all providers with a unified interface."""

    def __init__(self, rate_limiter: TPMRateLimiter) -> None:
        if rate_limiter is None:
            raise ValueError("rate_limiter is required; pass TPMRateLimiter(tpm_limit=...)")
        self._rate_limiter = rate_limiter

    async def complete(self, request: LLMRequest) -> LLMResponse:
        """Execute a non-streaming LLM call with TPM-aware scheduling and 429 retry.

        Args:
            request: The LLM request parameters.

        Returns:
            Full LLMResponse with content and usage stats.

        Raises:
            TPMBudgetExceededError: After one failed 429 retry.
            ContextLengthExceededError: On context overflow (not retried).
        """
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]
        input_tokens = TokenCounter.count(request.system_prompt) + TokenCounter.count(
            request.user_prompt
        )
        estimated = input_tokens + request.max_tokens

        reservation_id = await self._rate_limiter.wait_if_needed(estimated)

        try:
            response = await self._call_with_retry(request, messages, estimated)
        except litellm.ContextWindowExceededError as exc:  # type: ignore[attr-defined]
            await self._rate_limiter.release(reservation_id)
            raise ContextLengthExceededError(
                model=request.model,
                input_tokens=input_tokens,
                output_cap=request.max_tokens,
                context_window=get_context_window(request.model),
                phase=request.phase or "complete",
            ) from exc
        except TPMBudgetExceededError:
            await self._rate_limiter.release(reservation_id)
            raise

        usage = UsageStats(
            prompt_tokens=response.usage.prompt_tokens,
            completion_tokens=response.usage.completion_tokens,
            total_tokens=response.usage.total_tokens,
        )
        self._rate_limiter.record(usage.total_tokens, reservation_id)

        return LLMResponse(
            content=response.choices[0].message.content or "",
            usage=usage,
            model=request.model,
        )

    async def _call_with_retry(
        self,
        request: LLMRequest,
        messages: list[dict[str, str]],
        estimated: int,
    ) -> Any:  # noqa: ANN401 — litellm.ModelResponse is not exported in litellm's type stubs
        """Call litellm with one 429 retry, honouring the retry-after response header.

        Uses tenacity per coding guidelines. stop_after_attempt(2) = 1 initial + 1 retry,
        matching spec §4.3. TPMBudgetExceededError is raised via retry_error_callback after
        both attempts fail.
        """

        def _wait(retry_state: tenacity.RetryCallState) -> float:
            assert retry_state.outcome is not None
            exc = retry_state.outcome.exception()
            return (
                _extract_retry_after(exc)
                if isinstance(exc, BaseException)
                else float(_DEFAULT_RETRY_AFTER)
            )

        def _before_sleep(retry_state: tenacity.RetryCallState) -> None:
            assert retry_state.outcome is not None
            exc = retry_state.outcome.exception()
            wait = (
                _extract_retry_after(exc)
                if isinstance(exc, BaseException)
                else float(_DEFAULT_RETRY_AFTER)
            )
            log.warning(
                "llm_429_retry",
                model=request.model,
                retry_after=wait,
                attempt=retry_state.attempt_number,
                error=str(exc),
            )

        def _on_retry_exhausted(retry_state: tenacity.RetryCallState) -> None:
            raise TPMBudgetExceededError(
                model=request.model,
                estimated_tokens=estimated,
                rolling_window_tokens=self._rate_limiter.window_sum(),
                tpm_limit=self._rate_limiter.tpm_limit or 0,
            )

        async for attempt in tenacity.AsyncRetrying(
            retry=tenacity.retry_if_exception_type(litellm.RateLimitError),  # type: ignore[attr-defined]
            wait=_wait,
            stop=tenacity.stop_after_attempt(2),
            before_sleep=_before_sleep,
            retry_error_callback=_on_retry_exhausted,
        ):
            with attempt:
                return await litellm.acompletion(
                    model=request.model,
                    messages=messages,
                    max_tokens=request.max_tokens,
                )
        raise AssertionError("unreachable")

    async def stream(self, request: LLMRequest) -> AsyncIterator[LLMChunk]:
        """Execute a streaming LLM call with TPM-aware scheduling.

        Args:
            request: The LLM request parameters.

        Yields:
            LLMChunk instances as tokens arrive.

        Raises:
            ContextLengthExceededError: On context overflow.
        """
        messages = [
            {"role": "system", "content": request.system_prompt},
            {"role": "user", "content": request.user_prompt},
        ]
        input_tokens = TokenCounter.count(request.system_prompt) + TokenCounter.count(
            request.user_prompt
        )
        estimated = input_tokens + request.max_tokens
        reservation_id = await self._rate_limiter.wait_if_needed(estimated)

        actual_completion_tokens = 0
        try:
            response = await litellm.acompletion(
                model=request.model,
                messages=messages,
                max_tokens=request.max_tokens,
                stream=True,
            )
            async for part in response:
                delta = part.choices[0].delta.content or ""
                finish_reason = part.choices[0].finish_reason
                if delta:
                    actual_completion_tokens += TokenCounter.count(delta)
                yield LLMChunk(delta=delta, finish_reason=finish_reason)
        except litellm.ContextWindowExceededError as exc:  # type: ignore[attr-defined]
            raise ContextLengthExceededError(
                model=request.model,
                input_tokens=input_tokens,
                output_cap=request.max_tokens,
                context_window=get_context_window(request.model),
                phase=request.phase or "stream",
            ) from exc
        finally:
            actual_total = input_tokens + actual_completion_tokens
            self._rate_limiter.record(actual_total, reservation_id)
