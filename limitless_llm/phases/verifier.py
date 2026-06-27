from __future__ import annotations

import structlog

from limitless_llm.backends.base import LLMBackend
from limitless_llm.core.rate_limiter import TPMRateLimiter
from limitless_llm.core.token_counter import TokenCounter, get_context_window
from limitless_llm.exceptions import ContextLengthExceededError
from limitless_llm.models.requests import LLMRequest
from limitless_llm.types import TokenCount

log = structlog.get_logger(__name__)

# Covers system persona + headers; same purpose as the pipeline-level constant.
_SYSTEM_OVERHEAD: TokenCount = 500

_VERIFIER_SYSTEM_PROMPT = (
    "You are a verification assistant checking the quality and completeness "
    "of a merged document summary."
)

_VERIFIER_USER_TEMPLATE = """\
Review the following merged output against the original ledger of extracted facts.

Identify any:
- Facts present in the ledger that are absent from the merged output
- Contradictions between the ledger and the merged output
- Conflict markers ([CONFLICT: ...]) that require human attention

LEDGER:
{ledger}

MERGED OUTPUT:
{merged_output}

Provide a brief verification report."""


class VerificationPass:
    """Runs a final verification call comparing the merged output against the ledger."""

    def __init__(
        self,
        backend: LLMBackend,
        rate_limiter: TPMRateLimiter,
        model: str,
        max_output_tokens: TokenCount,
    ) -> None:
        if rate_limiter is None:
            raise ValueError("rate_limiter is required")
        self._backend = backend
        self._rate_limiter = rate_limiter
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._context_window = get_context_window(model)

    async def verify(self, merged_output: str, ledger: str) -> str:
        """Run a verification pass comparing merged_output against ledger.

        Performs a pre-call budget check before sending to the API - an improvement
        over spec §10 which defers detection to the API response. If ledger +
        merged_output exceed the context window, ContextLengthExceededError is raised
        immediately rather than spending a network round-trip on a call that will fail.

        Args:
            merged_output: The final merged text from HierarchicalMerge.
            ledger: Accumulated ledger of extracted facts from all chunks.

        Returns:
            The model's verification report text.

        Raises:
            ContextLengthExceededError: If the combined input exceeds the context window.
        """
        user_prompt = _VERIFIER_USER_TEMPLATE.format(
            ledger=ledger,
            merged_output=merged_output,
        )
        input_tokens = TokenCounter.count(_VERIFIER_SYSTEM_PROMPT) + TokenCounter.count(user_prompt)

        if input_tokens + self._max_output_tokens + _SYSTEM_OVERHEAD > self._context_window:
            raise ContextLengthExceededError(
                model=self._model,
                input_tokens=input_tokens,
                output_cap=self._max_output_tokens,
                context_window=self._context_window,
                phase="verification",
            )

        request = LLMRequest(
            model=self._model,
            system_prompt=_VERIFIER_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=self._max_output_tokens,
            phase="verification",
        )

        log.info("verification_start", input_tokens=input_tokens)
        response = await self._backend.complete(request)
        log.info("verification_complete", output_tokens=response.usage.completion_tokens)
        return response.content
