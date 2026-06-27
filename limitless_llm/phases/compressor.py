from __future__ import annotations

import structlog

from limitless_llm.backends.base import LLMBackend
from limitless_llm.core.rate_limiter import TPMRateLimiter
from limitless_llm.core.token_counter import TokenCounter
from limitless_llm.exceptions import ContextLengthExceededError, TPMBudgetExceededError
from limitless_llm.models.requests import LLMRequest
from limitless_llm.types import ChunkIndex, TokenCount

log = structlog.get_logger(__name__)

# Hard cap on compressed summary length in tokens. Enforced by truncation after
# generation, not solely by instruction-following.
SUMMARY_TOKEN_CAP: TokenCount = 300

# Verbatim compression prompt - must not be paraphrased; defined in spec §6.2.
_COMPRESSION_SYSTEM_PROMPT = (
    "You are a compression assistant. Your sole task is to produce a concise summary."
)

_COMPRESSION_USER_TEMPLATE = """\
You will receive:
  PRIOR SUMMARY: a summary of everything processed before the current section (may be empty)
  CURRENT OUTPUT: the extracted output from the section just processed

Produce a single updated summary that captures:
  - All named entities (people, organisations, systems, identifiers)
  - All defined terms and their definitions
  - All constraints, rules, deadlines, and numeric values
  - Any unresolved questions or flagged conflicts from either input

Requirements:
  - Maximum 300 tokens. If you cannot fit everything, prioritise constraints and defined terms\
 over narrative context.
  - Plain prose only. No bullet points, no headers, no markdown.
  - Do not add commentary, preamble, or closing remarks.
  - Output the summary text only.

PRIOR SUMMARY:
{prior_summary}

CURRENT OUTPUT:
{chunk_output}"""


def _truncate_at_sentence_boundary(text: str, token_cap: TokenCount) -> tuple[str, int]:
    """Truncate text at the last sentence boundary before token_cap.

    Returns the truncated text and the number of tokens dropped.
    """
    if TokenCounter.count(text) <= token_cap:
        return text, 0

    # Split into sentences by period/exclamation/question mark followed by space.
    import re

    sentences = re.split(r"(?<=[.!?])\s+", text)
    result: list[str] = []
    running = 0
    for sentence in sentences:
        cost = TokenCounter.count(sentence)
        if running + cost > token_cap:
            break
        result.append(sentence)
        running += cost

    if not result:
        # First sentence already exceeds the cap; return it as the smallest
        # atomic unit we can trim to (spec R3-#2: slight overrun permitted).
        first = sentences[0]
        return first, TokenCounter.count(text) - TokenCounter.count(first)

    truncated = " ".join(result)
    dropped = TokenCounter.count(text) - TokenCounter.count(truncated)
    return truncated, dropped


class Compressor:
    """Maintains the rolling compressed prior summary between chunk calls."""

    def __init__(self, backend: LLMBackend, rate_limiter: TPMRateLimiter, model: str) -> None:
        if rate_limiter is None:
            raise ValueError("rate_limiter is required")
        self._backend = backend
        self._rate_limiter = rate_limiter
        self._model = model
        self._current_summary: str = ""

    @property
    def current_summary(self) -> str:
        """The current compressed summary text."""
        return self._current_summary

    @property
    def current_summary_tokens(self) -> TokenCount:
        """Token count of the current compressed summary."""
        return TokenCounter.count(self._current_summary)

    async def update(self, chunk_output: str, chunk_index: ChunkIndex) -> None:
        """Update the compressed summary after a chunk call completes.

        On TPMBudgetExceededError the existing summary is kept unchanged and a
        warning is logged - the pipeline continues with degraded context for this
        chunk, but does not abort.

        On any other error the exception propagates.

        Args:
            chunk_output: The LLM's output for the chunk that just completed.
            chunk_index: Index of the chunk, used in log context.
        """
        user_prompt = _COMPRESSION_USER_TEMPLATE.format(
            prior_summary=self._current_summary or "(empty)",
            chunk_output=chunk_output,
        )
        request = LLMRequest(
            model=self._model,
            system_prompt=_COMPRESSION_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=SUMMARY_TOKEN_CAP + 50,  # small buffer; we truncate after
            phase="compression",
        )

        try:
            response = await self._backend.complete(request)
        except TPMBudgetExceededError:
            log.warning(
                "compressor_tpm_exceeded_using_prior_summary",
                chunk_index=chunk_index,
                current_summary_tokens=self.current_summary_tokens,
            )
            return
        except ContextLengthExceededError:
            log.error(
                "compressor_context_length_exceeded",
                chunk_index=chunk_index,
            )
            raise

        raw = response.content
        if TokenCounter.count(raw) > SUMMARY_TOKEN_CAP:
            truncated, dropped = _truncate_at_sentence_boundary(raw, SUMMARY_TOKEN_CAP)
            log.warning(
                "compressor_summary_truncated",
                chunk_index=chunk_index,
                tokens_dropped=dropped,
                truncated_tokens=TokenCounter.count(truncated),
            )
            self._current_summary = truncated
        else:
            self._current_summary = raw

        log.debug(
            "compressor_updated",
            chunk_index=chunk_index,
            summary_tokens=self.current_summary_tokens,
        )
