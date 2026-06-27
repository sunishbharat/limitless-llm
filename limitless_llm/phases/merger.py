from __future__ import annotations

import re

import structlog

from limitless_llm.backends.base import LLMBackend
from limitless_llm.core.rate_limiter import TPMRateLimiter
from limitless_llm.core.token_counter import TokenCounter, get_context_window
from limitless_llm.exceptions import MergeInputTooLargeError
from limitless_llm.models.requests import LLMRequest
from limitless_llm.types import TokenCount

log = structlog.get_logger(__name__)

# Token budget for the merge prompt template boilerplate (LEFT/RIGHT SECTION headers, etc.)
_MERGE_PROMPT_OVERHEAD: TokenCount = 300
# Token budget for the system persona + headers shared across all calls.
_SYSTEM_OVERHEAD: TokenCount = 500

# Verbatim merge prompt - must not be paraphrased; defined in spec §8.4.
_MERGE_SYSTEM_PROMPT = "You are merging two extracted summaries of adjacent sections of a document."

_MERGE_USER_TEMPLATE = """\
Your tasks:

1. Combine the information from both sections into a single coherent output.

2. If the two sections contain contradictory information about the same fact \
(e.g., conflicting dates, amounts, names, or rules), flag it with:
   [CONFLICT: left says "<value>", right says "<value>" - preserved for human review]
   Do not resolve the conflict yourself.

3. If either input already contains a [CONFLICT: ...] marker, copy it verbatim into \
your output. Do not attempt to summarise, resolve, or reformat existing conflict markers.

4. Do not drop any [CONFLICT: ...] markers from either input.

LEFT SECTION:
{left}

RIGHT SECTION:
{right}"""

_CONFLICT_PATTERN = re.compile(r"\[CONFLICT:[^\]]+\]")


def _extract_conflicts(text: str) -> list[str]:
    return _CONFLICT_PATTERN.findall(text)


def _truncate_at_sentence_boundary(text: str, token_budget: TokenCount) -> tuple[str, int]:
    """Truncate to the last sentence that fits within token_budget.

    Returns (truncated_text, tokens_dropped).
    """
    if TokenCounter.count(text) <= token_budget:
        return text, 0
    sentences = re.split(r"(?<=[.!?])\s+", text)
    result: list[str] = []
    running: TokenCount = 0
    for sentence in sentences:
        cost = TokenCounter.count(sentence)
        if running + cost > token_budget:
            break
        result.append(sentence)
        running += cost
    truncated = " ".join(result) if result else sentences[0]
    dropped = TokenCounter.count(text) - TokenCounter.count(truncated)
    return truncated, dropped


class HierarchicalMerge:
    """Merges a list of chunk outputs into one via a balanced binary tree reduction.

    Tree structure (spec §8.1):
    - Pair adjacent outputs; carry the last unpaired output forward unchanged.
    - Repeat until one output remains.
    - Depth is ceil(log2(N)), bounding context accumulation.
    """

    def __init__(
        self,
        backend: LLMBackend,
        rate_limiter: TPMRateLimiter,
        model: str,
        max_output_tokens: TokenCount,
        *,
        include_conflict_summary: bool = True,
    ) -> None:
        if rate_limiter is None:
            raise ValueError("rate_limiter is required")
        self._backend = backend
        self._rate_limiter = rate_limiter
        self._model = model
        self._max_output_tokens = max_output_tokens
        self._context_window = get_context_window(model)
        self._include_conflict_summary = include_conflict_summary

    async def merge(self, chunk_outputs: list[str]) -> str:
        """Reduce chunk_outputs to a single merged string.

        Args:
            chunk_outputs: Ordered list of per-chunk LLM outputs.

        Returns:
            Final merged text with a conflict summary appended if conflicts exist.

        Raises:
            MergeInputTooLargeError: If even a single input exceeds the usable budget.
        """
        if not chunk_outputs:
            return ""
        if len(chunk_outputs) == 1:
            return chunk_outputs[0]

        level = chunk_outputs
        tree_level = 0
        while len(level) > 1:
            next_level: list[str] = []
            for i in range(0, len(level), 2):
                if i + 1 >= len(level):
                    next_level.append(level[i])
                else:
                    merged = await self._merge_two(level[i], level[i + 1], tree_level, i // 2)
                    next_level.append(merged)
            level = next_level
            tree_level += 1

        final = level[0]
        if self._include_conflict_summary:
            return self._append_conflict_summary(final)
        return final

    async def _merge_two(self, left: str, right: str, tree_level: int, pair_index: int) -> str:
        left_tokens = TokenCounter.count(left)
        right_tokens = TokenCounter.count(right)
        total_input = left_tokens + right_tokens + _MERGE_PROMPT_OVERHEAD + _SYSTEM_OVERHEAD

        if total_input + self._max_output_tokens > self._context_window:
            left, right, left_tokens, right_tokens = self._overflow_fallback(
                left, right, left_tokens, right_tokens, tree_level, pair_index
            )

        user_prompt = _MERGE_USER_TEMPLATE.format(left=left, right=right)
        request = LLMRequest(
            model=self._model,
            system_prompt=_MERGE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            max_tokens=self._max_output_tokens,
            phase="merge",
        )

        log.info(
            "merge_call",
            tree_level=tree_level,
            pair_index=pair_index,
            left_tokens=left_tokens,
            right_tokens=right_tokens,
        )
        response = await self._backend.complete(request)
        return response.content

    def _overflow_fallback(
        self,
        left: str,
        right: str,
        left_tokens: TokenCount,
        right_tokens: TokenCount,
        tree_level: int,
        pair_index: int,
    ) -> tuple[str, str, TokenCount, TokenCount]:
        """Truncate the larger input to fit within the context budget.

        Raises MergeInputTooLargeError if even one input alone exceeds the budget.
        """
        budget = (
            self._context_window
            - self._max_output_tokens
            - _MERGE_PROMPT_OVERHEAD
            - _SYSTEM_OVERHEAD
        )

        # Degenerate case: single input already exceeds the entire available budget.
        if left_tokens > budget and right_tokens > budget:
            raise MergeInputTooLargeError(
                model=self._model,
                left_tokens=left_tokens,
                right_tokens=right_tokens,
                context_window=self._context_window,
                max_output_tokens=self._max_output_tokens,
            )

        if left_tokens >= right_tokens:
            new_left, dropped = _truncate_at_sentence_boundary(left, budget - right_tokens)
            log.warning(
                "merge_input_truncated",
                tree_level=tree_level,
                pair_index=pair_index,
                side="left",
                tokens_dropped=dropped,
            )
            return new_left, right, TokenCounter.count(new_left), right_tokens
        else:
            new_right, dropped = _truncate_at_sentence_boundary(right, budget - left_tokens)
            log.warning(
                "merge_input_truncated",
                tree_level=tree_level,
                pair_index=pair_index,
                side="right",
                tokens_dropped=dropped,
            )
            return left, new_right, left_tokens, TokenCounter.count(new_right)

    def _append_conflict_summary(self, text: str) -> str:
        """Scan text for [CONFLICT:] markers and append a structured summary section."""
        conflicts = _extract_conflicts(text)
        if not conflicts:
            return text
        lines = ["## Conflicts Requiring Human Review", ""]
        lines.append(
            "The following contradictions were detected across document sections "
            "and could not be automatically resolved:"
        )
        lines.append("")
        for i, marker in enumerate(conflicts, start=1):
            lines.append(f"{i}. {marker}")
        return text + "\n\n" + "\n".join(lines)
