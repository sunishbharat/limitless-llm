from __future__ import annotations

from typing import Any

import structlog
from llama_index.core.node_parser import SentenceSplitter

from limitless_llm.core.token_counter import TokenCounter
from limitless_llm.types import TokenCount

log = structlog.get_logger(__name__)


class StructuralSplitter:
    """Splits document text into token-bounded chunks using llama-index SentenceSplitter.

    SentenceSplitter respects sentence boundaries, preventing mid-sentence cuts.
    Code block boundary protection is a Phase 2 concern; see spec §10.
    """

    def __init__(self, chunk_size: TokenCount, chunk_overlap: int = 0) -> None:
        self._chunk_size = chunk_size

        # SentenceSplitter.tokenizer must return a list (it calls len() on the result).
        # Wrap TokenCounter.count by returning a list of that many dummy elements.
        def _list_tokenizer(text: str) -> list[int]:
            return [0] * TokenCounter.count(text)

        self._splitter = SentenceSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            tokenizer=_list_tokenizer,
        )

    def split(self, text: str) -> list[str]:
        """Split text into chunks no larger than chunk_size tokens.

        Args:
            text: The full document text to split.

        Returns:
            Ordered list of chunk strings.
        """
        nodes = self._splitter.get_nodes_from_documents([_make_document(text)])
        chunks = [node.get_content() for node in nodes]
        log.info(
            "chunker_split",
            chunk_count=len(chunks),
            chunk_size_limit=self._chunk_size,
        )
        return chunks


def _tail(words: list[str], token_limit: TokenCount) -> list[str]:
    """Return the largest suffix of words whose token count does not exceed token_limit.

    Uses incremental per-word counting to avoid O(N^2) re-tokenisation of the
    growing candidate string.

    Edge case: if a single word exceeds token_limit, include it anyway (it cannot
    be split further) and log a warning so the caller knows the tail is over-budget.

    Args:
        words: Tokenised word list from the previous chunk's text.
        token_limit: Maximum tokens the tail may consume.

    Returns:
        Suffix word list.
    """
    tail: list[str] = []
    running_tokens: TokenCount = 0
    for word in reversed(words):
        word_tokens = TokenCounter.count(word)
        if running_tokens + word_tokens > token_limit:
            if not tail:
                log.warning(
                    "_tail_single_word_over_budget",
                    word_prefix=word[:40],
                    word_tokens=word_tokens,
                    token_limit=token_limit,
                )
                return [word]
            break
        tail = [word] + tail
        running_tokens += word_tokens
    return tail


def build_tail(previous_chunk_text: str, token_limit: TokenCount = 200) -> str:
    """Return the token-bounded tail of the previous chunk as a string.

    Args:
        previous_chunk_text: The full text of the preceding chunk.
        token_limit: Maximum tokens to include in the tail (default 200).

    Returns:
        Tail string, or empty string if previous_chunk_text is empty.
    """
    if not previous_chunk_text:
        return ""
    words = previous_chunk_text.split()
    tail_words = _tail(words, token_limit)
    return " ".join(tail_words)


def _make_document(text: str) -> Any:  # noqa: ANN401 — llama_index.core.Document has no type stubs
    """Wrap text in a llama-index Document object."""
    from llama_index.core import Document

    return Document(text=text)
