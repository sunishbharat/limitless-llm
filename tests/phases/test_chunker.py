from __future__ import annotations

import pytest

from limitless_llm.core.token_counter import TokenCounter
from limitless_llm.phases.chunker import StructuralSplitter, _tail, build_tail


def test_tail_returns_suffix_within_budget() -> None:
    words = ["alpha", "beta", "gamma", "delta", "epsilon"]
    result = _tail(words, token_limit=10)
    assert len(result) > 0
    assert TokenCounter.count(" ".join(result)) <= 10


def test_tail_returns_single_oversized_word() -> None:
    # A very long token-dense string that exceeds the budget by itself.
    # _tail logs a structlog warning and still returns the word rather than an empty list.
    long_word = "supercalifragilistic" * 20
    result = _tail([long_word], token_limit=1)
    assert result == [long_word]


def test_tail_empty_words_returns_empty() -> None:
    assert _tail([], token_limit=50) == []


def test_build_tail_empty_previous() -> None:
    assert build_tail("") == ""


def test_build_tail_respects_token_limit() -> None:
    text = " ".join(["word"] * 500)
    tail = build_tail(text, token_limit=20)
    assert TokenCounter.count(tail) <= 20


def test_structural_splitter_produces_chunks_within_limit() -> None:
    # Build a document longer than one chunk.
    sentence = "This is a test sentence. " * 200
    splitter = StructuralSplitter(chunk_size=100)
    chunks = splitter.split(sentence)
    assert len(chunks) > 1
    for chunk in chunks:
        # Allow a small tolerance for the splitter's internal decisions.
        assert TokenCounter.count(chunk) <= 150


def test_structural_splitter_single_chunk_small_doc() -> None:
    text = "Short document."
    splitter = StructuralSplitter(chunk_size=500)
    chunks = splitter.split(text)
    assert len(chunks) == 1
    assert chunks[0] == text
