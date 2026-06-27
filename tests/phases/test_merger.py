from __future__ import annotations

import pytest

from limitless_llm.backends.mock_backend import MockBackend
from limitless_llm.core.rate_limiter import TPMRateLimiter
from limitless_llm.exceptions import MergeInputTooLargeError
from limitless_llm.phases.merger import HierarchicalMerge, _extract_conflicts


def _make_merger(mock_backend: MockBackend) -> HierarchicalMerge:
    return HierarchicalMerge(
        backend=mock_backend,
        rate_limiter=TPMRateLimiter(tpm_limit=None),
        model="groq/llama-3.3-70b-versatile",
        max_output_tokens=500,
    )


async def test_single_chunk_returns_as_is(mock_backend: MockBackend) -> None:
    merger = _make_merger(mock_backend)
    result = await merger.merge(["only one chunk"])
    assert result == "only one chunk"


async def test_empty_list_returns_empty(mock_backend: MockBackend) -> None:
    merger = _make_merger(mock_backend)
    result = await merger.merge([])
    assert result == ""


async def test_two_chunks_calls_backend_once(mock_backend: MockBackend) -> None:
    merger = _make_merger(mock_backend)
    mock_backend.set_responses(["merged result"])
    result = await merger.merge(["left text", "right text"])
    assert result == "merged result"
    assert len(mock_backend.calls) == 1


async def test_four_chunks_balanced_tree(mock_backend: MockBackend) -> None:
    # N=4: two pair merges at level 0, one final merge at level 1 = 3 calls total.
    merger = _make_merger(mock_backend)
    mock_backend.set_responses(["m12", "m34", "final"])
    result = await merger.merge(["c1", "c2", "c3", "c4"])
    assert result == "final"
    assert len(mock_backend.calls) == 3


async def test_five_chunks_carry_forward(mock_backend: MockBackend) -> None:
    # N=5 tree:
    #   level 0: merge(c1,c2)->m12, merge(c3,c4)->m34, c5 carried  => ["m12","m34","c5"]
    #   level 1: merge(m12,m34)->m1234, c5 carried                  => ["m1234","c5"]
    #   level 2: merge(m1234,c5)->final                              => ["final"]
    # Total: 4 backend calls
    merger = _make_merger(mock_backend)
    mock_backend.set_responses(["m12", "m34", "m1234", "final"])
    result = await merger.merge(["c1", "c2", "c3", "c4", "c5"])
    assert result == "final"
    assert len(mock_backend.calls) == 4


async def test_conflict_markers_appended_to_output(mock_backend: MockBackend) -> None:
    merger = _make_merger(mock_backend)
    merged_with_conflict = (
        'Facts here. [CONFLICT: left says "March 15", right says "April 1" '
        '- preserved for human review]'
    )
    mock_backend.set_responses([merged_with_conflict])
    result = await merger.merge(["left", "right"])
    assert "## Conflicts Requiring Human Review" in result
    assert "March 15" in result


def test_extract_conflicts_finds_markers() -> None:
    text = (
        'Some text [CONFLICT: left says "A", right says "B" - preserved for human review] '
        'more text [CONFLICT: left says "X", right says "Y" - preserved for human review]'
    )
    conflicts = _extract_conflicts(text)
    assert len(conflicts) == 2


def test_extract_conflicts_empty_text() -> None:
    assert _extract_conflicts("no conflicts here") == []


async def test_overflow_fallback_truncates_larger_input(mock_backend: MockBackend) -> None:
    """When merge inputs exceed the usable budget, the larger one is truncated."""
    import datetime
    import limitless_llm.core.token_counter as tc
    from limitless_llm.core.token_counter import _RegistryEntry
    from limitless_llm.phases.merger import _MERGE_PROMPT_OVERHEAD, _SYSTEM_OVERHEAD

    # budget = context_window - max_output_tokens - _MERGE_PROMPT_OVERHEAD - _SYSTEM_OVERHEAD
    # We need: left_tokens > budget, right_tokens <= budget, left + right > budget (overflow fires).
    # Fixed overhead = 300 + 500 = 800. Use context_window=1000, max_output=50 -> budget=150.
    # left ~ 200 tokens (> 150), right ~ 10 tokens (< 150), combined ~ 210 > 150.
    context_window = 1000
    max_output = 50
    tc._REGISTRY["test/overflow-merge"] = _RegistryEntry(
        context_window=context_window, tpm_limit=None, last_verified=datetime.date.today()
    )
    try:
        merger = HierarchicalMerge(
            backend=mock_backend,
            rate_limiter=TPMRateLimiter(tpm_limit=None),
            model="test/overflow-merge",
            max_output_tokens=max_output,
        )
        left = "word " * 200   # ~200 tokens, exceeds budget of 150
        right = "other " * 10  # ~10 tokens, fits within budget
        mock_backend.set_responses(["truncated merge result"])
        result = await merger.merge([left, right])
        assert "truncated merge result" in result
    finally:
        del tc._REGISTRY["test/overflow-merge"]


async def test_merge_input_too_large_raises(mock_backend: MockBackend) -> None:
    """MergeInputTooLargeError raised when both inputs each exceed the usable budget."""
    import datetime
    import limitless_llm.core.token_counter as tc
    from limitless_llm.core.token_counter import _RegistryEntry
    from limitless_llm.exceptions import MergeInputTooLargeError

    # budget = 1000 - 50 - 300 - 500 = 150; both inputs are ~200 tokens each.
    tc._REGISTRY["test/micro-merge"] = _RegistryEntry(
        context_window=1000, tpm_limit=None, last_verified=datetime.date.today()
    )
    try:
        merger = HierarchicalMerge(
            backend=mock_backend,
            rate_limiter=TPMRateLimiter(tpm_limit=None),
            model="test/micro-merge",
            max_output_tokens=50,
        )
        left = "alpha " * 200   # ~200 tokens > budget of 150
        right = "beta " * 200   # ~200 tokens > budget of 150
        with pytest.raises(MergeInputTooLargeError):
            await merger.merge([left, right])
    finally:
        del tc._REGISTRY["test/micro-merge"]


async def test_missing_rate_limiter_raises() -> None:
    with pytest.raises(ValueError, match="rate_limiter"):
        HierarchicalMerge(backend=None, rate_limiter=None, model="x", max_output_tokens=100)  # type: ignore[arg-type]
