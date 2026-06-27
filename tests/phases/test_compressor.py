from __future__ import annotations

import pytest

from limitless_llm.backends.mock_backend import MockBackend
from limitless_llm.core.rate_limiter import TPMRateLimiter
from limitless_llm.exceptions import TPMBudgetExceededError
from limitless_llm.phases.compressor import Compressor, SUMMARY_TOKEN_CAP
from limitless_llm.core.token_counter import TokenCounter


@pytest.fixture
def compressor(mock_backend: MockBackend) -> Compressor:
    return Compressor(
        backend=mock_backend,
        rate_limiter=TPMRateLimiter(tpm_limit=None),
        model="groq/llama-3.3-70b-versatile",
    )


async def test_initial_summary_is_empty(compressor: Compressor) -> None:
    assert compressor.current_summary == ""
    assert compressor.current_summary_tokens == 0


async def test_update_sets_summary(compressor: Compressor, mock_backend: MockBackend) -> None:
    mock_backend.set_responses(["Entity A defined as X. Constraint: deadline March 15."])
    await compressor.update("chunk output text", chunk_index=0)
    assert compressor.current_summary != ""
    assert TokenCounter.count(compressor.current_summary) <= SUMMARY_TOKEN_CAP


async def test_update_truncates_oversized_response(
    compressor: Compressor, mock_backend: MockBackend
) -> None:
    # Generate a response clearly over SUMMARY_TOKEN_CAP (300 tokens).
    # "This is a sentence. " is ~5 tokens; 100 repetitions = ~500 tokens.
    long_response = "This is a sentence. " * 100
    assert TokenCounter.count(long_response) > SUMMARY_TOKEN_CAP
    mock_backend.set_responses([long_response])
    await compressor.update("anything", chunk_index=0)
    assert TokenCounter.count(compressor.current_summary) <= SUMMARY_TOKEN_CAP


async def test_update_gracefully_degrades_on_tpm_exceeded(
    mock_backend: MockBackend,
) -> None:
    """TPMBudgetExceededError on compression must keep prior summary and not raise."""

    class FailingBackend:
        async def complete(self, request: object) -> None:  # type: ignore[override]
            raise TPMBudgetExceededError(
                model="groq/llama-3.3-70b-versatile",
                estimated_tokens=100,
                rolling_window_tokens=12_000,
                tpm_limit=12_000,
            )

        async def stream(self, request: object) -> None:  # type: ignore[override]
            raise NotImplementedError

    limiter = TPMRateLimiter(tpm_limit=None)
    comp = Compressor(backend=FailingBackend(), rate_limiter=limiter, model="groq/llama-3.3-70b-versatile")  # type: ignore[arg-type]
    comp._current_summary = "prior summary"
    await comp.update("new chunk output", chunk_index=1)
    # Summary must remain unchanged after the failure.
    assert comp.current_summary == "prior summary"


async def test_context_length_error_propagates(mock_backend: MockBackend) -> None:
    """ContextLengthExceededError on a compression call must propagate (not swallow)."""
    from limitless_llm.exceptions import ContextLengthExceededError

    class ContextErrorBackend:
        async def complete(self, request: object) -> None:  # type: ignore[override]
            raise ContextLengthExceededError(
                model="groq/llama-3.3-70b-versatile",
                input_tokens=200,
                output_cap=100,
                context_window=8192,
                phase="compression",
            )

        async def stream(self, request: object) -> None:  # type: ignore[override]
            raise NotImplementedError

    limiter = TPMRateLimiter(tpm_limit=None)
    comp = Compressor(backend=ContextErrorBackend(), rate_limiter=limiter, model="groq/llama-3.3-70b-versatile")  # type: ignore[arg-type]
    with pytest.raises(ContextLengthExceededError):
        await comp.update("some output", chunk_index=0)


async def test_missing_rate_limiter_raises() -> None:
    with pytest.raises(ValueError, match="rate_limiter"):
        Compressor(backend=None, rate_limiter=None, model="x")  # type: ignore[arg-type]
