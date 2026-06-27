from __future__ import annotations

import pytest

from limitless_llm.backends.mock_backend import MockBackend
from limitless_llm.core.rate_limiter import TPMRateLimiter
from limitless_llm.exceptions import ContextLengthExceededError
from limitless_llm.phases.verifier import VerificationPass


def _make_verifier(mock_backend: MockBackend, model: str = "groq/llama-3.3-70b-versatile") -> VerificationPass:
    return VerificationPass(
        backend=mock_backend,
        rate_limiter=TPMRateLimiter(tpm_limit=None),
        model=model,
        max_output_tokens=200,
    )


async def test_verify_returns_report(mock_backend: MockBackend) -> None:
    verifier = _make_verifier(mock_backend)
    mock_backend.set_responses(["All facts confirmed. No contradictions found."])
    report = await verifier.verify("merged output", "ledger content")
    assert "confirmed" in report


async def test_verify_raises_context_length_exceeded_when_too_large(
    mock_backend: MockBackend,
) -> None:
    # Use a tiny-context model backed by a custom registry entry.
    import limitless_llm.core.token_counter as tc
    import datetime
    from limitless_llm.core.token_counter import _RegistryEntry

    tc._REGISTRY["test/tiny-model"] = _RegistryEntry(
        context_window=10,
        tpm_limit=None,
        last_verified=datetime.date.today(),
    )
    try:
        verifier = VerificationPass(
            backend=mock_backend,
            rate_limiter=TPMRateLimiter(tpm_limit=None),
            model="test/tiny-model",
            max_output_tokens=5,
        )
        with pytest.raises(ContextLengthExceededError):
            await verifier.verify("a very long merged output " * 100, "ledger " * 100)
    finally:
        del tc._REGISTRY["test/tiny-model"]


async def test_missing_rate_limiter_raises() -> None:
    with pytest.raises(ValueError, match="rate_limiter"):
        VerificationPass(backend=None, rate_limiter=None, model="x", max_output_tokens=100)  # type: ignore[arg-type]
