from __future__ import annotations

import pytest

from limitless_llm.backends.mock_backend import MockBackend
from limitless_llm.core.rate_limiter import TPMRateLimiter


@pytest.fixture
def unlimited_limiter() -> TPMRateLimiter:
    """Rate limiter with no TPM cap - no-op for tests that don't test rate limiting."""
    return TPMRateLimiter(tpm_limit=None)


@pytest.fixture
def mock_backend(unlimited_limiter: TPMRateLimiter) -> MockBackend:
    """MockBackend wired to an unlimited limiter."""
    return MockBackend(rate_limiter=unlimited_limiter)
