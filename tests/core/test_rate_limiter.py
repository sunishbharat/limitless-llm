from __future__ import annotations

import asyncio
import time

import pytest
from freezegun import freeze_time

from limitless_llm.core.rate_limiter import TPMRateLimiter, _WINDOW_SECONDS


async def test_noop_when_tpm_limit_is_none() -> None:
    limiter = TPMRateLimiter(tpm_limit=None)
    rid = await limiter.wait_if_needed(99_999)
    limiter.record(99_999, rid)
    # No exception = no-op confirmed


async def test_reserve_and_record_within_budget() -> None:
    limiter = TPMRateLimiter(tpm_limit=10_000)
    rid = await limiter.wait_if_needed(5_000)
    assert len(limiter._log) == 1
    limiter.record(4_800, rid)
    assert limiter._log[0].tokens == 4_800
    assert not limiter._log[0].is_reservation


async def test_multiple_reservations_accumulate() -> None:
    limiter = TPMRateLimiter(tpm_limit=10_000)
    rid1 = await limiter.wait_if_needed(4_000)
    rid2 = await limiter.wait_if_needed(4_000)
    assert limiter._window_sum() == 8_000
    limiter.record(3_900, rid1)
    limiter.record(3_900, rid2)
    assert limiter._window_sum() == 7_800


async def test_reservation_replaced_by_uuid_not_timestamp() -> None:
    """record() must find the entry by reservation_id, not by position."""
    limiter = TPMRateLimiter(tpm_limit=20_000)
    rid1 = await limiter.wait_if_needed(1_000)
    rid2 = await limiter.wait_if_needed(2_000)
    limiter.record(900, rid1)
    # rid2 entry must still be at 2000, not overwritten
    remaining = [e for e in limiter._log if e.reservation_id == rid2]
    assert len(remaining) == 1
    assert remaining[0].tokens == 2_000


async def test_unknown_reservation_id_does_not_raise() -> None:
    limiter = TPMRateLimiter(tpm_limit=10_000)
    # Should log a warning but not raise
    limiter.record(500, "nonexistent-uuid")


async def test_expired_entries_evicted() -> None:
    limiter = TPMRateLimiter(tpm_limit=10_000)
    rid = await limiter.wait_if_needed(5_000)
    limiter.record(5_000, rid)
    # Manually backdate the entry timestamp to simulate window expiry
    limiter._log[0] = limiter._log[0].__class__(
        reservation_id=limiter._log[0].reservation_id,
        tokens=limiter._log[0].tokens,
        timestamp=time.monotonic() - _WINDOW_SECONDS - 1,
        is_reservation=False,
    )
    async with limiter._lock:
        limiter._evict_expired()
    assert len(limiter._log) == 0


async def test_budget_exceeded_causes_sleep(monkeypatch: pytest.MonkeyPatch) -> None:
    """When window is full, wait_if_needed must sleep until the oldest entry expires."""
    slept: list[float] = []

    async def fake_sleep(seconds: float) -> None:
        slept.append(seconds)
        # Simulate time passing by backdating all log entries.
        for i, e in enumerate(limiter._log):
            limiter._log[i] = e.__class__(
                reservation_id=e.reservation_id,
                tokens=e.tokens,
                timestamp=e.timestamp - _WINDOW_SECONDS - 1,
                is_reservation=e.is_reservation,
            )

    monkeypatch.setattr(asyncio, "sleep", fake_sleep)

    limiter = TPMRateLimiter(tpm_limit=5_000)
    rid1 = await limiter.wait_if_needed(5_000)
    # Window is now full; next call should trigger a sleep
    rid2 = await limiter.wait_if_needed(1_000)
    assert len(slept) >= 1
    limiter.record(5_000, rid1)
    limiter.record(900, rid2)
