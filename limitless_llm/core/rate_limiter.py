from __future__ import annotations

import asyncio
import time
import uuid
from dataclasses import dataclass, field

import structlog

from limitless_llm.types import ReservationId, TokenCount

log = structlog.get_logger(__name__)

# Minimum sleep duration to prevent spinning on sub-millisecond remainders.
_MIN_SLEEP_SECONDS = 0.1
# Rolling window width in seconds.
_WINDOW_SECONDS = 60.0


@dataclass
class _Entry:
    reservation_id: ReservationId
    tokens: TokenCount
    timestamp: float
    is_reservation: bool = True


@dataclass
class TPMRateLimiter:
    """Async rolling-window TPM rate limiter.

    When tpm_limit is None (local Ollama, paid OpenAI), this is a complete
    no-op - zero lock acquisitions, zero overhead.

    Reservations are keyed by UUID so that a retry sleeping 60+ seconds does
    not accidentally remove the wrong entry when record() is called.
    """

    tpm_limit: TokenCount | None
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)
    _log: list[_Entry] = field(default_factory=list, init=False, repr=False)

    async def wait_if_needed(self, estimated_tokens: TokenCount) -> ReservationId:
        """Check rolling window and sleep until budget is available, then reserve.

        Args:
            estimated_tokens: Expected total tokens for the upcoming call
                (measured input + max_output_tokens).

        Returns:
            A reservation_id UUID string. Pass this to record() after the call.
        """
        if self.tpm_limit is None:
            return str(uuid.uuid4())

        reservation_id = str(uuid.uuid4())

        async with self._lock:
            while True:
                self._evict_expired()
                current = self._window_sum()

                if current + estimated_tokens <= self.tpm_limit:
                    break

                # If the window is empty but estimated_tokens still exceeds the limit,
                # a single call's cost exceeds the TPM ceiling - nothing to wait for.
                # Proceed and let the 429 retry path handle it if the API rejects.
                if not self._log:
                    log.warning(
                        "tpm_single_call_exceeds_limit",
                        estimated_tokens=estimated_tokens,
                        tpm_limit=self.tpm_limit,
                    )
                    break

                # Find the oldest entry; sleep until it exits the window.
                oldest_ts = min(e.timestamp for e in self._log)
                window_exit_at = oldest_ts + _WINDOW_SECONDS
                sleep_for = max(_MIN_SLEEP_SECONDS, window_exit_at - time.monotonic())

                log.debug(
                    "tpm_wait",
                    estimated_tokens=estimated_tokens,
                    window_tokens=current,
                    tpm_limit=self.tpm_limit,
                    sleep_seconds=round(sleep_for, 2),
                )

                self._lock.release()
                try:
                    await asyncio.sleep(sleep_for)
                finally:
                    await self._lock.acquire()

            self._log.append(
                _Entry(
                    reservation_id=reservation_id,
                    tokens=estimated_tokens,
                    timestamp=time.monotonic(),
                    is_reservation=True,
                )
            )
            log.debug(
                "tpm_reserved",
                reservation_id=reservation_id,
                estimated_tokens=estimated_tokens,
                window_tokens=self._window_sum(),
            )

        return reservation_id

    def record(self, actual_tokens: TokenCount, reservation_id: ReservationId) -> None:
        """Replace the reservation with actual token usage after a successful call.

        Must be called after every successful LLM call (including retry successes).
        Do not call with actual_tokens=0 on a failed attempt - call once with the
        successful response's token count.

        Args:
            actual_tokens: Token count from the response's usage field.
            reservation_id: The UUID returned by wait_if_needed.
        """
        if self.tpm_limit is None:
            return

        for i, entry in enumerate(self._log):
            if entry.reservation_id == reservation_id:
                self._log[i] = _Entry(
                    reservation_id=reservation_id,
                    tokens=actual_tokens,
                    timestamp=entry.timestamp,
                    is_reservation=False,
                )
                log.debug(
                    "tpm_recorded",
                    reservation_id=reservation_id,
                    actual_tokens=actual_tokens,
                )
                return

        # Reservation expired from the window during a long retry sleep - log and move on.
        log.warning(
            "tpm_reservation_not_found",
            reservation_id=reservation_id,
            actual_tokens=actual_tokens,
        )

    def window_sum(self) -> TokenCount:
        """Return the current rolling-window token sum for diagnostic use.

        Called outside the lock in the error path; the value is approximate
        since it is not atomic with any pending reservation.
        """
        if self.tpm_limit is None:
            return 0
        self._evict_expired()
        return self._window_sum()

    def _evict_expired(self) -> None:
        """Remove entries older than the rolling window. Must be called under lock."""
        cutoff = time.monotonic() - _WINDOW_SECONDS
        self._log = [e for e in self._log if e.timestamp >= cutoff]

    def _window_sum(self) -> TokenCount:
        """Sum all token entries currently in the rolling window. Must be called under lock."""
        return sum(e.tokens for e in self._log)
