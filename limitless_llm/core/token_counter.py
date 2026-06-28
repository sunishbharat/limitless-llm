from __future__ import annotations

import datetime
import warnings
from dataclasses import dataclass

import tiktoken

from limitless_llm.types import ModelName, TokenCount

# cl100k_base is used as a cross-model approximation. LLaMA 3.x uses a different BPE
# vocabulary; on technical/symbol-heavy text this can undercount by 5-15%. SYSTEM_OVERHEAD
# is deliberately sized to absorb this error. Increase to 750 if context-length errors
# appear in practice on technical content (spec §11).
_ENCODING = tiktoken.get_encoding("cl100k_base")

# Token budget constants shared across pipeline and chunk-sizing helpers (spec §3.1, §11).
# SYSTEM_OVERHEAD covers both template boilerplate and cl100k_base approximation error.
SYSTEM_OVERHEAD: TokenCount = 500
# Literal tail of the prior chunk carried forward for adjacent-sentence continuity (spec §6.1).
TAIL_TOKENS: TokenCount = 200

# Staleness threshold: warn if a registry entry is older than this many days.
_STALENESS_DAYS = 90


@dataclass(frozen=True)
class _RegistryEntry:
    context_window: int
    tpm_limit: int | None  # None = unlimited (paid API or local)
    last_verified: datetime.date


# Model registry. Update `last_verified` whenever provider limits are confirmed.
# Context windows and TPM limits are separate concerns - do not confuse them.
_REGISTRY: dict[str, _RegistryEntry] = {
    # last verified: 2026-06-27 (Groq free plan; TPM confirmed 12,000)
    "groq/llama-3.3-70b-versatile": _RegistryEntry(
        context_window=128_000,
        tpm_limit=12_000,
        last_verified=datetime.date(2026, 6, 27),
    ),
    # last verified: 2026-06-27 (Groq free plan; TPM corrected from 20,000 to 6,000)
    "groq/llama-3.1-8b-instant": _RegistryEntry(
        context_window=128_000,
        tpm_limit=6_000,
        last_verified=datetime.date(2026, 6, 27),
    ),
    # last verified: 2025-06-27 (groq/llama3-70b-8192 no longer listed on Groq free plan
    # as of 2026-06-27 - may be deprecated. Keeping entry with conservative fallback TPM
    # to avoid breaking existing configs; verify before relying on this model.)
    "groq/llama3-70b-8192": _RegistryEntry(
        context_window=8_192,
        tpm_limit=6_000,
        last_verified=datetime.date(2026, 6, 27),
    ),
    # last verified: 2026-06-27
    "openai/gpt-4o": _RegistryEntry(
        context_window=128_000,
        tpm_limit=None,
        last_verified=datetime.date(2026, 6, 27),
    ),
    # last verified: 2026-06-27
    "ollama/llama3.2": _RegistryEntry(
        context_window=32_768,
        tpm_limit=None,
        last_verified=datetime.date(2026, 6, 27),
    ),
    # last verified: 2026-06-27
    "openai/minimax-m3": _RegistryEntry(
        context_window=40_960,
        tpm_limit=None,
        last_verified=datetime.date(2026, 6, 27),
    ),
}

# Fallback values for unknown models. Conservative to avoid over-budgeting.
# TPM fallback corrected from 10,000 (which exceeded Groq free-tier) to 6,000.
_FALLBACK_CONTEXT_WINDOW = 8_192
_FALLBACK_TPM_LIMIT = 6_000


def _check_staleness(model: ModelName, entry: _RegistryEntry) -> None:
    """Emit a warning if the registry entry is older than _STALENESS_DAYS."""
    age = (datetime.date.today() - entry.last_verified).days
    if age > _STALENESS_DAYS:
        warnings.warn(
            f"Model registry entry for '{model}' was last verified on "
            f"{entry.last_verified} ({age} days ago). "
            f"TPM and context window values may be outdated. "
            f"Run with --refresh-limits to fetch current values, "
            f"or update token_counter.py manually.",
            stacklevel=3,
        )


def get_context_window(model: ModelName) -> TokenCount:
    """Return the context window size in tokens for the given model.

    Args:
        model: LiteLLM model identifier.

    Returns:
        Context window size. Falls back to {_FALLBACK_CONTEXT_WINDOW} for unknown models.
    """
    entry = _REGISTRY.get(model)
    if entry is None:
        return _FALLBACK_CONTEXT_WINDOW
    _check_staleness(model, entry)
    return entry.context_window


def get_tpm_limit(model: ModelName) -> TokenCount | None:
    """Return the TPM limit for the given model, or None if unlimited.

    Args:
        model: LiteLLM model identifier.

    Returns:
        Tokens-per-minute limit, or None for local/paid-API models with no limit.
        Falls back to {_FALLBACK_TPM_LIMIT} for unknown models.
    """
    entry = _REGISTRY.get(model)
    if entry is None:
        return _FALLBACK_TPM_LIMIT
    _check_staleness(model, entry)
    return entry.tpm_limit


def audit_registry() -> list[dict[str, object]]:
    """Return a list of all registry entries with their staleness in days.

    Used by the --refresh-limits CLI subcommand.
    """
    today = datetime.date.today()
    results = []
    for model, entry in _REGISTRY.items():
        age = (today - entry.last_verified).days
        results.append(
            {
                "model": model,
                "context_window": entry.context_window,
                "tpm_limit": entry.tpm_limit,
                "last_verified": str(entry.last_verified),
                "age_days": age,
                "stale": age > _STALENESS_DAYS,
            }
        )
    return results


def derive_chunk_size(model_name: ModelName, max_output_tokens: int) -> int:
    """Return a safe baseline chunk size derived from the model's TPM limit.

    For unlimited providers (paid API, local) returns 6000. For throttled providers
    uses half the TPM window minus max_output_tokens and SYSTEM_OVERHEAD, floored at
    200 so StructuralSplitter remains useful.

    The half-window (// 2) reserves capacity for compression calls that share the
    same 60-second rolling window (spec §4.5).

    Args:
        model_name: LiteLLM model identifier.
        max_output_tokens: Per-call output budget in tokens.

    Returns:
        Baseline chunk size in tokens.
    """
    tpm = get_tpm_limit(model_name)
    if tpm is None:
        return 6000
    return max(200, (tpm // 2) - max_output_tokens - SYSTEM_OVERHEAD)


class TokenCounter:
    """Counts tokens using the cl100k_base encoding as a cross-model approximation."""

    @staticmethod
    def count(text: str) -> TokenCount:
        """Return the token count for the given text string.

        Args:
            text: Text to tokenize.

        Returns:
            Number of tokens according to cl100k_base encoding.
        """
        return len(_ENCODING.encode(text))
