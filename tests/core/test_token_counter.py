from __future__ import annotations

import datetime

import pytest

from limitless_llm.core.token_counter import (
    TokenCounter,
    _FALLBACK_CONTEXT_WINDOW,
    _FALLBACK_TPM_LIMIT,
    _STALENESS_DAYS,
    audit_registry,
    get_context_window,
    get_tpm_limit,
)


def test_get_context_window_known_model() -> None:
    assert get_context_window("groq/llama-3.3-70b-versatile") == 128_000


def test_get_context_window_unknown_model() -> None:
    assert get_context_window("unknown/model-xyz") == _FALLBACK_CONTEXT_WINDOW


def test_get_tpm_limit_known_model() -> None:
    assert get_tpm_limit("groq/llama-3.3-70b-versatile") == 12_000


def test_get_tpm_limit_unlimited_model() -> None:
    assert get_tpm_limit("openai/gpt-4o") is None


def test_get_tpm_limit_unknown_model() -> None:
    assert get_tpm_limit("unknown/model-xyz") == _FALLBACK_TPM_LIMIT


def test_token_counter_empty_string() -> None:
    assert TokenCounter.count("") == 0


def test_token_counter_nonempty() -> None:
    count = TokenCounter.count("hello world")
    assert count > 0


def test_token_counter_longer_text_has_more_tokens() -> None:
    short = TokenCounter.count("hello")
    long = TokenCounter.count("hello world this is a longer sentence with many words")
    assert long > short


def test_audit_registry_returns_all_models() -> None:
    results = audit_registry()
    models = {r["model"] for r in results}
    assert "groq/llama-3.3-70b-versatile" in models
    assert "openai/gpt-4o" in models


def test_audit_registry_has_age_and_stale_fields() -> None:
    results = audit_registry()
    for r in results:
        assert "age_days" in r
        assert "stale" in r
        assert isinstance(r["age_days"], int)
        assert isinstance(r["stale"], bool)


def test_staleness_warning_emitted(monkeypatch: pytest.MonkeyPatch) -> None:
    """Verify that a warning is emitted for a stale registry entry."""
    import limitless_llm.core.token_counter as tc

    stale_date = datetime.date.today() - datetime.timedelta(days=_STALENESS_DAYS + 10)
    from limitless_llm.core.token_counter import _RegistryEntry

    original = tc._REGISTRY.copy()
    tc._REGISTRY["groq/llama-3.3-70b-versatile"] = _RegistryEntry(
        context_window=128_000,
        tpm_limit=12_000,
        last_verified=stale_date,
    )
    try:
        with pytest.warns(UserWarning, match="last verified"):
            get_context_window("groq/llama-3.3-70b-versatile")
    finally:
        tc._REGISTRY["groq/llama-3.3-70b-versatile"] = original["groq/llama-3.3-70b-versatile"]
