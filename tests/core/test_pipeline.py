from __future__ import annotations

import pytest

from limitless_llm.backends.mock_backend import MockBackend
from limitless_llm.core.pipeline import PipelineFactory, _validate_startup
from limitless_llm.core.rate_limiter import TPMRateLimiter
from limitless_llm.exceptions import StartupValidationError
from limitless_llm.models.config import ModelConfig, PipelineConfig


def _small_config(input_text: str, model: str = "groq/llama-3.3-70b-versatile") -> PipelineConfig:
    return PipelineConfig(
        model=ModelConfig(model=model, max_output_tokens=200, baseline_chunk_size=500),
        input_text=input_text,
    )


def test_validate_startup_passes_when_within_budget() -> None:
    # 500 + 200 + 200 + 500 = 1400 << 128000
    _validate_startup("groq/llama-3.3-70b-versatile", 500, 200, 128_000)


def test_validate_startup_fails_when_over_budget() -> None:
    with pytest.raises(StartupValidationError):
        _validate_startup("groq/llama3-70b-8192", 8_000, 1_000, 8_192)


async def test_pipeline_single_chunk(mock_backend: MockBackend) -> None:
    # Short text -> one chunk -> one chunk call + one compression + one merge (passthrough) + verify
    mock_backend.set_responses([
        "extracted output",       # chunk call
        "compressed summary",     # compression call
        "verification passed",    # verification call
    ])
    config = _small_config("This is a short document.")
    runner = PipelineFactory.build(config, backend=mock_backend)
    result = await runner.run()
    assert "extracted output" in result


async def test_pipeline_respects_chunk_count(mock_backend: MockBackend) -> None:
    # Build a document that will produce multiple chunks at a small chunk_size.
    sentence = "Alpha beta gamma delta epsilon. " * 50
    # chunk_size=50 should produce several chunks.
    config = PipelineConfig(
        model=ModelConfig(model="groq/llama-3.3-70b-versatile", max_output_tokens=50, baseline_chunk_size=50),
        input_text=sentence,
    )
    # Supply enough mock responses: N chunk + N compression + (N-1 merge pairs) + 1 verify.
    # We'll just supply a large number of identical responses.
    mock_backend.set_responses(["output"] * 50)
    runner = PipelineFactory.build(config, backend=mock_backend)
    result = await runner.run()
    assert len(result) > 0


async def test_pipeline_includes_verification_report(mock_backend: MockBackend) -> None:
    mock_backend.set_responses([
        "chunk output",
        "summary",
        "all good",
    ])
    config = _small_config("Short document text here.")
    runner = PipelineFactory.build(config, backend=mock_backend)
    result = await runner.run()
    assert "Verification Report" in result


async def test_pipeline_factory_build_returns_runner() -> None:
    config = _small_config("x")
    runner = PipelineFactory.build(config)
    assert hasattr(runner, "run")
