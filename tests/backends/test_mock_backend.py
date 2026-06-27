from __future__ import annotations

import pytest

from limitless_llm.backends.mock_backend import MockBackend
from limitless_llm.core.rate_limiter import TPMRateLimiter
from limitless_llm.models.requests import LLMRequest


def _request(content: str = "hello") -> LLMRequest:
    return LLMRequest(
        model="groq/llama-3.3-70b-versatile",
        system_prompt="You are helpful.",
        user_prompt=content,
        max_tokens=100,
    )


async def test_complete_returns_queued_response(mock_backend: MockBackend) -> None:
    mock_backend.set_responses(["The answer is 42."])
    resp = await mock_backend.complete(_request())
    assert resp.content == "The answer is 42."
    assert resp.usage.total_tokens > 0


async def test_complete_records_calls(mock_backend: MockBackend) -> None:
    mock_backend.set_responses(["ok"])
    req = _request("test question")
    await mock_backend.complete(req)
    assert len(mock_backend.calls) == 1
    assert mock_backend.calls[0] is req


async def test_complete_empty_queue_returns_empty_string(mock_backend: MockBackend) -> None:
    resp = await mock_backend.complete(_request())
    assert resp.content == ""


async def test_stream_yields_words(mock_backend: MockBackend) -> None:
    mock_backend.set_responses(["hello world foo"])
    chunks = []
    async for chunk in mock_backend.stream(_request()):
        chunks.append(chunk)
    text = "".join(c.delta for c in chunks)
    assert "hello" in text
    assert "world" in text


async def test_stream_last_chunk_has_finish_reason(mock_backend: MockBackend) -> None:
    mock_backend.set_responses(["one two three"])
    chunks = []
    async for chunk in mock_backend.stream(_request()):
        chunks.append(chunk)
    assert chunks[-1].finish_reason == "stop"


async def test_missing_rate_limiter_raises() -> None:
    with pytest.raises(ValueError, match="rate_limiter"):
        MockBackend(rate_limiter=None)  # type: ignore[arg-type]
