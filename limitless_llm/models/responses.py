from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class UsageStats(BaseModel):
    """Token usage reported by the LLM API."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    prompt_tokens: int = Field(..., description="Input tokens consumed")
    completion_tokens: int = Field(..., description="Output tokens generated")
    total_tokens: int = Field(..., description="Total tokens (prompt + completion)")


class LLMResponse(BaseModel):
    """A complete (non-streaming) response from an LLM backend."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    content: str = Field(..., description="The model's text output")
    usage: UsageStats = Field(..., description="Token usage for this call")
    model: str = Field(..., description="Model that produced the response")


class LLMChunk(BaseModel):
    """A single token chunk from a streaming response."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    delta: str = Field(..., description="Incremental text fragment")
    finish_reason: str | None = Field(
        default=None,
        description="Set on the final chunk; None for intermediate chunks",
    )
