from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class LLMRequest(BaseModel):
    """A single request to an LLM backend."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(..., description="LiteLLM model identifier")
    system_prompt: str = Field(..., description="System-role message content")
    user_prompt: str = Field(..., description="User-role message content")
    max_tokens: int = Field(..., description="Maximum output tokens for this call")
    phase: str | None = Field(
        default=None,
        description="Pipeline phase for error context: chunk, compression, merge, verification",
    )
