from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field


class ErrorDetail(BaseModel):
    """Structured error information returned alongside a failed pipeline run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    error_type: str = Field(..., description="Exception class name")
    message: str = Field(..., description="Human-readable error message")
    phase: str | None = Field(default=None, description="Pipeline phase where the error occurred")
    chunk_index: int | None = Field(default=None, description="Chunk index if applicable")
