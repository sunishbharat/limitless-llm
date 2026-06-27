from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ModelConfig(BaseModel):
    """Configuration for a specific LLM model."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: str = Field(
        ...,
        description="LiteLLM model identifier (e.g. 'groq/llama-3.3-70b-versatile')",
    )
    max_output_tokens: int = Field(
        default=1500,
        description="Maximum tokens to request in model output",
    )
    baseline_chunk_size: int = Field(
        default=6000,
        description="Starting chunk size in tokens; shrinks dynamically as ledger grows",
    )

    @field_validator("max_output_tokens", "baseline_chunk_size")
    @classmethod
    def must_be_positive(cls, v: int) -> int:
        if v <= 0:
            raise ValueError("must be positive")
        return v


class PipelineConfig(BaseModel):
    """Top-level configuration for a pipeline run."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    model: ModelConfig = Field(..., description="Model configuration")
    input_text: str = Field(..., description="Document text to process")
    system_prompt: str = Field(
        default="You are a helpful assistant.",
        description="System prompt / persona sent with every chunk call",
    )
    chunk_prompt_template: str = Field(
        default=(
            "Process the following document section and extract the key information.\n\n"
            "{compressed_summary}"
            "{tail}"
            "DOCUMENT SECTION:\n{chunk}"
            "\n\nLEDGER SO FAR:\n{ledger}"
        ),
        description="Chunk template; placeholders: {compressed_summary}, {tail}, {chunk}, {ledger}",
    )
