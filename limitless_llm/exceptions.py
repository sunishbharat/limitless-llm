from __future__ import annotations


class LimitlessLLMError(Exception):
    """Base class for all limitless-llm errors."""


class TPMBudgetExceededError(LimitlessLLMError):
    """Raised when the TPM budget is exhausted after all retries."""

    def __init__(
        self,
        model: str,
        estimated_tokens: int,
        rolling_window_tokens: int,
        tpm_limit: int,
    ) -> None:
        self.model = model
        self.estimated_tokens = estimated_tokens
        self.rolling_window_tokens = rolling_window_tokens
        self.tpm_limit = tpm_limit
        super().__init__(
            f"TPM budget exceeded for model '{model}': estimated {estimated_tokens} tokens, "
            f"rolling window at {rolling_window_tokens}/{tpm_limit} TPM"
        )


class ContextLengthExceededError(LimitlessLLMError):
    """Raised when an API call exceeds the model's context window."""

    def __init__(
        self,
        model: str,
        input_tokens: int,
        output_cap: int,
        context_window: int,
        phase: str,
        chunk_index: int | None = None,
    ) -> None:
        self.model = model
        self.input_tokens = input_tokens
        self.output_cap = output_cap
        self.context_window = context_window
        self.phase = phase
        self.chunk_index = chunk_index
        location = f" (chunk {chunk_index})" if chunk_index is not None else ""
        super().__init__(
            f"Context length exceeded in phase '{phase}'{location}: "
            f"{input_tokens} input + {output_cap} output cap = {input_tokens + output_cap} "
            f"tokens, but context window is {context_window} for model '{model}'"
        )


class ChunkTooLargeError(LimitlessLLMError):
    """Raised when a chunk cannot be split to fit within the per-call TPM budget."""

    def __init__(self, text_tokens: int, target_tokens: int) -> None:
        self.text_tokens = text_tokens
        self.target_tokens = target_tokens
        super().__init__(
            f"Chunk of {text_tokens} tokens cannot be split to fit within "
            f"{target_tokens} tokens per call even at the 200-token minimum sub-chunk size. "
            f"Reduce system_prompt length or use a model with a higher TPM limit."
        )


class MergeInputTooLargeError(LimitlessLLMError):
    """Raised when a single merge input alone exceeds the usable context budget."""

    def __init__(
        self,
        model: str,
        left_tokens: int,
        right_tokens: int,
        context_window: int,
        max_output_tokens: int,
    ) -> None:
        self.model = model
        self.left_tokens = left_tokens
        self.right_tokens = right_tokens
        self.context_window = context_window
        self.max_output_tokens = max_output_tokens
        super().__init__(
            f"Merge inputs too large for model '{model}': "
            f"left={left_tokens}, right={right_tokens} tokens; "
            f"context_window={context_window}, max_output={max_output_tokens}"
        )


class ModelNotFoundError(LimitlessLLMError):
    """Raised when a model is not found in the registry and no fallback applies."""


class StartupValidationError(LimitlessLLMError):
    """Raised when the startup invariant check fails before any LLM call."""

    def __init__(
        self,
        baseline_chunk_size: int,
        max_output_tokens: int,
        tail_tokens: int,
        system_overhead: int,
        context_window: int,
        model: str,
    ) -> None:
        total = baseline_chunk_size + max_output_tokens + tail_tokens + system_overhead
        super().__init__(
            f"Startup validation failed for model '{model}': "
            f"baseline_chunk ({baseline_chunk_size}) + max_output ({max_output_tokens}) "
            f"+ tail ({tail_tokens}) + overhead ({system_overhead}) = {total} "
            f"> context_window ({context_window}). "
            f"Reduce max_output_tokens or use a model with a larger context window."
        )
