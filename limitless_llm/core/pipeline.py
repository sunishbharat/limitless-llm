from __future__ import annotations

from typing import cast

import structlog
import structlog.contextvars

from limitless_llm.backends.base import LLMBackend
from limitless_llm.backends.litellm_backend import LiteLLMBackend
from limitless_llm.core.rate_limiter import TPMRateLimiter
from limitless_llm.core.token_counter import TokenCounter, get_context_window, get_tpm_limit
from limitless_llm.exceptions import StartupValidationError
from limitless_llm.models.config import PipelineConfig
from limitless_llm.models.requests import LLMRequest
from limitless_llm.phases.chunker import StructuralSplitter, build_tail
from limitless_llm.phases.compressor import Compressor
from limitless_llm.phases.merger import HierarchicalMerge
from limitless_llm.phases.verifier import VerificationPass
from limitless_llm.types import TokenCount

log = structlog.get_logger(__name__)

# Token budget constants - see spec §3.1 and §11.
# system_overhead covers both template boilerplate and cl100k_base approximation error
# (5-15% undercount on technical text). Increasing to 750 is the first mitigation step
# if context-length errors are observed in practice.
_SYSTEM_OVERHEAD: TokenCount = 500
_TAIL_TOKENS: TokenCount = 200


def _validate_startup(
    model: str,
    baseline_chunk_size: TokenCount,
    max_output_tokens: TokenCount,
    context_window: TokenCount,
) -> None:
    total = baseline_chunk_size + max_output_tokens + _TAIL_TOKENS + _SYSTEM_OVERHEAD
    if total > context_window:
        raise StartupValidationError(
            baseline_chunk_size=baseline_chunk_size,
            max_output_tokens=max_output_tokens,
            tail_tokens=_TAIL_TOKENS,
            system_overhead=_SYSTEM_OVERHEAD,
            context_window=context_window,
            model=model,
        )


def _compute_available_chunk_tokens(
    context_window: TokenCount,
    current_ledger_tokens: TokenCount,
    max_output_tokens: TokenCount,
    current_summary_tokens: TokenCount,
) -> TokenCount:
    """Compute available tokens for chunk content given current pipeline state.

    Formula from spec §3.1.
    """
    return (
        context_window
        - current_ledger_tokens
        - max_output_tokens
        - _SYSTEM_OVERHEAD
        - _TAIL_TOKENS
        - current_summary_tokens
    )


async def run_with_chunking_if_needed(
    config: PipelineConfig,
    backend: LLMBackend | None = None,
) -> str:
    """Entry point for the full Phase 1 pipeline.

    Creates a single TPMRateLimiter and threads it through every component.
    Returns the final merged (and verified, if possible) output.

    Args:
        config: Full pipeline configuration.
        backend: Optional pre-built backend (defaults to LiteLLMBackend).

    Returns:
        Final merged output string.
    """
    model_name = config.model.model
    max_output_tokens = config.model.max_output_tokens
    baseline_chunk_size = config.model.baseline_chunk_size
    context_window = get_context_window(model_name)

    _validate_startup(model_name, baseline_chunk_size, max_output_tokens, context_window)

    rate_limiter = TPMRateLimiter(tpm_limit=get_tpm_limit(model_name))

    _backend = cast(
        LLMBackend,
        backend if backend is not None else LiteLLMBackend(rate_limiter=rate_limiter),
    )

    compressor = Compressor(backend=_backend, rate_limiter=rate_limiter, model=model_name)
    merger = HierarchicalMerge(
        backend=_backend,
        rate_limiter=rate_limiter,
        model=model_name,
        max_output_tokens=max_output_tokens,
        include_conflict_summary=config.include_conflict_summary,
    )
    verifier = VerificationPass(
        backend=_backend,
        rate_limiter=rate_limiter,
        model=model_name,
        max_output_tokens=max_output_tokens,
    )

    splitter = StructuralSplitter(chunk_size=baseline_chunk_size)
    chunks = splitter.split(config.input_text)

    log.info("pipeline_start", model=model_name, chunk_count=len(chunks))

    ledger = ""
    chunk_outputs: list[str] = []
    previous_chunk_text = ""

    for idx, chunk_text in enumerate(chunks):
        structlog.contextvars.bind_contextvars(chunk_index=idx, model=model_name, phase="chunk")

        # Compute dynamic budget before this call.
        current_ledger_tokens = TokenCounter.count(ledger)
        current_summary_tokens = compressor.current_summary_tokens
        available = _compute_available_chunk_tokens(
            context_window, current_ledger_tokens, max_output_tokens, current_summary_tokens
        )

        if available < max_output_tokens:
            raise RuntimeError(
                f"Available chunk token budget ({available}) fell below minimum viable size "
                f"({max_output_tokens}) on chunk {idx}. "
                f"Reduce max_output_tokens or enable ledger pruning (Phase 2)."
            )

        tail = build_tail(previous_chunk_text, _TAIL_TOKENS)
        summary_prefix = (
            f"PRIOR CONTEXT SUMMARY:\n{compressor.current_summary}\n\n"
            if compressor.current_summary
            else ""
        )
        tail_prefix = f"PRIOR SECTION TAIL:\n{tail}\n\n" if tail else ""

        user_prompt = config.chunk_prompt_template.format(
            compressed_summary=summary_prefix,
            tail=tail_prefix,
            chunk=chunk_text,
            ledger=ledger or "(empty)",
        )

        request = LLMRequest(
            model=model_name,
            system_prompt=config.system_prompt,
            user_prompt=user_prompt,
            max_tokens=max_output_tokens,
            phase="chunk",
        )

        log.info("chunk_call", chunk_index=idx, chunk_tokens=TokenCounter.count(chunk_text))
        response = await _backend.complete(request)
        chunk_output = response.content
        chunk_outputs.append(chunk_output)

        # Update ledger by appending this chunk's output.
        ledger = (ledger + "\n" + chunk_output).strip()

        # Update compressed summary for the next chunk.
        structlog.contextvars.bind_contextvars(phase="compression")
        await compressor.update(chunk_output, idx)

        previous_chunk_text = chunk_text

    structlog.contextvars.clear_contextvars()
    log.info("pipeline_merge_start", chunk_count=len(chunk_outputs))
    structlog.contextvars.bind_contextvars(phase="merge", model=model_name)

    merged = await merger.merge(chunk_outputs)

    structlog.contextvars.bind_contextvars(phase="verification")
    # Read include_verification_report directly from config here rather than forwarding
    # it to the VerificationPass constructor: the orchestrator owns the call-vs-skip
    # decision so the TPM budget is preserved *before* any work is done. (The
    # complementary include_conflict_summary flag is forwarded to HierarchicalMerge
    # because the conflict summary is a string scan appended after the merge tree -
    # no LLM call, no TPM cost - and the merge component already needs to know the
    # decision when it runs.)
    if config.include_verification_report:
        try:
            verification_report = await verifier.verify(merged, ledger)
            log.info("verification_done")
            final = merged + "\n\n---\n\n## Verification Report\n\n" + verification_report
        except Exception as exc:
            log.error("verification_failed", error=str(exc))
            final = merged
    else:
        log.info("verification_skipped")
        final = merged

    structlog.contextvars.clear_contextvars()
    log.info("pipeline_complete")
    return final


class PipelineFactory:
    """Assembles and returns a configured pipeline runner."""

    @staticmethod
    def build(config: PipelineConfig, backend: LLMBackend | None = None) -> PipelineRunner:
        """Build a PipelineRunner for the given config.

        Args:
            config: Pipeline configuration.
            backend: Optional pre-built backend (for testing).

        Returns:
            A PipelineRunner bound to the config and backend.
        """
        return PipelineRunner(config=config, backend=backend)


class PipelineRunner:
    """Thin wrapper that holds config and can be run asynchronously."""

    def __init__(self, config: PipelineConfig, backend: LLMBackend | None = None) -> None:
        self._config = config
        self._backend = backend

    async def run(self) -> str:
        """Execute the pipeline and return the final output."""
        return await run_with_chunking_if_needed(self._config, self._backend)
