from __future__ import annotations

import logging as _logging
from typing import cast

import structlog
import structlog.contextvars

from limitless_llm.backends.base import LLMBackend
from limitless_llm.backends.litellm_backend import LiteLLMBackend
from limitless_llm.core.rate_limiter import TPMRateLimiter
from limitless_llm.core.token_counter import (
    SYSTEM_OVERHEAD,
    TAIL_TOKENS,
    TokenCounter,
    get_context_window,
    get_tpm_limit,
)
from limitless_llm.exceptions import ChunkTooLargeError, StartupValidationError
from limitless_llm.models.config import PipelineConfig
from limitless_llm.models.requests import LLMRequest
from limitless_llm.phases.chunker import StructuralSplitter, build_tail
from limitless_llm.phases.compressor import Compressor
from limitless_llm.phases.merger import HierarchicalMerge
from limitless_llm.phases.verifier import VerificationPass
from limitless_llm.types import TokenCount

log = structlog.get_logger(__name__)

# Oldest ledger tokens are dropped once this cap is exceeded. Sized to consume at most
# 25% of the smallest throttled-provider TPM window (6,000 TPM after Groq correction),
# leaving room for system prompt, chunk content, and max_output_tokens in the same call.
# The compressor (spec §6.2) preserves long-term facts; the ledger provides recent context.
_LEDGER_CAP: TokenCount = 1500


def _validate_startup(
    model: str,
    baseline_chunk_size: TokenCount,
    max_output_tokens: TokenCount,
    context_window: TokenCount,
) -> None:
    total = baseline_chunk_size + max_output_tokens + TAIL_TOKENS + SYSTEM_OVERHEAD
    if total > context_window:
        raise StartupValidationError(
            baseline_chunk_size=baseline_chunk_size,
            max_output_tokens=max_output_tokens,
            tail_tokens=TAIL_TOKENS,
            system_overhead=SYSTEM_OVERHEAD,
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
        - SYSTEM_OVERHEAD
        - TAIL_TOKENS
        - current_summary_tokens
    )


def _append_ledger(ledger: str, addition: str) -> str:
    """Append addition to ledger, trimming oldest tokens when the result exceeds _LEDGER_CAP.

    Retains the most recent content. Uses incremental per-word token counting to avoid
    O(N^2) re-tokenization of the full string on each iteration.
    """
    merged = f"{ledger}\n{addition}".strip() if ledger else addition
    if TokenCounter.count(merged) <= _LEDGER_CAP:
        return merged
    words = merged.split()
    kept: list[str] = []
    running: TokenCount = 0
    for word in reversed(words):
        cost = TokenCounter.count(word)
        if running + cost > _LEDGER_CAP:
            break
        kept.append(word)
        running += cost
    # If no words fit (a single word exceeds the cap), keep the last word to preserve
    # some context rather than returning an empty ledger.
    return " ".join(reversed(kept)) if kept else words[-1]


def _rechunk_to_fit(text: str, target_chunk_tokens: TokenCount) -> list[str]:
    """Split text into sub-chunks each fitting within target_chunk_tokens.

    Halves the chunk size on each attempt until all sub-chunks fit or the 200-token
    floor is reached.

    Args:
        text: Chunk text whose prompt would exceed the TPM window.
        target_chunk_tokens: Maximum tokens allowed per sub-chunk.

    Returns:
        List of sub-chunk strings, each within target_chunk_tokens.

    Raises:
        ChunkTooLargeError: If text cannot be split to fit even at the 200-token floor.
    """
    size = max(200, target_chunk_tokens)
    while size >= 200:
        subs = StructuralSplitter(chunk_size=size).split(text)
        if all(TokenCounter.count(s) <= target_chunk_tokens for s in subs):
            return subs
        size //= 2
    raise ChunkTooLargeError(
        text_tokens=TokenCounter.count(text), target_tokens=target_chunk_tokens
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
    _logging.getLogger("limitless_llm").setLevel(
        _logging.DEBUG if config.verbose else _logging.ERROR
    )

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

        # Compute dynamic context-window budget before this call (spec §3.1).
        current_ledger_tokens = TokenCounter.count(ledger)
        current_summary_tokens = compressor.current_summary_tokens
        available = _compute_available_chunk_tokens(
            context_window, current_ledger_tokens, max_output_tokens, current_summary_tokens
        )

        if available < max_output_tokens:
            raise RuntimeError(
                f"Available chunk token budget ({available}) fell below minimum viable size "
                f"({max_output_tokens}) on chunk {idx}. "
                f"Reduce max_output_tokens or system_prompt length."
            )

        # Compute summary prefix once per chunk (shared across all sub-chunks).
        summary_prefix = (
            f"PRIOR CONTEXT SUMMARY:\n{compressor.current_summary}\n\n"
            if compressor.current_summary
            else ""
        )

        # Check if the full prompt would exceed the TPM window and split if needed.
        # Measuring actual prompt overhead (including variable system_prompt) rather than
        # using fixed constants ensures the check is accurate for large review-mode prompts.
        tpm_limit = rate_limiter.tpm_limit
        sub_chunks: list[str]
        if tpm_limit is not None:
            empty_user_prompt = config.chunk_prompt_template.format(
                compressed_summary=summary_prefix,
                tail="",
                chunk="",
                ledger=ledger or "(empty)",
            )
            prompt_overhead = (
                TokenCounter.count(config.system_prompt)
                + TokenCounter.count(empty_user_prompt)
                + TAIL_TOKENS
            )
            tpm_available_for_chunk = tpm_limit - prompt_overhead - max_output_tokens
            if TokenCounter.count(chunk_text) > tpm_available_for_chunk:
                sub_chunks = _rechunk_to_fit(chunk_text, max(200, tpm_available_for_chunk))
                log.info(
                    "chunk_rechunked",
                    chunk_index=idx,
                    sub_chunk_count=len(sub_chunks),
                    tpm_available=tpm_available_for_chunk,
                )
            else:
                sub_chunks = [chunk_text]
        else:
            sub_chunks = [chunk_text]

        chunk_output = ""
        for sub_idx, sub_text in enumerate(sub_chunks):
            tail_src = previous_chunk_text if sub_idx == 0 else sub_chunks[sub_idx - 1]
            tail = build_tail(tail_src, TAIL_TOKENS)
            tail_prefix = f"PRIOR SECTION TAIL:\n{tail}\n\n" if tail else ""

            user_prompt = config.chunk_prompt_template.format(
                compressed_summary=summary_prefix,
                tail=tail_prefix,
                chunk=sub_text,
                ledger=ledger or "(empty)",
            )
            request = LLMRequest(
                model=model_name,
                system_prompt=config.system_prompt,
                user_prompt=user_prompt,
                max_tokens=max_output_tokens,
                phase="chunk",
            )
            log.info(
                "chunk_call",
                chunk_index=idx,
                sub_chunk_index=sub_idx,
                chunk_tokens=TokenCounter.count(sub_text),
            )
            response = await _backend.complete(request)
            chunk_output = (
                (chunk_output + "\n" + response.content).strip()
                if chunk_output
                else response.content
            )

        chunk_outputs.append(chunk_output)

        # Cap ledger to prevent unbounded growth - drops oldest content when exceeded.
        ledger = _append_ledger(ledger, chunk_output)

        # Update compressed summary for the next chunk.
        structlog.contextvars.bind_contextvars(phase="compression")
        await compressor.update(chunk_output, idx)

        previous_chunk_text = sub_chunks[-1]

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
