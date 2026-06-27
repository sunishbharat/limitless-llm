# Contributing to limitless-llm

Thank you for your interest in contributing. This document describes open feature areas, how to set up the project locally, and what to expect from the pull request process.

---

## Open feature areas

These are known gaps in the current implementation where community contributions are most welcome. Each is self-contained enough to be tackled independently.

### 1. Structure-aware document splitter

**Problem:** The current splitter (`StructuralSplitter` in `limitless_llm/phases/chunker.py`) uses sentence boundaries but does not protect code block boundaries. Documents where more than ~20% of content is code blocks will have those blocks bisected at chunk boundaries, which produces degraded output.

**What a contribution here looks like:**
- A splitter that detects fenced code blocks (` ``` `) and treats them as atomic units that cannot be split
- Falls back to sentence-based splitting for prose content between code blocks
- Implements the same `split(text: str) -> list[str]` interface as `StructuralSplitter` so it is a drop-in replacement
- Includes tests covering: pure prose, pure code, mixed documents, code blocks that exceed chunk size

---

### 2. Ledger pruning

**Problem:** The pipeline accumulates all chunk outputs into a running ledger. On very long documents (50+ chunks) this ledger can exceed the model context window and the pipeline aborts. There is currently no mechanism to prune older or lower-priority ledger entries to stay within budget.

**What a contribution here looks like:**
- A pruning strategy that removes or summarises older ledger entries when the ledger approaches the context window limit
- Pruning must preserve named entities, defined terms, constraints, and numeric values - narrative context is the lowest priority to drop
- The pruning step runs inside `limitless_llm/core/pipeline.py` before the per-chunk budget check
- Includes tests that verify the ledger stays within bounds across a simulated long-document run

---

### 3. Redis backend for multi-process rate limiting

**Problem:** The `TPMRateLimiter` (`limitless_llm/core/rate_limiter.py`) uses an in-memory store. Two processes running against the same API key do not share TPM state and can trigger 429 errors even with the limiter active.

**What a contribution here looks like:**
- An optional Redis backend using `limits[redis]` (the `limits` library already supports this)
- Activated via a new `redis_url` parameter on `TPMRateLimiter` - if absent, falls back to in-memory (no behaviour change for existing users)
- Documented in README with a one-liner showing how to enable it
- Integration test that spawns two limiters sharing a Redis instance and verifies combined usage does not exceed the TPM cap

---

### 4. Streaming progress output

**Problem:** Processing a large document takes several minutes with no feedback. Users have no visibility into which chunk is being processed, how many remain, or what the current TPM usage is.

**What a contribution here looks like:**
- An optional `progress_callback` parameter on `PipelineFactory.build()` or `runner.run()`
- Called after each chunk with a progress snapshot (chunk index, total chunks, current TPM window usage)
- The `LLMBackend` protocol already defines a `stream()` method - this work can use it to yield token-level progress if desired
- No change to the default (non-streaming) behaviour

---

## Getting started

```bash
git clone https://github.com/your-org/limitless-llm.git
cd limitless-llm
uv sync
```

Run the checks locally before opening a PR:

```bash
uv run ruff check limitless_llm/
uv run ruff format --check limitless_llm/
uv run mypy --strict limitless_llm/
uv run python -m pytest
```

---

## Pull request guidelines

- Open an issue first for significant changes so the approach can be discussed before you write code
- Keep PRs focused - one feature or fix per PR
- All new code must pass `ruff`, `mypy --strict`, and have test coverage for both happy-path and error-path
- No new dependencies without discussion in the issue first - each new package adds installation overhead for all users
- Follow the code style in `docs/limitless_llm_coding_guidelines.md`: async-first, typed, structlog for logging, Pydantic for boundary models

---

## Reporting bugs

Open a [GitHub Issue](https://github.com/your-org/limitless-llm/issues) with:
- The model and provider you were using
- The approximate document size in tokens (rough estimate is fine)
- The full error message and traceback
- Whether the failure was a 429, a context-length error, or something else
