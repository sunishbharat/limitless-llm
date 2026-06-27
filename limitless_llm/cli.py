from __future__ import annotations

import asyncio
import pathlib

import structlog
import typer

from limitless_llm.core.pipeline import PipelineFactory
from limitless_llm.core.token_counter import audit_registry
from limitless_llm.models.config import ModelConfig, PipelineConfig

app = typer.Typer(help="limitless-llm: process large documents on free-tier LLM APIs")
log = structlog.get_logger(__name__)


@app.command()
def run(
    input_file: pathlib.Path = typer.Argument(..., help="Path to the input document"),
    model: str = typer.Option(
        "groq/llama-3.3-70b-versatile",
        "--model",
        "-m",
        help="LiteLLM model identifier",
    ),
    max_output_tokens: int = typer.Option(
        1500,
        "--max-output-tokens",
        help="Maximum tokens per LLM output call",
    ),
    baseline_chunk_size: int = typer.Option(
        6000,
        "--chunk-size",
        help="Baseline chunk size in tokens",
    ),
    output_file: pathlib.Path | None = typer.Option(
        None,
        "--output",
        "-o",
        help="Write output to this file instead of stdout",
    ),
) -> None:
    """Process a document through the TPM-aware pipeline."""
    if not input_file.exists():
        typer.echo(f"Error: input file not found: {input_file}", err=True)
        raise typer.Exit(code=1)

    input_text = input_file.read_text(encoding="utf-8")

    config = PipelineConfig(
        model=ModelConfig(
            model=model,
            max_output_tokens=max_output_tokens,
            baseline_chunk_size=baseline_chunk_size,
        ),
        input_text=input_text,
    )

    runner = PipelineFactory.build(config)
    result = asyncio.run(runner.run())

    if output_file:
        output_file.write_text(result, encoding="utf-8")
        typer.echo(f"Output written to {output_file}")
    else:
        typer.echo(result)


@app.command("refresh-limits")
def refresh_limits() -> None:
    """Audit the model registry for stale entries and report their age.

    Groq does not expose a live limits API, so this prints the registry with
    last-verified dates and exits with a non-zero code if any entry is stale.
    """
    entries = audit_registry()
    any_stale = False
    for entry in entries:
        stale_marker = " [STALE]" if entry["stale"] else ""
        typer.echo(
            f"{entry['model']}: context_window={entry['context_window']}, "
            f"tpm_limit={entry['tpm_limit']}, "
            f"last_verified={entry['last_verified']} ({entry['age_days']} days ago)"
            f"{stale_marker}"
        )
        if entry["stale"]:
            any_stale = True

    if any_stale:
        typer.echo(
            "\nWARNING: One or more registry entries are stale (>90 days). "
            "Update token_counter.py manually with current provider values.",
            err=True,
        )
        raise typer.Exit(code=1)


def main() -> None:
    """Sync entry point for the CLI - the only place asyncio.run() is called."""
    app()
