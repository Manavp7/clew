"""clew command-line interface.

Subcommands map onto the pipeline stages. Heavier stages (ingest/extract/
resolve/project/eval) are wired during Phase 1; this module keeps the command
surface stable so the Makefile and docs do not churn.
"""

from __future__ import annotations

import typer
from rich.console import Console

app = typer.Typer(help="Canonical Intelligence Engine (clew) CLI", no_args_is_help=True)
console = Console()

ingest_app = typer.Typer(help="Ingest documents from sources")
extract_app = typer.Typer(help="Extract mentions + claims")
resolve_app = typer.Typer(help="Entity resolution")
project_app = typer.Typer(help="Rebuild projections from the ledger")
eval_app = typer.Typer(help="Evaluation harness")

app.add_typer(ingest_app, name="ingest")
app.add_typer(extract_app, name="extract")
app.add_typer(resolve_app, name="resolve")
app.add_typer(project_app, name="project")
app.add_typer(eval_app, name="eval")


@app.command()
def version() -> None:
    """Print the clew version."""
    from clew import __version__

    console.print(f"clew {__version__}")


@ingest_app.command("edgar")
def ingest_edgar(
    form: str = typer.Option("13D", help="13D or 13G"),
    limit: int = typer.Option(100, help="Max filings to ingest"),
    year: int | None = typer.Option(None, help="Filter by filing year"),
    quarter: int | None = typer.Option(None, help="Filter by quarter (1-4)"),
) -> None:
    """Ingest SEC EDGAR 13D/13G filings into the ledger."""
    from clew.ingest.edgar import EdgarConnector
    from clew.ingest.service import ingest_documents

    console.print(f"[bold]Ingesting[/] {limit} {form} filings from EDGAR ...")
    summary = ingest_documents(
        EdgarConnector(), limit, form=form, year=year, quarter=quarter
    )
    console.print(summary)


if __name__ == "__main__":
    app()
