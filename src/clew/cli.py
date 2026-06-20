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
reconcile_app = typer.Typer(help="Ledger reconciliation: supersession + contradictions")
analytics_app = typer.Typer(help="Graph analytics (centrality, communities, interlocks)")

app.add_typer(ingest_app, name="ingest")
app.add_typer(extract_app, name="extract")
app.add_typer(resolve_app, name="resolve")
app.add_typer(project_app, name="project")
app.add_typer(eval_app, name="eval")
app.add_typer(reconcile_app, name="reconcile")
app.add_typer(analytics_app, name="analytics")


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


@extract_app.command("mentions")
def extract_mentions_cmd(
    threshold: float = typer.Option(0.5, help="GLiNER confidence threshold"),
    limit: int | None = typer.Option(None, help="Limit number of documents"),
) -> None:
    """Detect mentions (GLiNER) for ingested documents."""
    from clew.extract.service import run_mentions

    console.print("[bold]Detecting mentions[/] ...")
    console.print(run_mentions(threshold=threshold, limit=limit))


@extract_app.command("claims")
def extract_claims_cmd(
    use_llm: bool | None = typer.Option(None, help="Force LLM on/off (default: auto)"),
    limit: int | None = typer.Option(None, help="Limit number of documents"),
) -> None:
    """Extract + write claims (rule baseline + optional LLM)."""
    from clew.extract.service import run_claims

    console.print("[bold]Extracting claims[/] ...")
    console.print(run_claims(use_llm=use_llm, limit=limit))


@resolve_app.command("run")
def resolve_run_cmd(
    backend: str = typer.Option("default", help="ER backend: default | splink"),
) -> None:
    """Resolve mentions + headers into canonical entities."""
    from clew.resolve.service import run_resolution

    console.print("[bold]Resolving entities[/] ...")
    console.print(run_resolution(backend=backend))


@project_app.command("graph")
def project_graph_cmd(
    out: str = typer.Option("data/graph.json", help="Output JSON path"),
) -> None:
    """Rebuild the NetworkX graph projection from the ledger and export JSON."""
    import json
    from pathlib import Path

    from clew.db.session import read_session
    from clew.project.graph import build_graph, graph_to_json

    with read_session() as session:
        g = build_graph(session)
        data = graph_to_json(g)
    Path(out).parent.mkdir(parents=True, exist_ok=True)
    Path(out).write_text(json.dumps(data, indent=2))
    console.print(
        f"graph: {g.number_of_nodes()} nodes, {g.number_of_edges()} edges -> {out}"
    )


@project_app.command("vectors")
def project_vectors_cmd(
    only_missing: bool = typer.Option(False, help="Only embed entities without a vector"),
) -> None:
    """Rebuild entity embeddings into pgvector."""
    from clew.project.vectors import build_entity_vectors

    console.print("[bold]Embedding entities[/] ...")
    console.print(build_entity_vectors(only_missing=only_missing))


@project_app.command("all")
def project_all_cmd() -> None:
    """Rebuild all projections (graph + vectors)."""
    project_vectors_cmd(only_missing=False)
    project_graph_cmd(out="data/graph.json")


@eval_app.command("all")
def eval_all_cmd() -> None:
    """Run all evaluation stages and persist versioned eval_run rows."""
    from clew.eval.run import eval_all

    console.print(eval_all())


@eval_app.command("er")
def eval_er_cmd() -> None:
    """Run entity-resolution evaluation."""
    from clew.eval.run import eval_er

    console.print(eval_er())


@eval_app.command("extraction")
def eval_extraction_cmd() -> None:
    """Run extraction evaluation against per-filing gold."""
    from clew.eval.run import eval_extraction

    console.print(eval_extraction())


@eval_app.command("citation")
def eval_citation_cmd() -> None:
    """Run citation-accuracy evaluation."""
    from clew.eval.run import eval_citation

    console.print(eval_citation())


@reconcile_app.command("supersede")
def reconcile_supersede_cmd() -> None:
    """Link consecutive ownership claims (set valid_to + superseded_by)."""
    from clew.db.session import write_session
    from clew.ledger.supersession import reconcile_supersessions

    with write_session() as session:
        result = reconcile_supersessions(session)
    console.print(result)


@reconcile_app.command("contradictions")
def reconcile_contradictions_cmd() -> None:
    """Detect + materialize contradictions into the contradiction table."""
    from clew.db.session import write_session
    from clew.ledger.contradiction import materialize_contradictions

    with write_session() as session:
        result = materialize_contradictions(session)
    console.print(result)


@reconcile_app.command("all")
def reconcile_all_cmd() -> None:
    """Reconcile supersessions, then materialize remaining contradictions."""
    from clew.db.session import write_session
    from clew.ledger.contradiction import materialize_contradictions
    from clew.ledger.supersession import reconcile_supersessions

    with write_session() as session:
        sup = reconcile_supersessions(session)
        contra = materialize_contradictions(session)
    console.print({"supersession": sup, "contradictions": contra})


@analytics_app.command("summary")
def analytics_summary_cmd() -> None:
    """Graph-wide analytics summary (nodes/edges/components/communities/interlocks)."""
    from clew.analytics.graph_metrics import summary
    from clew.db.session import read_session
    from clew.project.graph import build_graph

    with read_session() as session:
        g = build_graph(session)
    console.print(summary(g))


@analytics_app.command("central")
def analytics_central_cmd(
    metric: str = typer.Option("pagerank", help="pagerank | degree | betweenness"),
    limit: int = typer.Option(20),
) -> None:
    """Most central entities in the ownership graph."""
    from clew.analytics.graph_metrics import centrality
    from clew.db.session import read_session
    from clew.project.graph import build_graph

    with read_session() as session:
        g = build_graph(session)
    for r in centrality(g, metric=metric, limit=limit):
        console.print(f"  {r['score']:.5f}  {r['label']} ({r['type']})")


@analytics_app.command("interlocks")
def analytics_interlocks_cmd(
    min_targets: int = typer.Option(2), limit: int = typer.Option(50)
) -> None:
    """Holders connected to multiple issuers (ownership interlocks)."""
    from clew.analytics.graph_metrics import interlocks
    from clew.db.session import read_session
    from clew.project.graph import build_graph

    with read_session() as session:
        g = build_graph(session)
    for r in interlocks(g, min_targets=min_targets, limit=limit):
        issuers = [i["label"] for i in r["issuers"]]
        console.print(f"  {r['issuer_count']}  {r['label']} -> {issuers}")


@app.command()
def search(query: str, limit: int = 10) -> None:
    """Semantic entity search over the vector projection."""
    from clew.project.vectors import search_entities

    for r in search_entities(query, limit=limit):
        console.print(f"  {r['score']:.3f}  {r['id']}  {r['canonical_name']} ({r['type']})")


if __name__ == "__main__":
    app()
