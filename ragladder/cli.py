"""ragladder command-line interface.

CLI-first and config-driven: a study is defined in study.yaml; these commands
execute it. Keep the surface small — run / compare / inspect / stats.

Milestone 1 implements `run` for the dense rung; the other commands are declared
so the surface is stable, but exit with a clear "not yet" until their milestone.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

import ragladder.adapters  # noqa: F401  (registers built-in adapters)
from ragladder.config import StudyConfig
from ragladder.data import load_dataset
from ragladder.eval.ab import ab_diff
from ragladder.eval.diagnostics import rerank_threshold_sweep, stage_attribution
from ragladder.eval.stats import corpus_length_stats, truncation_stats
from ragladder.eval.summary import summarize
from ragladder.pipeline.runner import Retriever, run_study
from ragladder.registry import get_embedder
from ragladder.report import (
    render_compare,
    render_diagnostics,
    render_inspect,
    render_run,
    render_stats,
    render_summary,
    write_report_html,
    write_results_json,
)

app = typer.Typer(
    add_completion=False,
    help="Evaluate which stage of your RAG retrieval pipeline earns its keep.",
    no_args_is_help=True,
)
console = Console()


@app.command()
def run(
    study: Path = typer.Argument(..., help="Path to study.yaml"),
    output: Path = typer.Option(Path("out"), "--output", "-o", help="Output directory"),
    k: int = typer.Option(None, "--k", help="Override final top-k"),
    limit: int = typer.Option(None, "--limit", help="Run only the first N queries (smoke test)"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable the embedding cache"),
    diagnostics: bool = typer.Option(
        False, "--diagnostics", help="Print per-embedder stage attribution + reranker threshold sweep"
    ),
):
    """Run a study: every embedder × the ablation ladder → tables + results.json."""
    cfg = StudyConfig.from_yaml(study)
    if k is not None:
        cfg.k = k

    dataset = load_dataset(cfg.corpus, cfg.queries, cfg.qrels)
    retriever = Retriever(
        cfg, dataset, limit=limit, cache_dir=None if no_cache else ".cache/embeddings"
    )
    n_q = len(retriever.queries)
    console.print(
        f"[dim]running {len(cfg.embedders)} embedder(s) over {n_q} queries "
        f"(pipeline: {' → '.join(cfg.pipeline)})…[/dim]"
    )
    result = run_study(
        cfg, retriever=retriever, progress=lambda m: console.print(f"[dim]· {m}[/dim]")
    )

    render_run(result, console)

    summary = summarize(result)
    if summary is not None:
        console.print()
        render_summary(summary, console)

    diag: dict = {}
    if diagnostics:
        for i, emb_cfg in enumerate(cfg.embedders, start=1):
            console.print(f"[dim]· diagnostics [{i}/{len(cfg.embedders)}] {emb_cfg.name}…[/dim]")
            traces = retriever.traces(emb_cfg)
            attribution = stage_attribution(traces, retriever.metric_k, cfg.n_rerank)
            sweep = rerank_threshold_sweep(traces, retriever.metric_k)
            diag[emb_cfg.name] = (attribution, sweep)
            render_diagnostics(emb_cfg.name, attribution, sweep, console)

    # Corpus stats for the machine-readable output (models are already loaded).
    length_stats = corpus_length_stats(dataset.corpus)
    truncation = truncation_stats(
        dataset.corpus, [(e.name, retriever.embedder(e)) for e in cfg.embedders]
    )

    output.mkdir(parents=True, exist_ok=True)
    results_path = write_results_json(
        result, output / "results.json",
        length_stats=length_stats, truncation=truncation, diagnostics=diag or None,
    )
    report_path = write_report_html(
        result, output / "report.html", diagnostics=diag or None, summary=summary
    )
    console.print(f"\n[dim]wrote {results_path}[/dim]")
    console.print(f"[dim]wrote {report_path}[/dim]")


@app.command()
def compare(
    study: Path = typer.Argument(..., help="Path to study.yaml"),
    a: str = typer.Option(..., "--a", help="Variant A (embedder name)"),
    b: str = typer.Option(..., "--b", help="Variant B (embedder name)"),
    limit: int = typer.Option(None, "--limit", help="Run only the first N queries"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable the embedding cache"),
):
    """A/B two embedders through the full pipeline, with per-query rescued/lost."""
    cfg = StudyConfig.from_yaml(study)
    by_name = {e.name: e for e in cfg.embedders}
    for name in (a, b):
        if name not in by_name:
            console.print(
                f"[red]unknown embedder {name!r}[/red]; study has: {sorted(by_name)}"
            )
            raise typer.Exit(code=2)

    dataset = load_dataset(cfg.corpus, cfg.queries, cfg.qrels)
    retriever = Retriever(
        cfg, dataset, limit=limit, cache_dir=None if no_cache else ".cache/embeddings"
    )
    stages = cfg.pipeline  # compare both variants through the full configured pipeline
    rankings_a = retriever.rankings(by_name[a], stages)
    rankings_b = retriever.rankings(by_name[b], stages)

    result = ab_diff(
        rankings_a, rankings_b, retriever.qrels, retriever.metric_k, a, b,
        prr_k=retriever.prr_k,
    )
    render_compare(result, console)


@app.command()
def inspect(
    study: Path = typer.Argument(..., help="Path to study.yaml"),
    query: str = typer.Option(None, "--query", help="Query text to diagnose (any text)"),
    query_id: str = typer.Option(None, "--query-id", help="Diagnose a dataset query by its _id"),
    embedder: str = typer.Option(None, "--embedder", help="Which embedder (default: first)"),
    top: int = typer.Option(10, "--top", help="Rows to show per stage"),
    no_cache: bool = typer.Option(False, "--no-cache", help="Disable the embedding cache"),
):
    """Single-query diagnostic: what each stage retrieved, its scores, and which
    results are correct vs wrong per qrels. Pass a dataset query (`--query-id`, or
    `--query` text that matches one) to get right/wrong marks, or any free text."""
    if not query and not query_id:
        console.print("[red]provide --query <text> or --query-id <id>[/red]")
        raise typer.Exit(code=2)

    cfg = StudyConfig.from_yaml(study)
    dataset = load_dataset(cfg.corpus, cfg.queries, cfg.qrels)
    retriever = Retriever(cfg, dataset, cache_dir=None if no_cache else ".cache/embeddings")

    by_name = {e.name: e for e in cfg.embedders}
    if embedder and embedder not in by_name:
        console.print(f"[red]unknown embedder {embedder!r}[/red]; study has: {sorted(by_name)}")
        raise typer.Exit(code=2)
    emb_cfg = by_name[embedder] if embedder else cfg.embedders[0]

    # Resolve the query + its qrels (if it's a dataset query).
    by_id = {q.id: q for q in dataset.queries}
    by_text = {q.text: q for q in dataset.queries}
    if query_id:
        if query_id not in by_id:
            console.print(f"[red]unknown query id {query_id!r}[/red]; ids: {sorted(by_id)}")
            raise typer.Exit(code=2)
        q = by_id[query_id]
        text, relevant, qid = q.text, dataset.qrels.get(q.id, set()), q.id
    elif query in by_text:  # exact match to a dataset query -> use its qrels
        q = by_text[query]
        text, relevant, qid = q.text, dataset.qrels.get(q.id, set()), q.id
    elif query in by_id:  # convenience: `--query q1` resolves to that query id
        q = by_id[query]
        console.print(f"[dim](interpreting --query {query!r} as query id {q.id})[/dim]")
        text, relevant, qid = q.text, dataset.qrels.get(q.id, set()), q.id
    else:  # free-text query, not in the dataset -> no qrels to judge against
        text, relevant, qid = query, None, "(ad-hoc)"

    trace = retriever.retrieve_text(emb_cfg, text, relevant=relevant, query_id=qid)
    render_inspect(text, emb_cfg.name, trace, dataset.corpus, retriever.metric_k, top, console)


@app.command()
def stats(
    corpus: Path = typer.Argument(..., help="Path to corpus.jsonl"),
    against: Path = typer.Option(None, "--against", help="study.yaml for per-model truncation"),
):
    """Corpus length distribution + (with --against) per-model truncation rate."""
    from ragladder.data.beir import _load_corpus  # corpus-only; no queries/qrels needed

    corpus_obj = _load_corpus(corpus)
    length_stats = corpus_length_stats(corpus_obj)

    truncation = []
    if against is not None:
        cfg = StudyConfig.from_yaml(against)
        embedders = [
            (e.name, get_embedder(e.type)(e)) for e in cfg.embedders
        ]
        truncation = truncation_stats(corpus_obj, embedders)

    render_stats(length_stats, truncation, console)


if __name__ == "__main__":  # pragma: no cover
    app()
