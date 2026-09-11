"""Terminal tables via `rich`.

Two views, per the README:
  * Model comparison — one row per embedder (native dim, max tokens, metrics).
  * Ablation ladder — per embedder, one row per rung with the marginal Δ.
"""

from __future__ import annotations

from rich.console import Console
from rich.table import Table

from ragladder.eval.ab import ABResult
from ragladder.eval.diagnostics import StageAttribution, ThresholdSweep
from ragladder.eval.stats import LengthStats, TruncationStat
from ragladder.eval.summary import Summary
from ragladder.pipeline.runner import LadderResult, RunResult, best_rung


def _primary_metric(metrics_requested: list[str]) -> str:
    """The metric whose marginal Δ drives the ladder. Prefer MRR (docs)."""
    if "mrr" in metrics_requested:
        return "mrr"
    for m in metrics_requested:
        if m.startswith("recall@"):
            return "recall@k"
    return "mrr"


def _implemented_rungs(ladder: LadderResult):
    return [r for r in ladder.rungs if r.implemented and r.metrics is not None]


def render_run(result: RunResult, console: Console | None = None) -> None:
    console = console or Console()
    ds = result.dataset_summary
    prr_note = f" · PRR@{result.prr_k}" if result.prr_k and result.prr_k != result.k else ""
    console.print(
        f"[bold]ragladder[/bold]  corpus={ds['corpus_size']} docs · "
        f"queries={ds['n_queries']} ({ds['n_judged']} judged) · k={result.k}{prr_note}"
    )

    _render_comparison(result, console)
    console.print()
    _render_ladders(result, console)


def _render_comparison(result: RunResult, console: Console) -> None:
    primary = _primary_metric(result.metrics_requested)
    metric_key = "recall" if primary == "recall@k" else "mrr"
    table = Table(title=f"Model comparison (each model's best rung by {primary})")
    table.add_column("embedder", style="bold")
    table.add_column("best config")
    table.add_column("dim", justify="right")
    table.add_column("max_tok", justify="right")
    prr_k = result.prr_k or result.k
    table.add_column(f"recall@{result.k}", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column(f"PRR@{prr_k}", justify="right")
    table.add_column("avg_wrong", justify="right")

    for ladder in result.ladders:
        rung = best_rung(ladder, metric=metric_key)
        if rung is None:
            table.add_row(ladder.embedder, "—", str(ladder.dim or "?"), "—", "—", "—", "—", "—")
            continue
        best = rung.metrics
        table.add_row(
            ladder.embedder,
            " → ".join(rung.stages),
            str(ladder.dim or "?"),
            str(ladder.max_input_tokens or "?"),
            f"{best.recall_at_k:.3f}",
            f"{best.mrr:.3f}",
            f"{best.perfect_retrieval_rate:.3f}",
            f"{best.avg_wrong_count:.2f}",
        )
    console.print(table)


def _render_ladders(result: RunResult, console: Console) -> None:
    primary = _primary_metric(result.metrics_requested)
    for ladder in result.ladders:
        table = Table(
            title=f"Ablation ladder — {ladder.embedder}  "
            f"(Δ on {primary})"
        )
        prr_k = result.prr_k or result.k
        table.add_column("rung")
        table.add_column("configuration")
        table.add_column(f"recall@{result.k}", justify="right")
        table.add_column("MRR", justify="right")
        table.add_column(f"PRR@{prr_k}", justify="right")
        table.add_column(f"Δ {primary}", justify="right")

        prev_value: float | None = None
        for i, rung in enumerate(ladder.rungs):
            config_str = " → ".join(rung.stages)
            if not rung.implemented or rung.metrics is None:
                table.add_row(str(i), config_str, "—", "—", "—", f"[dim]{rung.note}[/dim]")
                continue
            m = rung.metrics
            value = m.mrr if primary == "mrr" else m.recall_at_k
            if prev_value is None:
                delta = "—"
            else:
                d = value - prev_value
                delta = f"[green]+{d:.3f}[/green]" if d >= 0 else f"[red]{d:.3f}[/red]"
            prev_value = value
            table.add_row(
                str(i),
                config_str,
                f"{m.recall_at_k:.3f}",
                f"{m.mrr:.3f}",
                f"{m.perfect_retrieval_rate:.3f}",
                delta,
            )
        console.print(table)


def render_summary(summary: Summary, console: Console | None = None) -> None:
    console = console or Console()
    w = summary.weights
    table = Table(
        title=f"Summary — balanced score = {w['mrr']}·MRR' + {w['recall']}·recall' "
        f"+ {w['clean']}·clean'  (' = normalized across models)"
    )
    table.add_column("")
    table.add_column("embedder", style="bold")
    table.add_column("best config")
    table.add_column("score", justify="right")
    table.add_column(f"recall@{summary.k}", justify="right")
    table.add_column("MRR", justify="right")
    table.add_column(f"PRR@{summary.prr_k}", justify="right")
    table.add_column("avg_wrong", justify="right")
    for i, r in enumerate(summary.rows, start=1):
        star = "⭐" if r.is_favorite else str(i)
        table.add_row(
            star,
            r.embedder,
            r.best_config,
            f"[bold]{r.score:.3f}[/bold]" if r.is_favorite else f"{r.score:.3f}",
            f"{r.recall_at_k:.3f}",
            f"{r.mrr:.3f}",
            f"{r.perfect_retrieval_rate:.3f}",
            f"{r.avg_wrong_count:.2f}",
        )
    console.print(table)
    console.print(
        f"⭐ favorite: [bold]{summary.favorite}[/bold]  ·  "
        f"best recall: {summary.best_recall}  ·  best MRR: {summary.best_mrr}  ·  "
        f"cleanest context: {summary.cleanest}"
    )
    q = summary.quality
    color = {"strong": "green", "good": "green", "moderate": "yellow", "weak": "red"}[q.level]
    console.print(f"[{color}]Retrieval quality: {q.label.upper()}[/{color}] — {q.headline}")
    console.print(f"[dim]{q.rationale}[/dim]")
    if summary.caveat:
        console.print(f"[yellow]⚠ {summary.caveat}[/yellow]")


def _fmt_rank(rank) -> str:
    return "—" if rank is None else str(rank)


def render_compare(result: ABResult, console: Console | None = None) -> None:
    console = console or Console()
    a, b = result.a_name, result.b_name

    summary = Table(title=f"A/B — {a} vs {b}  (k={result.k})")
    summary.add_column("metric")
    summary.add_column(a, justify="right")
    summary.add_column(b, justify="right")
    summary.add_column("Δ (B−A)", justify="right")
    prr_k = result.metrics_a.prr_k
    for label, va, vb in (
        (f"recall@{result.k}", result.metrics_a.recall_at_k, result.metrics_b.recall_at_k),
        ("MRR", result.metrics_a.mrr, result.metrics_b.mrr),
        (f"PRR@{prr_k}", result.metrics_a.perfect_retrieval_rate, result.metrics_b.perfect_retrieval_rate),
        (f"avg_wrong@{prr_k}", result.metrics_a.avg_wrong_count, result.metrics_b.avg_wrong_count),
    ):
        d = vb - va
        color = "green" if d >= 0 else "red"
        summary.add_row(label, f"{va:.3f}", f"{vb:.3f}", f"[{color}]{d:+.3f}[/{color}]")
    console.print(summary)

    console.print(
        f"\n[bold]{len(result.rescued)} rescued[/bold] by {b} · "
        f"[bold]{len(result.lost)} lost[/bold] · {result.n_unchanged} unchanged"
    )

    if result.rescued or result.lost:
        detail = Table(title="Per-query attribution (first-relevant rank)")
        detail.add_column("query")
        detail.add_column(f"rank in {a}", justify="right")
        detail.add_column(f"rank in {b}", justify="right")
        detail.add_column("verdict")
        for d in result.rescued:
            detail.add_row(d.query_id, _fmt_rank(d.rank_a), _fmt_rank(d.rank_b), "[green]rescued[/green]")
        for d in result.lost:
            detail.add_row(d.query_id, _fmt_rank(d.rank_a), _fmt_rank(d.rank_b), "[red]lost[/red]")
        console.print(detail)


def render_stats(
    length_stats: dict[str, LengthStats],
    truncation: list[TruncationStat],
    console: Console | None = None,
) -> None:
    console = console or Console()

    lengths = Table(title="Corpus length distribution")
    lengths.add_column("unit")
    lengths.add_column("docs", justify="right")
    lengths.add_column("min", justify="right")
    lengths.add_column("p50", justify="right")
    lengths.add_column("mean", justify="right")
    lengths.add_column("p90", justify="right")
    lengths.add_column("max", justify="right")
    for s in length_stats.values():
        lengths.add_row(
            s.unit, str(s.count), str(s.min), f"{s.p50:.0f}", f"{s.mean:.0f}",
            f"{s.p90:.0f}", str(s.max),
        )
    console.print(lengths)

    if not truncation:
        console.print("[dim]No embedders given (--against) — skipping truncation rate.[/dim]")
        return

    trunc = Table(title="Per-model truncation rate")
    trunc.add_column("embedder")
    trunc.add_column("max_tok", justify="right")
    trunc.add_column("tok p50", justify="right")
    trunc.add_column("tok p90", justify="right")
    trunc.add_column("tok max", justify="right")
    trunc.add_column("truncated", justify="right")
    for t in truncation:
        pct = t.rate * 100
        color = "red" if pct > 0 else "green"
        trunc.add_row(
            t.embedder,
            str(t.max_input_tokens or "?"),
            f"{t.token_p50:.0f}",
            f"{t.token_p90:.0f}",
            str(t.token_max),
            f"[{color}]{t.truncated}/{t.total} ({pct:.0f}%)[/{color}]",
        )
    console.print(trunc)


def render_inspect(
    query_text: str,
    embedder: str,
    trace,
    corpus,
    k: int,
    top: int = 10,
    console: Console | None = None,
) -> None:
    """Single-query diagnostic: what each stage retrieved, scored, right vs wrong."""
    console = console or Console()
    resolved = (
        f"query_id={trace.query_id} (dataset query)"
        if trace.query_id != "(ad-hoc)"
        else "query_id=(ad-hoc — text not found in dataset)"
    )
    console.print(f"[bold]inspect[/bold]  embedder={embedder}  ·  {resolved}  ·  k={k}")
    console.print(f'[dim]query:[/dim] "{query_text}"')
    if trace.relevant:
        console.print(f"[dim]relevant (qrels):[/dim] {', '.join(sorted(trace.relevant))}")
    else:
        console.print("[dim]no qrels for this query — nothing marked right/wrong[/dim]")

    stages = [
        ("dense", trace.dense),
        ("bm25", trace.bm25),
        ("fused (RRF)", trace.fused),
        ("reranked", trace.reranked),
    ]
    for name, items in stages:
        if not items:
            continue
        first = next((i + 1 for i, (d, _) in enumerate(items) if d in trace.relevant), None)
        where = (
            f"first relevant at rank {first}"
            if first
            else ("no relevant in list" if trace.relevant else "")
        )
        table = Table(title=f"{name}  ({where})" if where else name)
        table.add_column("#", justify="right")
        if trace.relevant:
            table.add_column("", justify="center")  # ✓ / ✗
        table.add_column("doc_id")
        table.add_column("score", justify="right")
        table.add_column("title")
        for rank, (doc_id, score) in enumerate(items[:top], start=1):
            title = corpus.titles.get(doc_id, "") if hasattr(corpus, "titles") else ""
            row = [str(rank)]
            if trace.relevant:
                hit = doc_id in trace.relevant
                row.append("[green]✓[/green]" if hit else "[red]✗[/red]")
            row += [doc_id, f"{score:.4g}", title]
            style = "green" if (trace.relevant and doc_id in trace.relevant) else None
            table.add_row(*row, style=style)
        console.print(table)


def render_diagnostics(
    embedder: str,
    attribution: StageAttribution,
    sweep: ThresholdSweep | None,
    console: Console | None = None,
) -> None:
    console = console or Console()
    a = attribution
    console.print(f"\n[bold]Diagnostics — {embedder}[/bold]")

    # --- #2 Stage attribution ---
    attr = Table(title=f"Stage attribution (k={a.k}, pool=n_rerank={a.n_rerank})")
    attr.add_column("signal")
    attr.add_column("value", justify="right")
    attr.add_row("candidate-pool recall (ceiling)", f"{a.pool_recall:.3f}")
    attr.add_row("recall@k before rerank", f"{a.fused_recall_at_k:.3f}")
    attr.add_row("recall@k after rerank", f"{a.final_recall_at_k:.3f}")
    attr.add_row("relevant rescued INTO top-k by rerank", f"[green]{a.rescued_by_rerank}[/green]")
    attr.add_row("relevant dropped OUT of top-k by rerank", f"[red]{a.dropped_by_rerank}[/red]")
    attr.add_row("relevant in pool but left below k", str(a.reachable_missed))
    console.print(attr)

    src = Table(title="Where relevant docs were found (retriever attribution)")
    src.add_column("dense only", justify="right")
    src.add_column("BM25 only", justify="right")
    src.add_column("both", justify="right")
    src.add_column("missed by both", justify="right")
    s = a.source
    src.add_row(str(s.dense_only), str(s.bm25_only), str(s.both), str(s.missed))
    console.print(src)

    # --- #1 Reranker score-threshold analysis ---
    if sweep is None:
        console.print("[dim]No reranker in pipeline — score-threshold analysis skipped.[/dim]")
        return

    st = sweep.stats

    def _score(v: float) -> str:  # reranker scores span very different scales by model
        return f"{v:.3g}"

    console.print(
        f"[dim]reranker score — relevant median {_score(st.relevant_median)} "
        f"(p25 {_score(st.relevant_p25)}) · wrong median {_score(st.wrong_median)} "
        f"(p90 {_score(st.wrong_p90)}) · separation {_score(st.separation)}[/dim]"
    )
    sweep_tbl = Table(title=f"Reranker score-threshold sweep (floor on top-{sweep.k} context)")
    sweep_tbl.add_column("min score", justify="right")
    sweep_tbl.add_column("recall kept", justify="right")
    sweep_tbl.add_column("avg wrong/query", justify="right")
    sweep_tbl.add_column("clean-context rate", justify="right")
    for row in sweep.rows:
        thr = "none" if row.threshold == float("-inf") else f"{row.threshold:.3g}"
        sweep_tbl.add_row(
            thr,
            f"{row.recall_retained:.3f}",
            f"{row.avg_wrong_kept:.2f}",
            f"{row.perfect_rate:.3f}",
        )
    console.print(sweep_tbl)
