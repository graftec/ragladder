"""Self-contained `report.html` — the shareable artifact where the ladder chart lives.

No external assets, no JS libraries: inline CSS and a hand-built inline SVG bar
chart, so the file works offline and can be emailed or committed as-is. The hero
visual is the ablation ladder — the metric after each added stage, per embedder,
with the marginal Δ called out.
"""

from __future__ import annotations

import html
from pathlib import Path

from ragladder.eval.diagnostics import StageAttribution, ThresholdSweep
from ragladder.eval.summary import Summary
from ragladder.pipeline.runner import LadderResult, RunResult, best_rung

# Per-embedder diagnostics: name -> (stage attribution, reranker sweep or None).
Diagnostics = dict[str, tuple[StageAttribution, "ThresholdSweep | None"]]


def _implemented(ladder: LadderResult):
    return [r for r in ladder.rungs if r.implemented and r.metrics is not None]


def _th(label: str, tip: str) -> str:
    """A table header with a hover tooltip (native `title`; no JS)."""
    return f'<th title="{html.escape(tip)}">{html.escape(label)}</th>'


def _tooltips(k: int, prr_k: int, primary_label: str) -> dict[str, str]:
    return {
        "embedder": "The embedding model being evaluated.",
        "dim": "Native embedding vector dimension of the model.",
        "max_tok": "Maximum input tokens the model reads; longer documents are "
        "silently truncated before embedding.",
        "recall": f"Share of a query's relevant documents that appear in the top {k}. "
        "Higher is better (max 1.0).",
        "mrr": "Mean Reciprocal Rank: 1 / rank of the first correct document, averaged "
        "over queries. 1.0 means a correct doc is always ranked first.",
        "prr": f"Perfect Retrieval Rate: share of queries whose top {prr_k} contain zero "
        "wrong documents — i.e. a clean context with no potential to mislead the LLM.",
        "avg_wrong": f"Average number of irrelevant documents among the top {prr_k} "
        "retrieved (the context 'noise' handed to the LLM). Lower is better.",
        "rung": "A step in the ablation ladder; each rung adds one retrieval stage on "
        "top of the previous one.",
        "configuration": "The retrieval stages active at this rung "
        "(e.g. dense, then +BM25, then +reranker).",
        "delta": f"Marginal change in {primary_label} versus the previous rung — the "
        "value this single stage added. Flat or negative means it isn't earning its cost.",
    }


def _primary(metrics_requested: list[str]) -> str:
    return "mrr" if "mrr" in metrics_requested else "recall"


def _bars(values: list[float], labels: list[str], vmax: float) -> str:
    """A minimal vertical SVG bar chart (one group)."""
    w, h, pad_b, pad_l = 460, 200, 34, 34
    plot_h = h - pad_b - 10
    n = max(1, len(values))
    slot = (w - pad_l - 10) / n
    bar_w = slot * 0.6
    parts = [
        f'<svg viewBox="0 0 {w} {h}" role="img" class="chart">',
        f'<line x1="{pad_l}" y1="{h - pad_b}" x2="{w - 10}" y2="{h - pad_b}" class="axis"/>',
    ]
    for i, (val, label) in enumerate(zip(values, labels)):
        x = pad_l + i * slot + (slot - bar_w) / 2
        bh = (val / vmax) * plot_h if vmax else 0
        y = (h - pad_b) - bh
        parts.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bh:.1f}" '
            f'class="bar" rx="2"/>'
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 5:.1f}" class="val">{val:.3f}</text>'
        )
        parts.append(
            f'<text x="{x + bar_w / 2:.1f}" y="{h - pad_b + 16:.1f}" class="xl">'
            f"{html.escape(label)}</text>"
        )
    parts.append("</svg>")
    return "".join(parts)


def _ladder_chart(ladder: LadderResult, primary: str, vmax: float) -> str:
    rungs = _implemented(ladder)
    values = [getattr(r.metrics, "mrr" if primary == "mrr" else "recall_at_k") for r in rungs]
    labels = [" → ".join(r.stages) for r in rungs]
    return _bars(values, labels, vmax)


def _metric_table(result: RunResult) -> str:
    prr_k = result.prr_k or result.k
    primary = _primary(result.metrics_requested)
    rows = []
    for ladder in result.ladders:
        rung = best_rung(ladder, metric="recall" if primary == "recall" else "mrr")
        if rung is None:
            continue
        best = rung.metrics
        rows.append(
            f"<tr><td>{html.escape(ladder.embedder)}</td>"
            f"<td>{html.escape(' → '.join(rung.stages))}</td>"
            f"<td>{ladder.dim or '?'}</td>"
            f"<td>{ladder.max_input_tokens or '?'}</td>"
            f"<td>{best.recall_at_k:.3f}</td>"
            f"<td>{best.mrr:.3f}</td>"
            f"<td>{best.perfect_retrieval_rate:.3f}</td>"
            f"<td>{best.avg_wrong_count:.2f}</td></tr>"
        )
    tt = _tooltips(result.k, prr_k, "")
    header = (
        _th("embedder", tt["embedder"])
        + _th("best config", "The ladder rung that scored highest for this model — not "
              "necessarily the full pipeline (strong embedders can peak at dense-only).")
        + _th("dim", tt["dim"])
        + _th("max_tok", tt["max_tok"])
        + _th(f"recall@{result.k}", tt["recall"])
        + _th("MRR", tt["mrr"])
        + _th(f"PRR@{prr_k}", tt["prr"])
        + _th(f"avg_wrong@{prr_k}", tt["avg_wrong"])
    )
    return (
        f"<table><thead><tr>{header}</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table>"
    )


def _ladder_section(ladder: LadderResult, result: RunResult, primary: str) -> str:
    prr_k = result.prr_k or result.k
    rows = []
    prev = None
    for i, rung in enumerate(ladder.rungs):
        config = " → ".join(rung.stages)
        if not rung.implemented or rung.metrics is None:
            rows.append(
                f'<tr class="pending"><td>{i}</td><td>{html.escape(config)}</td>'
                f'<td colspan="4">{html.escape(rung.note)}</td></tr>'
            )
            continue
        m = rung.metrics
        val = m.mrr if primary == "mrr" else m.recall_at_k
        if prev is None:
            delta = "—"
        else:
            d = val - prev
            cls = "up" if d >= 0 else "down"
            delta = f'<span class="{cls}">{d:+.3f}</span>'
        prev = val
        rows.append(
            f"<tr><td>{i}</td><td>{html.escape(config)}</td>"
            f"<td>{m.recall_at_k:.3f}</td><td>{m.mrr:.3f}</td>"
            f"<td>{m.perfect_retrieval_rate:.3f}</td><td>{delta}</td></tr>"
        )
    vmax = _chart_vmax(result, primary)
    primary_label = "MRR" if primary == "mrr" else f"recall@{result.k}"
    tt = _tooltips(result.k, prr_k, primary_label)
    header = (
        _th("rung", tt["rung"])
        + _th("configuration", tt["configuration"])
        + _th(f"recall@{result.k}", tt["recall"])
        + _th("MRR", tt["mrr"])
        + _th(f"PRR@{prr_k}", tt["prr"])
        + _th(f"Δ {primary}", tt["delta"])
    )
    return (
        f"<h3>{html.escape(ladder.embedder)}</h3>"
        f'<div class="row">'
        f"{_ladder_chart(ladder, primary, vmax)}"
        f"<table><thead><tr>{header}</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _chart_vmax(result: RunResult, primary: str) -> float:
    vals = [
        getattr(r.metrics, "mrr" if primary == "mrr" else "recall_at_k")
        for ladder in result.ladders
        for r in _implemented(ladder)
    ]
    return max([*vals, 0.001]) * 1.15


_CSS = """
:root { color-scheme: light dark; }
* { box-sizing: border-box; }
body { font: 15px/1.5 -apple-system, system-ui, sans-serif; margin: 0;
  padding: 2rem; max-width: 900px; margin-inline: auto;
  color: #1a1a1a; background: #fafafa; }
h1 { font-size: 1.6rem; margin: 0 0 .25rem; }
h2 { font-size: 1.15rem; margin: 2rem 0 .5rem; border-bottom: 2px solid #e0e0e0; padding-bottom: .25rem; }
h3 { font-size: 1rem; margin: 1.5rem 0 .5rem; }
.sub { color: #666; margin: 0 0 1rem; }
.note { color: #555; margin: .25rem 0 1rem; font-size: .9rem; max-width: 62ch; }
.note b { color: #1a1a1a; font-weight: 600; }
table { border-collapse: collapse; width: 100%; margin: .5rem 0; font-variant-numeric: tabular-nums; }
th, td { padding: .35rem .6rem; text-align: right; border-bottom: 1px solid #e8e8e8; }
th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) { text-align: left; }
thead th { border-bottom: 2px solid #d0d0d0; font-weight: 600; }
th[title] { text-decoration: underline dotted; text-underline-offset: 3px; cursor: help; }
tr.pending td { color: #999; font-style: italic; }
tr.favorite td { background: #eef6ff; font-weight: 600; }
.note.warn { color: #9a5b00; }
.verdict { display: flex; gap: .75rem; align-items: center; margin: .5rem 0 1rem;
  padding: .75rem 1rem; border: 1px solid #bcdcff; border-left: 4px solid #2b6cb0;
  border-radius: 6px; background: #eef6ff; }
.verdict .trophy { font-size: 1.6rem; line-height: 1; }
.verdict .vhead { font-size: 1.15rem; }
.verdict .vsub { color: #4a5568; font-size: .85rem; margin-top: .15rem; }
.verdict .vquality { font-size: .9rem; margin-top: .4rem; }
.verdict .vrat { color: #4a5568; }
.qbadge { display: inline-block; font-size: .72rem; font-weight: 700; letter-spacing: .03em;
  text-transform: uppercase; padding: .1rem .45rem; border-radius: 999px; vertical-align: middle;
  margin-left: .35rem; }
.q-strong { background: #1a7f4b; color: #fff; }
.q-good { background: #2f855a; color: #fff; }
.q-moderate { background: #c98a00; color: #fff; }
.q-weak { background: #c53030; color: #fff; }
.row { display: flex; gap: 1.5rem; align-items: center; flex-wrap: wrap; }
.chart { width: 460px; max-width: 100%; height: auto; }
.bar { fill: #2b6cb0; }
.axis { stroke: #bbb; stroke-width: 1; }
.val { fill: #333; font-size: 11px; text-anchor: middle; }
.xl { fill: #666; font-size: 10px; text-anchor: middle; }
.up { color: #2f855a; font-weight: 600; }
.down { color: #c53030; font-weight: 600; }
footer { margin-top: 2.5rem; color: #999; font-size: .8rem; }
@media (prefers-color-scheme: dark) {
  body { color: #e8e8e8; background: #1a1a1a; }
  h2 { border-color: #333; } th, td { border-color: #2a2a2a; } thead th { border-color: #444; }
  .bar { fill: #4c9be8; } .val { fill: #ccc; } .sub, .xl { fill: #999; }
  .note { color: #b8b8b8; } .note b { color: #e8e8e8; }
  tr.favorite td { background: #16324f; }
  .note.warn { color: #e0a34a; }
  .verdict { background: #16324f; border-color: #2c5a8c; border-left-color: #4c9be8; }
  .verdict .vsub, .verdict .vrat { color: #a8c3e0; }
}
"""


def _score(v: float) -> str:
    return f"{v:.3g}"  # reranker scores span very different scales by model


def _attr_table(a: StageAttribution) -> str:
    rows = [
        ("candidate-pool recall (ceiling)", f"{a.pool_recall:.3f}",
         f"Recall of the {a.n_rerank}-candidate pool — the most the reranker could reach."),
        ("recall@k before rerank", f"{a.fused_recall_at_k:.3f}",
         "Recall of the top-k from fusion, before the reranker runs."),
        ("recall@k after rerank", f"{a.final_recall_at_k:.3f}",
         "Recall of the final top-k after reranking."),
        ("relevant rescued INTO top-k", f'<span class="up">{a.rescued_by_rerank}</span>',
         "Relevant docs the reranker pulled into the top-k."),
        ("relevant dropped OUT of top-k", f'<span class="down">{a.dropped_by_rerank}</span>',
         "Relevant docs the reranker pushed out of the top-k."),
        ("relevant in pool but left below k", str(a.reachable_missed),
         "Relevant docs available in the pool that the reranker failed to lift into the top-k."),
    ]
    body = "".join(
        f'<tr><td title="{html.escape(tip)}">{html.escape(label)}</td>'
        f"<td>{val}</td></tr>"
        for label, val, tip in rows
    )
    return (
        f"<table><thead><tr><th>signal (k={a.k})</th><th>value</th></tr></thead>"
        f"<tbody>{body}</tbody></table>"
    )


def _source_table(a: StageAttribution) -> str:
    s = a.source
    tips = {
        "dense only": "Relevant docs only the dense (semantic) retriever found.",
        "BM25 only": "Relevant docs only the BM25 (lexical) retriever found — BM25's unique value.",
        "both": "Relevant docs found by both retrievers.",
        "missed by both": "Relevant docs neither retriever surfaced — the retrieval ceiling problem.",
    }
    head = "".join(_th(k, v) for k, v in tips.items())
    body = f"<tr><td>{s.dense_only}</td><td>{s.bm25_only}</td><td>{s.both}</td><td>{s.missed}</td></tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _sweep_html(sweep: ThresholdSweep) -> str:
    st = sweep.stats
    note = (
        f'<p class="note">Reranker score — relevant median <b>{_score(st.relevant_median)}</b> '
        f"vs wrong median <b>{_score(st.wrong_median)}</b> (separation {_score(st.separation)}). "
        f"Each row applies a minimum-score floor to the top-{sweep.k} context.</p>"
    )
    head = (
        _th("min score", "Keep only reranked docs scoring at or above this floor.")
        + _th("recall kept", "Share of relevant docs still retained after the floor.")
        + _th("avg wrong/query", "Average irrelevant docs left in the context per query. Lower is better.")
        + _th("clean-context rate", "Share of queries whose kept context has zero wrong docs.")
    )
    body = ""
    for row in sweep.rows:
        thr = "none" if row.threshold == float("-inf") else _score(row.threshold)
        body += (
            f"<tr><td>{thr}</td><td>{row.recall_retained:.3f}</td>"
            f"<td>{row.avg_wrong_kept:.2f}</td><td>{row.perfect_rate:.3f}</td></tr>"
        )
    return note + f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def _diagnostics_html(diagnostics: Diagnostics) -> str:
    if not diagnostics:
        return ""
    note = (
        "For each model: <b>stage attribution</b> — where relevant docs come from "
        "(dense vs BM25) and what reranking changed — and, when a reranker is present, a "
        "<b>score-threshold sweep</b> showing how a minimum-score floor trades retained "
        "recall against a cleaner (less wrong-context) result."
    )
    blocks = [
        '<h2>Diagnostics <span class="sub">(choosing the best combination)</span></h2>',
        f'<p class="note">{note}</p>',
    ]
    for name, (attr, sweep) in diagnostics.items():
        blocks.append(f"<h3>{html.escape(name)}</h3>")
        blocks.append(_attr_table(attr))
        blocks.append(_source_table(attr))
        if sweep is not None:
            blocks.append(_sweep_html(sweep))
        else:
            blocks.append('<p class="note">No reranker in pipeline — score-threshold analysis skipped.</p>')
    return "".join(blocks)


def _summary_html(summary: Summary | None) -> str:
    if summary is None:
        return ""
    w = summary.weights
    rows = ""
    for i, r in enumerate(summary.rows, start=1):
        marker = "⭐" if r.is_favorite else str(i)
        cls = ' class="favorite"' if r.is_favorite else ""
        rows += (
            f"<tr{cls}><td>{marker}</td><td>{html.escape(r.embedder)}</td>"
            f"<td>{html.escape(r.best_config)}</td>"
            f"<td><b>{r.score:.3f}</b></td><td>{r.recall_at_k:.3f}</td>"
            f"<td>{r.mrr:.3f}</td><td>{r.perfect_retrieval_rate:.3f}</td>"
            f"<td>{r.avg_wrong_count:.2f}</td></tr>"
        )
    head = (
        "<th></th>"
        + _th("embedder", "The embedding model, at its best-scoring rung.")
        + _th("best config", "The ladder rung that scored highest for this model — not "
              "necessarily the full pipeline.")
        + _th("score", f"Balanced score = {w['mrr']}·MRR' + {w['recall']}·recall' "
              f"+ {w['clean']}·clean' (' = min-max normalized across the models here; "
              "clean' rewards fewer wrong docs).")
        + _th(f"recall@{summary.k}", "Share of relevant docs in the top-k.")
        + _th("MRR", "Rank of the first correct doc (1.0 = always first).")
        + _th(f"PRR@{summary.prr_k}", "Share of queries with a fully clean top context.")
        + _th("avg_wrong", "Average wrong docs in the context. Lower is better.")
    )
    fav = html.escape(summary.favorite)
    top = summary.rows[0]
    q = summary.quality
    badge = (
        f'<span class="qbadge q-{q.level}">Retrieval: {html.escape(q.label)}</span>'
    )
    verdict = (
        f'<div class="verdict"><span class="trophy">🏆</span>'
        f'<div><div class="vhead">Favorite: <b>{fav}</b> {badge}</div>'
        f'<div class="vsub">highest balanced score ({top.score:.3f}) · '
        f"best recall: <b>{html.escape(summary.best_recall)}</b> · "
        f"best MRR: <b>{html.escape(summary.best_mrr)}</b> · "
        f"cleanest context: <b>{html.escape(summary.cleanest)}</b></div>"
        f'<div class="vquality">{html.escape(q.headline)} '
        f"<span class=\"vrat\">{html.escape(q.rationale)}</span></div></div></div>"
    )
    note = (
        "The ⭐ favorite is picked by the balanced score in the table below; the per-axis "
        "winners are called out in case you weight recall, ranking, or context precision "
        "differently."
    )
    caveat = (
        f'<p class="note warn">⚠ {html.escape(summary.caveat)}</p>' if summary.caveat else ""
    )
    return (
        '<h2>Summary <span class="sub">(favorite by balanced score)</span></h2>'
        f"{verdict}"
        f'<p class="note">{note}</p>'
        f"<table><thead><tr>{head}</tr></thead><tbody>{rows}</tbody></table>"
        f"{caveat}"
    )


def render_report_html(
    result: RunResult,
    diagnostics: Diagnostics | None = None,
    summary: Summary | None = None,
) -> str:
    primary = _primary(result.metrics_requested)
    primary_label = "MRR" if primary == "mrr" else f"recall@{result.k}"
    ds = result.dataset_summary
    prr_k = result.prr_k or result.k
    ladders_html = "".join(
        _ladder_section(ladder, result, primary) for ladder in result.ladders
    )

    intro_note = (
        "This report evaluates the <b>retrieval</b> layer of a RAG pipeline: given each "
        "query, which documents come back, judged against known-correct answers. All "
        "metrics use <b>exact</b> nearest-neighbor search, so no approximate index "
        "distorts the numbers."
    )
    comparison_note = (
        "Each row is one embedding model at its best pipeline configuration. "
        f"<b>dim</b> / <b>max_tok</b> are the model's native vector size and input-token "
        f"limit. <b>recall@{result.k}</b>: share of relevant documents found in the top "
        f"{result.k}. <b>MRR</b>: how high the first correct document ranks (1.0 = always "
        f"first). <b>PRR@{prr_k}</b> (Perfect Retrieval Rate): share of queries whose top "
        f"{prr_k} contain <b>zero</b> wrong documents — a clean context. "
        f"<b>avg_wrong@{prr_k}</b>: average number of irrelevant documents in the top {prr_k}."
    )
    ladder_note = (
        "Each rung adds one retrieval stage on top of the rung above it "
        "(dense &rarr; +BM25 &rarr; +reranker). The <b>&Delta;</b> column is the point: it "
        f"shows how much that single stage changed <b>{primary_label}</b>. A stage with a "
        f"flat or negative &Delta; is not earning its added latency and cost. The chart plots "
        f"{primary_label} for each rung, so you can see the lift (or lack of it) at a glance."
    )

    return f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ragladder report</title><style>{_CSS}</style></head>
<body>
<h1>ragladder — retrieval evaluation</h1>
<p class="sub">corpus {ds['corpus_size']} docs · {ds['n_queries']} queries
({ds['n_judged']} judged) · recall/MRR @ k={result.k} · PRR @ k={prr_k}</p>
<p class="note">{intro_note}</p>

{_summary_html(summary)}

<h2>Model comparison <span class="sub">(best rung)</span></h2>
<p class="note">{comparison_note}</p>
{_metric_table(result)}

<h2>Ablation ladder <span class="sub">(Δ on {primary} — the marginal lift of each stage)</span></h2>
<p class="note">{ladder_note}</p>
{ladders_html}

{_diagnostics_html(diagnostics or {})}

<footer>Generated by ragladder. Metrics computed with exact nearest-neighbor search.</footer>
</body></html>"""


def write_report_html(
    result: RunResult,
    path: str | Path,
    diagnostics: Diagnostics | None = None,
    summary: Summary | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_report_html(result, diagnostics, summary), encoding="utf-8")
    return path
