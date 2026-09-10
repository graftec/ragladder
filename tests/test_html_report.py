"""report.html generation — self-contained, no external assets."""

from ragladder.config import StudyConfig
from ragladder.pipeline.runner import run_study
from ragladder.report.html import render_report_html, write_report_html


def _run(tiny_dir):
    cfg = StudyConfig.from_yaml(tiny_dir / "study.yaml")
    cfg.pipeline = ["dense", "bm25", "rerank"]
    cfg.reranker = {"type": "lexical"}
    return run_study(cfg, cache_dir=None)


def test_html_is_self_contained(tiny_dir):
    html = render_report_html(_run(tiny_dir))
    assert html.startswith("<!doctype html>")
    # No external resources (CSP-friendly / offline).
    assert "http://" not in html.replace('lang="en"', "")
    assert "src=" not in html and "<link" not in html
    assert "<svg" in html  # the ladder chart is inline SVG


def test_html_contains_metrics(tiny_dir):
    html = render_report_html(_run(tiny_dir))
    assert "ragladder" in html
    assert "Ablation ladder" in html
    assert "recall@5" in html  # tiny study k=5
    assert "bow" in html  # embedder name
    # explanatory notes for each section
    assert html.count('class="note"') >= 3
    assert "Perfect Retrieval Rate" in html
    assert "earning its added latency" in html
    # column-header hover tooltips
    assert html.count("<th title=") >= 6
    assert "Mean Reciprocal Rank" in html


def test_write_report_html(tmp_path, tiny_dir):
    path = write_report_html(_run(tiny_dir), tmp_path / "report.html")
    assert path.is_file()
    assert path.read_text(encoding="utf-8").count("<svg") >= 1


def test_html_diagnostics_section(tiny_dir):
    from ragladder.config import StudyConfig
    from ragladder.data import load_dataset
    from ragladder.eval.diagnostics import rerank_threshold_sweep, stage_attribution
    from ragladder.pipeline.runner import Retriever
    from ragladder.report.html import render_report_html

    cfg = StudyConfig.from_yaml(tiny_dir / "study.yaml")
    cfg.pipeline = ["dense", "bm25", "rerank"]
    cfg.reranker = {"type": "lexical"}
    ds = load_dataset(tiny_dir / "corpus.jsonl", tiny_dir / "queries.jsonl", tiny_dir / "qrels.jsonl")
    r = Retriever(cfg, ds, cache_dir=None)
    result = run_study(cfg, retriever=r)

    emb = cfg.embedders[0]
    traces = r.traces(emb)
    diag = {emb.name: (stage_attribution(traces, r.metric_k, cfg.n_rerank),
                       rerank_threshold_sweep(traces, r.metric_k))}

    # Without diagnostics: no section. With: a Diagnostics section appears.
    assert "Diagnostics" not in render_report_html(result)
    html = render_report_html(result, diagnostics=diag)
    assert "Diagnostics" in html
    assert "candidate-pool recall (ceiling)" in html
    assert "min score" in html  # threshold sweep table
    assert "http://" not in html.replace('lang="en"', "")  # still self-contained


def test_html_summary_verdict(tiny_dir):
    from ragladder.eval.summary import summarize
    from ragladder.report.html import render_report_html

    result = _run(tiny_dir)
    summary = summarize(result)
    html = render_report_html(result, summary=summary)
    assert "<h2>Summary" in html
    assert 'class="verdict"' in html  # prominent favorite banner
    assert "Favorite:" in html
    assert summary.favorite in html
    assert 'class="favorite"' in html  # highlighted leaderboard row
