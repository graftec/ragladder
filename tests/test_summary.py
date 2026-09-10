"""Cross-embedder summary: balanced score, favorite, per-axis winners, noise flag."""

from ragladder.eval.metrics import MetricResult
from ragladder.eval.summary import EmbedderScore, assess_quality, summarize
from ragladder.pipeline.runner import LadderResult, RungResult, RunResult


def _rung(recall, mrr, prr, wrong, k=10):
    m = MetricResult(k=k, n_queries=10, recall_at_k=recall, mrr=mrr,
                     perfect_retrieval_rate=prr, avg_wrong_count=wrong, prr_k=3)
    return RungResult(stages=["dense"], label="dense", implemented=True, metrics=m)


def _result(embedders, n_judged=10):
    ladders = [
        LadderResult(embedder=name, dim=1024, max_input_tokens=512, rungs=[_rung(*vals)])
        for name, vals in embedders
    ]
    return RunResult(
        k=10, prr_k=3, metrics_requested=["recall@10", "mrr"],
        dataset_summary={"corpus_size": 100, "n_queries": n_judged, "n_judged": n_judged},
        ladders=ladders,
    )


def test_favorite_is_balanced_score_winner():
    # B dominates on MRR and recall and ties on wrong -> should win.
    result = _result([
        ("A", (0.45, 0.26, 0.0, 2.6)),
        ("B", (0.50, 0.33, 0.0, 2.6)),
    ], n_judged=50)  # large enough to avoid small-n caveat
    s = summarize(result)
    assert s.favorite == "B"
    assert s.rows[0].embedder == "B" and s.rows[0].is_favorite
    assert s.best_recall == "B" and s.best_mrr == "B"
    assert not s.within_noise  # clear margin


def test_per_axis_winners_can_differ():
    # A: best MRR; B: best recall; C: cleanest context.
    result = _result([
        ("A", (0.30, 0.90, 0.0, 5.0)),
        ("B", (0.90, 0.30, 0.0, 5.0)),
        ("C", (0.50, 0.50, 0.5, 1.0)),
    ], n_judged=50)
    s = summarize(result)
    assert s.best_mrr == "A"
    assert s.best_recall == "B"
    assert s.cleanest == "C"


def test_small_n_and_close_margin_flagged():
    result = _result([
        ("A", (0.50, 0.331, 0.0, 2.6)),
        ("B", (0.50, 0.330, 0.0, 2.6)),  # essentially identical
    ], n_judged=10)
    s = summarize(result)
    assert s.within_noise  # margin far below threshold
    assert "10 judged queries" in s.caveat


def test_single_embedder_has_no_margin():
    s = summarize(_result([("solo", (0.5, 0.3, 0.0, 2.0))], n_judged=50))
    assert s.favorite == "solo" and s.margin == 0.0 and not s.within_noise


def test_none_when_no_metrics():
    empty = RunResult(k=10, prr_k=3, metrics_requested=[], dataset_summary={"n_judged": 0})
    assert summarize(empty) is None


def _score(recall, mrr, wrong):
    return EmbedderScore("x", recall, mrr, 0.0, wrong, 1.0, 1.0, 1.0, 1.0)


def test_quality_strong_vs_weak():
    strong = assess_quality(_score(0.95, 0.88, 0.5), k=10, prr_k=3)
    assert strong.level == "strong" and strong.label == "Strong"

    weak = assess_quality(_score(0.20, 0.11, 2.8), k=10, prr_k=3)
    assert weak.level == "weak"
    assert "weak level" in weak.rationale  # both axes weak
    assert "noisy" in weak.rationale  # 2.8 of top 3 irrelevant


def test_quality_overall_is_weaker_axis():
    # Great ranking (MRR 0.9) but poor coverage (recall 0.3) -> not "strong".
    r = assess_quality(_score(0.30, 0.90, 1.0), k=10, prr_k=3)
    assert r.level in {"weak", "moderate"}
    assert "Coverage is the bottleneck" in r.rationale


def test_summary_includes_quality():
    result = _result([("A", (0.95, 0.88, 0.0, 1.0)), ("B", (0.50, 0.30, 0.0, 5.0))], n_judged=50)
    s = summarize(result)
    assert s.quality.level == "strong"  # favorite A is strong on both axes
