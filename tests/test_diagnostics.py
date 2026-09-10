"""Stage attribution + reranker score-threshold analysis."""

from ragladder.eval.diagnostics import (
    QueryTrace,
    rerank_threshold_sweep,
    stage_attribution,
)


def _trace(qid, relevant, dense, bm25, fused, reranked):
    return QueryTrace(
        query_id=qid,
        relevant=set(relevant),
        dense=dense,
        bm25=bm25,
        fused=fused,
        reranked=reranked,
    )


def test_stage_attribution_pool_ceiling_and_rescue():
    # relevant "a": in the pool at rank 3 (below k=2 pre-rerank), rerank lifts to #1.
    t = _trace(
        "q1",
        {"a"},
        dense=[("x", 0.9), ("y", 0.8), ("a", 0.5)],
        bm25=[("a", 4.0), ("z", 1.0)],
        fused=[("x", 0.3), ("y", 0.2), ("a", 0.15), ("z", 0.1)],
        reranked=[("a", 9.0), ("x", 1.0), ("y", 0.5), ("z", 0.2)],
    )
    attr = stage_attribution([t], k=2, n_rerank=4)

    assert attr.pool_recall == 1.0  # "a" is in the pool
    assert attr.fused_recall_at_k == 0.0  # "a" was rank 3, below k=2, before rerank
    assert attr.final_recall_at_k == 1.0  # rerank pulled it to #1
    assert attr.rescued_by_rerank == 1
    assert attr.dropped_by_rerank == 0
    # "a" found by both dense and bm25
    assert attr.source.both == 1
    assert attr.source.dense_only == 0 and attr.source.bm25_only == 0


def test_source_attribution_splits_branches():
    t = _trace(
        "q1",
        {"a", "b", "c"},
        dense=[("a", 0.9), ("c", 0.7)],  # a: dense; c: dense+bm25
        bm25=[("b", 3.0), ("c", 2.0)],  # b: bm25 only
        fused=[("a", 0.3), ("b", 0.2), ("c", 0.4)],
        reranked=[],
    )
    attr = stage_attribution([t], k=3, n_rerank=3)
    assert attr.source.dense_only == 1  # a
    assert attr.source.bm25_only == 1  # b
    assert attr.source.both == 1  # c


def test_threshold_sweep_tradeoff():
    # Two queries; reranker scores separate relevant (high) from wrong (low).
    traces = [
        _trace("q1", {"a"}, [], [], [], [("a", 5.0), ("x", 1.0), ("y", 0.5)]),
        _trace("q2", {"b"}, [], [], [], [("b", 4.0), ("z", 0.8), ("w", 0.2)]),
    ]
    sweep = rerank_threshold_sweep(traces, k=3, n_thresholds=5)
    assert sweep is not None
    assert sweep.stats.separation > 0  # relevant scores well above wrong

    baseline = sweep.rows[0]  # no floor
    assert baseline.threshold == float("-inf")
    assert baseline.recall_retained == 1.0
    assert baseline.avg_wrong_kept == 2.0  # 2 wrong per query at k=3
    assert baseline.perfect_rate == 0.0  # every context has wrong docs

    strict = sweep.rows[-1]  # highest floor
    assert strict.avg_wrong_kept <= baseline.avg_wrong_kept  # floor removes wrong docs


def test_sweep_none_without_reranker():
    t = _trace("q1", {"a"}, [("a", 0.9)], [], [("a", 0.3)], [])
    assert rerank_threshold_sweep([t], k=3) is None


def test_diagnostics_end_to_end(tiny_dir):
    from ragladder.config import StudyConfig
    from ragladder.data import load_dataset
    from ragladder.pipeline.runner import Retriever

    cfg = StudyConfig.from_yaml(tiny_dir / "study.yaml")
    cfg.pipeline = ["dense", "bm25", "rerank"]
    cfg.reranker = {"type": "lexical"}
    ds = load_dataset(tiny_dir / "corpus.jsonl", tiny_dir / "queries.jsonl", tiny_dir / "qrels.jsonl")
    r = Retriever(cfg, ds, cache_dir=None)

    traces = r.traces(cfg.embedders[0])
    assert len(traces) == 3
    assert all(t.reranked for t in traces)  # rerank ran

    attr = stage_attribution(traces, r.metric_k, cfg.n_rerank)
    assert attr.pool_recall >= attr.final_recall_at_k  # ceiling is an upper bound
    sweep = rerank_threshold_sweep(traces, r.metric_k)
    assert sweep is not None and sweep.rows[0].threshold == float("-inf")
