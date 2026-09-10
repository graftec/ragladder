"""End-to-end spine: config -> data -> embed -> store -> dense -> metrics -> report.

Uses the bag-of-words embedder (conftest) so the whole pipeline runs offline
with meaningful lexical rankings.
"""

from ragladder.config import StudyConfig
from ragladder.pipeline.runner import run_study
from ragladder.report.results import run_to_dict


def test_run_study_dense_rung(tiny_dir):
    cfg = StudyConfig.from_yaml(tiny_dir / "study.yaml")
    result = run_study(cfg, cache_dir=None)

    assert result.k == 5
    assert result.dataset_summary["corpus_size"] == 10
    assert result.dataset_summary["n_judged"] == 3

    assert len(result.ladders) == 1
    ladder = result.ladders[0]
    assert ladder.embedder == "bow"
    assert ladder.dim == 50  # fixed vocab size

    # Only the dense rung exists in this study, and it is implemented.
    assert len(ladder.rungs) == 1
    dense = ladder.rungs[0]
    assert dense.implemented
    assert dense.stages == ["dense"]

    m = dense.metrics
    # Each query's distinctive words point unambiguously at its relevant doc.
    assert m.recall_at_k == 1.0
    assert m.mrr == 1.0
    # k=5 but each query has 1 relevant -> top-5 carries wrong docs -> PRR 0.
    assert m.perfect_retrieval_rate == 0.0
    assert m.avg_wrong_count == 4.0  # 5 retrieved - 1 relevant


def test_full_ladder_executes_all_rungs(tiny_dir):
    cfg = StudyConfig.from_yaml(tiny_dir / "study.yaml")
    cfg.pipeline = ["dense", "bm25", "rerank"]  # dense -> +bm25 -> +rerank
    cfg.reranker = {"type": "lexical"}  # deterministic fake reranker (conftest)
    result = run_study(cfg, cache_dir=None)

    rungs = result.ladders[0].rungs
    assert len(rungs) == 3
    assert [r.label for r in rungs] == ["dense", "+bm25", "+rerank"]
    assert all(r.implemented and r.metrics is not None for r in rungs)
    assert rungs[1].stages == ["dense", "bm25"]  # fusion is implicit at this rung

    # On this unambiguous fixture every rung already finds the right doc first.
    for r in rungs:
        assert r.metrics.recall_at_k == 1.0
        assert r.metrics.mrr == 1.0


def test_rerank_without_reranker_config_errors(tiny_dir):
    import pytest

    cfg = StudyConfig.from_yaml(tiny_dir / "study.yaml")
    cfg.pipeline = ["dense", "rerank"]
    cfg.reranker = None
    with pytest.raises(ValueError, match="no reranker is configured"):
        run_study(cfg, cache_dir=None)


def test_results_json_serializable(tiny_dir):
    cfg = StudyConfig.from_yaml(tiny_dir / "study.yaml")
    result = run_study(cfg, cache_dir=None)
    d = run_to_dict(result)
    assert d["embedders"][0]["ladder"][0]["metrics"]["recall@k"] == 1.0


def test_limit_flag(tiny_dir):
    cfg = StudyConfig.from_yaml(tiny_dir / "study.yaml")
    result = run_study(cfg, limit=1, cache_dir=None)
    assert result.dataset_summary["n_queries"] == 1
