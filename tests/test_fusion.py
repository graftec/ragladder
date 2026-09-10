"""Reciprocal Rank Fusion — hand-computed."""

import math

from ragladder.pipeline.fusion import reciprocal_rank_fusion


def test_rrf_rewards_agreement():
    # 'a' is rank 1 in both lists -> highest fused score.
    dense = ["a", "b", "c"]
    bm25 = ["a", "c", "d"]
    fused = reciprocal_rank_fusion([dense, bm25], k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert ids[0] == "a"
    # 'a' score = 1/61 + 1/61; 'c' = 1/63 + 1/62; 'b' = 1/62; 'd' = 1/63
    scores = dict(fused)
    assert math.isclose(scores["a"], 2 / 61)
    assert math.isclose(scores["c"], 1 / 63 + 1 / 62)
    assert scores["a"] > scores["c"] > scores["b"] > scores["d"]


def test_rrf_dedups_across_lists():
    fused = reciprocal_rank_fusion([["a", "b"], ["a", "b"]], k=60)
    ids = [doc_id for doc_id, _ in fused]
    assert ids == ["a", "b"]  # each appears once, merged


def test_rrf_deterministic_tie_break():
    # equal fused scores -> sorted by id for reproducibility
    fused = reciprocal_rank_fusion([["b", "a"], ["a", "b"]], k=60)
    scores = dict(fused)
    assert math.isclose(scores["a"], scores["b"])
    assert [doc_id for doc_id, _ in fused] == ["a", "b"]
