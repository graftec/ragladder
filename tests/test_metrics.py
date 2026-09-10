"""Metrics correctness against hand-computed values — the differentiator must be right."""

import math

from ragladder.eval.metrics import (
    evaluate,
    is_perfect,
    mrr,
    perfect_retrieval_rate,
    recall_at_k,
    reciprocal_rank,
    wrong_count,
)


def test_recall_partial_and_full():
    # 2 relevant, 1 of them in top-3 -> 0.5
    assert recall_at_k(["a", "x", "y"], {"a", "b"}, k=3) == 0.5
    # both relevant in top-3 -> 1.0
    assert recall_at_k(["a", "b", "y"], {"a", "b"}, k=3) == 1.0
    # relevant present but beyond k -> 0.0
    assert recall_at_k(["x", "y", "z", "a"], {"a"}, k=3) == 0.0


def test_recall_no_relevant_is_zero():
    assert recall_at_k(["a"], set(), k=3) == 0.0


def test_reciprocal_rank():
    assert reciprocal_rank(["x", "a", "y"], {"a"}, k=5) == 0.5  # first hit at rank 2
    assert reciprocal_rank(["a", "x"], {"a"}, k=5) == 1.0
    assert reciprocal_rank(["x", "y"], {"a"}, k=5) == 0.0  # miss
    assert reciprocal_rank(["x", "y", "z", "a"], {"a"}, k=3) == 0.0  # beyond k


def test_wrong_count_and_perfect():
    # top-3, 1 relevant found + 2 wrong
    assert wrong_count(["a", "x", "y"], {"a"}, k=3) == 2
    assert not is_perfect(["a", "x", "y"], {"a"}, k=3)
    # clean context: only the relevant doc in top-k
    assert wrong_count(["a"], {"a"}, k=3) == 0
    assert is_perfect(["a"], {"a"}, k=3)


def test_aggregate_evaluate():
    rankings = {
        "q1": ["a", "x", "y"],  # rel {a}: recall 1, rr 1, wrong 2, not perfect
        "q2": ["x", "b", "y"],  # rel {b}: recall 1, rr 1/2, wrong 2, not perfect
        "q3": ["c"],            # rel {c}: recall 1, rr 1, wrong 0, PERFECT
    }
    qrels = {"q1": {"a"}, "q2": {"b"}, "q3": {"c"}}
    res = evaluate(rankings, qrels, k=3)

    assert res.n_queries == 3
    assert math.isclose(res.recall_at_k, 1.0)
    assert math.isclose(res.mrr, (1.0 + 0.5 + 1.0) / 3)
    assert math.isclose(res.perfect_retrieval_rate, 1 / 3)
    assert math.isclose(res.avg_wrong_count, (2 + 2 + 0) / 3)


def test_queries_without_judgments_are_skipped():
    rankings = {"q1": ["a"], "q2": ["b"]}
    qrels = {"q1": {"a"}, "q2": set()}  # q2 has no judgments
    res = evaluate(rankings, qrels, k=3)
    assert res.n_queries == 1  # only q1 counts
    assert mrr(rankings, qrels, k=3) == 1.0
    assert perfect_retrieval_rate(rankings, qrels, k=3) == 1.0


def test_empty_qrels_is_zero_not_crash():
    res = evaluate({}, {}, k=3)
    assert res.n_queries == 0
    assert res.recall_at_k == 0.0


def test_separate_prr_k():
    # relevant doc at rank 1, plus noise; recall@5 sees it, PRR@1 is clean.
    rankings = {"q1": ["a", "x", "y", "z", "w"]}
    qrels = {"q1": {"a"}}
    res = evaluate(rankings, qrels, k=5, prr_k=1)
    assert res.k == 5 and res.prr_k == 1
    assert res.recall_at_k == 1.0  # found within top-5
    assert res.perfect_retrieval_rate == 1.0  # top-1 is exactly the relevant doc
    assert res.avg_wrong_count == 0.0  # measured at prr_k=1

    # Same rankings, PRR at k=5 -> 4 wrong docs, no longer perfect.
    res5 = evaluate(rankings, qrels, k=5)
    assert res5.prr_k == 5
    assert res5.perfect_retrieval_rate == 0.0
    assert res5.avg_wrong_count == 4.0
