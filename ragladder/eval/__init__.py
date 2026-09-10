"""Evaluation: standard IR metrics + the precision / wrong-context family."""

from ragladder.eval.metrics import (
    MetricResult,
    evaluate,
    mrr,
    perfect_retrieval_rate,
    recall_at_k,
    wrong_count,
)

__all__ = [
    "MetricResult",
    "evaluate",
    "mrr",
    "perfect_retrieval_rate",
    "recall_at_k",
    "wrong_count",
]
