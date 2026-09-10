"""Retrieval metrics — deterministic, no randomness (see the README).

Two families:
  * Standard IR   : recall@k, MRR  (table stakes)
  * Precision / wrong-context : wrong-document count/rate, Perfect Retrieval
    Rate (the differentiator) — what are we letting *into* the LLM's context?

Every function takes a single query's ranked list of retrieved doc ids (rank 1
first) and the set of relevant ids for that query. `evaluate()` aggregates
across queries. Queries with no relevant judgments are skipped in aggregation
(they cannot inform recall/MRR), matching BEIR convention.
"""

from __future__ import annotations

from dataclasses import dataclass


def recall_at_k(ranked: list[str], relevant: set[str], k: int) -> float:
    """Fraction of relevant docs that appear in the top-k."""
    if not relevant:
        return 0.0
    top = set(ranked[:k])
    return len(top & relevant) / len(relevant)


def reciprocal_rank(ranked: list[str], relevant: set[str], k: int) -> float:
    """1 / rank of the first relevant doc within top-k; 0 if none."""
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return 1.0 / rank
    return 0.0


def wrong_count(ranked: list[str], relevant: set[str], k: int) -> int:
    """How many of the top-k retrieved docs are NOT relevant (context poison)."""
    return sum(1 for doc_id in ranked[:k] if doc_id not in relevant)


def is_perfect(ranked: list[str], relevant: set[str], k: int) -> bool:
    """True when the top-k contains zero wrong docs — a clean context."""
    return wrong_count(ranked, relevant, k) == 0


def mrr(rankings: dict[str, list[str]], qrels: dict[str, set[str]], k: int) -> float:
    rrs = [
        reciprocal_rank(rankings.get(q, []), rel, k)
        for q, rel in qrels.items()
        if rel
    ]
    return sum(rrs) / len(rrs) if rrs else 0.0


def perfect_retrieval_rate(
    rankings: dict[str, list[str]], qrels: dict[str, set[str]], k: int
) -> float:
    judged = [(q, rel) for q, rel in qrels.items() if rel]
    if not judged:
        return 0.0
    perfect = sum(1 for q, rel in judged if is_perfect(rankings.get(q, []), rel, k))
    return perfect / len(judged)


@dataclass
class MetricResult:
    """Aggregate metrics for one (embedder, ladder-rung) run.

    recall@k and MRR are measured at the ranking cutoff `k`; the precision
    metrics (Perfect Retrieval Rate, avg wrong-doc count) are measured at
    `prr_k`, which is often smaller (e.g. k=10 for recall, prr_k=3 for PRR)
    because context precision matters most near the top.
    """

    k: int
    n_queries: int
    recall_at_k: float
    mrr: float
    perfect_retrieval_rate: float
    avg_wrong_count: float
    prr_k: int = 0

    def __post_init__(self):
        if not self.prr_k:
            self.prr_k = self.k

    def as_dict(self) -> dict:
        return {
            "k": self.k,
            "prr_k": self.prr_k,
            "n_queries": self.n_queries,
            "recall@k": round(self.recall_at_k, 4),
            "mrr": round(self.mrr, 4),
            "perfect_retrieval_rate": round(self.perfect_retrieval_rate, 4),
            "avg_wrong_count": round(self.avg_wrong_count, 4),
        }


def evaluate(
    rankings: dict[str, list[str]],
    qrels: dict[str, set[str]],
    k: int,
    prr_k: int | None = None,
) -> MetricResult:
    """Aggregate all launch metrics over the queries that carry judgments.

    recall@k and MRR use `k`; PRR and avg wrong-doc count use `prr_k`
    (defaults to `k`).
    """
    prr_k = prr_k or k
    judged = [(q, rel) for q, rel in qrels.items() if rel]
    n = len(judged)
    if n == 0:
        return MetricResult(k, 0, 0.0, 0.0, 0.0, 0.0, prr_k=prr_k)

    recall = sum(recall_at_k(rankings.get(q, []), rel, k) for q, rel in judged) / n
    rr = sum(reciprocal_rank(rankings.get(q, []), rel, k) for q, rel in judged) / n
    perfect = sum(
        1 for q, rel in judged if is_perfect(rankings.get(q, []), rel, prr_k)
    ) / n
    avg_wrong = sum(
        wrong_count(rankings.get(q, []), rel, prr_k) for q, rel in judged
    ) / n

    return MetricResult(
        k=k,
        n_queries=n,
        recall_at_k=recall,
        mrr=rr,
        perfect_retrieval_rate=perfect,
        avg_wrong_count=avg_wrong,
        prr_k=prr_k,
    )
