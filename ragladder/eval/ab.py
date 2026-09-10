"""A/B diff — compare two variants query-by-query.

The average hides the distribution of a change's effects. This classifies each
query as **rescued** (B ranks the first relevant doc higher than A, including
A-missed/B-hit), **lost** (A better than B), or unchanged — so you see whether a
change helps broadly or just trades wins for losses.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragladder.eval.metrics import MetricResult, evaluate


def first_relevant_rank(ranked: list[str], relevant: set[str], k: int) -> int | None:
    """1-based rank of the first relevant doc within top-k; None if none."""
    for rank, doc_id in enumerate(ranked[:k], start=1):
        if doc_id in relevant:
            return rank
    return None


@dataclass
class QueryDiff:
    query_id: str
    rank_a: int | None  # rank of first relevant in A (None = missed within k)
    rank_b: int | None
    status: str  # "rescued" | "lost" | "unchanged"


@dataclass
class ABResult:
    a_name: str
    b_name: str
    k: int
    metrics_a: MetricResult
    metrics_b: MetricResult
    rescued: list[QueryDiff]
    lost: list[QueryDiff]
    n_unchanged: int

    @property
    def diffs(self) -> list[QueryDiff]:
        return self.rescued + self.lost


def _rank_value(rank: int | None) -> float:
    """Missed (None) sorts as worst, so a hit always beats a miss."""
    return float("inf") if rank is None else float(rank)


def ab_diff(
    a_rankings: dict[str, list[str]],
    b_rankings: dict[str, list[str]],
    qrels: dict[str, set[str]],
    k: int,
    a_name: str = "A",
    b_name: str = "B",
    prr_k: int | None = None,
) -> ABResult:
    rescued: list[QueryDiff] = []
    lost: list[QueryDiff] = []
    n_unchanged = 0

    for q_id, relevant in qrels.items():
        if not relevant:
            continue
        rank_a = first_relevant_rank(a_rankings.get(q_id, []), relevant, k)
        rank_b = first_relevant_rank(b_rankings.get(q_id, []), relevant, k)
        va, vb = _rank_value(rank_a), _rank_value(rank_b)
        if vb < va:
            rescued.append(QueryDiff(q_id, rank_a, rank_b, "rescued"))
        elif va < vb:
            lost.append(QueryDiff(q_id, rank_a, rank_b, "lost"))
        else:
            n_unchanged += 1

    # Sort so the biggest swings surface first.
    rescued.sort(key=lambda d: _rank_value(d.rank_a) - _rank_value(d.rank_b), reverse=True)
    lost.sort(key=lambda d: _rank_value(d.rank_b) - _rank_value(d.rank_a), reverse=True)

    return ABResult(
        a_name=a_name,
        b_name=b_name,
        k=k,
        metrics_a=evaluate(a_rankings, qrels, k, prr_k=prr_k),
        metrics_b=evaluate(b_rankings, qrels, k, prr_k=prr_k),
        rescued=rescued,
        lost=lost,
        n_unchanged=n_unchanged,
    )
