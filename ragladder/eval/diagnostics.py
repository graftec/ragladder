"""Deep diagnostics for choosing the best pipeline combination.

Two questions the headline metrics don't answer:

  1. **Reranker score-threshold analysis** — cross-encoder scores are calibrated
     enough to threshold. If you keep only candidates scoring above some cutoff,
     how much wrong context do you drop, and how much recall do you keep? This
     turns raw scores into an actionable config lever for context precision.

  2. **Stage attribution** — where does performance come from, and what is the
     ceiling? The candidate-pool recall bounds what the reranker could possibly
     achieve; the rescue/drop counts show what it actually did; the source split
     (dense-only / BM25-only / both) shows whether each retriever earns its keep.

These consume `QueryTrace` objects (per-query, per-stage results *with scores*)
produced by the retriever. No metric here is random; everything is deterministic.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ragladder.eval.metrics import recall_at_k


@dataclass
class QueryTrace:
    """One query's results at each stage, scores included."""

    query_id: str
    relevant: set[str]
    dense: list[tuple[str, float]]  # (id, cosine) top_n; empty if dense not run
    bm25: list[tuple[str, float]]  # (id, bm25 score) top_n; empty if bm25 not run
    fused: list[tuple[str, float]]  # post-RRF pool (or single branch) with scores
    reranked: list[tuple[str, float]]  # post-rerank pool with scores; empty if no rerank


# ---------------------------------------------------------------------------
# #2 Stage attribution
# ---------------------------------------------------------------------------


@dataclass
class SourceAttribution:
    """Counts of *relevant* docs by which retriever branch surfaced them."""

    dense_only: int
    bm25_only: int
    both: int
    missed: int  # relevant docs no retriever surfaced within top_n


@dataclass
class StageAttribution:
    k: int
    n_rerank: int
    pool_recall: float  # recall of the candidate pool = ceiling for the reranker
    fused_recall_at_k: float  # recall of top-k BEFORE reranking
    final_recall_at_k: float  # recall of top-k AFTER reranking (or fused if no rerank)
    rescued_by_rerank: int  # relevant docs rerank pulled into top-k
    dropped_by_rerank: int  # relevant docs rerank pushed out of top-k
    reachable_missed: int  # relevant docs in the pool the reranker left below k
    source: SourceAttribution


def _recall(ids: list[str], relevant: set[str]) -> float:
    return recall_at_k(ids, relevant, len(ids)) if relevant else 0.0


def stage_attribution(traces: list[QueryTrace], k: int, n_rerank: int) -> StageAttribution:
    judged = [t for t in traces if t.relevant]
    n = len(judged) or 1

    pool_recall = fused_r = final_r = 0.0
    rescued = dropped = reachable_missed = 0
    d_only = b_only = both = missed = 0

    for t in judged:
        pool_ids = [i for i, _ in t.fused[:n_rerank]]
        fused_topk = [i for i, _ in t.fused[:k]]
        has_rerank = bool(t.reranked)
        final_topk = [i for i, _ in (t.reranked[:k] if has_rerank else t.fused[:k])]

        pool_recall += _recall(pool_ids, t.relevant)
        fused_r += _recall(fused_topk, t.relevant)
        final_r += _recall(final_topk, t.relevant)

        if has_rerank:
            fused_set, final_set = set(fused_topk), set(final_topk)
            rescued += len((final_set & t.relevant) - fused_set)
            dropped += len((fused_set & t.relevant) - final_set)
            pool_set = set(pool_ids)
            reachable_missed += len((pool_set & t.relevant) - final_set)

        dense_ids = {i for i, _ in t.dense}
        bm25_ids = {i for i, _ in t.bm25}
        for rel in t.relevant:
            in_d, in_b = rel in dense_ids, rel in bm25_ids
            if in_d and in_b:
                both += 1
            elif in_d:
                d_only += 1
            elif in_b:
                b_only += 1
            else:
                missed += 1

    return StageAttribution(
        k=k,
        n_rerank=n_rerank,
        pool_recall=pool_recall / n,
        fused_recall_at_k=fused_r / n,
        final_recall_at_k=final_r / n,
        rescued_by_rerank=rescued,
        dropped_by_rerank=dropped,
        reachable_missed=reachable_missed,
        source=SourceAttribution(dense_only=d_only, bm25_only=b_only, both=both, missed=missed),
    )


# ---------------------------------------------------------------------------
# #1 Reranker score-threshold analysis
# ---------------------------------------------------------------------------


@dataclass
class RerankScoreStats:
    relevant_median: float
    relevant_p25: float
    wrong_median: float
    wrong_p90: float
    separation: float  # relevant_median - wrong_median; higher = more separable


@dataclass
class ThresholdRow:
    threshold: float
    recall_retained: float  # relevant kept / total relevant
    avg_wrong_kept: float  # mean wrong docs surviving the floor, per query
    perfect_rate: float  # share of queries with zero wrong kept (clean context)
    kept_relevant: int
    kept_wrong: int


@dataclass
class ThresholdSweep:
    k: int
    stats: RerankScoreStats
    rows: list[ThresholdRow]


def _percentile(values: list[float], q: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=float), q)) if values else 0.0


def rerank_threshold_sweep(
    traces: list[QueryTrace], k: int, n_thresholds: int = 9
) -> ThresholdSweep | None:
    """Sweep a score floor applied to the reranked top-k. Returns None if the
    traces carry no reranker scores (rerank stage not in the pipeline)."""
    judged = [t for t in traces if t.relevant and t.reranked]
    if not judged:
        return None

    rel_scores: list[float] = []
    wrong_scores: list[float] = []
    topk_scores: list[float] = []
    for t in judged:
        for doc_id, score in t.reranked:  # full pool, for separation stats
            (rel_scores if doc_id in t.relevant else wrong_scores).append(score)
        topk_scores += [s for _, s in t.reranked[:k]]

    stats = RerankScoreStats(
        relevant_median=_percentile(rel_scores, 50),
        relevant_p25=_percentile(rel_scores, 25),
        wrong_median=_percentile(wrong_scores, 50),
        wrong_p90=_percentile(wrong_scores, 90),
        separation=_percentile(rel_scores, 50) - _percentile(wrong_scores, 50),
    )

    total_relevant = sum(len(t.relevant) for t in judged)
    # Sweep thresholds across the top-k score range (ascending = cut more).
    qs = np.linspace(0, 90, n_thresholds)
    thresholds = sorted({round(_percentile(topk_scores, q), 6) for q in qs})
    # Prepend a no-floor baseline so the first row == plain top-k.
    thresholds = [float("-inf"), *thresholds]

    rows: list[ThresholdRow] = []
    for t_floor in thresholds:
        kept_rel = kept_wrong = clean_queries = 0
        for t in judged:
            kept = [(i, s) for i, s in t.reranked[:k] if s >= t_floor]
            wrong_here = sum(1 for i, _ in kept if i not in t.relevant)
            kept_rel += sum(1 for i, _ in kept if i in t.relevant)
            kept_wrong += wrong_here
            clean_queries += wrong_here == 0
        rows.append(
            ThresholdRow(
                threshold=t_floor,
                recall_retained=kept_rel / total_relevant if total_relevant else 0.0,
                avg_wrong_kept=kept_wrong / len(judged),
                perfect_rate=clean_queries / len(judged),
                kept_relevant=kept_rel,
                kept_wrong=kept_wrong,
            )
        )
    return ThresholdSweep(k=k, stats=stats, rows=rows)
