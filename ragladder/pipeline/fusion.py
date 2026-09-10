"""Reciprocal Rank Fusion (RRF).

Merges the dense and BM25 ranked lists into ONE deduplicated candidate list.
A document's fused score is the sum, over each list it appears in, of
1 / (k + rank). Rank-based (not score-based) fusion sidesteps the fact that
cosine similarities and BM25 scores live on incomparable scales.

    rrf_score(d) = Σ_lists  1 / (k + rank_of_d_in_list)      (k=60 by default)
"""

from __future__ import annotations


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], k: int = 60
) -> list[tuple[str, float]]:
    """Fuse several ranked id lists (rank 1 first) into one, best score first."""
    scores: dict[str, float] = {}
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked, start=1):
            scores[doc_id] = scores.get(doc_id, 0.0) + 1.0 / (k + rank)
    # Sort by fused score desc; ties broken by id for determinism.
    return sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
