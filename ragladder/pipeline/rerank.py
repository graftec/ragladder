"""Rerank stage: apply a cross-encoder to the merged candidate list.

Runs in series after fusion. For each query it takes the top-`n_rerank` fused
candidates, looks up their text, and asks the reranker to re-sort them.
"""

from __future__ import annotations

from ragladder.adapters.rerankers.base import Candidate, Reranker
from ragladder.data import Corpus


def rerank_candidates(
    reranker: Reranker,
    corpus: Corpus,
    query_text: str,
    candidates: list[tuple[str, float]],
    n_rerank: int,
) -> list[tuple[str, float]]:
    pool = candidates[:n_rerank]
    docs = [Candidate(id=doc_id, text=corpus.texts[doc_id], score=score) for doc_id, score in pool]
    reranked = reranker.rerank(query_text, docs)
    return [(c.id, c.score) for c in reranked]
