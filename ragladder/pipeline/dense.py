"""Dense retrieval stage: embed the corpus, embed each query, exact-search.

This is the base rung of the ablation ladder. It embeds the corpus once (the
main cost — cached by the runner), indexes it into the store, then searches per
query. Returns per-query ranked candidate lists of (doc_id, score).
"""

from __future__ import annotations

import numpy as np

from ragladder.adapters.embedders.base import Embedder
from ragladder.data import Corpus, Query
from ragladder.registry import get_store


def dense_search(
    embedder: Embedder,
    store_cfg: dict,
    corpus: Corpus,
    queries: list[Query],
    corpus_vectors: np.ndarray,
    top_n: int,
) -> dict[str, list[tuple[str, float]]]:
    store_type = store_cfg.get("type", "inmemory_exact")
    store = get_store(store_type)(**{k: v for k, v in store_cfg.items() if k != "type"})
    store.index(corpus.ids, corpus_vectors)

    query_vectors = embedder.embed([q.text for q in queries], kind="query")
    results: dict[str, list[tuple[str, float]]] = {}
    for q, qv in zip(queries, query_vectors):
        results[q.id] = store.search(qv, top_n)
    return results
