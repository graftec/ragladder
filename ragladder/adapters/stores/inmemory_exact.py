"""Default store: brute-force EXACT nearest-neighbor over an in-memory matrix.

Exact by design (the README): production vector DBs use approximate search
(HNSW/IVF) for speed, but in *evaluation* that approximation is a confound — a
recall drop could be the embedding model or the index. Exact search removes the
index as a variable. At demo scale (~1,500 docs) this is milliseconds.

Similarity is cosine, implemented as a dot product over L2-normalized vectors.
"""

from __future__ import annotations

import numpy as np

from ragladder.registry import register_store


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0.0, 1.0, norms)  # avoid divide-by-zero on null vectors
    return matrix / norms


@register_store("inmemory_exact")
class InMemoryExactStore:
    def __init__(self, **options):
        self._ids: list[str] = []
        self._matrix: np.ndarray | None = None  # (n_docs, dim), L2-normalized

    def index(self, ids: list[str], vectors: np.ndarray) -> None:
        vectors = np.asarray(vectors, dtype=np.float32)
        if vectors.ndim != 2:
            raise ValueError(f"expected 2-D vectors, got shape {vectors.shape}")
        if len(ids) != vectors.shape[0]:
            raise ValueError(
                f"ids/vectors length mismatch: {len(ids)} ids vs {vectors.shape[0]} rows"
            )
        self._ids = list(ids)
        self._matrix = _normalize(vectors)

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        if self._matrix is None:
            raise RuntimeError("store.search called before index()")
        q = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        norm = np.linalg.norm(q)
        if norm != 0.0:
            q = q / norm
        scores = self._matrix @ q  # cosine similarity, one per doc

        k = min(k, len(self._ids))
        if k <= 0:
            return []
        # argpartition for the top-k, then sort just those — O(n) + O(k log k).
        top_idx = np.argpartition(-scores, k - 1)[:k]
        top_idx = top_idx[np.argsort(-scores[top_idx])]
        return [(self._ids[i], float(scores[i])) for i in top_idx]
