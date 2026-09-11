"""Voyage AI reranker (rerank-2.5 / rerank-3 / ...).

An API cross-encoder alternative to the local `cross_encoder` reranker. Voyage
returns calibrated 0–1 relevance scores, which also make the score-threshold
diagnostics far more readable than a local reranker's raw logits.

Requires the optional dependency (`pip install ragladder[voyage]`) and an API key
in the env var named by config (default VOYAGE_API_KEY). Both are checked lazily.
"""

from __future__ import annotations

import os

from ragladder.adapters.rerankers.base import Candidate
from ragladder.registry import register_reranker


@register_reranker("voyage")
class VoyageReranker:
    def __init__(
        self,
        model: str = "rerank-2.5",
        api_key_env: str = "VOYAGE_API_KEY",
        name: str | None = None,
        **options,
    ):
        self.model_id = model
        self.name = name or model
        self._api_key_env = api_key_env
        self._options = options
        self._client = None

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            import voyageai
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "voyageai is required for reranker type 'voyage'. "
                "Install with: pip install ragladder[voyage]"
            ) from e
        api_key = os.environ.get(self._api_key_env)
        if not api_key:
            raise RuntimeError(
                f"missing API key: set the {self._api_key_env} environment variable "
                f"for reranker {self.name!r}"
            )
        self._client = voyageai.Client(api_key=api_key)

    def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        if not candidates:
            return []
        self._ensure_client()
        resp = self._client.rerank(
            query, [c.text for c in candidates], model=self.model_id, top_k=len(candidates)
        )
        rescored = [
            Candidate(
                id=candidates[r.index].id,
                text=candidates[r.index].text,
                score=float(r.relevance_score),
            )
            for r in resp.results
        ]
        # Voyage returns results sorted by relevance, but sort defensively.
        rescored.sort(key=lambda c: c.score, reverse=True)
        return rescored
