"""API embedder via Voyage AI (e.g. voyage-law-2, a legal-domain model).

Text is sent to Voyage's API; the vectors come back and all search stays local
(exact, in-memory) as usual. Voyage uses an `input_type` ("query"/"document")
rather than text prefixes, so this adapter maps `kind` to that instead of using
query_prefix/doc_prefix.

Requires the optional dependency (`pip install ragladder[voyage]`) and an API
key in the env var named by the config's `api_key_env` (default VOYAGE_API_KEY).
Both are checked lazily so importing the package never needs either.
"""

from __future__ import annotations

import os

import numpy as np

from ragladder.adapters.embedders.base import BaseEmbedder
from ragladder.config import EmbedderConfig
from ragladder.registry import register_embedder

# Voyage caps how many texts one request may carry.
_MAX_BATCH = 128


@register_embedder("voyage")
class VoyageEmbedder(BaseEmbedder):
    def __init__(self, cfg: EmbedderConfig):
        super().__init__(cfg)
        self._client = None
        self.dim = int(cfg.options.get("dim", 1024))  # voyage-law-2 native dim
        self.max_input_tokens = int(cfg.options.get("max_input_tokens", 16000))
        self._api_key_env = cfg.api_key_env or "VOYAGE_API_KEY"

    def _ensure_client(self):
        if self._client is not None:
            return
        try:
            import voyageai
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "voyageai is required for embedder type 'voyage'. "
                "Install with: pip install ragladder[voyage]"
            ) from e
        api_key = os.environ.get(self._api_key_env)
        if not api_key:
            raise RuntimeError(
                f"missing API key: set the {self._api_key_env} environment variable "
                f"for embedder {self.name!r}"
            )
        self._client = voyageai.Client(api_key=api_key)

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        input_type = "query" if kind == "query" else "document"
        if kind not in ("query", "document"):
            raise ValueError(f"kind must be 'query' or 'document', got {kind!r}")
        self._ensure_client()

        vectors: list[list[float]] = []
        for start in range(0, len(texts), _MAX_BATCH):
            batch = texts[start : start + _MAX_BATCH]
            resp = self._client.embed(batch, model=self.cfg.model, input_type=input_type)
            vectors.extend(resp.embeddings)
        return np.asarray(vectors, dtype=np.float32)
