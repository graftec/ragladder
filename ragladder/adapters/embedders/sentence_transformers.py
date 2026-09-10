"""Local embedder via the `sentence-transformers` library (e5, bge-m3, ...).

Handles the E5 family's mandatory "query: " / "passage: " prefixes through the
config-driven prefix mechanism in BaseEmbedder. The model is loaded lazily so
that merely constructing the adapter (e.g. to read `dim`) is cheap and importing
the package never triggers a model download.
"""

from __future__ import annotations

import numpy as np

from ragladder.adapters.embedders.base import BaseEmbedder
from ragladder.config import EmbedderConfig
from ragladder.registry import register_embedder


@register_embedder("sentence_transformers")
class SentenceTransformerEmbedder(BaseEmbedder):
    def __init__(self, cfg: EmbedderConfig):
        super().__init__(cfg)
        self._model = None
        # Populated on first load; sensible defaults until then.
        self.dim = int(cfg.options.get("dim", 0))
        self.max_input_tokens = int(cfg.options.get("max_input_tokens", 0))

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as e:  # pragma: no cover
            raise ImportError(
                "sentence-transformers is required for embedder type "
                "'sentence_transformers'. Install with: pip install sentence-transformers"
            ) from e
        device = self.cfg.options.get("device")  # None -> library auto-selects
        self._model = SentenceTransformer(self.cfg.model, device=device)
        # Renamed from get_sentence_embedding_dimension() in newer versions.
        get_dim = getattr(self._model, "get_embedding_dimension", None) or (
            self._model.get_sentence_embedding_dimension
        )
        self.dim = int(get_dim())
        max_seq = getattr(self._model, "max_seq_length", None)
        if max_seq:
            self.max_input_tokens = int(max_seq)

    def token_lengths(self, texts: list[str]) -> list[int]:
        """Real subword token counts via the model's tokenizer (no truncation)."""
        self._ensure_model()
        encoded = self._model.tokenizer(list(texts), add_special_tokens=True)
        return [len(ids) for ids in encoded["input_ids"]]

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        self._ensure_model()
        prefix = self._prefix(kind)
        prepared = [prefix + t for t in texts] if prefix else list(texts)
        batch_size = int(self.cfg.options.get("batch_size", 32))
        vecs = self._model.encode(
            prepared,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=False,  # the store normalizes; keep raw here
            show_progress_bar=False,
        )
        return np.asarray(vecs, dtype=np.float32)
