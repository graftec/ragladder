"""The Embedder interface.

The core knows only "text in, vectors out." Each adapter hides the rest:
local-vs-API, query/document prefixes, native dimension, batching, rate limits.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from ragladder.config import EmbedderConfig


@runtime_checkable
class Embedder(Protocol):
    name: str  # display name from config
    dim: int  # native output dimension (reported, never tuned at launch)
    max_input_tokens: int  # for the truncation-rate stat

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        """Embed texts. `kind` is 'query' or 'document'; the adapter applies any
        model-specific prefix. Returns an (n, dim) float array."""
        ...


class BaseEmbedder:
    """Convenience base carrying the config and prefix handling."""

    max_input_tokens: int = 0
    dim: int = 0

    def __init__(self, cfg: EmbedderConfig):
        self.cfg = cfg
        self.name = cfg.name

    def _prefix(self, kind: str) -> str:
        if kind == "query":
            return self.cfg.query_prefix
        if kind == "document":
            return self.cfg.doc_prefix
        raise ValueError(f"kind must be 'query' or 'document', got {kind!r}")

    def embed(self, texts: list[str], kind: str) -> np.ndarray:  # pragma: no cover
        raise NotImplementedError

    def token_lengths(self, texts: list[str]) -> list[int]:
        """Token count per text, for the truncation stat.

        Default is a whitespace approximation; adapters with a real (subword)
        tokenizer should override this — the whole point of the truncation stat
        is real token counts, since subword tokenization is the hidden confound.
        """
        return [len(t.split()) for t in texts]
