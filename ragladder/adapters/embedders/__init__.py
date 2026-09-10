"""Embedder adapters. Import registers the built-in types."""

# Registering imports (kept at bottom to avoid a circular import with base).
from ragladder.adapters.embedders import sentence_transformers as _st  # noqa: F401
from ragladder.adapters.embedders import voyage as _voyage  # noqa: F401
from ragladder.adapters.embedders.base import Embedder

__all__ = ["Embedder"]
