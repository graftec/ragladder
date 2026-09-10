"""The Store interface: hold document vectors, return nearest by similarity."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np


@runtime_checkable
class Store(Protocol):
    def index(self, ids: list[str], vectors: np.ndarray) -> None:
        """Store document vectors under their ids."""
        ...

    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]:
        """Return the top-k (doc_id, score) pairs, highest score first."""
        ...
