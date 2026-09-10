"""The Reranker interface.

A cross-encoder reads query+document *together* and scores their relevance,
then re-sorts. Unlike the bi-encoder embedder (which scores query and doc
independently), it runs in series on an already-merged candidate list.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@dataclass
class Candidate:
    """A retrieved document handed to the reranker: id + the text to score."""

    id: str
    text: str
    score: float = 0.0  # upstream (fusion) score; replaced by the rerank score


@runtime_checkable
class Reranker(Protocol):
    name: str

    def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        """Return the candidates re-sorted by relevance, highest first, with each
        candidate's `score` set to the reranker's score."""
        ...
