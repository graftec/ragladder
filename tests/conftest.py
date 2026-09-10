"""Shared test fixtures.

A deterministic bag-of-words embedder lets the whole retrieval spine run
end-to-end with zero model downloads and meaningful (lexical) rankings. It is
registered under type "bag_of_words" so a study.yaml can select it, but it is a
test artifact — not a shipped adapter.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pytest

from ragladder.adapters.embedders.base import BaseEmbedder
from ragladder.adapters.rerankers.base import Candidate
from ragladder.config import EmbedderConfig
from ragladder.registry import register_embedder, register_reranker

FIXTURES = Path(__file__).parent / "fixtures"
TINY = FIXTURES / "tiny"

_WORD = re.compile(r"[a-zA-Z]+")

# A fixed vocabulary covering the tiny fixture so query/doc vectors are aligned.
_VOCAB = sorted(
    {
        "cat", "feline", "pet", "purr", "whiskers",
        "dog", "canine", "bark", "loyal", "puppy",
        "car", "engine", "wheel", "drive", "road",
        "boat", "sail", "ocean", "water", "harbor",
        "python", "programming", "code", "function", "loop",
        "bread", "bake", "flour", "oven", "dough",
        "guitar", "music", "string", "chord", "melody",
        "mountain", "hike", "trail", "summit", "climb",
        "rain", "cloud", "weather", "storm", "umbrella",
        "coffee", "espresso", "caffeine", "brew", "mug",
    }
)
_INDEX = {w: i for i, w in enumerate(_VOCAB)}


@register_embedder("bag_of_words")
class BagOfWordsEmbedder(BaseEmbedder):
    """Deterministic lexical embedder over a fixed vocabulary (cosine == overlap)."""

    def __init__(self, cfg: EmbedderConfig):
        super().__init__(cfg)
        self.dim = len(_VOCAB)
        self.max_input_tokens = 512

    def embed(self, texts: list[str], kind: str) -> np.ndarray:
        self._prefix(kind)  # validates kind
        out = np.zeros((len(texts), self.dim), dtype=np.float32)
        for row, text in enumerate(texts):
            for word in _WORD.findall(text.lower()):
                idx = _INDEX.get(word)
                if idx is not None:
                    out[row, idx] += 1.0
        return out


@register_reranker("lexical")
class LexicalReranker:
    """Deterministic fake reranker: score = word overlap with the query.

    Real enough to reorder candidates meaningfully, with no model download.
    """

    def __init__(self, model: str = "lexical", name: str | None = None, **options):
        self.name = name or model

    def rerank(self, query: str, candidates: list[Candidate]) -> list[Candidate]:
        q_words = set(_WORD.findall(query.lower()))
        scored = [
            Candidate(
                id=c.id,
                text=c.text,
                score=float(len(q_words & set(_WORD.findall(c.text.lower())))),
            )
            for c in candidates
        ]
        scored.sort(key=lambda c: (c.score, c.id), reverse=True)
        return scored


@pytest.fixture
def tiny_dir() -> Path:
    return TINY


@pytest.fixture
def bow_config() -> EmbedderConfig:
    return EmbedderConfig(name="bow", type="bag_of_words", model="bow")
