"""Corpus statistics + per-model truncation rate (the README `stats`).

Embedding models silently truncate text past their max input tokens (e5 ≈ 512,
bge-m3 ≈ 8192, voyage-law-2 ≈ 16000). If a chunk of your corpus is being cut
off, a model comparison is measuring truncation as much as model quality. This
surfaces that confound instead of burying it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ragladder.data import Corpus


@dataclass
class LengthStats:
    unit: str  # "characters" | "whitespace words"
    count: int
    min: int
    p50: float
    mean: float
    p90: float
    max: int


def _summarize(values: list[int], unit: str) -> LengthStats:
    arr = np.asarray(values, dtype=float)
    return LengthStats(
        unit=unit,
        count=len(values),
        min=int(arr.min()),
        p50=float(np.percentile(arr, 50)),
        mean=float(arr.mean()),
        p90=float(np.percentile(arr, 90)),
        max=int(arr.max()),
    )


def corpus_length_stats(corpus: Corpus) -> dict[str, LengthStats]:
    docs = corpus.documents
    chars = [len(t) for t in docs]
    words = [len(t.split()) for t in docs]
    return {
        "characters": _summarize(chars, "characters"),
        "words": _summarize(words, "whitespace words"),
    }


@dataclass
class TruncationStat:
    embedder: str
    max_input_tokens: int
    total: int
    truncated: int
    rate: float  # fraction of docs exceeding max_input_tokens
    token_p50: float
    token_p90: float
    token_max: int


def truncation_stats(
    corpus: Corpus, embedders: list[tuple[str, object]]
) -> list[TruncationStat]:
    """For each (name, embedder), count docs whose token length exceeds the model
    max. `embedder` must expose `token_lengths(texts)` and `max_input_tokens`."""
    docs = corpus.documents
    out: list[TruncationStat] = []
    for name, embedder in embedders:
        # token_lengths() lazily loads the model, which populates max_input_tokens,
        # so count first, then read the limit.
        lengths = embedder.token_lengths(docs)
        max_tokens = int(getattr(embedder, "max_input_tokens", 0) or 0)
        arr = np.asarray(lengths, dtype=float)
        truncated = int(sum(1 for length in lengths if max_tokens and length > max_tokens))
        out.append(
            TruncationStat(
                embedder=name,
                max_input_tokens=max_tokens,
                total=len(docs),
                truncated=truncated,
                rate=truncated / len(docs) if docs else 0.0,
                token_p50=float(np.percentile(arr, 50)),
                token_p90=float(np.percentile(arr, 90)),
                token_max=int(arr.max()),
            )
        )
    return out
