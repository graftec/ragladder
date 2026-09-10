"""BM25 lexical retrieval stage (rank-bm25).

The keyword-search branch that runs parallel to dense retrieval over the *same*
corpus. Tokenization is deliberately simple (lowercase, Unicode word tokens) so
it works across languages including German legal text without extra deps.
"""

from __future__ import annotations

import re

from rank_bm25 import BM25Okapi

from ragladder.data import Corpus, Query

_TOKEN = re.compile(r"\w+", re.UNICODE)


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class BM25Index:
    """Tokenize the corpus once, then score queries against it."""

    def __init__(self, corpus: Corpus):
        self.ids = corpus.ids
        self._bm25 = BM25Okapi([tokenize(t) for t in corpus.documents])

    def search(self, query_text: str, top_n: int) -> list[tuple[str, float]]:
        scores = self._bm25.get_scores(tokenize(query_text))
        # rank by score desc; stable order for reproducibility on ties.
        order = sorted(range(len(scores)), key=lambda i: (-scores[i], i))
        return [(self.ids[i], float(scores[i])) for i in order[:top_n]]


def bm25_search(
    corpus: Corpus, queries: list[Query], top_n: int
) -> dict[str, list[tuple[str, float]]]:
    index = BM25Index(corpus)
    return {q.id: index.search(q.text, top_n) for q in queries}
