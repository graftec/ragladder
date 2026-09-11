"""Voyage reranker adapter: registration, lazy failure, and stubbed rerank."""

import pytest

from ragladder.adapters.rerankers.base import Candidate
from ragladder.registry import get_reranker


def test_registered():
    r = get_reranker("voyage")(model="rerank-2.5")
    assert r.name == "rerank-2.5"


def test_missing_key_or_package_raises(monkeypatch):
    monkeypatch.delenv("VOYAGE_API_KEY", raising=False)
    r = get_reranker("voyage")()
    with pytest.raises((ImportError, RuntimeError)):
        r.rerank("q", [Candidate("d1", "text")])


def test_empty_candidates_short_circuits():
    assert get_reranker("voyage")().rerank("q", []) == []


def test_stubbed_client_resorts_by_relevance():
    r = get_reranker("voyage")(model="rerank-2.5")

    class _Res:
        def __init__(self, index, score):
            self.index = index
            self.relevance_score = score

    class _Resp:
        def __init__(self):
            # deliberately out of order to prove we re-sort
            self.results = [_Res(1, 0.2), _Res(0, 0.9), _Res(2, 0.5)]

    class _StubClient:
        def rerank(self, query, documents, model, top_k):
            assert model == "rerank-2.5" and top_k == 3
            return _Resp()

    r._client = _StubClient()  # bypass _ensure_client (no package/key needed)
    cands = [Candidate("a", "ta"), Candidate("b", "tb"), Candidate("c", "tc")]
    out = r.rerank("q", cands)
    # index 0 (a) has the top score 0.9 → first; then index 2 (c) 0.5; then index 1 (b) 0.2
    assert [c.id for c in out] == ["a", "c", "b"]
    assert out[0].score == 0.9
