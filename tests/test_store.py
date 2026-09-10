"""Exact in-memory store: cosine ranking, top-k, edge cases."""

import numpy as np
import pytest

from ragladder.adapters.stores.inmemory_exact import InMemoryExactStore


def test_ranks_by_cosine():
    store = InMemoryExactStore()
    store.index(
        ["a", "b", "c"],
        np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype=np.float32),
    )
    hits = store.search(np.array([1.0, 0.0]), k=3)
    ids = [h[0] for h in hits]
    assert ids[0] == "a"  # identical direction -> cosine 1
    assert ids[-1] == "b"  # orthogonal -> cosine 0
    assert hits[0][1] == pytest.approx(1.0)


def test_magnitude_does_not_matter():
    store = InMemoryExactStore()
    store.index(["a", "b"], np.array([[10.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    hits = store.search(np.array([0.5, 0.0]), k=2)  # same direction as 'a', tiny magnitude
    assert hits[0][0] == "a"
    assert hits[0][1] == pytest.approx(1.0)


def test_topk_clamped_to_corpus():
    store = InMemoryExactStore()
    store.index(["a", "b"], np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    assert len(store.search(np.array([1.0, 1.0]), k=100)) == 2


def test_search_before_index_raises():
    with pytest.raises(RuntimeError):
        InMemoryExactStore().search(np.array([1.0]), k=1)


def test_length_mismatch_raises():
    store = InMemoryExactStore()
    with pytest.raises(ValueError):
        store.index(["a"], np.array([[1.0], [2.0]], dtype=np.float32))
