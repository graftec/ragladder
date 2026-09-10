"""BM25 lexical retrieval on the tiny fixture."""

from ragladder.data import load_dataset
from ragladder.pipeline.bm25 import BM25Index, bm25_search, tokenize


def test_tokenize_lowercases_and_splits():
    assert tokenize("Feline, WHISKERS purr!") == ["feline", "whiskers", "purr"]


def test_bm25_retrieves_lexical_match(tiny_dir):
    ds = load_dataset(
        tiny_dir / "corpus.jsonl", tiny_dir / "queries.jsonl", tiny_dir / "qrels.jsonl"
    )
    results = bm25_search(ds.corpus, ds.queries, top_n=3)
    # each query's distinctive words point at its relevant doc
    assert results["q1"][0][0] == "d1"
    assert results["q2"][0][0] == "d2"
    assert results["q3"][0][0] == "d5"


def test_bm25_index_top_n_limit(tiny_dir):
    ds = load_dataset(
        tiny_dir / "corpus.jsonl", tiny_dir / "queries.jsonl", tiny_dir / "qrels.jsonl"
    )
    index = BM25Index(ds.corpus)
    assert len(index.search("feline whiskers", top_n=2)) == 2
