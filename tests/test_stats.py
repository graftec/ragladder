"""Corpus length stats + truncation rate."""

from ragladder.data import load_dataset
from ragladder.eval.stats import corpus_length_stats, truncation_stats


def test_corpus_length_stats(tiny_dir):
    ds = load_dataset(
        tiny_dir / "corpus.jsonl", tiny_dir / "queries.jsonl", tiny_dir / "qrels.jsonl"
    )
    stats = corpus_length_stats(ds.corpus)
    assert stats["words"].count == 10
    assert stats["words"].min == 5  # every tiny doc is 5 words
    assert stats["words"].max == 5
    assert stats["characters"].max >= stats["characters"].min


def test_truncation_rate(tiny_dir, bow_config):
    from tests.conftest import BagOfWordsEmbedder

    ds = load_dataset(
        tiny_dir / "corpus.jsonl", tiny_dir / "queries.jsonl", tiny_dir / "qrels.jsonl"
    )
    emb = BagOfWordsEmbedder(bow_config)
    emb.max_input_tokens = 3  # each 5-word doc exceeds 3 -> all truncated

    (stat,) = truncation_stats(ds.corpus, [("bow", emb)])
    assert stat.total == 10
    assert stat.truncated == 10
    assert stat.rate == 1.0
    assert stat.token_max == 5


def test_no_truncation_when_under_limit(tiny_dir, bow_config):
    from tests.conftest import BagOfWordsEmbedder

    ds = load_dataset(
        tiny_dir / "corpus.jsonl", tiny_dir / "queries.jsonl", tiny_dir / "qrels.jsonl"
    )
    emb = BagOfWordsEmbedder(bow_config)
    emb.max_input_tokens = 100  # generous -> nothing truncated

    (stat,) = truncation_stats(ds.corpus, [("bow", emb)])
    assert stat.truncated == 0
    assert stat.rate == 0.0
