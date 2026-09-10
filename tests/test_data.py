"""BEIR loader: JSONL + TSV qrels, plus the silent-breakage validations."""

import pytest

from ragladder.data import load_dataset


def test_load_tiny(tiny_dir):
    ds = load_dataset(
        tiny_dir / "corpus.jsonl", tiny_dir / "queries.jsonl", tiny_dir / "qrels.jsonl"
    )
    assert len(ds.corpus) == 10
    assert len(ds.queries) == 3
    assert ds.qrels["q1"] == {"d1"}
    assert ds.corpus.texts["d5"].startswith("python")
    assert ds.corpus.metadata["d1"]["topic"] == "animals"


def test_corpus_hash_is_stable_and_content_sensitive(tiny_dir):
    ds1 = load_dataset(
        tiny_dir / "corpus.jsonl", tiny_dir / "queries.jsonl", tiny_dir / "qrels.jsonl"
    )
    ds2 = load_dataset(
        tiny_dir / "corpus.jsonl", tiny_dir / "queries.jsonl", tiny_dir / "qrels.jsonl"
    )
    assert ds1.corpus.content_hash() == ds2.corpus.content_hash()
    ds2.corpus.texts["d1"] = "changed"
    assert ds2.corpus.content_hash() != ds1.corpus.content_hash()


def test_qrels_tsv(tmp_path, tiny_dir):
    tsv = tmp_path / "qrels.tsv"
    tsv.write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq1\td3\t0\nq2\td2\t1\n")
    ds = load_dataset(tiny_dir / "corpus.jsonl", tiny_dir / "queries.jsonl", tsv)
    assert ds.qrels["q1"] == {"d1"}  # score 0 excluded
    assert ds.qrels["q2"] == {"d2"}


def test_dangling_qrel_id_rejected(tmp_path, tiny_dir):
    bad = tmp_path / "qrels.jsonl"
    bad.write_text('{"query_id": "q1", "relevant": ["does_not_exist"]}\n')
    with pytest.raises(ValueError, match="not in the corpus"):
        load_dataset(tiny_dir / "corpus.jsonl", tiny_dir / "queries.jsonl", bad)


def test_duplicate_corpus_id_rejected(tmp_path, tiny_dir):
    dup = tmp_path / "corpus.jsonl"
    dup.write_text('{"_id": "d1", "text": "a"}\n{"_id": "d1", "text": "b"}\n')
    with pytest.raises(ValueError, match="duplicate corpus"):
        load_dataset(dup, tiny_dir / "queries.jsonl", tiny_dir / "qrels.jsonl")
