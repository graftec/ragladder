"""Single-query inspection via Retriever.retrieve_text (the engine of `inspect`)."""

from ragladder.config import StudyConfig
from ragladder.data import load_dataset
from ragladder.pipeline.runner import Retriever


def _retriever(tiny_dir, pipeline, reranker=None):
    cfg = StudyConfig.from_yaml(tiny_dir / "study.yaml")
    cfg.pipeline = pipeline
    if reranker:
        cfg.reranker = reranker
    ds = load_dataset(
        tiny_dir / "corpus.jsonl", tiny_dir / "queries.jsonl", tiny_dir / "qrels.jsonl"
    )
    return Retriever(cfg, ds, cache_dir=None), cfg, ds


def test_retrieve_text_full_pipeline(tiny_dir):
    r, cfg, ds = _retriever(tiny_dir, ["dense", "bm25", "rerank"], {"type": "lexical"})
    q = ds.queries[0]  # "feline whiskers purr" -> d1
    trace = r.retrieve_text(cfg.embedders[0], q.text, relevant=ds.qrels[q.id], query_id=q.id)

    assert trace.query_id == q.id
    assert trace.dense and trace.bm25 and trace.fused and trace.reranked
    # each stage list is (id, score) pairs
    assert all(isinstance(s, float) for _, s in trace.reranked)
    # the relevant doc is retrieved and surfaces at the top after reranking
    assert "d1" in {i for i, _ in trace.reranked}
    assert trace.reranked[0][0] == "d1"


def test_retrieve_text_adhoc_no_qrels(tiny_dir):
    r, cfg, _ = _retriever(tiny_dir, ["dense", "bm25"])
    trace = r.retrieve_text(cfg.embedders[0], "canine puppy bark")  # not a dataset query obj
    assert trace.relevant == set()  # nothing to judge against
    assert trace.dense and trace.fused
    assert not trace.reranked  # no rerank stage


def test_dense_only_has_no_bm25_or_rerank(tiny_dir):
    r, cfg, ds = _retriever(tiny_dir, ["dense"])
    q = ds.queries[2]
    trace = r.retrieve_text(cfg.embedders[0], q.text, relevant=ds.qrels[q.id])
    assert trace.dense and not trace.bm25 and not trace.reranked
    assert trace.fused == trace.dense  # single branch, no fusion


def test_store_reused_across_dataset_and_adhoc(tiny_dir):
    # dataset ranking and ad-hoc retrieval of the same text agree (same store).
    r, cfg, ds = _retriever(tiny_dir, ["dense"])
    q = ds.queries[0]
    batch = r.rankings(cfg.embedders[0], ["dense"])[q.id]
    adhoc = r.retrieve_text(cfg.embedders[0], q.text).dense
    assert [i for i, _ in adhoc][: len(batch)] == batch
