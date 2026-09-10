"""results.json now carries corpus_stats + diagnostics; must stay valid JSON."""

import json

from ragladder.config import StudyConfig
from ragladder.data import load_dataset
from ragladder.eval.diagnostics import rerank_threshold_sweep, stage_attribution
from ragladder.eval.stats import corpus_length_stats, truncation_stats
from ragladder.pipeline.runner import Retriever, run_study
from ragladder.report.results import run_to_dict


def _setup(tiny_dir):
    cfg = StudyConfig.from_yaml(tiny_dir / "study.yaml")
    cfg.pipeline = ["dense", "bm25", "rerank"]
    cfg.reranker = {"type": "lexical"}
    ds = load_dataset(tiny_dir / "corpus.jsonl", tiny_dir / "queries.jsonl", tiny_dir / "qrels.jsonl")
    r = Retriever(cfg, ds, cache_dir=None)
    result = run_study(cfg, retriever=r)
    return cfg, ds, r, result


def test_results_json_without_extras_has_null_sections(tiny_dir):
    _, _, _, result = _setup(tiny_dir)
    d = run_to_dict(result)
    assert d["corpus_stats"] is None
    assert d["diagnostics"] is None


def test_results_json_with_stats_and_diagnostics(tiny_dir):
    cfg, ds, r, result = _setup(tiny_dir)
    emb = cfg.embedders[0]

    length_stats = corpus_length_stats(ds.corpus)
    truncation = truncation_stats(ds.corpus, [(emb.name, r.embedder(emb))])
    traces = r.traces(emb)
    diag = {emb.name: (stage_attribution(traces, r.metric_k, cfg.n_rerank),
                       rerank_threshold_sweep(traces, r.metric_k))}

    d = run_to_dict(result, length_stats=length_stats, truncation=truncation, diagnostics=diag)

    # corpus stats
    assert d["corpus_stats"]["length"]["words"]["count"] == 10
    assert d["corpus_stats"]["truncation"][0]["embedder"] == "bow"

    # diagnostics
    diag_out = d["diagnostics"][emb.name]
    assert "stage_attribution" in diag_out
    assert diag_out["stage_attribution"]["source"]["dense_only"] >= 0

    # the no-floor baseline threshold (-inf) must serialize as null, not Infinity
    rows = diag_out["reranker_threshold_sweep"]["rows"]
    assert rows[0]["threshold"] is None

    # whole thing must be strictly valid JSON (no NaN/Infinity)
    text = json.dumps(d, allow_nan=False)
    assert "Infinity" not in text
