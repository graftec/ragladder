"""Compose stages per config and build the ablation ladder.

Two concepts that the README is emphatic must not be conflated:

  * Execution (one config): dense ∥ bm25 -> fuse (RRF) -> rerank -> top-k.
  * Ladder (comparing configs): dense, then +bm25, then +rerank — each a
    separate run, to attribute the marginal lift of each stage.

The ladder rungs are the cumulative prefixes of `pipeline`. Each rung executes
the sub-pipeline formed by its stages: the retrieval branches present (dense,
bm25) are fused with RRF when there is more than one, then reranked if the rung
includes `rerank`. Fusion is implicit (triggered by >1 retriever), not a rung of
its own.

`Retriever` holds the per-study fixed work (embedding cache, BM25 index, the
reranker) so `run_study` (the ladder) and `compare` (A/B) share one code path.
"""

from __future__ import annotations

import hashlib
import re
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from ragladder.config import EmbedderConfig, StudyConfig, parse_metric_k, parse_recall_k
from ragladder.data import Dataset, load_dataset
from ragladder.eval.diagnostics import QueryTrace
from ragladder.eval.metrics import MetricResult, evaluate
from ragladder.pipeline.bm25 import BM25Index
from ragladder.pipeline.fusion import reciprocal_rank_fusion
from ragladder.pipeline.rerank import rerank_candidates
from ragladder.registry import get_embedder, get_reranker, get_store

IMPLEMENTED_STAGES = {"dense", "bm25", "rerank"}
RETRIEVAL_STAGES = ("dense", "bm25")


@dataclass
class RungResult:
    stages: list[str]  # cumulative stages at this rung, e.g. ["dense", "bm25"]
    label: str  # the marginal stage: "dense", "+bm25", "+rerank"
    implemented: bool
    metrics: MetricResult | None = None
    note: str = ""


@dataclass
class LadderResult:
    embedder: str
    dim: int
    max_input_tokens: int
    rungs: list[RungResult] = field(default_factory=list)


@dataclass
class RunResult:
    k: int
    metrics_requested: list[str]
    dataset_summary: dict
    prr_k: int = 0
    ladders: list[LadderResult] = field(default_factory=list)


def _ladder_rungs(pipeline: list[str]) -> list[tuple[list[str], str]]:
    """Cumulative prefixes with a label for each rung's marginal stage."""
    rungs: list[tuple[list[str], str]] = []
    for i, stage in enumerate(pipeline):
        stages = pipeline[: i + 1]
        label = stage if i == 0 else f"+{stage}"
        rungs.append((stages, label))
    return rungs


def _cache_path(cache_dir: Path, model: str, doc_prefix: str, corpus_hash: str) -> Path:
    safe_model = re.sub(r"[^A-Za-z0-9._-]", "_", model)
    prefix_tag = hashlib.sha256(doc_prefix.encode("utf-8")).hexdigest()[:8]
    return cache_dir / f"{safe_model}__{prefix_tag}__{corpus_hash}.npy"


class Retriever:
    """Runs the retrieval pipeline for a study; caches the per-study fixed work."""

    def __init__(
        self,
        cfg: StudyConfig,
        dataset: Dataset,
        limit: int | None = None,
        cache_dir: str | Path | None = ".cache/embeddings",
    ):
        self.cfg = cfg
        self.dataset = dataset
        self.corpus = dataset.corpus
        self.queries = dataset.queries[:limit] if limit else dataset.queries
        self.qrels = {q.id: dataset.qrels.get(q.id, set()) for q in self.queries}
        self._corpus_hash = self.corpus.content_hash()
        self._cache_dir = Path(cache_dir) if cache_dir else None

        # recall@k / MRR use `metric_k`; PRR + avg-wrong use `prr_k`, which can be
        # smaller (e.g. "perfect_retrieval_rate@3") since precision matters near
        # the top. Both are parsed from the metrics list, defaulting to cfg.k.
        recall_k = next(
            (parse_recall_k(m, cfg.k) for m in cfg.metrics if m.startswith("recall@")),
            cfg.k,
        )
        self.metric_k = max(cfg.k, recall_k)
        self.prr_k = next(
            (
                v
                for m in cfg.metrics
                if (v := parse_metric_k(m, "perfect_retrieval_rate", cfg.k)) is not None
            ),
            cfg.k,
        )
        # Keep enough candidates for whichever cutoff is larger.
        self.top_n = max(self.metric_k, self.prr_k, cfg.n_rerank)

        if "rerank" in cfg.pipeline and not cfg.reranker:
            raise ValueError("pipeline includes 'rerank' but no reranker is configured")
        self._reranker = _build_reranker(cfg.reranker) if "rerank" in cfg.pipeline else None

        # BM25 is embedder-independent — build the index once; keep it so ad-hoc
        # queries (inspect) can be scored, not just the dataset queries.
        self._bm25_index = BM25Index(self.corpus) if "bm25" in cfg.pipeline else None
        self._bm25 = (
            {q.id: self._bm25_index.search(q.text, self.top_n) for q in self.queries}
            if self._bm25_index is not None
            else {}
        )
        self._embedders: dict[str, object] = {}
        self._stores: dict[str, tuple] = {}  # name -> (embedder, indexed store)
        self._dense: dict[str, dict[str, list[tuple[str, float]]]] = {}

    def embedder(self, emb_cfg: EmbedderConfig):
        if emb_cfg.name not in self._embedders:
            self._embedders[emb_cfg.name] = get_embedder(emb_cfg.type)(emb_cfg)
        return self._embedders[emb_cfg.name]

    def _dense_state(self, emb_cfg: EmbedderConfig):
        """Return (embedder, corpus-indexed store), built once and cached so the
        same store serves the whole dataset AND ad-hoc inspect queries."""
        if emb_cfg.name in self._stores:
            return self._stores[emb_cfg.name]
        embedder = self.embedder(emb_cfg)
        corpus_vectors = _embed_corpus(
            embedder, self.corpus, emb_cfg.doc_prefix, self._corpus_hash, self._cache_dir
        )
        store_type = self.cfg.store.get("type", "inmemory_exact")
        store = get_store(store_type)(
            **{k: v for k, v in self.cfg.store.items() if k != "type"}
        )
        store.index(self.corpus.ids, corpus_vectors)
        self._stores[emb_cfg.name] = (embedder, store)
        return embedder, store

    def _dense_candidates(self, emb_cfg: EmbedderConfig):
        if emb_cfg.name in self._dense:
            return self._dense[emb_cfg.name]
        embedder, store = self._dense_state(emb_cfg)
        query_vectors = embedder.embed([q.text for q in self.queries], kind="query")
        cands = {
            q.id: store.search(qv, self.top_n)
            for q, qv in zip(self.queries, query_vectors)
        }
        self._dense[emb_cfg.name] = cands
        return cands

    def retrieve_text(
        self,
        emb_cfg: EmbedderConfig,
        text: str,
        relevant: set[str] | None = None,
        stages: list[str] | None = None,
        query_id: str = "(ad-hoc)",
    ) -> QueryTrace:
        """Run the pipeline for ONE arbitrary query text — the engine of `inspect`.

        Works for a query in the dataset (pass its qrels as `relevant`) or a
        brand-new question typed at the CLI (no `relevant` → nothing marked wrong).
        """
        stages = stages or self.cfg.pipeline
        active = [s for s in RETRIEVAL_STAGES if s in stages]
        fusion_k = int(self.cfg.fusion.get("k", 60))

        dense_list: list[tuple[str, float]] = []
        bm25_list: list[tuple[str, float]] = []
        branches: list[list[tuple[str, float]]] = []
        if "dense" in active:
            embedder, store = self._dense_state(emb_cfg)
            qv = embedder.embed([text], kind="query")[0]
            dense_list = store.search(qv, self.top_n)
            branches.append(dense_list)
        if "bm25" in active and self._bm25_index is not None:
            bm25_list = self._bm25_index.search(text, self.top_n)
            branches.append(bm25_list)

        if len(branches) == 1:
            fused = branches[0]
        elif branches:
            fused = reciprocal_rank_fusion(
                [[i for i, _ in b] for b in branches], k=fusion_k
            )
        else:
            fused = []

        reranked: list[tuple[str, float]] = []
        if "rerank" in stages and self._reranker is not None:
            reranked = rerank_candidates(
                self._reranker, self.corpus, text, fused, self.cfg.n_rerank
            )

        return QueryTrace(
            query_id=query_id,
            relevant=relevant or set(),
            dense=dense_list,
            bm25=bm25_list,
            fused=fused,
            reranked=reranked,
        )

    def rankings(self, emb_cfg: EmbedderConfig, stages: list[str]) -> dict[str, list[str]]:
        """Per-query ranked doc ids for one embedder through the given stages."""
        dense = self._dense_candidates(emb_cfg)
        return _rankings_for_rung(
            stages=stages,
            queries=self.queries,
            corpus=self.corpus,
            dense_candidates=dense,
            bm25_candidates=self._bm25,
            fusion_k=int(self.cfg.fusion.get("k", 60)),
            reranker=self._reranker,
            n_rerank=self.cfg.n_rerank,
            metric_k=max(self.metric_k, self.prr_k),  # keep enough for both cutoffs
        )

    def traces(self, emb_cfg: EmbedderConfig, stages: list[str] | None = None) -> list[QueryTrace]:
        """Per-query, per-stage results *with scores* — the input to diagnostics.

        Unlike `rankings` (ids only), this keeps the dense/BM25/fused/reranked
        candidates and their scores so score-threshold and attribution analyses
        can run. Defaults to the full configured pipeline.
        """
        stages = stages or self.cfg.pipeline
        dense = self._dense_candidates(emb_cfg)
        active = [s for s in RETRIEVAL_STAGES if s in stages]
        do_rerank = "rerank" in stages
        fusion_k = int(self.cfg.fusion.get("k", 60))

        out: list[QueryTrace] = []
        for q in self.queries:
            dense_list = dense.get(q.id, []) if "dense" in active else []
            bm25_list = self._bm25.get(q.id, []) if "bm25" in active else []

            branches = []
            if "dense" in active:
                branches.append(dense_list)
            if "bm25" in active:
                branches.append(bm25_list)
            if len(branches) == 1:
                fused = branches[0]
            else:
                fused = reciprocal_rank_fusion(
                    [[i for i, _ in b] for b in branches], k=fusion_k
                )

            reranked: list[tuple[str, float]] = []
            if do_rerank:
                reranked = rerank_candidates(
                    self._reranker, self.corpus, q.text, fused, self.cfg.n_rerank
                )

            out.append(
                QueryTrace(
                    query_id=q.id,
                    relevant=self.qrels.get(q.id, set()),
                    dense=dense_list,
                    bm25=bm25_list,
                    fused=fused,
                    reranked=reranked,
                )
            )
        return out


def run_study(
    cfg: StudyConfig,
    dataset: Dataset | None = None,
    limit: int | None = None,
    cache_dir: str | Path | None = ".cache/embeddings",
    retriever: Retriever | None = None,
    progress: Callable[[str], None] | None = None,
) -> RunResult:
    """Run the study. `progress`, if given, is called with human-readable status
    strings as each embedder and ladder rung is processed — useful for long runs."""
    def _tell(msg: str) -> None:
        if progress:
            progress(msg)

    if retriever is None:
        if dataset is None:
            dataset = load_dataset(cfg.corpus, cfg.queries, cfg.qrels)
        retriever = Retriever(cfg, dataset, limit=limit, cache_dir=cache_dir)

    result = RunResult(
        k=retriever.metric_k,
        prr_k=retriever.prr_k,
        metrics_requested=cfg.metrics,
        dataset_summary={
            "corpus_size": len(retriever.corpus),
            "n_queries": len(retriever.queries),
            "n_judged": sum(1 for rel in retriever.qrels.values() if rel),
        },
    )

    n_emb = len(cfg.embedders)
    for i, emb_cfg in enumerate(cfg.embedders, start=1):
        tag = f"[{i}/{n_emb}] {emb_cfg.name}"
        _tell(f"{tag}: embedding corpus + dense search")
        embedder = retriever.embedder(emb_cfg)
        # Ensure dense (and thus the model's dim) is populated before reading dim.
        retriever._dense_candidates(emb_cfg)
        ladder = LadderResult(
            embedder=emb_cfg.name,
            dim=getattr(embedder, "dim", 0),
            max_input_tokens=getattr(embedder, "max_input_tokens", 0),
        )

        for stages, label in _ladder_rungs(cfg.pipeline):
            unimplemented = sorted(set(stages) - IMPLEMENTED_STAGES)
            if unimplemented:
                ladder.rungs.append(
                    RungResult(
                        stages=stages,
                        label=label,
                        implemented=False,
                        note=f"stage(s) not implemented yet: {', '.join(unimplemented)}",
                    )
                )
                continue
            _tell(f"{tag}: rung {' → '.join(stages)}")
            rankings = retriever.rankings(emb_cfg, stages)
            metrics = evaluate(
                rankings, retriever.qrels, retriever.metric_k, prr_k=retriever.prr_k
            )
            ladder.rungs.append(
                RungResult(stages=stages, label=label, implemented=True, metrics=metrics)
            )

        best = next((r.metrics for r in reversed(ladder.rungs) if r.metrics), None)
        done = f" (recall@{result.k}={best.recall_at_k:.3f})" if best else ""
        _tell(f"{tag}: done{done}")
        result.ladders.append(ladder)

    return result


def _rankings_for_rung(
    stages: list[str],
    queries,
    corpus,
    dense_candidates: dict[str, list[tuple[str, float]]],
    bm25_candidates: dict[str, list[tuple[str, float]]],
    fusion_k: int,
    reranker,
    n_rerank: int,
    metric_k: int,
) -> dict[str, list[str]]:
    active_retrievers = [s for s in RETRIEVAL_STAGES if s in stages]
    do_rerank = "rerank" in stages
    rankings: dict[str, list[str]] = {}

    for q in queries:
        branches: list[list[tuple[str, float]]] = []
        if "dense" in active_retrievers:
            branches.append(dense_candidates.get(q.id, []))
        if "bm25" in active_retrievers:
            branches.append(bm25_candidates.get(q.id, []))

        if len(branches) == 1:
            merged = branches[0]
        else:
            merged = reciprocal_rank_fusion(
                [[doc_id for doc_id, _ in lst] for lst in branches], k=fusion_k
            )

        if do_rerank:
            merged = rerank_candidates(reranker, corpus, q.text, merged, n_rerank)

        rankings[q.id] = [doc_id for doc_id, _ in merged][:metric_k]

    return rankings


def _build_reranker(reranker_cfg: dict):
    cfg = dict(reranker_cfg)
    rtype = cfg.pop("type")
    return get_reranker(rtype)(**cfg)


def _embed_corpus(embedder, corpus, doc_prefix, corpus_hash, cache_dir: Path | None) -> np.ndarray:
    model_id = getattr(getattr(embedder, "cfg", None), "model", None) or embedder.name
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
        path = _cache_path(cache_dir, model_id, doc_prefix, corpus_hash)
        if path.is_file():
            vecs = np.load(path)
            # Populate dim if the adapter hasn't loaded its model.
            if not getattr(embedder, "dim", 0):
                embedder.dim = int(vecs.shape[1])
            return vecs

    vecs = embedder.embed(corpus.documents, kind="document")
    vecs = np.asarray(vecs, dtype=np.float32)
    if cache_dir is not None:
        np.save(path, vecs)
    return vecs
