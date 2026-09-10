"""Load and validate a `study.yaml` into typed config objects.

Config-first is a core principle (see the README): a study is defined in
YAML, the CLI just executes it. This module is the single place that knows the
on-disk shape of that file, so the rest of the package works with dataclasses.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class EmbedderConfig:
    """One embedding model to evaluate. Each becomes a column in the report."""

    name: str  # display name, also the A/B selector (e.g. "e5-large")
    type: str  # adapter type resolved via the registry (e.g. "sentence_transformers")
    model: str  # the underlying model id (e.g. "intfloat/multilingual-e5-large")
    query_prefix: str = ""  # e.g. E5's "query: "
    doc_prefix: str = ""  # e.g. E5's "passage: "
    api_key_env: str | None = None  # env var holding the API key, for API models
    options: dict = field(default_factory=dict)  # adapter-specific extras


@dataclass
class StudyConfig:
    """A whole experiment: dataset + models + pipeline + metrics."""

    corpus: Path
    queries: Path
    qrels: Path
    embedders: list[EmbedderConfig]
    pipeline: list[str]  # e.g. ["dense", "bm25", "rerank"]; prefixes form the ladder
    store: dict
    fusion: dict
    reranker: dict | None
    metrics: list[str]
    k: int
    n_rerank: int
    source_path: Path  # where this config was loaded from (for path resolution)

    @classmethod
    def from_yaml(cls, path: str | Path) -> StudyConfig:
        path = Path(path).resolve()
        if not path.is_file():
            raise FileNotFoundError(f"study config not found: {path}")
        raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
        return cls.from_dict(raw, source_path=path)

    @classmethod
    def from_dict(cls, raw: dict, source_path: Path) -> StudyConfig:
        base_dir = source_path.parent

        for required in ("corpus", "queries", "qrels", "embedders"):
            if required not in raw:
                raise ValueError(f"study config missing required key: {required!r}")

        embedders = [_parse_embedder(e) for e in raw["embedders"]]
        if not embedders:
            raise ValueError("study config needs at least one embedder")

        pipeline = list(raw.get("pipeline", ["dense"]))
        if not pipeline:
            raise ValueError("pipeline must list at least one stage")
        if pipeline[0] != "dense":
            # dense is the base rung of the ablation ladder; everything fuses onto it.
            raise ValueError(
                f"pipeline must start with 'dense' (the base rung); got {pipeline!r}"
            )

        return cls(
            corpus=_resolve(raw["corpus"], base_dir),
            queries=_resolve(raw["queries"], base_dir),
            qrels=_resolve(raw["qrels"], base_dir),
            embedders=embedders,
            pipeline=pipeline,
            store=dict(raw.get("store", {"type": "inmemory_exact"})),
            fusion=dict(raw.get("fusion", {"type": "rrf", "k": 60})),
            reranker=dict(raw["reranker"]) if raw.get("reranker") else None,
            metrics=list(raw.get("metrics", ["recall@10", "mrr", "perfect_retrieval_rate"])),
            k=int(raw.get("k", 10)),
            n_rerank=int(raw.get("n_rerank", 50)),
            source_path=source_path,
        )


def _parse_embedder(raw: dict) -> EmbedderConfig:
    known = {"name", "type", "model", "query_prefix", "doc_prefix", "api_key_env"}
    for required in ("name", "type"):
        if required not in raw:
            raise ValueError(f"embedder entry missing required key {required!r}: {raw!r}")
    return EmbedderConfig(
        name=raw["name"],
        type=raw["type"],
        model=raw.get("model", raw["name"]),
        query_prefix=raw.get("query_prefix", ""),
        doc_prefix=raw.get("doc_prefix", ""),
        api_key_env=raw.get("api_key_env"),
        options={k: v for k, v in raw.items() if k not in known},
    )


def _resolve(rel: str, base_dir: Path) -> Path:
    """Resolve a dataset path.

    Study configs are written with dataset paths that may be relative either to
    the config file itself (self-contained fixtures) or to the working directory
    (the demo run from repo root). Try both; prefer whichever exists.
    """
    p = Path(rel)
    if p.is_absolute():
        return p
    candidates = [base_dir / p, Path.cwd() / p]
    for c in candidates:
        if c.exists():
            return c.resolve()
    # Fall back to the config-relative path so error messages point somewhere sane.
    return (base_dir / p).resolve()


def parse_recall_k(metric: str, default_k: int) -> int:
    """Extract N from a 'recall@N' metric string, falling back to default_k."""
    m = re.fullmatch(r"recall@(\d+)", metric.strip())
    return int(m.group(1)) if m else default_k


def parse_metric_k(metric: str, name: str, default_k: int) -> int | None:
    """Extract N from a '<name>@N' metric string; None if it doesn't match `name`.

    Accepts both the bare name (uses default_k) and '<name>@N'. Returns None for
    unrelated metric strings so callers can tell "not this metric" from "@k".
    """
    metric = metric.strip()
    if metric == name:
        return default_k
    m = re.fullmatch(rf"{re.escape(name)}@(\d+)", metric)
    return int(m.group(1)) if m else None
