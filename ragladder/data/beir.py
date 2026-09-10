"""BEIR-compatible dataset loading.

A dataset is three files (see the README):
  - corpus.jsonl   : {"_id", "title"?, "text", "metadata"?}   the retrievable units
  - queries.jsonl  : {"_id", "text", "metadata"?}             the questions
  - qrels          : which corpus ids are relevant per query  (TSV or JSONL)

The tool never parses domain sources; converting a corpus *into* this format is
an upstream, domain-specific step (see the demo-side converter under data/).
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Corpus:
    ids: list[str]
    titles: dict[str, str]
    texts: dict[str, str]
    metadata: dict[str, dict] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def documents(self) -> list[str]:
        """Texts in `ids` order — the strings that get embedded."""
        return [self.texts[i] for i in self.ids]

    def content_hash(self) -> str:
        """Stable hash of (id, text) pairs — the cache key for embeddings."""
        h = hashlib.sha256()
        for i in self.ids:
            h.update(i.encode("utf-8"))
            h.update(b"\x00")
            h.update(self.texts[i].encode("utf-8"))
            h.update(b"\x00")
        return h.hexdigest()[:16]


@dataclass
class Query:
    id: str
    text: str
    metadata: dict = field(default_factory=dict)


@dataclass
class Dataset:
    corpus: Corpus
    queries: list[Query]
    qrels: dict[str, set[str]]  # query_id -> set of relevant corpus ids

    def validate(self) -> None:
        """Fail loud on the silent-breakage cases the docs warn about."""
        corpus_ids = set(self.corpus.ids)
        if len(corpus_ids) != len(self.corpus.ids):
            raise ValueError("corpus contains duplicate _id values")

        query_ids = {q.id for q in self.queries}
        unknown_queries = set(self.qrels) - query_ids
        if unknown_queries:
            raise ValueError(
                f"qrels reference {len(unknown_queries)} unknown query id(s), "
                f"e.g. {sorted(unknown_queries)[:3]}"
            )

        dangling = {
            doc_id
            for rels in self.qrels.values()
            for doc_id in rels
            if doc_id not in corpus_ids
        }
        if dangling:
            raise ValueError(
                f"qrels reference {len(dangling)} corpus id(s) not in the corpus, "
                f"e.g. {sorted(dangling)[:3]} — stable ids are the linchpin"
            )


def load_dataset(corpus_path: Path, queries_path: Path, qrels_path: Path) -> Dataset:
    corpus = _load_corpus(Path(corpus_path))
    queries = _load_queries(Path(queries_path))
    qrels = _load_qrels(Path(qrels_path))
    ds = Dataset(corpus=corpus, queries=queries, qrels=qrels)
    ds.validate()
    return ds


def _iter_jsonl(path: Path):
    if not path.is_file():
        raise FileNotFoundError(f"dataset file not found: {path}")
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"{path}:{lineno}: invalid JSON ({e})") from e


def _load_corpus(path: Path) -> Corpus:
    ids: list[str] = []
    titles: dict[str, str] = {}
    texts: dict[str, str] = {}
    metadata: dict[str, dict] = {}
    seen: set[str] = set()
    for obj in _iter_jsonl(path):
        doc_id = obj.get("_id")
        if doc_id is None or "text" not in obj:
            raise ValueError(f"{path}: corpus entry needs '_id' and 'text': {obj!r}")
        doc_id = str(doc_id)
        if doc_id in seen:
            raise ValueError(f"{path}: duplicate corpus _id {doc_id!r}")
        seen.add(doc_id)
        ids.append(doc_id)
        texts[doc_id] = obj["text"]
        if obj.get("title"):
            titles[doc_id] = obj["title"]
        if obj.get("metadata"):
            metadata[doc_id] = obj["metadata"]
    if not ids:
        raise ValueError(f"{path}: corpus is empty")
    return Corpus(ids=ids, titles=titles, texts=texts, metadata=metadata)


def _load_queries(path: Path) -> list[Query]:
    queries: list[Query] = []
    seen: set[str] = set()
    for obj in _iter_jsonl(path):
        q_id = obj.get("_id")
        if q_id is None or "text" not in obj:
            raise ValueError(f"{path}: query entry needs '_id' and 'text': {obj!r}")
        q_id = str(q_id)
        if q_id in seen:
            raise ValueError(f"{path}: duplicate query _id {q_id!r}")
        seen.add(q_id)
        queries.append(Query(id=q_id, text=obj["text"], metadata=obj.get("metadata", {})))
    if not queries:
        raise ValueError(f"{path}: no queries loaded")
    return queries


def _load_qrels(path: Path) -> dict[str, set[str]]:
    """Load qrels from either JSONL ({query_id, relevant:[...]}) or BEIR TSV."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"qrels file not found: {path}")
    if path.suffix.lower() in {".tsv", ".txt"}:
        return _load_qrels_tsv(path)
    return _load_qrels_jsonl(path)


def _load_qrels_jsonl(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    for obj in _iter_jsonl(path):
        q_id = obj.get("query_id", obj.get("_id"))
        rel = obj.get("relevant")
        if q_id is None or rel is None:
            raise ValueError(
                f"{path}: qrels entry needs 'query_id' and 'relevant': {obj!r}"
            )
        qrels.setdefault(str(q_id), set()).update(str(r) for r in rel)
    if not qrels:
        raise ValueError(f"{path}: no qrels loaded")
    return qrels


def _load_qrels_tsv(path: Path) -> dict[str, set[str]]:
    qrels: dict[str, set[str]] = {}
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            line = line.strip()
            if not line:
                continue
            parts = line.split("\t")
            if parts[0].lower() in {"query-id", "query_id", "qid"}:  # header
                continue
            if len(parts) < 3:
                raise ValueError(f"{path}:{lineno}: expected 3 TSV columns, got {parts!r}")
            q_id, doc_id, score = parts[0], parts[1], parts[2]
            try:
                relevant = float(score) > 0
            except ValueError as e:
                raise ValueError(f"{path}:{lineno}: non-numeric score {score!r}") from e
            if relevant:
                qrels.setdefault(str(q_id), set()).add(str(doc_id))
    if not qrels:
        raise ValueError(f"{path}: no relevant judgments in qrels")
    return qrels
