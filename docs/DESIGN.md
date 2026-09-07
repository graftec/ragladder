# Design & architecture

## Principles

1. **Scalpel, not suite.** One question answered well: *which retrieval stage earns its keep, and is context precision suffering?*
2. **The tool measures; it does not ingest.** Input is always the BEIR JSONL contract. No domain parsing or chunking in the core.
3. **Config-first, CLI-driven.** A `study.yaml` defines the experiment; the CLI runs it. A thin Python API sits underneath.
4. **Everything swappable is a named adapter** selected in config: embedder, reranker, store. Same interface + registry pattern for all three.
5. **Exact by default.** Retrieval uses brute-force exact nearest-neighbor so an approximate index never confounds the metric.

## Target package layout

```
ragladder/
  __init__.py
  cli.py                 # run / compare / inspect / stats (typer or click)
  config.py              # load & validate study.yaml
  registry.py            # name -> adapter class, for embedders/rerankers/stores
  data/
    beir.py              # load corpus.jsonl / queries.jsonl / qrels
  adapters/
    embedders/
      base.py            # Embedder interface
      sentence_transformers.py
      voyage.py
      openai.py
    rerankers/
      base.py            # Reranker interface
      cross_encoder.py
    stores/
      base.py            # Store interface
      inmemory_exact.py  # default: numpy / faiss-flat exact search
      chroma.py          # optional
  pipeline/
    dense.py             # embedding retrieval
    bm25.py              # rank-bm25 lexical retrieval
    fusion.py            # reciprocal rank fusion (k=60)
    rerank.py            # apply reranker to merged candidates
    runner.py            # compose stages per config; build the ladder
  eval/
    metrics.py           # recall@k, mrr, perfect_retrieval_rate, wrong-doc counts
    ladder.py            # per-stage marginal lift
    ab.py                # a/b diff with per-query rescued/lost
    stats.py             # corpus token lengths + per-model truncation rate
  report/
    tables.py            # rich terminal tables
    html.py              # report.html (ladder chart lives here)
    results.py           # results.json (machine-readable, CI-friendly)
```

## Adapter interfaces (conceptual)

### Embedder
The core knows only "text in, vectors out." The adapter hides local-vs-API, query/doc prefixes, dimension, batching, rate limits.

```python
class Embedder:
    name: str
    dim: int                      # native output dimension (reported, not tuned)
    max_input_tokens: int         # for truncation stats
    def embed(self, texts: list[str], kind: str) -> np.ndarray: ...
        # kind in {"query", "document"}; adapter adds e5 prefixes etc.
```

Examples: `sentence_transformers` (e5, bge-m3 — local), `voyage` (voyage-law-2 — API), `openai`.

### Reranker
Cross-encoder: reads query+doc *together*, scores relevance, re-sorts.

```python
class Reranker:
    name: str
    def rerank(self, query: str, candidates: list[Doc]) -> list[Doc]: ...
```

Example: `cross_encoder` (`BAAI/bge-reranker-v2-m3`).

### Store
Holds document vectors and returns nearest by similarity.

```python
class Store:
    def index(self, ids: list[str], vectors: np.ndarray) -> None: ...
    def search(self, query_vector: np.ndarray, k: int) -> list[tuple[str, float]]: ...
```

Backends: `inmemory_exact` (default — cosine over an L2-normalized matrix; exact), `chroma` (optional). Other vector DBs are a documented extension point, not built at launch.

## Retrieval dataflow (single run)

```
query
 ├─► dense (Store.search over embedded corpus)  → list A
 └─► bm25  (rank-bm25 over tokenized corpus)     → list B     (A and B computed in parallel)
        │
     fuse: reciprocal rank fusion (k=60)  →  ONE merged, deduplicated candidate list (~N_RERANK)
        │
     rerank: cross-encoder re-sorts the merged list  (serial; needs candidates first)
        │
     top-k  →  metrics
```

- dense and bm25 are **parallel, independent** branches over the *same* corpus.
- fusion merges into **one** list; duplicates (found by both) collapse to a single entry.
- the reranker operates on that **single merged list**, not on A and B separately.

## The ladder vs execution — don't conflate

- **Execution** (one config): the dataflow above.
- **Ablation ladder** (comparing configs): run with `[dense]`, then `[dense, bm25]`, then `[dense, bm25, rerank]`, reporting the metric at each rung to attribute the marginal lift. Three runs, not one; not the execution order.

## Exact-search rationale

Production vector DBs use **approximate** nearest-neighbor (HNSW/IVF) for speed, trading a little recall. In *evaluation* that approximation is a confound: a recall drop could be the embedding model or the index. Exact search removes the index as a variable. At demo scale (~1,500 docs) exact search is milliseconds; even 100k docs is fast with numpy/faiss-flat. Hence exact is the principled default and part of the tool's identity.

## Caching

Embedding the corpus is the main cost. Cache vectors to disk keyed by `(embedder.name, corpus content hash)` so re-runs and ladders don't recompute. Deterministic; no randomness in the eval path.

## Two "k"s (implementation note)

There are really two cutoffs: how many candidates the first stage passes to the reranker (`n_rerank`, e.g. 50) and the final `k` used for metrics (e.g. 10). Config exposes `k`; `n_rerank` can default (e.g. 50) and be overridable later.

## Non-goals (launch)
- No multi-vector-DB backends beyond in-memory + Chroma.
- No document ingestion / chunking in the core.
- No generation, no LLM-as-judge.
- No enrichment or query reformulation stages (deferred — see ROADMAP).
- No embedding-dimension tuning (report native dim only).
