# Roadmap

Scope is deliberately staged. The launch is a **scalpel**: embedding-model comparison + ablation ladder + wrong-context metrics on raw documents. Everything else is a follow-up (which doubles as future article content — see `POSITIONING.md`).

## v0 — scaffold (current)
- [x] Project skeleton, docs, positioning, design specs, `CLAUDE.md`
- [ ] `pyproject.toml` finalized, dependencies pinned
- [ ] Empty package modules created

## v1 — launch (MVP)

Core loop, mostly generalizing/packaging existing scripts from the origin project.

- [ ] **Data layer**: BEIR loaders (corpus/queries/qrels jsonl + qrels tsv).
- [ ] **Demo dataset**: `data/build_or_dataset.py` converts SR-220 XML → BEIR jsonl (demo-side, not in package). Ship `or-corpus/queries/qrels`.
- [ ] **Embedder adapters**: `sentence_transformers` (e5, bge-m3), `voyage` (voyage-law-2). Registry + config selection. Handle e5 query/doc prefixes.
- [ ] **Store**: `inmemory_exact` (numpy cosine over L2-normalized matrix). Embedding cache keyed by (model, corpus hash).
- [ ] **Pipeline stages**: `dense`, `bm25` (rank-bm25), `fusion` (RRF k=60), `rerank` (cross-encoder bge-reranker-v2-m3).
- [ ] **Metrics**: recall@k, MRR, wrong-doc count/rate, Perfect Retrieval Rate.
- [ ] **Ladder**: per-stage marginal lift.
- [ ] **A/B diff**: per-query rescued/lost.
- [ ] **`stats`**: token-length distribution + per-model truncation rate.
- [ ] **CLI**: `run`, `compare`, `inspect`, `stats` (typer/click).
- [ ] **Reports**: rich terminal tables, `results.json`, `report.html` (ladder chart).
- [ ] **Docs**: `DEVELOPMENT.md`, finalize README quick-start against real commands.
- [ ] **The article**: run the OR study, write up the hero finding.

### Explicitly OUT of v1
- Multi-vector-DB backends (Qdrant/pgvector/Weaviate/Pinecone). In-memory + optional Chroma only.
- Document ingestion / chunking in the core (demo-side only).
- Generation / LLM-as-judge.
- Enrichment and query reformulation stages.
- Embedding-dimension tuning (report native dim only).
- CI quality gate (`--gate`).

## v2+ — follow-ups (each a potential post)

Prioritize by article value; add behind the existing adapter/stage interfaces so the core stays a scalpel.

1. **Query reformulation** stage (LLM rewrites the query into juristic anchors). The origin project's biggest win (MRR ~0.30 → ~0.45). Post: *"Can query reformulation close the fact-pattern↔doctrine gap?"*
2. **Embedding enrichment** experiment (embed LLM-generated summaries/questions/topics vs raw text). Prior data suggests it barely helps — a counterintuitive post.
3. **LoRA-fine-tuned reranker** vs off-the-shelf (hard-negative mining exists in the origin project). Post: *"Does fine-tuning a reranker beat off-the-shelf on legal text?"* Training needs a rented GPU, not the laptop.
4. **Dimension & quantization** cost-vs-quality sweep (Matryoshka truncation on MRL models only — voyage-3 / OpenAI-3; never naive-truncate e5/bge-m3; + int8/binary). Post: *"Halve your vectors for ~2% quality loss."*
5. **More embedder/reranker/store adapters** (OpenAI, Cohere, Qdrant, pgvector) as demand appears — ragtune-parity, low priority.
6. **CI quality gate** + bootstrap confidence intervals for the metrics.
7. **Results viewer** (richer HTML / small web UI) if the report.html proves popular.

## Cleanup carried from the origin project
- Strip any secrets / API keys; use env vars (`VOYAGE_API_KEY`, `ANTHROPIC_API_KEY`) only.
- Remove internal absolute paths (the origin eval scripts hardcoded an external `OR_Aufgaben_und_Lösungen.md`).
- Reconcile the origin project's two metric frameworks (hit@k/MRR vs "Perfect Retrieval Rate") into the single family in `METRICS.md`.
