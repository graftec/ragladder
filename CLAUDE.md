# CLAUDE.md — context for AI-assisted development of `ragladder`

This file orients a Claude Code session working in this repository. Read it first.

## What this project is

`ragladder` is a small, opinionated **RAG retrieval-layer evaluation tool**. It is being extracted from a larger private project (a Swiss-law RAG server) to serve as a **portfolio / thought-leadership artifact** that positions the author as a RAG specialist (public repo + an accompanying article). The Swiss legal corpus is only the *demo dataset*; the tool itself is domain-agnostic.

## The one-sentence definition

> A domain-agnostic RAG retrieval evaluation harness whose signature is (a) **stage-by-stage ablation** — "which pipeline stage earns its keep" — and (b) **precision-first metrics** that measure wrong-context, not just recall; shipped with Swiss legal German (Obligationenrecht) as the bundled demo dataset.

## The wedge (what makes it NOT a ragtune clone)

- **Stage attribution / the "ablation ladder"** is the hero feature: add one retrieval stage at a time (dense → +BM25 → +rerank) and report the *marginal lift* of each. Most tools score one pipeline; this attributes value per stage.
- **Precision / wrong-context metrics**: a `Perfect Retrieval Rate` (share of queries with zero wrong docs) and wrong-document counts, alongside recall@k / MRR. The insight: a single wrong doc in context causes confident hallucination.
- **Exact search by default**: retrieval uses brute-force exact nearest-neighbor (numpy / faiss-flat), so approximate-index (ANN) recall loss never confounds the measurement. This is a *principled* choice, framed as a feature.

## Guardrails — keep it a scalpel, not a suite

These are deliberate scope decisions. Do **not** casually violate them; if a change seems to require it, flag it.

1. **No multi-vector-DB support at launch.** Default backend is in-memory exact search; ChromaDB is an *optional* adapter. Qdrant/pgvector/Weaviate/etc. are a documented extension point, not built now. Rationale: connector breadth is ragtune's turf and not our differentiator.
2. **The tool does NOT ingest domain sources.** No XML/PDF/HTML parsing, no chunking pipeline in the core. Input is always the BEIR JSONL contract (corpus/queries/qrels). Domain conversion (e.g. OR XML → JSONL) is a *demo-side* script under `examples/` or `data/`, never in the `ragladder` package.
3. **Config-first.** A study is defined in `study.yaml` (reproducible, versionable). The CLI executes it. Don't push substantive knobs into CLI flags.
4. **Everything swappable is a named adapter** chosen in config: embedders, reranker, store. One consistent interface + registry pattern for all three.
5. **Launch scope excludes** enrichment (embedding LLM-generated text) and query reformulation (LLM rewriting the query). Those were part of the original project and are deliberately deferred to follow-up posts. Keep the launch about the *standard* retrieval stack on *raw* documents.
6. **Do not add generation.** This evaluates retrieval only. No LLM answer generation, no LLM-as-judge. (That's Ragas/DeepEval territory.)

## Architecture (target)

```
ragladder/
  adapters/
    embedders/     # sentence_transformers, voyage, openai, ... — one method: embed(texts, kind)
    rerankers/     # cross_encoder, ...
    stores/        # inmemory_exact (default), chroma (optional)
  pipeline/        # dense, bm25, fusion (RRF), rerank — stages composed per config
  eval/            # metrics (recall@k, mrr, perfect_retrieval_rate, wrong-doc), ladder, a/b diff
  data/            # BEIR loaders (corpus/queries/qrels jsonl)
  report/          # terminal tables (rich), results.json, report.html
  cli.py           # run / compare / inspect / stats
```

See `docs/DESIGN.md` for the adapter interfaces and retrieval dataflow, and `docs/` generally for the authoritative specs.

## Retrieval dataflow (one run)

`dense ∥ bm25` run in **parallel** (each searches the whole corpus) → **fuse** (Reciprocal Rank Fusion, k=60) into **one** merged, deduplicated candidate list → **rerank** (cross-encoder, runs in series on the merged list) → top-k → metrics.

Note the distinction:
- **Execution** (one run): parallel-then-serial as above.
- **Ladder** (comparing runs): dense only, then dense+bm25, then dense+bm25+rerank — three configs measured to attribute lift. The ladder is NOT the execution order.

## Key technical decisions already made

- **Embedding model** is the primary comparison axis. Demo compares `intfloat/multilingual-e5-large` (baseline), `voyage-law-2` (API, legal-domain), `BAAI/bge-m3`.
- **Vector dimension** is NOT a tuning variable for launch (shrinking dims lowers *cost*, not quality). Just report each model's native dimension as a column. Never naively truncate a non-Matryoshka model (e5, bge-m3 are not MRL-trained); truncation is valid only for MRL models (voyage-3 family, OpenAI text-embedding-3) and is a *future* cost-vs-quality experiment.
- **Input length matters and is handled.** Embedding models silently truncate text past their max input tokens (e5 ≈ 512, bge-m3 ≈ 8192, voyage-law-2 ≈ 16000). The `stats` command reports per-model truncation rate so this hidden confound in model comparison is surfaced, not buried.
- **Reranker** is off-the-shelf `BAAI/bge-reranker-v2-m3`. A LoRA-fine-tuned reranker is a *future* follow-up, not launch.

## Stack

Python 3.10+. Core deps: `sentence-transformers`, `rank-bm25`, `numpy`, `pyyaml`, `rich`, a CLI lib (`typer` or `click`). Optional: `chromadb`, `faiss-cpu`, `voyageai`. Developed on Apple Silicon (MPS); must also run CPU-only.

## Conventions

- Keep the core dependency-light; heavy/optional backends behind extras (`pip install ragladder[voyage,chroma]`).
- Deterministic evaluation (no randomness in metrics). Cache embeddings to disk keyed by (model, corpus hash) to avoid recompute.
- Match existing code style once code exists; until then, standard idiomatic Python, type hints, small modules.

## Provenance / honesty

- Findings and numbers in any writeup must be produced by actually running the tool — never fabricated. Honesty is at the level of *results*, not keystrokes.
- The bundled OR corpus is public federal law (SR 220) — safe to redistribute.
