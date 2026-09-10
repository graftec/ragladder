# CLAUDE.md — context for AI-assisted development of `ragladder`

Orientation for a coding session in this repo. For what the tool is and how to use it, read `README.md` first; this file adds the design intent and guardrails that aren't obvious from the code.

## One-sentence definition

A domain-agnostic RAG **retrieval-layer** evaluation harness whose signature is (a) **stage-by-stage ablation** — the "ablation ladder", measuring which pipeline stage earns its keep — and (b) **precision-first metrics** that measure wrong-context, not just recall; shipped with Swiss legal German (Obligationenrecht) as the bundled demo dataset.

## Guardrails — keep it a scalpel, not a suite

Deliberate scope decisions. Don't casually violate them; if a change seems to require it, flag it.

1. **Default backend is in-memory exact search.** ChromaDB is an optional adapter; other vector DBs are a documented extension point, not built. Exact-by-default is a principled choice (no ANN recall loss confounding the measurement), not a limitation to fix.
2. **The tool does not ingest domain sources.** No XML/PDF/HTML parsing or chunking in the core. Input is always the BEIR JSONL contract (corpus/queries/qrels). Domain conversion lives in demo-side scripts under `data/`, never in the `ragladder` package.
3. **Config-first.** A study is defined in `study.yaml` (reproducible, versionable); the CLI executes it. Don't push substantive knobs into CLI flags.
4. **Everything swappable is a named adapter** chosen in config: embedders, reranker, store — one interface + registry pattern per kind.
5. **No LLM enrichment or query reformulation *stages* in the core** (deferred). A manually reformulated query set (`data/or-queries-juristic.jsonl`) is a *dataset-side* probe of that idea and is fine; an automated stage is not in scope yet.
6. **Retrieval only. No generation, no LLM-as-judge.**

## Architecture

```
ragladder/
  cli.py                 # run / compare / inspect / stats (typer)
  config.py              # load & validate study.yaml
  registry.py            # name -> adapter class (embedders / rerankers / stores)
  data/beir.py           # BEIR loaders (corpus/queries/qrels; jsonl + qrels tsv)
  adapters/
    embedders/           # sentence_transformers, voyage  (embed(texts, kind))
    rerankers/           # cross_encoder                   (rerank(query, candidates))
    stores/              # inmemory_exact (default), chroma (optional)
  pipeline/              # dense, bm25, fusion (RRF), rerank; runner + Retriever
  eval/                  # metrics, ab (A/B diff), stats, diagnostics, summary
  report/                # tables (rich), results.json, html (self-contained report)
data/                    # demo-side converters + the OR BEIR dataset + study configs
```

## Retrieval dataflow vs the ladder (don't conflate)

- **Execution** (one config): `dense ∥ bm25` run in parallel over the whole corpus → **fuse** (Reciprocal Rank Fusion, k=60) into one deduplicated candidate list → **rerank** (cross-encoder, serial) → top-k → metrics.
- **Ladder** (comparing configs): `[dense]`, then `[dense, bm25]`, then `[dense, bm25, rerank]` — separate runs, to attribute the marginal lift of each stage. The rungs are the cumulative prefixes of `pipeline`; fusion is implicit (triggered by >1 retriever), not a rung.

## Key technical decisions

- **Embedding model is the primary comparison axis.** Demo compares `intfloat/multilingual-e5-large`, `BAAI/bge-m3`, and Voyage models (`voyage-law-2`, `voyage-4-large`).
- **Vector dimension is not a tuning variable at launch** — report each model's native dim as a column. Never naively truncate a non-Matryoshka model (e5, bge-m3); truncation is valid only for MRL models (Voyage 3/4 family, OpenAI text-embedding-3) and is a future cost-vs-quality experiment.
- **Input length is handled and surfaced.** Models silently truncate past their max tokens; `stats` reports per-model truncation rate so this confound is visible, not buried.
- **Reranker** is off-the-shelf `BAAI/bge-reranker-v2-m3`. A fine-tuned reranker is a future follow-up.
- **Metrics.** recall@k and MRR (ranking/coverage); Perfect Retrieval Rate + avg wrong-doc count (precision/wrong-context). PRR's cutoff is configurable (`perfect_retrieval_rate@N`) and independent of `k`.

## Conventions

- Python 3.10+, type hints, small modules. Core deps light; heavy/optional backends behind extras (`voyage`, `chroma`, `faiss`, `openai`, `dev`).
- Deterministic evaluation (no randomness in metrics). Cache embeddings to disk keyed by `(model, corpus hash)`.
- Lint/format with `ruff`; tests with `pytest`. Keep the suite green.

## Provenance / honesty

- Findings and numbers in any writeup must come from actually running the tool — never fabricated.
- The bundled OR corpus is public federal law (SR 220), safe to redistribute.
