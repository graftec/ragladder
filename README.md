# ragladder

**Measure which stage of your RAG retrieval pipeline actually earns its keep — and whether it's feeding your LLM the wrong context.**

`ragladder` is a small, opinionated evaluation tool for the **retrieval** layer of Retrieval-Augmented Generation (RAG) systems. Its two signatures:

1. **The ablation ladder.** It adds one retrieval stage at a time — dense → +BM25 → +rerank — and reports the *marginal* lift of each, so you can see what a stage is worth before paying for its latency and cost.
2. **Precision, not just recall.** Beyond "did we find the right document?" it measures whether the context is being *poisoned* with wrong ones — a **Perfect Retrieval Rate** (share of queries with zero wrong documents) and average wrong-document counts, alongside recall@k / MRR.

It is domain-agnostic (BEIR-compatible input), runs on a laptop (exact in-memory search by default — no vector database required), and ships with a **Swiss legal German** demo dataset (the Code of Obligations, *Obligationenrecht* / OR).

---

## Why the retrieval layer

In RAG, retrieved documents *become* the LLM's context. A single wrong document doesn't just waste space — it can make the model hallucinate confidently. Yet retrieval is usually tuned on faith: teams stack hybrid search and reranking without measuring what each contributes. `ragladder` makes that contribution visible and defensible, and evaluates it with **exact** nearest-neighbor search so an approximate index's recall loss never confounds the number.

---

## Install

```bash
pip install -e .                 # core
pip install -e ".[voyage]"       # + Voyage API embedder
pip install -e ".[chroma,dev]"   # + optional backends / dev tools
```

Python 3.10+. Local models use `sentence-transformers`; runs on Apple Silicon (MPS) or CPU.

---

## Quick start

Everything is driven by a **study config** plus a BEIR-style dataset (three JSONL files).

```bash
# Full study: every embedder × the ablation ladder → tables + results.json + report.html
ragladder run data/or-study.yaml --output out

# Add per-embedder diagnostics (stage attribution + reranker score-threshold sweep)
ragladder run data/or-study.yaml --output out --diagnostics

# A/B two embedders through the full pipeline, with per-query rescued/lost
ragladder compare data/or-study.yaml --a e5-large --b bge-m3

# Single-query diagnostic: what each stage retrieved, its scores, right vs wrong
ragladder inspect data/or-study.yaml --query-id q1 --embedder bge-m3

# Corpus length distribution + per-model truncation rate
ragladder stats data/or-corpus.jsonl --against data/or-study.yaml
```

### The study config

```yaml
corpus:  or-corpus.jsonl
queries: or-queries.jsonl
qrels:   or-qrels.jsonl

embedders:                         # each becomes a column; comparing them is the point
  - {name: e5-large, type: sentence_transformers, model: intfloat/multilingual-e5-large,
     query_prefix: "query: ", doc_prefix: "passage: "}
  - {name: bge-m3,   type: sentence_transformers, model: BAAI/bge-m3}

pipeline: [dense, bm25, rerank]    # prefixes of this list form the ladder rungs
reranker: {type: cross_encoder, model: BAAI/bge-reranker-v2-m3}
store:   {type: inmemory_exact}    # default: exact search, no ANN confound
fusion:  {type: rrf, k: 60}

metrics: [recall@10, mrr, perfect_retrieval_rate@3]
k: 10
n_rerank: 50                       # candidates passed into the reranker
```

Anything substantive lives in the config (reproducible, versionable); the CLI just executes it.

---

## How retrieval runs

```
                query
          ┌───────┴───────┐
          ▼               ▼
        dense           bm25          parallel: meaning-search + keyword-search
          └───────┬───────┘
                  ▼
              fuse (RRF)              merge into ONE deduplicated candidate list
                  ▼
                rerank                cross-encoder re-sorts the merged list
                  ▼
              top-k  →  metrics
```

The **ablation ladder** is different from this execution order: it compares *separate runs* — `[dense]`, then `[dense, bm25]`, then `[dense, bm25, rerank]` — reporting the metric at each rung to attribute the marginal lift of each stage.

---

## Data format (BEIR)

Three JSONL files. The tool never parses domain sources (XML, PDF, …); converting your corpus *into* this format is an upstream, domain-specific step (see `data/build_or_dataset.py` for the OR demo).

- **`corpus.jsonl`** — `{"_id", "title"?, "text", "metadata"?}`. Only `text` is embedded; `metadata` is for display/filtering.
- **`queries.jsonl`** — `{"_id", "text", "metadata"?}`.
- **`qrels`** — relevance judgments, as JSONL (`{"query_id", "relevant": [ids]}`) or BEIR TSV. Stable `_id`s are the linchpin: they must match across files.

---

## Metrics

| Metric | What it says |
|---|---|
| **recall@k** | Share of a query's relevant documents found in the top-`k`. Coverage. |
| **MRR** | Rank of the first correct document (1.0 = always first). Ranking quality. |
| **Perfect Retrieval Rate (PRR@k)** | Share of queries whose top-`k` contain **zero** wrong documents — a clean context. Precision. |
| **avg wrong-doc count** | Average irrelevant documents in the top-`k` — the context "noise" handed to the LLM. |

`recall@k` / `mrr` use `k`; PRR and avg-wrong can use a smaller cutoff (e.g. `perfect_retrieval_rate@3`).

---

## Output

`run` produces three synchronized views:

- **Terminal** (`rich`): a summary verdict, the model-comparison table, and one ablation ladder per embedder.
- **`results.json`**: the full machine-readable results — metrics, cross-embedder summary, corpus/truncation stats, and (with `--diagnostics`) the deep diagnostics.
- **`report.html`**: a self-contained, shareable report (inline SVG ladder charts, no external assets).

The **summary** ranks embedders by a balanced score, picks a favorite, names the per-axis winners, and rates the best config's absolute strength (strong / good / moderate / weak) — with a flag when the query set is too small to trust small gaps.

The **diagnostics** (opt-in) add, per embedder: *stage attribution* (candidate-pool recall ceiling, what reranking rescued or dropped, and which retriever found each relevant doc) and a *reranker score-threshold sweep* (how a minimum-score floor trades retained recall against a cleaner context).

---

## Adapters

Everything swappable is a named adapter chosen in config, via a small registry:

- **Embedders** — `sentence_transformers` (e5, bge-m3, …), `voyage` (API; e.g. voyage-law-2).
- **Reranker** — `cross_encoder` (e.g. bge-reranker-v2-m3).
- **Store** — `inmemory_exact` (default; exact cosine), `chroma` (optional).

Adding a model or backend is a config change; adding a new *kind* of backend is a small adapter plus a registered name.

---

## The demo dataset

The Swiss *Obligationenrecht* (SR 220) ships as `data/or-corpus.jsonl` (~1,600 articles), `data/or-queries.jsonl`, and `data/or-qrels.jsonl`, produced by the demo-side converters in `data/` from the official Fedlex XML and a question set. The corpus is public federal law, safe to redistribute.

---

## Development

```bash
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest        # test suite
ruff check .  # lint
```

Deterministic evaluation (no randomness in metrics). Corpus embeddings are cached to disk keyed by `(model, corpus hash)`, so re-runs and ladders don't recompute.

---

## License

MIT © 2026 Alain Graf. The bundled Swiss law demo corpus derives from the official *Obligationenrecht* (SR 220), which is public federal law.

*Built with the help of [Claude Code](https://claude.com/claude-code).*
