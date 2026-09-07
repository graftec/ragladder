# ragladder

**Measure which stage of your RAG retrieval pipeline actually earns its keep — and whether it's feeding your LLM the wrong context.**

`ragladder` is a small, opinionated evaluation tool for the **retrieval** layer of Retrieval-Augmented Generation (RAG) systems. Where most eval tools score *a* pipeline, `ragladder` builds an **ablation ladder**: it adds one retrieval stage at a time (dense → +BM25 → +rerank) and reports the *marginal* lift of each stage, so you can see what's worth its latency and cost. It also reports **precision / wrong-context metrics** — not just "did we find the right document?" but "are we poisoning the context window with wrong ones?"

It is domain-agnostic (BEIR-compatible input), runs on a laptop (exact in-memory search by default — no vector database required), and ships with a **Swiss legal German** demo dataset (the Code of Obligations, *Obligationenrecht* / OR).

> Status: **early scaffold.** This repository currently contains the design, documentation, and project skeleton. Implementation is pending — see [`docs/ROADMAP.md`](docs/ROADMAP.md).

---

## Why another RAG eval tool?

Most RAG evaluation tooling (Ragas, DeepEval, TruLens, …) focuses on **generation quality** (faithfulness, answer relevance) via LLM-as-judge. The retrieval layer — which decides *what the LLM even gets to see* — is comparatively under-served. The one close neighbor, [ragtune](https://github.com/metawake/ragtune), is a recall-first retrieval debugger.

`ragladder` stakes out a narrower, sharper claim:

1. **Stage attribution.** Teams stack hybrid search, query reformulation, and reranking on faith. `ragladder`'s headline output is the **ablation ladder** — the metric after each added stage — so the *marginal value* of every stage is visible and defensible.
2. **Precision, not just recall.** In high-stakes domains a single wrong document in context makes an LLM hallucinate confidently. `ragladder` reports a **Perfect Retrieval Rate** (share of queries with *zero* wrong documents retrieved) and wrong-document counts alongside the usual recall@k / MRR.
3. **Rigorous by default.** Retrieval is evaluated with **exact** nearest-neighbor search, so an approximate-index's recall loss never confounds the measurement. No vector DB required to get a trustworthy number.

It is deliberately a **scalpel, not a suite** — see the guardrails in [`CLAUDE.md`](CLAUDE.md).

---

## What it does (at a glance)

```
                query
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
     dense                bm25          ← parallel: meaning-search + keyword-search
        │                   │
        └────────┬──────────┘
                 ▼
             fuse (RRF)                 ← merge into ONE candidate list
                 ▼
             rerank                     ← cross-encoder re-sorts the merged list
                 ▼
             top-k  →  metrics
```

- **Compare embedding models** head-to-head (e.g. `multilingual-e5-large` vs `voyage-law-2` vs `bge-m3`).
- **Ablation ladder**: dense → +BM25 → +rerank, with per-stage lift.
- **A/B compare** two configs with query-level "rescued / lost" attribution.
- **Inspect** a single query: what was retrieved, at what score, which were right/wrong.
- **Corpus stats**: token-length distribution and per-model **truncation rate** (does your embedding model even read the whole document?).

---

## Installation

> Not yet published. Planned:

```bash
pip install ragladder
```

For development, see [`docs/DEVELOPMENT.md`](docs/DEVELOPMENT.md) (to be written).

---

## Quick start

Everything is driven by a **study config** (`study.yaml`) plus a BEIR-style dataset (three JSONL files). See [`examples/study.yaml`](examples/study.yaml).

```bash
# Run the full study: every embedder × every pipeline stage → ladder + comparison tables + report
ragladder run examples/study.yaml

# A/B: which queries did each variant rescue vs lose?
ragladder compare examples/study.yaml --a e5-large --b voyage-law-2

# Single-query diagnostic
ragladder inspect examples/study.yaml --query "Wer haftet, wenn ein Dritter die Schuld erfüllt?"

# Corpus stats: token lengths + per-model truncation rate
ragladder stats data/or-corpus.jsonl --against examples/study.yaml
```

Outputs: pretty terminal tables, a machine-readable `results.json`, and a shareable `report.html` (where the ladder chart lives).

---

## Data format

`ragladder` consumes the **BEIR** convention — three JSONL files:

- `corpus.jsonl` — the retrievable units (`_id`, `title`, `text`, `metadata`)
- `queries.jsonl` — the questions (`_id`, `text`, `metadata`)
- `qrels` — relevance judgments (which corpus `_id`s are correct per query)

The tool never parses domain sources (XML, PDF, …); converting your corpus *into* this format is an upstream, domain-specific step. See [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md).

---

## Documentation

| Doc | Contents |
|---|---|
| [`docs/POSITIONING.md`](docs/POSITIONING.md) | The project's purpose, the story/paper angle, the demo dataset |
| [`docs/DESIGN.md`](docs/DESIGN.md) | Architecture: adapters, retrieval flow, exact-search rationale |
| [`docs/DATA_FORMAT.md`](docs/DATA_FORMAT.md) | The BEIR JSONL contract (corpus / queries / qrels) |
| [`docs/METRICS.md`](docs/METRICS.md) | recall@k, MRR, Perfect Retrieval Rate, wrong-document metrics |
| [`docs/CLI.md`](docs/CLI.md) | Command surface: `run` / `compare` / `inspect` / `stats` |
| [`docs/ROADMAP.md`](docs/ROADMAP.md) | Launch scope and staged follow-ups |
| [`CLAUDE.md`](CLAUDE.md) | Context & guardrails for AI-assisted development |

---

## License

MIT © 2026 Alain Graf. See [`LICENSE`](LICENSE).

The bundled Swiss law demo corpus derives from the official *Obligationenrecht* (SR 220), which is public federal law.
