# Metrics

`ragladder` reports two families: standard IR metrics (table stakes) and **precision / wrong-context** metrics (the differentiator). All are computed deterministically over the top-`k` retrieved documents per query, judged against `qrels`.

## Standard IR metrics

### Recall@k
Of the documents that are relevant for a query, what fraction appear in the top-k?

```
recall@k = |relevant ∩ top_k| / |relevant|
```

Interpretation: "did we surface the right documents at all, within k?" With multiple relevant docs per query (common in legal), this rewards finding more of them.

### MRR (Mean Reciprocal Rank)
For each query, take the rank of the *first* relevant document; reciprocal it; average over queries.

```
MRR = mean( 1 / rank_of_first_relevant )     (0 if none in top-k)
```

Interpretation: "how high up is the first correct hit?" Rewards putting a right answer near the top.

### (Later) NDCG@k
Rewards putting *more* relevant docs *higher*, using graded relevance. Enabled only when `qrels` carry graded scores. Not required for launch.

## Precision / wrong-context metrics — the differentiator

The premise: in RAG, retrieved documents become the LLM's context. A **wrong** document doesn't just waste space — it actively causes confident hallucination. So we measure what's being let *into* the context, not only whether the right thing was found.

### Wrong-document count / rate
Per query, how many of the top-k retrieved documents are **not** relevant?

```
wrong_count(q)      = |top_k \ relevant|
wrong_rate(q)       = wrong_count(q) / k
avg_wrong_count     = mean over queries
```

### Perfect Retrieval Rate (headline)
The share of queries whose top-k contains **zero** wrong documents — i.e. the context was clean.

```
perfect_retrieval_rate = (# queries with wrong_count == 0) / (# queries)
```

Interpretation: "how often did we hand the LLM a context with no poison in it?" This is deliberately strict and precision-oriented; it can rank strategies very differently from recall/MRR (a strategy that finds the right doc *and* three wrong ones scores well on recall but poorly here).

> Note: with a fixed `k` and multiple relevant docs, "wrong" = retrieved-but-not-in-qrels. Consider reporting Perfect Retrieval Rate at a small `k` (e.g. k=3 or k=5) where context precision matters most, in addition to the recall/MRR at k=10.

## The ablation ladder (not a metric — a presentation)

For a fixed embedder, report a chosen metric after each added stage:

| Rung | Configuration | MRR | Δ (marginal lift) |
|---|---|---|---|
| 0 | dense only | 0.30 | — |
| 1 | + BM25 (RRF) | 0.34 | +0.04 |
| 2 | + reranker | 0.41 | +0.07 |

The Δ column *is* the point: it attributes value to each stage. A stage with a flat or negative Δ is not earning its latency/cost.

## A/B diff

Compare two configs on the same queries and classify each query:
- **rescued** — B ranks the first relevant doc higher than A (or A missed, B hit).
- **lost** — A was better than B.
- unchanged otherwise.

Report counts + the specific queries, so a change's *distribution* of effects is visible, not just its average.

## Reporting conventions

- Always show a **naive baseline** (plain dense, top-k) next to the best config so *relative* improvement is visible even when absolute numbers are modest.
- Report each embedder's **native dimension** and **max input tokens** as columns, plus its **truncation rate** on this corpus (see `stats`), so cross-model comparisons are honest about confounds.
- Keep native-dimension model comparisons and any (future) truncation sweeps in **separate tables** — they answer different questions.
