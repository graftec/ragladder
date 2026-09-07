# CLI reference

`ragladder` is CLI-first and config-driven. A **study** is defined in `study.yaml` (see `examples/study.yaml` and below); the CLI executes it. A thin Python API sits underneath for notebook/library use.

Keep the command surface small — four commands.

## `run` — the workhorse

```bash
ragladder run study.yaml [--output out/] [--k 10] [--limit N]
```

Runs every embedder × every pipeline stage defined in the config. Produces:
- **terminal**: the ablation ladder + model-comparison tables (via `rich`)
- **`results.json`**: machine-readable (for CI/regression, re-plotting)
- **`report.html`**: shareable report — where the ladder chart lives (the marketing artifact)

Flags:
- `--output` — output directory (default `./out`)
- `--k` — override final top-k
- `--limit N` — run only the first N queries (fast smoke test)

## `compare` — A/B with per-query attribution

```bash
ragladder compare study.yaml --a e5-large --b voyage-law-2
```

Runs two variants on the same queries and reports which queries each **rescued** vs **lost**, plus the aggregate deltas. Generalizes the original project's `compare_pipelines.py`. Variants can be two embedders, or two pipeline configs.

## `inspect` — single-query diagnostic

```bash
ragladder inspect study.yaml --query "Wer haftet, wenn ein Dritter die Schuld erfüllt?"
```

Shows, for one query: what was retrieved at each stage, the scores, and which results are correct vs wrong per `qrels`. The debugging counterpart to ragtune's `explain`.

## `stats` — corpus & truncation stats

```bash
ragladder stats data/or-corpus.jsonl [--against study.yaml]
```

Reports the corpus token-length distribution and, with `--against`, the **per-model truncation rate** — what fraction of documents exceed each embedder's max input tokens (and thus get silently cut off). Surfaces the input-length confound before you trust a model comparison.

## `--gate` (future, not launch)

```bash
ragladder run study.yaml --gate "mrr>=0.40"
```

Non-zero exit if a threshold fails, for use as a CI quality gate. Deferred.

## The study config

```yaml
# study.yaml
corpus:  data/or-corpus.jsonl
queries: data/or-queries.jsonl
qrels:   data/or-qrels.jsonl

embedders:
  - {name: e5-large,     type: sentence_transformers, model: intfloat/multilingual-e5-large,
     query_prefix: "query: ", doc_prefix: "passage: "}
  - {name: voyage-law-2, type: voyage, model: voyage-law-2, api_key_env: VOYAGE_API_KEY}
  - {name: bge-m3,       type: sentence_transformers, model: BAAI/bge-m3}

pipeline: [dense, bm25, rerank]         # stages; also defines the ablation ladder order
reranker: {type: cross_encoder, model: BAAI/bge-reranker-v2-m3}

store:   {type: inmemory_exact}         # default; or {type: chroma, path: ...}
fusion:  {type: rrf, k: 60}

metrics: [recall@10, mrr, perfect_retrieval_rate]
k: 10
n_rerank: 50                            # candidates passed into the reranker
```

Notes:
- Anything substantive goes in the config, not flags — reproducibility.
- `embedders` is a list precisely because comparing them is the point; each becomes a column.
- `pipeline` order drives the ladder (dense → +bm25 → +rerank).
- Adapters (`type:`) resolve through the registry, so adding a model/backend is config, and adding a *new* backend is a small adapter + a registered name.
