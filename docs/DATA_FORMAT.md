# Data format — BEIR-compatible JSONL

`ragladder` consumes the **BEIR** convention. Adopting a standard (rather than a bespoke schema) means the tool works with dozens of existing public datasets out of the box and signals fluency in the field. Converting your own domain sources *into* this format is an upstream, domain-specific step the tool never performs.

A dataset is **three files**.

## 1. `corpus.jsonl`

One JSON object per line. Each object is a **retrievable unit** — whatever granularity you chose upstream (article, paragraph, chunk). The tool does not chunk; it retrieves whatever units you give it.

```json
{"_id": "art_1", "title": "Art. 1 OR", "text": "Zum Abschlusse eines Vertrages ist die übereinstimmende gegenseitige Willensäusserung der Parteien erforderlich. ...", "metadata": {"hierarchy": "OR > Erste Abteilung > Erster Titel", "law": "OR", "article_num": "1", "language": "de"}}
```

- `_id` (string, **required**) — stable, permanent, unique. `qrels` references these. If IDs drift, evaluation breaks silently.
- `title` (string, optional) — short label; may be prepended to `text` when embedding if you choose.
- `text` (string, **required**) — **the string that gets embedded.** If you want hierarchy breadcrumbs or the title in the vector, bake them into `text` when generating the file (an upstream choice). The tool embeds `text` verbatim.
- `metadata` (object, optional) — **not embedded.** Used for display, filtering, and reporting only.

## 2. `queries.jsonl`

```json
{"_id": "q1", "text": "Wer haftet, wenn ein Dritter die Schuld erfüllt?", "metadata": {"source": "OR_Aufgaben"}}
```

- `_id` (string, **required**) — stable, unique.
- `text` (string, **required**) — the query, embedded with `kind="query"`.
- `metadata` (object, optional).

## 3. `qrels` — relevance judgments

Which corpus `_id`s are correct for each query. Two accepted forms:

**TSV** (BEIR-native):
```
query-id	corpus-id	score
q1	art_68	1
q1	art_69	1
```

**JSONL** (convenience):
```json
{"query_id": "q1", "relevant": ["art_68", "art_69", "art_70"]}
```

- `score` is the relevance grade. Binary (1 = relevant) is the core case and drives recall@k / MRR / Perfect Retrieval Rate. Graded scores (2, 3, …) enable NDCG later.
- A query may have **multiple** relevant docs (common in legal — several articles answer one Sachverhalt).

## Conventions

- **JSONL, not one big JSON array** — streamable and scalable.
- **Stable IDs are the linchpin.** They must be permanent and consistent across corpus and qrels.
- **`text` = "what to embed"; `metadata` = "everything else."** Keep the embed string clean and deliberate.

## The demo dataset (OR)

The Swiss Obligationenrecht demo ships as:
```
data/or-corpus.jsonl     # ~1,500 OR articles
data/or-queries.jsonl    # legal questions / Sachverhalte
data/or-qrels.jsonl      # question -> expected article ids
```

These are produced by a **demo-side converter** (e.g. `data/build_or_dataset.py`) that parses the official SR-220 XML into the BEIR format. That converter is domain-specific and lives with the demo — it is **not** part of the `ragladder` package. Anyone can drop in their own three files and run the tool unchanged.
