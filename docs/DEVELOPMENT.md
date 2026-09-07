# Development

> Placeholder — fill in once implementation begins (see `ROADMAP.md`).

## Setup (planned)

```bash
git clone https://github.com/graftec/ragladder
cd ragladder
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev,voyage,chroma]"
```

## Building the demo dataset

The Swiss OR demo corpus is generated from the official SR-220 XML by a demo-side
converter (not part of the package):

```bash
python data/build_or_dataset.py --xml path/to/SR-220-DE.xml --out data/
# -> data/or-corpus.jsonl, data/or-queries.jsonl, data/or-qrels.jsonl
```

## Running the test suite (planned)

```bash
pytest
```

## Conventions

- Keep the core dependency-light; optional backends behind extras.
- Deterministic evaluation — no randomness in the metric path.
- Cache embeddings by (model, corpus hash) to avoid recompute.
- Respect the guardrails in `../CLAUDE.md` (scalpel, not suite).
- Type hints, small modules, `ruff` for lint/format.
