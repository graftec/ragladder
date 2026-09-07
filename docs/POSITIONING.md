# Positioning & narrative

This document captures *why* `ragladder` exists and the story it is meant to tell. It is the strategic brief behind the tool.

## Goal

A small open-source tool + an accompanying article that position the author as a **RAG specialist** on LinkedIn / Medium. The proven template is [ragtune](https://github.com/metawake/ragtune) + its Medium post ("I built a RAG tuning tool and discovered intuition fails on legal text"): a focused tool, a clear repo, and one memorable, counterintuitive finding.

`ragladder` follows that container but stands on ground ragtune doesn't occupy.

## Differentiation vs the field

- **Generation-quality eval** (Ragas, DeepEval, TruLens, promptfoo, Phoenix) — crowded; LLM-as-judge on faithfulness/answer relevance. **Not our lane.**
- **Retrieval-layer eval** — thin. ragtune is the close neighbor: recall-first, English benchmarks.
- **`ragladder`'s claim**: the sharpest answer to one question — *"which stage of my retrieval pipeline earns its keep, and is it feeding the model wrong context?"*

One-line pitch: *"ragtune tells you your retrieval recall. `ragladder` tells you which pipeline stage earns its cost — and whether you're poisoning the LLM's context with wrong documents."*

## The two signature ideas

1. **The ablation ladder** (stage attribution). Start from a naive baseline, add one stage at a time, measure the lift at each rung. A tall rung = the stage is worth its cost; a flat rung = it isn't. This is the hero screenshot.
2. **Precision / wrong-context** (the `Perfect Retrieval Rate` metric). Recall asks "did we find the needle?" `ragladder` also asks "how much hay did we shove into the context window?" — because one wrong statute makes an LLM answer wrongly with confidence.

## The demo dataset & the finding

The bundled demo is the **Swiss Code of Obligations** (Obligationenrecht / OR, SR 220) — German-language, ~1,500 articles, official public federal law. It's a *hard*, non-English, high-stakes domain almost nobody has benchmarked publicly.

The launch study compares **embedding models** (`multilingual-e5-large` vs `voyage-law-2` vs `bge-m3`) on raw OR articles, across the ablation ladder.

**Candidate hero findings** (to be confirmed by actually running it):
- **Legal-specialist vs general multilingual embedder on German civil law.** `voyage-law-2` is legal-domain but English/common-law-leaning; the corpus is German civil law. Either outcome is a strong headline — *"domain-specific ≠ language-specific"* if the general model wins.
- **Input-length truncation.** `e5-large` reads only ~512 tokens; long articles get silently cut. Long-context models (`bge-m3` 8k, `voyage-law-2` 16k) read the whole article. Part of any quality gap may simply be *"e5 couldn't read the whole statute."* Honest, legal-specific, shareable.
- **Stage attribution.** Which of BM25 / rerank actually earns its keep on legal text? (Legal text is full of exact terms and article numbers, so BM25 may punch above its reputation.)

### Framing guardrails for the writeup
- Lead with **relative** improvement over a naive baseline (e.g. "+X% MRR"), not absolute numbers — absolute retrieval numbers on hard legal German are modest and that's fine.
- Make the **difficulty legible** (the fact-pattern↔doctrine vocabulary gap; long articles) so modest absolutes read as "hard problem, real progress," not "weak."
- Honesty at the level of **findings**, not keystrokes. AI-assisted development is fine; an optional one-line acknowledgment belongs in the README, not the paper.

## Staged content (one tool → several posts)

The launch is deliberately scoped small; the cut features become follow-ups:
1. **Launch:** embedding-model comparison + ablation ladder + wrong-context metric on raw OR.
2. Follow-up: *"Can query reformulation close the fact-pattern↔doctrine gap?"* (the original project's biggest win: MRR ~0.30 → ~0.45 via Claude reformulation).
3. Follow-up: *"Does embedding LLM-enrichment instead of raw statute help?"* (spoiler from prior data: barely — a counterintuitive result).
4. Follow-up: *"Does a fine-tuned (LoRA) reranker beat off-the-shelf on legal text?"*
5. Follow-up: *"Halve your vectors for ~2% quality loss"* (Matryoshka dimension / quantization cost-vs-quality).

See `ROADMAP.md` for how these map to scope.

## Prior art / reference
- ragtune: https://github.com/metawake/ragtune (retrieval debugger; recall-first). Note: the author (`metawake`) also owns the PyPI name `ragprobe` and a related tool — avoid name collisions with that ecosystem.
- BEIR: the data-format and dataset-compatibility standard we adopt.
