#!/usr/bin/env python3
"""Convert the OR question CSV into BEIR `or-queries.jsonl` + `or-qrels.jsonl`.

Demo-side, like build_or_dataset.py — domain-specific, not part of the package.

Input CSV columns:
    question  — a legal Sachverhalt / question (German)
    articles  — comma-separated OR article numbers that answer it (e.g. "74,189")

Article number N maps to corpus id `art_N` (the eId used in or-corpus.jsonl).
The script validates every referenced id against the corpus so a typo can't
silently break evaluation (stable ids are the linchpin — see the README).

Usage:
    python data/build_or_queries.py \
        --csv    resources/or_questions.csv \
        --corpus data/or-corpus.jsonl \
        --queries data/or-queries.jsonl \
        --qrels   data/or-qrels.jsonl
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def _corpus_ids(corpus_path: Path) -> set[str]:
    return {json.loads(line)["_id"] for line in corpus_path.open(encoding="utf-8") if line.strip()}


def convert(
    csv_path: Path, corpus_ids: set[str], source: str = "OR_Aufgaben"
) -> tuple[list[dict], list[dict]]:
    queries: list[dict] = []
    qrels: list[dict] = []
    missing: list[str] = []

    with csv_path.open(encoding="utf-8") as fh:
        for i, row in enumerate(csv.DictReader(fh), start=1):
            question = (row.get("question") or "").strip()
            if not question:
                continue
            q_id = f"q{i}"
            article_nums = [a.strip() for a in (row.get("articles") or "").split(",") if a.strip()]
            relevant = [f"art_{n}" for n in article_nums]
            missing += [rid for rid in relevant if rid not in corpus_ids]

            queries.append(
                {
                    "_id": q_id,
                    "text": question,
                    "metadata": {"source": source, "articles": article_nums},
                }
            )
            qrels.append({"query_id": q_id, "relevant": relevant})

    if missing:
        raise SystemExit(
            f"ERROR: {len(missing)} referenced article id(s) not in corpus: "
            f"{sorted(set(missing))}. Fix the CSV or rebuild the corpus."
        )
    return queries, qrels


def _write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="OR question CSV -> BEIR queries + qrels")
    ap.add_argument(
        "--csv",
        type=Path,
        default=Path("/Users/alaingraf/Documents/KAIMU/projects/kaimu_pyt_rag_server/resources/or_questions.csv"),
    )
    ap.add_argument("--corpus", type=Path, default=Path("data/or-corpus.jsonl"))
    ap.add_argument("--queries", type=Path, default=Path("data/or-queries.jsonl"))
    ap.add_argument("--qrels", type=Path, default=Path("data/or-qrels.jsonl"))
    ap.add_argument("--source", default="OR_Aufgaben", help="metadata.source tag (e.g. 'gen' for LLM-generated)")
    args = ap.parse_args()

    corpus_ids = _corpus_ids(args.corpus)
    queries, qrels = convert(args.csv, corpus_ids, source=args.source)

    _write_jsonl(queries, args.queries)
    _write_jsonl(qrels, args.qrels)

    total_rel = sum(len(q["relevant"]) for q in qrels)
    print(f"wrote {len(queries)} queries -> {args.queries}")
    print(f"wrote {len(qrels)} qrels ({total_rel} relevance judgments) -> {args.qrels}")


if __name__ == "__main__":
    main()
