"""Machine-readable results.json — for CI/regression and re-plotting."""

from __future__ import annotations

import json
import math
from pathlib import Path

from ragladder.eval.diagnostics import StageAttribution, ThresholdSweep
from ragladder.eval.stats import LengthStats, TruncationStat
from ragladder.eval.summary import summarize
from ragladder.pipeline.runner import RunResult

# Per-embedder diagnostics: name -> (stage attribution, sweep or None).
Diagnostics = dict[str, tuple[StageAttribution, "ThresholdSweep | None"]]


def _summary_dict(result: RunResult) -> dict | None:
    s = summarize(result)
    if s is None:
        return None
    return {
        "favorite": s.favorite,
        "quality": {
            "level": s.quality.level,
            "label": s.quality.label,
            "headline": s.quality.headline,
            "rationale": s.quality.rationale,
        },
        "rule": f"{s.weights['mrr']}*mrr' + {s.weights['recall']}*recall' "
        f"+ {s.weights['clean']}*clean' (normalized across models)",
        "best_recall": s.best_recall,
        "best_mrr": s.best_mrr,
        "cleanest": s.cleanest,
        "margin": round(s.margin, 4),
        "within_noise": s.within_noise,
        "caveat": s.caveat,
        "ranking": [
            {"embedder": r.embedder, "score": round(r.score, 4), "mrr": round(r.mrr, 4),
             "recall@k": round(r.recall_at_k, 4), "avg_wrong": round(r.avg_wrong_count, 4)}
            for r in s.rows
        ],
    }


def _length_dict(length_stats: dict[str, LengthStats]) -> dict:
    return {
        unit: {
            "unit": s.unit, "count": s.count, "min": s.min,
            "p50": round(s.p50, 1), "mean": round(s.mean, 1), "p90": round(s.p90, 1), "max": s.max,
        }
        for unit, s in length_stats.items()
    }


def _truncation_dict(truncation: list[TruncationStat]) -> list[dict]:
    return [
        {
            "embedder": t.embedder, "max_input_tokens": t.max_input_tokens,
            "total": t.total, "truncated": t.truncated, "rate": round(t.rate, 4),
            "token_p50": round(t.token_p50, 1), "token_p90": round(t.token_p90, 1),
            "token_max": t.token_max,
        }
        for t in truncation
    ]


def _attribution_dict(a: StageAttribution) -> dict:
    return {
        "k": a.k, "n_rerank": a.n_rerank,
        "pool_recall": round(a.pool_recall, 4),
        "fused_recall_at_k": round(a.fused_recall_at_k, 4),
        "final_recall_at_k": round(a.final_recall_at_k, 4),
        "rescued_by_rerank": a.rescued_by_rerank,
        "dropped_by_rerank": a.dropped_by_rerank,
        "reachable_missed": a.reachable_missed,
        "source": {
            "dense_only": a.source.dense_only, "bm25_only": a.source.bm25_only,
            "both": a.source.both, "missed": a.source.missed,
        },
    }


def _sweep_dict(sweep: ThresholdSweep | None) -> dict | None:
    if sweep is None:
        return None
    st = sweep.stats
    return {
        "k": sweep.k,
        "score_stats": {
            "relevant_median": st.relevant_median, "relevant_p25": st.relevant_p25,
            "wrong_median": st.wrong_median, "wrong_p90": st.wrong_p90,
            "separation": st.separation,
        },
        "rows": [
            {
                # -inf (the no-floor baseline) -> null, so the JSON stays valid.
                "threshold": None if math.isinf(r.threshold) else r.threshold,
                "recall_retained": round(r.recall_retained, 4),
                "avg_wrong_kept": round(r.avg_wrong_kept, 4),
                "perfect_rate": round(r.perfect_rate, 4),
                "kept_relevant": r.kept_relevant, "kept_wrong": r.kept_wrong,
            }
            for r in sweep.rows
        ],
    }


def _diagnostics_dict(diagnostics: Diagnostics | None) -> dict | None:
    if not diagnostics:
        return None
    return {
        name: {
            "stage_attribution": _attribution_dict(attr),
            "reranker_threshold_sweep": _sweep_dict(sweep),
        }
        for name, (attr, sweep) in diagnostics.items()
    }


def run_to_dict(
    result: RunResult,
    length_stats: dict[str, LengthStats] | None = None,
    truncation: list[TruncationStat] | None = None,
    diagnostics: Diagnostics | None = None,
) -> dict:
    corpus_stats = None
    if length_stats is not None:
        corpus_stats = {
            "length": _length_dict(length_stats),
            "truncation": _truncation_dict(truncation or []),
        }
    return {
        "k": result.k,
        "metrics_requested": result.metrics_requested,
        "dataset": result.dataset_summary,
        "summary": _summary_dict(result),
        "corpus_stats": corpus_stats,
        "diagnostics": _diagnostics_dict(diagnostics),
        "embedders": [
            {
                "embedder": ladder.embedder,
                "dim": ladder.dim,
                "max_input_tokens": ladder.max_input_tokens,
                "ladder": [
                    {
                        "rung": i,
                        "stages": rung.stages,
                        "label": rung.label,
                        "implemented": rung.implemented,
                        "note": rung.note,
                        "metrics": rung.metrics.as_dict() if rung.metrics else None,
                    }
                    for i, rung in enumerate(ladder.rungs)
                ],
            }
            for ladder in result.ladders
        ],
    }


def write_results_json(
    result: RunResult,
    path: str | Path,
    length_stats: dict[str, LengthStats] | None = None,
    truncation: list[TruncationStat] | None = None,
    diagnostics: Diagnostics | None = None,
) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = run_to_dict(result, length_stats, truncation, diagnostics)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return path
