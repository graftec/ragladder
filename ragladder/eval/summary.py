"""Cross-embedder summary — a transparent leaderboard with a picked favorite.

"Best" is multi-dimensional, so this does three things instead of crowning a
single winner blindly:

  * ranks embedders (at their best rung) by a **balanced score**
    ``0.4·MRR' + 0.3·recall' + 0.3·clean'`` where ' is min-max normalized across
    the models being compared and *clean* rewards a low wrong-doc count;
  * still names the **per-axis winners** (best recall, best MRR, cleanest
    context), because different use cases weight these differently;
  * flags when the favorite's lead is **within noise** — essential when the
    query set is small, where a fraction of a point is one query flipping.

The scoring rule is fixed and stated so the pick is defensible, not magic.
"""

from __future__ import annotations

from dataclasses import dataclass

from ragladder.eval.metrics import MetricResult
from ragladder.pipeline.runner import RunResult

# Balanced-blend weights (documented in the report). MRR leads; recall and a
# clean (low wrong-context) result split the rest.
W_MRR, W_RECALL, W_CLEAN = 0.4, 0.3, 0.3

# Raw MRR gap (favorite vs runner-up) below which the lead is called "within
# noise". Keyed off the raw metric, NOT the balanced score: min-max normalization
# stretches any tiny difference to a full 0..1 spread, so the score margin would
# wildly overstate how separated two near-tied models are.
NOISE_MRR_GAP = 0.05

# Absolute quality bands for the best config. MRR = ranking quality (how near the
# top the first correct doc lands); recall@k = coverage (how much of the answer
# set is found at all). Descending thresholds map to levels 3..0. These are
# deliberate, stated cut-offs — reasonable IR rules of thumb, not universal law.
_MRR_BANDS = [(0.70, 3), (0.50, 2), (0.30, 1)]  # else 0
_RECALL_BANDS = [(0.80, 3), (0.60, 2), (0.40, 1)]  # else 0
_LEVELS = {3: "strong", 2: "good", 1: "moderate", 0: "weak"}


@dataclass
class EmbedderScore:
    embedder: str
    recall_at_k: float
    mrr: float
    perfect_retrieval_rate: float
    avg_wrong_count: float
    recall_norm: float
    mrr_norm: float
    clean_norm: float
    score: float
    is_favorite: bool = False


@dataclass
class QualityRating:
    """Absolute quality of the best config — is the retrieval any good at all?"""

    level: str  # "strong" | "good" | "moderate" | "weak"
    label: str  # display form, e.g. "Strong"
    headline: str  # one-line verdict
    rationale: str  # what drives it / where the bottleneck is


@dataclass
class Summary:
    k: int
    prr_k: int
    n_judged: int
    weights: dict
    rows: list[EmbedderScore]  # sorted by score, descending
    favorite: str
    best_recall: str
    best_mrr: str
    cleanest: str
    margin: float  # raw MRR gap between favorite and runner-up (0.0 if one model)
    within_noise: bool
    caveat: str
    quality: QualityRating


def _best_metrics(result: RunResult) -> list[tuple[str, MetricResult]]:
    out = []
    for ladder in result.ladders:
        rungs = [r for r in ladder.rungs if r.implemented and r.metrics is not None]
        if rungs:
            out.append((ladder.embedder, rungs[-1].metrics))
    return out


def _band(value: float, bands: list[tuple[float, int]]) -> int:
    for threshold, level in bands:
        if value >= threshold:
            return level
    return 0


def assess_quality(best: EmbedderScore, k: int, prr_k: int) -> QualityRating:
    """Rate the absolute strength of the best config from its metrics.

    Overall = the weaker of ranking (MRR) and coverage (recall@k) — a retrieval
    layer isn't "strong" if it ranks well but misses half the answers, or finds
    them but buries them. The rationale names whichever is the bottleneck.
    """
    ranking = _band(best.mrr, _MRR_BANDS)
    coverage = _band(best.recall_at_k, _RECALL_BANDS)
    overall = min(ranking, coverage)
    label = _LEVELS[overall].capitalize()

    approx_rank = f"~{round(1 / best.mrr)}" if best.mrr > 0 else "not in top-k"
    headline = (
        f"{label}: the best config finds {best.recall_at_k:.0%} of relevant documents "
        f"(recall@{k}) and ranks the first correct one at position {approx_rank} "
        f"(MRR {best.mrr:.2f})."
    )

    if coverage < ranking:
        bottleneck = (
            f"Coverage is the bottleneck — {best.recall_at_k:.0%} recall@{k} means many "
            "relevant documents are never retrieved (a retrieval/embedding limit, not "
            "ranking)."
        )
    elif ranking < coverage:
        bottleneck = (
            f"Ranking is the bottleneck — relevant docs are retrieved but the first one "
            f"sits around position {approx_rank}; a reranker or better fusion should help."
        )
    else:
        bottleneck = f"Coverage and ranking are balanced at the {label.lower()} level."

    # Precision side-note: how noisy is the context near the top.
    if prr_k and best.avg_wrong_count > 0.7 * prr_k:
        bottleneck += (
            f" Context is also noisy: ~{best.avg_wrong_count:.1f} of the top {prr_k} "
            "documents are irrelevant."
        )

    return QualityRating(level=_LEVELS[overall], label=label, headline=headline, rationale=bottleneck)


def _minmax(values: list[float]) -> list[float]:
    lo, hi = min(values), max(values)
    if hi == lo:  # no discrimination on this axis; treat all as neutral-best
        return [1.0] * len(values)
    return [(v - lo) / (hi - lo) for v in values]


def summarize(result: RunResult) -> Summary | None:
    entries = _best_metrics(result)
    if not entries:
        return None

    recalls = [m.recall_at_k for _, m in entries]
    mrrs = [m.mrr for _, m in entries]
    wrongs = [m.avg_wrong_count for _, m in entries]

    recall_n = _minmax(recalls)
    mrr_n = _minmax(mrrs)
    clean_n = _minmax([-w for w in wrongs])  # fewer wrong docs -> higher clean'

    rows = [
        EmbedderScore(
            embedder=name,
            recall_at_k=m.recall_at_k,
            mrr=m.mrr,
            perfect_retrieval_rate=m.perfect_retrieval_rate,
            avg_wrong_count=m.avg_wrong_count,
            recall_norm=recall_n[i],
            mrr_norm=mrr_n[i],
            clean_norm=clean_n[i],
            score=W_MRR * mrr_n[i] + W_RECALL * recall_n[i] + W_CLEAN * clean_n[i],
        )
        for i, (name, m) in enumerate(entries)
    ]
    rows.sort(key=lambda r: r.score, reverse=True)
    rows[0].is_favorite = True

    # Honest closeness = raw MRR gap between the favorite and the runner-up.
    margin = abs(rows[0].mrr - rows[1].mrr) if len(rows) > 1 else 0.0
    n_judged = result.dataset_summary.get("n_judged", 0)
    within_noise = len(rows) > 1 and margin < NOISE_MRR_GAP

    caveats = []
    if within_noise:
        caveats.append(
            f"favorite's MRR lead over #2 is {margin:.3f} (< {NOISE_MRR_GAP}) — treat as a tie"
        )
    if n_judged and n_judged < 30:
        caveats.append(f"only {n_judged} judged queries — differences are statistically weak")

    return Summary(
        k=result.k,
        prr_k=result.prr_k or result.k,
        n_judged=n_judged,
        weights={"mrr": W_MRR, "recall": W_RECALL, "clean": W_CLEAN},
        rows=rows,
        favorite=rows[0].embedder,
        best_recall=max(rows, key=lambda r: r.recall_at_k).embedder,
        best_mrr=max(rows, key=lambda r: r.mrr).embedder,
        cleanest=min(rows, key=lambda r: r.avg_wrong_count).embedder,
        margin=margin,
        within_noise=within_noise,
        caveat="; ".join(caveats),
        quality=assess_quality(rows[0], result.k, result.prr_k or result.k),
    )
