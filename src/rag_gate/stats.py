"""Paired bootstrap over questions, so the gate fails on real regressions rather than noise."""

from __future__ import annotations

import random
from typing import Callable, Sequence

# Each metric as a per-question (numerator, denominator) pair. The metric is sum(num) / sum(den)
# over questions; for plain query-averaged metrics the denominator is 1. Denominators depend only
# on labels, so they are the same for baseline and candidate.
PerCase = Callable[[dict], tuple[float, float]]


def _citations_ok(case: dict) -> tuple[float, float]:
    cited = case["cited_ids"]
    bad = set(case["citations_not_retrieved"]) | set(case["citations_not_relevant"])
    return float(sum(c not in bad for c in cited)), float(len(cited))


PER_CASE: dict[str, PerCase] = {
    "hit_at_k": lambda c: (float(c["hit"]), 1.0),
    "mrr_at_k": lambda c: (float(c["reciprocal_rank"]), 1.0),
    "recall_at_k": lambda c: (float(c["recall"]), 1.0),
    "ndcg_at_k": lambda c: (float(c["ndcg"]), 1.0),
    "context_coverage_at_k": lambda c: (float(not c["missing_context_ids"]), 1.0),
    "hard_negative_top1_rate": lambda c: (
        (float(c["hard_negative_first"]), 1.0) if c["has_hard_negatives"] else (0.0, 0.0)
    ),
    "citation_precision": _citations_ok,
    "answers_fully_supported_rate": lambda c: (
        (float(not c["citations_not_retrieved"]), 1.0) if c["cited_ids"] else (0.0, 0.0)
    ),
}


def paired_deltas(
    metric: str, baseline_cases: Sequence[dict], candidate_cases: Sequence[dict]
) -> list[tuple[float, float]]:
    """Per-question (candidate - baseline numerator, denominator), paired by question id."""

    by_id = {case["id"]: case for case in candidate_cases}
    if set(by_id) != {case["id"] for case in baseline_cases}:
        raise ValueError("baseline and candidate reports must score the same questions")
    extract = PER_CASE[metric]
    pairs = []
    for before in baseline_cases:
        num_b, den = extract(before)
        num_c, _ = extract(by_id[before["id"]])
        pairs.append((num_c - num_b, den))
    return pairs


def bootstrap_ci(
    pairs: Sequence[tuple[float, float]], *, resamples: int = 10_000, seed: int = 0, alpha: float = 0.05
) -> tuple[float, float] | None:
    """Percentile (1 - alpha) confidence interval for the candidate - baseline delta.

    Questions are resampled with replacement; the same seed gives the same resamples for every
    metric. Resamples with no qualifying questions are skipped. None if no question qualifies.
    """

    if resamples < 1 or not 0 < alpha < 1:
        raise ValueError("resamples must be positive and alpha in (0, 1)")
    if not any(den for _, den in pairs):
        return None
    deltas = [d for d, _ in pairs]
    weights = [w for _, w in pairs]
    n = len(pairs)
    rng = random.Random(seed)
    population = range(n)
    stats: list[float] = []
    for _ in range(resamples):
        sample = rng.choices(population, k=n)
        total = sum(weights[i] for i in sample)
        if total:
            stats.append(sum(deltas[i] for i in sample) / total)
    stats.sort()
    low = stats[int((alpha / 2) * (len(stats) - 1))]
    high = stats[int(round((1 - alpha / 2) * (len(stats) - 1)))]
    return round(low, 4), round(high, 4)
