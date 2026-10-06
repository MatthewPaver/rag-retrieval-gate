"""Run a configuration against a dataset, and compare a candidate run with a baseline run."""

from __future__ import annotations

from dataclasses import asdict
from typing import Sequence

from .data import Dataset
from .metrics import aggregate, score_case
from .retrievers import Encoder, RetrieverConfig, build_retriever

# Metrics where a fall is a regression. hard_negative_top1_rate is the odd one out: a rise is bad.
HIGHER_IS_BETTER = (
    "hit_at_k",
    "mrr_at_k",
    "recall_at_k",
    "ndcg_at_k",
    "context_coverage_at_k",
    "citation_precision",
    "answers_fully_supported_rate",
)
LOWER_IS_BETTER = ("hard_negative_top1_rate",)


def run(
    dataset: Dataset,
    config: RetrieverConfig,
    *,
    k: int,
    encoder: Encoder | None = None,
    allow_download: bool = False,
) -> dict:
    if k < 1:
        raise ValueError("k must be positive")
    retriever = build_retriever(config, dataset.documents, encoder=encoder, allow_download=allow_download)
    rankings = retriever.search_many([case.query for case in dataset.cases], k)
    results = [score_case(case, ranked, k) for case, ranked in zip(dataset.cases, rankings, strict=True)]
    return {
        "dataset": {"name": dataset.name, "documents": len(dataset.documents), "cases": len(dataset.cases)},
        "config": asdict(config),
        "k": k,
        "metrics": aggregate(results),
        "cases": [result.to_dict() for result in results],
    }


def compare(baseline: dict, candidate: dict, *, max_drop: float) -> dict:
    """Return per-metric deltas and the list of regressions beyond `max_drop` (absolute)."""

    if baseline["dataset"] != candidate["dataset"] or baseline["k"] != candidate["k"]:
        raise ValueError("baseline and candidate must be scored on the same dataset and k")
    if max_drop < 0:
        raise ValueError("max_drop must be >= 0")
    rows, regressions = [], []
    for metric in (*HIGHER_IS_BETTER, *LOWER_IS_BETTER):
        before, after = baseline["metrics"].get(metric), candidate["metrics"].get(metric)
        if before is None or after is None:
            continue
        delta = round(after - before, 4)
        worse_by = -delta if metric in HIGHER_IS_BETTER else delta
        failed = worse_by > max_drop + 1e-9
        rows.append({"metric": metric, "baseline": before, "candidate": after, "delta": delta, "failed": failed})
        if failed:
            regressions.append(metric)
    before_cases = {case["id"]: case["passed"] for case in baseline["cases"]}
    newly_failing = [case["id"] for case in candidate["cases"] if before_cases.get(case["id"]) and not case["passed"]]
    return {
        "baseline": baseline["config"]["name"],
        "candidate": candidate["config"]["name"],
        "dataset": baseline["dataset"],
        "k": baseline["k"],
        "max_drop": max_drop,
        "metrics": rows,
        "regressions": regressions,
        "newly_failing_cases": newly_failing,
        "passed": not regressions,
    }


def format_comparison(result: dict, *, limit_cases: int = 10) -> str:
    lines = [
        f"dataset={result['dataset']['name']} documents={result['dataset']['documents']} "
        f"cases={result['dataset']['cases']} k={result['k']} max_drop={result['max_drop']}",
        f"{'metric':<30}{result['baseline'][:20]:>22}{result['candidate'][:20]:>22}{'delta':>10}",
    ]
    for row in result["metrics"]:
        flag = "  FAIL" if row["failed"] else ""
        lines.append(f"{row['metric']:<30}{row['baseline']:>22.4f}{row['candidate']:>22.4f}{row['delta']:>+10.4f}{flag}")
    failing: Sequence[str] = result["newly_failing_cases"]
    if failing:
        shown = ", ".join(failing[:limit_cases]) + (" ..." if len(failing) > limit_cases else "")
        lines.append(f"newly failing cases ({len(failing)}): {shown}")
    verdict = "PASS" if result["passed"] else "FAIL: " + ", ".join(result["regressions"])
    lines.append(f"gate: {verdict}")
    return "\n".join(lines)
