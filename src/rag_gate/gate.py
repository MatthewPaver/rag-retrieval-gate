"""Run a configuration against a dataset, and compare a candidate run with a baseline run."""

from __future__ import annotations

import math
import os
import platform
import subprocess
import sys
import time
from dataclasses import asdict
from typing import Sequence

from .data import Dataset
from .metrics import aggregate, score_case
from .retrievers import Encoder, PairScorer, RetrieverConfig, build_retriever
from .stats import bootstrap_ci, paired_deltas

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


def _percentile(sorted_values: Sequence[float], q: float) -> float:
    """Nearest-rank percentile of an ascending list."""

    return sorted_values[max(math.ceil(q * len(sorted_values)) - 1, 0)]


def _peak_rss_mb() -> float | None:
    """Peak resident memory of this process so far (it never goes down within a process)."""

    try:
        import resource
    except ImportError:  # pragma: no cover - Windows
        return None
    peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return round(peak / (1024 * 1024 if sys.platform == "darwin" else 1024), 1)


def _model_size(models: Sequence[object]) -> tuple[int | None, float | None]:
    """Parameter count and in-memory weight size (MB) of any torch models; None for BM25 / fakes."""

    params = size = 0
    found = False
    for model in models:
        module = getattr(model, "model", model)  # CrossEncoder wraps a torch module in .model
        parameters = getattr(module, "parameters", None)
        if not callable(parameters):
            continue
        for tensor in parameters():
            found = True
            params += tensor.numel()
            size += tensor.numel() * tensor.element_size()
    return (params, round(size / 1e6, 1)) if found else (None, None)


def hardware() -> dict:
    cpu = platform.processor() or platform.machine()
    try:
        if sys.platform == "darwin":
            cpu = subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                 text=True, timeout=5).stdout.strip() or cpu
        elif os.path.exists("/proc/cpuinfo"):
            with open("/proc/cpuinfo", encoding="utf-8") as handle:
                cpu = next((line.split(":", 1)[1].strip() for line in handle if line.startswith("model name")), cpu)
    except (OSError, subprocess.SubprocessError):
        pass
    return {"cpu": cpu, "cpu_count": os.cpu_count(), "os": f"{platform.system()} {platform.release()}",
            "python": platform.python_version()}


def run(
    dataset: Dataset,
    config: RetrieverConfig,
    *,
    k: int,
    encoder: Encoder | None = None,
    scorer: PairScorer | None = None,
    allow_download: bool = False,
    device: str = "cpu",
) -> dict:
    """Score one configuration. Queries run one at a time so latency is per question, as served."""

    if k < 1:
        raise ValueError("k must be positive")
    started = time.perf_counter()
    retriever = build_retriever(config, dataset.documents, encoder=encoder, scorer=scorer,
                                allow_download=allow_download, device=device)
    build_s = time.perf_counter() - started
    rankings, latencies = [], []
    for case in dataset.cases:
        started = time.perf_counter()
        rankings.append(retriever.search_many([case.query], k)[0])
        latencies.append(time.perf_counter() - started)
    latencies.sort()
    params, model_mb = _model_size(getattr(retriever, "models", []))
    results = [score_case(case, ranked, k) for case, ranked in zip(dataset.cases, rankings, strict=True)]
    return {
        "dataset": {"name": dataset.name, "documents": len(dataset.documents), "cases": len(dataset.cases)},
        "config": asdict(config),
        "k": k,
        "metrics": aggregate(results),
        "performance": {
            "index_build_s": round(build_s, 2),  # includes loading any model from local disk
            "query_p50_ms": round(_percentile(latencies, 0.50) * 1000, 2),
            "query_p95_ms": round(_percentile(latencies, 0.95) * 1000, 2),
            "peak_rss_mb": _peak_rss_mb(),
            "model_params": params,
            "model_mb": model_mb,
            "device": device if getattr(retriever, "models", []) else "cpu",
        },
        "hardware": hardware(),
        "cases": [result.to_dict() for result in results],
    }


def compare(
    baseline: dict,
    candidate: dict,
    *,
    max_drop: float,
    bootstrap: bool = True,
    alpha: float = 0.05,
    resamples: int = 10_000,
    seed: int = 0,
) -> dict:
    """Per-metric deltas, paired-bootstrap confidence intervals and a verdict for each metric.

    A metric fails when the candidate is worse by more than `max_drop` (absolute) and, with
    `bootstrap`, the (1 - alpha) interval for the delta excludes zero on the worse side. Without
    `bootstrap` the threshold alone decides, which is the only meaningful mode on a tiny fixture.
    """

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
        sign = -1 if metric in HIGHER_IS_BETTER else 1  # sign * delta > 0 means worse
        worse_by = sign * delta
        ci = None
        if bootstrap:
            ci = bootstrap_ci(paired_deltas(metric, baseline["cases"], candidate["cases"]),
                              resamples=resamples, seed=seed, alpha=alpha)
        if ci is None:
            significantly_worse = significantly_better = not bootstrap
        else:
            significantly_worse = min(sign * ci[0], sign * ci[1]) > 0
            significantly_better = max(sign * ci[0], sign * ci[1]) < 0
        if worse_by > 1e-9 and not significantly_worse:
            verdict = "not significant"
        elif worse_by > max_drop + 1e-9:
            verdict = "FAIL"
        elif worse_by > 1e-9:
            verdict = "within max-drop"
        elif bootstrap and significantly_better and delta != 0:
            verdict = "better"
        else:
            verdict = "ok"
        failed = verdict == "FAIL"
        rows.append({"metric": metric, "baseline": before, "candidate": after, "delta": delta,
                     "ci_low": ci[0] if ci else None, "ci_high": ci[1] if ci else None,
                     "verdict": verdict, "failed": failed})
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
        "bootstrap": {"resamples": resamples, "seed": seed, "alpha": alpha} if bootstrap else None,
        "metrics": rows,
        "performance": {"baseline": baseline.get("performance"), "candidate": candidate.get("performance")},
        "hardware": candidate.get("hardware") or baseline.get("hardware"),
        "regressions": regressions,
        "newly_failing_cases": newly_failing,
        "passed": not regressions,
    }


def format_comparison(result: dict, *, limit_cases: int = 10) -> str:
    boot = result.get("bootstrap")
    ci_label = f"{100 * (1 - boot['alpha']):g}% CI" if boot else "CI"
    header = (f"dataset={result['dataset']['name']} documents={result['dataset']['documents']} "
              f"cases={result['dataset']['cases']} k={result['k']} max_drop={result['max_drop']}")
    if boot:
        header += f" bootstrap={boot['resamples']} seed={boot['seed']} alpha={boot['alpha']}"
    else:
        header += " bootstrap=off"
    lines = [
        header,
        f"{'metric':<30}{result['baseline'][:20]:>22}{result['candidate'][:20]:>22}{'delta':>10}"
        f"{ci_label:>22}  verdict",
    ]
    for row in result["metrics"]:
        ci = f"[{row['ci_low']:+.4f}, {row['ci_high']:+.4f}]" if row.get("ci_low") is not None else "-"
        lines.append(f"{row['metric']:<30}{row['baseline']:>22.4f}{row['candidate']:>22.4f}"
                     f"{row['delta']:>+10.4f}{ci:>22}  {row.get('verdict', 'FAIL' if row['failed'] else 'ok')}")
    perf = result.get("performance") or {}
    if perf.get("baseline") and perf.get("candidate"):
        lines.append(f"{'cost (not gated)':<30}{'build s':>10}{'p50 ms':>10}{'p95 ms':>10}{'peak RSS MB':>13}{'model MB':>10}")
        for side in ("baseline", "candidate"):
            p = perf[side]
            lines.append(f"{result[side][:28]:<30}{p['index_build_s']:>10.2f}{p['query_p50_ms']:>10.2f}"
                         f"{p['query_p95_ms']:>10.2f}{_num(p['peak_rss_mb']):>13}{_num(p['model_mb']):>10}")
    if result.get("hardware"):
        hw = result["hardware"]
        lines.append(f"hardware: {hw['cpu']}, {hw['cpu_count']} cores, {hw['os']}, Python {hw['python']}")
    failing: Sequence[str] = result["newly_failing_cases"]
    if failing:
        shown = ", ".join(failing[:limit_cases]) + (" ..." if len(failing) > limit_cases else "")
        lines.append(f"newly failing cases ({len(failing)}): {shown}")
    verdict = "PASS" if result["passed"] else "FAIL: " + ", ".join(result["regressions"])
    lines.append(f"gate: {verdict}")
    return "\n".join(lines)


def _num(value: float | None) -> str:
    return "-" if value is None else f"{value:.1f}"
