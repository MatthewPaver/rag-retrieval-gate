"""Score each config in its own process and print the README results table (quality, CI, cost).

Each config runs as a separate `rag-gate run`, so peak memory is that config's alone. Every
candidate is then compared with the first config (the baseline) using the gate's default
bootstrap settings. Usage:

    python scripts/results_table.py --dataset data/beir/scifact --k 10 \
        configs/baseline.json configs/candidate-minilm.json ...
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from rag_gate.gate import compare

METRICS = ("hit_at_k", "mrr_at_k", "ndcg_at_k")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("configs", nargs="+", type=Path, help="baseline first, then candidates")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--max-drop", type=float, default=0.02)
    parser.add_argument("--reports", type=Path, default=Path("reports"))
    parser.add_argument("--reuse", action="store_true", help="reuse existing reports instead of re-running")
    parser.add_argument("--allow-download", action="store_true")
    args = parser.parse_args()

    reports = []
    for config in args.configs:
        out = args.reports / f"{args.dataset.name}-{config.stem}.json"
        if not (args.reuse and out.exists()):
            command = [
                sys.executable,
                "-m",
                "rag_gate",
                "run",
                "--dataset",
                str(args.dataset),
                "--k",
                str(args.k),
                "--config",
                str(config),
                "--output",
                str(out),
            ]
            if args.allow_download:
                command.append("--allow-download")
            print("$", " ".join(command[1:]), file=sys.stderr)
            subprocess.run(command, check=True)
        reports.append(json.loads(out.read_text(encoding="utf-8")))

    baseline = reports[0]
    hw = baseline["hardware"]
    print(
        f"{baseline['dataset']['name']}, k={args.k}, {baseline['dataset']['cases']} queries; "
        f"95% CI = paired bootstrap of candidate - baseline, 10,000 resamples, seed 0. "
        f"Hardware: {hw['cpu']}, {hw['cpu_count']} cores, {hw['os']}, Python {hw['python']}, CPU only.\n"
    )
    print(
        "| Configuration | hit@10 | MRR@10 (Δ, 95% CI) | nDCG@10 (Δ, 95% CI) | Gate | Build s | p50 / p95 ms | Peak RSS MB | Model MB |"
    )
    print("| --- | --- | --- | --- | --- | --- | --- | --- | --- |")
    for report in reports:
        perf = report["performance"]
        model_mb = "-" if perf["model_mb"] is None else f"{perf['model_mb']:.0f}"
        cost = (
            f"{perf['index_build_s']:.1f} | {perf['query_p50_ms']:.1f} / {perf['query_p95_ms']:.1f} | "
            f"{perf['peak_rss_mb']:.0f} | {model_mb}"
        )
        name = report["config"]["name"]
        m = report["metrics"]
        if report is baseline:
            print(
                f"| `{name}` | {m['hit_at_k']:.4f} | {m['mrr_at_k']:.4f} | {m['ndcg_at_k']:.4f} | baseline | {cost} |"
            )
            continue
        result = compare(baseline, report, max_drop=args.max_drop)
        rows = {row["metric"]: row for row in result["metrics"]}

        def cell(metric: str) -> str:
            r = rows[metric]
            return f"{r['candidate']:.4f} ({r['delta']:+.4f}, [{r['ci_low']:+.4f}, {r['ci_high']:+.4f}])"

        verdict = "PASS" if result["passed"] else "FAIL: " + ", ".join(result["regressions"])
        better = [metric for metric, row in rows.items() if row["verdict"] == "better"]
        if result["passed"] and better:
            verdict += f" (significantly better: {', '.join(better)})"
        print(f"| `{name}` | {m['hit_at_k']:.4f} | {cell('mrr_at_k')} | {cell('ndcg_at_k')} | {verdict} | {cost} |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
