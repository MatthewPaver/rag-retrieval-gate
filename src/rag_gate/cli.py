"""rag-gate: score retrieval configurations and fail CI when a candidate regresses.

Exit codes: 0 = pass, 1 = candidate significantly worse by more than --max-drop, 2 = bad input or usage.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .data import Dataset, load_dataset
from .gate import compare, format_comparison, run
from .retrievers import RetrieverConfig


def _load_config(path: Path) -> RetrieverConfig:
    return RetrieverConfig.from_dict(json.loads(path.read_text(encoding="utf-8")))


def _load(path: Path, max_cases: int | None) -> Dataset:
    dataset = load_dataset(path)
    if max_cases is not None:
        dataset = Dataset(dataset.name, dataset.documents, dataset.cases[:max_cases])
    return dataset


def _write(path: Path | None, payload: dict) -> None:
    if path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def _summary(report: dict) -> str:
    metrics = ", ".join(f"{name}={value}" for name, value in report["metrics"].items() if value is not None)
    return f"{report['config']['name']} on {report['dataset']['name']} (k={report['k']}): {metrics}"


def _gate_options(args: argparse.Namespace) -> dict:
    return {
        "max_drop": args.max_drop,
        "bootstrap": not args.no_bootstrap,
        "alpha": args.alpha,
        "resamples": args.resamples,
        "seed": args.seed,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rag-gate", description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="command", required=True)

    def dataset_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dataset", type=Path, required=True, help="JSON fixture or BEIR directory")
        p.add_argument("--k", type=int, default=10, help="retrieval cut-off (default 10)")
        p.add_argument(
            "--max-cases", type=int, help="score only the first N cases (file order; BEIR is sorted by query id)"
        )
        p.add_argument(
            "--allow-download",
            action="store_true",
            help="let dense retrievers download models; default is local cache only",
        )
        p.add_argument("--device", default="cpu", help="torch device for dense / re-rank models (default cpu)")
        p.add_argument("--output", type=Path, help="write the full JSON report here")

    p_run = sub.add_parser("run", help="score one configuration")
    p_run.add_argument("--config", type=Path, required=True)
    dataset_args(p_run)

    def gate_args(p: argparse.ArgumentParser) -> None:
        p.add_argument(
            "--max-drop",
            type=float,
            default=0.02,
            help="smallest absolute drop in a metric that counts as a regression (default 0.02)",
        )
        p.add_argument(
            "--alpha",
            type=float,
            default=0.05,
            help="a drop must also be significant: the (1 - alpha) bootstrap CI excludes zero (default 0.05)",
        )
        p.add_argument("--resamples", type=int, default=10_000, help="paired bootstrap resamples (default 10000)")
        p.add_argument("--seed", type=int, default=0, help="bootstrap seed (default 0)")
        p.add_argument(
            "--no-bootstrap",
            action="store_true",
            help="threshold only, no significance test; for tiny fixtures where a bootstrap has no power",
        )

    p_cmp = sub.add_parser("compare", help="compare two saved run reports")
    p_cmp.add_argument("baseline", type=Path)
    p_cmp.add_argument("candidate", type=Path)
    p_cmp.add_argument("--output", type=Path)
    gate_args(p_cmp)

    p_gate = sub.add_parser("gate", help="run baseline and candidate configurations, then compare")
    p_gate.add_argument("--baseline", type=Path, required=True)
    p_gate.add_argument("--candidate", type=Path, required=True)
    gate_args(p_gate)
    dataset_args(p_gate)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "compare":
            result = compare(
                json.loads(args.baseline.read_text(encoding="utf-8")),
                json.loads(args.candidate.read_text(encoding="utf-8")),
                **_gate_options(args),
            )
            _write(args.output, result)
            print(format_comparison(result))
            return 0 if result["passed"] else 1

        dataset = _load(args.dataset, args.max_cases)
        if args.command == "run":
            report = run(
                dataset, _load_config(args.config), k=args.k, allow_download=args.allow_download, device=args.device
            )
            _write(args.output, report)
            print(_summary(report))
            return 0

        options = {"k": args.k, "allow_download": args.allow_download, "device": args.device}
        baseline = run(dataset, _load_config(args.baseline), **options)
        candidate = run(dataset, _load_config(args.candidate), **options)
        result = compare(baseline, candidate, **_gate_options(args))
        _write(args.output, {**result, "baseline_report": baseline, "candidate_report": candidate})
        print(format_comparison(result))
        return 0 if result["passed"] else 1
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"rag-gate: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
