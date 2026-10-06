"""rag-gate: score retrieval configurations and fail CI when a candidate regresses.

Exit codes: 0 = pass, 1 = candidate regressed beyond the threshold, 2 = bad input or usage.
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
    metrics = ", ".join(
        f"{name}={value}" for name, value in report["metrics"].items() if value is not None
    )
    return f"{report['config']['name']} on {report['dataset']['name']} (k={report['k']}): {metrics}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rag-gate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    def dataset_args(p: argparse.ArgumentParser) -> None:
        p.add_argument("--dataset", type=Path, required=True, help="JSON fixture or BEIR directory")
        p.add_argument("--k", type=int, default=10, help="retrieval cut-off (default 10)")
        p.add_argument("--max-cases", type=int, help="score only the first N cases (file order; BEIR is sorted by query id)")
        p.add_argument("--allow-download", action="store_true",
                       help="let dense retrievers download models; default is local cache only")
        p.add_argument("--output", type=Path, help="write the full JSON report here")

    p_run = sub.add_parser("run", help="score one configuration")
    p_run.add_argument("--config", type=Path, required=True)
    dataset_args(p_run)

    p_cmp = sub.add_parser("compare", help="compare two saved run reports")
    p_cmp.add_argument("baseline", type=Path)
    p_cmp.add_argument("candidate", type=Path)
    p_cmp.add_argument("--max-drop", type=float, default=0.02)
    p_cmp.add_argument("--output", type=Path)

    p_gate = sub.add_parser("gate", help="run baseline and candidate configurations, then compare")
    p_gate.add_argument("--baseline", type=Path, required=True)
    p_gate.add_argument("--candidate", type=Path, required=True)
    p_gate.add_argument("--max-drop", type=float, default=0.02,
                        help="largest tolerated absolute drop in any metric (default 0.02)")
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
                max_drop=args.max_drop,
            )
            _write(args.output, result)
            print(format_comparison(result))
            return 0 if result["passed"] else 1

        dataset = _load(args.dataset, args.max_cases)
        if args.command == "run":
            report = run(dataset, _load_config(args.config), k=args.k, allow_download=args.allow_download)
            _write(args.output, report)
            print(_summary(report))
            return 0

        baseline = run(dataset, _load_config(args.baseline), k=args.k, allow_download=args.allow_download)
        candidate = run(dataset, _load_config(args.candidate), k=args.k, allow_download=args.allow_download)
        result = compare(baseline, candidate, max_drop=args.max_drop)
        _write(args.output, {**result, "baseline_report": baseline, "candidate_report": candidate})
        print(format_comparison(result))
        return 0 if result["passed"] else 1
    except (OSError, ValueError, KeyError, RuntimeError, json.JSONDecodeError) as exc:
        print(f"rag-gate: error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
