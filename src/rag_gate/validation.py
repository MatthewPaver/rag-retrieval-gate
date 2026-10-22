"""Fail closed on incomplete or incomparable saved reports."""

from __future__ import annotations

import math
import re

from .identity import METRIC_SCHEMA, SCHEMA_VERSION, digest
from .stats import PER_CASE

OPTIONAL = {"hard_negative_top1_rate", "citation_precision", "answers_fully_supported_rate"}


def finite_unit(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value) and 0 <= value <= 1


def validate_report(report: dict) -> None:
    if (
        not isinstance(report, dict)
        or report.get("schema_version") != SCHEMA_VERSION
        or report.get("metric_schema") != METRIC_SCHEMA
    ):
        raise ValueError("unsupported or missing report schema; regenerate both run reports")
    identity = report.get("source_identity")
    if (
        not isinstance(identity, dict)
        or identity.get("algorithm") != "sha256-canonical-json-v1"
        or any(
            not isinstance(identity.get(key), str) or not re.fullmatch(r"[0-9a-f]{64}", identity[key])
            for key in ("corpus", "queries", "labels")
        )
    ):
        raise ValueError("missing or invalid source fingerprints; regenerate both run reports")
    if not isinstance(report.get("tool_version"), str) or not report["tool_version"]:
        raise ValueError("missing tool version")
    config = report.get("config")
    if not isinstance(config, dict) or not isinstance(config.get("name"), str) or not config["name"]:
        raise ValueError("invalid report configuration")
    if report.get("config_identity") != digest(config):
        raise ValueError("configuration fingerprint does not match the saved configuration")
    if type(report.get("k")) is not int or report["k"] < 1:
        raise ValueError("report k must be positive")
    dataset, cases, metrics = report.get("dataset"), report.get("cases"), report.get("metrics")
    if not isinstance(dataset, dict) or type(dataset.get("documents")) is not int or dataset["documents"] < 1:
        raise ValueError("invalid dataset metadata")
    if not isinstance(cases, list) or not cases or dataset.get("cases") != len(cases):
        raise ValueError("report must contain the declared non-empty case set")
    if not isinstance(metrics, dict) or not metrics:
        raise ValueError("missing required metrics")
    ids = set()
    for case in cases:
        if not isinstance(case, dict) or not isinstance(case.get("id"), str) or not case["id"] or case["id"] in ids:
            raise ValueError("case IDs must be present and unique")
        ids.add(case["id"])
        for field in ("hit", "passed", "has_hard_negatives", "hard_negative_first"):
            if type(case.get(field)) is not bool:
                raise ValueError(f"case {case['id']}: invalid {field}")
        for field in ("reciprocal_rank", "recall", "ndcg"):
            if not finite_unit(case.get(field)):
                raise ValueError(f"case {case['id']}: invalid {field}")
        for field in (
            "retrieved_ids",
            "missing_context_ids",
            "cited_ids",
            "citations_not_retrieved",
            "citations_not_relevant",
        ):
            values = case.get(field)
            if not isinstance(values, list) or any(not isinstance(value, str) for value in values):
                raise ValueError(f"case {case['id']}: invalid {field}")
    for metric, extract in PER_CASE.items():
        if metric not in metrics:
            raise ValueError(f"missing required metric: {metric}")
        value = metrics[metric]
        pairs = [extract(case) for case in cases]
        denominator = sum(pair[1] for pair in pairs)
        applicable = denominator > 0
        if value is None and metric in OPTIONAL and not applicable:
            continue
        if not applicable or not finite_unit(value):
            raise ValueError(f"invalid metric: {metric}; use null only when no labels qualify")
        expected = sum(pair[0] for pair in pairs) / denominator
        if not (math.isclose(value, expected, abs_tol=1e-12) or math.isclose(value, round(expected, 4), abs_tol=1e-12)):
            raise ValueError(f"metric {metric} does not match per-case evidence; regenerate the run report")
