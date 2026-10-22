"""Canonical content identities for portable, unsigned run reports."""

from __future__ import annotations

import hashlib
import json
from importlib.metadata import PackageNotFoundError, version

from .data import Dataset

SCHEMA_VERSION = 2
METRIC_SCHEMA = 1
try:
    TOOL_VERSION = version("rag-retrieval-gate")
except PackageNotFoundError:
    TOOL_VERSION = "0.1.0"


def digest(value: object) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def dataset_identity(dataset: Dataset) -> dict[str, str]:
    corpus = digest([(doc.id, doc.text) for doc in sorted(dataset.documents, key=lambda doc: doc.id)])
    cases = sorted(dataset.cases, key=lambda case: case.id)
    queries = digest([(case.id, case.query) for case in cases])
    labels = digest(
        [
            (
                case.id,
                sorted(case.relevant_ids),
                sorted(case.required_context_ids),
                sorted(case.hard_negative_ids),
                case.answer,
            )
            for case in cases
        ]
    )
    return {"algorithm": "sha256-canonical-json-v1", "corpus": corpus, "queries": queries, "labels": labels}
