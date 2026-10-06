"""Load labelled retrieval datasets: the committed JSON fixture format or a local BEIR directory."""

from __future__ import annotations

import csv
import json
import re
from dataclasses import dataclass
from pathlib import Path

CITATION_RE = re.compile(r"\[([A-Za-z0-9_.:-]+)\]")


@dataclass(frozen=True)
class Document:
    id: str
    text: str


@dataclass(frozen=True)
class Case:
    """One labelled question.

    relevant_ids: documents that answer the question.
    required_context_ids: evidence that must reach the context window (e.g. a superseded rule
        the answer has to acknowledge). Defaults to relevant_ids.
    hard_negative_ids: plausible near-matches that must not rank first.
    answer: optional reference answer whose [doc-id] citations are checked against what the
        configuration actually retrieves.
    """

    id: str
    query: str
    relevant_ids: frozenset[str]
    required_context_ids: frozenset[str] = frozenset()
    hard_negative_ids: frozenset[str] = frozenset()
    answer: str | None = None

    @property
    def citations(self) -> tuple[str, ...]:
        return tuple(CITATION_RE.findall(self.answer or ""))


@dataclass(frozen=True)
class Dataset:
    name: str
    documents: list[Document]
    cases: list[Case]


def _id_list(case_id: str, row: dict, key: str, known: set[str]) -> frozenset[str]:
    values = row.get(key, [])
    if not isinstance(values, list) or any(not isinstance(v, str) for v in values):
        raise ValueError(f"case {case_id!r}: {key} must be a list of document IDs")
    unknown = set(values) - known
    if unknown:
        raise ValueError(f"case {case_id!r}: {key} contains unknown document IDs {sorted(unknown)}")
    return frozenset(values)


def load_json_fixture(path: Path) -> Dataset:
    """Load and validate a {"corpus": [...], "cases": [...]} file. Raises ValueError on bad labels."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("fixture must be an object with 'corpus' and 'cases'")
    for collection, text_key in (("corpus", "text"), ("cases", "query")):
        rows = payload.get(collection)
        if not isinstance(rows, list) or not rows:
            raise ValueError(f"{collection} must be a non-empty list")
        seen: set[str] = set()
        for row in rows:
            if not isinstance(row, dict):
                raise ValueError(f"{collection} entries must be objects")
            for key in ("id", text_key):
                if not isinstance(row.get(key), str) or not row[key].strip():
                    raise ValueError(f"{collection}.{key} must be a non-empty string")
            if row["id"] in seen:
                raise ValueError(f"duplicate {collection} id: {row['id']!r}")
            seen.add(row["id"])

    documents = [Document(row["id"], row["text"]) for row in payload["corpus"]]
    known = {doc.id for doc in documents}
    cases: list[Case] = []
    for row in payload["cases"]:
        relevant = _id_list(row["id"], row, "relevant_ids", known)
        if not relevant:
            raise ValueError(f"case {row['id']!r}: relevant_ids must not be empty")
        required = _id_list(row["id"], row, "required_context_ids", known)
        negatives = _id_list(row["id"], row, "hard_negative_ids", known)
        if negatives & (relevant | required):
            raise ValueError(f"case {row['id']!r}: hard negatives overlap relevant or required evidence")
        answer = row.get("answer")
        if answer is not None:
            if not isinstance(answer, str) or not answer.strip():
                raise ValueError(f"case {row['id']!r}: answer must be a non-empty string")
            cited = set(CITATION_RE.findall(answer))
            if not cited:
                raise ValueError(f"case {row['id']!r}: answer must cite at least one [doc-id]")
            if cited - known:
                raise ValueError(f"case {row['id']!r}: answer cites unknown documents {sorted(cited - known)}")
        cases.append(Case(row["id"], row["query"], relevant, required, negatives, answer))
    return Dataset(path.stem, documents, cases)


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_beir(dataset_dir: Path, *, split: str = "test") -> Dataset:
    """Load a BEIR-format directory (corpus.jsonl, queries.jsonl, qrels/<split>.tsv).

    Only queries with at least one positive qrel are kept. BEIR has no reference answers,
    so citation metrics are not available for these datasets.
    """

    documents = [
        Document(
            str(row["_id"]),
            " ".join(part for part in (str(row.get("title", "")), str(row.get("text", ""))) if part),
        )
        for row in _jsonl(dataset_dir / "corpus.jsonl")
    ]
    queries = {str(row["_id"]): str(row["text"]) for row in _jsonl(dataset_dir / "queries.jsonl")}
    qrels: dict[str, set[str]] = {}
    with (dataset_dir / "qrels" / f"{split}.tsv").open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle, delimiter="\t"):
            if int(row["score"]) > 0:
                qrels.setdefault(str(row["query-id"]), set()).add(str(row["corpus-id"]))
    cases = [
        Case(query_id, queries[query_id], frozenset(relevant))
        for query_id, relevant in sorted(qrels.items())
        if query_id in queries
    ]
    if not documents or not cases:
        raise ValueError(f"incomplete BEIR dataset at {dataset_dir}")
    return Dataset(dataset_dir.name, documents, cases)


def load_dataset(path: Path) -> Dataset:
    if path.is_dir():
        return load_beir(path)
    if path.suffix == ".json":
        return load_json_fixture(path)
    raise ValueError(f"{path} is neither a .json fixture nor a BEIR directory")
