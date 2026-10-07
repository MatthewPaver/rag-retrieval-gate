"""Per-case and aggregate retrieval / citation metrics. Every metric is cut off at k."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence

from .data import Case


@dataclass(frozen=True)
class CaseResult:
    id: str
    retrieved_ids: list[str]
    hit: bool
    reciprocal_rank: float
    recall: float
    ndcg: float
    missing_relevant_ids: list[str]
    missing_context_ids: list[str]
    hard_negative_first: bool
    has_hard_negatives: bool
    cited_ids: list[str]
    citations_not_retrieved: list[str]
    citations_not_relevant: list[str]

    @property
    def passed(self) -> bool:
        return (
            self.hit
            and not self.missing_context_ids
            and not self.hard_negative_first
            and not self.citations_not_retrieved
            and not self.citations_not_relevant
        )

    def to_dict(self) -> dict:
        return {**asdict(self), "passed": self.passed}


def score_case(case: Case, retrieved: Sequence[str], k: int) -> CaseResult:
    ranked = list(retrieved)[:k]
    if len(ranked) != len(set(ranked)):
        raise ValueError(f"case {case.id}: retriever returned duplicate document IDs")
    found = case.relevant_ids.intersection(ranked)
    first_hit = next((rank for rank, doc in enumerate(ranked, 1) if doc in case.relevant_ids), None)
    required = case.required_context_ids or case.relevant_ids
    cited = list(dict.fromkeys(case.citations))
    supporting = case.relevant_ids | case.required_context_ids
    dcg = sum(1 / math.log2(rank + 1) for rank, doc in enumerate(ranked, 1) if doc in case.relevant_ids)
    ideal = sum(1 / math.log2(rank + 1) for rank in range(1, min(len(case.relevant_ids), k) + 1))
    return CaseResult(
        id=case.id,
        retrieved_ids=ranked,
        hit=first_hit is not None,
        reciprocal_rank=1 / first_hit if first_hit else 0.0,
        recall=len(found) / len(case.relevant_ids),
        ndcg=dcg / ideal,
        missing_relevant_ids=sorted(case.relevant_ids - found),
        missing_context_ids=sorted(required - set(ranked)),
        hard_negative_first=bool(ranked) and ranked[0] in case.hard_negative_ids,
        has_hard_negatives=bool(case.hard_negative_ids),
        cited_ids=cited,
        citations_not_retrieved=[c for c in cited if c not in ranked],
        citations_not_relevant=[c for c in cited if c not in supporting],
    )


def _mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 4) if values else None


def aggregate(results: Sequence[CaseResult]) -> dict[str, float | int | None]:
    """Query-averaged metrics.

    hard_negative_top1_rate is averaged over cases that label hard negatives; citation metrics over
    cases that carry a reference answer. Each is None when no case qualifies (e.g. BEIR).
    """

    if not results:
        raise ValueError("no cases to score")
    with_negatives = [r for r in results if r.has_hard_negatives]
    cited = [r for r in results if r.cited_ids]
    citations = [
        c not in r.citations_not_retrieved and c not in r.citations_not_relevant for r in cited for c in r.cited_ids
    ]
    return {
        "cases": len(results),
        "hit_at_k": _mean([float(r.hit) for r in results]),
        "mrr_at_k": _mean([r.reciprocal_rank for r in results]),
        "recall_at_k": _mean([r.recall for r in results]),
        "ndcg_at_k": _mean([r.ndcg for r in results]),
        "context_coverage_at_k": _mean([float(not r.missing_context_ids) for r in results]),
        "hard_negative_top1_rate": _mean([float(r.hard_negative_first) for r in with_negatives]),
        "citation_precision": _mean([float(ok) for ok in citations]),
        "answers_fully_supported_rate": _mean([float(not r.citations_not_retrieved) for r in cited]),
        "cases_passed": sum(r.passed for r in results),
    }
