import pytest

from rag_gate.data import Case
from rag_gate.metrics import aggregate, score_case


def case(**kw):
    defaults = dict(id="c", query="q", relevant_ids=frozenset({"rel"}))
    defaults.update(kw)
    return Case(**defaults)


def test_metrics_respect_the_cutoff():
    result = score_case(case(), ["other", "rel"], k=1)
    assert (result.hit, result.reciprocal_rank, result.recall) == (False, 0.0, 0.0)
    assert score_case(case(), ["other", "rel"], k=2).reciprocal_rank == 0.5


def test_required_context_and_hard_negatives_are_reported():
    c = case(required_context_ids=frozenset({"rel", "old"}), hard_negative_ids=frozenset({"decoy"}))
    result = score_case(c, ["decoy", "rel"], k=3)
    assert result.hard_negative_first
    assert result.missing_context_ids == ["old"]
    assert not result.passed
    metrics = aggregate([result])
    assert metrics["hard_negative_top1_rate"] == 1.0
    assert metrics["context_coverage_at_k"] == 0.0


def test_citation_must_be_retrieved_and_labelled_as_evidence():
    c = case(answer="Claim one [rel]. Claim two [decoy].", hard_negative_ids=frozenset({"decoy"}))
    result = score_case(c, ["rel", "decoy"], k=2)
    assert result.citations_not_retrieved == []
    assert result.citations_not_relevant == ["decoy"]
    assert aggregate([result])["citation_precision"] == 0.5

    dropped = score_case(c, ["rel"], k=1)
    assert dropped.citations_not_retrieved == ["decoy"]
    assert aggregate([dropped])["answers_fully_supported_rate"] == 0.0


def test_metrics_without_answers_or_negatives_are_none_not_zero():
    metrics = aggregate([score_case(case(), ["rel"], k=10)])
    assert metrics["citation_precision"] is None
    assert metrics["hard_negative_top1_rate"] is None
    assert metrics["ndcg_at_k"] == 1.0


def test_duplicate_rankings_cannot_inflate_scores():
    with pytest.raises(ValueError, match="duplicate"):
        score_case(case(), ["rel", "rel"], k=2)
