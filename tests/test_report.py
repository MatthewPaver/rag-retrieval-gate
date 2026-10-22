import copy
import json
from html.parser import HTMLParser

import pytest
from conftest import ROOT
from test_gate import cli

from rag_gate.data import Dataset, Document, load_dataset
from rag_gate.gate import compare, run
from rag_gate.report import render_html
from rag_gate.retrievers import RetrieverConfig


@pytest.fixture
def reports():
    dataset = load_dataset(ROOT / "fixtures/policies.json")
    config = RetrieverConfig.from_dict(json.loads((ROOT / "configs/baseline.json").read_text()))
    baseline = run(dataset, config, k=3)
    return baseline, copy.deepcopy(baseline)


@pytest.mark.parametrize("bad", [None, True, "0.9", float("nan"), float("inf"), -0.1, 1.1])
def test_invalid_required_metric_never_passes(reports, bad):
    baseline, candidate = reports
    candidate["metrics"]["mrr_at_k"] = bad
    with pytest.raises(ValueError, match="invalid metric"):
        compare(baseline, candidate, max_drop=0.02)


def test_missing_metrics_and_legacy_identity_are_rejected(reports):
    baseline, candidate = reports
    candidate["metrics"] = {}
    with pytest.raises(ValueError, match="metrics"):
        compare(baseline, candidate, max_drop=0.02)
    candidate = copy.deepcopy(baseline)
    del candidate["source_identity"]
    with pytest.raises(ValueError, match="fingerprints"):
        compare(baseline, candidate, max_drop=0.02)


def test_content_change_with_same_counts_cannot_be_compared(reports):
    baseline, _ = reports
    dataset = load_dataset(ROOT / "fixtures/policies.json")
    changed = Dataset(
        dataset.name, [Document(doc.id, doc.text + " altered") for doc in dataset.documents], dataset.cases
    )
    config = RetrieverConfig.from_dict(baseline["config"])
    candidate = run(changed, config, k=3)
    with pytest.raises(ValueError, match="same dataset content"):
        compare(baseline, candidate, max_drop=0.02)


def test_different_case_ids_and_invalid_per_case_scores_are_rejected(reports):
    baseline, candidate = reports
    candidate["cases"][0]["id"] = "different-question"
    with pytest.raises(ValueError, match="same questions"):
        compare(baseline, candidate, max_drop=0.02)
    candidate = copy.deepcopy(baseline)
    candidate["cases"][0]["reciprocal_rank"] = float("nan")
    with pytest.raises(ValueError, match="reciprocal_rank"):
        compare(baseline, candidate, max_drop=0.02)


def test_delta_is_not_rounded_before_threshold(reports):
    baseline, candidate = reports
    baseline["metrics"]["mrr_at_k"] = 0.9
    candidate["metrics"]["mrr_at_k"] = 0.87999
    for case in baseline["cases"]:
        case["reciprocal_rank"] = 0.9
    for case in candidate["cases"]:
        case["reciprocal_rank"] = 0.87999
    result = compare(baseline, candidate, max_drop=0.02, bootstrap=False)
    assert "mrr_at_k" in result["regressions"]


def test_portable_html_escapes_inputs_and_contains_case_evidence(reports):
    baseline, candidate = reports
    result = compare(baseline, candidate, max_drop=0.02, bootstrap=False)
    result["candidate"] = '<script>alert("x")</script>'
    output = render_html(result)
    HTMLParser().feed(output)
    assert "<script>" not in output
    assert "&lt;script&gt;" in output
    assert "No detected regression" in output
    assert "Ranking metrics can regress even when every case passes" in output
    assert "data:application/json;base64," in output
    assert "Not estimated" in output
    assert 'src="http' not in output and 'href="http' not in output


def test_html_cli_failure_invalid_and_json_roundtrip(tmp_path):
    out, html = tmp_path / "result.json", tmp_path / "result.html"
    result = cli(
        "gate",
        "--dataset",
        "fixtures/policies.json",
        "--k",
        "3",
        "--baseline",
        "configs/baseline.json",
        "--candidate",
        "configs/candidate-chunk-12.json",
        "--no-bootstrap",
        "--output",
        out,
        "--html",
        html,
    )
    assert result.returncode == 1, result.stderr
    assert "Candidate regresses" in html.read_text()
    report = json.loads(out.read_text())
    assert report["source_identity"] and report["reproduce"].startswith("rag-gate gate")
    baseline = tmp_path / "baseline.json"
    baseline.write_text(json.dumps(report["baseline_report"]))
    candidate = tmp_path / "candidate.json"
    candidate.write_text(json.dumps(report["candidate_report"]))
    saved = cli("compare", baseline, candidate, "--no-bootstrap", "--html", html)
    assert saved.returncode == 1
    broken = json.loads(candidate.read_text())
    broken["metrics"] = {}
    candidate.write_text(json.dumps(broken))
    invalid = cli("compare", baseline, candidate, "--html", html)
    assert invalid.returncode == 2
    assert "Invalid comparison" in html.read_text() and "No gate verdict" in html.read_text()
    assert "Candidate regresses" not in html.read_text()


def test_max_cases_zero_is_a_usage_error():
    result = cli("run", "--dataset", "fixtures/policies.json", "--config", "configs/baseline.json", "--max-cases", "0")
    assert result.returncode == 2 and "max-cases must be positive" in result.stderr


def test_saved_aggregate_must_match_case_evidence(reports):
    baseline, candidate = reports
    candidate["cases"][0]["reciprocal_rank"] = 0.0
    with pytest.raises(ValueError, match="does not match per-case evidence"):
        compare(baseline, candidate, max_drop=0.02, bootstrap=False)
