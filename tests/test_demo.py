"""Public examples must be genuine gate outputs, with complete downloadable evidence."""

import json
import subprocess
import sys

from conftest import ROOT

from rag_gate.validation import validate_report


def test_public_demo_builds_real_pass_and_fail_journeys(tmp_path):
    subprocess.run([sys.executable, "scripts/build_demo.py", "--output", str(tmp_path)], cwd=ROOT, check=True)
    bundle = json.loads((tmp_path / "evidence.json").read_text())
    assert bundle["documents"] == json.loads((ROOT / "fixtures/policies.json").read_text())["corpus"]
    for scenario, stem, passed, html in (
        ("regression", "comparison", False, "report.html"),
        ("unchanged", "unchanged", True, "unchanged-report.html"),
    ):
        comparison = bundle["scenarios"][scenario]
        assert comparison == json.loads((tmp_path / f"{stem}.json").read_text())
        assert comparison["passed"] is passed
        for run in (comparison["baseline_report"], comparison["candidate_report"]):
            validate_report(run)
            assert len(run["cases"]) == 6
            assert all(case["passed"] for case in run["cases"])
        assert comparison["baseline_report"]["source_identity"] == comparison["candidate_report"]["source_identity"]
        assert (tmp_path / html).read_text().startswith("<!doctype html>")
    before = bundle["scenarios"]["regression"]["baseline_report"]["cases"][2]
    after = bundle["scenarios"]["regression"]["candidate_report"]["cases"][2]
    assert before["retrieved_ids"][:2] == ["retention-current", "retention-superseded"]
    assert after["retrieved_ids"][:2] == ["retention-superseded", "retention-current"]
    assert bundle["scenarios"]["unchanged"]["candidate_report"]["cases"][2]["retrieved_ids"] == before["retrieved_ids"]
    for name in (
        "index.html",
        "product.js",
        "product.css",
        "policies.json",
        "baseline.json",
        "candidate-chunk-12.json",
    ):
        assert (tmp_path / name).stat().st_size > 0
