import json
import subprocess
import sys

import pytest

from rag_gate.gate import compare
from conftest import ROOT


def report(name, **metrics):
    base = {"hit_at_k": 1.0, "mrr_at_k": 0.9, "hard_negative_top1_rate": 0.0, "citation_precision": None}
    base.update(metrics)
    return {"dataset": {"name": "d", "documents": 2, "cases": 1}, "k": 3, "config": {"name": name},
            "metrics": base, "cases": [{"id": "c1", "passed": base["hit_at_k"] == 1.0}]}


def test_drop_within_threshold_passes_and_beyond_fails():
    assert compare(report("a"), report("b", mrr_at_k=0.89), max_drop=0.02)["passed"]
    result = compare(report("a"), report("b", mrr_at_k=0.8, hit_at_k=0.0), max_drop=0.02)
    assert result["regressions"] == ["hit_at_k", "mrr_at_k"]
    assert result["newly_failing_cases"] == ["c1"]


def test_a_rise_in_hard_negatives_at_rank_one_is_a_regression():
    result = compare(report("a"), report("b", hard_negative_top1_rate=0.5), max_drop=0.02)
    assert result["regressions"] == ["hard_negative_top1_rate"]


def test_reports_from_different_datasets_or_k_cannot_be_compared():
    other = report("b")
    other["k"] = 5
    with pytest.raises(ValueError, match="same dataset"):
        compare(report("a"), other, max_drop=0.02)


def cli(*args):
    return subprocess.run([sys.executable, "-m", "rag_gate", *map(str, args)],
                          cwd=ROOT, capture_output=True, text=True, timeout=60)


def test_cli_gate_passes_for_equivalent_config_and_exits_one_on_regression(tmp_path):
    common = ["--dataset", "fixtures/policies.json", "--k", "3", "--baseline", "configs/baseline.json"]
    ok = cli("gate", *common, "--candidate", "configs/candidate-bm25-b05.json")
    assert ok.returncode == 0, ok.stdout + ok.stderr
    assert "gate: PASS" in ok.stdout

    out = tmp_path / "gate.json"
    bad = cli("gate", *common, "--candidate", "configs/candidate-chunk-12.json", "--output", out)
    assert bad.returncode == 1
    assert "mrr_at_k" in json.loads(out.read_text())["regressions"]


def test_cli_run_then_compare_round_trips_saved_reports(tmp_path):
    for name in ("baseline", "candidate-chunk-12"):
        result = cli("run", "--dataset", "fixtures/policies.json", "--k", "3",
                     "--config", f"configs/{name}.json", "--output", tmp_path / f"{name}.json")
        assert result.returncode == 0, result.stderr
    loose = cli("compare", tmp_path / "baseline.json", tmp_path / "candidate-chunk-12.json", "--max-drop", "0.1")
    assert loose.returncode == 0


def test_cli_input_errors_exit_two_without_a_traceback(tmp_path):
    bad = tmp_path / "bad.json"
    bad.write_text("{}")
    result = cli("run", "--dataset", bad, "--config", "configs/baseline.json")
    assert result.returncode == 2
    assert "Traceback" not in result.stderr
