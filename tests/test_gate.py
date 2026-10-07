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
    assert compare(report("a"), report("b", mrr_at_k=0.89), max_drop=0.02, bootstrap=False)["passed"]
    result = compare(report("a"), report("b", mrr_at_k=0.8, hit_at_k=0.0), max_drop=0.02, bootstrap=False)
    assert result["regressions"] == ["hit_at_k", "mrr_at_k"]
    assert result["newly_failing_cases"] == ["c1"]


def test_a_rise_in_hard_negatives_at_rank_one_is_a_regression():
    result = compare(report("a"), report("b", hard_negative_top1_rate=0.5), max_drop=0.02, bootstrap=False)
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
    assert "gate: PASS" in ok.stdout and "query_p50_ms" not in ok.stdout and "p95 ms" in ok.stdout

    out = tmp_path / "gate.json"
    bad = cli("gate", *common, "--candidate", "configs/candidate-chunk-12.json", "--no-bootstrap", "--output", out)
    assert bad.returncode == 1
    assert "mrr_at_k" in json.loads(out.read_text())["regressions"]

    # On six questions the same one-question drop is not statistically distinguishable from noise.
    noisy = cli("gate", *common, "--candidate", "configs/candidate-chunk-12.json")
    assert noisy.returncode == 0, noisy.stdout
    assert "not significant" in noisy.stdout


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


def case(case_id, hit=True, rr=1.0):
    return {"id": case_id, "hit": hit, "reciprocal_rank": rr, "recall": float(hit), "ndcg": rr,
            "missing_context_ids": [] if hit else ["x"], "hard_negative_first": False, "has_hard_negatives": False,
            "cited_ids": [], "citations_not_retrieved": [], "citations_not_relevant": [], "passed": hit}


def scored(name, cases):
    n = len(cases)
    metrics = {"hit_at_k": round(sum(c["hit"] for c in cases) / n, 4),
               "mrr_at_k": round(sum(c["reciprocal_rank"] for c in cases) / n, 4)}
    return {"dataset": {"name": "d", "documents": 9, "cases": n}, "k": 10, "config": {"name": name},
            "metrics": metrics, "cases": cases}


def verdicts(result):
    return {row["metric"]: row["verdict"] for row in result["metrics"]}


def test_identical_runs_pass_with_a_zero_width_interval():
    cases = [case(f"q{i}", hit=i % 3 > 0, rr=1 / (1 + i % 4)) for i in range(200)]
    result = compare(scored("a", cases), scored("b", cases), max_drop=0.02)
    assert result["passed"] and set(verdicts(result).values()) == {"ok"}
    assert all(row["ci_low"] == row["ci_high"] == 0 for row in result["metrics"])


def test_a_known_regression_fails_with_an_interval_below_zero():
    before = [case(f"q{i}") for i in range(200)]
    after = [case(f"q{i}", hit=i >= 40, rr=1.0 if i >= 40 else 0.0) for i in range(200)]  # 20% of questions lost
    result = compare(scored("a", before), scored("b", after), max_drop=0.02)
    assert not result["passed"] and result["regressions"] == ["hit_at_k", "mrr_at_k"]
    row = result["metrics"][0]
    assert row["delta"] == -0.2 and row["ci_high"] < 0 and row["verdict"] == "FAIL"


def test_a_small_noisy_drop_passes_as_not_significant():
    before = [case(f"q{i}", rr=0.5) for i in range(30)]
    after = [case(f"q{i}", rr=0.5) for i in range(30)]
    after[0] = case("q0", hit=False, rr=0.0)  # one question in thirty: a 0.033 drop in hit rate
    after[1] = case("q1", rr=1.0)  # and one unrelated improvement, so MRR barely moves
    result = compare(scored("a", before), scored("b", after), max_drop=0.02)
    hit = result["metrics"][0]
    assert result["passed"] and hit["delta"] < -0.02 and hit["ci_low"] < 0 <= hit["ci_high"]
    assert verdicts(result) == {"hit_at_k": "not significant", "mrr_at_k": "ok"}


def test_bootstrap_is_reproducible_for_a_fixed_seed():
    before = [case(f"q{i}") for i in range(50)]
    after = [case(f"q{i}", hit=i % 7 > 0, rr=1.0 if i % 7 else 0.0) for i in range(50)]
    first = compare(scored("a", before), scored("b", after), max_drop=0.02, seed=7)
    again = compare(scored("a", before), scored("b", after), max_drop=0.02, seed=7)
    assert first["metrics"] == again["metrics"]
