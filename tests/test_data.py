import json

import pytest
from conftest import ROOT

from rag_gate.data import load_beir, load_dataset, load_json_fixture


def test_committed_fixture_loads_with_answers_and_hard_negatives():
    dataset = load_json_fixture(ROOT / "fixtures/policies.json")
    assert len(dataset.cases) == 6
    assert all(case.citations for case in dataset.cases)
    assert any(case.hard_negative_ids for case in dataset.cases)


@pytest.mark.parametrize(
    "overrides, message",
    [
        ({"relevant_ids": ["missing"]}, "unknown"),
        ({"relevant_ids": []}, "must not be empty"),
        ({"relevant_ids": "refund"}, "list"),
        ({"query": " "}, "query"),
        ({"hard_negative_ids": ["refund"]}, "overlap"),
        ({"answer": "No citation here."}, "cite"),
        ({"answer": "Cites a ghost [ghost]."}, "unknown"),
    ],
)
def test_bad_labels_are_rejected_with_a_specific_message(small_fixture, overrides, message):
    with pytest.raises(ValueError, match=message):
        load_json_fixture(small_fixture(**overrides))


def test_duplicate_ids_are_rejected(small_fixture):
    path = small_fixture()
    payload = json.loads(path.read_text())
    payload["corpus"].append(payload["corpus"][0])
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="duplicate"):
        load_json_fixture(path)


def test_beir_loader_keeps_only_queries_with_positive_qrels(tmp_path):
    (tmp_path / "qrels").mkdir()
    (tmp_path / "corpus.jsonl").write_text(json.dumps({"_id": "d1", "title": "Title", "text": "Body"}) + "\n")
    (tmp_path / "queries.jsonl").write_text(
        json.dumps({"_id": "q1", "text": "query"}) + "\n" + json.dumps({"_id": "q2", "text": "unused"}) + "\n"
    )
    (tmp_path / "qrels" / "test.tsv").write_text("query-id\tcorpus-id\tscore\nq1\td1\t1\nq2\td1\t0\n")

    dataset = load_beir(tmp_path)

    assert [d.text for d in dataset.documents] == ["Title Body"]
    assert [(c.id, set(c.relevant_ids), c.answer) for c in dataset.cases] == [("q1", {"d1"}, None)]
    assert load_dataset(tmp_path).name == tmp_path.name
