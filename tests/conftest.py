import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def small_fixture(tmp_path):
    def make(**case_overrides):
        case = {
            "id": "refund-q",
            "query": "refund window",
            "relevant_ids": ["refund"],
            "answer": "Refunds are accepted within thirty days [refund].",
        }
        case.update(case_overrides)
        path = tmp_path / "cases.json"
        path.write_text(
            json.dumps(
                {
                    "corpus": [
                        {"id": "refund", "text": "Refund window is thirty days."},
                        {"id": "shipping", "text": "Shipping takes two weeks."},
                    ],
                    "cases": [case],
                }
            )
        )
        return path

    return make
