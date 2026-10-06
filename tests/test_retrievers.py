import numpy as np
import pytest

from rag_gate.data import Document
from rag_gate.retrievers import BM25Retriever, DenseRetriever, RetrieverConfig, build_retriever, chunk

DOCS = [
    Document("keys", "Production API keys rotate every 90 days."),
    Document("leave", "Paid parental leave lasts 16 weeks."),
]


def test_bm25_ranks_the_overlapping_document_and_returns_nothing_for_no_overlap():
    retriever = BM25Retriever(chunk(DOCS, None))
    assert retriever.search("When do production API keys rotate?", 2) == ["keys"]
    assert retriever.search("volcano", 2) == []


def test_chunking_splits_with_overlap_and_collapses_back_to_unique_documents():
    long_doc = Document("long", " ".join(f"w{i}" for i in range(10)))
    chunks = chunk([long_doc], 4, 1)
    assert [text for _, text in chunks] == ["w0 w1 w2 w3", "w3 w4 w5 w6", "w6 w7 w8 w9"]
    retriever = BM25Retriever(chunks + chunk(DOCS, None))
    assert retriever.search("w3 w6 keys", 5) == ["long", "keys"]


class FakeEncoder:
    """Deterministic bag-of-letters vectors so dense retrieval runs without a model download."""

    def encode(self, texts, **_):
        vectors = np.zeros((len(texts), 26))
        for row, text in enumerate(texts):
            for char in text.lower():
                if "a" <= char <= "z":
                    vectors[row, ord(char) - 97] += 1
        return vectors / np.linalg.norm(vectors, axis=1, keepdims=True)


def test_dense_retriever_uses_the_injected_encoder():
    retriever = DenseRetriever(chunk(DOCS, None), FakeEncoder())
    assert retriever.search_many(["parental leave weeks"], 1) == [["leave"]]


def test_dense_model_loading_is_cache_only_unless_download_is_allowed(monkeypatch):
    import sys
    import types

    calls = []

    def fake_model(name, **options):
        calls.append((name, options))
        return FakeEncoder()

    monkeypatch.setitem(sys.modules, "sentence_transformers", types.SimpleNamespace(SentenceTransformer=fake_model))
    config = RetrieverConfig.from_dict({"name": "d", "retriever": "dense", "model": "m"})
    build_retriever(config, DOCS)
    build_retriever(config, DOCS, allow_download=True)
    assert calls == [("m", {"local_files_only": True}), ("m", {"local_files_only": False})]


@pytest.mark.parametrize("raw, message", [
    ({"name": "x", "retriever": "tfidf"}, "retriever"),
    ({"name": "x", "retriever": "dense"}, "model"),
    ({"name": "x", "chunk_words": 0}, "chunk_words"),
    ({"name": "x", "chunk_words": 4, "chunk_overlap": 4}, "chunk_overlap"),
    ({"name": "x", "typo": 1}, "unknown"),
])
def test_invalid_configs_are_rejected(raw, message):
    with pytest.raises(ValueError, match=message):
        RetrieverConfig.from_dict(raw)
