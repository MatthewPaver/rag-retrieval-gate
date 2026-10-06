"""Retrievers under test. A configuration chooses one, plus optional chunking."""

from __future__ import annotations

import math
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Protocol, Sequence

from .data import Document

TOKEN_RE = re.compile(r"[a-z0-9]+")
STOP_WORDS = frozenset(
    "a an and are as at be by can do does for from has how in is it of on or that the to was what when which with".split()
)


def tokenise(text: str) -> list[str]:
    return [t for t in TOKEN_RE.findall(text.casefold()) if t not in STOP_WORDS]


@dataclass(frozen=True)
class RetrieverConfig:
    name: str
    retriever: str = "bm25"  # "bm25" or "dense"
    model: str | None = None  # sentence-transformers model id, dense only
    chunk_words: int | None = None  # None = index whole documents
    chunk_overlap: int = 0
    bm25_k1: float = 1.2
    bm25_b: float = 0.75

    @classmethod
    def from_dict(cls, raw: dict) -> "RetrieverConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        config = cls(**raw)
        if config.retriever not in {"bm25", "dense"}:
            raise ValueError("retriever must be 'bm25' or 'dense'")
        if config.retriever == "dense" and not config.model:
            raise ValueError("dense retriever needs a 'model'")
        if config.chunk_words is not None and config.chunk_words < 1:
            raise ValueError("chunk_words must be positive")
        if not 0 <= config.chunk_overlap < (config.chunk_words or 1):
            raise ValueError("chunk_overlap must be >= 0 and smaller than chunk_words")
        return config


def chunk(documents: Sequence[Document], words: int | None, overlap: int = 0) -> list[tuple[str, str]]:
    """Split documents into (parent_id, text) windows of `words` words. None keeps documents whole."""

    if words is None:
        return [(doc.id, doc.text) for doc in documents]
    step = words - overlap
    chunks: list[tuple[str, str]] = []
    for doc in documents:
        tokens = doc.text.split()
        for start in range(0, max(len(tokens), 1), step):
            chunks.append((doc.id, " ".join(tokens[start : start + words])))
            if start + words >= len(tokens):
                break
    return chunks


def _collapse(parent_ids: Sequence[str], ranked_chunk_indices: Sequence[int], k: int) -> list[str]:
    """Map ranked chunks back to unique parent documents, keeping each document's best rank."""

    seen: list[str] = []
    for index in ranked_chunk_indices:
        parent = parent_ids[index]
        if parent not in seen:
            seen.append(parent)
            if len(seen) == k:
                break
    return seen


class Retriever(Protocol):
    def search_many(self, queries: Sequence[str], k: int) -> list[list[str]]: ...


class BM25Retriever:
    """Okapi BM25 over an inverted index. Pure Python, deterministic, ties broken by chunk order."""

    def __init__(self, chunks: list[tuple[str, str]], k1: float = 1.2, b: float = 0.75):
        self.parent_ids = [parent for parent, _ in chunks]
        self.k1, self.b = k1, b
        self.postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
        self.lengths: list[int] = []
        for index, (_, text) in enumerate(chunks):
            tokens = tokenise(text)
            self.lengths.append(len(tokens))
            for term, tf in Counter(tokens).items():
                self.postings[term].append((index, tf))
        n = len(chunks)
        self.avg_len = (sum(self.lengths) / n) if n else 0.0
        self.idf = {
            term: math.log(1 + (n - len(p) + 0.5) / (len(p) + 0.5)) for term, p in self.postings.items()
        }

    def search(self, query: str, k: int) -> list[str]:
        scores: dict[int, float] = defaultdict(float)
        for term in set(tokenise(query)):
            idf = self.idf.get(term)
            if idf is None:
                continue
            for index, tf in self.postings[term]:
                norm = self.k1 * (1 - self.b + self.b * self.lengths[index] / (self.avg_len or 1))
                scores[index] += idf * tf * (self.k1 + 1) / (tf + norm)
        ranked = sorted(scores, key=lambda i: (-scores[i], i))
        return _collapse(self.parent_ids, ranked, k)

    def search_many(self, queries: Sequence[str], k: int) -> list[list[str]]:
        return [self.search(query, k) for query in queries]


class Encoder(Protocol):
    def encode(self, sentences: Sequence[str], **kwargs) -> object: ...


class DenseRetriever:
    """Cosine similarity over normalised embeddings from any encoder with an `encode` method."""

    def __init__(self, chunks: list[tuple[str, str]], encoder: Encoder, batch_size: int = 64):
        import numpy as np

        self._np = np
        self.parent_ids = [parent for parent, _ in chunks]
        self.encoder = encoder
        self.batch_size = batch_size
        self.matrix = self._encode([text for _, text in chunks])

    def _encode(self, texts: Sequence[str]):
        vectors = self._np.asarray(
            self.encoder.encode(list(texts), normalize_embeddings=True, batch_size=self.batch_size),
            dtype="float32",
        )
        if vectors.ndim != 2 or vectors.shape[0] != len(texts):
            raise ValueError("encoder returned an unexpected embedding shape")
        return vectors

    def search_many(self, queries: Sequence[str], k: int) -> list[list[str]]:
        scores = self._encode(queries) @ self.matrix.T
        results = []
        for row in scores:
            ranked = self._np.argsort(-row, kind="stable")
            results.append(_collapse(self.parent_ids, ranked.tolist(), k))
        return results


def load_sentence_transformer(model: str, *, allow_download: bool):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("dense retrieval needs: pip install 'rag-retrieval-gate[dense]'") from exc
    return SentenceTransformer(model, local_files_only=not allow_download)


def build_retriever(
    config: RetrieverConfig,
    documents: Sequence[Document],
    *,
    encoder: Encoder | None = None,
    allow_download: bool = False,
) -> Retriever:
    chunks = chunk(documents, config.chunk_words, config.chunk_overlap)
    if config.retriever == "bm25":
        return BM25Retriever(chunks, config.bm25_k1, config.bm25_b)
    encoder = encoder or load_sentence_transformer(config.model, allow_download=allow_download)
    return DenseRetriever(chunks, encoder)
