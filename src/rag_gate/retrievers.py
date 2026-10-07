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
    retriever: str = "bm25"  # "bm25", "dense" or "hybrid" (BM25 + dense, reciprocal rank fusion)
    model: str | None = None  # sentence-transformers model id, dense and hybrid only
    chunk_words: int | None = None  # None = index whole documents
    chunk_overlap: int = 0
    bm25_k1: float = 1.2
    bm25_b: float = 0.75
    rrf_k: int = 60  # RRF constant from Cormack, Clarke & Buttcher (2009); fixed, not tuned
    fusion_depth: int = 100  # documents taken from each retriever before fusion
    rerank_model: str | None = None  # optional cross-encoder applied to the first stage's top documents
    rerank_depth: int = 100  # documents re-scored by the cross-encoder (BEIR paper re-ranks BM25 top 100)

    @classmethod
    def from_dict(cls, raw: dict) -> "RetrieverConfig":
        known = set(cls.__dataclass_fields__)
        unknown = set(raw) - known
        if unknown:
            raise ValueError(f"unknown config keys: {sorted(unknown)}")
        config = cls(**raw)
        if config.retriever not in {"bm25", "dense", "hybrid"}:
            raise ValueError("retriever must be 'bm25', 'dense' or 'hybrid'")
        if config.retriever in {"dense", "hybrid"} and not config.model:
            raise ValueError(f"{config.retriever} retriever needs a 'model'")
        if config.rrf_k < 1 or config.fusion_depth < 1 or config.rerank_depth < 1:
            raise ValueError("rrf_k, fusion_depth and rerank_depth must be positive")
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
        self.models = [encoder]
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


class HybridRetriever:
    """Reciprocal rank fusion of two document rankings: score(d) = sum 1 / (rrf_k + rank).

    Each retriever contributes its top `depth` documents. Ties go to the document BM25 ranked first.
    """

    def __init__(self, sparse: Retriever, dense: Retriever, *, rrf_k: int = 60, depth: int = 100):
        self.sparse, self.dense, self.rrf_k, self.depth = sparse, dense, rrf_k, depth
        self.models = getattr(dense, "models", [])

    def search_many(self, queries: Sequence[str], k: int) -> list[list[str]]:
        depth = max(self.depth, k)
        results = []
        for sparse, dense in zip(self.sparse.search_many(queries, depth), self.dense.search_many(queries, depth), strict=True):
            scores: dict[str, float] = defaultdict(float)
            order: dict[str, int] = {}
            for ranking in (sparse, dense):
                for rank, doc_id in enumerate(ranking, 1):
                    scores[doc_id] += 1 / (self.rrf_k + rank)
                    order.setdefault(doc_id, len(order))
            results.append(sorted(scores, key=lambda d: (-scores[d], order[d]))[:k])
        return results


class PairScorer(Protocol):
    def predict(self, pairs: Sequence[tuple[str, str]], **kwargs) -> object: ...


class RerankRetriever:
    """Re-score the first stage's top `depth` documents with a cross-encoder over (query, full document)."""

    def __init__(self, first_stage: Retriever, scorer: PairScorer, documents: Sequence[Document], *, depth: int = 100):
        self.first_stage, self.scorer, self.depth = first_stage, scorer, depth
        self.text = {doc.id: doc.text for doc in documents}
        self.models = [*getattr(first_stage, "models", []), scorer]

    def search_many(self, queries: Sequence[str], k: int) -> list[list[str]]:
        results = []
        for query, candidates in zip(queries, self.first_stage.search_many(queries, max(self.depth, k)), strict=True):
            if not candidates:
                results.append([])
                continue
            scores = [float(s) for s in self.scorer.predict([(query, self.text[d]) for d in candidates])]
            ranked = sorted(range(len(candidates)), key=lambda i: (-scores[i], i))
            results.append([candidates[i] for i in ranked[:k]])
        return results


def load_sentence_transformer(model: str, *, allow_download: bool, device: str = "cpu"):
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("dense retrieval needs: pip install 'rag-retrieval-gate[dense]'") from exc
    return SentenceTransformer(model, local_files_only=not allow_download, device=device)


def load_cross_encoder(model: str, *, allow_download: bool, device: str = "cpu"):
    try:
        from sentence_transformers import CrossEncoder
    except ImportError as exc:  # pragma: no cover - depends on optional extra
        raise RuntimeError("re-ranking needs: pip install 'rag-retrieval-gate[dense]'") from exc
    return CrossEncoder(model, local_files_only=not allow_download, device=device)


def build_retriever(
    config: RetrieverConfig,
    documents: Sequence[Document],
    *,
    encoder: Encoder | None = None,
    scorer: PairScorer | None = None,
    allow_download: bool = False,
    device: str = "cpu",
) -> Retriever:
    chunks = chunk(documents, config.chunk_words, config.chunk_overlap)
    retriever: Retriever
    if config.retriever == "bm25":
        retriever = BM25Retriever(chunks, config.bm25_k1, config.bm25_b)
    else:
        encoder = encoder or load_sentence_transformer(config.model, allow_download=allow_download, device=device)
        retriever = DenseRetriever(chunks, encoder)
        if config.retriever == "hybrid":
            sparse = BM25Retriever(chunks, config.bm25_k1, config.bm25_b)
            retriever = HybridRetriever(sparse, retriever, rrf_k=config.rrf_k, depth=config.fusion_depth)
    if config.rerank_model:
        scorer = scorer or load_cross_encoder(config.rerank_model, allow_download=allow_download, device=device)
        retriever = RerankRetriever(retriever, scorer, documents, depth=config.rerank_depth)
    return retriever
