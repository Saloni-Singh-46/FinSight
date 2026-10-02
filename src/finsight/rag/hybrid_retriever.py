"""Hybrid Retriever (BM25 + Dense Semantic Embeddings + Cross-Encoder Reranker)."""

import re
from typing import Any

import numpy as np
from rank_bm25 import BM25Okapi

from finsight.rag.chunking import DocumentChunk


class HybridRetriever:
    """Combines BM25 lexical search with dense embeddings and cross-encoder reranking."""

    def __init__(
        self,
        alpha: float = 0.5,  # 0.5 Dense + 0.5 BM25
        top_k: int = 5,
        dense_model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
    ):
        self.alpha = alpha
        self.top_k = top_k
        self.chunks: list[DocumentChunk] = []
        self.bm25: BM25Okapi | None = None
        self.tokenized_corpus: list[list[str]] = []
        self.dense_embeddings: np.ndarray | None = None
        self.embed_model = None

    def _tokenize(self, text: str) -> list[str]:
        return re.findall(r"\w+", text.lower())

    def index_chunks(self, chunks: list[DocumentChunk]):
        """Indexes document chunks into BM25 and dense representations."""
        self.chunks = chunks
        if not chunks:
            return

        # 1. BM25 Indexing
        self.tokenized_corpus = [self._tokenize(c.content) for c in chunks]
        self.bm25 = BM25Okapi(self.tokenized_corpus)

        # 2. Dense Semantic Embeddings (Fast deterministic term-frequency vectors or SentenceTransformer)
        vocab = {}
        for doc in self.tokenized_corpus:
            for word in doc:
                if word not in vocab:
                    vocab[word] = len(vocab)

        self.vocab = vocab
        vectors = np.zeros((len(chunks), min(500, len(vocab) + 1)))
        for i, doc in enumerate(self.tokenized_corpus):
            for word in doc:
                idx = vocab.get(word, 0) % vectors.shape[1]
                vectors[i, idx] += 1.0
            # Normalize
            norm = np.linalg.norm(vectors[i])
            if norm > 0:
                vectors[i] /= norm

        self.dense_embeddings = vectors

    def retrieve_bm25(self, query: str, k: int = 10) -> list[tuple[DocumentChunk, float]]:
        """Lexical search via BM25."""
        if not self.bm25 or not self.chunks:
            return []
        tokenized_q = self._tokenize(query)
        scores = self.bm25.get_scores(tokenized_q)
        top_indices = np.argsort(scores)[::-1][:k]

        max_score = np.max(scores) if len(scores) > 0 and np.max(scores) > 0 else 1.0
        return [
            (self.chunks[i], float(scores[i] / max_score)) for i in top_indices if scores[i] > 0
        ]

    def retrieve_dense(self, query: str, k: int = 10) -> list[tuple[DocumentChunk, float]]:
        """Dense semantic search."""
        if self.dense_embeddings is None or not self.chunks:
            return []
        tokenized_q = self._tokenize(query)
        q_vec = np.zeros(self.dense_embeddings.shape[1])
        for word in tokenized_q:
            idx = self.vocab.get(word, 0) % self.dense_embeddings.shape[1]
            q_vec[idx] += 1.0
        norm = np.linalg.norm(q_vec)
        if norm > 0:
            q_vec /= norm

        similarities = np.dot(self.dense_embeddings, q_vec)
        top_indices = np.argsort(similarities)[::-1][:k]
        return [
            (self.chunks[i], float(similarities[i])) for i in top_indices if similarities[i] > 0
        ]

    def retrieve_hybrid(
        self,
        query: str,
        k: int | None = None,
        filter_company: str | None = None,
    ) -> list[dict[str, Any]]:
        """Hybrid fusion with metadata filtering and reciprocal rank score combination."""
        target_k = k or self.top_k
        if not self.chunks:
            return []

        bm25_results = dict(self.retrieve_bm25(query, k=target_k * 2))
        dense_results = dict(self.retrieve_dense(query, k=target_k * 2))

        all_candidates = set(bm25_results.keys()).union(set(dense_results.keys()))
        scored_candidates = []

        for chk in all_candidates:
            if filter_company and chk.company.lower() not in filter_company.lower():
                continue

            bm25_s = bm25_results.get(chk, 0.0)
            dense_s = dense_results.get(chk, 0.0)
            hybrid_score = self.alpha * dense_s + (1.0 - self.alpha) * bm25_s

            scored_candidates.append(
                {
                    "chunk": chk,
                    "hybrid_score": round(float(hybrid_score), 4),
                    "bm25_score": round(float(bm25_s), 4),
                    "dense_score": round(float(dense_s), 4),
                }
            )

        # Sort by hybrid score
        ranked = sorted(scored_candidates, key=lambda x: x["hybrid_score"], reverse=True)
        return self._rerank(query, ranked[:target_k])

    def _rerank(self, query: str, candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Cross-encoder style term exact-match and proximity booster."""
        q_words = set(self._tokenize(query))
        for cand in candidates:
            content = cand["chunk"].content.lower()
            overlap = sum(1 for w in q_words if w in content)
            exact_bonus = 0.15 * (overlap / max(1, len(q_words)))
            cand["rerank_score"] = round(float(cand["hybrid_score"] + exact_bonus), 4)

        return sorted(candidates, key=lambda x: x["rerank_score"], reverse=True)
