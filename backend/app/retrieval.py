"""Hybrid retrieval: FAISS dense (HNSW) + BM25 sparse, weighted fusion.

Fusion score = faiss_weight * dense_score + bm25_weight * sparse_score
(defaults 0.7 / 0.3, grid-searched in the dissertation; Recall@5 = 85%).
"""
import json
import pickle
from pathlib import Path

import numpy as np

from .config import settings
from .embeddings import encode_query

INDEX_DIR = settings.data_index_dir
FAISS_INDEX_FILE = "faiss.index"
META_FILE = "meta.jsonl"
BM25_FILE = "bm25.pkl"


class Retriever:
    def __init__(self):
        self._index = None
        self._bm25 = None
        self._meta = []

    # ---------------------------------------------------------- loading
    @property
    def index(self):
        if self._index is None:
            import faiss
            self._index = faiss.read_index(
                str(INDEX_DIR / FAISS_INDEX_FILE), faiss.IO_FLAG_MMAP
            )
        return self._index

    @property
    def bm25(self):
        if self._bm25 is None:
            with open(INDEX_DIR / BM25_FILE, "rb") as f:
                self._bm25 = pickle.load(f)
        return self._bm25

    @property
    def meta(self):
        if not self._meta:
            with open(INDEX_DIR / META_FILE, "r", encoding="utf-8") as f:
                self._meta = [json.loads(line) for line in f]
        return self._meta

    # ---------------------------------------------------------- search
    def search(self, query: str, top_k: int | None = None,
               region: str | None = None, after_year: int | None = None):
        """Return top_k chunks with metadata, hybrid-scored and filtered."""
        top_k = top_k or settings.retrieval_top_k
        dense_k = sparse_k = max(top_k * 5, 25)

        # dense: FAISS inner product on normalized embeddings ≈ cosine
        q = np.array([encode_query(query)], dtype="float32")
        d_scores, d_ids = self.index.search(q, dense_k)
        d_scores, d_ids = d_scores[0], d_ids[0]

        # sparse: BM25 over preprocessed tokens
        from .preprocess import tokenize
        s_scores = self.bm25.get_scores(tokenize(query))
        s_ids = np.argsort(s_scores)[::-1][:sparse_k]
        s_scores = s_scores[s_ids]

        # normalize each score list to [0, 1] before fusion
        def norm(a):
            a = np.asarray(a, dtype="float64")
            rng = a.max() - a.min()
            return (a - a.min()) / rng if rng > 0 else np.zeros_like(a)

        fused: dict[int, float] = {}
        for i, s in zip(d_ids, norm(d_scores)):
            if i >= 0:
                fused[int(i)] = fused.get(int(i), 0.0) + settings.faiss_weight * float(s)
        for i, s in zip(s_ids, norm(s_scores)):
            fused[int(i)] = fused.get(int(i), 0.0) + settings.bm25_weight * float(s)

        results = []
        for idx, score in sorted(fused.items(), key=lambda kv: -kv[1]):
            m = self.meta[idx]
            if region and m.get("region") and region.lower() not in m["region"].lower():
                continue
            year = m.get("year")
            if after_year and year and int(year) < after_year:
                continue
            results.append({"score": round(score, 4), **m})
            if len(results) >= top_k:
                break
        return results

    def temporal_search(self, query: str, top_k: int | None = None):
        """Guardrail variant: prefer peer-reviewed work from the last N years."""
        latest = self.meta and max(
            (int(m.get("year") or 0) for m in self.meta), default=0
        )
        after = (latest - settings.temporal_filter_years) if latest else None
        return self.search(query, top_k=top_k, after_year=after)


retriever = Retriever()
