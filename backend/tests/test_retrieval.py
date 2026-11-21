"""Integration tests for the retrieval stack against the built index.

These require data/index/faiss.index + bm25.pkl + meta.jsonl (run
preprocess.py && build_faiss_index.py first) and the sentence-transformers
model. They gate the hybrid retrieval behavior end-to-end.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

faiss = pytest.importorskip("faiss")

SOIL_QUERY = "How can I improve soil fertility without chemical fertilizers?"


def test_faiss_index_uses_inner_product():
    """RED regression: the index was built with the default L2 metric while
    the retriever treats scores as inner products (higher = better). L2 is
    smaller = closer, so the retriever returned the WORST matches.
    """
    index = faiss.read_index(
        str(Path(__file__).resolve().parent.parent.parent / "data" / "index" / "faiss.index"))
    assert index.metric_type == faiss.METRIC_INNER_PRODUCT, (
        "faiss.index must be built with METRIC_INNER_PRODUCT (metric_type=1), "
        f"got {index.metric_type} (0=L2)")


@pytest.mark.slow
def test_search_returns_relevant_top_results():
    from app.retrieval import retriever
    results = retriever.search(SOIL_QUERY, top_k=5)
    assert len(results) == 5
    top_titles = " ".join((r.get("title") or "").lower() for r in results[:3])
    agri_hits = sum(1 for r in results
                    if any(k in ((r.get("title") or "") + (r.get("text", "") or "")).lower()
                           for k in ("soil", "fertil", "compost", "manure", "nutrient", "crop")))
    assert agri_hits >= 3, f"only {agri_hits}/5 relevant: {[r.get('title') for r in results]}"
