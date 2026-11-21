"""Build the FAISS (HNSW) index + BM25 index + metadata file.

Reads  data/processed/chunks.jsonl  produced by preprocess.py and writes to
data/index/:  faiss.index, bm25.pkl, meta.jsonl

Usage:  python build_faiss_index.py   (from backend/)
"""
import json
import pickle
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings
from app.embeddings import encode
from app.preprocess import tokenize

INDEX_DIR = settings.data_index_dir
CHUNKS = settings.data_processed_dir / "chunks.jsonl"


def main():
    if not CHUNKS.exists():
        sys.exit(f"No {CHUNKS} — run preprocess.py first.")
    chunks = [json.loads(l) for l in open(CHUNKS, encoding="utf-8")]
    texts = [c["text"] for c in chunks]
    print(f"Encoding {len(texts)} chunks with all-MiniLM-L6-v2 …")
    vecs = np.array(encode(texts), dtype="float32")

    # FAISS HNSW index, inner product (embeddings are L2-normalized)
    import faiss
    index = faiss.IndexHNSWFlat(vecs.shape[1], 32)
    index.metric_type = faiss.METRIC_INNER_PRODUCT
    index.add(vecs)
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    # BM25 over preprocessed tokens
    from rank_bm25 import BM25Okapi
    print("Building BM25 index …")
    bm25 = BM25Okapi([tokenize(t) for t in texts])
    with open(INDEX_DIR / "bm25.pkl", "wb") as f:
        pickle.dump(bm25, f)

    # Metadata sidecar
    with open(INDEX_DIR / "meta.jsonl", "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps({k: c.get(k) for k in
                                ("title", "year", "region", "source_file")},
                               ensure_ascii=False) + "\n")

    print(f"Done → {INDEX_DIR} (faiss.index, bm25.pkl, meta.jsonl)")


if __name__ == "__main__":
    main()
