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

if hasattr(sys.stdout, "reconfigure"):  # Windows console (cp1252) → UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

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

    # FAISS exact inner-product index (embeddings are L2-normalized).
    # At this scale (tens of thousands × 384 dims) exact IP search takes only
    # a few ms — no ANN needed. Note: IndexHNSWFlat ignores a post-hoc
    # metric_type assignment; if you switch to HNSW later, pass the metric in
    # the constructor: IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT).
    import faiss
    index = faiss.IndexFlatIP(vecs.shape[1])
    index.add(vecs)
    np.save(INDEX_DIR / "vectors.npy", vecs)   # cache: rebuild without re-encoding
    faiss.write_index(index, str(INDEX_DIR / "faiss.index"))

    # BM25 over tokens stored by preprocess.py (falls back to re-tokenizing
    # older chunk files that don't carry tokens)
    from rank_bm25 import BM25Okapi
    if "tokens" in chunks[0] and chunks[0]["tokens"]:
        print("Building BM25 index from stored tokens …")
        tokenized = [c["tokens"] for c in chunks]
    else:
        print("Chunks carry no tokens — re-tokenizing (slow path) …")
        tokenized = [tokenize(t) for t in texts]
    bm25 = BM25Okapi(tokenized)
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
