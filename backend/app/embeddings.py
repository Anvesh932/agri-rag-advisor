"""Sentence-embedding generation (all-MiniLM-L6-v2, 384-dim)."""
from functools import lru_cache

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
EMBED_DIM = 384


@lru_cache(maxsize=1)
def get_encoder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(MODEL_NAME)


def encode(texts: list[str]):
    """Return list[list[float]] L2-normalized embeddings (required by FAISS IP)."""
    return get_encoder().encode(
        texts, normalize_embeddings=True, show_progress_bar=False
    ).tolist()


def encode_query(text: str):
    return encode([text])[0]
