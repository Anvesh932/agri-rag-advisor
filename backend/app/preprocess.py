"""Text preprocessing pipeline.

Tokenization (SpaCy), special-character removal, agricultural-context stop-word
removal, lemmatization (NLTK WordNet), and chunking into 200-token segments
with 50-token overlap — as specified in the dissertation (§5.2).
"""
import re
from functools import lru_cache

CHUNK_SIZE = 200   # tokens per chunk
CHUNK_OVERLAP = 50

# Domain terms that must NOT be treated as stop words
AGRI_KEEP = {
    "soil", "crop", "crops", "seed", "seeds", "water", "fertilizer", "fertilizer",
    "irrigation", "harvest", "monsoon", "compost", "manure", "pest", "disease",
}

_CUSTOM_STOP = {
    "also", "however", "therefore", "thus", "moreover", "whereas", "among",
    "within", "without", "toward", "towards", "based", "using", "used", "study",
    "results", "result", "show", "shown", "conclusion", "concluded",
}

_special_re = re.compile(r"[^a-z0-9\s\u0900-\u097F\u0980-\u09FF\u0B80-\u0BFF]")


@lru_cache(maxsize=1)
def _nlp():
    import spacy
    try:
        return spacy.load("en_core_web_sm")
    except OSError:  # model not downloaded
        return spacy.blank("en")


def clean_text(text: str) -> str:
    """Lowercase, strip special characters, collapse whitespace."""
    text = text.lower()
    text = _special_re.sub(" ", text)
    return re.sub(r"\s+", " ", text).strip()


def tokenize(text: str) -> list[str]:
    """SpaCy tokenization + stop-word removal (with agricultural keep-list)."""
    doc = _nlp()(clean_text(text))
    toks = []
    for t in doc:
        if t.is_space:
            continue
        if t.text in AGRI_KEEP:
            toks.append(t.text)
        elif t.is_stop or t.text in _CUSTOM_STOP:
            continue
        else:
            toks.append(t.lemma_ if t.lemma_ else t.text)
    return toks


def chunk_text(text: str, tokens: list[str] | None = None) -> list[str]:
    """Split into CHUNK_SIZE-token chunks with CHUNK_OVERLAP-token overlap."""
    if tokens is None:
        tokens = tokenize(text)
    if len(tokens) <= CHUNK_SIZE:
        return [" ".join(tokens)] if tokens else []
    step = CHUNK_SIZE - CHUNK_OVERLAP
    return [" ".join(tokens[i:i + CHUNK_SIZE]) for i in range(0, len(tokens), step)]


def preprocess_document(text: str) -> list[str]:
    """Full pipeline: clean → tokenize → chunk. Returns chunk strings."""
    return chunk_text(text, tokens=tokenize(text))
