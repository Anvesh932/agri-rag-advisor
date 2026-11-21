"""Multilingual support: mbart-large-50 on-the-fly translation.

Farmer languages: Hindi, Bengali, Tamil (+ English). Requests come in the
farmer's language, are translated to English for retrieval/generation, and the
answer is translated back.
"""
from functools import lru_cache

from .config import SUPPORTED_LANGS

MBART_MODEL = "facebook/mbart-large-50-many-to-many-mmt"


@lru_cache(maxsize=1)
def get_translator():
    from transformers import pipeline
    return pipeline("translation", model=MBART_MODEL)


def to_mbart(lang: str) -> str:
    return SUPPORTED_LANGS.get(lang, SUPPORTED_LANGS["en"])


def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate text between supported farmer languages."""
    if src_lang == tgt_lang or not text.strip():
        return text
    tr = get_translator()
    tr.tokenizer.src_lang = to_mbart(src_lang)
    out = tr(text, tgt_lang=to_mbart(tgt_lang), max_length=1024)
    return out[0]["translation_text"]
