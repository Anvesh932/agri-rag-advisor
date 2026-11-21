"""Multilingual support: NLLB-200 on-the-fly translation.

Farmer languages: Hindi, Bengali, Tamil (+ English). Requests come in the
farmer's language, are translated to English for retrieval/generation, and the
answer is translated back.

Implementation notes:
- NLLB-200-distilled-600M is used instead of mbart-large-50: mbart's
  Bengali→English output was incoherent ("post-harvest losses of paddy" →
  "reduce the government's revenue after the headquarters collapse"), which
  poisoned retrieval for Bengali questions. NLLB handles bn/hi/ta far better
  at the same free/local cost profile (~2.4 GB one-time download, CPU).
- Slow tokenizer + manual generate() for cross-version stability.
"""
import torch
from functools import lru_cache
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from .config import SUPPORTED_LANGS

NLLB_MODEL = "facebook/nllb-200-distilled-600M"


@lru_cache(maxsize=1)
def get_translator():
    tok = AutoTokenizer.from_pretrained(NLLB_MODEL, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
    model.eval()
    return tok, model


def to_mbart(lang: str) -> str:
    return SUPPORTED_LANGS.get(lang, SUPPORTED_LANGS["en"])


def translate(text: str, src_lang: str, tgt_lang: str) -> str:
    """Translate text between supported farmer languages."""
    if src_lang == tgt_lang or not text.strip():
        return text
    tok, model = get_translator()
    tok.src_lang = to_mbart(src_lang)
    encoded = tok(text, return_tensors="pt", truncation=True, max_length=1024)
    with torch.no_grad():
        generated = model.generate(
            **encoded,
            forced_bos_token_id=tok.convert_tokens_to_ids(to_mbart(tgt_lang)),
            max_length=1024,
        )
    return tok.batch_decode(generated, skip_special_tokens=True)[0].strip()
