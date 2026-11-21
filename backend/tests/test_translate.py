"""Tests for the multilingual translation layer (mbart-large-50)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

pytest.importorskip("transformers")

from app.translate import translate


@pytest.mark.slow
def test_translate_hindi_to_english():
    """RED regression: transformers v5 removed the 'translation' pipeline.

    Requirements now pin transformers<5; this test gates the round-trip.
    First run downloads ~2.4 GB (mbart-large-50).
    """
    out = translate("मानसून के बाद गेहूं की बुवाई का सही समय क्या है?", "hi", "en")
    assert out and len(out) > 10
    assert any(w in out.lower() for w in ("wheat", "sow", "sowing", "monsoon"))


@pytest.mark.slow
def test_translate_bengali_to_english_rice_question():
    """RED regression: mbart mangled Bengali→English completely ("post-harvest
    losses of paddy" → "reduce the government's revenue after the headquarters
    collapse"), making Bengali retrieval irrelevant. NLLB-200 must handle it."""
    out = translate("ধানের ফসল তোলার পরে ক্ষতি কমানোর উপায় কী?", "bn", "en")
    assert out and len(out) > 10
    low = out.lower()
    assert any(w in low for w in ("rice", "paddy", "harvest", "loss", "reduce", "crop")), out


@pytest.mark.slow
def test_translate_identity_short_circuits():
    from app.translate import translate
    assert translate("soil fertility", "en", "en") == "soil fertility"
