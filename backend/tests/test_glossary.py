"""Tests for the domain glossary that removes English code-mixing from
back-translated answers (quality-backlog item #2)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.glossary import apply_glossary, glossary_languages


def test_glossary_applies_hindi_terms():
    out = apply_glossary(
        "Use crop rotation and drip irrigation to increase yield. Add compost.", "hi")
    assert "फसल चक्र" in out            # crop rotation
    assert "टपक सिंचाई" in out           # drip irrigation
    assert "कम्पोस्ट" in out             # compost
    assert "Crop Rotation" not in out and "drip irrigation" not in out


def test_glossary_applies_bengali_terms():
    out = apply_glossary("Apply organic manure and green manure before sowing.", "bn")
    assert "জৈব সার" in out             # organic manure
    assert "সবুজ সার" in out            # green manure


def test_glossary_applies_tamil_terms():
    out = apply_glossary("In rainfed areas, practice mulching to conserve water.", "ta")
    assert "மழையை" in out or "மழை" in out   # rainfed
    assert "மல்சிங்" in out                   # mulching
    assert "rainfed" not in out.lower() or "மழை" in out


def test_glossary_case_insensitive_uppercase_terms():
    out = apply_glossary("RAINFED AREAS need drainage.", "hi")
    assert "rainfed" not in out.lower()
    assert "जल निकासी" in out           # drainage


def test_glossary_handles_inflected_forms():
    out = apply_glossary("Composting and mulching improve soil.", "hi")
    assert "कम्पोस्ट" in out
    assert "मल्चिंग" in out or "आच्छादन" in out
    assert "composting" not in out.lower() and "mulching" not in out.lower()


def test_glossary_noop_for_english():
    original = "Use crop rotation and compost."
    assert apply_glossary(original, "en") == original


def test_glossary_longest_phrases_win():
    # 'drip irrigation' must be matched before the bare word 'irrigation'
    out = apply_glossary("Use drip irrigation.", "hi")
    assert "टपक सिंचाई" in out
    assert "सिंचाई टपक" not in out      # no double-mangling
    assert out.count("सिंचाई") == 1


def test_glossary_supports_the_farmer_languages():
    assert set(glossary_languages()) >= {"hi", "bn", "ta"}
