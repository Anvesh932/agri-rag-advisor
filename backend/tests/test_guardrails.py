"""Tests for the ethical guardrails."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.guardrails import check_output


def test_check_output_clean_answer_has_no_flags():
    res = check_output("Improve soil fertility by adding compost and manure.")
    assert res == {"flags": [], "ok": True}


def test_check_output_flags_harmful_chemical_increase():
    res = check_output("You should increase pesticide application every week.")
    assert res["ok"] is False
    assert any("chemical use" in f for f in res["flags"])


def test_check_output_flags_gender_stereotyping():
    res = check_output("Women should only handle seed selection while men should plough.")
    assert res["ok"] is False


def test_check_output_handles_unicode_answers():
    # regression: real model answers contain non-ASCII (Hindi/Devanagari etc.)
    res = check_output("मिट्टी की सेहत के लिए compost डालें।")
    assert res["ok"] is True
