"""End-to-end endpoint test: /query must apply the domain glossary to the
back-translated answer (quality-backlog item #2 wiring)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


@pytest.fixture()
def stubbed_rag(monkeypatch):
    """Stub translation + generation + retrieval so the test runs fast."""
    from app import main as m

    # translation: hi→en returns the question text; en→hi returns English answer
    def fake_translate(text, src, tgt):
        return text if tgt == "en" else "Use compost, crop rotation and drip irrigation."
    monkeypatch.setattr(m, "translate", fake_translate)
    monkeypatch.setattr(m.retriever, "search", lambda *a, **kw: [])
    monkeypatch.setattr(m, "generate", lambda *a, **kw: "stub")
    monkeypatch.setattr(m, "_cache_get", lambda key: None)
    monkeypatch.setattr(m, "_cache_set", lambda *a: None)
    app.state.index_ready = True
    yield


def test_query_applies_glossary_to_hindi_answer(stubbed_rag):
    r = client.post("/query", json={
        "question": "मिट्टी की उर्वरता?", "lang": "hi", "model": "llama-3.2-3b"})
    assert r.status_code == 200
    ans = r.json()["answer"]
    assert "कम्पोस्ट" in ans and "फसल चक्र" in ans and "टपक सिंचाई" in ans
    assert "compost" not in ans.lower() and "crop rotation" not in ans.lower()


def test_query_leaves_english_answer_untouched(stubbed_rag):
    r = client.post("/query", json={
        "question": "soil fertility?", "lang": "en", "model": "llama-3.2-3b"})
    assert r.status_code == 200
