"""Tests for LLM provider routing (Ollama local vs HF remote).

Ollama is the free/local default: quantized GGUF models on the local GPU,
no API key, no spend. HF router stays available for benchmark reproduction.
"""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.config import MODEL_REGISTRY, settings
from app import llm


# ------------------------------------------------------------ model mapping

def test_ollama_registry_maps_benchmark_models():
    assert llm.ollama_model_name("llama-3.2-3b") == "llama3.2:3b"
    assert llm.ollama_model_name("mistral-7b") == "mistral:7b-instruct"


def test_ollama_registry_falls_back_to_default_for_missing_models():
    # phi-2 and falcon-7b are not in Ollama's registry; fall back instead of failing
    assert llm.ollama_model_name("phi-2") == "llama3.2:3b"
    assert llm.ollama_model_name("unknown-key") == "llama3.2:3b"


def test_registry_keeps_hf_ids_for_benchmarks():
    assert MODEL_REGISTRY["mistral-7b"]["hf_id"].startswith("mistralai/")
    assert MODEL_REGISTRY["llama-3.2-3b"]["hf_id"].startswith("meta-llama/")


# ------------------------------------------------------------- routing (mock)

def _patch_post(monkeypatch, calls):
    class Resp:
        def raise_for_status(self): pass
        def json(self): return {"message": {"content": "ok"},
                               "choices": [{"message": {"content": "ok"}}]}
    def fake_post(url, **kw):
        calls.append({"url": url, **kw})
        return Resp()
    monkeypatch.setattr(llm.requests, "post", fake_post)


def test_generate_routes_to_ollama_by_default(monkeypatch):
    calls = []
    _patch_post(monkeypatch, calls)
    out = llm.generate("llama-3.2-3b", "sys", "question", "context")
    assert out == "ok"
    assert calls[0]["url"].startswith(settings.ollama_base_url)
    assert "/api/chat" in calls[0]["url"]
    assert calls[0]["json"]["model"] == "llama3.2:3b"
    assert calls[0]["json"]["stream"] is False


def test_generate_routes_to_hf_when_provider_is_hf(monkeypatch):
    calls = []
    _patch_post(monkeypatch, calls)
    llm.generate("mistral-7b", "sys", "q", "c", provider="hf")
    assert calls[0]["url"].startswith(settings.hf_inference_base_url)
    assert "Mistral-7B" in calls[0]["url"]


def test_generate_accepts_explicit_ollama_provider(monkeypatch):
    calls = []
    _patch_post(monkeypatch, calls)
    llm.generate("mistral-7b", "sys", "q", "c", provider="ollama")
    assert calls[0]["url"].startswith(settings.ollama_base_url)


# ------------------------------------------------- integration (real Ollama)

@pytest.mark.slow
def test_generate_with_real_ollama():
    try:
        out = llm.generate("llama-3.2-3b", llm.SYSTEM_PROMPT,
                           "How can I improve soil fertility without chemical fertilizers?",
                           "Context: compost and manure improve soil organic matter.")
    except Exception as e:
        pytest.skip(f"Ollama not reachable: {e}")
    assert out and len(out) > 20
