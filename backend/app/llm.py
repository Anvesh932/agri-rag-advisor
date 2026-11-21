"""LLM generation via the Hugging Face inference router.

Supports the four evaluated models (see config.MODEL_REGISTRY) with an
OpenAI-compatible chat-completions call. When a local GPU with bitsandbytes is
available, models can be loaded quantized to 8-bit for edge-like deployment.
"""
import os
from functools import lru_cache

import requests

from .config import MODEL_REGISTRY, OLLAMA_DEFAULT, settings

LOCAL_MODEL_CACHE: dict[str, object] = {}


def resolve_model(key_or_hf_id: str) -> str:
    reg = MODEL_REGISTRY.get(key_or_hf_id)
    return reg["hf_id"] if reg else key_or_hf_id


def ollama_model_name(key_or_hf_id: str) -> str:
    """Map a benchmark-model key (or HF id) to a local Ollama model name.

    Models without an Ollama equivalent (falcon-7b, phi-2) fall back to the
    default lightweight model instead of failing the request.
    """
    reg = MODEL_REGISTRY.get(key_or_hf_id)
    if reg:
        return reg.get("ollama", OLLAMA_DEFAULT)
    if ":" in key_or_hf_id:          # already a direct Ollama model name
        return key_or_hf_id
    return OLLAMA_DEFAULT


def generate_ollama(ollama_model: str, system_prompt: str, question: str,
                    context: str, max_new_tokens: int = 512) -> str:
    """Local generation through Ollama's /api/chat (free, no API key)."""
    resp = requests.post(
        settings.ollama_base_url.rstrip("/") + "/api/chat",
        json={
            "model": ollama_model,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user",
                 "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            "stream": False,
            "options": {
                "temperature": 0.7,
                "top_p": 0.9,
                "repeat_penalty": 1.2,
                "num_predict": max_new_tokens,
            },
        },
        timeout=300,
    )
    resp.raise_for_status()
    return resp.json()["message"]["content"].strip()


def generate(model_key: str, system_prompt: str, question: str, context: str,
             provider: str | None = None, max_new_tokens: int = 512) -> str:
    """Provider router. Default comes from settings.llm_provider.

    "ollama" → local quantized GGUF models (free);
    "hf"     → HF inference router with the exact benchmark models.
    """
    provider = provider or settings.llm_provider
    if provider == "ollama":
        return generate_ollama(ollama_model_name(model_key), system_prompt,
                               question, context, max_new_tokens)
    return generate_remote(resolve_model(model_key), system_prompt,
                           question, context, max_new_tokens)


def generate_remote(hf_model: str, system_prompt: str, question: str,
                    context: str, max_new_tokens: int = 512) -> str:
    """Streaming-friendly completion through the HF inference router."""
    url = settings.hf_inference_base_url.rstrip("/") + f"/{hf_model}/v1"
    resp = requests.post(
        url,
        headers={
            "Authorization": f"Bearer {settings.hf_token}",
            "Model-Specific-Params": '{"temperature":0.7,"top_p":0.9,"repetition_penalty":1.2}',
        },
        json={
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"},
            ],
            "max_tokens": max_new_tokens,
            "stream": False,
        },
        timeout=120,
    )
    resp.raise_for_status()
    return resp.json()["choices"][0]["message"]["content"].strip()


def generate_local(hf_model: str, system_prompt: str, question: str,
                   context: str, load_in_8bit: bool = True) -> str:
    """Local generation with optional 8-bit quantization (LLM.int8()).

    Falls back to remote inference if local loading is not possible.
    """
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        if hf_model not in LOCAL_MODEL_CACHE:
            kwargs = {}
            if load_in_8bit and torch.cuda.is_available():
                kwargs = {"load_in_8bit": True, "device_map": "auto"}
            tok = AutoTokenizer.from_pretrained(hf_model)
            model = AutoModelForCausalLM.from_pretrained(hf_model, **kwargs)
            LOCAL_MODEL_CACHE[hf_model] = (tok, model)
        tok, model = LOCAL_MODEL_CACHE[hf_model]

        prompt = (
            f"{system_prompt}\n\nContext:\n{context}\n\nQuestion: {question}\nAnswer:"
        )
        inputs = tok(prompt, return_tensors="pt").to(model.device)
        out = model.generate(
            **inputs, max_new_tokens=512, temperature=0.7,
            top_p=0.9, repetition_penalty=1.2, do_sample=True,
        )
        return tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True).strip()
    except Exception:
        return generate_remote(hf_model, system_prompt, question, context)


SYSTEM_PROMPT = (
    "You are an agricultural advisor for smallholder farmers in South Asia. "
    "Answer ONLY from the provided context. Give actionable, region-aware advice "
    "on soil fertility, water optimization, and post-harvest loss reduction. "
    "If the context is insufficient, say what extra information is needed. "
    "Recommend sustainable practices aligned with SDG 12."
)
