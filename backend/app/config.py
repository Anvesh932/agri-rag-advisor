"""Central configuration, loaded from environment / .env."""
from pathlib import Path

from pydantic import field_validator
from pydantic_settings import BaseSettings

BASE_DIR = Path(__file__).resolve().parent.parent  # backend/


class Settings(BaseSettings):
    # LLM provider: "ollama" (free/local default) or "hf" (HF inference router)
    llm_provider: str = "ollama"
    ollama_base_url: str = "http://localhost:11434"
    hf_token: str = ""
    llm_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    hf_inference_base_url: str = "https://router.huggingface.co/hf-inference/models/"

    # Retrieval
    faiss_weight: float = 0.7          # hybrid fusion: 0.7 FAISS + 0.3 BM25
    bm25_weight: float = 0.3
    retrieval_top_k: int = 5
    temporal_filter_years: int = 5

    # Paths
    data_raw_dir: Path = BASE_DIR.parent / "data" / "raw"
    data_processed_dir: Path = BASE_DIR.parent / "data" / "processed"
    data_index_dir: Path = BASE_DIR.parent / "data" / "index"

    # Translation
    default_lang: str = "en"
    google_translate_api_key: str = ""

    # Voice input (local Whisper)
    whisper_model: str = "small"          # tiny | base | small | medium
    whisper_device: str = "cpu"           # "cpu" (safe default) or "cuda" (needs CUDA/cuDNN DLLs)
    whisper_compute: str = "int8"         # int8 on CPU; "float16" for cuda
    whisper_max_seconds: float = 60.0

    # Cache
    redis_url: str = "redis://localhost:6379/0"

    # App
    cors_origins: str = "http://localhost:5173"

    @field_validator("data_raw_dir", "data_processed_dir", "data_index_dir",
                     mode="after")
    @classmethod
    def _resolve_data_dir(cls, v: Path) -> Path:
        """Relative paths are resolved against the project root (parent of
        backend/), not the CWD — uvicorn is typically started from backend/."""
        return v if v.is_absolute() else (BASE_DIR.parent / v).resolve()

    class Config:
        env_file = (BASE_DIR.parent / ".env").as_posix()
        env_file_encoding = "utf-8"


settings = Settings()

# NLLB-200 language codes (en, hi, bn, ta farmer languages)
SUPPORTED_LANGS = {"en": "eng_Latn", "hi": "hin_Deva", "bn": "ben_Beng", "ta": "tam_Taml"}

# Model registry from the dissertation benchmark table
MODEL_REGISTRY = {
    "falcon-7b": {
        "hf_id": "tiiuae/falcon-7b-instruct",
        "bleu4": 0.75, "rouge_l": 0.71, "em": 0.81, "wh_per_query": 3.2,
        "note": "efficient; concise answers",
    },
    "llama-3.2-3b": {
        "hf_id": "meta-llama/Llama-3.2-3B-Instruct",
        "ollama": "llama3.2:3b",
        "bleu4": 0.72, "rouge_l": 0.68, "em": 0.78, "wh_per_query": 2.5,
        "note": "detailed, actionable advice (68% farmer preference)",
    },
    "mistral-7b": {
        "hf_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "ollama": "mistral:7b-instruct",
        "bleu4": 0.78, "rouge_l": 0.74, "em": 0.85, "wh_per_query": 3.8,
        "note": "best accuracy on the benchmark",
    },
    "phi-2": {
        "hf_id": "microsoft/phi-2",
        "bleu4": 0.70, "rouge_l": 0.66, "em": 0.76, "wh_per_query": 2.1,
        "note": "lightweight edge option (not in Ollama registry; falls back)",
    },
}

# Default Ollama model when a registry entry has no local equivalent
OLLAMA_DEFAULT = "llama3.2:3b"
