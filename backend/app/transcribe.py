"""Local speech-to-text with faster-whisper (CTranslate2).

Runs entirely on-device (CPU, or CUDA when available). The model loads lazily
on the first transcription request and is cached in-process.
"""
import io
import os
import tempfile
import wave
from functools import lru_cache

from fastapi import UploadFile

from .config import settings

MAX_UPLOAD_BYTES = 15 * 1024 * 1024   # ~60 s of compressed audio is far below this
SUPPORTED_AUDIO_TYPES = ("audio/", "video/webm", "video/mp4")


@lru_cache(maxsize=1)
def get_pipeline():
    from faster_whisper import WhisperModel
    # cpu/int8 works everywhere; switch to cuda/float16 via .env when the
    # CUDA toolkit DLLs (cublas, cudnn) are installed
    return WhisperModel(settings.whisper_model, device=settings.whisper_device,
                        compute_type=settings.whisper_compute)


def _wav_duration(data: bytes) -> float | None:
    """Duration of a WAV upload in seconds, or None if not a WAV file."""
    try:
        with wave.open(io.BytesIO(data), "rb") as w:
            return w.getnframes() / float(w.getframerate() or 1)
    except Exception:
        return None


async def validate_upload(file: UploadFile) -> bytes:
    """Read and sanity-check the upload. Raises ValueError with a message."""
    data = await file.read()
    if not data:
        raise ValueError("The uploaded audio is empty.")
    if len(data) > MAX_UPLOAD_BYTES:
        raise ValueError("Audio too large (limit ~60 seconds).")
    ctype = (file.content_type or "").lower()
    if not any(ctype.startswith(t) for t in SUPPORTED_AUDIO_TYPES):
        raise ValueError("Unsupported file type — please upload an audio recording.")

    dur = _wav_duration(data)
    if dur is not None and dur > settings.whisper_max_seconds:
        raise ValueError(
            f"Recording too long ({dur:.0f}s) — please keep it under "
            f"{settings.whisper_max_seconds:.0f} seconds.")
    return data


def transcribe(data: bytes) -> dict:
    """Transcribe audio bytes. Returns {text, language, duration}."""
    model = get_pipeline()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        tmp.write(data)
        tmp_path = tmp.name
    try:
        segments_iter, info = model.transcribe(tmp_path, vad_filter=True)
        segments = list(segments_iter)
        text = "".join(s.text for s in segments).strip()
        return {"text": text,
                "language": getattr(info, "language", None),
                "duration": round(float(getattr(info, "duration", 0.0) or 0.0), 2)}
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass
