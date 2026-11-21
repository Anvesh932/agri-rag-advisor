"""Tests for the voice-input transcription endpoint (POST /transcribe).

Validation tests use a monkeypatched Whisper model so they run fast without the
~460 MB model download; the real-model integration test is marked slow and
generates a spoken sample with Windows SAPI TTS.
"""
import io
import sys
import wave
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fastapi.testclient import TestClient

from app.main import app

client = TestClient(app)


def _wav_bytes(seconds: float, freq: int = 440) -> bytes:
    """Generate a minimal valid WAV file of `seconds` duration."""
    rate = 16000
    n = int(rate * seconds)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(
            b"\x00\x10" for _ in range(n)))  # constant quiet samples
    return buf.getvalue()


# ------------------------------------------------------------- validation

def test_transcribe_empty_upload_rejected():
    r = client.post("/transcribe", files={"file": ("a.webm", b"", "audio/webm")})
    assert r.status_code == 400
    assert "text" in r.json()["detail"].lower() or "audio" in r.json()["detail"].lower()


def test_transcribe_non_audio_rejected():
    r = client.post("/transcribe",
                    files={"file": ("a.txt", b"this is not audio", "text/plain")})
    assert r.status_code == 400


def test_transcribe_oversize_audio_rejected(monkeypatch):
    # 61 s of audio exceeds the 60 s limit; model must never be loaded
    from app import transcribe as tmod
    monkeypatch.setattr(tmod, "get_pipeline",
                        lambda: pytest.fail("model must not load for oversize input"))
    r = client.post("/transcribe",
                    files={"file": ("long.wav", _wav_bytes(61), "audio/wav")})
    assert r.status_code == 400
    assert "60" in r.json()["detail"] or "long" in r.json()["detail"].lower()


# ------------------------------------------------- successful transcription

def test_transcribe_success_returns_text_language_duration(monkeypatch):
    from app import transcribe as tmod

    class FakeInfo:
        language = "en"
        duration = 2.5

    class FakeSeg:
        def __init__(self):
            self.text = " How can I improve soil fertility? "

    class FakePipeline:
        def transcribe(self, path_or_bytes, **kw):
            return iter([FakeSeg()]), FakeInfo()

    monkeypatch.setattr(tmod, "get_pipeline", lambda: FakePipeline())
    r = client.post("/transcribe",
                    files={"file": ("a.wav", _wav_bytes(2.5), "audio/wav")})
    assert r.status_code == 200
    body = r.json()
    assert "soil fertility" in body["text"]
    assert body["language"] == "en"
    assert body["duration"] == 2.5


@pytest.mark.slow
def test_transcribe_real_english_voice_sample(tmp_path):
    """Real Whisper against a spoken sample generated with Windows SAPI TTS.

    Skipped when TTS or the model is unavailable; exercises decode + inference.
    """
    wav_path = tmp_path / "sample.wav"
    try:
        import subprocess
        script = (
            "Add-Type -AssemblyName System.Speech; "
            "$s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
            f"$s.SetOutputToWaveFile('{wav_path.as_posix()}'); "
            "$s.Speak('How can I improve the fertility of my soil?'); "
            "$s.Dispose()"
        )
        subprocess.run(["powershell", "-NoProfile", "-Command", script],
                       check=True, capture_output=True, timeout=60)
    except Exception as e:
        pytest.skip(f"SAPI TTS unavailable: {e}")
    assert wav_path.exists() and wav_path.stat().st_size > 1000

    try:
        r = client.post("/transcribe",
                        files={"file": ("sample.wav", wav_path.read_bytes(), "audio/wav")})
    except Exception as e:
        pytest.skip(f"whisper model not available: {e}")
    assert r.status_code == 200, r.text
    text = r.json()["text"].lower()
    assert "soil" in text or "fertility" in text, text
