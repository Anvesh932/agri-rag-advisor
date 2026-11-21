"""Ollama auto-start: make sure the local LLM server is up before serving.

On Windows Ollama is typically installed at
%LOCALAPPDATA%\\Programs\\Ollama\\ollama.exe and may not be running when the
API starts; ensure_ollama() probes the server and launches it if needed.
"""
import os
import shutil
import subprocess
import time
from pathlib import Path

import requests

# Thin wrappers so tests can monkeypatch without touching the real server
_get = lambda url, timeout: requests.get(url, timeout=timeout)

EXE_CANDIDATES = [
    shutil.which("ollama"),
    os.path.expandvars(r"%LOCALAPPDATA%\Programs\Ollama\ollama.exe"),
    r"C:\Program Files\Ollama\ollama.exe",
]


def find_ollama_exe() -> str | None:
    for c in EXE_CANDIDATES:
        if c and Path(c).exists():
            return c
    return None


def _spawn_ollama(exe: str) -> None:
    """Launch `ollama serve` detached so it survives this process."""
    flags = 0
    if os.name == "nt":  # detach on Windows
        flags = subprocess.DETACHED_PROCESS | subprocess.CREATE_NEW_PROCESS_GROUP
    subprocess.Popen(
        [exe, "serve"],
        stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=flags, close_fds=True,
    )


def ensure_ollama(base_url: str = "http://localhost:11434",
                  wait_seconds: int = 45) -> bool:
    """Return True if the Ollama server answers, launching it if needed."""
    base = base_url.rstrip("/")
    url = f"{base}/api/version"

    def up() -> bool:
        try:
            _get(url, timeout=2).raise_for_status()
            return True
        except Exception:
            return False

    if up():
        return True

    exe = find_ollama_exe()
    if exe:
        _spawn_ollama(exe)

    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if up():
            return True
        time.sleep(1)
    return False
