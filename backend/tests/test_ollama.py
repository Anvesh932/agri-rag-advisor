"""Tests for Ollama auto-start (backend boots the LLM server if it's down)."""
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app import ollama as omod
from app.ollama import ensure_ollama


def test_ensure_returns_true_when_server_already_running():
    # the real Ollama server is expected to be running on this machine
    assert ensure_ollama("http://localhost:11434", wait_seconds=5) is True


def test_ensure_spawns_server_when_down(monkeypatch):
    """Server unreachable → we try to spawn it and wait for it to answer."""
    state = {"calls": 0}

    class FakeResp:
        def raise_for_status(self): pass
        def json(self): return {"version": "0.0.0-test"}

    def fake_get(url, timeout=None):
        state["calls"] += 1
        if state["calls"] <= 2:          # first two probes: server still starting
            raise ConnectionError("refused")
        return FakeResp()

    spawned = []
    monkeypatch.setattr(omod, "_get", fake_get)
    monkeypatch.setattr(omod, "_spawn_ollama", lambda exe: spawned.append(exe))
    monkeypatch.setattr(omod.time, "sleep", lambda s: None)

    assert ensure_ollama("http://localhost:11434", wait_seconds=5) is True
    assert len(spawned) == 1             # it tried to launch the server once
    assert state["calls"] >= 3           # and polled until it was up


def test_ensure_gives_up_after_timeout(monkeypatch):
    """Server never comes up → return False, don't hang or raise."""
    def fake_get(url, timeout=None):
        raise ConnectionError("refused")
    spawned = []
    monkeypatch.setattr(omod, "_get", fake_get)
    monkeypatch.setattr(omod, "_spawn_ollama", lambda exe: spawned.append(exe))
    monkeypatch.setattr(omod.time, "sleep", lambda s: None)

    assert ensure_ollama("http://localhost:11434", wait_seconds=1) is False
    assert len(spawned) == 1


def test_ensure_finds_executable():
    exe = omod.find_ollama_exe()
    assert exe and Path(exe).exists()
