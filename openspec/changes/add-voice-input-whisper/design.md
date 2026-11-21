# Design

## Context

Voice input spans two modules (FastAPI backend + React frontend) and adds a new
heavy dependency (Whisper). The project is constraint-bound to free, local
execution (no cloud APIs) and already runs Ollama locally for the same reason.

## Goals / Non-Goals

- Goals: local speech-to-text for hi/bn/ta/en; minimal UI friction; no impact on
  the existing query pipeline.
- Non-Goals: streaming/realtime ASR, speaker diarization, wake words, voice
  *output* (TTS is a separate future change).

## Decisions

### Library: faster-whisper (CTranslate2) over openai-whisper
- ~4x faster on CPU at equal accuracy, much lower memory; supports CUDA
  transparently if available; pip-installable on Windows without build pain
  (unlike openai-whisper's optional Triton/torch extras).
- Model `small` (244M params, ~460 MB) as default: materially better
  hi/bn/ta accuracy than `base` while still CPU-feasible (~2-5 s for a 10 s
  clip). Configurable via `WHISPER_MODEL`.
- Model loads lazily on first transcription request (keeps API startup fast);
  cached in-process thereafter.

### Endpoint: POST /transcribe (multipart file upload)
- Accept WAV/MP3/M4A/WEBM/OGG (Whisper decodes via ffmpeg bundled in
  faster-whisper's audio pipeline; browser recordings are WEBM/OGG).
- Enforce ≤ 60 s audio (whisper's typical question length; protects RAM).
- Response: `{text, language, duration}`; 4xx with message on empty/undecodable.
- TDD: unit tests around validation + decode; integration test marked slow
  (needs model download) asserting an English sample transcribes correctly.

### Frontend: MediaRecorder → transcript into input box
- `navigator.mediaDevices.getUserMedia({audio:true})` + MediaRecorder (webm).
- Press-and-hold mic button (or click to start/stop for accessibility); on stop,
  POST blob to `/transcribe` via the existing `/api` proxy; put `text` into the
  question textarea; never auto-submit (per spec).
- States: idle → recording (red pulse) → transcribing (spinner) → idle/error.
- Errors (permission denied, transcription failure) shown inline; typing always
  remains available.

### Risks / Trade-offs
- First non-English query also triggers the ~2.3 GB mbart download (existing
  behavior) — voice makes this more likely; acceptable, one-time, free.
- Browser compatibility: MediaRecorder + getUserMedia need HTTPS or localhost —
  fine for local dev; document for future LAN/cloud deployment.
- Whisper hallucinates silence into text occasionally; mitigate by trimming
  leading/trailing silence client-side is out of scope — accept for now.

## Migration Plan
Additive only: new endpoint, new optional frontend control, new config keys.
No changes to `/query`. Rollback = revert commit; nothing persisted.

## Open Questions
- none
