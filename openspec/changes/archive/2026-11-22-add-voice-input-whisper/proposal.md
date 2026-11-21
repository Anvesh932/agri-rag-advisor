# Proposal

## Why

The major project report identified that 45% of test users struggled with the
text-based UI due to low digital literacy. Voice input (speech-to-text) is the
single highest-impact accessibility feature for smallholder farmers, and Whisper
runs locally for free (offline, no API cost), consistent with the project's
low-resource, no-spend deployment constraint.

## What Changes

- Add a `POST /transcribe` backend endpoint that accepts an uploaded audio file
  and returns transcribed text using a locally-run Whisper model
  (faster-whisper / CTranslate2, CPU or CUDA, no cloud calls).
- Add a microphone button to the React chat UI: press to record, release to
  transcribe; the transcript is placed into the question box for the farmer to
  review/edit before submitting (not auto-sent).
- Whisper language auto-detects hi/bn/ta/en; transcript stays in the spoken
  language and flows through the existing translate → retrieve → generate →
  translate-back pipeline unchanged.
- New `WHISPER_MODEL` config (default `small`, ~460 MB download on first use);
  model loads lazily on first request so startup stays fast.

## Capabilities

### New Capabilities

- `voice-input`: Farmer speech-to-text capture: backend transcription endpoint
  (Whisper local) and frontend record-to-question-box flow.

### Modified Capabilities

- none — the RAG query pipeline itself is unchanged; voice is a new input
  modality feeding the existing `/query` endpoint.
