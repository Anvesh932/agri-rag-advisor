# Tasks

## 1. Backend: transcription endpoint

- [x] 1.1 Add `faster-whisper` to requirements; add `WHISPER_MODEL` config (default `small`)
- [x] 1.2 Write RED tests for `/transcribe` validation: empty file → 4xx, oversize (>60 s) → 4xx, non-audio → 4xx, clean error messages
- [x] 1.3 Implement transcription service (lazy model load, cached; decode via faster-whisper; return `{text, language, duration}`) — GREEN
- [x] 1.4 Add slow integration test: real Whisper `small` transcribes an English voice sample correctly (skipped when model not downloaded)
- [x] 1.5 Full test suite green

## 2. Frontend: record-to-question-box

- [x] 2.1 Add mic button with idle/recording/transcribing/error states to the chat UI
- [x] 2.2 MediaRecorder capture → POST blob to `/api/transcribe` via existing proxy
- [x] 2.3 Place transcript into the question textarea for review; never auto-submit
- [x] 2.4 Inline error handling (permission denied, transcription failure); typing flow unaffected
- [ ] 2.5 Manual verification against the running stack

## 3. Multilingual quality pass

- [x] 3.1 Ask 3 real farming questions each in Hindi, Bengali, Tamil through the API (voice and/or text)
- [x] 3.2 Record where answers are generic/wrong/region-mismatched into `docs/quality-backlog.md`
- [x] 3.3 Fix the cheapest high-impact issue found (e.g., prompt tweak), re-verify

## 4. Evaluation harness run

- [ ] 4.1 Run `evaluation/evaluate.py` against the local Ollama models (llama3.2:3b and mistral:7b-instruct)
- [ ] 4.2 Save results table into `docs/` and summarize in README

## 5. Docker

- [ ] 5.1 Verify `docker compose build` succeeds for api + frontend + redis
- [ ] 5.2 Verify `docker compose up` serves health endpoint and frontend
