# Spec Delta

## Purpose
Voice input lets a farmer ask farming questions by speaking instead of typing,
removing the text-entry barrier that 45% of test users hit in the major project's
UI trials. Transcription runs locally (no cloud, no cost) and feeds the existing
multilingual advisory pipeline.

## ADDED Requirements

### Requirement: Transcribe uploaded audio to text
The system SHALL expose a transcription endpoint that accepts an audio upload
and returns the spoken text, running speech recognition entirely on-device.

#### Scenario: English question recorded via microphone
- **WHEN** a valid WAV/MP3/WEBM/OGG audio file containing the spoken question
  "How can I improve soil fertility?" is uploaded to the transcription endpoint
- **THEN** the response contains transcribed text with those words (or a close
  paraphrase) and the detected source language
- **AND** no network call to any speech service occurs

#### Scenario: Non-English speech is auto-detected
- **WHEN** the uploaded audio contains a question spoken in Hindi, Bengali, or
  Tamil
- **THEN** the transcription endpoint returns the transcript in that same
  language and reports the detected language code
- **AND** the transcript can flow through the existing multilingual query
  pipeline (translate → retrieve → generate → translate back) without manual
  language selection

#### Scenario: Unsupported or empty upload
- **WHEN** the upload is empty, not decodable audio, or exceeds 60 seconds
- **THEN** the endpoint responds with a client-error status and a clear message
- **AND** the service remains available for subsequent requests

### Requirement: Record voice in the chat UI
The UI SHALL let the farmer record a question with a microphone control and
place the transcript into the question box for review before sending.

#### Scenario: Farmer records and reviews before asking
- **WHEN** the farmer presses the microphone control, speaks, and releases it
- **THEN** the recorded audio is transcribed and the transcript text appears in
  the question input box
- **AND** the question is NOT submitted automatically; the farmer can edit or
  discard the text before pressing the existing ask button

#### Scenario: Recording feedback and failure
- **WHEN** recording starts, is in progress, or finishes
- **THEN** the UI shows a visible recording indicator and reverts to idle state
  when finished
- **AND** if the microphone is denied or transcription fails, the UI shows an
  understandable message while the typed-input flow keeps working
