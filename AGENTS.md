# Project Instructions — Agri-RAG Advisor

## Standing instructions (apply to this project and future ones)

- Every new project on this laptop gets **OpenSpec initialized** early
  (`openspec init <path> --tools pi`) and uses the OpenSpec propose → apply →
  archive workflow for features (not for trivial fixes).
- The **agent-skills** pack (addyosmani/agent-skills) is installed globally at
  `~/.pi/agent/skills` — rely on it (e.g., test-driven-development,
  code-review-and-quality, spec-driven-development) and keep it updated
  (`npx skills update`).
- Development follows the **test-driven-development skill**: failing test
  first, minimal implementation to green, refactor.
- LLM inference stays **free and local** (Ollama) unless the user asks
  otherwise; never introduce paid API calls without asking.

## Project facts

- Stack: FastAPI backend (`backend/`), React/Vite frontend (`frontend/`),
  RAG over AGRIS ODS corpus (22,874 chunks, FAISS + BM25 hybrid),
  LLMs via Ollama (llama3.2:3b, mistral:7b-instruct).
- Tests: `cd backend && .venv\Scripts\python.exe -m pytest tests/ -q`
  (slow/integration tests use the `slow` marker).
- Start: `.\start.ps1` from the project root.
