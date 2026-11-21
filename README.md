# Agri-RAG Advisor

Region-specific agricultural advisory system for smallholder farmers in South Asia,
built with Retrieval-Augmented Generation (RAG) and open LLMs, aligned with
**UN SDG 12: Responsible Consumption and Production**.

Based on the major project dissertation *"Integrating LLMs and RAG in Sustainable
Agriculture"* (Anvesh Tewari, VIPS-TC, GGSIP University) — see `docs/`.

## Architecture

```
User query (text/voice, hi/bn/ta/en)
        │
        ▼
Translation (mbart-large-50) ──► FastAPI backend
        │
        ▼
Hybrid Retrieval:  0.7 * FAISS dense (all-MiniLM-L6-v2, HNSW)
                 + 0.3 * BM25 sparse (keyword precision)
        │            ── knowledge base: AGROVOC-expanded terms → AGRIS abstracts (2015–2024)
        ▼
LLM generation (Hugging Face, 8-bit quantized)
   Falcon-7B-Instruct | Llama-3.2-3B-Instruct | Mistral-7B | Phi-2
        │            ── guardrails: temporal filters, climate/soil niche filters
        ▼
Back-translation → farmer, in local language
```

## Repository layout

```
backend/          FastAPI service + RAG pipeline (Python)
  app/            main.py, retrieval, embeddings, llm, translate, guardrails, preprocess
  build_faiss_index.py   builds data/index/faiss.index from processed corpus
frontend/         React (Vite) multilingual chat UI
evaluation/       BLEU-4 / ROUGE-L / Exact Match harness + sample queries
data/             raw/ AGRIS XML, processed/ chunks, index/ FAISS + BM25 artifacts
docs/             Source papers & dissertation (PDF/DOCX)
```

## Quickstart (Windows)

The easy way — from the project root in PowerShell:

```powershell
cd D:\Projects\agri-rag-advisor
.\start.ps1        # opens backend + frontend in their own windows
```

Then open **http://localhost:5173**.

Or manually:

```powershell
# Backend (from backend/ — auto-starts Ollama if it isn't running)
cd D:\Projects\agri-rag-advisor\backend
.\.venv\Scripts\python.exe -m uvicorn app.main:app --port 8000

# Frontend (second terminal)
cd D:\Projects\agri-rag-advisor\frontend
npm run dev
```

### Backend

```bash
cd backend
source .venv/bin/activate        # Linux/macOS (Windows: .venv\Scripts\activate)
pip install -r requirements.txt
cp ../.env.example .env         # fill in HF token etc.

# 1. place AGRIS XML exports in ../data/raw/, then:
python preprocess.py            # → ../data/processed/chunks.jsonl
python build_faiss_index.py     # → ../data/index/

# 2. serve
uvicorn app.main:app --reload --port 8000
```

### Frontend

```bash
cd frontend
npm install
npm run dev       # http://localhost:5173
```

### Docker

```bash
docker compose up --build
```

## Evaluation

Local (Ollama) numbers vs the dissertation benchmark: see `docs/eval-results.md`.

```bash
python evaluation/evaluate.py --queries evaluation/sample_queries.json
```

Reference benchmarks from the dissertation (1,000-query test set):

| Model               | BLEU-4 | ROUGE-L | EM   | Wh/query |
|---------------------|--------|---------|------|----------|
| Mistral-7B          | 0.78   | 0.74    | 85%  | 3.8      |
| Falcon-7B-Instruct  | 0.75   | 0.71    | 81%  | 3.2      |
| Llama-3.2-3B        | 0.72   | 0.68    | 78%  | 2.5      |
| Phi-2               | 0.70   | 0.66    | 76%  | 2.1      |

Retrieval: Recall@5 = 85% (FAO test questions), ~120 ms mean latency with Redis caching.

## Environment variables

See `.env.example`. Requires a Hugging Face token (`HF_TOKEN`) with access to the
selected models. Optional: `REDIS_URL`, `GOOGLE_TRANSLATE_API_KEY`.
