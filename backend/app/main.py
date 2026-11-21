"""FastAPI entrypoint: /query endpoint (text in → multilingual advisory out)."""
import hashlib
import json
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from .config import MODEL_REGISTRY, settings
from .guardrails import apply_guardrails, check_output
from .llm import SYSTEM_PROMPT, generate
from .retrieval import retriever
from .translate import translate


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Fail fast if the index hasn't been built yet (soft-fail: keep serving 503s per query)
    app.state.index_ready = (
        settings.data_index_dir / "faiss.index").exists()
    yield


app = FastAPI(title="Agri-RAG Advisor", version="0.1.0", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_methods=["*"],
    allow_headers=["*"],
)


class QueryRequest(BaseModel):
    question: str
    lang: str = "en"                    # en | hi | bn | ta
    model: str = "llama-3.2-3b"         # registry key or HF id
    region: str | None = Field(default=None, description="e.g. 'Indo-Gangetic Plain'")


class QueryResponse(BaseModel):
    answer: str
    sources: list[dict]
    guardrails: dict
    model: str


def _cache_key(req: QueryRequest) -> str:
    raw = json.dumps([req.question.lower(), req.lang, req.model, req.region])
    return hashlib.sha256(raw.encode()).hexdigest()


def _cache_get(key: str) -> str | None:
    try:
        import redis
        r = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=0.5)
        return r.get(f"ans:{key}")
    except Exception:
        return None


def _cache_set(key: str, value: str, ttl: int = 86400) -> None:
    try:
        import redis
        r = redis.Redis.from_url(settings.redis_url, socket_connect_timeout=0.5)
        r.set(f"ans:{key}", value, ex=ttl)
    except Exception:
        pass


@app.get("/health")
def health():
    return {"ok": True, "index_ready": app.state.index_ready,
            "models": list(MODEL_REGISTRY)}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    # 1. translate to English for retrieval + generation
    q_en = translate(req.question, req.lang, "en")

    # 2. hybrid retrieval + guardrails
    chunks = retriever.search(q_en, region=req.region) if app.state.index_ready else []
    chunks = apply_guardrails(q_en, chunks)
    context = "\n\n".join(f"[{c.get('title','')!r} ({c.get('year','n/a')})] {c.get('text','')}"
                          for c in chunks)

    # 3. generation (Redis-cached for frequent queries, e.g. sowing dates)
    key = _cache_key(req)
    cached = _cache_get(key)
    answer_en = cached or generate(req.model, SYSTEM_PROMPT, q_en, context)
    if not cached:
        _cache_set(key, answer_en)

    # 4. translate back to the farmer's language
    answer = translate(answer_en, "en", req.lang)

    return QueryResponse(
        answer=answer,
        sources=[{"title": c.get("title"), "year": c.get("year"),
                  "score": c.get("score"), "flag": c.get("flag")}
                 for c in chunks],
        guardrails=check_output(answer),
        model=req.model,
    )
