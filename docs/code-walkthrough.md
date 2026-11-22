# Code Walkthrough — Agri-RAG Advisor

The deepest level: file by file, function by function, with real code excerpts
and the reasoning behind every line that matters. Backend in full detail,
frontend in slightly less.

Read `understanding.md` (narrative) and `detailedexplanation.md` (concepts)
first — this one assumes you know what the pieces are and explains how the
*code* implements them.

---

# PART A — BACKEND

## A1. `backend/app/config.py` — one source of truth

Every tunable lives here as a pydantic `Settings` object loaded from `.env`:

```python
class Settings(BaseSettings):
    llm_provider: str = "ollama"          # "ollama" (free/local) or "hf"
    ollama_base_url: str = "http://localhost:11434"
    hf_token: str = ""
    llm_model: str = "meta-llama/Llama-3.2-3B-Instruct"
    hf_inference_base_url: str = "https://router.huggingface.co/hf-inference/models/"

    faiss_weight: float = 0.7             # hybrid fusion: 0.7 FAISS + 0.3 BM25
    bm25_weight: float = 0.3
    retrieval_top_k: int = 5
    temporal_filter_years: int = 5

    data_raw_dir: Path = BASE_DIR.parent / "data" / "raw"
    ...
    whisper_model: str = "small"
    whisper_device: str = "cpu"
    whisper_compute: str = "int8"
    whisper_max_seconds: float = 60.0
```

Two things worth being able to explain:

**1. The path validator.** Relative paths like `./data/index` in `.env` must
resolve against the *project root*, not the process's current directory
(uvicorn is usually started from `backend/`, Docker from `/app`):

```python
@field_validator("data_raw_dir", "data_processed_dir", "data_index_dir",
                 mode="after")
@classmethod
def _resolve_data_dir(cls, v: Path) -> Path:
    return v if v.is_absolute() else (BASE_DIR.parent / v).resolve()
```

This bug produced `index_ready: false` in Docker: `./data/index` resolved to
`/data/index` instead of `/app/data/index`. Test: `tests/test_config.py`.

**2. The model registry.** One dict maps benchmark names to concrete artifacts:

```python
MODEL_REGISTRY = {
    "llama-3.2-3b": {"hf_id": "meta-llama/Llama-3.2-3B-Instruct",
                     "ollama": "llama3.2:3b", ...},
    "mistral-7b":   {"hf_id": "mistralai/Mistral-7B-Instruct-v0.2",
                     "ollama": "mistral:7b-instruct", ...},
    ...
}
```

The same registry serves both providers (HF ids for benchmarks, Ollama names
for free local runs) and carries the dissertation's reference metrics for
context. `SUPPORTED_LANGS` maps farmer language codes to NLLB codes:
`{"en": "eng_Latn", "hi": "hin_Deva", "bn": "ben_Beng", "ta": "tam_Taml"}`.

---

## A2. `backend/app/preprocess.py` — text → tokens → chunks

Three public functions:

```python
def clean_text(text):       # lowercase, strip specials, collapse whitespace
def tokenize(text):         # SpaCy + keep-lists + lemmatization
def chunk_text(text, tokens=None)  # 200-token chunks, 50-token overlap
```

`clean_text` uses one compiled regex that whitelists scripts we care about —
ASCII plus Devanagari (`\u0900-\u097F`), Bengali (`\u0980-\u09FF`), Tamil
(`\u0B80-\u0BFF`) — everything else becomes a space:

```python
_special_re = re.compile(r"[^a-z0-9\s\u0900-\u097F\u0980-\u09FF\u0B80-\u0BFF]")
```

`tokenize` is where the two domain decisions live:

```python
def _tokens_from_doc(doc) -> list[str]:
    for t in doc:
        if t.like_num or t.text in AGRI_KEEP:    # numbers and agri terms survive
            toks.append(t.text)
        elif t.is_stop or t.text in _CUSTOM_STOP:
            continue                              # generic stopwords die
        else:
            toks.append(t.lemma_ if t.lemma_ else t.text)   # fertilizers → fertilizer
```

`like_num` matters: SpaCy's generic stop list contains "one", "two", "three" —
but "three varieties" is information in farming. Dropping numbers silently
degraded the corpus, caught by `test_tokenize_removes_stopwords_but_keeps_agri_terms`.

The singleton `_nlp()` deliberately **excludes parser and NER**:

```python
return spacy.load("en_core_web_sm", exclude=["parser", "ner"])
```

Dependency parsing and named-entity recognition are the two most expensive
pipeline components and we use neither — tokenization + lemmatization only
(needs the tagger). Batched form:

```python
def tokenize_batch(texts, batch_size=128):
    return [_tokens_from_doc(doc) for doc in nlp.pipe(cleaned, batch_size=batch_size)]
```

`nlp.pipe` processes documents in batches on one pipeline pass — this took
preprocessing from ~14.9 s per 200 records to under 10 s (performance-gated by
`test_preprocess_corpus_performance_gate`, written RED before the optimization).

`chunk_text` implements the overlap math:

```python
CHUNK_SIZE, CHUNK_OVERLAP = 200, 50
step = CHUNK_SIZE - CHUNK_OVERLAP            # stride 150
return [" ".join(tokens[i:i + CHUNK_SIZE]) for i in range(0, len(tokens), step)]
```

Test: last 50 tokens of chunk 1 must equal first 50 of chunk 2
(`test_chunk_text_long_input_respects_size_and_overlap`).

---

## A3. `backend/app/embeddings.py` — text → 384 floats

```python
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

@lru_cache(maxsize=1)
def get_encoder():
    return SentenceTransformer(MODEL_NAME)

def encode(texts):
    return get_encoder().encode(texts, normalize_embeddings=True).tolist()
```

`@lru_cache(maxsize=1)` makes the model a process-wide singleton — it's ~90 MB
on disk and you want it loaded exactly once, lazily (first call pays, every
later call reuses). `normalize_embeddings=True` gives unit-length vectors,
which is what makes the inner-product index behave as cosine similarity.
`EMBED_DIM = 384` is a constant the FAISS index is built against — changing
the model means rebuilding the index.

---

## A4. `backend/build_faiss_index.py` — offline indexing

Run once (after `preprocess.py`):

1. Read `chunks.jsonl`.
2. Encode all chunk texts → `(22874, 384)` float32 matrix.
3. Build the exact-search index and save everything:

```python
index = faiss.IndexFlatIP(vecs.shape[1])   # exact inner product
index.add(vecs)
np.save(INDEX_DIR / "vectors.npy", vecs)   # cache: rebuild without re-encoding
faiss.write_index(index, str(INDEX_DIR / "faiss.index"))
```

Then BM25 — **from stored tokens, not re-tokenization**:

```python
if "tokens" in chunks[0] and chunks[0]["tokens"]:
    tokenized = [c["tokens"] for c in chunks]       # fast path
else:
    tokenized = [tokenize(t) for t in texts]        # legacy fallback (slow)
bm25 = BM25Okapi(tokenized)
```

The `vectors.npy` cache and token reuse both exist because the embedding step
alone took ~14 minutes on CPU; a rebuild that re-does it for no reason wastes
a quarter hour every time.

Finally `meta.jsonl`: one line per chunk with `{title, year, region,
source_file}` — retrieval returns *metadata* + text, and the FAISS index only
stores vectors, so metadata must be re-aligned by position (index row `i` ↔
line `i` of `meta.jsonl`). That positional contract is why the build writes
all three files in one pass.

---

## A5. `backend/app/retrieval.py` — the hybrid search engine

The `Retriever` class uses **lazy properties** so the API process doesn't load
300 MB of indexes at import time:

```python
@property
def index(self):
    if self._index is None:
        self._index = faiss.read_index(str(INDEX_DIR / "faiss.index"),
                                       faiss.IO_FLAG_MMAP)
    return self._index
```

`IO_FLAG_MMAP` memory-maps the index — the OS pages vectors in on demand
instead of reading 35 MB up front.

### `search()` — the heart of the system, step by step

```python
top_k = top_k or settings.retrieval_top_k          # 5
dense_k = sparse_k = max(top_k * 5, 25)            # over-fetch 25 candidates
```

**Dense leg:**

```python
q = np.array([encode_query(query)], dtype="float32")   # 1×384, normalized
d_scores, d_ids = self.index.search(q, dense_k)        # top-25 by cosine
```

**Sparse leg:**

```python
s_scores = self.bm25.get_scores(tokenize(query))       # score ALL 22874 chunks
s_ids = np.argsort(s_scores)[::-1][:sparse_k]          # descending top-25
```

Note the asymmetry: FAISS returns top-k natively; BM25 scores everything and
we argsort ourselves. `[::-1]` flips ascending→descending.

**Score normalization** — the two scorers live on different scales (cosine
0–1; BM25 can hit 30+), so fusing raw numbers would let BM25 dominate. Min-max
squashes each list to 0–1 *over the candidates actually fetched*:

```python
def norm(a):
    rng = a.max() - a.min()
    return (a - a.min()) / rng if rng > 0 else np.zeros_like(a)
```

(The `rng > 0` guard handles the all-equal-scores edge case → zeros instead of
NaN.)

**Fusion via dict accumulation** — a chunk found by *both* retrievers gets
*both* contributions added, which is exactly the desired behavior (agreement
boosts rank):

```python
fused: dict[int, float] = {}
for i, s in zip(d_ids, norm(d_scores)):
    fused[int(i)] = fused.get(int(i), 0.0) + settings.faiss_weight * float(s)  # ×0.7
for i, s in zip(s_ids, norm(s_scores)):
    fused[int(i)] = fused.get(int(i), 0.0) + settings.bm25_weight * float(s)   # ×0.3
```

**Filters before top-k** (so filters can't shrink the result below top_k):

```python
for idx, score in sorted(fused.items(), key=lambda kv: -kv[1]):
    if region and region.lower() not in m["region"].lower():  continue
    if after_year and int(year) < after_year:                 continue
    results.append(...); if len(results) >= top_k: break
```

`temporal_search()` computes the year floor dynamically — the latest year in
the corpus minus `temporal_filter_years` (5) — so the guardrail survives new
data being added.

A module-level `retriever = Retriever()` singleton is imported by `main.py`
and `guardrails.py` — one shared set of loaded indexes per process.

---

## A6. `backend/app/llm.py` — provider router + prompt

```python
def generate(model_key, system_prompt, question, context,
             provider=None, max_new_tokens=512):
    provider = provider or settings.llm_provider
    if provider == "ollama":
        return generate_ollama(ollama_model_name(model_key), ...)
    return generate_remote(resolve_model(model_key), ...)
```

`ollama_model_name` maps registry keys to local names with a deliberate
fallback chain — models without an Ollama equivalent (falcon-7b, phi-2) fall
back to the default rather than 500-ing:

```python
reg = MODEL_REGISTRY.get(key_or_hf_id)
if reg: return reg.get("ollama", OLLAMA_DEFAULT)
if ":" in key_or_hf_id: return key_or_hf_id   # already an Ollama name
return OLLAMA_DEFAULT                          # "llama3.2:3b"
```

(The `":"` check distinguishes direct Ollama names like `mistral:7b-instruct`
from unknown benchmark keys.)

`generate_ollama` posts the OpenAI-style chat payload with the dissertation's
sampling settings encoded as Ollama options:

```python
"options": {"temperature": 0.7, "top_p": 0.9,
            "repeat_penalty": 1.2, "num_predict": max_new_tokens}
```

`SYSTEM_PROMPT` is the grounding contract (quoted in full in
`detailedexplanation.md` §3.2) — "Answer ONLY from the provided context … if
insufficient, say what extra information is needed."

`generate_local` (kept but unused by default) shows the bitsandbytes 8-bit
path for CUDA machines — the code is the documentation of how you'd run the
*uncompressed-pipeline* models on-GPU with `load_in_8bit=True, device_map="auto"`.

---

## A7. `backend/app/translate.py` — NLLB wrapper

```python
NLLB_MODEL = "facebook/nllb-200-distilled-600M"

@lru_cache(maxsize=1)
def get_translator():
    tok = AutoTokenizer.from_pretrained(NLLB_MODEL, use_fast=False)
    model = AutoModelForSeq2SeqLM.from_pretrained(NLLB_MODEL)
    model.eval()
    return tok, model
```

Three deliberate choices:

1. **`use_fast=False`** — mbart (and NLLB in this transformers era) fails
   fast-tokenizer conversion because the repo ships SentencePiece models, not
   `tokenizer.json`; the slow tokenizer avoids the whole class of breakage.
2. **Manual `generate()` instead of `pipeline()`** — transformers v5 removed
   the `translation` pipeline; explicit calls survive version churn.
3. **`model.eval()` + `torch.no_grad()`** — disables dropout and gradient
   bookkeeping: inference should be deterministic-ish and lean.

The translate call:

```python
tok.src_lang = to_mbart(src_lang)                # e.g. "ben_Beng"
encoded = tok(text, return_tensors="pt", truncation=True, max_length=1024)
generated = model.generate(
    **encoded,
    forced_bos_token_id=tok.convert_tokens_to_ids(to_mbart(tgt_lang)),
    max_length=1024)
return tok.batch_decode(generated, skip_special_tokens=True)[0].strip()
```

`forced_bos_token_id` pins the decoder's first token to the target language
code — the mechanism that makes one 600M model serve all 200 language pairs.
The `src == tgt` short-circuit at the top saves two model calls for English
queries (English farmers never trigger a 2.4 GB model load).

---

## A8. `backend/app/glossary.py` — deterministic de-code-mixing

Data-driven: one dict of 22 regex patterns → 3 languages, e.g.

```python
"crop rotation": {"hi": "फसल चक्र", "bn": "ফসল চক্র", "ta": "பயிர் சுழற்சி"},
"compost(?:ing|ed|s)?": {"hi": "कम्पोस्ट", ...},   # inflections folded in
```

Ordering is the subtle part — compound phrases must be replaced before their
components or "drip irrigation" becomes "टपक सिंचाई सिंचाई"-style mangles:

```python
_ORDERED_PATTERNS = sorted(GLOSSARY.keys(), key=lambda p: -len(p.split()))
```

Application is plain `regex.sub(term, out)` with word boundaries and
`IGNORECASE` — deterministic, instant, and fully testable (no model in the
loop). English short-circuits: the glossary only fires for hi/bn/ta.

---

## A9. `backend/app/guardrails.py` — ethics filters

Three mechanisms, all transparent (flag, don't silently drop):

1. **Temporal** — `temporal_search()` re-queries with `after_year = latest_year − 5`.
2. **Niche/region** — regex pairs like
   `r"\bwinter crops?\b.*\btropical\b"` flag context-mismatched content.
3. **Output bias scan** — `check_output(answer)` runs `_SUSPECT` patterns over
   the final text (e.g. `increase … (pesticide|herbicide)` in either word
   order, gender-stereotyped framing) and returns `{flags, ok}` that the API
   passes through to the UI.

The temporal+nicheline composition:

```python
def apply_guardrails(query, chunks):
    return niche_filter(query, temporal_filter(query, chunks))
```

---

## A10. `backend/app/transcribe.py` — voice in

```python
@lru_cache(maxsize=1)
def get_pipeline():
    return WhisperModel(settings.whisper_model,        # "small"
                        device=settings.whisper_device,   # "cpu"
                        compute_type=settings.whisper_compute)  # "int8"
```

Validation happens *before* the model can load (a 61-second WAV must be
rejected without paying a 460 MB model load):

```python
async def validate_upload(file) -> bytes:
    data = await file.read()
    if not data:                       raise ValueError("The uploaded audio is empty.")
    if len(data) > MAX_UPLOAD_BYTES:   raise ValueError("Audio too large ...")
    if not ctype.startswith(("audio/", "video/webm", "video/mp4")):
                                       raise ValueError("Unsupported file type ...")
    dur = _wav_duration(data)          # WAV headers → exact duration
    if dur and dur > settings.whisper_max_seconds:
        raise ValueError(f"Recording too long ({dur:.0f}s) ...")
```

(WAV duration is checkable from headers alone; for webm/opus we lean on the
size limit — browsers cap recordings well under it.)

Transcription writes bytes to a temp file (faster-whisper wants a path),
decodes via PyAV, runs Whisper with `vad_filter=True` (silence trimming),
joins segment texts, and returns `{text, language, duration}`. The temp file
is deleted in a `finally` block.

Why `device="cpu", compute_type="int8"` as defaults: the CUDA path needs
cuBLAS/cuDNN DLLs the user may not have — discovered live when a test failed
with "Library cublas64_12.dll is not found". CPU int8 is universally safe and
fast enough for ≤60 s clips.

---

## A11. `backend/app/ollama.py` — self-healing dependency

```python
def ensure_ollama(base_url="http://localhost:11434", wait_seconds=45) -> bool:
    if up(): return True                    # probe /api/version
    exe = find_ollama_exe()                 # PATH → %LOCALAPPDATA% → Program Files
    if exe: _spawn_ollama(exe)              # detached `ollama serve`
    deadline = time.time() + wait_seconds
    while time.time() < deadline:
        if up(): return True
        time.sleep(1)
    return False                            # report, don't crash
```

`_spawn_ollama` uses `DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP` on Windows
so the spawned server survives the API process. Tests monkeypatch `_get` and
`_spawn_ollama` to simulate "server down → spawned → comes up" and "never
comes up → False" without touching a real server.

---

## A12. `backend/app/main.py` — the API surface

### Lifespan (startup checks)

```python
@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.index_ready = (settings.data_index_dir / "faiss.index").exists()
    app.state.ollama_ready = (settings.llm_provider != "ollama"
                              or ensure_ollama(settings.ollama_base_url))
    yield
```

Both checks are *soft* — the server starts anyway and reports readiness on
`/health`, so a half-configured machine gets actionable diagnostics instead of
a crashed process.

### The `/query` handler — five steps, commented in code

```python
def query(req: QueryRequest):                       # pydantic-validated body
    q_en = translate(req.question, req.lang, "en")  # 1. → English

    chunks = retriever.search(q_en, region=req.region) if app.state.index_ready else []
    chunks = apply_guardrails(q_en, chunks)         # 2. retrieve + filter

    context = "\n\n".join(                          #    formatted evidence block
        f"[{c.get('title','')!r} ({c.get('year','n/a')})] {c.get('text','')}"
        for c in chunks)

    key = _cache_key(req)                           # 3. sha256(question.lower,
    cached = _cache_get(key)                        #       lang, model, region)
    answer_en = cached or generate(req.model, SYSTEM_PROMPT, q_en, context)
    if not cached: _cache_set(key, answer_en, ttl=86400)   # 24 h

    answer = translate(answer_en, "en", req.lang)   # 4. back-translation
    answer = apply_glossary(answer, req.lang)       #    + de-code-mixing

    return QueryResponse(                           # 5. answer + sources +
        answer=answer,                              #    guardrail flags
        sources=[{"title": c.get("title"), "year": ..., "score": ..., "flag": ...}
                 for c in chunks],
        guardrails=check_output(answer), model=req.model)
```

Details worth pointing at in an interview:

- **Cache key includes language+model+region** — the same English question in
  Hindi vs Tamil is two cache entries (back-translations differ).
- **Redis helpers swallow all exceptions** — cache is an optimization, never a
  dependency. `socket_connect_timeout=0.5` keeps a dead Redis from stalling
  requests.
- **Context formatting embeds title+year inline** so the model can cite
  naturally ("per 'Study…' (2022)").

### `/transcribe`

```python
@app.post("/transcribe", response_model=TranscribeResponse)
async def transcribe_endpoint(file: UploadFile = File(...)):
    try:
        data = await validate_upload(file)          # empty/size/type/duration
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    try:
        return TranscribeResponse(**transcribe(data))
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not decode the audio: {e}")
```

Validation errors are 4xx with *human-readable* messages (they surface in the
UI); decode failures are also 4xx because a bad upload is the user's problem,
not a server fault.

---

## A13. `backend/preprocess.py` + `backend/fetch_agris.py` — the data pipeline

`preprocess.py`:

```python
def process_corpus(docs, batch_size=128) -> list[dict]:
    for i in range(0, len(docs), batch_size):
        token_lists = tokenize_batch([d["abstract"] for d in docs[i:i+batch_size]])
        for d, tokens in zip(batch, token_lists):
            for chunk in chunk_text(d["abstract"], tokens=tokens):
                chunks.append({"text": chunk, "tokens": tokens, ...})
```

Note `tokens=tokens` — `chunk_text` accepts pre-tokenized input so the batch
pass and chunking share one tokenization. The main loop prints progress every
1,000 docs (a hang with no output was the original complaint that triggered
the optimization) and reconfigures stdout to UTF-8 (cp1252 crashes on "→").

`fetch_agris.py` (data acquisition, stdlib-only so it runs pre-venv):

1. Download the ODS catalog XML → map subset codes to provider names.
2. Select South Asia by code prefix (`IN/IM/NP/PK/BD/LK/BH`) *or* provider
   name keyword, minus a blacklist ("ocean", "caribbean" — matching caught
   "Bluefields **Indian** & Caribbean University" otherwise).
3. Download each subset with **cache + backoff**:

```python
if dest.exists() and dest.stat().st_size > 0: return        # resume-safe
for attempt in range(1, retries + 1):
    try: stream to file (1 MB chunks)
    except HTTPError as e:
        if e.code in (429, 500, 502, 503): time.sleep(30 * attempt); continue
```

4. Stream-parse with `iterparse`, filter (§2.4 of detailedexplanation), write
   the normalized `agris_filtered.xml`.

---

## A14. Backend tests — what each file locks down

| File | Locks down |
|---|---|
| `test_preprocess.py` (14) | cleaning, tokenization decisions, chunk math, AGRIS parsing, **performance gate**, cp1252 print regression |
| `test_retrieval.py` | index metric (IP not L2) + on-topic relevance (slow, real index) |
| `test_llm.py` | provider routing (mocked HTTP), Ollama name mapping + fallbacks, real generation (slow) |
| `test_translate.py` | NLLB Hindi + Bengali quality gates (slow), English no-op |
| `test_glossary.py` (10) | term replacement per language, case-insensitivity, inflections, longest-first ordering, English no-op |
| `test_query_flow.py` | glossary **wired into** `/query` (stubbed RAG, fast) |
| `test_transcribe.py` | upload validation (no model load), transcription contract, real Whisper on SAPI-generated speech (slow) |
| `test_ollama.py` | auto-start logic with mocked spawn/probe |
| `test_config.py` | relative-path resolution, absolute passthrough |
| `test_guardrails.py` | flag patterns incl. word-order and Unicode answers |

Fast suite: ~30 s. Slow suite adds model downloads/loads; runs on demand via
`pytest -m slow`.

---

# PART B — FRONTEND (slightly less detailed)

## B1. `frontend/src/api.js` — the transport layer

```javascript
export async function askQuestion({ question, lang, model, region }) {
  const res = await fetch('/api/query', { method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ question, lang, model, region: region || null }) })
  if (!res.ok) throw new Error(`API error ${res.status}`)
  return res.json()
}

export async function transcribe(blob) {
  const form = new FormData()
  form.append('file', blob, 'recording.webm')
  const res = await fetch('/api/transcribe', { method: 'POST', body: form })
  if (!res.ok) { let msg = ...; try { msg = (await res.json()).detail || msg } catch {}
                 throw new Error(msg) }
  return res.json()
}
```

Everything goes through the relative `/api` prefix — in dev, Vite proxies it
to `:8000` (`vite.config.js`); in Docker, nginx does (see B4). That one
abstraction is what made the dev↔Docker swap seamless once nginx was fixed.
Error messages prefer the backend's `detail` field — the user sees "Recording
too long (95s)…" not "HTTP 400".

## B2. `frontend/src/App.jsx` — one screen, one state machine

State hooks: `question, lang, model, region, result, loading, error` for the
chat, plus the recording machine: `recState` (`idle | recording |
transcribing`) and `micError`.

Recording lifecycle (press-and-hold, with click fallback for accessibility):

```javascript
async function startRecording() {
  const stream = await navigator.mediaDevices.getUserMedia({ audio: true })
  const rec = new MediaRecorder(stream)
  chunksRef.current = []
  rec.ondataavailable = (e) => chunksRef.current.push(e.data)
  rec.onstop = async () => {
    stream.getTracks().forEach(t => t.stop())     // release the mic!
    setRecState('transcribing')
    const { text } = await transcribe(new Blob(chunksRef.current, ...))
    setQuestion(q => q ? q + ' ' : '' + text)     // into the box, NOT submitted
    setRecState('idle')
  }
  rec.start(); setRecState('recording')
}
```

Design decisions per the spec:

- transcript **appends into the textarea** — farmer reviews/edits before
  pressing "Get advice" (no auto-submit),
- mic is explicitly released on stop (the trailing green dot otherwise stays
  "in use", which browsers indicate system-wide),
- every failure path (`micError`) tells the user typing still works.

The ask flow:

```javascript
async function submit(e) {
  setLoading(true); setError(null); setResult(null)
  try { setResult(await askQuestion({ question, lang, model, region })) }
  catch (err) { setError(err.message) } finally { setLoading(false) }
}
```

Rendering the trust UI:

```jsx
{result && (
  <section>
    <p>{result.answer}</p>
    {result.guardrails?.flags?.length > 0 && <p>⚠ Guardrails: ...</p>}
    <details><summary>Sources ({result.sources.length})</summary>
      {result.sources.map(s => <li>{s.title} ({s.year}) — score {s.score}</li>)}
    </details>
  </section>
)}
```

Sources are collapsed by default (visual calm) but one click away — citations
are present *and* unobtrusive.

## B3. `frontend/src/main.jsx` + `index.html` — 12 lines of boilerplate

React 18 `createRoot`, one stylesheet-free inline style object in `App.jsx`
(no CSS framework — a single-screen tool doesn't need one).

## B4. `frontend/vite.config.js` + `nginx.conf` — the dev/prod proxy pair

Dev (Vite):

```javascript
server: { port: 5173,
  proxy: { '/api': { target: 'http://localhost:8000',
                     rewrite: p => p.replace(/^\/api/, '') } } }
```

Prod (nginx in the container):

```nginx
location /api/ {
    proxy_pass http://api:8000/;        # compose service name, not localhost
    client_max_body_size 20m;           # voice uploads
    proxy_read_timeout 600s;            # local LLM generation is slow
}
```

Both do the same job (strip `/api`, forward to the backend) — the nginx
version's `api:8000` resolves through Docker's internal DNS, and the timeouts
match the reality of 30–160 s multilingual answers.

## B5. `frontend/Dockerfile` — two-stage build

```dockerfile
FROM node:20-alpine AS build
COPY package.json ./
RUN npm install
COPY . .
RUN npm run build            # → /app/dist (static bundle)

FROM nginx:alpine
COPY --from=build /app/dist /usr/share/nginx/html
COPY nginx.conf /etc/nginx/conf.d/default.conf
```

Node toolchain exists only in the build stage; the final image is just nginx +
static files (~50 MB).

---

# PART C — INFRA GLUE

## C1. `docker-compose.yml`

```yaml
api:
  build: { context: ., dockerfile: backend/Dockerfile }   # root context!
  environment:
    OLLAMA_BASE_URL: http://host.docker.internal:11434    # host Ollama
    REDIS_URL: redis://redis:6379/0                       # compose redis
    DATA_INDEX_DIR: /app/data/index                       # absolute, pinned
  extra_hosts: ["host.docker.internal:host-gateway"]
  volumes: [./data:/app/data]                             # reuse host indexes
```

Three container-lessons encoded here: build context must contain everything
COPYed (hence root context); `localhost` inside a container is the container
itself; relative paths resolve differently — pin them.

## C2. `start.ps1` / `start-backend.ps1` / `start-frontend.ps1`

One-command startup: `start.ps1` spawns two PowerShell windows running the
other scripts. `start-backend.ps1` exists because of two user-facing failure
modes: wrong working directory and the "venv not really activated" trap — the
script calls the venv's `python.exe` **by absolute path**, which cannot pick
up the wrong interpreter.

---

# PART D — HOW TO TALK ABOUT THIS CODE

**"Walk me through what happens when a farmer asks a question."** — Trace
A12's `/query` handler top to bottom: pydantic validation → NLLB translate →
hybrid retrieve (A5) → guardrails (A9) → prompt assembly → provider router
(A6) → Redis cache → back-translate → glossary (A8) → response with sources.

**"Where would it break first at 1000 concurrent users?"** — the LLM is the
bottleneck (seconds per query, local GPU). Next: FAISS search is
single-threaded per query but trivially parallel; Redis cache absorbs repeats.
The fix path: more GPU replicas behind a queue, then ANN (HNSW) when the
corpus outgrows exact search.

**"What's the riskiest line of code?"** — arguably
`fused[int(i)] = fused.get(int(i), 0.0) + weight * s`: correctness of the
whole system depends on the two score lists being normalized to comparable
scales *over the same candidate sets*. Change the over-fetch depth and the
min-max behavior changes. It's tested, but it's the line I'd watch.
