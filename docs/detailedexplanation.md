# Detailed Explanation — How Agri-RAG Advisor Actually Works

Companion to `understanding.md` (the short version). This goes deep: every
stage of the system, what the technology is, why we chose it, how it works
under the hood, and where the real code lives. Read it top to bottom and you
should be able to answer "how does X work?" for every X in this project.

All numbers and file paths below are the *actual* ones in this repo.

---

# Part 1 — The problem and why RAG

## 1.1 What goes wrong with a plain LLM

A large language model (LLM) like Llama-3 or Mistral generates text by
predicting the next word based on patterns it absorbed during training. Two
consequences:

1. **Its knowledge has a cutoff date.** A model trained in 2023 doesn't know
   2024 research.
2. **It can hallucinate.** When unsure, it doesn't say "I don't know" — it
   produces *plausible-sounding* text. For farming advice, a confidently wrong
   sowing date or fertilizer dose can cost a farmer real money.

## 1.2 Three ways to inject knowledge into an LLM

| Approach | What it is | Why we didn't use it (here) |
|---|---|---|
| **Fine-tuning** | Continue training the model on domain text | Expensive (GPU-days), can't cite sources, goes stale when literature updates, risks forgetting |
| **Long-context** | Paste entire documents into the prompt | Our corpus is 22,874 chunks; impossible to fit, slow, costly |
| **RAG** | Retrieve a few relevant chunks at question time, put them in the prompt | Cheap, updatable, *citable* (we can show which papers were used) |

**RAG = Retrieval-Augmented Generation.** The model answers an open-book exam:
it still writes the sentences, but only from the pages we hand it.

## 1.3 The pipeline in one line

```
question → translate → retrieve relevant chunks → stuff into prompt →
LLM answers → translate back → show with sources
```

Everything else in this doc is a deep-dive on one arrow of that line.

---

# Part 2 — Building the knowledge base

Code: `backend/fetch_agris.py`, `backend/preprocess.py`,
`backend/build_faiss_index.py`. Data flows `data/raw → data/processed → data/index`.

## 2.1 Where the documents come from (AGRIS + AGROVOC)

- **AGRIS** is FAO's international database of agricultural research — 16M+
  records. We use its **Open Data Set (ODS)**: downloadable XML files, split
  per data provider (one file per institute, e.g. `AGRIS.ODS.IN0.xml` =
  Indian Council of Agricultural Research, ~141 MB, ~80k records), licensed
  CC BY 4.0.
- **AGROVOC** is FAO's multilingual agricultural thesaurus. The dissertation's
  method: start from 55 seed terms, expand to ~515 related concepts using
  embedding nearest-neighbors ("biofertilizer" ↔ "vermicompost"), then use the
  expanded terms to search. Our pipeline implements the *search and filter*
  part; concept expansion is encoded in the topic-keyword filter.

## 2.2 The XML format (Dublin Core)

Each AGRIS record is Dublin Core metadata — flat tags:

```xml
<dctypes:BibliographicResource xml:id="64e8bd60...">
  <dc:title>Export and import of bamboo...</dc:title>
  <dc:description>Bamboo trade has played a vital role...</dc:description>  ← abstract
  <dc:date>2023</dc:date>
  <dc:subject schema="AGROVOC">import</dc:subject>
</dctypes:BibliographicResource>
```

Namespaces matter: `dc:` = `http://purl.org/dc/elements/1.1/`,
`dctypes:` = `http://purl.org/dc/dcmitype/`. A one-character namespace mistake
silently matches zero records (that was a real bug — "kept 0 records" until we
found the typo).

## 2.3 Streaming parse (why not load the file?)

A 141 MB XML parsed with `ET.parse()` would build a full DOM tree in RAM
(several GB with Python object overhead). Instead we use
`xml.etree.ElementTree.iterparse` — an *event stream* that hands us each
`</BibliographicResource>` closing tag as it finishes, letting us grab the
fields and **free the element immediately** (`elem.clear()`). Memory stays
flat regardless of file size. This is the standard trick for large XML.

## 2.4 The filter (81k → 19.5k)

From each South Asia provider subset we keep a record only if:

1. It has a **title and an abstract** (no abstract = nothing to embed),
2. Publication year is **2015–2024** — recency is a quality proxy and matches
   the dissertation's window,
3. Title/abstract mentions at least one topic keyword from the three advisory
   domains (stems, so "fertil" matches fertilizer/fertiliser/fertilization):
   `soil, fertil, compost, manure, nutrient, mulch, irrigat, water, drip,
   rainwater, drainage, drought, harvest, storage loss, food loss, grain
   storage, post-harvest`.

Result: **19,562 records → `data/raw/agris_filtered.xml`** (a normalized
`<record><title/><abstract/><date/><region/>` format our preprocess step reads).

## 2.5 Preprocessing and chunking

Code: `backend/app/preprocess.py`. For each abstract:

1. **Clean**: lowercase, strip special characters (keep Devanagari/Bengali/
   Tamil Unicode ranges), collapse whitespace.
2. **Tokenize + stop-word removal** (SpaCy `en_core_web_sm`): drop "the/is/also"
   — but two deliberate exceptions:
   - **Agri keep-list**: "soil", "crop", "monsoon", "compost"… are technically
     stop-listed in generic dictionaries in some contexts; we force-keep them
     because they ARE the domain.
   - **Numbers**: `token.like_num` tokens are kept ("three varieties",
     "20 kg") — quantity words matter in farming advice.
3. **Lemmatize**: "fertilizers" → "fertilizer", so BM25 matches plural/singular.
4. **Chunk**: 200 tokens per chunk, 50-token overlap between consecutive
   chunks (stride 150). Why overlap? If a key idea straddles a boundary, one
   chunk still contains it whole. Why 200? Small enough that retrieval returns
   *focused* passages (not whole abstracts diluting the match), large enough
   to carry context.

Batching: we process 1,000 documents at a time with `nlp.pipe(batch)` and the
parser/NER components **excluded** (tokenization+lemmatization needs only the
tagger) — roughly 2× faster than full pipeline, and measurable: the test
`test_preprocess_corpus_performance_gate` requires 200 records < 10 s (it was
14.9 s before batching — a TDD-driven optimization).

Output: `data/processed/chunks.jsonl` — one JSON line per chunk:
`{text, tokens, title, year, region, source_file}`. The `tokens` field is
stored so the BM25 index can be built **without re-tokenizing everything a
second time** (that duplication would have doubled a 12-minute build).

## 2.6 Embeddings (the core representation)

Code: `backend/app/embeddings.py`.

**What an embedding is**: a neural network converts text into a list of
numbers — a point in a 384-dimensional space — such that *texts with similar
meaning get nearby points*. `all-MiniLM-L6-v2` is a small, fast
sentence-transformer: it reads a sentence, and outputs 384 floats where
"soil fertility improves with compost" and "organic manure enriches soil" land
close together even though they share almost no words.

Key details:

- **384 dimensions**: small enough to search fast, big enough to carry meaning.
- **L2 normalization**: every vector is scaled to length 1. This matters
  because with unit vectors, **inner product (dot product) = cosine
  similarity** — the standard "how aligned are these meanings" measure, ranging
  from −1 (opposite) to +1 (identical). Our embeddings are all in the same
  direction half-space so scores are typically 0.2–0.7 for related text.
- We normalize at encode time (`normalize_embeddings=True`) so the index and
  every query are consistent.

## 2.7 The FAISS index

Code: `backend/build_faiss_index.py` → `data/index/faiss.index`.

**FAISS** (Facebook AI Similarity Search) is a library for fast
nearest-neighbor search over millions of vectors.

Our index: **`IndexFlatIP`** — "flat" means brute-force comparison of the
query against *every* vector (no approximation), "IP" means inner product.

- 22,874 vectors × 384 floats ≈ 35 MB — tiny by FAISS standards.
- Brute force over 22.8k vectors = ~22.8k multiplications of 384 dims —
  **milliseconds**. Exact, deterministic, zero tuning.
- We also save `vectors.npy` (the raw matrix) so a rebuild doesn't need to
  re-run the 14-minute embedding step.

**The bug that taught us FAISS internals** (worth retelling in interviews):
we originally used `IndexHNSWFlat` — HNSW = *Hierarchical Navigable Small
World*, an approximate graph index where nodes link to neighbors and search
walks the graph. We built it with default L2 metric, then *assigned*
`index.metric_type = METRIC_INNER_PRODUCT` after construction — which
Python happily accepted but the underlying C++ graph ignored. Search returned
**L2 distances** (where smaller = closer) but our retriever sorted
**descending** as if they were similarity scores. Net effect: the retriever
returned the *worst* matches — fish-anatomy papers for a soil question.
Diagnosis: reconstructed vectors from the index, computed cosine by hand
(`query · stored_vector`), showed the true neighbors were perfect while the
index's returned ones had *negative* similarity. Fix: `IndexFlatIP` (exact),
plus a regression test asserting the metric and a relevance test asserting
on-topic results.

## 2.8 BM25 (the other retriever)

Code: same build script → `data/index/bm25.pkl` (built with `rank_bm25`).

**BM25** ("Best Matching 25") is the ranking function behind classic search
engines. For a query term `t` in document `d`:

```
score ≈ IDF(t) × (tf × (k1+1)) / (tf + k1 × (1 − b + b × |d|/avgdl))
```

- **tf** = how often the term appears in the document,
- **IDF** = how *rare* the term is across the corpus (rare terms are
  informative; "the" gets ~zero weight, "vermicompost" gets a lot),
- **k1 (1.5)** saturates term frequency (the 5th occurrence adds less than the
  1st),
- **b (0.75)** normalizes for document length (long docs don't win just by
  having more words).

It operates on the *token lists* we stored in `chunks.jsonl` — that's why
tokenization consistency between index-time and query-time matters: the same
`tokenize()` function runs on both sides.

## 2.9 Hybrid fusion (0.7 dense + 0.3 sparse)

Code: `backend/app/retrieval.py`, class `Retriever.search()`.

The two scorers produce numbers on different scales (FAISS inner products
0–1; BM25 unbounded 0–30+). Before fusing we **min-max normalize** each score
list over the candidate set, so both span 0–1, then:

```
final = 0.7 × dense_score + 0.3 × sparse_score
```

Why 0.7/0.3? Grid-searched in the dissertation over 500 validation queries —
best Recall@5 (85%, the fraction of queries where a relevant chunk appears in
the top 5). Why both at all: dense catches meaning ("soil health" ↔ "organic
matter"), sparse nails exact technical strings ("NPK 20-20-20", "drip
irrigation") that embeddings blur.

Pipeline details:

- Fetch `top_k × 5` candidates from each retriever (25 each by default),
- fuse, sort descending, then apply **filters** before returning `top_k = 5`:
  region match (if the user specified one) and year floor (temporal guardrail).

---

# Part 3 — Generation (the LLM side)

Code: `backend/app/llm.py`, `backend/app/main.py`.

## 3.1 Serving: Ollama and quantization

**Ollama** runs open models locally, exposing an HTTP API
(`http://localhost:11434/api/chat`). Under the hood it runs **GGUF**-format
models via llama.cpp — models whose weights are stored in **4-bit integers**.

**Quantization** = representing 16/32-bit floating-point weights as low-bit
integers. Memory math for a 7B-parameter model:
- 16-bit float: 7B × 2 bytes ≈ 14 GB
- 8-bit: ≈ 7 GB, 4-bit: ≈ 3.5–4.5 GB

That's the difference between "needs an A100 server" and "runs on your laptop
GPU or a Raspberry Pi". Accuracy loss from int4/int8 quantization is typically
1–3% on benchmarks — the dissertation measured <5% for 8-bit.

Our provider router (`generate()`): `settings.llm_provider` picks
`"ollama"` (default, free) or `"hf"` (Hugging Face's hosted inference — used
only to reproduce the paper's exact benchmark models). Registry keys map
benchmark names to concrete models (`mistral-7b` → `mistral:7b-instruct`
locally, `mistralai/Mistral-7B-Instruct-v0.2` remotely), with graceful
fallback when a model has no local equivalent.

## 3.2 The prompt (this is where grounding happens)

System prompt (`SYSTEM_PROMPT` in `llm.py`):

> "You are an agricultural advisor for smallholder farmers in South Asia.
> Answer ONLY from the provided context. Give actionable, region-aware advice
> on soil fertility, water optimization, and post-harvest loss reduction.
> If the context is insufficient, say what extra information is needed.
> Recommend sustainable practices aligned with SDG 12."

User message:

```
Context:
[Retrieved abstract 1]
[Retrieved abstract 2] ...

Question: <farmer's translated question>
```

Three things this buys us: (1) the model *cannot* bring in outside "knowledge"
without contradicting instructions, (2) citations are natural — the papers are
literally in the prompt, (3) "if insufficient, say so" gives the model an
honest escape hatch instead of hallucinating.

## 3.3 Sampling parameters (and why they matter)

We send: `temperature 0.7, top_p 0.9, repetition_penalty 1.2`.

- **temperature** scales the random sampling: 0 = always pick the most likely
  word (deterministic, repetitive); 1.0 = full distribution. 0.7 = mostly
  sensible with some variation.
- **top_p (nucleus sampling)**: sample only from the smallest set of words
  covering 90% of probability mass — cuts the long tail of weird tokens.
- **repetition_penalty 1.2**: penalizes tokens already used — counters the
  "loop" failure mode of small models on technical text (we saw a Bengali
  answer repeat "সঙ্গে সঙ্গে সঙ্গে…" without it).

## 3.4 Caching

`/query` computes a SHA-256 of (question, language, model, region) and caches
the answer in Redis for 24 h. Frequent questions ("best sowing time for
wheat") return in milliseconds. Failure mode is graceful: if Redis is down,
the app computes normally (cache helpers swallow connection errors).

---

# Part 4 — Translation layer

Code: `backend/app/translate.py`, `backend/app/config.py` (`SUPPORTED_LANGS`).

## 4.1 Why translate at all?

The knowledge base is English. The farmer speaks Hindi/Bengali/Tamil. Rather
than building separate per-language corpora, we translate the *question* to
English, run the whole English pipeline, and translate the *answer* back.
One retrieval index, four languages.

## 4.2 The model: NLLB-200-distilled-600M

A **seq2seq transformer**: an encoder reads the source sentence into a
meaning representation; a decoder writes the target sentence token by token,
conditioned on the encoder output and the tokens it has written so far
("teacher forcing" during training).

- **NLLB** ("No Language Left Behind", Meta) covers 200 languages with
  language codes like `eng_Latn`, `hin_Deva`, `ben_Beng`, `tam_Taml`.
- **Distilled 600M** = compressed from the 54B-parameter teacher model into
  600M parameters — runs on CPU in a few seconds per paragraph.
- We **force the target language** with `forced_bos_token_id` — the decoder's
  first generated token is pinned to the target-language code, which is how
  one model does many→many translation.

## 4.3 The mbart → NLLB story (great failure story)

Original design: mbart-large-50-many-to-many-mmt. Testing with real questions
exposed that mbart's **Bengali→English** was catastrophically wrong:
"ধানের ফসল তোলার পরে ক্ষতি কমানোর উপায় কী?" (how to reduce post-harvest
losses of paddy) → *"What can we do to reduce the government's revenue after
the headquarters collapse?"*. Retrieval then matched irrelevant papers and the
LLM hallucinated. NLLB translates the same sentence correctly. Regression
test added: the Bengali test asserts the English output contains
rice/paddy/harvest/loss vocabulary.

Also learned: mbart's repo lacks a `tokenizer.json`, so "fast tokenizer"
conversion fails across transformers versions — we use the **slow
SentencePiece tokenizer** with manual `model.generate()` instead of the
high-level `pipeline()` (which transformers v5 removed entirely).

## 4.4 Glossary de-code-mixing

Code: `backend/app/glossary.py`.

Even with good translation, back-translated answers kept English domain terms
("Crop Rotation", "RAINFED AREAS", "mulching"). The glossary runs *after*
back-translation: 22 agricultural term patterns × 3 languages, applied
longest-phrase-first (so "drip irrigation" replaces before bare
"irrigation" — order matters, tested), case-insensitive, with inflected forms
folded into the pattern (`compost(?:ing|ed|s)?`). English answers are skipped
(no-op). TDD: 10 tests including a boundary case ("composter" must not be
mangled by "compost").

---

# Part 5 — Voice input

Code: `backend/app/transcribe.py`, endpoint in `backend/app/main.py`,
frontend `frontend/src/App.jsx`.

## 5.1 Whisper, briefly

OpenAI's open speech recognition model. Architecture: an **encoder** processes
a log-mel spectrogram (audio converted to an image-like time×frequency
representation), a **decoder** emits text tokens — and it was trained on 680k
hours with *multilingual task tokens*, so one model transcribes English,
Hindi, Bengali, Tamil and more, auto-detecting the language.

## 5.2 faster-whisper (CTranslate2)

OpenAI's reference implementation is PyTorch — heavy and slow on CPU.
**CTranslate2** re-implements transformer inference with int8 quantization and
optimized kernels: ~4× faster on CPU at the same accuracy, far lower memory.
Our `WhisperModel("small", device="cpu", compute_type="int8")`:

- **small** = 244M parameters (~460 MB download, one-time), materially better
  on hi/bn/ta than the 74M "base" model,
- **cpu/int8** default — the CUDA path needs cuBLAS/cuDNN DLLs installed;
  we learned this when a live call failed with "Library cublas64_12.dll is
  not found" and made device/compute config options instead,
- **lazy load** (`@lru_cache`): the 460 MB model loads on first use, not at
  server startup,
- **VAD filter** (voice activity detection) trims silence — also counters
  Whisper's known habit of hallucinating text during silence.

## 5.3 The browser side

`navigator.mediaDevices.getUserMedia({audio:true})` → browser records with
**MediaRecorder** → produces a **webm/Opus** blob → `POST /api/transcribe`
(multipart) → backend decodes (via PyAV, which bundles FFmpeg codecs — this is
why the `av` library version pin mattered: v19 removed an API
faster-whisper's decoder calls), runs Whisper, returns
`{text, language, duration}`.

Rules we specified in the OpenSpec spec and implemented:
- ≤ 60 s recordings rejected with a clear message (checked on WAV duration at
  upload; size limit as backstop for compressed formats),
- empty/undecodable → HTTP 400 with a readable message,
- **transcript goes into the text box for review — never auto-submitted**,
- recording states: idle → recording (red) → transcribing (spinner) →
  back to idle, with permission-denied and failure messages that keep the
  typing flow usable.

---

# Part 6 — Guardrails and ethics

Code: `backend/app/guardrails.py`.

- **Temporal filter**: retrieval prefers sources within the last 5 years of
  the corpus's latest year (outdated advice is a documented failure mode —
  11% of errors in the dissertation's analysis).
- **Niche/region filter**: chunks whose text contradicts the query context
  (e.g. "winter crops … tropical") get flagged; flagged items are surfaced to
  the user, not silently dropped (transparency over silent censorship).
- **Output bias scan**: regexes check the *generated answer* for
  "increase pesticide/herbicide", gender-stereotyped framing; flags appear in
  the API response and the UI.
- **Human oversight**: sources are always shown; the farmer can read the
  evidence themselves. (Full safety would need agronomist-in-the-loop review —
  documented as future work, not claimed as done.)

---

# Part 7 — Backend engineering

Code: `backend/app/main.py`, `backend/app/config.py`.

- **FastAPI**: async-capable Python web framework; `pydantic` models
  (`QueryRequest`, `TranscribeResponse`) validate every request body — wrong
  fields never reach business logic.
- **Lifespan hook**: on startup the app checks the FAISS index exists
  (`index_ready`) and **auto-starts Ollama if it isn't running**
  (`app/ollama.py`: probe `:11434/api/version`, spawn `ollama serve` detached
  if down, poll up to 45 s) — verified by killing Ollama and confirming the
  API resurrected it.
- **CORS**: the React dev server (port 5173) is allowed to call the API
  (port 8000) — browsers block cross-origin requests without it.
- **Paths**: relative data paths are resolved against the *project root*, not
  the current working directory (a bug when uvicorn started from `backend/`;
  caught by `test_config.py`).

---

# Part 8 — Frontend

`frontend/src/App.jsx` (React + Vite).

- One page: textarea + mic button, language select (English/हिन्दी/বাংলা/தமிழ்),
  model select (the four benchmark models), optional region field.
- The mic button carries the state machine: idle → recording (MediaRecorder
  capturing) → transcribing (POST in flight) → idle, with error text for
  permission denial and transcription failure.
- Answers render with a **Sources** section (title, year, retrieval score,
  guardrail flags) — the citation UI is the trust UI.
- Dev mode: Vite dev-server proxies `/api/*` → `localhost:8000`.
  Production (Docker): **nginx** serves the built assets and proxies `/api/`
  to the `api` container — forgetting this was our "404 in Docker" bug.

---

# Part 9 — Testing discipline

- **TDD loop**: failing test (RED) → minimal implementation (GREEN) →
  refactor. Every bug in Part 2.7, 4.3, 5.2 was reproduced by a failing test
  *before* the fix.
- **45+ tests** across `backend/tests/`:
  - unit: tokenizer/chunking math, glossary ordering, config path resolution,
    guardrail patterns,
  - integration (marked `slow`): real Whisper transcribing a spoken sample
    (generated with Windows SAPI TTS), real NLLB Hindi/Bengali translation,
    real Ollama generation, real retrieval relevance,
  - performance gate: 200 records must preprocess in < 10 s.
- Why keep both kinds: unit tests run in seconds on every change; slow
  integration tests prove the *real* stack works, not just the mocks.

---

# Part 10 — Evaluation (know the metrics cold)

- **BLEU-4**: n-gram (1,2,3,4-gram) *precision* of the generated answer vs
  reference(s), clipped (each n-gram counted no more times than it appears in
  the reference), geometric mean, times a **brevity penalty** (answers shorter
  than the reference get punished). With ONE reference and a long, freely
  worded answer, BLEU collapses toward 0 even for a good answer — our local
  result (0.000) is exactly this artifact.
- **ROUGE-L**: longest common subsequence between generated and reference —
  order-aware overlap, length-recall oriented. Our 0.065–0.094 says "some
  overlap, mostly different wording".
- **Exact Match**: literal string equality — 0 unless the model parrots the
  reference.
- **Recall@5**: of the queries where a relevant document exists, in what
  fraction does it appear in the top-5 retrieved. The dissertation reports
  85% for the hybrid retriever.
- **BERTScore** (the upgrade path we recommend): embeds generated and
  reference sentences and computes cosine similarity per token pair —
  *semantic* matching that tolerates paraphrase; the right fix for our
  single-reference setup.

---

# Part 11 — Deployment & operations

- **Local (no Docker)**: `.\start.ps1` → two windows: uvicorn (auto-starts
  Ollama) + Vite dev server.
- **Docker**: `docker compose up` — three containers: `api` (Python image,
  volume-mounted `data/`), `frontend` (nginx serving the built React bundle
  and proxying `/api`), `redis` (cache). Container-specific traps we fixed:
  relative paths differ (pinned absolute paths in compose), `localhost`
  inside a container is the container (Ollama reached via
  `host.docker.internal`), nginx had no API proxy.
- **GitHub**: public repo, history scrubbed of personal/copyright documents
  (`git-filter-repo`), sources documented in `docs/SOURCES.md` instead.

---

# Part 12 — Rapid-fire glossary

| Term | One-line meaning |
|---|---|
| RAG | retrieve documents at question time and ground the LLM's answer in them |
| Embedding | text → vector of numbers; similar meaning = nearby vector |
| Cosine similarity | angle between two vectors; 1 = same meaning direction |
| FAISS | library for fast nearest-neighbor vector search |
| IndexFlatIP | brute-force exact inner-product search (inner product = cosine on unit vectors) |
| HNSW | approximate graph-based ANN index (fast at millions of vectors) |
| BM25 | keyword scoring with TF saturation, IDF, length normalization |
| Hybrid retrieval | blend dense + sparse scores (ours: 0.7/0.3) |
| Chunking | splitting documents into overlapping passages for precise retrieval |
| all-MiniLM-L6-v2 | small sentence-embedding model, 384 dims |
| Quantization | storing weights as int8/int4 → 2–4× less memory, ~1–3% accuracy cost |
| GGUF / Ollama | quantized model file format / local model server |
| NLLB-200 | Meta's 200-language translation model (seq2seq, 600M distilled) |
| mbart-50 | 50-language translation model; weak Bengali→English (why we switched) |
| Whisper / faster-whisper | speech-to-text model / its 4×-faster CTranslate2 reimplementation |
| VAD | voice-activity detection — trims silence before transcription |
| BLEU / ROUGE-L / EM | n-gram precision / longest-common-subsequence / literal match metrics |
| Recall@5 | fraction of queries whose relevant doc appears in top-5 results |
| BERTScore | semantic metric via embedding similarity (paraphrase-tolerant) |
| TDD | write the failing test first, then the code |
| OpenSpec | spec-driven workflow: propose → spec → design → tasks → apply → archive |

---

# Part 13 — Ten questions you should be ready to answer

1. **Why not just ask ChatGPT?** No sources, generic answers, cost, privacy,
   needs internet.
2. **How exactly does retrieval "know" what's relevant?** Embedding cosine
   similarity for meaning + BM25 keyword scores for precision, fused 0.7/0.3.
3. **Why 0.7/0.3?** Grid search on validation queries (dissertation); the
   ratio is a tunable, not a law.
4. **What happens when nothing relevant is retrieved?** The prompt instructs
   the model to say what's missing instead of inventing an answer.
5. **How do you prevent hallucination?** Strict context-only prompt +
   citations shown + guardrail scans; remaining risk is documented, not denied.
6. **Why is your measured BLEU zero when the paper says 0.78?** Different test
   sets and metric conditions; single-reference BLEU punishes verbose answers;
   we separated literature values from measured values in the README.
7. **How would you make evaluation proper?** 20–50 curated query/reference
   pairs + BERTScore, then scale; or reproduce the paper's harness on HF.
8. **How does voice work end to end?** MediaRecorder (webm/opus) → multipart
   upload → PyAV decode → faster-whisper int8 CPU → text + language → into the
   text box for review.
9. **What broke in production and how did you find it?** Three stories:
   inverted FAISS metric (proved with hand-computed cosine), Bengali
   mistranslation (found testing in Bengali), Docker networking (localhost ≠
   host).
10. **What would you do with real users?** Agronomist-in-the-loop review,
    field trials with farmers, feedback capture loop into the corpus, TTS for
    voice output, and region-aware personalization.
