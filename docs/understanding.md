# Interview Prep — Agri-RAG Advisor, Explained Simply

Everything this project does, in plain language, with the "why" behind every
decision — written so you can explain it and defend it in an interview.

---

## 1. The one-sentence pitch

"We built an app where a smallholder farmer in South Asia can ask a farming
question — by typing or **speaking**, in Hindi, Bengali, or Tamil — and get an
answer grounded in real scientific papers, with the sources cited, running
entirely on cheap hardware for free."

## 2. The problem

- Most AI chatbots give **generic** farming advice ("use fertilizers", "irrigate
  properly") that ignores the farmer's region, soil, and climate.
- Farmers can't read long English reports, and many struggle with typing at all
  (45% of test users in the dissertation's field trials).
- Cloud AI costs money per query and needs internet — not realistic for rural,
  low-resource settings.

So the goal: **region-specific + local-language + local-hardware + zero-cost**.

## 3. The core idea: RAG (Retrieval-Augmented Generation)

A language model alone makes things up ("hallucinates"). RAG fixes this:

1. Farmer asks a question.
2. We **retrieve** the most relevant paragraphs from a trusted knowledge base
   (scientific agricultural papers).
3. We paste those paragraphs into the model's prompt and say: "Answer ONLY
   from this context."
4. The model answers using real evidence, and we show which papers were used.

Think of it as an "open-book exam" instead of a closed-book one.

## 4. How the knowledge base was built

1. **AGROVOC** (FAO's agricultural thesaurus): we started from 55 seed terms
   like "fertilizer", "irrigation".
2. Expanded them to ~515 related concepts using embedding similarity (word
   neighbors like "biofertilizer" → "vermicompost").
3. **AGRIS** (FAO's database of agricultural research): downloaded the Open
   Data Set — per-country XML files (~100 MB each).
4. Kept only records from South Asia (India, Nepal, Pakistan, Bangladesh, Sri
   Lanka, Bhutan), from **2015–2024** (recent = trustworthy), containing an
   abstract and at least one topic keyword (soil, water, harvest...).
5. Result: **19,562 documents → 22,874 text chunks**.

## 5. Text preprocessing (why not raw text?)

Raw abstracts are messy. We:
- lowercase and strip special characters,
- remove **stop words** ("the", "is") but KEEP agricultural terms and numbers
  ("soil", "three varieties" — numbers matter in farming),
- **lemmatize** ("fertilizers" → "fertilizer") so different word forms match,
- split text into **200-token chunks with 50-token overlap** so no idea is cut
  in half at a chunk boundary.

## 6. Embeddings and the two retrievers (the heart of it)

**Embeddings**: each chunk is converted into a list of 384 numbers (a vector)
using the `all-MiniLM-L6-v2` sentence-transformer model. Similar meanings end
up as nearby vectors — "soil health" lands near "organic matter".

We search with TWO methods in parallel and blend the results:

| Method | What it's good at | How it works |
|---|---|---|
| **FAISS (dense)** | meaning, synonyms | nearest-neighbor search over vectors |
| **BM25 (sparse)** | exact keywords | classic keyword scoring (like search engines) |

**Why both?** FAISS finds "soil fertility" ↔ "organic manure" (semantics),
BM25 nails precise terms like "drip irrigation" (keywords). We fuse scores
**0.7 × dense + 0.3 × sparse** (weights from the dissertation's grid search).

**FAISS detail you may be quizzed on:** our index is `IndexFlatIP` — *exact*
inner-product search on normalized vectors (inner product of normalized
vectors = cosine similarity, higher = more similar). At 22,874 × 384 floats
(~35 MB) exact search takes milliseconds, so we don't need approximate ANN.
(We originally used HNSW, an approximate graph index, and hit a subtle bug —
see section 11.)

## 7. The language model (generation)

- Models run **locally via Ollama** — quantized (compressed) versions of
  Llama-3.2-3B and Mistral-7B-Instruct fit on a 6 GB laptop GPU or CPU.
- Quantization = storing model weights in 8-bit/4-bit integers instead of
  16/32-bit floats → 2–4× less memory, tiny accuracy loss. This is the same
  trick that lets the system run on a Raspberry Pi-class device.
- The prompt is strict: *"Answer ONLY from the provided context, be actionable
  and region-aware; if context is insufficient, say so."* This is what makes
  answers grounded instead of generic.
- HF (Hugging Face router) remains available as an alternative provider for
  exact benchmark reproduction, but the default is free/local.

## 8. Language translation (the multilingual layer)

- Farmer asks in Hindi/Bengali/Tamil → we translate the question to **English**
  (the knowledge base language) → retrieve → generate → translate the answer
  **back**.
- Original design used **mbart-large-50**. Field-testing found Bengali was
  **broken**: "how to reduce post-harvest losses of paddy" came back as
  "reduce the government's revenue after the headquarters collapse" → the
  retrieval got garbage → the answer was garbage.
- **Fix: switched to NLLB-200-distilled-600M** (Meta's 200-language model),
  which translates Bengali correctly. Lesson: *always test in the target
  language; translation quality is not uniform across languages.*

## 9. Voice input (accessibility)

- Farmer presses a mic button and speaks; **Whisper** (OpenAI's open speech
  model, run locally via `faster-whisper`/CTranslate2) transcribes it.
- Auto-detects the language (hi/bn/ta/en), caps recordings at 60 s.
- The transcript is placed in the text box **for review before sending** — the
  farmer stays in control.
- Two hardening lessons: `faster-whisper` needs a compatible `av` library
  version, and CUDA paths need cuBLAS/cuDNN DLLs — we default to CPU (int8),
  which is fast enough for short clips.

## 10. Guardrails (ethics layer)

- **Temporal filter**: prefer studies from the last 5 years (old advice can be
  harmful).
- **Niche filters**: flag answers that conflict with the query's region
  (e.g., winter crops in a tropical zone).
- **Bias scan**: regex checks for harmful patterns (recommend *increasing*
  pesticide use; gender-stereotyped framing). Flagged items are shown as
  warnings.
- **Redis cache**: frequent questions ("best sowing time for wheat") are
  cached, cutting latency; silently skips if Redis isn't running.

## 11. Bugs we actually found and fixed (great interview stories)

1. **Inverted retrieval ranking** — the FAISS HNSW index was built with L2
   distance, but our code sorted scores descending as if they were inner
   products (higher = better). L2 is *lower = closer*, so the retriever returned
   the **worst** matches (fish-anatomy papers for a soil question). Proved it
   with a test, switched to exact `IndexFlatIP` (cosine via inner product).
2. **mbart Bengali mistranslation** (section 8) — fixed with NLLB-200.
3. **Version conflicts in the ML stack** — transformers v5 removed the
   `translation` pipeline; library `av` v19 removed an API faster-whisper needs;
   `datasets`/`huggingface_hub` version matrix conflicts. Lesson: ML dependency
   stacks are fragile; pin versions and let tests catch drift.
4. **Windows cp1252 console** crashed on Unicode characters in logs — fixed by
   forcing UTF-8 stdout.
5. **Docker-specific bugs** — nginx frontend had no `/api` proxy (404s), data
   paths resolved differently inside containers, `localhost` in a container is
   not the host (needed `host.docker.internal` to reach Ollama).
6. **Path resolution** — relative paths resolved against the wrong directory
   depending on where uvicorn was started; made paths resolve against the
   project root.

## 12. How we worked (process — interviewers love this)

- **TDD (test-driven development)**: for every feature and bug, we wrote a
  *failing test first* (red), then the minimal code to pass (green), then
  refactored. Final count: **45+ tests** (unit + integration, fast/slow marked).
  Every bug in section 11 was caught or verified by a test.
- **OpenSpec workflow**: features go through *propose → spec → design → tasks
  → apply → archive*. The voice-input feature has a spec, design doc, and task
  list that were implemented and then archived into a permanent capability spec.
- **Agent skills**: engineering practices (code review, TDD, spec-driven dev)
  encoded as reusable skill files the AI agent follows.
- **Quality backlog**: multilingual field-testing produced a written backlog
  (Bengali fix shipped; glossary for code-mixing shipped).
- **Everything is committed to GitHub** with descriptive commits.

## 13. Architecture in one picture

```
Farmer (browser, hi/bn/ta/en, voice or text)
   │  React UI (Vite; nginx in Docker proxies /api)
   ▼
FastAPI backend (/query, /transcribe, /health)
   │  1. Whisper transcribes voice → text          (local, faster-whisper)
   │  2. NLLB-200 translates hi/bn/ta → English     (local)
   │  3. Hybrid retrieval: 0.7·FAISS + 0.3·BM25     (22.8k AGRIS chunks)
   │  4. Guardrails: temporal + niche + bias filters
   │  5. LLM generates grounded answer              (Ollama, quantized, local)
   │  6. NLLB translates answer back + glossary de-code-mixing
   │  7. Redis cache for frequent questions
   ▼
Farmer sees: answer in their language + cited sources + guardrail flags
```

## 14. Metrics — read this before any interview

- The **dissertation's table** (BLEU-4 0.78 etc.) is literature-reported, from
  a 1,000-query test set with tuned references and unquantized models. **This
  repo did not reproduce those numbers.**
- What we **actually measured locally** (3 sample queries): BLEU-4 = 0,
  ROUGE-L ≈ 0.09. That is expected, not alarming: single-reference BLEU is
  brutally harsh on long, freely-worded answers. It is a *metric limitation*,
  not proof the answers are bad.
- Honest answer if asked: "The paper's numbers come from a much larger,
  purpose-built test set; my local harness uses 3 reference answers, so the
  BLEU comparison is apples-to-oranges. The right next step is a proper test
  set or semantic metrics like BERTScore."

## 15. Likely interview questions & crisp answers

**Q: Why RAG instead of fine-tuning the LLM?**
RAG grounds answers in citable, updatable sources without retraining; fine-
tuning can't cite sources and goes stale when literature updates.

**Q: Why hybrid retrieval instead of just FAISS?**
Semantics alone miss exact technical terms; keywords alone miss paraphrases.
0.7/0.3 fusion got the best Recall@5 in the dissertation's grid search.

**Q: Why local models? Why quantization?**
Rural deployment = no reliable internet + zero budget. Quantization (int8/int4)
cuts memory 2–4× so a 7B model runs on a 6 GB GPU or a Raspberry Pi-class
device, with minor accuracy loss.

**Q: How do you know retrieval works?**
Relevance smoke tests on real queries (top hits are on-topic), Recall@5 target
85% from the dissertation methodology, and unit + integration tests around the
retriever (including the metric bug regression test).

**Q: How do you handle hallucination?**
Strict system prompt ("answer only from context"), retrieval-grounded context,
guardrail scans, and sources displayed to the user so they can verify.

**Q: What about wrong/dangerous advice to a farmer?**
Temporal filters (recent science only), niche filters (region mismatch flags),
bias scan, sources shown, and the spec explicitly keeps the human in control
(transcript review, no auto-send). Full safety would need expert-in-the-loop
review and field trials — documented as future work.

**Q: Why FastAPI?** Async-friendly, automatic OpenAPI docs, pydantic
validation, Python-native to the ML stack.

**Q: Why 200-token chunks with 50 overlap?** Chunks must be small enough to
retrieve *precisely* but overlap prevents cutting an idea at a boundary.

**Q: What is cosine similarity / why inner product?** Cosine measures the angle
between two vectors (1 = same direction). If vectors are L2-normalized, cosine
= inner product, which FAISS computes extremely fast.

**Q: What would you improve next?** (from the backlog) semantic metrics for
evaluation, a proper 50-query test set, glossary coverage, TTS voice output,
and expert-in-the-loop field trials.

**Q: Biggest lesson?** Test in the real conditions: the Bengali failure, the
retrieval inversion, and the Docker-only bugs were all invisible until we ran
the *actual* stack end-to-end in the *actual* languages.
