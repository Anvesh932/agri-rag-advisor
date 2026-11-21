# Multilingual Quality Backlog

Findings from testing real farming questions in Hindi (hi), Bengali (bn), and
Tamil (ta) through the full pipeline (translate → FAISS+BM25 retrieval →
llama3.2:3b → translate back), 2026-11-21. Raw request/response pairs live in
`docs/quality/q1…q6.{txt,json}`.

## Fixed during this pass

1. **Bengali retrieval poisoned by mbart (CRITICAL — FIXED)**
   mbart-large-50 translated "ধানের ফসল তোলার পরে ক্ষতি কমানোর উপায় কী?" (how to
   reduce post-harvest losses of paddy) as *"reduce the government's revenue
   after the headquarters collapse"*. Retrieval matched irrelevant papers and
   the LLM answered incoherently. **Fix:** switched the translator to
   NLLB-200-distilled-600M (same free/local cost). Bengali and all other
   languages now retrieve relevant sources and answer coherently.

## Open items (priority order)

2. **Code-mixing in answers (MEDIUM)** — back-translation leaves English
   technical terms untranslated inside Hindi/Tamil/Bengali answers
   ("RAINFED AREAS", "Crop Rotation", "Organic Manure"). Low-literacy users may
   not parse them. *Idea:* prompt the LLM to write plain-language terms, or
   post-process a small domain glossary (compost→कम्पोस्ट/মালচ etc.).

3. **Study-title back-translation noise (LOW-MEDIUM)** — the model cites paper
   titles verbatim inside the answer and NLLB mangles them ("GRAIN YILD",
   "सब्प्रांत" for sub-tropical). *Idea:* instruct the model to cite as
   "(Study, 2022)" without translating titles, or render citations in English.

4. **Terminology drift in back-translation (MEDIUM)** — "organic manure" came
   back as "कृषि उर्वरक" (fertilizer) in Hindi Q2 — reverses the user's core
   constraint (no chemical fertilizer). A domain glossary for the answer
   back-translation would prevent meaning-flips.

5. **Hindi wheat-sowing answer hedged instead of answering (LOW)** — Q1 gave a
   literature review of two studies rather than one crisp recommendation
   (Nov sowing after monsoon). *Idea:* sharpen the system prompt ("give ONE
   actionable recommendation first, then evidence").

6. **Latency of multilingual queries (LOW, accepted for now)** — 40–160 s per
   non-English query on CPU: NLLB runs on CPU and long answers dominate
   back-translation time. *Idea:* run NLLB on CUDA when the toolkit DLLs are
   installed, or cache translations.

7. **Bengali diacritic rendering (COSMETIC)** — some NLLB Bengali output
   carries zero-width joiner artifacts ("প ্ র"), likely font/normalization;
   verify on a real Bengali-capable display before fixing.
