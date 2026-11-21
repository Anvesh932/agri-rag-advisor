# Local Evaluation Results (2026-11-21)

Run against the local Ollama models on the 3 sample queries in
`evaluation/sample_queries.json` (single reference answers from the
dissertation case study). Command:

```bash
backend/.venv/Scripts/python.exe evaluation/run_eval.py \
  --queries evaluation/sample_queries.json --model llama-3.2-3b
```

| Model              | BLEU-4 | ROUGE-L | Exact Match |
|--------------------|--------|---------|-------------|
| llama-3.2-3b       | 0.000  | 0.094   | 0.000       |
| mistral:7b-instruct| 0.000  | 0.065   | 0.000       |

## Reading these numbers

Single-reference BLEU with verbose generated answers is brutally penalized
(brevity penalty + no n-gram overlap with a one-sentence reference), so near-zero
BLEU does **not** mean the answers are useless — it means they are much longer
and differently worded than the reference. The dissertation's benchmark table
(0.72–0.78 BLEU) used a 1,000-query test set with tuned reference answers and
the unquantized models; reproducing those exact numbers requires the HF
benchmark path (`LLM_PROVIDER=hf`).

What the numbers still tell us locally:

- mistral scored *lower* ROUGE-L than llama here — consistent with it producing
  even longer answers on the same 3 queries.
- For meaningful local model comparison, grow the query set
  (`sample_queries.json`) with reference-length answers, and/or add semantic
  similarity (BERTScore/embedding cosine) which tolerates paraphrase.

## Suggested next evaluation steps

1. Expand the test set to 20–50 query/reference pairs drawn from the
   dissertation case studies.
2. Add answer-length normalization (or use BERTScore).
3. Re-run the 4-model benchmark on HF when exact paper numbers are needed.
