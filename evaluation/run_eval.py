"""Evaluation harness: BLEU-4, ROUGE-L, Exact Match over a query set.

Usage:
  python evaluation/evaluate.py --queries evaluation/sample_queries.json \
      [--model mistral-7b] [--api http://localhost:8000]

Scores generated answers against reference answers, reproducing the
dissertation's benchmark table format.
"""
import argparse
import json
import urllib.request
from pathlib import Path

SUPPORTED = {"bleu": None, "rouge": None}


def load_metrics():
    try:
        from evaluate import load
        bleu = load("bleu")
        rouge = load("rouge")
        return bleu, rouge
    except ImportError:
        import sacrebleu, rouge_score  # noqa
        raise


def ask(api: str, q: dict, model: str) -> str:
    body = json.dumps({"question": q["question"], "lang": "en", "model": model}).encode()
    req = urllib.request.Request(
        f"{api.rstrip('/')}/query", data=body,
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as r:
        return json.load(r)["answer"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", default=str(Path(__file__).parent / "sample_queries.json"))
    ap.add_argument("--api", default="http://localhost:8000")
    ap.add_argument("--model", default="mistral-7b")
    args = ap.parse_args()

    queries = json.loads(Path(args.queries).read_text(encoding="utf-8"))
    bleu, rouge = load_metrics()

    preds, refs = [], []
    for q in queries:
        print(f"Q: {q['question'][:70]}…")
        preds.append(ask(args.api, q, args.model))
        refs.append(q["reference"])

    bleu_res = bleu.compute(predictions=preds, references=[[r] for r in refs])
    rouge_res = rouge.compute(predictions=preds, references=refs, use_stemmer=True)
    em = sum(p.strip().lower() == r.strip().lower()
             for p, r in zip(preds, refs)) / len(refs)

    print(json.dumps({
        "model": args.model,
        "n_queries": len(refs),
        "BLEU-4": round(bleu_res["bleu"], 3),
        "ROUGE-L": round(rouge_res["rougeL"], 3),
        "Exact Match": round(em, 3),
    }, indent=2))


if __name__ == "__main__":
    main()
