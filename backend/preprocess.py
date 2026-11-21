"""Parse AGRIS XML exports from data/raw/, preprocess, and write chunks.jsonl.

AGRIS exports are XML; each record has title, abstract, year, region/country
metadata (per the dissertation §4.2: xml.etree.ElementTree parsing).

Usage:  python preprocess.py   (run from backend/ after placing exports in ../data/raw/)
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows console (cp1252) → UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings
from app.preprocess import chunk_text, tokenize_batch

REGION_HINTS = ["south asia", "india", "bangladesh", "nepal", "pakistan",
                "sri lanka", "bhutan", "indo-gangetic"]


def parse_agris_xml(path: Path) -> list[dict]:
    records = []
    try:
        root = ET.parse(path).getroot()
    except ET.ParseError as e:
        print(f"  ! skipping {path.name}: XML parse error {e}")
        return records
    for rec in root.iter():
        if not rec.tag.lower().endswith("record"):
            continue
        text = lambda tag: "".join(
            "".join(el.itertext()) for el in rec.iter() if el.tag.lower().endswith(tag)
        ).strip()
        title, abstract, year = text("title"), text("abstract"), text("date")
        if not (title and abstract):
            continue
        # region: use the <region> element written by fetch_agris.py when present,
        # otherwise fall back to keyword matching in title/abstract
        region_el = next((el for el in rec.iter() if el.tag.lower().endswith("region")), None)
        region = (region_el.text.strip() if region_el is not None and region_el.text else None)
        if not region:
            region = next((h for h in REGION_HINTS
                           if h in (title + " " + abstract).lower()), None)
        if not region:
            continue
        records.append({"title": title, "abstract": abstract,
                        "year": year[:4] if year else None, "region": region,
                        "source_file": path.name})
    return records


def process_corpus(docs: list[dict], batch_size: int = 128) -> list[dict]:
    """Tokenize (batched) and chunk a list of AGRIS records.

    Returns chunk dicts: {text, tokens, title, year, region, source_file}.
    `tokens` are stored so build_faiss_index.py can build the BM25 index
    without re-tokenizing the whole corpus.
    """
    chunks: list[dict] = []
    for i in range(0, len(docs), batch_size):
        batch = docs[i:i + batch_size]
        token_lists = tokenize_batch([d["abstract"] for d in batch],
                                     batch_size=batch_size)
        for d, tokens in zip(batch, token_lists):
            for chunk in chunk_text(d["abstract"], tokens=tokens):
                chunks.append({"text": chunk, "tokens": tokens,
                               "title": d["title"], "year": d["year"],
                               "region": d["region"],
                               "source_file": d["source_file"]})
    return chunks


def main():
    raw, out = settings.data_raw_dir, settings.data_processed_dir
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "chunks.jsonl"
    files = sorted(raw.glob("*.xml"))

    # parse all regionally relevant records first
    docs: list[dict] = []
    for path in files:
        recs = parse_agris_xml(path)
        print(f"{path.name}: {len(recs)} regionally relevant records")
        docs.extend(recs)
    print(f"\nTotal: {len(docs)} documents; tokenizing in batches …")

    # batched tokenization (fast), then chunk and write
    n_chunks = 0
    with open(dest, "w", encoding="utf-8") as f:
        for i in range(0, len(docs), 1000):
            batch_chunks = process_corpus(docs[i:i + 1000])
            for c in batch_chunks:
                f.write(json.dumps(c, ensure_ascii=False) + "\n")
                n_chunks += 1
            print(f"  {min(i + 1000, len(docs))}/{len(docs)} documents done")

    print(f"\nDone: {len(docs)} documents → {n_chunks} chunks → {dest}")


if __name__ == "__main__":
    main()
