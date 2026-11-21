"""Parse AGRIS XML exports from data/raw/, preprocess, and write chunks.jsonl.

AGRIS exports are XML; each record has title, abstract, year, region/country
metadata (per the dissertation §4.2: xml.etree.ElementTree parsing).

Usage:  python preprocess.py   (run from backend/ after placing exports in ../data/raw/)
"""
import json
import sys
import xml.etree.ElementTree as ET
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from app.config import settings
from app.preprocess import preprocess_document

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
            el.itertext() for el in rec.iter() if el.tag.lower().endswith(tag)
        ).strip()
        title, abstract, year = text("title"), text("abstract"), text("date")
        if not (title and abstract):
            continue
        region = next((h for h in REGION_HINTS
                       if h in (title + " " + abstract).lower()), None)
        if not region:
            continue  # keep only regionally relevant (South Asia) docs
        records.append({"title": title, "abstract": abstract,
                        "year": year[:4] if year else None, "region": region,
                        "source_file": path.name})
    return records


def main():
    raw, out = settings.data_raw_dir, settings.data_processed_dir
    out.mkdir(parents=True, exist_ok=True)
    dest = out / "chunks.jsonl"
    n_docs = n_chunks = 0
    with open(dest, "w", encoding="utf-8") as f:
        for path in sorted(raw.glob("*.xml")):
            docs = parse_agris_xml(path)
            print(f"{path.name}: {len(docs)} regionally relevant records")
            for d in docs:
                for chunk in preprocess_document(d["abstract"]):
                    f.write(json.dumps(
                        {"text": chunk, "title": d["title"], "year": d["year"],
                         "region": d["region"], "source_file": d["source_file"]},
                        ensure_ascii=False) + "\n")
                n_docs += 1
                n_chunks += len(
                    preprocess_document(d["abstract"])) if d else 0
    print(f"\nDone: {n_docs} documents → {n_chunks} chunks → {dest}")


if __name__ == "__main__":
    main()
