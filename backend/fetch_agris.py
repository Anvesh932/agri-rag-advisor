"""Fetch AGRIS data for the knowledge base.

Source: FAO AGRIS Open Data Set (ODS), CC BY 4.0 — https://www.fao.org/agris/agris-ods
The ODS is split into per-data-provider XML files listed in the catalog:
https://agris.fao.org/ods/AGRIS.ODS.xml

Steps:
  1. download the ODS catalog and map provider subset codes → provider names
  2. select South Asia providers (India, Nepal, Pakistan, Bangladesh,
     Sri Lanka, Bhutan + 'South Asia' in the name)
  3. stream each subset XML (~100 MB each; parsed incrementally, not loaded whole)
  4. keep records with: title + abstract, year 2015–2024, and at least one
     topic keyword from the three advisory domains (soil fertility, water
     optimization, post-harvest loss) — mirroring the dissertation filter
     (81,000 abstracts → 3,500 regionally relevant documents)
  5. write data/raw/agris_filtered.xml in the normalized <record> format
     expected by preprocess.py

Usage:  python fetch_agris.py [--all-south-asia] [--limit-per-file N]
"""
import argparse
import re
import sys
import urllib.request
import xml.etree.ElementTree as ET
from pathlib import Path

if hasattr(sys.stdout, "reconfigure"):  # Windows console (cp1252) → UTF-8
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

CATALOG_URL = "https://agris.fao.org/ods/AGRIS.ODS.xml"
ODS_BASE = "https://agris.fao.org/ods/"

# stdlib-only so the fetch works before the venv/deps are installed
BASE_DIR = Path(__file__).resolve().parent
RAW_DIR = BASE_DIR.parent / "data" / "raw"

NS = {
    "dcat": "http://www.w3.org/ns/dcat#",
    "dc": "http://purl.org/dc/elements/1.1/",
    "dct": "http://purl.org/dc/terms/",
    "dctypes": "http://purl.org/dcmitype/",
}

SOUTH_ASIA_NAME_KEYWORDS = ["india", "nepal", "pakistan", "bangladesh",
                            "sri lanka", "bhutan", "south asia"]
# Subset code prefixes used by AGRIS ODS for South Asian providers.
# (IN/IM = India, NP = Nepal, PK = Pakistan, BD = Bangladesh, LK = Sri Lanka,
#  BH = Bhutan)
SOUTH_ASIA_CODE_PREFIXES = ("IN", "IM", "NP", "PK", "BD", "LK", "BH")
# Names that only *contain* a keyword but are not South Asian
EXCLUDE_NAME = ["ocean", "caribbean", "africa", "indian ocean"]

# Advisory-domain topic filter (title/abstract must mention at least one).
# Stems on purpose: 'fertil' matches fertilizer/fertiliser/fertilization, etc.
TOPIC_KEYWORDS = [
    "soil", "fertil", "compost", "manure", "nutrient", "mulch",
    "irrigat", "water", "drip", "rainwater", "drainage", "drought",
    "harvest", "storage loss", "food loss", "grain storage", "post-harvest",
]

YEAR_MIN, YEAR_MAX = 2015, 2024

ODS_DIR = RAW_DIR / "ods"
FILTERED_OUT = RAW_DIR / "agris_filtered.xml"

# ODS record: dctypes:BibliographicResource with dc:title / dc:description / dc:date
REC_TAG = "{http://purl.org/dc/dcmitype/}BibliographicResource"


def fetch_catalog() -> list[dict]:
    """Return [{code, provider, url}] from the ODS catalog."""
    print("Downloading ODS catalog …")
    xml = urllib.request.urlopen(
        urllib.request.Request(CATALOG_URL, headers={"User-Agent": "Mozilla/5.0"}),
        timeout=120).read().decode("utf-8", errors="ignore")
    root = ET.fromstring(xml)
    subsets = []
    for ds in root.iter("{http://www.w3.org/ns/dcat#}Dataset"):
        code = ds.get("{http://www.w3.org/XML/1998/namespace}id") or ds.get("id") or ""
        creator = (ds.findtext("dc:creator", default="", namespaces=NS) or "").strip()
        dist = ds.find("dcat:distribution/dcat:Distribution", NS)
        url = dist.findtext("dcat:downloadURL", default="", namespaces=NS) if dist is not None else ""
        if url and code:
            subsets.append({"code": code, "provider": creator, "url": url})
    print(f"Catalog: {len(subsets)} provider subsets")
    return subsets


def select_south_asia(subsets: list[dict]) -> list[dict]:
    sel = []
    for s in subsets:
        name = s["provider"].lower()
        code_ok = s["code"].upper().startswith(SOUTH_ASIA_CODE_PREFIXES)
        name_ok = any(k in name for k in SOUTH_ASIA_NAME_KEYWORDS)
        excluded = any(k in name for k in EXCLUDE_NAME)
        if (code_ok or name_ok) and not excluded:
            sel.append(s)
    return sel


def download_with_cache(url: str, dest: Path, retries: int = 4):
    """Download to dest (skip if present); retry with backoff on 429/5xx."""
    if dest.exists() and dest.stat().st_size > 0:
        print(f"    cached: {dest.name} ({dest.stat().st_size // 1048576} MB)")
        return
    import time
    for attempt in range(1, retries + 1):
        try:
            print(f"    downloading {dest.name} …")
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=600) as resp, open(dest, "wb") as f:
                while True:
                    chunk = resp.read(1 << 20)
                    if not chunk:
                        break
                    f.write(chunk)
            return
        except urllib.error.HTTPError as e:
            if e.code in (429, 500, 502, 503) and attempt < retries:
                wait = 30 * attempt
                print(f"    HTTP {e.code} — backing off {wait}s (attempt {attempt}/{retries})")
                time.sleep(wait)
                continue
            raise
    raise RuntimeError(f"failed to download {url}")


def stream_records(path: Path):
    """Yield dicts {title, abstract, year, subjects} from a local ODS subset."""
    with open(path, "rb") as fh:
        for event, elem in ET.iterparse(fh, events=("end",)):
            if elem.tag != REC_TAG:
                continue
            title = elem.findtext("dc:title", default="", namespaces=NS).strip()
            abstract = elem.findtext("dc:description", default="", namespaces=NS).strip()
            date = (elem.findtext("dc:date", default="", namespaces=NS) or "").strip()
            subjects = [s.text for s in elem.findall("dc:subject", NS) if s.text]
            year_m = re.search(r"\d{4}", date)
            yield {"title": title, "abstract": abstract,
                   "year": int(year_m.group()) if year_m else None,
                   "subjects": subjects}
            elem.clear()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-per-file", type=int, default=None,
                    help="max kept records per subset (for quick tests)")
    args = ap.parse_args()

    RAW_DIR.mkdir(parents=True, exist_ok=True)
    ODS_DIR.mkdir(parents=True, exist_ok=True)

    subsets = select_south_asia(fetch_catalog())
    print(f"South Asia subsets: {len(subsets)}")
    for s in subsets:
        print(f"  {s['code']:6s} {s['provider'][:70]}")

    kept = 0
    ns_min = YEAR_MIN
    with open(FILTERED_OUT, "w", encoding="utf-8") as out:
        out.write("<?xml version='1.0' encoding='utf-8'?>\n<records>\n")
        for s in subsets:
            got = 0
            print(f"\n--- {s['code']} ({s['provider']}) …")
            cache_path = ODS_DIR / f"AGRIS.ODS.{s['code']}.xml"
            try:
                download_with_cache(s["url"], cache_path)
            except Exception as e:
                print(f"    ! skipping {s['code']}: {e}")
                continue
            for rec in stream_records(cache_path):
                y = rec["year"]
                if not rec["title"] or not rec["abstract"]:
                    continue
                if y is None or not (ns_min <= y <= YEAR_MAX):
                    continue
                hay = (rec["title"] + " " + rec["abstract"]).lower()
                if not any(k in hay for k in TOPIC_KEYWORDS):
                    continue
                esc = lambda t: t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
                out.write(
                    "<record>\n"
                    f"  <title>{esc(rec['title'])}</title>\n"
                    f"  <abstract>{esc(rec['abstract'])}</abstract>\n"
                    f"  <date>{y}</date>\n"
                    f"  <region>{esc(s['provider'])}</region>\n"
                    f"  <subjects>{esc('; '.join(rec['subjects']))}</subjects>\n"
                    "</record>\n")
                kept += 1
                got += 1
                if args.limit_per_file and got >= args.limit_per_file:
                    break
            print(f"    kept {got} records")
            import time
            time.sleep(3)  # be polite to the FAO server between large downloads
        out.write("</records>\n")

    print(f"\nDone: {kept} filtered records → {FILTERED_OUT}")
    print("Next:  python preprocess.py  &&  python build_faiss_index.py")


if __name__ == "__main__":
    main()
