"""Tests for the preprocessing pipeline (backend/app/preprocess.py).

Covers correctness of the cleaning/tokenizing/chunking/AGRIS-parsing steps and
a performance gate so the full 19.5k-record corpus preprocesses in minutes, not
tens of minutes (TDD: performance test written first, RED, then optimization).
"""
import sys
import time
import xml.etree.ElementTree as ET
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.preprocess import (CHUNK_OVERLAP, CHUNK_SIZE, chunk_text, clean_text,
                            preprocess_document, tokenize)
from preprocess import parse_agris_xml, process_corpus

# ---------------------------------------------------------------- clean_text

def test_clean_text_lowercases_and_strips_specials():
    assert clean_text("Soil-Health & Fertility!!") == "soil health fertility"


def test_clean_text_keeps_indic_scripts():
    assert "मिट्टी" in clean_text("मिट्टी की सेहत")


def test_clean_text_collapses_whitespace():
    assert clean_text("a\n\n  b\tc") == "a b c"


# ----------------------------------------------------------------- tokenize

def test_tokenize_removes_stopwords_but_keeps_agri_terms():
    toks = tokenize("The soil and the crop should be irrigated during the monsoon")
    assert "soil" in toks and "crop" in toks and "monsoon" in toks
    assert "the" not in toks and "should" not in toks


def test_tokenize_returns_lemmas_for_plural_nouns():
    assert "fertilizer" in tokenize("organic fertilizers improve soil")


def test_tokenize_handles_empty_input():
    assert tokenize("") == []
    assert tokenize("!!! ???") == []


# ---------------------------------------------------------------- chunk_text

def test_chunk_text_short_input_returns_single_chunk():
    assert chunk_text("one two three") == ["one two three"]


def test_chunk_text_long_input_respects_size_and_overlap():
    tokens = [f"w{i}" for i in range(CHUNK_SIZE * 2 + 10)]
    chunks = chunk_text(" ".join(tokens))
    assert len(chunks) >= 3
    first, second = chunks[0].split(), chunks[1].split()
    assert len(first) == CHUNK_SIZE
    # overlap: last OVERLAP tokens of chunk 1 == first OVERLAP of chunk 2
    assert first[-CHUNK_OVERLAP:] == second[:CHUNK_OVERLAP]


def test_preprocess_document_chunks_match_token_size():
    text = " ".join(["soil fertility management"] * 200)
    for chunk in preprocess_document(text):
        assert len(chunk.split()) <= CHUNK_SIZE


# ----------------------------------------------------------- parse_agris_xml

@pytest.fixture()
def agris_xml(tmp_path):
    xml = """<?xml version='1.0' encoding='utf-8'?>
    <records>
      <record>
        <title>Improving soil fertility in South Asia</title>
        <abstract>Compost and green manure raise yields without chemical fertilizer.</abstract>
        <date>2022</date>
        <region>Indian Council of Agricultural Research</region>
      </record>
      <record>
        <title>No abstract here</title>
        <date>2021</date>
      </record>
    </records>"""
    p = tmp_path / "sample.xml"
    p.write_text(xml, encoding="utf-8")
    return p


def test_parse_agris_xml_keeps_records_with_title_abstract_region(agris_xml):
    recs = parse_agris_xml(agris_xml)
    assert len(recs) == 1
    assert recs[0]["title"] == "Improving soil fertility in South Asia"
    assert recs[0]["year"] == "2022"
    assert "india" in recs[0]["region"].lower()


def test_parse_agris_xml_skips_records_without_abstract(agris_xml):
    titles = [r["title"] for r in parse_agris_xml(agris_xml)]
    assert all("No abstract" not in t for t in titles)


def test_console_print_survives_windows_cp1252(capsys):
    """Regression: the final 'Done' print crashed with UnicodeEncodeError on
    Windows consoles (cp1252) because of the '→' character."""
    code = (
        "import sys; sys.path.insert(0, r'{}'); "
        "import preprocess; "
        "print('Done: 3 documents \u2192 9 chunks')"
    ).format(Path(__file__).resolve().parent.parent)
    import subprocess
    env_path = Path(sys.executable)
    res = subprocess.run([str(env_path), "-c", code], capture_output=True,
                         text=True, encoding="utf-8", errors="replace")
    assert res.returncode == 0, res.stderr
    assert "9 chunks" in res.stdout


# ----------------------------------------------------------------- speed

def _make_corpus_xml(n: int, tmp_path: Path) -> Path:
    abstract = ("Field experiments on soil fertility with compost, manure and "
                "irrigation scheduling for rice and wheat were conducted during "
                "the monsoon season to reduce post-harvest storage loss. ") * 8
    parts = ["<?xml version='1.0' encoding='utf-8'?><records>"]
    for i in range(n):
        parts.append(
            f"<record><title>Study {i} on soil fertility</title>"
            f"<abstract>{abstract}</abstract><date>2020</date>"
            "<region>Indian Council of Agricultural Research</region></record>")
    parts.append("</records>")
    p = tmp_path / "corpus.xml"
    p.write_text("".join(parts), encoding="utf-8")
    return p


def test_preprocess_corpus_performance_gate(tmp_path):
    """200 records must preprocess in < 10 s (19.5k full corpus ≈ minutes).

    RED on the original implementation (full SpaCy pipeline per record);
    GREEN after batching + disabling parser/NER.
    """
    corpus = _make_corpus_xml(200, tmp_path)
    recs = parse_agris_xml(corpus)
    assert len(recs) == 200
    start = time.perf_counter()
    chunks = process_corpus(recs)
    elapsed = time.perf_counter() - start
    assert len(chunks) > 0
    assert elapsed < 10, f"processing 200 records took {elapsed:.1f}s (>10s)"


def test_process_corpus_stores_tokens_for_bm25(tmp_path):
    """Chunk dicts must carry their token lists so the BM25 index can be built
    without re-tokenizing the whole corpus a second time."""
    corpus = _make_corpus_xml(3, tmp_path)
    chunks = process_corpus(parse_agris_xml(corpus))
    assert chunks
    sample = chunks[0]
    assert "tokens" in sample
    expected = tokenize(sample["text"])
    assert sample["tokens"] == expected
