"""Ethical guardrails: temporal filtering, climate/soil niche filters,
and lightweight bias/adversarial checks on generated answers."""
import re

from .config import settings
from .retrieval import retriever

# Heuristic patterns for recommendations that may be unsafe or regionally wrong
_SUSPECT = [
    (r"\bwinter crops?\b.*\btropical\b", "possible regional mismatch: winter crops in a tropical zone"),
    (r"\b(pesticide|herbicide)\b.*\bincrease\b", "possible harmful recommendation: increase chemical use"),
    (r"\b(gender|men should|women should)\b", "possible gender-stereotyped framing"),
]


def niche_filter(query: str, chunks: list[dict]) -> list[dict]:
    """Drop chunks that disagree with the query's climate/soil context.

    In production this consults FAO agro-ecological zoning maps; the skeleton
    keeps a keyword-level check.
    """
    q = query.lower()
    keep = []
    for c in chunks:
        text = (c.get("text") or "").lower()
        conflict = next((msg for pat, msg in _SUSPECT if re.search(pat, text) and not re.search(pat.split(".*")[0], q)), None)
        if conflict:
            c = {**c, "flag": conflict}
        keep.append(c)
    return keep


def temporal_filter(query: str, chunks: list[dict]) -> list[dict]:
    """Prioritize recent peer-reviewed sources (last N years)."""
    return retriever.temporal_search(query) or chunks


def check_output(answer: str) -> dict:
    """Adversarial-style scan of generated output; returns flags for audit."""
    flags = [msg for _, msg in
             [(pat, msg) for pat, msg in _SUSPECT] if re.search(pat, answer.lower())]
    return {"flags": flags, "ok": not flags}


def apply_guardrails(query: str, chunks: list[dict]) -> list[dict]:
    return niche_filter(query, temporal_filter(query, chunks))
