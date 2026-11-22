"""Domain glossary: replace English agricultural terms that survive NLLB
back-translation with their target-language equivalents.

Fixes quality-backlog item #2 (code-mixing): answers in Hindi/Bengali/Tamil
kept English jargon like "Crop Rotation", "RAINFED AREAS", "mulching" inside
otherwise-translated text, which low-literacy farmers may not parse.

The glossary is applied AFTER back-translation, longest-phrase-first,
case-insensitively, with common inflected forms folded in.
"""
import re
from functools import lru_cache

# English term (lowercase, inflections folded via regex) → target-language term
GLOSSARY: dict[str, dict[str, str]] = {
    "crop rotation": {
        "hi": "फसल चक्र", "bn": "ফসল চক্র", "ta": "பயிர் சுழற்சி",
    },
    "drip irrigation": {
        "hi": "टपक सिंचाई", "bn": "ড্রিপ সেচ", "ta": "சொட்டு நீர் பாசனம்",
    },
    "rainwater harvesting": {
        "hi": "वर्षा जल संचयन", "bn": "বৃষ্টির পানি সংগ্রহ", "ta": "மழைநீர் சேமிப்பு",
    },
    "organic manure": {
        "hi": "जैविक खाद", "bn": "জৈব সার", "ta": "இயற்கை உரம்",
    },
    "green manure": {
        "hi": "हरी खाद", "bn": "সবুজ সার", "ta": "பச்சை உரம்",
    },
    "organic fertilizer": {
        "hi": "जैविक उर्वरक", "bn": "জৈব সার", "ta": "இயற்கை உரம்",
    },
    "biofertilizer": {
        "hi": "जैव उर्वरक", "bn": "জীবাণু সার", "ta": "உயிர் உரம்",
    },
    "irrigation": {
        "hi": "सिंचाई", "bn": "সেচ", "ta": "பாசனம்",
    },
    "manure": {
        "hi": "खाद", "bn": "সার", "ta": "உரம்",
    },
    "fertilizer": {
        "hi": "उर्वरक", "bn": "সার", "ta": "உரம்",
    },
    "compost(?:ing|ed|s)?": {
        "hi": "कम्पोस्ट", "bn": "কম্পোস্ট", "ta": "கம்போஸ்ட்",
    },
    "mulch(?:ing|ed|es)?": {
        "hi": "मल्चिंग", "bn": "মালচিং", "ta": "மல்சிங்",
    },
    "rainfed": {
        "hi": "वर्षा आधारित", "bn": "বৃষ্টিনির্ভর", "ta": "மழையை ஆதாரமாகக் கொண்ட",
    },
    "drainage": {
        "hi": "जल निकासी", "bn": "নিষ্কাশন", "ta": "வடிகால்",
    },
    "yield": {
        "hi": "उपज", "bn": "ফলন", "ta": "விளைச்சல்",
    },
    "post-harvest": {
        "hi": "फसल कटाई के बाद", "bn": "ফসল তোলার পরে", "ta": "அறுவடைக்குப் பிறகு",
    },
    "vermicompost(?:ing)?": {
        "hi": "वर्मीकम्पोस्ट", "bn": "ভার্মিকম্পোস্ট", "ta": "வர்மிகம்போஸ்ட்",
    },
    "nitrogen": {
        "hi": "नाइट्रोजन", "bn": "নাইট্রোজেন", "ta": "நைட்ரஜன்",
    },
    "phosphorus": {
        "hi": "फास्फोरस", "bn": "ফসফরাস", "ta": "பாஸ்பரஸ்",
    },
    "potassium": {
        "hi": "पोटैशियम", "bn": "পটাশ", "ta": "பொட்டாசியம்",
    },
    "cover crop": {
        "hi": "आच्छादक फसल", "bn": "আচ্ছাদক ফসল", "ta": "மூடு பயிர்",
    },
    "intercropping": {
        "hi": "अंतराफसली", "bn": "আন্তঃফসল চাষ", "ta": "இடைப்பயிர்",
    },
}

# Compound phrases must be replaced before their components
_ORDERED_PATTERNS = sorted(GLOSSARY.keys(), key=lambda p: -len(p.split()))


def glossary_languages() -> list[str]:
    """Languages the glossary covers (the farmer languages, not English)."""
    return sorted({lang for m in GLOSSARY.values() for lang in m})


@lru_cache(maxsize=1)
def _compiled():
    return [(re.compile(rf"\b{p}\b", re.IGNORECASE), p) for p in _ORDERED_PATTERNS]


def apply_glossary(text: str, lang: str) -> str:
    """Replace English domain terms in `text` with `lang` equivalents.

    English text is returned unchanged (it IS the target language).
    """
    if lang == "en":
        return text
    out = text
    for regex, pattern in _compiled():
        term = GLOSSARY[pattern].get(lang)
        if term:
            out = regex.sub(term, out)
    return out
