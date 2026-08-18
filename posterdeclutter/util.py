from __future__ import annotations

import re
import unicodedata

_WS = re.compile(r"\s+")
_NON_WORD = re.compile(r"[^a-z0-9 ]+")

# Ligatures and dashes that OCR loves to emit.
_REPLACEMENTS = {
    "ﬀ": "ff", "ﬁ": "fi", "ﬂ": "fl", "ﬃ": "ffi", "ﬄ": "ffl",
    "‐": "-", "‑": "-", "‒": "-", "–": "-", "—": "-",
    "‘": "'", "’": "'", "“": '"', "”": '"',
    " ": " ",
}


def clean_text(text: str) -> str:
    """Normalise OCR output without changing its words."""
    text = unicodedata.normalize("NFKC", text)
    for src, dst in _REPLACEMENTS.items():
        text = text.replace(src, dst)
    text = text.replace("­", "")  # soft hyphen
    return _WS.sub(" ", text).strip()


def normalise(text: str) -> str:
    """Aggressive form used only for comparing two strings."""
    text = clean_text(text).lower()
    text = _NON_WORD.sub(" ", text)
    return _WS.sub(" ", text).strip()


def tokens(text: str) -> list:
    return normalise(text).split()


def similarity(a: str, b: str) -> float:
    """Token-level F1 between two strings, 0..1.

    Chosen over difflib because OCR drops and inserts whole words, and word
    order in a recognised title is reliable enough that ordering adds nothing.
    """
    ta, tb = tokens(a), tokens(b)
    if not ta or not tb:
        return 0.0
    from collections import Counter

    ca, cb = Counter(ta), Counter(tb)
    overlap = sum((ca & cb).values())
    if not overlap:
        return 0.0
    precision = overlap / len(ta)
    recall = overlap / len(tb)
    return 2 * precision * recall / (precision + recall)


def slugify(text: str, max_len: int = 60) -> str:
    slug = normalise(text).replace(" ", "-")
    slug = re.sub(r"-+", "-", slug).strip("-")
    return slug[:max_len] or "untitled"
