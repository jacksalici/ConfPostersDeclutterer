"""Pick the title out of a page of OCR lines - no model involved.

A conference poster has a strong visual grammar: the title is the largest text,
it sits at the top, it is not an email/affiliation/section heading, and it is
between a handful and ~30 words long. That is enough to decide deterministically.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import List, Optional, Sequence

from .ocr import Line
from .util import clean_text

SECTION_WORDS = {
    "abstract", "introduction", "background", "motivation", "methods", "method",
    "methodology", "results", "result", "discussion", "conclusion", "conclusions",
    "references", "reference", "acknowledgements", "acknowledgments", "outlook",
    "future work", "summary", "contact", "bibliography", "appendix", "data",
    "dataset", "setup", "analysis", "related work", "contributions", "overview",
}

AFFILIATION_WORDS = {
    "university", "universite", "universita", "universidad", "universitat",
    "institute", "institut", "department", "dept", "laboratory", "laboratoire",
    "lab", "labs", "school", "faculty", "college", "centre", "center", "cnrs",
    "infn", "cern", "desy", "mit", "eth", "epfl", "inria", "max planck",
    "academy", "hospital", "clinic", "foundation", "gmbh", "inc", "ltd",
}

# Matched on word boundaries, never as substrings: "eth" must not fire inside
# "Ethnography", nor "lab" inside "collaborative".
AFFILIATION_RE = re.compile(
    r"(?<![a-z])(%s)(?![a-z])" % "|".join(sorted(map(re.escape, AFFILIATION_WORDS), key=len, reverse=True))
)

POSTER_CHROME = re.compile(
    r"^(poster|board|session|track|id|no\.?|paper|abstract|fig\.?|figure|table|eq\.?"
    r"|equation|panel|step)\s*[#:.]?\s*[a-z]?[-\s]?\d+",
    re.IGNORECASE,
)
EMAIL = re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+")
URL = re.compile(r"(https?://|www\.|arxiv\.org|doi\.org|github\.com)", re.IGNORECASE)
# Lookarounds keep the DOI 10.1145/3292500.3330701 from reading as an arXiv id.
ARXIV_ID = re.compile(r"(?:arxiv[:\s]*)?(?<!\d)(\d{4}\.\d{4,5})(?!\d)(v\d+)?", re.IGNORECASE)
DOI_ID = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>,;)\]]+)", re.IGNORECASE)
AUTHOR_LINE = re.compile(
    r"^([A-Z][\w.'-]*\.?\s+){1,3}[A-Z][\w.'-]+([\d*†‡§,]|,\s*[A-Z])", re.UNICODE
)


@dataclass
class TitleCandidate:
    text: str
    score: float
    lines: List[Line]

    @property
    def word_count(self) -> int:
        return len(self.text.split())


@dataclass
class PageReading:
    title: Optional[str]
    title_score: float
    candidates: List[TitleCandidate]
    arxiv_ids: List[str]
    full_text: str
    dois: List[str] = None  # set in read_page; kept last for backward compatibility


def _is_junk(text: str) -> bool:
    low = text.lower().strip(" .:-|")
    if not low:
        return True
    if low in SECTION_WORDS:
        return True
    if EMAIL.search(text) or URL.search(text) or DOI_ID.search(text):
        return True
    if POSTER_CHROME.match(text):
        return True
    letters = sum(ch.isalpha() for ch in text)
    if letters < max(3, len(text) * 0.5):
        return True
    if AFFILIATION_RE.search(low) and len(low.split()) < 14:
        return True
    return False


def _looks_like_authors(text: str) -> bool:
    if AUTHOR_LINE.match(text):
        return True
    # "A. Rossi, B. Bianchi, C. Verdi" - many commas, mostly capitalised tokens.
    parts = [p.strip() for p in text.split(",") if p.strip()]
    if len(parts) >= 3:
        shortish = sum(1 for p in parts if len(p.split()) <= 4)
        capped = sum(1 for p in parts if p[:1].isupper())
        if shortish == len(parts) and capped >= len(parts) - 1:
            return True
    return False


def group_lines(lines: Sequence[Line]) -> List[List[Line]]:
    """Merge vertically adjacent lines of comparable size into one block."""
    blocks: List[List[Line]] = []
    for line in sorted(lines, key=lambda l: (l.y, l.x)):
        if not blocks:
            blocks.append([line])
            continue
        prev = blocks[-1][-1]
        size_ratio = line.h / prev.h if prev.h else 99.0
        gap = line.y - (prev.y + prev.h)
        overlaps = min(line.x + line.w, prev.x + prev.w) - max(line.x, prev.x) > 0
        if 0.7 <= size_ratio <= 1.4 and gap <= 1.2 * prev.h and overlaps:
            blocks[-1].append(line)
        else:
            blocks.append([line])
    return blocks


def score_block(block: Sequence[Line], max_height: float) -> Optional[TitleCandidate]:
    text = clean_text(" ".join(l.text for l in block))
    if not text or _is_junk(text):
        return None
    words = text.split()
    if not (2 <= len(words) <= 25):
        return None
    # Titles do not end in a full stop; a long sentence that does is body text,
    # and accepting it would hide the fact that no title was found at all.
    if len(words) > 12 and text.rstrip().endswith((".", "!", "?")):
        return None
    if _looks_like_authors(text):
        return None

    height = max(l.h for l in block)
    top = min(l.y for l in block)
    rel_size = height / max_height if max_height else 0.0
    # Two-word titles are real ("Deep Sets", "Random Forests") but so is every
    # stray label on a poster, so only accept one if it is the biggest text there.
    if len(words) < 3 and rel_size < 0.9:
        return None

    score = 0.0
    score += 3.0 * rel_size            # the title is the biggest thing on the poster
    score += 2.0 * max(0.0, 1.0 - top * 2.5)  # ...and it lives at the top
    if 5 <= len(words) <= 20:
        score += 0.6
    alpha = [w for w in words if w[:1].isalpha()]
    if alpha and sum(1 for w in alpha if w[:1].isupper()) / len(alpha) >= 0.5:
        score += 0.3                   # title case
    if text.isupper():
        score += 0.2
    if text.rstrip().endswith((".", ",", ";")):
        score -= 0.3                   # sentences are body text
    if sum(1 for l in block) > 4:
        score -= 0.5                   # a paragraph, not a heading
    score += 0.3 * min(sum(l.conf for l in block) / len(block), 1.0)
    return TitleCandidate(text=text.strip(" .-|"), score=round(score, 4), lines=list(block))


def read_page(lines: Sequence[Line]) -> PageReading:
    full_text = "\n".join(l.text for l in sorted(lines, key=lambda l: (l.y, l.x)))
    arxiv_ids = []
    for match in ARXIV_ID.finditer(full_text):
        if match.group(1) not in arxiv_ids:
            arxiv_ids.append(match.group(1))
    dois = []
    for match in DOI_ID.finditer(full_text):
        doi = match.group(1).rstrip(".,;")
        if doi not in dois:
            dois.append(doi)

    if not lines:
        return PageReading(None, 0.0, [], arxiv_ids, full_text, dois)

    max_height = max(l.h for l in lines)
    candidates = []
    for block in group_lines(lines):
        cand = score_block(block, max_height)
        if cand:
            candidates.append(cand)
    candidates.sort(key=lambda c: c.score, reverse=True)
    if not candidates:
        return PageReading(None, 0.0, [], arxiv_ids, full_text, dois)
    best = candidates[0]
    return PageReading(best.text, best.score, candidates[:5], arxiv_ids, full_text, dois)
