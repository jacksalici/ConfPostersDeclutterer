"""The one record type every lookup source returns."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import List, Optional, Sequence, Tuple


@dataclass
class Work:
    source: str                     # arxiv | openalex | crossref | llm+<verifier>
    ident: str                      # arXiv id, DOI, or OpenAlex id
    title: str
    url: str
    authors: List[str] = field(default_factory=list)
    summary: str = ""
    published: str = ""
    pdf_url: str = ""
    doi: str = ""
    categories: List[str] = field(default_factory=list)
    primary_category: str = ""      # only arXiv sets this (author-declared)
    subject: str = ""               # source-assigned topic, if any
    subject_confidence: float = 0.0
    venue: str = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Work":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


@dataclass
class Match:
    work: Optional[Work]
    score: float
    how: str = "none"   # id-on-poster | title-search | llm-verified | none

    def to_dict(self) -> dict:
        return {"work": self.work.to_dict() if self.work else None,
                "score": self.score, "how": self.how}


NO_MATCH = Match(None, 0.0, "none")


def report_candidates(log, source: str, scored: Sequence[Tuple[float, str]],
                      bar: float, top: int = 3) -> None:
    """Show what a source came back with and how close it got. Verbose only."""
    if log is None or not getattr(log, "verbose", False):
        return
    if not scored:
        log.detail("%s: no candidates" % source, indent=2)
        return
    ranked = sorted(scored, reverse=True)[:top]
    log.detail("%s: %d candidate(s), need %.2f" % (source, len(scored), bar), indent=2)
    for score, title in ranked:
        mark = "accept" if score >= bar else "  skip"
        log.detail("%s %.2f  %s" % (mark, score, title[:88]), indent=3)
