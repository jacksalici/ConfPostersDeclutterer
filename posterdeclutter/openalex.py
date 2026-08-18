"""OpenAlex lookup - covers the ~85% of conference work that never reaches arXiv.

Free, no key. Give a contact address (--mailto) to join their faster "polite
pool". OpenAlex also assigns each work a topic, which we surface as a *weak*
subfield signal: unlike an arXiv category, which the authors declared, the topic
is model-assigned.
"""

from __future__ import annotations

import urllib.error
import urllib.parse
from typing import List, Optional

from .http import Fetcher
from .util import clean_text, normalise, similarity
from .works import Match, Work, report_candidates

NAME = "openalex"
API = "https://api.openalex.org/works"


def _search_url(fetcher: Fetcher, title: str, per_page: int = 8) -> str:
    words = normalise(title).split()[:20]
    params = fetcher.polite(
        {"filter": "title.search:%s" % " ".join(words), "per-page": per_page}
    )
    return "%s?%s" % (API, urllib.parse.urlencode(params))


def _doi_url(fetcher: Fetcher, doi: str) -> str:
    doi = doi.strip().replace("https://doi.org/", "").lstrip("/")
    return "%s/https://doi.org/%s?%s" % (
        API, urllib.parse.quote(doi), urllib.parse.urlencode(fetcher.polite({}))
    )


def _abstract(inverted: Optional[dict], limit: int = 1800) -> str:
    """OpenAlex ships abstracts as {word: [positions]}. Put them back in order."""
    if not inverted:
        return ""
    slots = {}
    for word, positions in inverted.items():
        for position in positions:
            slots[position] = word
    if not slots:
        return ""
    text = " ".join(slots[i] for i in sorted(slots))
    return clean_text(text)[:limit]


def to_work(item: dict) -> Optional[Work]:
    title = clean_text(item.get("display_name") or item.get("title") or "")
    if not title:
        return None
    doi = (item.get("doi") or "").replace("https://doi.org/", "")
    # OpenAlex nests topic -> subfield -> field. The subfield tier is a coarse
    # Scopus bucket (quantum-information papers sit under "Artificial
    # Intelligence"), so the topic name is the one worth clustering on.
    topic = item.get("primary_topic") or {}
    name = clean_text(topic.get("display_name") or "")
    subfield = (topic.get("subfield") or {}).get("display_name", "")
    field = (topic.get("field") or {}).get("display_name", "")
    location = item.get("primary_location") or {}
    oa_location = item.get("best_oa_location") or {}
    return Work(
        source=NAME,
        ident=(item.get("id") or "").rsplit("/", 1)[-1],
        title=title,
        url=location.get("landing_page_url") or item.get("id") or "",
        authors=[clean_text(a.get("author", {}).get("display_name", ""))
                 for a in item.get("authorships", [])[:12]],
        summary=_abstract(item.get("abstract_inverted_index")),
        published=item.get("publication_date") or str(item.get("publication_year") or ""),
        pdf_url=oa_location.get("pdf_url") or "",
        doi=doi,
        categories=[c for c in (name, subfield, field) if c],
        subject=name or field,
        subject_confidence=float(topic.get("score") or 0.0),
        venue=clean_text((location.get("source") or {}).get("display_name", "")),
    )


def lookup_by_doi(fetcher: Fetcher, doi: str) -> Optional[Work]:
    try:
        return to_work(fetcher.get_json(_doi_url(fetcher, doi)))
    except (urllib.error.URLError, ValueError, OSError):
        return None


def search(fetcher: Fetcher, title: str, threshold: float = 0.72,
           max_results: int = 8, log=None) -> Match:
    try:
        payload = fetcher.get_json(_search_url(fetcher, title, max_results))
    except (urllib.error.URLError, ValueError, OSError):
        return Match(None, 0.0, "none")
    best: Optional[Work] = None
    best_score = 0.0
    scored = []
    for item in payload.get("results", []):
        work = to_work(item)
        if not work:
            continue
        # OpenAlex ranks by its own relevance; re-rank on title agreement.
        score = similarity(title, work.title)
        scored.append((score, work.title))
        if score > best_score:
            best, best_score = work, score
    report_candidates(log, NAME, scored, threshold)
    if best_score < threshold:
        return Match(None, round(best_score, 4), "none")
    return Match(best, round(best_score, 4), "title-search")
