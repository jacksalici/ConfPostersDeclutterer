"""Crossref lookup - the DOI registry. Best for published proceedings and
journal papers; weak on bare titles, which the match threshold handles by
declining rather than guessing."""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
from typing import Optional

from .http import Fetcher
from .util import clean_text, similarity
from .works import Match, Work, report_candidates

NAME = "crossref"
API = "https://api.crossref.org/works"
SELECT = "title,author,DOI,URL,issued,container-title,subject,abstract,type"
_TAGS = re.compile(r"<[^>]+>")


def _search_url(fetcher: Fetcher, title: str, rows: int = 5) -> str:
    params = fetcher.polite(
        {"query.bibliographic": title, "rows": rows, "select": SELECT}
    )
    return "%s?%s" % (API, urllib.parse.urlencode(params))


def _doi_url(fetcher: Fetcher, doi: str) -> str:
    doi = doi.strip().replace("https://doi.org/", "").lstrip("/")
    return "%s/%s" % (API, urllib.parse.quote(doi))


def to_work(item: dict) -> Optional[Work]:
    titles = item.get("title") or []
    title = clean_text(titles[0]) if titles else ""
    if not title:
        return None
    authors = []
    for person in item.get("author", [])[:12]:
        name = " ".join(filter(None, [person.get("given"), person.get("family")]))
        if name:
            authors.append(clean_text(name))
    parts = (item.get("issued", {}).get("date-parts") or [[]])[0]
    published = "-".join("%02d" % p if i else str(p) for i, p in enumerate(parts) if p)
    doi = item.get("DOI", "")
    subjects = item.get("subject") or []
    container = item.get("container-title") or []
    return Work(
        source=NAME,
        ident=doi,
        title=title,
        url=item.get("URL") or ("https://doi.org/%s" % doi if doi else ""),
        authors=authors,
        summary=clean_text(_TAGS.sub(" ", item.get("abstract", "")))[:1800],
        published=published,
        doi=doi,
        categories=subjects,
        subject=subjects[0] if subjects else "",
        venue=clean_text(container[0]) if container else "",
    )


def lookup_by_doi(fetcher: Fetcher, doi: str) -> Optional[Work]:
    try:
        payload = fetcher.get_json(_doi_url(fetcher, doi))
    except (urllib.error.URLError, ValueError, OSError):
        return None
    return to_work(payload.get("message", {}))


def search(fetcher: Fetcher, title: str, threshold: float = 0.72,
           max_results: int = 5, log=None) -> Match:
    try:
        payload = fetcher.get_json(_search_url(fetcher, title, max_results))
    except (urllib.error.URLError, ValueError, OSError):
        return Match(None, 0.0, "none")
    best: Optional[Work] = None
    best_score = 0.0
    scored = []
    for item in payload.get("message", {}).get("items", []):
        work = to_work(item)
        if not work:
            continue
        score = similarity(title, work.title)
        scored.append((score, work.title))
        if score > best_score:
            best, best_score = work, score
    report_candidates(log, NAME, scored, threshold)
    if best_score < threshold:
        return Match(None, round(best_score, 4), "none")
    return Match(best, round(best_score, 4), "title-search")
