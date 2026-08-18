"""arXiv lookup over the public Atom API (no key needed)."""

from __future__ import annotations

import urllib.error
import urllib.parse
import xml.etree.ElementTree as ET
from typing import List, Optional

from .http import Fetcher
from .util import clean_text, normalise, similarity
from .works import Match, Work, report_candidates

NAME = "arxiv"
API = "http://export.arxiv.org/api/query"
NS = {"a": "http://www.w3.org/2005/Atom", "arxiv": "http://arxiv.org/schemas/atom"}


def parse_feed(xml_text: str) -> List[Work]:
    root = ET.fromstring(xml_text)
    works = []
    for entry in root.findall("a:entry", NS):
        raw_id = (entry.findtext("a:id", "", NS) or "").strip()
        title = clean_text(entry.findtext("a:title", "", NS) or "")
        if not raw_id or not title:
            continue
        arxiv_id = raw_id.rsplit("/abs/", 1)[-1]
        primary = entry.find("arxiv:primary_category", NS)
        primary_category = primary.get("term") if primary is not None else ""
        categories = [c.get("term", "") for c in entry.findall("a:category", NS)]
        if primary_category and primary_category not in categories:
            categories.insert(0, primary_category)
        pdf_url = ""
        for link in entry.findall("a:link", NS):
            if link.get("title") == "pdf":
                pdf_url = link.get("href", "")
        doi = clean_text(entry.findtext("arxiv:doi", "", NS) or "")
        works.append(
            Work(
                source=NAME,
                ident=arxiv_id,
                title=title,
                url="https://arxiv.org/abs/%s" % arxiv_id,
                authors=[clean_text(a.findtext("a:name", "", NS) or "")
                         for a in entry.findall("a:author", NS)],
                summary=clean_text(entry.findtext("a:summary", "", NS) or ""),
                published=(entry.findtext("a:published", "", NS) or "")[:10],
                pdf_url=pdf_url or "https://arxiv.org/pdf/%s" % arxiv_id,
                doi=doi,
                categories=[c for c in categories if c],
                primary_category=primary_category or (categories[0] if categories else ""),
            )
        )
    return works


def _query_url(search_query: str, max_results: int = 10) -> str:
    return "%s?%s" % (API, urllib.parse.urlencode(
        {"search_query": search_query, "start": 0,
         "max_results": max_results, "sortBy": "relevance"}))


def _id_url(arxiv_id: str) -> str:
    return "%s?%s" % (API, urllib.parse.urlencode({"id_list": arxiv_id, "max_results": 1}))


def _title_query(title: str) -> str:
    # Keep every word: arXiv treats ti:"..." as an exact phrase, so dropping
    # stopwords ("is", "of") turns a hit into a miss.
    words = normalise(title).split()[:16]
    return 'ti:"%s"' % " ".join(words) if words else ""


def _loose_query(title: str) -> str:
    words = [w for w in normalise(title).split() if len(w) > 3][:10]
    return "all:(%s)" % " AND ".join(words) if words else ""


def lookup_by_id(fetcher: Fetcher, arxiv_id: str) -> Optional[Work]:
    try:
        works = parse_feed(fetcher.get(_id_url(arxiv_id)))
    except (urllib.error.URLError, ET.ParseError, OSError):
        return None
    return works[0] if works else None


def search(fetcher: Fetcher, title: str, threshold: float = 0.72,
           max_results: int = 10, log=None) -> Match:
    """Best arXiv match for an OCR'd title, or an empty match below threshold."""
    best: Optional[Work] = None
    best_score = 0.0
    scored = []
    for build in (_title_query, _loose_query):
        query = build(title)
        if not query:
            continue
        try:
            candidates = parse_feed(fetcher.get(_query_url(query, max_results)))
        except (urllib.error.URLError, ET.ParseError, OSError):
            continue
        for work in candidates:
            score = similarity(title, work.title)
            scored.append((score, work.title))
            if score > best_score:
                best, best_score = work, score
        if best_score >= 0.9:
            break
    report_candidates(log, NAME, scored, threshold)
    if best_score < threshold:
        return Match(None, round(best_score, 4), "none")
    return Match(best, round(best_score, 4), "title-search")
