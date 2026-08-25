"""Metadata from wherever a pasted link points.

arXiv ids and DOIs resolve through the indexes; everything else people paste -
openaccess.thecvf.com papers, publisher and anthology landing pages - lands
here. Most open-access hosts embed Highwire/Google-Scholar `citation_*` meta
tags on their abstract pages, which carry the title, authors, venue and
keywords without any text scraping. Whatever survives becomes a Work, so a
merged poster gets a real title, authors and a category instead of a bare
link. A paper PDF itself is unreadable to us; where a host has a predictable
HTML sibling (CVF Open Access does), we read that instead.
"""

from __future__ import annotations

import re
import urllib.error
import urllib.parse
from typing import Dict, List, Optional

from .http import Fetcher
from .util import clean_text
from .works import Work

NAME = "web"

# .../papers/Name_ICCV2023_paper.pdf has a sibling .../html/Name_ICCV2023_paper.html.
_CVF_PDF = re.compile(
    r"(https?://openaccess\.thecvf\.com/content/[^/]+)/papers/(\S+)\.pdf$", re.I)

_TAG = re.compile(r"<meta\s+[^>]*>", re.I)
_ATTR = re.compile(r"([\w:-]+)\s*=\s*(\"[^\"]*\"|'[^']*')")
_TITLE_TAG = re.compile(r"<title[^>]*>(.*?)</title>", re.I | re.S)
_ISO_DATE = re.compile(r"\d{4}-\d{1,2}(-\d{1,2})?")
_YEAR = re.compile(r"(19|20)\d{2}")


def _meta_tags(html_text: str) -> Dict[str, List[str]]:
    """Pull every <meta name=... content=...> into {lowercased name: [values]}."""
    found: Dict[str, List[str]] = {}
    for tag in _TAG.findall(html_text):
        attrs = {key.lower(): value.strip("\"'")
                 for key, value in _ATTR.findall(tag)}
        name = attrs.get("name") or attrs.get("property") or ""
        content = attrs.get("content")
        if not name or not content:
            continue
        found.setdefault(name.strip().lower(), []).append(content)
    return found


def _first(metas: Dict[str, List[str]], *names: str) -> str:
    for name in names:
        for value in metas.get(name, []):
            if value.strip():
                return value.strip()
    return ""


def _all(metas: Dict[str, List[str]], *names: str) -> List[str]:
    out = []
    for name in names:
        for value in metas.get(name, []):
            if value.strip():
                out.append(clean_text(value))
    return out


def _date(text: str) -> str:
    if not text:
        return ""
    iso = _ISO_DATE.search(text)
    if iso:
        return iso.group(0)
    year = _YEAR.search(text)
    return year.group(0) if year else ""


def _keywords(text: str) -> List[str]:
    if not text:
        return []
    parts = text.split(";")
    if len(parts) == 1:
        parts = text.split(",")
    return [clean_text(p.strip(" .")) for p in parts if p.strip(" .")]


def _html_title(html_text: str) -> str:
    found = _TITLE_TAG.search(html_text)
    if not found:
        return ""
    # "The Real Title | PMLR" - the site name after the pipe is noise.
    return clean_text(found.group(1).split("|")[0])


def _page_work(url: str, html_text: str) -> Optional[Work]:
    metas = _meta_tags(html_text)
    title = (clean_text(_first(metas, "citation_title", "dc.title"))
             or _html_title(html_text))
    if not title:
        return None
    authors = list(dict.fromkeys(_all(metas, "citation_author", "dc.creator")))
    doi = clean_text(_first(metas, "citation_doi")).replace("https://doi.org/", "")
    keywords = _keywords(_first(metas, "citation_keywords", "dc.subject",
                                "keywords", "dc.keywords"))
    return Work(
        source=NAME,
        ident=url,
        title=title,
        url=url,
        authors=authors[:12],
        summary=clean_text(_first(metas, "citation_abstract", "description",
                                  "og:description"))[:1800],
        published=_date(_first(metas, "citation_publication_date", "citation_date")),
        pdf_url=_first(metas, "citation_pdf_url"),
        doi=doi,
        categories=keywords,
        subject=keywords[0] if keywords else "",
        subject_confidence=0.5 if keywords else 0.0,
        venue=clean_text(_first(metas, "citation_conference_title",
                                "citation_journal_title")),
    )


def resolve(fetcher: Fetcher, url: str, log=None) -> Optional[Work]:
    """Analyse a pasted URL and return what metadata it carries, or None."""
    candidates = [url]
    cvf = _CVF_PDF.search(url)
    if cvf:
        # The paper PDF itself is unreadable to us; its HTML abstract page is not.
        candidates.insert(0, "%s/html/%s.html" % (cvf.group(1), cvf.group(2)))
    elif urllib.parse.urlsplit(url).path.lower().endswith(".pdf"):
        return None                      # a bare PDF from somewhere we cannot read

    for candidate in candidates:
        try:
            body = fetcher.get(candidate)
        except (urllib.error.URLError, OSError):
            continue
        if body[:1024].lstrip().startswith("%PDF"):
            continue                     # landed on the binary anyway
        work = _page_work(candidate, body)
        if work:
            return work
        if log is not None:
            log.detail("no citation metadata found in %s" % candidate, indent=2)
    return None
