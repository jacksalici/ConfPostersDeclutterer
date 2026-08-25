"""The manual round-trip: export what could not be matched, take back your links.

The tool writes `unmatched.csv` listing every poster it could not place. You fill
in the `link` column (an arXiv URL, a DOI, an OpenAlex ID, or any URL at all),
optionally correcting `title` and `subfield`, and pass the file back with
`--merge`. Manual rows win over everything: you looked at the poster, the tool
did not.
"""

from __future__ import annotations

import csv
import re
import urllib.parse
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

from . import arxiv, crossref, openalex, webpage
from .http import Fetcher
from .log import Log
from .works import Match, Work

FIELDS = ("image", "title", "link", "subfield", "note")
NAME = "manual"

# arxiv.org/abs/2401.12345 and the pre-2007 arxiv.org/abs/hep-ex/0123456 form.
_ARXIV_NEW = re.compile(r"arxiv\.org/(?:abs|pdf)/(?<!\d)(\d{4}\.\d{4,5})(?!\d)(v\d+)?", re.I)
_ARXIV_OLD = re.compile(r"arxiv\.org/(?:abs|pdf)/([a-z-]+(?:\.[A-Z]{2})?/\d{7})(v\d+)?", re.I)
_DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>,;)\]]+)", re.I)
_OPENALEX = re.compile(r"openalex\.org/(W\d+)", re.I)


def parse_link(link: str) -> Optional[Tuple[str, str]]:
    """Work out what kind of identifier a pasted link carries."""
    link = (link or "").strip()
    if not link:
        return None
    for pattern, kind in ((_ARXIV_NEW, "arxiv"), (_ARXIV_OLD, "arxiv"), (_OPENALEX, "openalex")):
        found = pattern.search(link)
        if found:
            return (kind, found.group(1))
    doi = _DOI.search(urllib.parse.unquote(link))
    if doi:
        return ("doi", doi.group(1).rstrip(".,;"))
    if link.lower().startswith(("http://", "https://")):
        return ("url", link)
    # A bare arXiv id pasted without the URL around it.
    bare = re.fullmatch(r"(?:arxiv:)?(\d{4}\.\d{4,5})(v\d+)?", link, re.I)
    if bare:
        return ("arxiv", bare.group(1))
    return None


# -- export ----------------------------------------------------------------

def write_unmatched(posters: Sequence, path: Path) -> int:
    """Write every poster still missing a paper. Returns how many."""
    rows = [p for p in posters if not p.work]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS)
        writer.writeheader()
        for poster in rows:
            writer.writerow({
                "image": Path(poster.image).name,
                "title": poster.title or "",
                "link": "",
                "subfield": "",
                "note": "; ".join(poster.notes),
            })
    return len(rows)


def read(path: Path) -> List[Dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8-sig") as handle:
        rows = []
        for row in csv.DictReader(handle):
            rows.append({k: (row.get(k) or "").strip() for k in FIELDS})
        return rows


# -- merge -----------------------------------------------------------------

def resolve_link(fetcher: Fetcher, kind: str, value: str, log: Log) -> Optional[Work]:
    """Turn a pasted identifier into a full record where we can."""
    if kind == "arxiv":
        return arxiv.lookup_by_id(fetcher, value)
    if kind == "doi":
        return openalex.lookup_by_doi(fetcher, value) or crossref.lookup_by_doi(fetcher, value)
    if kind == "openalex":
        try:
            return openalex.to_work(fetcher.get_json(
                "https://api.openalex.org/works/%s" % value))
        except (ValueError, OSError):
            return None
    if kind == "url":
        # OpenReview, CVF Open Access, publisher pages: read what they say.
        return webpage.resolve(fetcher, value, log)
    return None


def merge(posters: Sequence, rows: Sequence[Dict[str, str]], fetcher: Fetcher,
          classify=None, log: Optional[Log] = None) -> int:
    """Apply manual rows to the posters they name. Returns how many changed.

    A row is matched to a poster by image filename. Rows naming an unknown image
    are reported rather than silently dropped - a typo in a filename should not
    look like a link that quietly did nothing.
    """
    log = log or Log()
    by_name = {Path(p.image).name: p for p in posters}
    changed = 0
    unknown = []

    for row in rows:
        name = Path(row.get("image", "")).name
        poster = by_name.get(name)
        if not poster:
            if name:
                unknown.append(name)
            continue

        touched = False
        if row.get("title") and row["title"] != poster.title:
            poster.title = row["title"]
            poster.title_source = NAME
            touched = True

        link = row.get("link", "")
        parsed = parse_link(link)
        if parsed:
            kind, value = parsed
            work = resolve_link(fetcher, kind, value, log)
            if work:
                log.detail("%s: %s %s -> %r" % (name, kind, value, work.title[:60]), indent=1)
            else:
                # An unresolvable link is still the user's answer; keep it as given.
                work = Work(source=NAME, ident=value if kind != "url" else link,
                            title=row.get("title") or poster.title or Path(name).stem,
                            url=link if kind == "url" else _canonical(kind, value),
                            doi=value if kind == "doi" else "")
                log.detail("%s: keeping %s as a plain link" % (name, link), indent=1)
            poster.work = work.to_dict()
            poster.match_source = NAME
            poster.match_how = NAME
            poster.match_score = 1.0
            poster.title = work.title
            poster.title_source = NAME
            touched = True
        elif link:
            log.warn("could not make sense of the link for %s: %r" % (name, link))
            poster.notes.append("manual link not understood: %s" % link)

        if touched and classify:
            classify(poster)
        if row.get("subfield"):
            poster.subfield = row["subfield"]
            poster.subfield_source = NAME
            poster.subfield_confidence = 1.0
            poster.evidence = [NAME]
            touched = True
        if row.get("note"):
            note = "manual note: %s" % row["note"]
            if note not in poster.notes:
                poster.notes.append(note)

        changed += 1 if touched else 0

    if unknown:
        log.warn("%d row(s) name a photo that is not in this run: %s"
                 % (len(unknown), ", ".join(sorted(set(unknown))[:5])))
    return changed


def _canonical(kind: str, value: str) -> str:
    if kind == "arxiv":
        return "https://arxiv.org/abs/%s" % value
    if kind == "doi":
        return "https://doi.org/%s" % value
    if kind == "openalex":
        return "https://openalex.org/%s" % value
    return value
