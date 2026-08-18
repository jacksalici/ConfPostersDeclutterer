"""Where a poster's paper can be looked up, and in what order.

Every source exposes `search(fetcher, title, threshold) -> Match`, so they are
interchangeable. `resolve` tries the identifiers printed on the poster first
(they are the strongest evidence available), then each source in the order the
user asked for, and stops at the first match above the threshold. It never talks
to a model: the "llm" source proposes identifiers in a single batched call made
by the pipeline, and each proposal comes back through `verify_identifier`.
"""

from __future__ import annotations

from typing import Callable, Dict, List, Optional, Sequence, Tuple

from . import arxiv, crossref, openalex
from .http import Fetcher
from .log import Log
from .util import similarity
from .works import Match, Work

SEARCHERS: Dict[str, Callable[..., Match]] = {
    arxiv.NAME: arxiv.search,
    openalex.NAME: openalex.search,
    crossref.NAME: crossref.search,
}
# "llm" is handled separately: it proposes an identifier, it never supplies data.
NAMES = tuple(SEARCHERS) + ("llm",)
DEFAULT = (arxiv.NAME, openalex.NAME)

# Below this, an identifier printed on the poster is treated as a citation of
# somebody else's paper rather than as the poster's own.
AGREEMENT = 0.35

# A two-word title agrees 0.80 with a three-word one ("Random Forests" vs
# "Neural Random Forests"), so short titles need a higher bar than long ones.
SHORT_TITLE_WORDS = 4
SHORT_TITLE_PENALTY = 0.15


def effective_threshold(title: Optional[str], threshold: float) -> float:
    words = len((title or "").split())
    if 0 < words < SHORT_TITLE_WORDS:
        return min(0.95, threshold + SHORT_TITLE_PENALTY)
    return threshold


def parse_names(spec: str) -> List[str]:
    names = [n.strip().lower() for n in spec.split(",") if n.strip()]
    unknown = [n for n in names if n not in NAMES]
    if unknown:
        raise ValueError("unknown source(s): %s (choose from %s)"
                         % (", ".join(unknown), ", ".join(NAMES)))
    return names or list(DEFAULT)


def _agrees(poster_title: Optional[str], work: Work) -> Tuple[bool, float]:
    """An identifier we did not search for must corroborate the OCR'd title."""
    if not poster_title:
        return True, 1.0
    score = similarity(poster_title, work.title)
    return score >= AGREEMENT, round(score, 4)


def by_identifier(
    fetcher: Fetcher,
    poster_title: Optional[str],
    arxiv_ids: Sequence[str],
    dois: Sequence[str],
    names: Sequence[str],
    notes: List[str],
    log: Optional[Log] = None,
) -> Optional[Match]:
    """Resolve an arXiv ID or DOI read straight off the poster."""
    log = log or Log()
    if arxiv_ids or dois:
        log.detail("identifiers on the poster: %s"
                   % ", ".join(list(arxiv_ids) + list(dois)), indent=2)
    for arxiv_id in arxiv_ids if arxiv.NAME in names else ():
        work = arxiv.lookup_by_id(fetcher, arxiv_id)
        if not work:
            continue
        ok, score = _agrees(poster_title, work)
        if not ok:
            log.detail("arXiv:%s resolves to %r - disagrees (%.2f), treating as a citation"
                       % (arxiv_id, work.title[:60], score), indent=2)
            notes.append(
                "ignored arXiv:%s printed on the poster - its title (%r) does not match "
                "the poster title; probably a citation" % (arxiv_id, work.title)
            )
            continue
        log.detail("arXiv:%s agrees with the poster title (%.2f)" % (arxiv_id, score), indent=2)
        return Match(work, max(score, AGREEMENT), "id-on-poster")

    for doi in dois:
        for name in names:
            lookup = {openalex.NAME: openalex.lookup_by_doi,
                      crossref.NAME: crossref.lookup_by_doi}.get(name)
            if not lookup:
                continue
            work = lookup(fetcher, doi)
            if not work:
                continue
            ok, score = _agrees(poster_title, work)
            if not ok:
                log.detail("DOI %s resolves to %r - disagrees (%.2f), treating as a citation"
                           % (doi, work.title[:60], score), indent=2)
                notes.append(
                    "ignored DOI %s printed on the poster - its title (%r) does not match "
                    "the poster title; probably a citation" % (doi, work.title)
                )
                break
            log.detail("DOI %s agrees with the poster title (%.2f) via %s"
                       % (doi, score, name), indent=2)
            return Match(work, max(score, AGREEMENT), "id-on-poster")
    return None


def verify_identifier(
    fetcher: Fetcher,
    proposal: Tuple[str, str],
    poster_title: Optional[str],
    names: Sequence[str],
    threshold: float,
    notes: List[str],
    log: Optional[Log] = None,
) -> Optional[Match]:
    """Check an identifier the model proposed against the real record.

    The model only ever *proposes*; nothing it says reaches the report unless the
    identifier resolves and the resolved title agrees with the poster. That is
    what makes a hallucinated ID harmless.
    """
    log = log or Log()
    kind, value = proposal
    log.detail("verifying the model's proposal: %s %s" % (kind, value), indent=2)
    if kind == "arxiv":
        work = arxiv.lookup_by_id(fetcher, value)
    else:
        work = openalex.lookup_by_doi(fetcher, value) or crossref.lookup_by_doi(fetcher, value)
    if not work:
        log.detail("%s %s does not resolve - discarded" % (kind, value), indent=3)
        notes.append("llm proposed %s %s, which does not resolve - discarded" % (kind, value))
        return None

    score = similarity(poster_title, work.title) if poster_title else 0.0
    if score < effective_threshold(poster_title, threshold):
        log.detail("resolves to %r - disagrees (%.2f) - discarded"
                   % (work.title[:60], score), indent=3)
        notes.append("llm proposed %s %s (%r), which disagrees with the poster title "
                     "(%.2f) - discarded" % (kind, value, work.title, score))
        return None
    log.detail("verified: %r (%.2f)" % (work.title[:60], score), indent=3)
    return Match(work, round(score, 4), "llm-verified")


def resolve(
    fetcher: Fetcher,
    poster_title: Optional[str],
    arxiv_ids: Sequence[str],
    dois: Sequence[str],
    names: Sequence[str],
    threshold: float,
    notes: List[str],
    log: Optional[Log] = None,
) -> Match:
    log = log or Log()
    match = by_identifier(fetcher, poster_title, arxiv_ids, dois, names, notes, log)
    if match:
        return match

    best = Match(None, 0.0, "none")
    if not poster_title:
        log.detail("no title to search with; skipping the indexes", indent=2)
        return best

    bar = effective_threshold(poster_title, threshold)
    if bar != threshold:
        log.detail("short title (%d words): raising the bar to %.2f"
                   % (len(poster_title.split()), bar), indent=2)
    for name in names:
        searcher = SEARCHERS.get(name)
        if not searcher:
            continue
        candidate = searcher(fetcher, poster_title, bar, log=log)
        if candidate.work:
            return candidate
        if candidate.score > best.score:
            best = candidate
    log.detail("no source cleared %.2f (best %.2f)" % (bar, best.score), indent=2)
    return best
