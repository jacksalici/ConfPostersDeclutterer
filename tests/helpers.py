"""Shared test plumbing. No network, no AI - everything is a local fixture."""

from __future__ import annotations

import hashlib
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURES = ROOT / "fixtures"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def seed(cache_dir: Path, url: str, fixture: str) -> Path:
    """Pre-fill the Fetcher's on-disk cache so offline mode can serve `url`."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
    target = cache_dir / (key + ".body")
    shutil.copyfile(FIXTURES / fixture, target)
    return target


def copy_photos(destination: Path, names=None) -> Path:
    """Copy the fixture photo blob (images + OCR sidecars) into a temp dir."""
    destination.mkdir(parents=True, exist_ok=True)
    for item in sorted((FIXTURES / "photos").iterdir()):
        if names and item.name.split(".")[0] not in names:
            continue
        shutil.copyfile(item, destination / item.name)
    return destination


def seed_all(cache_dir: Path, fetcher, extra_empty=()) -> None:
    """Seed every lookup the fixture posters will make, for all three sources."""
    from posterdeclutter import arxiv, crossref, openalex

    qubit = "Superconducting Qubit Readout with Travelling-Wave Amplifiers"
    widgets = "A Study of Widgets in the Wild"

    seed(cache_dir, arxiv._id_url("2401.12345"), "arxiv_2401.12345.xml")
    seed(cache_dir, arxiv._query_url(arxiv._title_query(qubit)), "arxiv_qubit.xml")
    seed(cache_dir, openalex._search_url(fetcher, qubit), "openalex_qubit.json")
    seed(cache_dir, crossref._search_url(fetcher, widgets), "crossref_widgets.json")
    seed(cache_dir, crossref._doi_url(fetcher, "10.1145/3292500.3330701"),
         "crossref_doi_widgets.json")

    # Titles that must find nothing anywhere.
    blanks = [widgets, "An Ethnography of Poster Sessions",
              "Deep Sets for Jet Flavour Tagging at the LHC"] + list(extra_empty)
    for title in blanks:
        seed(cache_dir, arxiv._query_url(arxiv._title_query(title)), "arxiv_empty.xml")
        seed(cache_dir, arxiv._query_url(arxiv._loose_query(title)), "arxiv_empty.xml")
        seed(cache_dir, openalex._search_url(fetcher, title), "openalex_empty.json")
        seed(cache_dir, crossref._search_url(fetcher, title), "crossref_empty.json")
