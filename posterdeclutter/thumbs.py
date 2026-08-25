"""Poster images for the HTML report: a small thumbnail on the card, and the
photo behind it that a click opens.

That photo is not the camera original but a lightly recompressed copy, named
after the poster rather than after whatever the phone called it - a report
folder you can hand to someone else, without the gigabytes.

Uses `sips`, which ships with macOS - the same reason the default OCR backend is
Vision. Where it is missing, the report falls back to linking the original photo
rather than showing nothing.
"""

from __future__ import annotations

import base64
import mimetypes
import re
import shutil
import subprocess
import os
from pathlib import Path
from typing import Dict, Optional, Sequence

from .log import Log
from .util import slugify

MODES = ("files", "embed", "none")
DEFAULT_WIDTH = 560
# The full photo, gently: wide enough to read the smallest caption on a poster
# when opened full screen, and a quality that leaves the text crisp. This is
# housekeeping, not an optimisation - a 4 MB phone photo lands around 1 MB.
PHOTO_WIDTH = 2400
PHOTO_QUALITY = 90


def available() -> bool:
    return bool(shutil.which("sips"))


def pixel_width(path: Path) -> Optional[int]:
    proc = subprocess.run(["sips", "-g", "pixelWidth", str(path)],
                          capture_output=True, text=True)
    found = re.search(r"pixelWidth:\s*(\d+)", proc.stdout)
    return int(found.group(1)) if found else None


def _convert(source: Path, target: Path, width: int,
             quality: Optional[int] = None) -> Optional[Path]:
    """JPEG-ify `source` into `target`, at most `width` wide. None if it failed."""
    if not available() or not source.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    # -Z scales up as happily as down; a copy bigger than its source would be
    # daft, so only resize when there is something to shrink.
    original = pixel_width(source)
    options = ["-s", "format", "jpeg"]
    if quality is not None:
        options += ["-s", "formatOptions", str(quality)]
    if (original or 0) > width:
        options += ["-Z", str(width)]
    proc = subprocess.run(["sips"] + options + [str(source), "--out", str(target)],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not target.exists():
        return None
    return target


def make(source: Path, target: Path, width: int = DEFAULT_WIDTH) -> Optional[Path]:
    """Downscale `source` into `target`. Returns None if it could not be made."""
    return _convert(source, target, width)


def compress(source: Path, target: Path, width: int = PHOTO_WIDTH,
             quality: int = PHOTO_QUALITY) -> Optional[Path]:
    """A full-size but lightly recompressed copy of `source`."""
    return _convert(source, target, width, quality)


def as_data_uri(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    kind = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return "data:%s;base64,%s" % (kind, base64.b64encode(path.read_bytes()).decode("ascii"))


def _unique(folder: Path, stem: str, used: set) -> Path:
    """`folder/stem.jpg`, counting up while the name is taken - two photos can
    share a slug, and two posters can share a title."""
    target = folder / ("%s.jpg" % stem)
    counter = 2
    while target in used:
        target = folder / ("%s-%d.jpg" % (stem, counter))
        counter += 1
    used.add(target)
    return target


def photos(posters: Sequence, out_dir: Path, mode: str = "files",
           log: Optional[Log] = None) -> Dict[str, str]:
    """Write the click-through photos and return {poster image path: href}.

    One lightly compressed JPEG per poster in <out>/posters, named after the
    poster, so the report folder travels on its own. Where a copy cannot be
    made the camera original is linked instead, as it always was.
    """
    log = log or Log()
    if mode not in MODES:
        raise ValueError("unknown thumbnail mode %r (choose from %s)" % (mode, ", ".join(MODES)))
    if mode == "none" or not posters:
        return {}

    out_dir = Path(out_dir)
    folder = out_dir / "posters"
    hrefs: Dict[str, str] = {}
    used = set()
    made = missing = 0
    before = after = 0
    for poster in posters:
        original = Path(poster.image)
        if not original.exists():
            continue
        target = _unique(folder, slugify(poster.title or original.stem, 64), used)
        copy = compress(original, target)
        if copy:
            made += 1
            before += original.stat().st_size
            after += copy.stat().st_size
            hrefs[poster.image] = os.path.relpath(copy, out_dir)
        else:
            missing += 1
            hrefs[poster.image] = os.path.relpath(original, out_dir)
    if made:
        log.detail("photos: %d compressed into %s (%.1f MB -> %.1f MB)"
                   % (made, folder, before / 1e6, after / 1e6), indent=0)
    if missing:
        log.warn("could not compress %d photo(s) (%s); linking the originals instead"
                 % (missing, "sips is not available" if not available()
                    else "sips could not read them"))
    return hrefs


def prepare(posters: Sequence, out_dir: Path, mode: str = "files",
            log: Optional[Log] = None) -> Dict[str, str]:
    """Make the thumbnails and return {poster image path: src for the report}.

    `files` writes them next to the report and links relatively - small HTML,
    and the folder stays portable. `embed` inlines them so the page is a single
    file. Where a thumbnail cannot be made, the original photo is linked instead.
    """
    log = log or Log()
    if mode not in MODES:
        raise ValueError("unknown thumbnail mode %r (choose from %s)" % (mode, ", ".join(MODES)))
    if mode == "none" or not posters:
        return {}

    out_dir = Path(out_dir)
    folder = out_dir / "thumbs"
    sources: Dict[str, str] = {}
    used = set()
    made = missing = 0
    for poster in posters:
        original = Path(poster.image)
        target = _unique(folder, slugify(original.stem, 48), used)
        thumb = make(original, target)
        if thumb:
            made += 1
            sources[poster.image] = (
                as_data_uri(thumb) if mode == "embed"
                else os.path.relpath(thumb, out_dir)
            )
        elif original.exists():
            missing += 1
            sources[poster.image] = os.path.relpath(original, out_dir)
    if made:
        log.detail("thumbnails: %d made in %s" % (made, folder), indent=0)
    if missing:
        log.warn("could not make %d thumbnail(s) (%s); linking the full photos instead"
                 % (missing, "sips is not available" if not available()
                    else "sips could not read them"))
    return sources
