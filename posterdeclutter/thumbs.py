"""Poster thumbnails for the HTML report.

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


def available() -> bool:
    return bool(shutil.which("sips"))


def pixel_width(path: Path) -> Optional[int]:
    proc = subprocess.run(["sips", "-g", "pixelWidth", str(path)],
                          capture_output=True, text=True)
    found = re.search(r"pixelWidth:\s*(\d+)", proc.stdout)
    return int(found.group(1)) if found else None


def make(source: Path, target: Path, width: int = DEFAULT_WIDTH) -> Optional[Path]:
    """Downscale `source` into `target`. Returns None if it could not be made."""
    if not available() or not source.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    # -Z scales up as happily as down; a thumbnail bigger than its source would
    # be daft, so only resize when there is something to shrink.
    original = pixel_width(source)
    resize = ["-Z", str(width)] if (original or 0) > width else []
    proc = subprocess.run(
        ["sips", "-s", "format", "jpeg"] + resize + [str(source), "--out", str(target)],
        capture_output=True, text=True,
    )
    if proc.returncode != 0 or not target.exists():
        return None
    return target


def as_data_uri(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    kind = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return "data:%s;base64,%s" % (kind, base64.b64encode(path.read_bytes()).decode("ascii"))


def originals(posters: Sequence, out_dir: Path) -> Dict[str, str]:
    """{poster image path: href to the full photo}, relative to the report."""
    out = {}
    for poster in posters:
        path = Path(poster.image)
        if path.exists():
            out[poster.image] = os.path.relpath(path, out_dir)
    return out


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
        stem = slugify(original.stem, 48)
        target = folder / ("%s.jpg" % stem)
        counter = 2
        while target in used:            # two photos can share a slug
            target = folder / ("%s-%d.jpg" % (stem, counter))
            counter += 1
        used.add(target)
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
