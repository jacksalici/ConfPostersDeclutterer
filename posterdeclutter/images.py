"""The poster images the HTML report shows.

There are no thumbnails: the report shows the poster photo itself, compressed
just enough that a folder of phone photos is something you can send to someone.
Each one is named after its poster - poster01.jpg - and lands in <out>/posters,
so the report folder travels on its own and nothing in it is called IMG_4417.

Uses `sips`, which ships with macOS - the same reason the default OCR backend is
Vision. Where it is missing, the report links the original photos where they lie
rather than showing nothing.
"""

from __future__ import annotations

import base64
import mimetypes
import re
import shutil
import subprocess
import os
import tempfile
from pathlib import Path
from typing import Dict, Optional, Sequence

from .log import Log
from .util import slugify

MODES = ("files", "embed", "none")
# Gentle on purpose: wide enough to read the small print when the image is
# opened on its own, at a quality that leaves the text crisp. A 4 MB phone photo
# lands around 800 kB. --image-width 0 keeps the original size.
DEFAULT_WIDTH = 1800
DEFAULT_QUALITY = 85


def available() -> bool:
    return bool(shutil.which("sips"))


def pixel_width(path: Path) -> Optional[int]:
    proc = subprocess.run(["sips", "-g", "pixelWidth", str(path)],
                          capture_output=True, text=True)
    found = re.search(r"pixelWidth:\s*(\d+)", proc.stdout)
    return int(found.group(1)) if found else None


def compress(source: Path, target: Path, width: int = DEFAULT_WIDTH,
             quality: int = DEFAULT_QUALITY) -> Optional[Path]:
    """Write a JPEG copy of `source`, at most `width` wide. None if it failed."""
    if not available() or not source.exists():
        return None
    target.parent.mkdir(parents=True, exist_ok=True)
    options = ["-s", "format", "jpeg", "-s", "formatOptions", str(quality)]
    # --resampleWidth scales up as happily as down (and -Z would bound the
    # longest side, not the width); a copy bigger than its source would be
    # daft, so only resize when there is something to shrink.
    if width and (pixel_width(source) or 0) > width:
        options += ["--resampleWidth", str(width)]
    proc = subprocess.run(["sips"] + options + [str(source), "--out", str(target)],
                          capture_output=True, text=True)
    if proc.returncode != 0 or not target.exists():
        return None
    return target


def as_data_uri(path: Path) -> Optional[str]:
    if not path.exists():
        return None
    kind = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    return "data:%s;base64,%s" % (kind, base64.b64encode(path.read_bytes()).decode("ascii"))


def _unique(folder: Path, stem: str, used: set) -> Path:
    """`folder/stem.jpg`, counting up while the name is taken - poster ids are
    unique already, but a fallback slug need not be."""
    target = folder / ("%s.jpg" % stem)
    counter = 2
    while target in used:
        target = folder / ("%s-%d.jpg" % (stem, counter))
        counter += 1
    used.add(target)
    return target


def _stem(poster) -> str:
    """poster01 - or the photo's own name, for a record made before ids existed."""
    return poster.pid or slugify(Path(poster.image).stem, 48)


def prepare(posters: Sequence, out_dir: Path, mode: str = "files",
            width: int = DEFAULT_WIDTH, quality: int = DEFAULT_QUALITY,
            log: Optional[Log] = None) -> Dict[str, str]:
    """Write the images and return {poster image path: src for the report}.

    `files` writes them into <out>/posters and links them relatively, so the
    folder stays portable; `embed` inlines them so the page is a single file;
    `none` leaves them out. Where a copy cannot be made the original photo is
    linked instead.
    """
    log = log or Log()
    if mode not in MODES:
        raise ValueError("unknown image mode %r (choose from %s)" % (mode, ", ".join(MODES)))
    if mode == "none" or not posters:
        return {}

    out_dir = Path(out_dir)
    # An embedded page is meant to be the only file there is, so its images are
    # compressed somewhere temporary and inlined, not left in the folder.
    scratch = tempfile.mkdtemp(prefix="posterdeclutter-") if mode == "embed" else None
    folder = Path(scratch) if scratch else out_dir / "posters"
    sources: Dict[str, str] = {}
    used = set()
    made = kept = missing = 0
    before = after = 0
    for poster in posters:
        original = Path(poster.image)
        if not original.exists():
            continue
        target = _unique(folder, _stem(poster), used)
        if target.exists() and original.samefile(target):
            # Re-rendering a report in place: the compressed copy is the source.
            kept += 1
            copy = target
        else:
            copy = compress(original, target, width, quality)
            if copy:
                made += 1
                before += original.stat().st_size
                after += copy.stat().st_size
        if copy:
            sources[poster.image] = (as_data_uri(copy) if mode == "embed"
                                     else os.path.relpath(copy, out_dir))
        else:
            missing += 1
            sources[poster.image] = os.path.relpath(original, out_dir)
    if scratch:
        shutil.rmtree(scratch, ignore_errors=True)
    if made:
        log.detail("images: %d compressed (%.1f MB -> %.1f MB)%s"
                   % (made, before / 1e6, after / 1e6,
                      " and inlined" if scratch else " into %s" % folder), indent=0)
    if kept:
        log.detail("images: %d already compressed in %s" % (kept, folder), indent=0)
    if missing:
        log.warn("could not compress %d photo(s) (%s); linking the originals instead"
                 % (missing, "sips is not available" if not available()
                    else "sips could not read them"))
    return sources
