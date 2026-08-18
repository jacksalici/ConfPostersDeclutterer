"""OCR backends.

Every backend returns a list of ``Line`` objects with normalised geometry
(origin top-left, 0..1), so the title heuristic can reason about font size and
position regardless of which engine produced the text.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Optional, Sequence

from .util import clean_text

IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".heic", ".heif", ".tif", ".tiff", ".webp", ".bmp"}

_HERE = Path(__file__).resolve().parent
_SWIFT_SRC = _HERE / "vendor" / "vision_ocr.swift"


@dataclass
class Line:
    text: str
    conf: float = 1.0
    x: float = 0.0
    y: float = 0.0  # top edge, 0 = top of image
    w: float = 1.0
    h: float = 0.05

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Line":
        return cls(**{k: d[k] for k in ("text", "conf", "x", "y", "w", "h") if k in d})


class OCRError(RuntimeError):
    pass


def find_images(root: Path, recursive: bool = True) -> List[Path]:
    if root.is_file():
        return [root]
    pattern = "**/*" if recursive else "*"
    found = [
        p
        for p in sorted(root.glob(pattern))
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES and not p.name.startswith(".")
    ]
    return found


# --------------------------------------------------------------------------
# Backend: macOS Vision (default on darwin; no third-party dependency)
# --------------------------------------------------------------------------

def _vision_binary(cache_dir: Path) -> Path:
    """Compile the Swift helper once, reuse it afterwards."""
    binary = cache_dir / "vision_ocr"
    if binary.exists() and binary.stat().st_mtime >= _SWIFT_SRC.stat().st_mtime:
        return binary
    if not shutil.which("swiftc"):
        raise OCRError("swiftc not found - install Xcode command line tools, or use --ocr tesseract")
    cache_dir.mkdir(parents=True, exist_ok=True)
    proc = subprocess.run(
        ["swiftc", "-O", "-o", str(binary), str(_SWIFT_SRC)],
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        raise OCRError("failed to build the Vision helper:\n" + proc.stderr.strip())
    return binary


def vision_ocr(paths: Sequence[Path], cache_dir: Path) -> dict:
    """Batch OCR through Apple's Vision framework. Returns {path: [Line]}."""
    if sys.platform != "darwin":
        raise OCRError("the vision backend needs macOS; try --ocr tesseract")
    binary = _vision_binary(cache_dir)
    proc = subprocess.run(
        [str(binary)] + [str(p) for p in paths], capture_output=True, text=True
    )
    if proc.returncode != 0:
        raise OCRError("vision_ocr failed: " + proc.stderr.strip())
    out = {}
    for raw in proc.stdout.splitlines():
        raw = raw.strip()
        if not raw:
            continue
        record = json.loads(raw)
        lines = []
        for item in record.get("lines", []):
            # Vision's origin is bottom-left; flip to top-left.
            lines.append(
                Line(
                    text=clean_text(item["text"]),
                    conf=float(item.get("conf", 1.0)),
                    x=float(item["x"]),
                    y=1.0 - float(item["y"]) - float(item["h"]),
                    w=float(item["w"]),
                    h=float(item["h"]),
                )
            )
        out[record["path"]] = sorted(lines, key=lambda l: (l.y, l.x))
    return out


# --------------------------------------------------------------------------
# Backend: tesseract (cross-platform fallback)
# --------------------------------------------------------------------------

def tesseract_ocr(paths: Sequence[Path], cache_dir: Path) -> dict:
    if not shutil.which("tesseract"):
        raise OCRError("tesseract not found - `brew install tesseract`, or use --ocr vision")
    out = {}
    for path in paths:
        proc = subprocess.run(
            ["tesseract", str(path), "stdout", "tsv"], capture_output=True, text=True
        )
        if proc.returncode != 0:
            raise OCRError("tesseract failed on %s: %s" % (path, proc.stderr.strip()))
        out[str(path)] = _parse_tesseract_tsv(proc.stdout)
    return out


def _parse_tesseract_tsv(tsv: str) -> List[Line]:
    rows = [r.split("\t") for r in tsv.splitlines() if r.strip()]
    if not rows:
        return []
    header, rows = rows[0], rows[1:]
    idx = {name: i for i, name in enumerate(header)}
    page_w = page_h = 1.0
    grouped = {}
    for row in rows:
        if len(row) < len(header):
            continue
        level = int(row[idx["level"]])
        if level == 1:  # page geometry
            page_w = max(float(row[idx["width"]]), 1.0)
            page_h = max(float(row[idx["height"]]), 1.0)
            continue
        if level != 5:  # word
            continue
        text = row[idx["text"]].strip()
        if not text:
            continue
        conf = float(row[idx["conf"]])
        if conf < 0:
            continue
        key = tuple(row[idx[k]] for k in ("block_num", "par_num", "line_num"))
        grouped.setdefault(key, []).append(
            (
                text,
                conf / 100.0,
                float(row[idx["left"]]),
                float(row[idx["top"]]),
                float(row[idx["width"]]),
                float(row[idx["height"]]),
            )
        )
    lines = []
    for words in grouped.values():
        text = clean_text(" ".join(w[0] for w in words))
        if not text:
            continue
        left = min(w[2] for w in words)
        top = min(w[3] for w in words)
        right = max(w[2] + w[4] for w in words)
        bottom = max(w[3] + w[5] for w in words)
        lines.append(
            Line(
                text=text,
                conf=sum(w[1] for w in words) / len(words),
                x=left / page_w,
                y=top / page_h,
                w=(right - left) / page_w,
                h=(bottom - top) / page_h,
            )
        )
    return sorted(lines, key=lambda l: (l.y, l.x))


# --------------------------------------------------------------------------
# Backend: sidecar text files (offline tests, or hand-corrected transcripts)
# --------------------------------------------------------------------------

def sidecar_ocr(paths: Sequence[Path], cache_dir: Path) -> dict:
    """Read <image>.txt next to each image; each line becomes an OCR line.

    Line height is faked as decreasing with position so the title heuristic
    still has a size signal: the first line is treated as the largest.
    """
    out = {}
    for path in paths:
        sidecar = path.with_suffix(path.suffix + ".txt")
        if not sidecar.exists():
            sidecar = path.with_suffix(".txt")
        if not sidecar.exists():
            out[str(path)] = []
            continue
        raw_lines = [clean_text(l) for l in sidecar.read_text(encoding="utf-8").splitlines()]
        raw_lines = [l for l in raw_lines if l]
        lines = []
        cursor = 0.0
        for i, text in enumerate(raw_lines):
            height = 0.05 if i == 0 else 0.02  # first line stands in for the title
            lines.append(Line(text=text, conf=1.0, x=0.1, y=cursor, w=0.8, h=height))
            cursor += height + 0.01
        out[str(path)] = lines
    return out


BACKENDS = {
    "vision": vision_ocr,
    "tesseract": tesseract_ocr,
    "sidecar": sidecar_ocr,
}


def default_backend() -> str:
    if sys.platform == "darwin" and shutil.which("swiftc"):
        return "vision"
    if shutil.which("tesseract"):
        return "tesseract"
    return "sidecar"


def run_ocr(paths: Sequence[Path], backend: Optional[str], cache_dir: Path) -> dict:
    name = backend or default_backend()
    if name not in BACKENDS:
        raise OCRError("unknown OCR backend %r (choose from %s)" % (name, ", ".join(BACKENDS)))
    return BACKENDS[name](paths, cache_dir)
