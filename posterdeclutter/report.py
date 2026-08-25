"""Reports: JSON, Markdown, and a self-contained HTML page - clustered per subfield."""

from __future__ import annotations

import html
import json
import os
from collections import OrderedDict
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

from .subfields import UNCLASSIFIED


def cluster(posters: List) -> "OrderedDict[str, List]":
    groups: Dict[str, List] = {}
    for poster in posters:
        groups.setdefault(poster.subfield, []).append(poster)
    ordered = OrderedDict()
    # Biggest cluster first; "Unclassified" always last, whatever its size.
    for name in sorted(groups, key=lambda n: (n == UNCLASSIFIED, -len(groups[n]), n)):
        ordered[name] = sorted(groups[name], key=lambda p: (p.title or "~").lower())
    return ordered


def _stats(posters: List) -> dict:
    return {
        "posters": len(posters),
        "titled": sum(1 for p in posters if p.title),
        "linked": sum(1 for p in posters if p.work),
        "classified": sum(1 for p in posters if p.subfield != UNCLASSIFIED),
    }


def _href(poster, images: dict, base: Optional[Path] = None) -> str:
    """Where the report should point for this poster's photo: the compressed
    copy in posters/ if there is one, else the photo itself, relative to `base`
    - the folder the report is written to."""
    href = images.get(poster.image, "")
    if href and not href.startswith("data:"):   # an embedded page has no file to point at
        return href
    if base is None:
        return poster.image
    try:
        return os.path.relpath(poster.image, base)
    except ValueError:                   # different drive; nothing relative to say
        return poster.image


def write_json(posters: List, path: Path, images: Optional[dict] = None) -> Path:
    """The records, with `image` rewritten to the path the report links.

    Absolute paths into someone's Photos folder are of no use to a reader and
    do not survive the folder being moved or sent on; a relative one does, and
    is exactly what report.html and report.md show.
    """
    images = images or {}

    def record(poster) -> dict:
        out = poster.to_dict()
        out["image"] = _href(poster, images, path.parent)
        return out

    payload = {
        "generated": date.today().isoformat(),
        "stats": _stats(posters),
        "clusters": {
            name: [record(p) for p in group] for name, group in cluster(posters).items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def render_markdown(posters: List, conference: str = "", images: Optional[dict] = None,
                    base: Optional[Path] = None) -> str:
    """The reader wants the paper, not the machinery that found it: scores,
    match methods and confidences stay in report.json and in --verbose.

    `images` maps a poster's image path to the photo the report links, the same
    map report.json and report.html are built from. See images.prepare.
    """
    images = images or {}
    stats = _stats(posters)
    out = ["# Poster report%s" % (" - " + conference if conference else ""), ""]
    out.append("%d posters - %d titles recovered, %d linked to a paper."
               % (stats["posters"], stats["titled"], stats["linked"]))
    out.append("")
    groups = cluster(posters)
    out.append("## Contents")
    for name, group in groups.items():
        out.append("- %s (%d)" % (name, len(group)))
    out.append("")

    for name, group in groups.items():
        out.append("## %s" % name)
        out.append("")
        for poster in group:
            out.append("### %s" % (poster.title or "*(no title recovered)*"))
            out.append("")
            out.append("- Photo: [%s](%s)" % (_href(poster, images, base),
                                              _href(poster, images, base)))
            if poster.work:
                work = poster.work
                authors = ", ".join(work["authors"][:6])
                if len(work["authors"]) > 6:
                    authors += ", et al."
                label = "arXiv:%s" % work["ident"] if work["source"] == "arxiv" else work["ident"]
                out.append("- Paper: [%s](%s)" % (label, work["url"]))
                out.append("- Authors: %s" % (authors or "unknown"))
                if work["venue"]:
                    out.append("- Venue: %s" % work["venue"])
                if work["categories"]:
                    out.append("- Categories: %s" % ", ".join(work["categories"]))
                if work["published"]:
                    out.append("- Published: %s" % work["published"])
                if work["doi"]:
                    out.append("- DOI: https://doi.org/%s" % work["doi"])
                if work["pdf_url"]:
                    out.append("- PDF: %s" % work["pdf_url"])
                if work["summary"]:
                    out.append("")
                    out.append("> %s" % _clip(work["summary"], 420))
            else:
                out.append("- Paper: not found")
            for note in poster.notes:
                out.append("- Note: %s" % note)
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def write_markdown(posters: List, path: Path, conference: str = "",
                   images: Optional[dict] = None) -> Path:
    path.write_text(render_markdown(posters, conference, images, path.parent),
                    encoding="utf-8")
    return path


_CSS = """
:root { color-scheme: light dark; --fg:#16181d; --bg:#fbfbfa; --mut:#5c6270;
        --line:#e3e2de; --card:#ffffff; --accent:#9a4a2f; }
@media (prefers-color-scheme: dark) {
  :root { --fg:#e8e8e6; --bg:#16181b; --mut:#9aa0ac; --line:#2c3037; --card:#1d2025; --accent:#e08a63; }
}
* { box-sizing: border-box; }
body { margin:0; padding:2.5rem 1.25rem 5rem; background:var(--bg); color:var(--fg);
       font:16px/1.6 ui-sans-serif, -apple-system, Segoe UI, sans-serif; }
main { max-width: 60rem; margin: 0 auto; }
h1 { font-size:1.9rem; margin:0 0 .3rem; letter-spacing:-.02em; }
h2 { font-size:1.2rem; margin:2.6rem 0 .9rem; padding-bottom:.35rem;
     border-bottom:1px solid var(--line); }
.lede { color:var(--mut); margin:0 0 2rem; }
.toc { display:flex; flex-wrap:wrap; gap:.4rem; margin-bottom:1rem; padding:0; list-style:none; }
.toc a { text-decoration:none; font-size:.85rem; padding:.25rem .6rem; border:1px solid var(--line);
         border-radius:999px; color:var(--fg); background:var(--card); }
.card { background:var(--card); border:1px solid var(--line); border-radius:10px;
        padding:1rem 1.15rem; margin-bottom:.85rem;
        display:grid; grid-template-columns:auto 1fr; gap:1rem; align-items:start; }
.card.noshot { grid-template-columns:1fr; }
.shot { display:block; width:170px; border-radius:6px; border:1px solid var(--line);
        background:var(--bg); }
.shot img { display:block; width:100%; height:auto; border-radius:5px; }
.body { min-width:0; }
.card h3 { margin:0 0 .4rem; font-size:1.02rem; line-height:1.35; }
@media (max-width: 34rem) {
  .card { grid-template-columns:1fr; }
  .shot { width:100%; max-width:22rem; }
}
.meta { color:var(--mut); font-size:.85rem; margin:.15rem 0; word-break:break-word; }
.meta a { color:var(--accent); }
.abstract { font-size:.88rem; color:var(--mut); margin:.6rem 0 0;
            border-left:2px solid var(--line); padding-left:.8rem; }
.tag { font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; color:var(--mut); }
.unmatched h3 { color:var(--mut); }
footer { margin-top:3rem; padding-top:1rem; border-top:1px solid var(--line);
         color:var(--mut); font-size:.82rem; }
footer a { color:var(--accent); }
"""

_FOOTER = (
    "<footer>Report generated automatically with the "
    "<a href='https://github.com/jacksalici/ConfPostersDeclutterer'>conference poster "
    "declutterer</a> by <a href='https://jacksalici.com'>Jack</a>.</footer>"
)


def render_html(posters: List, conference: str = "", images: Optional[dict] = None) -> str:
    """Same restraint as the Markdown: the paper, not the machinery.

    `images` maps a poster's image path to whatever the page should load - a
    relative path to the compressed photo, or a data URI. See images.prepare.
    The card shows it small; a click opens the same file full size.
    """
    images = images or {}
    stats = _stats(posters)
    groups = cluster(posters)
    e = html.escape
    parts = [
        "<!doctype html><html lang=en><meta charset=utf-8>",
        "<meta name=viewport content='width=device-width,initial-scale=1'>",
        "<title>Poster report%s</title>" % (" - " + e(conference) if conference else ""),
        "<style>%s</style><main>" % _CSS,
        "<h1>Poster report%s</h1>" % (" &middot; " + e(conference) if conference else ""),
        "<p class=lede>%d posters &middot; %d titles recovered &middot; %d linked to a paper</p>"
        % (stats["posters"], stats["titled"], stats["linked"]),
        "<ul class=toc>",
    ]
    for name, group in groups.items():
        parts.append("<li><a href='#%s'>%s (%d)</a></li>" % (_anchor(name), e(name), len(group)))
    parts.append("</ul>")

    for name, group in groups.items():
        parts.append("<h2 id='%s'>%s</h2>" % (_anchor(name), e(name)))
        for poster in group:
            shot = images.get(poster.image)
            classes = ["card"]
            if not poster.work:
                classes.append("unmatched")
            if not shot:
                classes.append("noshot")
            parts.append("<article class='%s'>" % " ".join(classes))
            if shot:
                parts.append("<a class=shot href='%s'><img loading=lazy src='%s' alt='%s'></a>"
                             % (e(shot), e(shot),
                                e(poster.title or poster.pid or Path(poster.image).name)))
            parts.append("<div class=body>")
            parts.append("<h3>%s</h3>" % e(poster.title or "(no title recovered)"))
            if poster.work:
                work = poster.work
                authors = ", ".join(work["authors"][:6]) + (", et al." if len(work["authors"]) > 6 else "")
                label = "arXiv:%s" % work["ident"] if work["source"] == "arxiv" else work["ident"]
                bits = [w for w in (work["published"], work["venue"]) if w]
                parts.append("<p class=meta><a href='%s'>%s</a>%s</p>"
                             % (e(work["url"]), e(label),
                                " &middot; " + e(" &middot; ".join(bits)) if bits else ""))
                parts.append("<p class=meta>%s</p>" % e(authors or "unknown authors"))
                if work["categories"]:
                    parts.append("<p class=meta>%s</p>" % e(", ".join(work["categories"])))
                if work["summary"]:
                    parts.append("<p class=abstract>%s</p>" % e(_clip(work["summary"], 480)))
            else:
                parts.append("<p class=meta>Paper not found</p>")
            parts.append("<p class=tag>%s</p>" % e(poster.pid or Path(poster.image).name))
            parts.append("</div></article>")
    parts.append(_FOOTER)
    parts.append("</main></html>")
    return "\n".join(parts)


def write_html(posters: List, path: Path, conference: str = "",
               images: Optional[dict] = None) -> Path:
    path.write_text(render_html(posters, conference, images), encoding="utf-8")
    return path


def _anchor(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "…"
