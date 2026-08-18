"""Reports: JSON, Markdown, and a self-contained HTML page - clustered per subfield."""

from __future__ import annotations

import html
import json
from collections import OrderedDict
from datetime import date
from pathlib import Path
from typing import Dict, List

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


def write_json(posters: List, path: Path) -> Path:
    payload = {
        "generated": date.today().isoformat(),
        "stats": _stats(posters),
        "clusters": {
            name: [p.to_dict() for p in group] for name, group in cluster(posters).items()
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def render_markdown(posters: List, conference: str = "") -> str:
    stats = _stats(posters)
    out = ["# Poster report%s" % (" - " + conference if conference else ""), ""]
    out.append(
        "%d posters - %d titles recovered, %d matched on arXiv, %d classified."
        % (stats["posters"], stats["titled"], stats["linked"], stats["classified"])
    )
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
            title = poster.title or "*(no title recovered)*"
            out.append("### %s" % title)
            out.append("")
            out.append("- Photo: `%s`" % Path(poster.image).name)
            if poster.work:
                work = poster.work
                authors = ", ".join(work["authors"][:6])
                if len(work["authors"]) > 6:
                    authors += ", et al."
                label = "arXiv:%s" % work["ident"] if work["source"] == "arxiv" else work["ident"]
                out.append("- Paper: [%s](%s) via %s (%s, match %.2f)"
                           % (label, work["url"], work["source"], poster.match_how, poster.match_score))
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
                out.append("- Paper: no confident match (best score %.2f)" % poster.match_score)
            out.append("- Subfield via %s (confidence %.2f)%s"
                       % (poster.subfield_source, poster.subfield_confidence,
                          ": " + ", ".join(poster.evidence) if poster.evidence else ""))
            for note in poster.notes:
                out.append("- Note: %s" % note)
            out.append("")
    return "\n".join(out).rstrip() + "\n"


def write_markdown(posters: List, path: Path, conference: str = "") -> Path:
    path.write_text(render_markdown(posters, conference), encoding="utf-8")
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
        padding:1rem 1.15rem; margin-bottom:.85rem; }
.card h3 { margin:0 0 .4rem; font-size:1.02rem; line-height:1.35; }
.meta { color:var(--mut); font-size:.85rem; margin:.15rem 0; word-break:break-word; }
.meta a { color:var(--accent); }
.abstract { font-size:.88rem; color:var(--mut); margin:.6rem 0 0;
            border-left:2px solid var(--line); padding-left:.8rem; }
.tag { font-size:.72rem; text-transform:uppercase; letter-spacing:.06em; color:var(--mut); }
.unmatched h3 { color:var(--mut); }
"""


def render_html(posters: List, conference: str = "") -> str:
    stats = _stats(posters)
    groups = cluster(posters)
    e = html.escape
    parts = [
        "<!doctype html><html lang=en><meta charset=utf-8>",
        "<meta name=viewport content='width=device-width,initial-scale=1'>",
        "<title>Poster report%s</title>" % (" - " + e(conference) if conference else ""),
        "<style>%s</style><main>" % _CSS,
        "<h1>Poster report%s</h1>" % (" &middot; " + e(conference) if conference else ""),
        "<p class=lede>%d posters &middot; %d titles recovered &middot; %d matched on arXiv "
        "&middot; %d classified</p>" % (stats["posters"], stats["titled"], stats["linked"], stats["classified"]),
        "<ul class=toc>",
    ]
    for name, group in groups.items():
        parts.append("<li><a href='#%s'>%s (%d)</a></li>" % (_anchor(name), e(name), len(group)))
    parts.append("</ul>")

    for name, group in groups.items():
        parts.append("<h2 id='%s'>%s</h2>" % (_anchor(name), e(name)))
        for poster in group:
            classes = "card" if poster.work else "card unmatched"
            parts.append("<article class='%s'>" % classes)
            parts.append("<h3>%s</h3>" % e(poster.title or "(no title recovered)"))
            parts.append("<p class=meta>Photo: %s</p>" % e(Path(poster.image).name))
            if poster.work:
                work = poster.work
                authors = ", ".join(work["authors"][:6]) + (", et al." if len(work["authors"]) > 6 else "")
                label = "arXiv:%s" % work["ident"] if work["source"] == "arxiv" else work["ident"]
                bits = [w for w in (work["published"], work["venue"]) if w]
                parts.append(
                    "<p class=meta><a href='%s'>%s</a>%s &middot; %s &middot; match %.2f</p>"
                    % (e(work["url"]), e(label),
                       " &middot; " + e(" &middot; ".join(bits)) if bits else "",
                       e(work["source"]), poster.match_score)
                )
                parts.append("<p class=meta>%s</p>" % e(authors or "unknown authors"))
                if work["categories"]:
                    parts.append("<p class=meta>%s</p>" % e(", ".join(work["categories"])))
                if work["summary"]:
                    parts.append("<p class=abstract>%s</p>" % e(_clip(work["summary"], 480)))
            else:
                parts.append("<p class=meta>No confident match (best %.2f)</p>" % poster.match_score)
            parts.append("<p class=tag>%s &middot; %s &middot; confidence %.2f</p>"
                         % (e(poster.match_how), e(poster.subfield_source), poster.subfield_confidence))
            parts.append("</article>")
    parts.append("</main></html>")
    return "\n".join(parts)


def write_html(posters: List, path: Path, conference: str = "") -> Path:
    path.write_text(render_html(posters, conference), encoding="utf-8")
    return path


def _anchor(name: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "-" for ch in name).strip("-")


def _clip(text: str, limit: int) -> str:
    text = " ".join(text.split())
    if len(text) <= limit:
        return text
    return text[: limit - 1].rsplit(" ", 1)[0] + "…"
