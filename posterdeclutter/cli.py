"""Command line entry point: python3 -m posterdeclutter ..."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from . import __version__, images as images_mod, manual, report
from .llm import LLM, PROVIDERS, DEFAULT_MODEL
from .log import from_flags
from .pipeline import REDO_MODES, Poster, number
from .sources import DEFAULT as DEFAULT_SOURCES, NAMES as SOURCE_NAMES, parse_names
from .ocr import BACKENDS, OCRError, default_backend, find_images
from .pipeline import Pipeline


def add_image_options(parser) -> None:
    """How hard to squeeze the poster photos. The defaults are deliberately
    gentle; both are here for the two ends of the page-weight argument."""
    parser.add_argument("--image-width", type=int, default=images_mod.DEFAULT_WIDTH,
                        metavar="PX",
                        help="widest a poster image may be, in pixels (default: %d; "
                             "0 keeps the original size)" % images_mod.DEFAULT_WIDTH)
    parser.add_argument("--image-quality", type=int, default=images_mod.DEFAULT_QUALITY,
                        metavar="1-100",
                        help="JPEG quality for the poster images (default: %d)"
                             % images_mod.DEFAULT_QUALITY)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="posterdeclutter",
        description="Turn a blob of conference poster photos into a report clustered by subfield.",
    )
    parser.add_argument("--version", action="version", version=__version__)
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="OCR -> title -> arXiv -> report")
    run.add_argument("photos", type=Path, help="folder of poster photos (or a single image)")
    run.add_argument("-o", "--out", type=Path, default=Path("poster-report"),
                     help="output folder (default: ./poster-report)")
    run.add_argument("--ocr", choices=sorted(BACKENDS), default=None,
                     help="OCR backend (default: %s here)" % default_backend())
    run.add_argument("--sources", default=",".join(DEFAULT_SOURCES),
                     help="lookup sources to try, in order, comma separated: %s "
                          "(default: %s)" % (", ".join(SOURCE_NAMES), ",".join(DEFAULT_SOURCES)))
    run.add_argument("--mailto", default="",
                     help="contact address; joins the faster OpenAlex/Crossref polite pools")
    run.add_argument("--llm", choices=PROVIDERS, default="off",
                     help="optional LLM assist for the cases heuristics miss (default: off)")
    run.add_argument("--model", default=DEFAULT_MODEL, help="model id for --llm api")
    run.add_argument("--conference", default="", help="name to put in the report header")
    run.add_argument("--threshold", type=float, default=0.72,
                     help="title/arXiv similarity needed to accept a match (default: 0.72)")
    run.add_argument("--merge", type=Path, metavar="CSV",
                     help="a CSV of manual links to apply (see <out>/unmatched.csv). "
                          "Manual rows win over everything the tool worked out")
    run.add_argument("--images", choices=images_mod.MODES, default="files",
                     help="poster images in the HTML report: files (default, compressed "
                          "into <out>/posters as poster01.jpg and linked), embed (inlined, "
                          "for a single self-contained page), or none")
    add_image_options(run)
    run.add_argument("--offline", action="store_true",
                     help="use only cached lookup responses; never touch the network")
    run.add_argument("--redo", choices=REDO_MODES, default="none",
                     help="none: resume where you left off. research: keep the cached OCR "
                          "text and redo titles, lookups and clustering. all: re-OCR too")
    run.add_argument("--refresh-web", action="store_true",
                     help="drop cached lookup responses so --redo research really re-queries")
    run.add_argument("--fresh", action="store_true", help="alias for --redo all")
    run.add_argument("--no-recursive", action="store_true", help="do not descend into subfolders")
    noise = run.add_mutually_exclusive_group()
    noise.add_argument("-v", "--verbose", action="store_true",
                       help="explain every decision: queries run, cache hits, candidate "
                            "scores, why a match was taken or declined, and a run summary")
    noise.add_argument("-q", "--quiet", action="store_true", help="warnings only")

    ocr = sub.add_parser("ocr", help="dump OCR text and title candidates for one image")
    ocr.add_argument("image", type=Path)
    ocr.add_argument("--ocr", dest="backend", choices=sorted(BACKENDS), default=None)
    ocr.add_argument("--cache", type=Path, default=Path(".posterdeclutter"))
    ocr.add_argument("-v", "--verbose", action="store_true", help="also show the geometry")

    again = sub.add_parser("report",
                           help="re-render report.md/html from an earlier report.json")
    again.add_argument("json", type=Path, nargs="?", default=Path("report"),
                       help="the report.json, or the folder holding it (default: ./report)")
    again.add_argument("-o", "--out", type=Path, default=None,
                       help="output folder (default: next to the json)")
    again.add_argument("--conference", default="", help="name to put in the report header")
    again.add_argument("--images", choices=images_mod.MODES, default="files",
                       help="poster images in the HTML report: files (default, compressed "
                            "into <out>/posters as poster01.jpg and linked), embed (inlined, "
                            "for a single self-contained page), or none")
    add_image_options(again)
    noise = again.add_mutually_exclusive_group()
    noise.add_argument("-v", "--verbose", action="store_true",
                       help="say what is written where")
    noise.add_argument("-q", "--quiet", action="store_true", help="warnings only")
    return parser


def cmd_run(args) -> int:
    try:
        source_names = parse_names(args.sources)
    except ValueError as exc:
        print(str(exc), file=sys.stderr)
        return 1
    if "llm" in source_names and args.llm == "off":
        print("--sources includes 'llm' but --llm is off; pick a provider "
              "(e.g. --llm claude-cli)", file=sys.stderr)
        return 1

    log = from_flags(quiet=args.quiet, verbose=args.verbose)
    images = find_images(args.photos, recursive=not args.no_recursive)
    if not images:
        print("no images found under %s" % args.photos, file=sys.stderr)
        return 1
    log.head("%d photo(s) under %s" % (len(images), args.photos))

    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    pipeline = Pipeline(
        cache_dir=out / "cache",
        backend=args.ocr,
        llm=LLM(args.llm, model=args.model),
        offline=args.offline,
        threshold=args.threshold,
        source_names=source_names,
        mailto=args.mailto,
        log=log,
    )
    posters = pipeline.run(
        images,
        redo="all" if args.fresh else args.redo,
        refresh_web=args.refresh_web,
    )

    merged = 0
    if args.merge:
        if not args.merge.exists():
            print("no such file: %s" % args.merge, file=sys.stderr)
            return 1
        rows = manual.read(args.merge)
        log.head("merging %d manual row(s) from %s" % (len(rows), args.merge))
        merged = manual.merge(posters, rows, pipeline.fetcher,
                              classify=pipeline.classify, log=log)
        if merged:
            pipeline.persist(posters)

    number(posters)
    shots = images_mod.prepare(posters, out, mode=args.images, width=args.image_width,
                               quality=args.image_quality, log=log)

    report.write_json(posters, out / "report.json", shots)
    report.write_markdown(posters, out / "report.md", args.conference, shots)
    report.write_html(posters, out / "report.html", args.conference, shots)
    for name in ("report.json", "report.md", "report.html"):
        log.detail("wrote %s (%d bytes)" % (out / name, (out / name).stat().st_size), indent=0)

    still_missing = manual.write_unmatched(posters, out / "unmatched.csv")

    linked = sum(1 for p in posters if p.work)
    print("%d posters, %d matched (%s)%s"
          % (len(posters), linked, ", ".join(source_names),
             ", %d from --merge" % merged if merged else ""))
    print("report: %s" % (out / "report.md"))
    print("        %s" % (out / "report.html"))
    if still_missing:
        print("%d unmatched: add links in %s, then re-run with --merge %s"
              % (still_missing, out / "unmatched.csv", out / "unmatched.csv"))
    return 0


def cmd_report(args) -> int:
    """Rebuild report.md/report.html from a saved report.json. No OCR, no lookups."""
    log = from_flags(quiet=args.quiet, verbose=args.verbose)
    src = args.json
    if src.is_dir():
        src = src / "report.json"
    if not src.exists():
        print("no such file: %s" % src, file=sys.stderr)
        return 1
    try:
        payload = json.loads(src.read_text(encoding="utf-8"))
        posters = [Poster.from_dict(record)
                   for group in payload["clusters"].values() for record in group]
    except (ValueError, KeyError, TypeError, AttributeError):
        print("not a posterdeclutter report.json: %s" % src, file=sys.stderr)
        return 1
    if not posters:
        print("no poster records in %s" % src, file=sys.stderr)
        return 1

    # The paths in report.json are relative to it, which is the point of them:
    # the folder can be moved, or sent to someone, and still render.
    for poster in posters:
        if not Path(poster.image).is_absolute():
            poster.image = str(src.parent / poster.image)
    number(posters)

    out = args.out or src.parent
    out.mkdir(parents=True, exist_ok=True)
    log.head("re-rendering %d poster(s) from %s" % (len(posters), src))

    shots = images_mod.prepare(posters, out, mode=args.images, width=args.image_width,
                               quality=args.image_quality, log=log)
    report.write_markdown(posters, out / "report.md", args.conference, shots)
    report.write_html(posters, out / "report.html", args.conference, shots)
    for name in ("report.md", "report.html"):
        log.detail("wrote %s (%d bytes)" % (out / name, (out / name).stat().st_size), indent=0)

    print("%d posters, %d matched" % (len(posters),
                                      sum(1 for p in posters if p.work)))
    print("report: %s" % (out / "report.md"))
    print("        %s" % (out / "report.html"))
    return 0


def cmd_ocr(args) -> int:
    from .ocr import run_ocr
    from .titles import read_page

    raw = run_ocr([args.image], args.backend, args.cache / "ocr")
    lines = raw.get(str(args.image), [])
    reading = read_page(lines)
    if args.verbose:
        print("--- lines (x, y, w, h, confidence) ---")
        for line in lines:
            print("%.3f %.3f %.3f %.3f  %.2f  %s"
                  % (line.x, line.y, line.w, line.h, line.conf, line.text))
    print("--- text ---")
    print(reading.full_text)
    print("--- title candidates ---")
    for candidate in reading.candidates:
        print("%6.3f  %s" % (candidate.score, candidate.text))
    print("--- picked ---")
    print(reading.title or "(none)")
    if reading.arxiv_ids or reading.dois:
        print("--- identifiers on poster ---")
        for ident in list(reading.arxiv_ids) + list(reading.dois or []):
            print(ident)
    return 0


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.command == "run":
            return cmd_run(args)
        if args.command == "report":
            return cmd_report(args)
        return cmd_ocr(args)
    except OCRError as exc:
        print("OCR error: %s" % exc, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
