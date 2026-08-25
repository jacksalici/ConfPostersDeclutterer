"""Command line entry point: python3 -m posterdeclutter ..."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__, manual, report, thumbs
from .llm import LLM, PROVIDERS, DEFAULT_MODEL
from .log import from_flags
from .pipeline import REDO_MODES
from .sources import DEFAULT as DEFAULT_SOURCES, NAMES as SOURCE_NAMES, parse_names
from .ocr import BACKENDS, OCRError, default_backend, find_images
from .pipeline import Pipeline, organise


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
    run.add_argument("--organise", choices=("plan", "copy", "move", "symlink"), default="plan",
                     help="lay photos out as <out>/photos/<subfield>/<title>.jpg (default: plan)")
    run.add_argument("--merge", type=Path, metavar="CSV",
                     help="a CSV of manual links to apply (see <out>/unmatched.csv). "
                          "Manual rows win over everything the tool worked out")
    run.add_argument("--thumbnails", choices=thumbs.MODES, default="files",
                     help="poster images in the HTML report: files (default, written to "
                          "<out>/thumbs), embed (single self-contained page), or none")
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

    images = thumbs.prepare(posters, out, mode=args.thumbnails, log=log)

    report.write_json(posters, out / "report.json")
    report.write_markdown(posters, out / "report.md", args.conference)
    report.write_html(posters, out / "report.html", args.conference, images)
    for name in ("report.json", "report.md", "report.html"):
        log.detail("wrote %s (%d bytes)" % (out / name, (out / name).stat().st_size), indent=0)

    still_missing = manual.write_unmatched(posters, out / "unmatched.csv")

    pairs = organise(posters, out / "photos", mode=args.organise)
    if log.verbose:
        for source, target in pairs:
            log.detail("%s -> %s" % (Path(source).name, target), indent=0)
    if args.organise == "plan":
        (out / "organise-plan.txt").write_text(
            "\n".join("%s -> %s" % (s, t) for s, t in pairs) + "\n", encoding="utf-8"
        )

    linked = sum(1 for p in posters if p.work)
    print("%d posters, %d matched (%s)%s"
          % (len(posters), linked, ", ".join(source_names),
             ", %d from --merge" % merged if merged else ""))
    print("report: %s" % (out / "report.md"))
    print("        %s" % (out / "report.html"))
    if args.organise == "plan":
        print("organise plan (dry run): %s" % (out / "organise-plan.txt"))
    else:
        print("photos %sd into %s" % (args.organise, out / "photos"))
    if still_missing:
        print("%d unmatched: add links in %s, then re-run with --merge %s"
              % (still_missing, out / "unmatched.csv", out / "unmatched.csv"))
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
        return cmd_ocr(args)
    except OCRError as exc:
        print("OCR error: %s" % exc, file=sys.stderr)
        return 2
    except KeyboardInterrupt:
        print("interrupted", file=sys.stderr)
        return 130


if __name__ == "__main__":
    sys.exit(main())
