"""Orchestration: photos in, one record per poster out.

Two caches, invalidated independently, so that redoing the lookups never means
redoing the OCR:

  cache/lines/*.json    recognised text + geometry, keyed by image path and mtime
  cache/web/*.body      every HTTP response, keyed by URL
  cache/posters.json    the finished records
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
from collections import Counter
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional

from . import ocr as ocr_mod
from . import sources as sources_mod
from . import subfields as subfields_mod
from . import titles as titles_mod
from .http import Fetcher
from .llm import LLM, Ask, LLMError
from .log import Log, human_time
from .util import slugify

REDO_MODES = ("none", "research", "all")


@dataclass
class Poster:
    image: str
    pid: str = ""                       # poster01, poster02 - see number()
    title: Optional[str] = None
    title_source: str = "none"          # heuristic | llm | matched-paper | none
    title_score: float = 0.0
    candidates: List[str] = field(default_factory=list)
    ocr_lines: int = 0
    full_text: str = ""
    work: Optional[dict] = None         # Work.to_dict()
    match_score: float = 0.0
    match_source: str = "none"          # arxiv | openalex | crossref | none
    match_how: str = "none"             # id-on-poster | title-search | llm-verified | none
    subfield: str = subfields_mod.UNCLASSIFIED
    subfield_source: str = "none"       # arxiv | index | keywords | llm | none
    subfield_confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    notes: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "Poster":
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in d.items() if k in known})


class Pipeline:
    def __init__(
        self,
        cache_dir: Path,
        backend: Optional[str] = None,
        llm: Optional[LLM] = None,
        offline: bool = False,
        threshold: float = 0.72,
        source_names: Optional[List[str]] = None,
        mailto: str = "",
        log: Optional[Log] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.lines_dir = self.cache_dir / "lines"
        self.lines_dir.mkdir(parents=True, exist_ok=True)
        self.backend = backend
        self.llm = llm or LLM("off")
        self.threshold = threshold
        self.sources = list(source_names or sources_mod.DEFAULT)
        self.log = log or Log()
        self.fetcher = Fetcher(self.cache_dir / "web", offline=offline,
                               mailto=mailto, log=self.log)
        self.log.detail("sources: %s | threshold %.2f | ocr %s | llm %s"
                        % (", ".join(self.sources), threshold,
                           backend or ocr_mod.default_backend(), self.llm.provider), indent=0)

    # -- OCR cache ---------------------------------------------------------

    def _lines_path(self, image: Path) -> Path:
        try:
            stamp = "%d-%d" % (image.stat().st_mtime_ns, image.stat().st_size)
        except OSError:
            stamp = "0"
        key = hashlib.sha256(("%s|%s|%s" % (image.resolve(), stamp, self.backend or "auto"))
                             .encode("utf-8")).hexdigest()[:32]
        return self.lines_dir / (key + ".json")

    def read_lines(self, images: List[Path], reuse: bool = True) -> Dict[str, List]:
        """OCR, using the cached text for any image already recognised."""
        cached: Dict[str, List] = {}
        todo: List[Path] = []
        for image in images:
            path = self._lines_path(image)
            if reuse and path.exists():
                cached[str(image)] = [ocr_mod.Line.from_dict(d)
                                      for d in json.loads(path.read_text(encoding="utf-8"))]
            else:
                todo.append(image)

        if cached:
            self.log.info("OCR: reusing cached text for %d image(s)" % len(cached))
        if todo:
            self.log.info("OCR: %d image(s) via %s"
                          % (len(todo), self.backend or ocr_mod.default_backend()))
            with self.log.timed("OCR", indent=1):
                fresh = ocr_mod.run_ocr(todo, self.backend, self.cache_dir / "ocr")
            for image in todo:
                lines = fresh.get(str(image), [])
                self._lines_path(image).write_text(
                    json.dumps([l.to_dict() for l in lines]), encoding="utf-8")
                cached[str(image)] = lines
                self.log.detail("%s: %d line(s) recognised" % (image.name, len(lines)))
        return cached

    # -- one poster (no model involved) ------------------------------------

    def build_poster(self, image: Path, reading: titles_mod.PageReading) -> Poster:
        poster = Poster(
            image=str(image),
            ocr_lines=len(reading.full_text.splitlines()),
            full_text=reading.full_text,
            candidates=[c.text for c in reading.candidates],
        )
        if not reading.full_text.strip():
            poster.notes.append("no text recognised - is this a poster photo?")
            return poster

        if reading.title:
            poster.title = reading.title
            poster.title_score = reading.title_score
            poster.title_source = "heuristic"
            self.log.detail('title: "%s" (score %.2f)' % (poster.title, poster.title_score))
            for candidate in reading.candidates[1:3]:
                self.log.detail("runner-up %.2f  %s" % (candidate.score, candidate.text[:80]),
                                indent=2)
        else:
            self.log.detail("title: none of %d line(s) looked like one" % poster.ocr_lines)

        self.look_up(poster, reading.arxiv_ids, reading.dois or [])
        self.classify(poster)
        return poster

    def look_up(self, poster: Poster, arxiv_ids=(), dois=()) -> None:
        match = sources_mod.resolve(
            self.fetcher, poster.title, arxiv_ids, dois,
            self.sources, self.threshold, poster.notes, self.log,
        )
        self.apply_match(poster, match)
        if match.work:
            self.log.detail("matched %s:%s via %s (%.2f)"
                            % (match.work.source, match.work.ident, match.how, match.score))

    def apply_match(self, poster: Poster, match) -> None:
        poster.match_score = match.score
        poster.match_how = match.how
        if match.work:
            poster.work = match.work.to_dict()
            poster.match_source = match.work.source
            # A published title beats OCR every time.
            poster.title = match.work.title
            poster.title_source = "matched-paper"

    def classify(self, poster: Poster) -> None:
        work = poster.work or {}
        subfield, source, confidence, evidence = subfields_mod.classify(
            work.get("primary_category", ""),
            " ".join(filter(None, [poster.title or "", poster.full_text])),
            work.get("subject", ""),
            work.get("subject_confidence", 0.0),
        )
        poster.subfield = subfield
        poster.subfield_source = source
        poster.subfield_confidence = confidence
        poster.evidence = evidence
        self.log.detail("subfield: %s (via %s, %.2f)%s"
                        % (subfield, source, confidence,
                           " - " + ", ".join(evidence[:4]) if evidence else ""))

    # -- the batched model pass (exactly one request) -----------------------

    def _gaps(self, poster: Poster) -> List[str]:
        needs = []
        if not poster.full_text.strip():
            return needs                       # nothing to work from
        if not poster.title:
            needs.append("title")
        if not poster.work and "llm" in self.sources:
            needs.append("id")
        if poster.subfield == subfields_mod.UNCLASSIFIED:
            needs.append("subfield")
        return needs

    def assist(self, posters: List[Poster]) -> int:
        """One request for the whole batch, then re-derive what it unblocked.

        Returns the number of posters the answers actually changed.
        """
        if not self.llm.enabled:
            return 0
        asks, indexed = [], {}
        for index, poster in enumerate(posters, 1):
            needs = self._gaps(poster)
            if not needs:
                continue
            indexed[index] = poster
            asks.append(Ask(index=index, title=poster.title,
                            text=poster.full_text, needs=needs))
        if not asks:
            self.log.info("llm: nothing left unresolved, no request made")
            return 0

        self.log.info("llm: one request covering %d poster(s)" % len(asks))
        for ask in asks:
            self.log.detail("poster %d needs %s" % (ask.index, ", ".join(ask.needs)))
        options = sorted({name for name, _ in subfields_mod.KEYWORD_RULES})
        try:
            with self.log.timed("llm request"):
                answers = self.llm.assist(asks, options)
            self.log.detail("parsed %d answer(s) from the reply" % len(answers))
        except LLMError as exc:
            self.log.warn("llm request failed (%s); keeping the deterministic result" % exc)
            for poster in indexed.values():
                poster.notes.append("llm request failed: %s" % exc)
            return 0

        changed = 0
        for index, poster in indexed.items():
            answer = answers.get(index)
            if not answer:
                poster.notes.append("llm returned nothing usable for this poster")
                continue
            if self._apply_answer(poster, answer):
                changed += 1
        missing = [i for i in indexed if i not in answers]
        if missing:
            self.log.info("llm: no usable answer for %d poster(s)" % len(missing))
        self.log.info("llm: %d poster(s) improved" % changed)
        return changed

    def _apply_answer(self, poster: Poster, answer) -> bool:
        changed = False
        self.log.detail("%s: llm answered title=%r id=%r subfield=%r"
                        % (Path(poster.image).name, answer.title,
                           answer.identifier, answer.subfield))
        if answer.title and not poster.title:
            poster.title = answer.title
            poster.title_source = "llm"
            changed = True
            # A title is only useful if we then go and look it up - deterministically.
            self.look_up(poster)

        if answer.identifier and not poster.work:
            match = sources_mod.verify_identifier(
                self.fetcher, answer.identifier, poster.title,
                self.sources, self.threshold, poster.notes, self.log,
            )
            if match:
                self.apply_match(poster, match)
                changed = True

        if changed:
            self.classify(poster)          # the new paper may name its own subfield

        if answer.subfield and poster.subfield == subfields_mod.UNCLASSIFIED:
            poster.subfield = answer.subfield
            poster.subfield_source = "llm"
            poster.subfield_confidence = 0.5
            poster.evidence = ["llm"]
            changed = True
        return changed

    # -- the run -----------------------------------------------------------

    def persist(self, posters: List[Poster]) -> None:
        """Write the records back, so a later resume keeps merged edits."""
        state_path = self.cache_dir / "posters.json"
        done = {}
        if state_path.exists():
            for item in json.loads(state_path.read_text(encoding="utf-8")):
                done[item["image"]] = Poster.from_dict(item)
        for poster in posters:
            done[poster.image] = poster
        self._save(state_path, done)

    @staticmethod
    def _save(path: Path, done: Dict[str, Poster]) -> None:
        path.write_text(json.dumps([p.to_dict() for p in done.values()], indent=2),
                        encoding="utf-8")

    def run(self, images: List[Path], redo: str = "none",
            refresh_web: bool = False) -> List[Poster]:
        if redo not in REDO_MODES:
            raise ValueError("unknown redo mode %r (choose from %s)"
                             % (redo, ", ".join(REDO_MODES)))
        started = time.time()
        if refresh_web:
            dropped = self.fetcher.clear_cache()
            self.log.info("dropped %d cached web response(s)" % dropped)

        state_path = self.cache_dir / "posters.json"
        done: Dict[str, Poster] = {}
        if redo == "none" and state_path.exists():
            for item in json.loads(state_path.read_text(encoding="utf-8")):
                done[item["image"]] = Poster.from_dict(item)

        todo = [p for p in images if str(p) not in done]
        if done:
            self.log.detail("resuming: %d poster(s) already done, %d to go"
                            % (len(done), len(todo)), indent=0)
        if not todo:
            self.log.info("nothing to do (%d posters already done)" % len(done))
            return [done[str(p)] for p in images if str(p) in done]

        # This is the whole point of `--redo research`: keep the recognised
        # text, throw away the conclusions drawn from it.
        lines = self.read_lines(todo, reuse=(redo != "all"))
        fresh = []
        for index, image in enumerate(todo, 1):
            self.log.info("[%d/%d] %s" % (index, len(todo), image.name))
            poster_started = time.time()
            reading = titles_mod.read_page(lines.get(str(image), []))
            poster = self.build_poster(image, reading)
            done[str(image)] = poster
            fresh.append(poster)
            self._save(state_path, done)
            self.log.detail("done in %s" % human_time(time.time() - poster_started))

        # Everything deterministic is finished. Whatever is still missing across
        # the whole batch goes into a single request.
        if self.assist(fresh):
            self._save(state_path, done)

        posters = [done[str(p)] for p in images if str(p) in done]
        self.summarise(posters, time.time() - started)
        return posters

    def summarise(self, posters: List[Poster], elapsed: float) -> None:
        self.log.info("lookups: %d live, %d cached; llm requests: %d; %s elapsed"
                      % (self.fetcher.live_requests, self.fetcher.cache_hits,
                         self.llm.calls, human_time(elapsed)))
        if not self.log.verbose or not posters:
            return
        self.log.detail("matched by source:", indent=0)
        by_source = Counter(p.match_source if p.work else "unmatched" for p in posters)
        for name, count in by_source.most_common():
            self.log.detail("%4d  %s" % (count, name))
        self.log.detail("subfield decided by:", indent=0)
        for name, count in Counter(p.subfield_source for p in posters).most_common():
            self.log.detail("%4d  %s" % (count, name))
        self.log.detail("clusters:", indent=0)
        for name, count in Counter(p.subfield for p in posters).most_common():
            self.log.detail("%4d  %s" % (count, name))
        flagged = [p for p in posters if p.notes]
        if flagged:
            self.log.detail("%d poster(s) with notes:" % len(flagged), indent=0)
            for poster in flagged:
                self.log.detail("%s" % Path(poster.image).name)
                for note in poster.notes:
                    self.log.detail(note, indent=3)


def number(posters: List[Poster]) -> List[Poster]:
    """Give every poster a stable name - poster01, poster02, ... - which is what
    its files are called wherever they land: thumbs/, posters/, photos/.

    A poster that already carries one keeps it, so re-rendering an old report
    renames nothing and the paths in report.json stay good.
    """
    width = max(2, len(str(len(posters))))
    taken = {p.pid for p in posters if p.pid}
    counter = 0
    for poster in posters:
        if poster.pid:
            continue
        counter += 1
        while ("poster%0*d" % (width, counter)) in taken:
            counter += 1
        poster.pid = "poster%0*d" % (width, counter)
        taken.add(poster.pid)
    return posters


def organise(posters: List[Poster], destination: Path, mode: str = "copy") -> List[tuple]:
    """Lay the photos out as <destination>/<subfield>/poster<id>.<ext>.

    Returns the (source, target) pairs; with mode="plan" nothing is written.
    """
    if mode not in ("plan", "copy", "move", "symlink"):
        raise ValueError("unknown organise mode %r" % mode)
    pairs = []
    used = set()
    for poster in posters:
        source = Path(poster.image)
        folder = destination / slugify(poster.subfield)
        stem = poster.pid or slugify(poster.title or source.stem)
        target = folder / (stem + source.suffix.lower())
        counter = 2
        while target in used or (mode != "plan" and target.exists() and target != source):
            target = folder / ("%s-%d%s" % (stem, counter, source.suffix.lower()))
            counter += 1
        used.add(target)
        pairs.append((source, target))
        if mode == "plan":
            continue
        folder.mkdir(parents=True, exist_ok=True)
        if mode == "copy":
            shutil.copy2(source, target)
        elif mode == "move":
            shutil.move(str(source), str(target))
        else:
            if target.is_symlink():
                target.unlink()
            target.symlink_to(source.resolve())
    return pairs
