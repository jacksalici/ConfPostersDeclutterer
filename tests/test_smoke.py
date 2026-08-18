"""Offline smoke tests. They exercise the whole pipeline without touching the
network and without asking any model anything: OCR comes from sidecar text files
and every lookup response comes from fixture files seeded into the HTTP cache.

Run with:  python3 -m unittest discover -s tests -t . -v
"""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from tests.helpers import FIXTURES, copy_photos, seed, seed_all

from posterdeclutter import (arxiv, crossref, llm, ocr, openalex, report,
                             sources, subfields, titles, util)
from posterdeclutter.cli import main
from posterdeclutter.http import Fetcher
from posterdeclutter.log import QUIET, VERBOSE, Log, from_flags, human_time
from posterdeclutter.pipeline import Pipeline, organise


def fetcher(cache: Path) -> Fetcher:
    return Fetcher(cache, offline=True)


class TestUtil(unittest.TestCase):
    def test_similarity_is_order_insensitive_but_content_sensitive(self):
        self.assertEqual(util.similarity("Deep Sets", "deep sets"), 1.0)
        self.assertGreater(util.similarity("Deep Sets for Jet Tagging",
                                           "Deep Sets for Jet Tagging at the LHC"), 0.75)
        self.assertLess(util.similarity("Deep Sets for Jet Tagging",
                                        "Galaxy cluster mass calibration"), 0.1)

    def test_clean_text_repairs_ocr_artifacts(self):
        self.assertEqual(util.clean_text("eﬃcient   ﬂow–rate"), "efficient flow-rate")

    def test_slugify(self):
        self.assertEqual(util.slugify("Deep Sets, for Jet Tagging!"), "deep-sets-for-jet-tagging")
        self.assertEqual(util.slugify("???"), "untitled")


class TestOCRParsing(unittest.TestCase):
    def test_tesseract_tsv_groups_words_into_lines(self):
        tsv = "\n".join([
            "level\tpage_num\tblock_num\tpar_num\tline_num\tword_num\tleft\ttop\twidth\theight\tconf\ttext",
            "1\t1\t0\t0\t0\t0\t0\t0\t1000\t2000\t-1\t",
            "5\t1\t1\t1\t1\t1\t100\t50\t180\t40\t96\tDeep",
            "5\t1\t1\t1\t1\t2\t300\t50\t120\t40\t95\tSets",
            "5\t1\t2\t1\t1\t1\t100\t400\t200\t20\t90\tAbstract",
        ])
        lines = ocr._parse_tesseract_tsv(tsv)
        self.assertEqual([l.text for l in lines], ["Deep Sets", "Abstract"])
        self.assertAlmostEqual(lines[0].y, 0.025, places=3)
        self.assertGreater(lines[0].h, lines[1].h)

    def test_sidecar_backend_reads_transcripts(self):
        with tempfile.TemporaryDirectory() as tmp:
            photos = copy_photos(Path(tmp) / "photos")
            images = ocr.find_images(photos)
            self.assertEqual(len(images), 5)
            result = ocr.sidecar_ocr(images, Path(tmp))
            first = result[str(photos / "IMG_0001.jpg")]
            self.assertTrue(first[0].text.startswith("Deep Sets"))
            self.assertEqual(first[0].y, 0.0)
            self.assertGreater(first[0].h, first[-1].h)

    def test_lines_round_trip_through_the_cache_format(self):
        line = ocr.Line(text="Deep Sets", conf=0.9, x=0.1, y=0.2, w=0.5, h=0.06)
        self.assertEqual(ocr.Line.from_dict(line.to_dict()), line)


class TestTitleHeuristic(unittest.TestCase):
    def _read(self, name):
        image = FIXTURES / "photos" / name
        return titles.read_page(ocr.sidecar_ocr([image], FIXTURES)[str(image)])

    def test_picks_the_title_over_authors_and_sections(self):
        reading = self._read("IMG_0001.jpg")
        self.assertEqual(reading.title, "Deep Sets for Jet Flavour Tagging at the LHC")

    def test_finds_identifiers_printed_on_the_poster(self):
        self.assertEqual(self._read("IMG_0001.jpg").arxiv_ids, ["2401.12345"])
        self.assertEqual(self._read("IMG_0004.jpg").dois, ["10.1145/3292500.3330701"])
        self.assertEqual(self._read("IMG_0002.jpg").arxiv_ids, [])

    def test_a_doi_is_not_misread_as_an_arxiv_id(self):
        # 10.1145/3292500.3330701 contains "2500.33307", which looks arXiv-shaped.
        self.assertEqual(self._read("IMG_0004.jpg").arxiv_ids, [])

    def test_rejects_junk_lines(self):
        self.assertTrue(titles._is_junk("Abstract"))
        self.assertTrue(titles._is_junk("Contact: a.rossi@unibo.it"))
        self.assertTrue(titles._is_junk("Department of Physics, University of Bologna"))
        self.assertTrue(titles._is_junk("Poster #B12"))
        self.assertTrue(titles._is_junk("doi 10.1145/3292500.3330701"))
        self.assertFalse(titles._is_junk("Deep Sets for Jet Flavour Tagging at the LHC"))

    def test_a_sentence_is_not_a_title(self):
        # Accepting an abstract paragraph as the title would hide the fact that
        # no title was found - and hide it from the batched model pass too.
        lines = [ocr.Line("Fig. 1", 1.0, 0.1, 0.02, 0.8, 0.05),
                 ocr.Line("We propose the Transformer, a network architecture based solely "
                          "on attention mechanisms, dispensing with recurrence entirely.",
                          1.0, 0.1, 0.10, 0.8, 0.02)]
        self.assertIsNone(titles.read_page(lines).title)
        # A long title without terminal punctuation is still accepted.
        keep = [ocr.Line("Deep Sets for Jet Flavour Tagging at the Large Hadron Collider",
                         1.0, 0.1, 0.02, 0.8, 0.05)]
        self.assertIsNotNone(titles.read_page(keep).title)

    def test_affiliation_words_match_whole_words_only(self):
        # "eth" (ETH Zurich) must not fire inside "Ethnography", nor "lab"
        # inside "collaborative", nor "mit" inside "transmit".
        for good in ("An Ethnography of Poster Sessions",
                     "Collaborative Filtering at Scale",
                     "Limits on Transmit Power",
                     "Incidence of Drift in Sensor Arrays"):
            self.assertFalse(titles._is_junk(good), good)
        for bad in ("Institute for Quantum Optics, ETH Zurich",
                    "Max Planck Institute for Astrophysics",
                    "Department of Physics, University of Bologna"):
            self.assertTrue(titles._is_junk(bad), bad)

    def test_detects_author_lines(self):
        self.assertTrue(titles._looks_like_authors("A. Rossi, B. Bianchi, C. Verdi, D. Neri"))
        self.assertFalse(titles._looks_like_authors("Deep Sets for Jet Flavour Tagging at the LHC"))

    def test_returns_nothing_when_there_is_nothing(self):
        self.assertIsNone(self._read("IMG_0003.jpg").title)


class TestHTTPCache(unittest.TestCase):
    def test_offline_serves_seeded_responses_and_refuses_the_rest(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            url = arxiv._id_url("2401.12345")
            seed(cache, url, "arxiv_2401.12345.xml")
            f = fetcher(cache)
            self.assertIn("Deep Sets", f.get(url))
            self.assertEqual(f.cache_hits, 1)
            self.assertEqual(f.live_requests, 0)
            with self.assertRaises(Exception):
                f.get("https://api.openalex.org/works?nope")

    def test_clear_cache_empties_it(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            seed(cache, "u1", "arxiv_empty.xml")
            seed(cache, "u2", "arxiv_empty.xml")
            self.assertEqual(fetcher(cache).clear_cache(), 2)
            self.assertEqual(list(cache.glob("*.body")), [])

    def test_mailto_joins_the_polite_pool(self):
        self.assertEqual(Fetcher(Path("/tmp"), mailto="a@b.c").polite({"x": 1}),
                         {"x": 1, "mailto": "a@b.c"})
        self.assertEqual(Fetcher(Path("/tmp")).polite({"x": 1}), {"x": 1})


class TestLogging(unittest.TestCase):
    def _capture(self, level):
        import io
        stream = io.StringIO()
        return Log(level, stream), stream

    def test_levels_gate_what_is_written(self):
        for level, expect in ((QUIET, {"warn"}),
                              (1, {"info", "head", "warn"}),
                              (VERBOSE, {"info", "head", "detail", "warn"})):
            log, stream = self._capture(level)
            log.info("info"); log.head("head"); log.detail("detail"); log.warn("warn")
            written = {w.strip().lstrip("! ") for w in stream.getvalue().splitlines()}
            self.assertEqual(written, expect, level)

    def test_warnings_survive_quiet(self):
        log, stream = self._capture(QUIET)
        log.warn("tesseract not found")
        self.assertIn("tesseract not found", stream.getvalue())

    def test_from_flags(self):
        self.assertEqual(from_flags().level, 1)
        self.assertEqual(from_flags(quiet=True).level, QUIET)
        self.assertEqual(from_flags(verbose=True).level, VERBOSE)
        self.assertTrue(from_flags(verbose=True).verbose)
        self.assertFalse(from_flags().verbose)

    def test_detail_is_indented_for_readability(self):
        log, stream = self._capture(VERBOSE)
        log.detail("one"); log.detail("two", indent=2)
        self.assertEqual(stream.getvalue().splitlines(), ["  one", "    two"])

    def test_human_time(self):
        self.assertEqual([human_time(x) for x in (0.004, 0.42, 12.5, 185)],
                         ["4ms", "420ms", "12.5s", "3m05s"])

    def test_timed_reports_only_when_verbose(self):
        log, stream = self._capture(VERBOSE)
        with log.timed("OCR"):
            pass
        self.assertIn("OCR took", stream.getvalue())
        quiet, quiet_stream = self._capture(1)
        with quiet.timed("OCR"):
            pass
        self.assertEqual(quiet_stream.getvalue(), "")

    def test_the_fetcher_reports_cache_hits_and_live_requests(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            seed(cache, "https://example.org/a", "arxiv_empty.xml")
            log, stream = self._capture(VERBOSE)
            Fetcher(cache, offline=True, log=log).get("https://example.org/a")
            self.assertIn("cached  https://example.org/a", stream.getvalue())

    def test_candidate_scores_are_explained(self):
        from posterdeclutter.works import report_candidates
        log, stream = self._capture(VERBOSE)
        report_candidates(log, "arxiv", [(0.91, "A Good Match"), (0.30, "Something Else")], 0.72)
        written = stream.getvalue()
        self.assertIn("arxiv: 2 candidate(s), need 0.72", written)
        self.assertIn("accept 0.91  A Good Match", written)
        self.assertIn("skip 0.30  Something Else", written)

    def test_candidate_reporting_is_silent_below_verbose(self):
        from posterdeclutter.works import report_candidates
        log, stream = self._capture(1)
        report_candidates(log, "arxiv", [(0.91, "A Good Match")], 0.72)
        self.assertEqual(stream.getvalue(), "")


class TestArxivSource(unittest.TestCase):
    def test_parses_an_atom_feed(self):
        works = arxiv.parse_feed((FIXTURES / "arxiv_2401.12345.xml").read_text())
        self.assertEqual(len(works), 1)
        work = works[0]
        self.assertEqual(work.source, "arxiv")
        self.assertEqual(work.ident, "2401.12345v1")
        self.assertEqual(work.primary_category, "hep-ex")
        self.assertIn("cs.LG", work.categories)
        self.assertEqual(work.authors[0], "A. Rossi")
        self.assertTrue(work.url.startswith("https://arxiv.org/abs/"))

    def test_title_query_keeps_stopwords(self):
        # ti:"..." is an exact phrase, so dropping "is"/"you" turns a hit into a miss.
        self.assertEqual(arxiv._title_query("Attention Is All You Need"),
                         'ti:"attention is all you need"')

    def test_rejects_a_weak_match(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            title = "A completely unrelated poster about sourdough"
            seed(cache, arxiv._query_url(arxiv._title_query(title)), "arxiv_2401.12345.xml")
            seed(cache, arxiv._query_url(arxiv._loose_query(title)), "arxiv_empty.xml")
            match = arxiv.search(fetcher(cache), title)
            self.assertIsNone(match.work)
            self.assertLess(match.score, 0.72)


class TestOpenAlexSource(unittest.TestCase):
    def test_reconstructs_the_inverted_abstract(self):
        self.assertEqual(openalex._abstract({"We": [0], "are": [1], "here": [2]}),
                         "We are here")
        self.assertEqual(openalex._abstract(None), "")

    def test_maps_a_record_and_prefers_the_topic_over_the_scopus_bucket(self):
        payload = json.loads((FIXTURES / "openalex_qubit.json").read_text())
        work = openalex.to_work(payload["results"][0])
        self.assertEqual(work.source, "openalex")
        self.assertEqual(work.doi, "10.1103/physrevx.13.041020")
        self.assertEqual(work.venue, "Physical Review X")
        # "Artificial Intelligence" is the coarse Scopus subfield for this paper.
        self.assertEqual(work.subject, "Quantum Information and Cryptography")
        self.assertEqual(work.summary, "We characterise a readout chain.")

    def test_search_reranks_on_title_agreement(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            f = fetcher(cache)
            title = "Superconducting Qubit Readout with Travelling-Wave Amplifiers"
            seed(cache, openalex._search_url(f, title), "openalex_qubit.json")
            match = openalex.search(f, title)
            self.assertEqual(match.work.title, title)   # not the unrelated result
            self.assertEqual(match.score, 1.0)


class TestCrossrefSource(unittest.TestCase):
    def test_maps_a_record_and_strips_jats_markup(self):
        payload = json.loads((FIXTURES / "crossref_widgets.json").read_text())
        work = crossref.to_work(payload["message"]["items"][0])
        self.assertEqual(work.source, "crossref")
        self.assertEqual(work.doi, "10.1145/3292500.3330701")
        self.assertEqual(work.authors, ["J. Doe", "K. Smith"])
        self.assertEqual(work.published, "2019-08-04")
        self.assertEqual(work.venue, "Proceedings of KDD '19")
        self.assertNotIn("<jats", work.summary)
        self.assertIn("widgets", work.summary)


class TestSourceSelection(unittest.TestCase):
    def test_parse_names_validates(self):
        self.assertEqual(sources.parse_names("arxiv, openalex"), ["arxiv", "openalex"])
        self.assertEqual(sources.parse_names(""), list(sources.DEFAULT))
        with self.assertRaises(ValueError):
            sources.parse_names("arxiv,google-scholar")

    def test_short_titles_face_a_higher_bar(self):
        # "Random Forests" scores 0.80 against "Neural Random Forests" - close
        # enough to pass the normal threshold, and the wrong paper.
        self.assertEqual(sources.effective_threshold("Random Forests", 0.72), 0.87)
        self.assertEqual(sources.effective_threshold("Deep Sets for Jet Tagging", 0.72), 0.72)
        self.assertEqual(sources.effective_threshold(None, 0.72), 0.72)
        self.assertGreater(sources.effective_threshold("Random Forests", 0.72),
                           util.similarity("Random Forests", "Neural Random Forests"))

    def test_first_source_that_clears_the_threshold_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            f = fetcher(cache)
            title = "Superconducting Qubit Readout with Travelling-Wave Amplifiers"
            seed(cache, arxiv._query_url(arxiv._title_query(title)), "arxiv_qubit.xml")
            seed(cache, openalex._search_url(f, title), "openalex_qubit.json")
            notes = []
            match = sources.resolve(f, title, [], [], ["arxiv", "openalex"], 0.72, notes)
            self.assertEqual(match.work.source, "arxiv")
            reversed_ = sources.resolve(f, title, [], [], ["openalex", "arxiv"], 0.72, notes)
            self.assertEqual(reversed_.work.source, "openalex")

    def test_a_doi_on_the_poster_resolves_through_crossref(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            f = fetcher(cache)
            seed(cache, crossref._doi_url(f, "10.1145/3292500.3330701"),
                 "crossref_doi_widgets.json")
            notes = []
            match = sources.resolve(f, "A Study of Widgets in the Wild", [],
                                    ["10.1145/3292500.3330701"], ["crossref"], 0.72, notes)
            self.assertEqual(match.how, "id-on-poster")
            self.assertEqual(match.work.doi, "10.1145/3292500.3330701")
            self.assertEqual(notes, [])

    def test_an_identifier_that_disagrees_with_the_title_is_treated_as_a_citation(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            f = fetcher(cache)
            seed(cache, arxiv._id_url("2401.12345"), "arxiv_2401.12345.xml")
            notes = []
            match = sources.resolve(f, "An unrelated poster about sourdough starters",
                                    ["2401.12345"], [], ["arxiv"], 0.72, notes)
            self.assertIsNone(match.work)
            self.assertTrue(any("probably a citation" in n for n in notes), notes)


class _StubLLM(llm.LLM):
    """A fake model, so the LLM paths are tested without asking anything."""

    def __init__(self, reply: str = ""):
        super().__init__("claude-cli")
        self.reply = reply
        self.prompts = []

    def ask(self, prompt: str, max_tokens: int = 4096) -> str:
        self.calls += 1
        self.prompts.append(prompt)
        return self.reply


class TestLLMIsOptional(unittest.TestCase):
    def test_disabled_by_default_and_never_called(self):
        model = llm.LLM()
        self.assertFalse(model.enabled)
        with self.assertRaises(llm.LLMError):
            model.ask("hello")
        self.assertEqual(model.calls, 0)

    def test_unknown_provider_is_rejected(self):
        with self.assertRaises(llm.LLMError):
            llm.LLM("gpt-by-carrier-pigeon")

    def test_identifier_parsing(self):
        self.assertEqual(llm.parse_identifier("arXiv:2401.12345v2"), ("arxiv", "2401.12345"))
        self.assertEqual(llm.parse_identifier("https://doi.org/10.1038/s41586-021-03819-2."),
                         ("doi", "10.1038/s41586-021-03819-2"))
        self.assertIsNone(llm.parse_identifier("NONE"))
        self.assertIsNone(llm.parse_identifier("I'm not sure, sorry"))


class TestOneRequestPerRun(unittest.TestCase):
    def _asks(self, count):
        return [llm.Ask(index=i, title=None, text="ocr text %d" % i, needs=["title"])
                for i in range(1, count + 1)]

    def test_a_whole_batch_costs_exactly_one_request(self):
        model = _StubLLM("\n".join('{"n": %d, "title": "Poster %d"}' % (i, i)
                                   for i in range(1, 26)))
        answers = model.assist(self._asks(25), ["Machine Learning"])
        self.assertEqual(model.calls, 1)
        self.assertEqual(len(answers), 25)
        self.assertEqual(answers[7].title, "Poster 7")

    def test_nothing_to_ask_means_no_request_at_all(self):
        model = _StubLLM("should never be sent")
        self.assertEqual(model.assist([], ["Machine Learning"]), {})
        self.assertEqual(model.assist([llm.Ask(1, None, "text", [])], []), {})
        self.assertEqual(model.calls, 0)

    def test_the_prompt_carries_every_poster_and_its_needs(self):
        model = _StubLLM("")
        model.assist([llm.Ask(1, None, "first poster text", ["title", "id"]),
                      llm.Ask(2, "Known", "second poster text", ["subfield"])],
                     ["Machine Learning", "Quantum Physics"])
        prompt = model.prompts[0]
        self.assertIn("--- poster 1 (need: title, id) ---", prompt)
        self.assertIn("--- poster 2 (need: subfield) ---", prompt)
        self.assertIn("first poster text", prompt)
        self.assertIn("Title read off the poster: Known", prompt)
        self.assertIn("- Quantum Physics", prompt)

    def test_excerpts_shrink_so_one_prompt_stays_bounded(self):
        self.assertEqual(llm.excerpt_budget(1), llm.MAX_EXCERPT)
        self.assertEqual(llm.excerpt_budget(10_000), llm.MIN_EXCERPT)
        big = [llm.Ask(i, None, "x" * 5000, ["title"]) for i in range(1, 501)]
        self.assertLess(len(llm.build_prompt(big, [])), llm.PROMPT_BUDGET + 60_000)

    def test_reply_parsing_survives_fences_prose_and_junk(self):
        answers = llm.parse_batch(
            "```json\n"
            '{"n": 1, "title": "Widgets", "id": "arXiv:2401.12345", "subfield": null}\n'
            "not json at all\n"
            '{"n": 2, "title": null, "id": "nope", "subfield": "Quantum Physics"}\n'
            '{"n": 99, "title": "out of range"}\n'
            "```", expected={1, 2})
        self.assertEqual(set(answers), {1, 2})
        self.assertEqual(answers[1].identifier, ("arxiv", "2401.12345"))
        self.assertIsNone(answers[1].subfield)
        self.assertIsNone(answers[2].identifier)      # "nope" is not an identifier
        self.assertEqual(answers[2].subfield, "Quantum Physics")

    def test_a_json_array_reply_also_parses(self):
        answers = llm.parse_batch('[{"n": 1, "title": "A"}, {"n": 2, "title": "B"}]')
        self.assertEqual(answers[2].title, "B")

    def test_an_empty_or_garbage_reply_yields_nothing(self):
        self.assertEqual(llm.parse_batch(""), {})
        self.assertEqual(llm.parse_batch("I could not identify any of these."), {})


class TestProposedIdentifiersAreVerified(unittest.TestCase):
    def test_a_proposal_must_agree_with_the_poster(self):
        with tempfile.TemporaryDirectory() as tmp:
            cache = Path(tmp)
            f = fetcher(cache)
            seed(cache, arxiv._id_url("2401.12345"), "arxiv_2401.12345.xml")
            title = "Deep Sets for Jet Flavour Tagging at the LHC"

            notes = []
            match = sources.verify_identifier(f, ("arxiv", "2401.12345"), title,
                                              ["arxiv"], 0.72, notes)
            self.assertEqual(match.how, "llm-verified")
            self.assertEqual(match.work.ident, "2401.12345v1")

            notes = []
            self.assertIsNone(sources.verify_identifier(
                f, ("arxiv", "2401.12345"), "Sourdough starters of Lombardy",
                ["arxiv"], 0.72, notes))
            self.assertTrue(any("disagrees" in n for n in notes), notes)

    def test_an_invented_identifier_is_discarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            f = fetcher(Path(tmp))   # nothing seeded: the lookup cannot resolve
            notes = []
            self.assertIsNone(sources.verify_identifier(
                f, ("arxiv", "9999.99999"), "Some poster", ["arxiv"], 0.72, notes))
            self.assertTrue(any("does not resolve" in n for n in notes), notes)


class TestSubfields(unittest.TestCase):
    def test_arxiv_category_wins(self):
        name, source, confidence, _ = subfields.classify("hep-ex", "irrelevant", "Some Topic", 0.9)
        self.assertEqual(name, "High Energy Physics - Experiment")
        self.assertEqual(source, "arxiv")
        self.assertEqual(confidence, 1.0)

    def test_index_topic_beats_ocr_keywords(self):
        name, source, confidence, _ = subfields.classify(
            "", "neural network trained with gradient descent", "Quantum Information", 0.9)
        self.assertEqual(name, "Quantum Information")
        self.assertEqual(source, "index")
        self.assertEqual(confidence, 0.9)

    def test_unknown_category_falls_back_to_the_archive_name(self):
        self.assertEqual(subfields.name_for_category("cond-mat.zzz"), "Condensed Matter")
        self.assertEqual(subfields.name_for_category("wat.99"), "wat.99")

    def test_keyword_rules_classify_without_any_match(self):
        name, source, confidence, hits = subfields.classify(
            "", "We train a graph neural network with gradient descent on a large training set.")
        self.assertEqual(name, "Machine Learning")
        self.assertEqual(source, "keywords")
        self.assertGreater(confidence, 0.0)
        self.assertTrue(hits)

    def test_no_signal_means_unclassified_not_a_guess(self):
        name, source, _, _ = subfields.classify("", "Blurry photo")
        self.assertEqual(name, subfields.UNCLASSIFIED)
        self.assertEqual(source, "none")

    def test_keyword_matching_is_word_bounded(self):
        name, _, _, _ = subfields.classify("", "the slab pileups were unrelated to physics")
        self.assertEqual(name, subfields.UNCLASSIFIED)


class TestPipelineEndToEnd(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.photos = copy_photos(self.tmp / "photos")
        self.out = self.tmp / "out"
        seed_all(self.out / "cache" / "web", Fetcher(self.out / "cache" / "web", offline=True))

    def tearDown(self):
        self._tmp.cleanup()

    def _pipeline(self, **kwargs):
        kwargs.setdefault("source_names", ["arxiv", "openalex", "crossref"])
        return Pipeline(cache_dir=self.out / "cache", backend="sidecar",
                        offline=True, log=Log(QUIET), **kwargs)

    def _run(self, redo="none", **kwargs):
        return self._pipeline(**kwargs).run(ocr.find_images(self.photos), redo=redo)

    def _by_name(self, posters):
        return {Path(p.image).name: p for p in posters}

    def test_full_run_links_through_every_source(self):
        posters = self._by_name(self._run())
        self.assertEqual(len(posters), 5)

        jets = posters["IMG_0001.jpg"]
        self.assertEqual(jets.match_source, "arxiv")
        self.assertEqual(jets.match_how, "id-on-poster")
        self.assertEqual(jets.title_source, "matched-paper")
        self.assertEqual(jets.subfield, "High Energy Physics - Experiment")
        self.assertEqual(jets.subfield_source, "arxiv")

        qubits = posters["IMG_0002.jpg"]
        self.assertEqual(qubits.match_source, "arxiv")
        self.assertEqual(qubits.match_how, "title-search")
        self.assertEqual(qubits.subfield, "Quantum Physics")

        widgets = posters["IMG_0004.jpg"]
        self.assertEqual(widgets.match_source, "crossref")
        self.assertEqual(widgets.match_how, "id-on-poster")
        self.assertEqual(widgets.work["venue"], "Proceedings of KDD '19")
        self.assertEqual(widgets.subfield, "Human-Computer Interaction")
        self.assertEqual(widgets.subfield_source, "index")

        blank = posters["IMG_0003.jpg"]
        self.assertIsNone(blank.work)
        self.assertEqual(blank.subfield, subfields.UNCLASSIFIED)

    def test_source_order_changes_which_index_answers(self):
        posters = self._by_name(self._run(source_names=["openalex", "arxiv"]))
        qubits = posters["IMG_0002.jpg"]
        self.assertEqual(qubits.match_source, "openalex")
        self.assertEqual(qubits.subfield, "Quantum Information and Cryptography")

    def test_reports_render_from_every_source(self):
        posters = self._run()
        report.write_json(posters, self.out / "report.json")
        payload = json.loads((self.out / "report.json").read_text())
        self.assertEqual(payload["stats"]["linked"], 3)
        self.assertEqual(payload["stats"]["posters"], 5)

        markdown = report.render_markdown(posters, "TestConf 2026")
        self.assertIn("# Poster report - TestConf 2026", markdown)
        self.assertIn("arxiv.org/abs/2401.12345v1", markdown)
        self.assertIn("via crossref", markdown)
        self.assertIn("Proceedings of KDD", markdown)
        self.assertIn("## Unclassified", markdown)

        page = report.render_html(posters)
        self.assertTrue(page.startswith("<!doctype html>"))
        self.assertIn("Human-Computer Interaction", page)
        self.assertNotIn("<script", page)

    def test_unclassified_cluster_sorts_last(self):
        self.assertEqual(list(report.cluster(self._run()))[-1], subfields.UNCLASSIFIED)

    def test_resume_skips_work_already_done(self):
        first = self._run()
        self.assertTrue((self.out / "cache" / "posters.json").exists())
        again = self._run()
        self.assertEqual([p.to_dict() for p in first], [p.to_dict() for p in again])

    def _no_ocr_allowed(self):
        """Replace the OCR entry point with a tripwire, so a run that reuses the
        cached text succeeds and a run that re-recognises anything fails loudly."""
        import posterdeclutter.pipeline as pipeline_mod

        calls = []
        original = pipeline_mod.ocr_mod.run_ocr

        def tripwire(images, backend, cache_dir):
            calls.append([str(i) for i in images])
            return original(images, backend, cache_dir)

        pipeline_mod.ocr_mod.run_ocr = tripwire
        self.addCleanup(setattr, pipeline_mod.ocr_mod, "run_ocr", original)
        return calls

    def test_redo_research_keeps_the_cached_text_and_redoes_the_rest(self):
        self._run()
        cached = sorted((self.out / "cache" / "lines").glob("*.json"))
        self.assertEqual(len(cached), 5)
        stamps = {p: p.stat().st_mtime_ns for p in cached}

        calls = self._no_ocr_allowed()
        posters = self._run(redo="research")

        self.assertEqual(calls, [])                      # nothing was re-recognised
        self.assertEqual(len(posters), 5)                # but everything was re-derived
        self.assertEqual(self._by_name(posters)["IMG_0001.jpg"].match_source, "arxiv")
        self.assertEqual({p: p.stat().st_mtime_ns for p in cached}, stamps)

    def test_redo_all_re_runs_the_ocr(self):
        self._run()
        calls = self._no_ocr_allowed()
        self._run(redo="all")
        self.assertEqual(len(calls), 1)
        self.assertEqual(len(calls[0]), 5)

    def test_refresh_web_drops_the_cached_lookups(self):
        self._run()
        pipeline = self._pipeline()
        # Offline + emptied cache => no lookup can succeed, proving they re-ran.
        posters = pipeline.run(ocr.find_images(self.photos), redo="research", refresh_web=True)
        self.assertTrue(all(p.work is None for p in posters))

    def test_edited_photos_are_re_ocred_even_on_resume(self):
        self._run()
        sidecar = self.photos / "IMG_0003.jpg.txt"
        sidecar.write_text("Galaxy Cluster Mass Calibration with Weak Lensing\n")
        (self.photos / "IMG_0003.jpg").write_bytes(b"changed")
        posters = self._by_name(self._run(redo="research"))
        self.assertEqual(posters["IMG_0003.jpg"].title,
                         "Galaxy Cluster Mass Calibration with Weak Lensing")

    def test_organise_plan_is_a_dry_run(self):
        posters = self._run()
        pairs = organise(posters, self.out / "photos", mode="plan")
        self.assertEqual(len(pairs), 5)
        self.assertFalse((self.out / "photos").exists())
        target = dict(pairs)[self.photos / "IMG_0001.jpg"]
        self.assertEqual(target.parent.name, "high-energy-physics-experiment")
        self.assertTrue(target.name.startswith("deep-sets-for-jet-flavour-tagging"))

    def test_organise_copy_writes_files_and_keeps_originals(self):
        for source, target in organise(self._run(), self.out / "photos", mode="copy"):
            self.assertTrue(target.exists(), target)
            self.assertTrue(source.exists(), source)


class TestBatchedAssistInThePipeline(unittest.TestCase):
    """The model is consulted once per run, after every deterministic pass."""

    JETS = "Deep Sets for Jet Flavour Tagging at the LHC"
    LENSING = "Galaxy Cluster Mass Calibration with Weak Lensing"

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.tmp = Path(self._tmp.name)
        self.photos = copy_photos(self.tmp / "photos")
        self.out = self.tmp / "out"
        web = self.out / "cache" / "web"
        seed_all(web, Fetcher(web, offline=True), extra_empty=[self.LENSING])

    def tearDown(self):
        self._tmp.cleanup()

    def _run(self, model, sources_=("arxiv", "openalex", "crossref", "llm")):
        pipeline = Pipeline(cache_dir=self.out / "cache", backend="sidecar", offline=True,
                            llm=model, source_names=list(sources_), log=Log(QUIET))
        posters = pipeline.run(ocr.find_images(self.photos))
        return {Path(p.image).name: p for p in posters}

    def test_one_request_covers_every_unresolved_poster(self):
        model = _StubLLM("")
        self._run(model)
        self.assertEqual(model.calls, 1)
        prompt = model.prompts[0]
        # Two posters are unresolved; the three that resolved are not in the prompt.
        self.assertIn("--- poster 3 ", prompt)
        self.assertIn("--- poster 5 ", prompt)
        self.assertNotIn("--- poster 1 ", prompt)
        self.assertNotIn("--- poster 4 ", prompt)

    def test_no_request_when_the_deterministic_pass_left_no_gaps(self):
        model = _StubLLM("")
        # Only the posters that already resolve end to end.
        for name in ("IMG_0003", "IMG_0005"):
            (self.photos / (name + ".jpg")).unlink()
            (self.photos / (name + ".jpg.txt")).unlink()
        posters = self._run(model)
        self.assertEqual(len(posters), 3)
        self.assertEqual(model.calls, 0)

    def test_a_supplied_title_is_looked_up_and_reclassified(self):
        model = _StubLLM('{"n": 3, "title": "%s", "id": null, "subfield": null}' % self.LENSING)
        posters = self._run(model)
        lensing = posters["IMG_0003.jpg"]
        self.assertEqual(model.calls, 1)
        self.assertEqual(lensing.title, self.LENSING)
        self.assertEqual(lensing.title_source, "llm")
        self.assertIsNone(lensing.work)                       # nothing indexed matches
        # ...but the title was still searched, and then classified from its words.
        self.assertEqual(lensing.subfield, "Astrophysics")
        self.assertEqual(lensing.subfield_source, "keywords")

    def test_a_proposed_identifier_is_verified_then_used(self):
        # Poster 3 has no readable title, so the model supplies both; the ID is
        # then checked against the title it just gave us.
        model = _StubLLM('{"n": 3, "title": "%s", "id": "arXiv:2401.12345", "subfield": null}'
                         % self.JETS)
        poster = self._run(model)["IMG_0003.jpg"]
        self.assertEqual(poster.match_how, "llm-verified")
        self.assertEqual(poster.work["ident"], "2401.12345v1")
        self.assertEqual(poster.title_source, "matched-paper")
        self.assertEqual(poster.subfield, "High Energy Physics - Experiment")
        self.assertEqual(poster.subfield_source, "arxiv")

    def test_an_identifier_is_checked_against_the_title_we_already_read(self):
        # Poster 5 has a perfectly good OCR'd title; an ID naming a different
        # paper must lose to it, not overwrite it.
        model = _StubLLM('{"n": 5, "title": "%s", "id": "arXiv:2401.12345", "subfield": null}'
                         % self.JETS)
        ethnography = self._run(model)["IMG_0005.jpg"]
        self.assertEqual(ethnography.title, "An Ethnography of Poster Sessions")
        self.assertIsNone(ethnography.work)
        self.assertTrue(any("disagrees" in n for n in ethnography.notes), ethnography.notes)

    def test_a_wrong_identifier_is_rejected_and_noted(self):
        model = _StubLLM('{"n": 5, "title": null, "id": "arXiv:2401.12345", '
                         '"subfield": "Machine Learning"}')
        ethnography = self._run(model)["IMG_0005.jpg"]
        self.assertIsNone(ethnography.work)
        self.assertTrue(any("disagrees" in n for n in ethnography.notes), ethnography.notes)
        # The subfield it offered is still used - that answer stands on its own.
        self.assertEqual(ethnography.subfield, "Machine Learning")
        self.assertEqual(ethnography.subfield_source, "llm")

    def test_a_failed_request_leaves_the_deterministic_result_intact(self):
        class Broken(_StubLLM):
            def ask(self, prompt, max_tokens=4096):
                self.calls += 1
                raise llm.LLMError("claude not found on PATH")

        posters = self._run(Broken())
        self.assertEqual(posters["IMG_0001.jpg"].match_source, "arxiv")
        self.assertTrue(any("llm request failed" in n
                            for n in posters["IMG_0005.jpg"].notes))

    def test_answers_survive_into_the_saved_state(self):
        model = _StubLLM('{"n": 3, "title": "%s", "id": null, "subfield": null}' % self.LENSING)
        self._run(model)
        saved = json.loads((self.out / "cache" / "posters.json").read_text())
        by_name = {Path(p["image"]).name: p for p in saved}
        self.assertEqual(by_name["IMG_0003.jpg"]["title"], self.LENSING)

    def test_the_llm_source_only_asks_for_ids_when_it_is_listed(self):
        model = _StubLLM("")
        self._run(model, sources_=("arxiv",))
        self.assertNotIn("need: title, id", model.prompts[0])
        self.assertIn("need: title, subfield", model.prompts[0])


class TestCLI(unittest.TestCase):
    def _prepared(self, root: Path):
        photos = copy_photos(root / "photos")
        out = root / "out"
        seed_all(out / "cache" / "web", Fetcher(out / "cache" / "web", offline=True))
        return photos, out

    def test_run_command_produces_all_three_reports(self):
        with tempfile.TemporaryDirectory() as tmp:
            photos, out = self._prepared(Path(tmp))
            code = main(["run", str(photos), "-o", str(out), "--ocr", "sidecar",
                         "--offline", "--quiet", "--conference", "TestConf",
                         "--sources", "arxiv,openalex,crossref"])
            self.assertEqual(code, 0)
            for name in ("report.json", "report.md", "report.html", "organise-plan.txt"):
                self.assertTrue((out / name).exists(), name)
            self.assertFalse((out / "photos").exists())  # plan mode must not write

    def test_redo_research_via_the_cli(self):
        with tempfile.TemporaryDirectory() as tmp:
            photos, out = self._prepared(Path(tmp))
            argv = ["run", str(photos), "-o", str(out), "--ocr", "sidecar", "--offline", "--quiet"]
            self.assertEqual(main(argv), 0)
            self.assertEqual(main(argv + ["--redo", "research"]), 0)
            payload = json.loads((out / "report.json").read_text())
            self.assertEqual(payload["stats"]["posters"], 5)

    def test_unknown_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            photos, out = self._prepared(Path(tmp))
            self.assertEqual(main(["run", str(photos), "-o", str(out),
                                   "--sources", "google-scholar"]), 1)

    def test_llm_source_without_a_provider_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            photos, out = self._prepared(Path(tmp))
            self.assertEqual(main(["run", str(photos), "-o", str(out),
                                   "--sources", "arxiv,llm"]), 1)

    def test_verbose_and_quiet_change_how_much_is_written(self):
        import contextlib, io

        def run(flag):
            with tempfile.TemporaryDirectory() as tmp:
                photos, out = self._prepared(Path(tmp))
                err = io.StringIO()
                with contextlib.redirect_stderr(err):
                    code = main(["run", str(photos), "-o", str(out),
                                 "--ocr", "sidecar", "--offline"] + flag)
                self.assertEqual(code, 0)
                return err.getvalue()

        quiet, normal, verbose = run(["-q"]), run([]), run(["-v"])
        self.assertEqual(quiet.strip(), "")
        self.assertIn("[1/5]", normal)
        self.assertIn("[1/5]", verbose)
        self.assertGreater(len(verbose), len(normal) * 2)
        # Things only -v explains:
        self.assertIn("candidate(s), need", verbose)
        self.assertIn("matched by source:", verbose)
        self.assertIn("subfield:", verbose)
        self.assertNotIn("candidate(s), need", normal)

    def test_verbose_and_quiet_are_mutually_exclusive(self):
        import contextlib, io

        with tempfile.TemporaryDirectory() as tmp:
            photos, out = self._prepared(Path(tmp))
            with contextlib.redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit):
                    main(["run", str(photos), "-o", str(out), "-v", "-q"])

    def test_run_on_an_empty_folder_fails_loudly(self):
        with tempfile.TemporaryDirectory() as tmp:
            empty = Path(tmp) / "empty"
            empty.mkdir()
            self.assertEqual(main(["run", str(empty), "-o", str(Path(tmp) / "out")]), 1)

    def test_ocr_command_dumps_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            photos = copy_photos(Path(tmp) / "photos")
            self.assertEqual(main(["ocr", str(photos / "IMG_0001.jpg"), "--ocr", "sidecar",
                                   "--cache", str(Path(tmp) / "cache")]), 0)


if __name__ == "__main__":
    unittest.main()
