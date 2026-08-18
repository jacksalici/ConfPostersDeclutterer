"""Optional LLM assist. Off by default - the pipeline is deterministic without it.

**Exactly one request per run.** The pipeline finishes every deterministic pass
first, collects whatever is still missing across the whole batch, and asks once.
A run over 300 posters costs one call, not 300.

Three providers, all interchangeable:

  api         Anthropic Messages API (SDK if installed, else stdlib HTTPS)
  claude-cli  `claude -p` on this machine - uses your existing session, no API key
  codex-cli   `codex exec` on this machine

The model is only ever asked for three narrow things, always with a strict output
contract, and any answer that does not parse is discarded.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import urllib.request
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence, Tuple

DEFAULT_MODEL = "claude-opus-5"
API_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
PROVIDERS = ("off", "api", "claude-cli", "codex-cli")

# Keep the single prompt to a sane size however many posters need help.
PROMPT_BUDGET = 120_000
MIN_EXCERPT = 200
MAX_EXCERPT = 1_200


class LLMError(RuntimeError):
    pass


@dataclass
class Ask:
    """What is still missing for one poster."""
    index: int
    title: Optional[str]
    text: str
    needs: List[str] = field(default_factory=list)   # title | id | subfield


@dataclass
class Answer:
    title: Optional[str] = None
    identifier: Optional[Tuple[str, str]] = None     # ("arxiv"|"doi", value)
    subfield: Optional[str] = None


class LLM:
    def __init__(
        self,
        provider: str = "off",
        model: str = DEFAULT_MODEL,
        timeout: float = 300.0,
        verbose: bool = False,
    ):
        if provider not in PROVIDERS:
            raise LLMError("unknown llm provider %r (choose from %s)" % (provider, ", ".join(PROVIDERS)))
        self.provider = provider
        self.model = model
        self.timeout = timeout
        self.verbose = verbose
        self.calls = 0

    @property
    def enabled(self) -> bool:
        return self.provider != "off"

    # -- transports --------------------------------------------------------

    def ask(self, prompt: str, max_tokens: int = 4096) -> str:
        if not self.enabled:
            raise LLMError("LLM is disabled")
        self.calls += 1
        if self.provider == "api":
            return self._ask_api(prompt, max_tokens)
        if self.provider == "claude-cli":
            return self._ask_cli(["claude", "-p", prompt])
        return self._ask_cli(["codex", "exec", prompt])

    def _ask_api(self, prompt: str, max_tokens: int) -> str:
        key = os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise LLMError("ANTHROPIC_API_KEY is not set (or use --llm claude-cli)")
        body = {
            "model": self.model,
            "max_tokens": max_tokens,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:  # official SDK when it happens to be installed
            import anthropic  # type: ignore

            client = anthropic.Anthropic(api_key=key)
            with client.messages.stream(**body) as stream:
                response = stream.get_final_message()
            return "".join(b.text for b in response.content if b.type == "text").strip()
        except ImportError:
            pass
        request = urllib.request.Request(
            API_URL,
            data=json.dumps(body).encode("utf-8"),
            headers={
                "content-type": "application/json",
                "x-api-key": key,
                "anthropic-version": API_VERSION,
            },
        )
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
        return "".join(
            block.get("text", "") for block in payload.get("content", []) if block.get("type") == "text"
        ).strip()

    def _ask_cli(self, argv: Sequence[str]) -> str:
        if not shutil.which(argv[0]):
            raise LLMError("%s not found on PATH" % argv[0])
        # stdin must be closed: `claude -p` otherwise waits several seconds for
        # piped input that is never coming.
        proc = subprocess.run(list(argv), capture_output=True, text=True,
                              timeout=self.timeout, stdin=subprocess.DEVNULL)
        if proc.returncode != 0:
            raise LLMError("%s failed: %s" % (argv[0], proc.stderr.strip()[:400]))
        return proc.stdout.strip()

    # -- the one call ------------------------------------------------------

    def assist(self, asks: Sequence[Ask], options: Sequence[str]) -> Dict[int, Answer]:
        """Resolve every outstanding gap in the batch with a single request."""
        asks = [a for a in asks if a.needs]
        if not asks:
            return {}
        prompt = build_prompt(asks, options)
        reply = self.ask(prompt, max_tokens=min(16_000, 512 + 240 * len(asks)))
        return parse_batch(reply, {a.index for a in asks})


# -- prompt ----------------------------------------------------------------

NEED_TEXT = {
    "title": '"title": the paper title exactly as printed on the poster, or null',
    "id": '"id": the paper\'s arXiv ID ("arXiv:2401.12345") or DOI ("10.1145/3292500"), '
          "only if you genuinely recognise the paper - otherwise null. Never guess: "
          "every ID is checked against the real record and a wrong one is discarded",
    "subfield": '"subfield": one option copied verbatim from the list below, or null',
}


def excerpt_budget(count: int) -> int:
    """Share the prompt budget across however many posters need help."""
    if count <= 0:
        return MAX_EXCERPT
    return max(MIN_EXCERPT, min(MAX_EXCERPT, PROMPT_BUDGET // count))


def build_prompt(asks: Sequence[Ask], options: Sequence[str]) -> str:
    needed = sorted({need for ask in asks for need in ask.needs})
    limit = excerpt_budget(len(asks))
    lines = [
        "You are helping tidy up photographs of academic conference posters.",
        "Automated processing could not fully resolve the %d posters below." % len(asks),
        "",
        "Reply with one JSON object per line - one line per poster, nothing else, "
        "no code fences, no commentary:",
        '{"n": <poster number>, %s}'
        % ", ".join('"%s": <value or null>' % n for n in needed),
        "",
        "Field rules:",
    ]
    lines += ["- " + NEED_TEXT[n] for n in needed]
    lines += [
        "- Answer only the fields listed on a poster's 'need' line; use null for the rest.",
        "- Use null rather than a guess. A null costs nothing; a wrong answer is worse "
        "than none.",
    ]
    if "subfield" in needed:
        lines += ["", "Subfield options:"] + ["- " + o for o in options]
    lines.append("")
    for ask in asks:
        lines.append("--- poster %d (need: %s) ---" % (ask.index, ", ".join(ask.needs)))
        if ask.title:
            lines.append("Title read off the poster: %s" % ask.title)
        lines.append("OCR text:")
        lines.append(ask.text[:limit].strip() or "(no text recognised)")
        lines.append("")
    return "\n".join(lines)


# -- reply parsing ---------------------------------------------------------

ARXIV_ID = re.compile(r"(?:arxiv[:\s]*)?(?<!\d)(\d{4}\.\d{4,5})(?!\d)(?:v\d+)?", re.I)
DOI = re.compile(r"\b(10\.\d{4,9}/[^\s\"'<>,;)\]]+)", re.I)
_FENCE = re.compile(r"^\s*```[a-z]*\s*$|^\s*```\s*$", re.I)


def parse_identifier(reply: Optional[str]) -> Optional[Tuple[str, str]]:
    """Pull an arXiv ID or DOI out of a model reply, or give up."""
    reply = (reply or "").strip()
    if not reply or reply.upper().startswith("NONE"):
        return None
    doi = DOI.search(reply)
    if doi:
        return ("doi", doi.group(1).rstrip(".,;"))
    arxiv_id = ARXIV_ID.search(reply)
    if arxiv_id:
        return ("arxiv", arxiv_id.group(1))
    return None


def _clean(value) -> Optional[str]:
    if not isinstance(value, str):
        return None
    value = value.strip().strip('"').strip()
    if not value or value.lower() in ("null", "none", "n/a", "unknown"):
        return None
    return value


def parse_batch(reply: str, expected: Optional[set] = None) -> Dict[int, Answer]:
    """Read back one JSON object per line, tolerating fences and stray prose.

    Anything unparseable is skipped rather than guessed at, so a partly mangled
    reply still yields the answers that did come back cleanly.
    """
    answers: Dict[int, Answer] = {}
    records = []
    for line in (reply or "").splitlines():
        line = line.strip()
        if not line or _FENCE.match(line):
            continue
        if line.startswith("["):  # a whole JSON array on one line
            try:
                records.extend(json.loads(line))
                continue
            except ValueError:
                pass
        start = line.find("{")
        if start < 0:
            continue
        try:
            records.append(json.loads(line[start:]))
        except ValueError:
            continue

    for record in records:
        if not isinstance(record, dict):
            continue
        try:
            index = int(record.get("n"))
        except (TypeError, ValueError):
            continue
        if expected is not None and index not in expected:
            continue
        answers[index] = Answer(
            title=_clean(record.get("title")),
            identifier=parse_identifier(_clean(record.get("id"))),
            subfield=_clean(record.get("subfield")),
        )
    return answers
