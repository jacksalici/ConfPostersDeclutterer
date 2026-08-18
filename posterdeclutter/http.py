"""One tiny cached HTTP client, shared by every lookup source.

Responses are cached on disk by URL, so re-running is free and `--offline`
works. Each host gets its own minimum interval between live requests, because
arXiv asks for one request every three seconds and the others do not.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Dict, Optional

from .log import Log, human_time

USER_AGENT = "posterdeclutter/0.2 (conference poster tidier; +https://arxiv.org/help/api)"

# Seconds to wait between live requests, per host. arXiv's is a published rule.
HOST_INTERVALS: Dict[str, float] = {
    "export.arxiv.org": 3.0,
    "api.openalex.org": 0.15,
    "api.crossref.org": 0.15,
}
DEFAULT_INTERVAL = 1.0


class Fetcher:
    def __init__(
        self,
        cache_dir: Path,
        offline: bool = False,
        timeout: float = 30.0,
        mailto: str = "",
        log: Optional[Log] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.offline = offline
        self.timeout = timeout
        self.mailto = mailto  # joins the OpenAlex/Crossref "polite pools"
        self.log = log or Log()
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._last_call: Dict[str, float] = {}
        self.live_requests = 0
        self.cache_hits = 0

    def clear_cache(self) -> int:
        """Drop every cached response. Returns how many were removed."""
        count = len(list(self.cache_dir.glob("*.body")))
        shutil.rmtree(self.cache_dir, ignore_errors=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        return count

    def _wait(self, host: str) -> None:
        interval = HOST_INTERVALS.get(host, DEFAULT_INTERVAL)
        gap = interval - (time.time() - self._last_call.get(host, 0.0))
        if gap > 0:
            time.sleep(gap)

    def get(self, url: str) -> str:
        key = hashlib.sha256(url.encode("utf-8")).hexdigest()[:32]
        path = self.cache_dir / (key + ".body")
        if path.exists():
            self.cache_hits += 1
            self.log.detail("cached  %s" % _short(url), indent=2)
            return path.read_text(encoding="utf-8")
        if self.offline:
            self.log.detail("offline %s (not cached)" % _short(url), indent=2)
            raise urllib.error.URLError("offline mode: %s is not cached" % url)

        host = urllib.parse.urlsplit(url).netloc
        waited = time.time()
        self._wait(host)
        waited = time.time() - waited
        headers = {"User-Agent": USER_AGENT, "Accept": "application/json, application/atom+xml"}
        if self.mailto:
            headers["User-Agent"] = "%s mailto:%s" % (USER_AGENT, self.mailto)
        request = urllib.request.Request(url, headers=headers)
        started = time.time()
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            body = response.read().decode("utf-8", "replace")
        self._last_call[host] = time.time()
        self.live_requests += 1
        path.write_text(body, encoding="utf-8")
        self.log.detail(
            "GET     %s (%s%s, %s)"
            % (_short(url), human_time(time.time() - started),
               " after %s rate-limit wait" % human_time(waited) if waited > 0.05 else "",
               _size(len(body))),
            indent=2,
        )
        return body

    def get_json(self, url: str) -> dict:
        return json.loads(self.get(url))

    def polite(self, params: dict) -> dict:
        """Add the contact address the OpenAlex/Crossref polite pools want."""
        if self.mailto:
            params = dict(params, mailto=self.mailto)
        return params


def _short(url: str, limit: int = 110) -> str:
    """URLs are long and mostly boilerplate; show the interesting tail."""
    url = urllib.parse.unquote(url)
    return url if len(url) <= limit else url[: limit - 1] + "\u2026"


def _size(count: int) -> str:
    return "%dB" % count if count < 1024 else "%.1fkB" % (count / 1024.0)
