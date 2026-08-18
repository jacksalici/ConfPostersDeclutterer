"""Levelled logging to stderr, so stdout stays parseable.

Three levels: -q prints only warnings, the default prints one line per poster,
-v explains every decision - which query ran, whether it was cached, what scored
what, and why a match was accepted or declined.
"""

from __future__ import annotations

import sys
import time
from contextlib import contextmanager

QUIET, NORMAL, VERBOSE = 0, 1, 2

_DIM, _BOLD, _WARN, _OFF = "\033[2m", "\033[1m", "\033[33m", "\033[0m"


class Log:
    def __init__(self, level: int = NORMAL, stream=None):
        self.level = level
        self.stream = stream or sys.stderr
        self.colour = hasattr(self.stream, "isatty") and self.stream.isatty()

    @property
    def verbose(self) -> bool:
        return self.level >= VERBOSE

    def _write(self, text: str, style: str = "") -> None:
        if self.colour and style:
            text = "%s%s%s" % (style, text, _OFF)
        print(text, file=self.stream)
        self.stream.flush()

    def info(self, message: str) -> None:
        """One line per poster or per stage. Suppressed by -q."""
        if self.level >= NORMAL:
            self._write(message)

    def head(self, message: str) -> None:
        if self.level >= NORMAL:
            self._write(message, _BOLD)

    def detail(self, message: str, indent: int = 1) -> None:
        """Why something happened. Only with -v."""
        if self.level >= VERBOSE:
            self._write("%s%s" % ("  " * indent, message), _DIM)

    def warn(self, message: str) -> None:
        """Something went wrong but the run continues. Always shown."""
        self._write("! %s" % message, _WARN)

    @contextmanager
    def timed(self, label: str, indent: int = 1):
        """Time a stage and report it, but only when -v is on."""
        started = time.time()
        try:
            yield
        finally:
            self.detail("%s took %s" % (label, human_time(time.time() - started)), indent)


def human_time(seconds: float) -> str:
    if seconds < 1:
        return "%dms" % round(seconds * 1000)
    if seconds < 60:
        return "%.1fs" % seconds
    return "%dm%02ds" % (int(seconds // 60), int(seconds % 60))


def from_flags(quiet: bool = False, verbose: bool = False, stream=None) -> Log:
    level = QUIET if quiet else (VERBOSE if verbose else NORMAL)
    return Log(level, stream)
