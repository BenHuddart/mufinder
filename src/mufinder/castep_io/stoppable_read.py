"""Read and search a large CASTEP output in pieces, so a caller can stop part-way.

A parser given a ``should_stop`` callable asks it between pieces of fixed size and
raises :class:`ReadStopped` once it answers true, so how long a stop waits is set
by one piece, not by the size of the file. Without a callable each helper falls
straight through to the plain library call, so a caller that never stops reads
exactly as before.

Pieces are cut so the result is the same as reading the whole file at once:

- the text is read through one text stream, which decodes and translates line
  endings across the cuts;
- lines are split at a newline, so no line is cut in two;
- a pattern is searched window by window, each window running on
  :data:`LONGEST_MATCH_CHARACTERS` past its cut and keeping only the matches that
  start before it. A match is found whole while it is shorter than that, and the
  patterns run over CASTEP output match within a line or two.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Iterator
from itertools import chain, islice
from pathlib import Path
import re

#: Asked between pieces; answering true stops the read.
StopCheck = Callable[[], bool]

#: Characters read, or split into lines, between two stop checks.
CHARACTERS_PER_CHECK = 1 << 22
#: Lines walked between two stop checks.
LINES_PER_CHECK = 1 << 15
#: How far past its cut a search window runs, so a match that starts before the
#: cut is found whole. Far longer than any line CASTEP writes.
LONGEST_MATCH_CHARACTERS = 1 << 16


class ReadStopped(Exception):
    """Raised by a read whose ``should_stop`` answered true part-way through."""


def stop_if_asked(should_stop: StopCheck | None) -> None:
    """Raise :class:`ReadStopped` if ``should_stop`` answers true."""

    if should_stop is not None and should_stop():
        raise ReadStopped


def read_text(path: Path, should_stop: StopCheck | None = None) -> str:
    """Return ``path`` as ``Path.read_text(encoding="utf-8", errors="replace")`` does."""

    if should_stop is None:
        return path.read_text(encoding="utf-8", errors="replace")
    pieces: list[str] = []
    with path.open(encoding="utf-8", errors="replace") as handle:
        while True:
            stop_if_asked(should_stop)
            piece = handle.read(CHARACTERS_PER_CHECK)
            if not piece:
                break
            pieces.append(piece)
    return "".join(pieces)


def split_lines(text: str, should_stop: StopCheck | None = None) -> list[str]:
    """Return ``text.splitlines()``."""

    if should_stop is None:
        return text.splitlines()
    lines: list[str] = []
    start = 0
    while start < len(text):
        stop_if_asked(should_stop)
        newline = text.find("\n", start + CHARACTERS_PER_CHECK)
        end = len(text) if newline < 0 else newline + 1
        lines.extend(text[start:end].splitlines())
        start = end
    return lines


def checked_lines(lines: list[str], should_stop: StopCheck | None = None) -> Iterable[str]:
    """Iterate ``lines``, asking ``should_stop`` before each run of lines."""

    if should_stop is None:
        return lines
    return chain.from_iterable(line_runs(lines, should_stop))


def line_runs(lines: list[str], should_stop: StopCheck | None = None) -> Iterator[Iterable[str]]:
    """Yield ``lines`` in runs of :data:`LINES_PER_CHECK`, asking ``should_stop`` before each.

    Each run must be used up before the next is taken: they share one iterator,
    so the lines are walked without being copied.
    """

    remaining = iter(lines)
    for _start in range(0, len(lines), LINES_PER_CHECK):
        stop_if_asked(should_stop)
        yield islice(remaining, LINES_PER_CHECK)


def ask_every_run(walked: int, should_stop: StopCheck | None) -> None:
    """Ask ``should_stop`` once ``walked`` lines make up another run of lines."""

    if should_stop is not None and walked and not walked % LINES_PER_CHECK:
        stop_if_asked(should_stop)


def next_line_index(
    lines: list[str],
    start: int,
    wanted: Callable[[str], bool],
    should_stop: StopCheck | None = None,
) -> int:
    """Return the index of the first line from ``start`` on that is ``wanted``, else ``len(lines)``."""

    step = LINES_PER_CHECK if should_stop is not None else max(len(lines), 1)
    for run_start in range(start, len(lines), step):
        stop_if_asked(should_stop)
        for index in range(run_start, min(run_start + step, len(lines))):
            if wanted(lines[index]):
                return index
    return len(lines)


def finditer(
    pattern: re.Pattern[str], text: str, should_stop: StopCheck | None = None
) -> Iterator[re.Match[str]]:
    """Yield the matches ``pattern.finditer(text)`` does, a window at a time.

    ``pattern`` must not match an empty string nor look past the end of its
    match (``$``, ``\\Z``, a lookahead, ``\\b``): a window ends where the text
    does not, and a match shorter than :data:`LONGEST_MATCH_CHARACTERS` is then
    the same in the window as in the whole text.
    """

    if should_stop is None:
        yield from pattern.finditer(text)
        return
    position = 0
    while position < len(text):
        stop_if_asked(should_stop)
        cut = min(len(text), position + CHARACTERS_PER_CHECK)
        for match in pattern.finditer(
            text, position, min(len(text), cut + LONGEST_MATCH_CHARACTERS)
        ):
            if match.start() >= cut:
                break
            yield match
            position = match.end()
        position = max(position, cut)


def search(
    pattern: re.Pattern[str], text: str, should_stop: StopCheck | None = None
) -> re.Match[str] | None:
    """Return what ``pattern.search(text)`` does."""

    return next(finditer(pattern, text, should_stop), None)


__all__ = [
    "CHARACTERS_PER_CHECK",
    "LINES_PER_CHECK",
    "LONGEST_MATCH_CHARACTERS",
    "ReadStopped",
    "StopCheck",
    "ask_every_run",
    "checked_lines",
    "finditer",
    "line_runs",
    "next_line_index",
    "read_text",
    "search",
    "split_lines",
]
