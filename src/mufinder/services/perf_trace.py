"""Lightweight stage-timing instrumentation for GUI-responsiveness diagnostics.

Long-running GUI operations (dashboard rebuilds, project saves) are made of
several stages, any one of which can dominate the wall-clock time on a slow
project or disk. This module gives both ``mufinder.services`` and
``mufinder.app`` a shared, dependency-free way to time named stages and log
the result, so we can see where time goes without hand-rolled ``time.time()``
calls scattered across call sites. It lives in ``mufinder.services`` (rather
than ``mufinder.app``) because ``mufinder.services`` may not import
``mufinder.app``, while the reverse is allowed.

Overhead is kept negligible: timing itself is always a cheap
``time.perf_counter()`` call, but message formatting is skipped entirely
unless the target logger is enabled for the level that will actually be
emitted.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
import logging
import time

#: A stage or total duration above this many milliseconds escalates the
#: summary log line from DEBUG to WARNING.
WARNING_THRESHOLD_MS = 50.0

#: A total duration above this many milliseconds escalates the summary log
#: line to WARNING even when every individual stage stayed under threshold.
TOTAL_WARNING_THRESHOLD_MS = 200.0


@dataclass(slots=True)
class StageTiming:
    """One named stage's measured duration."""

    name: str
    duration_ms: float


@dataclass(slots=True)
class PerfTrace:
    """Collects named stage timings for one traced operation.

    Use :meth:`stage` as a context manager around each stage, then call
    :meth:`log_summary` once at the end to emit a single log line covering
    every stage plus the total.
    """

    label: str
    logger: logging.Logger
    stages: list[StageTiming] = field(default_factory=list)
    _start: float = field(default_factory=time.perf_counter)

    @contextmanager
    def stage(self, name: str) -> Iterator[None]:
        """Time one named stage and record its duration."""

        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            self.stages.append(StageTiming(name=name, duration_ms=elapsed_ms))

    @property
    def total_ms(self) -> float:
        """Elapsed time since this trace was created, in milliseconds."""

        return (time.perf_counter() - self._start) * 1000.0

    def log_summary(self, *, extra: str = "") -> None:
        """Emit one log line summarizing every stage and the total.

        WARNING is used when any single stage exceeds
        :data:`WARNING_THRESHOLD_MS` or the total exceeds
        :data:`TOTAL_WARNING_THRESHOLD_MS`; DEBUG otherwise. Message
        formatting only happens once the chosen level is confirmed enabled.
        """

        total_ms = self.total_ms
        is_slow = total_ms > TOTAL_WARNING_THRESHOLD_MS or any(
            stage.duration_ms > WARNING_THRESHOLD_MS for stage in self.stages
        )
        level = logging.WARNING if is_slow else logging.DEBUG
        if not self.logger.isEnabledFor(level):
            return
        stage_text = ", ".join(f"{stage.name}={stage.duration_ms:.1f}ms" for stage in self.stages)
        suffix = f" ({extra})" if extra else ""
        self.logger.log(
            level,
            "%s%s: total=%.1fms [%s]",
            self.label,
            suffix,
            total_ms,
            stage_text,
        )
