"""Tests for the stage-timing helper in services.perf_trace."""

from __future__ import annotations

import logging
import time

from mufinder.services.perf_trace import (
    TOTAL_WARNING_THRESHOLD_MS,
    WARNING_THRESHOLD_MS,
    PerfTrace,
    StageTiming,
)


def test_stage_context_manager_records_named_durations() -> None:
    trace = PerfTrace(label="test", logger=logging.getLogger("mufinder.test.perf_trace"))

    with trace.stage("alpha"):
        pass
    with trace.stage("beta"):
        pass

    names = [stage.name for stage in trace.stages]
    assert names == ["alpha", "beta"]
    assert all(stage.duration_ms >= 0.0 for stage in trace.stages)


def test_stage_records_elapsed_time() -> None:
    trace = PerfTrace(label="test", logger=logging.getLogger("mufinder.test.perf_trace"))

    with trace.stage("slow"):
        time.sleep(0.01)

    assert trace.stages[0].duration_ms >= 10.0


def test_log_summary_uses_debug_when_fast(caplog) -> None:
    logger = logging.getLogger("mufinder.test.perf_trace.fast")
    trace = PerfTrace(label="fast flush", logger=logger)
    with trace.stage("quick"):
        pass

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        trace.log_summary()

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.DEBUG
    assert "fast flush" in caplog.records[0].message
    assert "quick=" in caplog.records[0].message


def test_log_summary_escalates_to_warning_when_a_stage_is_slow(caplog) -> None:
    logger = logging.getLogger("mufinder.test.perf_trace.slow_stage")
    trace = PerfTrace(label="slow stage flush", logger=logger)
    trace.stages.append(StageTiming(name="heavy", duration_ms=WARNING_THRESHOLD_MS + 1.0))

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        trace.log_summary()

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING


def test_log_summary_escalates_to_warning_when_total_is_slow(caplog) -> None:
    logger = logging.getLogger("mufinder.test.perf_trace.slow_total")
    trace = PerfTrace(label="slow total flush", logger=logger)
    # No individual stage crosses the per-stage threshold, but the elapsed
    # total (since trace construction) can still trip the total threshold.
    trace._start -= (TOTAL_WARNING_THRESHOLD_MS + 50.0) / 1000.0

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        trace.log_summary()

    assert len(caplog.records) == 1
    assert caplog.records[0].levelno == logging.WARNING


def test_log_summary_skips_formatting_when_level_disabled() -> None:
    logger = logging.getLogger("mufinder.test.perf_trace.disabled")
    logger.setLevel(logging.CRITICAL)
    trace = PerfTrace(label="disabled flush", logger=logger)
    with trace.stage("quick"):
        pass

    # Should not raise even though no handler/level is configured to emit it;
    # this also documents that isEnabledFor gates formatting, not just emission.
    trace.log_summary()


def test_log_summary_includes_extra_suffix(caplog) -> None:
    logger = logging.getLogger("mufinder.test.perf_trace.extra")
    trace = PerfTrace(label="save", logger=logger)
    with trace.stage("write"):
        pass

    with caplog.at_level(logging.DEBUG, logger=logger.name):
        trace.log_summary(extra="entity_files=3")

    assert "entity_files=3" in caplog.records[0].message
