"""Tests for running Python's cyclic garbage collector on the GUI thread only."""

from __future__ import annotations

import gc
import threading
import time
import weakref

from PySide6.QtCore import QObject

from mufinder.app.gui_garbage_collector import GuiGarbageCollector, install_gui_garbage_collector


class _Holder:
    """A reference cycle around a QObject, which only the cyclic collector frees."""

    def __init__(self) -> None:
        self.qobject = QObject()
        self.cycle = self


def _garbage_cycle(freed_on: list[str]) -> None:
    holder = _Holder()
    weakref.finalize(holder, lambda: freed_on.append(threading.current_thread().name))


def _allocate_past_every_threshold() -> list[list[list[object]]]:
    # Each live container allocation advances the young generation's count and
    # a freed one takes it back, so the caller keeps the lists until it has
    # checked; together they pass every generation's threshold.
    return [[[] for _ in range(1_000)] for _ in range(200)]


def _allocate_in_worker() -> list[object]:
    kept: list[object] = []
    worker = threading.Thread(target=lambda: kept.append(_allocate_past_every_threshold()), name="test-worker")
    worker.start()
    worker.join()
    return kept


def _pump_until(app, condition, *, timeout_s: float = 5.0) -> None:
    deadline = time.monotonic() + timeout_s
    while not condition() and time.monotonic() < deadline:
        app.processEvents()
        time.sleep(0.005)


def test_a_worker_thread_never_frees_qobjects_while_the_collector_runs(qapp) -> None:
    collector = GuiGarbageCollector(interval_ms=10)
    freed_on: list[str] = []
    collector.start()
    try:
        _garbage_cycle(freed_on)
        kept = _allocate_in_worker()

        assert freed_on == []

        _pump_until(qapp, lambda: bool(freed_on))
        del kept
    finally:
        collector.stop()

    assert freed_on == [threading.main_thread().name]


def test_the_collector_only_collects_once_a_threshold_is_passed(qapp) -> None:
    collector = GuiGarbageCollector()
    collector.start()
    try:
        gc.collect()

        assert collector.collect_if_due() is None

        kept = _allocate_past_every_threshold()

        assert collector.collect_if_due() in (0, 1, 2)
        del kept
    finally:
        collector.stop()


def test_stop_turns_automatic_collection_back_on(without_session_garbage_collector) -> None:
    assert gc.isenabled()
    collector = GuiGarbageCollector()

    collector.start()

    assert not gc.isenabled()

    collector.stop()

    assert gc.isenabled()


def test_stop_leaves_automatic_collection_off_when_it_was_off_before(without_session_garbage_collector) -> None:
    gc.disable()
    collector = GuiGarbageCollector()
    try:
        collector.start()
        collector.stop()

        assert not gc.isenabled()
    finally:
        gc.enable()


def test_automatic_collection_stays_off_until_the_last_collector_stops(without_session_garbage_collector) -> None:
    first = GuiGarbageCollector()
    second = GuiGarbageCollector()
    first.start()
    second.start()

    first.stop()

    assert not gc.isenabled()
    assert second.is_running()

    second.stop()

    assert gc.isenabled()


def test_the_test_session_frees_cycles_on_the_gui_thread_only(qapp) -> None:
    assert not gc.isenabled()


def test_install_starts_a_collector_owned_by_the_application(qapp) -> None:
    collector = install_gui_garbage_collector(qapp)
    try:
        assert collector.parent() is qapp
        assert collector.is_running()
        assert not gc.isenabled()
    finally:
        collector.stop()
        collector.setParent(None)
        collector.deleteLater()
