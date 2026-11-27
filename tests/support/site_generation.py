"""Helpers for tests that run the muon site-generation worker thread."""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager
import threading
import time

import pytest
from PySide6.QtWidgets import QApplication

from mufinder.app.workflow_controllers.site_generation import SiteGenerationController
from mufinder.services.project_service import ProjectService


@contextmanager
def held_generation_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[threading.Event, threading.Event]]:
    """Hold each generation worker once it has computed its sites, until released.

    Yields ``(computed, release)``: ``computed`` is set when the worker holds,
    and setting ``release`` lets it hand its sites back. Leaving the block
    releases any worker still held.
    """

    computed = threading.Event()
    release = threading.Event()
    original = ProjectService.compute_candidate_sites

    def held(request, **kwargs):
        outcome = original(request, **kwargs)
        computed.set()
        release.wait(10)
        return outcome

    monkeypatch.setattr(ProjectService, "compute_candidate_sites", staticmethod(held))
    try:
        yield computed, release
    finally:
        release.set()


def wait_for_generation(qapp: QApplication, controller: SiteGenerationController) -> None:
    """Process events until ``controller``'s generation thread has ended and been handled."""

    for _ in range(2000):
        qapp.processEvents()
        if controller.thread is None:
            return
        time.sleep(0.005)
    raise AssertionError("the generation worker did not finish")
