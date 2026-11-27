"""Tests for the site-generation controller: the Generation page's starts."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
from pathlib import Path
import threading

import pytest
import shiboken6
from PySide6.QtCore import QCoreApplication, QEvent, QObject, Qt, QThread, QTimer, Slot
from PySide6.QtWidgets import QApplication, QWidget

from mufinder.app.workflow_controllers import site_generation as controller_module
from mufinder.app.workflow_controllers.site_generation import (
    SiteGenerationController,
    SiteGenerationDeps,
)
from mufinder.domain.models import MuonSite, Structure
from mufinder.science.site_generation import SiteGenerationError
from mufinder.services.project_service import ProjectService
from tests.support.controllers import make_context
from tests.support.site_generation import held_generation_worker, wait_for_generation
from tests.support.structures import cubic_lattice

_RANDOM = {
    "selection_mode": "count",
    "count": 2,
    "min_muon_atom": 0.8,
    "max_attempts": 5000,
    "seed": "3",
    "include_symmetry_equivalents": False,
}
_MANUAL = {"coord_system": "fractional", "x": 0.25, "y": 0.25, "z": 0.5}


class _FakeView:
    """Records what the controller shows on the page, and on which thread."""

    def __init__(self) -> None:
        self.calls: list[tuple] = []
        self.threads: set[int] = set()

    def _record(self, *call: object) -> None:
        self.threads.add(threading.get_ident())
        self.calls.append(call)

    def set_generation_busy(self, busy: bool, message: str = "") -> None:
        self._record("busy", busy, message)

    def set_generation_progress(self, generated: int, expected: int, message: str = "") -> None:
        self._record("progress", generated, expected, message)

    def set_generation_controls_enabled(self, enabled: bool) -> None:
        self._record("controls", enabled)

    def set_random_muon_muon_distance(self, distance: float) -> None:
        self._record("distance", distance)

    def select_site_ids(self, site_ids: list[str]) -> None:
        self._record("select", list(site_ids))


@dataclass
class _Harness:
    """A controller on a real project, recording everything it asks the window for."""

    service: ProjectService
    document: object
    parent: QWidget
    view: _FakeView = field(default_factory=_FakeView)
    statuses: list[str] = field(default_factory=list)
    persisted: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    closing: bool = False
    _controller: SiteGenerationController | None = None

    @property
    def controller(self) -> SiteGenerationController:
        if self._controller is None:
            context = make_context(
                project_service=self.service, statuses=self.statuses, persisted=self.persisted
            )
            self._controller = SiteGenerationController(
                # The controller reads whichever project the harness has open.
                replace(context, document=lambda: self.document),
                SiteGenerationDeps(
                    dialog_parent=lambda: self.parent,
                    refresh_generation_table=lambda: self.calls.append("table"),
                    refresh_journal_view=lambda: self.calls.append("journal"),
                    shutting_down=lambda: self.closing,
                ),
                self.view,  # type: ignore[arg-type]
            )
        return self._controller


def _cubic_host() -> Structure:
    return Structure(
        label="Host",
        lattice_vectors=cubic_lattice(8.0),
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )


@pytest.fixture
def harness(qapp: QApplication, tmp_path: Path) -> Iterator[_Harness]:
    service = ProjectService()
    document = service.create_project(tmp_path / "generation.mufinder", "Generation")
    service.add_structure(document, _cubic_host())
    parent = QWidget()
    harness = _Harness(service=service, document=document, parent=parent)
    yield harness
    if harness._controller is not None and harness._controller.thread is not None:
        harness._controller.stop_for_close()
    parent.deleteLater()


@pytest.fixture
def messages(monkeypatch: pytest.MonkeyPatch) -> list[tuple[str, str]]:
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        controller_module.QMessageBox,
        "information",
        lambda _parent, title, text: shown.append(("information", title)),
    )
    monkeypatch.setattr(
        controller_module.QMessageBox,
        "warning",
        lambda _parent, title, text: shown.append(("warning", title)),
    )
    return shown


@pytest.fixture
def held_worker(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[threading.Event, threading.Event]]:
    with held_generation_worker(monkeypatch) as events:
        yield events


def _finish(qapp: QApplication, controller: SiteGenerationController) -> None:
    wait_for_generation(qapp, controller)


def test_generate_without_a_project_says_so(harness: _Harness, messages) -> None:
    harness.document = None

    harness.controller.generate("manual", dict(_MANUAL))

    assert messages == [("information", "No project")]
    assert harness.statuses == []


def test_manual_generation_adds_the_start_saves_and_refreshes(harness: _Harness, messages) -> None:
    harness.controller.generate("manual", dict(_MANUAL))

    (site,) = harness.document.sites.values()
    assert site.initial_fractional_position == pytest.approx([0.25, 0.25, 0.5])
    assert harness.persisted == ["Muon sites were generated, but autosave failed."]
    assert harness.calls == ["table", "journal"]
    assert harness.statuses == ["Generated 1 muon site"]
    assert harness.controller.thread is None
    assert messages == []


def test_a_blocked_generation_warns_and_changes_nothing(
    harness: _Harness, messages, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args, **_kwargs):
        raise SiteGenerationError("boom")

    monkeypatch.setattr(harness.service, "generate_candidate_sites", boom)

    harness.controller.generate("manual", dict(_MANUAL))

    assert messages == [("warning", "Generation blocked")]
    assert harness.persisted == []
    assert harness.calls == []
    assert harness.statuses == []


def test_the_status_names_a_shortfall_from_the_requested_count(
    harness: _Harness, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(harness.service, "requested_site_count", lambda *_a, **_k: 3)

    harness.controller.generate("manual", dict(_MANUAL))

    assert harness.statuses == [
        "Generated 1 muon site (requested 3; constraints prevented full count)"
    ]


def test_a_worker_run_is_handled_on_the_gui_thread(harness: _Harness, qapp, messages) -> None:
    """The worker's signals reach the controller queued, on the thread that started it."""

    controller = harness.controller
    controller.generate("random", dict(_RANDOM))

    assert controller.thread is not None
    assert harness.view.calls[0] == ("busy", True, "Preparing random generation…")
    assert harness.statuses == ["Generating random muon sites…"]
    worker = controller.thread
    _finish(qapp, controller)

    assert worker is not None
    assert harness.view.threads == {threading.get_ident()}
    assert len(harness.document.sites) == 2
    assert harness.persisted == ["Muon sites were generated, but autosave failed."]
    assert harness.calls == ["table", "journal"]
    progress = [call for call in harness.view.calls if call[0] == "progress"]
    assert progress[-1] == ("progress", 2, 2, "Generated 2 sites")
    assert all(call[3] == "Generated %v/%m sites" for call in progress[:-1])
    assert [call[0] for call in harness.view.calls[-3:]] == ["progress", "distance", "controls"]
    assert harness.view.calls[-1] == ("controls", True)
    assert harness.statuses[-1].startswith("Generated 2 muon sites (estimated spacing ")
    assert messages == []


def test_a_request_that_cannot_be_prepared_starts_no_worker(
    harness: _Harness, messages, monkeypatch: pytest.MonkeyPatch
) -> None:
    def boom(*_args, **_kwargs):
        raise ValueError("no host")

    monkeypatch.setattr(harness.service, "prepare_candidate_sites", boom)

    harness.controller.generate("near_element", {"element": "Si"})

    assert messages == [("warning", "Generation blocked")]
    assert harness.controller.thread is None
    assert harness.view.calls == []


def test_a_second_request_waits_for_the_running_one(
    harness: _Harness, qapp, messages, held_worker
) -> None:
    computed, release = held_worker
    harness.controller.generate("random", dict(_RANDOM))
    assert computed.wait(10)

    harness.controller.generate("manual", dict(_MANUAL))

    assert messages == [("information", "Generation in progress")]
    release.set()
    _finish(qapp, harness.controller)
    assert len(harness.document.sites) == 2


def test_cancelling_asks_first_and_discards_the_sites(
    harness: _Harness, qapp, messages, held_worker, monkeypatch: pytest.MonkeyPatch
) -> None:
    computed, release = held_worker
    questions: list[str] = []

    def confirm(_parent, title, *_args, **_kwargs) -> bool:
        questions.append(title)
        return True

    monkeypatch.setattr(controller_module.workflow_table_utils, "confirm_destructive", confirm)
    harness.controller.generate("random", dict(_RANDOM))
    assert computed.wait(10)

    harness.controller.cancel_generation()
    # A second request while the first is pending asks nothing.
    harness.controller.cancel_generation()

    assert questions == ["Cancel generation"]
    assert harness.view.calls[-1] == ("progress", 0, 1, "Cancelling…")
    release.set()
    _finish(qapp, harness.controller)

    assert harness.document.sites == {}
    assert ("busy", False, "Generation cancelled") in harness.view.calls
    assert harness.view.calls[-1] == ("controls", True)
    assert harness.statuses[-1] == "Muon site generation cancelled; no sites were added"


def test_declining_to_cancel_keeps_the_run(
    harness: _Harness, qapp, held_worker, monkeypatch: pytest.MonkeyPatch
) -> None:
    computed, release = held_worker
    monkeypatch.setattr(
        controller_module.workflow_table_utils, "confirm_destructive", lambda *_a, **_k: False
    )
    harness.controller.generate("random", dict(_RANDOM))
    assert computed.wait(10)

    harness.controller.cancel_generation()
    release.set()
    _finish(qapp, harness.controller)

    assert len(harness.document.sites) == 2


def test_cancel_without_a_run_does_nothing(harness: _Harness, monkeypatch) -> None:
    asked: list[str] = []
    monkeypatch.setattr(
        controller_module.workflow_table_utils,
        "confirm_destructive",
        lambda *_a, **_k: asked.append("asked"),
    )

    harness.controller.cancel_generation()

    assert asked == []


def test_a_host_change_is_refused_only_while_a_run_computes(
    harness: _Harness, qapp, messages, held_worker
) -> None:
    computed, release = held_worker
    assert harness.controller.blocks_host_change() is False

    harness.controller.generate("random", dict(_RANDOM))
    assert computed.wait(10)

    assert harness.controller.blocks_host_change() is True
    assert messages == [("information", "Generation in progress")]
    release.set()
    _finish(qapp, harness.controller)
    assert harness.controller.blocks_host_change() is False


def test_a_run_started_in_a_closed_project_is_discarded(
    harness: _Harness, qapp, tmp_path: Path, held_worker
) -> None:
    computed, release = held_worker
    first = harness.document
    harness.controller.generate("random", dict(_RANDOM))
    assert computed.wait(10)
    harness.document = harness.service.create_project(tmp_path / "second.mufinder", "Second")

    release.set()
    _finish(qapp, harness.controller)

    assert first.sites == {}
    assert harness.document.sites == {}
    assert ("busy", False, "Ready") in harness.view.calls
    assert harness.statuses[-1] == (
        "Discarded the muon site generation started in the previously open project"
    )


def test_a_failed_run_warns_and_leaves_the_controls_off(
    harness: _Harness, qapp, messages, monkeypatch: pytest.MonkeyPatch
) -> None:
    def fail(_request, **_kwargs):
        raise SiteGenerationError("no room")

    monkeypatch.setattr(ProjectService, "compute_candidate_sites", staticmethod(fail))

    harness.controller.generate("near_element", {"element": "Si", "bond_distance": 1.0})
    _finish(qapp, harness.controller)

    assert messages == [("warning", "Generation blocked")]
    assert harness.statuses[-1] == "Near-element site generation failed"
    assert harness.view.calls[-1] == ("busy", False, "Generation failed")
    assert ("controls", True) not in harness.view.calls


@pytest.mark.parametrize(
    ("error", "text"),
    [
        (
            IndexError("index 4 is out of bounds for axis 0 with size 3."),
            "Muon site generation stopped: index 4 is out of bounds for axis 0 with size 3. "
            "No sites were added.",
        ),
        (
            ZeroDivisionError(),
            "Muon site generation stopped because of an unexpected error. No sites were added.",
        ),
        (
            KeyError("fractional"),
            "Muon site generation stopped because of an unexpected error. No sites were added.",
        ),
        (
            RuntimeError("singular matrix.\nwhile reducing the cell"),
            "Muon site generation stopped: singular matrix. No sites were added.",
        ),
    ],
)
def test_an_unexpected_worker_error_warns_in_plain_words(
    harness: _Harness,
    qapp,
    monkeypatch: pytest.MonkeyPatch,
    caplog: pytest.LogCaptureFixture,
    error: Exception,
    text: str,
) -> None:
    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        controller_module.QMessageBox,
        "warning",
        lambda _parent, title, text: shown.append((title, text)),
    )

    def fail(_request, **_kwargs):
        raise error

    monkeypatch.setattr(ProjectService, "compute_candidate_sites", staticmethod(fail))

    with caplog.at_level("ERROR"):
        harness.controller.generate("near_element", {"element": "Si", "bond_distance": 1.0})
        _finish(qapp, harness.controller)

    assert shown == [("Generation failed", text)]
    assert type(error).__name__ in caplog.text
    assert type(error).__name__ not in text
    assert harness.statuses[-1] == "Near-element site generation failed"
    assert harness.view.calls[-1] == ("busy", False, "Generation failed")
    assert harness.document.sites == {}
    assert harness.persisted == []


def test_nothing_is_applied_while_the_window_closes(
    harness: _Harness, qapp, messages, held_worker
) -> None:
    computed, release = held_worker
    harness.controller.generate("random", dict(_RANDOM))
    assert computed.wait(10)
    harness.closing = True

    release.set()
    _finish(qapp, harness.controller)

    assert harness.document.sites == {}
    assert harness.persisted == []
    assert harness.statuses == ["Generating random muon sites…"]


def test_stop_for_close_cancels_and_waits_for_the_run(harness: _Harness, qapp, held_worker) -> None:
    computed, release = held_worker
    harness.controller.generate("random", dict(_RANDOM))
    assert computed.wait(10)
    worker = harness.controller.thread
    assert worker is not None
    # A direct connection runs on the worker's thread as it finishes, whatever loop runs here.
    finished: list[bool] = []
    worker.finished.connect(lambda: finished.append(True), Qt.ConnectionType.DirectConnection)
    release.set()

    harness.controller.stop_for_close()

    # The thread deletes itself once finished, so it is not called after the stop.
    assert finished == [True]
    assert not shiboken6.isValid(worker) or worker.isFinished()
    assert harness.controller.thread is None
    qapp.processEvents()
    assert harness.document.sites == {}


class _DeleteOnFinish(QObject):
    """Runs a finished thread's own ``deleteLater`` at once, as the close's loop may.

    Connected after the controller's ``deleteLater``, so the deferred deletion
    is already posted when this runs; it is delivered here rather than left to
    whichever loop gets to it.
    """

    def __init__(self, thread: QThread) -> None:
        super().__init__()
        self._thread = thread

    @Slot()
    def delete(self) -> None:
        QCoreApplication.sendPostedEvents(self._thread, QEvent.Type.DeferredDelete)


def test_stop_for_close_survives_the_thread_deleting_itself_while_it_waits(
    harness: _Harness, qapp, held_worker
) -> None:
    """The finished thread's ``deleteLater`` can run inside the close's event loop.

    The run is still held when the close starts waiting, and is released from
    that loop; the thread is then deleted in the loop, as a deferred deletion
    may be, before the close joins it.
    """

    computed, release = held_worker
    harness.controller.generate("random", dict(_RANDOM))
    assert computed.wait(10)
    worker = harness.controller.thread
    assert worker is not None
    finished: list[bool] = []
    worker.finished.connect(lambda: finished.append(True), Qt.ConnectionType.DirectConnection)
    deleter = _DeleteOnFinish(worker)
    worker.finished.connect(deleter.delete)
    QTimer.singleShot(50, release.set)

    harness.controller.stop_for_close()

    assert finished == [True]
    assert not shiboken6.isValid(worker)
    assert harness.controller.thread is None
    qapp.processEvents()
    assert harness.document.sites == {}


def test_stop_for_close_without_a_run_does_nothing(harness: _Harness) -> None:
    harness.controller.stop_for_close()

    assert harness.controller.thread is None
    assert harness.view.calls == []


@pytest.mark.parametrize(
    ("method", "mode", "text"),
    [
        ("random", "separation", "Generated %v sites"),
        ("random", "distance", "Generated %v sites"),
        ("random", "spacing", "Generated %v sites"),
        ("random", "count", "Generated %v/%m sites"),
        ("near_element", "", "Generated %v/%m sites"),
    ],
)
def test_progress_text_omits_the_total_for_a_separation_run(
    harness: _Harness, method: str, mode: str, text: str
) -> None:
    harness.controller.generation_progressed(method, 6, 13, mode)

    assert harness.view.calls == [("progress", 6, 13, text)]


def test_delete_sites_removes_generated_starts_only(harness: _Harness, messages) -> None:
    candidate = MuonSite(label="Candidate", status="generated", generation_method="manual")
    relaxed = MuonSite(label="Relaxed", status="relaxed", generation_method="manual")
    harness.document.sites[candidate.id] = candidate
    harness.document.sites[relaxed.id] = relaxed

    harness.controller.delete_sites([str(candidate.id), str(relaxed.id), "not-a-uuid"])

    assert candidate.id not in harness.document.sites
    assert relaxed.id in harness.document.sites
    assert harness.persisted == ["Muon sites were deleted, but autosave failed."]
    assert harness.calls == ["table", "journal"]
    assert harness.statuses == ["Deleted 1 generated muon site"]


def test_delete_sites_is_refused_while_a_run_computes(
    harness: _Harness, qapp, messages, held_worker
) -> None:
    candidate = MuonSite(label="Candidate", status="generated", generation_method="manual")
    harness.document.sites[candidate.id] = candidate
    computed, release = held_worker
    harness.controller.generate("random", dict(_RANDOM))
    assert computed.wait(10)

    harness.controller.delete_sites([str(candidate.id)])

    assert messages == [("information", "Generation in progress")]
    assert candidate.id in harness.document.sites
    assert harness.persisted == []
    assert harness.calls == []
    release.set()
    _finish(qapp, harness.controller)

    harness.controller.delete_sites([str(candidate.id)])
    assert candidate.id not in harness.document.sites


def test_a_failed_run_refuses_nothing_while_its_thread_winds_down(
    harness: _Harness, qapp, messages, held_worker
) -> None:
    candidate = MuonSite(label="Candidate", status="generated", generation_method="manual")
    harness.document.sites[candidate.id] = candidate
    computed, release = held_worker
    harness.controller.generate("random", dict(_RANDOM))
    assert computed.wait(10)
    # The failure has been reported; the thread has not ended yet.
    harness.controller.generation_failed("random", "no room")

    harness.controller.delete_sites([str(candidate.id)])

    assert messages == [("warning", "Generation blocked")]
    assert candidate.id not in harness.document.sites
    release.set()
    _finish(qapp, harness.controller)


def test_cluster_positions_is_refused_while_a_run_computes(
    harness: _Harness, qapp, messages, held_worker
) -> None:
    site = MuonSite(
        label="Candidate",
        status="generated",
        generation_method="manual",
        initial_fractional_position=[0.1, 0.2, 0.3],
        initial_cartesian_position=[0.8, 1.6, 2.4],
    )
    harness.document.sites[site.id] = site
    payload = {"site_ids": [str(site.id)], "anchor_fractional": [0.9, 0.8, 0.7]}
    computed, release = held_worker
    harness.controller.generate("random", dict(_RANDOM))
    assert computed.wait(10)

    harness.controller.cluster_positions(payload)

    assert messages == [("information", "Generation in progress")]
    assert site.initial_fractional_position == pytest.approx([0.1, 0.2, 0.3])
    assert harness.persisted == []
    release.set()
    _finish(qapp, harness.controller)

    harness.controller.cluster_positions(payload)
    assert site.initial_fractional_position == pytest.approx([0.9, 0.8, 0.7])


def test_delete_sites_that_removes_nothing_does_not_save(harness: _Harness) -> None:
    relaxed = MuonSite(label="Relaxed", status="relaxed", generation_method="manual")
    harness.document.sites[relaxed.id] = relaxed

    harness.controller.delete_sites([str(relaxed.id)])

    assert harness.persisted == []
    assert harness.calls == ["table", "journal"]
    assert harness.statuses == ["Deleted 0 generated muon sites"]


def test_delete_sites_ignores_a_request_with_no_valid_id(harness: _Harness) -> None:
    harness.controller.delete_sites(["not-a-uuid"])

    assert harness.calls == []
    assert harness.statuses == []


def test_delete_sites_without_a_project_says_so(harness: _Harness, messages) -> None:
    harness.document = None

    harness.controller.delete_sites(["x"])

    assert messages == [("information", "No project")]


def test_cluster_positions_moves_selects_and_saves(harness: _Harness, messages) -> None:
    site = MuonSite(
        label="Candidate",
        status="generated",
        generation_method="manual",
        initial_fractional_position=[0.1, 0.2, 0.3],
        initial_cartesian_position=[0.8, 1.6, 2.4],
    )
    harness.document.sites[site.id] = site

    harness.controller.cluster_positions(
        {"site_ids": [str(site.id)], "anchor_fractional": [0.9, 0.8, 0.7]}
    )

    assert site.initial_fractional_position == pytest.approx([0.9, 0.8, 0.7])
    assert harness.persisted == ["Muon positions were clustered, but autosave failed."]
    assert harness.calls == ["table", "journal"]
    assert harness.view.calls == [("select", [str(site.id)])]
    assert harness.statuses == ["Clustered 1 muon position"]
    assert messages == []


def test_cluster_positions_warns_when_blocked(harness: _Harness, messages) -> None:
    harness.controller.cluster_positions({"site_ids": [], "anchor_fractional": [0.5]})

    assert messages == [("warning", "Clustering blocked")]
    assert harness.persisted == []
    assert harness.statuses == []


def test_cluster_positions_ignores_a_payload_that_is_not_a_dict(harness: _Harness) -> None:
    harness.controller.cluster_positions(["not", "a", "dict"])

    assert harness.calls == []


def test_cluster_positions_without_a_project_says_so(harness: _Harness, messages) -> None:
    harness.document = None

    harness.controller.cluster_positions({})

    assert messages == [("information", "No project")]


def test_store_form_state_keeps_the_form_in_ui_state(harness: _Harness) -> None:
    harness.controller.store_form_state({"method": "random"})

    assert harness.document.ui_state["muon_generation_form"] == {"method": "random"}


def test_store_form_state_is_ignored_while_the_window_closes(harness: _Harness) -> None:
    harness.closing = True

    harness.controller.store_form_state({"method": "random"})

    assert "muon_generation_form" not in harness.document.ui_state
