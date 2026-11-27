"""Tests for the imported-relaxations controller, driven without a main window."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field, replace
import os
from pathlib import Path
import shutil
import sys
import threading

import pytest
import shiboken6
from PySide6.QtCore import Qt, QTimer
from PySide6.QtWidgets import QApplication, QDialog, QProgressDialog, QWidget

from mufinder.app import blocking_call
from mufinder.app.muon_relaxation_widget import MuonRelaxationWidget
from mufinder.app.relaxation_import_dialog import _LIVE_SCANS
from mufinder.app.workflow_controllers.imported_relaxations import (
    ImportedRelaxationsController,
    ImportedRelaxationsDeps,
)
from mufinder.domain.models import Structure
from mufinder.services.project_service import ProjectDocument, ProjectService
from mufinder.services.relaxation_import import scan_relaxation_folder
from mufinder.services.relaxation_import_commit import stage_relaxation_import
from mufinder.workflow.muon_site import ACTIVE_MUON_RELAXATION_DEFINITION_KEY
from tests.services.test_relaxation_import_scan import HOST_LATTICE, build_folder
from tests.support.controllers import make_context, muon_host_document
from tests.support.qt import pump_until

_CONTROLLER = "mufinder.app.workflow_controllers.imported_relaxations"


@dataclass
class _Harness:
    """A controller with no project, recording what it asks the window for."""

    parent: QWidget
    controller: ImportedRelaxationsController
    import_enabled: list[bool] = field(default_factory=list)
    statuses: list[str] = field(default_factory=list)
    cleared: list[bool] = field(default_factory=list)


def _harness(parent: QWidget) -> _Harness:
    import_enabled: list[bool] = []
    statuses: list[str] = []
    cleared: list[bool] = []
    deps = ImportedRelaxationsDeps(
        dialog_parent=lambda: parent,
        set_import_enabled=import_enabled.append,
        clear_status=lambda: cleared.append(True),
        active_muon_relaxation_definition=lambda: None,
        refresh_muon_relaxation_rows=lambda: None,
        refresh_jobs_table=lambda: None,
        refresh_muon_sites_summary=lambda: None,
        show_kept_folders=lambda _kept: None,
    )
    controller = ImportedRelaxationsController(
        make_context(statuses=statuses), deps, MuonRelaxationWidget()
    )
    return _Harness(parent, controller, import_enabled, statuses, cleared)


def _host() -> Structure:
    return Structure(
        label="Host",
        origin="imported",
        lattice_vectors=[list(row) for row in HOST_LATTICE],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )


def test_importing_without_a_project_says_so(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    shown: list[str] = []
    monkeypatch.setattr(
        "mufinder.app.workflow_controllers.imported_relaxations.QMessageBox.information",
        lambda _parent, title, _text: shown.append(title),
    )
    harness = _harness(QWidget())

    harness.controller.import_relaxations()

    assert shown == ["No project open"]
    assert not harness.controller.scan_in_progress


def test_removing_without_a_project_says_so(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    shown: list[str] = []
    monkeypatch.setattr(
        "mufinder.app.workflow_controllers.imported_relaxations.QMessageBox.information",
        lambda _parent, title, _text: shown.append(title),
    )
    harness = _harness(QWidget())

    harness.controller.remove_imported_relaxations()

    assert shown == ["No project open"]


def test_a_scan_ending_after_its_window_is_gone_leaves_it_alone(qapp, tmp_path: Path) -> None:
    """The scan's signals go nowhere once the window that owns them is destroyed."""

    parent = QWidget()
    harness = _harness(parent)
    root = build_folder(tmp_path, sites=(1,))
    harness.controller._begin_scan(root, _host())
    assert harness.statuses == [f"Scanning {root.name} for CASTEP relaxations…"]
    assert harness.import_enabled == [False]

    shiboken6.delete(parent)
    assert pump_until(lambda: not _LIVE_SCANS, timeout=15.0)

    assert harness.import_enabled == [False]
    assert harness.cleared == []


# --- Committing an import: the copies run off the GUI thread ---------------------


@dataclass
class _Project:
    """A project with a host, a controller over it, and what the controller told the user."""

    document: ProjectDocument
    host: Structure
    controller: ImportedRelaxationsController
    messages: list[tuple[str, str, str]]
    parent: QWidget

    @property
    def provenance(self) -> list[str]:
        path = self.document.project_root / "provenance.jsonl"
        return path.read_text(encoding="utf-8").splitlines() if path.exists() else []

    def commit(self, folder: Path) -> None:
        scan = scan_relaxation_folder(folder, host_lattice=HOST_LATTICE)
        self.controller.commit_import(scan, list(scan.importable_runs), self.host)


@pytest.fixture()
def project(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Project]:
    service = ProjectService()
    document, host = muon_host_document(service, tmp_path)
    messages: list[tuple[str, str, str]] = []
    for kind in ("information", "warning"):
        monkeypatch.setattr(
            f"{_CONTROLLER}.QMessageBox.{kind}",
            lambda _parent, title, text, *_args, _kind=kind, **_kwargs: messages.append(
                (_kind, title, text)
            ),
        )
    parent = QWidget()
    definitions: list = []

    def active_definition():
        return definitions[-1] if definitions else None

    def remember_definition() -> None:
        definitions[:] = [
            definition
            for definition in document.calculations.values()
            if str(definition.id) == document.ui_state.get(ACTIVE_MUON_RELAXATION_DEFINITION_KEY)
        ]

    deps = ImportedRelaxationsDeps(
        dialog_parent=lambda: parent,
        set_import_enabled=lambda _enabled: None,
        clear_status=lambda: None,
        active_muon_relaxation_definition=active_definition,
        refresh_muon_relaxation_rows=remember_definition,
        refresh_jobs_table=lambda: None,
        refresh_muon_sites_summary=lambda: None,
        show_kept_folders=lambda _kept: None,
    )
    controller = ImportedRelaxationsController(
        make_context(document=document, project_service=service), deps, MuonRelaxationWidget()
    )
    yield _Project(document, host, controller, messages, parent)
    parent.deleteLater()


def _tree(root: Path) -> dict[str, bytes | None]:
    return {
        path.relative_to(root).as_posix(): (path.read_bytes() if path.is_file() else None)
        for path in sorted(root.rglob("*"))
    }


def _entity_counts(document: ProjectDocument) -> tuple[int, ...]:
    return tuple(
        len(entities)
        for entities in (
            document.structures,
            document.sites,
            document.jobs,
            document.runs,
            document.results,
        )
    )


def _busy_dialogs() -> list[QProgressDialog]:
    return [
        widget
        for widget in QApplication.topLevelWidgets()
        if isinstance(widget, QProgressDialog) and widget.isVisible()
    ]


def test_the_copies_run_off_the_gui_thread_behind_a_busy_dialog(
    project: _Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A slow copy leaves the window responsive, with a busy dialog naming the import.

    The copy waits for a timer that only fires while the GUI thread runs its
    event loop, so a copy on the GUI thread would never return.
    """

    folder = build_folder(tmp_path / "runs", sites=(1, 2))
    real_copy = shutil.copy2
    copy_threads: set[threading.Thread] = set()
    sites_during_copy: set[int] = set()
    seen: list[tuple[str, str, bool]] = []
    gui_ticks = threading.Event()

    def watch_for_the_dialog() -> None:
        dialogs = _busy_dialogs()
        if dialogs:
            dialog = dialogs[0]
            seen.append((dialog.windowTitle(), dialog.labelText(), dialog.isModal()))
            gui_ticks.set()
        else:
            QTimer.singleShot(10, watch_for_the_dialog)

    def slow_copy(source, target, *args, **kwargs):
        copy_threads.add(threading.current_thread())
        sites_during_copy.add(len(project.document.sites))
        if not gui_ticks.wait(10):
            raise AssertionError("the GUI thread stopped running its event loop")
        return real_copy(source, target, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", slow_copy)
    QTimer.singleShot(0, watch_for_the_dialog)

    project.commit(folder)

    assert copy_threads and threading.main_thread() not in copy_threads
    # No entity exists until every copy has finished.
    assert sites_during_copy == {0}
    assert seen == [("Import CASTEP relaxations", "Copying 2 relaxations into the project…", True)]
    assert _busy_dialogs() == []
    assert len(project.document.sites) == 2
    assert [message[1] for message in project.messages] == ["Import complete"]


def test_a_copy_that_fails_part_way_imports_nothing(
    project: _Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The files copied before the failure go; another import's folder of the same name stays."""

    folder = build_folder(tmp_path / "runs", sites=(1, 2))
    project.commit(folder)
    files_before = _tree(project.document.project_root / "files")
    entities_before = _entity_counts(project.document)
    provenance_before = project.provenance
    rows_before = project.controller._view.collect_state()
    project.messages.clear()

    real_copy = shutil.copy2
    calls: list[Path] = []

    def failing_copy(source, target, *args, **kwargs):
        calls.append(Path(target))
        if len(calls) == 6:
            raise OSError(28, "No space left on device")
        return real_copy(source, target, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", failing_copy)

    project.commit(folder)

    assert len(calls) == 6
    assert project.messages == [
        (
            "warning",
            "Import failed",
            "Could not write the imported files into the project: no space left on device. "
            "Nothing was imported.",
        )
    ]
    assert _tree(project.document.project_root / "files") == files_before
    assert _entity_counts(project.document) == entities_before
    assert project.provenance == provenance_before
    assert project.controller._view.collect_state() == rows_before


_CAN_DENY_WRITES = sys.platform != "win32" and hasattr(os, "geteuid") and os.geteuid() != 0
_needs_permissions = pytest.mark.skipif(
    not _CAN_DENY_WRITES, reason="needs a user that read-only folders stop"
)


def _innermost(paths: set[str]) -> list[str]:
    return sorted(
        path for path in paths if not any(other.startswith(f"{path}/") for other in paths)
    )


@_needs_permissions
@pytest.mark.parametrize(
    ("locked", "after_file"),
    [
        ("job", "first"),  # files in the job folder cannot be deleted
        ("charge", "middle"),  # the next job folder cannot be made, nor this one removed
        ("job", "last"),  # the manifest cannot be written
    ],
)
def test_a_destination_made_read_only_part_way_names_what_stays(
    project: _Project,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    locked: str,
    after_file: str,
) -> None:
    """Everything this import made that may be removed is removed; the rest is named."""

    folder = build_folder(tmp_path / "runs", sites=(1, 2))
    scan = scan_relaxation_folder(folder, host_lattice=HOST_LATTICE)
    per_run = sum(
        path is not None
        for path in (
            scan.runs[0].castep_path,
            scan.runs[0].out_cell_path,
            scan.runs[0].input_cell_path,
            scan.runs[0].run_param_path,
        )
    )
    lock_after = {"first": 1, "middle": per_run, "last": 2 * per_run}[after_file]
    root = project.document.project_root
    tree_before = _tree(root)
    entities_before = _entity_counts(project.document)

    real_copy = shutil.copy2
    calls: list[Path] = []
    locked_folders: list[Path] = []

    def copy_then_lock(source, target, *args, **kwargs):
        result = real_copy(source, target, *args, **kwargs)
        calls.append(Path(target))
        if len(calls) == lock_after:
            job = Path(target).parent
            if job.name == "original_inputs":
                job = job.parent
            folder_to_lock = job if locked == "job" else job.parent
            folder_to_lock.chmod(0o555)
            locked_folders.append(folder_to_lock)
        return result

    monkeypatch.setattr(shutil, "copy2", copy_then_lock)
    try:
        project.commit(folder)
    finally:
        for folder_to_lock in locked_folders:
            folder_to_lock.chmod(0o755)

    added = {path for path in _tree(root) if path not in tree_before}
    remaining = _innermost(added)
    assert remaining, "the lock should stop some of the clean-up"
    assert all(_tree(root)[path] == tree_before.get(path) for path in tree_before)
    listed = "\n".join(f"- {path}" for path in remaining[:5])
    if len(remaining) > 5:
        listed += f"\n- and {len(remaining) - 5} more ({len(remaining)} in all)"
    assert project.messages == [
        (
            "warning",
            "Import failed",
            "Could not write the imported files into the project: permission denied. "
            "Nothing was imported.\n\nMuFinder could not remove everything this import had "
            f"created, so {'this is' if len(remaining) == 1 else 'these are'} still in the "
            f"project:\n{listed}",
        )
    ]
    assert _entity_counts(project.document) == entities_before
    assert project.provenance == []


def test_an_import_after_a_cleaned_up_failure_matches_a_clean_import(
    project: _Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing is left to push the next import's job folders to other names."""

    folder = build_folder(tmp_path / "runs", sites=(1, 2))
    clean = tmp_path / "clean"
    scan = scan_relaxation_folder(folder, host_lattice=HOST_LATTICE)
    stage_relaxation_import(scan=scan, runs=scan.importable_runs, output_root=clean)
    expected = sorted(_tree(clean))

    real_copy = shutil.copy2
    calls: list[Path] = []

    def failing_copy(source, target, *args, **kwargs):
        calls.append(Path(target))
        if len(calls) == 6:
            raise OSError(28, "No space left on device")
        return real_copy(source, target, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", failing_copy)
    project.commit(folder)
    monkeypatch.setattr(shutil, "copy2", real_copy)
    assert not (project.document.project_root / "files" / "muon_relaxation").exists()

    project.commit(folder)

    output_root = project.document.project_root / "files" / "muon_relaxation"
    assert sorted(_tree(output_root)) == expected
    assert [message[1] for message in project.messages] == ["Import failed", "Import complete"]


@_needs_permissions
def test_a_run_file_that_cannot_be_read_is_named(project: _Project, tmp_path: Path) -> None:
    folder = build_folder(tmp_path / "runs", sites=(1, 2))
    scan = scan_relaxation_folder(folder, host_lattice=HOST_LATTICE)
    unreadable = folder / "results" / "Host_2.castep"
    files_before = _tree(project.document.project_root)
    unreadable.chmod(0o000)
    try:
        project.controller.commit_import(scan, list(scan.importable_runs), project.host)
    finally:
        unreadable.chmod(0o644)

    assert project.messages == [
        (
            "warning",
            "Import failed",
            "Could not read 'results/Host_2.castep' from the chosen folder: permission "
            "denied. Nothing was imported.",
        )
    ]
    assert _tree(project.document.project_root) == files_before
    assert len(project.document.sites) == 0


def test_cancel_stops_between_files_and_imports_nothing(
    project: _Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    folder = build_folder(tmp_path / "runs", sites=(1, 2))
    project.commit(folder)
    files_before = _tree(project.document.project_root / "files")
    entities_before = _entity_counts(project.document)
    provenance_before = project.provenance
    project.messages.clear()

    real_copy = shutil.copy2
    real_run = blocking_call.run_blocking_call
    calls: list[Path] = []
    cancel_events: list[threading.Event] = []

    def run_recording_the_cancel_event(parent, op, **kwargs):
        cancel_events.append(kwargs["cancel_event"])
        return real_run(parent, op, **kwargs)

    def cancel_the_busy_dialog() -> None:
        dialogs = _busy_dialogs()
        if dialogs:
            dialogs[0].cancel()
        else:
            QTimer.singleShot(10, cancel_the_busy_dialog)

    def copy_until_cancelled(source, target, *args, **kwargs):
        calls.append(Path(target))
        result = real_copy(source, target, *args, **kwargs)
        if len(calls) == 2:
            # Hold the second file until the user has pressed Cancel.
            assert cancel_events[0].wait(10)
        return result

    monkeypatch.setattr(
        f"{_CONTROLLER}.blocking_call.run_blocking_call", run_recording_the_cancel_event
    )
    monkeypatch.setattr(shutil, "copy2", copy_until_cancelled)
    QTimer.singleShot(0, cancel_the_busy_dialog)

    project.commit(folder)

    # The copy stopped before its third file.
    assert len(calls) == 2
    assert project.messages == [("information", "Import cancelled", "Nothing was imported.")]
    assert _tree(project.document.project_root / "files") == files_before
    assert _entity_counts(project.document) == entities_before
    assert project.provenance == provenance_before


def test_a_copy_that_outlasts_the_wait_after_cancel_removes_its_own_files(
    project: _Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """When a file takes longer to copy than the wait after Cancel, the worker cleans up itself."""

    from mufinder.app.blocking_call import BlockingCallCancelled

    release = threading.Event()
    workers: list[threading.Thread] = []
    real_copy = shutil.copy2

    def slow_copy(source, target, *args, **kwargs):
        assert release.wait(10)
        return real_copy(source, target, *args, **kwargs)

    def cancel_at_once(_parent, op, *, cancel_event, **_kwargs):
        worker = threading.Thread(target=lambda: _swallow(op))
        workers.append(worker)
        worker.start()
        cancel_event.set()
        raise BlockingCallCancelled("cancelled")

    monkeypatch.setattr(shutil, "copy2", slow_copy)
    monkeypatch.setattr(f"{_CONTROLLER}.blocking_call.run_blocking_call", cancel_at_once)
    monkeypatch.setattr(f"{_CONTROLLER}.blocking_call.wait_for_event", lambda *_a, **_k: False)
    files_root = project.document.project_root / "files"
    files_before = _tree(files_root)

    project.commit(build_folder(tmp_path / "runs", sites=(1,)))

    assert project.messages == [
        (
            "information",
            "Import cancelled",
            "Nothing was imported. The files copied so far are removed once the file being "
            "copied is done.",
        )
    ]
    release.set()
    workers[0].join(10)
    assert _tree(files_root) == files_before
    assert len(project.document.sites) == 0


def test_a_copy_cancelled_during_its_last_file_removes_its_own_files(
    project: _Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No checkpoint follows the last file, so the worker checks for Cancel once it is done."""

    from mufinder.app.blocking_call import BlockingCallCancelled
    from mufinder.services import relaxation_import_commit

    release = threading.Event()
    workers: list[threading.Thread] = []
    real_stage = relaxation_import_commit.stage_relaxation_import

    def stage_then_hold(**kwargs):
        staged = real_stage(**kwargs)
        assert release.wait(10)
        return staged

    def cancel_at_once(_parent, op, *, cancel_event, **_kwargs):
        worker = threading.Thread(target=lambda: _swallow(op))
        workers.append(worker)
        worker.start()
        cancel_event.set()
        raise BlockingCallCancelled("cancelled")

    monkeypatch.setattr(f"{_CONTROLLER}.stage_relaxation_import", stage_then_hold)
    monkeypatch.setattr(f"{_CONTROLLER}.blocking_call.run_blocking_call", cancel_at_once)
    monkeypatch.setattr(f"{_CONTROLLER}.blocking_call.wait_for_event", lambda *_a, **_k: False)
    files_root = project.document.project_root / "files"
    files_before = _tree(files_root)

    project.commit(build_folder(tmp_path / "runs", sites=(1,)))
    release.set()
    workers[0].join(10)

    assert _tree(files_root) == files_before
    assert len(project.document.sites) == 0


def _swallow(op) -> None:
    try:
        op()
    except Exception:  # noqa: BLE001 - run_blocking_call drops a cancelled call's error
        pass


def test_a_copy_cancelled_as_it_finishes_keeps_none_of_its_files(
    project: _Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The worker can finish just after the user cancels; its result is dropped, so its files go."""

    from mufinder.app.blocking_call import BlockingCallCancelled

    def finish_then_cancel(_parent, op, **_kwargs):
        op()
        raise BlockingCallCancelled("cancelled")

    monkeypatch.setattr(f"{_CONTROLLER}.blocking_call.run_blocking_call", finish_then_cancel)
    files_root = project.document.project_root / "files"
    files_before = _tree(files_root)

    project.commit(build_folder(tmp_path / "runs", sites=(1, 2)))

    assert project.messages == [("information", "Import cancelled", "Nothing was imported.")]
    assert _tree(files_root) == files_before
    assert len(project.document.sites) == 0
    assert project.provenance == []


# --- A scan that outlives its project ------------------------------------------------


@dataclass
class _Switchable:
    """A controller whose open project the test changes while a real scan thread runs."""

    service: ProjectService
    open_document: list[ProjectDocument | None]
    first: ProjectDocument
    second: ProjectDocument
    controller: ImportedRelaxationsController
    messages: list[tuple[str, str, str]]
    reviewed: list[int]
    import_enabled: list[bool]
    cleared: list[bool]
    release: threading.Event

    def switch_to(self, document: ProjectDocument | None) -> None:
        self.open_document[0] = document

    def finish(self) -> None:
        self.release.set()
        assert pump_until(lambda: not self.controller.scan_in_progress, timeout=15.0)
        assert pump_until(lambda: not _LIVE_SCANS, timeout=15.0)


def _snapshot(document: ProjectDocument) -> tuple[object, ...]:
    files = document.project_root / "files"
    provenance = document.project_root / "provenance.jsonl"
    return (
        _tree(files) if files.exists() else {},
        _entity_counts(document),
        provenance.read_text(encoding="utf-8") if provenance.exists() else "",
        dict(document.ui_state),
    )


@pytest.fixture()
def switchable(qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[_Switchable]:
    """A scan started in one project, held on its worker thread until the test releases it.

    The review accepts every importable run, so a scan that is not discarded
    imports into whatever project is open when it completes.
    """

    from mufinder.app import relaxation_import_dialog

    service = ProjectService()
    first, _host_one = muon_host_document(service, tmp_path / "first")
    second, _host_two = muon_host_document(service, tmp_path / "second")
    open_document: list[ProjectDocument | None] = [first]
    messages: list[tuple[str, str, str]] = []
    for kind in ("information", "warning"):
        monkeypatch.setattr(
            f"{_CONTROLLER}.QMessageBox.{kind}",
            lambda _parent, title, text, *_args, _kind=kind, **_kwargs: messages.append(
                (_kind, title, text)
            ),
        )
    reviewed: list[int] = []

    class _AcceptEverything:
        def __init__(self, scan, parent=None) -> None:
            self._scan = scan
            reviewed.append(len(scan.runs))

        def exec(self) -> QDialog.DialogCode:
            return QDialog.DialogCode.Accepted

        def selected_runs(self) -> list:
            return list(self._scan.importable_runs)

    monkeypatch.setattr(f"{_CONTROLLER}.RelaxationImportDialog", _AcceptEverything)
    folder = build_folder(tmp_path / "runs", sites=(1, 2))
    monkeypatch.setattr(
        f"{_CONTROLLER}.QFileDialog.getExistingDirectory", lambda *_args, **_kwargs: str(folder)
    )
    release = threading.Event()
    real_scan = relaxation_import_dialog.scan_relaxation_folder
    scan_threads: list[threading.Thread] = []

    def held_scan(*args, **kwargs):
        scan_threads.append(threading.current_thread())
        assert release.wait(15), "the test never released the scan"
        return real_scan(*args, **kwargs)

    monkeypatch.setattr(relaxation_import_dialog, "scan_relaxation_folder", held_scan)

    parent = QWidget()
    import_enabled: list[bool] = []
    cleared: list[bool] = []
    deps = ImportedRelaxationsDeps(
        dialog_parent=lambda: parent,
        set_import_enabled=import_enabled.append,
        clear_status=lambda: cleared.append(True),
        active_muon_relaxation_definition=lambda: None,
        refresh_muon_relaxation_rows=lambda: None,
        refresh_jobs_table=lambda: None,
        refresh_muon_sites_summary=lambda: None,
        show_kept_folders=lambda _kept: None,
    )
    context = replace(make_context(project_service=service), document=lambda: open_document[0])
    controller = ImportedRelaxationsController(context, deps, MuonRelaxationWidget())
    yield _Switchable(
        service,
        open_document,
        first,
        second,
        controller,
        messages,
        reviewed,
        import_enabled,
        cleared,
        release,
    )
    release.set()
    pump_until(lambda: not _LIVE_SCANS, timeout=15.0)
    assert threading.main_thread() not in scan_threads
    parent.deleteLater()


_PROJECT_CHANGED = (
    "information",
    "Scan discarded",
    "The scan of 'runs' was discarded because a different project was opened while it ran. "
    "Nothing was imported. Start the import again from the project you want to import into.",
)
_PROJECT_CLOSED = (
    "information",
    "Scan discarded",
    "The scan of 'runs' was discarded because the project was closed while it ran. "
    "Nothing was imported.",
)
_HOST_CHANGED = (
    "information",
    "Scan discarded",
    "The scan of 'runs' was discarded because the project's host structure changed while it "
    "ran, and the runs were checked against the previous host. Nothing was imported. Start "
    "the import again to check them against the current host.",
)


def test_a_scan_imports_into_the_project_it_started_in(switchable: _Switchable) -> None:
    """The control case: with nothing changed, the scan reviews and imports as before."""

    switchable.controller.import_relaxations()
    assert switchable.controller.scan_in_progress

    switchable.finish()

    assert switchable.reviewed == [2]
    assert len(switchable.first.sites) == 2
    assert [message[1] for message in switchable.messages] == ["Import complete"]
    assert switchable.import_enabled == [False, True]


@pytest.mark.parametrize("replacement", ["the same project reopened", "a new project", "none"])
def test_a_scan_that_outlives_its_project_is_discarded(
    switchable: _Switchable, replacement: str
) -> None:
    """Opening, creating or closing a project during the scan drops its result.

    Reopening the same folder counts as a different project: the scan belongs
    to the document it started with, not to a name or a path.
    """

    other: ProjectDocument | None
    if replacement == "the same project reopened":
        other = switchable.service.load_project(switchable.first.project_root)
    elif replacement == "a new project":
        other = switchable.second
    else:
        other = None
    before = [_snapshot(document) for document in (switchable.first, switchable.second)]
    switchable.controller.import_relaxations()

    switchable.switch_to(other)
    switchable.finish()

    assert switchable.reviewed == []
    assert switchable.messages == [_PROJECT_CLOSED if other is None else _PROJECT_CHANGED]
    assert [_snapshot(document) for document in (switchable.first, switchable.second)] == before
    if other is not None:
        assert _entity_counts(other)[1:] == (0, 0, 0, 0)
    # The action follows the project: back on with one open, off with none.
    assert switchable.import_enabled == [False, other is not None]
    assert switchable.cleared == [True]
    assert not switchable.controller.scan_in_progress


def test_a_scan_whose_host_changed_is_discarded(switchable: _Switchable) -> None:
    """The scan matched the runs against the old host's lattice and species."""

    switchable.controller.import_relaxations()
    new_host = Structure(
        label="Bigger host",
        origin="imported",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Ge"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    switchable.service.add_structure(switchable.first, new_host)
    switchable.first.project.host_structure_id = new_host.id
    before = _snapshot(switchable.first)

    switchable.finish()

    assert switchable.reviewed == []
    assert switchable.messages == [_HOST_CHANGED]
    assert _snapshot(switchable.first) == before
    assert switchable.import_enabled == [False, True]


def test_a_host_edited_in_place_during_the_scan_is_a_changed_host(
    switchable: _Switchable,
) -> None:
    switchable.controller.import_relaxations()
    host = switchable.service.active_host_structure(switchable.first)
    host.lattice_vectors = [[4.5, 0.0, 0.0], [0.0, 4.5, 0.0], [0.0, 0.0, 4.5]]

    switchable.finish()

    assert switchable.reviewed == []
    assert switchable.messages == [_HOST_CHANGED]
    assert len(switchable.first.sites) == 0


def test_a_scan_that_fails_after_its_project_changed_only_says_it_was_discarded(
    switchable: _Switchable, monkeypatch: pytest.MonkeyPatch
) -> None:
    from mufinder.app import relaxation_import_dialog

    def _held_failure(*_args, **_kwargs):
        assert switchable.release.wait(15)
        raise RuntimeError("disk on fire")

    monkeypatch.setattr(relaxation_import_dialog, "scan_relaxation_folder", _held_failure)
    switchable.controller.import_relaxations()
    switchable.switch_to(switchable.second)

    switchable.finish()

    assert switchable.messages == [_PROJECT_CHANGED]


def test_a_scan_ending_while_the_window_closes_shows_and_writes_nothing(
    switchable: _Switchable,
) -> None:
    before = _snapshot(switchable.first)
    switchable.controller.import_relaxations()

    switchable.controller.hold_for_close()
    switchable.release.set()
    assert pump_until(lambda: not _LIVE_SCANS, timeout=15.0)

    assert switchable.reviewed == []
    assert switchable.messages == []
    assert _snapshot(switchable.first) == before
    assert switchable.import_enabled == [False]


@pytest.mark.parametrize("outcome", ["completed", "failed"])
def test_a_scan_that_ended_during_a_cancelled_close_ends_as_usual(
    switchable: _Switchable, monkeypatch: pytest.MonkeyPatch, outcome: str
) -> None:
    from mufinder.app import relaxation_import_dialog

    if outcome == "failed":

        def _held_failure(*_args, **_kwargs):
            assert switchable.release.wait(15)
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(relaxation_import_dialog, "scan_relaxation_folder", _held_failure)
    switchable.controller.import_relaxations()
    switchable.controller.hold_for_close()
    switchable.release.set()
    assert pump_until(lambda: not _LIVE_SCANS, timeout=15.0)
    assert switchable.messages == []

    switchable.controller.resume_after_cancelled_close()

    assert not switchable.controller.scan_in_progress
    assert switchable.import_enabled == [False, True]
    if outcome == "completed":
        assert switchable.reviewed == [2]
        assert len(switchable.first.sites) == 2
        assert [message[1] for message in switchable.messages] == ["Import complete"]
    else:
        assert switchable.messages == [
            ("warning", "Import failed", "Could not scan the folder:\ndisk on fire")
        ]


def test_closing_stops_the_scan_and_ignores_a_late_ending(switchable: _Switchable) -> None:
    """Nothing the scan reports after the stop is held, shown or imported."""

    switchable.controller.import_relaxations()
    ((thread, _worker),) = _LIVE_SCANS
    # A direct connection runs on the scan thread as it finishes, whatever loop runs here.
    finished: list[bool] = []
    thread.finished.connect(lambda: finished.append(True), Qt.ConnectionType.DirectConnection)
    switchable.controller.hold_for_close()
    QTimer.singleShot(50, switchable.release.set)

    switchable.controller.stop_for_close()
    switchable.controller.scan_completed(object())
    switchable.controller.scan_failed("late")

    # The thread deletes itself once finished, so it is not called after the stop.
    assert finished == [True]
    assert not shiboken6.isValid(thread) or thread.isFinished()
    assert not switchable.controller.scan_in_progress
    assert pump_until(lambda: not _LIVE_SCANS, timeout=5.0)
    switchable.controller.resume_after_cancelled_close()
    assert switchable.reviewed == []
    assert switchable.messages == []
    assert len(switchable.first.sites) == 0


def test_a_cancelled_close_with_no_scan_changes_nothing(switchable: _Switchable) -> None:
    switchable.controller.hold_for_close()
    switchable.controller.resume_after_cancelled_close()

    switchable.controller.import_relaxations()
    switchable.finish()

    assert switchable.reviewed == [2]


def test_the_project_cannot_change_while_the_files_are_copied(
    project: _Project, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The copy runs behind an application-modal busy dialog, so no menu can open a project.

    Checked while the copy is running: the busy dialog is the active modal
    widget and blocks every other window, the main window included.
    """

    folder = build_folder(tmp_path / "runs", sites=(1,))
    real_copy = shutil.copy2
    seen: list[tuple[bool, bool]] = []
    looked = threading.Event()

    def look() -> None:
        dialogs = _busy_dialogs()
        if not dialogs:
            QTimer.singleShot(10, look)
            return
        seen.append(
            (
                QApplication.activeModalWidget() is dialogs[0],
                dialogs[0].windowModality() == Qt.WindowModality.ApplicationModal,
            )
        )
        looked.set()

    def copy_once_looked(source, target, *args, **kwargs):
        assert looked.wait(10)
        return real_copy(source, target, *args, **kwargs)

    monkeypatch.setattr(shutil, "copy2", copy_once_looked)
    QTimer.singleShot(0, look)

    project.commit(folder)

    assert seen == [(True, True)]
    assert len(project.document.sites) == 1
