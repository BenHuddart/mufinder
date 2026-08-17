"""The legacy-import review dialog and the action that drives it.

The dialog is a review step, not a confirmation: an import writes read-only
entities and resolves genuine ambiguities on the user's behalf, so what it
resolved has to be visible *before* anything is committed. Most of these tests
are therefore about what the user is shown, not just what is selected.

Fixtures are hand-authored: see ``docs/legacy_import.rst``.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from mufinder.app.legacy_import_dialog import (
    LegacyImportDialog,
    run_notes,
    run_summary_lines,
)
from mufinder.app.main_window import MainWindow
from mufinder.domain.models import Project, Structure
from mufinder.services.legacy_import import scan_legacy_folder
from mufinder.services.project_service import ProjectDocument

from tests.services.test_legacy_import_scan import HOST_LATTICE, _legacy_param, build_folder


@pytest.fixture()
def window(qapp) -> MainWindow:
    return MainWindow()


@pytest.fixture()
def host_structure() -> Structure:
    return Structure(
        label="Host",
        origin="imported",
        lattice_vectors=[list(row) for row in HOST_LATTICE],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )


def _document(tmp_path: Path, host: Structure) -> ProjectDocument:
    project = Project(name="T")
    project.host_structure_id = host.id
    document = ProjectDocument(project_root=tmp_path / "project", project=project)
    document.structures[host.id] = host
    return document


def _scan(folder: Path):
    return scan_legacy_folder(folder, host_lattice=HOST_LATTICE)


class TestSelection:
    def test_converged_runs_are_preselected(self, qapp, tmp_path: Path) -> None:
        """The wizard offers everything and pre-ticks what is safe."""

        dialog = LegacyImportDialog(_scan(build_folder(tmp_path, sites=(1, 2))))

        assert len(dialog.selected_runs()) == 2

    def test_an_unconverged_run_is_offered_but_not_preselected(
        self, qapp, tmp_path: Path
    ) -> None:
        scan = _scan(build_folder(tmp_path, sites=(1,), castep_kwargs={"converged": False}))
        dialog = LegacyImportDialog(scan)

        assert dialog._table.rowCount() == 1
        assert dialog.selected_runs() == []

    def test_a_run_that_cannot_be_imported_is_shown_but_not_selectable(
        self, qapp, tmp_path: Path
    ) -> None:
        """Shown rather than hidden, so the user sees why it is absent."""

        folder = build_folder(tmp_path, sites=(1, 2))
        next(folder.glob("results/*_2-out.cell")).unlink()
        dialog = LegacyImportDialog(_scan(folder))

        assert dialog._table.rowCount() == 2
        assert [run.site_index for run in dialog.selected_runs()] == [1]

    def test_select_none_clears_the_selection(self, qapp, tmp_path: Path) -> None:
        dialog = LegacyImportDialog(_scan(build_folder(tmp_path, sites=(1, 2))))

        dialog._on_select_none()

        assert dialog.selected_runs() == []

    def test_select_converged_restores_the_default(self, qapp, tmp_path: Path) -> None:
        dialog = LegacyImportDialog(_scan(build_folder(tmp_path, sites=(1, 2))))
        dialog._on_select_none()

        dialog._on_select_converged()

        assert len(dialog.selected_runs()) == 2

    def test_ok_is_disabled_with_nothing_selected(self, qapp, tmp_path: Path) -> None:
        from PySide6.QtWidgets import QDialogButtonBox

        dialog = LegacyImportDialog(_scan(build_folder(tmp_path, sites=(1,))))
        ok = dialog._buttons.button(QDialogButtonBox.StandardButton.Ok)

        assert ok.isEnabled()
        dialog._on_select_none()
        assert not ok.isEnabled()


class TestWhatTheUserIsTold:
    def test_a_stale_input_tree_is_surfaced_before_committing(self, tmp_path: Path) -> None:
        """The headline hazard. The scan resolves it in favour of the output,
        which is right, but it is the user's science and they should see it."""

        folder = build_folder(tmp_path, sites=(1,), param_text=_legacy_param(xc="LDA"))
        scan = _scan(folder)

        summary = " ".join(run_summary_lines(scan))
        notes = run_notes(scan.runs[0])

        assert "disagreement" in summary
        assert "XC_FUNCTIONAL" in notes
        assert "PBE" in notes and "LDA" in notes

    def test_an_undetermined_charge_state_is_surfaced(self, tmp_path: Path) -> None:
        folder = build_folder(
            tmp_path,
            sites=(1,),
            castep_kwargs={"charge": "  7.000"},
            param_text=_legacy_param(charge_line=""),
        )

        summary = " ".join(run_summary_lines(_scan(folder)))

        assert "Charge state could not be determined" in summary
        assert "positive muons" in summary

    def test_a_missing_cif_is_surfaced(self, tmp_path: Path) -> None:
        scan = _scan(build_folder(tmp_path, sites=(1,), include_cif=False))

        assert any("No CIF" in line for line in run_summary_lines(scan))

    def test_a_blocked_run_explains_itself(self, tmp_path: Path) -> None:
        folder = build_folder(tmp_path, sites=(1,))
        next(folder.glob("results/*-out.cell")).unlink()

        notes = run_notes(_scan(folder).runs[0])

        assert "-out.cell" in notes

    def test_a_clean_folder_produces_no_alarming_summary(self, tmp_path: Path) -> None:
        scan = _scan(build_folder(tmp_path, sites=(1,)))

        assert not any("disagreement" in line for line in run_summary_lines(scan))


class TestTheAction:
    def test_it_refuses_without_a_project(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        window._document = None
        shown: list[str] = []
        monkeypatch.setattr(
            "mufinder.app.main_window.QMessageBox.information",
            lambda *args, **kwargs: shown.append(args[1]),
        )

        window._on_import_legacy_relaxations()

        assert shown == ["No project open"]

    def test_it_refuses_without_a_host_structure(
        self, window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Every imported structure hangs off the active host, or its downstream
        jobs fail host resolution at submission time."""

        project = Project(name="T")
        window._document = ProjectDocument(project_root=tmp_path, project=project)
        shown: list[str] = []
        monkeypatch.setattr(
            "mufinder.app.main_window.QMessageBox.information",
            lambda *args, **kwargs: shown.append(args[1]),
        )
        picked: list[str] = []
        monkeypatch.setattr(
            "mufinder.app.main_window.QFileDialog.getExistingDirectory",
            lambda *args, **kwargs: picked.append("called") or "",
        )

        window._on_import_legacy_relaxations()

        assert shown == ["No host structure"]
        # The folder picker never opened: refusing early beats refusing after
        # the user has chosen a folder.
        assert picked == []

    def test_committing_a_selection_imports_and_attaches_it(
        self,
        window: MainWindow,
        tmp_path: Path,
        host_structure: Structure,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """The whole flow below the dialog: commit, merge, refresh, persist."""

        window._document = _document(tmp_path, host_structure)
        monkeypatch.setattr(
            "mufinder.app.main_window.QMessageBox.information", lambda *args, **kwargs: None
        )
        monkeypatch.setattr(window, "_persist_document_ui_state", lambda **kwargs: None)
        monkeypatch.setattr(window, "_refresh_muon_sites_summary", lambda **kwargs: None)

        scan = _scan(build_folder(tmp_path / "legacy", sites=(1, 2)))
        window._commit_legacy_import(scan, list(scan.importable_runs), host_structure)

        definition = window._active_muon_relaxation_definition()
        assert definition is not None
        assert len(definition.parameters["job_ids_by_label"]) == 2
        assert len(window._document.sites) == 2
        assert all(job.status.value == "completed" for job in window._document.jobs.values())

    def test_the_completion_message_reports_skips_and_notes(
        self,
        window: MainWindow,
        tmp_path: Path,
        host_structure: Structure,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window._document = _document(tmp_path, host_structure)
        messages: list[str] = []
        monkeypatch.setattr(
            "mufinder.app.main_window.QMessageBox.information",
            lambda *args, **kwargs: messages.append(args[2]),
        )
        monkeypatch.setattr(window, "_persist_document_ui_state", lambda **kwargs: None)
        monkeypatch.setattr(window, "_refresh_muon_sites_summary", lambda **kwargs: None)

        folder = build_folder(tmp_path / "legacy", sites=(1, 2), muon_label="H")
        next(folder.glob("results/*_2-out.cell")).unlink()
        scan = _scan(folder)
        window._commit_legacy_import(scan, list(scan.runs), host_structure)

        assert messages
        body = messages[0]
        assert "Imported 1 relaxation(s)." in body
        assert "1 were skipped." in body
        # The bare-H relabelling is reported rather than applied silently.
        assert "relabelled" in body


class TestScanWorker:
    def test_the_worker_reports_a_scan(self, qapp, tmp_path: Path) -> None:
        """Scanning runs off the GUI thread, so it is exercised directly here
        rather than through a spun event loop."""

        from mufinder.app.legacy_import_dialog import LegacyScanWorker

        received: list[object] = []
        worker = LegacyScanWorker(build_folder(tmp_path, sites=(1,)), HOST_LATTICE)
        worker.completed.connect(received.append)

        worker.run()

        assert len(received) == 1
        assert [run.site_index for run in received[0].runs] == [1]

    def test_the_worker_reports_failure_instead_of_raising(
        self, qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """An exception crossing a thread boundary would be lost."""

        from mufinder.app import legacy_import_dialog
        from mufinder.app.legacy_import_dialog import LegacyScanWorker

        def _boom(*args, **kwargs):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(legacy_import_dialog, "scan_legacy_folder", _boom)
        failures: list[str] = []
        worker = LegacyScanWorker(tmp_path, None)
        worker.failed.connect(failures.append)

        worker.run()

        assert failures == ["disk on fire"]


class TestRemovalAction:
    def _imported_window(self, window: MainWindow, tmp_path: Path, host: Structure):
        from mufinder.services.legacy_import_commit import commit_legacy_import

        window._document = _document(tmp_path, host)
        scan = _scan(build_folder(tmp_path / "legacy", sites=(1, 2)))
        output_root = window._document.project_root / "files" / "muon_relaxation"
        outcome = commit_legacy_import(
            window._document,
            scan=scan,
            runs=scan.importable_runs,
            host_structure=host,
            output_root=output_root,
        )
        window.merge_imported_relaxation_rows(outcome, output_directory=output_root)
        return outcome

    def test_it_reports_when_nothing_is_imported(
        self, window: MainWindow, tmp_path: Path, host_structure: Structure,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        window._document = _document(tmp_path, host_structure)
        shown: list[str] = []
        monkeypatch.setattr(
            "mufinder.app.main_window.QMessageBox.information",
            lambda *args, **kwargs: shown.append(args[1]),
        )

        window._on_remove_imported_relaxations()

        assert shown == ["Nothing imported"]

    def test_batches_are_listed_with_their_source_and_size(
        self, window: MainWindow, tmp_path: Path, host_structure: Structure
    ) -> None:
        self._imported_window(window, tmp_path, host_structure)

        batches = window._imported_relaxation_batches()

        assert len(batches) == 1
        label, count, _batch_id = batches[0]
        assert count == 2
        assert label == "legacy"

    def test_removal_refuses_while_something_still_uses_the_sites(
        self, window: MainWindow, tmp_path: Path, host_structure: Structure,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from uuid import uuid4

        from mufinder.domain.models import CalculationDefinition
        from mufinder.domain.enums import WorkflowStage as Stage

        outcome = self._imported_window(window, tmp_path, host_structure)
        window._document.calculations[uuid4()] = CalculationDefinition(
            workflow_stage=Stage.QUANTUM_CORRECTIONS,
            calc_type="castep_quantum_phonon",
            input_site_ids=[outcome.imported[0].site_id],
        )
        warned: list[str] = []
        monkeypatch.setattr(
            "mufinder.app.main_window.QMessageBox.warning",
            lambda *args, **kwargs: warned.append(args[2]),
        )
        monkeypatch.setattr(
            "mufinder.app.main_window.QInputDialog.getItem",
            lambda *args, **kwargs: (args[3][0], True),
        )
        asked: list[int] = []
        monkeypatch.setattr(
            "mufinder.app.main_window.QMessageBox.question",
            lambda *args, **kwargs: asked.append(1),
        )

        window._on_remove_imported_relaxations()

        assert warned and "quantum_corrections" in warned[0]
        # Refused before the confirmation, so the user is never asked to confirm
        # something that cannot happen.
        assert asked == []
        assert outcome.imported[0].site_id in window._document.sites

    def test_removal_drops_the_rows_and_the_entities(
        self, window: MainWindow, tmp_path: Path, host_structure: Structure,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from PySide6.QtWidgets import QMessageBox

        outcome = self._imported_window(window, tmp_path, host_structure)
        monkeypatch.setattr(
            "mufinder.app.main_window.QInputDialog.getItem",
            lambda *args, **kwargs: (args[3][0], True),
        )
        monkeypatch.setattr(
            "mufinder.app.main_window.QMessageBox.question",
            lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
        )
        monkeypatch.setattr(window, "_persist_document_ui_state", lambda **kwargs: None)
        monkeypatch.setattr(window, "_refresh_muon_sites_summary", lambda **kwargs: None)

        window._on_remove_imported_relaxations()

        assert window._document.sites == {}
        assert window._document.jobs == {}
        definition = window._active_muon_relaxation_definition()
        assert definition.parameters["configuration"]["rows"] == []
        assert definition.parameters["job_ids_by_label"] == {}
        # The widget no longer shows them either.
        assert window._muon_relaxation_view.collect_state()["rows"] == []

    def test_declining_the_confirmation_changes_nothing(
        self, window: MainWindow, tmp_path: Path, host_structure: Structure,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        from PySide6.QtWidgets import QMessageBox

        outcome = self._imported_window(window, tmp_path, host_structure)
        monkeypatch.setattr(
            "mufinder.app.main_window.QInputDialog.getItem",
            lambda *args, **kwargs: (args[3][0], True),
        )
        monkeypatch.setattr(
            "mufinder.app.main_window.QMessageBox.question",
            lambda *args, **kwargs: QMessageBox.StandardButton.No,
        )

        window._on_remove_imported_relaxations()

        assert len(window._document.sites) == 2
        assert all(item.job_directory.is_dir() for item in outcome.imported)


class TestScanProgressReporting:
    def test_the_worker_relays_progress(self, qapp, tmp_path: Path) -> None:
        from mufinder.app.legacy_import_dialog import LegacyScanWorker

        seen: list[tuple[int, int]] = []
        worker = LegacyScanWorker(build_folder(tmp_path, sites=(1, 2)), HOST_LATTICE)
        worker.progressed.connect(lambda done, total: seen.append((done, total)))

        worker.run()

        assert seen == [(1, 2), (2, 2)]

    def test_the_status_bar_names_the_folder_and_position(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        window._legacy_scan_root = tmp_path / "legacy"

        window._on_legacy_scan_progress(3, 36)

        assert window.statusBar().currentMessage() == "Scanning legacy: relaxation 3 of 36..."


class TestScanRunsOnAThread:
    """The scan must survive being started and left to run.

    Every other test in this file drives the worker synchronously by calling
    ``run()``, which exercises the parsing but never the threading. That gap hid
    a real defect: a ``QObject`` moved to a thread is parentless, so nothing but
    a Python reference keeps it alive, and the worker was a local that died the
    moment the starter returned. The scan never ran, and the GUI sat forever on
    a status message that never changed.

    These drive the real signal path through an event loop, so the same mistake
    fails here instead of in the application.
    """

    def _run_to_completion(self, folder: Path, timeout_ms: int = 15000):
        """Drive a real scan and return (events, receiver thread ids).

        Receivers are bound slots on a ``QObject`` that lives on the main
        thread, matching what ``MainWindow`` actually connects. That is not
        incidental: Qt picks the connection type from the receiver's thread
        affinity, so a plain function or lambda has no receiver object and is
        invoked **directly on the worker thread**. Connecting one here would
        make this class silently test direct delivery while claiming to test
        queued delivery, and it would stop catching regressions in the path the
        application really uses.
        """

        import threading

        from PySide6.QtCore import QEventLoop, QObject, QTimer, Slot

        from mufinder.app.legacy_import_dialog import start_legacy_scan

        events: list[tuple] = []
        threads: dict[str, int] = {}
        loop = QEventLoop()

        class Receiver(QObject):
            @Slot(object)
            def completed(self, scan) -> None:
                threads["completed"] = threading.get_ident()
                events.append(("completed", scan))
                loop.quit()

            @Slot(str)
            def failed(self, message: str) -> None:
                threads["failed"] = threading.get_ident()
                events.append(("failed", message))
                loop.quit()

            @Slot(int, int)
            def progressed(self, done: int, total: int) -> None:
                threads.setdefault("progressed", threading.get_ident())
                events.append(("progress", done, total))

        receiver = Receiver()
        start_legacy_scan(
            None,
            folder,
            HOST_LATTICE,
            ["Si"],
            receiver.completed,
            receiver.failed,
            receiver.progressed,
        )
        QTimer.singleShot(timeout_ms, loop.quit)
        loop.exec()
        return events, threads

    def test_a_started_scan_completes_without_being_held_by_the_caller(
        self, qapp, tmp_path: Path
    ) -> None:
        events, _threads = self._run_to_completion(build_folder(tmp_path, sites=(1, 2)))

        kinds = [item[0] for item in events]
        assert "completed" in kinds, f"scan never completed; got {events!r}"
        scan = next(item[1] for item in events if item[0] == "completed")
        assert len(scan.runs) == 2

    def test_progress_is_delivered_across_the_thread_boundary(
        self, qapp, tmp_path: Path
    ) -> None:
        """Queued delivery onto the main thread, asserted rather than assumed.

        The thread-identity check is the point of this test. Without it the
        assertions below pass just as happily when the callbacks run directly on
        the worker thread, which is what a plain-function receiver would do --
        so the test would keep passing while no longer testing its own name.
        """

        import threading

        main_thread = threading.get_ident()
        events, threads = self._run_to_completion(build_folder(tmp_path, sites=(1, 2, 3)))

        assert threads["progressed"] == main_thread
        assert threads["completed"] == main_thread

        progress = [item for item in events if item[0] == "progress"]
        assert progress == [("progress", 1, 3), ("progress", 2, 3), ("progress", 3, 3)]
        # Progress always lands before completion, so the final message is never
        # left showing a partial count.
        assert [item[0] for item in events][-1] == "completed"

    def test_a_failing_scan_reports_across_the_thread_boundary(
        self, qapp, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mufinder.app import legacy_import_dialog

        def _boom(*args, **kwargs):
            raise RuntimeError("disk on fire")

        monkeypatch.setattr(legacy_import_dialog, "scan_legacy_folder", _boom)

        import threading

        main_thread = threading.get_ident()
        events, threads = self._run_to_completion(build_folder(tmp_path, sites=(1,)))

        assert [item[0] for item in events] == ["failed"]
        assert events[0][1] == "disk on fire"
        assert threads["failed"] == main_thread

    def test_the_registry_is_emptied_once_the_thread_exits(
        self, qapp, tmp_path: Path
    ) -> None:
        """A registry that never drains would leak a thread per import."""

        from PySide6.QtCore import QDeadlineTimer

        from mufinder.app.legacy_import_dialog import _LIVE_SCANS

        self._run_to_completion(build_folder(tmp_path, sites=(1,)))

        # Cleanup is asynchronous: the loop quits on `completed`, which fires
        # before the thread has finished, so drain rather than sample once.
        deadline = QDeadlineTimer(5000)
        while _LIVE_SCANS and not deadline.hasExpired():
            qapp.processEvents()

        assert _LIVE_SCANS == set(), "a scan leaked its thread and worker"
