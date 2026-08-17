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
