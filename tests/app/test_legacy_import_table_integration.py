"""Imported relaxations in the Structural Relaxations table.

Three things are pinned here, in order of how much damage their absence would
do:

1. Imported jobs cannot be reset or resubmitted. The reset cleanup keeps only
   ``.cell`` and ``.param`` and recursively removes subdirectories, so running
   it on an imported directory would destroy the ``.castep``, the relaxed cell,
   and the whole archive of originals — irreplaceable data.
2. Imported rows cannot be continued. A continuation rebuilds the cell from the
   *initial* muon position and relies on a checkpoint an import does not have,
   so it would silently restart from scratch while presenting itself as a warm
   start.
3. Imported rows render read-only, and merging a batch produces rows whose
   signature matches the widget's — without which every imported row renders
   "ready" with no energy.

Fixtures are hand-authored: see ``docs/legacy_import.rst``.
"""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from mufinder.app.main_window import MainWindow, job_is_imported
from mufinder.domain.constants import IMPORTED_JOB_TAG
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import Job, utc_now


@pytest.fixture()
def window(qapp) -> MainWindow:
    return MainWindow()


def _imported_job(status: JobStatus = JobStatus.COMPLETED) -> Job:
    return Job(
        label="Legacy Site 1",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=status,
        tags=["muon_relaxation", "castep_positivemuon_relaxation", IMPORTED_JOB_TAG],
    )


def _native_job(status: JobStatus = JobStatus.FAILED) -> Job:
    return Job(
        label="Site 1",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=status,
        tags=["muon_relaxation", "castep_positivemuon_relaxation"],
    )


class TestJobPredicate:
    def test_the_tag_identifies_an_imported_job(self) -> None:
        assert job_is_imported(_imported_job()) is True
        assert job_is_imported(_native_job()) is False

    def test_a_job_with_no_tags_is_not_imported(self) -> None:
        job = _native_job()
        job.tags = []
        assert job_is_imported(job) is False


class TestResetIsBlocked:
    def test_an_imported_job_is_never_reset_or_cleaned(
        self, window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The data-loss guard.

        ``_apply_failed_job_reset`` is the shared core of both the dashboard
        Reset action and the lazy reset a dismissed row performs on resubmit, so
        it is guarded directly rather than only at the call sites.
        """

        job = _imported_job(status=JobStatus.FAILED)
        directory = tmp_path / "positivemuon" / "Host_1"
        directory.mkdir(parents=True)
        (directory / "Host_1.castep").write_text("results", encoding="utf-8")
        (directory / "legacy_source").mkdir()
        (directory / "legacy_source" / "Host_1.cell").write_text("original", encoding="utf-8")

        monkeypatch.setattr(window, "_job_output_directory", lambda _job: directory)
        monkeypatch.setattr(window, "_refresh_views_after_job_reset", lambda: None)
        monkeypatch.setattr(window, "_record_job_reset", lambda **kwargs: None)

        reset_count, errors = window._apply_failed_job_reset(
            [job], cleanup_choice=True, detach_convergence=False
        )

        assert reset_count == 0
        assert errors == []
        assert job.status is JobStatus.FAILED
        # The irreplaceable files are still there.
        assert (directory / "Host_1.castep").is_file()
        assert (directory / "legacy_source" / "Host_1.cell").is_file()

    def test_a_native_failed_job_still_resets(
        self, window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The guard must not disable recovery for jobs MuFinder did run."""

        job = _native_job(status=JobStatus.FAILED)
        directory = tmp_path / "positivemuon" / "Site_1"
        directory.mkdir(parents=True)
        (directory / "Site_1.castep").write_text("results", encoding="utf-8")

        monkeypatch.setattr(window, "_job_output_directory", lambda _job: directory)
        monkeypatch.setattr(window, "_refresh_views_after_job_reset", lambda: None)
        monkeypatch.setattr(window, "_record_job_reset", lambda **kwargs: None)

        reset_count, _errors = window._apply_failed_job_reset(
            [job], cleanup_choice=True, detach_convergence=False
        )

        assert reset_count == 1
        assert job.status is JobStatus.DEFINED
        assert not (directory / "Site_1.castep").exists()

    def test_resubmission_preparation_skips_imported_jobs(
        self, window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A dismissed imported job must not be resurrected by a row submit.

        Dismissing a failed job makes its row read "ready" again, and the row's
        own submit is then the recovery path -- which resets the job and cleans
        its directory. For an imported job that would destroy the results.
        """

        from mufinder.domain.models import Project
        from mufinder.services.project_service import ProjectDocument

        job = _imported_job(status=JobStatus.FAILED)
        job.dismissed_at = utc_now()
        window._document = ProjectDocument(
            project_root=tmp_path / "project", project=Project(name="T")
        )
        window._document.jobs[job.id] = job

        prompted: list[int] = []
        monkeypatch.setattr(
            window,
            "_prompt_failed_job_cleanup",
            lambda count: (prompted.append(count), True)[1],
        )

        assert window._prepare_jobs_for_resubmission([job]) is True
        # No cleanup prompt means the reset core was never reached for this job.
        assert prompted == []
        assert job.status is JobStatus.FAILED

    def test_resubmission_preparation_still_recovers_a_native_job(
        self, window: MainWindow, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        from mufinder.domain.models import Project
        from mufinder.services.project_service import ProjectDocument

        job = _native_job(status=JobStatus.FAILED)
        job.dismissed_at = utc_now()
        window._document = ProjectDocument(
            project_root=tmp_path / "project", project=Project(name="T")
        )
        window._document.jobs[job.id] = job

        prompted: list[int] = []
        monkeypatch.setattr(
            window,
            "_prompt_failed_job_cleanup",
            lambda count: (prompted.append(count), False)[1],
        )
        monkeypatch.setattr(window, "_refresh_views_after_job_reset", lambda: None)
        monkeypatch.setattr(window, "_record_job_reset", lambda **kwargs: None)
        monkeypatch.setattr(window, "_persist_document_ui_state", lambda **kwargs: None)
        monkeypatch.setattr(window, "_job_output_directory", lambda _job: None)

        assert window._prepare_jobs_for_resubmission([job]) is True
        assert prompted == [1]
        assert job.status is JobStatus.DEFINED


class TestContinuationIsBlocked:
    def test_an_imported_row_is_not_continuable(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        controller = window._muon_relaxation_controller
        monkeypatch.setattr(controller, "jobs_for_keys", lambda keys: [_imported_job()])

        assert controller.row_is_continuable("positivemuon:Legacy Site 1") is False

    def test_a_native_completed_row_is_still_continuable(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        controller = window._muon_relaxation_controller
        monkeypatch.setattr(
            controller, "jobs_for_keys", lambda keys: [_native_job(status=JobStatus.COMPLETED)]
        )

        assert controller.row_is_continuable("positivemuon:Site 1") is True


class TestReadOnlyRendering:
    def _state(self, *, imported: bool) -> dict[str, object]:
        return {
            "rows": [
                {
                    "site_id": str(uuid4()),
                    "label": "Legacy Site 1" if imported else "Site 1",
                    "fractional_position": [0.1, 0.2, 0.3],
                    "charge_state": "positive_muon",
                    "supercell_matrix": None,
                    "parameter_values": {},
                    "added_advanced": [],
                    "imported": imported,
                    "import_batch_id": str(uuid4()) if imported else None,
                }
            ]
        }

    def test_an_imported_rows_parameters_are_read_only(self, window: MainWindow) -> None:
        widget = window._muon_relaxation_view
        widget.set_form_state(self._state(imported=True))
        row_id = next(iter(widget._row_imported))

        assert widget._row_parameters_locked((row_id,)) is True

    def test_a_native_rows_parameters_are_editable(self, window: MainWindow) -> None:
        widget = window._muon_relaxation_view
        widget.set_form_state(self._state(imported=False))
        row_ids = tuple(widget._row_site_ids)

        assert widget._row_parameters_locked(row_ids) is False

    def test_an_imported_row_cannot_be_removed_from_the_table(
        self, window: MainWindow, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Imported relaxations own the only copy of their results, so removal
        goes through the cascade that first checks downstream references."""

        widget = window._muon_relaxation_view
        widget.set_form_state(self._state(imported=True))
        row_id = next(iter(widget._row_imported))
        monkeypatch.setattr(
            "mufinder.app.muon_relaxation_widget.QMessageBox.information",
            lambda *args, **kwargs: None,
        )
        removed: list[list[str]] = []
        widget.remove_requested.connect(lambda labels: removed.append(labels))

        widget._remove_rows_by_id((row_id,))

        assert removed == []
        assert widget._row_imported == {row_id}

    def test_the_label_of_an_imported_row_is_not_editable(self, window: MainWindow) -> None:
        from PySide6.QtCore import Qt

        widget = window._muon_relaxation_view
        widget.set_form_state(self._state(imported=True))
        table = widget._table_for_charge_state("positive_muon")
        item = table.item(0, widget._LABEL_COLUMN)

        assert item is not None
        assert not (item.flags() & Qt.ItemFlag.ItemIsEditable)
        assert "read-only" in item.toolTip()

    def test_the_label_text_is_left_alone(self, window: MainWindow) -> None:
        """The label is read back as the row's identity, so it must not be
        decorated to show the imported state."""

        widget = window._muon_relaxation_view
        widget.set_form_state(self._state(imported=True))
        table = widget._table_for_charge_state("positive_muon")

        assert table.item(0, widget._LABEL_COLUMN).text() == "Legacy Site 1"


class TestMergingABatch:
    """The app-layer half of an import: rows have to reach both the definition
    and the widget, with signatures that agree."""

    def _import_into(self, window: MainWindow, tmp_path: Path):
        from mufinder.domain.models import Project, Structure
        from mufinder.services.legacy_import import scan_legacy_folder
        from mufinder.services.legacy_import_commit import commit_legacy_import
        from mufinder.services.project_service import ProjectDocument

        from tests.services.test_legacy_import_scan import HOST_LATTICE, build_folder

        host = Structure(
            label="Host",
            origin="imported",
            lattice_vectors=[list(row) for row in HOST_LATTICE],
            species=["Si"],
            fractional_positions=[[0.0, 0.0, 0.0]],
        )
        project = Project(name="T")
        project.host_structure_id = host.id
        document = ProjectDocument(project_root=tmp_path / "project", project=project)
        document.structures[host.id] = host
        window._document = document

        folder = build_folder(tmp_path / "legacy", sites=(1, 2))
        scan = scan_legacy_folder(folder, host_lattice=HOST_LATTICE)
        output_root = document.project_root / "files" / "muon_relaxation"
        outcome = commit_legacy_import(
            document,
            scan=scan,
            runs=scan.importable_runs,
            host_structure=host,
            output_root=output_root,
        )
        merged = window.merge_imported_relaxation_rows(outcome, output_directory=output_root)
        return document, outcome, merged

    def test_merging_creates_a_definition_and_binds_every_job(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        document, outcome, merged = self._import_into(window, tmp_path)

        assert merged is True
        definition = window._active_muon_relaxation_definition()
        assert definition is not None
        mapping = definition.parameters["job_ids_by_label"]
        assert len(mapping) == 2
        for item in outcome.imported:
            key = f"positivemuon:{item.label}"
            assert mapping[key] == str(item.job_id)
            assert definition.parameters["job_seed_names_by_label"][key] == item.seed_name
            assert item.site_id in definition.input_site_ids

    def test_the_merged_rows_are_marked_imported(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        _document, _outcome, _merged = self._import_into(window, tmp_path)

        config = window._deserialize_muon_relaxation_config(
            window._active_muon_relaxation_definition().parameters["configuration"]
        )
        assert config is not None
        assert len(config.rows) == 2
        assert all(row.imported for row in config.rows)

    def test_the_widget_and_the_definition_agree_on_every_signature(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        """The condition that decides whether a row renders with its real
        status. If these disagree the row silently reads "ready" with no energy,
        which is the whole failure mode the mirroring exists to prevent.
        """

        _document, _outcome, _merged = self._import_into(window, tmp_path)

        definition_rows = window._deserialize_muon_relaxation_config(
            window._active_muon_relaxation_definition().parameters["configuration"]
        ).rows
        form_rows = window._muon_relaxation_form_config().rows

        definition_signatures = {
            window._muon_relaxation_row_key(row): window._muon_relaxation_row_signature(row)
            for row in definition_rows
        }
        form_signatures = {
            window._muon_relaxation_row_key(row): window._muon_relaxation_row_signature(row)
            for row in form_rows
        }

        assert definition_signatures
        assert form_signatures == definition_signatures

    def test_the_mirrored_positions_are_not_zeroed(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        """The two shapes name the position differently; a straight pass-through
        parses cleanly and silently zeroes it."""

        _document, _outcome, _merged = self._import_into(window, tmp_path)

        form_rows = window._muon_relaxation_form_config().rows
        assert all(
            any(abs(value) > 1e-9 for value in row.initial_fractional_position)
            for row in form_rows
        )

    def test_merging_preserves_existing_rows(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        """Re-registering the definition would arrive with an empty job mapping
        and sever every existing row's binding."""

        document, _outcome, _merged = self._import_into(window, tmp_path)
        definition = window._active_muon_relaxation_definition()
        before = dict(definition.parameters["job_ids_by_label"])

        from mufinder.services.legacy_import import scan_legacy_folder
        from mufinder.services.legacy_import_commit import commit_legacy_import
        from tests.services.test_legacy_import_scan import HOST_LATTICE, build_folder

        host = document.structures[document.project.host_structure_id]
        folder = build_folder(tmp_path / "legacy2", sites=(3,), seed="Other")
        scan = scan_legacy_folder(folder, host_lattice=HOST_LATTICE)
        outcome = commit_legacy_import(
            document,
            scan=scan,
            runs=scan.importable_runs,
            host_structure=host,
            output_root=document.project_root / "files" / "muon_relaxation",
            existing_labels=[item for item in before],
        )
        window.merge_imported_relaxation_rows(
            outcome, output_directory=document.project_root / "files" / "muon_relaxation"
        )

        after = window._active_muon_relaxation_definition().parameters["job_ids_by_label"]
        assert before.items() <= after.items()
        assert len(after) == 3
