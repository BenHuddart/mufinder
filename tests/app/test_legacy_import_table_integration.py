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


class TestImportedRowParameters:
    """An imported row reports the parameters its run actually used.

    The row is what a user reads to learn what a legacy calculation was run
    with. Showing the workflow's current defaults instead is not a cosmetic
    problem: the defaults are plausible values for the same material, so a wrong
    cutoff or tolerance reads as fact.
    """

    def _import_into(self, window: MainWindow, tmp_path: Path):
        return TestMergingABatch()._import_into(window, tmp_path)

    def _row_for(self, window: MainWindow, label: str):
        config = window._muon_relaxation_form_config()
        return next(row for row in config.rows if row.label == label)

    def test_the_row_carries_what_the_run_used(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        _document, outcome, _merged = self._import_into(window, tmp_path)

        row = self._row_for(window, outcome.imported[0].label)
        values = dict(row.parameter_values)

        assert float(getattr(values["cut_off_energy"], "value", values["cut_off_energy"])) == 600.0
        assert values["xc_functional"] == "PBE"
        assert tuple(values["kpoint_mp_grid"]) == (2, 2, 2)

    def test_a_value_matching_todays_default_is_still_recorded(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        """Stripping values that match a default would make what the row stores
        depend on ambient state that can change. Defaults move; the run cannot.
        """

        _document, outcome, _merged = self._import_into(window, tmp_path)

        values = dict(self._row_for(window, outcome.imported[0].label).parameter_values)

        # Present in the run's recovery *and* equal to the workflow default.
        assert "max_scf_cycles" in values
        assert "elec_energy_tol" in values

    def test_the_losing_half_of_an_exclusive_pair_is_dropped(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        """The fixture states BASIS_PRECISION in its .param while the output
        echo reports the CUT_OFF_ENERGY derived from it. Keeping both would
        describe a calculation CASTEP would reject."""

        _document, outcome, _merged = self._import_into(window, tmp_path)

        values = dict(self._row_for(window, outcome.imported[0].label).parameter_values)

        assert "cut_off_energy" in values
        assert "basis_precision" not in values

    def test_the_asserted_keywords_are_not_presented_as_row_choices(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        """TASK, CHARGE, FIX_ALL_CELL and WRITE_CELL_STRUCTURE are asserted by
        the import and shown as locked values, not chosen by the row."""

        _document, outcome, _merged = self._import_into(window, tmp_path)

        values = dict(self._row_for(window, outcome.imported[0].label).parameter_values)

        for name in ("charge", "fix_all_cell", "write_cell_structure"):
            assert name not in values

    def test_the_dialog_shows_no_parameter_the_run_did_not_use(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        """The effective values seed the read-only dialog. Layering the
        workflow defaults underneath would present parameters the historical run
        never used as though it had."""

        _document, outcome, _merged = self._import_into(window, tmp_path)
        widget = window._muon_relaxation_view
        row_id = next(iter(widget._row_imported))

        effective = widget._effective_params_for_row(row_id)

        stored = set(widget._row_parameter_values[row_id])
        # "task" is the workflow's own locked value, not a recovered one.
        assert set(effective) - {"task"} == stored

    def test_no_kpoint_spacing_is_invented_for_an_imported_row(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        """The suggested spacing comes from the project reference lattice, not
        from the row's own recovered grid, so pre-filling it would display a
        spacing the run never used."""

        _document, _outcome, _merged = self._import_into(window, tmp_path)
        widget = window._muon_relaxation_view
        widget.set_project_defaults(
            cutoff_energy=800.0,
            kpoint_grid=(4, 4, 4),
            kpoint_reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        )
        row_id = next(iter(widget._row_imported))

        common = widget._common_params_for_rows((row_id,))

        assert "kpoint_mp_spacing" not in common
        assert tuple(common["kpoint_mp_grid"]) == (2, 2, 2)

    def test_the_configured_indicator_reports_the_row_as_customized(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        _document, _outcome, _merged = self._import_into(window, tmp_path)
        widget = window._muon_relaxation_view
        row_id = next(iter(widget._row_imported))

        assert widget._row_parameter_values[row_id]


class TestSignatureStability:
    """The imported row must survive changes to ambient project state.

    The definition row is a frozen snapshot taken at import; the form row is
    re-normalised on every project-default and host-lattice change. If
    normalisation depended on that ambient state the two would drift apart, and
    a signature mismatch drops the completed import to a row that renders
    "ready" with no energy.
    """

    def _import_into(self, window: MainWindow, tmp_path: Path):
        return TestMergingABatch()._import_into(window, tmp_path)

    def _signatures(self, window: MainWindow):
        definition_rows = window._deserialize_muon_relaxation_config(
            window._active_muon_relaxation_definition().parameters["configuration"]
        ).rows
        form_rows = window._muon_relaxation_form_config().rows
        return (
            {
                window._muon_relaxation_row_key(row): window._muon_relaxation_row_signature(row)
                for row in definition_rows
            },
            {
                window._muon_relaxation_row_key(row): window._muon_relaxation_row_signature(row)
                for row in form_rows
            },
        )

    def test_signatures_survive_defaults_moving_onto_the_imported_values(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        """The realistic trigger: adopting the very cutoff and grid a
        convergence run on this system recommends, which the legacy run also
        used."""

        _document, _outcome, _merged = self._import_into(window, tmp_path)
        definition_before, form_before = self._signatures(window)
        assert form_before == definition_before

        window._muon_relaxation_view.set_project_defaults(
            cutoff_energy=600.0,
            kpoint_grid=(2, 2, 2),
            kpoint_reference_lattice=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        )
        window._muon_relaxation_view.set_host_lattice(
            [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
        )

        definition_after, form_after = self._signatures(window)
        assert form_after == definition_after
        assert form_after == form_before

    def test_signatures_survive_a_save_and_reload(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        """The stored values are structured (units, triplets); a round trip
        through the persisted form must coerce them back identically."""

        document, _outcome, _merged = self._import_into(window, tmp_path)
        definition_before, form_before = self._signatures(window)
        assert form_before == definition_before

        import json

        from mufinder.persistence.codec import to_jsonable

        # The real save path: encoded, written as JSON, read back.
        reloaded = json.loads(
            json.dumps(to_jsonable(document.ui_state["muon_relaxation_form"]))
        )
        window._muon_relaxation_view.set_form_state(reloaded)

        _definition_after, form_after = self._signatures(window)
        assert form_after == definition_before

    def test_a_keyword_the_catalog_no_longer_knows_is_kept(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        """Catalog discovery is environment-dependent, so the same project can
        meet a thinner one elsewhere. Dropping a name from the widget's state
        while the definition keeps it would desynchronise the two."""

        document, _outcome, _merged = self._import_into(window, tmp_path)
        state = window._muon_relaxation_view.collect_state()
        for row in state["rows"]:
            if row.get("imported"):
                row["parameter_values"]["not_a_known_keyword"] = 7

        window._muon_relaxation_view.set_form_state(state)

        kept = [
            row
            for row in window._muon_relaxation_view.collect_state()["rows"]
            if row.get("imported")
        ]
        assert kept
        assert all(row["parameter_values"]["not_a_known_keyword"] == 7 for row in kept)


class TestNativeRowsAreUnaffected:
    """An imported row's recovered values must never block a native row.

    Expansion and validation both run over the whole configuration, and both
    turn a single bad row into a result that looks like "nothing to submit".
    """

    def _import_into(self, window: MainWindow, tmp_path: Path):
        return TestMergingABatch()._import_into(window, tmp_path)

    def test_expansion_ignores_imported_rows(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        from mufinder.app.workflow_controllers.muon_relaxation import (
            _without_imported_rows,
        )

        _document, _outcome, _merged = self._import_into(window, tmp_path)
        config = window._muon_relaxation_form_config()
        assert any(row.imported for row in config.rows)

        stripped = _without_imported_rows(config)

        assert stripped.rows == ()

    def test_a_config_with_no_imported_rows_is_returned_unchanged(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        """Identity matters: the common case must not pay for a rebuild."""

        from mufinder.app.relaxation_types import MuonRelaxationConfig
        from mufinder.app.workflow_controllers.muon_relaxation import (
            _without_imported_rows,
        )

        config = MuonRelaxationConfig(rows=())

        assert _without_imported_rows(config) is config

    def test_an_imported_row_cannot_veto_the_run_buttons(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        """A recovered value the current schema dislikes would otherwise abort
        config building and disable submission for every native row."""

        _document, _outcome, _merged = self._import_into(window, tmp_path)
        widget = window._muon_relaxation_view
        row_id = next(iter(widget._row_imported))
        widget._row_parameter_values[row_id]["cut_off_energy"] = "not a number at all"

        assert widget._build_config_or_warn() is not None


class TestReadOnlyDialogContents:
    """What the read-only dialog actually renders for an imported row."""

    def _import_into(self, window: MainWindow, tmp_path: Path):
        return TestMergingABatch()._import_into(window, tmp_path)

    def _dialog_for_imported_row(self, window: MainWindow):
        """Build the dialog the widget would open, without exec()ing it."""

        from mufinder.app.param_dialog import ParameterDialog

        widget = window._muon_relaxation_view
        row_id = next(iter(widget._row_imported))
        dialog = ParameterDialog(
            specs=widget._specs,
            workflow_stage="muon_relaxation",
            read_only=True,
        )
        dialog.set_values(widget._common_params_for_rows((row_id,)))
        dialog.set_advanced_names(widget._row_added_advanced.get(row_id, set()))
        return widget, row_id, dialog

    def test_a_recovered_advanced_keyword_renders_without_being_listed(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        """``added_advanced`` is deliberately stored empty: the dialog promotes
        any advanced keyword carrying an active value. Computing the list at
        import time instead would tie the row signature to the runtime keyword
        catalog, which differs between machines.
        """

        _document, _outcome, _merged = self._import_into(window, tmp_path)
        widget = window._muon_relaxation_view
        row_id = next(iter(widget._row_imported))

        # Chosen from the live spec set rather than named, so the test does not
        # depend on how the runtime keyword catalog classifies any one keyword.
        advanced = next(
            spec.name
            for spec in widget._specs
            if "muon_relaxation" in spec.advanced_workflows
            and "muon_relaxation" not in spec.basic_workflows
            and spec.value_type == "int"
            and not spec.mutually_exclusive_with
        )
        widget._row_parameter_values[row_id][advanced] = 7

        _widget, _row_id, dialog = self._dialog_for_imported_row(window)

        assert widget._row_added_advanced.get(row_id, set()) == set()
        assert advanced in dialog.advanced_names()
        assert advanced in dialog._rows

    def test_the_recovered_cutoff_is_the_value_on_display(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        _document, _outcome, _merged = self._import_into(window, tmp_path)

        _widget, _row_id, dialog = self._dialog_for_imported_row(window)

        shown = dialog._rows["cut_off_energy"].editor.value()
        assert float(getattr(shown, "value", shown)) == 600.0

    def test_no_spacing_row_is_active_for_an_imported_row(
        self, window: MainWindow, tmp_path: Path
    ) -> None:
        _document, _outcome, _merged = self._import_into(window, tmp_path)

        _widget, _row_id, dialog = self._dialog_for_imported_row(window)

        assert "kpoint_mp_spacing" not in dialog.collect_values()
