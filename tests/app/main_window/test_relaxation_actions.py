"""Pristine relaxation actions in the main window."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace

import pytest

import mufinder.app.main_window as main_window_module
import mufinder.services.job_artifacts as job_artifacts_module
from mufinder.app.workflow_controllers.base import output_predates_run
from mufinder.castep_io.output_parser import CastepResult
from mufinder.app.main_window import MainWindow
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, ComparisonTable, Job, Structure
from mufinder.services.project_service import ProjectService


def test_main_window_local_finish_pristine_relaxation_uses_first_three_lattice_parameters(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "local_finish_pristine_relax.mufinder", "LocalFinishPristineRelax")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    job = service.add_job(
        doc,
        label="pristine_relax_row1",
        workflow_stage=main_window_module.WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    job.status = JobStatus.RUNNING
    run = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.RUNNING)
    doc.runs[run.id] = run
    window._document = doc
    window._active_run_by_job[job.id] = run.id

    output_dir = doc.project_root / "files" / "pristine_relaxation" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{job.label}-out.cell").write_text("dummy", encoding="utf-8")

    # The relaxed -out.cell is parsed once by the off-thread collector
    # (mufinder.services.job_artifacts), so patch the collector's parse helpers
    # rather than main_window's (which no longer parses secondary outputs).
    monkeypatch.setattr(
        job_artifacts_module,
        "parse_cell_file",
        lambda _path: SimpleNamespace(
            lattice_vectors=((5.1, 0.0, 0.0), (0.0, 5.2, 0.0), (0.0, 0.0, 5.3)),
            species=("Si",),
            fractional_positions=((0.0, 0.0, 0.0),),
        ),
    )
    monkeypatch.setattr(
        job_artifacts_module,
        "lattice_parameters",
        lambda _vectors: (5.1, 5.2, 5.3, 90.0, 90.0, 90.0),
    )

    parsed = CastepResult(
        completed=True,
        total_energy=-11.0,
        warnings=(),
        geom_converged=True,
    )
    window._on_local_job_finished(job.id, parsed)

    assert job.status == JobStatus.COMPLETED
    assert run.status == JobStatus.COMPLETED
    assert run.result_id is not None
    result = doc.results[run.result_id]
    assert result.data["lattice_a"] == pytest.approx(5.1)
    assert result.data["lattice_b"] == pytest.approx(5.2)
    assert result.data["lattice_c"] == pytest.approx(5.3)
    assert len(result.derived_structure_ids) == 1
    window.close()


def test_main_window_local_finish_pristine_relaxation_missing_out_cell_fails_job(
    main_window, tmp_path: Path
) -> None:
    """PristineRelaxationController.validate_artifacts, dispatched from
    _on_local_job_finished, must fail a pristine job whose relaxed structure
    output is absent BEFORE build_result is ever reached."""

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "pristine_missing_out.mufinder", "PristineMissingOut")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    job = service.add_job(
        doc,
        label="pristine_relax_row1",
        workflow_stage=main_window_module.WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    job.status = JobStatus.RUNNING
    run = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.RUNNING)
    doc.runs[run.id] = run
    window._document = doc
    window._active_run_by_job[job.id] = run.id

    # Output directory exists but the relaxed -out.cell was NOT written.
    output_dir = doc.project_root / "files" / "pristine_relaxation" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)

    parsed = CastepResult(
        completed=True,
        total_energy=-11.0,
        warnings=(),
        geom_converged=True,  # clears the geom prologue so we reach the out.cell guard
    )
    window._on_local_job_finished(job.id, parsed)

    assert job.status == JobStatus.FAILED
    assert run.result_id is None
    assert not any(
        structure.origin == "pristine_relaxation" for structure in doc.structures.values()
    )
    window.close()


_PRISTINE_OUT_CELL_FIRST = """
%BLOCK LATTICE_CART
ang
5.1 0.0 0.0
0.0 5.2 0.0
0.0 0.0 5.3
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Si 0.0 0.0 0.0
%ENDBLOCK POSITIONS_FRAC
"""


_PRISTINE_OUT_CELL_CONTINUED = """
%BLOCK LATTICE_CART
ang
5.4 0.0 0.0
0.0 5.5 0.0
0.0 0.0 5.6
%ENDBLOCK LATTICE_CART

%BLOCK POSITIONS_FRAC
Si 0.01 0.02 0.03
%ENDBLOCK POSITIONS_FRAC
"""


def _pristine_relaxation_job_fixture(window: MainWindow, service: ProjectService, tmp_path: Path, name: str):
    """Build a document with one running pristine-relaxation job and its output dir."""

    doc = service.create_project(tmp_path / f"{name}.mufinder", name)
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    job = service.add_job(
        doc,
        label="pristine_relax_row1",
        workflow_stage=main_window_module.WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    window._document = doc
    output_dir = doc.project_root / "files" / "pristine_relaxation" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    return doc, host, job, output_dir


def _start_pristine_run(window: MainWindow, service: ProjectService, doc, job, run_index: int):
    """Take ``job`` through submit → start and register a fresh active run."""

    service.submit_job(job)
    service.start_job(job)
    run = CalculationRun(job_id=job.id, run_index=run_index, status=JobStatus.RUNNING)
    run.started_at = job.started_at
    doc.runs[run.id] = run
    window._active_run_by_job[job.id] = run.id
    return run


def test_main_window_local_finish_relaxation_stale_out_cell_fails_job(
    main_window, tmp_path: Path
) -> None:
    """A continued run that dies before writing a new ``-out.cell`` leaves the
    PREVIOUS run's file in the re-used job directory, where the completion path
    would happily validate it as this run's geometry. The relaxation
    controller's stale-output check fails the job instead, so no result is built from a superseded file."""

    window = main_window
    service = ProjectService()
    doc, _host, job, output_dir = _pristine_relaxation_job_fixture(
        window, service, tmp_path, "StaleOutCell"
    )

    # The previous run's relaxed structure, left behind in the re-used directory.
    (output_dir / f"{job.label}-out.cell").write_text(
        _PRISTINE_OUT_CELL_FIRST, encoding="utf-8"
    )
    run = _start_pristine_run(window, service, doc, job, run_index=2)
    # This run started well after that file was written.
    run.started_at = datetime.now(UTC) + timedelta(minutes=5)

    parsed = CastepResult(
        completed=True,
        total_energy=-11.0,
        warnings=(),
        geom_converged=True,  # clears the geom prologue so we reach the stale guard
    )
    window._on_local_job_finished(job.id, parsed)

    assert job.status == JobStatus.FAILED
    assert "predates this run" in (job.error_message or "")
    assert run.result_id is None
    assert not any(
        structure.origin == "pristine_relaxation" for structure in doc.structures.values()
    )
    window.close()


def test_main_window_stale_output_guard_skips_jobs_with_no_start_time(tmp_path: Path) -> None:
    """The guard fails open. A job the poller never observed running has no
    ``started_at`` to compare against, so its output is accepted rather than a
    job being failed on no evidence. Sub-slack differences are also accepted, so
    filesystem mtime granularity cannot false-fail a fresh output."""

    path = tmp_path / "seed-out.cell"
    path.write_text("cell", encoding="utf-8")
    written = datetime.fromtimestamp(path.stat().st_mtime, UTC)

    assert output_predates_run(path, None) is False
    assert output_predates_run(None, written) is False
    # Written after the run started, and written a hair before it: both fresh.
    assert output_predates_run(path, written - timedelta(minutes=5)) is False
    assert output_predates_run(path, written + timedelta(seconds=1)) is False
    # A whole run older: stale.
    assert output_predates_run(path, written + timedelta(minutes=5)) is True
    # An unreadable path is not evidence of staleness either.
    assert output_predates_run(tmp_path / "absent", written) is False


def test_main_window_pristine_relaxation_recompletion_keeps_trail_and_references(
    main_window, tmp_path: Path
) -> None:
    """Drive a full continue → running → re-complete cycle through the real
    completion path.

    The pristine-relaxation workflow-trail milestone is table-driven
    (``_workflow_pristine_relaxation_complete`` reads ``document.comparison_tables``),
    **not** job-driven, so it stays complete across the whole cycle — continuing a
    job neither un-completes nor re-completes it. What the cycle really has to
    protect is the reference that milestone stands for: the comparison table's
    ``dft_structure_ids`` must still resolve, which holds only because the second
    completion updates the derived structure in place instead of minting a new one.
    """

    window = main_window
    service = ProjectService()
    doc, host, job, output_dir = _pristine_relaxation_job_fixture(
        window, service, tmp_path, "PristineRecompletion"
    )
    parsed = CastepResult(
        completed=True, total_energy=-11.0, warnings=(), geom_converged=True
    )

    run_one = _start_pristine_run(window, service, doc, job, run_index=1)
    (output_dir / f"{job.label}-out.cell").write_text(
        _PRISTINE_OUT_CELL_FIRST, encoding="utf-8"
    )
    window._on_local_job_finished(job.id, parsed)

    assert job.status == JobStatus.COMPLETED
    first_result = doc.results[run_one.result_id]
    derived_id = first_result.derived_structure_ids[0]
    structure_count = len(doc.structures)

    # A comparison table built from that first completion.
    table = ComparisonTable(
        label="Comparison 1",
        experimental_structure_id=host.id,
        dft_structure_ids=[derived_id],
        dft_labels=[job.label],
    )
    service.add_comparison_table(doc, table)
    assert MainWindow._workflow_pristine_relaxation_complete(doc) is True

    # Continue the completed job and start its second run.
    service.prepare_job_continuation(job)
    assert job.status == JobStatus.DEFINED
    run_two = _start_pristine_run(window, service, doc, job, run_index=2)
    assert job.status == JobStatus.RUNNING
    # Table-driven, so the milestone does not dip while the continued job runs.
    assert MainWindow._workflow_pristine_relaxation_complete(doc) is True

    (output_dir / f"{job.label}-out.cell").write_text(
        _PRISTINE_OUT_CELL_CONTINUED, encoding="utf-8"
    )
    window._on_local_job_finished(job.id, parsed)

    assert job.status == JobStatus.COMPLETED
    assert MainWindow._workflow_pristine_relaxation_complete(doc) is True

    # One derived structure, same id, refreshed geometry; both Results agree.
    second_result = doc.results[run_two.result_id]
    assert second_result.id != first_result.id
    assert second_result.derived_structure_ids == [derived_id]
    assert len(doc.structures) == structure_count
    assert second_result.data["lattice_a"] == pytest.approx(5.4)
    derived = doc.structures[derived_id]
    assert derived.lattice_vectors == [[5.4, 0.0, 0.0], [0.0, 5.5, 0.0], [0.0, 0.0, 5.6]]

    # The comparison table's reference survived the cycle intact.
    assert table.dft_structure_ids == [derived_id]
    assert service.validate_document(doc) == []
    window.close()


def test_on_pristine_relaxation_configured_surfaces_kpoint_message(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A supercell row rescales the project k-point grid; the adjustment message
    # from AdapterExpansion.messages must be surfaced to the user, not dropped.
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "pristine_msg.mufinder", "PristineMsg")

    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    doc.project.recommended_kpoint_grid = (4, 4, 4)
    doc.project.recommended_kpoint_lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]
    window._document = doc

    infos: list[tuple[str, str]] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, text, *args, **kwargs: infos.append((title, text)),
    )

    window._on_pristine_relaxation_configured(
        main_window_module.PristineRelaxationConfig(
            name_prefix="pristine_relax",
            rows=(
                main_window_module.PristineRelaxationRow(
                    label="super",
                    fix_all_cell=False,
                    supercell_matrix=((2, 0, 0), (0, 1, 0), (0, 0, 1)),
                    parameter_values={},
                ),
            ),
        )
    )

    assert any("K-point grid adjusted" in text for _title, text in infos)
    window.close()


def test_main_window_form_pristine_relaxation_shows_ready_before_submission(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "pristine_form_ready.mufinder", "PristineFormReady")

    doc.ui_state["pristine_relaxation_form"] = {
        "name_prefix": "pristine_relax",
        "rows": [
            {"label": "row_1", "fix_all_cell": False, "params": {}, "added_advanced": []},
            {"label": "row_2", "fix_all_cell": True, "params": {}, "added_advanced": []},
        ],
    }

    window._document = doc
    window._refresh_ui_after_project_change()

    assert window._jobs_table.rowCount() == 0
    assert window._pristine_relaxation_view._table.item(0, window._pristine_relaxation_view._STATUS_COLUMN).text() == "ready"
    assert window._pristine_relaxation_view._table.item(1, window._pristine_relaxation_view._STATUS_COLUMN).text() == "ready"
    window.close()


def test_main_window_pristine_form_edit_hides_stale_definition_status(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "pristine_form_stale.mufinder", "PristineFormStale")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    row = main_window_module.PristineRelaxationRow(label="row_1", parameter_values={})
    definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_pristine_relaxation_config(
            main_window_module.PristineRelaxationConfig(name_prefix="pristine_relax", rows=(row,))
        ),
        output_directory=doc.project_root / "files" / "pristine_relaxation",
    )
    running_job = service.add_job(
        doc,
        label="pristine_relax_row_1",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    running_job.status = JobStatus.RUNNING
    definition.parameters["job_ids_by_label"] = {"row_1": str(running_job.id)}
    doc.ui_state[window._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY] = str(definition.id)
    doc.ui_state["pristine_relaxation_form"] = {
        "name_prefix": "pristine_relax",
        "rows": [
            {"label": "row_1", "fix_all_cell": True, "params": {}, "added_advanced": []},
        ],
    }

    window._refresh_ui_after_project_change()

    assert window._pristine_relaxation_view._table.item(0, window._pristine_relaxation_view._STATUS_COLUMN).text() == "ready"
    window.close()


def test_main_window_pristine_form_edit_preserves_matching_row_statuses(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "pristine_form_partial_status.mufinder", "PristineFormPartialStatus")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    spinpol_row = main_window_module.PristineRelaxationRow(
        label="spinpol",
        parameter_values={"spin_polarized": True},
    )
    nospin_row = main_window_module.PristineRelaxationRow(label="nospin", parameter_values={})
    definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_pristine_relaxation_config(
            main_window_module.PristineRelaxationConfig(
                name_prefix="pristine_relax",
                rows=(spinpol_row, nospin_row),
            )
        ),
        output_directory=doc.project_root / "files" / "pristine_relaxation",
    )
    completed_job = service.add_job(
        doc,
        label="pristine_relax_spinpol",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    completed_job.status = JobStatus.COMPLETED
    reset_job = service.add_job(
        doc,
        label="pristine_relax_nospin",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    reset_job.status = JobStatus.DEFINED
    definition.parameters["job_ids_by_label"] = {
        "spinpol": str(completed_job.id),
        "nospin": str(reset_job.id),
    }
    doc.ui_state[window._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY] = str(definition.id)
    doc.ui_state["pristine_relaxation_form"] = {
        "name_prefix": "pristine_relax",
        "rows": [
            {
                "label": "spinpol",
                "fix_all_cell": False,
                "params": {"spin_polarized": True},
                "added_advanced": [],
            },
            {
                "label": "nospin",
                "fix_all_cell": False,
                "params": {"nextra_bands": 17},
                "added_advanced": ["nextra_bands"],
            },
        ],
    }

    window._refresh_ui_after_project_change()

    assert window._pristine_relaxation_view._table.item(0, window._pristine_relaxation_view._STATUS_COLUMN).text() == "completed"
    assert window._pristine_relaxation_view._table.item(1, window._pristine_relaxation_view._STATUS_COLUMN).text() == "ready"
    window.close()


def test_pristine_relaxation_preserves_running_row_when_adding_job(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "pristine_running.mufinder", "PristineRunning")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    row_1 = main_window_module.PristineRelaxationRow(label="row_1", parameter_values={})
    row_2 = main_window_module.PristineRelaxationRow(label="row_2", parameter_values={})
    old_config = main_window_module.PristineRelaxationConfig(name_prefix="pristine_relax", rows=(row_1,))
    old_definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_pristine_relaxation_config(old_config),
        output_directory=doc.project_root / "files" / "pristine_relaxation",
    )
    running_job = service.add_job(
        doc,
        label="pristine_relax_row_1",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    running_job.status = JobStatus.RUNNING
    old_definition.parameters["job_ids_by_label"] = {"row_1": str(running_job.id)}

    new_config = main_window_module.PristineRelaxationConfig(name_prefix="pristine_relax", rows=(row_1, row_2))
    new_definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_pristine_relaxation_config(new_config),
        output_directory=doc.project_root / "files" / "pristine_relaxation",
    )
    doc.ui_state[window._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY] = str(new_definition.id)

    submitted: list[Job] = []
    monkeypatch.setattr(window, "_submit_jobs_on_cluster", lambda jobs: submitted.extend(jobs))

    window._pristine_relaxation_controller.submit(["row_2"], target="cluster")

    states = window._pristine_relaxation_controller.build_states()
    assert states["row_1"].status == JobStatus.RUNNING.value
    assert states["row_1"].job_id == running_job.id
    assert len(submitted) == 1
    assert submitted[0].label == "pristine_relax_row_2"
    assert new_definition.parameters["job_ids_by_label"]["row_1"] == str(running_job.id)
    window.close()


def test_pristine_relaxation_row_parameter_edit_keeps_row_bound_to_its_job(
    main_window, tmp_path: Path
) -> None:
    """In-place parameter edits must not sever the row→job binding.

    Registering a new definition would give the row an empty
    ``job_ids_by_label``; and because pristine relaxation has no active-status
    escape hatch in ``build_states``, a form/definition signature mismatch would
    drop the row to a jobless "ready". This drives the real widget, so the
    table's own parameter normalization (implicit project defaults are stripped)
    is exercised end to end.
    """

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "pristine_param_edit.mufinder", "PristineParamEdit")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    row = main_window_module.PristineRelaxationRow(
        label="row_1", parameter_values={"geom_max_iter": 120}
    )
    definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_pristine_relaxation_config(
            main_window_module.PristineRelaxationConfig(name_prefix="pristine_relax", rows=(row,))
        ),
        output_directory=doc.project_root / "files" / "pristine_relaxation",
    )
    doc.ui_state[window._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY] = str(definition.id)
    job = service.add_job(
        doc,
        label="pristine_relax_row_1",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    job.status = JobStatus.COMPLETED
    definition.parameters["job_ids_by_label"] = {"row_1": str(job.id)}
    window._restore_pristine_relaxation_form_from_definition()
    assert window._pristine_relaxation_controller.build_states()["row_1"].job_id == job.id

    updated = window._pristine_relaxation_controller.update_row_parameters(
        "row_1", {"geom_max_iter": 500, "task": "GEOMETRYOPTIMIZATION"}
    )

    assert updated is True
    # No new definition; the binding is intact and the row still resolves its job.
    assert [
        candidate.id
        for candidate in doc.calculations.values()
        if candidate.workflow_stage == WorkflowStage.PRISTINE_RELAXATION
    ] == [definition.id]
    assert definition.parameters["job_ids_by_label"] == {"row_1": str(job.id)}
    states = window._pristine_relaxation_controller.build_states()
    assert states["row_1"].job_id == job.id
    assert states["row_1"].status == JobStatus.COMPLETED.value

    config = window._deserialize_pristine_relaxation_config(
        definition.parameters["configuration"]
    )
    assert config.rows[0].parameter_values["geom_max_iter"] == 500
    # The widget is the normalizer of record: it strips the implicit ``task``
    # default, so the definition must not keep it either.
    assert "task" not in config.rows[0].parameter_values
    form_config = window._pristine_relaxation_form_config()
    assert window._pristine_relaxation_row_signature(
        form_config.rows[0]
    ) == window._pristine_relaxation_row_signature(config.rows[0])
    assert (
        doc.ui_state["pristine_relaxation_form"]["rows"][0]["params"]["geom_max_iter"]
        == 500
    )
    window.close()


def test_pristine_relaxation_deleted_row_does_not_reinherit_stale_status(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "pristine_deleted_stale.mufinder", "PristineDeletedStale")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id
    window._document = doc

    row = main_window_module.PristineRelaxationRow(label="row_1", parameter_values={})
    old_definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_pristine_relaxation_config(
            main_window_module.PristineRelaxationConfig(name_prefix="pristine_relax", rows=(row,))
        ),
        output_directory=doc.project_root / "files" / "pristine_relaxation",
    )
    running_job = service.add_job(
        doc,
        label="pristine_relax_row_1",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    running_job.status = JobStatus.RUNNING
    old_definition.parameters["job_ids_by_label"] = {"row_1": str(running_job.id)}

    window._on_remove_pristine_relaxation_rows(["row_1"])

    new_definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_pristine_relaxation_config(
            main_window_module.PristineRelaxationConfig(name_prefix="pristine_relax", rows=(row,))
        ),
        output_directory=doc.project_root / "files" / "pristine_relaxation",
    )
    doc.ui_state[window._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY] = str(new_definition.id)

    states = window._pristine_relaxation_controller.build_states()

    assert states["row_1"].status == "ready"
    assert new_definition.parameters.get("job_ids_by_label", {}) == {}
    assert old_definition.id not in doc.calculations
    window.close()


def test_pristine_relaxation_remove_row_deletes_associated_files_and_job(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "pristine_remove_files.mufinder", "PristineRemoveFiles")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    window._document = doc

    row = main_window_module.PristineRelaxationRow(label="row_1", parameter_values={})
    definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_pristine_relaxation_config(
            main_window_module.PristineRelaxationConfig(name_prefix="pristine_relax", rows=(row,))
        ),
        output_directory=doc.project_root / "files" / "pristine_relaxation",
    )
    job = service.add_job(
        doc,
        label="pristine_relax_row_1",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    definition.parameters["job_ids_by_label"] = {"row_1": str(job.id)}
    job_dir = doc.project_root / "files" / "pristine_relaxation" / job.label
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / f"{job.label}.cell").write_text("cell", encoding="utf-8")
    monkeypatch.setattr(window, "_prompt_job_file_disposition", lambda **_kwargs: "delete")

    window._on_remove_pristine_relaxation_rows(["row_1"])

    assert not job_dir.exists()
    assert job.id not in doc.jobs
    assert definition.id not in doc.calculations
    window.close()


def test_pristine_relaxation_remove_row_cancel_keeps_files_and_restores_form(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "pristine_remove_cancel.mufinder", "PristineRemoveCancel")
    host = Structure(
        label="Host",
        lattice_vectors=[[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]],
        species=["Si"],
        fractional_positions=[[0.0, 0.0, 0.0]],
    )
    service.add_structure(doc, host)
    window._document = doc

    row = main_window_module.PristineRelaxationRow(label="row_1", parameter_values={})
    definition = service.register_pristine_relaxation_definition(
        doc,
        input_structure_id=host.id,
        parameters=window._serialize_pristine_relaxation_config(
            main_window_module.PristineRelaxationConfig(name_prefix="pristine_relax", rows=(row,))
        ),
        output_directory=doc.project_root / "files" / "pristine_relaxation",
    )
    doc.ui_state[window._ACTIVE_PRISTINE_RELAXATION_DEFINITION_KEY] = str(definition.id)
    job = service.add_job(
        doc,
        label="pristine_relax_row_1",
        workflow_stage=WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    definition.parameters["job_ids_by_label"] = {"row_1": str(job.id)}
    job_dir = doc.project_root / "files" / "pristine_relaxation" / job.label
    job_dir.mkdir(parents=True, exist_ok=True)
    (job_dir / f"{job.label}.cell").write_text("cell", encoding="utf-8")
    window._pristine_relaxation_view.set_form_state({"name_prefix": "pristine_relax", "rows": []})
    monkeypatch.setattr(window, "_prompt_job_file_disposition", lambda **_kwargs: None)

    window._on_remove_pristine_relaxation_rows(["row_1"])

    assert job_dir.exists()
    assert job.id in doc.jobs
    assert definition.id in doc.calculations
    assert window._pristine_relaxation_view.collect_state()["rows"][0]["label"] == "row_1"
    window.close()
