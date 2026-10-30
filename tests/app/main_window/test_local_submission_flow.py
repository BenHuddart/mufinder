"""Submitting jobs locally from the main window and handling local-runner events."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

import mufinder.app.main_window as main_window_module
from mufinder.castep_io.output_parser import CastepResult
from mufinder.castep_setup import CastepSetup
from mufinder.domain.enums import JobStatus
from mufinder.domain.models import CalculationDefinition, CalculationRun, Job, Structure
from mufinder.services.project_service import ProjectService


def test_main_window_submit_jobs_locally_creates_run_and_calls_runner(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "local_submit_ui.mufinder", "LocalSubmit")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    output_dir = doc.project_root / "files" / "convergence_cutoff"
    (output_dir / job.label).mkdir(parents=True, exist_ok=True)

    service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "sweep_values": [300.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=output_dir,
        job_ids_by_cutoff={main_window_module.MainWindow._cutoff_key(300.0): str(job.id)},
    )

    window._document = doc

    monkeypatch.setattr(
        main_window_module,
        "load_castep_setup",
        lambda _settings: CastepSetup(executable_path="/usr/bin/castep", mpirun_path=""),
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(num_cores=1, parallel=False, max_concurrent=1)

    monkeypatch.setattr(main_window_module, "LocalSubmissionDialog", _FakeDialog)

    submit_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        window._local_runner,
        "submit",
        lambda **kwargs: submit_calls.append(kwargs),
    )

    window._submit_jobs_locally([job])

    assert job.status == JobStatus.SUBMITTING
    assert len(submit_calls) == 1
    assert submit_calls[0]["seed_name"] == job.label
    assert submit_calls[0]["job_dir"] == output_dir / job.label
    assert job.num_processes == 1
    runs = [run for run in doc.runs.values() if run.job_id == job.id]
    assert len(runs) == 1
    assert doc.provenance[-1].action == "submit_jobs"
    assert doc.provenance[-1].parameters["job_labels"] == [job.label]

    window.close()


def test_main_window_submit_jobs_locally_uses_definition_output_for_non_convergence_stage(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "local_submit_generic.mufinder", "LocalSubmitGeneric")

    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    job = service.add_job(
        doc,
        label="muon_relax_site_1",
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type="castep_geom_opt",
    )
    output_root = doc.project_root / "files" / "muon_relaxation"
    (output_root / job.label).mkdir(parents=True, exist_ok=True)

    definition = CalculationDefinition(
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type="castep_geom_opt",
        parameters={
            "output_directory": str(output_root.relative_to(doc.project_root)),
            "job_labels": [job.label],
        },
        input_structure_ids=[host.id],
    )
    doc.calculations[definition.id] = definition

    window._document = doc

    monkeypatch.setattr(
        main_window_module,
        "load_castep_setup",
        lambda _settings: CastepSetup(executable_path="/usr/bin/castep", mpirun_path=""),
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(num_cores=1, parallel=False, max_concurrent=1)

    monkeypatch.setattr(main_window_module, "LocalSubmissionDialog", _FakeDialog)

    submit_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        window._local_runner, "submit", lambda **kwargs: submit_calls.append(kwargs)
    )

    window._submit_jobs_locally([job])

    assert len(submit_calls) == 1
    assert submit_calls[0]["job_dir"] == output_root / job.label
    window.close()


def test_main_window_submit_jobs_locally_resets_execution_target_after_cluster_reset(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "local_submit_resets_target.mufinder", "LocalSubmitResetsTarget"
    )

    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        execution_target="my-cluster",
    )
    output_dir = doc.project_root / "files" / "convergence_cutoff" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    job.status = JobStatus.FAILED
    job.remote_job_id = "12345"
    job.remote_directory = "/remote/conv_ecut_300"
    service.reset_failed_job(job)
    window._document = doc

    monkeypatch.setattr(
        main_window_module,
        "load_castep_setup",
        lambda _settings: CastepSetup(executable_path="/usr/bin/castep", mpirun_path=""),
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return self.DialogCode.Accepted

        def config(self):
            return SimpleNamespace(parallel=False, max_concurrent=1, num_cores=1)

    monkeypatch.setattr(main_window_module, "LocalSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(window._local_runner, "submit", lambda **kwargs: None)

    window._submit_jobs_locally([job])

    assert job.execution_target == "local"
    assert doc.provenance[-1].parameters["execution_target"] == "local"
    window.close()


def test_main_window_submit_jobs_locally_warns_if_castep_unconfigured(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, answered_choices
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "submit_warn_no_exec.mufinder", "WarnNoExec")
    job = service.add_job(
        doc,
        label="job_a",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    window._document = doc

    monkeypatch.setattr(
        main_window_module,
        "load_castep_setup",
        lambda _settings: CastepSetup(executable_path="", mpirun_path=""),
    )

    monkeypatch.setattr(
        main_window_module,
        "LocalSubmissionDialog",
        lambda **_kwargs: pytest.fail("No submission dialog without an executable"),
    )

    window._submit_jobs_locally([job])

    (prompt,) = answered_choices
    assert prompt["title"] == "CASTEP executable missing"
    assert prompt["choices"] == [("setup", "Open Setup > CASTEP…"), ("cancel", "Cancel")]
    assert (prompt["default"], prompt["escape"]) == ("cancel", "cancel")
    window.close()


def test_main_window_submit_from_dashboard_dispatches_local_and_cluster(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "submit_dashboard_mix.mufinder", "SubmitDashboardMix")

    local_job = service.add_job(
        doc,
        label="local_job",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        execution_target="local",
    )
    cluster_job = service.add_job(
        doc,
        label="cluster_job",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        execution_target="alpha",
    )
    window._document = doc

    local_calls: list[list[Job]] = []
    cluster_calls: list[tuple[list[Job], str | None]] = []
    monkeypatch.setattr(window, "_submit_jobs_locally", lambda jobs: local_calls.append(jobs))
    monkeypatch.setattr(
        window,
        "_submit_jobs_on_cluster",
        lambda jobs, default_cluster_name=None: cluster_calls.append((jobs, default_cluster_name)),
    )

    window._on_submit_from_dashboard([local_job.id, cluster_job.id])

    assert local_calls == [[local_job]]
    assert cluster_calls == [([cluster_job], "alpha")]
    window.close()


def test_main_window_submit_jobs_locally_warns_missing_output_paths(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "submit_missing_output.mufinder", "SubmitMissingOutput")
    job = service.add_job(
        doc,
        label="job_a",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    window._document = doc

    monkeypatch.setattr(
        main_window_module,
        "load_castep_setup",
        lambda _settings: CastepSetup(executable_path="/usr/bin/castep", mpirun_path=""),
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(num_cores=1, parallel=False, max_concurrent=1)

    monkeypatch.setattr(main_window_module, "LocalSubmissionDialog", _FakeDialog)

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    window._submit_jobs_locally([job])

    assert "Missing job directories" in warnings
    window.close()


def test_main_window_local_status_handlers_update_job_and_run(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "local_statuses.mufinder", "LocalStatuses")
    job = service.add_job(
        doc,
        label="job_state",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    run = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.SUBMITTING)
    doc.runs[run.id] = run
    window._document = doc
    window._active_run_by_job[job.id] = run.id
    job.status = JobStatus.SUBMITTING

    window._on_local_job_status_changed(job.id, JobStatus.QUEUED.value)
    assert job.status == JobStatus.QUEUED
    assert run.status == JobStatus.QUEUED

    window._on_local_job_started(job.id)
    assert job.status == JobStatus.RUNNING
    assert run.status == JobStatus.RUNNING
    assert run.started_at is not None

    window._on_local_job_failed(job.id, "boom")
    assert job.status == JobStatus.FAILED
    assert run.status == JobStatus.FAILED
    assert doc.provenance[-1].action == "job_failed"

    job.status = JobStatus.RUNNING
    run.status = JobStatus.RUNNING
    window._on_local_job_cancelled(job.id)
    assert job.status == JobStatus.CANCELLED
    assert run.status == JobStatus.CANCELLED
    assert doc.provenance[-1].action == "job_cancelled"
    window.close()


def test_main_window_local_finish_creates_result_and_completes_job(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "local_finish.mufinder", "LocalFinish")
    job = service.add_job(
        doc,
        label="job_finish",
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type="castep_geom_opt",
    )
    job.status = JobStatus.RUNNING
    run = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.RUNNING)
    doc.runs[run.id] = run
    window._document = doc
    window._active_run_by_job[job.id] = run.id

    output_dir = doc.project_root / "files" / "muon_relaxation" / job.label
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / f"{job.label}.cell").write_text("cell", encoding="utf-8")
    (output_dir / f"{job.label}.param").write_text("param", encoding="utf-8")
    (output_dir / f"{job.label}.castep").write_text("castep", encoding="utf-8")

    parsed = CastepResult(
        completed=True,
        total_energy=-10.5,
        warnings=("warning",),
        geom_converged=True,
    )
    window._on_local_job_finished(job.id, parsed)

    assert job.status == JobStatus.COMPLETED
    assert run.status == JobStatus.COMPLETED
    assert run.result_id is not None
    result = doc.results[run.result_id]
    assert result.data["total_energy"] == -10.5
    assert any(path.endswith(f"{job.label}.castep") for path in result.output_files)
    assert window._jobs_table.rowCount() == 0
    assert doc.provenance[-1].action == "job_completed"
    window.close()


def test_main_window_job_output_directory_fallback_by_stage(main_window, tmp_path: Path) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "output_fallback.mufinder", "OutputFallback")
    job = service.add_job(
        doc,
        label="job_path",
        workflow_stage=main_window_module.WorkflowStage.MUON_RELAXATION,
        calc_type="castep_geom_opt",
    )
    window._document = doc

    fallback = doc.project_root / "files" / "muon_relaxation" / job.label
    fallback.mkdir(parents=True, exist_ok=True)

    resolved = window._job_output_directory(job)

    assert resolved == fallback
    window.close()


def test_main_window_close_invokes_runner_shutdown(
    main_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    called: list[bool] = []
    monkeypatch.setattr(window._local_runner, "shutdown", lambda: called.append(True))

    window.close()

    assert called == [True]


def test_main_window_submit_jobs_locally_handles_no_eligible_defined_jobs(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "submit_no_defined.mufinder", "SubmitNoDefined")
    job = service.add_job(
        doc,
        label="job_done",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    job.status = JobStatus.COMPLETED
    window._document = doc

    infos: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    window._submit_jobs_locally([job])

    assert infos == ["No eligible jobs"]
    window.close()


def test_main_window_submit_jobs_locally_dialog_rejected(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "submit_dialog_reject.mufinder", "SubmitDialogReject")
    job = service.add_job(
        doc,
        label="job_a",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    (doc.project_root / "files" / "convergence_cutoff" / job.label).mkdir(
        parents=True, exist_ok=True
    )
    window._document = doc

    monkeypatch.setattr(
        main_window_module,
        "load_castep_setup",
        lambda _settings: CastepSetup(executable_path="/usr/bin/castep", mpirun_path=""),
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 0

    monkeypatch.setattr(main_window_module, "LocalSubmissionDialog", _FakeDialog)

    submit_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        window._local_runner, "submit", lambda **kwargs: submit_calls.append(kwargs)
    )

    window._submit_jobs_locally([job])

    assert submit_calls == []
    assert job.status == JobStatus.DEFINED
    window.close()
