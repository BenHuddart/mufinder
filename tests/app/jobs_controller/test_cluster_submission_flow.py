"""Submitting jobs to a cluster through the jobs controller, and the cluster connection around it."""

from __future__ import annotations

from mufinder.services.job_monitoring import job_status_word
import dataclasses
import threading
from pathlib import Path
from types import SimpleNamespace
from uuid import uuid4

import pytest
from PySide6.QtCore import QTimer
from PySide6.QtGui import QAction

import mufinder.app.main_window as main_window_module
import mufinder.app.jobs_controller as jobs_controller_module
from mufinder.app import workflow_table_utils
from mufinder.app.main_window import MainWindow
from mufinder.domain.cluster import ClusterConfig
from mufinder.domain.enums import EndReason, JobActivity, JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job, MuonSite, Result, Structure
from mufinder.services.job_execution_controller import LAST_CLUSTER_SUBMISSION_KEY
from mufinder.services.project_service import ProjectService
from mufinder.services.ssh_client import SSHCredentials
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE, QUANTUM_SINGLEPOINT_CALC_TYPE
from tests.support.main_window import (
    CONTINUATION_CLUSTER,
    add_cluster_input_job,
    patch_continuation_cluster_dialog,
    patch_continuation_cluster_runner,
)
from tests.support.qt import pump_until


def test_main_window_submit_jobs_on_cluster_creates_run_and_calls_runner(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_submit_ui.mufinder", "ClusterSubmit")

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

    cluster = ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="alpha",
                working_directory="/remote/work",
                username="alice",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(jobs_controller_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(
        jobs_controller_module, "load_cluster_config", lambda _settings, _name: cluster
    )
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)

    connect_calls: list[object] = []
    submit_calls: list[dict[str, object]] = []
    infos: list[str] = []
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "connect_to_cluster",
        lambda config, password=None: connect_calls.append((config.name, password)),
    )
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(
        window._cluster_runner._runner, "remote_directory_exists", lambda _path: False
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "submit",
        lambda **kwargs: submit_calls.append(kwargs) or "99123",
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    window._jobs.submit_jobs_on_cluster([job])

    assert connect_calls == [("alpha", None)]
    assert infos == ["Cluster connection successful"]
    assert len(submit_calls) == 1
    assert submit_calls[0]["seed_name"] == job.label
    assert job_status_word(job) == "queued"
    assert job.execution_target == "alpha"
    assert job.remote_job_id == "99123"
    assert job.remote_directory == f"/remote/work/{job.label}"
    assert job.num_processes == 16
    runs = [run for run in doc.runs.values() if run.job_id == job.id]
    assert len(runs) == 1
    assert runs[0].status == JobStatus.ACTIVE

    reloaded = service.load_project(doc.project_root)
    persisted_job = reloaded.jobs[job.id]
    assert job_status_word(persisted_job) == "queued"
    assert persisted_job.remote_job_id == "99123"
    assert persisted_job.remote_directory == f"/remote/work/{job.label}"
    persisted_runs = [run for run in reloaded.runs.values() if run.job_id == job.id]
    assert len(persisted_runs) == 1
    assert persisted_runs[0].status == JobStatus.ACTIVE
    assert reloaded.provenance[-1].parameters["jobs"] == [
        {
            "job_id": str(job.id),
            "label": job.label,
            "status": "queued",
            "execution_target": "alpha",
            "remote_job_id": "99123",
            "remote_directory": f"/remote/work/{job.label}",
            "num_processes": 16,
            "submitted_at": job.submitted_at.isoformat() if job.submitted_at else None,
        }
    ]
    window.close()


def test_main_window_submit_jobs_on_cluster_passes_jump_host_credentials(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_jump_submit_ui.mufinder", "ClusterJumpSubmit")

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

    cluster = ClusterConfig(
        name="Redwood",
        hostname="glamdring.physics.ox.ac.uk",
        username="glam-user",
        auth_method="password",
        default_working_directory="/mnt/redwoodfs/glam-user",
        queue_system="slurm",
        cores_per_node=24,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
        jump_host_enabled=True,
        jump_hostname="bastion.physics.ox.ac.uk",
        jump_username="physics-user",
        jump_auth_method="password",
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="Redwood",
                working_directory="/mnt/redwoodfs/glam-user",
                username="glam-user",
                num_processes=24,
                walltime="01:00:00",
            )

        def credentials(self) -> SSHCredentials:
            return SSHCredentials(target_secret="glam-secret", jump_secret="physics-secret")

    connect_calls: list[tuple[str, SSHCredentials | None, str | None]] = []
    submit_calls: list[dict[str, object]] = []
    monkeypatch.setattr(jobs_controller_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(
        jobs_controller_module, "load_cluster_config", lambda _settings, _name: cluster
    )
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "connect_to_cluster",
        lambda config, password=None, credentials=None: connect_calls.append(
            (config.name, credentials, password)
        ),
    )
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(
        window._cluster_runner._runner, "remote_directory_exists", lambda _path: False
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "submit",
        lambda **kwargs: submit_calls.append(kwargs) or "99123",
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox, "information", lambda *_args, **_kwargs: None
    )

    window._jobs.submit_jobs_on_cluster([job])

    assert len(connect_calls) == 1
    assert connect_calls[0][0] == "Redwood"
    assert connect_calls[0][1] == SSHCredentials(
        target_secret="glam-secret", jump_secret="physics-secret"
    )
    assert connect_calls[0][2] is None
    assert len(submit_calls) == 1
    window.close()


def test_main_window_resubmits_failed_cluster_job_through_its_row_with_rebuilt_inputs(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_resubmit_ui.mufinder", "ClusterResubmit")

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
        execution_target="alpha",
    )
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
    job.remote_job_id = "old99123"
    job.remote_directory = "/remote/work/conv_ecut_300"
    job.error_message = "previous scheduler failure"
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

    cluster = ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="alpha",
                working_directory="/remote/work",
                username="alice",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(jobs_controller_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(
        jobs_controller_module, "load_cluster_config", lambda _settings, _name: cluster
    )
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "connect_to_cluster",
        lambda _config, password=None: None,
    )
    # A sweep point has no checkpoint to continue from, so it starts again in
    # its own remote folder, which is deleted (with no overwrite prompt) once
    # the scheduler confirms the previous run has finished.
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda **_kwargs: pytest.fail("starting again reuses the job's own remote folder"),
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remote_directory_exists",
        lambda _path: pytest.fail("starting again needs no existence check"),
    )
    asked_scheduler: list[str] = []
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "scheduler_job_is_active",
        lambda remote_job_id: asked_scheduler.append(remote_job_id) or False,
    )
    deleted: list[str] = []
    monkeypatch.setattr(window._cluster_runner._runner, "remove_remote_directory", deleted.append)
    submit_calls: list[dict[str, object]] = []
    infos: list[str] = []
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "submit",
        lambda **kwargs: submit_calls.append(kwargs) or "new99123",
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    window._jobs._on_resubmit_from_dashboard([job.id])

    assert infos == ["Cluster connection successful"]
    # Asked before anything is cleared, and again just before the delete.
    assert asked_scheduler == ["old99123", "old99123"]
    assert deleted == ["/remote/work/conv_ecut_300"]
    assert len(submit_calls) == 1
    assert submit_calls[0]["remote_directory"] == "/remote/work/conv_ecut_300"
    # The inputs are rebuilt from the row and uploaded.
    assert (output_dir / job.label / f"{job.label}.param").is_file()
    assert job_status_word(job) == "queued"
    assert job.remote_job_id == "new99123"
    assert job.remote_directory == "/remote/work/conv_ecut_300"
    assert job.error_message is None
    runs = [run for run in doc.runs.values() if run.job_id == job.id]
    assert len(runs) == 1
    assert runs[0].status == JobStatus.ACTIVE
    window.close()


def test_main_window_resubmit_passes_updated_resources_to_runner(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "cluster_resubmit_resources.mufinder", "ClusterResubmit"
    )

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
        execution_target="alpha",
    )
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
    job.remote_job_id = "old99123"
    job.remote_directory = "/remote/work/conv_ecut_300"
    job.error_message = "previous scheduler failure"
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

    cluster = ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="alpha",
                working_directory="/remote/work",
                username="alice",
                num_processes=32,
                walltime="168:00:00",
                mem_per_cpu_gb=8.0,
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(jobs_controller_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(
        jobs_controller_module, "load_cluster_config", lambda _settings, _name: cluster
    )
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "connect_to_cluster",
        lambda _config, password=None: None,
    )
    # A sweep point has no checkpoint to continue from, so it starts again in
    # its own remote folder, which is deleted (with no overwrite prompt) once
    # the scheduler confirms the previous run has finished.
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda **_kwargs: pytest.fail("starting again reuses the job's own remote folder"),
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remote_directory_exists",
        lambda _path: pytest.fail("starting again needs no existence check"),
    )
    asked_scheduler: list[str] = []
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "scheduler_job_is_active",
        lambda remote_job_id: asked_scheduler.append(remote_job_id) or False,
    )
    deleted: list[str] = []
    monkeypatch.setattr(window._cluster_runner._runner, "remove_remote_directory", deleted.append)
    submit_calls: list[dict[str, object]] = []
    infos: list[str] = []
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "submit",
        lambda **kwargs: submit_calls.append(kwargs) or "new99123",
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    window._jobs._on_resubmit_from_dashboard([job.id])

    assert infos == ["Cluster connection successful"]
    assert len(submit_calls) == 1
    assert submit_calls[0]["walltime"] == "168:00:00"
    assert submit_calls[0]["mem_per_cpu_gb"] == 8.0
    assert submit_calls[0]["num_processes"] == 32
    assert submit_calls[0]["remote_directory"] == "/remote/work/conv_ecut_300"
    assert job.num_processes == 32
    assert job_status_word(job) == "queued"
    assert job.remote_job_id == "new99123"
    assert job.remote_directory == "/remote/work/conv_ecut_300"
    assert job.error_message is None
    runs = [run for run in doc.runs.values() if run.job_id == job.id]
    assert len(runs) == 1
    assert runs[0].status == JobStatus.ACTIVE
    window.close()


def test_main_window_submit_jobs_on_cluster_prefills_from_last_submission_defaults(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_prefill_last.mufinder", "ClusterPrefill")
    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    doc.ui_state[LAST_CLUSTER_SUBMISSION_KEY] = {
        "cluster_name": "Redwood",
        "working_directory": "/mnt/redwoodfs/alice/project_x",
        "username": "alice",
        # Legacy key: the retired ``Job name prefix`` field was saved here.
        "job_name_prefix": "kpoint",
        "num_processes": 32,
        "walltime": "",
        "queue_name": "redwood-priority",
        "memory_gb": 10.0,
        "comment": "batch-b",
    }
    window._document = doc

    alpha = ClusterConfig(
        name="SCARF",
        hostname="scarf.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/work/scarf/alice",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )
    redwood = ClusterConfig(
        name="Redwood",
        hostname="redwood.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/mnt/redwoodfs/alice",
        queue_system="addqueue",
        cores_per_node=32,
        default_walltime="",
        template_script_path="",
        default_queue_name="redwood",
        default_memory_gb=7.5,
    )

    captured_kwargs: dict[str, object] = {}

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **kwargs) -> None:
            captured_kwargs.update(kwargs)

        def exec(self) -> int:
            return 0

    monkeypatch.setattr(jobs_controller_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(
        main_window_module, "load_cluster_configs", lambda _settings: [alpha, redwood]
    )
    monkeypatch.setattr(
        jobs_controller_module, "load_cluster_configs", lambda _settings: [alpha, redwood]
    )

    window._jobs.submit_jobs_on_cluster([job])

    assert captured_kwargs["default_cluster_name"] == "Redwood"
    initial_config = captured_kwargs["initial_config"]
    assert initial_config is not None
    assert initial_config.cluster_name == "Redwood"
    assert initial_config.working_directory == "/mnt/redwoodfs/alice/project_x"
    assert initial_config.num_processes == 32
    assert initial_config.queue_name == "redwood-priority"
    assert initial_config.memory_gb == 10.0
    window.close()


def test_main_window_submit_jobs_on_cluster_prefills_same_workflow_without_reset(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "cluster_prefill_same_workflow.mufinder", "ClusterPrefillSame"
    )
    job = service.add_job(
        doc,
        label="conv_ecut_350",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    doc.ui_state[LAST_CLUSTER_SUBMISSION_KEY] = {
        "cluster_name": "Redwood",
        "working_directory": "/mnt/redwoodfs/alice/project_x/convergence_cutoff/custom",
        "username": "alice",
        "num_processes": 40,
        "walltime": "",
        "queue_name": "redwood-priority",
        "memory_gb": 10.0,
        "comment": "batch-b",
        "workflow_type": "convergence_cutoff",
    }
    window._document = doc

    redwood = ClusterConfig(
        name="Redwood",
        hostname="redwood.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/mnt/redwoodfs/alice",
        queue_system="addqueue",
        cores_per_node=32,
        default_walltime="",
        template_script_path="",
        default_queue_name="redwood",
        default_memory_gb=7.5,
    )

    captured_kwargs: dict[str, object] = {}

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **kwargs) -> None:
            captured_kwargs.update(kwargs)

        def exec(self) -> int:
            return 0

    monkeypatch.setattr(jobs_controller_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [redwood])
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [redwood])

    window._jobs.submit_jobs_on_cluster([job])

    initial_config = captured_kwargs["initial_config"]
    assert initial_config is not None
    assert (
        initial_config.working_directory
        == "/mnt/redwoodfs/alice/project_x/convergence_cutoff/custom"
    )
    assert initial_config.username == "alice"
    assert initial_config.num_processes == 40
    assert initial_config.queue_name == "redwood-priority"
    assert initial_config.memory_gb == 10.0
    assert captured_kwargs["workflow_type"] == "convergence_cutoff"
    window.close()


@pytest.mark.parametrize(
    ("label", "stage", "calc_type", "workflow_type"),
    [
        (
            "pristine_relax_01",
            "PRISTINE_RELAXATION",
            "castep_pristine_relaxation",
            "pristine_relaxation",
        ),
        ("uep_01", "UEP", "castep_uep_source", "uep"),
    ],
)
def test_main_window_submit_jobs_on_cluster_resets_workdir_on_workflow_switch(
    main_window,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    label: str,
    stage: str,
    calc_type: str,
    workflow_type: str,
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "cluster_prefill_workflow_switch.mufinder", "ClusterPrefillSwitch"
    )
    job = service.add_job(
        doc,
        label=label,
        workflow_stage=main_window_module.WorkflowStage[stage],
        calc_type=calc_type,
    )
    doc.ui_state[LAST_CLUSTER_SUBMISSION_KEY] = {
        "cluster_name": "Redwood",
        "working_directory": "/mnt/redwoodfs/alice/project_x/convergence_cutoff/custom",
        "username": "alice",
        "num_processes": 40,
        "walltime": "",
        "queue_name": "redwood-priority",
        "memory_gb": 10.0,
        "comment": "batch-b",
        "workflow_type": "convergence_cutoff",
    }
    window._document = doc

    redwood = ClusterConfig(
        name="Redwood",
        hostname="redwood.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/mnt/redwoodfs/alice",
        queue_system="addqueue",
        cores_per_node=32,
        default_walltime="",
        template_script_path="",
        default_queue_name="redwood",
        default_memory_gb=7.5,
    )

    captured_kwargs: dict[str, object] = {}

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **kwargs) -> None:
            captured_kwargs.update(kwargs)

        def exec(self) -> int:
            return 0

    monkeypatch.setattr(jobs_controller_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [redwood])
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [redwood])

    window._jobs.submit_jobs_on_cluster([job])

    initial_config = captured_kwargs["initial_config"]
    assert initial_config is not None
    assert initial_config.working_directory == ""
    assert initial_config.username == "alice"
    assert initial_config.num_processes == 40
    assert initial_config.queue_name == "redwood-priority"
    assert initial_config.memory_gb == 10.0
    assert captured_kwargs["workflow_type"] == workflow_type
    window.close()


def test_main_window_submit_jobs_on_cluster_persists_last_submission_defaults(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_persist_last.mufinder", "ClusterPersist")
    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    # Legacy key: the retired ``Job name prefix`` field was saved here.
    doc.ui_state[LAST_CLUSTER_SUBMISSION_KEY] = {
        "cluster_name": "Redwood",
        "job_name_prefix": "conv",
    }
    window._document = doc

    redwood = ClusterConfig(
        name="Redwood",
        hostname="redwood.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/mnt/redwoodfs/alice",
        queue_system="addqueue",
        cores_per_node=32,
        default_walltime="",
        template_script_path="",
        default_queue_name="redwood",
        default_memory_gb=7.5,
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="Redwood",
                working_directory="/mnt/redwoodfs/alice/project_y",
                username="alice",
                num_processes=24,
                walltime="",
                queue_name="redwood-priority",
                memory_gb=12.0,
                comment="night-run",
            )

        def password(self) -> str:
            return ""

    warnings: list[str] = []
    monkeypatch.setattr(jobs_controller_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [redwood])
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [redwood])
    monkeypatch.setattr(
        jobs_controller_module, "load_cluster_config", lambda _settings, _name: redwood
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "connect_to_cluster",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )
    monkeypatch.setattr(
        main_window_module,
        "show_detailed_warning",
        lambda _parent, title, _message, **_kwargs: warnings.append(title),
    )
    monkeypatch.setattr(
        jobs_controller_module,
        "show_detailed_warning",
        lambda _parent, title, _message, **_kwargs: warnings.append(title),
    )

    window._jobs.submit_jobs_on_cluster([job])

    assert warnings == ["Cluster connection failed"]
    persisted = doc.ui_state[LAST_CLUSTER_SUBMISSION_KEY]
    assert persisted["cluster_name"] == "Redwood"
    assert persisted["working_directory"] == "/mnt/redwoodfs/alice/project_y"
    assert persisted["username"] == "alice"
    assert "job_name_prefix" not in persisted
    assert persisted["num_processes"] == 24
    assert persisted["queue_name"] == "redwood-priority"
    assert persisted["memory_gb"] == 12.0
    assert persisted["comment"] == "night-run"
    assert persisted["workflow_type"] == "convergence_cutoff"
    window.close()


def test_main_window_submit_jobs_on_cluster_connection_failure_warns(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_connect_fail.mufinder", "ClusterConnectFail")
    job = service.add_job(
        doc,
        label="conv_ecut_300",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    window._document = doc

    cluster = ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="alpha",
                working_directory="/remote/work",
                username="alice",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(jobs_controller_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(
        jobs_controller_module, "load_cluster_config", lambda _settings, _name: cluster
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "connect_to_cluster",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("boom")),
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    window._jobs.submit_jobs_on_cluster([job])

    assert warnings == ["Cluster connection failed"]
    window.close()


def test_main_window_submit_jobs_on_cluster_blocks_unsafe_remote_labels(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_unsafe_label.mufinder", "ClusterUnsafeLabel")
    job = service.add_job(
        doc,
        label="bad label",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    window._document = doc

    cluster = ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="alpha",
                working_directory="/remote/work",
                username="alice",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(jobs_controller_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(
        jobs_controller_module, "load_cluster_config", lambda _settings, _name: cluster
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module,
        "show_detailed_warning",
        lambda _parent, title, _message, **_kwargs: warnings.append(title),
    )
    monkeypatch.setattr(
        jobs_controller_module,
        "show_detailed_warning",
        lambda _parent, title, _message, **_kwargs: warnings.append(title),
    )

    window._jobs.submit_jobs_on_cluster([job])

    assert warnings == ["Unsafe job labels"]
    window.close()


def test_main_window_submit_jobs_on_cluster_existing_remote_dir_yes_deletes(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "cluster_existing_dir_yes.mufinder", "ClusterExistingDirYes"
    )
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

    cluster = ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="alpha",
                working_directory="/remote/work",
                username="alice",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(jobs_controller_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(
        jobs_controller_module, "load_cluster_config", lambda _settings, _name: cluster
    )
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)

    monkeypatch.setattr(
        window._cluster_runner._runner, "connect_to_cluster", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(
        window._cluster_runner._runner, "remote_directory_exists", lambda _path: True
    )
    deleted: list[str] = []
    monkeypatch.setattr(
        window._cluster_runner._runner, "remove_remote_directory", lambda path: deleted.append(path)
    )
    monkeypatch.setattr(window._cluster_runner._runner, "submit", lambda **_kwargs: "99123")
    monkeypatch.setattr(window._jobs, "_prompt_remote_directory_overwrite", lambda *_args: "yes")
    monkeypatch.setattr(
        main_window_module.QMessageBox, "information", lambda *_args, **_kwargs: None
    )

    window._jobs.submit_jobs_on_cluster([job])

    assert deleted == ["/remote/work/conv_ecut_300"]
    window.close()


def test_main_window_submit_jobs_on_cluster_sends_a_quantum_phonon_job_to_its_own_directory(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "cluster_quantum_continuation.mufinder", "ClusterQuantumContinuation"
    )
    window._document = doc

    relaxed_structure = Structure(
        label="site relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    doc.structures[relaxed_structure.id] = relaxed_structure

    source_job = Job(
        label="mu-site",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    doc.jobs[source_job.id] = source_job
    run = CalculationRun(
        job_id=source_job.id, status=JobStatus.COMPLETED, convergence_achieved=True
    )
    result = Result(
        calculation_run_id=run.id,
        result_type=source_job.calc_type,
        data={"total_energy": -1.0},
        derived_structure_ids=[relaxed_structure.id],
    )
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result
    site = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run.id, total_energy_eV=-1.0)
    doc.sites[site.id] = site

    definition = window._quantum_corrections_controller.ensure_definition([str(site.id)])
    assert definition is not None
    definition.parameters["phonon_job_ids_by_label"] = {}
    definition.parameters["job_seed_names_by_label"] = {str(site.id): "quantum_siteA"}
    config = definition.parameters.setdefault("configuration", {})
    assert isinstance(config, dict)
    row_settings = config.setdefault("row_settings", {})
    assert isinstance(row_settings, dict)
    row_settings[str(site.id)] = {
        "parameter_values": {},
        "added_advanced": [],
        "pseudopotential_family": "norm_conserving",
        "num_farms": 4,
    }

    phonon_job = service.add_job(
        doc,
        label="quantum_phonon_siteA",
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=QUANTUM_PHONON_CALC_TYPE,
    )
    definition.parameters["phonon_job_ids_by_label"][str(site.id)] = str(phonon_job.id)
    definition.parameters["job_ids_by_label"] = definition.parameters["phonon_job_ids_by_label"]

    quantum_dir = doc.project_root / "files" / "quantum_corrections" / "quantum_siteA"
    quantum_dir.mkdir(parents=True, exist_ok=True)

    cluster = ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="alpha",
                working_directory="/remote/work",
                username="alice",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(jobs_controller_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(
        jobs_controller_module, "load_cluster_config", lambda _settings, _name: cluster
    )
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)
    monkeypatch.setattr(
        window._cluster_runner._runner, "connect_to_cluster", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )

    checked_paths: list[str] = []
    deleted: list[str] = []
    submit_calls: list[dict[str, object]] = []
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remote_directory_exists",
        lambda path: checked_paths.append(path) or False,
    )
    monkeypatch.setattr(
        window._cluster_runner._runner, "remove_remote_directory", lambda path: deleted.append(path)
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "submit",
        lambda **kwargs: submit_calls.append(kwargs) or "99123",
    )
    monkeypatch.setattr(
        window._jobs,
        "_prompt_remote_directory_overwrite",
        lambda *_args: pytest.fail("a fresh remote directory needs no overwrite prompt"),
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox, "information", lambda *_args, **_kwargs: None
    )

    window._jobs.submit_jobs_on_cluster([phonon_job])

    # One fresh job: its own directory is checked before upload, and task farming applies.
    assert checked_paths == ["/remote/work/quantum_siteA"]
    assert deleted == []
    assert len(submit_calls) == 1
    assert submit_calls[0]["remote_directory"] == "/remote/work/quantum_siteA"
    assert submit_calls[0]["task_farming_enabled"] is True
    assert submit_calls[0]["num_farms"] == 4
    assert phonon_job.remote_directory == "/remote/work/quantum_siteA"
    assert phonon_job.execution_target == "alpha"

    window.close()


def _continue_cluster_job(service: ProjectService, job: Job, remote_directory: str) -> None:
    """Take a job through a failed cluster run and back to DEFINED for continuation."""

    job.execution_target = "alpha"
    job.remote_directory = remote_directory
    job.remote_job_id = "55501"
    job.status = JobStatus.ENDED
    job.end_reason = EndReason.UNKNOWN
    service.prepare_job_continuation(job)
    assert job.status == JobStatus.DEFINED
    assert job.remote_directory == remote_directory


def test_main_window_submit_jobs_on_cluster_reuses_continued_job_remote_directory(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_continuation.mufinder", "ClusterContinuation")
    window._document = doc

    job = add_cluster_input_job(service, doc, "conv_ecut_300")
    _continue_cluster_job(service, job, "/remote/work/conv_ecut_300")

    patch_continuation_cluster_dialog(monkeypatch)
    recorded = patch_continuation_cluster_runner(monkeypatch, window, directory_exists=True)
    monkeypatch.setattr(
        window._jobs,
        "_prompt_remote_directory_overwrite",
        lambda *_args: pytest.fail(
            "a continued job must never prompt to overwrite its own directory"
        ),
    )

    window._jobs.submit_jobs_on_cluster([job])

    # The preserved directory is never probed, never offered for deletion, and
    # never deleted: it holds the .check file the continuation resumes from.
    assert recorded["checked"] == []
    assert recorded["deleted"] == []
    assert len(recorded["submitted"]) == 1
    submitted = recorded["submitted"][0]
    assert submitted["remote_directory"] == "/remote/work/conv_ecut_300"
    assert submitted["allow_existing_remote_directory"] is True
    # Fresh .cell/.param are uploaded; only the reuse flag would suppress that.
    assert job.remote_directory == "/remote/work/conv_ecut_300"
    assert job.execution_target == "alpha"

    window.close()


def test_main_window_submit_jobs_on_cluster_continuation_keeps_original_seed_name(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "cluster_continuation_seed.mufinder", "ClusterContinuationSeed"
    )
    window._document = doc

    job = add_cluster_input_job(service, doc, "conv_ecut_300")
    original_seed = window._job_seed_name(job)
    assert original_seed == "conv_ecut_300"
    _continue_cluster_job(service, job, "/remote/work/conv_ecut_300")

    # The remote basename now drives the seed, and must agree with the local one.
    assert window._job_seed_name(job) == original_seed

    patch_continuation_cluster_dialog(monkeypatch)
    recorded = patch_continuation_cluster_runner(monkeypatch, window, directory_exists=True)
    monkeypatch.setattr(
        window._jobs,
        "_prompt_remote_directory_overwrite",
        lambda *_args: pytest.fail(
            "a continued job must never prompt to overwrite its own directory"
        ),
    )

    window._jobs.submit_jobs_on_cluster([job])

    assert len(recorded["submitted"]) == 1
    submitted = recorded["submitted"][0]
    assert submitted["seed_name"] == original_seed
    # The runner uploads {seed}.cell/{seed}.param from this directory, so a seed
    # drift would fail the submission outright.
    local_dir = Path(submitted["local_job_dir"])
    assert (local_dir / f"{original_seed}.cell").is_file()
    assert (local_dir / f"{original_seed}.param").is_file()

    window.close()


def test_main_window_submit_jobs_on_cluster_mixed_batch_checks_only_the_new_job(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_mixed_batch.mufinder", "ClusterMixedBatch")
    window._document = doc

    continued = add_cluster_input_job(service, doc, "conv_ecut_300")
    _continue_cluster_job(service, continued, "/remote/work/conv_ecut_300")
    fresh = add_cluster_input_job(service, doc, "conv_ecut_400")
    assert fresh.status == JobStatus.DEFINED
    assert fresh.remote_directory is None

    patch_continuation_cluster_dialog(monkeypatch)
    recorded = patch_continuation_cluster_runner(monkeypatch, window, directory_exists=False)
    monkeypatch.setattr(
        window._jobs,
        "_prompt_remote_directory_overwrite",
        lambda *_args: pytest.fail("no directory exists, so nothing should prompt"),
    )

    window._jobs.submit_jobs_on_cluster([continued, fresh])

    # Exactly the fresh job's computed target is probed: the continued job is
    # excluded, the fresh job's existence check is unchanged.
    assert recorded["checked"] == ["/remote/work/conv_ecut_400"]
    assert recorded["deleted"] == []
    by_directory = {call["remote_directory"]: call for call in recorded["submitted"]}
    assert set(by_directory) == {"/remote/work/conv_ecut_300", "/remote/work/conv_ecut_400"}
    assert by_directory["/remote/work/conv_ecut_300"]["allow_existing_remote_directory"] is True
    assert by_directory["/remote/work/conv_ecut_400"]["allow_existing_remote_directory"] is False

    window.close()


def test_main_window_submit_jobs_on_cluster_does_not_submit_a_job_whose_directory_check_raised(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_check_error.mufinder", "ClusterCheckError")
    window._document = doc

    unchecked = add_cluster_input_job(service, doc, "conv_ecut_300")
    checked = add_cluster_input_job(service, doc, "conv_ecut_400")

    patch_continuation_cluster_dialog(monkeypatch)
    recorded = patch_continuation_cluster_runner(monkeypatch, window, directory_exists=False)

    def _directory_exists(path: str) -> bool:
        if path.endswith("conv_ecut_300"):
            raise OSError("SFTP stat failed")
        return False

    monkeypatch.setattr(
        window._cluster_runner._runner, "remote_directory_exists", _directory_exists
    )
    monkeypatch.setattr(
        window._jobs,
        "_prompt_remote_directory_overwrite",
        lambda *_args: pytest.fail("no directory is known to exist, so nothing should prompt"),
    )
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        main_window_module,
        "show_detailed_warning",
        lambda _parent, title, message, **_kwargs: warnings.append((title, message)),
    )
    monkeypatch.setattr(
        jobs_controller_module,
        "show_detailed_warning",
        lambda _parent, title, message, **_kwargs: warnings.append((title, message)),
    )

    window._jobs.submit_jobs_on_cluster([unchecked, checked])

    assert [call["remote_directory"] for call in recorded["submitted"]] == [
        "/remote/work/conv_ecut_400"
    ]
    assert unchecked.status == JobStatus.DEFINED
    assert unchecked.remote_directory is None
    assert job_status_word(checked) == "queued"
    assert warnings == [
        (
            "Cluster submission failed",
            "Some jobs failed to submit:\n- conv_ecut_300: not submitted because its remote directory "
            "could not be checked (SFTP stat failed)",
        )
    ]
    assert window.statusBar().currentMessage() == "Submitted 1 cluster job on alpha"

    window.close()


def test_main_window_submit_jobs_on_cluster_continuation_survives_target_collision(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A colliding batch job must not drag a preserved directory into the prompt.

    Overwrite decisions are matched by target string, so a second job whose
    computed target equals a continued job's preserved directory would otherwise
    let one prompt answer skip or delete the continuation.
    """

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_collision.mufinder", "ClusterCollision")
    window._document = doc

    continued = add_cluster_input_job(service, doc, "conv_ecut_300")
    collider = add_cluster_input_job(service, doc, "conv_ecut_400")
    # The continued job was originally submitted into the directory the collider
    # now computes for itself.
    _continue_cluster_job(service, continued, "/remote/work/conv_ecut_400")

    patch_continuation_cluster_dialog(monkeypatch)
    recorded = patch_continuation_cluster_runner(monkeypatch, window, directory_exists=True)
    prompted: list[str] = []
    monkeypatch.setattr(
        window._jobs,
        "_prompt_remote_directory_overwrite",
        lambda label, _target: prompted.append(label) or "no",
    )

    window._jobs.submit_jobs_on_cluster([continued, collider])

    assert prompted == ["conv_ecut_400"]
    assert recorded["deleted"] == []
    assert len(recorded["submitted"]) == 1
    assert recorded["submitted"][0]["seed_name"] == "conv_ecut_400"
    assert recorded["submitted"][0]["remote_directory"] == "/remote/work/conv_ecut_400"
    assert recorded["submitted"][0]["allow_existing_remote_directory"] is True
    assert job_status_word(continued) == "queued"
    assert collider.status == JobStatus.DEFINED

    window.close()


def test_main_window_submit_jobs_on_cluster_collision_overwrite_spares_continued_directory(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Answering "yes" for a colliding job must not delete a preserved directory.

    The overwrite decision belongs to the colliding job, but the deletion it
    authorises would land on the directory holding the continuation's .check file.
    """

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_collision_yes.mufinder", "ClusterCollisionYes")
    window._document = doc

    continued = add_cluster_input_job(service, doc, "conv_ecut_300")
    collider = add_cluster_input_job(service, doc, "conv_ecut_400")
    _continue_cluster_job(service, continued, "/remote/work/conv_ecut_400")

    patch_continuation_cluster_dialog(monkeypatch)
    recorded = patch_continuation_cluster_runner(monkeypatch, window, directory_exists=True)
    monkeypatch.setattr(window._jobs, "_prompt_remote_directory_overwrite", lambda *_args: "yes")

    window._jobs.submit_jobs_on_cluster([continued, collider])

    assert recorded["deleted"] == []
    assert len(recorded["submitted"]) == 2
    assert job_status_word(continued) == "queued"
    assert job_status_word(collider) == "queued"

    window.close()


def test_main_window_submit_jobs_on_cluster_yes_to_all_requires_confirmation(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(
        tmp_path / "cluster_yes_all_confirm.mufinder", "ClusterYesAllConfirm"
    )
    host = Structure(
        label="Host",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "Si"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
    )
    service.add_structure(doc, host)
    doc.project.host_structure_id = host.id

    jobs: list[Job] = []
    for cutoff in (300, 400):
        job = service.add_job(
            doc,
            label=f"conv_ecut_{cutoff}",
            workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
            calc_type="castep_convergence_cutoff",
        )
        jobs.append(job)
    output_dir = doc.project_root / "files" / "convergence_cutoff"
    for job in jobs:
        (output_dir / job.label).mkdir(parents=True, exist_ok=True)
    service.register_convergence_cutoff_definition(
        doc,
        host_structure_id=host.id,
        parameters={
            "name_prefix": "conv",
            "sweep_values": [300.0, 400.0],
            "shared_params": {"task": "SINGLEPOINT"},
            "overrides": [],
        },
        output_directory=output_dir,
        job_ids_by_cutoff={
            main_window_module.MainWindow._cutoff_key(300.0): str(jobs[0].id),
            main_window_module.MainWindow._cutoff_key(400.0): str(jobs[1].id),
        },
    )
    window._document = doc

    cluster = ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="alpha",
                working_directory="/remote/work",
                username="alice",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(jobs_controller_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(
        jobs_controller_module, "load_cluster_config", lambda _settings, _name: cluster
    )
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)

    monkeypatch.setattr(
        window._cluster_runner._runner, "connect_to_cluster", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(
        window._cluster_runner._runner, "remote_directory_exists", lambda _path: True
    )
    deleted: list[str] = []
    monkeypatch.setattr(
        window._cluster_runner._runner, "remove_remote_directory", lambda path: deleted.append(path)
    )
    monkeypatch.setattr(window._cluster_runner._runner, "submit", lambda **_kwargs: "99123")

    prompt_calls: list[str] = []
    confirm_calls: list[list[str]] = []
    monkeypatch.setattr(
        window._jobs,
        "_prompt_remote_directory_overwrite",
        lambda *_args: prompt_calls.append("prompt") or "yes_to_all",
    )
    monkeypatch.setattr(
        window._jobs,
        "_confirm_bulk_remote_delete",
        lambda paths: confirm_calls.append(list(paths)) or (len(paths) == 2),
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox, "information", lambda *_args, **_kwargs: None
    )

    window._jobs.submit_jobs_on_cluster(jobs)

    assert prompt_calls == ["prompt"]
    assert len(confirm_calls) == 1
    assert sorted(deleted) == ["/remote/work/conv_ecut_300", "/remote/work/conv_ecut_400"]
    window.close()


def test_main_window_connect_with_trust_handles_jump_and_target_host_keys(
    main_window,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = main_window
    config = ClusterConfig(
        name="Redwood",
        hostname="glamdring.physics.ox.ac.uk",
        username="glam-user",
        auth_method="password",
        jump_host_enabled=True,
        jump_hostname="bastion.physics.ox.ac.uk",
        jump_username="physics-user",
        jump_auth_method="password",
    )
    credentials = SSHCredentials(target_secret="glam", jump_secret="phys")
    prompts: list[str] = []
    connect_attempts: list[int] = []

    def _connect(_config, *, credentials=None, password=None):
        _ = credentials, password
        connect_attempts.append(len(connect_attempts) + 1)
        if len(connect_attempts) == 1:
            raise jobs_controller_module.SSHUnknownHostKeyError(
                hostname="bastion.physics.ox.ac.uk",
                port=22,
                algorithm="ssh-ed25519",
                fingerprint_sha256="SHA256:bastion",
                key_base64="YmFzdGlvbg==",
                endpoint="jump",
            )
        if len(connect_attempts) == 2:
            raise jobs_controller_module.SSHUnknownHostKeyError(
                hostname="glamdring.physics.ox.ac.uk",
                port=22,
                algorithm="ssh-ed25519",
                fingerprint_sha256="SHA256:glamdring",
                key_base64="Z2xhbWRyaW5n",
                endpoint="target",
            )

    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", _connect)
    monkeypatch.setattr(
        jobs_controller_module,
        "prompt_to_trust_host_key",
        lambda _parent, _config, exc: prompts.append(exc.endpoint) or True,
    )

    window._jobs._connect_to_cluster_with_trust(config, credentials=credentials)

    assert connect_attempts == [1, 2, 3]
    assert prompts == ["jump", "target"]
    assert (
        window._settings.value(
            "trusted_host_keys/Redwood/jump/bastion.physics.ox.ac.uk/algorithm",
            "",
            str,
        )
        == "ssh-ed25519"
    )
    assert (
        window._settings.value(
            "trusted_host_keys/Redwood/glamdring.physics.ox.ac.uk/algorithm",
            "",
            str,
        )
        == "ssh-ed25519"
    )
    window.close()


def test_main_window_remote_directory_prompt_mappings(
    main_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window

    sequence = iter(["yes", "yes_to_all", "no_to_all", "no"])
    prompts: list[tuple[list[tuple[str, str]], str, str]] = []

    def _choose(_parent, _title, _text, choices, *, default, escape, **_kwargs) -> str:
        prompts.append((list(choices), default, escape))
        return next(sequence)

    monkeypatch.setattr(workflow_table_utils, "confirm_choice", _choose)

    assert window._jobs._prompt_remote_directory_overwrite("job", "/remote/job") == "yes"
    assert window._jobs._prompt_remote_directory_overwrite("job", "/remote/job") == "yes_to_all"
    assert window._jobs._prompt_remote_directory_overwrite("job", "/remote/job") == "no_to_all"
    assert window._jobs._prompt_remote_directory_overwrite("job", "/remote/job") == "no"
    choices, default, escape = prompts[0]
    assert choices == [
        ("yes", "Delete directory"),
        ("yes_to_all", "Delete all"),
        ("no", "Keep directory"),
        ("no_to_all", "Keep all"),
    ]
    assert (default, escape) == ("no", "no")
    window.close()


def test_main_window_confirm_bulk_remote_delete_branches(
    main_window, monkeypatch: pytest.MonkeyPatch, confirmed_destructive_actions
) -> None:
    window = main_window

    assert window._jobs._confirm_bulk_remote_delete([]) is True
    assert confirmed_destructive_actions == []

    assert window._jobs._confirm_bulk_remote_delete(["/b", "/a", "/a"]) is True
    (prompt,) = confirmed_destructive_actions
    assert prompt["accept_label"] == "Delete 2 directories"
    assert prompt["detailed_text"] == "- /a\n- /b"

    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *_a, **_k: False)
    assert window._jobs._confirm_bulk_remote_delete(["/a"]) is False
    window.close()


def test_main_window_submit_jobs_on_cluster_no_to_all_skips_deletion(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_no_all_skip.mufinder", "ClusterNoAll")
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

    cluster = ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="alpha",
                working_directory="/remote/work",
                username="alice",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(jobs_controller_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(
        jobs_controller_module, "load_cluster_config", lambda _settings, _name: cluster
    )
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)

    monkeypatch.setattr(
        window._cluster_runner._runner, "connect_to_cluster", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(
        window._cluster_runner._runner, "remote_directory_exists", lambda _path: True
    )
    deleted: list[str] = []
    monkeypatch.setattr(
        window._cluster_runner._runner, "remove_remote_directory", lambda path: deleted.append(path)
    )
    submitted: list[dict[str, object]] = []
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "submit",
        lambda **kwargs: submitted.append(kwargs) or "99123",
    )
    monkeypatch.setattr(
        window._jobs, "_prompt_remote_directory_overwrite", lambda *_args: "no_to_all"
    )
    monkeypatch.setattr(
        main_window_module.QMessageBox, "information", lambda *_args, **_kwargs: None
    )

    window._jobs.submit_jobs_on_cluster([job])

    assert deleted == []
    assert len(submitted) == 0
    assert job.status == JobStatus.DEFINED
    window.close()


def test_main_window_submit_jobs_on_cluster_no_skips_submission(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_no_skip.mufinder", "ClusterNo")
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

    cluster = ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="alpha",
                working_directory="/remote/work",
                username="alice",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(jobs_controller_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(
        jobs_controller_module, "load_cluster_config", lambda _settings, _name: cluster
    )
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)

    monkeypatch.setattr(
        window._cluster_runner._runner, "connect_to_cluster", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(
        window._cluster_runner._runner, "remote_directory_exists", lambda _path: True
    )
    deleted: list[str] = []
    monkeypatch.setattr(
        window._cluster_runner._runner, "remove_remote_directory", lambda path: deleted.append(path)
    )
    submitted: list[dict[str, object]] = []
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "submit",
        lambda **kwargs: submitted.append(kwargs) or "99123",
    )
    monkeypatch.setattr(window._jobs, "_prompt_remote_directory_overwrite", lambda *_args: "no")
    monkeypatch.setattr(
        main_window_module.QMessageBox, "information", lambda *_args, **_kwargs: None
    )

    window._jobs.submit_jobs_on_cluster([job])

    assert deleted == []
    assert len(submitted) == 0
    assert job.status == JobStatus.DEFINED
    window.close()


def test_main_window_submit_jobs_on_cluster_reuses_connection_without_success_dialog(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_reuse_conn.mufinder", "ClusterReuse")
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

    cluster = ClusterConfig(
        name="alpha",
        hostname="alpha.example",
        username="alice",
        auth_method="key",
        key_path="/tmp/key",
        default_working_directory="/remote/work",
        queue_system="slurm",
        cores_per_node=64,
        default_walltime="01:00:00",
        template_script_path="/templates/slurm.sh",
    )

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="alpha",
                working_directory="/remote/work",
                username="alice",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(jobs_controller_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(
        jobs_controller_module, "load_cluster_config", lambda _settings, _name: cluster
    )
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)
    monkeypatch.setattr(
        type(window._cluster_runner),
        "is_connected",
        property(lambda _self: True),
    )
    monkeypatch.setattr(
        type(window._cluster_runner),
        "connected_cluster_name",
        property(lambda _self: "alpha"),
    )
    monkeypatch.setattr(
        type(window._cluster_runner),
        "connected_username",
        property(lambda _self: "alice"),
    )

    monkeypatch.setattr(
        window._cluster_runner._runner, "connect_to_cluster", lambda *_args, **_kwargs: None
    )
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(
        window._cluster_runner._runner, "remote_directory_exists", lambda _path: False
    )
    monkeypatch.setattr(window._cluster_runner._runner, "submit", lambda **_kwargs: "99123")

    infos: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    window._jobs.submit_jobs_on_cluster([job])

    assert "Cluster connection successful" not in infos
    window.close()


def _legacy_quantum_phonon_job_after_cluster_singlepoint(window: MainWindow, tmp_path: Path) -> Job:
    """Seed a defined phonon job whose row's legacy singlepoint ran on ``alpha``."""

    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_legacy_target.mufinder", "QuantumLegacyTarget")
    window._document = doc
    relaxed_structure = Structure(
        label="site relaxed",
        origin="muon_relaxation",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
        species=["Si", "H:mu"],
        fractional_positions=[[0.0, 0.0, 0.0], [0.25, 0.25, 0.25]],
        properties={"source_workflow_stage": WorkflowStage.MUON_RELAXATION.value},
    )
    doc.structures[relaxed_structure.id] = relaxed_structure
    source_job = Job(
        label="mu-site",
        workflow_stage=WorkflowStage.MUON_RELAXATION,
        calc_type="castep_positivemuon_relaxation",
        status=JobStatus.COMPLETED,
    )
    doc.jobs[source_job.id] = source_job
    run = CalculationRun(
        job_id=source_job.id, status=JobStatus.COMPLETED, convergence_achieved=True
    )
    result = Result(
        calculation_run_id=run.id,
        result_type=source_job.calc_type,
        data={"total_energy": -1.0},
        derived_structure_ids=[relaxed_structure.id],
    )
    run.result_id = result.id
    doc.runs[run.id] = run
    doc.results[result.id] = result
    site = MuonSite(label="siteA", status="relaxed", relaxation_run_id=run.id, total_energy_eV=-1.0)
    doc.sites[site.id] = site

    definition = window._quantum_corrections_controller.ensure_definition([str(site.id)])
    assert definition is not None
    definition.parameters["job_seed_names_by_label"] = {str(site.id): "quantum_siteA"}
    singlepoint_job = service.add_job(
        doc,
        label="quantum_singlepoint_siteA",
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=QUANTUM_SINGLEPOINT_CALC_TYPE,
    )
    singlepoint_job.status = JobStatus.COMPLETED
    singlepoint_job.execution_target = "alpha"
    singlepoint_job.remote_directory = "/remote/work/quantum_siteA"
    definition.parameters["singlepoint_job_ids_by_label"] = {str(site.id): str(singlepoint_job.id)}
    phonon_job = service.add_job(
        doc,
        label="quantum_phonon_siteA",
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=QUANTUM_PHONON_CALC_TYPE,
    )
    definition.parameters["phonon_job_ids_by_label"] = {str(site.id): str(phonon_job.id)}
    definition.parameters["job_ids_by_label"] = definition.parameters["phonon_job_ids_by_label"]
    (doc.project_root / "files" / "quantum_corrections" / "quantum_siteA").mkdir(
        parents=True, exist_ok=True
    )
    return phonon_job


def test_main_window_cluster_submission_sends_a_legacy_phonon_job_to_any_cluster(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A phonon job is one fresh CASTEP run, so where a legacy singlepoint ran does not matter."""

    window = main_window
    phonon_job = _legacy_quantum_phonon_job_after_cluster_singlepoint(window, tmp_path)
    beta = dataclasses.replace(CONTINUATION_CLUSTER, name="beta", hostname="beta.example")

    class _FakeDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, **_kwargs) -> None:
            pass

        def exec(self) -> int:
            return 1

        def config(self):
            return SimpleNamespace(
                cluster_name="beta",
                working_directory="/remote/work",
                username="alice",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    infos: list[str] = []
    monkeypatch.setattr(jobs_controller_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [beta])
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [beta])
    monkeypatch.setattr(
        jobs_controller_module, "load_cluster_config", lambda _settings, _name: beta
    )
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)
    recorded = patch_continuation_cluster_runner(monkeypatch, window, directory_exists=False)
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, _title, text: infos.append(text),
    )

    window._jobs.submit_jobs_on_cluster([phonon_job])

    assert not any("stage" in text.lower() for text in infos)
    assert recorded["checked"] == ["/remote/work/quantum_siteA"]
    assert [call["remote_directory"] for call in recorded["submitted"]] == [
        "/remote/work/quantum_siteA"
    ]
    assert phonon_job.execution_target == "beta"
    window.close()


def test_main_window_local_submission_runs_a_legacy_phonon_job_after_a_cluster_singlepoint(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The local backend refuses no phonon job by target: it goes straight on to CASTEP setup."""

    window = main_window
    phonon_job = _legacy_quantum_phonon_job_after_cluster_singlepoint(window, tmp_path)
    setups: list[object] = []
    infos: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, _title, text: infos.append(text),
    )
    monkeypatch.setattr(main_window_module.QMessageBox, "warning", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        jobs_controller_module,
        "load_castep_setup",
        lambda settings: (
            setups.append(settings) or SimpleNamespace(executable_path="", mpirun_path="")
        ),
    )

    window._jobs.submit_jobs_locally([phonon_job])

    assert len(setups) == 1
    assert infos == []
    window.close()


def _queued_cluster_job(window: MainWindow, tmp_path: Path) -> Job:
    service = window._project_service
    doc = service.create_project(tmp_path / "cluster_recovery.mufinder", "ClusterRecovery")
    job = service.add_job(
        doc,
        label="long_queue_job",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        execution_target="alpha",
    )
    service.submit_job(job)
    service.mark_job_active(job, JobActivity.QUEUED, remote_job_id="4242")
    job.remote_directory = "/remote/work/long_queue_job"
    run = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.ACTIVE)
    doc.runs[run.id] = run
    window._document = doc
    window._job_execution.active_run_by_job[job.id] = run.id
    return job


def test_main_window_batch_timeout_stops_the_worker_submitting(
    main_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    requests: list[str] = []
    # The worker never reports progress, so the inactivity watchdog fires.
    monkeypatch.setattr(window._cluster_runner, "submit_batch", lambda _items: 7)
    monkeypatch.setattr(
        window._cluster_runner, "abandon_batch", lambda batch: requests.append(("abandon", batch))
    )
    monkeypatch.setattr(
        window._cluster_runner, "cancel_batch", lambda batch: requests.append(("cancel", batch))
    )
    _use_a_fast_submission_watchdog(monkeypatch)

    submitted, failed = window._jobs._run_cluster_submit_batch([{"job_id": None}], {})

    assert submitted == []
    # The watchdog gives the batch up before the unseen jobs are cancelled;
    # closing the progress dialog then asks the worker to stop as Cancel does.
    assert requests == [("abandon", 7), ("cancel", 7)]
    assert failed == [
        "- Cluster submission stopped responding after 3 minutes; 1 pending job was cancelled."
    ]
    window.close()


def test_main_window_batch_dialog_ignores_the_signals_of_another_batch(
    main_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    # A batch given up on earlier still drains on the worker; its result
    # and its finish must not reach the dialog of the batch submitted after it.
    window = main_window
    runner = window._cluster_runner
    job_id = uuid4()
    prepared = SimpleNamespace(job=SimpleNamespace(id=job_id))
    finalized: list[str] = []
    cancelled: list[object] = []
    monkeypatch.setattr(
        window._job_execution,
        "finalize_cluster_submission",
        lambda _prepared, remote: finalized.append(remote),
    )
    monkeypatch.setattr(
        window._job_execution,
        "fail_cluster_submission",
        lambda _prepared, error: SimpleNamespace(
            submitted_job=None, failed_message=f"- {error}", journal_changed=False
        ),
    )
    monkeypatch.setattr(window._job_execution, "cancel_job", cancelled.append)

    def submit_batch(_items) -> int:
        def earlier_batch() -> None:
            runner.submit_batch_progress.emit(1, 1, 1, "stale")
            runner.job_submit_finished.emit(1, job_id, "999", "")
            runner.submit_batch_finished.emit(1)

        def own_batch() -> None:
            runner.job_submit_finished.emit(2, job_id, None, "own result")
            runner.submit_batch_finished.emit(2)

        QTimer.singleShot(0, earlier_batch)
        QTimer.singleShot(50, own_batch)
        return 2

    monkeypatch.setattr(runner, "submit_batch", submit_batch)

    submitted, failed = window._jobs._run_cluster_submit_batch(
        [{"job_id": job_id}], {job_id: prepared}
    )

    assert submitted == []
    assert finalized == []
    assert failed == ["- own result"]
    assert cancelled == []
    window.close()


def _use_a_fast_submission_watchdog(monkeypatch: pytest.MonkeyPatch) -> None:
    class _FastTimer(main_window_module.QTimer):
        def setInterval(self, _msec: int) -> None:
            super().setInterval(10)

    monkeypatch.setattr(main_window_module, "QTimer", _FastTimer)
    monkeypatch.setattr(jobs_controller_module, "QTimer", _FastTimer)


def test_main_window_cancels_a_job_submitted_after_the_batch_timed_out(
    main_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    runner = window._cluster_runner._runner
    scheduler_cancels: list[str] = []
    runner._ssh = SimpleNamespace(is_connected=True)
    runner._scheduler = SimpleNamespace(
        cancel=lambda _ssh, remote_job_id: scheduler_cancels.append(remote_job_id)
    )
    gate = threading.Event()

    def stuck_submit(**_kwargs) -> str:
        gate.wait(5.0)
        return "4242"

    monkeypatch.setattr(runner, "submit", stuck_submit)
    _use_a_fast_submission_watchdog(monkeypatch)
    job_id = uuid4()

    # The submit outlasts the dialog's wait, then succeeds.
    _submitted, failed = window._jobs._run_cluster_submit_batch(
        [{"job_id": job_id, "label": "late", "submit_kwargs": {}}], {}
    )
    gate.set()

    assert failed[-1].startswith("- Cluster submission stopped responding")
    assert pump_until(lambda: scheduler_cancels == ["4242"])
    window.close()


def test_main_window_warns_when_a_late_submitted_job_cannot_be_cancelled(
    main_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        main_window_module,
        "show_detailed_warning",
        lambda _parent, title, message, **_kwargs: warnings.append((title, message)),
    )
    monkeypatch.setattr(
        jobs_controller_module,
        "show_detailed_warning",
        lambda _parent, title, message, **_kwargs: warnings.append((title, message)),
    )

    window._cluster_runner.abandoned_job_cancel_failed.emit(uuid4(), "4242", "scancel failed")

    assert len(warnings) == 1
    title, message = warnings[0]
    assert title == "Cluster cancellation failed"
    assert "scheduler job 4242" in message
    assert "scancel failed" in message
    window.close()


def test_main_window_missing_username_opens_the_cluster_profile(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The submission dialog's offer reaches the setup dialog, on the right cluster."""

    window = main_window
    window._document = ProjectService().create_project(tmp_path / "profile.mufinder", "Profile")
    cluster = ClusterConfig(name="SCARF", hostname="ui1.scarf.rl.ac.uk", username="alice")
    opened: list[str | None] = []

    class _FakeSetupDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, *, settings, parent, cluster_name) -> None:  # noqa: ANN001
            opened.append(cluster_name)

        def exec(self) -> int:
            return 1

    edit_callbacks: list[object] = []

    class _FakeSubmissionDialog:
        DialogCode = SimpleNamespace(Accepted=1)

        def __init__(self, *, edit_cluster, **_kwargs) -> None:  # noqa: ANN001
            edit_callbacks.append(edit_cluster)

        def exec(self) -> int:
            return 0

    monkeypatch.setattr(main_window_module, "ClusterSetupDialog", _FakeSetupDialog)
    monkeypatch.setattr(jobs_controller_module, "ClusterSubmissionDialog", _FakeSubmissionDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(jobs_controller_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(window._keyword_catalog_guard, "confirm_submission", lambda: True)

    assert window._jobs._choose_cluster_submission([], "SCARF") is None

    (edit_cluster,) = edit_callbacks
    assert edit_cluster(window, "SCARF") == [cluster]
    assert opened == ["SCARF"]

    # The Setup menu opens the dialog with no cluster chosen.
    setup_action = next(
        action for action in window.findChildren(QAction) if action.text() == "HPC Clusters…"
    )
    setup_action.trigger()
    assert opened == ["SCARF", None]
    window.close()
