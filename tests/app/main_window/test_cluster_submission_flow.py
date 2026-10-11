"""Submitting jobs to a cluster from the main window, and the cluster connection around it."""

from __future__ import annotations

import dataclasses
from pathlib import Path
from types import SimpleNamespace

import pytest

import mufinder.app.main_window as main_window_module
from mufinder.app.cluster_submission_dialog import ClusterSubmissionConfig
from mufinder.app.main_window import MainWindow
from mufinder.domain.cluster import ClusterConfig
from mufinder.domain.enums import JobStatus, WorkflowStage
from mufinder.domain.models import CalculationRun, Job, MuonSite, Result, Structure
from mufinder.services.project_service import ProjectService
from mufinder.services.ssh_client import SSHCredentials
from mufinder.workflow.quantum import QUANTUM_PHONON_CALC_TYPE, QUANTUM_SINGLEPOINT_CALC_TYPE
from tests.support.main_window import (
    CONTINUATION_CLUSTER,
    add_cluster_input_job,
    patch_continuation_cluster_dialog,
    patch_continuation_cluster_runner,
)


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
                job_name_prefix="conv",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)
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
    monkeypatch.setattr(window._cluster_runner._runner, "remote_directory_exists", lambda _path: False)
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

    window._submit_jobs_on_cluster([job])

    assert connect_calls == [("alpha", None)]
    assert infos == ["Cluster connection successful"]
    assert len(submit_calls) == 1
    assert submit_calls[0]["seed_name"] == job.label
    assert job.status == JobStatus.QUEUED
    assert job.execution_target == "alpha"
    assert job.remote_job_id == "99123"
    assert job.remote_directory == f"/remote/work/{job.label}"
    assert job.num_processes == 16
    runs = [run for run in doc.runs.values() if run.job_id == job.id]
    assert len(runs) == 1
    assert runs[0].status == JobStatus.QUEUED

    reloaded = service.load_project(doc.project_root)
    persisted_job = reloaded.jobs[job.id]
    assert persisted_job.status == JobStatus.QUEUED
    assert persisted_job.remote_job_id == "99123"
    assert persisted_job.remote_directory == f"/remote/work/{job.label}"
    persisted_runs = [run for run in reloaded.runs.values() if run.job_id == job.id]
    assert len(persisted_runs) == 1
    assert persisted_runs[0].status == JobStatus.QUEUED
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
                job_name_prefix="conv",
                num_processes=24,
                walltime="01:00:00",
            )

        def credentials(self) -> SSHCredentials:
            return SSHCredentials(target_secret="glam-secret", jump_secret="physics-secret")

    connect_calls: list[tuple[str, SSHCredentials | None, str | None]] = []
    submit_calls: list[dict[str, object]] = []
    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "connect_to_cluster",
        lambda config, password=None, credentials=None: connect_calls.append((config.name, credentials, password)),
    )
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(window._cluster_runner._runner, "remote_directory_exists", lambda _path: False)
    monkeypatch.setattr(window._cluster_runner._runner, "submit", lambda **kwargs: submit_calls.append(kwargs) or "99123")
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *_args, **_kwargs: None)

    window._submit_jobs_on_cluster([job])

    assert len(connect_calls) == 1
    assert connect_calls[0][0] == "Redwood"
    assert connect_calls[0][1] == SSHCredentials(target_secret="glam-secret", jump_secret="physics-secret")
    assert connect_calls[0][2] is None
    assert len(submit_calls) == 1
    window.close()


def test_main_window_resubmits_failed_cluster_job_without_replacing_remote_files(
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
    job.status = JobStatus.FAILED
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
                job_name_prefix="conv",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "connect_to_cluster",
        lambda _config, password=None: None,
    )
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda **_kwargs: pytest.fail("resubmission should reuse the stored remote directory"),
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remote_directory_exists",
        lambda _path: pytest.fail("resubmission should not check for overwrite"),
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remove_remote_directory",
        lambda _path: pytest.fail("resubmission should not delete remote files"),
    )
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

    window._on_resubmit_from_dashboard([job.id])

    assert infos == ["Cluster connection successful"]
    assert len(submit_calls) == 1
    assert submit_calls[0]["remote_directory"] == "/remote/work/conv_ecut_300"
    assert submit_calls[0]["reuse_existing_files"] is True
    assert job.status == JobStatus.QUEUED
    assert job.remote_job_id == "new99123"
    assert job.remote_directory == "/remote/work/conv_ecut_300"
    assert job.error_message is None
    runs = [run for run in doc.runs.values() if run.job_id == job.id]
    assert len(runs) == 1
    assert runs[0].status == JobStatus.QUEUED
    window.close()


def test_main_window_resubmit_passes_updated_resources_to_runner(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_resubmit_resources.mufinder", "ClusterResubmit")

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
    job.status = JobStatus.FAILED
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
                job_name_prefix="conv",
                num_processes=32,
                walltime="168:00:00",
                mem_per_cpu_gb=8.0,
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "connect_to_cluster",
        lambda _config, password=None: None,
    )
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda **_kwargs: pytest.fail("resubmission should reuse the stored remote directory"),
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remote_directory_exists",
        lambda _path: pytest.fail("resubmission should not check for overwrite"),
    )
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "remove_remote_directory",
        lambda _path: pytest.fail("resubmission should not delete remote files"),
    )
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

    window._on_resubmit_from_dashboard([job.id])

    assert infos == ["Cluster connection successful"]
    assert len(submit_calls) == 1
    assert submit_calls[0]["walltime"] == "168:00:00"
    assert submit_calls[0]["mem_per_cpu_gb"] == 8.0
    assert submit_calls[0]["num_processes"] == 32
    assert submit_calls[0]["reuse_existing_files"] is True
    assert submit_calls[0]["remote_directory"] == "/remote/work/conv_ecut_300"
    assert job.num_processes == 32
    assert job.status == JobStatus.QUEUED
    assert job.remote_job_id == "new99123"
    assert job.remote_directory == "/remote/work/conv_ecut_300"
    assert job.error_message is None
    runs = [run for run in doc.runs.values() if run.job_id == job.id]
    assert len(runs) == 1
    assert runs[0].status == JobStatus.QUEUED
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
    doc.ui_state[main_window_module.MainWindow._LAST_CLUSTER_SUBMISSION_CONFIG_KEY] = {
        "cluster_name": "Redwood",
        "working_directory": "/mnt/redwoodfs/alice/project_x",
        "username": "alice",
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

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [alpha, redwood])

    window._submit_jobs_on_cluster([job])

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
    doc = service.create_project(tmp_path / "cluster_prefill_same_workflow.mufinder", "ClusterPrefillSame")
    job = service.add_job(
        doc,
        label="conv_ecut_350",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
    )
    doc.ui_state[main_window_module.MainWindow._LAST_CLUSTER_SUBMISSION_CONFIG_KEY] = {
        "cluster_name": "Redwood",
        "working_directory": "/mnt/redwoodfs/alice/project_x/convergence_cutoff/custom",
        "username": "alice",
        "job_name_prefix": "cutoff_scan",
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

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [redwood])

    window._submit_jobs_on_cluster([job])

    initial_config = captured_kwargs["initial_config"]
    assert initial_config is not None
    assert initial_config.working_directory == "/mnt/redwoodfs/alice/project_x/convergence_cutoff/custom"
    assert initial_config.job_name_prefix == "cutoff_scan"
    assert initial_config.username == "alice"
    assert initial_config.num_processes == 40
    assert initial_config.queue_name == "redwood-priority"
    assert initial_config.memory_gb == 10.0
    assert captured_kwargs["workflow_type"] == "convergence_cutoff"
    window.close()


def test_main_window_submit_jobs_on_cluster_resets_workdir_and_prefix_on_workflow_switch(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_prefill_workflow_switch.mufinder", "ClusterPrefillSwitch")
    job = service.add_job(
        doc,
        label="pristine_relax_01",
        workflow_stage=main_window_module.WorkflowStage.PRISTINE_RELAXATION,
        calc_type="castep_pristine_relaxation",
    )
    doc.ui_state[main_window_module.MainWindow._LAST_CLUSTER_SUBMISSION_CONFIG_KEY] = {
        "cluster_name": "Redwood",
        "working_directory": "/mnt/redwoodfs/alice/project_x/convergence_cutoff/custom",
        "username": "alice",
        "job_name_prefix": "cutoff_scan",
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

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [redwood])

    window._submit_jobs_on_cluster([job])

    initial_config = captured_kwargs["initial_config"]
    assert initial_config is not None
    assert initial_config.working_directory == ""
    assert initial_config.job_name_prefix == "pristine_relax"
    assert initial_config.username == "alice"
    assert initial_config.num_processes == 40
    assert initial_config.queue_name == "redwood-priority"
    assert initial_config.memory_gb == 10.0
    assert captured_kwargs["workflow_type"] == "pristine_relaxation"
    window.close()


def test_main_window_submit_uep_jobs_on_cluster_resets_stale_prefix_on_workflow_switch(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_prefill_uep_switch.mufinder", "ClusterPrefillUepSwitch")
    job = service.add_job(
        doc,
        label="uep_01",
        workflow_stage=main_window_module.WorkflowStage.UEP,
        calc_type="castep_uep_source",
    )
    doc.ui_state[main_window_module.MainWindow._LAST_CLUSTER_SUBMISSION_CONFIG_KEY] = {
        "cluster_name": "Redwood",
        "working_directory": "/mnt/redwoodfs/alice/project_x/convergence_cutoff/custom",
        "username": "alice",
        "job_name_prefix": "cutoff_scan",
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

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [redwood])

    window._submit_jobs_on_cluster([job])

    initial_config = captured_kwargs["initial_config"]
    assert initial_config is not None
    assert initial_config.working_directory == ""
    assert initial_config.job_name_prefix == "uep"
    assert initial_config.username == "alice"
    assert initial_config.num_processes == 40
    assert initial_config.queue_name == "redwood-priority"
    assert initial_config.memory_gb == 10.0
    assert captured_kwargs["workflow_type"] == "uep"
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
                job_name_prefix="pristine_relax",
                num_processes=24,
                walltime="",
                queue_name="redwood-priority",
                memory_gb=12.0,
                comment="night-run",
            )

        def password(self) -> str:
            return ""

    warnings: list[str] = []
    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [redwood])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: redwood)
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

    window._submit_jobs_on_cluster([job])

    assert warnings == ["Cluster connection failed"]
    persisted = doc.ui_state[main_window_module.MainWindow._LAST_CLUSTER_SUBMISSION_CONFIG_KEY]
    assert persisted["cluster_name"] == "Redwood"
    assert persisted["working_directory"] == "/mnt/redwoodfs/alice/project_y"
    assert persisted["username"] == "alice"
    assert persisted["job_name_prefix"] == "pristine_relax"
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
                job_name_prefix="conv",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)
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

    window._submit_jobs_on_cluster([job])

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
                job_name_prefix="conv",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)

    warnings: list[str] = []
    monkeypatch.setattr(
        main_window_module,
        "show_detailed_warning",
        lambda _parent, title, _message, **_kwargs: warnings.append(title),
    )

    window._submit_jobs_on_cluster([job])

    assert warnings == ["Unsafe job labels"]
    window.close()


def test_main_window_submit_jobs_on_cluster_existing_remote_dir_yes_deletes(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_existing_dir_yes.mufinder", "ClusterExistingDirYes")
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
                job_name_prefix="conv",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)

    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(window._cluster_runner._runner, "remote_directory_exists", lambda _path: True)
    deleted: list[str] = []
    monkeypatch.setattr(window._cluster_runner._runner, "remove_remote_directory", lambda path: deleted.append(path))
    monkeypatch.setattr(window._cluster_runner._runner, "submit", lambda **_kwargs: "99123")
    monkeypatch.setattr(window, "_prompt_remote_directory_overwrite", lambda *_args: "yes")
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *_args, **_kwargs: None)

    window._submit_jobs_on_cluster([job])

    assert deleted == ["/remote/work/conv_ecut_300"]
    window.close()


def test_main_window_submit_jobs_on_cluster_reuses_quantum_continuation_directory(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_quantum_continuation.mufinder", "ClusterQuantumContinuation")
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
    run = CalculationRun(job_id=source_job.id, status=JobStatus.COMPLETED, convergence_achieved=True)
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
    definition.parameters["singlepoint_job_ids_by_label"] = {}
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

    singlepoint_job = service.add_job(
        doc,
        label="quantum_singlepoint_siteA",
        workflow_stage=WorkflowStage.QUANTUM_CORRECTIONS,
        calc_type=QUANTUM_SINGLEPOINT_CALC_TYPE,
    )
    singlepoint_job.status = JobStatus.COMPLETED
    singlepoint_job.execution_target = "alpha"
    singlepoint_job.remote_directory = "/remote/work/quantum_siteA"
    definition.parameters["singlepoint_job_ids_by_label"][str(site.id)] = str(singlepoint_job.id)

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
                job_name_prefix="quantum",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)
    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", lambda *_args, **_kwargs: None)
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
    monkeypatch.setattr(window._cluster_runner._runner, "remove_remote_directory", lambda path: deleted.append(path))
    monkeypatch.setattr(window._cluster_runner._runner, "submit", lambda **kwargs: submit_calls.append(kwargs) or "99123")
    monkeypatch.setattr(
        window,
        "_prompt_remote_directory_overwrite",
        lambda *_args: pytest.fail("continuation submissions should not prompt to delete the remote directory"),
    )
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *_args, **_kwargs: None)

    window._submit_jobs_on_cluster([phonon_job])

    assert checked_paths == []
    assert deleted == []
    assert len(submit_calls) == 1
    assert submit_calls[0]["remote_directory"] == "/remote/work/quantum_siteA"
    assert submit_calls[0]["reuse_existing_files"] is False
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
    job.status = JobStatus.FAILED
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
        window,
        "_prompt_remote_directory_overwrite",
        lambda *_args: pytest.fail("a continued job must never prompt to overwrite its own directory"),
    )

    window._submit_jobs_on_cluster([job])

    # The preserved directory is never probed, never offered for deletion, and
    # never deleted: it holds the .check file the continuation resumes from.
    assert recorded["checked"] == []
    assert recorded["deleted"] == []
    assert len(recorded["submitted"]) == 1
    submitted = recorded["submitted"][0]
    assert submitted["remote_directory"] == "/remote/work/conv_ecut_300"
    assert submitted["allow_existing_remote_directory"] is True
    # Fresh .cell/.param are uploaded; only the reuse flag would suppress that.
    assert submitted["reuse_existing_files"] is False
    assert job.remote_directory == "/remote/work/conv_ecut_300"
    assert job.execution_target == "alpha"

    window.close()


def test_main_window_submit_jobs_on_cluster_continuation_keeps_original_seed_name(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_continuation_seed.mufinder", "ClusterContinuationSeed")
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
        window,
        "_prompt_remote_directory_overwrite",
        lambda *_args: pytest.fail("a continued job must never prompt to overwrite its own directory"),
    )

    window._submit_jobs_on_cluster([job])

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
        window,
        "_prompt_remote_directory_overwrite",
        lambda *_args: pytest.fail("no directory exists, so nothing should prompt"),
    )

    window._submit_jobs_on_cluster([continued, fresh])

    # Exactly the fresh job's computed target is probed: the continued job is
    # excluded, the fresh job's existence check is unchanged.
    assert recorded["checked"] == ["/remote/work/conv_ecut_400"]
    assert recorded["deleted"] == []
    by_directory = {call["remote_directory"]: call for call in recorded["submitted"]}
    assert set(by_directory) == {"/remote/work/conv_ecut_300", "/remote/work/conv_ecut_400"}
    assert by_directory["/remote/work/conv_ecut_300"]["allow_existing_remote_directory"] is True
    assert by_directory["/remote/work/conv_ecut_400"]["allow_existing_remote_directory"] is False
    assert by_directory["/remote/work/conv_ecut_400"]["reuse_existing_files"] is False

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

    monkeypatch.setattr(window._cluster_runner._runner, "remote_directory_exists", _directory_exists)
    monkeypatch.setattr(
        window,
        "_prompt_remote_directory_overwrite",
        lambda *_args: pytest.fail("no directory is known to exist, so nothing should prompt"),
    )
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(
        main_window_module,
        "show_detailed_warning",
        lambda _parent, title, message, **_kwargs: warnings.append((title, message)),
    )

    window._submit_jobs_on_cluster([unchecked, checked])

    assert [call["remote_directory"] for call in recorded["submitted"]] == ["/remote/work/conv_ecut_400"]
    assert unchecked.status == JobStatus.DEFINED
    assert unchecked.remote_directory is None
    assert checked.status == JobStatus.QUEUED
    assert warnings == [
        (
            "Cluster submission failed",
            "Some jobs failed to submit:\n- conv_ecut_300: not submitted because its remote directory "
            "could not be checked (SFTP stat failed)",
        )
    ]
    assert window.statusBar().currentMessage() == "Submitted 1 cluster job(s) on alpha"

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
        window,
        "_prompt_remote_directory_overwrite",
        lambda label, _target: prompted.append(label) or "no",
    )

    window._submit_jobs_on_cluster([continued, collider])

    assert prompted == ["conv_ecut_400"]
    assert recorded["deleted"] == []
    assert len(recorded["submitted"]) == 1
    assert recorded["submitted"][0]["seed_name"] == "conv_ecut_400"
    assert recorded["submitted"][0]["remote_directory"] == "/remote/work/conv_ecut_400"
    assert recorded["submitted"][0]["allow_existing_remote_directory"] is True
    assert continued.status == JobStatus.QUEUED
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
    monkeypatch.setattr(window, "_prompt_remote_directory_overwrite", lambda *_args: "yes")

    window._submit_jobs_on_cluster([continued, collider])

    assert recorded["deleted"] == []
    assert len(recorded["submitted"]) == 2
    assert continued.status == JobStatus.QUEUED
    assert collider.status == JobStatus.QUEUED

    window.close()


def test_main_window_submit_jobs_on_cluster_dashboard_resubmit_still_reuses_files(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The continuation branch must not capture the dashboard resubmit path.

    Dashboard resubmission passes FAILED/STOPPED/CANCELLED jobs, which the continuation
    branch ignores, and keeps its own reuse-the-remote-inputs semantics.
    """

    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_resubmit.mufinder", "ClusterResubmit")
    window._document = doc

    job = add_cluster_input_job(service, doc, "conv_ecut_300")
    job.execution_target = "alpha"
    job.remote_directory = "/remote/work/conv_ecut_300"
    job.status = JobStatus.FAILED

    patch_continuation_cluster_dialog(monkeypatch)
    recorded = patch_continuation_cluster_runner(monkeypatch, window, directory_exists=True)
    monkeypatch.setattr(
        window,
        "_prompt_remote_directory_overwrite",
        lambda *_args: pytest.fail("a resubmission into its own directory must not prompt"),
    )

    window._submit_jobs_on_cluster(
        [job],
        eligible_statuses={JobStatus.FAILED, JobStatus.CANCELLED},
        reuse_existing_remote_files=True,
    )

    assert recorded["checked"] == []
    assert recorded["deleted"] == []
    assert len(recorded["submitted"]) == 1
    submitted = recorded["submitted"][0]
    assert submitted["remote_directory"] == "/remote/work/conv_ecut_300"
    assert submitted["reuse_existing_files"] is True
    assert submitted["allow_existing_remote_directory"] is False

    window.close()


def test_main_window_submit_jobs_on_cluster_yes_to_all_requires_confirmation(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_yes_all_confirm.mufinder", "ClusterYesAllConfirm")
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
                job_name_prefix="conv",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)

    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(window._cluster_runner._runner, "remote_directory_exists", lambda _path: True)
    deleted: list[str] = []
    monkeypatch.setattr(window._cluster_runner._runner, "remove_remote_directory", lambda path: deleted.append(path))
    monkeypatch.setattr(window._cluster_runner._runner, "submit", lambda **_kwargs: "99123")

    prompt_calls: list[str] = []
    confirm_calls: list[list[str]] = []
    monkeypatch.setattr(
        window,
        "_prompt_remote_directory_overwrite",
        lambda *_args: prompt_calls.append("prompt") or "yes_to_all",
    )
    monkeypatch.setattr(
        window,
        "_confirm_bulk_remote_delete",
        lambda paths: confirm_calls.append(list(paths)) or (len(paths) == 2),
    )
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *_args, **_kwargs: None)

    window._submit_jobs_on_cluster(jobs)

    assert prompt_calls == ["prompt"]
    assert len(confirm_calls) == 1
    assert sorted(deleted) == ["/remote/work/conv_ecut_300", "/remote/work/conv_ecut_400"]
    window.close()


def test_main_window_cluster_status_handlers_update_job_and_run(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_status_ui.mufinder", "ClusterStatus")
    job = service.add_job(
        doc,
        label="cluster_job",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_CUTOFF,
        calc_type="castep_convergence_cutoff",
        execution_target="alpha",
    )
    service.submit_job(job)
    run = CalculationRun(job_id=job.id, run_index=1, status=JobStatus.SUBMITTING)
    doc.runs[run.id] = run
    window._document = doc

    window._on_cluster_job_status_changed(job.id, JobStatus.QUEUED.value)
    assert job.status == JobStatus.QUEUED

    window._on_cluster_job_status_changed(job.id, JobStatus.RUNNING.value)
    assert job.status == JobStatus.RUNNING
    assert run.status == JobStatus.RUNNING

    window._on_cluster_job_status_changed(job.id, JobStatus.RETRIEVING.value)
    assert job.status == JobStatus.RETRIEVING

    window._on_cluster_job_status_changed(job.id, JobStatus.STOPPED.value)
    assert job.status == JobStatus.STOPPED
    assert run.status == JobStatus.STOPPED
    assert run.completed_at is not None
    assert doc.provenance[-1].action == "job_stopped"

    window.close()


def test_main_window_refresh_ui_seeds_cluster_tracking_for_active_remote_jobs(
    main_window,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_tracking_seed.mufinder", "ClusterTrackingSeed")

    tracked_job = service.add_job(
        doc,
        label="kpoint_convergence_kgrid_15x15x15",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_KPOINT,
        calc_type="castep_convergence_kpoint",
        execution_target="Redwood",
    )
    tracked_job.status = JobStatus.RUNNING
    tracked_job.remote_job_id = "661459"
    tracked_job.remote_directory = "/mnt/redwoodfs/alice/project/kpoint_convergence_kgrid_15x15x15"

    local_job = service.add_job(
        doc,
        label="local_job",
        workflow_stage=main_window_module.WorkflowStage.CONVERGENCE_KPOINT,
        calc_type="castep_convergence_kpoint",
        execution_target="local",
    )
    local_job.status = JobStatus.RUNNING

    job_dir = doc.project_root / "files" / tracked_job.workflow_stage.value / tracked_job.label
    job_dir.mkdir(parents=True, exist_ok=True)

    clear_calls: list[str] = []
    track_calls: list[dict[str, object]] = []
    monkeypatch.setattr(window._cluster_runner, "clear_tracked_jobs", lambda: clear_calls.append("clear"))
    monkeypatch.setattr(
        window._cluster_runner,
        "track_existing_job",
        lambda **kwargs: track_calls.append(kwargs),
    )

    window._document = doc
    window._refresh_ui_after_project_change()

    assert clear_calls == ["clear"]
    assert len(track_calls) == 1
    assert track_calls[0]["job_id"] == tracked_job.id
    assert track_calls[0]["remote_job_id"] == "661459"
    assert track_calls[0]["cluster_name"] == "Redwood"
    assert track_calls[0]["status"] == JobStatus.RUNNING
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
            raise main_window_module.SSHUnknownHostKeyError(
                hostname="bastion.physics.ox.ac.uk",
                port=22,
                algorithm="ssh-ed25519",
                fingerprint_sha256="SHA256:bastion",
                key_base64="YmFzdGlvbg==",
                endpoint="jump",
            )
        if len(connect_attempts) == 2:
            raise main_window_module.SSHUnknownHostKeyError(
                hostname="glamdring.physics.ox.ac.uk",
                port=22,
                algorithm="ssh-ed25519",
                fingerprint_sha256="SHA256:glamdring",
                key_base64="Z2xhbWRyaW5n",
                endpoint="target",
            )

    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", _connect)
    monkeypatch.setattr(
        main_window_module,
        "prompt_to_trust_host_key",
        lambda _parent, _config, exc: prompts.append(exc.endpoint) or True,
    )

    window._connect_to_cluster_with_trust(config, credentials=credentials)

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

    sequence = iter(
        [
            main_window_module.QMessageBox.StandardButton.Yes,
            main_window_module.QMessageBox.StandardButton.YesToAll,
            main_window_module.QMessageBox.StandardButton.NoToAll,
            main_window_module.QMessageBox.StandardButton.No,
        ]
    )
    monkeypatch.setattr(main_window_module.QMessageBox, "exec", lambda _self: next(sequence))

    assert window._prompt_remote_directory_overwrite("job", "/remote/job") == "yes"
    assert window._prompt_remote_directory_overwrite("job", "/remote/job") == "yes_to_all"
    assert window._prompt_remote_directory_overwrite("job", "/remote/job") == "no_to_all"
    assert window._prompt_remote_directory_overwrite("job", "/remote/job") == "no"
    window.close()


def test_main_window_confirm_bulk_remote_delete_branches(
    main_window, monkeypatch: pytest.MonkeyPatch
) -> None:
    window = main_window

    assert window._confirm_bulk_remote_delete([]) is True

    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "exec",
        lambda _self: main_window_module.QMessageBox.StandardButton.Yes,
    )
    assert window._confirm_bulk_remote_delete(["/b", "/a", "/a"]) is True

    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "exec",
        lambda _self: main_window_module.QMessageBox.StandardButton.No,
    )
    assert window._confirm_bulk_remote_delete(["/a"]) is False
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
                job_name_prefix="conv",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)

    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(window._cluster_runner._runner, "remote_directory_exists", lambda _path: True)
    deleted: list[str] = []
    monkeypatch.setattr(window._cluster_runner._runner, "remove_remote_directory", lambda path: deleted.append(path))
    submitted: list[dict[str, object]] = []
    monkeypatch.setattr(window._cluster_runner._runner, "submit", lambda **kwargs: submitted.append(kwargs) or "99123")
    monkeypatch.setattr(window, "_prompt_remote_directory_overwrite", lambda *_args: "no_to_all")
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *_args, **_kwargs: None)

    window._submit_jobs_on_cluster([job])

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
                job_name_prefix="conv",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)
    monkeypatch.setattr(main_window_module.QApplication, "processEvents", lambda: None)

    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(window._cluster_runner._runner, "remote_directory_exists", lambda _path: True)
    deleted: list[str] = []
    monkeypatch.setattr(window._cluster_runner._runner, "remove_remote_directory", lambda path: deleted.append(path))
    submitted: list[dict[str, object]] = []
    monkeypatch.setattr(window._cluster_runner._runner, "submit", lambda **kwargs: submitted.append(kwargs) or "99123")
    monkeypatch.setattr(window, "_prompt_remote_directory_overwrite", lambda *_args: "no")
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda *_args, **_kwargs: None)

    window._submit_jobs_on_cluster([job])

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
                job_name_prefix="conv",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [cluster])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: cluster)
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

    monkeypatch.setattr(window._cluster_runner._runner, "connect_to_cluster", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        window._cluster_runner,
        "remote_job_directory",
        lambda *, working_directory, seed_name: f"{working_directory}/{seed_name}",
    )
    monkeypatch.setattr(window._cluster_runner._runner, "remote_directory_exists", lambda _path: False)
    monkeypatch.setattr(window._cluster_runner._runner, "submit", lambda **_kwargs: "99123")

    infos: list[str] = []
    monkeypatch.setattr(
        main_window_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    window._submit_jobs_on_cluster([job])

    assert "Cluster connection successful" not in infos
    window.close()


def test_main_window_round_trips_mem_per_cpu_in_last_submission_config(
    main_window, tmp_path: Path
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_mem_per_cpu.mufinder", "ClusterMem")
    window._document = doc

    window._persist_last_cluster_submission_config(
        ClusterSubmissionConfig(
            cluster_name="SCARF",
            working_directory="/work4/isis/alice/project_z",
            username="alice",
            job_name_prefix="positivemuon",
            num_processes=144,
            walltime="12:00:00",
            mem_per_cpu_gb=6.0,
        ),
        workflow_type="muon_relaxation/positivemuon",
    )

    persisted = doc.ui_state[main_window_module.MainWindow._LAST_CLUSTER_SUBMISSION_CONFIG_KEY]
    assert persisted["mem_per_cpu_gb"] == 6.0

    restored = window._load_last_cluster_submission_config(
        workflow_type="muon_relaxation/positivemuon"
    )
    assert restored is not None
    assert restored.mem_per_cpu_gb == 6.0
    window.close()


@pytest.mark.parametrize("stored", [None, "", "bad-memory", 0, -2.0])
def test_main_window_loads_absent_or_invalid_mem_per_cpu_as_unset(
    main_window, tmp_path: Path, stored: object
) -> None:
    window = main_window
    service = ProjectService()
    doc = service.create_project(tmp_path / "cluster_mem_per_cpu_bad.mufinder", "ClusterMem")
    window._document = doc
    doc.ui_state[main_window_module.MainWindow._LAST_CLUSTER_SUBMISSION_CONFIG_KEY] = {
        "cluster_name": "SCARF",
        "num_processes": 144,
        "mem_per_cpu_gb": stored,
    }

    restored = window._load_last_cluster_submission_config()

    assert restored is not None
    assert restored.mem_per_cpu_gb is None
    window.close()


def _quantum_phonon_job_after_cluster_singlepoint(window: MainWindow, tmp_path: Path) -> Job:
    """Seed a ready quantum phonon job whose stage-1 singlepoint ran on ``alpha``."""

    service = ProjectService()
    doc = service.create_project(tmp_path / "quantum_stage_two_target.mufinder", "QuantumStageTwoTarget")
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
    run = CalculationRun(job_id=source_job.id, status=JobStatus.COMPLETED, convergence_achieved=True)
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
    return phonon_job


def test_main_window_cluster_submission_refuses_phonon_job_on_a_different_cluster(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Picking another cluster in the dialog must not cold-start a continuation.

    The stage-1 checkpoint lives in ``alpha``'s directory, so a phonon job sent to
    ``beta`` would start without it and CASTEP would fail; it is refused before
    MuFinder connects, and the message names the cluster that holds it.
    """

    window = main_window
    phonon_job = _quantum_phonon_job_after_cluster_singlepoint(window, tmp_path)
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
                job_name_prefix="quantum",
                num_processes=16,
                walltime="01:00:00",
            )

        def password(self) -> str:
            return ""

    infos: list[str] = []
    monkeypatch.setattr(main_window_module, "ClusterSubmissionDialog", _FakeDialog)
    monkeypatch.setattr(main_window_module, "load_cluster_configs", lambda _settings: [beta])
    monkeypatch.setattr(main_window_module, "load_cluster_config", lambda _settings, _name: beta)
    monkeypatch.setattr(
        window._cluster_runner._runner,
        "connect_to_cluster",
        lambda *_args, **_kwargs: pytest.fail("a refused submission must not connect"),
    )
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda _parent, _title, text: infos.append(text))

    window._submit_jobs_on_cluster([phonon_job])

    assert phonon_job.status == JobStatus.DEFINED
    assert len(infos) == 1
    assert "not submitted to beta" in infos[0]
    assert "quantum_phonon_siteA (stage 1 ran on alpha)" in infos[0]
    window.close()


def test_main_window_local_submission_refuses_phonon_job_after_a_cluster_singlepoint(
    main_window, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The dashboard route bypasses the Quantum Corrections page, so the backend guards too."""

    window = main_window
    phonon_job = _quantum_phonon_job_after_cluster_singlepoint(window, tmp_path)
    infos: list[str] = []
    monkeypatch.setattr(main_window_module.QMessageBox, "information", lambda _parent, _title, text: infos.append(text))
    monkeypatch.setattr(
        main_window_module,
        "load_castep_setup",
        lambda _settings: pytest.fail("a refused local submission must stop before CASTEP setup"),
    )

    window._submit_jobs_locally([phonon_job])

    assert phonon_job.status == JobStatus.DEFINED
    assert len(infos) == 1
    assert "quantum_phonon_siteA (stage 1 ran on alpha)" in infos[0]
    window.close()
