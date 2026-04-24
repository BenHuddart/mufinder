"""Tests for cluster submission dialog."""

from __future__ import annotations

import pytest

import mufinder.app.cluster_submission_dialog as dialog_module
from mufinder.app.cluster_submission_dialog import ClusterSubmissionConfig, ClusterSubmissionDialog
from mufinder.cluster_setup import ClusterConfig


def _configs() -> list[ClusterConfig]:
    return [
        ClusterConfig(
            name="alpha",
            hostname="alpha.example",
            username="alice",
            auth_method="key",
            key_path="/keys/alice",
            default_working_directory="/work/alice",
            queue_system="slurm",
            cores_per_node=64,
            default_walltime="02:00:00",
            template_script_path="/templates/alpha.sh",
        ),
        ClusterConfig(
            name="beta",
            hostname="beta.example",
            username="bob",
            auth_method="password",
            key_path="",
            default_working_directory="/scratch/bob",
            queue_system="slurm",
            cores_per_node=32,
            default_walltime="01:00:00",
            template_script_path="/templates/beta.sh",
        ),
        ClusterConfig(
            name="redwood",
            hostname="glamdring.physics.ox.ac.uk",
            username="charlie",
            auth_method="key",
            key_path="",
            default_working_directory="/mnt/redwoodfs/charlie",
            queue_system="addqueue",
            cores_per_node=32,
            default_walltime="",
            template_script_path="",
            default_queue_name="redwood",
            default_memory_gb=7.5,
            castep_executable="/mnt/redwoodfs/muon/castep.mpi",
        ),
    ]


def test_cluster_submission_dialog_populates_defaults(qapp) -> None:
    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=_configs())

    config = dialog.config()

    assert config.cluster_name == "alpha"
    assert config.working_directory == "/work/alice"
    assert config.username == "alice"
    assert config.num_processes == 64
    assert config.walltime == "02:00:00"
    dialog.close()


def test_cluster_submission_dialog_switches_cluster_defaults(qapp) -> None:
    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=_configs())

    dialog._cluster_combo.setCurrentText("beta")
    config = dialog.config()

    assert config.cluster_name == "beta"
    assert config.working_directory == "/scratch/bob"
    assert config.username == "bob"
    assert config.num_processes == 32
    assert dialog._password_edit.isHidden() is False
    dialog.close()


def test_cluster_submission_dialog_password_auth_hint_has_vertical_headroom(qapp) -> None:
    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=_configs())

    dialog._cluster_combo.setCurrentText("beta")

    assert "Password authentication is enabled" in dialog._auth_hint.text()
    assert bool(dialog._auth_hint.alignment() & dialog_module.Qt.AlignTop)
    assert dialog._auth_hint.minimumHeight() >= dialog._auth_hint.fontMetrics().lineSpacing() + 6
    assert dialog._auth_hint.sizeHint().height() >= dialog._auth_hint.fontMetrics().lineSpacing()
    dialog.close()


def test_cluster_submission_dialog_default_cluster_name(qapp) -> None:
    dialog = ClusterSubmissionDialog(
        job_labels=["job_a"],
        cluster_configs=_configs(),
        default_cluster_name="beta",
    )

    assert dialog.selected_cluster_name() == "beta"
    dialog.close()


def test_cluster_submission_dialog_applies_initial_config_for_matching_cluster(qapp) -> None:
    initial = ClusterSubmissionConfig(
        cluster_name="redwood",
        working_directory="/mnt/redwoodfs/custom/project_a",
        username="charlie",
        job_name_prefix="pristine_relax",
        num_processes=24,
        walltime="",
        queue_name="redwood-priority",
        memory_gb=12.0,
        comment="night-run",
    )

    dialog = ClusterSubmissionDialog(
        job_labels=["job_a"],
        cluster_configs=_configs(),
        initial_config=initial,
    )

    config = dialog.config()
    assert dialog.selected_cluster_name() == "redwood"
    assert config.working_directory == "/mnt/redwoodfs/custom/project_a"
    assert config.username == "charlie"
    assert config.job_name_prefix == "pristine_relax"
    assert config.num_processes == 24
    assert config.queue_name == "redwood-priority"
    assert config.memory_gb == 12.0
    assert config.comment == "night-run"
    dialog.close()


def test_cluster_submission_dialog_does_not_apply_initial_config_to_other_cluster(qapp) -> None:
    initial = ClusterSubmissionConfig(
        cluster_name="redwood",
        working_directory="/mnt/redwoodfs/custom/project_a",
        username="charlie",
        job_name_prefix="pristine_relax",
        num_processes=24,
        walltime="",
        queue_name="redwood-priority",
        memory_gb=12.0,
        comment="night-run",
    )

    dialog = ClusterSubmissionDialog(
        job_labels=["job_a"],
        cluster_configs=_configs(),
        default_cluster_name="alpha",
        initial_config=initial,
    )

    config = dialog.config()
    assert dialog.selected_cluster_name() == "alpha"
    assert config.working_directory == "/work/alice"
    assert config.username == "alice"
    assert config.job_name_prefix == "convergence"
    assert config.num_processes == 64
    assert config.walltime == "02:00:00"
    dialog.close()


def test_cluster_submission_dialog_builds_project_workflow_workdir(qapp) -> None:
    dialog = ClusterSubmissionDialog(
        job_labels=["job_a"],
        cluster_configs=_configs(),
        project_name="My Project",
        workflow_type="convergence_cutoff",
    )

    assert dialog.config().working_directory == "/work/alice/My_Project/convergence_cutoff"
    dialog.close()


def test_cluster_submission_dialog_preserves_nested_muon_workdir_and_prefix(qapp) -> None:
    dialog = ClusterSubmissionDialog(
        job_labels=["Random Site 1"],
        cluster_configs=_configs(),
        project_name="My Project",
        workflow_type="muon_relaxation/positivemuon",
    )

    config = dialog.config()

    assert config.working_directory == "/work/alice/My_Project/muon_relaxation/positivemuon"
    assert config.job_name_prefix == "positivemuon"
    dialog.close()


def test_cluster_submission_dialog_key_auth_allows_blank_passphrase(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=_configs())

    accepted: list[bool] = []
    monkeypatch.setattr(dialog, "accept", lambda: accepted.append(True))

    dialog._cluster_combo.setCurrentText("alpha")
    dialog._password_edit.setText("")
    dialog._accept_if_valid()

    assert accepted == [True]
    dialog.close()


def test_cluster_submission_dialog_validation_warns_for_missing_fields(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=_configs())

    warnings: list[str] = []
    monkeypatch.setattr(
        dialog_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    dialog._working_directory_edit.setText("")
    dialog._accept_if_valid()
    assert warnings[-1] == "Missing working directory"

    dialog._working_directory_edit.setText("/work/alice")
    dialog._walltime_edit.setText("")
    dialog._accept_if_valid()
    assert warnings[-1] == "Missing walltime"

    dialog._cluster_combo.setCurrentText("beta")
    dialog._password_edit.setText("")
    dialog._accept_if_valid()
    assert warnings[-1] == "Missing password"
    dialog.close()


def test_cluster_submission_dialog_active_connection_hides_password_and_accepts(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = ClusterSubmissionDialog(
        job_labels=["job_a"],
        cluster_configs=_configs(),
        default_cluster_name="beta",
        connected_cluster_name="beta",
        connected_username="bob",
    )

    accepted: list[bool] = []
    monkeypatch.setattr(dialog, "accept", lambda: accepted.append(True))

    assert dialog._password_edit.isVisible() is False
    dialog._accept_if_valid()

    assert accepted == [True]
    dialog.close()


def test_cluster_submission_dialog_addqueue_defaults_and_optional_walltime(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=_configs())
    dialog._cluster_combo.setCurrentText("redwood")
    dialog._walltime_edit.setText("")
    dialog._comment_edit.setText("overnight")

    accepted: list[bool] = []
    monkeypatch.setattr(dialog, "accept", lambda: accepted.append(True))
    dialog._accept_if_valid()

    config = dialog.config()
    assert accepted == [True]
    assert config.queue_name == "redwood"
    assert config.memory_gb == 7.5
    assert config.comment == "overnight"
    assert dialog._queue_name_edit.isHidden() is False
    dialog.close()
