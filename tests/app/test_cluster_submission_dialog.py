"""Tests for cluster submission dialog."""

from __future__ import annotations

from dataclasses import replace

import pytest

import mufinder.app.cluster_submission_dialog as dialog_module
import mufinder.services.keychain as keychain_module
from mufinder.app.cluster_submission_dialog import ClusterSubmissionConfig, ClusterSubmissionDialog
from mufinder.domain.cluster import ClusterConfig
from mufinder.services.ssh_client import SSHCredentials


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


def _jump_password_config() -> ClusterConfig:
    return ClusterConfig(
        name="Redwood",
        hostname="glamdring.physics.ox.ac.uk",
        username="glam-user",
        auth_method="password",
        default_working_directory="/mnt/redwoodfs/glam-user",
        queue_system="addqueue",
        cores_per_node=24,
        default_walltime="",
        template_script_path="",
        jump_host_enabled=True,
        jump_hostname="bastion.physics.ox.ac.uk",
        jump_username="physics-user",
        jump_auth_method="password",
    )


def _keychain_password_config() -> ClusterConfig:
    return replace(_jump_password_config(), use_macos_keychain_passwords=True)


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


def test_cluster_submission_dialog_shows_jump_host_credentials(qapp) -> None:
    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=[_jump_password_config()])

    assert dialog._jump_secret_edit.isHidden() is False
    assert dialog._target_secret_label.text() == "Redwood password"
    assert dialog._jump_secret_label.text() == "Physics password"
    assert "Physics password" in dialog._auth_hint.text()
    assert "above" in dialog._auth_hint.text()
    assert "below" not in dialog._auth_hint.text()

    dialog._target_secret_edit.setText("glam-secret")
    dialog._jump_secret_edit.setText("physics-secret")

    credentials = dialog.credentials()
    assert credentials.target_secret == "glam-secret"
    assert credentials.jump_secret == "physics-secret"
    dialog.close()


def test_cluster_submission_dialog_validates_jump_host_password_first(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=[_jump_password_config()])

    warnings: list[str] = []
    monkeypatch.setattr(
        dialog_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    dialog._accept_if_valid()
    assert warnings[-1] == "Missing jump host password"

    dialog._jump_secret_edit.setText("physics-secret")
    dialog._accept_if_valid()
    assert warnings[-1] == "Missing cluster password"
    dialog.close()


def test_cluster_submission_dialog_keychain_mode_hides_manual_credentials(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keychain_module.sys, "platform", "darwin")
    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=[_keychain_password_config()])

    monkeypatch.setattr(
        dialog_module,
        "resolve_cluster_credentials",
        lambda *_args, **_kwargs: SSHCredentials(target_secret="glam-secret", jump_secret="physics-secret"),
    )
    accepted: list[bool] = []
    monkeypatch.setattr(dialog, "accept", lambda: accepted.append(True))

    assert dialog._target_secret_edit.isHidden() is True
    assert dialog._jump_secret_edit.isHidden() is True
    assert "macOS Keychain passwords are enabled" in dialog._auth_hint.text()

    dialog._accept_if_valid()

    assert accepted == [True]
    assert dialog.credentials() == SSHCredentials(target_secret="glam-secret", jump_secret="physics-secret")
    dialog.close()


def test_cluster_submission_dialog_keyring_mode_names_linux_store(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keychain_module.sys, "platform", "linux")
    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=[_keychain_password_config()])

    assert dialog._target_secret_edit.isHidden() is True
    assert dialog._jump_secret_edit.isHidden() is True
    assert "System keyring passwords are enabled" in dialog._auth_hint.text()
    assert "from the keyring" in dialog._auth_hint.text()
    dialog.close()


def test_cluster_submission_dialog_compacts_large_job_summary(qapp) -> None:
    job_labels = [f"cutoff_convergence_ecut_{index}" for index in range(12)]

    dialog = ClusterSubmissionDialog(job_labels=job_labels, cluster_configs=[_jump_password_config()])

    summary = dialog._summary_label.text()
    assert "12 job(s)" in summary
    assert "cutoff_convergence_ecut_0" in summary
    assert "cutoff_convergence_ecut_3" in summary
    assert "cutoff_convergence_ecut_4" not in summary
    assert "and 8 more" in summary
    dialog.close()


def test_cluster_submission_dialog_password_auth_hint_has_vertical_headroom(qapp) -> None:
    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=_configs())

    dialog._cluster_combo.setCurrentText("beta")

    assert "Password authentication is enabled" in dialog._auth_hint.text()
    assert bool(dialog._auth_hint.alignment() & dialog_module.Qt.AlignTop)
    assert dialog._auth_hint.minimumHeight() >= dialog._auth_hint.fontMetrics().lineSpacing() + 6
    assert dialog._auth_hint.sizeHint().height() >= dialog._auth_hint.fontMetrics().lineSpacing()
    dialog.close()


def test_cluster_submission_dialog_auth_hint_reserves_wrapped_keychain_height(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keychain_module.sys, "platform", "darwin")
    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=[_keychain_password_config()])

    dialog._auth_hint.setFixedWidth(320)
    dialog._set_auth_hint_text(dialog._auth_hint.text())

    assert dialog._auth_hint.minimumHeight() >= dialog._auth_hint.sizeHint().height()
    assert dialog._auth_hint.minimumHeight() > dialog._auth_hint.fontMetrics().lineSpacing() + 6
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


def test_cluster_submission_dialog_defaults_pristine_relaxation_prefix(qapp) -> None:
    dialog = ClusterSubmissionDialog(
        job_labels=["relax_1"],
        cluster_configs=_configs(),
        project_name="My Project",
        workflow_type="pristine_relaxation",
    )

    config = dialog.config()

    assert config.working_directory == "/work/alice/My_Project/pristine_relaxation"
    assert config.job_name_prefix == "pristine_relax"
    dialog.close()


def test_cluster_submission_dialog_defaults_uep_prefix(qapp) -> None:
    dialog = ClusterSubmissionDialog(
        job_labels=["uep_1"],
        cluster_configs=_configs(),
        project_name="My Project",
        workflow_type="uep",
    )

    config = dialog.config()

    assert config.working_directory == "/work/alice/My_Project/uep"
    assert config.job_name_prefix == "uep"
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
    assert warnings[-1] == "Missing cluster password"
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
    assert dialog._queue_combo.isHidden() is False
    dialog.close()


def test_cluster_submission_dialog_requests_no_memory_by_default(qapp) -> None:
    # Unchecked by default so an upgrade never silently changes a generated
    # batch script that used to carry no memory directive.
    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=_configs())

    assert dialog._mem_per_cpu_check.isChecked() is False
    assert dialog._mem_per_cpu_edit.isEnabled() is False
    assert dialog.config().mem_per_cpu_gb is None
    dialog.close()


def test_cluster_submission_dialog_reports_enabled_mem_per_cpu(qapp) -> None:
    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=_configs())

    dialog._mem_per_cpu_check.setChecked(True)
    dialog._mem_per_cpu_edit.setValue(6.0)

    assert dialog._mem_per_cpu_edit.isEnabled() is True
    assert dialog.config().mem_per_cpu_gb == 6.0
    dialog.close()


def test_cluster_submission_dialog_mem_per_cpu_is_hidden_for_addqueue(qapp) -> None:
    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=_configs())

    assert dialog._mem_per_cpu_check.isHidden() is False

    dialog._cluster_combo.setCurrentText("redwood")

    assert dialog._mem_per_cpu_check.isHidden() is True
    assert dialog._mem_per_cpu_edit.isHidden() is True
    dialog.close()


def test_cluster_submission_dialog_drops_mem_per_cpu_for_addqueue(qapp) -> None:
    # addqueue takes memory through its own -m argument, so even a stale checked
    # state must not leak a per-CPU request into an addqueue submission.
    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=_configs())
    dialog._mem_per_cpu_check.setChecked(True)
    dialog._mem_per_cpu_edit.setValue(6.0)

    dialog._cluster_combo.setCurrentText("redwood")
    dialog._mem_per_cpu_check.setChecked(True)

    assert dialog.config().mem_per_cpu_gb is None
    assert dialog.config().memory_gb == 7.5
    dialog.close()


def test_cluster_submission_dialog_seeds_mem_per_cpu_from_cluster_default(qapp) -> None:
    configs = list(_configs())
    configs[0] = replace(configs[0], default_mem_per_cpu_gb=8.0)

    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=configs)

    assert dialog._mem_per_cpu_check.isChecked() is True
    assert dialog._mem_per_cpu_edit.isEnabled() is True
    assert dialog.config().mem_per_cpu_gb == 8.0
    dialog.close()


def test_cluster_submission_dialog_restores_mem_per_cpu_from_initial_config(qapp) -> None:
    initial = ClusterSubmissionConfig(
        cluster_name="alpha",
        working_directory="/work/alice/project_a",
        username="alice",
        job_name_prefix="positivemuon",
        num_processes=144,
        walltime="12:00:00",
        mem_per_cpu_gb=6.0,
    )

    dialog = ClusterSubmissionDialog(
        job_labels=["job_a"],
        cluster_configs=_configs(),
        initial_config=initial,
    )

    assert dialog._mem_per_cpu_check.isChecked() is True
    assert dialog.config().mem_per_cpu_gb == 6.0
    dialog.close()


def test_cluster_submission_dialog_clears_mem_per_cpu_when_switching_clusters(qapp) -> None:
    configs = list(_configs())
    configs[0] = replace(configs[0], default_mem_per_cpu_gb=8.0)
    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=configs)

    dialog._cluster_combo.setCurrentText("beta")

    assert dialog._mem_per_cpu_check.isChecked() is False
    assert dialog.config().mem_per_cpu_gb is None
    dialog.close()


def test_cluster_dialog_blocks_when_farms_exceed_processes(qapp) -> None:
    # farms > MPI ranks is a hard CASTEP abort, not a slow run.
    dialog = ClusterSubmissionDialog(
        job_labels=["barrier_siteA_to_siteB_neb"],
        cluster_configs=_configs(),
        farm_counts={"barrier_siteA_to_siteB_neb": 6},
    )
    dialog._num_processes_spin.setValue(4)

    ok_button = dialog._button_box.button(dialog_module.QDialogButtonBox.StandardButton.Ok)
    assert ok_button.isEnabled() is False
    assert "at least 6" in dialog._farm_warning.text()

    dialog._num_processes_spin.setValue(12)
    assert ok_button.isEnabled() is True
    assert dialog._farm_warning.text() == ""
    dialog.close()


def test_cluster_dialog_warns_on_uneven_farm_division(qapp) -> None:
    dialog = ClusterSubmissionDialog(
        job_labels=["barrier_siteA_to_siteB_neb"],
        cluster_configs=_configs(),
        farm_counts={"barrier_siteA_to_siteB_neb": 5},
    )
    dialog._num_processes_spin.setValue(12)

    ok_button = dialog._button_box.button(dialog_module.QDialogButtonBox.StandardButton.Ok)
    assert ok_button.isEnabled() is True
    assert "idle" in dialog._farm_warning.text().lower()
    dialog.close()


def _psi_config(**overrides) -> ClusterConfig:  # noqa: ANN003
    from mufinder.domain.cluster import ClusterQueue

    values = dict(
        name="PSI",
        hostname="login.psi.example",
        username="dana",
        default_working_directory="/data/dana",
        queue_system="slurm",
        cores_per_node=44,
        default_walltime="01:00:00",
        template_script_path="/templates/psi.sh",
        default_mem_per_cpu_gb=4.0,
        default_queue_name="daily",
        queues=(
            ClusterQueue(name="hourly", default_walltime="00:59:00", max_walltime="01:00:00"),
            ClusterQueue(
                name="daily",
                default_walltime="12:00:00",
                max_walltime="1-00:00:00",
                cores_per_node=88,
                default_mem_per_cpu_gb=8.0,
            ),
            ClusterQueue(name="general"),
        ),
    )
    values.update(overrides)
    return ClusterConfig(**values)


def _queue_items(dialog: ClusterSubmissionDialog) -> list[str]:
    return [dialog._queue_combo.itemText(index) for index in range(dialog._queue_combo.count())]


def test_slurm_submission_offers_the_profile_queues_and_starts_on_the_default(qapp) -> None:
    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=[_psi_config()])

    assert dialog._queue_combo.isHidden() is False
    assert _queue_items(dialog) == ["daily", "hourly", "general"]
    config = dialog.config()
    assert config.queue_name == "daily"
    assert (config.walltime, config.num_processes, config.mem_per_cpu_gb) == ("12:00:00", 88, 8.0)
    assert dialog._queue_hint.text() == "Maximum walltime on this queue: 1-00:00:00."
    dialog.close()


def test_picking_a_queue_applies_its_defaults(qapp) -> None:
    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=[_psi_config()])

    dialog._queue_combo.setCurrentIndex(_queue_items(dialog).index("general"))
    dialog._queue_combo.activated.emit(dialog._queue_combo.currentIndex())

    config = dialog.config()
    assert config.queue_name == "general"
    # "general" overrides nothing, so the profile values apply.
    assert (config.walltime, config.num_processes, config.mem_per_cpu_gb) == ("01:00:00", 44, 4.0)
    assert dialog._queue_hint.isHidden() is True
    dialog.close()


def test_typed_queue_name_is_submitted_as_is(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=[_psi_config()])
    accepted: list[bool] = []
    monkeypatch.setattr(dialog, "accept", lambda: accepted.append(True))

    dialog._queue_combo.setEditText("debug")
    dialog._accept_if_valid()

    assert accepted == [True]
    assert dialog.config().queue_name == "debug"
    dialog.close()


def test_slurm_submission_restores_the_last_used_queue(qapp) -> None:
    initial = ClusterSubmissionConfig(
        cluster_name="PSI",
        working_directory="/data/dana/project",
        username="dana",
        job_name_prefix="convergence",
        num_processes=22,
        walltime="00:30:00",
        queue_name="hourly",
        mem_per_cpu_gb=2.0,
    )

    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=[_psi_config()], initial_config=initial)

    config = dialog.config()
    assert config.queue_name == "hourly"
    assert (config.walltime, config.num_processes, config.mem_per_cpu_gb) == ("00:30:00", 22, 2.0)
    assert dialog._queue_hint.text() == "Maximum walltime on this queue: 01:00:00."
    dialog.close()


def test_submission_refuses_a_walltime_beyond_the_queue_maximum(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=[_psi_config()])
    warnings: list[tuple[str, str]] = []
    monkeypatch.setattr(dialog_module.QMessageBox, "warning", lambda _p, title, text: warnings.append((title, text)))
    accepted: list[bool] = []
    monkeypatch.setattr(dialog, "accept", lambda: accepted.append(True))

    dialog._walltime_edit.setText("2-00:00:00")
    dialog._accept_if_valid()

    assert accepted == []
    assert warnings == [
        ("Invalid walltime", "The walltime 2-00:00:00 is longer than the 1-00:00:00 maximum for queue 'daily'.")
    ]
    dialog.close()


@pytest.mark.parametrize(("walltime", "queue", "title"), [("soon", "daily", "Invalid walltime"), ("01:00:00", "a b", "Invalid queue")])
def test_submission_refuses_unreadable_walltime_or_queue(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
    walltime: str,
    queue: str,
    title: str,
) -> None:
    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=[_psi_config()])
    warnings: list[str] = []
    monkeypatch.setattr(dialog_module.QMessageBox, "warning", lambda _p, t, _text: warnings.append(t))
    monkeypatch.setattr(dialog, "accept", lambda: pytest.fail("accepted"))

    dialog._queue_combo.setEditText(queue)
    dialog._walltime_edit.setText(walltime)
    dialog._accept_if_valid()

    assert warnings == [title]
    dialog.close()


def test_submission_hint_says_when_the_template_sets_the_partition(qapp, tmp_path) -> None:  # noqa: ANN001
    template = tmp_path / "fixed.sh"
    template.write_text("#!/bin/bash\n#SBATCH --partition=fixed\n$castep_command\n", encoding="utf-8")

    dialog = ClusterSubmissionDialog(
        job_labels=["job_a"],
        cluster_configs=[_psi_config(template_script_path=str(template))],
    )

    assert "sets its own partition" in dialog._queue_hint.text()
    dialog.close()


def test_profile_without_queues_submits_no_slurm_partition(qapp) -> None:
    dialog = ClusterSubmissionDialog(job_labels=["job_a"], cluster_configs=_configs(), default_cluster_name="alpha")

    assert _queue_items(dialog) == []
    assert dialog.config().queue_name == ""
    dialog.close()
