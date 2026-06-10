"""Tests for cluster setup dialog."""

from __future__ import annotations

import pytest

import mufinder.app.cluster_setup_dialog as dialog_module
from mufinder.cluster_setup import load_cluster_configs
from mufinder.app.cluster_setup_dialog import ClusterSetupDialog
from mufinder.services.ssh_client import SSHCredentials


class _FakeSettings:
    def __init__(self) -> None:
        self._values: dict[str, object] = {}

    def value(self, key: str, default=None, value_type=None):  # noqa: ANN001
        value = self._values.get(key, default)
        if value_type is str:
            return str(value)
        return value

    def setValue(self, key: str, value) -> None:  # noqa: N802, ANN001
        self._values[key] = value


class _FakeConnection:
    def __init__(self, _config, *, settings=None) -> None:
        _ = settings
        self.connected = False
        self.password_used: str | None = None

    @property
    def is_connected(self) -> bool:
        return self.connected

    def connect(self, password: str | None = None) -> None:
        self.password_used = password
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False


def test_cluster_setup_dialog_adds_default_cluster(qapp) -> None:
    dialog = ClusterSetupDialog(settings=_FakeSettings())

    assert dialog._cluster_combo.count() == 2
    assert dialog._name_edit.text() == "SCARF"
    assert dialog._hostname_edit.text() == "ui1.scarf.rl.ac.uk"
    assert dialog._queue_combo.currentText() == "SLURM"
    assert dialog._keyword_mode_combo.currentText() == "scheduler"
    assert dialog._keyword_shell_init_edit.text() == "module load CASTEP"
    dialog.close()


def test_cluster_setup_dialog_saves_configuration(qapp) -> None:
    settings = _FakeSettings()
    dialog = ClusterSetupDialog(settings=settings)

    dialog._name_edit.setText("alpha")
    dialog._hostname_edit.setText("alpha.example")
    dialog._username_edit.setText("alice")
    dialog._auth_combo.setCurrentText("password")
    dialog._working_dir_edit.setText("/work/alice")
    dialog._template_path_edit.setText("/templates/slurm.sh")
    dialog._keyword_mode_combo.setCurrentText("scheduler")
    dialog._keyword_shell_init_edit.setText("module load CASTEP")
    dialog._keyword_python_edit.setText("python")

    dialog._accept_and_save()

    loaded = load_cluster_configs(settings)
    assert len(loaded) == 2
    alpha = next(config for config in loaded if config.name == "alpha")
    assert alpha.hostname == "alpha.example"
    assert alpha.auth_method == "password"
    assert alpha.castep_executable == "castep.mpi"
    assert alpha.keyword_execution_mode == "scheduler"
    assert alpha.keyword_shell_init == "module load CASTEP"
    assert alpha.keyword_python_command == "python"
    dialog.close()


def test_cluster_setup_dialog_addqueue_does_not_require_template(qapp) -> None:
    settings = _FakeSettings()
    dialog = ClusterSetupDialog(settings=settings)

    dialog._name_edit.setText("redwood")
    dialog._hostname_edit.setText("glamdring.physics.ox.ac.uk")
    dialog._username_edit.setText("alice")
    dialog._queue_combo.setCurrentText("ADDQUEUE")
    dialog._walltime_edit.setText("")
    dialog._template_path_edit.setText("")
    dialog._submit_command_edit.setText("/bin/addqueue")

    dialog._accept_and_save()

    loaded = load_cluster_configs(settings)
    names = [config.name for config in loaded]
    assert "redwood" in names
    saved = next(config for config in loaded if config.name == "redwood")
    assert saved.queue_system == "ADDQUEUE"
    assert saved.default_walltime == ""
    dialog.close()


def test_cluster_setup_dialog_loads_redwood_jump_host_defaults(qapp) -> None:
    dialog = ClusterSetupDialog(settings=_FakeSettings())
    dialog._cluster_combo.setCurrentText("Redwood")

    assert dialog._hostname_edit.text() == "glamdring.physics.ox.ac.uk"
    assert dialog._auth_combo.currentText() == "password"
    assert dialog._jump_enabled_check.isChecked() is True
    assert dialog._jump_hostname_edit.text() == "bastion.physics.ox.ac.uk"
    assert dialog._jump_username_edit.text() == "physics_username"
    assert dialog._jump_auth_combo.currentText() == "password"
    dialog.close()


def test_cluster_setup_dialog_saves_jump_host_configuration(qapp) -> None:
    settings = _FakeSettings()
    dialog = ClusterSetupDialog(settings=settings)

    dialog._name_edit.setText("alpha")
    dialog._hostname_edit.setText("alpha.example")
    dialog._username_edit.setText("alice")
    dialog._auth_combo.setCurrentText("password")
    dialog._jump_enabled_check.setChecked(True)
    dialog._jump_hostname_edit.setText("bastion.example")
    dialog._jump_port_spin.setValue(2222)
    dialog._jump_username_edit.setText("physics-alice")
    dialog._jump_auth_combo.setCurrentText("key")
    dialog._jump_key_path_edit.setText("/keys/physics")
    dialog._working_dir_edit.setText("/work/alice")
    dialog._template_path_edit.setText("/templates/slurm.sh")

    dialog._accept_and_save()

    saved = next(config for config in load_cluster_configs(settings) if config.name == "alpha")
    assert saved.jump_host_enabled is True
    assert saved.jump_hostname == "bastion.example"
    assert saved.jump_port == 2222
    assert saved.jump_username == "physics-alice"
    assert saved.jump_auth_method == "key"
    assert saved.jump_key_path == "/keys/physics"
    dialog.close()


def test_cluster_setup_dialog_test_connection_uses_ssh_client(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = ClusterSetupDialog(settings=_FakeSettings())

    infos: list[str] = []
    monkeypatch.setattr(dialog_module, "SSHConnection", _FakeConnection)
    monkeypatch.setattr(dialog_module, "prompt_for_cluster_credentials", lambda *_args, **_kwargs: SSHCredentials())
    monkeypatch.setattr(
        dialog_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    dialog._hostname_edit.setText("alpha.example")
    dialog._username_edit.setText("alice")
    dialog._template_path_edit.setText("/templates/slurm.sh")
    dialog._test_connection()

    assert infos == ["Connection successful"]
    dialog.close()


def test_cluster_setup_dialog_test_connection_passes_key_passphrase(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = ClusterSetupDialog(settings=_FakeSettings())

    captured_passwords: list[str | None] = []

    class _CaptureConnection(_FakeConnection):
        def connect(self, password: str | None = None) -> None:
            captured_passwords.append(password)
            super().connect(password=password)

    monkeypatch.setattr(dialog_module, "SSHConnection", _CaptureConnection)
    monkeypatch.setattr(
        dialog_module,
        "prompt_for_cluster_credentials",
        lambda *_args, **_kwargs: SSHCredentials(target_secret="my-passphrase"),
    )
    monkeypatch.setattr(
        dialog_module.QMessageBox,
        "information",
        lambda *_args, **_kwargs: None,
    )

    dialog._hostname_edit.setText("alpha.example")
    dialog._username_edit.setText("alice")
    dialog._template_path_edit.setText("/templates/slurm.sh")
    dialog._auth_combo.setCurrentText("key")
    dialog._test_connection()

    assert captured_passwords == ["my-passphrase"]
    dialog.close()


def test_cluster_setup_dialog_test_connection_trusts_unknown_host_key(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = ClusterSetupDialog(settings=_FakeSettings())

    infos: list[str] = []
    prompted: list[str] = []

    class _UnknownHostConnection(_FakeConnection):
        attempts = 0

        def connect(self, password: str | None = None) -> None:
            self.__class__.attempts += 1
            if self.__class__.attempts == 1:
                raise dialog_module.SSHUnknownHostKeyError(
                    hostname="alpha.example",
                    port=22,
                    algorithm="ssh-ed25519",
                    fingerprint_sha256="SHA256:test",
                    key_base64="dGVzdA==",
                )
            super().connect(password=password)

    monkeypatch.setattr(dialog_module, "SSHConnection", _UnknownHostConnection)
    monkeypatch.setattr(dialog_module, "prompt_to_trust_host_key", lambda *_args, **_kwargs: prompted.append("yes") or True)
    monkeypatch.setattr(dialog_module, "prompt_for_cluster_credentials", lambda *_args, **_kwargs: SSHCredentials())
    monkeypatch.setattr(
        dialog_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    dialog._hostname_edit.setText("alpha.example")
    dialog._username_edit.setText("alice")
    dialog._template_path_edit.setText("/templates/slurm.sh")
    dialog._test_connection()

    assert prompted == ["yes"]
    assert infos == ["Connection successful"]
    assert dialog._settings.value("trusted_host_keys/SCARF/alpha.example/algorithm", "", str) == "ssh-ed25519"
    dialog.close()


def test_cluster_setup_dialog_remove_cluster_and_auth_controls(qapp) -> None:
    dialog = ClusterSetupDialog(settings=_FakeSettings())

    dialog._add_cluster()
    assert dialog._cluster_combo.count() >= 2

    dialog._auth_combo.setCurrentText("password")
    dialog._refresh_auth_controls()
    assert dialog._key_path_edit.isEnabled() is False

    dialog._remove_cluster()
    assert dialog._cluster_combo.count() >= 1
    dialog.close()


def test_cluster_setup_dialog_validation_blocks_invalid_save(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = ClusterSetupDialog(settings=_FakeSettings())

    warnings: list[str] = []
    monkeypatch.setattr(
        dialog_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    dialog._name_edit.setText("alpha")
    dialog._hostname_edit.setText("")
    dialog._username_edit.setText("")
    dialog._template_path_edit.setText("")

    dialog._accept_and_save()

    assert warnings == ["Invalid cluster configuration"]
    dialog.close()


def test_cluster_setup_dialog_username_change_can_sync_working_dir(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = ClusterSetupDialog(settings=_FakeSettings())

    monkeypatch.setattr(
        dialog_module.QMessageBox,
        "question",
        lambda *_args, **_kwargs: dialog_module.QMessageBox.StandardButton.Yes,
    )

    dialog._username_edit.setText("alice")
    dialog._working_dir_edit.setText("/work4/isis/username")
    dialog._maybe_sync_working_dir_for_username()

    assert dialog._working_dir_edit.text() == "/work4/isis/alice"
    dialog.close()
