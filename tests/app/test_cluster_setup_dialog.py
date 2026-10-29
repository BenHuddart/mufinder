"""Tests for cluster setup dialog."""

from __future__ import annotations


import pytest

import mufinder.app.cluster_setup_dialog as dialog_module
from mufinder.app import workflow_table_utils
import mufinder.services.keychain as keychain_module
from mufinder.cluster_setup import load_cluster_configs
from mufinder.app.cluster_setup_dialog import ClusterSetupDialog
from mufinder.services.ssh_client import SSHCredentials, SSHUnknownHostKeyError
from tests.support.qt import pump_until


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


@pytest.fixture(autouse=True)
def _fail_on_unexpected_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Turn an unexpected modal warning into a failure instead of a hang."""

    def _raise(_parent, title, message, *_args, **_kwargs):  # noqa: ANN001
        raise AssertionError(f"Unexpected warning {title!r}: {message}")

    monkeypatch.setattr(dialog_module.QMessageBox, "warning", _raise)


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

    assert dialog._cluster_list.count() == 2
    assert dialog._name_edit.text() == "SCARF"
    assert dialog._hostname_edit.text() == "ui1.scarf.rl.ac.uk"
    assert dialog._queue_system_combo.currentText() == "SLURM"
    assert dialog._keyword_mode_combo.currentText() == "Scheduler job (recommended)"
    assert dialog._keyword_mode_combo.currentData() == "scheduler"
    assert [
        dialog._keyword_mode_combo.itemData(i) for i in range(dialog._keyword_mode_combo.count())
    ] == [
        "scheduler",
        "direct",
    ]
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
    dialog._keyword_mode_combo.setCurrentText("Login node")
    dialog._keyword_shell_init_edit.setText("module load CASTEP")

    dialog._accept_and_save()

    loaded = load_cluster_configs(settings)
    assert len(loaded) == 2
    alpha = next(config for config in loaded if config.name == "alpha")
    assert alpha.hostname == "alpha.example"
    assert alpha.auth_method == "password"
    assert alpha.castep_executable == "castep.mpi"
    assert alpha.keyword_execution_mode == "direct"
    assert alpha.keyword_shell_init == "module load CASTEP"
    dialog.close()


def test_cluster_setup_dialog_defaults_new_clusters_to_scheduler_keyword_generation(qapp) -> None:
    """The login node is opt-in: running CASTEP there is forbidden on some sites."""

    dialog = ClusterSetupDialog(settings=_FakeSettings())

    dialog._set_keyword_mode("direct")
    assert dialog._keyword_mode_combo.currentData() == "direct"
    dialog._add_cluster()
    assert dialog._keyword_mode_combo.currentData() == "scheduler"
    assert dialog._updated_selected_config().keyword_execution_mode == "scheduler"

    # An unrecognised stored value falls back to the scheduler too.
    dialog._set_keyword_mode("mystery")
    assert dialog._keyword_mode_combo.currentData() == "scheduler"
    dialog.close()


def test_cluster_setup_dialog_addqueue_does_not_require_template(qapp) -> None:
    settings = _FakeSettings()
    dialog = ClusterSetupDialog(settings=settings)

    dialog._name_edit.setText("glamdring")
    dialog._hostname_edit.setText("glamdring.physics.ox.ac.uk")
    dialog._username_edit.setText("alice")
    dialog._queue_system_combo.setCurrentText("ADDQUEUE")
    dialog._walltime_edit.setText("")
    dialog._template_path_edit.setText("")
    dialog._submit_command_edit.setText("/bin/addqueue")

    dialog._accept_and_save()

    loaded = load_cluster_configs(settings)
    saved = next(config for config in loaded if config.name == "glamdring")
    assert saved.queue_system == "ADDQUEUE"
    assert saved.default_walltime == ""
    dialog.close()


def test_cluster_setup_dialog_slurm_template_is_optional(qapp) -> None:
    """A blank SLURM template means the built-in one, so it saves without one."""

    settings = _FakeSettings()
    dialog = ClusterSetupDialog(settings=settings)

    dialog._name_edit.setText("alpha")
    dialog._hostname_edit.setText("alpha.example")
    dialog._username_edit.setText("alice")
    dialog._template_path_edit.setText("")
    assert dialog._template_path_edit.placeholderText() == "Built-in SLURM template"

    dialog._accept_and_save()

    saved = next(config for config in load_cluster_configs(settings) if config.name == "alpha")
    assert saved.template_script_path == ""
    dialog.close()


def test_cluster_setup_dialog_refuses_names_that_differ_only_in_case(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = ClusterSetupDialog(settings=_FakeSettings())
    warnings: list[str] = []
    monkeypatch.setattr(
        dialog_module.QMessageBox, "warning", lambda _p, _t, message: warnings.append(message)
    )

    dialog._name_edit.setText("redwood")
    dialog._accept_and_save()

    assert warnings and "Duplicate cluster name: Redwood" in warnings[0]
    dialog.close()


def test_cluster_setup_dialog_loads_redwood_jump_host_defaults(qapp) -> None:
    dialog = ClusterSetupDialog(settings=_FakeSettings())
    dialog._select_cluster("Redwood")

    assert dialog._hostname_edit.text() == "glamdring.physics.ox.ac.uk"
    assert dialog._auth_combo.currentText() == "password"
    assert dialog._jump_enabled_check.isChecked() is True
    assert dialog._jump_hostname_edit.text() == "bastion.physics.ox.ac.uk"
    assert dialog._jump_username_edit.text() == "physics_username"
    assert dialog._jump_auth_combo.currentText() == "password"
    assert dialog._keychain_check.isChecked() is False
    dialog.close()


def test_cluster_setup_dialog_groups_settings_into_tabs(qapp) -> None:
    dialog = ClusterSetupDialog(settings=_FakeSettings())

    assert [dialog._tabs.tabText(index) for index in range(dialog._tabs.count())] == [
        "Connection",
        "Scheduler",
        "Queues",
        "Keyword generation",
    ]
    assert dialog._ssh_form.indexOf(dialog._hostname_edit) >= 0
    assert dialog._jump_form.indexOf(dialog._jump_key_path_row) >= 0
    assert dialog._cluster_form.indexOf(dialog._working_dir_edit) >= 0
    assert dialog._tabs.widget(2).isAncestorOf(dialog._queue_table)
    assert dialog._tabs.objectName() == "mufinderTabs"
    dialog.close()


def test_cluster_setup_dialog_uses_transparent_inline_rows(qapp) -> None:
    dialog = ClusterSetupDialog(settings=_FakeSettings())

    assert dialog._key_path_row.objectName() == "mufinderInlineGroup"
    assert dialog._jump_key_path_row.objectName() == "mufinderInlineGroup"
    assert dialog._keychain_row.objectName() == "mufinderInlineGroup"
    assert dialog._template_path_row.objectName() == "mufinderInlineGroup"
    # The theme styles inline rows; a dialog-local sheet hid the checkbox indicators.
    assert dialog.styleSheet() == ""
    dialog.close()


def test_cluster_setup_dialog_shows_keychain_option_for_macos_redwood(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keychain_module.sys, "platform", "darwin")
    dialog = ClusterSetupDialog(settings=_FakeSettings())
    dialog._select_cluster("Redwood")

    assert dialog._keychain_row.isHidden() is False
    assert dialog._keychain_check.text() == "Use macOS Keychain passwords"
    assert dialog._keychain_help_button.toolTip() == "Show macOS Keychain setup commands"

    messages: list[str] = []
    monkeypatch.setattr(
        dialog_module.QMessageBox,
        "information",
        lambda _parent, _title, message: messages.append(message),
    )
    dialog._show_keychain_help()

    assert "security add-generic-password -a physics_username -s bastion -w" in messages[0]
    assert "security add-generic-password -a username -s glamdring -w" in messages[0]
    dialog.close()


def test_cluster_setup_dialog_shows_keyring_option_for_linux_redwood(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keychain_module.sys, "platform", "linux")
    dialog = ClusterSetupDialog(settings=_FakeSettings())
    dialog._select_cluster("Redwood")

    assert dialog._keychain_row.isHidden() is False
    assert dialog._keychain_check.text() == "Use system keyring passwords"
    assert dialog._keychain_help_button.toolTip() == "Show system keyring setup commands"

    shown: list[tuple[str, str]] = []
    monkeypatch.setattr(
        dialog_module.QMessageBox,
        "information",
        lambda _parent, title, message: shown.append((title, message)),
    )
    dialog._show_keychain_help()

    assert shown[0][0] == "System keyring setup"
    assert "secret-tool store" in shown[0][1]
    assert "service bastion account physics_username" in shown[0][1]
    assert "service glamdring account username" in shown[0][1]
    dialog.close()


def test_cluster_setup_dialog_hides_keychain_option_without_supported_keyring(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(keychain_module.sys, "platform", "win32")
    dialog = ClusterSetupDialog(settings=_FakeSettings())
    dialog._select_cluster("Redwood")
    dialog._keychain_check.setChecked(True)
    dialog._refresh_keychain_controls()

    assert dialog._keychain_row.isHidden() is True
    assert dialog._keychain_check.isChecked() is False
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


@pytest.mark.parametrize("platform", ["darwin", "linux"])
def test_cluster_setup_dialog_saves_keychain_setting_for_eligible_redwood(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
    platform: str,
) -> None:
    monkeypatch.setattr(keychain_module.sys, "platform", platform)
    settings = _FakeSettings()
    dialog = ClusterSetupDialog(settings=settings)
    dialog._select_cluster("Redwood")
    dialog._keychain_check.setChecked(True)

    dialog._accept_and_save()

    saved = next(config for config in load_cluster_configs(settings) if config.name == "Redwood")
    assert saved.use_macos_keychain_passwords is True
    dialog.close()


def test_cluster_setup_dialog_test_connection_uses_ssh_client(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = ClusterSetupDialog(settings=_FakeSettings())

    infos: list[str] = []
    monkeypatch.setattr(dialog_module, "SSHConnection", _FakeConnection)
    monkeypatch.setattr(
        dialog_module, "prompt_for_cluster_credentials", lambda *_args, **_kwargs: SSHCredentials()
    )
    monkeypatch.setattr(
        dialog_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    dialog._hostname_edit.setText("alpha.example")
    dialog._username_edit.setText("alice")
    dialog._template_path_edit.setText("/templates/slurm.sh")
    dialog._test_connection()

    assert pump_until(lambda: bool(infos), timeout=5.0)
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

    assert pump_until(lambda: bool(captured_passwords), timeout=5.0)
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
                raise SSHUnknownHostKeyError(
                    hostname="alpha.example",
                    port=22,
                    algorithm="ssh-ed25519",
                    fingerprint_sha256="SHA256:test",
                    key_base64="dGVzdA==",
                )
            super().connect(password=password)

    monkeypatch.setattr(dialog_module, "SSHConnection", _UnknownHostConnection)
    monkeypatch.setattr(
        dialog_module,
        "prompt_to_trust_host_key",
        lambda *_args, **_kwargs: prompted.append("yes") or True,
    )
    monkeypatch.setattr(
        dialog_module, "prompt_for_cluster_credentials", lambda *_args, **_kwargs: SSHCredentials()
    )
    monkeypatch.setattr(
        dialog_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    dialog._hostname_edit.setText("alpha.example")
    dialog._username_edit.setText("alice")
    dialog._template_path_edit.setText("/templates/slurm.sh")
    dialog._test_connection()

    assert pump_until(lambda: bool(infos), timeout=5.0)
    assert prompted == ["yes"]
    assert infos == ["Connection successful"]
    assert (
        dialog._settings.value("trusted_host_keys/SCARF/alpha.example/algorithm", "", str)
        == "ssh-ed25519"
    )
    dialog.close()


def test_cluster_setup_dialog_test_connection_trusts_jump_and_target_host_keys(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = ClusterSetupDialog(settings=_FakeSettings())

    infos: list[str] = []
    prompted: list[str] = []

    class _UnknownHostConnection(_FakeConnection):
        attempts = 0

        def connect(self, password: str | None = None, **_kwargs) -> None:
            _ = password
            self.__class__.attempts += 1
            if self.__class__.attempts == 1:
                raise SSHUnknownHostKeyError(
                    hostname="bastion.physics.ox.ac.uk",
                    port=22,
                    algorithm="ssh-ed25519",
                    fingerprint_sha256="SHA256:bastion",
                    key_base64="YmFzdGlvbg==",
                    endpoint="jump",
                )
            if self.__class__.attempts == 2:
                raise SSHUnknownHostKeyError(
                    hostname="glamdring.physics.ox.ac.uk",
                    port=22,
                    algorithm="ssh-ed25519",
                    fingerprint_sha256="SHA256:glamdring",
                    key_base64="Z2xhbWRyaW5n",
                    endpoint="target",
                )
            self.connected = True

    monkeypatch.setattr(dialog_module, "SSHConnection", _UnknownHostConnection)
    monkeypatch.setattr(
        dialog_module,
        "prompt_to_trust_host_key",
        lambda _parent, _config, exc: prompted.append(exc.endpoint) or True,
    )
    monkeypatch.setattr(
        dialog_module,
        "prompt_for_cluster_credentials",
        lambda *_args, **_kwargs: SSHCredentials(target_secret="glam", jump_secret="phys"),
    )
    monkeypatch.setattr(
        dialog_module.QMessageBox,
        "information",
        lambda _parent, title, _message: infos.append(title),
    )

    dialog._select_cluster("Redwood")
    dialog._test_connection()

    assert pump_until(lambda: bool(infos), timeout=5.0)
    assert prompted == ["jump", "target"]
    assert infos == ["Connection successful"]
    assert (
        dialog._settings.value(
            "trusted_host_keys/Redwood/jump/bastion.physics.ox.ac.uk/algorithm",
            "",
            str,
        )
        == "ssh-ed25519"
    )
    assert (
        dialog._settings.value(
            "trusted_host_keys/Redwood/glamdring.physics.ox.ac.uk/algorithm",
            "",
            str,
        )
        == "ssh-ed25519"
    )
    dialog.close()


def test_cluster_setup_dialog_remove_cluster_and_auth_controls(qapp) -> None:
    dialog = ClusterSetupDialog(settings=_FakeSettings())

    dialog._add_cluster()
    assert dialog._cluster_list.count() >= 2

    dialog._auth_combo.setCurrentText("password")
    dialog._refresh_auth_controls()
    assert dialog._key_path_edit.isEnabled() is False

    dialog._remove_cluster()
    assert dialog._cluster_list.count() >= 1
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
    answered_choices,
) -> None:
    dialog = ClusterSetupDialog(settings=_FakeSettings())

    # The stub answers with the default, Update directory.
    dialog._username_edit.setText("alice")
    dialog._working_dir_edit.setText("/work4/isis/username")
    dialog._maybe_sync_working_dir_for_username()

    assert dialog._working_dir_edit.text() == "/work4/isis/alice"
    (prompt,) = answered_choices
    assert prompt["choices"] == [("update", "Update directory"), ("keep", "Keep directory")]
    assert (prompt["default"], prompt["escape"]) == ("update", "keep")

    monkeypatch.setattr(workflow_table_utils, "confirm_choice", lambda *_a, **_k: "keep")
    dialog._username_edit.setText("bob")
    dialog._maybe_sync_working_dir_for_username()

    assert dialog._working_dir_edit.text() == "/work4/isis/alice"
    dialog.close()


def test_cluster_setup_dialog_saves_default_mem_per_cpu_for_slurm(qapp) -> None:
    settings = _FakeSettings()
    dialog = ClusterSetupDialog(settings=settings)

    dialog._name_edit.setText("alpha")
    dialog._hostname_edit.setText("alpha.example")
    dialog._username_edit.setText("alice")
    dialog._template_path_edit.setText("/templates/slurm.sh")
    dialog._default_mem_per_cpu_check.setChecked(True)
    dialog._default_mem_per_cpu_edit.setValue(6.0)

    dialog._accept_and_save()

    alpha = next(config for config in load_cluster_configs(settings) if config.name == "alpha")
    assert alpha.default_mem_per_cpu_gb == 6.0
    dialog.close()


def test_cluster_setup_dialog_default_mem_per_cpu_is_unset_by_default(qapp) -> None:
    settings = _FakeSettings()
    dialog = ClusterSetupDialog(settings=settings)

    dialog._name_edit.setText("alpha")
    dialog._hostname_edit.setText("alpha.example")
    dialog._username_edit.setText("alice")
    dialog._template_path_edit.setText("/templates/slurm.sh")

    assert dialog._default_mem_per_cpu_check.isChecked() is False
    assert dialog._default_mem_per_cpu_edit.isEnabled() is False

    dialog._accept_and_save()

    alpha = next(config for config in load_cluster_configs(settings) if config.name == "alpha")
    assert alpha.default_mem_per_cpu_gb is None
    dialog.close()


def test_cluster_setup_dialog_hides_default_mem_per_cpu_for_addqueue(qapp) -> None:
    settings = _FakeSettings()
    dialog = ClusterSetupDialog(settings=settings)

    dialog._name_edit.setText("glamdring")
    dialog._hostname_edit.setText("glamdring.physics.ox.ac.uk")
    dialog._username_edit.setText("alice")
    dialog._default_mem_per_cpu_check.setChecked(True)
    dialog._default_mem_per_cpu_edit.setValue(6.0)
    dialog._queue_system_combo.setCurrentText("ADDQUEUE")
    dialog._walltime_edit.setText("")
    dialog._submit_command_edit.setText("/bin/addqueue")

    assert dialog._default_mem_per_cpu_check.isHidden() is True
    assert dialog._default_mem_per_cpu_edit.isHidden() is True

    dialog._accept_and_save()

    saved = next(config for config in load_cluster_configs(settings) if config.name == "glamdring")
    assert saved.default_mem_per_cpu_gb is None
    dialog.close()


def test_cluster_setup_dialog_round_trips_default_mem_per_cpu(qapp) -> None:
    settings = _FakeSettings()
    dialog = ClusterSetupDialog(settings=settings)
    dialog._name_edit.setText("alpha")
    dialog._hostname_edit.setText("alpha.example")
    dialog._username_edit.setText("alice")
    dialog._template_path_edit.setText("/templates/slurm.sh")
    dialog._default_mem_per_cpu_check.setChecked(True)
    dialog._default_mem_per_cpu_edit.setValue(7.5)
    dialog._accept_and_save()
    dialog.close()

    reopened = ClusterSetupDialog(settings=settings)
    reopened._select_cluster("alpha")

    assert reopened._default_mem_per_cpu_check.isChecked() is True
    assert reopened._default_mem_per_cpu_edit.value() == 7.5
    reopened.close()


def _fill_valid_cluster(dialog: ClusterSetupDialog, name: str = "PSI") -> None:
    dialog._name_edit.setText(name)
    dialog._hostname_edit.setText("login.psi.example")
    dialog._username_edit.setText("dana")


def _set_queue_cell(dialog: ClusterSetupDialog, row: int, column: int, text: str) -> None:
    dialog._queue_table.item(row, column).setText(text)


def test_cluster_setup_dialog_saves_queues_with_overrides(qapp) -> None:
    from mufinder.domain.cluster import ClusterQueue

    settings = _FakeSettings()
    dialog = ClusterSetupDialog(settings=settings)
    _fill_valid_cluster(dialog)

    dialog._add_queue()
    _set_queue_cell(dialog, 0, dialog_module._QUEUE_NAME_COLUMN, "hourly")
    _set_queue_cell(dialog, 0, dialog_module._QUEUE_WALLTIME_COLUMN, "00:59:00")
    _set_queue_cell(dialog, 0, dialog_module._QUEUE_MAX_WALLTIME_COLUMN, "01:00:00")
    dialog._add_queue()
    _set_queue_cell(dialog, 1, dialog_module._QUEUE_NAME_COLUMN, "daily")
    _set_queue_cell(dialog, 1, dialog_module._QUEUE_WALLTIME_COLUMN, "12:00:00")
    _set_queue_cell(dialog, 1, dialog_module._QUEUE_MAX_WALLTIME_COLUMN, "1-00:00:00")
    _set_queue_cell(dialog, 1, dialog_module._QUEUE_CORES_COLUMN, "88")
    _set_queue_cell(dialog, 1, dialog_module._QUEUE_MEMORY_COLUMN, "8")
    _set_queue_cell(dialog, 1, dialog_module._QUEUE_EXECUTABLE_COLUMN, "/opt/castep/castep.mpi")
    _set_queue_cell(dialog, 1, dialog_module._QUEUE_TEMPLATE_COLUMN, "/templates/daily.sh")
    assert [
        dialog._default_queue_combo.itemText(i) for i in range(dialog._default_queue_combo.count())
    ] == [
        "hourly",
        "daily",
    ]
    dialog._default_queue_combo.setCurrentText("daily")

    dialog._accept_and_save()

    saved = next(config for config in load_cluster_configs(settings) if config.name == "PSI")
    assert saved.default_queue_name == "daily"
    assert saved.queues == (
        ClusterQueue(name="hourly", default_walltime="00:59:00", max_walltime="01:00:00"),
        ClusterQueue(
            name="daily",
            default_walltime="12:00:00",
            max_walltime="1-00:00:00",
            cores_per_node=88,
            default_mem_per_cpu_gb=8.0,
            castep_executable="/opt/castep/castep.mpi",
            template_script_path="/templates/daily.sh",
        ),
    )
    dialog.close()


def test_cluster_setup_dialog_shows_a_single_queue_profile_as_one_row(qapp) -> None:
    dialog = ClusterSetupDialog(settings=_FakeSettings())

    dialog._select_cluster("Redwood")

    assert dialog._queue_table.rowCount() == 1
    assert dialog._queue_table.item(0, dialog_module._QUEUE_NAME_COLUMN).text() == "redwood"
    assert dialog._default_queue_combo.currentText() == "redwood"
    assert dialog._queue_table.isColumnHidden(dialog_module._QUEUE_TEMPLATE_COLUMN) is True
    memory_header = dialog._queue_table.horizontalHeaderItem(dialog_module._QUEUE_MEMORY_COLUMN)
    assert memory_header.text() == "Memory per core (GB)"
    dialog.close()


def test_cluster_setup_dialog_addqueue_queue_memory_is_per_core(qapp) -> None:
    settings = _FakeSettings()
    dialog = ClusterSetupDialog(settings=settings)
    dialog._select_cluster("Redwood")
    dialog._username_edit.setText("charlie")
    dialog._jump_username_edit.setText("charlie")
    _set_queue_cell(dialog, 0, dialog_module._QUEUE_MEMORY_COLUMN, "15")

    dialog._accept_and_save()

    saved = next(config for config in load_cluster_configs(settings) if config.name == "Redwood")
    (queue,) = saved.queues
    assert (queue.name, queue.default_memory_gb, queue.default_mem_per_cpu_gb) == (
        "redwood",
        15.0,
        None,
    )
    dialog.close()


@pytest.mark.parametrize(
    ("column", "text", "expected"),
    [
        (dialog_module._QUEUE_NAME_COLUMN, "two words", "is not a queue name"),
        (dialog_module._QUEUE_NAME_COLUMN, "", "has no name"),
        (dialog_module._QUEUE_MAX_WALLTIME_COLUMN, "forever", "is not a walltime"),
        (dialog_module._QUEUE_WALLTIME_COLUMN, "2-00:00:00", "longer than the 1-00:00:00 maximum"),
        (
            dialog_module._QUEUE_MAX_WALLTIME_COLUMN,
            "06:00:00",
            "uses the cluster's default walltime",
        ),
        (
            dialog_module._QUEUE_CORES_COLUMN,
            "many",
            "cores per node must be a positive whole number",
        ),
        (dialog_module._QUEUE_MEMORY_COLUMN, "-1", "memory must be a positive number of GB"),
    ],
)
def test_cluster_setup_dialog_refuses_invalid_queues(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
    column: int,
    text: str,
    expected: str,
) -> None:
    settings = _FakeSettings()
    dialog = ClusterSetupDialog(settings=settings)
    warnings: list[str] = []
    monkeypatch.setattr(
        dialog_module.QMessageBox, "warning", lambda _p, _t, message: warnings.append(message)
    )
    _fill_valid_cluster(dialog)
    dialog._add_queue()
    _set_queue_cell(dialog, 0, dialog_module._QUEUE_MAX_WALLTIME_COLUMN, "1-00:00:00")
    _set_queue_cell(dialog, 0, column, text)

    dialog._accept_and_save()

    assert len(warnings) == 1 and expected in warnings[0]
    assert "PSI" not in [config.name for config in load_cluster_configs(settings)]
    dialog.close()


def test_cluster_setup_dialog_refuses_duplicate_queue_names(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog = ClusterSetupDialog(settings=_FakeSettings())
    warnings: list[str] = []
    monkeypatch.setattr(
        dialog_module.QMessageBox, "warning", lambda _p, _t, message: warnings.append(message)
    )
    _fill_valid_cluster(dialog)
    dialog._add_queue()
    dialog._add_queue()
    _set_queue_cell(dialog, 1, dialog_module._QUEUE_NAME_COLUMN, "queue")

    dialog._accept_and_save()

    assert warnings and "The queue 'queue' is listed twice." in warnings[0]
    dialog.close()


def test_cluster_setup_dialog_keeps_edits_when_switching_clusters(qapp) -> None:
    settings = _FakeSettings()
    dialog = ClusterSetupDialog(settings=settings)
    dialog._username_edit.setText("scarf-user")
    dialog._add_queue()

    dialog._select_cluster("Redwood")
    dialog._select_cluster("SCARF")

    assert dialog._username_edit.text() == "scarf-user"
    assert dialog._queue_table.rowCount() == 1
    dialog.close()


def test_cluster_setup_dialog_stays_on_a_cluster_with_invalid_queues(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dialog = ClusterSetupDialog(settings=_FakeSettings())
    warnings: list[str] = []
    monkeypatch.setattr(
        dialog_module.QMessageBox, "warning", lambda _p, title, _m: warnings.append(title)
    )
    dialog._add_queue()
    _set_queue_cell(dialog, 0, dialog_module._QUEUE_CORES_COLUMN, "lots")

    dialog._select_cluster("Redwood")

    assert warnings == ["Invalid queue settings"]
    assert dialog._name_edit.text() == "SCARF"
    assert dialog._cluster_list.currentItem().text() == "SCARF"
    assert dialog._queue_table.item(0, dialog_module._QUEUE_CORES_COLUMN).text() == "lots"
    dialog.close()


def test_cluster_setup_dialog_remove_queue_is_confirmed(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
    confirmed_destructive_actions: list[dict[str, str]],
) -> None:
    from mufinder.app import workflow_table_utils

    dialog = ClusterSetupDialog(settings=_FakeSettings())
    dialog._select_cluster("Redwood")
    dialog._queue_table.selectRow(0)

    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *_a, **_k: False)
    dialog._remove_queue()
    assert dialog._queue_table.rowCount() == 1

    monkeypatch.undo()
    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *_a, **_k: True)
    dialog._queue_table.selectRow(0)
    dialog._remove_queue()
    assert dialog._queue_table.rowCount() == 0
    assert dialog._default_queue_combo.isEnabled() is False
    dialog.close()


def test_cluster_setup_dialog_remove_cluster_is_confirmed(
    qapp,
    monkeypatch: pytest.MonkeyPatch,
    confirmed_destructive_actions: list[dict[str, str]],
) -> None:
    from mufinder.app import workflow_table_utils

    dialog = ClusterSetupDialog(settings=_FakeSettings())

    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *_a, **_k: False)
    dialog._remove_cluster()
    assert dialog._cluster_list.count() == 2

    monkeypatch.setattr(workflow_table_utils, "confirm_destructive", lambda *_a, **_k: True)
    dialog._remove_cluster()
    assert [dialog._cluster_list.item(i).text() for i in range(dialog._cluster_list.count())] == [
        "Redwood"
    ]
    assert dialog._name_edit.text() == "Redwood"
    dialog.close()


@pytest.mark.parametrize("key", ["delete", "backspace"])
def test_cluster_setup_dialog_delete_keys_remove_the_selected_cluster_and_queue(
    qapp,
    key: str,
    confirmed_destructive_actions: list[dict[str, str]],
) -> None:
    from PySide6.QtCore import Qt
    from PySide6.QtGui import QKeySequence

    dialog = ClusterSetupDialog(settings=_FakeSettings())
    sequence = (
        QKeySequence(QKeySequence.StandardKey.Delete)
        if key == "delete"
        else QKeySequence(Qt.Key.Key_Backspace)
    )

    list_shortcut = next(s for s in dialog._cluster_list_shortcuts if s.key() == sequence)
    assert list_shortcut.context() == Qt.ShortcutContext.WidgetShortcut
    list_shortcut.activated.emit()
    assert dialog._cluster_list.count() == 1

    dialog._queue_table.selectRow(0)
    queue_shortcut = next(s for s in dialog._queue_table_shortcuts if s.key() == sequence)
    assert queue_shortcut.context() == Qt.ShortcutContext.WidgetShortcut
    queue_shortcut.activated.emit()
    assert dialog._queue_table.rowCount() == 0
    assert [prompt["title"] for prompt in confirmed_destructive_actions] == [
        "Remove cluster",
        "Remove queue",
    ]
    dialog.close()


def test_cluster_setup_dialog_duplicates_a_cluster(qapp) -> None:
    settings = _FakeSettings()
    dialog = ClusterSetupDialog(settings=settings)
    dialog._select_cluster("Redwood")

    dialog._duplicate_cluster()

    assert dialog._name_edit.text() == "Redwood copy"
    assert dialog._hostname_edit.text() == "glamdring.physics.ox.ac.uk"
    assert dialog._queue_table.rowCount() == 1
    assert [dialog._cluster_list.item(i).text() for i in range(dialog._cluster_list.count())] == [
        "SCARF",
        "Redwood",
        "Redwood copy",
    ]
    dialog.close()


def test_cluster_setup_dialog_list_follows_name_edits(qapp) -> None:
    dialog = ClusterSetupDialog(settings=_FakeSettings())

    dialog._name_edit.setFocus()
    dialog._name_edit.selectAll()
    from PySide6.QtTest import QTest

    QTest.keyClicks(dialog._name_edit, "PSI")

    assert dialog._cluster_list.item(0).text() == "PSI"
    dialog.close()


def test_cluster_setup_dialog_renaming_the_default_queue_keeps_it_default(qapp) -> None:
    settings = _FakeSettings()
    dialog = ClusterSetupDialog(settings=settings)
    _fill_valid_cluster(dialog)
    dialog._add_queue()
    dialog._add_queue()
    _set_queue_cell(dialog, 0, dialog_module._QUEUE_NAME_COLUMN, "hourly")
    _set_queue_cell(dialog, 1, dialog_module._QUEUE_NAME_COLUMN, "daily")
    dialog._default_queue_combo.setCurrentText("daily")

    _set_queue_cell(dialog, 1, dialog_module._QUEUE_NAME_COLUMN, "day")
    assert dialog._default_queue_combo.currentText() == "day"

    # Renaming another queue leaves the default alone.
    _set_queue_cell(dialog, 0, dialog_module._QUEUE_NAME_COLUMN, "hour")
    assert dialog._default_queue_combo.currentText() == "day"

    dialog._accept_and_save()
    saved = next(config for config in load_cluster_configs(settings) if config.name == "PSI")
    assert saved.default_queue_name == "day"
    assert [queue.name for queue in saved.queues] == ["hour", "day"]
    dialog.close()
