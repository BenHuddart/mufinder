"""Dialog for configuring remote HPC cluster connections."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtWidgets import (
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QInputDialog,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.cluster_ui_utils import (
    format_cluster_connection_error,
    prompt_to_trust_host_key,
    show_detailed_warning,
)
from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.cluster_setup import ClusterConfig, load_cluster_configs, save_cluster_configs
from mufinder.services.ssh_client import SSHConnection, SSHUnknownHostKeyError, save_trusted_host_key


class ClusterSetupDialog(QDialog):
    """Manage persisted cluster configurations."""

    def __init__(self, *, settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._configs: list[ClusterConfig] = load_cluster_configs(settings)
        self._last_username = ""

        self.setWindowTitle("Remote Cluster Connections")
        self.resize(760, 520)

        layout = QVBoxLayout(self)

        top_row = QHBoxLayout()
        top_row.addWidget(QLabel("Saved clusters"))
        self._cluster_combo = QComboBox(self)
        top_row.addWidget(self._cluster_combo, 1)
        self._add_button = QPushButton("Add", self)
        self._remove_button = QPushButton("Remove", self)
        top_row.addWidget(self._add_button)
        top_row.addWidget(self._remove_button)
        layout.addLayout(top_row)

        form_group = QGroupBox("Connection Settings", self)
        form = QFormLayout(form_group)

        self._name_edit = QLineEdit(form_group)
        self._hostname_edit = QLineEdit(form_group)
        self._port_spin = QSpinBox(form_group)
        self._port_spin.setRange(1, 65535)
        self._port_spin.setValue(22)
        self._username_edit = QLineEdit(form_group)

        self._auth_combo = QComboBox(form_group)
        self._auth_combo.addItems(["key", "password"])

        self._key_path_edit = QLineEdit(form_group)
        key_row = QWidget(form_group)
        key_layout = QHBoxLayout(key_row)
        key_layout.setContentsMargins(0, 0, 0, 0)
        key_layout.addWidget(self._key_path_edit, 1)
        self._key_browse_button = QPushButton("Browse", key_row)
        key_layout.addWidget(self._key_browse_button)

        self._working_dir_edit = QLineEdit(form_group)

        self._queue_combo = QComboBox(form_group)
        self._queue_combo.addItems(["SLURM", "ADDQUEUE"])

        self._cores_per_node_spin = QSpinBox(form_group)
        self._cores_per_node_spin.setRange(1, 4096)
        self._cores_per_node_spin.setValue(1)

        self._walltime_edit = QLineEdit(form_group)
        self._walltime_edit.setPlaceholderText("HH:MM:SS")

        self._template_path_edit = QLineEdit(form_group)
        template_row = QWidget(form_group)
        template_layout = QHBoxLayout(template_row)
        template_layout.setContentsMargins(0, 0, 0, 0)
        template_layout.addWidget(self._template_path_edit, 1)
        self._template_browse_button = QPushButton("Browse", template_row)
        template_layout.addWidget(self._template_browse_button)

        self._castep_exec_edit = QLineEdit(form_group)
        self._default_queue_edit = QLineEdit(form_group)
        self._default_memory_spin = DecimalLineEdit(form_group)
        self._default_memory_spin.setRange(0.1, 2048.0)
        self._default_memory_spin.setDecimals(2)
        self._default_memory_spin.setSingleStep(0.5)
        self._default_memory_spin.setValue(7.5)
        self._submit_command_edit = QLineEdit(form_group)
        self._queue_check_command_edit = QLineEdit(form_group)
        self._cancel_command_edit = QLineEdit(form_group)
        self._keyword_mode_combo = QComboBox(form_group)
        self._keyword_mode_combo.addItems(["direct", "scheduler"])
        self._keyword_shell_init_edit = QLineEdit(form_group)
        self._keyword_python_edit = QLineEdit(form_group)
        self._keyword_python_edit.setPlaceholderText("python3")

        form.addRow("Cluster name", self._name_edit)
        form.addRow("Host name", self._hostname_edit)
        form.addRow("Port", self._port_spin)
        form.addRow("Username", self._username_edit)
        form.addRow("Authentication method", self._auth_combo)
        form.addRow("SSH key path", key_row)
        form.addRow("Default remote working directory", self._working_dir_edit)
        form.addRow("Queue system", self._queue_combo)
        form.addRow("Cores per node", self._cores_per_node_spin)
        form.addRow("Default walltime", self._walltime_edit)
        self._template_label = QLabel("Template script", form_group)
        form.addRow(self._template_label, template_row)
        form.addRow("CASTEP executable", self._castep_exec_edit)
        self._default_queue_label = QLabel("Default queue / partition", form_group)
        form.addRow(self._default_queue_label, self._default_queue_edit)
        self._default_memory_label = QLabel("Default memory per core (GB)", form_group)
        form.addRow(self._default_memory_label, self._default_memory_spin)
        form.addRow("Submit command path", self._submit_command_edit)
        form.addRow("Queue check command path", self._queue_check_command_edit)
        form.addRow("Cancel command path", self._cancel_command_edit)
        self._keyword_mode_label = QLabel("Keyword generation mode", form_group)
        form.addRow(self._keyword_mode_label, self._keyword_mode_combo)
        self._keyword_shell_init_label = QLabel("Keyword shell init", form_group)
        form.addRow(self._keyword_shell_init_label, self._keyword_shell_init_edit)
        self._keyword_python_label = QLabel("Keyword Python command", form_group)
        form.addRow(self._keyword_python_label, self._keyword_python_edit)

        layout.addWidget(form_group)

        actions_row = QHBoxLayout()
        self._test_button = QPushButton("Test Connection", self)
        actions_row.addWidget(self._test_button)
        actions_row.addStretch(1)
        layout.addLayout(actions_row)

        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        layout.addWidget(self._button_box)

        self._cluster_combo.currentIndexChanged.connect(lambda *_: self._load_selected_config())
        self._add_button.clicked.connect(self._add_cluster)
        self._remove_button.clicked.connect(self._remove_cluster)
        self._auth_combo.currentTextChanged.connect(lambda *_: self._refresh_auth_controls())
        self._queue_combo.currentTextChanged.connect(lambda *_: self._refresh_scheduler_controls())
        self._keyword_mode_combo.currentTextChanged.connect(lambda *_: self._refresh_keyword_controls())
        self._username_edit.editingFinished.connect(self._maybe_sync_working_dir_for_username)
        self._key_browse_button.clicked.connect(self._browse_key_path)
        self._template_browse_button.clicked.connect(self._browse_template_path)
        self._test_button.clicked.connect(self._test_connection)

        self._button_box.accepted.connect(self._accept_and_save)
        self._button_box.rejected.connect(self.reject)

        self._populate_cluster_combo()
        self._load_selected_config()

    def configs(self) -> list[ClusterConfig]:
        return list(self._configs)

    def _populate_cluster_combo(self) -> None:
        self._cluster_combo.blockSignals(True)
        self._cluster_combo.clear()
        for config in self._configs:
            self._cluster_combo.addItem(config.name)
        self._cluster_combo.blockSignals(False)

        if self._cluster_combo.count() == 0:
            self._add_cluster()

    def _selected_index(self) -> int:
        return self._cluster_combo.currentIndex()

    def _selected_config(self) -> ClusterConfig | None:
        index = self._selected_index()
        if index < 0 or index >= len(self._configs):
            return None
        return self._configs[index]

    def _add_cluster(self) -> None:
        existing = {config.name for config in self._configs}
        counter = 1
        while True:
            name = f"Cluster {counter}"
            if name not in existing:
                break
            counter += 1

        self._configs.append(
            ClusterConfig(
                name=name,
                hostname="",
                port=22,
                username="",
                auth_method="key",
                key_path="",
                default_working_directory="",
                queue_system="SLURM",
                cores_per_node=1,
                default_walltime="01:00:00",
                template_script_path="",
                castep_executable="castep.mpi",
                default_queue_name="",
                default_memory_gb=None,
                submit_command_path="",
                queue_check_command_path="",
                cancel_command_path="",
                keyword_execution_mode="direct",
                keyword_shell_init="",
                keyword_python_command="python3",
            )
        )
        self._populate_cluster_combo()
        self._cluster_combo.setCurrentIndex(len(self._configs) - 1)
        self._load_selected_config()

    def _remove_cluster(self) -> None:
        index = self._selected_index()
        if index < 0 or index >= len(self._configs):
            return
        self._configs.pop(index)
        self._populate_cluster_combo()
        self._load_selected_config()

    def _load_selected_config(self) -> None:
        config = self._selected_config()
        if config is None:
            self._name_edit.setText("")
            self._hostname_edit.setText("")
            self._port_spin.setValue(22)
            self._username_edit.setText("")
            self._auth_combo.setCurrentText("key")
            self._key_path_edit.setText("")
            self._working_dir_edit.setText("")
            self._queue_combo.setCurrentText("SLURM")
            self._cores_per_node_spin.setValue(1)
            self._walltime_edit.setText("01:00:00")
            self._template_path_edit.setText("")
            self._castep_exec_edit.setText("castep.mpi")
            self._default_queue_edit.setText("")
            self._default_memory_spin.setValue(7.5)
            self._submit_command_edit.setText("")
            self._queue_check_command_edit.setText("")
            self._cancel_command_edit.setText("")
            self._keyword_mode_combo.setCurrentText("direct")
            self._keyword_shell_init_edit.setText("")
            self._keyword_python_edit.setText("python3")
            self._last_username = ""
            self._refresh_auth_controls()
            self._refresh_scheduler_controls()
            self._refresh_keyword_controls()
            return

        self._name_edit.setText(config.name)
        self._hostname_edit.setText(config.hostname)
        self._port_spin.setValue(int(config.port))
        self._username_edit.setText(config.username)
        self._auth_combo.setCurrentText(config.auth_method)
        self._key_path_edit.setText(config.key_path)
        self._working_dir_edit.setText(config.default_working_directory)
        self._queue_combo.setCurrentText(config.queue_system.upper())
        self._cores_per_node_spin.setValue(int(config.cores_per_node))
        self._walltime_edit.setText(config.default_walltime)
        self._template_path_edit.setText(config.template_script_path)
        self._castep_exec_edit.setText(config.castep_executable)
        self._default_queue_edit.setText(config.default_queue_name)
        self._default_memory_spin.setValue(7.5 if config.default_memory_gb is None else float(config.default_memory_gb))
        self._submit_command_edit.setText(config.submit_command_path)
        self._queue_check_command_edit.setText(config.queue_check_command_path)
        self._cancel_command_edit.setText(config.cancel_command_path)
        self._keyword_mode_combo.setCurrentText(config.keyword_execution_mode)
        self._keyword_shell_init_edit.setText(config.keyword_shell_init)
        self._keyword_python_edit.setText(config.keyword_python_command)
        self._last_username = config.username
        self._refresh_auth_controls()
        self._refresh_scheduler_controls()
        self._refresh_keyword_controls()

    def _maybe_sync_working_dir_for_username(self) -> None:
        old_username = self._last_username.strip()
        new_username = self._username_edit.text().strip()
        self._last_username = new_username

        if not old_username or not new_username or old_username == new_username:
            return

        working_dir = self._working_dir_edit.text().strip()
        expected_prefix = f"/work4/isis/{old_username}"
        if not working_dir.startswith(expected_prefix):
            return

        answer = QMessageBox.question(
            self,
            "Update working directory?",
            "The username changed. Update the default working directory to match it?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.Yes,
        )
        if answer != QMessageBox.StandardButton.Yes:
            return

        suffix = working_dir[len(expected_prefix) :]
        self._working_dir_edit.setText(f"/work4/isis/{new_username}{suffix}")

    def _refresh_auth_controls(self) -> None:
        key_enabled = self._auth_combo.currentText().strip().lower() == "key"
        self._key_path_edit.setEnabled(key_enabled)
        self._key_browse_button.setEnabled(key_enabled)

    def _refresh_scheduler_controls(self) -> None:
        queue_name = self._queue_combo.currentText().strip().lower()
        is_slurm = queue_name == "slurm"
        is_addqueue = queue_name == "addqueue"
        self._template_label.setVisible(is_slurm)
        self._template_path_edit.setVisible(is_slurm)
        self._template_browse_button.setVisible(is_slurm)
        self._default_queue_label.setVisible(is_addqueue)
        self._default_queue_edit.setVisible(is_addqueue)
        self._default_memory_label.setVisible(is_addqueue)
        self._default_memory_spin.setVisible(is_addqueue)

    def _refresh_keyword_controls(self) -> None:
        is_scheduler = self._keyword_mode_combo.currentText().strip().lower() == "scheduler"
        self._keyword_python_label.setVisible(is_scheduler)
        self._keyword_python_edit.setVisible(is_scheduler)

    def _browse_key_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select SSH private key", str(Path.home()), "All Files (*)")
        if path:
            self._key_path_edit.setText(path)

    def _browse_template_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select submission template", str(Path.home()), "Shell Scripts (*.sh);;All Files (*)")
        if path:
            self._template_path_edit.setText(path)

    def _updated_selected_config(self) -> ClusterConfig | None:
        existing = self._selected_config()
        if existing is None:
            return None

        queue_system = self._queue_combo.currentText().strip() or "SLURM"
        walltime = self._walltime_edit.text().strip()
        if queue_system.strip().lower() != "addqueue" and not walltime:
            walltime = "01:00:00"
        if queue_system.strip().lower() == "addqueue":
            default_queue_name = self._default_queue_edit.text().strip()
            default_memory_gb: float | None = float(self._default_memory_spin.value())
        else:
            default_queue_name = ""
            default_memory_gb = None

        return replace(
            existing,
            name=self._name_edit.text().strip(),
            hostname=self._hostname_edit.text().strip(),
            port=self._port_spin.value(),
            username=self._username_edit.text().strip(),
            auth_method=self._auth_combo.currentText().strip().lower() or "key",
            key_path=self._key_path_edit.text().strip(),
            default_working_directory=self._working_dir_edit.text().strip(),
            queue_system=queue_system,
            cores_per_node=self._cores_per_node_spin.value(),
            default_walltime=walltime,
            template_script_path=self._template_path_edit.text().strip(),
            castep_executable=self._castep_exec_edit.text().strip() or "castep.mpi",
            default_queue_name=default_queue_name,
            default_memory_gb=default_memory_gb,
            submit_command_path=self._submit_command_edit.text().strip(),
            queue_check_command_path=self._queue_check_command_edit.text().strip(),
            cancel_command_path=self._cancel_command_edit.text().strip(),
            keyword_execution_mode=self._keyword_mode_combo.currentText().strip().lower() or "direct",
            keyword_shell_init=self._keyword_shell_init_edit.text().strip(),
            keyword_python_command=self._keyword_python_edit.text().strip() or "python3",
        )

    def _sync_selected_config_from_form(self) -> None:
        updated = self._updated_selected_config()
        index = self._selected_index()
        if updated is None or index < 0 or index >= len(self._configs):
            return
        self._configs[index] = updated
        self._cluster_combo.setItemText(index, updated.name or f"Cluster {index + 1}")

    def _test_connection(self) -> None:
        self._sync_selected_config_from_form()
        config = self._selected_config()
        if config is None:
            return
        if not config.hostname:
            QMessageBox.warning(self, "Missing hostname", "Enter a host name before testing the connection.")
            return
        if not config.username:
            QMessageBox.warning(self, "Missing username", "Enter a username before testing the connection.")
            return

        password: str | None = None
        if config.auth_method == "password":
            raw_password, accepted = QInputDialog.getText(
                self,
                "Cluster password",
                "Enter cluster password:",
                QLineEdit.EchoMode.Password,
            )
            if not accepted or not raw_password.strip():
                QMessageBox.warning(self, "Missing password", "Password is required for this auth method.")
                return
            password = raw_password.strip()
        elif config.auth_method == "key":
            raw_passphrase, accepted = QInputDialog.getText(
                self,
                "Key passphrase",
                "Enter key passphrase (required for encrypted private keys):",
                QLineEdit.EchoMode.Password,
            )
            if accepted and raw_passphrase.strip():
                password = raw_passphrase.strip()

        try:
            self._test_connection_with_trust(config, password=password)
        except Exception as exc:  # noqa: BLE001
            show_detailed_warning(
                self,
                "Connection failed",
                format_cluster_connection_error(config, exc),
                informative_text="Open Details for the full SSH connection error and recovery guidance.",
            )
            return

        QMessageBox.information(self, "Connection successful", "SSH connection test succeeded.")

    def _test_connection_with_trust(self, config: ClusterConfig, *, password: str | None) -> None:
        connection = SSHConnection(config, settings=self._settings)
        try:
            connection.connect(password=password)
        except SSHUnknownHostKeyError as exc:
            if not prompt_to_trust_host_key(self, config, exc):
                raise ConnectionError("Connection cancelled because the SSH host key was not trusted.") from exc
            save_trusted_host_key(
                self._settings,
                config,
                algorithm=exc.algorithm,
                key_base64=exc.key_base64,
            )
            connection = SSHConnection(config, settings=self._settings)
            connection.connect(password=password)
        finally:
            if connection.is_connected:
                connection.disconnect()

    def _validate_configs(self) -> list[str]:
        issues: list[str] = []
        seen: set[str] = set()
        for config in self._configs:
            if not config.name:
                issues.append("Cluster name is required.")
                continue
            if config.name in seen:
                issues.append(f"Duplicate cluster name: {config.name}")
            seen.add(config.name)
            if not config.hostname:
                issues.append(f"Cluster '{config.name}' is missing hostname.")
            if not config.username:
                issues.append(f"Cluster '{config.name}' is missing username.")
            if config.queue_system.strip().lower() == "slurm" and not config.template_script_path:
                issues.append(f"Cluster '{config.name}' is missing template script path.")
        return issues

    def _accept_and_save(self) -> None:
        self._sync_selected_config_from_form()
        issues = self._validate_configs()
        if issues:
            QMessageBox.warning(self, "Invalid cluster configuration", "\n".join(f"- {issue}" for issue in issues))
            return

        save_cluster_configs(self._settings, self._configs)
        self.accept()
