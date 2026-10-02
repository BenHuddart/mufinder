"""Dialog for configuring remote HPC cluster connections."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.cluster_connection_check import ClusterConnectionCheck
from mufinder.app.cluster_ui_utils import (
    format_cluster_connection_error,
    prompt_for_cluster_credentials,
    prompt_to_trust_host_key,
    show_detailed_warning,
)
from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.app.theme import style_help_button, style_inline_group
from mufinder.cluster_setup import ClusterConfig, load_cluster_configs, save_cluster_configs
from mufinder.services.keychain import is_glamdring_keychain_supported, keychain_setup_help
from mufinder.services.ssh_client import SSHConnection, SSHUnknownHostKeyError, save_trusted_host_key

#: Pre-filled default per-CPU memory request for SLURM cluster profiles. Mirrors
#: ``cluster_submission_dialog._DEFAULT_MEM_PER_CPU_GB``.
_DEFAULT_MEM_PER_CPU_GB = 6.0


class ClusterSetupDialog(QDialog):
    """Manage persisted cluster configurations."""

    def __init__(self, *, settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._configs: list[ClusterConfig] = load_cluster_configs(settings)
        self._last_username = ""

        self.setWindowTitle("Remote Cluster Connections")
        self.resize(980, 520)
        self.setStyleSheet(
            """
            #mufinderInlineGroup,
            #mufinderInlineGroup QLabel {
                background: transparent;
            }
            QCheckBox#mufinderInlineGroup {
                background: transparent;
            }
            """
        )

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
        self._form_columns = QHBoxLayout(form_group)
        self._form_columns.setSpacing(28)
        self._form_columns.setContentsMargins(14, 18, 14, 14)
        self._ssh_form = QFormLayout()
        self._cluster_form = QFormLayout()
        for form in (self._ssh_form, self._cluster_form):
            form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
            form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
            form.setHorizontalSpacing(12)
            form.setVerticalSpacing(8)
        self._form_columns.addLayout(self._ssh_form, 1)
        self._form_columns.addLayout(self._cluster_form, 1)

        self._name_edit = QLineEdit(form_group)
        self._hostname_edit = QLineEdit(form_group)
        self._port_spin = QSpinBox(form_group)
        self._port_spin.setRange(1, 65535)
        self._port_spin.setValue(22)
        self._username_edit = QLineEdit(form_group)

        self._auth_combo = QComboBox(form_group)
        self._auth_combo.addItems(["key", "password"])

        self._key_path_edit = QLineEdit(form_group)
        self._key_path_row = QWidget(form_group)
        style_inline_group(self._key_path_row)
        key_layout = QHBoxLayout(self._key_path_row)
        key_layout.setContentsMargins(0, 0, 0, 0)
        key_layout.addWidget(self._key_path_edit, 1)
        self._key_browse_button = QPushButton("Browse", self._key_path_row)
        key_layout.addWidget(self._key_browse_button)

        self._jump_enabled_check = QCheckBox("Connect through SSH jump host", form_group)
        style_inline_group(self._jump_enabled_check)
        self._jump_hostname_edit = QLineEdit(form_group)
        self._jump_port_spin = QSpinBox(form_group)
        self._jump_port_spin.setRange(1, 65535)
        self._jump_port_spin.setValue(22)
        self._jump_username_edit = QLineEdit(form_group)
        self._jump_auth_combo = QComboBox(form_group)
        self._jump_auth_combo.addItems(["key", "password"])
        self._jump_key_path_edit = QLineEdit(form_group)
        self._jump_key_path_row = QWidget(form_group)
        style_inline_group(self._jump_key_path_row)
        jump_key_layout = QHBoxLayout(self._jump_key_path_row)
        jump_key_layout.setContentsMargins(0, 0, 0, 0)
        jump_key_layout.addWidget(self._jump_key_path_edit, 1)
        self._jump_key_browse_button = QPushButton("Browse", self._jump_key_path_row)
        jump_key_layout.addWidget(self._jump_key_browse_button)
        self._keychain_row = QWidget(form_group)
        style_inline_group(self._keychain_row)
        keychain_layout = QHBoxLayout(self._keychain_row)
        keychain_layout.setContentsMargins(0, 0, 0, 0)
        self._keychain_check = QCheckBox("Use macOS Keychain passwords", self._keychain_row)
        style_inline_group(self._keychain_check)
        keychain_layout.addWidget(self._keychain_check, 1)
        self._keychain_help_button = QToolButton(self._keychain_row)
        self._keychain_help_button.setText("?")
        self._keychain_help_button.setToolTip("Show macOS Keychain setup commands")
        style_help_button(self._keychain_help_button)
        keychain_layout.addWidget(self._keychain_help_button)

        self._working_dir_edit = QLineEdit(form_group)

        self._queue_combo = QComboBox(form_group)
        self._queue_combo.addItems(["SLURM", "ADDQUEUE"])

        self._cores_per_node_spin = QSpinBox(form_group)
        self._cores_per_node_spin.setRange(1, 4096)
        self._cores_per_node_spin.setValue(1)

        self._walltime_edit = QLineEdit(form_group)
        self._walltime_edit.setPlaceholderText("HH:MM:SS")

        self._template_path_edit = QLineEdit(form_group)
        self._template_path_row = QWidget(form_group)
        style_inline_group(self._template_path_row)
        template_layout = QHBoxLayout(self._template_path_row)
        template_layout.setContentsMargins(0, 0, 0, 0)
        template_layout.addWidget(self._template_path_edit, 1)
        self._template_browse_button = QPushButton("Browse", self._template_path_row)
        template_layout.addWidget(self._template_browse_button)

        self._castep_exec_edit = QLineEdit(form_group)
        self._default_queue_edit = QLineEdit(form_group)
        self._default_memory_spin = DecimalLineEdit(form_group)
        self._default_memory_spin.setRange(0.1, 2048.0)
        self._default_memory_spin.setDecimals(2)
        self._default_memory_spin.setSingleStep(0.5)
        self._default_memory_spin.setValue(7.5)
        self._default_mem_per_cpu_check = QCheckBox(
            "Request default memory per CPU (GB)",
            form_group,
        )
        self._default_mem_per_cpu_edit = DecimalLineEdit(form_group)
        self._default_mem_per_cpu_edit.setRange(0.1, 2048.0)
        self._default_mem_per_cpu_edit.setDecimals(2)
        self._default_mem_per_cpu_edit.setSingleStep(0.5)
        self._default_mem_per_cpu_edit.setValue(_DEFAULT_MEM_PER_CPU_GB)
        self._default_mem_per_cpu_edit.setToolTip(
            "Pre-fills the submission dialog's #SBATCH --mem-per-cpu request. "
            "Leave unchecked to let the cluster's own memory default apply."
        )
        self._submit_command_edit = QLineEdit(form_group)
        self._queue_check_command_edit = QLineEdit(form_group)
        self._cancel_command_edit = QLineEdit(form_group)
        self._keyword_mode_combo = QComboBox(form_group)
        self._keyword_mode_combo.addItem("Scheduler job (recommended)", "scheduler")
        self._keyword_mode_combo.addItem("Login node", "direct")
        self._keyword_mode_combo.setToolTip(
            "Where MuFinder runs CASTEP to build the keyword catalog. A scheduler job reuses this "
            "cluster's job script and is safe where running CASTEP on the login node is forbidden or "
            "discouraged. The login node is quicker when it is permitted, because nothing waits in a queue."
        )
        self._keyword_shell_init_edit = QLineEdit(form_group)
        self._keyword_shell_init_edit.setPlaceholderText("Optional, e.g. module load CASTEP")
        self._keyword_shell_init_edit.setToolTip(
            "Extra commands to run before CASTEP when generating keywords. Usually unnecessary: "
            "the environment setup in this cluster's job script template is reused automatically."
        )

        self._ssh_form.addRow("Cluster name", self._name_edit)
        self._ssh_form.addRow("Host name", self._hostname_edit)
        self._ssh_form.addRow("Port", self._port_spin)
        self._ssh_form.addRow("Username", self._username_edit)
        self._ssh_form.addRow("Authentication method", self._auth_combo)
        self._ssh_form.addRow("SSH key path", self._key_path_row)
        self._ssh_form.addRow("", self._jump_enabled_check)
        self._jump_hostname_label = QLabel("Jump host name", form_group)
        self._ssh_form.addRow(self._jump_hostname_label, self._jump_hostname_edit)
        self._jump_port_label = QLabel("Jump host port", form_group)
        self._ssh_form.addRow(self._jump_port_label, self._jump_port_spin)
        self._jump_username_label = QLabel("Jump host username", form_group)
        self._ssh_form.addRow(self._jump_username_label, self._jump_username_edit)
        self._jump_auth_label = QLabel("Jump host authentication method", form_group)
        self._ssh_form.addRow(self._jump_auth_label, self._jump_auth_combo)
        self._jump_key_label = QLabel("Jump host SSH key path", form_group)
        self._ssh_form.addRow(self._jump_key_label, self._jump_key_path_row)
        self._ssh_form.addRow("", self._keychain_row)
        self._cluster_form.addRow("Default remote working directory", self._working_dir_edit)
        self._cluster_form.addRow("Queue system", self._queue_combo)
        self._cluster_form.addRow("Cores per node", self._cores_per_node_spin)
        self._cluster_form.addRow("Default walltime", self._walltime_edit)
        self._template_label = QLabel("Template script", form_group)
        self._cluster_form.addRow(self._template_label, self._template_path_row)
        self._cluster_form.addRow("CASTEP executable", self._castep_exec_edit)
        self._default_queue_label = QLabel("Default queue / partition", form_group)
        self._cluster_form.addRow(self._default_queue_label, self._default_queue_edit)
        self._default_memory_label = QLabel("Default memory per core (GB)", form_group)
        self._cluster_form.addRow(self._default_memory_label, self._default_memory_spin)
        self._cluster_form.addRow(self._default_mem_per_cpu_check, self._default_mem_per_cpu_edit)
        self._cluster_form.addRow("Submit command path", self._submit_command_edit)
        self._cluster_form.addRow("Queue check command path", self._queue_check_command_edit)
        self._cluster_form.addRow("Cancel command path", self._cancel_command_edit)
        self._keyword_mode_label = QLabel("Generate keywords using", form_group)
        self._cluster_form.addRow(self._keyword_mode_label, self._keyword_mode_combo)
        self._keyword_shell_init_label = QLabel("Keyword shell init (optional)", form_group)
        self._cluster_form.addRow(self._keyword_shell_init_label, self._keyword_shell_init_edit)

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
        self._jump_enabled_check.toggled.connect(lambda *_: self._refresh_jump_host_controls())
        self._hostname_edit.textChanged.connect(lambda *_: self._refresh_keychain_controls())
        self._auth_combo.currentTextChanged.connect(lambda *_: self._refresh_keychain_controls())
        self._jump_enabled_check.toggled.connect(lambda *_: self._refresh_keychain_controls())
        self._jump_hostname_edit.textChanged.connect(lambda *_: self._refresh_keychain_controls())
        self._jump_auth_combo.currentTextChanged.connect(lambda *_: self._refresh_keychain_controls())
        self._jump_auth_combo.currentTextChanged.connect(lambda *_: self._refresh_jump_host_controls())
        self._queue_combo.currentTextChanged.connect(lambda *_: self._refresh_scheduler_controls())
        self._default_mem_per_cpu_check.toggled.connect(self._default_mem_per_cpu_edit.setEnabled)
        self._default_mem_per_cpu_edit.setEnabled(self._default_mem_per_cpu_check.isChecked())
        self._username_edit.editingFinished.connect(self._maybe_sync_working_dir_for_username)
        self._key_browse_button.clicked.connect(self._browse_key_path)
        self._jump_key_browse_button.clicked.connect(self._browse_jump_key_path)
        self._keychain_help_button.clicked.connect(self._show_keychain_help)
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
                keyword_execution_mode="scheduler",
                keyword_shell_init="",
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
            self._jump_enabled_check.setChecked(False)
            self._jump_hostname_edit.setText("")
            self._jump_port_spin.setValue(22)
            self._jump_username_edit.setText("")
            self._jump_auth_combo.setCurrentText("key")
            self._jump_key_path_edit.setText("")
            self._keychain_check.setChecked(False)
            self._working_dir_edit.setText("")
            self._queue_combo.setCurrentText("SLURM")
            self._cores_per_node_spin.setValue(1)
            self._walltime_edit.setText("01:00:00")
            self._template_path_edit.setText("")
            self._castep_exec_edit.setText("castep.mpi")
            self._default_queue_edit.setText("")
            self._default_memory_spin.setValue(7.5)
            self._apply_default_mem_per_cpu(None)
            self._submit_command_edit.setText("")
            self._queue_check_command_edit.setText("")
            self._cancel_command_edit.setText("")
            self._set_keyword_mode("scheduler")
            self._keyword_shell_init_edit.setText("")
            self._last_username = ""
            self._refresh_auth_controls()
            self._refresh_jump_host_controls()
            self._refresh_keychain_controls()
            self._refresh_scheduler_controls()
            return

        self._name_edit.setText(config.name)
        self._hostname_edit.setText(config.hostname)
        self._port_spin.setValue(int(config.port))
        self._username_edit.setText(config.username)
        self._auth_combo.setCurrentText(config.auth_method)
        self._key_path_edit.setText(config.key_path)
        self._jump_enabled_check.setChecked(config.jump_host_enabled)
        self._jump_hostname_edit.setText(config.jump_hostname)
        self._jump_port_spin.setValue(int(config.jump_port))
        self._jump_username_edit.setText(config.jump_username)
        self._jump_auth_combo.setCurrentText(config.jump_auth_method)
        self._jump_key_path_edit.setText(config.jump_key_path)
        self._keychain_check.setChecked(config.use_macos_keychain_passwords)
        self._working_dir_edit.setText(config.default_working_directory)
        self._queue_combo.setCurrentText(config.queue_system.upper())
        self._cores_per_node_spin.setValue(int(config.cores_per_node))
        self._walltime_edit.setText(config.default_walltime)
        self._template_path_edit.setText(config.template_script_path)
        self._castep_exec_edit.setText(config.castep_executable)
        self._default_queue_edit.setText(config.default_queue_name)
        self._default_memory_spin.setValue(7.5 if config.default_memory_gb is None else float(config.default_memory_gb))
        self._apply_default_mem_per_cpu(config.default_mem_per_cpu_gb)
        self._submit_command_edit.setText(config.submit_command_path)
        self._queue_check_command_edit.setText(config.queue_check_command_path)
        self._cancel_command_edit.setText(config.cancel_command_path)
        self._set_keyword_mode(config.keyword_execution_mode)
        self._keyword_shell_init_edit.setText(config.keyword_shell_init)
        self._last_username = config.username
        self._refresh_auth_controls()
        self._refresh_jump_host_controls()
        self._refresh_keychain_controls()
        self._refresh_scheduler_controls()

    def _set_keyword_mode(self, mode: str) -> None:
        index = self._keyword_mode_combo.findData(mode.strip().lower())
        self._keyword_mode_combo.setCurrentIndex(index if index >= 0 else 0)

    def _apply_default_mem_per_cpu(self, mem_per_cpu_gb: float | None) -> None:
        """Seed the default per-CPU memory row, treating ``None`` as "no request"."""

        enabled = mem_per_cpu_gb is not None and float(mem_per_cpu_gb) > 0
        self._default_mem_per_cpu_check.setChecked(enabled)
        self._default_mem_per_cpu_edit.setValue(
            float(mem_per_cpu_gb) if enabled else _DEFAULT_MEM_PER_CPU_GB
        )
        self._default_mem_per_cpu_edit.setEnabled(enabled)

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

    def _refresh_jump_host_controls(self) -> None:
        enabled = self._jump_enabled_check.isChecked()
        key_enabled = enabled and self._jump_auth_combo.currentText().strip().lower() == "key"
        for widget in (
            self._jump_hostname_label,
            self._jump_hostname_edit,
            self._jump_port_label,
            self._jump_port_spin,
            self._jump_username_label,
            self._jump_username_edit,
            self._jump_auth_label,
            self._jump_auth_combo,
            self._jump_key_label,
            self._jump_key_path_edit,
            self._jump_key_browse_button,
        ):
            widget.setEnabled(enabled)
        self._jump_key_path_edit.setEnabled(key_enabled)
        self._jump_key_browse_button.setEnabled(key_enabled)

    def _refresh_keychain_controls(self) -> None:
        eligible = self._keychain_is_available_for_form()
        self._keychain_row.setVisible(eligible)
        self._keychain_check.setEnabled(eligible)
        self._keychain_help_button.setEnabled(eligible)
        if not eligible:
            self._keychain_check.setChecked(False)

    def _keychain_is_available_for_form(self) -> bool:
        return is_glamdring_keychain_supported(
            ClusterConfig(
                name=self._name_edit.text().strip() or "Redwood",
                hostname=self._hostname_edit.text().strip(),
                username=self._username_edit.text().strip(),
                auth_method=self._auth_combo.currentText().strip().lower() or "key",
                jump_host_enabled=self._jump_enabled_check.isChecked(),
                jump_hostname=self._jump_hostname_edit.text().strip(),
                jump_username=self._jump_username_edit.text().strip(),
                jump_auth_method=self._jump_auth_combo.currentText().strip().lower() or "key",
            )
        )

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
        self._default_mem_per_cpu_check.setVisible(is_slurm)
        self._default_mem_per_cpu_edit.setVisible(is_slurm)

    def _browse_key_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select SSH private key", str(Path.home()), "All Files (*)")
        if path:
            self._key_path_edit.setText(path)

    def _browse_jump_key_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select jump-host SSH private key", str(Path.home()), "All Files (*)")
        if path:
            self._jump_key_path_edit.setText(path)

    def _browse_template_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "Select submission template", str(Path.home()), "Shell Scripts (*.sh);;All Files (*)")
        if path:
            self._template_path_edit.setText(path)

    def _show_keychain_help(self) -> None:
        QMessageBox.information(
            self,
            "macOS Keychain setup",
            keychain_setup_help(self._updated_selected_config() or self._selected_config() or ClusterConfig(name="", hostname="")),
        )

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
        if queue_system.strip().lower() == "slurm" and self._default_mem_per_cpu_check.isChecked():
            default_mem_per_cpu_gb: float | None = float(self._default_mem_per_cpu_edit.value())
        else:
            default_mem_per_cpu_gb = None

        return replace(
            existing,
            name=self._name_edit.text().strip(),
            hostname=self._hostname_edit.text().strip(),
            port=self._port_spin.value(),
            username=self._username_edit.text().strip(),
            auth_method=self._auth_combo.currentText().strip().lower() or "key",
            key_path=self._key_path_edit.text().strip(),
            jump_host_enabled=self._jump_enabled_check.isChecked(),
            jump_hostname=self._jump_hostname_edit.text().strip(),
            jump_port=self._jump_port_spin.value(),
            jump_username=self._jump_username_edit.text().strip(),
            jump_auth_method=self._jump_auth_combo.currentText().strip().lower() or "key",
            jump_key_path=self._jump_key_path_edit.text().strip(),
            use_macos_keychain_passwords=self._keychain_check.isChecked() and self._keychain_is_available_for_form(),
            default_working_directory=self._working_dir_edit.text().strip(),
            queue_system=queue_system,
            cores_per_node=self._cores_per_node_spin.value(),
            default_walltime=walltime,
            template_script_path=self._template_path_edit.text().strip(),
            castep_executable=self._castep_exec_edit.text().strip() or "castep.mpi",
            default_queue_name=default_queue_name,
            default_memory_gb=default_memory_gb,
            default_mem_per_cpu_gb=default_mem_per_cpu_gb,
            submit_command_path=self._submit_command_edit.text().strip(),
            queue_check_command_path=self._queue_check_command_edit.text().strip(),
            cancel_command_path=self._cancel_command_edit.text().strip(),
            keyword_execution_mode=str(self._keyword_mode_combo.currentData() or "scheduler"),
            keyword_shell_init=self._keyword_shell_init_edit.text().strip(),
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

        credentials = prompt_for_cluster_credentials(self, config)
        if credentials is None:
            QMessageBox.warning(self, "Missing credentials", "Credentials are required for this connection.")
            return

        # The connect runs on a worker thread behind a cancellable progress
        # dialog: paramiko can block for over a minute against an unreachable
        # host and must never freeze the GUI. The factory and trust prompt are
        # passed from this module's namespace so tests can monkeypatch them.
        check = ClusterConnectionCheck(
            self,
            config,
            credentials,
            self._settings,
            title="Testing connection",
            connection_factory=SSHConnection,
            trust_prompt=lambda parent, cfg, exc: prompt_to_trust_host_key(parent, cfg, exc),
        )

        def _on_succeeded() -> None:
            QMessageBox.information(self, "Connection successful", "SSH connection test succeeded.")

        def _on_failed(exc: object) -> None:
            show_detailed_warning(
                self,
                "Connection failed",
                format_cluster_connection_error(config, exc),
                informative_text="Open Details for the full SSH connection error and recovery guidance.",
            )

        check.succeeded.connect(_on_succeeded)
        check.failed.connect(_on_failed)
        check.succeeded.connect(check.deleteLater)
        check.failed.connect(check.deleteLater)
        check.cancelled.connect(check.deleteLater)
        check.start()

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
            if config.jump_host_enabled:
                if not config.jump_hostname:
                    issues.append(f"Cluster '{config.name}' is missing jump host name.")
                if not config.jump_username:
                    issues.append(f"Cluster '{config.name}' is missing jump host username.")
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
