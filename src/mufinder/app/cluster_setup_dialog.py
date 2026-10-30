"""Dialog for configuring remote HPC cluster connections."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QListWidget,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app import workflow_table_utils
from mufinder.app.cluster_connection_check import ClusterConnectionCheck
from mufinder.app.cluster_ui_utils import (
    format_cluster_connection_error,
    prompt_for_cluster_credentials,
    prompt_to_trust_host_key,
    show_detailed_warning,
)
from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.app.theme import (
    style_help_button,
    style_hint_text,
    style_inline_group,
    style_tab_widget,
    track_item_view_corners,
)
from mufinder.app.workflow_table_utils import bind_delete_shortcuts
from mufinder.cluster_setup import load_cluster_configs, save_cluster_configs
from mufinder.domain.cluster import ClusterConfig, ClusterQueue
from mufinder.services.cluster_queues import queue_issues, walltime_issue
from mufinder.services.keychain import (
    credential_store_for_display,
    is_glamdring_keychain_supported,
    keychain_setup_help,
)
from mufinder.services.ssh_client import SSHConnection

#: Pre-filled default per-CPU memory request for SLURM cluster profiles. Mirrors
#: ``cluster_submission_dialog._DEFAULT_MEM_PER_CPU_GB``.
_DEFAULT_MEM_PER_CPU_GB = 6.0

# Columns of the queue table. Every column but the name is an override; a
# blank cell uses the profile's value from the Scheduler tab.
_QUEUE_NAME_COLUMN = 0
_QUEUE_WALLTIME_COLUMN = 1
_QUEUE_MAX_WALLTIME_COLUMN = 2
_QUEUE_CORES_COLUMN = 3
_QUEUE_MEMORY_COLUMN = 4
_QUEUE_EXECUTABLE_COLUMN = 5
_QUEUE_TEMPLATE_COLUMN = 6
_QUEUE_HEADERS = (
    "Name",
    "Default walltime",
    "Max walltime",
    "Cores per node",
    "Memory (GB)",
    "CASTEP executable",
    "Template script",
)
_SLURM_MEMORY_HEADER = "Memory per CPU (GB)"
_ADDQUEUE_MEMORY_HEADER = "Memory per core (GB)"


def _new_cluster_config(name: str) -> ClusterConfig:
    return ClusterConfig(
        name=name,
        hostname="",
        queue_system="SLURM",
        default_walltime="01:00:00",
        keyword_execution_mode="scheduler",
    )


class ClusterSetupDialog(QDialog):
    """Manage persisted cluster configurations.

    Saved clusters are listed on the left; the selected one is edited in the
    tabs on the right. Edits are kept per cluster while switching between
    them and are saved together on ``OK``.
    """

    def __init__(self, *, settings, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._settings = settings
        self._configs: list[ClusterConfig] = load_cluster_configs(settings)
        self._last_username = ""
        # The cluster whose values the form shows, so its edits can be kept
        # before another cluster is loaded.
        self._loaded_index = -1
        self._queue_row_names: list[str] = []

        self.setWindowTitle("Remote Cluster Connections")
        self.resize(1000, 580)

        layout = QVBoxLayout(self)
        body = QHBoxLayout()
        body.setSpacing(14)
        layout.addLayout(body, 1)

        body.addWidget(self._build_cluster_list_panel())

        self._tabs = QTabWidget(self)
        style_tab_widget(self._tabs)
        self._tabs.addTab(self._build_connection_tab(), "Connection")
        self._tabs.addTab(self._build_scheduler_tab(), "Scheduler")
        self._tabs.addTab(self._build_queues_tab(), "Queues")
        self._tabs.addTab(self._build_keywords_tab(), "Keyword generation")
        body.addWidget(self._tabs, 1)

        bottom_row = QHBoxLayout()
        self._test_button = QPushButton("Test Connection", self)
        self._test_button.setToolTip("Connect to the selected cluster with the settings shown.")
        bottom_row.addWidget(self._test_button)
        bottom_row.addStretch(1)
        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        bottom_row.addWidget(self._button_box)
        layout.addLayout(bottom_row)

        self._connect_signals()
        self._populate_cluster_list()
        self._load_selected_config()

    # ------------------------------------------------------------------ layout

    def _build_cluster_list_panel(self) -> QWidget:
        panel = QWidget(self)
        panel.setFixedWidth(190)
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(0, 0, 0, 0)
        panel_layout.addWidget(QLabel("Saved clusters", panel))
        self._cluster_list = QListWidget(panel)
        self._cluster_list.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        panel_layout.addWidget(self._cluster_list, 1)
        buttons = QHBoxLayout()
        self._add_button = QPushButton("Add", panel)
        self._add_button.setToolTip("Add a new cluster.")
        self._duplicate_button = QPushButton("Duplicate", panel)
        self._duplicate_button.setToolTip("Add a copy of the selected cluster.")
        self._remove_button = QPushButton("Remove", panel)
        self._remove_button.setToolTip("Remove the selected cluster (Delete).")
        for button in (self._add_button, self._duplicate_button, self._remove_button):
            buttons.addWidget(button)
        panel_layout.addLayout(buttons)
        return panel

    @staticmethod
    def _new_form() -> QFormLayout:
        form = QFormLayout()
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.ExpandingFieldsGrow)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(8)
        return form

    def _path_row(self, parent: QWidget, edit: QLineEdit) -> tuple[QWidget, QPushButton]:
        row = QWidget(parent)
        style_inline_group(row)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(0, 0, 0, 0)
        row_layout.addWidget(edit, 1)
        button = QPushButton("Browse", row)
        row_layout.addWidget(button)
        return row, button

    def _build_connection_tab(self) -> QWidget:
        tab = QWidget(self)
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(14, 14, 14, 14)

        self._ssh_form = self._new_form()
        self._name_edit = QLineEdit(tab)
        self._hostname_edit = QLineEdit(tab)
        self._hostname_edit.setPlaceholderText("e.g. login.cluster.example.ac.uk")
        self._port_spin = QSpinBox(tab)
        self._port_spin.setRange(1, 65535)
        self._port_spin.setValue(22)
        self._username_edit = QLineEdit(tab)
        self._auth_combo = QComboBox(tab)
        self._auth_combo.addItems(["key", "password"])
        self._key_path_edit = QLineEdit(tab)
        self._key_path_edit.setPlaceholderText("Default SSH keys and agent")
        self._key_path_row, self._key_browse_button = self._path_row(tab, self._key_path_edit)

        self._keychain_row = QWidget(tab)
        style_inline_group(self._keychain_row)
        keychain_layout = QHBoxLayout(self._keychain_row)
        keychain_layout.setContentsMargins(0, 0, 0, 0)
        store = credential_store_for_display()
        self._keychain_check = QCheckBox(f"Use {store.name} passwords", self._keychain_row)
        keychain_layout.addWidget(self._keychain_check, 1)
        self._keychain_help_button = QToolButton(self._keychain_row)
        self._keychain_help_button.setText("?")
        self._keychain_help_button.setToolTip(f"Show {store.name} setup commands.")
        style_help_button(self._keychain_help_button)
        keychain_layout.addWidget(self._keychain_help_button)

        self._ssh_form.addRow("Cluster name", self._name_edit)
        self._ssh_form.addRow("Host name", self._hostname_edit)
        self._ssh_form.addRow("Port", self._port_spin)
        self._ssh_form.addRow("Username", self._username_edit)
        self._ssh_form.addRow("Authentication method", self._auth_combo)
        self._ssh_form.addRow("SSH key path", self._key_path_row)
        self._ssh_form.addRow("", self._keychain_row)
        tab_layout.addLayout(self._ssh_form)

        jump_group = QGroupBox("Jump host", tab)
        jump_layout = QVBoxLayout(jump_group)
        self._jump_enabled_check = QCheckBox("Connect through an SSH jump host", jump_group)
        jump_layout.addWidget(self._jump_enabled_check)
        self._jump_form = self._new_form()
        self._jump_hostname_edit = QLineEdit(jump_group)
        self._jump_port_spin = QSpinBox(jump_group)
        self._jump_port_spin.setRange(1, 65535)
        self._jump_port_spin.setValue(22)
        self._jump_username_edit = QLineEdit(jump_group)
        self._jump_auth_combo = QComboBox(jump_group)
        self._jump_auth_combo.addItems(["key", "password"])
        self._jump_key_path_edit = QLineEdit(jump_group)
        self._jump_key_path_edit.setPlaceholderText("Default SSH keys and agent")
        self._jump_key_path_row, self._jump_key_browse_button = self._path_row(
            jump_group, self._jump_key_path_edit
        )
        self._jump_hostname_label = QLabel("Host name", jump_group)
        self._jump_form.addRow(self._jump_hostname_label, self._jump_hostname_edit)
        self._jump_port_label = QLabel("Port", jump_group)
        self._jump_form.addRow(self._jump_port_label, self._jump_port_spin)
        self._jump_username_label = QLabel("Username", jump_group)
        self._jump_form.addRow(self._jump_username_label, self._jump_username_edit)
        self._jump_auth_label = QLabel("Authentication method", jump_group)
        self._jump_form.addRow(self._jump_auth_label, self._jump_auth_combo)
        self._jump_key_label = QLabel("SSH key path", jump_group)
        self._jump_form.addRow(self._jump_key_label, self._jump_key_path_row)
        jump_layout.addLayout(self._jump_form)
        tab_layout.addWidget(jump_group)
        tab_layout.addStretch(1)
        return tab

    def _build_scheduler_tab(self) -> QWidget:
        tab = QWidget(self)
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(14, 14, 14, 14)

        self._cluster_form = self._new_form()
        self._queue_system_combo = QComboBox(tab)
        self._queue_system_combo.addItems(["SLURM", "ADDQUEUE"])
        self._working_dir_edit = QLineEdit(tab)
        self._working_dir_edit.setPlaceholderText("e.g. /work/username")
        self._template_path_edit = QLineEdit(tab)
        self._template_path_edit.setPlaceholderText("Built-in SLURM template")
        self._template_path_row, self._template_browse_button = self._path_row(
            tab, self._template_path_edit
        )
        self._castep_exec_edit = QLineEdit(tab)
        self._castep_exec_edit.setPlaceholderText("castep.mpi")
        self._cores_per_node_spin = QSpinBox(tab)
        self._cores_per_node_spin.setRange(1, 4096)
        self._cores_per_node_spin.setValue(1)
        self._walltime_edit = QLineEdit(tab)
        self._walltime_edit.setPlaceholderText("HH:MM:SS")
        self._default_memory_spin = DecimalLineEdit(tab)
        self._default_memory_spin.setRange(0.1, 2048.0)
        self._default_memory_spin.setDecimals(2)
        self._default_memory_spin.setSingleStep(0.5)
        self._default_memory_spin.setValue(7.5)
        self._default_mem_per_cpu_check = QCheckBox("Request default memory per CPU (GB)", tab)
        self._default_mem_per_cpu_edit = DecimalLineEdit(tab)
        self._default_mem_per_cpu_edit.setRange(0.1, 2048.0)
        self._default_mem_per_cpu_edit.setDecimals(2)
        self._default_mem_per_cpu_edit.setSingleStep(0.5)
        self._default_mem_per_cpu_edit.setValue(_DEFAULT_MEM_PER_CPU_GB)
        self._default_mem_per_cpu_edit.setToolTip(
            "Pre-fills the submission dialog's #SBATCH --mem-per-cpu request. "
            "Leave unchecked to let the cluster's own memory default apply."
        )

        self._cluster_form.addRow("Queue system", self._queue_system_combo)
        self._cluster_form.addRow("Default remote working directory", self._working_dir_edit)
        self._template_label = QLabel("Template script", tab)
        self._cluster_form.addRow(self._template_label, self._template_path_row)
        self._cluster_form.addRow("CASTEP executable", self._castep_exec_edit)
        self._cluster_form.addRow("Cores per node", self._cores_per_node_spin)
        self._walltime_label = QLabel("Default walltime", tab)
        self._cluster_form.addRow(self._walltime_label, self._walltime_edit)
        self._default_memory_label = QLabel("Default memory per core (GB)", tab)
        self._cluster_form.addRow(self._default_memory_label, self._default_memory_spin)
        self._cluster_form.addRow(self._default_mem_per_cpu_check, self._default_mem_per_cpu_edit)
        tab_layout.addLayout(self._cluster_form)

        scheduler_note = QLabel(
            "These values apply to every queue. A queue on the Queues tab can override them.",
            tab,
        )
        scheduler_note.setWordWrap(True)
        style_hint_text(scheduler_note)
        tab_layout.addWidget(scheduler_note)

        commands_group = QGroupBox("Scheduler commands", tab)
        commands_form = self._new_form()
        commands_group.setLayout(commands_form)
        self._submit_command_edit = QLineEdit(commands_group)
        self._queue_check_command_edit = QLineEdit(commands_group)
        self._cancel_command_edit = QLineEdit(commands_group)
        for edit in (
            self._submit_command_edit,
            self._queue_check_command_edit,
            self._cancel_command_edit,
        ):
            edit.setPlaceholderText("Found on the cluster's PATH")
        commands_form.addRow("Submit command path", self._submit_command_edit)
        commands_form.addRow("Queue check command path", self._queue_check_command_edit)
        commands_form.addRow("Cancel command path", self._cancel_command_edit)
        tab_layout.addWidget(commands_group)
        tab_layout.addStretch(1)
        return tab

    def _build_queues_tab(self) -> QWidget:
        tab = QWidget(self)
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(14, 14, 14, 14)

        intro = QLabel(
            "List the queues (partitions) you submit to. The submission dialog offers them and "
            "starts on the default queue. Leave a cell blank to use the value from the Scheduler tab.",
            tab,
        )
        intro.setWordWrap(True)
        style_hint_text(intro)
        tab_layout.addWidget(intro)

        default_row = QHBoxLayout()
        default_row.addWidget(QLabel("Default queue", tab))
        self._default_queue_combo = QComboBox(tab)
        self._default_queue_combo.setMinimumWidth(180)
        default_row.addWidget(self._default_queue_combo)
        default_row.addStretch(1)
        tab_layout.addLayout(default_row)

        self._queue_table = QTableWidget(0, len(_QUEUE_HEADERS), tab)
        track_item_view_corners(self._queue_table)
        self._queue_table.setHorizontalHeaderLabels(list(_QUEUE_HEADERS))
        self._queue_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._queue_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._queue_table.verticalHeader().setVisible(False)
        header = self._queue_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setStretchLastSection(True)
        tooltips = {
            _QUEUE_NAME_COLUMN: "The partition or queue name passed to the scheduler",
            _QUEUE_WALLTIME_COLUMN: "Walltime the submission dialog starts with on this queue",
            _QUEUE_MAX_WALLTIME_COLUMN: "Longest walltime this queue accepts; longer submissions are refused",
            _QUEUE_CORES_COLUMN: "Cores per node on this queue's nodes",
            _QUEUE_EXECUTABLE_COLUMN: "CASTEP executable to run on this queue",
            _QUEUE_TEMPLATE_COLUMN: "Job script template for this queue",
        }
        for column, tooltip in tooltips.items():
            item = self._queue_table.horizontalHeaderItem(column)
            if item is not None:
                item.setToolTip(tooltip)
        tab_layout.addWidget(self._queue_table, 1)

        queue_buttons = QHBoxLayout()
        self._add_queue_button = QPushButton("Add Queue", tab)
        self._remove_queue_button = QPushButton("Remove Queue", tab)
        self._remove_queue_button.setToolTip("Remove the selected queue (Delete).")
        queue_buttons.addWidget(self._add_queue_button)
        queue_buttons.addWidget(self._remove_queue_button)
        queue_buttons.addStretch(1)
        tab_layout.addLayout(queue_buttons)
        return tab

    def _build_keywords_tab(self) -> QWidget:
        tab = QWidget(self)
        tab_layout = QVBoxLayout(tab)
        tab_layout.setContentsMargins(14, 14, 14, 14)
        form = self._new_form()
        self._keyword_mode_combo = QComboBox(tab)
        self._keyword_mode_combo.addItem("Scheduler job (recommended)", "scheduler")
        self._keyword_mode_combo.addItem("Login node", "direct")
        self._keyword_mode_combo.setToolTip(
            "Where MuFinder runs CASTEP to build the keyword catalog. A scheduler job reuses this "
            "cluster's job script and is safe where running CASTEP on the login node is forbidden or "
            "discouraged. The login node is quicker when it is permitted, because nothing waits in a queue."
        )
        self._keyword_shell_init_edit = QLineEdit(tab)
        self._keyword_shell_init_edit.setPlaceholderText("Optional, e.g. module load CASTEP")
        self._keyword_shell_init_edit.setToolTip(
            "Extra commands to run before CASTEP when generating keywords. Usually unnecessary: "
            "the environment setup in this cluster's job script template is reused automatically."
        )
        self._keyword_mode_label = QLabel("Generate keywords using", tab)
        form.addRow(self._keyword_mode_label, self._keyword_mode_combo)
        self._keyword_shell_init_label = QLabel("Keyword shell init (optional)", tab)
        form.addRow(self._keyword_shell_init_label, self._keyword_shell_init_edit)
        tab_layout.addLayout(form)
        note = QLabel(
            "MuFinder reads CASTEP's keyword descriptions from this cluster when you build the "
            "keyword catalog. A scheduler job runs on the default queue.",
            tab,
        )
        note.setWordWrap(True)
        style_hint_text(note)
        tab_layout.addWidget(note)
        tab_layout.addStretch(1)
        return tab

    def _connect_signals(self) -> None:
        self._cluster_list.currentRowChanged.connect(
            lambda *_: self._on_cluster_selection_changed()
        )
        self._add_button.clicked.connect(self._add_cluster)
        self._duplicate_button.clicked.connect(self._duplicate_cluster)
        self._remove_button.clicked.connect(self._remove_cluster)
        self._cluster_list_shortcuts = bind_delete_shortcuts(
            self._cluster_list,
            self._remove_cluster,
            context=Qt.ShortcutContext.WidgetShortcut,
        )
        self._name_edit.textEdited.connect(self._rename_selected_list_item)
        self._auth_combo.currentTextChanged.connect(lambda *_: self._refresh_auth_controls())
        self._jump_enabled_check.toggled.connect(lambda *_: self._refresh_jump_host_controls())
        self._hostname_edit.textChanged.connect(lambda *_: self._refresh_keychain_controls())
        self._auth_combo.currentTextChanged.connect(lambda *_: self._refresh_keychain_controls())
        self._jump_enabled_check.toggled.connect(lambda *_: self._refresh_keychain_controls())
        self._jump_hostname_edit.textChanged.connect(lambda *_: self._refresh_keychain_controls())
        self._jump_auth_combo.currentTextChanged.connect(
            lambda *_: self._refresh_keychain_controls()
        )
        self._jump_auth_combo.currentTextChanged.connect(
            lambda *_: self._refresh_jump_host_controls()
        )
        self._queue_system_combo.currentTextChanged.connect(
            lambda *_: self._refresh_scheduler_controls()
        )
        self._default_mem_per_cpu_check.toggled.connect(self._default_mem_per_cpu_edit.setEnabled)
        self._default_mem_per_cpu_edit.setEnabled(self._default_mem_per_cpu_check.isChecked())
        self._username_edit.editingFinished.connect(self._maybe_sync_working_dir_for_username)
        self._key_browse_button.clicked.connect(self._browse_key_path)
        self._jump_key_browse_button.clicked.connect(self._browse_jump_key_path)
        self._keychain_help_button.clicked.connect(self._show_keychain_help)
        self._template_browse_button.clicked.connect(self._browse_template_path)
        self._add_queue_button.clicked.connect(self._add_queue)
        self._remove_queue_button.clicked.connect(self._remove_queue)
        self._queue_table_shortcuts = bind_delete_shortcuts(
            self._queue_table,
            self._remove_queue,
            context=Qt.ShortcutContext.WidgetShortcut,
        )
        self._queue_table.itemChanged.connect(self._on_queue_item_changed)
        self._queue_table.itemSelectionChanged.connect(self._refresh_queue_buttons)
        self._test_button.clicked.connect(self._test_connection)
        self._button_box.accepted.connect(self._accept_and_save)
        self._button_box.rejected.connect(self.reject)

    # ---------------------------------------------------------- cluster list

    def configs(self) -> list[ClusterConfig]:
        return list(self._configs)

    def _populate_cluster_list(self, *, select: int | None = None) -> None:
        self._cluster_list.blockSignals(True)
        self._cluster_list.clear()
        for index, config in enumerate(self._configs):
            self._cluster_list.addItem(config.name or f"Cluster {index + 1}")
        if self._configs:
            target = 0 if select is None else max(0, min(select, len(self._configs) - 1))
            self._cluster_list.setCurrentRow(target)
        self._cluster_list.blockSignals(False)

        if not self._configs:
            self._add_cluster()

    def _selected_index(self) -> int:
        return self._cluster_list.currentRow()

    def _selected_config(self) -> ClusterConfig | None:
        index = self._selected_index()
        if index < 0 or index >= len(self._configs):
            return None
        return self._configs[index]

    def _select_cluster(self, name: str) -> bool:
        """Select the cluster called ``name``; return whether it exists."""

        matches = self._cluster_list.findItems(name, Qt.MatchFlag.MatchExactly)
        if not matches:
            return False
        self._cluster_list.setCurrentItem(matches[0])
        return True

    def _can_leave_loaded_cluster(self) -> bool:
        """Refuse to leave a cluster whose queue table has problems.

        Storing the table drops cells it cannot read, so the user fixes them
        first rather than losing them silently.
        """

        if not 0 <= self._loaded_index < len(self._configs):
            return True
        _queues, issues = self._queues_from_table()
        if not issues:
            return True
        self._tabs.setCurrentWidget(self._queue_table.parentWidget())
        QMessageBox.warning(
            self,
            "Invalid queue settings",
            "Fix these queue settings first:\n" + "\n".join(f"- {issue}" for issue in issues),
        )
        return False

    def _on_cluster_selection_changed(self) -> None:
        if not self._can_leave_loaded_cluster():
            self._cluster_list.blockSignals(True)
            self._cluster_list.setCurrentRow(self._loaded_index)
            self._cluster_list.blockSignals(False)
            return
        self._store_loaded_config()
        self._load_selected_config()

    def _store_loaded_config(self) -> None:
        """Keep the form's edits on the cluster they were made to."""

        index = self._loaded_index
        if index < 0 or index >= len(self._configs):
            return
        updated = self._config_from_form(self._configs[index])
        self._configs[index] = updated
        item = self._cluster_list.item(index)
        if item is not None:
            item.setText(updated.name or f"Cluster {index + 1}")

    def _rename_selected_list_item(self, text: str) -> None:
        item = self._cluster_list.currentItem()
        if item is not None:
            item.setText(text.strip() or f"Cluster {self._selected_index() + 1}")

    def _unique_name(self, base: str) -> str:
        existing = {config.name.casefold() for config in self._configs}
        if base.casefold() not in existing:
            return base
        counter = 2
        while f"{base} {counter}".casefold() in existing:
            counter += 1
        return f"{base} {counter}"

    def _add_cluster(self) -> None:
        if not self._can_leave_loaded_cluster():
            return
        self._store_loaded_config()
        existing = {config.name for config in self._configs}
        counter = 1
        while f"Cluster {counter}" in existing:
            counter += 1
        self._configs.append(_new_cluster_config(f"Cluster {counter}"))
        self._loaded_index = -1
        self._populate_cluster_list(select=len(self._configs) - 1)
        self._load_selected_config()
        self._tabs.setCurrentIndex(0)
        self._name_edit.setFocus()
        self._name_edit.selectAll()

    def _duplicate_cluster(self) -> None:
        if not self._can_leave_loaded_cluster():
            return
        self._store_loaded_config()
        config = self._selected_config()
        if config is None:
            return
        copy = replace(config, name=self._unique_name(f"{config.name} copy"))
        index = self._selected_index() + 1
        self._configs.insert(index, copy)
        self._loaded_index = -1
        self._populate_cluster_list(select=index)
        self._load_selected_config()

    def _remove_cluster(self) -> None:
        index = self._selected_index()
        if index < 0 or index >= len(self._configs):
            return
        name = self._name_edit.text().strip() or self._configs[index].name
        if not workflow_table_utils.confirm_destructive(
            self,
            "Remove cluster",
            f"Remove the cluster '{name}'?",
            "Its settings are deleted when you press OK. Jobs already submitted to it keep running.",
        ):
            return
        self._configs.pop(index)
        self._loaded_index = -1
        self._populate_cluster_list(select=index)
        self._load_selected_config()

    # ------------------------------------------------------------- form state

    def _load_selected_config(self) -> None:
        config = self._selected_config()
        self._loaded_index = self._selected_index() if config is not None else -1
        if config is None:
            config = _new_cluster_config("")
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
        self._queue_system_combo.setCurrentText(config.queue_system.upper())
        self._cores_per_node_spin.setValue(int(config.cores_per_node))
        self._walltime_edit.setText(config.default_walltime)
        self._template_path_edit.setText(config.template_script_path)
        self._castep_exec_edit.setText(config.castep_executable)
        self._default_memory_spin.setValue(
            7.5 if config.default_memory_gb is None else float(config.default_memory_gb)
        )
        self._apply_default_mem_per_cpu(config.default_mem_per_cpu_gb)
        self._submit_command_edit.setText(config.submit_command_path)
        self._queue_check_command_edit.setText(config.queue_check_command_path)
        self._cancel_command_edit.setText(config.cancel_command_path)
        self._set_keyword_mode(config.keyword_execution_mode)
        self._keyword_shell_init_edit.setText(config.keyword_shell_init)
        self._load_queues(config)
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

        answer = workflow_table_utils.confirm_choice(
            self,
            "Update working directory?",
            "The username changed. Update the default working directory to match it?",
            [("update", "Update directory"), ("keep", "Keep directory")],
            default="update",
            escape="keep",
        )
        if answer != "update":
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

    def _is_addqueue_form(self) -> bool:
        return self._queue_system_combo.currentText().strip().lower() == "addqueue"

    def _refresh_scheduler_controls(self) -> None:
        is_addqueue = self._is_addqueue_form()
        is_slurm = self._queue_system_combo.currentText().strip().lower() == "slurm"
        self._template_label.setVisible(is_slurm)
        self._template_path_row.setVisible(is_slurm)
        self._default_memory_label.setVisible(is_addqueue)
        self._default_memory_spin.setVisible(is_addqueue)
        self._default_mem_per_cpu_check.setVisible(is_slurm)
        self._default_mem_per_cpu_edit.setVisible(is_slurm)
        self._walltime_edit.setPlaceholderText("Optional (HH:MM:SS)" if is_addqueue else "HH:MM:SS")
        memory_header = self._queue_table.horizontalHeaderItem(_QUEUE_MEMORY_COLUMN)
        if memory_header is not None:
            memory_header.setText(_ADDQUEUE_MEMORY_HEADER if is_addqueue else _SLURM_MEMORY_HEADER)
            memory_header.setToolTip(
                "Memory per core passed to addqueue -m on this queue"
                if is_addqueue
                else "#SBATCH --mem-per-cpu request on this queue"
            )
        # addqueue submits without a job script, so there is no template to override.
        self._queue_table.setColumnHidden(_QUEUE_TEMPLATE_COLUMN, not is_slurm)

    # ------------------------------------------------------------------ queues

    def _load_queues(self, config: ClusterConfig) -> None:
        """Show the profile's queues; a single-queue profile becomes one row."""

        queues = list(config.queues)
        if not queues and config.default_queue_name.strip():
            queues = [ClusterQueue(name=config.default_queue_name.strip())]
        self._queue_table.blockSignals(True)
        self._queue_table.setRowCount(0)
        for queue in queues:
            self._append_queue_row(queue)
        self._queue_table.blockSignals(False)
        self._refresh_default_queue_combo(config.default_queue_name.strip())
        self._refresh_queue_buttons()

    def _append_queue_row(self, queue: ClusterQueue) -> int:
        row = self._queue_table.rowCount()
        self._queue_table.insertRow(row)
        memory = (
            queue.default_memory_gb
            if queue.default_memory_gb is not None
            else queue.default_mem_per_cpu_gb
        )
        values = {
            _QUEUE_NAME_COLUMN: queue.name,
            _QUEUE_WALLTIME_COLUMN: queue.default_walltime,
            _QUEUE_MAX_WALLTIME_COLUMN: queue.max_walltime,
            _QUEUE_CORES_COLUMN: "" if queue.cores_per_node is None else str(queue.cores_per_node),
            _QUEUE_MEMORY_COLUMN: "" if memory is None else format(float(memory), "g"),
            _QUEUE_EXECUTABLE_COLUMN: queue.castep_executable,
            _QUEUE_TEMPLATE_COLUMN: queue.template_script_path,
        }
        for column, text in values.items():
            self._queue_table.setItem(row, column, QTableWidgetItem(text))
        return row

    def _queue_cell(self, row: int, column: int) -> str:
        item = self._queue_table.item(row, column)
        return item.text().strip() if item is not None else ""

    def _queue_names_in_table(self) -> list[str]:
        names: list[str] = []
        for row in range(self._queue_table.rowCount()):
            name = self._queue_cell(row, _QUEUE_NAME_COLUMN)
            if name and name not in names:
                names.append(name)
        return names

    def _refresh_default_queue_combo(self, preferred: str | None = None) -> None:
        current = self._default_queue_combo.currentText() if preferred is None else preferred
        names = self._queue_names_in_table()
        # Each row's name as the combo last saw it, so a rename can be followed.
        self._queue_row_names = [
            self._queue_cell(row, _QUEUE_NAME_COLUMN) for row in range(self._queue_table.rowCount())
        ]
        self._default_queue_combo.blockSignals(True)
        self._default_queue_combo.clear()
        self._default_queue_combo.addItems(names)
        if current in names:
            self._default_queue_combo.setCurrentText(current)
        self._default_queue_combo.setEnabled(bool(names))
        self._default_queue_combo.blockSignals(False)

    def _refresh_queue_buttons(self) -> None:
        self._remove_queue_button.setEnabled(
            bool(self._queue_table.selectionModel().selectedRows())
        )

    def _on_queue_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != _QUEUE_NAME_COLUMN:
            return
        row = item.row()
        previous = self._queue_row_names[row] if row < len(self._queue_row_names) else ""
        renamed_default = bool(previous) and previous == self._default_queue_combo.currentText()
        # Renaming the default queue keeps it the default under its new name.
        self._refresh_default_queue_combo(item.text().strip() if renamed_default else None)

    def _add_queue(self) -> None:
        existing = set(self._queue_names_in_table())
        counter = 1
        name = "queue"
        while name in existing:
            counter += 1
            name = f"queue{counter}"
        self._queue_table.blockSignals(True)
        row = self._append_queue_row(ClusterQueue(name=name))
        self._queue_table.blockSignals(False)
        self._refresh_default_queue_combo()
        self._queue_table.selectRow(row)
        self._queue_table.setCurrentCell(row, _QUEUE_NAME_COLUMN)
        self._queue_table.editItem(self._queue_table.item(row, _QUEUE_NAME_COLUMN))

    def _remove_queue(self) -> None:
        rows = sorted({index.row() for index in self._queue_table.selectionModel().selectedRows()})
        if not rows:
            return
        row = rows[0]
        name = self._queue_cell(row, _QUEUE_NAME_COLUMN) or "this queue"
        if not workflow_table_utils.confirm_destructive(
            self,
            "Remove queue",
            f"Remove the queue '{name}' from this cluster?",
        ):
            return
        self._queue_table.removeRow(row)
        self._refresh_default_queue_combo()
        self._refresh_queue_buttons()

    def _queues_from_table(self) -> tuple[tuple[ClusterQueue, ...], list[str]]:
        """Read the queue table into records, with the problems that block saving.

        The table only translates cells; ``queue_issues`` holds the rules. A
        number cell that does not parse is read as ``0`` so the rules report it
        as not positive.
        """

        is_addqueue = self._is_addqueue_form()
        queues: list[ClusterQueue] = []
        for row in range(self._queue_table.rowCount()):
            cores_text = self._queue_cell(row, _QUEUE_CORES_COLUMN)
            try:
                cores = int(cores_text) if cores_text else None
            except ValueError:
                cores = 0
            memory_text = self._queue_cell(row, _QUEUE_MEMORY_COLUMN)
            try:
                memory = float(memory_text) if memory_text else None
            except ValueError:
                memory = 0.0
            queues.append(
                ClusterQueue(
                    name=self._queue_cell(row, _QUEUE_NAME_COLUMN),
                    default_walltime=self._queue_cell(row, _QUEUE_WALLTIME_COLUMN),
                    max_walltime=self._queue_cell(row, _QUEUE_MAX_WALLTIME_COLUMN),
                    cores_per_node=cores,
                    default_memory_gb=memory if is_addqueue else None,
                    default_mem_per_cpu_gb=None if is_addqueue else memory,
                    castep_executable=self._queue_cell(row, _QUEUE_EXECUTABLE_COLUMN),
                    template_script_path=""
                    if is_addqueue
                    else self._queue_cell(row, _QUEUE_TEMPLATE_COLUMN),
                )
            )
        issues = queue_issues(queues, profile_walltime=self._walltime_edit.text().strip())
        return tuple(queues), issues

    # ------------------------------------------------------------- utilities

    def _browse_key_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select SSH private key", str(Path.home()), "All Files (*)"
        )
        if path:
            self._key_path_edit.setText(path)

    def _browse_jump_key_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select jump-host SSH private key", str(Path.home()), "All Files (*)"
        )
        if path:
            self._jump_key_path_edit.setText(path)

    def _browse_template_path(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "Select submission template",
            str(Path.home()),
            "Shell Scripts (*.sh);;All Files (*)",
        )
        if path:
            self._template_path_edit.setText(path)

    def _show_keychain_help(self) -> None:
        QMessageBox.information(
            self,
            f"{credential_store_for_display().sentence_name} setup",
            keychain_setup_help(
                self._updated_selected_config()
                or self._selected_config()
                or ClusterConfig(name="", hostname="")
            ),
        )

    def _config_from_form(self, existing: ClusterConfig) -> ClusterConfig:
        queue_system = self._queue_system_combo.currentText().strip() or "SLURM"
        is_addqueue = queue_system.lower() == "addqueue"
        walltime = self._walltime_edit.text().strip()
        if not is_addqueue and not walltime:
            walltime = "01:00:00"
        default_memory_gb: float | None = (
            float(self._default_memory_spin.value()) if is_addqueue else None
        )
        if queue_system.lower() == "slurm" and self._default_mem_per_cpu_check.isChecked():
            default_mem_per_cpu_gb: float | None = float(self._default_mem_per_cpu_edit.value())
        else:
            default_mem_per_cpu_gb = None
        queues, _issues = self._queues_from_table()
        default_queue_name = self._default_queue_combo.currentText().strip() if queues else ""

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
            use_macos_keychain_passwords=self._keychain_check.isChecked()
            and self._keychain_is_available_for_form(),
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
            queues=queues,
        )

    def _updated_selected_config(self) -> ClusterConfig | None:
        existing = self._selected_config()
        if existing is None:
            return None
        return self._config_from_form(existing)

    def _sync_selected_config_from_form(self) -> None:
        self._loaded_index = self._selected_index()
        self._store_loaded_config()

    def _test_connection(self) -> None:
        self._sync_selected_config_from_form()
        config = self._selected_config()
        if config is None:
            return
        if not config.hostname:
            QMessageBox.warning(
                self, "Missing hostname", "Enter a host name before testing the connection."
            )
            return
        if not config.username:
            QMessageBox.warning(
                self, "Missing username", "Enter a username before testing the connection."
            )
            return

        credentials = prompt_for_cluster_credentials(self, config)
        if credentials is None:
            QMessageBox.warning(
                self, "Missing credentials", "Credentials are required for this connection."
            )
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
            if config.name.casefold() in seen:
                issues.append(f"Duplicate cluster name: {config.name}")
            seen.add(config.name.casefold())
            if not config.hostname:
                issues.append(f"Cluster '{config.name}' is missing hostname.")
            if not config.username:
                issues.append(f"Cluster '{config.name}' is missing username.")
            if config.jump_host_enabled:
                if not config.jump_hostname:
                    issues.append(f"Cluster '{config.name}' is missing jump host name.")
                if not config.jump_username:
                    issues.append(f"Cluster '{config.name}' is missing jump host username.")
            walltime_problem = walltime_issue(config.default_walltime)
            if walltime_problem:
                issues.append(f"Cluster '{config.name}': {walltime_problem}")
        return issues

    def _accept_and_save(self) -> None:
        self._sync_selected_config_from_form()
        # Only the cluster on screen can have queue problems: leaving a
        # cluster is refused until its queue table is valid.
        _queues, queue_issues = self._queues_from_table()
        name = self._name_edit.text().strip() or "this cluster"
        issues = [
            *self._validate_configs(),
            *(f"Cluster '{name}': {issue}" for issue in queue_issues),
        ]
        if issues:
            QMessageBox.warning(
                self, "Invalid cluster configuration", "\n".join(f"- {issue}" for issue in issues)
            )
            return

        save_cluster_configs(self._settings, self._configs)
        self.accept()
