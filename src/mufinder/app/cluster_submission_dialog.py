"""Dialog for remote cluster submission runtime settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import re

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QLineEdit,
    QMessageBox,
    QSpinBox,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
    QComboBox,
)

from PySide6.QtCore import Qt

from mufinder.app.cluster_ui_utils import resolve_cluster_credentials
from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.app.farm_validation import check_farm_counts
from mufinder.app.theme import style_hint_text
from mufinder.domain.cluster import ClusterConfig
from mufinder.services.cluster_queues import is_valid_queue_name, walltime_issue
from mufinder.services.keychain import credential_store_for_display, is_glamdring_keychain_supported
from mufinder.services.script_template import template_sets_partition
from mufinder.services.ssh_client import SSHCredentials


#: Pre-filled per-CPU memory request, used when a SLURM cluster profile does not
#: define its own default. Chosen to clear the ~5.1 GB peak RSS observed for
#: muon-relaxation CASTEP tasks, which exceeds the ~3.9 GB/task SCARF default.
_DEFAULT_MEM_PER_CPU_GB = 6.0


@dataclass(frozen=True, slots=True)
class ClusterSubmissionConfig:
    """Runtime configuration chosen by the user for cluster submission."""

    cluster_name: str
    working_directory: str
    username: str
    job_name_prefix: str
    num_processes: int
    walltime: str
    queue_name: str = ""
    memory_gb: float | None = None
    mem_per_cpu_gb: float | None = None
    comment: str = ""


class ClusterSubmissionDialog(QDialog):
    """Collect runtime options for remote cluster execution."""

    def __init__(
        self,
        *,
        job_labels: list[str],
        cluster_configs: list[ClusterConfig],
        default_cluster_name: str | None = None,
        initial_config: ClusterSubmissionConfig | None = None,
        project_name: str = "",
        workflow_type: str = "",
        connected_cluster_name: str | None = None,
        connected_username: str | None = None,
        farm_counts: Mapping[str, int] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._job_labels = list(job_labels)
        # Per-job CASTEP task-farm counts, keyed by job label; validated against
        # the requested rank count (see mufinder.app.farm_validation).
        self._farm_counts = {str(label): int(count) for label, count in (farm_counts or {}).items()}
        self._cluster_configs = {config.name: config for config in cluster_configs}
        self._project_name = self._sanitize_path_component(project_name)
        self._workflow_type = self._sanitize_path_fragment(workflow_type)
        self._initial_config = initial_config
        self._connected_cluster_name = (connected_cluster_name or "").strip().casefold()
        self._connected_username = (connected_username or "").strip()
        self._resolved_credentials: SSHCredentials | None = None

        self.setWindowTitle("Run Calculations on a Remote Cluster")
        self.resize(620, 480)

        layout = QVBoxLayout(self)

        self._summary_label = QLabel(self._job_summary_text())
        self._summary_label.setWordWrap(True)
        layout.addWidget(self._summary_label)

        form_group = QGroupBox("Submission Settings", self)
        form = QFormLayout(form_group)
        self._form = form

        self._cluster_combo = QComboBox(form_group)
        for config in cluster_configs:
            self._cluster_combo.addItem(config.name)
        form.addRow("Remote cluster", self._cluster_combo)

        # Editable, so a queue the profile does not list can still be typed.
        # Picking a listed queue applies its defaults; typing does not.
        self._queue_name_label = QLabel("Queue or partition", form_group)
        self._queue_combo = QComboBox(form_group)
        self._queue_combo.setEditable(True)
        self._queue_combo.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)
        self._queue_combo.lineEdit().setPlaceholderText("Scheduler default")
        form.addRow(self._queue_name_label, self._queue_combo)
        self._queue_hint = QLabel(form_group)
        self._queue_hint.setWordWrap(True)
        style_hint_text(self._queue_hint)
        form.addRow("", self._queue_hint)

        self._working_directory_edit = QLineEdit(form_group)
        form.addRow("Remote working directory", self._working_directory_edit)

        self._username_edit = QLineEdit(form_group)
        form.addRow("Username", self._username_edit)

        self._job_name_prefix_edit = QLineEdit(form_group)
        self._job_name_prefix_edit.setText(self._default_job_name_prefix())
        form.addRow("Job name prefix", self._job_name_prefix_edit)

        self._num_processes_spin = QSpinBox(form_group)
        self._num_processes_spin.setRange(1, 4096)
        self._num_processes_spin.setValue(1)
        form.addRow("Processor cores per job", self._num_processes_spin)

        self._walltime_edit = QLineEdit(form_group)
        self._walltime_edit.setPlaceholderText("HH:MM:SS")
        form.addRow("Maximum runtime", self._walltime_edit)

        self._memory_label = QLabel("Memory per core (GB)", form_group)
        self._memory_spin = DecimalLineEdit(form_group)
        self._memory_spin.setRange(0.1, 2048.0)
        self._memory_spin.setDecimals(2)
        self._memory_spin.setSingleStep(0.5)
        self._memory_spin.setValue(7.5)
        form.addRow(self._memory_label, self._memory_spin)

        # SLURM path: the checkbox doubles as the row label so "off" (no
        # #SBATCH memory directive, scheduler default applies) stays distinct
        # from "0 GB", which the value field cannot express.
        self._mem_per_cpu_check = QCheckBox("Memory per CPU (GB)", form_group)
        self._mem_per_cpu_edit = DecimalLineEdit(form_group)
        self._mem_per_cpu_edit.setRange(0.1, 2048.0)
        self._mem_per_cpu_edit.setDecimals(2)
        self._mem_per_cpu_edit.setSingleStep(0.5)
        self._mem_per_cpu_edit.setValue(_DEFAULT_MEM_PER_CPU_GB)
        self._mem_per_cpu_edit.setToolTip(
            "Requested memory per allocated CPU, emitted as #SBATCH --mem-per-cpu. "
            "Raise this when jobs are killed by the cluster's out-of-memory handler."
        )
        form.addRow(self._mem_per_cpu_check, self._mem_per_cpu_edit)

        self._comment_label = QLabel("Submission note", form_group)
        self._comment_edit = QLineEdit(form_group)
        self._comment_edit.setPlaceholderText("Optional note for this submission")
        form.addRow(self._comment_label, self._comment_edit)

        self._target_secret_label = QLabel("Cluster password or key passphrase", form_group)
        self._target_secret_edit = QLineEdit(form_group)
        self._target_secret_edit.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow(self._target_secret_label, self._target_secret_edit)
        self._password_label = self._target_secret_label
        self._password_edit = self._target_secret_edit

        self._jump_secret_label = QLabel("Jump host password or key passphrase", form_group)
        self._jump_secret_edit = QLineEdit(form_group)
        self._jump_secret_edit.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow(self._jump_secret_label, self._jump_secret_edit)

        self._farm_warning = QLabel(form_group)
        self._farm_warning.setWordWrap(True)
        form.addRow(self._farm_warning)

        self._auth_hint = QLabel(form_group)
        self._auth_hint.setWordWrap(True)
        # Keep the hint top-aligned with explicit headroom to avoid glyph clipping.
        self._auth_hint.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._auth_hint.setContentsMargins(0, 0, 0, 0)
        self._auth_hint.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._auth_hint.setMinimumHeight(self._auth_hint.fontMetrics().lineSpacing() + 6)
        # Add the label as a full-row widget so it isn't cramped by the form's label column
        form.addRow(self._auth_hint)

        layout.addWidget(form_group)

        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        layout.addWidget(self._button_box)

        self._button_box.accepted.connect(self._accept_if_valid)
        self._button_box.rejected.connect(self.reject)
        self._cluster_combo.currentTextChanged.connect(lambda *_: self._apply_cluster_defaults())
        self._queue_combo.activated.connect(lambda *_: self._on_queue_activated())
        self._queue_combo.editTextChanged.connect(lambda *_: self._refresh_queue_hint())
        self._username_edit.textChanged.connect(lambda *_: self._refresh_connection_state())
        self._mem_per_cpu_check.toggled.connect(self._mem_per_cpu_edit.setEnabled)
        self._mem_per_cpu_edit.setEnabled(self._mem_per_cpu_check.isChecked())
        self._num_processes_spin.valueChanged.connect(lambda *_: self._refresh_farm_validation())

        if default_cluster_name:
            index = self._cluster_combo.findText(default_cluster_name)
            if index >= 0:
                self._cluster_combo.setCurrentIndex(index)
        elif self._initial_config is not None and self._initial_config.cluster_name:
            index = self._cluster_combo.findText(self._initial_config.cluster_name)
            if index >= 0:
                self._cluster_combo.setCurrentIndex(index)

        self._apply_cluster_defaults()
        self._apply_initial_config(default_cluster_name=default_cluster_name)
        self._refresh_farm_validation()

    def _refresh_farm_validation(self) -> None:
        """Block or warn when the requested ranks cannot serve the farm counts."""

        verdict = check_farm_counts(
            self._farm_counts,
            ranks=self._num_processes_spin.value(),
            rank_label="processor cores",
        )
        ok_button = self._button_box.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setEnabled(not verdict.blocked)
        self._farm_warning.setText(verdict.blocking_message or verdict.warning_message)

    def selected_cluster_name(self) -> str:
        return self._cluster_combo.currentText().strip()

    def selected_cluster_config(self) -> ClusterConfig | None:
        return self._cluster_configs.get(self.selected_cluster_name())

    def password(self) -> str:
        return self._target_secret_edit.text()

    def credentials(self) -> SSHCredentials:
        return self._resolved_credentials or self._manual_credentials()

    def _selected_mem_per_cpu_gb(self) -> float | None:
        """Return the per-CPU memory request, or ``None`` to emit no directive."""

        if self._is_addqueue_cluster() or not self._mem_per_cpu_check.isChecked():
            return None
        value = float(self._mem_per_cpu_edit.value())
        return value if value > 0 else None

    def selected_queue_name(self) -> str:
        return self._queue_combo.currentText().strip()

    def config(self) -> ClusterSubmissionConfig:
        addqueue = self._is_addqueue_cluster()
        return ClusterSubmissionConfig(
            cluster_name=self.selected_cluster_name(),
            working_directory=self._working_directory_edit.text().strip(),
            username=self._username_edit.text().strip(),
            job_name_prefix=self._job_name_prefix_edit.text().strip()
            or self._default_job_name_prefix(),
            num_processes=self._num_processes_spin.value(),
            walltime=self._walltime_edit.text().strip(),
            queue_name=self.selected_queue_name(),
            memory_gb=float(self._memory_spin.value()) if addqueue else None,
            mem_per_cpu_gb=self._selected_mem_per_cpu_gb(),
            comment=self._comment_edit.text().strip() if addqueue else "",
        )

    def _apply_cluster_defaults(self) -> None:
        config = self.selected_cluster_config()
        if config is None:
            self._working_directory_edit.setText("")
            self._username_edit.setText("")
            self._queue_combo.clear()
            self._num_processes_spin.setValue(1)
            self._walltime_edit.setText("01:00:00")
            self._apply_mem_per_cpu_default(None)
            self._set_auth_hint_text("No cluster selected.")
            self._set_password_row_visible(True)
            self._refresh_scheduler_controls()
            return

        self._working_directory_edit.setText(self._default_submission_working_directory(config))
        self._username_edit.setText(config.username)
        self._queue_combo.blockSignals(True)
        self._queue_combo.clear()
        self._queue_combo.addItems(config.queue_names())
        self._queue_combo.setEditText(config.default_queue_name)
        self._queue_combo.blockSignals(False)
        self._apply_queue_defaults()
        self._comment_edit.setText("")
        self._target_secret_edit.setText("")
        self._jump_secret_edit.setText("")
        self._resolved_credentials = None

        if self._has_active_connection_for_selected():
            active_user = self._username_edit.text().strip() or self._connected_username
            self._set_auth_hint_text(
                f"Active SSH connection detected for this cluster as '{active_user}'. "
                "Passwords/passphrases are not required for this submission."
            )
        elif self._uses_keychain_credentials(config):
            store = credential_store_for_display()
            self._set_auth_hint_text(
                f"{store.sentence_name} passwords are enabled for this cluster. "
                f"MuFinder will read the Physics and Glamdring passwords from the {store.short_name} when you submit."
            )
        elif config.jump_host_enabled:
            self._set_auth_hint_text(self._jump_host_auth_hint(config))
        elif config.auth_method == "password":
            self._set_auth_hint_text(
                "Password authentication is enabled for this cluster. "
                "Enter your password when submitting."
            )
        else:
            self._set_auth_hint_text(
                "This cluster uses SSH keys. Enter a passphrase only if your private key is encrypted."
            )
        self._refresh_scheduler_controls()
        self._refresh_connection_state()

    def _selected_queue_config(self) -> ClusterConfig | None:
        """The selected cluster as it applies to the selected queue."""

        config = self.selected_cluster_config()
        if config is None:
            return None
        return config.for_queue(self.selected_queue_name())

    def _on_queue_activated(self) -> None:
        """Apply a picked queue's defaults; leave the fields alone for a typed one.

        An editable combo also reports Enter on typed text as an activation. A
        name the profile does not list has no defaults of its own, so the
        resources the user set stay as they are.
        """

        config = self.selected_cluster_config()
        if config is not None and self.selected_queue_name() in config.queue_names():
            self._apply_queue_defaults()
        else:
            self._refresh_queue_hint()

    def _apply_queue_defaults(self) -> None:
        """Fill the resource fields from the selected queue's defaults."""

        config = self._selected_queue_config()
        if config is None:
            return
        self._num_processes_spin.setValue(max(1, int(config.cores_per_node)))
        self._walltime_edit.setText(config.default_walltime)
        self._memory_spin.setValue(
            7.5 if config.default_memory_gb is None else float(config.default_memory_gb)
        )
        self._apply_mem_per_cpu_default(config.default_mem_per_cpu_gb)
        self._refresh_queue_hint()

    def _refresh_queue_hint(self) -> None:
        """Say what limits the queue has and whether the template overrides it."""

        config = self.selected_cluster_config()
        queue_name = self.selected_queue_name()
        notes: list[str] = []
        if config is not None:
            max_walltime = config.max_walltime_for_queue(queue_name)
            if max_walltime:
                notes.append(f"Maximum walltime on this queue: {max_walltime}.")
            queue_config = config.for_queue(queue_name)
            if queue_name and template_sets_partition(
                queue_config.template_script_path,
                cluster_name=config.name,
                queue_system=config.queue_system,
            ):
                notes.append(
                    "The job script template sets its own partition, which takes precedence."
                )
        self._queue_hint.setText(" ".join(notes))
        self._form.setRowVisible(self._queue_hint, bool(notes))

    def _apply_initial_config(self, *, default_cluster_name: str | None) -> None:
        config = self._initial_config
        if config is None:
            return

        selected_cluster_name = self.selected_cluster_name().strip()
        initial_cluster_name = config.cluster_name.strip()
        if not selected_cluster_name or not initial_cluster_name:
            return
        if selected_cluster_name.casefold() != initial_cluster_name.casefold():
            return
        if (
            default_cluster_name
            and selected_cluster_name.casefold() != default_cluster_name.strip().casefold()
        ):
            return

        if config.queue_name.strip():
            self._queue_combo.setEditText(config.queue_name.strip())
            self._apply_queue_defaults()
        if config.working_directory.strip():
            self._working_directory_edit.setText(config.working_directory.strip())
        if config.username.strip():
            self._username_edit.setText(config.username.strip())
        if config.job_name_prefix.strip():
            self._job_name_prefix_edit.setText(config.job_name_prefix.strip())
        if int(config.num_processes) > 0:
            self._num_processes_spin.setValue(int(config.num_processes))
        self._walltime_edit.setText(config.walltime.strip())

        if self._is_addqueue_cluster():
            if config.memory_gb is not None:
                self._memory_spin.setValue(float(config.memory_gb))
            self._comment_edit.setText(config.comment.strip())
        else:
            self._apply_mem_per_cpu_default(config.mem_per_cpu_gb)

        self._refresh_connection_state()

    def _apply_mem_per_cpu_default(self, mem_per_cpu_gb: float | None) -> None:
        """Seed the per-CPU memory row, treating ``None`` as "do not request"."""

        enabled = mem_per_cpu_gb is not None and float(mem_per_cpu_gb) > 0
        self._mem_per_cpu_check.setChecked(enabled)
        self._mem_per_cpu_edit.setValue(
            float(mem_per_cpu_gb) if enabled else _DEFAULT_MEM_PER_CPU_GB
        )
        self._mem_per_cpu_edit.setEnabled(enabled)

    def _set_auth_hint_text(self, text: str) -> None:
        self._auth_hint.setText(text)
        minimum_height = self._auth_hint.fontMetrics().lineSpacing() + 6
        self._auth_hint.setMinimumHeight(max(minimum_height, self._auth_hint.sizeHint().height()))
        self._auth_hint.updateGeometry()

    def _job_summary_text(self) -> str:
        job_count = len(self._job_labels)
        prefix = f"You are about to submit {job_count} job(s)"
        if not self._job_labels:
            return f"{prefix}."
        displayed_labels = self._job_labels[:4]
        summary = ", ".join(displayed_labels)
        remaining = job_count - len(displayed_labels)
        if remaining > 0:
            return f"{prefix}: {summary}, and {remaining} more."
        return f"{prefix}: {summary}."

    def _set_password_row_visible(self, visible: bool) -> None:
        self._refresh_credential_rows(visible)

    def _refresh_credential_rows(self, visible: bool) -> None:
        config = self.selected_cluster_config()
        target_visible = visible
        jump_visible = visible and config is not None and config.jump_host_enabled

        if config is not None:
            self._target_secret_label.setText(self._secret_label(config.name, config.auth_method))
            self._jump_secret_label.setText(self._jump_secret_label_text(config))

        if config is not None and self._uses_keychain_credentials(config):
            target_visible = False
            jump_visible = False

        for widget in (self._target_secret_label, self._target_secret_edit):
            widget.setVisible(target_visible)
            widget.setEnabled(target_visible)
        for widget in (self._jump_secret_label, self._jump_secret_edit):
            widget.setVisible(jump_visible)
            widget.setEnabled(jump_visible)

    def _has_active_connection_for_selected(self) -> bool:
        config = self.selected_cluster_config()
        if config is None or not self._connected_cluster_name:
            return False
        if config.name.strip().casefold() != self._connected_cluster_name:
            return False
        if not self._connected_username:
            return True
        return self._username_edit.text().strip() == self._connected_username

    def _refresh_connection_state(self) -> None:
        reuse_connection = self._has_active_connection_for_selected()
        self._refresh_credential_rows(not reuse_connection)

    @staticmethod
    def _secret_text(edit: QLineEdit) -> str | None:
        text = edit.text().strip()
        return text or None

    def _manual_credentials(self) -> SSHCredentials:
        return SSHCredentials(
            target_secret=self._secret_text(self._target_secret_edit),
            jump_secret=self._secret_text(self._jump_secret_edit),
        )

    @staticmethod
    def _secret_label(system_name: str, auth_method: str) -> str:
        if auth_method.strip().lower() == "password":
            return f"{system_name} password"
        return f"{system_name} key passphrase"

    def _jump_secret_label_text(self, config: ClusterConfig) -> str:
        system_name = "Physics" if self._is_physics_bastion(config) else "Jump host"
        return self._secret_label(system_name, config.jump_auth_method)

    @staticmethod
    def _is_physics_bastion(config: ClusterConfig) -> bool:
        return config.jump_hostname.strip().casefold() == "bastion.physics.ox.ac.uk"

    def _jump_host_auth_hint(self, config: ClusterConfig) -> str:
        target_label = (
            "password" if config.auth_method.strip().lower() == "password" else "key passphrase"
        )
        jump_system = "Physics" if self._is_physics_bastion(config) else "jump-host"
        jump_label = (
            "password"
            if config.jump_auth_method.strip().lower() == "password"
            else "key passphrase"
        )
        optional_note = (
            " Key passphrase fields can be left blank when the corresponding private key is not encrypted."
            if "key passphrase" in {target_label, jump_label}
            else ""
        )
        return (
            "This cluster connects through an SSH jump host. "
            f"Use the {jump_system} {jump_label} and cluster {target_label} fields above."
            f"{optional_note}"
        )

    def _uses_keychain_credentials(self, config: ClusterConfig) -> bool:
        return config.use_macos_keychain_passwords and is_glamdring_keychain_supported(config)

    def _is_addqueue_cluster(self) -> bool:
        config = self.selected_cluster_config()
        if config is None:
            return False
        return config.queue_system.strip().lower() == "addqueue"

    def _refresh_scheduler_controls(self) -> None:
        addqueue = self._is_addqueue_cluster()
        self._memory_label.setVisible(addqueue)
        self._memory_spin.setVisible(addqueue)
        self._comment_label.setVisible(addqueue)
        self._comment_edit.setVisible(addqueue)
        # --mem-per-cpu is a SLURM batch-script directive; addqueue takes its
        # memory request through the "Memory per core" argument above instead.
        self._mem_per_cpu_check.setVisible(not addqueue)
        self._mem_per_cpu_edit.setVisible(not addqueue)
        if addqueue:
            self._walltime_edit.setPlaceholderText("Optional for addqueue (HH:MM:SS)")
        else:
            self._walltime_edit.setPlaceholderText("HH:MM:SS")

    def _default_submission_working_directory(self, config: ClusterConfig) -> str:
        base = config.default_working_directory.strip().rstrip("/")
        if not base:
            return ""
        result = base
        for component in (self._project_name, self._workflow_type):
            if not component:
                continue
            result = f"{result}/{component.strip('/')}"
        return result

    @staticmethod
    def _sanitize_path_component(value: str) -> str:
        collapsed = re.sub(r"\s+", "_", value.strip())
        return re.sub(r"[^A-Za-z0-9._-]", "_", collapsed)

    @classmethod
    def _sanitize_path_fragment(cls, value: str) -> str:
        parts = [
            cls._sanitize_path_component(part) for part in value.strip().split("/") if part.strip()
        ]
        return "/".join(part for part in parts if part)

    def _default_job_name_prefix(self) -> str:
        normalized = self._workflow_type.strip("/")
        workflow_defaults = {
            "pristine_relaxation": "pristine_relax",
            "uep": "uep",
        }
        if normalized in workflow_defaults:
            return workflow_defaults[normalized]
        last_component = normalized.rsplit("/", 1)[-1]
        if last_component in {"positivemuon", "muonium"}:
            return last_component
        return "convergence"

    def _accept_if_valid(self) -> None:
        config = self.config()
        if not config.cluster_name:
            QMessageBox.warning(self, "Missing cluster", "Choose which configured cluster to use.")
            return
        if not config.working_directory:
            QMessageBox.warning(
                self,
                "Missing working directory",
                "Enter the remote working directory where MuFinder should create job folders.",
            )
            return
        if not config.walltime and not self._is_addqueue_cluster():
            QMessageBox.warning(
                self,
                "Missing walltime",
                "Enter a maximum runtime in HH:MM:SS format.",
            )
            return
        if config.queue_name and not is_valid_queue_name(config.queue_name):
            QMessageBox.warning(
                self,
                "Invalid queue",
                f"'{config.queue_name}' is not a queue name. Use letters, digits, '.', '_' and '-', "
                "with commas between several partitions.",
            )
            return
        cluster = self.selected_cluster_config()
        issue = walltime_issue(
            config.walltime,
            max_walltime=cluster.max_walltime_for_queue(config.queue_name)
            if cluster is not None
            else "",
            queue_name=config.queue_name,
        )
        if issue:
            QMessageBox.warning(self, "Invalid walltime", issue)
            return

        selected = self.selected_cluster_config()
        if selected is not None and not self._has_active_connection_for_selected():
            manual_credentials = self._manual_credentials()
            if self._uses_keychain_credentials(selected):
                resolved = resolve_cluster_credentials(
                    self,
                    selected,
                    manual_credentials=manual_credentials,
                    prompt_optional_key=False,
                )
                if resolved is None:
                    return
                self._resolved_credentials = resolved
            elif (
                selected.jump_host_enabled
                and selected.jump_auth_method == "password"
                and not manual_credentials.jump_secret
            ):
                QMessageBox.warning(
                    self,
                    "Missing jump host password",
                    "Enter the jump-host password to continue with this submission.",
                )
                return
            if (
                not self._uses_keychain_credentials(selected)
                and selected.auth_method == "password"
                and not manual_credentials.target_secret
            ):
                QMessageBox.warning(
                    self,
                    "Missing cluster password",
                    "Enter the cluster password to continue with this submission.",
                )
                return

        self.accept()
