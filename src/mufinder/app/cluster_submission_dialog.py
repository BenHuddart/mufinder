"""Dialog for remote cluster submission runtime settings."""

from __future__ import annotations

from dataclasses import dataclass
import re

from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
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

from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.cluster_setup import ClusterConfig


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
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._job_labels = list(job_labels)
        self._cluster_configs = {config.name: config for config in cluster_configs}
        self._project_name = self._sanitize_path_component(project_name)
        self._workflow_type = self._sanitize_path_fragment(workflow_type)
        self._initial_config = initial_config
        self._connected_cluster_name = (connected_cluster_name or "").strip().casefold()
        self._connected_username = (connected_username or "").strip()

        self.setWindowTitle("Run Calculations on a Remote Cluster")
        self.resize(620, 480)

        layout = QVBoxLayout(self)

        summary = QLabel(
            f"You are about to submit {len(self._job_labels)} job(s): "
            + ", ".join(self._job_labels[:6])
            + ("..." if len(self._job_labels) > 6 else "")
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        form_group = QGroupBox("Submission Settings", self)
        form = QFormLayout(form_group)

        self._cluster_combo = QComboBox(form_group)
        for config in cluster_configs:
            self._cluster_combo.addItem(config.name)
        form.addRow("Remote cluster", self._cluster_combo)

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

        self._queue_name_label = QLabel("Queue or partition", form_group)
        self._queue_name_edit = QLineEdit(form_group)
        form.addRow(self._queue_name_label, self._queue_name_edit)

        self._memory_label = QLabel("Memory per core (GB)", form_group)
        self._memory_spin = DecimalLineEdit(form_group)
        self._memory_spin.setRange(0.1, 2048.0)
        self._memory_spin.setDecimals(2)
        self._memory_spin.setSingleStep(0.5)
        self._memory_spin.setValue(7.5)
        form.addRow(self._memory_label, self._memory_spin)

        self._comment_label = QLabel("Submission note", form_group)
        self._comment_edit = QLineEdit(form_group)
        self._comment_edit.setPlaceholderText("Optional note for this submission")
        form.addRow(self._comment_label, self._comment_edit)

        self._password_label = QLabel("Password or key passphrase", form_group)
        self._password_edit = QLineEdit(form_group)
        self._password_edit.setEchoMode(QLineEdit.EchoMode.Password)
        form.addRow(self._password_label, self._password_edit)

        self._auth_hint = QLabel(form_group)
        self._auth_hint.setWordWrap(True)
        # Keep the hint top-aligned with explicit headroom to avoid glyph clipping.
        self._auth_hint.setAlignment(Qt.AlignLeft | Qt.AlignTop)
        self._auth_hint.setContentsMargins(0, 0, 0, 0)
        self._auth_hint.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
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
        self._username_edit.textChanged.connect(lambda *_: self._refresh_connection_state())

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

    def selected_cluster_name(self) -> str:
        return self._cluster_combo.currentText().strip()

    def selected_cluster_config(self) -> ClusterConfig | None:
        return self._cluster_configs.get(self.selected_cluster_name())

    def password(self) -> str:
        return self._password_edit.text()

    def config(self) -> ClusterSubmissionConfig:
        addqueue = self._is_addqueue_cluster()
        return ClusterSubmissionConfig(
            cluster_name=self.selected_cluster_name(),
            working_directory=self._working_directory_edit.text().strip(),
            username=self._username_edit.text().strip(),
            job_name_prefix=self._job_name_prefix_edit.text().strip() or self._default_job_name_prefix(),
            num_processes=self._num_processes_spin.value(),
            walltime=self._walltime_edit.text().strip(),
            queue_name=self._queue_name_edit.text().strip() if addqueue else "",
            memory_gb=float(self._memory_spin.value()) if addqueue else None,
            comment=self._comment_edit.text().strip() if addqueue else "",
        )

    def _apply_cluster_defaults(self) -> None:
        config = self.selected_cluster_config()
        if config is None:
            self._working_directory_edit.setText("")
            self._username_edit.setText("")
            self._num_processes_spin.setValue(1)
            self._walltime_edit.setText("01:00:00")
            self._set_auth_hint_text("No cluster selected.")
            self._set_password_row_visible(True)
            self._refresh_scheduler_controls()
            return

        self._working_directory_edit.setText(self._default_submission_working_directory(config))
        self._username_edit.setText(config.username)
        self._num_processes_spin.setValue(max(1, int(config.cores_per_node)))
        self._walltime_edit.setText(config.default_walltime)
        self._queue_name_edit.setText(config.default_queue_name)
        self._memory_spin.setValue(7.5 if config.default_memory_gb is None else float(config.default_memory_gb))
        self._comment_edit.setText("")

        if self._has_active_connection_for_selected():
            active_user = self._username_edit.text().strip() or self._connected_username
            self._set_auth_hint_text(
                f"Active SSH connection detected for this cluster as '{active_user}'. "
                "Password/passphrase is not required for this submission."
            )
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
        if default_cluster_name and selected_cluster_name.casefold() != default_cluster_name.strip().casefold():
            return

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
            self._queue_name_edit.setText(config.queue_name.strip())
            if config.memory_gb is not None:
                self._memory_spin.setValue(float(config.memory_gb))
            self._comment_edit.setText(config.comment.strip())

        self._refresh_connection_state()

    def _set_auth_hint_text(self, text: str) -> None:
        self._auth_hint.setText(text)
        self._auth_hint.setMinimumHeight(self._auth_hint.fontMetrics().lineSpacing() + 6)
        self._auth_hint.updateGeometry()

    def _set_password_row_visible(self, visible: bool) -> None:
        self._password_label.setVisible(visible)
        self._password_edit.setVisible(visible)
        self._password_edit.setEnabled(visible)

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
        self._set_password_row_visible(not reuse_connection)

    def _is_addqueue_cluster(self) -> bool:
        config = self.selected_cluster_config()
        if config is None:
            return False
        return config.queue_system.strip().lower() == "addqueue"

    def _refresh_scheduler_controls(self) -> None:
        addqueue = self._is_addqueue_cluster()
        self._queue_name_label.setVisible(addqueue)
        self._queue_name_edit.setVisible(addqueue)
        self._memory_label.setVisible(addqueue)
        self._memory_spin.setVisible(addqueue)
        self._comment_label.setVisible(addqueue)
        self._comment_edit.setVisible(addqueue)
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
            cls._sanitize_path_component(part)
            for part in value.strip().split("/")
            if part.strip()
        ]
        return "/".join(part for part in parts if part)

    def _default_job_name_prefix(self) -> str:
        last_component = self._workflow_type.strip("/").rsplit("/", 1)[-1]
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

        selected = self.selected_cluster_config()
        if (
            selected is not None
            and selected.auth_method == "password"
            and not self._has_active_connection_for_selected()
            and not self.password().strip()
        ):
            QMessageBox.warning(
                self,
                "Missing password",
                "Enter the cluster password to continue with this submission.",
            )
            return

        self.accept()
