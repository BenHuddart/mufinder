"""Dialog for local CASTEP job submission runtime settings."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from PySide6.QtCore import QThread
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QLabel,
    QRadioButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.farm_validation import check_farm_counts


@dataclass(frozen=True, slots=True)
class SubmissionConfig:
    """Runtime configuration chosen by the user at submission time."""

    num_cores: int
    parallel: bool
    max_concurrent: int


class LocalSubmissionDialog(QDialog):
    """Collect runtime options for local CASTEP execution."""

    def __init__(
        self,
        *,
        job_labels: list[str],
        mpirun_configured: bool,
        farm_counts: Mapping[str, int] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)

        self._job_labels = list(job_labels)
        # Per-job CASTEP task-farm counts, keyed by job label. Farms are chosen
        # per row when the workflow is configured; the core count is chosen here,
        # so this dialog is the first place the two constraints meet.
        self._farm_counts = {str(label): int(count) for label, count in (farm_counts or {}).items()}
        self._mpirun_configured = bool(mpirun_configured)
        self._available_cores = max(1, QThread.idealThreadCount())

        self.setWindowTitle("Run Calculations Locally")
        self.resize(560, 420)

        layout = QVBoxLayout(self)

        summary = QLabel(
            f"You are about to run {len(self._job_labels)} job(s): "
            + ", ".join(self._job_labels[:6])
            + ("..." if len(self._job_labels) > 6 else "")
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        form_group = QGroupBox("Run Settings", self)
        form = QFormLayout(form_group)

        self._core_spin = QSpinBox(form_group)
        self._core_spin.setRange(1, self._available_cores)
        self._core_spin.setValue(1)
        form.addRow("Processor cores per job", self._core_spin)

        self._core_hint = QLabel(
            f"This computer reports {self._available_cores} logical cores.",
            form_group,
        )
        self._core_hint.setWordWrap(True)
        form.addRow("", self._core_hint)

        self._parallel_group = QGroupBox("How Jobs Should Run", self)
        parallel_layout = QFormLayout(self._parallel_group)
        self._sequential_radio = QRadioButton("Run jobs one after another", self._parallel_group)
        self._parallel_radio = QRadioButton("Run jobs at the same time", self._parallel_group)
        self._sequential_radio.setChecked(True)

        radio_row = QWidget(self._parallel_group)
        radio_layout = QVBoxLayout(radio_row)
        radio_layout.setContentsMargins(0, 0, 0, 0)
        radio_layout.addWidget(self._sequential_radio)
        radio_layout.addWidget(self._parallel_radio)
        parallel_layout.addRow("Execution mode", radio_row)

        self._max_concurrent_spin = QSpinBox(self._parallel_group)
        self._max_concurrent_spin.setRange(1, max(1, len(self._job_labels)))
        self._max_concurrent_spin.setValue(min(2, max(1, len(self._job_labels))))
        parallel_layout.addRow("Maximum jobs running together", self._max_concurrent_spin)

        layout.addWidget(form_group)
        if len(self._job_labels) > 1:
            layout.addWidget(self._parallel_group)
        else:
            self._parallel_group.setVisible(False)

        self._oversubscription_warning = QLabel("", self)
        self._oversubscription_warning.setWordWrap(True)
        layout.addWidget(self._oversubscription_warning)

        if not mpirun_configured:
            self._core_spin.setRange(1, 1)
            self._core_spin.setValue(1)
            self._core_hint.setText(
                "Set up mpirun in Setup -> CASTEP to enable multi-core local runs."
            )

        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        layout.addWidget(self._button_box)

        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)
        self._parallel_radio.toggled.connect(lambda *_: self._refresh_parallel_controls())
        self._core_spin.valueChanged.connect(lambda *_: self._update_oversubscription_warning())
        self._max_concurrent_spin.valueChanged.connect(
            lambda *_: self._update_oversubscription_warning()
        )

        self._refresh_parallel_controls()
        self._update_oversubscription_warning()

    def config(self) -> SubmissionConfig:
        """Return the selected submission runtime configuration."""

        parallel = len(self._job_labels) > 1 and self._parallel_radio.isChecked()
        max_concurrent = self._max_concurrent_spin.value() if parallel else 1
        return SubmissionConfig(
            num_cores=self._core_spin.value(),
            parallel=parallel,
            max_concurrent=max_concurrent,
        )

    def _refresh_parallel_controls(self) -> None:
        parallel_enabled = len(self._job_labels) > 1 and self._parallel_radio.isChecked()
        self._max_concurrent_spin.setEnabled(parallel_enabled)
        self._update_oversubscription_warning()

    def _update_oversubscription_warning(self) -> None:
        config = self.config()

        # Task farming first: a farm count above the core count is a certain
        # CASTEP abort, so it blocks OK rather than merely warning.
        verdict = check_farm_counts(
            self._farm_counts,
            ranks=config.num_cores,
            mpirun_configured=self._mpirun_configured,
        )
        ok_button = self._button_box.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setEnabled(not verdict.blocked)
        if verdict.blocking_message:
            self._oversubscription_warning.setText(verdict.blocking_message)
            return

        requested_total_cores = config.num_cores * max(1, config.max_concurrent)
        messages = [verdict.warning_message] if verdict.warning_message else []
        if requested_total_cores > self._available_cores:
            messages.append(
                "Requested cores exceed what this computer reports. "
                "Jobs may run slower because of oversubscription."
            )
        self._oversubscription_warning.setText(" ".join(messages))
