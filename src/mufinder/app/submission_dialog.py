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
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.farm_validation import check_farm_counts
from mufinder.app.theme import (
    refresh_widget_style,
    style_hint_text,
    style_section_title,
    style_warning_text,
)
from mufinder.domain.text import plural
from mufinder.services.local_runner import physical_core_count

# Job names listed in the heading's tooltip before the rest are counted.
_TOOLTIP_LABEL_LIMIT = 20


@dataclass(frozen=True, slots=True)
class SubmissionConfig:
    """Runtime configuration chosen by the user at submission time.

    ``max_concurrent`` is the number of jobs run at once; ``1`` runs them one
    after another, and ``parallel`` is true exactly when it is above 1.
    """

    num_cores: int
    parallel: bool
    max_concurrent: int


class LocalSubmissionDialog(QDialog):
    """Collect the cores per job and the jobs run at once for local CASTEP execution."""

    def __init__(
        self,
        *,
        job_labels: list[str],
        mpirun_configured: bool,
        farm_counts: Mapping[str, int] | None = None,
        stage_title: str | None = None,
        initial_cores: int | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """Build the dialog for ``job_labels``.

        ``stage_title`` names the page the jobs come from in the heading ("Run 6
        Cutoff Energy jobs"); without it the heading says plain "jobs".
        ``initial_cores`` is the core count last used in this project; without
        one, every physical core is offered (every logical core where the
        physical count cannot be read), since OpenMPI refuses more ranks than
        physical cores. A remembered count is capped the same way.
        """

        super().__init__(parent)

        self._job_labels = list(job_labels)
        # Per-job CASTEP task-farm counts, keyed by job label. Farms are chosen
        # per row when the workflow is configured; the core count is chosen here,
        # so this dialog is the first place the two constraints meet.
        self._farm_counts = {str(label): int(count) for label, count in (farm_counts or {}).items()}
        self._mpirun_configured = bool(mpirun_configured)
        self._available_cores = max(1, QThread.idealThreadCount())
        job_count = len(self._job_labels)

        self.setWindowTitle("Run Calculations Locally")
        self.resize(480, 0)

        layout = QVBoxLayout(self)

        noun = f"{stage_title} job" if stage_title else "job"
        self._heading = QLabel(f"Run {plural(job_count, noun)}", self)
        style_section_title(self._heading)
        self._heading.setWordWrap(True)
        self._heading.setToolTip(_labels_tooltip(self._job_labels))
        self._heading.setAccessibleDescription(", ".join(self._job_labels))
        layout.addWidget(self._heading)

        form_group = QGroupBox("Run settings", self)
        form = QFormLayout(form_group)

        self._core_spin = QSpinBox(form_group)
        if mpirun_configured:
            self._core_spin.setRange(1, self._available_cores)
            # A remembered count from a bigger computer is capped at this one's
            # physical cores too, so reopening a project never offers a launch
            # mpirun refuses.
            usable_cores = min(
                physical_core_count() or self._available_cores, self._available_cores
            )
            cores = usable_cores if initial_cores is None else int(initial_cores)
            self._core_spin.setValue(min(max(1, cores), usable_cores))
        else:
            self._core_spin.setRange(1, 1)
            self._core_spin.setValue(1)
        form.addRow("Cores per job", self._core_spin)

        self._jobs_at_once_spin = QSpinBox(form_group)
        self._jobs_at_once_spin.setRange(1, max(1, job_count))
        self._jobs_at_once_spin.setValue(1)
        self._jobs_at_once_spin.setToolTip("1 runs the jobs one after another.")
        if job_count > 1:
            form.addRow("Jobs at once", self._jobs_at_once_spin)
        else:
            self._jobs_at_once_spin.setVisible(False)

        self._cores_line = QLabel(form_group)
        self._cores_line.setWordWrap(True)
        form.addRow("", self._cores_line)

        self._mpirun_hint: QLabel | None = None
        if not mpirun_configured:
            self._mpirun_hint = QLabel(
                "Set up mpirun in Setup > CASTEP to enable multi-core local runs.",
                form_group,
            )
            self._mpirun_hint.setWordWrap(True)
            style_hint_text(self._mpirun_hint)
            form.addRow("", self._mpirun_hint)

        layout.addWidget(form_group)

        self._farm_warning = QLabel("", self)
        self._farm_warning.setWordWrap(True)
        style_warning_text(self._farm_warning)
        layout.addWidget(self._farm_warning)
        layout.addStretch(1)

        self._button_box = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            parent=self,
        )
        run_button = self._button_box.button(QDialogButtonBox.StandardButton.Ok)
        if run_button is not None:
            run_button.setText(f"Run {plural(job_count, 'job')}")
        layout.addWidget(self._button_box)

        self._button_box.accepted.connect(self.accept)
        self._button_box.rejected.connect(self.reject)
        self._core_spin.valueChanged.connect(lambda *_: self._refresh_checks())
        self._jobs_at_once_spin.valueChanged.connect(lambda *_: self._refresh_checks())

        self._refresh_checks()

    def config(self) -> SubmissionConfig:
        """Return the selected submission runtime configuration."""

        jobs_at_once = self._jobs_at_once_spin.value() if len(self._job_labels) > 1 else 1
        return SubmissionConfig(
            num_cores=self._core_spin.value(),
            parallel=jobs_at_once > 1,
            max_concurrent=jobs_at_once,
        )

    def _refresh_checks(self) -> None:
        config = self.config()

        used_cores = config.num_cores * config.max_concurrent
        line = f"Uses {used_cores} of {self._available_cores} cores"
        oversubscribed = used_cores > self._available_cores
        if oversubscribed:
            line += ". Jobs that share cores run slower."
            style_warning_text(self._cores_line)
        else:
            style_hint_text(self._cores_line)
        self._cores_line.setText(line)
        refresh_widget_style(self._cores_line)

        # Task farming: a farm count above the core count is a certain CASTEP
        # abort, so it blocks the run button rather than merely warning.
        verdict = check_farm_counts(
            self._farm_counts,
            ranks=config.num_cores,
            mpirun_configured=self._mpirun_configured,
        )
        run_button = self._button_box.button(QDialogButtonBox.StandardButton.Ok)
        if run_button is not None:
            run_button.setEnabled(not verdict.blocked)
        self._farm_warning.setText(verdict.blocking_message or verdict.warning_message or "")
        self._farm_warning.setVisible(bool(self._farm_warning.text()))


def _labels_tooltip(labels: list[str]) -> str:
    """List the job names, one per line, counting those past the limit."""

    shown = labels[:_TOOLTIP_LABEL_LIMIT]
    lines = list(shown)
    hidden = len(labels) - len(shown)
    if hidden > 0:
        lines.append(f"and {hidden} more")
    return "\n".join(lines)
