"""Choose whether resubmitted relaxations continue from their checkpoint or start again.

Every relaxation ``.param`` MuFinder writes carries ``CONTINUATION : DEFAULT``,
so a relaxation resubmitted into its own run directory resumes from the
``{seed}.check`` file left there whenever one exists. Before this dialog that
happened silently: the outcome depended on a file the user could not see, and
starting again meant a separate Reset. The dashboard's Resubmit action now asks
once per batch, defaulting to the choice the jobs' recorded failure causes
suggest (see :func:`default_resubmit_mode`). The dialog performs no I/O: the
caller probes the checkpoints and acts on :meth:`ResubmitChoiceDialog.mode`.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

from PySide6.QtWidgets import (
    QButtonGroup,
    QDialog,
    QDialogButtonBox,
    QLabel,
    QRadioButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.theme import style_hint_text, style_section_title
from mufinder.domain.enums import FailureReason, JobStatus
from mufinder.services.cluster_runner import CheckpointProbeStatus
from mufinder.services.failure_diagnosis import failure_reason_label


class ResubmitMode(StrEnum):
    """How resubmitted relaxations treat the previous run's output."""

    CONTINUE = "continue"
    START_AGAIN = "start_again"


#: Causes that leave a healthy, merely interrupted run behind: the relaxation
#: was cut short (walltime, a node failure, MuFinder losing track of it) or
#: under-resourced (memory), so its checkpoint is worth resuming.
CONTINUE_FAVOURING_REASONS = frozenset(
    {
        FailureReason.WALLTIME,
        FailureReason.OUT_OF_MEMORY,
        FailureReason.NODE_FAILURE,
        FailureReason.INTERRUPTED,
    }
)


@dataclass(frozen=True, slots=True)
class ResubmitCandidate:
    """One relaxation offered the choice, with its probed checkpoint state."""

    label: str
    status: JobStatus
    reason: FailureReason | None
    checkpoint: CheckpointProbeStatus


def default_resubmit_mode(candidates: Sequence[ResubmitCandidate]) -> ResubmitMode:
    """Return the preselected mode for ``candidates``.

    Continue when any job's cause favours resuming. Starting again deletes
    output, so a mixed batch defaults to the choice that loses nothing; only a
    batch whose every cause points at the run itself (a CASTEP error, a
    cancellation, no recorded cause) defaults to starting again.
    """

    if any(candidate.reason in CONTINUE_FAVOURING_REASONS for candidate in candidates):
        return ResubmitMode.CONTINUE
    return ResubmitMode.START_AGAIN


def cause_summary(candidates: Sequence[ResubmitCandidate]) -> str:
    """Return e.g. ``"3 walltime, 1 CASTEP, 1 cancelled"`` in first-seen order."""

    counts = Counter(_cause_label(candidate) for candidate in candidates)
    return ", ".join(f"{count} {label}" for label, count in counts.items())


def checkpoint_summary(candidates: Sequence[ResubmitCandidate]) -> str:
    """Describe how many relaxations will actually resume from a checkpoint."""

    total = len(candidates)
    counts = Counter(candidate.checkpoint for candidate in candidates)
    found = counts[CheckpointProbeStatus.FOUND]
    absent = counts[CheckpointProbeStatus.ABSENT]
    unknown = counts[CheckpointProbeStatus.UNAVAILABLE]
    if found == total:
        if total == 1:
            return "The job has a checkpoint to resume from."
        return "Every job has a checkpoint to resume from."
    if absent == total:
        if total == 1:
            return "No checkpoint was found, so the run will start from its initial structure."
        return "No checkpoints were found, so the runs will start from their initial structures."
    if unknown == total:
        return (
            "Checkpoints could not be checked; a run without one starts from its initial structure."
        )
    parts = []
    if found:
        parts.append(f"{found} {'has' if found == 1 else 'have'} a checkpoint")
    if absent:
        parts.append(
            f"{absent} {'has' if absent == 1 else 'have'} none and will start from the initial structure"
        )
    if unknown:
        parts.append(f"{unknown} could not be checked")
    return "; ".join(parts) + "."


def _cause_label(candidate: ResubmitCandidate) -> str:
    if candidate.status == JobStatus.CANCELLED:
        return "cancelled"
    return failure_reason_label(candidate.reason) or "no recorded cause"


class ResubmitChoiceDialog(QDialog):
    """Ask whether resubmitted relaxations continue or start again."""

    def __init__(
        self,
        candidates: Sequence[ResubmitCandidate],
        *,
        other_job_count: int = 0,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Resubmit relaxations")
        self.setMinimumWidth(420)
        count = len(candidates)
        noun = "relaxation" if count == 1 else "relaxations"

        title = QLabel(f"Resubmit {count} {noun}")
        style_section_title(title)
        causes = QLabel(f"Causes: {cause_summary(candidates)}")
        style_hint_text(causes)

        self._continue_radio = QRadioButton("Continue from the checkpoint")
        continue_detail = QLabel(checkpoint_summary(candidates))
        continue_detail.setWordWrap(True)
        style_hint_text(continue_detail)

        self._start_again_radio = QRadioButton("Start again from the initial structure")
        start_again_detail = QLabel(
            "Deletes the previous run's output, including its checkpoint, on this "
            "computer and on the cluster. The input files are kept."
        )
        start_again_detail.setWordWrap(True)
        style_hint_text(start_again_detail)

        group = QButtonGroup(self)
        group.addButton(self._continue_radio)
        group.addButton(self._start_again_radio)
        if default_resubmit_mode(candidates) == ResubmitMode.CONTINUE:
            self._continue_radio.setChecked(True)
        else:
            self._start_again_radio.setChecked(True)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(causes)
        layout.addSpacing(6)
        layout.addWidget(self._continue_radio)
        layout.addWidget(_indented(continue_detail))
        layout.addWidget(self._start_again_radio)
        layout.addWidget(_indented(start_again_detail))

        if other_job_count:
            others = QLabel(
                f"{other_job_count} other {'job' if other_job_count == 1 else 'jobs'} "
                "will run again from the start; only relaxations can continue."
            )
            others.setWordWrap(True)
            style_hint_text(others)
            layout.addSpacing(6)
            layout.addWidget(others)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        self._resubmit_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._resubmit_button.setText("Resubmit…")
        self._cancel_button = buttons.button(QDialogButtonBox.StandardButton.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._start_again_radio.toggled.connect(self._update_default_button)
        self._update_default_button()

    def _update_default_button(self) -> None:
        """Make Enter cancel while the destructive choice is selected."""

        start_again = self._start_again_radio.isChecked()
        self._resubmit_button.setDefault(not start_again)
        self._resubmit_button.setAutoDefault(not start_again)
        self._cancel_button.setDefault(start_again)
        self._cancel_button.setAutoDefault(start_again)

    def mode(self) -> ResubmitMode:
        """Return the mode the user selected."""

        if self._start_again_radio.isChecked():
            return ResubmitMode.START_AGAIN
        return ResubmitMode.CONTINUE


def _indented(label: QLabel) -> QWidget:
    container = QWidget()
    layout = QVBoxLayout(container)
    layout.setContentsMargins(24, 0, 0, 4)
    layout.addWidget(label)
    return container
