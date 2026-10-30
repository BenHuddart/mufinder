"""Say why failed or stopped jobs did not finish, and offer what to do next.

Opened by the dashboard's ``Inspect`` and a workflow table's ``Inspect
Failure``. For one job the dialog shows its headline (the classified cause,
else the recorded failure message, see
:func:`~mufinder.services.failure_diagnosis.failure_headline`), the lines
under it and a collapsible log excerpt. For several jobs it lists one line per
job under each cause. The dialog performs no I/O beyond opening the output
folder: the caller builds the reports and acts on :meth:`FailedJobDialog.action`.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from enum import StrEnum
from pathlib import Path

from PySide6.QtCore import QSize, Qt
from PySide6.QtGui import QFontDatabase
from PySide6.QtWidgets import (
    QDialog,
    QDialogButtonBox,
    QLabel,
    QPlainTextEdit,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.theme import style_hint_text, style_section_title
from mufinder.domain.enums import JobStatus
from mufinder.domain.text import plural
from mufinder.services.failure_diagnosis import FailedJobReport, group_reports_by_headline

#: Job lines listed under one cause before the rest are counted.
JOB_LINES_PER_CAUSE = 8

LOG_EXCERPT_TEXT = "Log excerpt"
RESUBMIT_TEXT = "Resubmit…"
CONTINUE_RELAXATION_TEXT = "Continue relaxation…"
OPEN_FOLDER_TEXT = "Open Folder"


class FailedJobAction(StrEnum):
    """The follow-up the user chose; the dialog closes on either."""

    RESUBMIT = "resubmit"
    CONTINUE_RELAXATION = "continue_relaxation"


def failure_title(reports: Sequence[FailedJobReport]) -> str:
    """Return ``"<label> failed"`` for one job, ``"3 jobs stopped"`` for several."""

    statuses = {report.status for report in reports}
    if statuses == {JobStatus.STOPPED}:
        verb = "stopped"
    elif statuses == {JobStatus.FAILED}:
        verb = "failed"
    else:
        verb = "failed or stopped"
    if len(reports) == 1:
        return f"{reports[0].label} {verb}"
    return f"{plural(len(reports), 'job')} {verb}"


class FailedJobDialog(QDialog):
    """Headline, notes and log excerpt for failed jobs, with the next actions.

    ``can_resubmit`` offers ``Resubmit…`` and ``can_continue`` offers
    ``Continue relaxation…``; either closes the dialog with that
    :meth:`action`. ``Open Folder`` is offered for one job whose output folder
    exists and calls ``open_folder`` without closing the dialog.
    """

    def __init__(
        self,
        reports: Sequence[FailedJobReport],
        *,
        can_resubmit: bool,
        can_continue: bool,
        open_folder: Callable[[Path], None],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if not reports:
            raise ValueError("FailedJobDialog needs at least one report")
        self._reports = list(reports)
        self._action: FailedJobAction | None = None
        self.setWindowTitle("Failed job" if len(reports) == 1 else "Failed jobs")
        self.setMinimumWidth(480)

        layout = QVBoxLayout(self)
        title = QLabel(failure_title(self._reports))
        style_hint_text(title)
        layout.addWidget(title)
        if len(self._reports) == 1:
            self._add_single_job(layout, self._reports[0])
        else:
            self._add_job_groups(layout)

        self._log_toggle: QToolButton | None = None
        self._log_view: QPlainTextEdit | None = None
        excerpt = self._log_excerpt_text()
        if excerpt:
            self._add_log_excerpt(layout, excerpt)

        buttons = QDialogButtonBox()
        self.resubmit_button: QPushButton | None = None
        self.continue_button: QPushButton | None = None
        self.open_folder_button: QPushButton | None = None
        if can_resubmit:
            self.resubmit_button = buttons.addButton(
                RESUBMIT_TEXT, QDialogButtonBox.ButtonRole.ActionRole
            )
            self.resubmit_button.clicked.connect(lambda: self._finish(FailedJobAction.RESUBMIT))
        if can_continue:
            self.continue_button = buttons.addButton(
                CONTINUE_RELAXATION_TEXT, QDialogButtonBox.ButtonRole.ActionRole
            )
            self.continue_button.clicked.connect(
                lambda: self._finish(FailedJobAction.CONTINUE_RELAXATION)
            )
        folder = self._reports[0].output_directory if len(self._reports) == 1 else None
        if folder is not None:
            self.open_folder_button = buttons.addButton(
                OPEN_FOLDER_TEXT, QDialogButtonBox.ButtonRole.ActionRole
            )
            self.open_folder_button.setToolTip(f"Open {folder} in the file manager.")
            self.open_folder_button.clicked.connect(lambda: open_folder(folder))
        self.close_button = buttons.addButton(QDialogButtonBox.StandardButton.Close)
        self.close_button.setDefault(True)
        buttons.rejected.connect(self.reject)
        for button in buttons.buttons():
            button.setAccessibleName(button.text().replace("&", "").rstrip("…"))
        # Spare height goes below the text, not between its lines; once the
        # log excerpt is shown it takes the spare height instead.
        self._spacer_index = layout.count()
        layout.addStretch(1)
        layout.addWidget(buttons)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt override
        """Size the dialog for its wrapped text at its real width, not the narrowest."""

        width = max(self.minimumWidth(), super().sizeHint().width())
        return QSize(width, self.layout().totalHeightForWidth(width))

    def action(self) -> FailedJobAction | None:
        """Return the follow-up chosen, or ``None`` when the dialog was closed."""

        return self._action

    def _finish(self, action: FailedJobAction) -> None:
        self._action = action
        self.accept()

    def _add_single_job(self, layout: QVBoxLayout, report: FailedJobReport) -> None:
        headline = _wrapped_label(report.headline)
        style_section_title(headline)
        headline.setAccessibleName(f"Cause: {report.headline}")
        layout.addWidget(headline)
        for note in report.notes:
            label = _wrapped_label(note)
            style_hint_text(label)
            layout.addWidget(label)

    def _add_job_groups(self, layout: QVBoxLayout) -> None:
        for headline, reports in group_reports_by_headline(self._reports):
            cause = _wrapped_label(headline)
            font = cause.font()
            font.setBold(True)
            cause.setFont(font)
            layout.addSpacing(4)
            layout.addWidget(cause)
            lines = [f"• {report.label}" for report in reports[:JOB_LINES_PER_CAUSE]]
            hidden = len(reports) - len(lines)
            if hidden > 0:
                lines.append(f"…and {plural(hidden, 'more job')}")
            jobs = _wrapped_label("\n".join(lines))
            jobs.setContentsMargins(12, 0, 0, 0)
            layout.addWidget(jobs)

    def _log_excerpt_text(self) -> str:
        if len(self._reports) == 1:
            return self._reports[0].log_excerpt
        # Several jobs list only their labels above, so each job's notes lead
        # its section here.
        return "\n\n".join(
            "\n".join((f"=== {report.label} ===", *report.notes, report.log_excerpt)).rstrip()
            for report in self._reports
            if report.notes or report.log_excerpt
        )

    def _add_log_excerpt(self, layout: QVBoxLayout, excerpt: str) -> None:
        toggle = QToolButton()
        toggle.setText(LOG_EXCERPT_TEXT)
        toggle.setAccessibleName("Show the log excerpt")
        toggle.setCheckable(True)
        toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        toggle.setArrowType(Qt.ArrowType.RightArrow)
        toggle.setAutoRaise(True)

        view = QPlainTextEdit()
        view.setReadOnly(True)
        view.setAccessibleName(LOG_EXCERPT_TEXT)
        view.setLineWrapMode(QPlainTextEdit.LineWrapMode.NoWrap)
        view.setFont(QFontDatabase.systemFont(QFontDatabase.SystemFont.FixedFont))
        view.setPlainText(excerpt)
        view.setMinimumHeight(220)
        view.setVisible(False)

        def show_excerpt(shown: bool) -> None:
            toggle.setArrowType(Qt.ArrowType.DownArrow if shown else Qt.ArrowType.RightArrow)
            view.setVisible(shown)
            layout.setStretch(self._spacer_index, 0 if shown else 1)
            self.adjustSize()

        toggle.toggled.connect(show_excerpt)
        layout.addSpacing(6)
        layout.addWidget(toggle)
        layout.addWidget(view, 1)
        self._log_toggle = toggle
        self._log_view = view

    @property
    def log_toggle(self) -> QToolButton | None:
        """The ``Log excerpt`` disclosure button, ``None`` when there are no logs."""

        return self._log_toggle

    @property
    def log_view(self) -> QPlainTextEdit | None:
        """The read-only log excerpt, ``None`` when there are no logs."""

        return self._log_view


def _wrapped_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setTextFormat(Qt.TextFormat.PlainText)
    label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
    return label
