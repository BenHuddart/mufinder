"""The Job Dashboard: the table of active and ended jobs, its actions and menus.

``JobDashboard`` is the content of the main window's Job Dashboard dock. It
draws the jobs of the open project, keeps the running jobs' runtime cells
current, and offers each row's action button, its ``⋯`` menu, the table's
context menu and the Delete and Backspace keys. It decides which action applies
to which job and confirms Cancel and Dismiss before anything happens; it never
changes a job itself. A confirmed action leaves as ``action_requested(kind,
job_ids)`` (or ``cancel_and_dismiss_requested`` for the Delete keys) for the
window to carry out (see :doc:`/job_lifecycle`, Dashboard).
"""

from __future__ import annotations

from collections.abc import Callable, Collection, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from uuid import UUID

import shiboken6
from PySide6.QtCore import QSignalBlocker, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMenu,
    QPushButton,
    QSizePolicy,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.app import workflow_table_utils
from mufinder.app.geometry_progress_dialog import (
    STATUS_COLUMN_TOOLTIP,
    progress_status_text,
    progress_tooltip,
)
from mufinder.app.icons import icon
from mufinder.app.selection_snapshot import (
    RowControlsCell,
    RowControlsTable,
    SelectionSnapshotPushButton,
    resolve_target_keys,
)
from mufinder.app.theme import (
    SPACING,
    muted_text_brush,
    status_brush,
    use_square_top_table_corners,
)
from mufinder.app.workflow_submission import (
    RESUBMITTABLE_JOB_STATES,
    job_resumes_from_checkpoint,
)
from mufinder.app.workflow_table_utils import SelectionFeedback, partial_count_suffix
from mufinder.castep_io.geometry_progress import GeometryProgress
from mufinder.domain.enums import EndReason, JobActivity, JobStatus
from mufinder.domain.models import Job, utc_now
from mufinder.domain.text import plural
from mufinder.services.failure_diagnosis import failure_reason_summary, failure_status_reason
from mufinder.services.job_monitoring import (
    has_unretrieved_cluster_output,
    job_status_word,
    job_where_text,
    unretrieved_output_note,
)
from mufinder.services.job_progress import JobProgressMonitor, job_reports_geometry_progress
from mufinder.services.project_document import ProjectDocument
from mufinder.services.project_service import CONTINUABLE_JOB_STATUSES
from mufinder.services.project_summary import job_stage_key, job_stage_name

#: The job states the dashboard lists.
VISIBLE_DASHBOARD_STATES = frozenset({JobStatus.ACTIVE, JobStatus.ENDED})


def _set_item_text(item: QTableWidgetItem, text: str) -> bool:
    """Set ``item``'s text only when it differs; return whether it was written."""

    if item.text() == text:
        return False
    item.setText(text)
    return True


def _set_item_tooltip(item: QTableWidgetItem, tooltip: str) -> None:
    """Set ``item``'s tooltip only when it differs."""

    if item.toolTip() != tooltip:
        item.setToolTip(tooltip)


def _set_item_foreground(item: QTableWidgetItem, brush: QBrush) -> None:
    """Set ``item``'s foreground brush only when it differs."""

    if item.foreground() != brush:
        item.setForeground(brush)


@dataclass(frozen=True)
class JobDashboardDeps:
    """What the dashboard reads about jobs beyond the document, late-bound by its owner."""

    #: The open project, or ``None``.
    document: Callable[[], ProjectDocument | None]
    #: Whether ``Check now`` can look at the job now.
    can_check_now: Callable[[Job], bool]
    #: Why the job's status may be out of date, or ``None``.
    staleness: Callable[[Job, datetime], str | None]
    #: Whether the job's workflow row offers ``Continue relaxation…``.
    can_continue_relaxation: Callable[[Job], bool]
    #: Show a line in the window's status bar.
    show_status: Callable[[str], None]
    #: The widget confirmations and prompts are parented to (the main window).
    dialog_parent: Callable[[], QWidget]


class JobDashboard(QWidget):
    """The Job Dashboard dock's contents: the jobs table and its actions."""

    #: A confirmed action for jobs: ``(kind, job_ids)``. Kinds are ``cancel``,
    #: ``inspect``, ``check_now``, ``resubmit``, ``continue_relaxation``,
    #: ``reset``, ``dismiss`` and ``open``.
    action_requested = Signal(str, object)
    #: Delete or Backspace, confirmed: ``(job_ids to cancel, job_ids to dismiss)``.
    cancel_and_dismiss_requested = Signal(object, object)
    #: ``Show dismissed`` was ticked or unticked.
    show_dismissed_toggled = Signal(bool)

    LABEL_COLUMN = 0
    STAGE_COLUMN = 1
    STATUS_COLUMN = 2
    WHERE_COLUMN = 3
    RUNTIME_COLUMN = 4
    CORES_COLUMN = 5
    ACTION_COLUMN = 6
    # Columns sized to their contents by fit_columns; Job and Status stretch to
    # share the remaining width.
    CONTENT_SIZED_COLUMNS = (
        STAGE_COLUMN,
        WHERE_COLUMN,
        RUNTIME_COLUMN,
        CORES_COLUMN,
        ACTION_COLUMN,
    )
    # The status item's record of whether it carries the stale icon.
    _STALE_ROLE = Qt.ItemDataRole.UserRole + 1

    #: Menu entries, before the final "Open in <stage>".
    MENU_ACTIONS = (
        ("Cancel", "cancel"),
        ("Inspect…", "inspect"),
        ("Check now", "check_now"),
        ("Resubmit…", "resubmit"),
        ("Continue relaxation…", "continue_relaxation"),
        ("Reset…", "reset"),
        ("Dismiss", "dismiss"),
    )

    _WALLTIME_NEXT_STEP = "Resubmit to run it again, with a longer walltime if needed."
    _RELAXATION_WALLTIME_NEXT_STEP = (
        "Resubmit to continue from its checkpoint or start again, with a longer walltime if needed."
    )

    def __init__(
        self,
        deps: JobDashboardDeps,
        progress: JobProgressMonitor,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self._deps = deps
        self._progress = progress
        # Whether the dock is actually visible to the user. Driven by the dock's
        # visibilityChanged, which (unlike isHidden()) also reports False on
        # window minimize / tabbed-dock obscuring. Defaults True so an as-yet-
        # unshown window still arms the timer.
        self._visible_to_user = True
        # True while _sync_rows reconciles the rows.
        self._syncing = False
        self._cluster_elapsed_seconds_by_job: dict[UUID, int] = {}
        self._cluster_elapsed_checked_at: datetime | None = None

        self.runtime_timer = QTimer(self)
        self.runtime_timer.setInterval(1000)
        self.runtime_timer.timeout.connect(self.refresh_runtime_cells)

        self.table = RowControlsTable(0, 7)
        self.table.setHorizontalHeaderLabels(
            ["Job", "Stage", "Status", "Where", "Runtime", "Cores", "Action"]
        )
        self.table.horizontalHeaderItem(self.STATUS_COLUMN).setToolTip(STATUS_COLUMN_TOOLTIP)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.doubleClicked.connect(self._on_double_clicked)
        use_square_top_table_corners(self.table)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.open_context_menu)
        # Shown only while two or more rows are selected, so it costs no height
        # otherwise; it is what tells the user a row button acts on them all.
        self.selection_hint = QLabel("")
        self.selection_hint.setObjectName("hintText")
        self.selection_hint.setContentsMargins(8, 2, 8, 2)
        self.selection_hint.hide()
        self._selection_feedback = SelectionFeedback(
            self.table,
            multi_row_hint=self.selection_hint,
            multi_row_hint_text=lambda count: (
                f"{count} selected · row actions apply to all selected"
            ),
        )
        self._configure_columns()
        # The runtime tick updates only the rows in view, so bring rows that
        # scroll into view current at once rather than on the next tick.
        self.table.verticalScrollBar().valueChanged.connect(
            lambda _value: self.refresh_runtime_cells()
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        header = QHBoxLayout()
        header.setContentsMargins(0, 0, SPACING.sm, 0)
        header.addWidget(self.selection_hint, 1)
        self.show_dismissed_box = QCheckBox("Show dismissed", self)
        self.show_dismissed_box.setAccessibleName("Show dismissed jobs")
        self.show_dismissed_box.setToolTip(
            "Show the jobs dismissed from the dashboard, greyed, with their actions"
        )
        self.show_dismissed_box.toggled.connect(self.show_dismissed_toggled)
        header.addWidget(self.show_dismissed_box, 0, Qt.AlignmentFlag.AlignRight)
        layout.addLayout(header)
        layout.addWidget(self.table, 1)

        self.delete_shortcut: QShortcut | None = None
        self.backspace_shortcut: QShortcut | None = None

        self._progress.progress_changed.connect(self._on_progress_changed)

    def bind_delete_keys(self, owner: QWidget) -> None:
        """Bind Delete and Backspace to :meth:`delete_selected` while focus is inside ``owner``.

        The window passes the dock, so the keys also work from the dock's own
        title-bar controls. macOS delivers Delete as Backspace, so both are bound.
        """

        self.delete_shortcut = QShortcut(QKeySequence.StandardKey.Delete, owner)
        self.delete_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.delete_shortcut.activated.connect(self.delete_selected)
        self.backspace_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Backspace), owner)
        self.backspace_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self.backspace_shortcut.activated.connect(self.delete_selected)

    # ------------------------------------------------------------------ table

    def refresh(self) -> None:
        """Redraw the rows from the open project, or empty the table when none is open."""

        document = self._deps.document()
        if document is None:
            self._progress.follow(())
            self.table.setRowCount(0)
            self.runtime_timer.stop()
            self.clear_cluster_elapsed()
            return

        now = utc_now()
        desired_jobs = [
            job
            for job in document.jobs.values()
            if job.status in VISIBLE_DASHBOARD_STATES
            and (job.dismissed_at is None or self.show_dismissed_box.isChecked())
        ]
        self._progress.follow(desired_jobs)
        self._sync_rows(desired_jobs, now=now)
        self.sync_runtime_timer()

    def hide_dismissed(self) -> None:
        """Untick ``Show dismissed`` without redrawing, as a newly opened project starts."""

        with QSignalBlocker(self.show_dismissed_box):
            self.show_dismissed_box.setChecked(False)

    def primary_action(self, job: Job) -> tuple[str, str]:
        """Return the ``(kind, text)`` of the one action button that fits ``job``'s state."""

        if job.status == JobStatus.ACTIVE:
            return "cancel", "Cancel"
        if job.end_reason == EndReason.CANCELLED:
            return "resubmit", "Resubmit…"
        if has_unretrieved_cluster_output(job) and self._deps.can_check_now(job):
            return "check_now", "Check now"
        return "inspect", "Inspect…"

    @staticmethod
    def row_job_id(label_item) -> UUID | None:
        """Return the job id stashed on a label item's UserRole, or ``None``."""

        if label_item is None:
            return None
        try:
            return UUID(str(label_item.data(Qt.ItemDataRole.UserRole)))
        except (TypeError, ValueError, AttributeError):
            return None

    def job_id_at_row(self, row: int) -> UUID | None:
        """Return the job id shown on ``row``, or ``None``."""

        return self.row_job_id(self.table.item(row, self.LABEL_COLUMN))

    def _sync_rows(self, desired_jobs: list[Job], *, now) -> None:
        """Reconcile visible rows against ``desired_jobs`` by job id.

        Rows whose job disappeared are removed; brand-new jobs get a freshly
        built row inserted at the correct position; rows whose job id is
        unchanged are updated in place so their ``QTableWidgetItem`` and
        ``QPushButton`` cell widgets are never recreated (no widget churn, no
        stale button closures). A row is never repurposed for a different job
        id — mismatches are removed and re-inserted so each button's captured
        job id stays correct.
        """

        desired_ids = [job.id for job in desired_jobs]
        desired_id_set = set(desired_ids)
        table = self.table
        rows_changed = False
        changed_columns: set[int] = set()

        # One repaint for the whole pass, however many rows change.
        self._syncing = True
        table.setUpdatesEnabled(False)
        try:
            # Remove rows whose job is no longer desired. Iterate bottom-up so
            # row indexes stay valid as we delete.
            for row in range(table.rowCount() - 1, -1, -1):
                existing_id = self.job_id_at_row(row)
                if existing_id is None or existing_id not in desired_id_set:
                    table.removeRow(row)
                    rows_changed = True

            # Walk the desired order; align each row to the expected job id,
            # inserting new rows as needed. A row is never reused for a
            # different job id, so each button's captured job id stays correct.
            for target_row, job in enumerate(desired_jobs):
                current_id = self.job_id_at_row(target_row)
                if current_id == job.id:
                    changed_columns |= self._update_row(target_row, job, now=now)
                    continue
                self._insert_row(target_row, job, now=now)
                rows_changed = True

            # Trim any trailing rows left over (reordered leftovers the removal
            # pass above could not have caught).
            while table.rowCount() > len(desired_jobs):
                table.removeRow(table.rowCount() - 1)
                rows_changed = True

            self.fit_columns(self.CONTENT_SIZED_COLUMNS if rows_changed else changed_columns)
        finally:
            table.setUpdatesEnabled(True)
            self._syncing = False

    def _insert_row(self, row: int, job: Job, *, now) -> None:
        """Insert a fully built row for ``job`` at ``row``.

        Creates every ``QTableWidgetItem`` and action ``QPushButton`` once. Each
        button captures ``job.id`` in its click closure, so this row must only
        ever be used for this job (callers guarantee that via id-keyed
        reconciliation).
        """

        self.table.insertRow(row)

        label_item = QTableWidgetItem(job.label)
        label_item.setData(Qt.ItemDataRole.UserRole, str(job.id))
        self._apply_label_cell(label_item, job, now=now)
        self.table.setItem(row, self.LABEL_COLUMN, label_item)
        self.table.setItem(
            row, self.STAGE_COLUMN, QTableWidgetItem(job_stage_name(job.workflow_stage))
        )

        status_item = QTableWidgetItem()
        self._apply_status_cell(status_item, job, now=now)
        self.table.setItem(row, self.STATUS_COLUMN, status_item)

        self.table.setItem(row, self.WHERE_COLUMN, QTableWidgetItem(job_where_text(job)))
        self.table.setItem(row, self.RUNTIME_COLUMN, QTableWidgetItem(self._runtime_text(job)))
        self.table.setItem(
            row,
            self.CORES_COLUMN,
            QTableWidgetItem("" if job.num_processes is None else str(job.num_processes)),
        )
        self.table.setCellWidget(row, self.ACTION_COLUMN, self._make_action_cell(job))

    def _make_action_cell(self, job: Job) -> RowControlsCell:
        """Build a row's action button and its ``⋯`` menu button."""

        primary = SelectionSnapshotPushButton()
        primary.selection_snapshot_key = job.id
        # Outside a dialog a push button takes Return and Enter only when told to.
        primary.setAutoDefault(True)
        self._apply_primary_action(primary, job)
        primary.clicked.connect(
            lambda _checked=False, jid=job.id, btn=primary: self.dispatch_action(
                btn, jid, btn.dashboard_action
            )
        )
        more = SelectionSnapshotPushButton("⋯")
        more.selection_snapshot_key = job.id
        more.setObjectName("jobRowMoreButton")
        more.setAutoDefault(True)
        more.setToolTip("More actions")
        more.setAccessibleName(f"More actions for {job.label}")
        more.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        more.clicked.connect(
            lambda _checked=False, jid=job.id, btn=more: self.open_row_menu(btn, jid)
        )
        return RowControlsCell(primary, more)

    def _apply_primary_action(self, button: SelectionSnapshotPushButton, job: Job) -> bool:
        """Point a row's action button at the action ``job``'s state calls for.

        Returns whether its text changed, which can change the column's width.
        """

        kind, text = self.primary_action(job)
        button.dashboard_action = kind
        name = f"{text.rstrip('…')} {job.label}"
        if button.accessibleName() != name:
            button.setAccessibleName(name)
        if button.text() == text:
            return False
        button.setText(text)
        return True

    def _update_row(self, row: int, job: Job, *, now) -> set[int]:
        """Refresh an existing row in place without recreating items or widgets.

        Updates cell texts, the status cell and the action button, writing only
        the values that changed. The label item's job-id UserRole and the
        action cell's buttons (with their captured job id) are left untouched,
        so clicks keep dispatching the correct job. Returns the content-sized
        columns whose contents changed, for the caller to fit once.
        """

        table = self.table
        before = {
            column: item.text()
            for column in self.CONTENT_SIZED_COLUMNS
            if (item := table.item(row, column)) is not None
        }

        self._set_row_item_text(row, self.LABEL_COLUMN, job.label)
        label_item = table.item(row, self.LABEL_COLUMN)
        if label_item is not None:
            self._apply_label_cell(label_item, job, now=now)
        self._set_row_item_text(row, self.STAGE_COLUMN, job_stage_name(job.workflow_stage))

        status_item = table.item(row, self.STATUS_COLUMN)
        if status_item is not None:
            self._apply_status_cell(status_item, job, now=now)

        self._set_row_item_text(row, self.WHERE_COLUMN, job_where_text(job))
        self._set_row_item_text(row, self.RUNTIME_COLUMN, self._runtime_text(job))
        self._set_row_item_text(
            row,
            self.CORES_COLUMN,
            "" if job.num_processes is None else str(job.num_processes),
        )

        changed = {
            column
            for column, old_text in before.items()
            if table.item(row, column).text() != old_text
        }
        action_cell = table.cellWidget(row, self.ACTION_COLUMN)
        if action_cell is not None:
            primary, more = action_cell.row_controls
            if self._apply_primary_action(primary, job):
                changed.add(self.ACTION_COLUMN)
            more_name = f"More actions for {job.label}"
            if more.accessibleName() != more_name:
                more.setAccessibleName(more_name)
        return changed

    def _apply_label_cell(self, item: QTableWidgetItem, job: Job, *, now) -> None:
        """Grey a dismissed job's label, shown with ``Show dismissed``, and say when."""

        dismissed = job.dismissed_at is not None
        _set_item_foreground(item, muted_text_brush() if dismissed else QBrush())
        tooltip = (
            f"Dismissed {self.format_relative_age(job.dismissed_at, now=now)}" if dismissed else ""
        )
        _set_item_tooltip(item, tooltip)

    def _set_row_item_text(self, row: int, column: int, text: str) -> None:
        """Set a cell's text in place when it differs, or create the item if it is missing."""

        item = self.table.item(row, column)
        if item is None:
            self.table.setItem(row, column, QTableWidgetItem(text))
        else:
            _set_item_text(item, text)

    def _configure_columns(self) -> None:
        """Stretch Job and Status; size the rest to their contents.

        The content-sized columns are ``Fixed``, not ``ResizeToContents``: Qt
        re-measures a ``ResizeToContents`` column over every row on each cell
        change, which stalls a dashboard of a few hundred rows.
        ``fit_columns`` sizes them instead, at most once per column per
        refresh: every column when rows were added or removed, otherwise only
        the columns whose text changed.
        """

        header = self.table.horizontalHeader()
        for column in (self.LABEL_COLUMN, self.STATUS_COLUMN):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)
        for column in self.CONTENT_SIZED_COLUMNS:
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
        self.fit_columns(self.CONTENT_SIZED_COLUMNS)

    def fit_columns(self, columns=CONTENT_SIZED_COLUMNS) -> None:
        """Size each of ``columns`` to its contents (and header) once."""

        for column in columns:
            self.table.resizeColumnToContents(column)

    # ----------------------------------------------------------- status cell

    def _apply_status_cell(self, item: QTableWidgetItem, job: Job, *, now) -> None:
        """Show the job's state with its reason or progress inline.

        The cell reads ``failed · SCF did not converge``, ``stopped · walltime``
        or ``running · step 12 of 100``. Its tooltip explains the failure and
        carries the last check's age and source; an active cluster job whose
        status may be out of date gets a small icon, with the reason in the
        tooltip.
        """

        detail = ""
        tooltip_parts: list[str] = []
        progress = None
        if job.status == JobStatus.ENDED and job.end_reason != EndReason.CANCELLED:
            detail = failure_status_reason(job.end_reason, job.error_message)
            tooltip_parts.append(self.failure_tooltip(job))
        elif job.activity == JobActivity.RUNNING:
            progress = self._progress.progress_for(job.id)
            detail = progress_status_text(progress)
        stale_reason = self._deps.staleness(job, now)
        check_lines = self._last_check_lines(job, stale_reason, now=now)
        if check_lines:
            tooltip_parts.append("\n".join(check_lines))
        tooltip = "\n\n".join(tooltip_parts)
        if progress is not None:
            # The progress tooltip is rich text, so the plain lines join it as such.
            tooltip = "<br>".join(
                part
                for part in (progress_tooltip(progress), escape(tooltip).replace("\n", "<br>"))
                if part
            )

        word = job_status_word(job)
        _set_item_text(item, f"{word} · {detail}" if detail else word)
        _set_item_tooltip(item, tooltip)
        _set_item_foreground(item, status_brush(word))
        is_stale = stale_reason is not None
        if bool(item.data(self._STALE_ROLE)) != is_stale:
            item.setData(self._STALE_ROLE, is_stale)
            item.setIcon(icon("stale") if is_stale else QIcon())

    def _last_check_lines(self, job: Job, stale_reason: str | None, *, now) -> list[str]:
        """Return the status tooltip's lines on when and how the job was last checked."""

        lines: list[str] = []
        if job.last_checked_at is not None:
            lines.append(f"Last checked: {self.format_relative_age(job.last_checked_at, now=now)}")
        if job.last_status_source:
            lines.append(f"Source: {job.last_status_source}")
        if job.last_status_message:
            lines.append(f"Message: {job.last_status_message}")
        if stale_reason is not None:
            lines.append(f"Stale: {stale_reason}")
        return lines

    @classmethod
    def failure_tooltip(cls, job: Job) -> str:
        """Return the status tooltip's explanation of why ``job`` ended."""

        summary = failure_reason_summary(job.end_reason)
        message = (job.error_message or "").strip()
        if job.end_reason in (None, EndReason.UNKNOWN) and message:
            return f"{message}\n\nUse Inspect to see the job's logs."
        parts = [summary]
        if message and message != summary:
            parts.append(message)
        if job.end_reason == EndReason.WALLTIME:
            parts.append(
                cls._RELAXATION_WALLTIME_NEXT_STEP
                if job_resumes_from_checkpoint(job)
                else cls._WALLTIME_NEXT_STEP
            )
        return "\n\n".join(parts)

    def refresh_job_status(self, job_id: UUID) -> None:
        """Redraw the status cell of ``job_id``'s row, if it has one."""

        document = self._deps.document()
        job = document.jobs.get(job_id) if document is not None else None
        if job is None:
            return
        for row in range(self.table.rowCount()):
            if self.job_id_at_row(row) == job_id:
                item = self.table.item(row, self.STATUS_COLUMN)
                if item is not None:
                    self._apply_status_cell(item, job, now=utc_now())
                break

    def _on_progress_changed(self, job_id: UUID, _progress: GeometryProgress | None) -> None:
        self.refresh_job_status(job_id)

    @staticmethod
    def format_relative_age(then, *, now) -> str:
        """Render a compact relative age such as ``just now`` / ``2 min ago``."""

        if then is None:
            return "—"
        stamp = then if then.tzinfo is not None else then.replace(tzinfo=UTC)
        seconds = max(0, int((now - stamp).total_seconds()))
        if seconds < 45:
            return "just now"
        minutes = seconds // 60
        if minutes < 60:
            return f"{max(1, minutes)} min ago"
        hours = minutes // 60
        if hours < 24:
            return f"{hours} hr ago"
        days = hours // 24
        return f"{days} day{'s' if days != 1 else ''} ago"

    # ---------------------------------------------------------------- runtime

    def set_cluster_elapsed(
        self, elapsed_by_job_id: Mapping[object, object] | None, checked_at: object
    ) -> None:
        """Take the scheduler's elapsed time per job from a queue check.

        ``elapsed_by_job_id`` maps job ids to the scheduler's ``[D-]H:M:S`` text;
        ``None`` keeps the times already known. A running cluster job's runtime
        cell counts on from its elapsed time at ``checked_at``.
        """

        if isinstance(elapsed_by_job_id, Mapping):
            parsed: dict[UUID, int] = {}
            for job_id, elapsed in elapsed_by_job_id.items():
                try:
                    normalized_job_id = job_id if isinstance(job_id, UUID) else UUID(str(job_id))
                except (TypeError, ValueError, AttributeError):
                    continue
                seconds = self._parse_scheduler_elapsed_seconds(str(elapsed))
                if seconds is not None:
                    parsed[normalized_job_id] = seconds
            self._cluster_elapsed_seconds_by_job = parsed
        self._cluster_elapsed_checked_at = checked_at if isinstance(checked_at, datetime) else None

    def clear_cluster_elapsed(self) -> None:
        """Forget the scheduler's elapsed times, as when the cluster disconnects."""

        self._cluster_elapsed_seconds_by_job = {}
        self._cluster_elapsed_checked_at = None

    def refresh_runtime_cells(self) -> None:
        """Advance the runtime cells of the running jobs the user can see.

        Runs every second while a job is running, so it reads only the rows in
        the viewport and writes only the runtime cells of running or
        retrieving jobs whose text changed; scrolling brings newly shown rows
        current. The rest of the table is redrawn by ``refresh``.
        It writes a handful of cells, so it leaves repaint batching to Qt.
        """

        document = self._deps.document()
        if document is None:
            self.runtime_timer.stop()
            return
        if self._syncing:
            # A row pass is under way (a removal clamped the scroll bar); it
            # writes every runtime cell itself.
            return

        table = self.table
        runtime_grew = False
        for row in self._viewport_rows():
            runtime_item = table.item(row, self.RUNTIME_COLUMN)
            job_id = self.job_id_at_row(row)
            job = document.jobs.get(job_id) if job_id is not None else None
            if (
                runtime_item is None
                or job is None
                or job.activity not in {JobActivity.RUNNING, JobActivity.RETRIEVING}
            ):
                continue
            old_text = runtime_item.text()
            if _set_item_text(runtime_item, self._runtime_text(job)):
                runtime_grew = runtime_grew or len(runtime_item.text()) > len(old_text)
        if runtime_grew:
            self.fit_columns((self.RUNTIME_COLUMN,))

        if not self._has_live_runtime(document):
            self.runtime_timer.stop()

    def _viewport_rows(self) -> range:
        """Return the rows inside the table's viewport.

        Every row while the table is hidden (the dock is closed or not shown
        yet), so an update made then is not lost.
        """

        table = self.table
        count = table.rowCount()
        if count == 0:
            return range(0)
        if not table.isVisible():
            return range(count)
        height = table.viewport().height()
        first = table.rowAt(0)
        last = table.rowAt(height - 1) if height > 0 else -1
        if first < 0:
            first = 0
        if last < 0:
            last = count - 1
        return range(first, last + 1)

    def _job_has_live_runtime(self, job: Job) -> bool:
        """Return True when the job's runtime cell advances with the clock."""

        return job.activity in {JobActivity.RUNNING, JobActivity.RETRIEVING} and (
            job.started_at is not None
            or (job.execution_target != "local" and job.id in self._cluster_elapsed_seconds_by_job)
        )

    def _has_live_runtime(self, document: ProjectDocument) -> bool:
        return any(
            self._job_has_live_runtime(job)
            for job in document.jobs.values()
            if job.status in VISIBLE_DASHBOARD_STATES
        )

    def sync_runtime_timer(self) -> None:
        """Tick once a second while a running job is in view, and not otherwise."""

        document = self._deps.document()
        if document is None:
            self.runtime_timer.stop()
            return
        # No point ticking once a second to update a table the user cannot see;
        # set_visible_to_user re-arms it when the dock returns.
        if self._has_live_runtime(document) and self._visible_to_user:
            if not self.runtime_timer.isActive():
                self.runtime_timer.start()
        else:
            self.runtime_timer.stop()

    def set_visible_to_user(self, visible: bool) -> None:
        """Follow the dock's visibility: tick and read progress only while it is seen."""

        self._visible_to_user = bool(visible)
        self._progress.set_enabled(self._visible_to_user)
        # On becoming visible, bring the (possibly stale) runtime cells current
        # before re-arming. sync_runtime_timer stops the timer when hidden.
        if self._visible_to_user:
            self.refresh_runtime_cells()
        self.sync_runtime_timer()

    @staticmethod
    def _format_runtime_seconds(elapsed_seconds: int) -> str:
        elapsed_seconds = max(0, int(elapsed_seconds))
        hours, remainder = divmod(elapsed_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)
        if hours:
            return f"{hours:d}:{minutes:02d}:{seconds:02d}"
        return f"{minutes:02d}:{seconds:02d}"

    @staticmethod
    def _parse_scheduler_elapsed_seconds(value: str) -> int | None:
        text = str(value).strip()
        if not text:
            return None
        day_count = 0
        if "-" in text:
            day_text, remainder = text.split("-", 1)
            try:
                day_count = int(day_text)
            except ValueError:
                return None
            text = remainder
        parts = text.split(":")
        try:
            numbers = [int(part) for part in parts]
        except ValueError:
            return None
        if len(numbers) == 2:
            hours = 0
            minutes, seconds = numbers
        elif len(numbers) == 3:
            hours, minutes, seconds = numbers
        else:
            return None
        return day_count * 86400 + hours * 3600 + minutes * 60 + seconds

    def _runtime_text(self, job: Job) -> str:
        if (
            job.execution_target != "local"
            and job.activity in {JobActivity.RUNNING, JobActivity.RETRIEVING}
            and (base_elapsed := self._cluster_elapsed_seconds_by_job.get(job.id)) is not None
        ):
            checked_at = self._cluster_elapsed_checked_at
            if checked_at is None:
                return self._format_runtime_seconds(base_elapsed)
            delta_seconds = max(0, int((utc_now() - checked_at).total_seconds()))
            return self._format_runtime_seconds(base_elapsed + delta_seconds)
        if job.started_at is None:
            return ""
        end_time = job.completed_at or utc_now()
        elapsed_seconds = max(0, int((end_time - job.started_at).total_seconds()))
        return self._format_runtime_seconds(elapsed_seconds)

    # ---------------------------------------------------------------- actions

    def action_applies(self, kind: str, job: Job) -> bool:
        """Return whether action ``kind`` can act on ``job``.

        Single source of truth for the row controls' multi-row prompt and both
        menus.
        """

        if kind == "continue_relaxation":
            return job.status in CONTINUABLE_JOB_STATUSES and self._deps.can_continue_relaxation(
                job
            )
        if kind == "open":
            return True
        if kind == "check_now":
            # Needs somewhere to look and a live connection, not just a status.
            return self._deps.can_check_now(job)
        if kind == "inspect" and job.activity == JobActivity.RUNNING:
            # A running geometry optimisation opens its convergence progress.
            return job_reports_geometry_progress(job)
        if kind == "inspect":
            # A failed or stopped job opens its failure; a cancelled one has none.
            return job.status == JobStatus.ENDED and job.end_reason != EndReason.CANCELLED
        eligible_statuses = {
            "resubmit": RESUBMITTABLE_JOB_STATES,
            "cancel": {JobStatus.ACTIVE},
            "reset": {JobStatus.ENDED},
            "dismiss": {JobStatus.ENDED},
        }
        if kind == "dismiss" and job.dismissed_at is not None:
            return False
        return job.status in eligible_statuses[kind]

    def dispatch_action(
        self,
        button: SelectionSnapshotPushButton,
        job_id: UUID,
        kind: str,
        *,
        snapshot_keys: Sequence[UUID] | None = None,
    ) -> None:
        """Run a row control's action on the jobs it targets.

        ``snapshot_keys`` are the jobs the selection held when the control was
        pressed, for a caller that read them before rows could have moved.
        """

        document = self._deps.document()
        if document is None:
            return

        if snapshot_keys is None:
            snapshot_keys = self._job_ids_at_rows(button.selection_snapshot_rows)

        def eligibility(uid: UUID) -> bool:
            job = document.jobs.get(uid)
            return job is not None and self.action_applies(kind, job)

        actions = {
            "resubmit": "Resubmit",
            "check_now": "Check",
            "cancel": "Cancel",
            "inspect": "Inspect",
            "reset": "Reset",
            "dismiss": "Dismiss",
        }

        # Cancel and Dismiss act immediately, so a single-row click is confirmed
        # here; on a multi-row selection the three-way prompt is the
        # confirmation. Reset always asks its own clean-up question, which has
        # a Cancel, so it needs nothing extra.
        clicked_job = document.jobs.get(job_id)
        clicked_label = clicked_job.label if clicked_job is not None else str(job_id)
        single_confirmations = {
            "cancel": lambda: self.confirm_cancel_and_dismiss([clicked_label], []),
            "dismiss": lambda: self.confirm_cancel_and_dismiss(
                [],
                [clicked_label],
                unretrieved_note=unretrieved_output_note(
                    [clicked_job] if clicked_job is not None else []
                ),
            ),
        }

        target_ids = resolve_target_keys(
            button,
            clicked_key=job_id,
            snapshot_keys=snapshot_keys,
            eligibility=eligibility,
            parent=self._deps.dialog_parent(),
            action=actions[kind],
            noun=("job", "jobs"),
            confirm_unprompted=single_confirmations.get(kind),
        )
        if not target_ids:
            return

        self.action_requested.emit(kind, list(target_ids))

    def _job_ids_at_rows(self, rows: Iterable[int]) -> list[UUID]:
        """Return the job ids of ``rows``."""

        job_ids = (self.job_id_at_row(row) for row in rows)
        return [job_id for job_id in job_ids if job_id is not None]

    def _rows_of_jobs(self, job_ids: Collection[UUID]) -> tuple[int, ...]:
        """Return the rows now showing ``job_ids``."""

        return tuple(
            row for row in range(self.table.rowCount()) if self.job_id_at_row(row) in job_ids
        )

    def open_row_menu(self, button: SelectionSnapshotPushButton, job_id: UUID) -> QMenu | None:
        """Open a row's ``⋯`` menu: what applies to its job, less the action button's action.

        An entry follows the same rule as the action button: on a row in a
        multi-row selection it asks whether to act on the selection. The jobs
        are read now, since a refresh can move the rows while the menu is open.
        """

        document = self._deps.document()
        job = document.jobs.get(job_id) if document is not None else None
        if job is None:
            return None
        snapshot_keys = self._job_ids_at_rows(button.selection_snapshot_rows)
        primary_kind, _text = self.primary_action(job)

        def run(kind: str) -> None:
            if kind in {"continue_relaxation", "open"}:
                self.action_requested.emit(kind, [job_id])
            elif shiboken6.isValid(button):
                button.selection_snapshot_rows = self._rows_of_jobs(snapshot_keys)
                self.dispatch_action(button, job_id, kind, snapshot_keys=snapshot_keys)
            else:
                # A refresh removed the job's row while the menu was open.
                self._deps.show_status(f"{job.label} has left the dashboard; nothing changed")

        menu = QMenu(self.table)
        for text, kind in self.MENU_ACTIONS:
            if kind != primary_kind and self.action_applies(kind, job):
                menu.addAction(text).triggered.connect(lambda _checked=False, k=kind: run(k))
        menu.addSeparator()
        menu.addAction(f"Open in {job_stage_name(job.workflow_stage)}").triggered.connect(
            lambda _checked=False: run("open")
        )
        menu.exec(button.mapToGlobal(button.rect().bottomLeft()))
        menu.deleteLater()
        return menu

    def open_context_menu(self, position) -> QMenu | None:
        """Offer the actions for the selected jobs.

        A right-click on a row outside the selection selects it first, so the
        menu always describes what is highlighted. Each action is enabled when
        it applies to at least one selected job and acts on those jobs only;
        for a mixed selection the entry says how many ("Reset… (2 of 3)").
        Continue relaxation opens a dialog for one job, so it needs exactly one
        that qualifies, and Open needs the selected jobs to share a page.
        """

        if self._deps.document() is None:
            return None
        row = self.table.rowAt(position.y())
        if row < 0:
            return None
        selection_model = self.table.selectionModel()
        selected_rows = {index.row() for index in selection_model.selectedRows()}
        if row not in selected_rows:
            self.table.selectRow(row)
        jobs = self.selected_jobs()
        if not jobs:
            return None

        menu = QMenu(self.table)
        menu.setToolTipsVisible(True)
        for text, kind in self.MENU_ACTIONS:
            eligible_ids = [job.id for job in jobs if self.action_applies(kind, job)]
            action = menu.addAction(text + partial_count_suffix(len(eligible_ids), len(jobs)))
            action.setEnabled(bool(eligible_ids))
            if kind == "continue_relaxation" and len(eligible_ids) > 1:
                action.setEnabled(False)
                action.setToolTip("Continue one relaxation at a time.")
            action.triggered.connect(
                lambda _checked=False, k=kind, ids=eligible_ids: self.run_menu_action(k, ids)
            )
        menu.addSeparator()
        stages = {job.workflow_stage for job in jobs}
        one_stage = len({job_stage_key(stage) for stage in stages}) == 1
        open_action = menu.addAction(
            f"Open in {job_stage_name(jobs[0].workflow_stage)}" if one_stage else "Open in stage"
        )
        open_action.setEnabled(one_stage)
        if not one_stage:
            open_action.setToolTip("The selected jobs belong to different stages.")
        open_action.triggered.connect(
            lambda _checked=False, ids=[job.id for job in jobs]: self.run_menu_action("open", ids)
        )
        menu.exec(self.table.viewport().mapToGlobal(position))
        menu.deleteLater()
        return menu

    def _on_double_clicked(self, index) -> None:
        job_id = self.row_job_id(self.table.item(index.row(), 0))
        if job_id is not None:
            self.action_requested.emit("open", [job_id])

    def run_menu_action(self, kind: str, job_ids: list[UUID]) -> None:
        """Run a context-menu entry on ``job_ids``, confirming Cancel and Dismiss first."""

        if not job_ids:
            return
        # Cancel and Dismiss act immediately, so they are confirmed here; Reset
        # asks its own clean-up question, which has a Cancel.
        document = self._deps.document()
        if kind in {"cancel", "dismiss"} and document is not None:
            jobs = [job for job_id in job_ids if (job := document.jobs.get(job_id)) is not None]
            labels = [job.label for job in jobs]
            confirmed = (
                self.confirm_cancel_and_dismiss(labels, [])
                if kind == "cancel"
                else self.confirm_cancel_and_dismiss(
                    [], labels, unretrieved_note=unretrieved_output_note(jobs)
                )
            )
            if not confirmed:
                return
        self.action_requested.emit(kind, list(job_ids))

    def delete_selected(self) -> None:
        """Apply Delete/Backspace to the selected jobs.

        Active jobs are cancelled and failed, stopped and cancelled ones are
        dismissed. One summary states both before anything happens, so a
        single refusal leaves every selected job untouched.
        """

        jobs = self.selected_jobs()
        active_jobs = [job for job in jobs if job.status == JobStatus.ACTIVE]
        dismissable_jobs = [job for job in jobs if self.action_applies("dismiss", job)]
        if not active_jobs and not dismissable_jobs:
            return
        if not self.confirm_cancel_and_dismiss(
            [job.label for job in active_jobs],
            [job.label for job in dismissable_jobs],
            unretrieved_note=unretrieved_output_note(
                dismissable_jobs, prompt_job_count=len(active_jobs) + len(dismissable_jobs)
            ),
        ):
            return
        self.cancel_and_dismiss_requested.emit(
            [job.id for job in active_jobs], [job.id for job in dismissable_jobs]
        )

    def confirm_cancel_and_dismiss(
        self,
        cancel_labels: Sequence[str],
        dismiss_labels: Sequence[str],
        *,
        unretrieved_note: str = "",
    ) -> bool:
        """Ask once before cancelling and dismissing jobs; a lone job is named.

        ``unretrieved_note`` warns that some of the jobs to dismiss have output
        still on the cluster (:func:`~mufinder.services.job_monitoring.unretrieved_output_note`).
        """

        parent = self._deps.dialog_parent()

        def with_note(detail: str) -> str:
            return f"{detail}\n\n{unretrieved_note}" if unretrieved_note else detail

        if len(cancel_labels) + len(dismiss_labels) == 1:
            if cancel_labels:
                return workflow_table_utils.confirm_destructive(
                    parent,
                    "Cancel job",
                    f"Cancel job '{cancel_labels[0]}'?",
                    "",
                    "Cancel job",
                    "Keep running",
                )
            return workflow_table_utils.confirm_destructive(
                parent,
                "Dismiss job",
                f"Dismiss job '{dismiss_labels[0]}' from the dashboard?",
                with_note("The job stays in the project. Show it again with Show dismissed."),
                "Dismiss job",
            )
        cancel_phrase = plural(len(cancel_labels), "job")
        dismiss_phrase = plural(len(dismiss_labels), "job")
        dismiss_note = "Dismissed jobs stay in the project. Show them again with Show dismissed."
        if not dismiss_labels:
            return workflow_table_utils.confirm_destructive(
                parent,
                "Cancel jobs",
                f"Cancel {cancel_phrase}?",
                "",
                f"Cancel {cancel_phrase}",
                "Keep running",
            )
        if not cancel_labels:
            return workflow_table_utils.confirm_destructive(
                parent,
                "Dismiss jobs",
                f"Dismiss {dismiss_phrase} from the dashboard?",
                with_note(dismiss_note),
                f"Dismiss {dismiss_phrase}",
            )
        return workflow_table_utils.confirm_destructive(
            parent,
            "Cancel and dismiss jobs",
            f"Cancel {cancel_phrase} and dismiss {dismiss_phrase} from the dashboard?",
            with_note(dismiss_note),
            f"Cancel {len(cancel_labels)}, dismiss {len(dismiss_labels)}",
            "Keep them",
        )

    def selected_jobs(self) -> list[Job]:
        """Return the selected rows' jobs, or the current row's when none is selected."""

        document = self._deps.document()
        if document is None:
            return []

        selection_model = self.table.selectionModel()
        if selection_model is None:
            return []

        selected_rows = selection_model.selectedRows()
        if not selected_rows and self.table.currentRow() >= 0:
            selected_rows = [self.table.model().index(self.table.currentRow(), 0)]

        selected: list[Job] = []
        seen: set[UUID] = set()
        for index in selected_rows:
            job_id = self.job_id_at_row(index.row())
            if job_id is None:
                continue
            job = document.jobs.get(job_id)
            if job is None or job.id in seen:
                continue
            selected.append(job)
            seen.add(job.id)
        return selected


class ClusterQueueStatus:
    """The status bar's cluster widgets: the connection, the last queue check and Refresh Queue.

    The window adds each widget to its status bar; the jobs controller keeps
    them current. They show only while the project has cluster jobs to follow.
    """

    def __init__(self, parent: QWidget) -> None:
        self.connection_label = QLabel(parent)
        self.queue_checked_label = QLabel("Queue last checked: never", parent)
        self.refresh_button = QPushButton("Refresh Queue", parent)

    def widgets(self) -> tuple[QWidget, ...]:
        """Return the widgets in the order they sit in the status bar."""

        return (self.connection_label, self.queue_checked_label, self.refresh_button)
