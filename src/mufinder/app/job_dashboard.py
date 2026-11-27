"""The Job Dashboard: the table of active and ended jobs, its actions and menus.

``JobDashboard`` is the content of the main window's Job Dashboard tray
(:mod:`mufinder.app.job_dashboard_tray`), which puts its ``Show dismissed`` box,
selection hint and job summary in the tray's header strip. It draws the jobs of
the open project, only those in the states it is filtered by when the summary's
counts are ticked (``set_state_filter``), keeps the running jobs' runtime cells
current, and offers each row's action buttons, the table's context menu and the
Delete and Backspace keys. It decides which action applies to which job and
confirms Cancel and Dismiss before anything happens; it never changes a job
itself. A confirmed action leaves as ``action_requested(kind,
job_ids)`` (or ``cancel_and_dismiss_requested`` for the Delete keys) for the
window to carry out (see :doc:`/job_lifecycle`, Dashboard).
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from html import escape
from uuid import UUID

from PySide6.QtCore import (
    QEvent,
    QItemSelectionModel,
    QObject,
    QPoint,
    QSignalBlocker,
    QSize,
    Qt,
    QTimer,
    Signal,
)
from PySide6.QtGui import QBrush, QContextMenuEvent, QIcon, QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QStyle,
    QStyleOptionButton,
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
    muted_text_brush,
    status_brush,
    use_square_top_table_corners,
)
from mufinder.app.workflow_submission import RESUBMITTABLE_JOB_STATES
from mufinder.app.workflow_table_utils import SelectionFeedback, partial_count_suffix
from mufinder.castep_io.geometry_progress import GeometryProgress
from mufinder.domain.enums import EndReason, JobActivity, JobStatus
from mufinder.domain.models import Job, as_aware_utc, utc_now
from mufinder.domain.text import plural
from mufinder.services.failure_diagnosis import failure_reason_summary, failure_status_reason
from mufinder.services.job_monitoring import (
    awaits_cluster_answer,
    check_now_reason,
    has_unretrieved_cluster_output,
    job_resumes_from_checkpoint,
    job_status_counts,
    job_status_word,
    job_where_text,
    jobs_in_states,
    parse_scheduler_elapsed_seconds,
    unretrieved_output_note,
)
from mufinder.services.job_progress import JobProgressMonitor, job_reports_geometry_progress
from mufinder.services.project_document import ProjectDocument
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
    #: Why the job's status may be out of date, or ``None``.
    staleness: Callable[[Job, datetime], str | None]
    #: Whether ``Resubmit…`` can run the job's workflow row (it has one).
    can_resubmit: Callable[[Job], bool]
    #: The widget confirmations and prompts are parented to (the main window).
    dialog_parent: Callable[[], QWidget]


def _asks_for_menu(event: QEvent) -> bool:
    """Return whether ``event`` is the keyboard asking for a context menu.

    That is the context-menu key or Shift+F10, or a context-menu event the
    platform raised from the keyboard.
    """

    if event.type() == QEvent.Type.ContextMenu:
        return event.reason() == QContextMenuEvent.Reason.Keyboard
    if event.type() != QEvent.Type.KeyPress:
        return False
    modifiers = event.modifiers() & ~Qt.KeyboardModifier.KeypadModifier
    if event.key() == Qt.Key.Key_Menu:
        return modifiers == Qt.KeyboardModifier.NoModifier
    return event.key() == Qt.Key.Key_F10 and modifiers == Qt.KeyboardModifier.ShiftModifier


def _as_uuid(value: object) -> UUID | None:
    """``value`` as a job id, or ``None`` when it is not one."""

    try:
        return value if isinstance(value, UUID) else UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        return None


#: Each action's label on a row button and in the context menu.
ACTION_TEXTS = {
    "cancel": "Cancel",
    "resubmit": "Resubmit…",
    "inspect": "Inspect…",
    "check_now": "Check now",
    "dismiss": "Dismiss",
}

#: The actions each of a row's three slots can show. Slot 0 runs or stops the
#: job, slot 1 looks at it and slot 2 dismisses it.
ROW_SLOTS: tuple[tuple[str, ...], ...] = (
    ("cancel", "resubmit"),
    ("inspect", "check_now"),
    ("dismiss",),
)

DISMISS_TOOLTIP = "Dismiss this job from the dashboard. Show dismissed lists it again."


class RowSlotButton(SelectionSnapshotPushButton):
    """One slot of a dashboard row, as wide as the widest action its slot can show.

    The buttons of a slot therefore line up down the table whatever each row
    shows, and a slot that shows nothing on a row keeps its space blank.
    """

    def __init__(self, slot_texts: Sequence[str]) -> None:
        super().__init__()
        self.slot_texts = tuple(slot_texts)
        #: The action the button runs now (a key of ``ACTION_TEXTS``).
        self.dashboard_action = ""
        policy = self.sizePolicy()
        policy.setHorizontalPolicy(QSizePolicy.Policy.Fixed)
        policy.setRetainSizeWhenHidden(True)
        self.setSizePolicy(policy)

    def sizeHint(self) -> QSize:  # noqa: N802 - Qt API naming.
        hint = super().sizeHint()
        if not self.slot_texts:
            return hint
        # QPushButton's own measure, taken for each label the slot can show.
        option = QStyleOptionButton()
        self.initStyleOption(option)
        metrics = self.fontMetrics()
        width = hint.width() if self.text() else 0
        for text in self.slot_texts:
            option.text = text
            size = metrics.size(Qt.TextFlag.TextShowMnemonic, text)
            contents = self.style().sizeFromContents(
                QStyle.ContentsType.CT_PushButton, option, size, self
            )
            width = max(width, contents.width())
        return QSize(width, hint.height())

    def minimumSizeHint(self) -> QSize:  # noqa: N802 - Qt API naming.
        return self.sizeHint()


class JobDashboard(QWidget):
    """The Job Dashboard dock's contents: the jobs table and its actions."""

    #: A confirmed action for jobs: ``(kind, job_ids)``. Kinds are ``cancel``,
    #: ``inspect``, ``check_now``, ``resubmit``, ``dismiss`` and ``open``.
    action_requested = Signal(str, object)
    #: Delete or Backspace, confirmed: ``(job_ids to cancel, job_ids to dismiss)``.
    cancel_and_dismiss_requested = Signal(object, object)
    #: ``Show dismissed`` was ticked or unticked.
    show_dismissed_toggled = Signal(bool)
    #: The listed jobs counted by state, ``(word, count)`` pairs, after each refresh.
    #: They count every listed job, whatever the state filter, and give each
    #: state the table is filtered by a pair even at count 0.
    summary_changed = Signal(object)
    #: The states the table is filtered by changed: the new ``state_filter()``.
    state_filter_changed = Signal(object)
    #: The height the rows need (``rows_height_hint``) may have changed: rows came
    #: or went, a row was resized, or the horizontal scroll bar showed or hid.
    rows_height_changed = Signal()

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

    #: Context-menu entries, before the final "Open in <stage>".
    MENU_ACTIONS = tuple(
        (ACTION_TEXTS[kind], kind)
        for kind in ("cancel", "inspect", "check_now", "resubmit", "dismiss")
    )

    _WALLTIME_NEXT_STEP = (
        "Resubmit to run its row again, with a longer walltime in the cluster "
        "submission dialog if needed."
    )
    _RESUMABLE_WALLTIME_NEXT_STEP = (
        "Resubmit to run its row again, continuing from its checkpoint or starting again, "
        "with a longer walltime in the cluster submission dialog if needed."
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
        # Per cluster job: the scheduler's elapsed seconds and when the check
        # that reported them ran (``None`` when it gave no time).
        self._cluster_elapsed_by_job: dict[UUID, tuple[int, datetime | None]] = {}
        # The status words the table is filtered by, in the summary's order; none lists all.
        self._state_filter: tuple[str, ...] = ()

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
        # The context-menu key and Shift+F10 open the menu on the current row,
        # from the table or from a row's buttons (see eventFilter).
        self.table.installEventFilter(self)
        self.table.verticalHeader().sectionResized.connect(
            lambda *_args: self.rows_height_changed.emit()
        )
        self.table.horizontalScrollBar().installEventFilter(self)
        # Shown only while two or more rows are selected; it is what tells the
        # user a row button acts on them all. The tray's header strip holds it.
        self.selection_hint = QLabel("", self)
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

        # The tray's header strip holds it, so it stays usable while the tray is collapsed.
        self.show_dismissed_box = QCheckBox("Show dismissed", self)
        self.show_dismissed_box.setAccessibleName("Show dismissed jobs")
        self.show_dismissed_box.setToolTip(
            "Show the jobs dismissed from the dashboard, greyed, with their actions"
        )
        self.show_dismissed_box.toggled.connect(self.show_dismissed_toggled)

        # The tray sets the table's height from rows_height_hint, and lets it be
        # dragged down to nothing, so the table asks for no minimum height.
        policy = self.table.sizePolicy()
        policy.setVerticalPolicy(QSizePolicy.Policy.Ignored)
        self.table.setSizePolicy(policy)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
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
            had_rows = self.table.rowCount() > 0
            self.table.setRowCount(0)
            self.runtime_timer.stop()
            self.clear_cluster_elapsed()
            self.summary_changed.emit(job_status_counts((), always=self._state_filter))
            if had_rows:
                self.rows_height_changed.emit()
            return

        now = utc_now()
        listed_jobs = [
            job
            for job in document.jobs.values()
            if job.status in VISIBLE_DASHBOARD_STATES
            and (job.dismissed_at is None or self.show_dismissed_box.isChecked())
        ]
        # Progress is followed for every listed job: an open progress dialog shows
        # it whether or not the filter shows the job's row.
        self._progress.follow(listed_jobs)
        # A job the state filter leaves out has no row, so nothing that acts on
        # rows (the selection, the row buttons, the menu, the Delete keys) reaches it.
        rows_changed = self._sync_rows(jobs_in_states(listed_jobs, self._state_filter), now=now)
        self.sync_runtime_timer()
        self.summary_changed.emit(job_status_counts(listed_jobs, always=self._state_filter))
        if rows_changed:
            self.rows_height_changed.emit()

    def rows_height_hint(self) -> int:
        """Return the height the table needs to show every row, and at least one.

        That is its header, its rows, its frame and its horizontal scroll bar
        while that shows; the tray fits itself to it.
        """

        table = self.table
        header = table.horizontalHeader()
        header_height = 0 if header.isHidden() else header.sizeHint().height()
        rows = table.verticalHeader()
        rows_height = rows.length() if table.rowCount() else rows.defaultSectionSize()
        scroll_bar = table.horizontalScrollBar()
        bar_height = scroll_bar.sizeHint().height() if scroll_bar.isVisibleTo(table) else 0
        return header_height + rows_height + bar_height + 2 * table.frameWidth()

    def hide_dismissed(self) -> None:
        """Untick ``Show dismissed`` without redrawing, as a newly opened project starts."""

        with QSignalBlocker(self.show_dismissed_box):
            self.show_dismissed_box.setChecked(False)

    def state_filter(self) -> tuple[str, ...]:
        """Return the status words the table is filtered by, in the summary's order.

        The table lists only the jobs in one of these states; with none it lists
        every job. The words are those of ``job_status_word``.
        """

        return self._state_filter

    def set_state_filter(self, words: Iterable[str]) -> None:
        """List only the jobs in the states ``words`` name, or every job when it is empty.

        The rows are redrawn and ``state_filter_changed`` emitted when the filter
        changes. Rows that stay keep their selection; the rows the filter hides
        leave it, and no row is current once the current row is hidden.
        """

        ordered = tuple(word for word, _count in job_status_counts((), always=words))
        if ordered == self._state_filter:
            return
        self._state_filter = ordered
        self.state_filter_changed.emit(ordered)
        current = self.job_id_at_row(self.table.currentRow())
        self.refresh()
        shown = self._job_ids_at_rows(range(self.table.rowCount()))
        if current is not None and current not in shown:
            # Qt makes a neighbouring row current, which the Delete keys and the
            # menu key would then act on with no row selected.
            self.table.selectionModel().clearCurrentIndex()

    def clear_state_filter(self) -> None:
        """List every job again, without redrawing, as a newly opened project starts.

        ``state_filter_changed`` is emitted; the rows follow on the next refresh.
        """

        if not self._state_filter:
            return
        self._state_filter = ()
        self.state_filter_changed.emit(())

    def slot_actions(self, job: Job) -> tuple[str | None, str | None, str | None]:
        """Return the action each of a row's three slots shows for ``job``, or ``None``.

        Slot 0 is ``Cancel`` on an active job and ``Resubmit…`` on an ended one
        its workflow row still holds. Slot 1 is ``Check now`` on a job whose
        output may still be on the cluster, else ``Inspect…`` where it applies.
        Slot 2 is Dismiss on an ended job not yet dismissed. A slot with no
        action is left blank rather than showing a button that does nothing.
        """

        if job.status == JobStatus.ACTIVE:
            run = "cancel"
        else:
            run = "resubmit" if self.action_applies("resubmit", job) else None
        look = next(
            (kind for kind in ("check_now", "inspect") if self.action_applies(kind, job)), None
        )
        dismiss = "dismiss" if self.action_applies("dismiss", job) else None
        return run, look, dismiss

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

    def _sync_rows(self, desired_jobs: list[Job], *, now) -> bool:
        """Reconcile visible rows against ``desired_jobs`` by job id; return whether rows came or went.

        Rows whose job disappeared are removed; brand-new jobs get a freshly
        built row inserted at the correct position; rows whose job id is
        unchanged are updated in place so their ``QTableWidgetItem`` and
        ``QPushButton`` cell widgets are never recreated (no widget churn, no
        stale button closures). A row is never repurposed for a different job
        id — mismatches are removed and re-inserted so each button's captured
        job id stays correct. An inserted row starts unselected, even between
        selected rows (Qt splits a selected range around it), so a row the state
        filter shows again is not selected.
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
        return rows_changed

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
        """Build a row's three action slots (see :meth:`slot_actions`)."""

        buttons: list[RowSlotButton] = []
        for kinds in ROW_SLOTS:
            if kinds == ("dismiss",):
                button = RowSlotButton(())
                button.setObjectName("jobRowDismissButton")
                button.setIcon(icon("dismiss"))
                button.setToolTip(DISMISS_TOOLTIP)
            else:
                button = RowSlotButton([ACTION_TEXTS[kind] for kind in kinds])
            button.selection_snapshot_key = job.id
            # Outside a dialog a push button takes Return and Enter only when told to.
            button.setAutoDefault(True)
            button.installEventFilter(self)
            button.clicked.connect(
                lambda _checked=False, jid=job.id, btn=button: self.dispatch_action(
                    btn, jid, btn.dashboard_action
                )
            )
            buttons.append(button)
        cell = RowControlsCell(*buttons)
        self._apply_slot_actions(cell, job)
        return cell

    def _apply_slot_actions(self, cell: RowControlsCell, job: Job) -> None:
        """Show in each of a row's slots the action ``job``'s state calls for, or nothing.

        A slot keeps its width whatever it shows, so the column's width never changes.
        """

        for button, kind in zip(cell.row_controls, self.slot_actions(job), strict=True):
            if kind is not None:
                button.dashboard_action = kind
                text = "" if kind == "dismiss" else ACTION_TEXTS[kind]
                if button.text() != text:
                    button.setText(text)
                name = f"{ACTION_TEXTS[kind].rstrip('…')} {job.label}"
                if button.accessibleName() != name:
                    button.setAccessibleName(name)
            if button.isHidden() != (kind is None):
                button.setHidden(kind is None)

    def _update_row(self, row: int, job: Job, *, now) -> set[int]:
        """Refresh an existing row in place without recreating items or widgets.

        Updates cell texts, the status cell and the action slots, writing only
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
            self._apply_slot_actions(action_cell, job)
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
        reason = check_now_reason(job)
        if job.status == JobStatus.ENDED and job.end_reason != EndReason.CANCELLED:
            detail = failure_status_reason(job.end_reason, job.error_message)
            tooltip_parts.append(self.failure_tooltip(job, check_now_reason=reason))
        elif job.activity == JobActivity.RUNNING:
            progress = self._progress.progress_for(job.id)
            detail = progress_status_text(progress)
        if reason:
            tooltip_parts.append(reason)
        stale_reason = self._deps.staleness(job, now)
        check_lines = self._last_check_lines(
            job, stale_reason, now=now, stamp_explained=bool(reason) and awaits_cluster_answer(job)
        )
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

    def _last_check_lines(
        self, job: Job, stale_reason: str | None, *, now, stamp_explained: bool = False
    ) -> list[str]:
        """Return the status tooltip's lines on when and how the job was last checked.

        ``stamp_explained`` leaves out the source and message when the tooltip
        already says in words what that stamp means.
        """

        lines: list[str] = []
        if job.last_checked_at is not None:
            lines.append(f"Last checked: {self.format_relative_age(job.last_checked_at, now=now)}")
        if not stamp_explained:
            if job.last_status_source:
                lines.append(f"Source: {job.last_status_source}")
            if job.last_status_message:
                lines.append(f"Message: {job.last_status_message}")
        if stale_reason is not None:
            lines.append(f"Stale: {stale_reason}")
        return lines

    @classmethod
    def failure_tooltip(cls, job: Job, *, check_now_reason: str = "") -> str:
        """Return the status tooltip's explanation of why ``job`` ended.

        ``check_now_reason`` is what the tooltip goes on to say about
        ``Check now``; when there is one, it replaces the pointer to Inspect.
        """

        summary = failure_reason_summary(job.end_reason)
        message = (job.error_message or "").strip()
        if job.end_reason in (None, EndReason.UNKNOWN) and message:
            if check_now_reason:
                return message
            return f"{message}\n\nUse Inspect to see the job's logs."
        parts = [summary]
        if message and message != summary:
            parts.append(message)
        if job.end_reason == EndReason.WALLTIME:
            parts.append(
                cls._RESUMABLE_WALLTIME_NEXT_STEP
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
        self,
        elapsed_by_job_id: Mapping[object, object] | None,
        checked_at: object,
        *,
        checked_job_ids: Iterable[object] | None = None,
    ) -> None:
        """Take the scheduler's elapsed time per job from a queue check.

        ``elapsed_by_job_id`` maps job ids to the scheduler's ``[D-]H:M:S`` text;
        ``None`` keeps the times already known. A running cluster job's runtime
        cell counts on from its elapsed time at ``checked_at``, the time of the
        check that reported it.

        ``checked_job_ids`` names the jobs the check looked at. Only their times
        change: each is set from the check, or forgotten when the check gave
        none, and every other job keeps the time its own last check gave.
        ``None`` means the check covered every job, so the times are replaced.
        """

        if not isinstance(elapsed_by_job_id, Mapping):
            return
        stamp = checked_at if isinstance(checked_at, datetime) else None
        reported: dict[UUID, tuple[int, datetime | None]] = {}
        for job_id, elapsed in elapsed_by_job_id.items():
            normalized_job_id = _as_uuid(job_id)
            seconds = parse_scheduler_elapsed_seconds(str(elapsed))
            if normalized_job_id is not None and seconds is not None:
                reported[normalized_job_id] = (seconds, stamp)
        if checked_job_ids is None:
            self._cluster_elapsed_by_job = reported
            return
        for job_id in checked_job_ids:
            if (normalized_job_id := _as_uuid(job_id)) is not None:
                self._cluster_elapsed_by_job.pop(normalized_job_id, None)
        self._cluster_elapsed_by_job.update(reported)

    def clear_cluster_elapsed(self) -> None:
        """Forget the scheduler's elapsed times, as when the cluster disconnects."""

        self._cluster_elapsed_by_job = {}

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

        Every row while the table is hidden (the tray is closed, collapsed or
        not shown yet), so an update made then is not lost.
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
            job.started_at is not None or self._scheduler_elapsed(job) is not None
        )

    def _scheduler_elapsed(self, job: Job) -> tuple[int, datetime | None] | None:
        """The scheduler's elapsed time for cluster ``job``'s current run, and when it was read.

        A time read before the job's latest submission belongs to an earlier
        run: a job leaves the checks once it ends, so its last time stays known
        when it is submitted again.
        """

        if job.execution_target == "local":
            return None
        known = self._cluster_elapsed_by_job.get(job.id)
        if known is None:
            return None
        _seconds, checked_at = known
        if (
            checked_at is not None
            and job.submitted_at is not None
            and as_aware_utc(checked_at) < as_aware_utc(job.submitted_at)
        ):
            return None
        return known

    def _has_live_runtime(self, document: ProjectDocument) -> bool:
        # The jobs with a row: a running job the state filter hides has no cell to tick.
        listed = (job for job in document.jobs.values() if job.status in VISIBLE_DASHBOARD_STATES)
        return any(
            self._job_has_live_runtime(job) for job in jobs_in_states(listed, self._state_filter)
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

    def set_visible_to_user(self, visible: bool, *, rows_shown: bool = True) -> None:
        """Follow whether the dashboard is seen: read progress while it is, tick while its rows are.

        ``visible`` is false while the tray is closed, obscured or minimised.
        ``rows_shown`` is false while the tray is collapsed to its header strip:
        the runtimes then stop ticking, but progress is still read, since an open
        progress dialog shows it.
        """

        self._progress.set_enabled(bool(visible))
        self._visible_to_user = bool(visible) and bool(rows_shown)
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

    def _runtime_text(self, job: Job) -> str:
        if (
            job.activity in {JobActivity.RUNNING, JobActivity.RETRIEVING}
            and (known := self._scheduler_elapsed(job)) is not None
        ):
            base_elapsed, checked_at = known
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

        if kind == "resubmit":
            # Resubmit runs the job's workflow row, so a job without one has none.
            return job.status in RESUBMITTABLE_JOB_STATES and self._deps.can_resubmit(job)
        if kind == "open":
            return True
        if kind == "check_now":
            # Only output that may still be on the cluster is worth a check from
            # here; Refresh Queue and the automatic checks cover every other job.
            # A cluster that is not connected is connected to first.
            return has_unretrieved_cluster_output(job)
        if kind == "inspect" and job.status == JobStatus.ACTIVE:
            if job.activity != JobActivity.RUNNING:
                return False
            # A running geometry optimisation opens its convergence progress.
            return job_reports_geometry_progress(job)
        if kind == "inspect":
            # A failed or stopped job opens its failure; a cancelled one has none.
            return job.status == JobStatus.ENDED and job.end_reason != EndReason.CANCELLED
        eligible_statuses = {
            "cancel": {JobStatus.ACTIVE},
            "dismiss": {JobStatus.ENDED},
        }
        if kind == "dismiss" and job.dismissed_at is not None:
            return False
        return job.status in eligible_statuses[kind]

    def dispatch_action(self, button: SelectionSnapshotPushButton, job_id: UUID, kind: str) -> None:
        """Run a row button's action on the jobs it targets."""

        document = self._deps.document()
        if document is None:
            return

        snapshot_keys = self._job_ids_at_rows(button.selection_snapshot_rows)

        def eligibility(uid: UUID) -> bool:
            job = document.jobs.get(uid)
            return job is not None and self.action_applies(kind, job)

        actions = {
            "resubmit": "Resubmit",
            "check_now": "Check",
            "cancel": "Cancel",
            "inspect": "Inspect",
            "dismiss": "Dismiss",
        }

        # Cancel and Dismiss act immediately, so a single-row click is confirmed
        # here; on a multi-row selection the three-way prompt is the
        # confirmation.
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

    def open_context_menu(self, position: QPoint) -> QMenu | None:
        """Offer the actions for the selected jobs, at ``position`` in the table's viewport.

        A right-click on a row outside the selection selects it first, so the
        menu always describes what is highlighted. Each action is enabled when
        it applies to at least one selected job and acts on those jobs only;
        for a mixed selection the entry says how many ("Dismiss (2 of 3)").
        Open needs the selected jobs to share a page.
        """

        row = self.table.rowAt(position.y())
        if row < 0:
            return None
        return self._open_context_menu_on_row(row, self.table.viewport().mapToGlobal(position))

    def open_context_menu_from_keyboard(self, anchor: QWidget | None = None) -> QMenu | None:
        """Open the context menu as the context-menu key or Shift+F10 does.

        ``anchor`` is the row button that has focus, if any: the menu is about
        its row and opens below it. Otherwise it is about the current row and
        opens below that row's ``Job`` cell.
        """

        table = self.table
        row = table.currentRow()
        if anchor is not None:
            row = next(
                (
                    candidate
                    for candidate in range(table.rowCount())
                    if anchor
                    in getattr(table.cellWidget(candidate, self.ACTION_COLUMN), "row_controls", ())
                ),
                row,
            )
        if row < 0:
            return None
        if anchor is not None:
            where = anchor.mapToGlobal(anchor.rect().bottomLeft())
        else:
            rect = table.visualRect(table.model().index(row, self.LABEL_COLUMN))
            where = table.viewport().mapToGlobal(rect.bottomLeft())
        return self._open_context_menu_on_row(row, where)

    def _open_context_menu_on_row(self, row: int, global_position: QPoint) -> QMenu | None:
        """Open the context menu for the selection, selecting ``row`` first when outside it."""

        if self._deps.document() is None:
            return None
        selection_model = self.table.selectionModel()
        selected_rows = {index.row() for index in selection_model.selectedRows()}
        if row not in selected_rows:
            # Not selectRow: that extends the selection while Shift is held, as
            # it is for Shift+F10.
            selection_model.setCurrentIndex(
                self.table.model().index(row, self.LABEL_COLUMN),
                QItemSelectionModel.SelectionFlag.ClearAndSelect
                | QItemSelectionModel.SelectionFlag.Rows,
            )
        jobs = self.selected_jobs()
        if not jobs:
            return None

        menu = QMenu(self.table)
        menu.setToolTipsVisible(True)
        for text, kind in self.MENU_ACTIONS:
            eligible_ids = [job.id for job in jobs if self.action_applies(kind, job)]
            action = menu.addAction(text + partial_count_suffix(len(eligible_ids), len(jobs)))
            action.setEnabled(bool(eligible_ids))
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
        menu.exec(global_position)
        menu.deleteLater()
        return menu

    def eventFilter(self, watched: QObject, event: QEvent) -> bool:  # noqa: N802 - Qt API naming.
        """Open the context menu from the keyboard, on the table or on a row's button.

        Also report the horizontal scroll bar showing or hiding, which changes the
        height the rows need.
        """

        if watched is self.table.horizontalScrollBar() and event.type() in (
            QEvent.Type.Show,
            QEvent.Type.Hide,
        ):
            self.rows_height_changed.emit()
            return False
        if (watched is self.table or isinstance(watched, RowSlotButton)) and _asks_for_menu(event):
            self.open_context_menu_from_keyboard(None if watched is self.table else watched)
            return True
        return super().eventFilter(watched, event)

    def _on_double_clicked(self, index) -> None:
        job_id = self.row_job_id(self.table.item(index.row(), 0))
        if job_id is not None:
            self.action_requested.emit("open", [job_id])

    def run_menu_action(self, kind: str, job_ids: list[UUID]) -> None:
        """Run a context-menu entry on ``job_ids``, confirming Cancel and Dismiss first.

        A job whose row went while the menu was open (it left the dashboard, or
        the states the table is filtered by) is left out: the menu acts on the
        rows shown.
        """

        shown = set(self._job_ids_at_rows(range(self.table.rowCount())))
        job_ids = [job_id for job_id in job_ids if job_id in shown]
        if not job_ids:
            return
        # Cancel and Dismiss act immediately, so they are confirmed here;
        # Resubmit asks its own questions, each with a Cancel.
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

        if self.table.isHidden():
            # The tray is collapsed: no row is in sight to be the one meant.
            return
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
