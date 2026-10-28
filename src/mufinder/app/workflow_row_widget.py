"""Row-table widget bases for the job-staging workflow widgets.

Pristine relaxation, muon relaxation and UEP each stage one CASTEP job per
table row. A row is identified by an integer row id stored in its label item's
``UserRole``, and submitted under its *submission key*: its label, or for muon
relaxation the label qualified by the row's charge state.

:class:`RowTableWidget` owns everything those widgets share: the table and its
row scaffold (label, supercell, status, Configure and remove cells), row ids,
labels and keys, the selection, the row menu and delete shortcuts, the
parameter dialog, removal, runtime-state refresh, config validation and
submission, the form-state plumbing and the page layout.
:class:`RelaxationRowTableWidget` adds the geometry-relaxation parameter
surface, the row parameters stored as a diff from the implicit defaults, and
the "Continue relaxation…" row action.

A subclass declares its columns and wording as class attributes and supplies
the workflow-specific parts through three public hooks, as the controller base
does: :meth:`RowTableWidget.create_row_cells` and
:meth:`RowTableWidget.refresh_row_cells` for its own columns, and
:meth:`RowTableWidget.config_for_rows` for its run configuration. It keeps its
own ``collect_state`` / ``set_form_state``, built from the shared helpers here.

:class:`SourceTableWidget` is the base for quantum corrections and energy
barriers, whose rows are project sources keyed by a source key rather than
staged labelled rows. It owns the key lookups, run and Configure targeting,
the row menu, the table fill with the per-row Configure button and the
action-state refresh; a subclass adds its own cells and actions through the
public hooks :meth:`SourceTableWidget.source_row_values`,
:meth:`SourceTableWidget.create_source_row_cells` and
:meth:`SourceTableWidget.refresh_source_actions`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Any

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.export_dialogs import STRUCTURE_FORMATS, JobFilesMenu, export_relaxed_structure
from mufinder.app.adapter_common import coerce_supercell_matrix
from mufinder.app.project_defaults_widget import ProjectDefaultsStrip
from mufinder.app.relaxation_types import SupercellMatrix
from mufinder.app.selection_snapshot import (
    SelectionSnapshotPushButton,
    SelectionSnapshotToolButton,
    resolve_target_keys,
)
from mufinder.app.theme import (
    status_brush,
    style_hint_text,
    style_section_subtitle,
    style_section_title,
)
from mufinder.app.workflow_dialogs import RowParameterRequest, run_row_parameter_dialog
from mufinder.app.workflow_parameters import (
    drop_default_conflicts,
    drop_explicit_conflicts,
    geometry_relaxation_parameter_specs,
    sanitize_parameter_mapping,
    seed_xc_functional_default,
    shared_parameter_values,
    workflow_default_params,
)
from mufinder.app.workflow_table_utils import (
    EXPORT_RELAXED_STRUCTURE_ACTION_TEXT,
    REMOVE_SELECTED_TEXT,
    RowMenuAction,
    SelectionFeedback,
    SupercellColumn,
    bind_delete_shortcuts,
    bind_reset_failed_context_menu,
    confirm_row_removal,
    create_workflow_table,
    selected_table_rows,
    set_workflow_table_item,
    standard_row_actions,
    style_workflow_configure_button,
    workflow_row_is_locked,
    workflow_row_status_text,
)
from mufinder.castep_io.gui_metadata import infer_task, validate_gui_values
from mufinder.castep_io.schema import KeywordSpec
from mufinder.castep_io.types import ValueWithUnit
from mufinder.services.structure_export import NOT_COMPLETED_REASON, RelaxedStructureExport

_LOCKED_LABEL_TOOLTIP = "Job label is locked once this row has been queued for execution."


@dataclass(frozen=True)
class RowEntry:
    """One row of a row table that has a row id: where it is and its label."""

    table: QTableWidget
    row: int
    row_id: int
    label: str


class RowTableWidget(QWidget):
    """Base for a workflow widget that stages one labelled job per table row.

    Class attributes a subclass sets:

    - the column indexes ``_STATUS_COLUMN``, ``_CONFIGURE_COLUMN`` and
      ``_REMOVE_COLUMN`` (label and supercell are columns 0 and 1) and the
      ``_COLUMN_HEADERS``;
    - ``_WORKFLOW_STAGE`` (its CASTEP task follows from it) and
      ``_PINNED_PARAMS``, values every row carries whatever the dialog says;
    - the parameter dialog's ``_DIALOG_SUBJECT``, ``_LOCKED_DIALOG_VALUES`` and
      fixed ``_DIALOG_INTRO`` (default: the included row labels);
    - ``_EMPTY_STATUS`` for a row with no job, ``_CONFIGURE_NOUN``,
      ``_ROW_LABEL_TEMPLATE``, ``_NAME_PREFIX_DEFAULT`` (``None``: no job-name
      prefix field), ``_ADD_ROW_TEXT`` (``None``: no Add button),
      ``_NO_ROWS_MESSAGE``, ``_TABLE_TOOLTIP`` and ``_READ_ONLY_ROW_TOOLTIP``.

    A subclass implements :meth:`create_row_cells`, :meth:`refresh_row_cells`,
    :meth:`config_for_rows`, ``collect_state`` and ``set_form_state``, plus the
    parameter surface (``_build_parameter_specs``, ``_initial_default_params``,
    ``_effective_params_for_row`` and ``set_project_defaults``) unless it
    inherits :class:`RelaxationRowTableWidget`'s. Its ``__init__`` calls
    ``super().__init__()``, builds its own widgets, then :meth:`_build_layout`
    and :meth:`_connect_signals`.
    """

    configure_requested = Signal(object)
    submit_local_requested = Signal(object)
    submit_cluster_requested = Signal(object)
    cancel_requested = Signal(object)
    inspect_failed_requested = Signal(object)
    reset_failed_requested = Signal(object)
    remove_requested = Signal(object)
    row_supercell_edit_requested = Signal(object)
    form_state_changed = Signal(dict)

    _LABEL_COLUMN = 0
    _SUPERCELL_COLUMN = 1
    _STATUS_COLUMN: int
    _CONFIGURE_COLUMN: int
    _REMOVE_COLUMN: int
    _COLUMN_HEADERS: tuple[str, ...] = ()

    _WORKFLOW_STAGE = ""
    _PINNED_PARAMS: Mapping[str, object] = {}
    _DIALOG_SUBJECT = "Parameters"
    _DIALOG_INTRO: str | None = None
    _LOCKED_DIALOG_VALUES: Mapping[str, str] = {}

    _EMPTY_STATUS = "not generated"
    _CONFIGURE_NOUN = "row"
    _ROW_LABEL_TEMPLATE = "row_{number}"
    _NAME_PREFIX_DEFAULT: str | None = None
    _ADD_ROW_TEXT: str | None = "Add Job"
    _NO_ROWS_MESSAGE = "Add one or more rows first."
    _TABLE_TOOLTIP = (
        "Right-click selected rows to run, cancel, inspect, reset, configure, or remove them."
    )
    _READ_ONLY_ROW_TOOLTIP = ""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        #: Open Containing Folder / Export Inputs/Outputs… in every row menu.
        self.job_files_menu = JobFilesMenu()
        self._restoring_form_state = False
        self._suppress_state_emission = False
        self._next_row_id = 1
        self._row_parameter_values: dict[int, dict[str, object]] = {}
        self._row_added_advanced: dict[int, set[str]] = {}
        self._runtime_states: dict[str, Any] = {}
        self._parameter_species_options: tuple[str, ...] = ()
        self._selection_feedbacks: list[SelectionFeedback] = []

        self.project_defaults_strip = ProjectDefaultsStrip()
        self._kpoint_message = QLabel("")
        self._kpoint_message.setWordWrap(True)
        style_hint_text(self._kpoint_message)
        self._kpoint_message.hide()
        self._supercells = SupercellColumn(
            self,
            column=self._SUPERCELL_COLUMN,
            manage_requested=self.row_supercell_edit_requested.emit,
            normalize_rows=self._normalize_all_row_parameter_values,
            refresh_rows=self._refresh_all_rows,
            state_changed=self._emit_state,
            eligibility=self._row_is_editable,
            kpoint_message=self._kpoint_message,
        )

        self._tables = self._create_tables()

        self._name_prefix = (
            QLineEdit(self._NAME_PREFIX_DEFAULT) if self._NAME_PREFIX_DEFAULT is not None else None
        )
        self._add_row_button = QPushButton(self._ADD_ROW_TEXT) if self._ADD_ROW_TEXT else None
        self._batch_configure_button = QPushButton("Configure Selected Parameters...")
        self._batch_configure_button.setEnabled(False)
        self._remove_selected_button = QPushButton(REMOVE_SELECTED_TEXT)
        self._selection_count_label = QLabel("")
        style_hint_text(self._selection_count_label)
        # SelectionFeedback sets the run buttons' text ("Run All ..." / "Run
        # Selected ...") from the selection.
        self._run_local_button = QPushButton()
        self._run_cluster_button = QPushButton()

        self._specs = self._build_parameter_specs()
        self._specs_by_name = {spec.name: spec for spec in self._specs}
        self._default_params = self._initial_default_params()

        self._delete_shortcut, self._backspace_shortcut = bind_delete_shortcuts(
            self, self._delete_selected_rows
        )

    # -- Hooks ---------------------------------------------------------------

    def create_row_cells(self, table: QTableWidget, row: int, row_id: int) -> None:
        """Fill the workflow's own cells of a newly inserted row."""

        raise NotImplementedError

    def refresh_row_cells(
        self,
        table: QTableWidget,
        row: int,
        row_id: int,
        runtime: Any,
        *,
        status: str,
        locked: bool,
    ) -> None:
        """Show ``runtime`` (``None`` before the row has a job) in the workflow's own cells."""

        raise NotImplementedError

    def config_for_rows(self, rows: Sequence[RowEntry]) -> object:
        """Return the run configuration for ``rows``, already validated."""

        raise NotImplementedError

    def collect_state(self) -> dict[str, Any]:
        raise NotImplementedError

    def _build_parameter_specs(self) -> list[KeywordSpec]:
        raise NotImplementedError

    def _initial_default_params(self) -> dict[str, object]:
        raise NotImplementedError

    def _effective_params_for_row(self, row_id: int) -> dict[str, object]:
        raise NotImplementedError

    def _stored_row_values(self, row_id: int, values: Mapping[str, object]) -> dict[str, object]:
        """Return what a row stores for ``values`` returned by the parameter dialog.

        By default the row keeps the dialog's values with the pinned ones.
        """

        del row_id
        return {**values, **self._pinned_params()}

    def _normalize_all_row_parameter_values(self) -> None:
        """Re-derive every row's stored values after the implicit defaults moved.

        Nothing to do while rows store the dialog's full values.
        """

    def _row_is_read_only(self, row_id: int) -> bool:
        """Return whether the row records a run that already happened (never edited)."""

        del row_id
        return False

    def _configurable_row_ids(self, row_ids: tuple[int, ...]) -> tuple[int, ...]:
        """Return the selected rows the batch Configure action may edit."""

        return row_ids

    def _removable_row_ids(self, row_ids: tuple[int, ...]) -> tuple[int, ...]:
        """Return the rows the table may remove itself."""

        return row_ids

    def _forget_row(self, row_id: int) -> None:
        """Drop the per-row state of a removed row."""

        self._supercells.matrices.pop(row_id, None)
        self._row_parameter_values.pop(row_id, None)
        self._row_added_advanced.pop(row_id, None)

    def _extra_row_actions(self) -> tuple[RowMenuAction, ...]:
        """Return the workflow's row-menu actions after the shared core."""

        return ()

    def _dialog_locked_values(self, row_ids: tuple[int, ...]) -> dict[str, str]:
        del row_ids
        return dict(self._LOCKED_DIALOG_VALUES)

    def _row_display_label(self, row: int, *, table: QTableWidget) -> str:
        """Return how the row is named in dialogs and warnings."""

        return self._label_at_row(row, table=table)

    def _submission_key_at_row(self, row: int, *, table: QTableWidget | None = None) -> str:
        """Return the key the row's job is submitted and tracked under."""

        return self._label_at_row(row, table=table)

    # -- Tables ----------------------------------------------------------------

    def _create_tables(self) -> tuple[QTableWidget, ...]:
        self._table = self._create_table()
        return (self._table,)

    def _create_table(
        self,
        header_factory: Callable[[QTableWidget], QHeaderView] | None = None,
    ) -> QTableWidget:
        """Return an editable job table with the workflow's columns.

        Label and supercell stretch, the remove column is a fixed 34 px, and
        every other column fits its contents. ``header_factory`` installs a
        custom horizontal header.
        """

        headers = self._COLUMN_HEADERS
        table = create_workflow_table(headers)
        if header_factory is not None:
            table.setHorizontalHeader(header_factory(table))
            table.setHorizontalHeaderLabels(list(headers))
        # Per-widget settings the shared helper does not cover (or that this
        # editable table overrides): extended multi-select, in-place editing,
        # a fixed row height, per-column resize modes, and a non-stretching
        # last (Remove) column.
        table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
            | QAbstractItemView.EditTrigger.SelectedClicked
        )
        table.verticalHeader().setDefaultSectionSize(32)
        header = table.horizontalHeader()
        header.setStretchLastSection(False)
        for column in range(len(headers)):
            stretch = column in (self._LABEL_COLUMN, self._SUPERCELL_COLUMN)
            header.setSectionResizeMode(
                column,
                QHeaderView.ResizeMode.Stretch
                if stretch
                else QHeaderView.ResizeMode.ResizeToContents,
            )
        header.setSectionResizeMode(self._REMOVE_COLUMN, QHeaderView.ResizeMode.Fixed)
        table.setColumnWidth(self._REMOVE_COLUMN, 34)
        return table

    def _all_tables(self) -> tuple[QTableWidget, ...]:
        return self._tables

    def _active_table(self) -> QTableWidget:
        return self._table

    # -- Layout and wiring -------------------------------------------------------

    def _build_layout(
        self,
        *,
        title: str,
        subtitle: str,
        table_group_title: str,
        preamble: Sequence[QWidget] = (),
        table_view: QWidget | None = None,
        control_leading: QWidget | None = None,
        run_extras: Sequence[QWidget] = (),
        separate_run_group: bool = True,
    ) -> None:
        """Lay out the page: header, ``preamble``, job table with controls, run row.

        The controls row holds ``control_leading`` (default: the Add button),
        the selection count, Remove Selected and Configure Selected
        Parameters. The run row holds the job-name prefix (when the widget has
        one), the run buttons and ``run_extras``, in its own "Run Jobs" group
        unless ``separate_run_group`` is false.
        """

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        header_row = QHBoxLayout()
        title_column = QVBoxLayout()
        title_column.setSpacing(3)
        title_label = QLabel(title)
        style_section_title(title_label)
        subtitle_label = QLabel(subtitle)
        subtitle_label.setWordWrap(True)
        style_section_subtitle(subtitle_label)
        title_column.addWidget(title_label)
        title_column.addWidget(subtitle_label)
        header_row.addLayout(title_column, 1)
        header_row.addWidget(self.project_defaults_strip, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(header_row)

        for widget in preamble:
            layout.addWidget(widget)
        layout.addWidget(self._kpoint_message)

        table_group = QGroupBox(table_group_title)
        table_layout = QVBoxLayout(table_group)
        table_layout.setContentsMargins(10, 8, 10, 8)
        table_layout.addWidget(table_view if table_view is not None else self._table)

        control_row = QHBoxLayout()
        leading = control_leading if control_leading is not None else self._add_row_button
        if leading is not None:
            control_row.addWidget(leading)
        control_row.addStretch(1)
        control_row.addWidget(self._selection_count_label)
        control_row.addWidget(self._remove_selected_button)
        control_row.addWidget(self._batch_configure_button)
        table_layout.addLayout(control_row)

        run_row = QHBoxLayout()
        if self._name_prefix is not None:
            run_row.addWidget(QLabel("Job name prefix"))
            run_row.addWidget(self._name_prefix, 1)
        else:
            run_row.addStretch(1)
        run_row.addWidget(self._run_local_button)
        run_row.addWidget(self._run_cluster_button)
        for widget in run_extras:
            run_row.addWidget(widget)

        if separate_run_group:
            layout.addWidget(table_group, 1)
            run_group = QGroupBox("Run Jobs")
            run_layout = QVBoxLayout(run_group)
            run_layout.setContentsMargins(10, 8, 10, 8)
            run_layout.addLayout(run_row)
            layout.addWidget(run_group)
        else:
            # One row cannot hold the selection controls, the job-name prefix
            # and both run buttons at laptop width, so the run row sits under
            # the controls row inside the table group.
            table_layout.addLayout(run_row)
            layout.addWidget(table_group, 1)

    def _connect_signals(self) -> None:
        for table in self._all_tables():
            table.itemChanged.connect(self._on_table_item_changed)
            table.setToolTip(self._TABLE_TOOLTIP)
            # A right-click makes the rows under it the selection in that
            # table, so the run/configure/remove entries reuse the
            # selection-based handlers. Each table binds its own menu with the
            # table captured: it acts on the rows it was raised over.
            leading_actions, trailing_actions = standard_row_actions(
                status_for_row=lambda row, bound=table: self._row_status(row, table=bound),
                run_local=lambda _rows: self._run_local_selected_rows(),
                run_cluster=lambda _rows: self._run_cluster_selected_rows(),
                cancel=lambda rows, bound=table: self.cancel_requested.emit(
                    self._submission_keys_at_rows(rows, table=bound)
                ),
                inspect=lambda rows, bound=table: self.inspect_failed_requested.emit(
                    self._submission_keys_at_rows(rows, table=bound)
                ),
                configure=lambda _rows: self._configure_selected_rows(),
                remove=lambda _rows, bound=table: self._delete_selected_rows(bound),
                confirm_parent=self,
            )
            bind_reset_failed_context_menu(
                table,
                key_for_row=lambda row, bound=table: self._reset_key_at_row(row, table=bound),
                status_for_row=lambda row, bound=table: self._row_status(row, table=bound),
                emit_reset=self.reset_failed_requested.emit,
                leading_actions=leading_actions,
                additional_actions=(
                    *trailing_actions,
                    *self._extra_row_actions(),
                    *self.job_files_menu.row_actions(self),
                ),
            )
            # The count label and run/remove buttons are shared by every table;
            # whichever is on show drives them.
            self._selection_feedbacks.append(
                SelectionFeedback(
                    table,
                    count_label=self._selection_count_label,
                    run_local_button=self._run_local_button,
                    run_cluster_button=self._run_cluster_button,
                    selection_buttons=(self._remove_selected_button,),
                    is_active=lambda bound=table: self._active_table() is bound,
                )
            )
            selection_model = table.selectionModel()
            if selection_model is not None:
                selection_model.selectionChanged.connect(
                    lambda *_: self._refresh_batch_button_state()
                )

        if self._add_row_button is not None:
            self._add_row_button.clicked.connect(lambda: self._add_row())
        if self._name_prefix is not None:
            self._name_prefix.textChanged.connect(lambda *_: self._emit_state())
        self._batch_configure_button.clicked.connect(lambda: self._configure_selected_rows())
        self._remove_selected_button.clicked.connect(lambda: self._delete_selected_rows())
        self._run_local_button.clicked.connect(lambda: self._run_local_selected_rows())
        self._run_cluster_button.clicked.connect(lambda: self._run_cluster_selected_rows())

    # -- Project state pushed in -------------------------------------------------

    def set_host_lattice(self, lattice_vectors: list[list[float]] | None) -> None:
        self._supercells.set_host_lattice(lattice_vectors)
        self._normalize_all_row_parameter_values()
        self._refresh_all_rows()

    def set_runtime_states(self, states: Mapping[str, Any]) -> None:
        self._runtime_states = {str(key): state for key, state in states.items()}
        self._refresh_all_rows()

    def set_supercell_options(self, options: list[dict[str, object]]) -> None:
        self._supercells.set_options(options)
        self._refresh_all_rows()

    def set_parameter_species_options(self, species_options: Sequence[str]) -> None:
        self._parameter_species_options = tuple(
            dict.fromkeys(option for item in species_options if (option := str(item).strip()))
        )

    def row_supercell_matrix(self, row_id: int) -> SupercellMatrix | None:
        return self._supercells.matrices.get(int(row_id))

    def set_row_supercell(
        self,
        row_id: int,
        matrix: SupercellMatrix | None,
        *,
        message: str | None = None,
    ) -> None:
        self._supercells.matrices[int(row_id)] = matrix
        location = self._row_location_for_id(int(row_id))
        if location is not None:
            table, row = location
            self._refresh_row(row, table=table)
        self._supercells.update_kpoint_message(int(row_id), message)
        self._emit_state()

    def clear_kpoint_message(self) -> None:
        self._kpoint_message.hide()
        self._kpoint_message.setText("")

    # -- Row lookup ------------------------------------------------------------

    def _row_id_at(self, row: int, *, table: QTableWidget | None = None) -> int | None:
        item = (table or self._table).item(row, self._LABEL_COLUMN)
        if item is None:
            return None
        try:
            return int(item.data(Qt.ItemDataRole.UserRole))
        except (TypeError, ValueError):
            return None

    def _label_at_row(self, row: int, *, table: QTableWidget | None = None) -> str:
        item = (table or self._table).item(row, self._LABEL_COLUMN)
        return item.text().strip() if item is not None else ""

    def _row_location_for_id(self, row_id: int) -> tuple[QTableWidget, int] | None:
        for table in self._all_tables():
            for row in range(table.rowCount()):
                if self._row_id_at(row, table=table) == int(row_id):
                    return table, row
        return None

    def _row_location_for_key(self, key: str) -> tuple[QTableWidget, int] | None:
        for table in self._all_tables():
            for row in range(table.rowCount()):
                if (
                    self._label_at_row(row, table=table)
                    and self._submission_key_at_row(row, table=table) == key
                ):
                    return table, row
        return None

    def _row_entries(self) -> Iterator[RowEntry]:
        for table in self._all_tables():
            for row in range(table.rowCount()):
                row_id = self._row_id_at(row, table=table)
                if row_id is not None:
                    yield RowEntry(table, row, row_id, self._label_at_row(row, table=table))

    def _submission_keys_at_rows(self, rows: Sequence[int], *, table: QTableWidget) -> list[str]:
        return [
            self._submission_key_at_row(row, table=table)
            for row in rows
            if self._label_at_row(row, table=table)
        ]

    def _reset_key_at_row(self, row: int, *, table: QTableWidget | None = None) -> str | None:
        """Return the row's submission key, or ``None`` for an unlabeled row."""

        if not self._label_at_row(row, table=table):
            return None
        return self._submission_key_at_row(row, table=table)

    def _row_status(self, row: int, *, table: QTableWidget | None = None) -> str:
        return workflow_row_status_text(table or self._table, row, self._STATUS_COLUMN)

    def _new_row_label(self, number: int | None = None) -> str:
        return self._ROW_LABEL_TEMPLATE.format(
            number=self._table.rowCount() + 1 if number is None else int(number)
        )

    # -- Runtime state and locking ---------------------------------------------

    def _runtime_state_at(self, row: int, *, table: QTableWidget) -> Any:
        state = self._runtime_states.get(self._submission_key_at_row(row, table=table))
        if state is None:
            state = self._runtime_states.get(self._label_at_row(row, table=table))
        return state

    def _runtime_status(self, runtime: Any) -> str:
        return (runtime.status if runtime is not None else "") or self._EMPTY_STATUS

    def _row_status_for_id(self, row_id: int) -> str:
        location = self._row_location_for_id(row_id)
        if location is None:
            return self._EMPTY_STATUS
        table, row = location
        return self._runtime_status(self._runtime_state_at(row, table=table))

    def _row_is_editable(self, row_id: int) -> bool:
        """Return whether the row's inline cells (supercell, row controls) may change."""

        if self._row_location_for_id(row_id) is None or self._row_is_read_only(row_id):
            return False
        return not workflow_row_is_locked(self._row_status_for_id(row_id))

    def _row_parameters_locked(self, row_ids: tuple[int, ...]) -> bool:
        return any(
            self._row_is_read_only(row_id)
            or workflow_row_is_locked(self._row_status_for_id(row_id))
            for row_id in row_ids
        )

    # -- Rows ------------------------------------------------------------------

    def _allocate_row_id(self) -> int:
        row_id = self._next_row_id
        self._next_row_id += 1
        self._supercells.matrices[row_id] = None
        self._row_parameter_values[row_id] = {}
        self._row_added_advanced[row_id] = set()
        return row_id

    def _insert_row(self, row_id: int, label: str, *, table: QTableWidget | None = None) -> int:
        """Append the row for ``row_id`` to ``table`` and return its index."""

        target = table or self._table
        row = target.rowCount()
        previous = target.blockSignals(True)
        try:
            target.insertRow(row)
            label_item = QTableWidgetItem(label)
            label_item.setData(Qt.ItemDataRole.UserRole, row_id)
            target.setItem(row, self._LABEL_COLUMN, label_item)
            target.setCellWidget(
                row, self._SUPERCELL_COLUMN, self._supercells.create_combo(target, row_id)
            )
            set_workflow_table_item(target, row, self._STATUS_COLUMN, self._EMPTY_STATUS)
            self.create_row_cells(target, row, row_id)

            configure_button = SelectionSnapshotPushButton("Configure...", target)
            configure_button.selection_snapshot_key = row_id
            configure_button.setSizePolicy(
                QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed
            )
            configure_button.clicked.connect(
                lambda _checked=False, btn=configure_button: self._dispatch_row_action(
                    btn, row_id, "configure", target
                )
            )
            target.setCellWidget(row, self._CONFIGURE_COLUMN, configure_button)

            remove_button = SelectionSnapshotToolButton(target)
            remove_button.selection_snapshot_key = row_id
            remove_button.setText("x")
            remove_button.setAutoRaise(True)
            remove_button.clicked.connect(
                lambda _checked=False, btn=remove_button: self._dispatch_row_action(
                    btn, row_id, "remove", target
                )
            )
            target.setCellWidget(row, self._REMOVE_COLUMN, remove_button)

            self._refresh_row(row, table=target)
        finally:
            target.blockSignals(previous)
        self._refresh_batch_button_state()
        return row

    def _add_row(self) -> None:
        """Add a row with the next default label (the Add button)."""

        row_id = self._allocate_row_id()
        self._insert_row(row_id, self._new_row_label())
        self._emit_state()

    def _set_float_item(
        self,
        row: int,
        column: int,
        value: float | None,
        *,
        formatter: str = ".8g",
        table: QTableWidget | None = None,
    ) -> None:
        text = "" if value is None else format(float(value), formatter)
        set_workflow_table_item(table or self._table, row, column, text)

    def _set_status_item(self, row: int, status: str, *, table: QTableWidget | None = None) -> None:
        target = table or self._table
        set_workflow_table_item(target, row, self._STATUS_COLUMN, status)
        item = target.item(row, self._STATUS_COLUMN)
        if item is not None:
            item.setForeground(status_brush(status))

    def _refresh_row(self, row: int, *, table: QTableWidget | None = None) -> None:
        target = table or self._table
        row_id = self._row_id_at(row, table=target)
        if row_id is None:
            return
        supercell_combo = target.cellWidget(row, self._SUPERCELL_COLUMN)
        if isinstance(supercell_combo, QComboBox):
            self._supercells.populate_combo(supercell_combo, row_id)

        runtime = self._runtime_state_at(row, table=target)
        status = self._runtime_status(runtime)
        read_only = self._row_is_read_only(row_id)
        locked = read_only or workflow_row_is_locked(status)

        label_item = target.item(row, self._LABEL_COLUMN)
        if label_item is not None:
            flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            if not locked:
                flags |= Qt.ItemFlag.ItemIsEditable
                label_item.setToolTip("")
            elif read_only:
                label_item.setToolTip(self._READ_ONLY_ROW_TOOLTIP)
            else:
                label_item.setToolTip(_LOCKED_LABEL_TOOLTIP)
            label_item.setFlags(flags)
            # A read-only row is marked by style rather than by text: the label
            # text is read back as the row's identity.
            font = label_item.font()
            font.setItalic(read_only)
            label_item.setFont(font)
        if isinstance(supercell_combo, QComboBox):
            supercell_combo.setEnabled(not locked)
        remove_button = target.cellWidget(row, self._REMOVE_COLUMN)
        if isinstance(remove_button, QToolButton):
            remove_button.setEnabled(not read_only)
            remove_button.setToolTip(self._READ_ONLY_ROW_TOOLTIP if read_only else "")

        self._set_status_item(row, status, table=target)
        self.refresh_row_cells(target, row, row_id, runtime, status=status, locked=locked)

        configure_button = target.cellWidget(row, self._CONFIGURE_COLUMN)
        if isinstance(configure_button, QPushButton):
            style_workflow_configure_button(
                configure_button,
                workflow_label="CASTEP",
                configured=bool(self._row_parameter_values.get(row_id, {})),
                locked=locked,
                noun=self._CONFIGURE_NOUN,
                used=True,
            )

    def _refresh_all_rows(self) -> None:
        for table in self._all_tables():
            previous = table.blockSignals(True)
            for row in range(table.rowCount()):
                self._refresh_row(row, table=table)
            table.blockSignals(previous)

    def _on_table_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != self._LABEL_COLUMN:
            return
        self._refresh_all_rows()
        self._emit_state()

    # -- Selection -------------------------------------------------------------

    def _selected_row_ids(self) -> tuple[int, ...]:
        table = self._active_table()
        return tuple(
            row_id
            for row in selected_table_rows(table)
            if (row_id := self._row_id_at(row, table=table)) is not None
        )

    def _selected_keys(self) -> list[str]:
        table = self._active_table()
        return self._submission_keys_at_rows(selected_table_rows(table), table=table)

    def _selected_or_all_keys(self) -> list[str]:
        table = self._active_table()
        rows = selected_table_rows(table) or tuple(range(table.rowCount()))
        return self._submission_keys_at_rows(rows, table=table)

    def _refresh_batch_button_state(self) -> None:
        self._batch_configure_button.setEnabled(bool(self._selected_row_ids()))
        for feedback in self._selection_feedbacks:
            feedback.refresh()

    # -- Row actions -----------------------------------------------------------

    def _dispatch_row_action(
        self,
        widget: SelectionSnapshotPushButton | SelectionSnapshotToolButton,
        row_id: int,
        kind: str,
        table: QTableWidget,
    ) -> None:
        snapshot_keys = [
            key
            for row in widget.selection_snapshot_rows
            if (key := self._row_id_at(row, table=table)) is not None
        ]
        if kind == "configure":
            target_row_ids = resolve_target_keys(
                widget,
                clicked_key=row_id,
                snapshot_keys=snapshot_keys,
                eligibility=lambda _rid: True,
                parent=self,
                action="Configure parameters for",
            )
            if target_row_ids:
                self._open_parameter_dialog(target_row_ids)
        elif kind == "remove":
            target_row_ids = resolve_target_keys(
                widget,
                clicked_key=row_id,
                snapshot_keys=snapshot_keys,
                eligibility=lambda _rid: True,
                parent=self,
                action="Remove",
                confirm_unprompted=lambda: self._confirm_removal((row_id,)),
            )
            if target_row_ids:
                self._remove_rows_by_id(target_row_ids)

    def _configure_selected_rows(self) -> None:
        row_ids = self._selected_row_ids()
        if not row_ids:
            QMessageBox.information(self, "No rows selected", "Select one or more rows first.")
            return
        row_ids = self._configurable_row_ids(row_ids)
        if row_ids:
            self._open_parameter_dialog(row_ids)

    def _confirm_removal(self, row_ids: tuple[int, ...]) -> bool:
        submitted = sum(
            1 for row_id in row_ids if workflow_row_is_locked(self._row_status_for_id(row_id))
        )
        return confirm_row_removal(self, len(row_ids), submitted_count=submitted)

    def _delete_selected_rows(self, table: QTableWidget | None = None) -> None:
        target = table or self._active_table()
        if target.state() == QAbstractItemView.State.EditingState:
            return
        rows = selected_table_rows(target)
        if not rows and target.currentRow() >= 0:
            rows = (target.currentRow(),)
        row_ids = tuple(
            row_id for row in rows if (row_id := self._row_id_at(row, table=target)) is not None
        )
        if not row_ids or not self._confirm_removal(row_ids):
            return
        self._remove_rows_by_id(row_ids)

    def _remove_rows_by_id(self, row_ids: tuple[int, ...]) -> None:
        removed = False
        removed_keys: list[str] = []
        for row_id in self._removable_row_ids(tuple(row_ids)):
            location = self._row_location_for_id(row_id)
            if location is None:
                continue
            table, row = location
            if self._label_at_row(row, table=table):
                removed_keys.append(self._submission_key_at_row(row, table=table))
            table.removeRow(row)
            self._forget_row(row_id)
            removed = True
        if not removed:
            return
        if removed_keys:
            self.remove_requested.emit(removed_keys)
        self._refresh_batch_button_state()
        self._emit_state()

    # -- Parameters ------------------------------------------------------------

    def _pinned_params(self) -> dict[str, object]:
        return {"task": infer_task(self._WORKFLOW_STAGE), **self._PINNED_PARAMS}

    def _common_params_for_rows(self, row_ids: tuple[int, ...]) -> dict[str, object]:
        if not row_ids:
            return dict(self._default_params)
        common = shared_parameter_values(
            self._effective_params_for_row(row_id) for row_id in row_ids
        )
        common.update(self._pinned_params())
        if any(self._row_is_read_only(row_id) for row_id in row_ids):
            # No pre-fill for a read-only row: the suggested spacing comes from
            # the project reference, not from what the row's run used, and the
            # read-only dialog reports only what the run did use.
            return common
        # Configure-dialog seeding only: pre-fill the inactive half of the
        # kpoint_mp_grid / kpoint_mp_spacing pair so the editor for whichever
        # radio the user toggles to shows a value derived from the project
        # default. The dialog drops the inactive choice in collect_values(),
        # so this never reaches the rendered .cell or validation.
        has_grid = "kpoint_mp_grid" in common
        has_spacing = "kpoint_mp_spacing" in common
        if has_grid and not has_spacing:
            suggested_spacing = self._supercells.suggested_spacing()
            if suggested_spacing is not None:
                common["kpoint_mp_spacing"] = suggested_spacing
        elif has_spacing and not has_grid:
            suggested_grid = self._supercells.suggested_grid(row_ids[0])
            if suggested_grid is not None:
                common["kpoint_mp_grid"] = suggested_grid
        return common

    def _dialog_intro_text(self, row_ids: tuple[int, ...]) -> str:
        if self._DIALOG_INTRO is not None:
            return self._DIALOG_INTRO
        labels = [
            self._row_display_label(location[1], table=location[0])
            for row_id in row_ids
            if (location := self._row_location_for_id(row_id)) is not None
        ]
        return "Included rows: " + ", ".join(labels)

    def _open_parameter_dialog(self, row_ids: tuple[int, ...]) -> None:
        result = run_row_parameter_dialog(
            RowParameterRequest(
                specs=self._specs,
                workflow_stage=self._WORKFLOW_STAGE,
                subject=self._DIALOG_SUBJECT,
                values=self._common_params_for_rows(row_ids),
                row_values={row_id: self._effective_params_for_row(row_id) for row_id in row_ids},
                row_advanced={
                    row_id: self._row_added_advanced.get(row_id, set()) for row_id in row_ids
                },
                sanitize=lambda values: sanitize_parameter_mapping(values, self._specs_by_name),
                locked_values=self._dialog_locked_values(row_ids),
                read_only=self._row_parameters_locked(row_ids),
                intro_text=self._dialog_intro_text(row_ids),
                species_options=self._parameter_species_options,
            ),
            parent=self,
        )
        if result is None:
            return
        for row_id, values in result.row_values.items():
            self._row_parameter_values[row_id] = self._stored_row_values(row_id, values)
            self._row_added_advanced[row_id] = result.row_advanced[row_id]
        self._refresh_all_rows()
        self._emit_state()

    # -- Config and submission ---------------------------------------------------

    def _build_config_or_warn(self) -> object | None:
        if not any(table.rowCount() for table in self._all_tables()):
            QMessageBox.warning(self, "No jobs", self._NO_ROWS_MESSAGE)
            return None
        task = str(self._pinned_params()["task"])
        rows: list[RowEntry] = []
        seen: set[str] = set()
        for entry in self._row_entries():
            if not entry.label:
                QMessageBox.warning(self, "Invalid label", "Job labels must not be empty.")
                return None
            key = self._submission_key_at_row(entry.row, table=entry.table)
            if key in seen:
                name = self._row_display_label(entry.row, table=entry.table)
                QMessageBox.warning(self, "Duplicate label", f"Duplicate job label '{name}'.")
                return None
            seen.add(key)
            # A read-only row describes a run that already happened and is
            # never submitted, so validating it against today's schema could
            # only block the runnable rows.
            if not self._row_is_read_only(entry.row_id):
                values = self._effective_params_for_row(entry.row_id)
                try:
                    errors = validate_gui_values(values, self._specs_by_name, task=task)
                except ValueError as exc:
                    errors = [str(exc)]
                if errors:
                    QMessageBox.warning(
                        self,
                        "Invalid parameters",
                        f"Row '{entry.label}':\n" + "\n".join(errors),
                    )
                    return None
            rows.append(entry)
        return self.config_for_rows(tuple(rows))

    def _run_local_selected_rows(self) -> None:
        self._submit_selected_rows(self.submit_local_requested)

    def _run_cluster_selected_rows(self) -> None:
        self._submit_selected_rows(self.submit_cluster_requested)

    def _submit_selected_rows(self, signal: Any) -> None:
        config = self._build_config_or_warn()
        if config is None:
            return
        keys = self._selected_or_all_keys()
        if not keys:
            QMessageBox.information(self, "No jobs selected", "Select one or more rows first.")
            return
        self.configure_requested.emit(config)
        signal.emit(keys)
        self._emit_state()

    # -- Form state ------------------------------------------------------------

    def _name_prefix_text(self) -> str:
        if self._name_prefix is None:
            return ""
        return self._name_prefix.text().strip() or str(self._NAME_PREFIX_DEFAULT or "")

    def _set_name_prefix(self, text: str) -> None:
        if self._name_prefix is None:
            return
        previous = self._name_prefix.blockSignals(True)
        self._name_prefix.setText(text)
        self._name_prefix.blockSignals(previous)

    def _supercell_matrix_state(self, row_id: int) -> list[list[int]] | None:
        matrix = self._supercells.matrices.get(row_id)
        return [list(line) for line in matrix] if matrix is not None else None

    def _restore_row_fields(
        self,
        row_id: int,
        entry: Mapping[str, object],
        *,
        params_key: str,
        keep_unknown: bool = False,
    ) -> None:
        """Restore a persisted row's supercell, parameter values and advanced names."""

        self._supercells.matrices[row_id] = coerce_supercell_matrix(entry.get("supercell_matrix"))
        params = entry.get(params_key, {})
        if isinstance(params, dict):
            self._row_parameter_values[row_id] = sanitize_parameter_mapping(
                {str(name): value for name, value in params.items()},
                self._specs_by_name,
                keep_unknown=keep_unknown,
            )
        added = entry.get("added_advanced", [])
        if isinstance(added, list):
            self._row_added_advanced[row_id] = {name for name in added if isinstance(name, str)}

    def _clear_rows(self) -> None:
        for row_id in tuple(self._row_parameter_values):
            self._forget_row(row_id)
        for table in self._all_tables():
            table.setRowCount(0)
        self._supercells.matrices.clear()
        self._next_row_id = 1

    @contextmanager
    def _restoring_form(self) -> Iterator[None]:
        """Rebuild every row from persisted state without emitting state.

        Clears the rows first; afterwards re-normalises the stored values,
        refreshes the rows, and emits the restored state once.
        """

        self._restoring_form_state = True
        previous = [table.blockSignals(True) for table in self._all_tables()]
        try:
            self._clear_rows()
            yield
            self._normalize_all_row_parameter_values()
            self._refresh_all_rows()
            self._refresh_batch_button_state()
        finally:
            for table, blocked in zip(self._all_tables(), previous, strict=True):
                table.blockSignals(blocked)
            self._restoring_form_state = False
        self._emit_state()

    @contextmanager
    def _suspend_state_emission(self) -> Iterator[None]:
        previous = self._suppress_state_emission
        self._suppress_state_emission = True
        try:
            yield
        finally:
            self._suppress_state_emission = previous

    def _emit_state(self) -> None:
        if self._restoring_form_state or self._suppress_state_emission:
            return
        self.form_state_changed.emit(self.collect_state())


class RelaxationRowTableWidget(RowTableWidget):
    """Row table for a geometry-relaxation workflow.

    Adds the relaxation parameter surface (the convergence keywords plus the
    geometry keywords, without ``_EXCLUDED_PARAMETER_NAMES``, starting from
    ``_PARAMETER_DEFAULT_OVERRIDES``), project-default seeding, row values
    stored as the diff from the row's implicit defaults, and the "Continue
    relaxation…" row action.

    A row's effective parameters are the defaults, then
    :meth:`_row_default_overrides`, then the row's own values (which win over
    a mutually exclusive default), the pinned task, and the row's suggested
    k-point grid when it chose neither grid nor spacing. A read-only row shows
    its stored values only.
    """

    #: Emits the row's submission key when "Continue relaxation…" is chosen
    #: from the row context menu. See :meth:`set_continuable_predicate` for how
    #: the menu decides whether that action is enabled for a given row.
    continue_relaxation_requested = Signal(object)

    _EXCLUDED_PARAMETER_NAMES: frozenset[str] = frozenset({"task", "fix_all_cell"})
    # Starting values that differ from the schema default in this workflow.
    _PARAMETER_DEFAULT_OVERRIDES: Mapping[str, object] = {
        "geom_max_iter": 300,
        "continuation": "DEFAULT",
    }
    _TABLE_TOOLTIP = (
        "Right-click selected rows to run, cancel, inspect, reset, configure, or remove "
        "them, or a completed/failed/stopped/cancelled row to continue its relaxation."
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        # Injected by the controller wiring (see MainWindow's
        # ``set_continuable_predicate`` call): resolves a row's job through the
        # controller and checks its real status, never the row's displayed
        # status text (see the "Continue relaxation…" menu-gating rule in
        # ``mufinder.app.workflow_table_utils.RowMenuAction``). Defaults to
        # "never continuable" so an unwired widget (e.g. in a test) never
        # offers an action it cannot fulfil.
        self._continuable_predicate: Callable[[object], bool] = lambda _key: False
        # Injected like the predicate above, with the controller's
        # ``relaxed_structure_export``. Unwired, no row has a structure to export.
        self._relaxed_structure_export: Callable[[object], RelaxedStructureExport] = lambda _key: (
            RelaxedStructureExport(None, NOT_COMPLETED_REASON)
        )

    def _build_parameter_specs(self) -> list[KeywordSpec]:
        return geometry_relaxation_parameter_specs(
            workflow_stage=self._WORKFLOW_STAGE,
            excluded_names=self._EXCLUDED_PARAMETER_NAMES,
            default_overrides=self._PARAMETER_DEFAULT_OVERRIDES,
        )

    def _initial_default_params(self) -> dict[str, object]:
        # kpoint_mp_grid and kpoint_mp_spacing are project-default-aware:
        # _effective_params_for_row injects the row's suggested grid from the
        # adopted project grid, so the dialog reflects the project default
        # rather than the bare schema default.
        defaults = drop_default_conflicts(
            workflow_default_params(
                self._specs,
                workflow_stage=self._WORKFLOW_STAGE,
                locked_names={"task"},
                skip_names=("kpoint_mp_grid", "kpoint_mp_spacing"),
            ),
            self._specs_by_name,
            locked_names={"task"},
        )
        self._schema_xc_functional = defaults.get("xc_functional")
        return defaults

    def set_project_defaults(
        self,
        *,
        cutoff_energy: float | None,
        kpoint_grid: tuple[int, int, int] | None,
        kpoint_reference_lattice: list[list[float]] | None,
        xc_functional: str | None = None,
    ) -> None:
        """Seed row defaults from the project defaults (shown by the strip)."""

        seed_xc_functional_default(
            self._default_params,
            xc_functional,
            schema_default=self._schema_xc_functional,
        )
        self._supercells.set_kpoint_reference(kpoint_grid, kpoint_reference_lattice)
        if cutoff_energy is None:
            self._default_params.pop("cut_off_energy", None)
        else:
            self._default_params["cut_off_energy"] = ValueWithUnit(float(cutoff_energy), "eV")
        self._default_params = drop_default_conflicts(
            self._default_params, self._specs_by_name, locked_names={"task"}
        )
        self._normalize_all_row_parameter_values()
        self._refresh_all_rows()

    def set_continuable_predicate(self, predicate: Callable[[object], bool]) -> None:
        """Inject the live "can this row's job be continued?" check.

        Called by the controller wiring with the controller's
        ``row_is_continuable``. Evaluated fresh every time a row context menu
        opens, not cached, so a status change since the last refresh (e.g. a
        dashboard dismissal) is always reflected in whether "Continue
        relaxation…" is enabled.
        """

        self._continuable_predicate = predicate

    def set_relaxed_structure_export_provider(
        self, provider: Callable[[object], RelaxedStructureExport]
    ) -> None:
        """Inject how a row key resolves to its exportable relaxed structure.

        Asked fresh each time a row menu opens, like
        :meth:`set_continuable_predicate`, to enable "Export Relaxed Structure…"
        and to say why it is disabled.
        """

        self._relaxed_structure_export = provider

    def _export_tooltip(self, keys: Sequence[object]) -> str:
        if len(keys) != 1:
            return "Select one completed relaxation to export its relaxed structure."
        export = self._relaxed_structure_export(keys[0])
        if export.structure is None:
            return export.unavailable_reason or ""
        return "Save the relaxed supercell as CIF or as a CASTEP .cell file."

    def continue_relaxation_dialog_seed(self, key: object) -> dict[str, object] | None:
        """Return the ``ContinueRelaxationDialog`` seed kwargs for one row.

        ``key`` is the row's submission key. Mirrors the parameter dialog's
        single-row seeding (the row's effective parameter values, its
        added-advanced set and the same locked keywords). MainWindow supplies
        everything this widget cannot know (the checkpoint probe, the rendered
        old/new ``.param`` text, and the job's execution target/directory)
        alongside these kwargs. Returns ``None`` when ``key`` does not resolve
        to a row.
        """

        location = self._row_location_for_key(str(key))
        if location is None:
            return None
        table, row = location
        row_id = self._row_id_at(row, table=table)
        if row_id is None:
            return None
        return {
            "specs": self._specs,
            "workflow_stage": self._WORKFLOW_STAGE,
            "parameter_values": self._effective_params_for_row(row_id),
            "added_advanced": tuple(sorted(self._row_added_advanced.get(row_id, set()))),
            "locked_values": self._dialog_locked_values((row_id,)),
            "species_options": self._parameter_species_options,
        }

    def _extra_row_actions(self) -> tuple[RowMenuAction, ...]:
        return (
            RowMenuAction(
                text="Continue relaxation…",
                is_enabled=lambda keys: len(keys) == 1 and self._continuable_predicate(keys[0]),
                trigger=lambda keys: self.continue_relaxation_requested.emit(keys[0]),
                separator_before=True,
            ),
            RowMenuAction(
                text=EXPORT_RELAXED_STRUCTURE_ACTION_TEXT,
                is_enabled=lambda keys: (
                    len(keys) == 1 and self._relaxed_structure_export(keys[0]).structure is not None
                ),
                trigger=lambda _keys: None,
                tooltip_for=self._export_tooltip,
                submenu=tuple(
                    RowMenuAction(
                        text=fmt.menu_text,
                        is_enabled=lambda keys: len(keys) == 1,
                        trigger=lambda keys, chosen=fmt: export_relaxed_structure(
                            self, self._relaxed_structure_export(keys[0]), chosen
                        ),
                    )
                    for fmt in STRUCTURE_FORMATS
                ),
            ),
        )

    def _row_default_overrides(self, row_id: int) -> dict[str, object]:
        """Return defaults the adapter applies to this row unless it says otherwise.

        They are shown as the row's values and never stripped from what it
        stores.
        """

        del row_id
        return {}

    def _effective_params_for_row(self, row_id: int) -> dict[str, object]:
        if self._row_is_read_only(row_id):
            # A read-only row shows what its run used and nothing else.
            # Layering today's defaults underneath would present parameters the
            # run never used as though it had, and the suggested grid would
            # overwrite the grid recovered from the run's own output.
            values = sanitize_parameter_mapping(
                dict(self._row_parameter_values.get(row_id, {})),
                self._specs_by_name,
                keep_unknown=True,
            )
            values.update(self._pinned_params())
            return values

        # The effective values are what would land in the rendered .cell: only
        # ONE of kpoint_mp_grid / kpoint_mp_spacing, never both, since they are
        # mutually exclusive in CASTEP. The configure dialog renders both with
        # a radio toggle and drops the inactive choice in collect_values(), so
        # a row's stored values are single-choice too. With neither set the
        # row gets its suggested grid (matching the radio default).
        values = dict(self._default_params)
        values.update(self._row_default_overrides(row_id))
        row_values = self._row_parameter_values.get(row_id, {})
        values.update(row_values)
        drop_explicit_conflicts(values, self._specs_by_name, explicit_names=set(row_values))
        values.update(self._pinned_params())
        if "kpoint_mp_grid" not in values and "kpoint_mp_spacing" not in values:
            suggested = self._supercells.suggested_grid(row_id)
            if suggested is not None:
                values["kpoint_mp_grid"] = suggested
        return drop_default_conflicts(values, self._specs_by_name, locked_names={"task"})

    def _implicit_params_for_row(self, row_id: int) -> dict[str, object]:
        implicit = dict(self._default_params)
        for name in self._row_default_overrides(row_id):
            # Never strip an overridden default: an explicit value is the only
            # way to turn the adapter's default off, and one stored before the
            # default was shown must survive normalisation or the row's
            # signature stops matching its definition.
            implicit.pop(name, None)
        implicit.update(self._pinned_params())
        suggested = self._supercells.suggested_grid(row_id)
        if suggested is not None:
            implicit["kpoint_mp_grid"] = suggested
        return drop_default_conflicts(implicit, self._specs_by_name, locked_names={"task"})

    def _stored_row_values(self, row_id: int, values: Mapping[str, object]) -> dict[str, object]:
        """Return ``values`` as the diff from the row's implicit defaults.

        A read-only row records a run that already happened, so what it stores
        must be a pure function of the import. Stripping compares against the
        *current* project defaults and suggested grid, and it is re-run over
        every row whenever those change, while the row's definition is a
        frozen snapshot; the day a project default came to equal an imported
        value, stripping would remove it here but not there, the row signature
        would stop matching, and a completed import would render as "ready".
        Coerce, never strip.
        """

        if self._row_is_read_only(row_id):
            return sanitize_parameter_mapping(values, self._specs_by_name, keep_unknown=True)
        cleaned = sanitize_parameter_mapping(values, self._specs_by_name)
        implicit = self._implicit_params_for_row(row_id)
        return {
            name: value
            for name, value in cleaned.items()
            if not (name in implicit and value == implicit[name])
        }

    def _normalize_all_row_parameter_values(self) -> None:
        for row_id in tuple(self._row_parameter_values):
            self._row_parameter_values[row_id] = self._stored_row_values(
                row_id, self._row_parameter_values.get(row_id, {})
            )


class SourceTableWidget(QWidget):
    """Base for a workflow widget whose table lists sources keyed by a source key.

    Quantum corrections and energy barriers do not stage labelled rows: each
    row is a source read from the project (a relaxed site, a diffusion path;
    see ``mufinder.services.source_candidates``), shown in table order from
    ``_source_rows`` and submitted, configured and tracked under that row's
    ``key``. A source row is a frozen dataclass with at least ``key``,
    ``status`` and ``configured``.

    The base owns the key lookups (``selected_source_keys`` and the
    row-to-key helpers), the run and Configure targeting (the selection, else
    every row), the source table's row menu, its fill with the status colour
    and the per-row Configure button, and the action-state refresh.

    Class attributes a subclass sets:

    - ``_SOURCE_STATUS_COLUMN`` and ``_SOURCE_CONFIGURE_COLUMN``;
    - ``_SOURCE_WORKFLOW_LABEL``, the workflow named on the Configure button;
    - ``_CONFIGURE_SIGNAL``, ``_SUBMIT_LOCAL_SIGNAL`` and
      ``_SUBMIT_CLUSTER_SIGNAL``, the names of its own signals.

    The subclass also declares ``cancel_requested`` and
    ``inspect_failed_requested``, builds ``_sources_table``,
    ``_batch_configure_button`` and ``_sources_feedback`` in its ``__init__``
    (connecting the table's ``itemSelectionChanged`` to ``_update_actions``),
    calls :meth:`_bind_source_row_menu`, and implements the public hooks
    :meth:`source_row_values`, :meth:`create_source_row_cells` and
    :meth:`refresh_source_actions` for what only it has.
    """

    _SOURCE_STATUS_COLUMN: int
    _SOURCE_CONFIGURE_COLUMN: int
    _SOURCE_WORKFLOW_LABEL = ""
    _CONFIGURE_SIGNAL = ""
    _SUBMIT_LOCAL_SIGNAL = ""
    _SUBMIT_CLUSTER_SIGNAL = ""

    _sources_table: QTableWidget
    _batch_configure_button: QPushButton
    _sources_feedback: SelectionFeedback

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._source_rows: list[Any] = []
        #: Open Containing Folder / Export Inputs/Outputs… in the row menu.
        self.job_files_menu = JobFilesMenu()

    # -- hooks -------------------------------------------------------------

    def source_row_values(self, row: Any) -> Sequence[str]:
        """Return the text of ``row``'s leading columns, from column 0."""

        raise NotImplementedError

    def create_source_row_cells(self, row_index: int, row: Any) -> None:
        """Add the workflow's own cells to a filled row (default: none)."""

    def refresh_source_actions(self, selected_rows: Sequence[Any]) -> None:
        """Enable the workflow's own actions for the selected source rows."""

    # -- keys --------------------------------------------------------------

    def selected_source_keys(self) -> list[str]:
        """Return selected source keys in table order."""

        return [row.key for row in self._selected_source_rows()]

    def _selected_source_rows(self) -> list[Any]:
        indexes = sorted({index.row() for index in self._sources_table.selectedIndexes()})
        return [self._source_rows[row] for row in indexes if 0 <= row < len(self._source_rows)]

    def _selected_or_all_source_keys(self) -> list[str]:
        """Return the rows the run buttons act on: the selection, else every row."""

        return self.selected_source_keys() or [row.key for row in self._source_rows if row.key]

    def _source_key_at_row(self, row: int) -> str | None:
        if not (0 <= row < len(self._source_rows)):
            return None
        return self._source_rows[row].key or None

    def _source_keys_at_rows(self, rows: Sequence[int]) -> list[str]:
        return [key for row in rows if (key := self._source_key_at_row(row))]

    def _source_row_status(self, row: int) -> str:
        return workflow_row_status_text(self._sources_table, row, self._SOURCE_STATUS_COLUMN)

    # -- actions -----------------------------------------------------------

    def _emit_submit_local(self) -> None:
        getattr(self, self._SUBMIT_LOCAL_SIGNAL).emit(self._selected_or_all_source_keys())

    def _emit_submit_cluster(self) -> None:
        getattr(self, self._SUBMIT_CLUSTER_SIGNAL).emit(self._selected_or_all_source_keys())

    def _emit_batch_configure(self) -> None:
        keys = self.selected_source_keys()
        if keys:
            getattr(self, self._CONFIGURE_SIGNAL).emit(keys)

    def _update_actions(self) -> None:
        selected_rows = self._selected_source_rows()
        # The run buttons follow the selection through SelectionFeedback.
        self._batch_configure_button.setEnabled(bool(selected_rows))
        self.refresh_source_actions(selected_rows)

    def _bind_source_row_menu(
        self,
        *,
        remove: Callable[[Sequence[int]], None] | None = None,
        emit_reset: Callable[[object], None] | None = None,
        extra_actions: Sequence[RowMenuAction] = (),
    ) -> None:
        """Bind the source table's right-click menu.

        A right-click makes the rows under it the selection, so the run,
        Configure and Remove entries reuse the selection-based handlers.
        ``extra_actions`` follow the shared entries.
        """

        leading_actions, trailing_actions = standard_row_actions(
            status_for_row=self._source_row_status,
            run_local=lambda _rows: self._emit_submit_local(),
            run_cluster=lambda _rows: self._emit_submit_cluster(),
            cancel=lambda rows: self.cancel_requested.emit(self._source_keys_at_rows(rows)),
            inspect=lambda rows: self.inspect_failed_requested.emit(
                self._source_keys_at_rows(rows)
            ),
            configure=lambda _rows: self._emit_batch_configure(),
            remove=remove,
            confirm_parent=self,
        )
        bind_reset_failed_context_menu(
            self._sources_table,
            key_for_row=self._source_key_at_row,
            status_for_row=self._source_row_status,
            emit_reset=emit_reset,
            leading_actions=leading_actions,
            additional_actions=(
                *trailing_actions,
                *extra_actions,
                *self.job_files_menu.row_actions(self),
            ),
        )

    # -- table -------------------------------------------------------------

    def _set_source_rows(self, rows: Sequence[Any]) -> None:
        """Rebuild the source table from ``rows`` and refresh the actions."""

        self._source_rows = list(rows)
        self._fill_source_table()
        self._update_actions()
        # The rebuild replaced every row's cell widgets.
        self._sources_feedback.refresh()

    def _fill_source_table(self) -> None:
        table = self._sources_table
        rows = self._source_rows
        previous = table.blockSignals(True)
        try:
            table.setRowCount(len(rows))
            for row_index, row in enumerate(rows):
                for column, value in enumerate(self.source_row_values(row)):
                    set_workflow_table_item(table, row_index, column, value)
                self._create_source_configure_button(row_index, row)
                self.create_source_row_cells(row_index, row)
            self._reapply_source_status_colors()
            table.resizeRowsToContents()
        finally:
            table.blockSignals(previous)

    def _reapply_source_status_colors(self) -> None:
        """Colour each Status cell with the project-standard per-status palette."""

        for row_index, row in enumerate(self._source_rows):
            status_item = self._sources_table.item(row_index, self._SOURCE_STATUS_COLUMN)
            if status_item is not None:
                status_item.setForeground(status_brush(row.status))

    def _create_source_configure_button(self, row_index: int, row: Any) -> None:
        table = self._sources_table
        button = SelectionSnapshotPushButton(table)
        button.selection_snapshot_key = row.key
        style_workflow_configure_button(
            button,
            workflow_label=self._SOURCE_WORKFLOW_LABEL,
            configured=row.configured,
            locked=workflow_row_is_locked(row.status),
        )

        def _on_configure_clicked(_checked: bool = False, key: str = row.key) -> None:
            keys = resolve_target_keys(
                button,
                clicked_key=key,
                snapshot_keys=self._source_keys_at_rows(button.selection_snapshot_rows),
                eligibility=lambda _key: True,
                parent=table,
                action="Configure parameters for",
            )
            if keys:
                getattr(self, self._CONFIGURE_SIGNAL).emit(list(keys))

        button.clicked.connect(_on_configure_clicked)
        table.setCellWidget(row_index, self._SOURCE_CONFIGURE_COLUMN, button)
