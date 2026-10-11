"""Sweep-keyed row tables for the cutoff and k-point convergence widgets.

A convergence sweep stages one CASTEP singlepoint per sweep point. The table
row is identified by its sweep value, typed into the key column: a cutoff
energy for the cutoff sweep, a Monkhorst-Pack grid for the k-point sweep. The
row is submitted, tracked and removed under that key.

:class:`SweepConvergenceWidget` owns everything the two sweeps share: the
simulation-cell picker, the table and its row scaffold (key, status, total
energy, max force, Configure and remove cells), the selection, the row menu
and delete shortcuts, per-row Configure and remove targeting, removal with its
confirmation, the parameter dialog, runtime-state refresh and the Examine
button, config validation and submission, and the form-state round trip.

A subclass declares its key codec, wording and persisted field names as class
attributes and implements three public hooks: :meth:`next_key` (the key a new
row starts from), :meth:`prompt_new_key` (what the Add button asks for) and
:meth:`generate_keys` (the range or spacing generator). Behaviour only one
sweep has is an override of a private base method with a default.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
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
    QVBoxLayout,
    QWidget,
)

from mufinder.app.convergence_types import ConvergenceConfig, SweepKeyCodec, SweepPointRunState
from mufinder.app.project_defaults_widget import ProjectDefaultsStrip
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
    style_warning_text,
)
from mufinder.app.workflow_dialogs import RowParameterRequest, run_row_parameter_dialog
from mufinder.app.workflow_parameters import (
    STRUCTURE_CONTROLLED_CELL_KEYWORDS,
    reorder_workflow_parameter_specs,
    sanitize_parameter_mapping,
    seed_xc_functional_default,
    workflow_default_params,
)
from mufinder.app.workflow_table_utils import (
    REMOVE_SELECTED_TEXT,
    SelectionFeedback,
    bind_delete_shortcuts,
    bind_reset_failed_context_menu,
    confirm_row_removal,
    create_workflow_table,
    selected_table_rows,
    standard_row_actions,
    style_workflow_configure_button,
    workflow_row_is_locked,
    workflow_row_status_text,
)
from mufinder.castep_io.gui_metadata import (
    get_gui_keyword_specs_for_files,
    infer_task,
    keyword_locked_in_workflow,
    validate_gui_values,
)

_ACTIVE_JOB_STATUSES = {"submitting", "queued", "running", "retrieving"}
_PROGRESSED_JOB_STATUSES = _ACTIVE_JOB_STATUSES | {"stopped", "completed", "failed", "cancelled"}
_EMPTY_STATUS = "not generated"


class SweepConvergenceWidget(QWidget):
    """Base for a convergence widget that stages one job per sweep value.

    Class attributes a subclass sets:

    - ``_CODEC``, the :class:`~mufinder.app.convergence_types.SweepKeyCodec`
      that parses, formats and normalises the row key;
    - ``_WORKFLOW_STAGE`` and ``_SWEEP_PARAMETER`` (the keyword the sweep
      varies, locked in the parameter dialog);
    - the persisted field names ``_KEYS_STATE_FIELD`` (the key list),
      ``_ROWS_STATE_FIELD`` (the configured rows) and ``_ROW_KEY_FIELD`` (the
      key inside one configured row), ``_NAME_PREFIX_DEFAULT`` and
      ``_DEFAULT_RANGE_STATE`` (the generator dialog's remembered values: a
      ``replace_existing`` flag and floats);
    - ``_REMOVE_SIGNAL``, the name of the subclass's own removal signal;
    - the wording: ``_TITLE``, ``_SUBTITLE``, ``_SWEEP_GROUP_TITLE``,
      ``_KEY_COLUMN_LABEL``, the button texts, the Examine tooltips, the row
      nouns, the dialog subject and intro, and the warning messages.

    Behaviour one sweep alone has is a private override: the cutoff sweep
    cancels a removed row's running job (``_CANCEL_ACTIVE_ON_REMOVE``), titles a
    single-row dialog by its cutoff (``_dialog_subject``), repairs task-only
    rows (``_is_default_parameter_set``) and restores legacy shared/override
    state (``_restore_legacy_parameters``); the k-point sweep persists the cell
    its grids were generated for (``_extra_state`` / ``_restore_extra_state``).
    """

    configure_requested = Signal(object)
    submit_local_requested = Signal(object)
    submit_cluster_requested = Signal(object)
    cancel_requested = Signal(object)
    inspect_failed_requested = Signal(object)
    reset_failed_requested = Signal(object)
    examine_requested = Signal()
    form_state_changed = Signal(dict)
    simulation_cell_create_requested = Signal()
    simulation_cell_selection_requested = Signal(object)
    simulation_cell_use_unit_cell_requested = Signal()

    _KEY_COLUMN = 0
    _STATUS_COLUMN = 1
    _TOTAL_ENERGY_COLUMN = 2
    _MAX_FORCE_COLUMN = 3
    _CONFIGURE_COLUMN = 4
    _REMOVE_COLUMN = 5
    _ROW_HEIGHT = 32
    _KEY_COLUMN_WIDTH = 132
    _CONFIGURE_COLUMN_MIN_WIDTH = 136
    _REMOVE_COLUMN_WIDTH = 34

    _CODEC: SweepKeyCodec[Any]
    _WORKFLOW_STAGE = ""
    _SWEEP_PARAMETER = ""
    _KEYS_STATE_FIELD = ""
    _ROWS_STATE_FIELD = ""
    _ROW_KEY_FIELD = ""
    _NAME_PREFIX_DEFAULT = ""
    _DEFAULT_RANGE_STATE: Mapping[str, object] = {"replace_existing": False}
    _REMOVE_SIGNAL = ""

    _TITLE = ""
    _SUBTITLE = ""
    _SWEEP_GROUP_TITLE = ""
    _KEY_COLUMN_LABEL = ""
    _ADD_BUTTON_TEXT = ""
    _GENERATE_BUTTON_TEXT = ""
    _EXAMINE_BUTTON_TEXT = ""
    _EXAMINE_READY_TOOLTIP = ""
    _EXAMINE_WAITING_TOOLTIP = ""
    _LOCKED_KEY_TOOLTIP = ""
    _REMOVE_ROW_TOOLTIP = ""
    _CONFIGURE_NOUN = "row"
    _REMOVAL_NOUN: tuple[str, str] = ("row", "rows")
    _CANCEL_ACTIVE_ON_REMOVE = False
    _DIALOG_SUBJECT = "Parameters"
    _DIALOG_INTRO_PREFIX = "Included rows: "
    _INVALID_PARAMETERS_PREFIX = "Row"
    _NO_SELECTION_MESSAGE = ("No rows selected", "Select one or more rows first.")
    _INVALID_KEYS_MESSAGE = ("Invalid sweep", "Sweep values are not valid.")
    _NO_KEYS_MESSAGE = ("No rows", "Add one or more rows first.")
    _DUPLICATE_KEYS_MESSAGE = ("Duplicate rows", "Sweep values must be unique.")

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._add_button = QPushButton(self._ADD_BUTTON_TEXT)
        self._generate_button = QPushButton(self._GENERATE_BUTTON_TEXT)
        self._batch_configure_button = QPushButton("Configure Selected Parameters...")
        self._batch_configure_button.setEnabled(False)
        self._remove_selected_button = QPushButton(REMOVE_SELECTED_TEXT)
        self._selection_count_label = QLabel("")
        style_hint_text(self._selection_count_label)
        self._examine_button = QPushButton(self._EXAMINE_BUTTON_TEXT)
        self._examine_button.setEnabled(False)
        # SelectionFeedback sets the run buttons' text ("Run All ..." / "Run
        # Selected ...") from the selection.
        self._run_local_button = QPushButton()
        self._run_cluster_button = QPushButton()
        self.project_defaults_strip = ProjectDefaultsStrip()
        # A warning line under the header, hidden until a subclass has
        # something to say (the k-point sweep: grids generated for another cell).
        self._warning_label = QLabel("")
        self._warning_label.setWordWrap(True)
        style_warning_text(self._warning_label)
        self._warning_label.hide()

        self._range_state: dict[str, object] = dict(self._DEFAULT_RANGE_STATE)
        self._parameter_species_options: tuple[str, ...] = ()
        self._runtime_states: dict[Hashable, SweepPointRunState] = {}
        self._parameter_values: dict[Hashable, dict[str, Any]] = {}
        self._added_advanced: dict[Hashable, set[str]] = {}
        self._restoring_form_state = False
        self._updating_simulation_cell = False

        self._table = self._create_table()

        self._name_prefix = QLineEdit(self._NAME_PREFIX_DEFAULT)
        self._simulation_cell_summary = QLabel("Using single unit cell.")
        self._simulation_cell_summary.setWordWrap(True)
        self._simulation_cell_combo = QComboBox()
        self._simulation_cell_create_button = QPushButton("Manage Supercells...")
        self._simulation_cell_use_unit_button = QPushButton("Use Unit Cell")

        specs = get_gui_keyword_specs_for_files(
            files=("param", "cell"),
            workflow_stage=self._WORKFLOW_STAGE,
            include_advanced=True,
        )
        self._specs = reorder_workflow_parameter_specs(
            [spec for spec in specs if spec.name not in {"task", "spin_polarised"}]
        )
        self._specs_by_name = {spec.name: spec for spec in self._specs}
        self._default_params = workflow_default_params(
            self._specs,
            workflow_stage=self._WORKFLOW_STAGE,
            locked_names=set(self._shared_locked_values()),
        )
        self._schema_xc_functional = self._default_params.get("xc_functional")

        self._build_layout()
        self._connect_signals()
        self._delete_shortcut, self._backspace_shortcut = bind_delete_shortcuts(
            self, self._remove_selected_or_current
        )

    # -- Hooks -----------------------------------------------------------------

    def next_key(self, previous: Any | None) -> Any:
        """Return the key a new row starts from, given the last row's key."""

        raise NotImplementedError

    def prompt_new_key(self) -> Any | None:
        """Return the key the Add button adds, or ``None`` to add nothing.

        By default the next key after the last row, without asking.
        """

        return self.next_key(self._last_key())

    def generate_keys(self) -> tuple[tuple[Any, ...], bool] | None:
        """Run the generator dialog; return the keys and whether they replace the list.

        ``None`` adds nothing (the dialog was cancelled or generation failed).
        The dialog's values are remembered in ``_range_state`` either way.
        """

        raise NotImplementedError

    def _dialog_subject(self, keys: tuple[Any, ...]) -> str:
        del keys
        return self._DIALOG_SUBJECT

    def _is_default_parameter_set(self, values: Mapping[str, Any]) -> bool:
        """Return whether sanitised ``values`` mean "this row uses the defaults"."""

        return values == self._default_params

    def _restore_legacy_parameters(self, state: Mapping[str, Any], keys: tuple[Any, ...]) -> None:
        """Restore rows from a state that has no ``_ROWS_STATE_FIELD`` but
        carries ``shared_params`` / ``overrides``. Default: nothing to restore."""

        del state, keys

    def _extra_state(self) -> dict[str, Any]:
        """Return the subclass's own persisted form-state fields."""

        return {}

    def _restore_extra_state(self, state: Mapping[str, Any]) -> None:
        """Restore the subclass's own fields; ``state`` may be empty."""

        del state

    # -- Layout and wiring -------------------------------------------------------

    def _create_table(self) -> QTableWidget:
        table = create_workflow_table(
            (
                self._KEY_COLUMN_LABEL,
                "Status",
                "Total Energy (eV)",
                "Max |Force| (eV/Ang)",
                "Parameters",
                "",
            )
        )
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
        table.verticalHeader().setDefaultSectionSize(self._ROW_HEIGHT)
        header = table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(self._KEY_COLUMN, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(self._STATUS_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self._TOTAL_ENERGY_COLUMN, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self._MAX_FORCE_COLUMN, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(self._CONFIGURE_COLUMN, QHeaderView.ResizeMode.Interactive)
        header.setSectionResizeMode(self._REMOVE_COLUMN, QHeaderView.ResizeMode.Fixed)
        table.setColumnWidth(self._KEY_COLUMN, self._KEY_COLUMN_WIDTH)
        table.setColumnWidth(self._CONFIGURE_COLUMN, self._CONFIGURE_COLUMN_MIN_WIDTH)
        table.setColumnWidth(self._REMOVE_COLUMN, self._REMOVE_COLUMN_WIDTH)
        return table

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        header_row = QHBoxLayout()
        header_row.setContentsMargins(0, 0, 0, 0)
        header_row.setSpacing(10)
        title_column = QVBoxLayout()
        title_column.setContentsMargins(0, 0, 0, 0)
        title_column.setSpacing(3)
        title = QLabel(self._TITLE)
        style_section_title(title)
        title_column.addWidget(title)
        subtitle = QLabel(self._SUBTITLE)
        style_section_subtitle(subtitle)
        subtitle.setWordWrap(True)
        title_column.addWidget(subtitle)
        header_row.addLayout(title_column, 1)
        header_row.addWidget(self.project_defaults_strip, 0, Qt.AlignmentFlag.AlignTop)
        layout.addLayout(header_row)

        layout.addWidget(self._warning_label)

        simulation_group = QGroupBox("Simulation Cell")
        simulation_layout = QVBoxLayout(simulation_group)
        simulation_layout.setContentsMargins(10, 8, 10, 8)
        simulation_layout.setSpacing(6)
        simulation_layout.addWidget(self._simulation_cell_summary)
        simulation_row = QHBoxLayout()
        simulation_row.setSpacing(8)
        simulation_row.addWidget(QLabel("Saved supercells"))
        simulation_row.addWidget(self._simulation_cell_combo, 1)
        simulation_row.addWidget(self._simulation_cell_create_button)
        simulation_row.addWidget(self._simulation_cell_use_unit_button)
        simulation_layout.addLayout(simulation_row)
        layout.addWidget(simulation_group)

        sweep_group = QGroupBox(self._SWEEP_GROUP_TITLE)
        sweep_layout = QVBoxLayout(sweep_group)
        sweep_layout.setContentsMargins(10, 8, 10, 8)
        sweep_layout.setSpacing(6)
        sweep_layout.addWidget(self._table)

        controls_row = QHBoxLayout()
        controls_row.setSpacing(6)
        controls_row.addWidget(self._add_button)
        controls_row.addWidget(self._generate_button)
        controls_row.addStretch(1)
        controls_row.addWidget(self._selection_count_label)
        controls_row.addWidget(self._remove_selected_button)
        controls_row.addWidget(self._batch_configure_button)
        sweep_layout.addLayout(controls_row)

        examine_row = QHBoxLayout()
        examine_row.addStretch(1)
        examine_row.addWidget(self._examine_button)
        sweep_layout.addLayout(examine_row)
        layout.addWidget(sweep_group, 1)

        run_group = QGroupBox("Run Jobs")
        run_layout = QVBoxLayout(run_group)
        run_layout.setContentsMargins(10, 8, 10, 8)
        run_layout.setSpacing(6)
        run_row = QHBoxLayout()
        run_row.setSpacing(8)
        run_row.addWidget(QLabel("Job name prefix"))
        run_row.addWidget(self._name_prefix, 1)
        run_row.addWidget(self._run_local_button)
        run_row.addWidget(self._run_cluster_button)
        run_layout.addLayout(run_row)
        layout.addWidget(run_group)

    def _connect_signals(self) -> None:
        self._add_button.clicked.connect(self._on_add_clicked)
        self._generate_button.clicked.connect(self._open_generator_dialog)
        self._batch_configure_button.clicked.connect(self._configure_selected)

        self._table.itemChanged.connect(self._on_table_changed)
        self._remove_selected_button.clicked.connect(self._remove_selected_or_current)
        self._table.setToolTip(
            "Right-click selected rows to run, cancel, inspect, reset, configure, or remove them."
        )
        # A right-click makes the rows under it the selection, so the menu's
        # run/configure/remove entries reuse the selection-based handlers.
        leading_actions, trailing_actions = standard_row_actions(
            status_for_row=self._row_status,
            run_local=lambda _rows: self._run_local_selected(),
            run_cluster=lambda _rows: self._run_cluster_selected(),
            cancel=lambda rows: self.cancel_requested.emit(self._keys_at_rows(rows)),
            inspect=lambda rows: self.inspect_failed_requested.emit(self._keys_at_rows(rows)),
            configure=lambda _rows: self._configure_selected(),
            remove=lambda _rows: self._remove_selected_or_current(),
            confirm_parent=self,
        )
        bind_reset_failed_context_menu(
            self._table,
            key_for_row=lambda row: self._key_at_row(row, strict=False),
            status_for_row=self._row_status,
            emit_reset=self.reset_failed_requested.emit,
            leading_actions=leading_actions,
            additional_actions=trailing_actions,
        )
        selection_model = self._table.selectionModel()
        if selection_model is not None:
            selection_model.selectionChanged.connect(lambda *_: self._refresh_batch_button_state())

        self._name_prefix.textChanged.connect(self._emit_state)
        self._examine_button.clicked.connect(self.examine_requested.emit)
        self._run_local_button.clicked.connect(self._run_local_selected)
        self._run_cluster_button.clicked.connect(self._run_cluster_selected)
        self._selection_feedback = SelectionFeedback(
            self._table,
            count_label=self._selection_count_label,
            run_local_button=self._run_local_button,
            run_cluster_button=self._run_cluster_button,
            selection_buttons=(self._remove_selected_button,),
        )

        self._simulation_cell_combo.currentIndexChanged.connect(self._on_simulation_cell_combo_changed)
        self._simulation_cell_create_button.clicked.connect(self.simulation_cell_create_requested.emit)
        self._simulation_cell_use_unit_button.clicked.connect(
            self.simulation_cell_use_unit_cell_requested.emit
        )

    # -- Project state pushed in -------------------------------------------------

    def set_xc_functional_default(self, xc_functional: str | None) -> None:
        """Use the project default XC functional for rows without their own value."""

        seed_xc_functional_default(
            self._default_params,
            xc_functional,
            schema_default=self._schema_xc_functional,
        )

    def set_simulation_cell_options(
        self,
        *,
        summary: str,
        assets: list[dict[str, str]],
        selected_structure_id: str | None,
        enabled: bool,
    ) -> None:
        self._updating_simulation_cell = True
        try:
            self._simulation_cell_summary.setText(summary)
            self._simulation_cell_combo.clear()
            self._simulation_cell_combo.addItem("Single unit cell", None)
            for asset in assets:
                label = str(asset.get("label", "")).strip() or "Saved supercell"
                structure_id = str(asset.get("structure_id", "")).strip()
                self._simulation_cell_combo.addItem(label, structure_id)

            selected_index = 0
            if selected_structure_id:
                matched = self._simulation_cell_combo.findData(str(selected_structure_id))
                if matched >= 0:
                    selected_index = matched
            self._simulation_cell_combo.setCurrentIndex(selected_index)

            self._simulation_cell_combo.setEnabled(enabled)
            self._simulation_cell_create_button.setEnabled(enabled)
            self._simulation_cell_use_unit_button.setEnabled(enabled)
        finally:
            self._updating_simulation_cell = False

    def set_parameter_species_options(self, species_options: Sequence[str]) -> None:
        self._parameter_species_options = tuple(
            dict.fromkeys(option for item in species_options if (option := str(item).strip()))
        )

    def set_runtime_states(self, states: Mapping[Any, SweepPointRunState]) -> None:
        """Update the rows' status, energy and force from project job state."""

        self._runtime_states = {
            key: value
            for raw, value in states.items()
            if (key := self._CODEC.coerce(raw)) is not None
        }
        self._refresh_all_rows()
        self._refresh_examine_button_state()

    def _on_simulation_cell_combo_changed(self, index: int) -> None:
        if self._updating_simulation_cell:
            return
        data = self._simulation_cell_combo.itemData(index)
        if not data:
            self.simulation_cell_use_unit_cell_requested.emit()
            return
        self.simulation_cell_selection_requested.emit(str(data))

    def _shared_locked_values(self) -> dict[str, str]:
        locked: dict[str, str] = {}
        for spec in self._specs:
            if keyword_locked_in_workflow(spec, workflow_stage=self._WORKFLOW_STAGE):
                locked[spec.name] = "Variable (set by workflow)"
        if self._SWEEP_PARAMETER in {spec.name for spec in self._specs}:
            locked[self._SWEEP_PARAMETER] = "Variable (set by sweep)"
        return locked

    # -- Rows and keys -----------------------------------------------------------

    def _key_at_row(self, row: int, *, strict: bool) -> Any | None:
        item = self._table.item(row, self._KEY_COLUMN)
        if item is None:
            return None
        text = item.text().strip()
        if not text:
            return None
        try:
            return self._CODEC.parse(text)
        except ValueError:
            if strict:
                raise
            return None

    def _last_key(self) -> Any | None:
        if self._table.rowCount() == 0:
            return None
        return self._key_at_row(self._table.rowCount() - 1, strict=False)

    def _collect_keys(self, *, strict: bool = True) -> tuple[Any, ...]:
        values: list[Any] = []
        for row in range(self._table.rowCount()):
            value = self._key_at_row(row, strict=strict)
            if value is not None:
                values.append(value)
        return tuple(values)

    def _keys_at_rows(self, rows: Sequence[int]) -> list[Any]:
        return [key for row in rows if (key := self._key_at_row(row, strict=False)) is not None]

    def _rows_for_keys(self, keys: Sequence[Any]) -> tuple[int, ...]:
        return tuple(
            row for row in range(self._table.rowCount()) if self._key_at_row(row, strict=False) in keys
        )

    def _selected_rows(self) -> tuple[int, ...]:
        return selected_table_rows(self._table)

    def _selected_keys(self) -> tuple[Any, ...]:
        return tuple(self._keys_at_rows(self._selected_rows()))

    def _selected_or_all_keys(self) -> tuple[Any, ...]:
        """Return the keys the run buttons act on: the selection, else every row.

        The buttons read "Run Selected ..." or "Run All ..." to match, so what
        they run is always stated on the button.
        """

        return self._selected_keys() or self._collect_keys(strict=False)

    def _row_status(self, row: int) -> str:
        return workflow_row_status_text(self._table, row, self._STATUS_COLUMN)

    def _runtime_status_at_row(self, row: int) -> str:
        key = self._key_at_row(row, strict=False)
        runtime = self._runtime_states.get(key) if key is not None else None
        return (runtime.status if runtime is not None else "").strip().lower()

    def _make_item(self, text: str, *, editable: bool) -> QTableWidgetItem:
        item = QTableWidgetItem(text)
        flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
        if editable:
            flags |= Qt.ItemFlag.ItemIsEditable
        item.setFlags(flags)
        return item

    def _insert_row(self, key: Any) -> None:
        row = self._table.rowCount()
        self._table.insertRow(row)
        self._table.setItem(row, self._KEY_COLUMN, self._make_item(self._CODEC.format(key), editable=True))
        self._table.setItem(row, self._STATUS_COLUMN, self._make_item(_EMPTY_STATUS, editable=False))
        self._table.setItem(row, self._TOTAL_ENERGY_COLUMN, self._make_item("", editable=False))
        self._table.setItem(row, self._MAX_FORCE_COLUMN, self._make_item("", editable=False))

        configure_button = SelectionSnapshotPushButton(self._table)
        configure_button.selection_snapshot_key = key
        self._style_configure_button(configure_button)
        configure_button.clicked.connect(
            lambda _checked=False, clicked=key, btn=configure_button: self._configure_button_clicked(
                btn, clicked
            )
        )
        self._table.setCellWidget(row, self._CONFIGURE_COLUMN, configure_button)

        remove_button = SelectionSnapshotToolButton(self._table)
        remove_button.selection_snapshot_key = key
        remove_button.setText("x")
        remove_button.setAutoRaise(True)
        remove_button.setToolTip(self._REMOVE_ROW_TOOLTIP)
        remove_button.clicked.connect(
            lambda _checked=False, clicked=key, btn=remove_button: self._remove_button_clicked(btn, clicked)
        )
        self._table.setCellWidget(row, self._REMOVE_COLUMN, remove_button)
        self._refresh_row(row)

    def _style_configure_button(self, button: QPushButton) -> None:
        metrics = button.fontMetrics()
        label_width = max(
            metrics.horizontalAdvance("Configure..."),
            metrics.horizontalAdvance("Configured..."),
        )
        button.setSizePolicy(QSizePolicy.Policy.MinimumExpanding, QSizePolicy.Policy.Fixed)
        button.setMinimumWidth(max(self._CONFIGURE_COLUMN_MIN_WIDTH, label_width + 24))
        button.setMinimumHeight(max(24, metrics.height() + 10))

    def _set_keys(self, keys: tuple[Any, ...]) -> None:
        self._table.blockSignals(True)
        self._table.setRowCount(0)
        for key in keys:
            self._insert_row(key)
        self._table.blockSignals(False)
        self._prune_configs()
        self._refresh_all_rows()
        self._refresh_batch_button_state()

    def _add_row(self, key: Any | None = None) -> None:
        """Append a row for ``key``, or for :meth:`next_key` after the last row."""

        target = self.next_key(self._last_key()) if key is None else self._CODEC.normalize(key)
        self._table.blockSignals(True)
        self._insert_row(target)
        self._table.blockSignals(False)
        self._refresh_all_rows()
        self._emit_state()

    def _on_add_clicked(self) -> None:
        key = self.prompt_new_key()
        if key is not None:
            self._add_row(key)

    def _open_generator_dialog(self) -> None:
        generated = self.generate_keys()
        if generated is None:
            return
        keys, replace_existing = generated
        merged = keys if replace_existing else tuple(dict.fromkeys(self._collect_keys(strict=False) + keys))
        self._set_keys(merged)
        self._emit_state()

    def _prune_configs(self) -> None:
        available = set(self._collect_keys(strict=False))
        for key in [key for key in self._parameter_values if key not in available]:
            self._parameter_values.pop(key, None)
            self._added_advanced.pop(key, None)
        for key in [key for key in self._added_advanced if key not in available]:
            self._added_advanced.pop(key, None)

    def _on_table_changed(self, item: QTableWidgetItem) -> None:
        if item.column() != self._KEY_COLUMN:
            return
        self._prune_configs()
        self._refresh_all_rows()
        self._emit_state()

    # -- Refresh -----------------------------------------------------------------

    def _refresh_all_rows(self) -> None:
        self._table.blockSignals(True)
        for row in range(self._table.rowCount()):
            self._refresh_row(row)
        self._table.blockSignals(False)

    def _refresh_row(self, row: int) -> None:
        key = self._key_at_row(row, strict=False)
        runtime = self._runtime_states.get(key) if key is not None else None

        status_text = _EMPTY_STATUS
        energy_text = ""
        force_text = ""
        if runtime is not None:
            if runtime.status:
                status_text = runtime.status
            if runtime.total_energy is not None:
                energy_text = f"{runtime.total_energy:.12g}"
            if runtime.max_force is not None:
                force_text = f"{runtime.max_force:.12g}"
        locked = workflow_row_is_locked(status_text.strip().lower())

        key_item = self._table.item(row, self._KEY_COLUMN)
        if key_item is not None:
            flags = Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable
            if locked:
                key_item.setToolTip(self._LOCKED_KEY_TOOLTIP)
            else:
                flags |= Qt.ItemFlag.ItemIsEditable
                key_item.setToolTip("")
            key_item.setFlags(flags)

        for column, text in (
            (self._STATUS_COLUMN, status_text),
            (self._TOTAL_ENERGY_COLUMN, energy_text),
            (self._MAX_FORCE_COLUMN, force_text),
        ):
            item = self._table.item(row, column)
            if item is None:
                item = self._make_item(text, editable=False)
                self._table.setItem(row, column, item)
            else:
                item.setText(text)
            if column == self._STATUS_COLUMN:
                item.setForeground(status_brush(status_text))

        configure_button = self._table.cellWidget(row, self._CONFIGURE_COLUMN)
        if isinstance(configure_button, QPushButton):
            style_workflow_configure_button(
                configure_button,
                workflow_label="CASTEP",
                configured=key is not None and self._is_customized(key),
                locked=locked,
                noun=self._CONFIGURE_NOUN,
                used=True,
            )

    def _refresh_batch_button_state(self) -> None:
        self._batch_configure_button.setEnabled(bool(self._selected_keys()))

    def _refresh_examine_button_state(self) -> None:
        completed_with_results = sum(
            1
            for runtime in self._runtime_states.values()
            if runtime.status.strip().lower() == "completed"
            and runtime.total_energy is not None
            and runtime.max_force is not None
        )
        enabled = completed_with_results >= 2
        self._examine_button.setEnabled(enabled)
        self._examine_button.setToolTip(
            self._EXAMINE_READY_TOOLTIP if enabled else self._EXAMINE_WAITING_TOOLTIP
        )

    # -- Parameters --------------------------------------------------------------

    def _sanitize(self, values: Mapping[str, Any]) -> dict[str, Any]:
        return sanitize_parameter_mapping(
            dict(values),
            self._specs_by_name,
            keep=("task",),
            drop=STRUCTURE_CONTROLLED_CELL_KEYWORDS,
        )

    def _effective_params_for_key(self, key: Any) -> dict[str, Any]:
        values = self._parameter_values.get(key)
        return dict(self._default_params) if values is None else dict(values)

    def _set_parameter_values(self, key: Any, values: Mapping[str, Any], advanced_names: set[str]) -> None:
        normalized = self._sanitize(values)
        if self._is_default_parameter_set(normalized) and not advanced_names:
            self._parameter_values.pop(key, None)
            self._added_advanced.pop(key, None)
            return
        self._parameter_values[key] = dict(normalized)
        if advanced_names:
            self._added_advanced[key] = set(advanced_names)
        else:
            self._added_advanced.pop(key, None)

    def _common_params(self, maps: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
        """Return the values every map shares, with the workflow's task."""

        if not maps:
            return dict(self._default_params)
        common = dict(maps[0])
        for candidate in maps[1:]:
            common = {
                name: value for name, value in common.items() if name in candidate and candidate[name] == value
            }
        if "task" not in common:
            common["task"] = infer_task(self._WORKFLOW_STAGE)
        return common

    def _is_customized(self, key: Any) -> bool:
        return key in self._parameter_values or key in self._added_advanced

    def _parameters_locked(self, keys: tuple[Any, ...]) -> bool:
        return any(
            workflow_row_is_locked(
                "" if (runtime := self._runtime_states.get(key)) is None else runtime.status.strip().lower()
            )
            for key in keys
        )

    def _configure_button_clicked(self, button: SelectionSnapshotPushButton, clicked_key: Any) -> None:
        keys = resolve_target_keys(
            button,
            clicked_key=clicked_key,
            snapshot_keys=self._keys_at_rows(button.selection_snapshot_rows),
            eligibility=lambda _key: True,
            parent=self,
            action="Configure parameters for",
        )
        if keys:
            self._open_parameter_dialog(keys)

    def _configure_selected(self) -> None:
        keys = self._selected_keys()
        if not keys:
            QMessageBox.information(self, *self._NO_SELECTION_MESSAGE)
            return
        self._open_parameter_dialog(keys)

    def _open_parameter_dialog(self, keys: tuple[Any, ...]) -> None:
        if not keys:
            return
        result = run_row_parameter_dialog(
            RowParameterRequest(
                specs=self._specs,
                workflow_stage=self._WORKFLOW_STAGE,
                subject=self._dialog_subject(keys),
                values=self._common_params([self._effective_params_for_key(key) for key in keys]),
                row_values={key: self._effective_params_for_key(key) for key in keys},
                row_advanced={key: self._added_advanced.get(key, set()) for key in keys},
                sanitize=self._sanitize,
                locked_values=self._shared_locked_values(),
                read_only=self._parameters_locked(keys),
                intro_text=self._DIALOG_INTRO_PREFIX
                + ", ".join(self._CODEC.describe(key) for key in sorted(keys)),
                species_options=self._parameter_species_options,
            ),
            parent=self,
        )
        if result is None:
            return
        for key, values in result.row_values.items():
            self._set_parameter_values(key, values, result.row_advanced[key])
        self._refresh_all_rows()
        self._emit_state()

    # -- Removal -----------------------------------------------------------------

    def _active_keys(self, rows: Sequence[int]) -> list[Any]:
        if not self._CANCEL_ACTIVE_ON_REMOVE:
            return []
        return [
            key
            for row in rows
            if (key := self._key_at_row(row, strict=False)) is not None
            and self._runtime_status_at_row(row) in _ACTIVE_JOB_STATUSES
        ]

    def _confirm_removal(self, rows: Sequence[int]) -> bool:
        submitted = sum(1 for row in rows if self._runtime_status_at_row(row) in _PROGRESSED_JOB_STATUSES)
        return confirm_row_removal(
            self,
            len(rows),
            noun=self._REMOVAL_NOUN,
            submitted_count=submitted,
            cancelling_count=len(self._active_keys(rows)),
        )

    def _remove_button_clicked(self, button: SelectionSnapshotToolButton, clicked_key: Any) -> None:
        asked_single: list[bool] = []
        keys = resolve_target_keys(
            button,
            clicked_key=clicked_key,
            snapshot_keys=self._keys_at_rows(button.selection_snapshot_rows),
            eligibility=lambda _key: True,
            parent=self,
            action="Remove",
            confirm_unprompted=lambda: asked_single.append(True)
            or self._confirm_removal(self._rows_for_keys((clicked_key,))),
        )
        rows = self._rows_for_keys(keys)
        if rows:
            # The three-way prompt is the confirmation for a multi-row removal,
            # but it says nothing about running jobs, so it is not consent to
            # cancel them: those rows are confirmed again with that spelled out.
            self._request_removals(rows, confirmed=bool(asked_single) or not self._active_keys(rows))

    def _request_removals(self, rows: tuple[int, ...], *, confirmed: bool = False) -> None:
        """Remove ``rows`` after one confirmation covering the whole batch.

        With ``_CANCEL_ACTIVE_ON_REMOVE``, rows whose job is still queued or
        running have it cancelled first; the confirmation says how many, so
        nothing is cancelled unannounced. A row whose key does not parse is
        dropped here; the others are emitted on the removal signal, and the
        window removes them through :meth:`remove_keys`. ``confirmed`` skips
        the prompt for a caller that has already asked.
        """

        if not rows:
            return
        if not confirmed and not self._confirm_removal(rows):
            return

        active = self._active_keys(rows)
        if active:
            self.cancel_requested.emit(active)

        keys = self._keys_at_rows(rows)
        invalid_rows = [row for row in rows if self._key_at_row(row, strict=False) is None]
        for row in sorted(invalid_rows, reverse=True):
            self._table.removeRow(row)
        if invalid_rows:
            self._after_rows_removed()
        if keys:
            getattr(self, self._REMOVE_SIGNAL).emit(keys)

    def _remove_selected_or_current(self) -> None:
        if self._table.state() == QAbstractItemView.State.EditingState:
            return
        selected_rows = self._selected_rows()
        if selected_rows:
            self._request_removals(selected_rows)
            return
        current_row = self._table.currentRow()
        if current_row >= 0:
            self._request_removals((current_row,))

    def remove_keys(self, keys: Sequence[Any]) -> None:
        """Remove the rows for ``keys`` and forget their parameters."""

        targets = {key for raw in keys if (key := self._CODEC.coerce(raw)) is not None}
        if not targets:
            return
        removed = False
        for row in reversed(range(self._table.rowCount())):
            key = self._key_at_row(row, strict=False)
            if key is None or key not in targets:
                continue
            self._parameter_values.pop(key, None)
            self._added_advanced.pop(key, None)
            self._table.removeRow(row)
            removed = True
        if removed:
            self._after_rows_removed()

    def _after_rows_removed(self) -> None:
        self._prune_configs()
        self._refresh_batch_button_state()
        self._emit_state()

    # -- Config and submission ---------------------------------------------------

    def build_config(self) -> ConvergenceConfig | None:
        try:
            keys = self._collect_keys()
        except ValueError:
            QMessageBox.warning(self, *self._INVALID_KEYS_MESSAGE)
            return None
        if not keys:
            QMessageBox.warning(self, *self._NO_KEYS_MESSAGE)
            return None
        if len(set(keys)) != len(keys):
            QMessageBox.warning(self, *self._DUPLICATE_KEYS_MESSAGE)
            return None

        effective = {key: self._effective_params_for_key(key) for key in keys}
        for key, values in effective.items():
            task = str(values.get("task", infer_task(self._WORKFLOW_STAGE)))
            errors = validate_gui_values(values, self._specs_by_name, task=task)
            if errors:
                QMessageBox.warning(
                    self,
                    "Invalid parameters",
                    f"{self._INVALID_PARAMETERS_PREFIX} {self._CODEC.describe(key)}:\n" + "\n".join(errors),
                )
                return None

        shared_params = self._common_params(list(effective.values()))
        return ConvergenceConfig(
            sweep_parameter=self._SWEEP_PARAMETER,
            sweep_values=keys,
            shared_params=shared_params,
            overrides=self._overrides(keys, effective, shared_params),
            name_prefix=self._name_prefix_text(),
        )

    @staticmethod
    def _overrides(
        keys: tuple[Any, ...],
        effective: Mapping[Any, Mapping[str, Any]],
        shared: Mapping[str, Any],
    ) -> dict[frozenset[Any], dict[str, object]]:
        """Group the keys whose values differ from ``shared`` in the same way."""

        grouped: list[tuple[set[Any], dict[str, object]]] = []
        for key in keys:
            diff = {name: value for name, value in effective[key].items() if shared.get(name) != value}
            if not diff:
                continue
            for members, payload in grouped:
                if payload == diff:
                    members.add(key)
                    break
            else:
                grouped.append(({key}, diff))
        return {frozenset(members): payload for members, payload in grouped}

    def _emit_config(self) -> None:
        config = self.build_config()
        if config is None:
            return
        self.configure_requested.emit(config)
        self._emit_state()

    def _run_local_selected(self) -> None:
        self._submit_selected(self.submit_local_requested)

    def _run_cluster_selected(self) -> None:
        self._submit_selected(self.submit_cluster_requested)

    def _submit_selected(self, signal: Any) -> None:
        config = self.build_config()
        if config is None:
            return
        selected = tuple(dict.fromkeys(self._selected_or_all_keys()))
        if not selected:
            QMessageBox.information(self, *self._NO_SELECTION_MESSAGE)
            return
        self.configure_requested.emit(config)
        signal.emit(list(selected))
        self._emit_state()

    # -- Form state --------------------------------------------------------------

    def _name_prefix_text(self) -> str:
        return self._name_prefix.text().strip() or self._NAME_PREFIX_DEFAULT

    def _set_name_prefix(self, text: str) -> None:
        previous = self._name_prefix.blockSignals(True)
        self._name_prefix.setText(text)
        self._name_prefix.blockSignals(previous)

    def _emit_state(self) -> None:
        if self._restoring_form_state:
            return
        self.form_state_changed.emit(self.collect_state())

    def collect_state(self) -> dict[str, Any]:
        to_state = self._CODEC.to_state
        return {
            "mode": "manual",
            "name_prefix": self._name_prefix_text(),
            "range": dict(self._range_state),
            self._KEYS_STATE_FIELD: [to_state(key) for key in self._collect_keys(strict=False)],
            self._ROWS_STATE_FIELD: [
                {
                    self._ROW_KEY_FIELD: to_state(key),
                    "params": dict(self._parameter_values[key]),
                    "added_advanced": sorted(self._added_advanced.get(key, set())),
                }
                for key in sorted(self._parameter_values)
            ],
            **self._extra_state(),
        }

    def _restore_range_state(self, raw: object) -> None:
        self._range_state = dict(self._DEFAULT_RANGE_STATE)
        if not isinstance(raw, dict):
            return
        for name, default in self._DEFAULT_RANGE_STATE.items():
            if name == "replace_existing":
                self._range_state[name] = bool(raw.get(name, default))
                continue
            try:
                self._range_state[name] = float(raw.get(name, default))
            except (TypeError, ValueError):
                continue

    def set_form_state(self, state: dict[str, Any]) -> None:
        self._restoring_form_state = True
        try:
            state = state or {}
            self._parameter_values = {}
            self._added_advanced = {}
            self._set_name_prefix(str(state.get("name_prefix", self._NAME_PREFIX_DEFAULT)))
            self._restore_range_state(state.get("range"))

            raw_keys = state.get(self._KEYS_STATE_FIELD, [])
            raw_rows = state.get(self._ROWS_STATE_FIELD, [])
            rows = [entry for entry in raw_rows if isinstance(entry, dict)] if isinstance(raw_rows, list) else []
            listed = [self._CODEC.coerce(raw) for raw in raw_keys] if isinstance(raw_keys, list) else []
            configured = [self._CODEC.coerce(entry.get(self._ROW_KEY_FIELD)) for entry in rows]
            keys = tuple(dict.fromkeys(key for key in (*listed, *configured) if key is not None))
            self._set_keys(keys)

            if rows:
                for entry, key in zip(rows, configured, strict=True):
                    if key is None:
                        continue
                    params = self._sanitize(
                        {str(name): raw for name, raw in dict(entry.get("params", {})).items()}
                    )
                    advanced_names = {
                        str(name) for name in entry.get("added_advanced", []) if isinstance(name, str)
                    }
                    self._set_parameter_values(key, params, advanced_names)
            elif "shared_params" in state or "overrides" in state:
                self._restore_legacy_parameters(state, keys)

            self._restore_extra_state(state)
            self._refresh_all_rows()
            self._refresh_batch_button_state()
        finally:
            self._restoring_form_state = False

        self._emit_state()


__all__ = ["SweepConvergenceWidget"]
