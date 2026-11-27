"""Host-structure panel and file import controls."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtCore import QEvent, QItemSelectionModel, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.app.eliding_label import ElidingComboBox
from mufinder.app.fact_card import FactCard
from mufinder.app.help_bubble import help_button
from mufinder.app.wrapping_layout import WrappingRowLayout
from mufinder.app.structure_render_guard import (
    NO_STRUCTURE,
    structure_geometry_fingerprint,
    viewer_structure_fingerprint,
)
from mufinder.app.theme import (
    SPACING,
    form_spacing,
    scaled_to_text,
    style_hint_text,
    style_section_subtitle,
    style_section_title,
    style_subsection_title,
)
from mufinder.app.workflow_table_utils import (
    create_workflow_table,
    fill_with_last_column,
    set_header_tooltips,
    table_natural_width,
)
from mufinder.domain.magnetic_properties import propagation_vector_from_structure
from mufinder.domain.models import Structure
from mufinder.services.host_structure_facts import StructureFacts, structure_facts
from mufinder.services.project_service import PromotableHost
from mufinder.services.supercell_assets import structure_atomic_spins


#: The facts card with no host loaded.
NO_HOST_TEXT = "No host structure loaded."
#: The atom table's spin column.
SPIN_HEADER = "Spin (ħ/2)"
#: What the propagation vector does, behind its help icon.
PROPAGATION_HELP = (
    "The propagation vector k sets how spins repeat from cell to cell: in a supercell each"
    " atom's spin is multiplied by cos(2π k·R), with R the lattice translation of its cell."
    " Leave it at 0 0 0 when the loaded cell is already the magnetic cell."
)


class HostStructureWidget(QWidget):
    """Display current host structure and allow loading from disk."""

    load_requested = Signal()
    magnetic_configuration_changed = Signal(object)
    supercell_editor_requested = Signal()
    active_host_change_requested = Signal(str)  # emits structure_id as str

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._title = QLabel("Host Structure")
        style_section_title(self._title)

        # The facts card: contents, space group and source, then the cell.
        self._facts_card = FactCard(accessible_name="Host structure facts")
        self._facts_card.set_message(NO_HOST_TEXT)
        self._shown_fact_groups: tuple[object, ...] | None = None
        # The facts of the geometry last shown; the space-group search runs again
        # only when the geometry changes, not on a spin edit or a refresh.
        self._facts: StructureFacts | None = None
        self._facts_geometry: tuple[object, ...] = NO_STRUCTURE

        self._current_structure: Structure | None = None
        self._atomic_spins: list[float] = []
        self._propagation_vector: list[float] = [0.0, 0.0, 0.0]
        self._updating = False
        self._table_seed: tuple[object, int] | None = None
        # Flush-path guard: skip the full atom-table + 3D viewer rebuild when the
        # rendered structure (geometry + spin/propagation properties) is
        # unchanged. Content-based — see structure_render_guard for the contract.
        self._structure_fingerprint: tuple[object, ...] = NO_STRUCTURE
        self._structure_fingerprint_valid = False

        # This table keeps row numbers (vertical header) as a visual atom index
        # and allows editing the Spin column in place, so it opts out of the
        # shared no-edit/hidden-header defaults via show_vertical_header/editable.
        # It also opts out of stretch-to-fill: a stretched last column would
        # shrink and elide before the horizontal scrollbar appears. Instead the
        # last column takes the spare width of the viewport and never goes
        # below its contents (_stretch_last_column).
        self._atom_table = create_workflow_table(
            ("Element", "x", "y", "z", SPIN_HEADER),
            selection_mode=QAbstractItemView.SelectionMode.ExtendedSelection,
            editable=True,
            show_vertical_header=True,
            default_resize_mode=QHeaderView.ResizeMode.ResizeToContents,
            stretch_last_section=False,
            accessible_name="Atoms",
        )
        set_header_tooltips(
            self._atom_table,
            {
                "x": "Fractional coordinate along a.",
                "y": "Fractional coordinate along b.",
                "z": "Fractional coordinate along c.",
                SPIN_HEADER: "Spin of each atom in units of ħ/2. Type a number to change it.",
            },
        )
        self._atom_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._atom_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self._atom_table.cellChanged.connect(self._on_table_cell_changed)
        self._last_column_content_width = 0
        self._atom_table.viewport().installEventFilter(self)
        # The header resizes the content-sized columns after the rows change;
        # the last column then takes up the new spare width.
        self._atom_table.horizontalHeader().sectionResized.connect(self._on_section_resized)

        # The components are in reciprocal-lattice units, along a*, b* and c*,
        # each named by a faint caption inside its box.
        self._propagation_x = DecimalLineEdit(caption="a*")
        self._propagation_y = DecimalLineEdit(caption="b*")
        self._propagation_z = DecimalLineEdit(caption="c*")
        for axis, widget in zip(
            "abc", (self._propagation_x, self._propagation_y, self._propagation_z), strict=True
        ):
            widget.setAccessibleName(f"Propagation Vector {axis}*")
            widget.setToolTip(f"Component of the propagation vector along {axis}*.")
            widget.setDecimals(4)
            widget.setRange(-1.0, 1.0)
            widget.setSingleStep(0.01)
            # As wide as its caption and the longest value it takes, -0.0000; only
            # in a window too narrow for the row and the viewer does it narrow,
            # down to a value like 0.5.
            widget.set_width_for_values("0.5", "-0.0000")
            widget.valueChanged.connect(lambda *_: self._on_propagation_changed())

        self._viewer = AtomicStructureViewer()
        self._viewer.selected_atoms_changed.connect(self._on_viewer_atoms_selected)
        self._syncing_selection = False

        self._load_button = QPushButton("Load Structure…")
        self._load_button.clicked.connect(self.load_requested.emit)
        self._supercell_button = QPushButton("Supercell Editor…")
        self._supercell_button.setEnabled(False)
        self._supercell_button.clicked.connect(self._on_supercell_preview_requested)

        # As wide as its longest entry, and narrower, eliding, in a narrow window.
        self._active_host_combo = ElidingComboBox()
        self._active_host_combo.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToContents)
        self._active_host_combo.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self._active_host_combo.setMinimumWidth(scaled_to_text(160))
        self._active_host_combo.currentIndexChanged.connect(self._on_active_host_combo_changed)

        self._build_layout()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(self._title)
        subtitle = QLabel(
            "Load the crystal structure every later stage works on, check its cell and"
            " atoms, and give the atoms spins for magnetic work."
        )
        style_section_subtitle(subtitle)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        action_row = QHBoxLayout()
        action_row.addWidget(self._load_button)
        action_row.addWidget(self._supercell_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        host_selector_row = QHBoxLayout()
        host_selector_label = QLabel("Active host")
        host_selector_label.setBuddy(self._active_host_combo)
        host_selector_row.addWidget(host_selector_label)
        host_selector_row.addWidget(self._active_host_combo)
        host_selector_row.addStretch(1)
        layout.addLayout(host_selector_row)
        host_help = QLabel(
            "Every later stage uses this structure. Spins and saved supercells are kept per host."
        )
        style_hint_text(host_help)
        host_help.setWordWrap(True)
        layout.addWidget(host_help)

        # As wide as the page: the structure's facts, a hairline, then the cell's.
        layout.addWidget(self._facts_card)

        # The spins and the propagation vector sit together: one row on top of
        # the atom table, with the table's left and right edges.
        magnetic_order = QLabel("Magnetic order")
        style_subsection_title(magnetic_order)
        self._propagation_label = QLabel("Propagation vector")
        self._propagation_label.setBuddy(self._propagation_x)
        self._propagation_help = help_button(
            PROPAGATION_HELP, accessible_name="About the propagation vector"
        )
        self._propagation_controls = QWidget()
        self._propagation_controls.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        controls_layout = QHBoxLayout(self._propagation_controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(SPACING.xs)
        for edit in (self._propagation_x, self._propagation_y, self._propagation_z):
            controls_layout.addWidget(edit)
        # The icon ends the row at the table's right edge.
        controls_layout.addStretch(1)
        controls_layout.addWidget(self._propagation_help, 0, Qt.AlignmentFlag.AlignVCenter)
        # The pane opens wide enough for this row on one line (_seed_splitter).
        # Only a window too narrow for the pane and the viewer's minimum wraps
        # it, the boxes and the icon going under the label.
        self._propagation_row = WrappingRowLayout(spacing=form_spacing())
        self._propagation_row.setContentsMargins(0, 0, 0, 0)
        self._propagation_row.addWidget(self._propagation_label)
        self._propagation_row.addWidget(self._propagation_controls)

        self._table_pane = QWidget()
        table_pane_layout = QVBoxLayout(self._table_pane)
        table_pane_layout.setContentsMargins(0, 0, 0, 0)
        table_pane_layout.addWidget(magnetic_order)
        table_pane_layout.addLayout(self._propagation_row)
        table_pane_layout.addWidget(self._atom_table, 1)

        # The table keeps the width it was seeded with; the viewer takes the
        # rest of the window and every later change in the window's width.
        self._splitter = QSplitter(Qt.Orientation.Horizontal)
        self._splitter.addWidget(self._table_pane)
        self._splitter.addWidget(self._viewer)
        self._splitter.setChildrenCollapsible(False)
        self._splitter.setStretchFactor(0, 0)
        self._splitter.setStretchFactor(1, 1)
        self._viewer.setMinimumWidth(380)
        self._seed_splitter()

        layout.addWidget(self._splitter, 1)

    def _fit_atom_columns(self) -> None:
        """Make each column as wide as its widest value, whatever the row count.

        ``ResizeToContents`` measures only the first ``resizeContentsPrecision``
        rows, and again on every resize event. Above that many rows the columns
        are fixed instead, at the width of the value that measures widest.
        """
        table = self._atom_table
        header = table.horizontalHeader()
        row_count = table.rowCount()
        last = table.columnCount() - 1
        if row_count <= header.resizeContentsPrecision():
            header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
            # The last column is sized by hand so it can take the spare width.
            header.setSectionResizeMode(last, QHeaderView.ResizeMode.Fixed)
            # Measure the other columns now, not at the header's next layout.
            header.resizeSections(QHeaderView.ResizeMode.ResizeToContents)
            self._last_column_content_width = max(
                header.sectionSizeHint(last), table.sizeHintForColumn(last)
            )
            self._stretch_last_column()
            return
        header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        table.ensurePolished()
        metrics = table.fontMetrics()
        grid_width = 1 if table.showGrid() else 0
        for column in range(table.columnCount()):
            row_of_text = {table.item(row, column).text(): row for row in range(row_count)}
            widest = row_of_text[max(row_of_text, key=metrics.horizontalAdvance)]
            cell_width = table.sizeHintForIndex(table.model().index(widest, column)).width()
            table.setColumnWidth(
                column, max(header.sectionSizeHint(column), cell_width + grid_width)
            )
        self._last_column_content_width = table.columnWidth(last)
        self._stretch_last_column()

    def _stretch_last_column(self) -> None:
        """Let the last column fill the viewport's spare width (``fill_with_last_column``)."""
        fill_with_last_column(self._atom_table, self._last_column_content_width)

    def _on_section_resized(self, column: int, _old: int, _new: int) -> None:
        if column != self._atom_table.columnCount() - 1:
            self._stretch_last_column()

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API naming.
        if watched is self._atom_table.viewport() and event.type() == QEvent.Type.Resize:
            self._stretch_last_column()
        return super().eventFilter(watched, event)

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        # The columns and the row are measured right only once the theme has
        # styled the table and the row's controls.
        super().showEvent(event)
        self._remeasure()

    def changeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().changeEvent(event)
        if event.type() in (QEvent.Type.StyleChange, QEvent.Type.FontChange):
            self._remeasure()

    def _remeasure(self) -> None:
        """Fit the columns and seed the split again, after the text or style changed."""
        if self._atom_table.rowCount():
            self._fit_atom_columns()
        self._seed_splitter()

    def _propagation_row_width(self) -> int:
        """The width of the propagation-vector row on one line, its boxes at full width."""
        return (
            self._propagation_label.sizeHint().width()
            + self._propagation_row.spacing()
            + self._propagation_controls.sizeHint().width()
        )

    def _seed_splitter(self) -> None:
        """Open the table pane as wide as the table or the propagation row needs, the wider.

        The table fills a pane wider than its columns (``fill_with_last_column``).
        A refresh of the same structure at the same width, such as a spin edit,
        keeps a split the user dragged.
        """
        width = max(table_natural_width(self._atom_table), self._propagation_row_width())
        structure = self._current_structure
        seed = (structure.id if structure is not None else None, width)
        if seed != self._table_seed:
            self._table_seed = seed
            self._splitter.setSizes([width, self._viewer.minimumWidth()])
        self._stretch_last_column()

    @staticmethod
    def _render_fingerprint(structure: Structure | None) -> tuple[object, ...]:
        """Fingerprint the fields ``set_structure`` renders. Content-based.

        Combines what the viewer draws (``viewer_structure_fingerprint``: the
        geometry, spins, atom colours and periodicity) with a ``repr``
        serialization of the spin/propagation ``properties`` the atom table and
        propagation row read. The ``repr`` string is built from the property
        *values* at fingerprint time, so later in-place mutation of those
        lists/dicts (the normal edit pattern for ``properties``) still produces
        a differing fingerprint — no object identities are involved anywhere.
        ``source`` is not part of it: a new source name only updates the facts
        card (``set_structure``). The parts are the geometry, the viewer's
        properties and the page's properties, so ``set_structure`` can tell a
        change only the viewer draws from one the table shows.
        """
        viewer = viewer_structure_fingerprint(structure)
        if structure is None:
            return viewer
        properties = structure.properties
        property_signature = repr(
            (
                properties.get("atomic_spins"),
                properties.get("initial_magnetic_moments"),
                properties.get("propagation_vector"),
                properties.get("magnetic_structure"),
            )
        )
        return (*viewer, property_signature)

    def set_structure(
        self,
        structure: Structure | None,
        *,
        source: str | None = None,
        force: bool = False,
    ) -> None:
        """Render the facts card and one atom-table row per atom.

        ``source`` says where the structure came from at the end of the facts
        strip's first line (``ProjectService.host_structure_source``); without
        it the strip shows the structure's source file, or its label.

        No-op (beyond storing the reference) when the rendered structure is
        unchanged since the last call, avoiding the atom-table repopulation,
        the space-group search and the full 3D viewer mesh rebuild on every
        dashboard flush.
        """
        if structure is not None and not source:
            source = structure.source_file or structure.label
        source = source or ""
        fingerprint = self._render_fingerprint(structure)
        if (
            not force
            and self._structure_fingerprint_valid
            and fingerprint == self._structure_fingerprint
        ):
            self._current_structure = structure
            # A new source name, such as a relaxation renamed by another one
            # completing, changes the facts card only, not the table or viewer.
            self._show_facts(structure, source)
            return
        if (
            not force
            and self._structure_fingerprint_valid
            and structure is not None
            and self._current_structure is not None
            and fingerprint[0] == self._structure_fingerprint[0]
            and fingerprint[2:] == self._structure_fingerprint[2:]
        ):
            # Only what the viewer alone draws changed (atom colours, periodicity):
            # redraw it with the camera and selection kept; the table is unchanged.
            self._structure_fingerprint = fingerprint
            self._current_structure = structure
            with self._viewer.batch_update():
                self._viewer.set_structure_preserving_view(structure)
                # The spins are unchanged, so Spins stays as the user left it.
                self._viewer.set_spins(self._atomic_spins, keep_spins_choice=True)
            self._show_facts(structure, source)
            return
        self._structure_fingerprint = fingerprint
        self._structure_fingerprint_valid = True
        same_structure = (
            self._current_structure is not None
            and structure is not None
            and self._current_structure.id == structure.id
        )
        self._current_structure = structure
        self._atom_table.setRowCount(0)
        # One viewer render for the structure and its spins.
        with self._viewer.batch_update():
            self._viewer.set_structure(structure)
            self._updating = True
            self._show_facts(structure, source)
            if structure is None:
                self._atomic_spins = []
                self._propagation_vector = [0.0, 0.0, 0.0]
                self._set_propagation_inputs(
                    self._propagation_vector, force=force or not same_structure
                )
                self._viewer.set_spins(None)
                self._supercell_button.setEnabled(False)
                self._updating = False
                return

            self._supercell_button.setEnabled(True)

            self._atomic_spins = structure_atomic_spins(structure)
            self._propagation_vector = list(propagation_vector_from_structure(structure))
            self._set_propagation_inputs(
                self._propagation_vector, force=force or not same_structure
            )

            for index, (species, frac) in enumerate(
                zip(structure.species, structure.fractional_positions),
                start=1,
            ):
                row = self._atom_table.rowCount()
                self._atom_table.insertRow(row)
                self._atom_table.setVerticalHeaderItem(row, QTableWidgetItem(str(index)))
                self._set_readonly_item(row, 0, species)
                self._set_readonly_item(row, 1, f"{float(frac[0]):.4f}")
                self._set_readonly_item(row, 2, f"{float(frac[1]):.4f}")
                self._set_readonly_item(row, 3, f"{float(frac[2]):.4f}")
                self._set_spin_item(row, self._atomic_spins[row])

            self._fit_atom_columns()
            self._seed_splitter()
            self._viewer.set_spins(self._atomic_spins)
            self._updating = False

    def _show_facts(self, structure: Structure | None, source: str) -> None:
        """Fill the facts card, searching for the space group only when the geometry changed."""

        if structure is None:
            if self._facts is None and self._facts_card.message() == NO_HOST_TEXT:
                return
            self._facts = None
            self._facts_geometry = NO_STRUCTURE
            self._shown_fact_groups = None
            self._facts_card.set_message(NO_HOST_TEXT)
            return
        geometry = structure_geometry_fingerprint(structure)
        if self._facts is None or geometry != self._facts_geometry:
            self._facts = structure_facts(structure)
            self._facts_geometry = geometry
        groups = self._facts.groups(source)
        if groups != self._shown_fact_groups:
            self._shown_fact_groups = groups
            self._facts_card.set_groups(groups)

    def set_promotable_hosts(
        self,
        hosts: list[PromotableHost],
        active_id: object | None,
    ) -> None:
        """Repopulate the active-host dropdown without triggering spurious signals."""
        self._active_host_combo.blockSignals(True)
        try:
            self._active_host_combo.clear()
            active_str = str(active_id) if active_id is not None else None
            current_index = 0
            for i, host in enumerate(hosts):
                self._active_host_combo.addItem(host.label, userData=str(host.structure_id))
                self._active_host_combo.setItemData(
                    i, host.provenance_hint, Qt.ItemDataRole.ToolTipRole
                )
                if active_str and str(host.structure_id) == active_str:
                    current_index = i
            if hosts:
                self._active_host_combo.setCurrentIndex(current_index)
            # A narrow window elides the entry; the tooltip keeps it whole.
            self._active_host_combo.setToolTip(self._active_host_combo.currentText())
        finally:
            self._active_host_combo.blockSignals(False)

    def _on_active_host_combo_changed(self, index: int) -> None:
        self._active_host_combo.setToolTip(self._active_host_combo.currentText())
        structure_id_str = self._active_host_combo.itemData(index)
        if structure_id_str:
            self.active_host_change_requested.emit(structure_id_str)

    def _on_table_selection_changed(self, *args) -> None:
        if self._syncing_selection:
            return
        selection_model = self._atom_table.selectionModel()
        selected_rows = (
            sorted({index.row() for index in selection_model.selectedRows()})
            if selection_model is not None
            else []
        )
        if not selected_rows and args:
            current_row = int(args[0])
            if current_row >= 0:
                selected_rows = [current_row]
        self._viewer.select_atoms(selected_rows)

    def _on_table_cell_changed(self, row: int, column: int) -> None:
        if self._updating or column != 4:
            return
        if row < 0 or row >= len(self._atomic_spins):
            return
        item = self._atom_table.item(row, column)
        if item is None:
            return

        previous = float(self._atomic_spins[row])
        try:
            spin_value = float(item.text())
        except ValueError:
            self._updating = True
            try:
                self._set_spin_item(row, previous)
            finally:
                self._updating = False
            return

        self._atomic_spins[row] = spin_value
        self._updating = True
        try:
            self._set_spin_item(row, spin_value)
        finally:
            self._updating = False
        self._fit_atom_columns()
        self._seed_splitter()
        self._viewer.set_spins(self._atomic_spins)
        self._emit_magnetic_configuration()

    def _on_viewer_atoms_selected(self, atom_indexes: Sequence[int]) -> None:
        selection_model = self._atom_table.selectionModel()
        if selection_model is None:
            return
        self._syncing_selection = True
        try:
            self._atom_table.clearSelection()
            valid_rows = [
                int(row) for row in atom_indexes if 0 <= int(row) < self._atom_table.rowCount()
            ]
            if not valid_rows:
                self._atom_table.setCurrentCell(-1, -1)
                return
            for row in valid_rows:
                index = self._atom_table.model().index(row, 0)
                selection_model.select(
                    index,
                    QItemSelectionModel.SelectionFlag.Select
                    | QItemSelectionModel.SelectionFlag.Rows,
                )
            selection_model.setCurrentIndex(
                self._atom_table.model().index(valid_rows[-1], 0),
                QItemSelectionModel.SelectionFlag.NoUpdate,
            )
        finally:
            self._syncing_selection = False

    def _set_propagation_inputs(self, vector: list[float], *, force: bool) -> None:
        for edit, value in zip(
            (self._propagation_x, self._propagation_y, self._propagation_z), vector, strict=True
        ):
            edit.refresh_value(value, force=force)

    def _set_readonly_item(self, row: int, column: int, text: str) -> None:
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._atom_table.setItem(row, column, item)

    def _set_spin_item(self, row: int, spin: float) -> None:
        item = self._atom_table.item(row, 4)
        if item is None:
            item = QTableWidgetItem()
            self._atom_table.setItem(row, 4, item)
        item.setText(f"{float(spin):.4f}")

    def _on_propagation_changed(self) -> None:
        if self._updating:
            return
        self._propagation_vector = [
            float(self._propagation_x.value()),
            float(self._propagation_y.value()),
            float(self._propagation_z.value()),
        ]
        self._emit_magnetic_configuration()

    def _on_supercell_preview_requested(self) -> None:
        if self._current_structure is None:
            return
        self.supercell_editor_requested.emit()

    def _emit_magnetic_configuration(self) -> None:
        if self._current_structure is None:
            return
        self.magnetic_configuration_changed.emit(
            {
                "atomic_spins": [float(value) for value in self._atomic_spins],
                "propagation_vector": [float(value) for value in self._propagation_vector],
            }
        )
