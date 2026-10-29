"""Host-structure panel and file import controls."""

from __future__ import annotations

from collections.abc import Sequence
from collections import Counter
from typing import Any

from PySide6.QtCore import QItemSelectionModel, QMargins, Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QGridLayout,
    QGroupBox,
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
from mufinder.app.structure_render_guard import (
    NO_STRUCTURE,
    structure_geometry_fingerprint,
)
from mufinder.app.theme import style_section_subtitle, style_section_title
from mufinder.app.workflow_table_utils import create_workflow_table
from mufinder.domain.models import Structure
from mufinder.services.project_service import PromotableHost


class HostStructureWidget(QWidget):
    """Display current host structure and allow loading from disk."""

    _TABLE_WIDTH_EXTRA = 28

    load_requested = Signal()
    magnetic_configuration_changed = Signal(object)
    supercell_editor_requested = Signal()
    active_host_change_requested = Signal(str)  # emits structure_id as str

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._title = QLabel("Host Structure")
        style_section_title(self._title)

        self._summary = QLabel("No host structure loaded.")
        self._summary.setWordWrap(True)

        self._current_structure: Structure | None = None
        self._atomic_spins: list[float] = []
        self._propagation_vector: list[float] = [0.0, 0.0, 0.0]
        self._supercell_matrix: list[list[int]] = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self._updating = False
        # Flush-path guard: skip the full atom-table + 3D viewer rebuild when the
        # rendered structure (geometry + spin/propagation properties) is
        # unchanged. Content-based — see structure_render_guard for the contract.
        self._structure_fingerprint: tuple[object, ...] = NO_STRUCTURE
        self._structure_fingerprint_valid = False

        self._file_label = QLabel("Source: -")
        self._atom_count_label = QLabel("Atoms: -")
        self._species_label = QLabel("Species: -")
        self._file_label.setWordWrap(True)
        self._species_label.setWordWrap(True)

        # This table keeps row numbers (vertical header) as a visual atom index
        # and allows editing the Spin column in place, so it opts out of the
        # shared no-edit/hidden-header defaults via show_vertical_header/editable.
        # It also opts out of stretch-to-fill: _lock_table_width relies on
        # resizeColumnsToContents(), which Stretch-mode columns ignore.
        self._atom_table = create_workflow_table(
            ("Element", "fX", "fY", "fZ", "Spin (hbar/2)"),
            selection_mode=QAbstractItemView.SelectionMode.ExtendedSelection,
            editable=True,
            show_vertical_header=True,
            default_resize_mode=QHeaderView.ResizeMode.Interactive,
        )
        self._atom_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self._atom_table.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Expanding)
        self._atom_table.itemSelectionChanged.connect(self._on_table_selection_changed)
        self._atom_table.cellChanged.connect(self._on_table_cell_changed)

        self._propagation_x = DecimalLineEdit()
        self._propagation_y = DecimalLineEdit()
        self._propagation_z = DecimalLineEdit()
        for widget in (self._propagation_x, self._propagation_y, self._propagation_z):
            widget.setDecimals(4)
            widget.setRange(-1.0, 1.0)
            widget.setSingleStep(0.01)
            widget.valueChanged.connect(lambda *_: self._on_propagation_changed())

        self._viewer = AtomicStructureViewer()
        self._viewer.selected_atoms_changed.connect(self._on_viewer_atoms_selected)
        self._syncing_selection = False

        self._load_button = QPushButton("Load Structure (.cif/.mcif/.cell)")
        self._load_button.clicked.connect(self.load_requested.emit)
        self._supercell_button = QPushButton("Supercell Editor...")
        self._supercell_button.setEnabled(False)
        self._supercell_button.clicked.connect(self._on_supercell_preview_requested)

        self._active_host_combo = QComboBox()
        self._active_host_combo.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed
        )
        self._active_host_combo.currentIndexChanged.connect(self._on_active_host_combo_changed)

        self._build_layout()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(self._title)
        subtitle = QLabel(
            "Load your crystal structure and inspect atoms before running calculations."
        )
        style_section_subtitle(subtitle)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        host_selector_row = QHBoxLayout()
        host_selector_label = QLabel("Active host structure:")
        host_selector_row.addWidget(host_selector_label)
        host_selector_row.addWidget(self._active_host_combo, 1)
        layout.addLayout(host_selector_row)

        action_row = QHBoxLayout()
        action_row.addWidget(self._load_button)
        action_row.addWidget(self._supercell_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        info = QGroupBox("Current Structure")
        info_layout = QGridLayout(info)
        info_layout.addWidget(self._summary, 0, 0)
        info_layout.addWidget(self._atom_count_label, 0, 1)
        info_layout.addWidget(self._file_label, 1, 0)
        info_layout.addWidget(self._species_label, 1, 1)
        info_layout.setColumnStretch(0, 2)
        info_layout.setColumnStretch(1, 1)

        top_panel = QWidget()
        top_layout = QVBoxLayout(top_panel)
        top_layout.addWidget(self._atom_table)

        propagation_row = QHBoxLayout()
        self._propagation_label = QLabel("Propagation vector (r.l.u.):")
        propagation_row.addWidget(self._propagation_label)
        propagation_row.addWidget(self._propagation_x)
        propagation_row.addWidget(self._propagation_y)
        propagation_row.addWidget(self._propagation_z)
        propagation_row.addStretch(1)
        self._propagation_row = propagation_row
        top_layout.addLayout(propagation_row)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(top_panel)
        splitter.addWidget(self._viewer)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 8)
        splitter.setSizes([320, 960])
        self._viewer.setMinimumWidth(380)

        layout.addWidget(info)
        layout.addWidget(splitter, 1)

    def _lock_table_width(self) -> None:
        header = self._atom_table.horizontalHeader()
        stretch_last = header.stretchLastSection()
        header.setStretchLastSection(False)
        self._atom_table.resizeColumnsToContents()
        total = self._atom_table.verticalHeader().width()
        for column in range(self._atom_table.columnCount()):
            width = self._atom_table.columnWidth(column)
            padded_width = width + 14
            self._atom_table.setColumnWidth(column, padded_width)
            total += padded_width
        total += (2 * self._atom_table.frameWidth()) + 12
        target_width = max(total, self._propagation_controls_width() + self._TABLE_WIDTH_EXTRA)
        # Cap the table at its content width so it never greedily expands, but
        # do not also lock the minimum to the same value: that would make the
        # splitter pane rigid and prevent the table from shrinking (with its
        # own horizontal scrollbar) on small screens.
        self._atom_table.setMaximumWidth(target_width)
        header.setStretchLastSection(stretch_last)

    def _propagation_controls_width(self) -> int:
        label_width = self._propagation_label.sizeHint().width()
        spinbox_widths = (
            self._propagation_x.sizeHint().width()
            + self._propagation_y.sizeHint().width()
            + self._propagation_z.sizeHint().width()
        )
        spacing = max(0, self._propagation_row.spacing())
        margins: QMargins = self._propagation_row.contentsMargins()
        control_count = 4  # Label + three spinboxes
        return (
            margins.left()
            + margins.right()
            + label_width
            + spinbox_widths
            + (spacing * (control_count - 1))
        )

    @staticmethod
    def _render_fingerprint(structure: Structure | None) -> tuple[object, ...]:
        """Fingerprint the fields ``set_structure`` renders. Content-based.

        Combines the content-based geometry fingerprint from
        ``structure_render_guard`` with a ``repr`` serialization of the
        spin/propagation ``properties`` this widget reads. The ``repr`` string is
        built from the property *values* at fingerprint time, so later in-place
        mutation of those lists/dicts (the normal edit pattern for
        ``properties``) still produces a differing fingerprint — no object
        identities are involved anywhere. ``source_file`` and ``label`` feed the
        info labels and are included so a rename still refreshes.
        """
        geometry = structure_geometry_fingerprint(structure)
        if structure is None:
            return geometry
        properties = structure.properties
        property_signature = repr(
            (
                properties.get("atomic_spins"),
                properties.get("initial_magnetic_moments"),
                properties.get("propagation_vector"),
                properties.get("magnetic_structure"),
            )
        )
        return (
            *geometry,
            structure.label,
            structure.source_file,
            property_signature,
        )

    def set_structure(self, structure: Structure | None) -> None:
        """Render structure metadata and one atom-table row per atom.

        No-op (beyond storing the reference) when the rendered structure is
        unchanged since the last call, avoiding the atom-table repopulation and
        full 3D viewer mesh rebuild on every dashboard flush.
        """
        fingerprint = self._render_fingerprint(structure)
        if self._structure_fingerprint_valid and fingerprint == self._structure_fingerprint:
            self._current_structure = structure
            return
        self._structure_fingerprint = fingerprint
        self._structure_fingerprint_valid = True
        self._current_structure = structure
        self._atom_table.setRowCount(0)
        # One viewer render for the structure and its spins.
        with self._viewer.batch_update():
            self._viewer.set_structure(structure)
            self._updating = True
            if structure is None:
                self._summary.setText("No host structure loaded.")
                self._file_label.setText("Source: -")
                self._atom_count_label.setText("Atoms: -")
                self._species_label.setText("Species: -")
                self._atomic_spins = []
                self._propagation_vector = [0.0, 0.0, 0.0]
                self._set_propagation_inputs(self._propagation_vector)
                self._viewer.set_spins(None)
                self._supercell_button.setEnabled(False)
                self._lock_table_width()
                self._updating = False
                return

            self._supercell_button.setEnabled(True)

            self._summary.setText(f"Loaded structure: {structure.label}")
            self._file_label.setText(f"Source: {structure.source_file or 'external/unspecified'}")
            self._atom_count_label.setText(f"Atoms: {len(structure.species)}")

            counts = Counter(structure.species)
            species_summary = ", ".join(f"{k}:{v}" for k, v in sorted(counts.items()))
            self._species_label.setText(f"Species: {species_summary}")

            self._atomic_spins = self._load_spins(structure)
            self._propagation_vector = self._load_propagation_vector(structure)
            self._set_propagation_inputs(self._propagation_vector)

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

            self._lock_table_width()
            self._viewer.set_spins(self._atomic_spins)
            self._updating = False

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
        finally:
            self._active_host_combo.blockSignals(False)

    def _on_active_host_combo_changed(self, index: int) -> None:
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
        self._viewer.set_spins(self._atomic_spins)
        self._emit_magnetic_configuration()

    def _on_viewer_atom_selected(self, atom_index: int) -> None:
        self._on_viewer_atoms_selected([] if atom_index < 0 else [atom_index])

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

    def _load_spins(self, structure: Structure) -> list[float]:
        atom_count = len(structure.species)
        raw = structure.properties.get("atomic_spins")
        if raw is None:
            raw = structure.properties.get("initial_magnetic_moments")

        spins = [0.0] * atom_count
        saw_numeric = False
        if isinstance(raw, list):
            for index, value in enumerate(raw[:atom_count]):
                try:
                    spins[index] = float(value)
                    saw_numeric = True
                except (TypeError, ValueError):
                    spins[index] = 0.0

        magnetic = structure.properties.get("magnetic_structure")
        if isinstance(magnetic, dict):
            moments = magnetic.get("moments")
            if isinstance(moments, list):
                fallback = [0.0] * atom_count
                saw_fallback = False
                for index, moment in enumerate(moments[:atom_count]):
                    if not isinstance(moment, dict):
                        continue
                    vector = moment.get("vector")
                    if not isinstance(vector, list) or len(vector) < 3:
                        continue
                    try:
                        fallback[index] = float(vector[2])
                        saw_fallback = True
                    except (TypeError, ValueError):
                        continue
                if saw_fallback:
                    if saw_numeric and self._has_nonzero_spin(spins):
                        return spins
                    return fallback
        if saw_numeric:
            return spins
        return spins

    @staticmethod
    def _has_nonzero_spin(values: Sequence[float], *, tolerance: float = 1e-12) -> bool:
        return any(abs(float(value)) > tolerance for value in values)

    def _load_propagation_vector(self, structure: Structure) -> list[float]:
        source: Any = structure.properties.get("propagation_vector")
        if source is None:
            magnetic = structure.properties.get("magnetic_structure")
            if isinstance(magnetic, dict):
                source = magnetic.get("propagation_vector")

        vector = [0.0, 0.0, 0.0]
        if isinstance(source, list) and len(source) == 3:
            for index, value in enumerate(source):
                try:
                    vector[index] = float(value)
                except (TypeError, ValueError):
                    vector[index] = 0.0
        return vector

    def _set_propagation_inputs(self, vector: list[float]) -> None:
        self._propagation_x.setValue(float(vector[0]))
        self._propagation_y.setValue(float(vector[1]))
        self._propagation_z.setValue(float(vector[2]))

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

    def set_project_supercell_matrix(self, matrix: list[list[int]] | None) -> None:
        if not isinstance(matrix, list) or len(matrix) != 3:
            self._supercell_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
            return
        parsed: list[list[int]] = []
        for row in matrix:
            if not isinstance(row, list) or len(row) != 3:
                self._supercell_matrix = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
                return
            parsed.append([int(row[0]), int(row[1]), int(row[2])])
        self._supercell_matrix = parsed

    def _emit_magnetic_configuration(self) -> None:
        if self._current_structure is None:
            return

        moments = [
            {
                "label": f"{species}{index + 1}",
                "vector": [0.0, 0.0, float(spin)],
                "coordinate_system": "cartesian",
            }
            for index, (species, spin) in enumerate(
                zip(self._current_structure.species, self._atomic_spins)
            )
            if abs(float(spin)) > 1e-12
        ]
        propagation = [float(value) for value in self._propagation_vector]

        magnetic_structure: dict[str, Any] | None = None
        if moments or any(abs(value) > 1e-12 for value in propagation):
            magnetic_structure = {
                "propagation_vector": propagation,
                "moments": moments,
            }

        self.magnetic_configuration_changed.emit(
            {
                "atomic_spins": [float(value) for value in self._atomic_spins],
                "propagation_vector": propagation,
                "magnetic_structure": magnetic_structure,
                "supercell_matrix": [list(row) for row in self._supercell_matrix],
            }
        )
