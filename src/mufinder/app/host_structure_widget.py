"""Host-structure panel and file import controls."""

from __future__ import annotations

from collections.abc import Sequence
from collections import Counter
from typing import Any

from PySide6.QtCore import QMargins, Qt, Signal
from PySide6.QtWidgets import (
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.app.supercell_dialog import SupercellPreviewDialog
from mufinder.app.theme import style_section_subtitle, style_section_title
from mufinder.domain.models import Structure


class HostStructureWidget(QWidget):
    """Display current host structure and allow loading from disk."""

    _TABLE_WIDTH_EXTRA = 28

    load_requested = Signal()
    magnetic_configuration_changed = Signal(object)

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

        self._file_label = QLabel("Source: -")
        self._atom_count_label = QLabel("Atoms: -")
        self._species_label = QLabel("Species: -")
        self._file_label.setWordWrap(True)
        self._species_label.setWordWrap(True)

        self._atom_table = QTableWidget(0, 5)
        self._atom_table.setHorizontalHeaderLabels(["Element", "fX", "fY", "fZ", "Spin (hbar/2)"])
        self._atom_table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self._atom_table.horizontalHeader().setStretchLastSection(True)
        self._atom_table.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Expanding)
        self._atom_table.currentCellChanged.connect(self._on_table_selection_changed)
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
        self._viewer.atom_selected.connect(self._on_viewer_atom_selected)
        self._syncing_selection = False

        self._load_button = QPushButton("Load Structure (.cif/.mcif/.cell)")
        self._load_button.clicked.connect(self.load_requested.emit)
        self._supercell_button = QPushButton("Supercell Preview...")
        self._supercell_button.setEnabled(False)
        self._supercell_button.clicked.connect(self._on_supercell_preview_requested)

        self._build_layout()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(self._title)
        subtitle = QLabel("Load your crystal structure and inspect atoms before running calculations.")
        style_section_subtitle(subtitle)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

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
        self._viewer.setMinimumWidth(560)

        layout.addWidget(info)
        layout.addWidget(splitter)

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
        self._atom_table.setMinimumWidth(target_width)
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

    def set_structure(self, structure: Structure | None) -> None:
        """Render structure metadata and first rows of atom coordinates."""
        self._current_structure = structure
        self._atom_table.setRowCount(0)
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

            if row >= 199:
                break

        self._lock_table_width()
        self._viewer.set_spins(self._atomic_spins)
        self._updating = False

    def _on_table_selection_changed(
        self,
        current_row: int,
        _current_column: int,
        _previous_row: int,
        _previous_column: int,
    ) -> None:
        if self._syncing_selection:
            return
        self._viewer.select_atom(current_row if current_row >= 0 else None)

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
        if atom_index < 0 or atom_index >= self._atom_table.rowCount():
            return
        self._syncing_selection = True
        try:
            self._atom_table.setCurrentCell(atom_index, 0)
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

        dialog = SupercellPreviewDialog(
            structure=self._current_structure,
            atomic_spins=self._atomic_spins,
            propagation_vector=self._propagation_vector,
            initial_matrix=self._supercell_matrix,
            parent=self,
        )
        if dialog.exec() == dialog.DialogCode.Accepted:
            self._supercell_matrix = dialog.matrix()
            self._emit_magnetic_configuration()

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
            for index, (species, spin) in enumerate(zip(self._current_structure.species, self._atomic_spins))
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
