"""Supercell preview dialog for magnetic structure inspection."""

from __future__ import annotations

from collections.abc import Sequence

from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
)

from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.domain.models import Structure
from mufinder.science.supercell import build_supercell_structure


class SupercellPreviewDialog(QDialog):
    """Display a generated supercell with propagated spin arrows."""

    def __init__(
        self,
        *,
        structure: Structure,
        atomic_spins: Sequence[float],
        propagation_vector: Sequence[float],
        initial_matrix: Sequence[Sequence[int]] | None = None,
        accept_button_text: str = "Use Matrix",
        show_unit_cell_button: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Supercell Preview")
        self.resize(980, 760)

        self._structure = structure
        self._atomic_spins = [float(value) for value in atomic_spins]
        self._propagation_vector = [float(value) for value in propagation_vector[:3]]
        while len(self._propagation_vector) < 3:
            self._propagation_vector.append(0.0)
        self._result_mode = "accepted"

        self._simple_nx = QSpinBox()
        self._simple_ny = QSpinBox()
        self._simple_nz = QSpinBox()
        for spinbox in (self._simple_nx, self._simple_ny, self._simple_nz):
            spinbox.setRange(1, 12)
            spinbox.setValue(1)

        self._advanced_toggle = QCheckBox("Advanced 3x3 matrix")
        self._advanced_toggle.setStyleSheet("QCheckBox { background: transparent; }")
        self._advanced_toggle.toggled.connect(self._on_advanced_toggled)

        self._advanced_cells: list[list[QSpinBox]] = []
        self._advanced_group = QGroupBox("Supercell Matrix")
        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._viewer = AtomicStructureViewer()

        self._generate_button = QPushButton("Generate Preview")
        self._generate_button.clicked.connect(self._generate_preview)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Close
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        ok_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        if ok_button is not None:
            ok_button.setText(accept_button_text)
        if show_unit_cell_button:
            unit_cell_button = buttons.addButton("Use Unit Cell", QDialogButtonBox.ButtonRole.ActionRole)
            unit_cell_button.clicked.connect(self._accept_unit_cell)

        self._build_advanced_matrix_grid()
        self._build_layout(buttons)

        matrix = initial_matrix or [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        self._apply_initial_matrix(matrix)
        self._on_advanced_toggled(self._advanced_toggle.isChecked())
        self._generate_preview()

    def matrix(self) -> list[list[int]]:
        """Return the currently configured supercell matrix."""

        if not self._advanced_toggle.isChecked():
            return [
                [int(self._simple_nx.value()), 0, 0],
                [0, int(self._simple_ny.value()), 0],
                [0, 0, int(self._simple_nz.value())],
            ]
        return [
            [int(cell.value()) for cell in row]
            for row in self._advanced_cells
        ]

    def result_mode(self) -> str:
        """Return which high-level action accepted the dialog."""

        return self._result_mode

    def accept(self) -> None:  # noqa: D401 - Qt API override.
        """Accept only when current matrix produces a valid supercell."""

        self._result_mode = "accepted"
        matrix = self.matrix()
        try:
            build_supercell_structure(
                self._structure,
                matrix,
                atomic_spins=self._atomic_spins,
                propagation_vector=self._propagation_vector,
            )
        except ValueError as exc:
            self._status_label.setText(f"Supercell generation failed: {exc}")
            return
        super().accept()

    def _build_layout(self, buttons: QDialogButtonBox) -> None:
        layout = QVBoxLayout(self)

        controls = QGroupBox("Supercell Controls")
        controls_layout = QVBoxLayout(controls)

        simple_row = QHBoxLayout()
        simple_row.addWidget(QLabel("Simple mode:"))
        simple_row.addWidget(QLabel("Nx"))
        simple_row.addWidget(self._simple_nx)
        simple_row.addWidget(QLabel("Ny"))
        simple_row.addWidget(self._simple_ny)
        simple_row.addWidget(QLabel("Nz"))
        simple_row.addWidget(self._simple_nz)
        simple_row.addStretch(1)
        controls_layout.addLayout(simple_row)

        controls_layout.addWidget(self._advanced_toggle)
        controls_layout.addWidget(self._advanced_group)

        action_row = QHBoxLayout()
        action_row.addWidget(self._generate_button)
        action_row.addStretch(1)
        controls_layout.addLayout(action_row)

        layout.addWidget(controls)
        layout.addWidget(self._status_label)
        layout.addWidget(self._viewer, stretch=1)
        layout.addWidget(buttons)

    def _build_advanced_matrix_grid(self) -> None:
        grid = QGridLayout(self._advanced_group)
        for row in range(3):
            matrix_row: list[QSpinBox] = []
            for column in range(3):
                cell = QSpinBox()
                cell.setRange(-12, 12)
                cell.setValue(1 if row == column else 0)
                matrix_row.append(cell)
                grid.addWidget(cell, row, column)
            self._advanced_cells.append(matrix_row)

    def _apply_initial_matrix(self, matrix: Sequence[Sequence[int]]) -> None:
        parsed = [[1, 0, 0], [0, 1, 0], [0, 0, 1]]
        if len(matrix) == 3 and all(len(row) == 3 for row in matrix):
            parsed = [[int(row[0]), int(row[1]), int(row[2])] for row in matrix]

        is_diagonal = (
            parsed[0][1] == 0
            and parsed[0][2] == 0
            and parsed[1][0] == 0
            and parsed[1][2] == 0
            and parsed[2][0] == 0
            and parsed[2][1] == 0
            and parsed[0][0] > 0
            and parsed[1][1] > 0
            and parsed[2][2] > 0
        )

        self._simple_nx.setValue(max(1, parsed[0][0]))
        self._simple_ny.setValue(max(1, parsed[1][1]))
        self._simple_nz.setValue(max(1, parsed[2][2]))

        for row in range(3):
            for column in range(3):
                self._advanced_cells[row][column].setValue(parsed[row][column])

        self._advanced_toggle.setChecked(not is_diagonal)

    def _on_advanced_toggled(self, enabled: bool) -> None:
        self._advanced_group.setVisible(enabled)

    def _accept_unit_cell(self) -> None:
        self._result_mode = "unit_cell"
        super().accept()

    def _generate_preview(self) -> None:
        matrix = self.matrix()
        try:
            expanded_structure, expanded_spins = build_supercell_structure(
                self._structure,
                matrix,
                atomic_spins=self._atomic_spins,
                propagation_vector=self._propagation_vector,
            )
        except ValueError as exc:
            self._status_label.setText(f"Supercell generation failed: {exc}")
            self._viewer.set_structure(None)
            return

        self._viewer.set_structure(expanded_structure)
        self._viewer.set_spins(expanded_spins)
        self._status_label.setText(
            "Matrix "
            f"[{matrix[0][0]} {matrix[0][1]} {matrix[0][2]}; "
            f"{matrix[1][0]} {matrix[1][1]} {matrix[1][2]}; "
            f"{matrix[2][0]} {matrix[2][1]} {matrix[2][2]}] "
            f"-> {len(expanded_structure.species)} atoms"
        )
