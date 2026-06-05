"""Supercell preview dialog for magnetic structure inspection."""

from __future__ import annotations

from collections.abc import Sequence
from uuid import UUID

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
    QDialogButtonBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.supercell_suggestion_dialog import SupercellSuggestionDialog
from mufinder.app.theme import style_section_subtitle, style_section_title
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import Structure
from mufinder.science.supercell import build_supercell_structure
from mufinder.science.supercell_suggestion import SupercellSuggestionError, suggest_supercell
from mufinder.services.project_service import ProjectDocument, ProjectService


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
            spinbox.setRange(1, 64)
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
        self._viewer_shutdown = False

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

    def _shutdown_viewer(self) -> None:
        if self._viewer_shutdown:
            return
        self._viewer_shutdown = True
        try:
            self._viewer.shutdown()
        except RuntimeError:
            pass

    def done(self, result: int) -> None:  # noqa: D401 - Qt API override.
        """Finish the dialog after releasing viewer resources."""

        self._shutdown_viewer()
        super().done(result)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        self._shutdown_viewer()
        super().closeEvent(event)

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
                cell.setRange(-64, 64)
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


class SupercellEditorDialog(QDialog):
    """Manage saved project supercells for one source structure."""

    _ID_COLUMN = 0
    _NAME_COLUMN = 1
    _MATRIX_COLUMN = 2
    _ATOMS_COLUMN = 3
    _USAGE_COLUMN = 4

    def __init__(
        self,
        *,
        document: ProjectDocument,
        project_service: ProjectService,
        source_structure: Structure,
        workflow_stage: WorkflowStage = WorkflowStage.CONVERGENCE_CUTOFF,
        initial_matrix: Sequence[Sequence[int]] | None = None,
        select_mode: bool = False,
        parent=None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Supercell Editor")
        self.resize(1180, 760)

        self._document = document
        self._project_service = project_service
        self._source_structure = source_structure
        self._workflow_stage = workflow_stage
        self._select_mode = bool(select_mode)
        self._selected_structure_id: UUID | None = None
        self._current_asset_id: UUID | None = None
        self._updating_table = False

        self._title = QLabel("Supercell Editor")
        style_section_title(self._title)
        self._subtitle = QLabel("Generate, preview, name, and manage reusable supercells for this project.")
        style_section_subtitle(self._subtitle)
        self._subtitle.setWordWrap(True)

        self._asset_table = QTableWidget(0, 5)
        self._asset_table.setHorizontalHeaderLabels(["Id", "Name", "Matrix", "Atoms", "Usage"])
        self._asset_table.setColumnHidden(self._ID_COLUMN, True)
        self._asset_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._asset_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._asset_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._asset_table.itemSelectionChanged.connect(self._on_asset_selection_changed)
        self._configure_asset_table_columns()

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Supercell name")

        self._simple_nx = QSpinBox()
        self._simple_ny = QSpinBox()
        self._simple_nz = QSpinBox()
        for spinbox in (self._simple_nx, self._simple_ny, self._simple_nz):
            spinbox.setRange(1, 64)
            spinbox.setValue(1)
            spinbox.valueChanged.connect(lambda *_: self._generate_preview())

        self._advanced_toggle = QCheckBox("Advanced 3x3 matrix")
        self._advanced_toggle.setStyleSheet("QCheckBox { background: transparent; }")
        self._advanced_toggle.toggled.connect(self._on_advanced_toggled)

        self._advanced_cells: list[list[QSpinBox]] = []
        self._advanced_group = QGroupBox("Supercell Matrix")
        self._build_advanced_matrix_grid()

        self._status_label = QLabel("")
        self._status_label.setWordWrap(True)
        self._viewer = AtomicStructureViewer()

        self._new_button = QPushButton("New")
        self._save_button = QPushButton("Save")
        self._save_as_button = QPushButton("Save As")
        self._delete_button = QPushButton("Delete")
        self._suggest_button = QPushButton("Suggest…")
        self._use_button = QPushButton("Use Selected")
        self._close_button = QPushButton("Close")
        self._viewer_shutdown = False

        self._new_button.clicked.connect(self._new_asset)
        self._save_button.clicked.connect(self._save_current)
        self._save_as_button.clicked.connect(self._save_as_asset)
        self._delete_button.clicked.connect(self._delete_current)
        self._suggest_button.clicked.connect(self._suggest_supercell)
        self._use_button.clicked.connect(self._use_selected)
        self._close_button.clicked.connect(self.reject)
        self._use_button.setVisible(self._select_mode)

        self._build_layout()
        self._apply_matrix(initial_matrix or [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self._refresh_assets()
        self._generate_preview()

    def _shutdown_viewer(self) -> None:
        if self._viewer_shutdown:
            return
        self._viewer_shutdown = True
        try:
            self._viewer.shutdown()
        except RuntimeError:
            pass

    def done(self, result: int) -> None:  # noqa: D401 - Qt API override.
        """Finish the dialog after releasing viewer resources."""

        self._shutdown_viewer()
        super().done(result)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        self._shutdown_viewer()
        super().closeEvent(event)

    def selected_structure_id(self) -> UUID | None:
        """Return the supercell selected for caller use."""

        return self._selected_structure_id

    def selected_matrix(self) -> list[list[int]] | None:
        """Return the selected saved supercell matrix."""

        if self._selected_structure_id is None:
            return None
        structure = self._document.structures.get(self._selected_structure_id)
        if structure is None:
            return None
        asset = self._project_service._supercell_asset(structure)
        return None if asset is None else [list(row) for row in asset.matrix]

    def _configure_asset_table_columns(self) -> None:
        header = self._asset_table.horizontalHeader()
        header.setSectionResizeMode(self._ID_COLUMN, QHeaderView.ResizeMode.Fixed)
        for column in (self._NAME_COLUMN, self._MATRIX_COLUMN, self._ATOMS_COLUMN, self._USAGE_COLUMN):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)

    def matrix(self) -> list[list[int]]:
        """Return the currently configured editor matrix."""

        if not self._advanced_toggle.isChecked():
            return [
                [int(self._simple_nx.value()), 0, 0],
                [0, int(self._simple_ny.value()), 0],
                [0, 0, int(self._simple_nz.value())],
            ]
        return [[int(cell.value()) for cell in row] for row in self._advanced_cells]

    def _build_layout(self) -> None:
        outer = QVBoxLayout(self)
        outer.addWidget(self._title)
        outer.addWidget(self._subtitle)

        splitter = QSplitter(Qt.Orientation.Horizontal)

        library_panel = QWidget()
        library_layout = QVBoxLayout(library_panel)
        library_layout.setContentsMargins(0, 0, 8, 0)
        library_title = QLabel("Saved Supercells")
        style_section_subtitle(library_title)
        library_layout.addWidget(library_title)
        library_layout.addWidget(self._asset_table)

        editor_panel = QWidget()
        editor_layout = QVBoxLayout(editor_panel)
        editor_layout.setContentsMargins(8, 0, 0, 0)

        controls = QGroupBox("Supercell")
        controls_layout = QVBoxLayout(controls)

        name_row = QHBoxLayout()
        name_row.addWidget(QLabel("Name"))
        name_row.addWidget(self._name_edit, 1)
        controls_layout.addLayout(name_row)

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
        action_row.addWidget(self._new_button)
        action_row.addWidget(self._save_button)
        action_row.addWidget(self._save_as_button)
        action_row.addWidget(self._delete_button)
        action_row.addWidget(self._suggest_button)
        action_row.addStretch(1)
        controls_layout.addLayout(action_row)

        editor_layout.addWidget(controls)
        editor_layout.addWidget(self._status_label)
        editor_layout.addWidget(self._viewer, 1)

        splitter.addWidget(library_panel)
        splitter.addWidget(editor_panel)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        splitter.setSizes([360, 820])
        outer.addWidget(splitter, 1)

        bottom_row = QHBoxLayout()
        bottom_row.addStretch(1)
        bottom_row.addWidget(self._use_button)
        bottom_row.addWidget(self._close_button)
        outer.addLayout(bottom_row)

    def _build_advanced_matrix_grid(self) -> None:
        grid = QGridLayout(self._advanced_group)
        for row in range(3):
            matrix_row: list[QSpinBox] = []
            for column in range(3):
                cell = QSpinBox()
                cell.setRange(-64, 64)
                cell.setValue(1 if row == column else 0)
                cell.valueChanged.connect(lambda *_: self._generate_preview())
                matrix_row.append(cell)
                grid.addWidget(cell, row, column)
            self._advanced_cells.append(matrix_row)

    @staticmethod
    def _matrix_label(matrix: Sequence[Sequence[int]]) -> str:
        rows = [[int(value) for value in row] for row in matrix]
        if (
            rows[0][1] == 0
            and rows[0][2] == 0
            and rows[1][0] == 0
            and rows[1][2] == 0
            and rows[2][0] == 0
            and rows[2][1] == 0
        ):
            return f"{rows[0][0]}x{rows[1][1]}x{rows[2][2]}"
        return (
            f"[{rows[0][0]} {rows[0][1]} {rows[0][2]}; "
            f"{rows[1][0]} {rows[1][1]} {rows[1][2]}; "
            f"{rows[2][0]} {rows[2][1]} {rows[2][2]}]"
        )

    def _default_name(self, matrix: Sequence[Sequence[int]]) -> str:
        return f"{self._matrix_label(matrix)} supercell"

    def _apply_matrix(self, matrix: Sequence[Sequence[int]]) -> None:
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
        self._on_advanced_toggled(self._advanced_toggle.isChecked())

    def _on_advanced_toggled(self, enabled: bool) -> None:
        self._advanced_group.setVisible(enabled)
        self._generate_preview()

    def _asset_usage_text(self, structure_id: UUID) -> str:
        references = self._project_service.supercell_asset_references(
            self._document,
            asset_structure_id=structure_id,
        )
        if not references:
            return "Available"
        if self._project_service.supercell_asset_is_frozen(
            self._document,
            asset_structure_id=structure_id,
        ):
            return "Used by configured jobs"
        return "Selected"

    def _refresh_assets(self, *, select_id: UUID | None = None) -> None:
        if select_id is None:
            select_id = self._current_asset_id

        assets = self._project_service.list_workflow_supercell_assets(
            self._document,
            host_structure_id=self._source_structure.id,
        )
        self._updating_table = True
        try:
            self._asset_table.setRowCount(0)
            selected_row = -1
            for asset in assets:
                row = self._asset_table.rowCount()
                self._asset_table.insertRow(row)
                self._asset_table.setItem(row, self._ID_COLUMN, QTableWidgetItem(str(asset.structure_id)))
                self._asset_table.setItem(row, self._NAME_COLUMN, QTableWidgetItem(asset.label))
                self._asset_table.setItem(row, self._MATRIX_COLUMN, QTableWidgetItem(self._matrix_label(asset.matrix)))
                structure = self._document.structures.get(asset.structure_id)
                atom_count = len(structure.species) if structure is not None else 0
                self._asset_table.setItem(row, self._ATOMS_COLUMN, QTableWidgetItem(str(atom_count)))
                self._asset_table.setItem(row, self._USAGE_COLUMN, QTableWidgetItem(self._asset_usage_text(asset.structure_id)))
                if asset.structure_id == select_id:
                    selected_row = row
            self._asset_table.resizeColumnsToContents()
            if selected_row >= 0:
                self._current_asset_id = select_id
                self._asset_table.selectRow(selected_row)
            elif self._asset_table.rowCount() > 0 and self._current_asset_id is not None:
                self._asset_table.clearSelection()
        finally:
            self._updating_table = False

        self._refresh_action_state()

    def _selected_table_asset_id(self) -> UUID | None:
        selected = self._asset_table.selectedItems()
        if not selected:
            return None
        row = selected[0].row()
        item = self._asset_table.item(row, self._ID_COLUMN)
        if item is None:
            return None
        try:
            return UUID(item.text())
        except ValueError:
            return None

    def _on_asset_selection_changed(self) -> None:
        if self._updating_table:
            return
        asset_id = self._selected_table_asset_id()
        self._current_asset_id = asset_id
        if asset_id is None:
            self._refresh_action_state()
            return
        structure = self._document.structures.get(asset_id)
        if structure is None:
            self._refresh_action_state()
            return
        asset = self._project_service._supercell_asset(structure)
        if asset is None:
            self._refresh_action_state()
            return
        self._name_edit.setText(asset.label)
        self._apply_matrix(asset.matrix)
        self._generate_preview()
        self._refresh_action_state()

    def _refresh_action_state(self) -> None:
        has_asset = self._current_asset_id is not None
        frozen = (
            self._project_service.supercell_asset_is_frozen(
                self._document,
                asset_structure_id=self._current_asset_id,
            )
            if self._current_asset_id is not None
            else False
        )
        self._save_button.setEnabled(True)
        self._save_button.setToolTip(
            "Creates a new copy when the selected supercell is already used by configured calculations."
            if frozen
            else ""
        )
        self._delete_button.setEnabled(has_asset and not frozen)
        self._delete_button.setToolTip(
            "Configured calculations already use this supercell, so it cannot be deleted."
            if frozen
            else ""
        )
        self._use_button.setEnabled(has_asset)

    def _new_asset(self) -> None:
        self._asset_table.clearSelection()
        self._current_asset_id = None
        self._apply_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self._name_edit.setText(self._default_name(self.matrix()))
        self._generate_preview()
        self._refresh_action_state()

    def _save_as_asset(self):
        matrix = self.matrix()
        label = self._name_edit.text().strip() or self._default_name(matrix)
        try:
            saved = self._project_service.create_workflow_supercell_asset(
                self._document,
                workflow_stage=self._workflow_stage,
                host_structure_id=self._source_structure.id,
                matrix=matrix,
                label=label,
                reuse_existing=False,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Supercell unavailable", str(exc))
            return None
        self._current_asset_id = saved.id
        self._refresh_assets(select_id=saved.id)
        self._status_label.setText(f"Saved '{saved.label}' with {len(saved.species)} atoms.")
        return saved

    def _save_current(self):
        matrix = self.matrix()
        label = self._name_edit.text().strip() or self._default_name(matrix)
        if self._current_asset_id is None:
            return self._save_as_asset()

        structure = self._document.structures.get(self._current_asset_id)
        asset = self._project_service._supercell_asset(structure) if structure is not None else None
        if asset is None:
            return self._save_as_asset()

        try:
            if self._project_service.supercell_asset_is_frozen(
                self._document,
                asset_structure_id=self._current_asset_id,
            ) and matrix != asset.matrix:
                saved = self._project_service.create_workflow_supercell_asset(
                    self._document,
                    workflow_stage=self._workflow_stage,
                    host_structure_id=self._source_structure.id,
                    matrix=matrix,
                    label=label,
                    source_asset_structure_id=self._current_asset_id,
                    reuse_existing=False,
                )
            else:
                saved = (
                    self._project_service.update_workflow_supercell_asset_matrix(
                        self._document,
                        asset_structure_id=self._current_asset_id,
                        matrix=matrix,
                        label=label,
                    )
                    if matrix != asset.matrix
                    else self._project_service.rename_workflow_supercell_asset(
                        self._document,
                        asset_structure_id=self._current_asset_id,
                        label=label,
                    )
                )
        except ValueError as exc:
            QMessageBox.warning(self, "Supercell unavailable", str(exc))
            return None

        self._current_asset_id = saved.id
        self._refresh_assets(select_id=saved.id)
        self._status_label.setText(f"Saved '{saved.label}' with {len(saved.species)} atoms.")
        return saved

    def _delete_current(self) -> None:
        if self._current_asset_id is None:
            return
        structure = self._document.structures.get(self._current_asset_id)
        if structure is None:
            return
        if QMessageBox.question(
            self,
            "Delete supercell",
            f"Delete saved supercell '{structure.label}'?",
        ) != QMessageBox.StandardButton.Yes:
            return
        try:
            removed = self._project_service.delete_workflow_supercell_asset(
                self._document,
                asset_structure_id=self._current_asset_id,
            )
        except ValueError as exc:
            QMessageBox.warning(self, "Supercell in use", str(exc))
            return
        self._current_asset_id = None
        self._refresh_assets()
        self._new_asset()
        self._status_label.setText(f"Deleted '{removed.label}'.")

    def _use_selected(self) -> None:
        asset_id = self._current_asset_id
        if asset_id is None:
            saved = self._save_current()
            asset_id = saved.id if saved is not None else None
        if asset_id is None:
            return
        self._selected_structure_id = asset_id
        self.accept()

    def _suggest_supercell(self) -> None:
        try:
            suggestion = suggest_supercell(
                self._source_structure.lattice_vectors,
                primitive_atom_count=len(self._source_structure.species),
            )
        except SupercellSuggestionError as exc:
            QMessageBox.warning(self, "Suggest Supercell", str(exc))
            return
        dialog = SupercellSuggestionDialog(suggestion=suggestion, parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        matrix = dialog.chosen_matrix()
        if matrix is not None:
            self._apply_matrix(matrix)

    def _generate_preview(self) -> None:
        matrix = self.matrix()
        try:
            expanded_structure, expanded_spins = build_supercell_structure(
                self._source_structure,
                matrix,
                atomic_spins=self._project_service.structure_atomic_spins(self._source_structure),
                propagation_vector=self._project_service.structure_propagation_vector(self._source_structure),
            )
        except ValueError as exc:
            self._status_label.setText(f"Supercell generation failed: {exc}")
            self._viewer.set_structure(None)
            return

        self._viewer.set_structure(expanded_structure)
        self._viewer.set_spins(expanded_spins)
        self._status_label.setText(
            f"Previewing {self._matrix_label(matrix)} -> {len(expanded_structure.species)} atoms."
        )
