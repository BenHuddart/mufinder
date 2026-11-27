"""Supercell Editor dialog: generate, preview, name and manage saved supercells."""

from __future__ import annotations

from collections.abc import Callable, Sequence
from contextlib import ExitStack
from itertools import chain
from uuid import UUID

from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QKeyEvent, QShowEvent
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QDialog,
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
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.app import workflow_table_utils
from mufinder.app.adapter_common import (
    IDENTITY_SUPERCELL,
    coerce_supercell_matrix,
    is_diagonal_scaling,
    supercell_label,
)
from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.blocking_call import run_blocking_call
from mufinder.app.fact_card import FactCard
from mufinder.app.help_bubble import help_button
from mufinder.app.magnetic_component_dialogs import q_vector_text
from mufinder.app.supercell_suggestion_dialog import SupercellSuggestionDialog
from mufinder.app.theme import style_section_subtitle, style_section_title, style_warning_text
from mufinder.app.viewer_dialog import StructureViewerDialog
from mufinder.app.workflow_table_utils import bind_delete_shortcuts, create_workflow_table
from mufinder.domain.enums import WorkflowStage
from mufinder.domain.models import Structure
from mufinder.domain.text import plural
from mufinder.science.supercell import (
    build_supercell_structure,
    normalize_supercell_matrix,
    supercell_fits_propagation_vector,
)
from mufinder.science.supercell_suggestion import SupercellSuggestionError, suggest_supercell
from mufinder.services.host_structure_facts import supercell_facts
from mufinder.services.journal import workflow_label
from mufinder.services.project_service import ProjectDocument, ProjectService, host_entry
from mufinder.services.supercell_assets import WorkflowSupercellAsset, supercell_asset

#: What the propagation-vector warning means, behind its help icon.
PROPAGATION_FIT_HELP = (
    "Each copy of an atom in a supercell has its spin multiplied by cos(2π k·R), with R the"
    " lattice translation of its cell. The spins repeat across the supercell's boundary only"
    " when k·A is a whole number for each supercell vector A, a row of the matrix. For"
    " k = (0.5, 0, 0) a 2x1x1 cell fits and a 3x1x1 cell does not."
)


#: Rows that pick a cell by its matrix are not selections of a saved supercell.
_ROW_CHOICES_NOT_COUNTED = (
    "Rows on the Pristine Relaxation, Muon Relaxations and UEP pages pick their cell by its"
    " matrix and are not counted here."
)


def _names(stages: Sequence[WorkflowStage]) -> str:
    """``Cutoff Convergence``, ``UEP and Muon Sites``, ``A, B and C``."""

    labels = [workflow_label(stage) for stage in stages]
    return " and ".join(filter(None, (", ".join(labels[:-1]), labels[-1])))


def _usage_summary(stages: Sequence[WorkflowStage]) -> str:
    """One workflow by name, several by their count, so the column stays narrow."""

    return workflow_label(stages[0]) if len(stages) == 1 else plural(len(stages), "workflow")


def _propagation_text(k_vector: Sequence[float]) -> str:
    """``(0.5, 0, 0)``, each component as the Magnetic Fields dialogs write it."""

    return f"({q_vector_text(k_vector)})"


class SupercellEditorDialog(StructureViewerDialog):
    """Manage saved project supercells for one source structure."""

    _ID_COLUMN = 0
    _NAME_COLUMN = 1
    _MATRIX_COLUMN = 2
    _ATOMS_COLUMN = 3
    _USAGE_COLUMN = 4
    # Stepper arrows, padding and line edit margins under the application theme,
    # around the widest value.
    _SCALING_BOX_CHROME = 70
    # The property holding a scaling box's own tooltip, shown while it is enabled.
    _SCALING_TOOLTIP = "mufinderScalingTooltip"
    #: The scaling boxes' tooltip while the general matrix sets the cell.
    GENERAL_MATRIX_SETS_CELL = "The general 3×3 matrix sets the cell while it is ticked."

    def __init__(
        self,
        *,
        document: ProjectDocument,
        project_service: ProjectService,
        source_structure: Structure,
        workflow_stage: WorkflowStage | None = None,
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
        self._previewed_matrix: list[list[int]] | None = None
        # The general matrix left by unticking, restored by ticking again; only while open.
        self._remembered_general_matrix: list[list[int]] | None = None
        self._updating_table = False

        self._title = QLabel("Supercell Editor")
        style_section_title(self._title)
        host = host_entry(document, source_structure.id)
        host_label = host.label if host is not None else source_structure.label
        self._subtitle = QLabel(f"Supercells of {host_label}")
        # The host's name is the user's text, never markup.
        self._subtitle.setTextFormat(Qt.TextFormat.PlainText)
        style_section_subtitle(self._subtitle)
        self._subtitle.setWordWrap(True)

        self._asset_table = create_workflow_table(
            ("Id", "Name", "Matrix", "Atoms", "Usage"),
            selection_mode=QAbstractItemView.SelectionMode.SingleSelection,
            stretch_last_section=False,
            accessible_name="Saved Supercells",
        )
        self._asset_table.setColumnHidden(self._ID_COLUMN, True)
        self._asset_table.itemSelectionChanged.connect(self._on_asset_selection_changed)
        self._configure_asset_table_columns()
        self._delete_shortcuts = bind_delete_shortcuts(
            self._asset_table, self._delete_from_keyboard
        )

        self._name_edit = QLineEdit()
        self._name_edit.setPlaceholderText("Supercell name")
        self._name_edit.setAccessibleName("Supercell name")

        self._simple_nx = QSpinBox()
        self._simple_ny = QSpinBox()
        self._simple_nz = QSpinBox()
        for spinbox, name, axis in (
            (self._simple_nx, "Nx", "a"),
            (self._simple_ny, "Ny", "b"),
            (self._simple_nz, "Nz", "c"),
        ):
            spinbox.setRange(1, 64)
            spinbox.setValue(1)
            spinbox.ensurePolished()
            metrics = spinbox.fontMetrics()
            spinbox.setFixedWidth(
                max(metrics.horizontalAdvance(str(n)) for n in range(1, spinbox.maximum() + 1))
                + self._SCALING_BOX_CHROME
            )
            spinbox.setAccessibleName(name)
            spinbox.setProperty(
                self._SCALING_TOOLTIP, f"{name}: repeats of the unit cell along {axis}."
            )
            spinbox.valueChanged.connect(lambda *_: self._on_scaling_changed())

        self._advanced_toggle = QCheckBox("General 3×3 matrix")
        self._advanced_toggle.toggled.connect(self._on_advanced_toggled)

        self._advanced_cells: list[list[QSpinBox]] = []
        self._advanced_group = QGroupBox("Supercell Matrix")
        self._build_advanced_matrix_grid()

        self._facts_card = FactCard(accessible_name="Previewed supercell")
        self._propagation_warning = QLabel()
        style_warning_text(self._propagation_warning)
        self._propagation_help = help_button(
            PROPAGATION_FIT_HELP, accessible_name="About fitting the propagation vector"
        )
        self._propagation_row = QWidget()
        self._propagation_row.setVisible(False)
        # Messages that are not facts (saved, deleted, generation failed); they sit in the
        # bottom row beside Close, so showing or clearing one moves nothing above it.
        self._status_label = QLabel("")
        # Supercell names are the user's text, never markup.
        self._status_label.setTextFormat(Qt.TextFormat.PlainText)
        self._status_label.setWordWrap(True)
        self._viewer = self.hold_viewer(AtomicStructureViewer())

        self._new_button = QPushButton("New")
        self._save_button = QPushButton("Save")
        self._save_as_button = QPushButton("Save As")
        self._delete_button = QPushButton("Delete")
        self._suggest_button = QPushButton("Suggest…")
        self._use_button = QPushButton("Use Selected")
        self._close_button = QPushButton("Close")

        self._new_button.clicked.connect(self._new_asset)
        self._save_button.clicked.connect(self._save_current)
        self._save_as_button.clicked.connect(self._save_as_asset)
        self._delete_button.clicked.connect(self._delete_current)
        self._suggest_button.clicked.connect(self._suggest_supercell)
        self._use_button.clicked.connect(self._use_selected)
        self._close_button.clicked.connect(self.reject)
        self._use_button.setVisible(self._select_mode)

        self._build_layout()
        # Return in the name or a scaling box must not press a button: the first
        # automatic default would be New, which discards the cell being edited.
        # keyPressEvent keeps Return working on a focused button.
        for button in self.findChildren(QPushButton):
            button.setAutoDefault(False)
        self._apply_matrix(initial_matrix or [[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self._open_on_saved_cell()

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
        asset = supercell_asset(structure)
        return None if asset is None else [list(row) for row in asset.matrix]

    def _configure_asset_table_columns(self) -> None:
        # The Name column takes the spare width, so the columns always reach the
        # table's right edge.
        header = self._asset_table.horizontalHeader()
        header.setSectionResizeMode(self._ID_COLUMN, QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(self._NAME_COLUMN, QHeaderView.ResizeMode.Stretch)
        for column in (self._MATRIX_COLUMN, self._ATOMS_COLUMN, self._USAGE_COLUMN):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)

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

        # The saved list and the buttons that act on it: New and Delete.
        library_panel = QWidget()
        library_layout = QVBoxLayout(library_panel)
        library_layout.setContentsMargins(0, 0, 8, 0)
        library_title = QLabel("Saved Supercells")
        style_section_subtitle(library_title)
        library_layout.addWidget(library_title)
        library_layout.addWidget(self._asset_table)
        list_actions = QHBoxLayout()
        list_actions.addWidget(self._new_button)
        list_actions.addWidget(self._delete_button)
        list_actions.addStretch(1)
        library_layout.addLayout(list_actions)

        # The cell being edited, the buttons that act on it, and its preview.
        editor_panel = QWidget()
        editor_layout = QVBoxLayout(editor_panel)
        editor_layout.setContentsMargins(8, 0, 0, 0)

        controls = QGroupBox("Supercell")
        controls_layout = QVBoxLayout(controls)

        scaling_row = QHBoxLayout()
        scaling_row.addWidget(self._simple_nx)
        scaling_row.addWidget(QLabel("×"))
        scaling_row.addWidget(self._simple_ny)
        scaling_row.addWidget(QLabel("×"))
        scaling_row.addWidget(self._simple_nz)
        scaling_row.addStretch(1)

        fields = QGridLayout()
        fields.setColumnStretch(1, 1)
        fields.addWidget(QLabel("Name"), 0, 0)
        fields.addWidget(self._name_edit, 0, 1)
        fields.addWidget(QLabel("Scaling"), 1, 0)
        fields.addLayout(scaling_row, 1, 1)
        fields.addWidget(self._advanced_toggle, 2, 1)
        controls_layout.addLayout(fields)
        controls_layout.addWidget(self._advanced_group)

        cell_actions = QHBoxLayout()
        cell_actions.addWidget(self._save_button)
        cell_actions.addWidget(self._save_as_button)
        cell_actions.addWidget(self._suggest_button)
        cell_actions.addStretch(1)
        controls_layout.addLayout(cell_actions)

        propagation_layout = QHBoxLayout(self._propagation_row)
        propagation_layout.setContentsMargins(0, 0, 0, 0)
        propagation_layout.addWidget(self._propagation_warning)
        propagation_layout.addWidget(self._propagation_help)
        propagation_layout.addStretch(1)

        editor_layout.addWidget(controls)
        editor_layout.addWidget(self._facts_card)
        # Below the card, so the warning coming and going resizes only the viewer.
        editor_layout.addWidget(self._propagation_row)
        editor_layout.addWidget(self._viewer, 1)

        splitter.addWidget(library_panel)
        splitter.addWidget(editor_panel)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 7)
        splitter.setSizes([440, 740])
        outer.addWidget(splitter, 1)

        bottom_row = QHBoxLayout()
        bottom_row.addWidget(self._status_label, 1)
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
                cell.setAccessibleName(f"Matrix row {row + 1} column {column + 1}")
                cell.valueChanged.connect(lambda *_: self._on_matrix_cell_changed())
                matrix_row.append(cell)
                grid.addWidget(cell, row, column)
            self._advanced_cells.append(matrix_row)

    @staticmethod
    def _matrix_label(matrix: Sequence[Sequence[int]]) -> str:
        # The editor names the unit cell by its scaling, so the Matrix column and
        # the default name read ``1x1x1`` rather than ``Unit cell``.
        return supercell_label(
            coerce_supercell_matrix(matrix), unit_cell="1x1x1", spell_out_custom=True
        )

    def _default_name(self, matrix: Sequence[Sequence[int]]) -> str:
        return f"{self._matrix_label(matrix)} supercell"

    def _apply_matrix(self, matrix: Sequence[Sequence[int]]) -> None:
        # A loaded cell replaces whatever general matrix an untick left behind.
        self._remembered_general_matrix = None
        rows = [list(row) for row in matrix]
        well_formed = len(rows) == 3 and all(len(row) == 3 for row in rows)
        parsed = (coerce_supercell_matrix(rows) if well_formed else None) or IDENTITY_SUPERCELL
        is_diagonal = is_diagonal_scaling(parsed)
        # Set every box with its signal blocked so the whole matrix change builds
        # one preview, not one per box.
        simple_boxes = (self._simple_nx, self._simple_ny, self._simple_nz)
        with ExitStack() as blocked:
            for widget in (*simple_boxes, *chain(*self._advanced_cells), self._advanced_toggle):
                blocked.enter_context(QSignalBlocker(widget))
            for axis, spinbox in enumerate(simple_boxes):
                spinbox.setValue(max(1, parsed[axis][axis]))
            for row in range(3):
                for column in range(3):
                    self._advanced_cells[row][column].setValue(parsed[row][column])
            self._advanced_toggle.setChecked(not is_diagonal)
        self._show_matrix_mode()
        self._generate_preview()

    @property
    def _scaling_boxes(self) -> tuple[QSpinBox, QSpinBox, QSpinBox]:
        return (self._simple_nx, self._simple_ny, self._simple_nz)

    def _show_matrix_mode(self) -> None:
        """Show the matrix boxes and disable the scaling boxes while the general matrix is ticked."""

        general = self._advanced_toggle.isChecked()
        self._advanced_group.setVisible(general)
        for spinbox in self._scaling_boxes:
            spinbox.setEnabled(not general)
            spinbox.setToolTip(
                self.GENERAL_MATRIX_SETS_CELL
                if general
                else str(spinbox.property(self._SCALING_TOOLTIP))
            )

    def _set_boxes(self, boxes: Sequence[QSpinBox], values: Sequence[int]) -> None:
        with ExitStack() as blocked:
            for box in boxes:
                blocked.enter_context(QSignalBlocker(box))
            for box, value in zip(boxes, values, strict=True):
                box.setValue(int(value))

    def _on_scaling_changed(self) -> None:
        # Disabled scaling boxes never drive the cell.
        if not self._advanced_toggle.isChecked():
            self._generate_preview()

    def _on_matrix_cell_changed(self) -> None:
        if not self._advanced_toggle.isChecked():
            return
        matrix = self.matrix()
        if is_diagonal_scaling(coerce_supercell_matrix(matrix)):
            # The disabled scaling boxes show a plain scaling; otherwise they keep theirs.
            self._set_boxes(self._scaling_boxes, [matrix[axis][axis] for axis in range(3)])
        self._generate_preview()

    def _on_advanced_toggled(self, enabled: bool) -> None:
        cells = list(chain(*self._advanced_cells))
        if enabled:
            remembered = self._remembered_general_matrix
            if remembered is not None:
                # A general matrix the user left by unticking comes back as it was.
                self._set_boxes(cells, list(chain(*remembered)))
            else:
                # The matrix starts from the scaling, so ticking never changes the cell.
                scaling = [int(box.value()) for box in self._scaling_boxes]
                self._set_boxes(
                    cells,
                    [
                        scaling[row] if row == column else 0
                        for row in range(3)
                        for column in range(3)
                    ],
                )
        else:
            general = [[int(cell.value()) for cell in row] for row in self._advanced_cells]
            plain = is_diagonal_scaling(coerce_supercell_matrix(general))
            # Only a matrix the scaling boxes cannot express is worth keeping.
            self._remembered_general_matrix = None if plain else general
        self._show_matrix_mode()
        self._generate_preview()

    def _asset_usage(self, structure_id: UUID) -> tuple[str, str]:
        """The Usage cell of a saved cell and its tooltip, which names every workflow."""

        with_jobs = self._project_service.workflows_with_jobs_using_supercell_asset(
            self._document, asset_structure_id=structure_id
        )
        selecting = self._project_service.workflows_selecting_supercell_asset(
            self._document, asset_structure_id=structure_id
        )
        details = []
        if with_jobs:
            details.append(
                f"Configured jobs of {_names(with_jobs)} use this supercell, so a changed"
                " matrix is saved as a copy."
            )
        if selecting:
            verb = "has" if len(selecting) == 1 else "have"
            details.append(f"{_names(selecting)} {verb} it selected.")
        if with_jobs:
            text = f"Jobs in {_usage_summary(with_jobs)}"
        elif selecting:
            text = f"Selected by {_usage_summary(selecting)}"
        else:
            text = "Not selected"
            details.append(
                "No workflow's simulation cell is this supercell and no configured job uses it."
            )
        details.append(_ROW_CHOICES_NOT_COUNTED)
        return text, " ".join(details)

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
                self._asset_table.setItem(
                    row, self._ID_COLUMN, QTableWidgetItem(str(asset.structure_id))
                )
                name_item = QTableWidgetItem(asset.label)
                name_item.setToolTip(asset.label)
                self._asset_table.setItem(row, self._NAME_COLUMN, name_item)
                self._asset_table.setItem(
                    row, self._MATRIX_COLUMN, QTableWidgetItem(self._matrix_label(asset.matrix))
                )
                structure = self._document.structures.get(asset.structure_id)
                atom_count = len(structure.species) if structure is not None else 0
                self._asset_table.setItem(
                    row, self._ATOMS_COLUMN, QTableWidgetItem(str(atom_count))
                )
                usage, usage_tooltip = self._asset_usage(asset.structure_id)
                usage_item = QTableWidgetItem(usage)
                usage_item.setToolTip(usage_tooltip)
                self._asset_table.setItem(row, self._USAGE_COLUMN, usage_item)
                if asset.structure_id == select_id:
                    selected_row = row
            if selected_row >= 0:
                self._current_asset_id = select_id
                self._asset_table.selectRow(selected_row)
            elif self._asset_table.rowCount() > 0 and self._current_asset_id is not None:
                self._clear_table_selection()
        finally:
            self._updating_table = False

        self._refresh_action_state()

    def _clear_table_selection(self) -> None:
        self._asset_table.clearSelection()
        self._asset_table.setCurrentItem(None)

    def _open_on_saved_cell(self) -> None:
        """Select the saved cell the opening matrix names, or leave every row unmarked."""

        asset = self._workflow_cell_with_opening_matrix()
        if asset is None:
            asset = self._project_service.latest_workflow_supercell_asset(
                self._document,
                host_structure_id=self._source_structure.id,
                matrix=self.matrix(),
            )
        if asset is None:
            self._refresh_assets()
        else:
            self._refresh_assets(select_id=asset.structure_id)
            self._name_edit.setText(asset.label)
        # A table that takes the opening focus frames its first row as current.
        self._name_edit.setFocus()

    def _workflow_cell_with_opening_matrix(self) -> WorkflowSupercellAsset | None:
        if self._workflow_stage is None:
            return None
        selection = self._project_service.get_workflow_supercell_selection(
            self._document, self._workflow_stage
        )
        structure = self._document.structures.get(selection.structure_id)
        asset = supercell_asset(structure) if structure is not None else None
        return asset if asset is not None and asset.matrix == self.matrix() else None

    def keyPressEvent(self, event: QKeyEvent) -> None:  # noqa: N802 - Qt API naming.
        if event.key() not in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            super().keyPressEvent(event)
            return
        focused = self.focusWidget()
        if isinstance(focused, QPushButton) and not event.isAutoRepeat():
            focused.click()

    def showEvent(self, event: QShowEvent) -> None:  # noqa: N802 - Qt API naming.
        super().showEvent(event)
        # QDialog gives the opening focus after this event, and that selects the name;
        # a stray key press must not replace it.
        QTimer.singleShot(0, self._name_edit.deselect)

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
        asset = supercell_asset(structure)
        if asset is None:
            self._refresh_action_state()
            return
        self._apply_matrix(asset.matrix)
        self._name_edit.setText(asset.label)
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
        self._save_button.setEnabled(has_asset)
        if not has_asset:
            save_tooltip = "Select a saved supercell to update it, or use 'Save As' to add one."
        elif frozen:
            save_tooltip = (
                "Creates a new copy when the selected supercell is already used by configured"
                " calculations."
            )
        else:
            save_tooltip = ""
        self._save_button.setToolTip(save_tooltip)
        self._delete_button.setEnabled(has_asset and not frozen)
        self._delete_button.setToolTip(
            "Configured calculations already use this supercell, so it cannot be deleted."
            if frozen
            else ""
        )
        matches_selection = self._selected_asset_matrix() == self.matrix()
        self._use_button.setEnabled(matches_selection)
        self._use_button.setToolTip(
            "Save the changed matrix before using this supercell."
            if has_asset and not matches_selection
            else ""
        )

    def _selected_asset_matrix(self) -> list[list[int]] | None:
        if self._current_asset_id is None:
            return None
        structure = self._document.structures.get(self._current_asset_id)
        asset = supercell_asset(structure) if structure is not None else None
        return None if asset is None else asset.matrix

    def _new_asset(self) -> None:
        self._clear_table_selection()
        self._current_asset_id = None
        self._apply_matrix([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
        self._name_edit.setText(self._default_name(self.matrix()))
        self._refresh_action_state()

    def _create_asset(
        self,
        matrix: list[list[int]],
        label: str,
        *,
        source_asset_structure_id: UUID | None = None,
    ) -> Structure:
        return self._project_service.create_workflow_supercell_asset(
            self._document,
            workflow_stage=self._workflow_stage,
            host_structure_id=self._source_structure.id,
            matrix=matrix,
            label=label,
            source_asset_structure_id=source_asset_structure_id,
            reuse_existing=False,
        )

    def _store(
        self,
        save: Callable[[], Structure | None],
        *,
        copy_of: str | None = None,
    ) -> Structure | None:
        """Run ``save``, then select what it saved and report it; ``None`` if nothing was.

        ``copy_of`` names the frozen cell a copy was saved in place of.
        """

        try:
            saved = save()
        except ValueError as exc:
            QMessageBox.warning(self, "Supercell unavailable", str(exc))
            return None
        if saved is None:
            return None
        self._current_asset_id = saved.id
        self._refresh_assets(select_id=saved.id)
        self._name_edit.setText(saved.label)
        atoms = plural(len(saved.species), "atom")
        self._status_label.setText(
            f"Saved '{saved.label}' with {atoms}"
            if copy_of is None
            else f"Saved copy '{saved.label}' with {atoms}; configured calculations use"
            f" '{copy_of}', which was not changed"
        )
        return saved

    def _save_as_asset(self) -> Structure | None:
        matrix = self.matrix()
        label = self._name_edit.text().strip() or self._default_name(matrix)
        return self._store(lambda: self._create_asset(matrix, label))

    def _save_current(self) -> Structure | None:
        matrix = self.matrix()
        label = self._name_edit.text().strip() or self._default_name(matrix)
        if self._current_asset_id is None:
            return None
        asset_id = self._current_asset_id

        structure = self._document.structures.get(asset_id)
        asset = supercell_asset(structure) if structure is not None else None
        if asset is None:
            return self._save_as_asset()

        changed = matrix != asset.matrix
        copied = changed and self._project_service.supercell_asset_is_frozen(
            self._document, asset_structure_id=asset_id
        )

        def save() -> Structure | None:
            normalize_supercell_matrix(matrix)
            if copied:
                return self._create_asset(
                    matrix,
                    self._project_service.distinct_supercell_asset_label(
                        self._document,
                        host_structure_id=self._source_structure.id,
                        label=label,
                    ),
                    source_asset_structure_id=asset_id,
                )
            if not changed:
                return self._project_service.rename_workflow_supercell_asset(
                    self._document, asset_structure_id=asset_id, label=label
                )
            if not self._confirm_shared_change(asset):
                return None
            return self._project_service.update_workflow_supercell_asset_matrix(
                self._document, asset_structure_id=asset_id, matrix=matrix, label=label
            )

        return self._store(save, copy_of=asset.label if copied else None)

    def _confirm_shared_change(self, asset: WorkflowSupercellAsset) -> bool:
        """Ask before a matrix change reaches workflows other than the one editing."""

        others = [
            stage
            for stage in self._project_service.workflows_selecting_supercell_asset(
                self._document, asset_structure_id=asset.structure_id
            )
            if stage != self._workflow_stage
        ]
        if not others:
            return True
        return workflow_table_utils.confirm_destructive(
            self,
            "Change shared supercell",
            f"Change the matrix of '{asset.label}' for {_names(others)}?",
            detail=(
                f"{plural(len(others), 'workflow has', 'workflows have')} this supercell selected"
                " and will use the new matrix for calculations configured from now on."
                " 'Save As' keeps it and adds a new supercell instead."
            ),
            accept_label="Change supercell",
        )

    def _delete_from_keyboard(self) -> None:
        """Delete or Backspace on the saved list: the Delete button's confirmed delete."""

        if self._delete_button.isEnabled():
            self._delete_current()

    def _delete_current(self) -> None:
        if self._current_asset_id is None:
            return
        structure = self._document.structures.get(self._current_asset_id)
        if structure is None:
            return
        if not workflow_table_utils.confirm_destructive(
            self,
            "Delete supercell",
            f"Delete saved supercell '{structure.label}'?",
            accept_label="Delete supercell",
        ):
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
        self._status_label.setText(f"Deleted '{removed.label}'")

    def _use_selected(self) -> None:
        if self._current_asset_id is None:
            return
        self._selected_structure_id = self._current_asset_id
        self.accept()

    def _suggest_supercell(self) -> None:
        try:
            source = self._source_structure
            suggestion = run_blocking_call(
                self,
                lambda: suggest_supercell(
                    source.lattice_vectors, primitive_atom_count=len(source.species)
                ),
                title="Suggest Supercell",
                description="Searching for supercells…",
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

    def _follow_matrix_with_automatic_name(self, matrix: list[list[int]]) -> None:
        previous = self._previewed_matrix
        if previous is not None and self._name_edit.text() == self._default_name(previous):
            self._name_edit.setText(self._default_name(matrix))
        self._previewed_matrix = matrix

    def _show_propagation_fit(self, matrix: list[list[int]], k_vector: Sequence[float]) -> None:
        fits = supercell_fits_propagation_vector(matrix, k_vector)
        if not fits:
            self._propagation_warning.setText(
                f"This supercell does not fit the propagation vector k = "
                f"{_propagation_text(k_vector)}."
            )
        self._propagation_row.setVisible(not fits)

    def _generate_preview(self) -> None:
        matrix = self.matrix()
        self._follow_matrix_with_automatic_name(matrix)
        self._refresh_action_state()
        matrix_fact = ("Matrix", self._matrix_label(matrix))
        k_vector = self._project_service.structure_propagation_vector(self._source_structure)
        try:
            expanded_structure, expanded_spins = build_supercell_structure(
                self._source_structure,
                matrix,
                atomic_spins=self._project_service.structure_atomic_spins(self._source_structure),
                propagation_vector=k_vector,
            )
        except ValueError as exc:
            self._facts_card.set_groups([[matrix_fact]])
            self._propagation_row.setVisible(False)
            self._status_label.setText(f"Supercell generation failed: {exc}")
            self._viewer.set_structure(None)
            return

        with self._viewer.batch_update():
            self._viewer.set_structure(expanded_structure)
            self._viewer.set_spins(expanded_spins)
        self._facts_card.set_groups([[matrix_fact, *supercell_facts(expanded_structure)]])
        self._show_propagation_fit(matrix, k_vector)
        # A message about an earlier cell does not describe this one.
        self._status_label.setText("")
