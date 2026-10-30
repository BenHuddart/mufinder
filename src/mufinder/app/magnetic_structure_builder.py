"""Magnetic Structure Builder dialog."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace
from html import escape

import numpy as np
from PySide6.QtCore import QItemSelection, QItemSelectionModel, Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.app import workflow_table_utils
from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.magnetic_component_dialogs import (
    MagneticComponentDialog,
    MagneticComponentTable,
    MagneticPresetDialog,
    component_moment_label,
    component_q_label,
    create_q_basis_table,
    fill_q_basis_table,
    q_vector_text,
    selector_label,
)
from mufinder.app.magnetic_texture_dialog import MagneticTextureDialog
from mufinder.app.screen_utils import clamped_size
from mufinder.app.theme import (
    current_palette,
    magnetic_structure_builder_stylesheet,
    style_hint_text,
    style_section_title,
    use_square_top_table_corners,
)
from mufinder.app.viewer_dialog import StructureViewerDialog
from mufinder.domain.models import Structure
from mufinder.domain.text import plural
from mufinder.science.magnetic_fields import classify_magnetic_model
from mufinder.science.magnetic_structure import (
    CHECKLIST_DONE,
    CHECKLIST_PROBLEM,
    MagneticComponent,
    MagneticStructureError,
    MagneticStructureModel,
    allocate_q_basis_row,
    allocate_texture_q_basis_rows,
    atom_phases_for_selection,
    component_propagation_coeff,
    component_propagation_rlu,
    component_q_basis_indices,
    duplicate_magnetic_components,
    enabled_magnetic_components,
    is_texture_component,
    magnetic_builder_checklist,
    magnetic_model_for_host,
    magnetic_model_from_legacy_structure,
    moments_in_supercell,
    move_magnetic_components,
    q_basis_slots,
    set_magnetic_components_enabled,
    set_q_basis_vector,
    tiled_supercell_structure,
)
from mufinder.services.project_document import active_host_structure
from mufinder.services.project_service import ProjectDocument


class MagneticStructureBuilderDialog(StructureViewerDialog):
    """Dialog for building and previewing a magnetic structure."""

    _COMPONENT_ID_ROLE = int(Qt.ItemDataRole.UserRole) + 1
    _COMPONENT_HEADERS = ("Enabled", "Component", "Type", "Apply to", "q", "Moment")
    _ENABLED_COLUMN = 0
    _LABEL_COLUMN = 1
    _Q_BASIS_VECTOR_COLUMN = 1
    _Q_BASIS_USERS_COLUMN = 2

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._document: ProjectDocument | None = None
        self._host_structure: Structure | None = None
        self._model = MagneticStructureModel()
        self._updating = False
        self._preview_atom_indices = np.empty((0,), dtype=int)

        self.setWindowTitle("Magnetic Structure Builder")
        self.resize(clamped_size(self, 1180, 760))
        self.setMinimumSize(920, 620)

        self._title = QLabel("Magnetic Structure Builder")
        style_section_title(self._title)

        self._preview_status = QLabel("Preview: -")
        self._classification = QLabel("Structure type: -")

        self._component_table = MagneticComponentTable(
            0,
            len(self._COMPONENT_HEADERS),
            delete_callback=self._remove_selected_components,
        )
        self._component_table.setHorizontalHeaderLabels(list(self._COMPONENT_HEADERS))
        self._component_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._component_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._component_table.setEditTriggers(
            QAbstractItemView.EditTrigger.DoubleClicked
            | QAbstractItemView.EditTrigger.EditKeyPressed
        )
        self._component_table.itemSelectionChanged.connect(self._on_component_selection_changed)
        self._component_table.itemChanged.connect(self._on_component_item_changed)
        self._configure_component_table_columns()
        use_square_top_table_corners(self._component_table)

        self._q_basis_table = create_q_basis_table(editable=True)
        self._q_basis_table.itemChanged.connect(self._on_q_basis_item_changed)
        self._q_basis_hint = QLabel(
            "Each component stores its q as a combination of these rows. Double-click a q vector to edit it; "
            "every component that uses the row follows it. Rows used by a texture change through 'Edit…'."
        )
        self._q_basis_hint.setWordWrap(True)
        style_hint_text(self._q_basis_hint)

        self._add_button = QPushButton("Add Component")
        self._add_button.clicked.connect(self._add_component)
        self._add_preset_button = QPushButton("Add Preset…")
        self._add_preset_button.setToolTip(
            "Build a helix, conical, cycloid or spin-density wave from a propagation direction and crystal axes."
        )
        self._add_preset_button.clicked.connect(self._add_preset)
        self._add_texture_button = QPushButton("Add Texture…")
        self._add_texture_button.clicked.connect(self._add_texture)
        self._edit_button = QPushButton("Edit…")
        self._edit_button.clicked.connect(self._edit_selected_component)
        self._duplicate_button = QPushButton("Duplicate")
        self._duplicate_button.setToolTip(
            "Add a copy of each selected component below its original."
        )
        self._duplicate_button.clicked.connect(self._duplicate_selected_components)
        self._remove_button = QPushButton("Remove")
        self._remove_button.clicked.connect(self._remove_selected_components)
        self._move_up_button = QPushButton("Move Up")
        self._move_up_button.clicked.connect(lambda: self._move_selected_components(-1))
        self._move_down_button = QPushButton("Move Down")
        self._move_down_button.clicked.connect(lambda: self._move_selected_components(1))
        self._build_from_host_button = QPushButton("Build From Host Spins")
        self._build_from_host_button.clicked.connect(self._build_from_host_spins)

        self._repeat_a = self._repeat_spinbox()
        self._repeat_b = self._repeat_spinbox()
        self._repeat_c = self._repeat_spinbox()
        for spinbox in (self._repeat_a, self._repeat_b, self._repeat_c):
            spinbox.valueChanged.connect(lambda *_: self._refresh_preview())

        self._checklist_labels: dict[str, QLabel] = {}
        self._checklist_group = QGroupBox("Checklist")
        checklist_layout = QVBoxLayout(self._checklist_group)
        checklist_layout.setSpacing(4)
        for key in ("components", "q_basis", "preview", "ready"):
            label = QLabel("")
            label.setTextFormat(Qt.TextFormat.RichText)
            label.setWordWrap(True)
            label.setStyleSheet("background: transparent;")
            self._checklist_labels[key] = label
            checklist_layout.addWidget(label)

        self._viewer = self.hold_viewer(AtomicStructureViewer())
        self._viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setStyleSheet(magnetic_structure_builder_stylesheet())

        self._build_layout()
        self._sync_enabled_state()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.addWidget(self._title)

        action_row = QHBoxLayout()
        action_row.addWidget(self._add_button)
        action_row.addWidget(self._add_preset_button)
        action_row.addWidget(self._add_texture_button)
        action_row.addWidget(self._edit_button)
        action_row.addWidget(self._duplicate_button)
        action_row.addWidget(self._remove_button)
        action_row.addWidget(self._move_up_button)
        action_row.addWidget(self._move_down_button)
        action_row.addWidget(self._build_from_host_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        supercell_group = QGroupBox("Preview Supercell")
        supercell_layout = QHBoxLayout(supercell_group)
        supercell_layout.addWidget(QLabel("a"))
        supercell_layout.addWidget(self._repeat_a)
        supercell_layout.addWidget(QLabel("b"))
        supercell_layout.addWidget(self._repeat_b)
        supercell_layout.addWidget(QLabel("c"))
        supercell_layout.addWidget(self._repeat_c)
        supercell_layout.addStretch(1)
        layout.addWidget(supercell_group)

        info_panel = QWidget()
        info_layout = QVBoxLayout(info_panel)
        info_layout.addWidget(self._checklist_group)
        info_layout.addWidget(self._preview_status)
        info_layout.addWidget(self._classification)
        info_layout.addWidget(self._component_table, 1)
        q_basis_group = QGroupBox("Propagation-Vector Basis")
        q_basis_layout = QVBoxLayout(q_basis_group)
        q_basis_layout.addWidget(self._q_basis_table)
        q_basis_layout.addWidget(self._q_basis_hint)
        info_layout.addWidget(q_basis_group)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(info_panel)
        splitter.addWidget(self._viewer)
        splitter.setChildrenCollapsible(False)
        splitter.setStretchFactor(0, 5)
        splitter.setStretchFactor(1, 6)
        splitter.setSizes([620, 560])
        layout.addWidget(splitter, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Close
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Use Structure")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        # Enter reaches the dialog after a preview-repeat spin box commits its value; it must not
        # press the default "Use Structure" button and close the builder.
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.accept()
            return
        super().keyPressEvent(event)

    def set_document(self, document: ProjectDocument | None) -> None:
        self._document = document
        self._host_structure = None if document is None else active_host_structure(document)
        self._preview_atom_indices = np.empty((0,), dtype=int)
        self._model = magnetic_model_for_host(
            None if document is None else document.project.magnetic_structure,
            self._host_structure,
        )
        self._populate_components()
        self._sync_enabled_state()
        self._refresh_preview()

    def set_context(
        self,
        host_structure: Structure | None,
        model: MagneticStructureModel | None,
        *,
        selected_indices: tuple[int, ...] = (),
    ) -> None:
        self._document = None
        self._host_structure = host_structure
        self._preview_atom_indices = np.empty((0,), dtype=int)
        self._model = model if model is not None else MagneticStructureModel()
        self._populate_components()
        self._sync_enabled_state()
        self._refresh_preview()
        if selected_indices:
            preview_indices = [
                index
                for index, base_index in enumerate(self._preview_atom_indices)
                if int(base_index) in set(int(value) for value in selected_indices)
            ]
            self._viewer.select_atoms(preview_indices)

    def model(self) -> MagneticStructureModel:
        return self._model

    @staticmethod
    def _repeat_spinbox() -> QSpinBox:
        spinbox = QSpinBox()
        spinbox.setRange(1, 12)
        spinbox.setValue(1)
        spinbox.setKeyboardTracking(False)
        return spinbox

    def _sync_enabled_state(self) -> None:
        has_host = self._host_structure is not None
        self._add_button.setEnabled(has_host)
        self._add_preset_button.setEnabled(has_host)
        self._add_texture_button.setEnabled(has_host)
        self._build_from_host_button.setEnabled(has_host)
        selected_ids = self._selected_component_ids() if has_host else ()
        selected_rows = self._selected_component_rows() if has_host else ()
        row_count = self._component_table.rowCount()
        self._edit_button.setEnabled(has_host and len(selected_ids) == 1)
        self._duplicate_button.setEnabled(has_host and bool(selected_ids))
        self._remove_button.setEnabled(has_host and bool(selected_ids))
        # A selection can move while it is not already packed against that end of the table.
        self._move_up_button.setEnabled(
            any(row != index for index, row in enumerate(selected_rows))
        )
        self._move_down_button.setEnabled(
            any(
                row != row_count - len(selected_rows) + index
                for index, row in enumerate(selected_rows)
            )
        )
        self._component_table.setEnabled(has_host)
        self._q_basis_table.setEnabled(has_host)

    def _configure_component_table_columns(self) -> None:
        header = self._component_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(self._ENABLED_COLUMN, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(self._LABEL_COLUMN, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(4, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.Stretch)

    def _populate_components(self) -> None:
        self._updating = True
        self._component_table.setRowCount(0)
        for component in self._model.components:
            row = self._component_table.rowCount()
            self._component_table.insertRow(row)
            enabled_item = QTableWidgetItem()
            enabled_item.setData(self._COMPONENT_ID_ROLE, component.component_id)
            enabled_item.setFlags(
                Qt.ItemFlag.ItemIsUserCheckable
                | Qt.ItemFlag.ItemIsEnabled
                | Qt.ItemFlag.ItemIsSelectable
            )
            self._component_table.setItem(row, self._ENABLED_COLUMN, enabled_item)
            self._set_item(
                row, self._LABEL_COLUMN, component.label, component.component_id, editable=True
            )
            self._set_item(row, 2, component.kind.replace("_", " ").title())
            self._set_item(row, 3, selector_label(component.selector, self._host_structure))
            self._set_item(row, 4, component_q_label(self._model, component))
            self._set_item(row, 5, component_moment_label(component))
            self._style_component_row(row, component.enabled)
        self._component_table.resizeRowsToContents()
        self._populate_q_basis()
        self._updating = False

    def _populate_q_basis(self) -> None:
        """Show each stored q-basis row, the components that use it, and whether it can be edited here."""

        table = self._q_basis_table
        fill_q_basis_table(table, self._model)
        for slot in q_basis_slots(self._model):
            vector_item = table.item(slot.index, self._Q_BASIS_VECTOR_COLUMN)
            if not slot.texture_component_ids:
                vector_item.setFlags(vector_item.flags() | Qt.ItemFlag.ItemIsEditable)
            if slot.texture_component_ids:
                vector_item.setToolTip(
                    "Used by a texture, whose moment directions follow its q vectors: edit the texture to change it."
                )
            elif slot.direction_component_ids:
                vector_item.setToolTip(
                    "Double-click to change the wavelength: a helix, cycloid or conical component uses this row, "
                    "so only edits that keep its wavevector directions are accepted."
                )
            elif slot.in_use:
                vector_item.setToolTip(
                    "Double-click to edit; every component that uses this row changes wavevector with it."
                )
            else:
                vector_item.setToolTip(
                    "Unused: double-click to edit. Adding a component with a new q fills an unused row."
                )
        self._highlight_selected_q_basis_rows()

    def _refresh_q_basis_panel(self) -> None:
        """Rebuild the q-basis panel after a component change made in place (rename or enable toggle)."""

        updating = self._updating
        self._updating = True
        self._populate_q_basis()
        self._updating = updating

    def _highlight_selected_q_basis_rows(self) -> None:
        """Embolden the q-basis rows that the selected components reference."""

        selected = set(self._selected_component_ids())
        used: set[int] = set()
        for component in self._model.components:
            if component.component_id in selected:
                used.update(component_q_basis_indices(component))
        updating = self._updating
        self._updating = True
        table = self._q_basis_table
        accent = QColor(current_palette().accent)
        for row in range(table.rowCount()):
            highlighted = row in used
            for column in range(table.columnCount()):
                item = table.item(row, column)
                if item is None:
                    continue
                font = item.font()
                font.setBold(highlighted)
                item.setFont(font)
            row_item = table.item(row, 0)
            if row_item is not None:
                if highlighted:
                    row_item.setForeground(accent)
                    row_item.setToolTip("Used by the selected component.")
                else:
                    row_item.setData(Qt.ItemDataRole.ForegroundRole, None)
                    row_item.setToolTip("")
        self._updating = updating

    def _on_q_basis_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating or item.column() != self._Q_BASIS_VECTOR_COLUMN:
            return
        index = item.row()
        try:
            q_rlu = _parse_q_vector_text(item.text())
            if not np.allclose(q_rlu, self._model.q_basis_rlu[index], atol=1e-12, rtol=0.0):
                self._model = set_q_basis_vector(self._model, index, q_rlu)
        except (MagneticStructureError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid propagation vector", str(exc))
        # Update in place: rebuilding the table from inside its own itemChanged signal is unsafe.
        self._updating = True
        item.setText(q_vector_text(self._model.q_basis_rlu[index]))
        for row, component in enumerate(self._model.components):
            q_item = self._component_table.item(row, 4)
            if q_item is not None:
                q_item.setText(component_q_label(self._model, component))
        self._component_table.resizeRowsToContents()
        self._updating = False
        self._refresh_preview()

    def _style_component_row(self, row: int, enabled: bool) -> None:
        """Show the row's enabled state on its checkbox and mute a disabled row's text."""

        enabled_item = self._component_table.item(row, self._ENABLED_COLUMN)
        if enabled_item is not None:
            enabled_item.setCheckState(
                Qt.CheckState.Checked if enabled else Qt.CheckState.Unchecked
            )
            enabled_item.setToolTip(
                "Included in the magnetic structure"
                if enabled
                else "Disabled: excluded from the preview, moments and fields"
            )
        muted = QColor(current_palette().text_muted)
        for column in range(self._LABEL_COLUMN, self._component_table.columnCount()):
            item = self._component_table.item(row, column)
            if item is None:
                continue
            if enabled:
                item.setData(Qt.ItemDataRole.ForegroundRole, None)
            else:
                item.setForeground(muted)

    def _set_item(
        self,
        row: int,
        column: int,
        text: str,
        component_id: str | None = None,
        *,
        editable: bool = False,
    ) -> None:
        item = QTableWidgetItem(text)
        if component_id is not None:
            item.setData(self._COMPONENT_ID_ROLE, component_id)
        if not editable:
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
        self._component_table.setItem(row, column, item)

    def _on_component_item_changed(self, item: QTableWidgetItem) -> None:
        if self._updating or item.column() not in (self._ENABLED_COLUMN, self._LABEL_COLUMN):
            return
        component_id = item.data(self._COMPONENT_ID_ROLE)
        if not isinstance(component_id, str):
            return
        if item.column() == self._ENABLED_COLUMN:
            self._set_component_enabled(
                item.row(), component_id, item.checkState() == Qt.CheckState.Checked
            )
            return
        label = item.text().strip()
        existing = next(
            (
                component
                for component in self._model.components
                if component.component_id == component_id
            ),
            None,
        )
        if existing is None:
            return
        if not label:
            self._updating = True
            item.setText(existing.label)
            self._updating = False
            return
        if label == existing.label:
            return
        components = tuple(
            replace(component, label=label) if component.component_id == component_id else component
            for component in self._model.components
        )
        self._model = replace(self._model, components=components)
        self._refresh_q_basis_panel()
        self._refresh_preview()
        self._sync_enabled_state()

    def _set_component_enabled(self, row: int, component_id: str, enabled: bool) -> None:
        # Restyle the row in place: rebuilding the table from inside its own itemChanged signal is unsafe.
        self._model = set_magnetic_components_enabled(self._model, (component_id,), enabled)
        self._updating = True
        self._style_component_row(row, enabled)
        self._updating = False
        self._refresh_q_basis_panel()
        self._refresh_preview()
        self._sync_enabled_state()

    def _on_component_selection_changed(self) -> None:
        self._highlight_selected_q_basis_rows()
        self._refresh_preview()
        self._sync_enabled_state()

    def _add_component(self) -> None:
        if self._host_structure is None:
            return
        dialog = MagneticComponentDialog(
            self._host_structure,
            selected_indices=self._selected_base_atom_indices(),
            model=self._model,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._append_dialog_component(dialog)

    def _add_preset(self) -> None:
        if self._host_structure is None:
            return
        dialog = MagneticPresetDialog(
            self._host_structure,
            model=self._model,
            selected_indices=self._selected_base_atom_indices(),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._append_dialog_component(dialog)

    def _append_dialog_component(
        self, dialog: MagneticComponentDialog | MagneticPresetDialog
    ) -> None:
        resolved = self._dialog_q_coefficients(dialog)
        if resolved is None:
            return
        q_basis, q_coeff = resolved
        try:
            component = dialog.component(q_coeff)
        except (MagneticStructureError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid magnetic component", str(exc))
            return
        self._set_model(
            replace(
                self._model, q_basis_rlu=q_basis, components=(*self._model.components, component)
            )
        )

    def _dialog_q_coefficients(
        self,
        dialog: MagneticComponentDialog | MagneticPresetDialog,
        *,
        ignored_component_ids: set[str] | None = None,
    ) -> tuple[tuple[tuple[float, float, float], ...], tuple[float, float, float]] | None:
        """Return the q basis and q coefficients for a dialog's q, warning when no basis row is free."""

        coeff = dialog.q_basis_coefficients()
        if coeff is not None:
            return tuple(self._model.q_basis_rlu), coeff
        q_basis, q_coeff = self._q_basis_for(
            dialog.q_vector(), ignored_component_ids=ignored_component_ids
        )
        if q_coeff is None:
            QMessageBox.warning(
                self,
                "No q-basis slot available",
                "This builder can store up to three independent propagation-vector basis directions.",
            )
            return None
        return q_basis, q_coeff

    def _add_texture(self) -> None:
        if self._host_structure is None:
            return
        dialog = MagneticTextureDialog(
            self._host_structure,
            selected_indices=self._selected_base_atom_indices(),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        q_basis, transform = self._q_basis_for_texture(dialog.q_basis_vectors())
        if transform is None:
            QMessageBox.warning(
                self,
                "No q-basis slot available",
                "This texture needs more independent propagation-vector basis directions than are available.",
            )
            return
        component = dialog.component(transform, q_basis)
        self._set_model(
            replace(
                self._model, q_basis_rlu=q_basis, components=(*self._model.components, component)
            )
        )

    def _edit_selected_component(self) -> None:
        if self._host_structure is None:
            return
        selected_ids = self._selected_component_ids()
        if len(selected_ids) != 1:
            return
        component_id = selected_ids[0]
        component = next(
            (
                candidate
                for candidate in self._model.components
                if candidate.component_id == component_id
            ),
            None,
        )
        if component is None:
            return
        if is_texture_component(component):
            self._edit_selected_texture_component(component)
            return
        dialog = MagneticComponentDialog(
            self._host_structure,
            selected_indices=self._selected_base_atom_indices(),
            component=component,
            q_vector=component_propagation_rlu(self._model, component),
            model=self._model,
            q_coeff=component_propagation_coeff(component),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        resolved = self._dialog_q_coefficients(dialog, ignored_component_ids={component_id})
        if resolved is None:
            return
        q_basis, q_coeff = resolved
        updated = replace(dialog.component(q_coeff), enabled=component.enabled)
        components = tuple(
            updated if candidate.component_id == component_id else candidate
            for candidate in self._model.components
        )
        self._set_model(replace(self._model, q_basis_rlu=q_basis, components=components))

    def _edit_selected_texture_component(self, component: MagneticComponent) -> None:
        if self._host_structure is None:
            return
        dialog = MagneticTextureDialog(
            self._host_structure,
            selected_indices=self._selected_base_atom_indices(),
            component=component,
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        q_basis, transform = self._q_basis_for_texture(
            dialog.q_basis_vectors(),
            ignored_component_ids={component.component_id},
        )
        if transform is None:
            QMessageBox.warning(
                self,
                "No q-basis slot available",
                "This texture needs more independent propagation-vector basis directions than are available.",
            )
            return
        updated = _with_kept_atom_phases(
            self._host_structure,
            replace(dialog.component(transform, q_basis), enabled=component.enabled),
            component,
        )
        components = tuple(
            updated if candidate.component_id == component.component_id else candidate
            for candidate in self._model.components
        )
        self._set_model(replace(self._model, q_basis_rlu=q_basis, components=components))

    def _q_basis_for(
        self,
        q_vector: tuple[float, float, float],
        *,
        ignored_component_ids: set[str] | None = None,
    ) -> tuple[tuple[tuple[float, float, float], ...], tuple[float, float, float] | None]:
        return allocate_q_basis_row(
            self._model, q_vector, ignored_component_ids=ignored_component_ids
        )

    def _q_basis_for_texture(
        self,
        q_vectors: Sequence[Sequence[float]],
        *,
        ignored_component_ids: set[str] | None = None,
    ) -> tuple[
        tuple[tuple[float, float, float], ...], tuple[tuple[float, float, float], ...] | None
    ]:
        return allocate_texture_q_basis_rows(
            self._model, q_vectors, ignored_component_ids=ignored_component_ids
        )

    def _build_from_host_spins(self) -> None:
        if self._host_structure is None:
            return
        count = len(self._model.components)
        if count and not workflow_table_utils.confirm_destructive(
            self,
            "Replace magnetic structure",
            f"Replace the {plural(count, 'component', 'components')} "
            "in the builder with components built from the host spins?",
            "The current components are discarded.",
            "Replace Components",
        ):
            return
        model = magnetic_model_from_legacy_structure(self._host_structure)
        self._set_model(model)

    def _remove_selected_components(self) -> None:
        selected_ids = self._selected_component_ids()
        if not selected_ids:
            return
        if not workflow_table_utils.confirm_row_removal(
            self, len(selected_ids), noun=("component", "components")
        ):
            return
        components = tuple(
            component
            for component in self._model.components
            if component.component_id not in selected_ids
        )
        self._set_model(replace(self._model, components=components))

    def _duplicate_selected_components(self) -> None:
        selected_ids = self._selected_component_ids()
        if not selected_ids:
            return
        model, copy_ids = duplicate_magnetic_components(self._model, selected_ids)
        self._set_model(model, selected_ids=copy_ids)

    def _move_selected_components(self, offset: int) -> None:
        selected_ids = self._selected_component_ids()
        if not selected_ids:
            return
        self._set_model(
            move_magnetic_components(self._model, selected_ids, offset), selected_ids=selected_ids
        )

    def _selected_component_rows(self) -> tuple[int, ...]:
        return tuple(
            sorted(index.row() for index in self._component_table.selectionModel().selectedRows())
        )

    def _selected_component_ids(self) -> tuple[str, ...]:
        ids: list[str] = []
        for row in self._selected_component_rows():
            item = self._component_table.item(row, self._LABEL_COLUMN)
            if item is None:
                continue
            component_id = item.data(self._COMPONENT_ID_ROLE)
            if isinstance(component_id, str):
                ids.append(component_id)
        return tuple(ids)

    def _select_components(self, component_ids: Sequence[str]) -> None:
        wanted = set(component_ids)
        table = self._component_table
        selection = QItemSelection()
        for row in range(table.rowCount()):
            item = table.item(row, self._LABEL_COLUMN)
            if item is not None and item.data(self._COMPONENT_ID_ROLE) in wanted:
                selection.select(
                    table.model().index(row, 0), table.model().index(row, table.columnCount() - 1)
                )
        table.selectionModel().select(selection, QItemSelectionModel.SelectionFlag.ClearAndSelect)

    def _set_model(
        self, model: MagneticStructureModel, *, selected_ids: Sequence[str] = ()
    ) -> None:
        self._model = model
        self._populate_components()
        if selected_ids:
            self._updating = True
            self._select_components(selected_ids)
            self._updating = False
        self._sync_enabled_state()
        self._refresh_preview()

    def _supercell_repeats(self) -> tuple[int, int, int]:
        return (
            int(self._repeat_a.value()),
            int(self._repeat_b.value()),
            int(self._repeat_c.value()),
        )

    def _selected_base_atom_indices(self) -> tuple[int, ...]:
        selected: list[int] = []
        for preview_index in self._viewer.selected_atom_indices():
            if 0 <= int(preview_index) < len(self._preview_atom_indices):
                selected.append(int(self._preview_atom_indices[int(preview_index)]))
        return tuple(sorted(set(selected)))

    def _refresh_checklist(self) -> None:
        """Tick off the builder's steps from the current model and flag what is wrong with it."""

        palette = current_palette()
        marks = {
            CHECKLIST_DONE: ("✓", palette.success),
            CHECKLIST_PROBLEM: ("⚠", palette.warning),
        }
        items = magnetic_builder_checklist(
            self._model,
            has_host=self._host_structure is not None,
            preview_repeats=self._supercell_repeats(),
            max_preview_repeat=self._repeat_a.maximum(),
        )
        for item in items:
            label = self._checklist_labels.get(item.key)
            if label is None:
                continue
            mark, colour = marks.get(item.state, ("○", palette.text_muted))
            label.setText(
                f'<span style="color:{colour}; font-weight:600;">{mark}</span>&nbsp; '
                f"<b>{escape(item.title)}</b> "
                f'<span style="color:{palette.text_muted};">{escape(item.detail)}</span>'
            )
            label.setProperty("checklistState", item.state)
            label.setToolTip(item.detail)

    def _refresh_preview(self) -> None:
        self._refresh_checklist()
        if self._updating or self._host_structure is None:
            if self._host_structure is None:
                self._preview_atom_indices = np.empty((0,), dtype=int)
                self._viewer.set_structure(None)
            return
        # One viewer render for the supercell and its moment arrows.
        with self._viewer.batch_update():
            self._show_preview_structure()

    def _show_preview_structure(self) -> None:
        component_ids = self._selected_component_ids() or None
        try:
            repeats = self._supercell_repeats()
            preview_structure, preview_atom_indices = tiled_supercell_structure(
                self._host_structure, repeats
            )
            self._preview_atom_indices = preview_atom_indices
            self._viewer.set_structure(preview_structure)
            if not self._model.components:
                self._viewer.set_moment_vectors(None)
                self._preview_status.setText("Preview: empty magnetic structure")
                self._classification.setText("Structure type: -")
                return
            enabled_ids = {
                component.component_id for component in enabled_magnetic_components(self._model)
            }
            if not enabled_ids:
                self._viewer.set_moment_vectors(None)
                self._preview_status.setText("Preview: every component is disabled")
                self._classification.setText("Structure type: -")
                return
            if component_ids is not None and enabled_ids.isdisjoint(component_ids):
                self._viewer.set_moment_vectors(None)
                self._preview_status.setText("Preview: the selected components are disabled")
            else:
                _positions, moments, _atom_indices = moments_in_supercell(
                    self._host_structure,
                    self._model,
                    repeats,
                    component_ids=component_ids,
                )
                self._viewer.set_moment_vectors(moments)
                mode = (
                    "selected components"
                    if component_ids is not None
                    else "full magnetic structure"
                )
                self._preview_status.setText(
                    f"Preview: {mode} ({repeats[0]}x{repeats[1]}x{repeats[2]})"
                )
            classification = classify_magnetic_model(self._host_structure, self._model)
            self._classification.setText(f"Structure type: {classification.replace('_', ' ')}")
        except ValueError as exc:
            self._viewer.set_moment_vectors(None)
            self._preview_status.setText(f"Preview unavailable: {exc}")
            self._classification.setText("Structure type: -")


def _with_kept_atom_phases(
    structure: Structure, updated: MagneticComponent, original: MagneticComponent
) -> MagneticComponent:
    """Return ``updated`` with ``original``'s atom phases on the atoms it still selects."""

    if not original.atom_phase_turns:
        return updated
    return replace(
        updated,
        atom_phase_turns=atom_phases_for_selection(
            structure, original.atom_phase_turns, updated.selector
        ),
    )


def _parse_q_vector_text(text: str) -> tuple[float, float, float]:
    """Read ``h, k, l`` (commas, spaces and surrounding brackets are all accepted)."""

    parts = text.strip().strip("()[]").replace(",", " ").split()
    if len(parts) != 3:
        raise ValueError(
            "Enter three numbers h, k, l in reciprocal-lattice units, for example 0.25, 0, 0."
        )
    try:
        return (float(parts[0]), float(parts[1]), float(parts[2]))
    except ValueError:
        raise ValueError(f"'{text.strip()}' is not three numbers h, k, l.") from None
