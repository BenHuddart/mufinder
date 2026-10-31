"""Magnetic Structure Builder component dialogs and the helpers they share.

The component, atom-phase and preset dialogs, the component table, the q-basis
table, the direction and decimal inputs, and the labels of atom selectors,
components and moments.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from html import escape

import numpy as np
from PySide6.QtCore import QEvent, QSize, Qt
from PySide6.QtGui import QColor, QKeySequence
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app import workflow_table_utils
from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.app.icons import icon
from mufinder.app.screen_utils import clamped_size
from mufinder.app.theme import (
    current_palette,
    feature_dialog_stylesheet,
    set_option_card_state,
    style_help_button,
    style_hint_text,
    style_inline_group,
    style_option_card,
    style_section_title,
    track_item_view_corners,
)
from mufinder.app.workflow_table_utils import create_workflow_table, format_table_number
from mufinder.domain.models import Structure
from mufinder.domain.text import plural
from mufinder.persistence.codec import float_triplet
from mufinder.science.magnetic_structure import (
    AtomSelector,
    MagneticComponent,
    MagneticStructureError,
    MagneticStructureModel,
    atom_phases_for_selection,
    build_collinear_component,
    build_conical_component,
    build_cycloid_component,
    build_helix_component,
    build_magnetic_preset_component,
    build_spin_density_wave_component,
    conical_component_geometry,
    constrained_secondary_direction,
    default_preset_reference_axis,
    is_texture_component,
    nonzero_unit_direction,
    perpendicular_unit_direction,
    propagation_direction,
    q_basis_combination_rlu,
    q_basis_slots,
    q_coeff_expression,
    reciprocal_axis_length,
    reciprocal_axis_q_rlu,
    reference_direction_perpendicular_to,
    resolve_atom_selector,
    rotation_handedness,
    unit_direction_or,
    unit_q_coeff,
    viable_atom_selectors,
    wavevector_rlu_from_inverse_angstrom,
)

_Q_BASIS_HEADERS = ("Row", "q (r.l.u.)", "Used by")


#: The x, y and z fields of a direction or vector.
DecimalTriple = tuple[DecimalLineEdit, DecimalLineEdit, DecimalLineEdit]


class AtomTargetCardsMixin:
    """The atom-target cards a component or texture dialog shows, and their selector.

    A dialog that mixes this in builds ``_all_atoms_radio``, ``_species_radio`` with
    ``_species_combo``, ``_equivalent_radio`` with ``_equivalent_combo`` and
    ``_selected_radio``, keeps the selected atom indices in ``_selected_indices``, and
    starts ``_selector_rows`` empty before building its cards.
    """

    #: Spacing between a card's button, hint and editor; ``None`` keeps the style's.
    _target_card_spacing: int | None = 6

    def _build_target_card(
        self, button: QRadioButton, hint_text: str, editor: QWidget | None = None
    ) -> QFrame:
        card = QFrame(self)
        style_option_card(card)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        if self._target_card_spacing is not None:
            card_layout.setSpacing(self._target_card_spacing)
        button.setParent(card)
        hint = QLabel(hint_text, card)
        hint.setWordWrap(True)
        style_hint_text(hint)
        card_layout.addWidget(button)
        card_layout.addWidget(hint)
        if editor is not None:
            editor.setParent(card)
            card_layout.addWidget(editor)
        self._selector_rows.append((button, card, hint))
        button.toggled.connect(self._refresh_target_card_styles)
        return card

    def _refresh_target_options(self) -> None:
        self._all_atoms_radio.setEnabled(True)
        self._species_radio.setEnabled(self._species_combo.count() > 0)
        self._equivalent_radio.setEnabled(self._equivalent_combo.count() > 0)
        self._selected_radio.setEnabled(bool(self._selected_indices))
        if not any(
            button.isChecked()
            for button in (
                self._all_atoms_radio,
                self._species_radio,
                self._equivalent_radio,
                self._selected_radio,
            )
        ):
            if self._selected_radio.isEnabled():
                self._selected_radio.setChecked(True)
            elif self._species_radio.isEnabled():
                self._species_radio.setChecked(True)
            else:
                self._all_atoms_radio.setChecked(True)
        self._refresh_target_card_styles()

    def _refresh_target_card_styles(self, *_args) -> None:
        for button, card, hint in self._selector_rows:
            set_option_card_state(
                card, selected=button.isChecked(), unavailable=not button.isEnabled()
            )
            hint.setEnabled(button.isEnabled())

    def _set_selector(self, selector: AtomSelector) -> None:
        if selector.kind == "species" and selector.values:
            index = self._species_combo.findText(str(selector.values[0]))
            if index >= 0:
                self._species_combo.setCurrentIndex(index)
                self._species_radio.setChecked(True)
                return
        if selector.kind == "symmetry_equivalent":
            for index in range(self._equivalent_combo.count()):
                candidate = self._equivalent_combo.itemData(index)
                if candidate == selector:
                    self._equivalent_combo.setCurrentIndex(index)
                    self._equivalent_radio.setChecked(True)
                    return
        if selector.kind == "indices" and selector.values:
            self._selected_radio.setChecked(True)
            return
        self._all_atoms_radio.setChecked(True)


class MagneticComponentDialog(AtomTargetCardsMixin, QDialog):
    """Dialog for creating a propagation-vector magnetic component."""

    _Q_SOURCE_VECTOR = "vector"
    _Q_SOURCE_COMBINATION = "combination"
    _TEXTURE_KINDS = {
        "Collinear": "collinear",
        "Spin-density wave": "spin_density_wave",
        "Helix": "helix",
        "Cycloid": "cycloid",
        "Conical": "conical",
    }

    def __init__(
        self,
        structure: Structure,
        *,
        selected_indices: tuple[int, ...] = (),
        component: MagneticComponent | None = None,
        q_vector: tuple[float, float, float] | None = None,
        model: MagneticStructureModel | None = None,
        q_coeff: Sequence[float] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(
            "Edit Magnetic Component" if component is not None else "Add Magnetic Component"
        )
        self.resize(clamped_size(self, 760, 820))
        self._structure = structure
        self._model = model if model is not None else MagneticStructureModel()
        if component is not None and component.selector.kind == "indices":
            selected_indices = (
                *selected_indices,
                *(int(value) for value in component.selector.values),
            )
        self._selected_indices = tuple(
            sorted(
                {
                    int(index)
                    for index in selected_indices
                    if 0 <= int(index) < len(structure.species)
                }
            )
        )
        self._editing_component_id = component.component_id if component is not None else None
        self._selector_rows: list[tuple[QRadioButton, QFrame, QLabel]] = []
        self._syncing_constrained_inputs = False
        self.setStyleSheet(feature_dialog_stylesheet(emphasised_radio_buttons=True))

        self._label_edit = QLineEdit("Magnetic component")
        self._label_edit.setMinimumWidth(420)
        self._kind_combo = QComboBox()
        self._kind_combo.addItems(list(self._TEXTURE_KINDS))
        self._component_info_button = QToolButton()
        style_help_button(self._component_info_button)
        self._component_info_button.setIcon(icon("help"))
        self._component_info_button.setIconSize(QSize(22, 22))
        self._component_info_button.setToolTip(
            "Explain this component in the propagation-vector formalism."
        )
        self._component_info_button.clicked.connect(self._show_component_info)

        self._amplitude = DecimalLineEdit(
            standard_notation=True, text="1.0", minimum=0.0, maximum=1000.0, placeholder="1.0"
        )
        self._cone_angle = DecimalLineEdit(
            standard_notation=True, text="45.0", minimum=0.0, maximum=180.0, placeholder="45"
        )

        self._primary_direction = direction_inputs((0.0, 0.0, 1.0))
        self._secondary_direction = direction_inputs((1.0, 0.0, 0.0))
        self._handedness_combo = QComboBox()
        self._handedness_combo.addItem("Right-handed", 1)
        self._handedness_combo.addItem("Left-handed", -1)
        self._constraint_hint = QLabel("")
        self._constraint_hint.setWordWrap(True)
        style_hint_text(self._constraint_hint)

        self._target_group = QButtonGroup(self)
        self._all_atoms_radio = QRadioButton("All atoms")
        self._species_radio = QRadioButton("Element")
        self._equivalent_radio = QRadioButton("Equivalent site")
        self._selected_radio = QRadioButton("Current selection")
        self._target_group.addButton(self._all_atoms_radio)
        self._target_group.addButton(self._species_radio)
        self._target_group.addButton(self._equivalent_radio)
        self._target_group.addButton(self._selected_radio)

        self._species_combo = QComboBox()
        for species in sorted(set(str(symbol) for symbol in structure.species)):
            self._species_combo.addItem(species)

        self._equivalent_combo = QComboBox()
        for selector in viable_atom_selectors(structure):
            if selector.kind == "symmetry_equivalent" and selector.values:
                self._equivalent_combo.addItem(selector_label(selector, structure), selector)

        self._q_x = DecimalLineEdit(
            standard_notation=True, text="0.0", minimum=-4.0, maximum=4.0, placeholder="qx"
        )
        self._q_y = DecimalLineEdit(
            standard_notation=True, text="0.0", minimum=-4.0, maximum=4.0, placeholder="qy"
        )
        self._q_z = DecimalLineEdit(
            standard_notation=True, text="0.0", minimum=-4.0, maximum=4.0, placeholder="qz"
        )
        self._q_source_combo = QComboBox()
        self._q_source_combo.addItem("Enter a vector (r.l.u.)", self._Q_SOURCE_VECTOR)
        stored_rows = [
            slot
            for slot in q_basis_slots(self._model)
            if not np.allclose(slot.q_rlu, 0.0, atol=1e-12)
        ]
        for slot in stored_rows:
            self._q_source_combo.addItem(
                f"{slot.label} = ({q_vector_text(slot.q_rlu)})", slot.index
            )
        if stored_rows:
            self._q_source_combo.addItem("Combination of basis rows", self._Q_SOURCE_COMBINATION)
        else:
            self._q_source_combo.setToolTip(
                "No propagation vector is stored yet; enter q as a vector."
            )
        self._q_coeff_inputs = tuple(
            DecimalLineEdit(
                standard_notation=True,
                text="0",
                minimum=-4.0,
                maximum=4.0,
                placeholder=f"n{index + 1}",
            )
            for index in range(3)
        )
        for index, edit in enumerate(self._q_coeff_inputs):
            stored = any(slot.index == index for slot in stored_rows)
            edit.setEnabled(stored)
            edit.setToolTip(
                f"Multiple of q{index + 1}"
                if stored
                else f"q{index + 1} holds no propagation vector"
            )
        self._q_expression = QLabel("")
        self._q_expression.setWordWrap(True)
        style_hint_text(self._q_expression)
        self._q_basis_table = create_q_basis_table(editable=False)
        fill_q_basis_table(self._q_basis_table, self._model)
        self._q_basis_table.resizeRowsToContents()
        self._q_basis_table.setFixedHeight(
            self._q_basis_table.horizontalHeader().sizeHint().height()
            + sum(self._q_basis_table.rowHeight(row) for row in range(3))
            + 2 * self._q_basis_table.frameWidth()
        )

        self._phase = DecimalLineEdit(
            standard_notation=True, text="0.0", minimum=-10.0, maximum=10.0, placeholder="0.0"
        )
        self._atom_phase_turns: dict[int, float] = (
            dict(component.atom_phase_turns) if component is not None else {}
        )
        self._atom_phase_summary = QLabel("")
        self._atom_phase_summary.setWordWrap(True)
        self._atom_phase_summary.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding
        )
        style_hint_text(self._atom_phase_summary)
        self._atom_phase_button = QPushButton("Per-Atom Phases…")
        self._atom_phase_button.setToolTip(
            "Give single atoms an extra phase in turns, added to the component's phase."
        )
        self._atom_phase_button.clicked.connect(self._edit_atom_phases)

        self._build_layout()
        self._refresh_target_options()
        for button, _card, _hint in self._selector_rows:
            button.toggled.connect(lambda *_: self._refresh_atom_phase_summary())
        for combo in (self._species_combo, self._equivalent_combo):
            combo.currentIndexChanged.connect(lambda *_: self._refresh_atom_phase_summary())
        self._q_source_combo.currentIndexChanged.connect(lambda *_: self._sync_q_from_basis())
        for edit in self._q_coeff_inputs:
            edit.textChanged.connect(lambda *_: self._sync_q_from_basis())
        self._kind_combo.currentTextChanged.connect(self._refresh_texture_controls)
        self._handedness_combo.currentIndexChanged.connect(
            lambda *_: self._sync_constrained_direction_inputs()
        )
        for edit in (*self._primary_direction, self._q_x, self._q_y, self._q_z):
            edit.textChanged.connect(lambda *_: self._sync_constrained_direction_inputs())
        if component is not None:
            self._load_component(component, q_vector or (0.0, 0.0, 0.0))
            self._select_q_source(q_coeff)
        self._sync_q_from_basis()
        self._refresh_texture_controls()
        self._refresh_atom_phase_summary()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(14)

        title = QLabel("Add Magnetic Component", self)
        style_section_title(title)
        layout.addWidget(title)

        form_group = QGroupBox("Component")
        form = QFormLayout(form_group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow("Label", self._label_edit)
        component_type_group = QWidget()
        style_inline_group(component_type_group)
        component_type_layout = QHBoxLayout(component_type_group)
        component_type_layout.setContentsMargins(0, 0, 0, 0)
        component_type_layout.addWidget(self._kind_combo, 1)
        component_type_layout.addWidget(self._component_info_button)
        form.addRow("Component type", component_type_group)
        amplitude_group = QWidget()
        style_inline_group(amplitude_group)
        amplitude_layout = QHBoxLayout(amplitude_group)
        amplitude_layout.setContentsMargins(0, 0, 0, 0)
        amplitude_layout.addWidget(self._amplitude)
        bohr_label = QLabel("μ<sub>B</sub>")
        bohr_label.setTextFormat(Qt.TextFormat.RichText)
        amplitude_layout.addWidget(bohr_label)
        form.addRow("Amplitude", amplitude_group)
        self._primary_direction_row_label = QLabel("Direction (x, y, z)")
        self._primary_direction_group = direction_group(self._primary_direction)
        form.addRow(self._primary_direction_row_label, self._primary_direction_group)
        self._secondary_direction_row_label = QLabel("Derived direction (x, y, z)")
        self._secondary_direction_group = direction_group(self._secondary_direction)
        form.addRow(self._secondary_direction_row_label, self._secondary_direction_group)
        self._handedness_row_label = QLabel("Handedness")
        form.addRow(self._handedness_row_label, self._handedness_combo)
        form.addRow("", self._constraint_hint)
        self._cone_angle_row_label = QLabel("Cone angle (degrees)")
        form.addRow(self._cone_angle_row_label, self._cone_angle)
        q_group = QWidget()
        style_inline_group(q_group)
        q_layout = QHBoxLayout(q_group)
        q_layout.setContentsMargins(0, 0, 0, 0)
        q_layout.addWidget(self._q_x)
        q_layout.addWidget(self._q_y)
        q_layout.addWidget(self._q_z)
        form.addRow("q from", self._q_source_combo)
        coeff_group = QWidget()
        style_inline_group(coeff_group)
        coeff_layout = QHBoxLayout(coeff_group)
        coeff_layout.setContentsMargins(0, 0, 0, 0)
        for index, edit in enumerate(self._q_coeff_inputs):
            coeff_layout.addWidget(edit)
            row_label = QLabel(f"q{index + 1}" + (" +" if index < 2 else ""))
            coeff_layout.addWidget(row_label)
        self._q_coeff_row_label = QLabel("Basis coefficients")
        self._q_coeff_group = coeff_group
        form.addRow(self._q_coeff_row_label, coeff_group)
        form.addRow("q (r.l.u.)", q_group)
        form.addRow("", self._q_expression)
        form.addRow("Phase (turns)", self._phase)
        atom_phase_group = QWidget()
        style_inline_group(atom_phase_group)
        atom_phase_layout = QHBoxLayout(atom_phase_group)
        atom_phase_layout.setContentsMargins(0, 0, 0, 0)
        atom_phase_layout.addWidget(self._atom_phase_summary, 1)
        atom_phase_layout.addWidget(self._atom_phase_button)
        form.addRow("Per-atom phases", atom_phase_group)
        layout.addWidget(form_group)

        q_basis_group = QGroupBox("Propagation-Vector Basis")
        q_basis_layout = QVBoxLayout(q_basis_group)
        q_basis_layout.addWidget(self._q_basis_table)
        layout.addWidget(q_basis_group)

        target_group = QGroupBox("Apply Component To")
        target_layout = QGridLayout(target_group)
        target_layout.setContentsMargins(14, 12, 14, 12)
        target_layout.setHorizontalSpacing(10)
        target_layout.setVerticalSpacing(10)
        target_layout.addWidget(
            self._build_target_card(
                self._all_atoms_radio,
                "Apply this component to every atom in the host structure.",
            ),
            0,
            0,
        )
        target_layout.addWidget(
            self._build_target_card(
                self._species_radio,
                "Apply to all atoms of one chemical species.",
                self._species_combo,
            ),
            0,
            1,
        )
        target_layout.addWidget(
            self._build_target_card(
                self._equivalent_radio,
                "Apply to the atoms in one symmetry-equivalent site.",
                self._equivalent_combo,
            ),
            1,
            0,
        )
        target_layout.addWidget(
            self._build_target_card(
                self._selected_radio,
                "Apply to the atoms selected in the preview.",
            ),
            1,
            1,
        )
        target_layout.setColumnStretch(0, 1)
        target_layout.setColumnStretch(1, 1)
        layout.addWidget(target_group)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _refresh_texture_controls(self) -> None:
        kind = self._TEXTURE_KINDS.get(self._kind_combo.currentText(), "collinear")
        rotating = kind in {"helix", "cycloid", "conical"}
        self._secondary_direction_group.setVisible(rotating)
        self._secondary_direction_row_label.setVisible(rotating)
        self._handedness_combo.setVisible(rotating)
        self._handedness_row_label.setVisible(rotating)
        self._constraint_hint.setVisible(rotating)
        for edit in self._secondary_direction:
            edit.setEnabled(rotating)
            edit.setReadOnly(rotating)
        if kind == "helix":
            self._primary_direction_row_label.setText("Reference direction in plane (x, y, z)")
            self._secondary_direction_row_label.setText("Derived transverse direction (x, y, z)")
            self._constraint_hint.setText(
                "The rotation plane is perpendicular to q. The reference direction is projected into that plane; "
                "the second axis is then fixed by q and the selected handedness."
            )
        elif kind == "cycloid":
            self._primary_direction_row_label.setText("Transverse direction (x, y, z)")
            self._secondary_direction_row_label.setText("Propagation-axis direction (x, y, z)")
            self._constraint_hint.setText(
                "The cycloid plane contains q. The transverse direction is projected perpendicular to q; "
                "the second axis is the propagation direction, with sign set by handedness."
            )
        elif kind == "conical":
            self._primary_direction_row_label.setText("Reference direction around cone (x, y, z)")
            self._secondary_direction_row_label.setText("Derived transverse direction (x, y, z)")
            self._constraint_hint.setText(
                "The cone axis is fixed parallel to q. The rotating axes lie in the plane perpendicular to q "
                "and are completed from the reference direction and handedness."
            )
        else:
            self._primary_direction_row_label.setText("Direction (x, y, z)")
        conical = kind == "conical"
        self._cone_angle.setVisible(conical)
        self._cone_angle_row_label.setVisible(conical)
        self._sync_constrained_direction_inputs()

    def _load_component(
        self, component: MagneticComponent, q_vector: tuple[float, float, float]
    ) -> None:
        self._label_edit.setText(component.label)
        kind_label = next(
            (label for label, kind in self._TEXTURE_KINDS.items() if kind == component.kind),
            "Collinear",
        )
        self._kind_combo.setCurrentText(kind_label)
        self._set_selector(component.selector)
        for edit, value in zip((self._q_x, self._q_y, self._q_z), q_vector, strict=True):
            edit.setText(f"{float(value):.8g}")
        self._phase.setText(f"{float(component.phase_turns):.8g}")
        self._load_component_vectors(component)

    def _load_component_vectors(self, component: MagneticComponent) -> None:
        if not component.terms:
            return
        if component.kind == "conical" and len(component.terms) >= 2:
            amplitude, cone_angle, real_direction, imag_direction, axis = (
                conical_component_geometry(
                    component, self._q_cartesian_direction(allow_fallback=True)
                )
            )
            self._amplitude.setText(f"{amplitude:.8g}")
            self._cone_angle.setText(f"{cone_angle:.8g}")
            set_direction_inputs(self._primary_direction, real_direction)
            self._set_handedness_from_axes(real_direction, imag_direction, axis)
            self._sync_constrained_direction_inputs()
            return
        coefficient = component.terms[0].coefficient
        real = np.asarray(coefficient.real, dtype=float)
        imag = np.asarray(coefficient.imaginary, dtype=float)
        real_norm = float(np.linalg.norm(real))
        imag_norm = float(np.linalg.norm(imag))
        if component.kind in {"helix", "cycloid"}:
            self._amplitude.setText(f"{(0.5 * (real_norm + imag_norm)):.8g}")
            set_direction_inputs(self._primary_direction, unit_direction_or(real, (0.0, 0.0, 1.0)))
            q_direction = self._q_cartesian_direction(allow_fallback=True)
            if q_direction is not None:
                if component.kind == "cycloid":
                    self._set_handedness(1 if float(np.dot(imag, q_direction)) >= 0.0 else -1)
                else:
                    self._set_handedness_from_axes(
                        np.asarray(unit_direction_or(real, (0.0, 0.0, 1.0)), dtype=float),
                        np.asarray(unit_direction_or(imag, (1.0, 0.0, 0.0)), dtype=float),
                        q_direction,
                    )
            self._sync_constrained_direction_inputs()
            return
        self._amplitude.setText(f"{real_norm:.8g}")
        set_direction_inputs(self._primary_direction, unit_direction_or(real, (0.0, 0.0, 1.0)))
        set_direction_inputs(self._secondary_direction, (1.0, 0.0, 0.0))

    def _set_handedness(self, sign: int) -> None:
        target = 1 if int(sign) >= 0 else -1
        index = self._handedness_combo.findData(target)
        if index >= 0:
            self._handedness_combo.setCurrentIndex(index)

    def _handedness_sign(self) -> int:
        value = self._handedness_combo.currentData()
        try:
            return 1 if int(value) >= 0 else -1
        except (TypeError, ValueError):
            return 1

    def _set_handedness_from_axes(
        self,
        primary: Sequence[float],
        secondary: Sequence[float],
        q_direction: Sequence[float],
    ) -> None:
        self._set_handedness(rotation_handedness(primary, secondary, q_direction))

    def _q_cartesian_direction(self, *, allow_fallback: bool = False) -> np.ndarray | None:
        try:
            q_rlu = self.q_vector()
        except ValueError:
            return None
        return propagation_direction(
            self._structure.lattice_vectors, q_rlu, allow_fallback=allow_fallback
        )

    def _projected_reference_direction(self, q_direction: np.ndarray) -> np.ndarray:
        try:
            raw = np.asarray(
                [
                    decimal_value(edit, "reference direction component")
                    for edit in self._primary_direction
                ],
                dtype=float,
            )
        except ValueError:
            return perpendicular_unit_direction(q_direction)
        return reference_direction_perpendicular_to(raw, q_direction)

    def _constrained_component_directions(
        self, kind: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q_direction = self._q_cartesian_direction()
        if q_direction is None:
            raise ValueError("q must be valid for helical, cycloidal, and conical components")
        primary = self._projected_reference_direction(q_direction)
        secondary = constrained_secondary_direction(
            kind, q_direction, primary, float(self._handedness_sign())
        )
        return primary, secondary, q_direction

    def _sync_constrained_direction_inputs(self) -> None:
        if self._syncing_constrained_inputs:
            return
        kind = self._TEXTURE_KINDS.get(self._kind_combo.currentText(), "collinear")
        if kind not in {"helix", "cycloid", "conical"}:
            return
        try:
            primary, secondary, _q_direction = self._constrained_component_directions(kind)
        except ValueError:
            return
        self._syncing_constrained_inputs = True
        try:
            set_direction_inputs(self._primary_direction, primary)
            set_direction_inputs(self._secondary_direction, secondary)
        finally:
            self._syncing_constrained_inputs = False

    def _show_component_info(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Component Representation")
        dialog.resize(720, 620)

        layout = QVBoxLayout(dialog)
        title = QLabel("Propagation-Vector Representation", dialog)
        style_section_title(title)
        layout.addWidget(title)

        browser = QTextBrowser(dialog)
        browser.setOpenExternalLinks(False)
        browser.setHtml(self._component_help_html())
        layout.addWidget(browser, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dialog)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def _component_help_html(self) -> str:
        kind_label = self._kind_combo.currentText()
        kind = self._TEXTURE_KINDS.get(kind_label, "collinear")
        amplitude = escape(self._amplitude.text().strip() or "A")
        primary = escape(self._direction_display(self._primary_direction))
        secondary = escape(self._direction_display(self._secondary_direction))
        q = tuple(
            escape(field.text().strip() or "0") for field in (self._q_x, self._q_y, self._q_z)
        )
        phase = escape(self._phase.text().strip() or "0")
        selector = escape(selector_label(self.selector(), self._structure))

        if kind == "spin_density_wave":
            coefficient = f"<b>S</b> = {amplitude} μ<sub>B</sub> <b>u</b><sub>1</sub>, where <b>u</b><sub>1</sub> = {primary}"
            real_space = (
                f"<b>m</b>(<b>r</b>) = {amplitude} μ<sub>B</sub> "
                f"cos[2π(<b>q</b>·<b>r</b> + {phase})] <b>u</b><sub>1</sub>"
            )
            example = (
                f"If <b>q</b> = (0, 0, 1/2), neighbouring cells along c alternate between "
                f"+{amplitude} μ<sub>B</sub> and -{amplitude} μ<sub>B</sub> along {primary}."
            )
        elif kind == "helix":
            coefficient = (
                f"<b>S</b> = {amplitude} μ<sub>B</sub> "
                f"(<b>u</b><sub>1</sub> + i <b>u</b><sub>2</sub>), where "
                f"<b>u</b><sub>1</sub> = {primary} and <b>u</b><sub>2</sub> = {secondary}"
            )
            real_space = (
                f"<b>m</b>(<b>r</b>) = {amplitude} μ<sub>B</sub> "
                f"[cos θ <b>u</b><sub>1</sub> + sin θ <b>u</b><sub>2</sub>]"
            )
            example = (
                "The spin keeps approximately constant size and rotates in the plane perpendicular to "
                "<b>q</b>. MuFinder projects the reference direction into this plane, then derives "
                "the second axis from <b>q</b> and the selected handedness."
            )
        elif kind == "cycloid":
            coefficient = (
                f"<b>S</b> = {amplitude} μ<sub>B</sub> "
                f"(<b>u</b><sub>1</sub> + i <b>u</b><sub>2</sub>), where "
                f"<b>u</b><sub>1</sub> = {primary} and <b>u</b><sub>2</sub> = {secondary}"
            )
            real_space = (
                f"<b>m</b>(<b>r</b>) = {amplitude} μ<sub>B</sub> "
                f"[cos θ <b>u</b><sub>1</sub> + sin θ <b>u</b><sub>2</sub>]"
            )
            example = (
                "The cycloidal rotation plane contains <b>q</b>. MuFinder projects the reference "
                "direction perpendicular to <b>q</b> and uses the propagation direction as the second "
                "axis, with sign set by handedness."
            )
        elif kind == "conical":
            cone_angle = escape(self._cone_angle.text().strip() or "α")
            coefficient = (
                f"<b>S</b><sub>0</sub> = {amplitude} cos({cone_angle}°) μ<sub>B</sub> "
                f"<b>u</b><sub>3</sub>, and <b>S</b><sub>q</sub> = {amplitude} sin({cone_angle}°) "
                f"μ<sub>B</sub> (<b>u</b><sub>1</sub> + i <b>u</b><sub>2</sub>)"
            )
            real_space = (
                f"<b>m</b>(<b>r</b>) = {amplitude} cos({cone_angle}°) μ<sub>B</sub> <b>u</b><sub>3</sub> + "
                f"{amplitude} sin({cone_angle}°) μ<sub>B</sub> "
                f"[cos θ <b>u</b><sub>1</sub> + sin θ <b>u</b><sub>2</sub>]"
            )
            example = (
                "This is stored as two Fourier terms in one component: a uniform q=0 moment along "
                "the cone axis and a finite-q helix in the perpendicular plane. MuFinder fixes the "
                "cone axis parallel to <b>q</b> and derives the second transverse axis from handedness."
            )
        else:
            coefficient = f"<b>S</b> = {amplitude} μ<sub>B</sub> <b>u</b><sub>1</sub>, where <b>u</b><sub>1</sub> = {primary}"
            real_space = (
                f"<b>m</b>(<b>r</b>) = {amplitude} μ<sub>B</sub> "
                f"cos[2π(<b>q</b>·<b>r</b> + {phase})] <b>u</b><sub>1</sub>"
            )
            example = (
                f"For <b>q</b> = (0, 0, 0), every selected atom has the same moment: "
                f"{amplitude} μ<sub>B</sub> along {primary}, shifted only by the phase."
            )

        return f"""
        <html>
        <body style="font-size: 13px;">
          <h2>{escape(kind_label)}</h2>
          <p>
            This component is applied to <b>{selector}</b>. The propagation vector entered in the dialog is
          </p>
          <p style="text-align: center; font-size: 15px;">
            <b>q</b> = ({q[0]}, {q[1]}, {q[2]}) in reciprocal lattice units.
          </p>
          <p>
            MuFinder stores the component as a complex Fourier coefficient. For a selected atom at fractional
            position <b>r</b>, and for translated copies in a preview supercell, the displayed moment is
          </p>
          <p style="text-align: center; font-size: 15px;">
            <b>m</b>(<b>r</b>) = Re[ <b>S</b> exp(-2πi(<b>q</b>·<b>r</b> + φ)) ].
          </p>
          <p>
            The phase parameter is φ = <b>{phase}</b> turns. One turn is a full period, so 0.25 is a quarter-period
            shift and 0.5 reverses a cosine component.
          </p>
          <p>
            Direction vectors are entered as Cartesian x, y, z components and are normalized before the amplitude
            is applied. For helical, cycloidal, and conical presets, MuFinder constrains these directions using
            the propagation-vector geometry before storing the Cartesian moment vector in μ<sub>B</sub>.
          </p>
          <h3>Coefficient Stored For This Preset</h3>
          <p style="text-align: center; font-size: 15px;">{coefficient}</p>
          <h3>Equivalent Real-Space Form</h3>
          <p style="text-align: center; font-size: 15px;">{real_space}</p>
          <p>
            Here θ = -2π(<b>q</b>·<b>r</b> + {phase}). The sign convention matches the complex exponential used by
            the internal model and by the field-calculation export path.
          </p>
          <h3>Example</h3>
          <p>{example}</p>
        </body>
        </html>
        """

    def _q_source(self) -> object:
        return self._q_source_combo.currentData()

    def _select_q_source(self, q_coeff: Sequence[float] | None) -> None:
        """Show an edited component's q as the basis row or combination it is stored as."""

        if q_coeff is None or self._q_source_combo.count() < 2:
            return
        coeff = tuple(float(value) for value in tuple(q_coeff)[:3])
        used = [index for index, value in enumerate(coeff) if abs(value) > 1e-12]
        if not used:
            return
        if len(used) == 1 and abs(coeff[used[0]] - 1.0) <= 1e-12:
            row = self._q_source_combo.findData(used[0])
            if row >= 0:
                self._q_source_combo.setCurrentIndex(row)
                return
        row = self._q_source_combo.findData(self._Q_SOURCE_COMBINATION)
        if row < 0:
            return
        for edit, value in zip(self._q_coeff_inputs, coeff, strict=True):
            edit.setText(f"{value:.8g}")
        self._q_source_combo.setCurrentIndex(row)

    def q_basis_coefficients(self) -> tuple[float, float, float] | None:
        """Return the chosen combination of q-basis rows, or ``None`` when q was entered as a vector."""

        source = self._q_source()
        if isinstance(source, int):
            return unit_q_coeff(source)
        if source == self._Q_SOURCE_COMBINATION:
            values = tuple(
                decimal_value(edit, f"q{index + 1} coefficient") if edit.isEnabled() else 0.0
                for index, edit in enumerate(self._q_coeff_inputs)
            )
            return (values[0], values[1], values[2])
        return None

    def _sync_q_from_basis(self) -> None:
        """Fill the r.l.u. fields from the chosen basis row or combination, and lock them while one is chosen."""

        source = self._q_source()
        from_basis = source != self._Q_SOURCE_VECTOR
        combination = source == self._Q_SOURCE_COMBINATION
        self._q_coeff_group.setVisible(combination)
        self._q_coeff_row_label.setVisible(combination)
        for edit in (self._q_x, self._q_y, self._q_z):
            edit.setReadOnly(from_basis)
        if not from_basis:
            self._q_expression.setText(
                "A new q fills an unused basis row; a q already in the basis reuses its row."
            )
            return
        try:
            coeff = self.q_basis_coefficients()
            q_rlu = q_basis_combination_rlu(self._model, coeff or (0.0, 0.0, 0.0))
        except (MagneticStructureError, ValueError) as exc:
            self._q_expression.setText(str(exc))
            return
        for edit, value in zip((self._q_x, self._q_y, self._q_z), q_rlu, strict=True):
            text = f"{float(value):.8g}"
            if edit.text() != text:
                edit.setText(text)
        self._q_expression.setText(
            f"q = {q_coeff_expression(coeff or (0.0, 0.0, 0.0))} = ({q_vector_text(q_rlu)}); "
            "the component follows these rows if they are edited."
        )

    def q_vector(self) -> tuple[float, float, float]:
        return (
            decimal_value(self._q_x, "q x"),
            decimal_value(self._q_y, "q y"),
            decimal_value(self._q_z, "q z"),
        )

    def _primary_unit_direction(self) -> np.ndarray:
        return unit_direction(self._primary_direction, "first direction")

    def _secondary_unit_direction(self) -> np.ndarray:
        return unit_direction(self._secondary_direction, "second direction")

    def selector(self) -> AtomSelector:
        if self._species_radio.isChecked():
            return AtomSelector.species(self._species_combo.currentText())
        if self._equivalent_radio.isChecked():
            selector = self._equivalent_combo.currentData()
            if isinstance(selector, AtomSelector):
                return selector
        if self._selected_radio.isChecked() and self._selected_indices:
            return AtomSelector.indices(self._selected_indices)
        return AtomSelector.all()

    def _target_atom_phases(self) -> dict[int, float]:
        """Return the per-atom phases on the atoms the current target selects."""

        return atom_phases_for_selection(self._structure, self._atom_phase_turns, self.selector())

    def _refresh_atom_phase_summary(self) -> None:
        phases = self._target_atom_phases()
        self._atom_phase_summary.setText(_atom_phase_text(phases) if phases else "None")

    def _edit_atom_phases(self) -> None:
        targeted = resolve_atom_selector(self._structure, self.selector())
        dialog = AtomPhaseDialog(self._structure, targeted, self._target_atom_phases(), parent=self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        targeted_set = set(targeted)
        kept = {
            index: phase
            for index, phase in self._atom_phase_turns.items()
            if index not in targeted_set
        }
        self._atom_phase_turns = {**kept, **dialog.atom_phase_turns()}
        self._refresh_atom_phase_summary()

    def component(self, q_coeff: tuple[float, float, float]) -> MagneticComponent:
        return replace(
            self._component_without_atom_phases(q_coeff),
            atom_phase_turns=self._target_atom_phases(),
        )

    def _component_without_atom_phases(
        self, q_coeff: tuple[float, float, float]
    ) -> MagneticComponent:
        amplitude = decimal_value(self._amplitude, "amplitude")
        label = self._label_edit.text().strip() or "Magnetic component"
        selector = self.selector()
        kind = self._TEXTURE_KINDS.get(self._kind_combo.currentText(), "collinear")
        if kind in {"helix", "cycloid", "conical"}:
            primary_direction, secondary_direction, propagation_direction = (
                self._constrained_component_directions(kind)
            )
            set_direction_inputs(self._primary_direction, primary_direction)
            set_direction_inputs(self._secondary_direction, secondary_direction)
        else:
            primary_direction = self._primary_unit_direction()
            secondary_direction = self._secondary_unit_direction()
            propagation_direction = np.cross(primary_direction, secondary_direction)
        primary = amplitude * primary_direction
        secondary = amplitude * secondary_direction
        kwargs = {
            "label": label,
            "selector": selector,
            "q_coeff": q_coeff,
            "phase_turns": decimal_value(self._phase, "phase"),
            "component_id": self._editing_component_id,
        }
        if kind == "spin_density_wave":
            return build_spin_density_wave_component(amplitude_vector=primary, **kwargs)
        if kind == "helix":
            return build_helix_component(
                real_axis_vector=primary, imaginary_axis_vector=secondary, **kwargs
            )
        if kind == "cycloid":
            return build_cycloid_component(
                real_axis_vector=primary, imaginary_axis_vector=secondary, **kwargs
            )
        if kind == "conical":
            return build_conical_component(
                total_moment=amplitude,
                real_axis_direction=primary_direction,
                imaginary_axis_direction=secondary_direction,
                static_axis_direction=propagation_direction,
                cone_angle_degrees=decimal_value(self._cone_angle, "cone angle"),
                **kwargs,
            )
        return build_collinear_component(moment=primary, **kwargs)

    @staticmethod
    def _direction_display(inputs: DecimalTriple) -> str:
        return "(" + ", ".join(edit.text().strip() or "0" for edit in inputs) + ")"

    def accept(self) -> None:  # noqa: D401 - Qt API override.
        """Accept the dialog only when decimal fields are valid."""

        try:
            decimal_value(self._amplitude, "amplitude")
            coeff = self.q_basis_coefficients()
            if coeff is not None:
                q_basis_combination_rlu(self._model, coeff)
            self.q_vector()
            decimal_value(self._phase, "phase")
            kind = self._TEXTURE_KINDS.get(self._kind_combo.currentText(), "collinear")
            if kind in {"helix", "cycloid", "conical"}:
                self._constrained_component_directions(kind)
            else:
                self._primary_unit_direction()
            if kind == "conical":
                decimal_value(self._cone_angle, "cone angle")
        except (MagneticStructureError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid magnetic component", str(exc))
            return
        super().accept()


class AtomPhaseDialog(QDialog):
    """Table of the atoms a component targets, each with an editable extra phase in turns."""

    _PHASE_COLUMN = 3
    _PHASE_LIMIT = 10.0

    def __init__(
        self,
        structure: Structure,
        atom_indices: Sequence[int],
        phases: Mapping[int, float],
        *,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Per-Atom Phases")
        self.resize(clamped_size(self, 620, 460))
        self.setStyleSheet(feature_dialog_stylesheet())
        self._atom_indices = tuple(int(index) for index in atom_indices)

        self._table = create_workflow_table(
            ("Atom", "Species", "Position (fractional)", "Extra phase (turns)"),
            resize_to_contents_columns=(0, 1),
            editable=True,
            selection_behavior=QAbstractItemView.SelectionBehavior.SelectItems,
            accessible_name="Per-Atom Phases",
        )
        self._table.setRowCount(len(self._atom_indices))
        for row, index in enumerate(self._atom_indices):
            labels = (
                str(index + 1),
                str(structure.species[index]),
                _fractional_position_label(structure.fractional_positions[index]),
            )
            for column, text in enumerate(labels):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(row, column, item)
            phase = float(phases.get(index, 0.0))
            self._table.setItem(
                row, self._PHASE_COLUMN, QTableWidgetItem(f"{phase:.8g}" if phase else "")
            )

        if self._atom_indices:
            self._table.setCurrentCell(0, self._PHASE_COLUMN)
        self._clear_button = QPushButton("Clear Phases")
        self._clear_button.clicked.connect(self._clear_phases)
        self._build_layout()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(14)
        title = QLabel("Per-Atom Phases", self)
        style_section_title(title)
        layout.addWidget(title)
        intro = QLabel(
            "An extra phase in turns for single atoms, added to the component's phase and to "
            "the q · r phase of the atom's position. A blank cell is 0. Only the atoms the "
            "component applies to are listed."
        )
        intro.setWordWrap(True)
        style_hint_text(intro)
        layout.addWidget(intro)
        layout.addWidget(self._table, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.addButton(self._clear_button, QDialogButtonBox.ButtonRole.ResetRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def atom_phase_turns(self) -> dict[int, float]:
        """Return the non-zero phases by unit-cell atom index, raising ``ValueError`` for a bad cell."""

        phases: dict[int, float] = {}
        for row, index in enumerate(self._atom_indices):
            item = self._table.item(row, self._PHASE_COLUMN)
            text = item.text().strip() if item is not None else ""
            if not text:
                continue
            value = _bounded_decimal(
                text, f"the phase of atom {index + 1}", -self._PHASE_LIMIT, self._PHASE_LIMIT
            )
            if value != 0.0:
                phases[index] = value
        return phases

    def _clear_phases(self) -> None:
        if not workflow_table_utils.confirm_destructive(
            self,
            "Clear Per-Atom Phases",
            "Clear the phase of every atom in this table?",
            "Nothing changes until you click OK.",
            "Clear Phases",
        ):
            return
        for row in range(self._table.rowCount()):
            self._table.setItem(row, self._PHASE_COLUMN, QTableWidgetItem(""))

    def accept(self) -> None:  # noqa: D401 - Qt API override.
        """Accept only when every phase cell holds a valid number."""

        try:
            self.atom_phase_turns()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid per-atom phase", str(exc))
            return
        super().accept()


class MagneticPresetDialog(QDialog):
    """Guided helper that builds a helix, conical, cycloid or spin-density wave from crystal axes."""

    _DELTA_LIMIT_RLU = 4.0
    _PROPAGATION_VECTOR = "vector"

    _KINDS = {
        "Helix": "helix",
        "Conical": "conical",
        "Cycloid": "cycloid",
        "Spin-density wave": "spin_density_wave",
    }
    _REFERENCE_LABELS = {
        "helix": "Moment at zero phase along",
        "conical": "Rotating moment at zero phase along",
        "cycloid": "Rotation plane holds q and",
        "spin_density_wave": "Moment along",
    }

    def __init__(
        self,
        structure: Structure,
        *,
        model: MagneticStructureModel | None = None,
        selected_indices: tuple[int, ...] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Add Magnetic Preset")
        self.resize(clamped_size(self, 640, 560))
        self._structure = structure
        self._model = model if model is not None else MagneticStructureModel()
        self._label_edited = False
        self.setStyleSheet(feature_dialog_stylesheet())

        self._kind_combo = QComboBox()
        for label, kind in self._KINDS.items():
            self._kind_combo.addItem(label, kind)
        self._propagation_combo = QComboBox()
        for axis in ("a", "b", "c"):
            self._propagation_combo.addItem(f"Along {axis}*", f"{axis}*")
        self._propagation_combo.addItem("Enter a vector (r.l.u.)", self._PROPAGATION_VECTOR)
        for slot in q_basis_slots(self._model):
            if not np.allclose(slot.q_rlu, 0.0, atol=1e-12):
                self._propagation_combo.addItem(
                    f"Basis row {slot.label} = ({q_vector_text(slot.q_rlu)})", slot.index
                )
        self._propagation_combo.setCurrentIndex(2)
        self._wavevector = DecimalLineEdit(
            standard_notation=True,
            text="0.25",
            minimum=-self._DELTA_LIMIT_RLU,
            maximum=self._DELTA_LIMIT_RLU,
            placeholder="δ",
        )
        # The exact value behind the text a unit switch wrote, so rounding the text never moves q.
        self._converted_wavevector: tuple[str, float] | None = None
        self._wavevector_unit = QComboBox()
        self._wavevector_unit.addItem("r.l.u.", "rlu")
        self._wavevector_unit.addItem("Å⁻¹", "inverse_angstrom")
        self._wavevector_group = QWidget()
        style_inline_group(self._wavevector_group)
        wavevector_layout = QHBoxLayout(self._wavevector_group)
        wavevector_layout.setContentsMargins(0, 0, 0, 0)
        wavevector_layout.addWidget(self._wavevector, 1)
        wavevector_layout.addWidget(self._wavevector_unit)
        self._wavevector_row_label = QLabel("")
        self._typed_q = direction_inputs(
            (0.0, 0.0, 0.0),
            minimum=-self._DELTA_LIMIT_RLU,
            maximum=self._DELTA_LIMIT_RLU,
            placeholders=("qx", "qy", "qz"),
        )
        for edit, axis in zip(self._typed_q, "xyz", strict=True):
            edit.setAccessibleName(f"q{axis} in r.l.u.")
        self._typed_q_group = direction_group(self._typed_q)
        self._typed_q_row_label = QLabel("q (r.l.u.)")
        self._typed_q_direction: np.ndarray | None = None
        self._reference_combo = QComboBox()
        for axis in ("a", "b", "c"):
            self._reference_combo.addItem(f"Crystal axis {axis}", axis)
        self._reference_row_label = QLabel("")
        self._handedness_combo = QComboBox()
        self._handedness_combo.addItem("Right-handed", 1)
        self._handedness_combo.addItem("Left-handed", -1)
        self._handedness_row_label = QLabel("Handedness")
        self._cone_angle = DecimalLineEdit(
            standard_notation=True, text="45.0", minimum=0.0, maximum=180.0, placeholder="45"
        )
        self._cone_angle_row_label = QLabel("Cone angle (degrees)")
        self._moment = DecimalLineEdit(
            standard_notation=True, text="1.0", minimum=0.0, maximum=1000.0, placeholder="1.0"
        )
        self._selector_combo = QComboBox()
        for selector in viable_atom_selectors(structure, selected_indices=selected_indices):
            self._selector_combo.addItem(selector_label(selector, structure), selector)
        if selected_indices:
            self._selector_combo.setCurrentIndex(self._selector_combo.count() - 1)
        elif self._selector_combo.count() > 1:
            self._selector_combo.setCurrentIndex(1)
        self._label_edit = QLineEdit()
        self._summary = QLabel("")
        self._summary.setWordWrap(True)
        style_hint_text(self._summary)

        self._build_layout()
        self._kind_combo.currentIndexChanged.connect(lambda *_: self._refresh())
        self._propagation_combo.currentIndexChanged.connect(
            lambda *_: self._refresh(reset_reference=True)
        )
        # δ only rescales q along the same axis, so it keeps the reference axis the user chose.
        self._wavevector.textChanged.connect(lambda *_: self._refresh())
        self._wavevector_unit.currentIndexChanged.connect(self._on_wavevector_unit_changed)
        # The summary follows every keystroke; the reference axis waits for the finished entry,
        # as the steps between two parallel vectors point elsewhere.
        for edit in self._typed_q:
            edit.textChanged.connect(lambda *_: self._refresh())
            edit.editingFinished.connect(self._on_typed_q_finished)
        for widget in (self._reference_combo, self._handedness_combo):
            widget.currentIndexChanged.connect(lambda *_: self._refresh())
        for edit in (self._cone_angle, self._moment):
            edit.textChanged.connect(lambda *_: self._refresh())
        self._label_edit.textEdited.connect(self._on_label_edited)
        self._refresh(reset_reference=True)

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(14)
        title = QLabel("Add Magnetic Preset", self)
        style_section_title(title)
        layout.addWidget(title)
        intro = QLabel(
            "Pick the propagation direction and how the moments sit relative to the crystal axes; "
            "the preset becomes an ordinary component you can edit afterwards."
        )
        intro.setWordWrap(True)
        style_hint_text(intro)
        layout.addWidget(intro)

        group = QGroupBox("Preset")
        form = QFormLayout(group)
        form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
        form.addRow("Structure", self._kind_combo)
        form.addRow("Propagation", self._propagation_combo)
        form.addRow(self._wavevector_row_label, self._wavevector_group)
        form.addRow(self._typed_q_row_label, self._typed_q_group)
        form.addRow(self._reference_row_label, self._reference_combo)
        form.addRow(self._handedness_row_label, self._handedness_combo)
        form.addRow(self._cone_angle_row_label, self._cone_angle)
        moment_group = QWidget()
        style_inline_group(moment_group)
        moment_layout = QHBoxLayout(moment_group)
        moment_layout.setContentsMargins(0, 0, 0, 0)
        moment_layout.addWidget(self._moment)
        bohr_label = QLabel("μ<sub>B</sub>")
        bohr_label.setTextFormat(Qt.TextFormat.RichText)
        moment_layout.addWidget(bohr_label)
        form.addRow("Moment", moment_group)
        form.addRow("Apply to", self._selector_combo)
        form.addRow("Label", self._label_edit)
        layout.addWidget(group)
        layout.addWidget(self._summary)
        layout.addStretch(1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Add Component")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _kind(self) -> str:
        return str(self._kind_combo.currentData())

    def _on_label_edited(self, *_args: object) -> None:
        self._label_edited = bool(self._label_edit.text().strip())

    def q_basis_coefficients(self) -> tuple[float, float, float] | None:
        """Return the stored basis row the preset propagates along, or ``None`` for a new q."""

        source = self._propagation_combo.currentData()
        return unit_q_coeff(source) if isinstance(source, int) else None

    def q_vector(self) -> tuple[float, float, float]:
        source = self._propagation_combo.currentData()
        if isinstance(source, int):
            return float_triplet(self._model.q_basis_rlu[source])
        if source == self._PROPAGATION_VECTOR:
            return direction_value(self._typed_q, "q")
        try:
            if self._wavevector_in_inverse_angstrom():
                return wavevector_rlu_from_inverse_angstrom(
                    self._structure.lattice_vectors,
                    self._wavevector_value("k"),
                    reciprocal_axis_q_rlu(str(source), 1.0),
                )
            return reciprocal_axis_q_rlu(str(source), self._wavevector_value("δ"))
        except MagneticStructureError as exc:
            raise ValueError(str(exc)) from None

    def _axis_source(self) -> str | None:
        """The reciprocal axis the preset propagates along, or ``None`` for a row or a typed vector."""

        source = self._propagation_combo.currentData()
        if isinstance(source, int) or source == self._PROPAGATION_VECTOR:
            return None
        return str(source)

    def _on_typed_q_finished(self) -> None:
        self._refresh(reset_reference=self._typed_q_turned())

    def _typed_q_turned(self) -> bool:
        """Note the typed q's direction; a new direction resets the reference axis, a new length keeps it."""

        try:
            q_rlu = np.array(direction_value(self._typed_q, "q"))
        except ValueError:
            return False
        length = float(np.linalg.norm(q_rlu))
        if length <= 1e-12:
            return False
        direction = q_rlu / length
        previous = self._typed_q_direction
        self._typed_q_direction = direction
        return previous is None or not np.allclose(np.cross(previous, direction), 0.0, atol=1e-9)

    def _wavevector_value(self, label: str) -> float:
        converted = self._converted_wavevector
        if converted is not None and converted[0] == self._wavevector.text():
            # The rounded text may sit a hair past the range its exact value lies on the edge of.
            limit = self._wavevector.maximum()
            if abs(converted[1]) <= limit * (1.0 + 1e-12):
                return converted[1]
        return decimal_value(self._wavevector, label)

    def _wavevector_in_inverse_angstrom(self) -> bool:
        return self._wavevector_unit.currentData() == "inverse_angstrom"

    def _on_wavevector_unit_changed(self, *_args: object) -> None:
        """Convert the typed wavevector to the new unit, so q stays the same."""

        inverse_angstrom = self._wavevector_in_inverse_angstrom()
        source = self._propagation_combo.currentData()
        try:
            previous = self._wavevector_value_in_previous_unit()
            axis_length = reciprocal_axis_length(self._structure.lattice_vectors, str(source))
        except (MagneticStructureError, ValueError):
            previous = None
        if previous is not None:
            converted = previous * axis_length if inverse_angstrom else previous / axis_length
            text = f"{converted:.6g}"
            self._wavevector.setText(text)
            self._converted_wavevector = (text, converted)
        self._refresh()

    def _wavevector_value_in_previous_unit(self) -> float:
        text = self._wavevector.text().strip()
        converted = self._converted_wavevector
        if converted is not None and converted[0] == text:
            return converted[1]
        return float(text)

    def _set_wavevector_range(self) -> None:
        """δ spans ±4 r.l.u.; k spans the same wavevectors, ±4 |a*| along the chosen axis.

        Matching ranges let every valid value convert to a valid value in the other unit.
        """

        limit = self._DELTA_LIMIT_RLU
        axis = self._axis_source()
        if self._wavevector_in_inverse_angstrom() and axis is not None:
            try:
                limit *= reciprocal_axis_length(self._structure.lattice_vectors, axis)
            except MagneticStructureError:
                pass
        self._wavevector.setRange(-limit, limit, keep_text=True)

    def _propagation_text(self) -> str:
        source = self._propagation_combo.currentData()
        if isinstance(source, int):
            return f"q{source + 1}"
        if source == self._PROPAGATION_VECTOR:
            try:
                return f"({q_vector_text(self.q_vector())})"
            except ValueError:
                return "a typed q"
        return str(source)

    def component(self, q_coeff: Sequence[float]) -> MagneticComponent:
        kind = self._kind()
        label = self._label_edit.text().strip() or self._default_label()
        return build_magnetic_preset_component(
            kind,
            lattice_vectors=self._structure.lattice_vectors,
            q_rlu=self.q_vector(),
            q_coeff=q_coeff,
            moment=decimal_value(self._moment, "moment"),
            reference_axis=str(self._reference_combo.currentData()),
            selector=self._selector_combo.currentData(),
            label=label,
            handedness=int(self._handedness_combo.currentData()),
            cone_angle_degrees=(
                decimal_value(self._cone_angle, "cone angle") if kind == "conical" else 45.0
            ),
        )

    def _default_label(self) -> str:
        return f"{self._kind_combo.currentText()} along {self._propagation_text()}"

    def _refresh(self, *, reset_reference: bool = False) -> None:
        kind = self._kind()
        from_axis = self._axis_source() is not None
        self._wavevector_group.setVisible(from_axis)
        self._wavevector_row_label.setVisible(from_axis)
        typed = self._propagation_combo.currentData() == self._PROPAGATION_VECTOR
        self._typed_q_group.setVisible(typed)
        self._typed_q_row_label.setVisible(typed)
        inverse_angstrom = self._wavevector_in_inverse_angstrom()
        self._wavevector_row_label.setText(
            "Wavevector k (Å⁻¹)" if inverse_angstrom else "Wavevector δ (r.l.u.)"
        )
        self._wavevector.setPlaceholderText("k" if inverse_angstrom else "δ")
        self._set_wavevector_range()
        rotating = kind != "spin_density_wave"
        self._handedness_combo.setVisible(rotating)
        self._handedness_row_label.setVisible(rotating)
        conical = kind == "conical"
        self._cone_angle.setVisible(conical)
        self._cone_angle_row_label.setVisible(conical)
        self._reference_row_label.setText(self._REFERENCE_LABELS[kind])
        if not self._label_edited:
            self._label_edit.setText(self._default_label())
        try:
            q_rlu = self.q_vector()
            if reset_reference:
                axis = default_preset_reference_axis(self._structure.lattice_vectors, q_rlu)
                self._reference_combo.blockSignals(True)
                self._reference_combo.setCurrentIndex(self._reference_combo.findData(axis))
                self._reference_combo.blockSignals(False)
            self.component(self.q_basis_coefficients() or (1.0, 0.0, 0.0))
        except (MagneticStructureError, ValueError) as exc:
            self._summary.setText(f"Cannot build this preset: {exc}")
            return
        axis = str(self._reference_combo.currentData())
        q_text = f"q = ({q_vector_text(q_rlu)})"
        if kind == "helix":
            text = f"Moments rotate in the plane perpendicular to {q_text}, starting along axis {axis}, projected into it."
        elif kind == "conical":
            text = (
                f"Moments precess on a cone about {q_text}; the rotating part starts along axis {axis}, "
                "projected perpendicular to q."
            )
        elif kind == "cycloid":
            text = f"Moments rotate in the plane that holds {q_text} and axis {axis}."
        else:
            text = f"Moments along {axis} are modulated in size with {q_text}."
        self._summary.setText(text)

    def accept(self) -> None:  # noqa: D401 - Qt API override.
        """Accept only when the preset builds."""

        try:
            self.component(self.q_basis_coefficients() or (1.0, 0.0, 0.0))
        except (MagneticStructureError, ValueError) as exc:
            QMessageBox.warning(self, "Invalid magnetic preset", str(exc))
            return
        super().accept()


class MagneticComponentTable(QTableWidget):
    """Component table with platform-aware deletion keys."""

    def __init__(
        self,
        rows: int,
        columns: int,
        *,
        delete_callback: Callable[[], None] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(rows, columns, parent)
        self._delete_callback = delete_callback
        track_item_view_corners(self)
        self.viewport().installEventFilter(self)

    def eventFilter(self, watched, event) -> bool:  # noqa: N802 - Qt API naming.
        if watched is self.viewport() and event.type() == QEvent.Type.KeyPress:
            if self._handle_delete_key(event):
                return True
        return super().eventFilter(watched, event)

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        if self._handle_delete_key(event):
            return
        super().keyPressEvent(event)

    def _handle_delete_key(self, event) -> bool:
        if not (
            event.matches(QKeySequence.StandardKey.Delete) or event.key() == Qt.Key.Key_Backspace
        ):
            return False
        if self.state() == QAbstractItemView.State.EditingState:
            return False
        if self._delete_callback is not None:
            self._delete_callback()
        event.accept()
        return True


def direction_inputs(
    values: Sequence[float],
    *,
    minimum: float = -100.0,
    maximum: float = 100.0,
    placeholders: Sequence[str] = ("x", "y", "z"),
) -> DecimalTriple:
    """Three decimal fields for a direction's components, showing ``values``."""

    x, y, z = (
        DecimalLineEdit(
            standard_notation=True,
            text=f"{float(value):g}",
            minimum=minimum,
            maximum=maximum,
            placeholder=placeholder,
        )
        for value, placeholder in zip(values, placeholders, strict=True)
    )
    return (x, y, z)


def inline_group(widgets: Sequence[QWidget]) -> QWidget:
    """Lay ``widgets`` out in one compact row with no margins of its own."""

    group = QWidget()
    style_inline_group(group)
    layout = QHBoxLayout(group)
    layout.setContentsMargins(0, 0, 0, 0)
    for widget in widgets:
        layout.addWidget(widget)
    return group


def direction_group(inputs: DecimalTriple) -> QWidget:
    return inline_group(inputs)


def decimal_value(edit: DecimalLineEdit, label: str) -> float:
    """The required value of ``edit``, inside its range, naming ``label`` on error."""

    text = edit.text().strip()
    if not text:
        raise ValueError(f"{label} is required")
    return _bounded_decimal(text, label, edit.minimum(), edit.maximum())


def direction_value(inputs: DecimalTriple, label: str) -> tuple[float, float, float]:
    """The three required components of a direction, naming ``label`` on error."""

    x, y, z = (decimal_value(edit, f"{label} component") for edit in inputs)
    return (x, y, z)


def unit_direction(inputs: DecimalTriple, label: str) -> np.ndarray:
    """The direction ``inputs`` give, as a unit vector; a zero direction is refused."""

    return nonzero_unit_direction(direction_value(inputs, label), label)


def set_direction_inputs(inputs: DecimalTriple, values: Sequence[float]) -> None:
    """Show ``values`` in ``inputs``, leaving a field whose text already matches untouched."""

    for edit, value in zip(inputs, values, strict=True):
        text = f"{float(value):.8g}"
        if edit.text() != text:
            edit.setText(text)


def _bounded_decimal(text: str, label: str, minimum: float, maximum: float) -> float:
    """Parse ``text`` as a finite decimal between ``minimum`` and ``maximum``, naming ``label`` on error."""

    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{label} must be a decimal number") from exc
    if not np.isfinite(value) or value < minimum or value > maximum:
        raise ValueError(f"{label} must be between {minimum:g} and {maximum:g}")
    return value


def _atom_phase_text(phases: Mapping[int, float]) -> str:
    """List per-atom phases as ``atom 2 +0.5 turns, atom 4 +0.5 turns``, atoms numbered from 1."""

    return ", ".join(
        f"atom {index + 1} {phase:+.8g} turns" for index, phase in sorted(phases.items())
    )


def selector_label(selector: AtomSelector, structure: Structure | None = None) -> str:
    if selector.kind == "all":
        return "All atoms"
    if selector.kind == "species":
        return "Species: " + ", ".join(str(value) for value in selector.values)
    if selector.kind == "indices":
        return "Selected atoms: " + ", ".join(str(int(value) + 1) for value in selector.values)
    if selector.kind == "symmetry_equivalent" and selector.values:
        if structure is not None:
            return _equivalent_selector_label(selector, structure)
        return f"Equivalent site containing atom {int(selector.values[0]) + 1}"
    return selector.kind.replace("_", " ").title()


def _equivalent_selector_label(selector: AtomSelector, structure: Structure) -> str:
    representative = int(selector.values[0])
    atom_count = len(structure.species)
    if representative < 0 or representative >= atom_count:
        return f"Equivalent site containing atom {representative + 1}"

    species = str(structure.species[representative])
    position = _fractional_position_label(structure.fractional_positions[representative])
    try:
        indices = resolve_atom_selector(structure, selector)
    except ValueError:
        indices = (representative,)
    count = len(indices)
    index_label = ", ".join(str(index + 1) for index in indices)
    atom_word = "atom" if count == 1 else "atoms"
    return (
        f"{species} site: atom {representative + 1} at ({position}); "
        f"{count} equivalent {atom_word} ({index_label})"
    )


def _fractional_position_label(position: object) -> str:
    values = np.asarray(position, dtype=float).reshape(3)
    return ", ".join(_compact_float(value) for value in values)


def _compact_float(value: float) -> str:
    if abs(value) < 5e-13:
        value = 0.0
    return f"{float(value):.5g}"


def component_kind_label(component: MagneticComponent) -> str:
    """The name a component's stored ``kind`` is shown under."""

    if component.kind == "legacy_moment":
        return "Moment"
    return component.kind.replace("_", " ").title()


def component_q_label(model: MagneticStructureModel, component: MagneticComponent) -> str:
    """List each distinct q a component uses as its q-basis combination and its r.l.u. vector."""

    values: list[str] = []
    for term in component.terms:
        vector = q_vector_text(model.q_coeff_to_rlu(term.q_coeff))
        expression = q_coeff_expression(term.q_coeff)
        value = f"({vector})" if expression == "0" else f"{expression} = ({vector})"
        if value not in values:
            values.append(value)
    return "; ".join(values) or "-"


def create_q_basis_table(*, editable: bool) -> QTableWidget:
    """Build the three-row ``Row`` / ``q (r.l.u.)`` / ``Used by`` table that shows a model's q basis."""

    table = create_workflow_table(
        _Q_BASIS_HEADERS,
        resize_to_contents_columns=(0,),
        selection_mode=QAbstractItemView.SelectionMode.NoSelection,
        editable=editable,
    )
    table.setRowCount(3)
    table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.setSizeAdjustPolicy(QAbstractItemView.SizeAdjustPolicy.AdjustToContents)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    return table


def fill_q_basis_table(table: QTableWidget, model: MagneticStructureModel) -> None:
    """Show each q-basis row, its vector and the components that use it, all read-only."""

    components = {component.component_id: component for component in model.components}
    for slot in q_basis_slots(model):
        users = [
            components[component_id].label
            + ("" if components[component_id].enabled else " (disabled)")
            for component_id in slot.component_ids
        ]
        users_item = QTableWidgetItem(", ".join(users) if users else "Unused")
        users_item.setToolTip("\n".join(users) if users else "No component references this row")
        if not users:
            users_item.setForeground(QColor(current_palette().text_muted))
        for column, item in enumerate(
            (QTableWidgetItem(slot.label), QTableWidgetItem(q_vector_text(slot.q_rlu)), users_item)
        ):
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(slot.index, column, item)


def q_vector_text(values: Sequence[float]) -> str:
    return ", ".join(
        format_table_number(value, absolute_tolerance=5e-13, relative_tolerance=0.0)
        for value in values
    )


def component_moment_label(component: MagneticComponent) -> str:
    if not component.terms:
        return "-"
    if is_texture_component(component):
        magnitudes = [
            float(np.linalg.norm(term.coefficient.to_array())) for term in component.terms
        ]
        mode = str(component.metadata.get("texture_mode", "texture")).replace("_", " ")
        return f"{mode.title()}\n{plural(len(component.terms), 'Fourier term')}\nmax {_format_moment_magnitude(max(magnitudes))}"
    if component.kind == "conical" and len(component.terms) >= 2:
        static = np.linalg.norm(np.asarray(component.terms[0].coefficient.real, dtype=float))
        rotating_real = np.linalg.norm(np.asarray(component.terms[1].coefficient.real, dtype=float))
        rotating_imag = np.linalg.norm(
            np.asarray(component.terms[1].coefficient.imaginary, dtype=float)
        )
        rotating = 0.5 * (rotating_real + rotating_imag)
        angle = np.degrees(np.arctan2(rotating, static))
        return (
            f"Conical α≈{angle:.3g}°\n"
            f"static {_format_moment_magnitude(static)}\n"
            f"rotating {_format_moment_magnitude(rotating)}"
        )
    coefficient = component.terms[0].coefficient
    real = ", ".join(f"{value:.3g}" for value in coefficient.real)
    imag = ", ".join(f"{value:.3g}" for value in coefficient.imaginary)
    if any(abs(value) > 1e-12 for value in coefficient.imaginary):
        return f"Re [{real}], Im [{imag}]"
    return f"[{real}]"


def _format_moment_magnitude(value: float) -> str:
    return f"{float(value):.3g} μB"
