"""Magnetic-field workflow page and magnetic-structure builder dialog."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import asdict, dataclass, replace
from html import escape
from pathlib import Path
from typing import Any
from uuid import UUID

import numpy as np
from matplotlib.ticker import ScalarFormatter
from matplotlib.transforms import Bbox
from PySide6.QtCore import (
    QEvent,
    QItemSelection,
    QItemSelectionModel,
    QPointF,
    QRectF,
    QSize,
    Qt,
    Signal,
)
from PySide6.QtGui import QColor, QDoubleValidator, QKeySequence, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QAbstractItemView,
    QButtonGroup,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFileDialog,
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
    QScrollArea,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextBrowser,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.atomic_structure_viewer import (
    AtomicStructureViewer,
    _configure_transparent_pyvista_background,
    _pyvista_modules,
    _shutdown_pyvista_widget,
)
from mufinder.app.icons import icon
from mufinder.app.pinned_sites_scope import PinnedSitesScopeBar
from mufinder.app.plot_canvas import PlotFrame, create_plot_canvas
from mufinder.app.sample_photo_dialog import SamplePhotoDialog, demag_length_input
from mufinder.app.screen_utils import clamped_size
from mufinder.app.theme import (
    current_palette,
    feature_dialog_stylesheet,
    set_option_card_state,
    style_help_button,
    style_hint_text,
    style_inline_group,
    style_option_card,
    style_plot_axes,
    style_plot_legend,
    style_section_subtitle,
    style_section_title,
    style_small_checked_tool_button,
    style_warning_text,
    use_square_top_table_corners,
    track_item_view_corners,
)
from mufinder.app import workflow_table_utils
from mufinder.app.workflow_table_utils import (
    create_workflow_table,
    format_table_number,
    format_vector_components,
)
from mufinder.domain.models import Structure
from mufinder.persistence.codec import to_jsonable
from mufinder.science.ase_bridge import host_elements, reduced_formula
from mufinder.science.demagnetization import (
    DemagnetizationError,
    DemagnetizingSample,
    fourier_demagnetizing_result,
    SampleOrientation,
    principal_demagnetizing_tensor,
    sample_orientation,
)
from mufinder.science.lattice import reciprocal_to_cartesian
from mufinder.science.magnetic_fields import (
    CONTACT_COUPLING_UNIT_OPTIONS,
    CONTACT_FIELD_T_PER_ANGSTROM_MINUS3_PER_MUB,
    MagneticFieldDependencyError,
    MagneticFieldDistributionResult,
    MagneticFieldError,
    MagneticFieldOptions,
    MUON_GYROMAGNETIC_RATIO_MHZ_PER_T,
    assigned_magnetic_atom_indices,
    calculate_commensurate_approximant_fields,
    calculate_commensurate_fields,
    calculate_dipolar_tensors_for_atoms,
    calculate_incommensurate_field_distributions,
    classify_magnetic_model,
    contact_coupling_to_angstrom_minus3,
    contact_shell_tie_checker,
    find_commensurate_approximant,
    muon_site_clearance_error,
)
from mufinder.science.magnetic_structure import (
    AtomSelector,
    atom_phases_for_selection,
    MagneticComponent,
    MagneticStructureModel,
    MagneticStructureError,
    CHECKLIST_DONE,
    CHECKLIST_PROBLEM,
    build_collinear_component,
    build_conical_component,
    build_cycloid_component,
    build_helix_component,
    build_hexagonal_skyrmion_texture,
    build_magnetic_preset_component,
    build_square_skyrmion_texture,
    build_spin_density_wave_component,
    build_texture_component,
    evaluate_texture_grid,
    component_q_basis_indices,
    default_preset_reference_axis,
    duplicate_magnetic_components,
    enabled_magnetic_components,
    is_texture_component,
    magnetic_builder_checklist,
    magnetic_model_from_legacy_structure,
    MagneticTexture,
    MagneticTextureTerm,
    moments_in_supercell,
    move_magnetic_components,
    normalize_texture_magnitude,
    prune_texture_terms,
    q_basis_combination_rlu,
    q_basis_slots,
    q_cartesian_direction,
    q_coeff_expression,
    reciprocal_axis_q_rlu,
    resolve_atom_selector,
    set_magnetic_components_enabled,
    set_q_basis_vector,
    skyrmion_q_basis_rlu,
    texture_diagnostics,
    topological_charge_magnetic_unit_cell,
    viable_atom_selectors,
    reciprocal_axis_length,
    wavevector_rlu_from_inverse_angstrom,
)
from mufinder.science.nuclear_relaxation import (
    NuclearRelaxationError,
    NuclearRelaxationOptions,
    calculate_nuclear_relaxation,
    calculate_nuclear_relaxation_with_distortion,
)
from mufinder.services.magnetic_field_results import (
    UEP_SITE_SOURCE,
    UNRELAXED_RESULTS_NOTE,
    is_unrelaxed_site_source,
)
from mufinder.services.muon_sites_summary import (
    MAGNETIC_FIELDS_SCOPE,
    NO_PINNED_SITES_NOTICE,
    has_pinned_sites,
    pinned_sites_by_representative,
    set_show_every_relaxed_site,
    show_every_relaxed_site,
)
from mufinder.services.project_service import ProjectDocument


_DIRECTION_TOLERANCE = 1e-6
_Q_BASIS_HEADERS = ("Row", "q (r.l.u.)", "Used by")
_UEP_ANALYSIS_STATES_KEY = "uep_analysis_states"
_RELAXATION_ANALYSIS_STATE_KEY = "relaxation_analysis_workspaces"
_SITE_OPTION_ROLE = int(Qt.ItemDataRole.UserRole) + 41
_DISTORTION_CORRECTED_FIELD = "field (distortion-corrected)"
_DISTORTION_CORRECTED_DISTRIBUTION = "distribution mean (distortion-corrected)"


def _distribution_label(calculation_parameters: object) -> str:
    """The Result column text of a distribution row, marking a distortion-corrected one."""

    corrected = isinstance(calculation_parameters, Mapping) and bool(
        calculation_parameters.get("distortion_corrected", False)
    )
    return _DISTORTION_CORRECTED_DISTRIBUTION if corrected else "distribution mean"


_UNRELAXED_ESTIMATE_NOTE = (
    "UEP sites are unrelaxed, so fields computed at them are estimates: "
    "the relaxed muon site can sit some distance from the UEP minimum."
)
_UNRELAXED_ESTIMATE_SUFFIX = " Computed at an unrelaxed UEP site, so these are estimates."
_DISTRIBUTION_RESULT_ROLE = int(Qt.ItemDataRole.UserRole) + 42
MAX_INCOMMENSURATE_PHASE_SAMPLES = 10_000_000
MAX_INCOMMENSURATE_LOOKUP_ANGLES = 1_000_000
MAX_APPROXIMANT_PERIOD = 64
MAX_CONTACT_NEIGHBOURS = 64
MIN_CONTACT_RADIUS_ANG = 0.1
MAX_CONTACT_RADIUS_ANG = 100.0
CONTACT_RADIUS_DECIMALS = 2
CONVERGENCE_GUIDANCE_TEXT = (
    "Each distribution is checked after it is calculated: halving the lookup angles or the samples "
    "must shift the field distribution by at most 1%, and every occupied histogram bin needs at "
    "least 20 samples. Select a distribution row to read the checks, then double any flagged "
    "setting and recalculate."
)
PHASE_SAMPLING_METHOD_OPTIONS: tuple[tuple[str, str], ...] = (
    ("auto", "Auto"),
    ("grid", "Grid"),
    ("montecarlo", "Monte Carlo"),
)
ADVANCED_DEMAG_SHAPE_OPTIONS: tuple[tuple[str, str], ...] = (
    ("ellipsoid", "Ellipsoid"),
    ("rectangular_prism", "Rectangular prism"),
    ("image_plate", "Plate from photo"),
)
# Offered only for a sample given as voxels through the Python API.
DEMAG_VOXEL_GRID_OPTION = ("voxel_grid", "Voxel grid")
ADVANCED_DEMAG_SHAPES = frozenset(
    {key for key, _label in ADVANCED_DEMAG_SHAPE_OPTIONS} | {DEMAG_VOXEL_GRID_OPTION[0]}
)
# The preview draws at most this many voxels, skipping evenly through larger grids.
_PREVIEW_MAX_VOXELS = 400_000
DEMAG_ORIENTATION_OPTIONS: tuple[tuple[str, str], ...] = (
    ("aligned", "Aligned with crystal axes"),
    ("directions", "Crystal directions"),
)
# Offered only for a sample whose rotation was saved without crystal directions.
DEMAG_SAVED_ROTATION_OPTION = ("saved", "Saved rotation")


@dataclass(frozen=True, slots=True)
class MagneticMuonSiteOption:
    """Muon-site option selected for magnetic-field calculation."""

    source: str
    label: str
    fractional_position: tuple[float, float, float]
    cartesian_position: tuple[float, float, float] | None = None
    cluster_label: str | None = None
    source_label: str | None = None
    distortion_structure_id: UUID | None = None
    properties: dict[str, object] | None = None
    include_distortion: bool = False


class MagneticComponentDialog(QDialog):
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
        self._apply_styles()

        self._label_edit = QLineEdit("Magnetic component")
        self._label_edit.setMinimumWidth(420)
        self._kind_combo = QComboBox()
        self._kind_combo.addItems(list(self._TEXTURE_KINDS))
        self._component_info_button = QToolButton()
        style_help_button(self._component_info_button)
        self._component_info_button.setIcon(icon("help"))
        self._component_info_button.setIconSize(QSize(22, 22))
        self._component_info_button.setToolTip(
            "Explain this component in the propagation-vector formalism"
        )
        self._component_info_button.clicked.connect(self._show_component_info)

        self._amplitude = self._decimal_input("1.0", minimum=0.0, maximum=1000.0, placeholder="1.0")
        self._cone_angle = self._decimal_input("45.0", minimum=0.0, maximum=180.0, placeholder="45")

        self._primary_direction = self._direction_inputs((0.0, 0.0, 1.0))
        self._secondary_direction = self._direction_inputs((1.0, 0.0, 0.0))
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
                self._equivalent_combo.addItem(_selector_label(selector, structure), selector)

        self._q_x = self._decimal_input("0.0", minimum=-4.0, maximum=4.0, placeholder="qx")
        self._q_y = self._decimal_input("0.0", minimum=-4.0, maximum=4.0, placeholder="qy")
        self._q_z = self._decimal_input("0.0", minimum=-4.0, maximum=4.0, placeholder="qz")
        self._q_source_combo = QComboBox()
        self._q_source_combo.addItem("Enter a vector (r.l.u.)", self._Q_SOURCE_VECTOR)
        stored_rows = [
            slot
            for slot in q_basis_slots(self._model)
            if not np.allclose(slot.q_rlu, 0.0, atol=1e-12)
        ]
        for slot in stored_rows:
            self._q_source_combo.addItem(
                f"{slot.label} = ({_q_vector_text(slot.q_rlu)})", slot.index
            )
        if stored_rows:
            self._q_source_combo.addItem("Combination of basis rows", self._Q_SOURCE_COMBINATION)
        else:
            self._q_source_combo.setToolTip(
                "No propagation vector is stored yet; enter q as a vector"
            )
        self._q_coeff_inputs = tuple(
            self._decimal_input("0", minimum=-4.0, maximum=4.0, placeholder=f"n{index + 1}")
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
        self._q_basis_table = _create_q_basis_table(editable=False)
        _fill_q_basis_table(self._q_basis_table, self._model)
        self._q_basis_table.resizeRowsToContents()
        self._q_basis_table.setFixedHeight(
            self._q_basis_table.horizontalHeader().sizeHint().height()
            + sum(self._q_basis_table.rowHeight(row) for row in range(3))
            + 2 * self._q_basis_table.frameWidth()
        )

        self._phase = self._decimal_input("0.0", minimum=-10.0, maximum=10.0, placeholder="0.0")
        self._atom_phase_turns: dict[int, float] = (
            dict(component.atom_phase_turns) if component is not None else {}
        )
        self._atom_phase_summary = QLabel("")
        self._atom_phase_summary.setWordWrap(True)
        self._atom_phase_summary.setSizePolicy(
            QSizePolicy.Policy.Preferred, QSizePolicy.Policy.MinimumExpanding
        )
        style_hint_text(self._atom_phase_summary)
        self._atom_phase_button = QPushButton("Per-Atom Phases...")
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
        bohr_label.setStyleSheet("background: transparent;")
        amplitude_layout.addWidget(bohr_label)
        form.addRow("Amplitude", amplitude_group)
        self._primary_direction_row_label = QLabel("Direction (x, y, z)")
        self._primary_direction_group = self._direction_group(self._primary_direction)
        form.addRow(self._primary_direction_row_label, self._primary_direction_group)
        self._secondary_direction_row_label = QLabel("Derived direction (x, y, z)")
        self._secondary_direction_group = self._direction_group(self._secondary_direction)
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
            row_label.setStyleSheet("background: transparent;")
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

    @staticmethod
    def _decimal_input(
        value: str,
        *,
        minimum: float,
        maximum: float,
        placeholder: str,
    ) -> QLineEdit:
        edit = QLineEdit(value)
        edit.setPlaceholderText(placeholder)
        edit.setProperty("minimum", minimum)
        edit.setProperty("maximum", maximum)
        validator = QDoubleValidator(minimum, maximum, 8, edit)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        edit.setValidator(validator)
        return edit

    @staticmethod
    def _direction_inputs(
        values: Sequence[float],
        *,
        minimum: float = -100.0,
        maximum: float = 100.0,
        placeholders: Sequence[str] = ("x", "y", "z"),
    ) -> tuple[QLineEdit, QLineEdit, QLineEdit]:
        return tuple(  # type: ignore[return-value]
            MagneticComponentDialog._decimal_input(
                f"{float(value):g}", minimum=minimum, maximum=maximum, placeholder=placeholder
            )
            for value, placeholder in zip(values, placeholders, strict=True)
        )

    @staticmethod
    def _direction_group(inputs: tuple[QLineEdit, QLineEdit, QLineEdit]) -> QWidget:
        group = QWidget()
        style_inline_group(group)
        layout = QHBoxLayout(group)
        layout.setContentsMargins(0, 0, 0, 0)
        for edit in inputs:
            layout.addWidget(edit)
        return group

    def _build_target_card(
        self, button: QRadioButton, hint_text: str, editor: QWidget | None = None
    ) -> QFrame:
        card = QFrame(self)
        style_option_card(card)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(6)
        button.setParent(card)
        hint = QLabel(hint_text, card)
        hint.setWordWrap(True)
        hint.setStyleSheet("background: transparent;")
        style_hint_text(hint)
        card_layout.addWidget(button)
        card_layout.addWidget(hint)
        if editor is not None:
            editor.setParent(card)
            card_layout.addWidget(editor)
        self._selector_rows.append((button, card, hint))
        button.toggled.connect(self._refresh_target_card_styles)
        return card

    def _apply_styles(self) -> None:
        self.setStyleSheet(
            feature_dialog_stylesheet(
                extra=(
                    "QRadioButton {"
                    "font-weight: 600;"
                    "spacing: 10px;"
                    "padding-top: 4px;"
                    "padding-bottom: 4px;"
                    "}"
                )
            )
        )

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

    def _load_component_vectors(self, component: MagneticComponent) -> None:
        if not component.terms:
            return
        if component.kind == "conical" and len(component.terms) >= 2:
            static = np.asarray(component.terms[0].coefficient.real, dtype=float)
            rotating = component.terms[1].coefficient
            real = np.asarray(rotating.real, dtype=float)
            imag = np.asarray(rotating.imaginary, dtype=float)
            real_direction = np.asarray(_fallback_direction(real, (0.0, 0.0, 1.0)), dtype=float)
            imag_direction = np.asarray(_fallback_direction(imag, (1.0, 0.0, 0.0)), dtype=float)
            q_direction = self._q_cartesian_direction(allow_fallback=True)
            if q_direction is None:
                axis = np.cross(real_direction, imag_direction)
            else:
                axis = q_direction
            axis_norm = float(np.linalg.norm(axis))
            if axis_norm <= _DIRECTION_TOLERANCE:
                axis = np.asarray((0.0, 0.0, 1.0), dtype=float)
            else:
                axis = axis / axis_norm
            signed_static = float(static @ axis)
            rotating_norm = 0.5 * (float(np.linalg.norm(real)) + float(np.linalg.norm(imag)))
            self._amplitude.setText(f"{float(np.hypot(signed_static, rotating_norm)):.8g}")
            self._cone_angle.setText(
                f"{float(np.degrees(np.arctan2(rotating_norm, signed_static))):.8g}"
            )
            self._set_direction_inputs(self._primary_direction, real_direction)
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
            self._set_direction_inputs(
                self._primary_direction, _fallback_direction(real, (0.0, 0.0, 1.0))
            )
            q_direction = self._q_cartesian_direction(allow_fallback=True)
            if q_direction is not None:
                if component.kind == "cycloid":
                    self._set_handedness(1 if float(np.dot(imag, q_direction)) >= 0.0 else -1)
                else:
                    self._set_handedness_from_axes(
                        np.asarray(_fallback_direction(real, (0.0, 0.0, 1.0)), dtype=float),
                        np.asarray(_fallback_direction(imag, (1.0, 0.0, 0.0)), dtype=float),
                        q_direction,
                    )
            self._sync_constrained_direction_inputs()
            return
        self._amplitude.setText(f"{real_norm:.8g}")
        self._set_direction_inputs(
            self._primary_direction, _fallback_direction(real, (0.0, 0.0, 1.0))
        )
        self._set_direction_inputs(self._secondary_direction, (1.0, 0.0, 0.0))

    @staticmethod
    def _set_direction_inputs(
        inputs: tuple[QLineEdit, QLineEdit, QLineEdit], values: Sequence[float]
    ) -> None:
        for edit, value in zip(inputs, values, strict=True):
            text = f"{float(value):.8g}"
            if edit.text() != text:
                edit.setText(text)

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
        primary_arr = np.asarray(primary, dtype=float)
        secondary_arr = np.asarray(secondary, dtype=float)
        q_arr = np.asarray(q_direction, dtype=float)
        sign = 1 if float(np.dot(np.cross(q_arr, primary_arr), secondary_arr)) >= 0.0 else -1
        self._set_handedness(sign)

    def _q_cartesian_direction(self, *, allow_fallback: bool = False) -> np.ndarray | None:
        try:
            q_rlu = np.asarray(self.q_vector(), dtype=float)
        except ValueError:
            return None
        if float(np.linalg.norm(q_rlu)) <= _DIRECTION_TOLERANCE:
            if allow_fallback:
                return np.asarray((0.0, 0.0, 1.0), dtype=float)
            raise ValueError("q must be non-zero for helical, cycloidal, and conical components")
        try:
            return q_cartesian_direction(self._structure.lattice_vectors, q_rlu)
        except MagneticStructureError:
            if allow_fallback:
                return np.asarray((0.0, 0.0, 1.0), dtype=float)
            raise ValueError("q must define a non-zero propagation direction") from None

    def _projected_reference_direction(self, q_direction: np.ndarray) -> np.ndarray:
        try:
            raw = np.asarray(
                [
                    self._decimal_value(edit, "reference direction component")
                    for edit in self._primary_direction
                ],
                dtype=float,
            )
        except ValueError:
            return _orthogonal_unit(q_direction)
        projected = raw - float(raw @ q_direction) * q_direction
        norm = float(np.linalg.norm(projected))
        if norm <= _DIRECTION_TOLERANCE:
            return _orthogonal_unit(q_direction)
        return projected / norm

    def _constrained_component_directions(
        self, kind: str
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        q_direction = self._q_cartesian_direction()
        if q_direction is None:
            raise ValueError("q must be valid for helical, cycloidal, and conical components")
        primary = self._projected_reference_direction(q_direction)
        sign = float(self._handedness_sign())
        if kind == "cycloid":
            secondary = sign * q_direction
        else:
            secondary = sign * np.cross(q_direction, primary)
            secondary_norm = float(np.linalg.norm(secondary))
            if secondary_norm <= _DIRECTION_TOLERANCE:
                secondary = _orthogonal_unit(primary)
            else:
                secondary = secondary / secondary_norm
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
            self._set_direction_inputs(self._primary_direction, primary)
            self._set_direction_inputs(self._secondary_direction, secondary)
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
        selector = escape(_selector_label(self.selector(), self._structure))

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
            return _unit_q_coeff(source)
        if source == self._Q_SOURCE_COMBINATION:
            values = tuple(
                self._decimal_value(edit, f"q{index + 1} coefficient") if edit.isEnabled() else 0.0
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
            f"q = {q_coeff_expression(coeff or (0.0, 0.0, 0.0))} = ({_q_vector_text(q_rlu)}); "
            "the component follows these rows if they are edited."
        )

    def q_vector(self) -> tuple[float, float, float]:
        return (
            self._decimal_value(self._q_x, "q x"),
            self._decimal_value(self._q_y, "q y"),
            self._decimal_value(self._q_z, "q z"),
        )

    def _primary_unit_direction(self) -> np.ndarray:
        return self._unit_direction(self._primary_direction, "first direction")

    def _secondary_unit_direction(self) -> np.ndarray:
        return self._unit_direction(self._secondary_direction, "second direction")

    def _validate_directions(self) -> tuple[np.ndarray, np.ndarray]:
        primary = self._primary_unit_direction()
        secondary = self._secondary_unit_direction()
        if abs(float(primary @ secondary)) > _DIRECTION_TOLERANCE:
            raise ValueError("first and second directions must be perpendicular")
        return primary, secondary

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
        amplitude = self._decimal_value(self._amplitude, "amplitude")
        label = self._label_edit.text().strip() or "Magnetic component"
        selector = self.selector()
        kind = self._TEXTURE_KINDS.get(self._kind_combo.currentText(), "collinear")
        if kind in {"helix", "cycloid", "conical"}:
            primary_direction, secondary_direction, propagation_direction = (
                self._constrained_component_directions(kind)
            )
            self._set_direction_inputs(self._primary_direction, primary_direction)
            self._set_direction_inputs(self._secondary_direction, secondary_direction)
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
            "phase_turns": self._decimal_value(self._phase, "phase"),
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
                cone_angle_degrees=self._decimal_value(self._cone_angle, "cone angle"),
                **kwargs,
            )
        return build_collinear_component(moment=primary, **kwargs)

    @staticmethod
    def _decimal_value(edit: QLineEdit, label: str) -> float:
        text = edit.text().strip()
        if not text:
            raise ValueError(f"{label} is required")
        return _bounded_decimal(
            text, label, float(edit.property("minimum")), float(edit.property("maximum"))
        )

    def _unit_direction(
        self, inputs: tuple[QLineEdit, QLineEdit, QLineEdit], label: str
    ) -> np.ndarray:
        vector = np.asarray(
            [self._decimal_value(edit, f"{label} component") for edit in inputs],
            dtype=float,
        )
        norm = float(np.linalg.norm(vector))
        if norm <= _DIRECTION_TOLERANCE:
            raise ValueError(f"{label} must be a non-zero direction")
        return vector / norm

    @staticmethod
    def _direction_display(inputs: tuple[QLineEdit, QLineEdit, QLineEdit]) -> str:
        return "(" + ", ".join(edit.text().strip() or "0" for edit in inputs) + ")"

    def accept(self) -> None:  # noqa: D401 - Qt API override.
        """Accept the dialog only when decimal fields are valid."""

        try:
            self._decimal_value(self._amplitude, "amplitude")
            coeff = self.q_basis_coefficients()
            if coeff is not None:
                q_basis_combination_rlu(self._model, coeff)
            self.q_vector()
            self._decimal_value(self._phase, "phase")
            kind = self._TEXTURE_KINDS.get(self._kind_combo.currentText(), "collinear")
            if kind in {"helix", "cycloid", "conical"}:
                self._constrained_component_directions(kind)
            else:
                self._primary_unit_direction()
            if kind == "conical":
                self._decimal_value(self._cone_angle, "cone angle")
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
        for slot in q_basis_slots(self._model):
            if not np.allclose(slot.q_rlu, 0.0, atol=1e-12):
                self._propagation_combo.addItem(
                    f"Basis row {slot.label} = ({_q_vector_text(slot.q_rlu)})", slot.index
                )
        self._propagation_combo.setCurrentIndex(2)
        self._wavevector = MagneticComponentDialog._decimal_input(
            "0.25",
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
        self._reference_combo = QComboBox()
        for axis in ("a", "b", "c"):
            self._reference_combo.addItem(f"Crystal axis {axis}", axis)
        self._reference_row_label = QLabel("")
        self._handedness_combo = QComboBox()
        self._handedness_combo.addItem("Right-handed", 1)
        self._handedness_combo.addItem("Left-handed", -1)
        self._handedness_row_label = QLabel("Handedness")
        self._cone_angle = MagneticComponentDialog._decimal_input(
            "45.0", minimum=0.0, maximum=180.0, placeholder="45"
        )
        self._cone_angle_row_label = QLabel("Cone angle (degrees)")
        self._moment = MagneticComponentDialog._decimal_input(
            "1.0", minimum=0.0, maximum=1000.0, placeholder="1.0"
        )
        self._selector_combo = QComboBox()
        for selector in viable_atom_selectors(structure, selected_indices=selected_indices):
            self._selector_combo.addItem(_selector_label(selector, structure), selector)
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
        bohr_label.setStyleSheet("background: transparent;")
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
        """Return the stored basis row the preset propagates along, or ``None`` for a new reciprocal-axis q."""

        source = self._propagation_combo.currentData()
        return _unit_q_coeff(source) if isinstance(source, int) else None

    def q_vector(self) -> tuple[float, float, float]:
        source = self._propagation_combo.currentData()
        if isinstance(source, int):
            return _tuple3_float(self._model.q_basis_rlu[source])
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

    def _wavevector_value(self, label: str) -> float:
        converted = self._converted_wavevector
        if converted is not None and converted[0] == self._wavevector.text():
            # The rounded text may sit a hair past the range its exact value lies on the edge of.
            limit = float(self._wavevector.property("maximum"))
            if abs(converted[1]) <= limit * (1.0 + 1e-12):
                return converted[1]
        return MagneticComponentDialog._decimal_value(self._wavevector, label)

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
        source = self._propagation_combo.currentData()
        if self._wavevector_in_inverse_angstrom() and not isinstance(source, int):
            try:
                limit *= reciprocal_axis_length(self._structure.lattice_vectors, str(source))
            except MagneticStructureError:
                pass
        minimum, maximum = -limit, limit
        self._wavevector.setProperty("minimum", minimum)
        self._wavevector.setProperty("maximum", maximum)
        validator = self._wavevector.validator()
        if isinstance(validator, QDoubleValidator):
            validator.setRange(minimum, maximum, 8)

    def _propagation_text(self) -> str:
        source = self._propagation_combo.currentData()
        return f"q{source + 1}" if isinstance(source, int) else str(source)

    def component(self, q_coeff: Sequence[float]) -> MagneticComponent:
        kind = self._kind()
        label = self._label_edit.text().strip() or self._default_label()
        return build_magnetic_preset_component(
            kind,
            lattice_vectors=self._structure.lattice_vectors,
            q_rlu=self.q_vector(),
            q_coeff=q_coeff,
            moment=MagneticComponentDialog._decimal_value(self._moment, "moment"),
            reference_axis=str(self._reference_combo.currentData()),
            selector=self._selector_combo.currentData(),
            label=label,
            handedness=int(self._handedness_combo.currentData()),
            cone_angle_degrees=(
                MagneticComponentDialog._decimal_value(self._cone_angle, "cone angle")
                if kind == "conical"
                else 45.0
            ),
        )

    def _default_label(self) -> str:
        return f"{self._kind_combo.currentText()} along {self._propagation_text()}"

    def _refresh(self, *, reset_reference: bool = False) -> None:
        kind = self._kind()
        from_axis = not isinstance(self._propagation_combo.currentData(), int)
        self._wavevector_group.setVisible(from_axis)
        self._wavevector_row_label.setVisible(from_axis)
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
        q_text = f"q = ({_q_vector_text(q_rlu)})"
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


class MagneticTexturePreview(QWidget):
    """Compact Qt-painted preview of a 2D texture slice."""

    _MIN_VIEW_FRACTION = 0.035

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._moments = np.empty((0, 0, 3), dtype=float)
        self._canvas_mode = "light"
        self._view_center = np.asarray([0.5, 0.5], dtype=float)
        self._view_fraction = 1.0
        self._initial_view_fraction = 1.0
        self._pan_start: QPointF | None = None
        self._pan_start_center = self._view_center.copy()
        self._box_start: QPointF | None = None
        self._box_end: QPointF | None = None
        self.setMouseTracking(True)
        self.setMinimumSize(300, 260)

    def set_canvas_mode(self, mode: str) -> None:
        self._canvas_mode = "dark" if mode == "dark" else "light"
        self.update()

    def set_moments(
        self, moments: np.ndarray | None, *, initial_view_fraction: float | None = None
    ) -> None:
        if moments is None:
            self._moments = np.empty((0, 0, 3), dtype=float)
            self.reset_view()
        else:
            arr = np.asarray(moments, dtype=float)
            self._moments = arr[:, :, 0, :] if arr.ndim == 4 else arr
            if initial_view_fraction is not None:
                self.reset_view(initial_view_fraction=initial_view_fraction)
        self.update()

    def reset_view(self, *, initial_view_fraction: float = 1.0) -> None:
        self._initial_view_fraction = self._bounded_view_fraction(initial_view_fraction)
        self._view_center = np.asarray([0.5, 0.5], dtype=float)
        self._view_fraction = self._initial_view_fraction
        self._pan_start = None
        self._box_start = None
        self._box_end = None
        self.update()

    @staticmethod
    def _mix_color(start: QColor, end: QColor, t: float) -> QColor:
        t = max(0.0, min(1.0, float(t)))
        return QColor(
            round(start.red() + (end.red() - start.red()) * t),
            round(start.green() + (end.green() - start.green()) * t),
            round(start.blue() + (end.blue() - start.blue()) * t),
            round(start.alpha() + (end.alpha() - start.alpha()) * t),
        )

    def _display_moments(self) -> np.ndarray:
        if self._moments.size == 0:
            return self._moments
        i0, i1, j0, j1 = self._visible_index_bounds()
        return self._moments[i0:i1, j0:j1]

    def _plot_rect(self) -> QRectF:
        bounds = QRectF(self.rect()).adjusted(12.0, 12.0, -12.0, -12.0)
        side = max(10.0, min(bounds.width(), bounds.height()))
        return QRectF(
            bounds.left() + max(0.0, (bounds.width() - side) / 2.0),
            bounds.top() + max(0.0, (bounds.height() - side) / 2.0),
            side,
            side,
        )

    def _view_bounds(self) -> tuple[float, float, float, float]:
        half = 0.5 * self._view_fraction
        low_x = float(self._view_center[0] - half)
        high_x = float(self._view_center[0] + half)
        low_y = float(self._view_center[1] - half)
        high_y = float(self._view_center[1] + half)
        return low_x, high_x, low_y, high_y

    def _visible_index_bounds(self) -> tuple[int, int, int, int]:
        nx, ny = self._moments.shape[:2]
        low_x, high_x, low_y, high_y = self._view_bounds()
        i0 = max(0, min(nx - 1, int(np.floor(low_x * nx))))
        i1 = max(i0 + 1, min(nx, int(np.ceil(high_x * nx))))
        j0 = max(0, min(ny - 1, int(np.floor(low_y * ny))))
        j1 = max(j0 + 1, min(ny, int(np.ceil(high_y * ny))))
        return i0, i1, j0, j1

    def _point_to_view_fraction(
        self, point: QPointF, plot_rect: QRectF | None = None
    ) -> np.ndarray:
        rect = self._plot_rect() if plot_rect is None else plot_rect
        if rect.width() <= 0.0 or rect.height() <= 0.0:
            return self._view_center.copy()
        local_x = (float(point.x()) - rect.left()) / rect.width()
        local_y = 1.0 - (float(point.y()) - rect.top()) / rect.height()
        local = np.asarray([local_x, local_y], dtype=float)
        local = np.clip(local, 0.0, 1.0)
        low_x, _high_x, low_y, _high_y = self._view_bounds()
        return np.asarray([low_x, low_y], dtype=float) + local * self._view_fraction

    def _clamp_view(self) -> None:
        self._view_fraction = self._bounded_view_fraction(self._view_fraction)
        half = 0.5 * self._view_fraction
        self._view_center = np.clip(self._view_center, half, 1.0 - half)

    def _bounded_view_fraction(self, value: float) -> float:
        return max(self._MIN_VIEW_FRACTION, min(1.0, float(value)))

    def _zoom_at(self, point: QPointF, factor: float) -> None:
        if self._moments.size == 0:
            return
        old_fraction = self._view_fraction
        new_fraction = self._bounded_view_fraction(old_fraction * float(factor))
        if abs(new_fraction - old_fraction) <= 1e-12:
            return
        rect = self._plot_rect()
        anchor = self._point_to_view_fraction(point, rect)
        local_x = (float(point.x()) - rect.left()) / max(rect.width(), 1e-12)
        local_y = 1.0 - (float(point.y()) - rect.top()) / max(rect.height(), 1e-12)
        local = np.clip(np.asarray([local_x, local_y], dtype=float), 0.0, 1.0)
        self._view_fraction = new_fraction
        self._view_center = anchor - (local - 0.5) * new_fraction
        self._clamp_view()
        self.update()

    def _pan_by_pixels(self, delta: QPointF) -> None:
        rect = self._plot_rect()
        if rect.width() <= 0.0 or rect.height() <= 0.0:
            return
        self._view_center = self._pan_start_center + np.asarray(
            [
                -float(delta.x()) / rect.width() * self._view_fraction,
                float(delta.y()) / rect.height() * self._view_fraction,
            ],
            dtype=float,
        )
        self._clamp_view()
        self.update()

    def _zoom_to_box(self, start: QPointF, end: QPointF) -> None:
        rect = self._plot_rect()
        if abs(start.x() - end.x()) < 6.0 or abs(start.y() - end.y()) < 6.0:
            return
        first = self._point_to_view_fraction(start, rect)
        second = self._point_to_view_fraction(end, rect)
        low = np.minimum(first, second)
        high = np.maximum(first, second)
        span = max(float(high[0] - low[0]), float(high[1] - low[1]))
        if span <= 1e-12:
            return
        self._view_fraction = self._bounded_view_fraction(span)
        self._view_center = 0.5 * (low + high)
        self._clamp_view()
        self.update()

    def _background_values(self, moments: np.ndarray | None = None) -> tuple[np.ndarray, str]:
        values = self._moments if moments is None else moments
        if values.size == 0:
            return np.empty((0, 0), dtype=float), "none"
        z_values = values[..., 2]
        if float(np.max(np.abs(z_values))) > 1e-10:
            return z_values, "out_of_plane"
        return np.linalg.norm(values, axis=-1), "magnitude"

    def _moment_color(
        self,
        value: float,
        *,
        minimum: float,
        maximum: float,
        mode: str,
    ) -> QColor:
        palette = current_palette()
        if self._canvas_mode == "dark":
            if mode == "out_of_plane":
                span = max(abs(minimum), abs(maximum), 1e-12)
                t = max(-1.0, min(1.0, value / span))
                if t < 0.0:
                    return self._mix_color(QColor("#22D3EE"), QColor("#8B5CF6"), abs(t))
                return self._mix_color(QColor("#34D399"), QColor("#FACC15"), t)
            t = (value - minimum) / max(maximum - minimum, 1e-12)
            return self._mix_color(QColor("#22D3EE"), QColor("#FACC15"), t)
        if mode == "out_of_plane":
            span = max(abs(minimum), abs(maximum), 1e-12)
            t = max(-1.0, min(1.0, value / span))
            if t < 0.0:
                return self._mix_color(QColor("#1F6F78"), QColor("#3F2D5C"), abs(t))
            return self._mix_color(QColor("#5D9A57"), QColor(palette.seal_gold), t)
        t = (value - minimum) / max(maximum - minimum, 1e-12)
        return self._mix_color(
            QColor("#1F6F78"),
            QColor(palette.accent if palette.mode == "light" else palette.seal_gold),
            t,
        )

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        palette = current_palette()
        painter.fillRect(self.rect(), QColor(palette.bg_surface))
        plot_rect = self._plot_rect()
        plot_background = QColor("#111827" if self._canvas_mode == "dark" else "#FFFDF8")
        painter.fillRect(plot_rect, plot_background)
        frame_color = QColor("#3D5068" if self._canvas_mode == "dark" else palette.border)
        painter.setPen(QPen(frame_color, 1.1))
        painter.drawRect(plot_rect)
        if self._moments.size == 0:
            painter.setPen(QColor("#A09484" if self._canvas_mode == "dark" else palette.text_muted))
            painter.drawText(plot_rect, Qt.AlignmentFlag.AlignCenter, "Texture preview")
            return

        moments = self._display_moments()
        nx, ny = moments.shape[:2]
        background_values, background_mode = self._background_values(moments)
        min_value = float(np.min(background_values))
        max_value = float(np.max(background_values))
        in_plane = np.linalg.norm(moments[..., :2], axis=-1)
        max_in_plane = float(np.max(in_plane))
        if max_in_plane <= 1e-12:
            return
        cell_w = plot_rect.width() / max(1, nx)
        cell_h = plot_rect.height() / max(1, ny)
        step = max(1, int(max(nx, ny) / 31))
        arrow_spacing = min(cell_w, cell_h) * step
        arrow_scale = 0.44 * arrow_spacing / max_in_plane
        head = max(1.5, min(3.4, arrow_spacing * 0.14))
        for i in range(0, nx, step):
            for j in range(0, ny, step):
                moment = moments[i, j]
                color = self._moment_color(
                    float(background_values[i, j]),
                    minimum=min_value,
                    maximum=max_value,
                    mode=background_mode,
                )
                painter.setPen(QPen(color, max(0.9, min(1.45, arrow_spacing * 0.06))))
                painter.setBrush(color)
                start = QPointF(
                    plot_rect.left() + (i + 0.5) * cell_w,
                    plot_rect.top() + (ny - j - 0.5) * cell_h,
                )
                dx = float(moment[0]) * arrow_scale
                dy = -float(moment[1]) * arrow_scale
                end = QPointF(start.x() + dx, start.y() + dy)
                length = float(np.hypot(dx, dy))
                if length <= 1e-6:
                    painter.drawEllipse(start, 1.0, 1.0)
                    continue
                painter.drawLine(start, end)
                if length > 1e-6:
                    ux, uy = dx / length, dy / length
                    left = QPointF(
                        end.x() - head * (ux + 0.55 * uy), end.y() - head * (uy - 0.55 * ux)
                    )
                    right = QPointF(
                        end.x() - head * (ux - 0.55 * uy), end.y() - head * (uy + 0.55 * ux)
                    )
                    painter.drawPolygon(QPolygonF([end, left, right]))
        if self._box_start is not None and self._box_end is not None:
            painter.setPen(
                QPen(
                    QColor("#FFFFFF" if self._canvas_mode == "dark" else palette.accent),
                    1.3,
                    Qt.PenStyle.DashLine,
                )
            )
            painter.setBrush(Qt.BrushStyle.NoBrush)
            painter.drawRect(
                QRectF(self._box_start, self._box_end).normalized().intersected(plot_rect)
            )

    def wheelEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        delta = event.angleDelta().y()
        if delta == 0:
            return
        factor = 0.82 if delta > 0 else 1.0 / 0.82
        self._zoom_at(event.position(), factor)
        event.accept()

    def mousePressEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        if self._moments.size == 0:
            return
        position = event.position()
        if not self._plot_rect().contains(position):
            return
        if event.button() == Qt.MouseButton.RightButton or (
            event.button() == Qt.MouseButton.LeftButton
            and event.modifiers() & Qt.KeyboardModifier.ShiftModifier
        ):
            self._box_start = position
            self._box_end = position
            self.update()
            event.accept()
            return
        if event.button() == Qt.MouseButton.LeftButton:
            self._pan_start = position
            self._pan_start_center = self._view_center.copy()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
            event.accept()

    def mouseMoveEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        position = event.position()
        if self._box_start is not None:
            self._box_end = position
            self.update()
            event.accept()
            return
        if self._pan_start is not None:
            self._pan_by_pixels(position - self._pan_start)
            event.accept()

    def mouseReleaseEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        position = event.position()
        if self._box_start is not None:
            start = self._box_start
            self._box_start = None
            self._box_end = None
            self._zoom_to_box(start, position)
            event.accept()
            return
        if self._pan_start is not None:
            self._pan_start = None
            self.unsetCursor()
            event.accept()

    def mouseDoubleClickEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        if event.button() == Qt.MouseButton.LeftButton:
            self.reset_view(initial_view_fraction=self._initial_view_fraction)
            event.accept()


def _evaluate_texture_preview_grid(
    texture: MagneticTexture,
    *,
    mode: str = "custom",
    core_direction: Sequence[float] = (0.0, 0.0, -1.0),
    resolution: int = 73,
    repeats: float = 3.0,
) -> np.ndarray:
    """Evaluate a continuous 2D phase window for the texture preview."""

    axis = np.linspace(-0.5 * repeats, 0.5 * repeats, int(resolution), endpoint=True)
    xx, yy = np.meshgrid(axis, axis, indexing="ij")
    core = np.asarray(core_direction, dtype=float)
    handed_plot_sign = -1.0 if core.size >= 3 and float(core[2]) < 0.0 else 1.0
    if mode == "hexagonal_skyrmion":
        preview_q_basis = np.array(
            (
                (1.0, 0.0),
                (-0.5, handed_plot_sign * np.sqrt(3.0) / 2.0),
                (0.0, 0.0),
            ),
            dtype=float,
        )
    elif mode == "square_skyrmion":
        preview_q_basis = np.array(
            (
                (1.0, 0.0),
                (0.0, handed_plot_sign),
                (0.0, 0.0),
            ),
            dtype=float,
        )
    else:
        preview_q_basis = np.array(
            (
                (1.0, 0.0),
                (0.0, 1.0),
                (0.0, 0.0),
            ),
            dtype=float,
        )
    moments = np.zeros((axis.size, axis.size, 3), dtype=complex)
    for term in texture.terms:
        q = np.asarray(term.q_coeff, dtype=float)
        q_xy = q[:3] @ preview_q_basis
        q_dot = q_xy[0] * xx + q_xy[1] * yy
        moments += (
            term.coefficient.to_array()[np.newaxis, np.newaxis, :]
            * np.exp(-2j * np.pi * q_dot)[..., np.newaxis]
        )
    return moments.real


def _beside_square_cell(cell_axis):
    """Return an axes locator that keeps a colour bar beside a fixed-aspect plot, as tall as it."""

    def locate(colorbar_axis, _renderer) -> Bbox:
        cell_axis.apply_aspect()
        cell = cell_axis.get_position()
        reserved = colorbar_axis.get_position(original=True)
        gap = reserved.x0 - cell_axis.get_position(original=True).x1
        return Bbox.from_bounds(cell.x1 + gap, cell.y0, reserved.width, cell.height)

    return locate


class CornerHelpGroupBox(QGroupBox):
    """Group box with a help button positioned in the title band."""

    def __init__(self, title: str, help_button: QToolButton, parent: QWidget | None = None) -> None:
        super().__init__(title, parent)
        self._help_button = help_button
        self._help_button.setParent(self)
        self._help_button.raise_()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().resizeEvent(event)
        self._position_help_button()

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().showEvent(event)
        self._position_help_button()

    def _position_help_button(self) -> None:
        size = self._help_button.sizeHint()
        self._help_button.resize(size)
        self._help_button.move(max(0, self.width() - size.width() - 12), 0)


class MagneticTextureDialog(QDialog):
    """Dialog for creating a grouped multi-q spin texture."""

    _PREVIEW_RESOLUTION = 161
    _PREVIEW_RENDER_REPEATS = 8.0
    _PREVIEW_INITIAL_REPEATS = 3.0
    _CHARGE_RESOLUTION = 64
    _CHARGE_MAX_RESOLUTION = 256
    _ARROW_LEGEND = "Colour shows the out-of-plane moment; arrows show the in-plane moment."
    _MAGNITUDE_LEGEND = (
        "Colour shows the moment magnitude |m| over one magnetic cell, on the u-v plane "
        "of the normalization grid."
    )

    _MODES = (
        ("Hexagonal skyrmion", "hexagonal_skyrmion"),
        ("Square skyrmion", "square_skyrmion"),
        ("Custom Fourier texture", "custom"),
    )

    def __init__(
        self,
        structure: Structure,
        *,
        selected_indices: tuple[int, ...] = (),
        component: MagneticComponent | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(
            "Edit Magnetic Texture" if component is not None else "Add Magnetic Texture"
        )
        self.resize(1040, 680)
        self.setMinimumSize(860, 560)
        self._structure = structure
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
        self._mode_rows: list[tuple[QRadioButton, QFrame, QLabel]] = []
        self._preview_error: str | None = None

        self._label_edit = QLineEdit("Magnetic texture")
        self._label_edit.setMinimumWidth(360)
        self._mode_group = QButtonGroup(self)
        self._mode_buttons: dict[str, QRadioButton] = {}
        for label, key in self._MODES:
            button = QRadioButton(label)
            self._mode_group.addButton(button)
            self._mode_buttons[key] = button

        self._amplitude = self._decimal_input("1.0", minimum=0.0, maximum=1000.0, placeholder="1.0")
        self._q1 = self._direction_inputs((0.05, 0.0, 0.0), minimum=-10.0, maximum=10.0)
        self._q2 = self._direction_inputs((0.0, 0.05, 0.0), minimum=-10.0, maximum=10.0)
        self._q3 = self._direction_inputs((0.0, 0.0, 0.0), minimum=-10.0, maximum=10.0)
        self._core_direction = self._direction_inputs((0.0, 0.0, -1.0))
        self._cone_angle = self._decimal_input("90.0", minimum=0.0, maximum=180.0, placeholder="90")
        self._background = self._decimal_input(
            "0.0", minimum=-1000.0, maximum=1000.0, placeholder="0"
        )
        self._skyrmion_type = QComboBox()
        self._skyrmion_type.addItems(["Bloch", "Neel"])
        self._helicity = QComboBox()
        self._helicity.addItem("+1", 1)
        self._helicity.addItem("-1", -1)

        self._normalise_check = QCheckBox("Normalize moment magnitude")
        self._normalise_check.setToolTip(
            "Use Fourier modulation to reduce spatial variation in |m|."
        )
        self._grid_u = self._integer_input(32, 4, 128)
        self._grid_v = self._integer_input(32, 4, 128)
        self._grid_w = self._integer_input(1, 1, 64)
        self._normalise_threshold = self._decimal_input(
            "0.001", minimum=1e-8, maximum=1.0, placeholder="0.001"
        )
        self._prune_mode = QComboBox()
        self._prune_mode.addItems(["No pruning", "Coefficient threshold", "Keep strongest N"])
        self._prune_threshold = self._decimal_input(
            "0.001", minimum=0.0, maximum=1000.0, placeholder="0.001"
        )
        self._prune_top_n = self._integer_input(32, 1, 10000)
        self._magnitude_info_button = QToolButton()
        style_help_button(self._magnitude_info_button)
        self._magnitude_info_button.setIcon(icon("help"))
        self._magnitude_info_button.setIconSize(QSize(22, 22))
        self._magnitude_info_button.setToolTip("Explain magnitude normalization and pruning")
        self._magnitude_info_button.clicked.connect(self._show_magnitude_pruning_info)

        self._custom_table = MagneticComponentTable(
            0, 9, delete_callback=self._remove_selected_texture_terms
        )
        self._custom_table.setHorizontalHeaderLabels(
            ["q1", "q2", "q3", "Re x", "Re y", "Re z", "Im x", "Im y", "Im z"]
        )
        self._custom_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._custom_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._custom_table.itemChanged.connect(lambda *_: self._refresh_preview())
        self._configure_custom_table_columns()
        use_square_top_table_corners(self._custom_table)
        self._add_term_button = QPushButton("Add Fourier Term")
        self._add_term_button.clicked.connect(self._add_custom_term_row)

        self._target_group = QButtonGroup(self)
        self._all_atoms_radio = QRadioButton("All atoms")
        self._species_radio = QRadioButton("Element")
        self._equivalent_radio = QRadioButton("Equivalent site")
        self._selected_radio = QRadioButton("Current selection")
        for button in (
            self._all_atoms_radio,
            self._species_radio,
            self._equivalent_radio,
            self._selected_radio,
        ):
            self._target_group.addButton(button)
        self._species_combo = QComboBox()
        for species in sorted(set(str(symbol) for symbol in structure.species)):
            self._species_combo.addItem(species)
        self._equivalent_combo = QComboBox()
        for selector in viable_atom_selectors(structure):
            if selector.kind == "symmetry_equivalent" and selector.values:
                self._equivalent_combo.addItem(_selector_label(selector, structure), selector)

        self._preview = MagneticTexturePreview(self)
        self._preview_canvas_group = QButtonGroup(self)
        self._preview_light_button = self._preview_canvas_button("Light canvas", "appearance_light")
        self._preview_dark_button = self._preview_canvas_button("Dark canvas", "appearance_dark")
        self._preview_light_button.setChecked(True)
        self._preview_canvas_group.buttonClicked.connect(self._refresh_preview_canvas_mode)
        self._magnitude_map_check = QCheckBox("Show |m| map")
        self._magnitude_map_check.setToolTip(
            "Show the moment magnitude over one magnetic cell, sampled on the normalization grid."
        )
        self._magnitude_map_check.toggled.connect(self._refresh_preview_page)
        palette = current_palette()
        self._magnitude_figure, self._magnitude_canvas = create_plot_canvas(
            figsize=(4.0, 3.6), facecolor=palette.bg_surface, minimum_size=(300, 260)
        )
        self._magnitude_frame = PlotFrame(
            self._magnitude_canvas, title="Moment Magnitude", parent=self
        )
        self._magnitude_mesh = None
        self._charge_cache: tuple[object, str] | None = None
        self._preview_stack = QStackedWidget(self)
        self._preview_stack.addWidget(self._preview)
        self._preview_stack.addWidget(self._magnitude_frame)
        self._diagnostics = QLabel("Texture: -")
        self._diagnostics.setWordWrap(True)
        style_hint_text(self._diagnostics)

        self._apply_styles()
        self._build_layout()
        self._connect_preview_updates()
        self._refresh_target_options()
        self._mode_buttons["hexagonal_skyrmion"].setChecked(True)
        self._add_custom_term_row(values=(1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0))
        if component is not None:
            self._load_component(component)
        self._refresh_mode_controls()
        self._refresh_preview()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 16, 18, 16)
        layout.setSpacing(8)
        title = QLabel("Add Magnetic Texture", self)
        style_section_title(title)
        subtitle = QLabel(
            "Define a spin texture in propagation-vector coefficient space, preview it, then apply it to atoms.",
            self,
        )
        subtitle.setWordWrap(True)
        style_section_subtitle(subtitle)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        controls = QWidget(self)
        self._controls_panel = controls
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 8, 0)
        controls_layout.setSpacing(8)

        component_group = QGroupBox("Texture")
        component_form = QFormLayout(component_group)
        component_form.addRow("Label", self._label_edit)
        component_form.addRow("Amplitude", self._amplitude)
        controls_layout.addWidget(component_group)

        mode_group = QGroupBox("Texture Type")
        mode_layout = QGridLayout(mode_group)
        for column, (label, key) in enumerate(self._MODES):
            mode_layout.addWidget(
                self._build_mode_card(self._mode_buttons[key], self._mode_hint(key)), 0, column
            )
        controls_layout.addWidget(mode_group)

        parameter_row = QGridLayout()
        q_group = QGroupBox("Propagation Basis")
        q_layout = QVBoxLayout(q_group)
        self._q1_row = self._vector_row("q1 (r.l.u.)", self._q1)
        self._q2_row = self._vector_row("q2 (r.l.u.)", self._q2)
        self._q3_row = self._vector_row("q3 (r.l.u.)", self._q3)
        self._generated_q_hint = QLabel(
            "Skyrmion presets generate the remaining q vectors from q1, the core direction, and the selected lattice.",
            q_group,
        )
        self._generated_q_hint.setWordWrap(True)
        style_hint_text(self._generated_q_hint)
        q_layout.addWidget(self._q1_row)
        q_layout.addWidget(self._q2_row)
        q_layout.addWidget(self._q3_row)
        q_layout.addWidget(self._generated_q_hint)
        parameter_row.addWidget(q_group, 0, 0)

        self._skyrmion_group = QGroupBox("Skyrmion Parameters")
        skyrmion_layout = QVBoxLayout(self._skyrmion_group)
        self._core_direction_row = self._vector_row("Core direction", self._core_direction)
        self._rotation_row = self._field_row("Rotation", self._skyrmion_type)
        self._helicity_row = self._field_row("Helicity", self._helicity)
        self._cone_angle_row = self._field_row("Cone angle", self._cone_angle)
        self._square_background_row = self._field_row("Square background", self._background)
        skyrmion_layout.addWidget(self._core_direction_row)
        skyrmion_layout.addWidget(self._rotation_row)
        skyrmion_layout.addWidget(self._helicity_row)
        skyrmion_layout.addWidget(self._cone_angle_row)
        skyrmion_layout.addWidget(self._square_background_row)
        parameter_row.addWidget(self._skyrmion_group, 0, 1)
        parameter_row.setColumnStretch(0, 1)
        parameter_row.setColumnStretch(1, 1)
        controls_layout.addLayout(parameter_row)

        self._custom_group = QGroupBox("Custom Fourier Terms")
        custom_layout = QVBoxLayout(self._custom_group)
        custom_layout.addWidget(self._custom_table)
        custom_layout.addWidget(self._add_term_button)
        controls_layout.addWidget(self._custom_group)

        normalize_group = CornerHelpGroupBox("Magnitude and Pruning", self._magnitude_info_button)
        normalize_layout = QVBoxLayout(normalize_group)
        normalize_form = QFormLayout()
        normalize_form.addRow(self._normalise_check)
        grid_group = QWidget()
        style_inline_group(grid_group)
        grid_layout = QHBoxLayout(grid_group)
        grid_layout.setContentsMargins(0, 0, 0, 0)
        grid_layout.addWidget(QLabel("u"))
        grid_layout.addWidget(self._grid_u)
        grid_layout.addWidget(QLabel("v"))
        grid_layout.addWidget(self._grid_v)
        grid_layout.addWidget(QLabel("w"))
        grid_layout.addWidget(self._grid_w)
        normalize_form.addRow("Grid", grid_group)
        normalize_form.addRow("Threshold", self._normalise_threshold)
        normalize_form.addRow("Pruning", self._prune_mode)
        normalize_form.addRow("Prune threshold", self._prune_threshold)
        normalize_form.addRow("Keep N", self._prune_top_n)
        normalize_layout.addLayout(normalize_form)
        controls_layout.addWidget(normalize_group)

        target_group = QGroupBox("Apply Texture To")
        target_layout = QGridLayout(target_group)
        target_layout.addWidget(
            self._build_target_card(self._all_atoms_radio, "Apply the texture to every atom."), 0, 0
        )
        target_layout.addWidget(
            self._build_target_card(
                self._species_radio, "Apply to all atoms of one element.", self._species_combo
            ),
            0,
            1,
        )
        target_layout.addWidget(
            self._build_target_card(
                self._equivalent_radio,
                "Apply to one symmetry-equivalent atom site.",
                self._equivalent_combo,
            ),
            1,
            0,
        )
        target_layout.addWidget(
            self._build_target_card(
                self._selected_radio, "Apply to atoms selected in the builder preview."
            ),
            1,
            1,
        )
        controls_layout.addWidget(target_group)
        controls_layout.addStretch(1)

        self._controls_scroll = QScrollArea(self)
        self._controls_scroll.setWidgetResizable(True)
        self._controls_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._controls_scroll.setWidget(controls)

        preview_panel = QWidget(self)
        preview_layout = QVBoxLayout(preview_panel)
        preview_header = QHBoxLayout()
        preview_title = QLabel("Spin Texture Preview")
        preview_header.addWidget(preview_title)
        preview_header.addStretch(1)
        preview_header.addWidget(self._magnitude_map_check)
        preview_header.addWidget(self._preview_light_button)
        preview_header.addWidget(self._preview_dark_button)
        preview_layout.addLayout(preview_header)
        self._preview_legend = QLabel(self._ARROW_LEGEND)
        self._preview_legend.setWordWrap(True)
        style_hint_text(self._preview_legend)
        preview_layout.addWidget(self._preview_legend)
        preview_layout.addWidget(self._preview_stack, 1)
        preview_layout.addWidget(self._diagnostics)
        splitter.addWidget(self._controls_scroll)
        splitter.addWidget(preview_panel)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 2)
        layout.addWidget(splitter, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    @staticmethod
    def _mode_hint(key: str) -> str:
        hints = {
            "hexagonal_skyrmion": "Three q vectors separated by 120 degrees in the core-normal plane.",
            "square_skyrmion": "Two perpendicular q vectors in the core-normal plane, with optional uniform background.",
            "custom": "Enter arbitrary complex Fourier coefficients and local q coefficients.",
        }
        return hints[key]

    def _build_mode_card(self, button: QRadioButton, hint_text: str) -> QFrame:
        card = QFrame(self)
        style_option_card(card)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        button.setParent(card)
        hint = QLabel(hint_text, card)
        hint.setWordWrap(True)
        hint.setStyleSheet("background: transparent;")
        style_hint_text(hint)
        card_layout.addWidget(button)
        card_layout.addWidget(hint)
        self._mode_rows.append((button, card, hint))
        button.toggled.connect(self._refresh_mode_controls)
        return card

    def _build_target_card(
        self, button: QRadioButton, hint_text: str, editor: QWidget | None = None
    ) -> QFrame:
        card = QFrame(self)
        style_option_card(card)
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        button.setParent(card)
        hint = QLabel(hint_text, card)
        hint.setWordWrap(True)
        hint.setStyleSheet("background: transparent;")
        style_hint_text(hint)
        card_layout.addWidget(button)
        card_layout.addWidget(hint)
        if editor is not None:
            editor.setParent(card)
            card_layout.addWidget(editor)
        self._selector_rows.append((button, card, hint))
        button.toggled.connect(self._refresh_target_card_styles)
        return card

    def _preview_canvas_button(self, tooltip: str, icon_name: str) -> QToolButton:
        button = QToolButton(self)
        style_small_checked_tool_button(button)
        button.setIcon(icon(icon_name))
        button.setToolTip(tooltip)
        button.setCheckable(True)
        button.setAutoRaise(False)
        self._preview_canvas_group.addButton(button)
        return button

    @staticmethod
    def _decimal_input(
        value: str, *, minimum: float, maximum: float, placeholder: str
    ) -> QLineEdit:
        return MagneticComponentDialog._decimal_input(
            value, minimum=minimum, maximum=maximum, placeholder=placeholder
        )

    @staticmethod
    def _integer_input(value: int, minimum: int, maximum: int) -> QSpinBox:
        spinbox = QSpinBox()
        spinbox.setRange(minimum, maximum)
        spinbox.setValue(value)
        spinbox.setMinimumWidth(86)
        return spinbox

    def _direction_inputs(
        self,
        values: tuple[float, float, float],
        *,
        minimum: float = -100.0,
        maximum: float = 100.0,
    ) -> tuple[QLineEdit, QLineEdit, QLineEdit]:
        return MagneticComponentDialog._direction_inputs(values, minimum=minimum, maximum=maximum)

    @staticmethod
    def _direction_group(inputs: tuple[QLineEdit, QLineEdit, QLineEdit]) -> QWidget:
        return MagneticComponentDialog._direction_group(inputs)

    def _vector_row(self, label: str, inputs: tuple[QLineEdit, QLineEdit, QLineEdit]) -> QWidget:
        row = QWidget(self)
        style_inline_group(row)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        label_widget = QLabel(label, row)
        label_widget.setMinimumWidth(88)
        layout.addWidget(label_widget)
        layout.addWidget(self._direction_group(inputs), 1)
        return row

    def _field_row(self, label: str, editor: QWidget) -> QWidget:
        row = QWidget(self)
        style_inline_group(row)
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        label_widget = QLabel(label, row)
        label_widget.setMinimumWidth(88)
        layout.addWidget(label_widget)
        layout.addWidget(editor, 1)
        return row

    def _apply_styles(self) -> None:
        self.setStyleSheet(feature_dialog_stylesheet(include_scroll_area=True))

    def _connect_preview_updates(self) -> None:
        for edit in (
            self._label_edit,
            self._amplitude,
            self._cone_angle,
            self._background,
            self._normalise_threshold,
            self._prune_threshold,
            *self._q1,
            *self._q2,
            *self._q3,
            *self._core_direction,
        ):
            edit.textChanged.connect(lambda *_: self._refresh_preview())
        for widget in (self._skyrmion_type, self._helicity, self._prune_mode):
            widget.currentIndexChanged.connect(lambda *_: self._refresh_preview())
        for widget in (self._normalise_check,):
            widget.toggled.connect(lambda *_: self._refresh_preview())
        for widget in (self._grid_u, self._grid_v, self._grid_w, self._prune_top_n):
            widget.valueChanged.connect(lambda *_: self._refresh_preview())

    def _configure_custom_table_columns(self) -> None:
        header = self._custom_table.horizontalHeader()
        for column in range(self._custom_table.columnCount()):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Stretch)

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

    def _refresh_mode_controls(self, *_args) -> None:
        mode = self.mode()
        custom = mode == "custom"
        square = mode == "square_skyrmion"
        skyrmion = mode in {"hexagonal_skyrmion", "square_skyrmion"}
        for edit in (*self._q2, *self._q3):
            edit.setEnabled(custom)
        self._q2_row.setVisible(custom)
        self._q3_row.setVisible(custom)
        self._generated_q_hint.setVisible(not custom)
        self._custom_group.setVisible(custom)
        self._skyrmion_group.setVisible(skyrmion)
        self._square_background_row.setVisible(square)
        self._custom_table.setEnabled(custom)
        self._add_term_button.setEnabled(custom)
        self._background.setEnabled(square)
        for button, card, _hint in self._mode_rows:
            set_option_card_state(card, selected=button.isChecked())
        self._refresh_preview()

    def _refresh_preview_canvas_mode(self, *_args) -> None:
        self._preview.set_canvas_mode("dark" if self._preview_dark_button.isChecked() else "light")

    def _refresh_preview_page(self, *_args) -> None:
        show_map = self._magnitude_map_check.isChecked()
        if not show_map and self._magnitude_frame.is_popped_out():
            # A popped-out map would stop updating once hidden; bring it back first.
            self._magnitude_frame.show_here()
        self._preview_stack.setCurrentIndex(1 if show_map else 0)
        self._preview_legend.setText(self._MAGNITUDE_LEGEND if show_map else self._ARROW_LEGEND)
        for button in (self._preview_light_button, self._preview_dark_button):
            button.setVisible(not show_map)
        if show_map:
            self._refresh_preview()

    def _draw_magnitude_map(self, texture: MagneticTexture | None) -> None:
        palette = current_palette()
        self._magnitude_figure.clear()
        self._magnitude_mesh = None
        axis = self._magnitude_figure.add_subplot(111)
        style_plot_axes(axis)
        # Grid lines would draw over the colour map.
        axis.grid(False)
        axis.set_xlabel("Fraction of a₁")
        axis.set_ylabel("Fraction of a₂")
        if texture is not None and texture.terms:
            nu, nv, _nw = self._grid_resolution()
            magnitudes = np.linalg.norm(
                evaluate_texture_grid(texture, (nu, nv, 1))[:, :, 0, :], axis=-1
            )
            self._magnitude_mesh = axis.pcolormesh(
                np.linspace(0.0, 1.0, nu + 1),
                np.linspace(0.0, 1.0, nv + 1),
                magnitudes.T,
                shading="flat",
                cmap="viridis",
            )
            # The cell is drawn square. The layout reserves room for the colour bar
            # and its labels; at draw time the bar is resized to the cell's height.
            axis.set_aspect("equal")
            colorbar = self._magnitude_figure.colorbar(self._magnitude_mesh, ax=axis)
            colorbar.ax.set_box_aspect(None)
            colorbar.ax.set_axes_locator(_beside_square_cell(axis))
            colorbar.set_label("|m| (μB)", color=palette.text_primary)
            colorbar.ax.tick_params(colors=palette.text_primary)
            colorbar.outline.set_edgecolor(palette.border)
        self._magnitude_canvas.draw_idle()

    def _topological_charge_text(self, texture: MagneticTexture) -> str:
        q_basis = tuple(
            tuple(reciprocal_to_cartesian(row, self._structure.lattice_vectors))
            for row in self.q_basis_vectors()
        )
        # Edits that leave the texture unchanged (label, target, map toggle) reuse
        # the last result instead of resampling the cell.
        key = (texture, q_basis)
        if self._charge_cache is not None and self._charge_cache[0] == key:
            return self._charge_cache[1]
        try:
            charge = topological_charge_magnetic_unit_cell(
                texture,
                q_basis,
                (self._CHARGE_RESOLUTION, self._CHARGE_RESOLUTION),
                max_resolution=self._CHARGE_MAX_RESOLUTION,
            )
            text = f"Topological charge per magnetic cell: {charge}."
        except MagneticStructureError as exc:
            text = f"Topological charge per magnetic cell: undefined ({exc})."
        self._charge_cache = (key, text)
        return text

    def _show_magnitude_pruning_info(self) -> None:
        dialog = QDialog(self)
        dialog.setWindowTitle("Magnitude Normalization and Pruning")
        dialog.resize(760, 640)

        layout = QVBoxLayout(dialog)
        title = QLabel("Magnitude Normalization and Pruning", dialog)
        style_section_title(title)
        subtitle = QLabel(
            "Advanced Fourier-texture controls for reducing moment-length variation.", dialog
        )
        subtitle.setWordWrap(True)
        style_section_subtitle(subtitle)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        browser = QTextBrowser(dialog)
        browser.setOpenExternalLinks(False)
        browser.setHtml(self._magnitude_pruning_help_html())
        layout.addWidget(browser, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, dialog)
        buttons.rejected.connect(dialog.reject)
        layout.addWidget(buttons)
        dialog.exec()

    def _magnitude_pruning_help_html(self) -> str:
        grid = f"{int(self._grid_u.value())} × {int(self._grid_v.value())} × {int(self._grid_w.value())}"
        threshold = escape(self._normalise_threshold.text().strip() or "0.001")
        prune_threshold = escape(self._prune_threshold.text().strip() or "0.001")
        keep_n = int(self._prune_top_n.value())
        prune_mode = escape(self._prune_mode.currentText())
        return f"""
        <html>
        <body style="font-size: 13px;">
          <p>
            These controls are adapted from the spin-texture modulation workflow in <b>mag2spectra</b>.
            They are useful when a compact multi-q ansatz has the right winding pattern but the local
            moment length varies too strongly across the texture.
          </p>

          <h3>Why Normalize?</h3>
          <p>
            A finite Fourier texture is evaluated as
          </p>
          <p style="text-align: center; font-size: 15px;">
            <b>m</b>(<b>r</b>) =
            Re[Σ<sub>j</sub> <b>S</b><sub>j</sub> exp(-2πi <b>q</b><sub>j</sub>·<b>r</b>)].
          </p>
          <p>
            Linear superposition can make |<b>m</b>(<b>r</b>)| breathe from point to point.
            For a local-moment model, it is often more physical for the direction to vary while
            the moment magnitude remains approximately fixed.
          </p>

          <h3>What MuFinder Does</h3>
          <p>
            When <b>Normalize moment magnitude</b> is enabled, MuFinder samples the current texture
            on a regular reduced-coordinate grid ({grid} at the moment), estimates an inverse
            magnitude field, and Fourier-transforms that field:
          </p>
          <p style="text-align: center; font-size: 15px;">
            <b>m</b>'(<b>r</b>) = <b>m</b>(<b>r</b>) f(<b>r</b>),
            &nbsp;&nbsp; f(<b>r</b>) ≈ A / |<b>m</b>(<b>r</b>)|.
          </p>
          <p>
            The Fourier coefficients of f are convolved with the original texture coefficients,
            producing a richer Fourier texture whose sampled |<b>m</b>| is closer to the chosen
            amplitude A. The threshold currently shown here is <b>{threshold}</b>; smaller values
            keep more modulation harmonics and usually improve constancy, but generate more terms.
          </p>

          <h3>Pruning</h3>
          <p>
            Normalization can create many Fourier terms. Pruning is a deliberate approximation used
            to keep the model manageable for preview, storage, and later field calculations.
          </p>
          <ul>
            <li><b>Coefficient threshold</b>: remove terms with coefficient norm below {prune_threshold} μ<sub>B</sub>.</li>
            <li><b>Keep strongest N</b>: keep only the {keep_n} largest coefficient norms.</li>
          </ul>
          <p>
            Current pruning mode: <b>{prune_mode}</b>.
          </p>

          <h3>Practical Guidance</h3>
          <ul>
            <li>Leave normalization off while drafting simple helices or unmodulated skyrmion presets.</li>
            <li>Turn it on when the diagnostics show a large |m| range or the preview suggests moment collapse.</li>
            <li>Use moderate grids first; raising the grid resolution improves spectral fidelity but costs more.</li>
            <li>After normalization, inspect the diagnostics and preview before pruning aggressively.</li>
          </ul>

          <h3>Caveats</h3>
          <p>
            If the original texture has exact or near-zero |<b>m</b>| points, the ideal inverse
            magnitude is singular. MuFinder guards this numerically, but the result may need many
            high-order harmonics. A small uniform background component can improve conditioning.
          </p>
          <p>
            The result is an approximation, not an analytic proof of constant magnitude. Pruning
            may reintroduce some moment-length variation.
          </p>
        </body>
        </html>
        """

    def mode(self) -> str:
        for key, button in self._mode_buttons.items():
            if button.isChecked():
                return key
        return "hexagonal_skyrmion"

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

    def q_basis_vectors(self) -> tuple[tuple[float, float, float], ...]:
        mode = self.mode()
        if mode == "custom":
            return (
                self._direction_value(self._q1, "q1"),
                self._direction_value(self._q2, "q2"),
                self._direction_value(self._q3, "q3"),
            )
        lattice = "square" if mode == "square_skyrmion" else "hexagonal"
        return skyrmion_q_basis_rlu(
            self._structure,
            q1_rlu=self._direction_value(self._q1, "q1"),
            core_direction=self._unit_direction(self._core_direction, "core direction"),
            lattice=lattice,
        )

    def texture(self) -> MagneticTexture:
        base = self._base_texture()
        if self._normalise_check.isChecked():
            base = normalize_texture_magnitude(
                base,
                grid_resolution=self._grid_resolution(),
                threshold=self._decimal_value(self._normalise_threshold, "normalization threshold"),
                target_magnitude=self._decimal_value(self._amplitude, "amplitude"),
            )
        mode = self._prune_mode.currentText()
        if mode == "Coefficient threshold":
            base = prune_texture_terms(
                base, threshold=self._decimal_value(self._prune_threshold, "prune threshold")
            )
        elif mode == "Keep strongest N":
            base = prune_texture_terms(base, top_n=int(self._prune_top_n.value()))
        return base

    def component(
        self,
        q_coeff_transform: Sequence[Sequence[float]],
        q_basis_rlu: Sequence[Sequence[float]],
    ) -> MagneticComponent:
        texture = self.texture()
        metadata = self._texture_metadata(texture, q_basis_rlu)
        return build_texture_component(
            label=self._label_edit.text().strip() or "Magnetic texture",
            selector=self.selector(),
            texture=texture,
            q_coeff_transform=q_coeff_transform,
            phase_turns=0.0,
            component_id=self._editing_component_id,
            metadata=metadata,
        )

    def _base_texture(self) -> MagneticTexture:
        mode = self.mode()
        if mode == "custom":
            return MagneticTexture(tuple(self._custom_terms())).combine_terms()
        q_basis = self.q_basis_vectors()
        q_dirs = self._q_basis_directions(q_basis)
        kwargs = {
            "magnitude": self._decimal_value(self._amplitude, "amplitude"),
            "core_direction": self._unit_direction(self._core_direction, "core direction"),
            "cone_angle_degrees": self._decimal_value(self._cone_angle, "cone angle"),
            "skyrmion_type": self._skyrmion_type.currentText().lower(),
            "helicity": int(self._helicity.currentData()),
        }
        if mode == "square_skyrmion":
            return build_square_skyrmion_texture(
                q_directions=q_dirs[:2],
                ferromagnetic_background=self._decimal_value(self._background, "square background"),
                **kwargs,
            )
        return build_hexagonal_skyrmion_texture(
            q_directions=(q_dirs[0], q_dirs[1], -(q_dirs[0] + q_dirs[1])), **kwargs
        )

    def _q_basis_directions(
        self, q_basis: Sequence[Sequence[float]]
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        reciprocal = np.linalg.inv(np.asarray(self._structure.lattice_vectors, dtype=float)).T
        dirs = []
        for row in q_basis:
            cart = np.asarray(row, dtype=float) @ reciprocal
            norm = float(np.linalg.norm(cart))
            dirs.append(cart / norm if norm > 1e-12 else np.zeros(3, dtype=float))
        return (dirs[0], dirs[1], dirs[2])

    def _texture_metadata(
        self, texture: MagneticTexture, q_basis_rlu: Sequence[Sequence[float]]
    ) -> dict[str, object]:
        before = texture_diagnostics(self._base_texture(), self._grid_resolution())
        after = texture_diagnostics(texture, self._grid_resolution())
        return {
            "type": "texture",
            "texture_mode": self.mode(),
            "q_basis_rlu": [[float(x) for x in row] for row in q_basis_rlu],
            "texture": texture.to_json(),
            "builder": {
                "amplitude": self._decimal_value(self._amplitude, "amplitude"),
                "q_basis_vectors": [[float(x) for x in row] for row in self.q_basis_vectors()],
                "core_direction": [
                    self._decimal_value(edit, "core direction component")
                    for edit in self._core_direction
                ],
                "cone_angle_degrees": self._decimal_value(self._cone_angle, "cone angle"),
                "skyrmion_type": self._skyrmion_type.currentText().lower(),
                "helicity": int(self._helicity.currentData()),
                "square_background": self._decimal_value(self._background, "square background"),
            },
            "normalization": {
                "enabled": bool(self._normalise_check.isChecked()),
                "grid_resolution": list(self._grid_resolution()),
                "threshold": self._decimal_value(
                    self._normalise_threshold, "normalization threshold"
                ),
            },
            "pruning": {
                "mode": self._prune_mode.currentText(),
                "threshold": self._decimal_value(self._prune_threshold, "prune threshold"),
                "top_n": int(self._prune_top_n.value()),
            },
            "diagnostics": {"before": asdict(before), "after": asdict(after)},
        }

    def _custom_terms(self) -> tuple[MagneticTextureTerm, ...]:
        terms: list[MagneticTextureTerm] = []
        for row in range(self._custom_table.rowCount()):
            values = [self._table_decimal(row, column) for column in range(9)]
            coefficient = np.asarray(values[3:6], dtype=float) + 1j * np.asarray(
                values[6:9], dtype=float
            )
            if np.allclose(coefficient, 0.0):
                continue
            terms.append(MagneticTextureTerm.from_vector(values[:3], coefficient))
        if not terms:
            raise ValueError("custom texture requires at least one non-zero Fourier term")
        return tuple(terms)

    def _add_custom_term_row(self, *, values: tuple[float, ...] | None = None) -> None:
        values = values or (0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0)
        row = self._custom_table.rowCount()
        self._custom_table.insertRow(row)
        for column, value in enumerate(values):
            item = QTableWidgetItem(f"{float(value):.8g}")
            self._custom_table.setItem(row, column, item)
        self._refresh_preview()

    def _remove_selected_texture_terms(self) -> None:
        rows = sorted(
            {index.row() for index in self._custom_table.selectionModel().selectedRows()},
            reverse=True,
        )
        for row in rows:
            self._custom_table.removeRow(row)
        self._refresh_preview()

    def _load_component(self, component: MagneticComponent) -> None:
        self._label_edit.setText(component.label)
        self._set_selector(component.selector)
        metadata = dict(component.metadata)
        mode = str(metadata.get("texture_mode", "custom"))
        builder = metadata.get("builder")
        has_builder = isinstance(builder, Mapping)
        if not has_builder and mode in {"hexagonal_skyrmion", "square_skyrmion"}:
            mode = "custom"
        if mode in self._mode_buttons:
            self._mode_buttons[mode].setChecked(True)
        q_basis = builder.get("q_basis_vectors") if has_builder else metadata.get("q_basis_rlu")
        if isinstance(q_basis, Sequence) and len(q_basis) >= 3:
            for inputs, row in zip((self._q1, self._q2, self._q3), q_basis[:3], strict=True):
                if isinstance(row, Sequence):
                    self._set_direction_inputs(inputs, row)

        if has_builder:
            self._load_texture_builder_metadata(builder)

        texture = (
            MagneticTexture.from_json(metadata.get("texture", {}))
            if has_builder
            else MagneticTexture(
                tuple(
                    MagneticTextureTerm.from_vector(term.q_coeff, term.coefficient.to_array())
                    for term in component.terms
                )
            ).combine_terms()
        )
        if texture.terms:
            self._custom_table.setRowCount(0)
            for term in texture.terms:
                fc = term.coefficient
                self._add_custom_term_row(values=(*term.q_coeff, *fc.real, *fc.imaginary))
        normalization = metadata.get("normalization")
        if has_builder and isinstance(normalization, Mapping):
            self._normalise_check.setChecked(bool(normalization.get("enabled", False)))
            grid = normalization.get("grid_resolution")
            if isinstance(grid, Sequence) and len(grid) >= 3:
                self._grid_u.setValue(int(grid[0]))
                self._grid_v.setValue(int(grid[1]))
                self._grid_w.setValue(int(grid[2]))
            self._set_line_edit_from_metadata(
                self._normalise_threshold, normalization.get("threshold")
            )
        pruning = metadata.get("pruning")
        if has_builder and isinstance(pruning, Mapping):
            mode_text = str(pruning.get("mode", "No pruning"))
            index = self._prune_mode.findText(mode_text)
            if index >= 0:
                self._prune_mode.setCurrentIndex(index)
            self._set_line_edit_from_metadata(self._prune_threshold, pruning.get("threshold"))
            if pruning.get("top_n") is not None:
                self._prune_top_n.setValue(int(pruning.get("top_n")))

    def _load_texture_builder_metadata(self, builder: Mapping[str, object]) -> None:
        self._set_line_edit_from_metadata(self._amplitude, builder.get("amplitude"))
        self._set_direction_from_metadata(self._core_direction, builder.get("core_direction"))
        self._set_line_edit_from_metadata(self._cone_angle, builder.get("cone_angle_degrees"))
        self._set_line_edit_from_metadata(self._background, builder.get("square_background"))
        skyrmion_type = str(builder.get("skyrmion_type", "")).strip().lower()
        for index in range(self._skyrmion_type.count()):
            if self._skyrmion_type.itemText(index).lower() == skyrmion_type:
                self._skyrmion_type.setCurrentIndex(index)
                break
        helicity = builder.get("helicity")
        if helicity is not None:
            index = self._helicity.findData(int(helicity))
            if index >= 0:
                self._helicity.setCurrentIndex(index)

    def _set_direction_from_metadata(
        self, inputs: tuple[QLineEdit, QLineEdit, QLineEdit], values: object
    ) -> None:
        if (
            isinstance(values, Sequence)
            and not isinstance(values, (str, bytes))
            and len(values) >= 3
        ):
            self._set_direction_inputs(inputs, values[:3])

    @staticmethod
    def _set_line_edit_from_metadata(edit: QLineEdit, value: object) -> None:
        if value is None:
            return
        edit.setText(f"{float(value):.8g}")

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

    def _refresh_preview(self) -> None:
        try:
            texture = self.texture()
            moments = _evaluate_texture_preview_grid(
                texture,
                mode=self.mode(),
                core_direction=self._unit_direction(self._core_direction, "core direction"),
                resolution=self._PREVIEW_RESOLUTION,
                repeats=self._PREVIEW_RENDER_REPEATS,
            )
            self._preview.set_moments(
                moments,
                initial_view_fraction=self._PREVIEW_INITIAL_REPEATS / self._PREVIEW_RENDER_REPEATS,
            )
            before = texture_diagnostics(self._base_texture(), self._grid_resolution())
            after = texture_diagnostics(texture, self._grid_resolution())
            self._diagnostics.setText(
                "Texture: "
                f"{before.term_count} term(s) -> {after.term_count} term(s); "
                f"|m| mean {after.magnitude_mean:.4g} μB, std {after.magnitude_std:.4g}, "
                f"range {after.magnitude_min:.4g}-{after.magnitude_max:.4g}.\n"
                f"{self._topological_charge_text(texture)}"
            )
            if self._magnitude_map_check.isChecked():
                self._draw_magnitude_map(texture)
            self._preview_error = None
        except Exception as exc:
            self._preview.set_moments(None)
            if self._magnitude_map_check.isChecked():
                self._draw_magnitude_map(None)
            self._diagnostics.setText(f"Texture preview unavailable: {exc}")
            self._preview_error = str(exc)

    def accept(self) -> None:  # noqa: D401 - Qt API override.
        """Accept only if the texture can be generated."""

        try:
            self.q_basis_vectors()
            self.texture()
        except Exception as exc:
            QMessageBox.warning(self, "Invalid magnetic texture", str(exc))
            return
        super().accept()

    def _grid_resolution(self) -> tuple[int, int, int]:
        return (int(self._grid_u.value()), int(self._grid_v.value()), int(self._grid_w.value()))

    def _direction_value(
        self, inputs: tuple[QLineEdit, QLineEdit, QLineEdit], label: str
    ) -> tuple[float, float, float]:
        return tuple(self._decimal_value(edit, f"{label} component") for edit in inputs)  # type: ignore[return-value]

    def _unit_direction(
        self, inputs: tuple[QLineEdit, QLineEdit, QLineEdit], label: str
    ) -> np.ndarray:
        vector = np.asarray(self._direction_value(inputs, label), dtype=float)
        norm = float(np.linalg.norm(vector))
        if norm <= _DIRECTION_TOLERANCE:
            raise ValueError(f"{label} must be a non-zero direction")
        return vector / norm

    def _table_decimal(self, row: int, column: int) -> float:
        item = self._custom_table.item(row, column)
        if item is None or not item.text().strip():
            raise ValueError(f"custom term row {row + 1}, column {column + 1} is required")
        return float(item.text())

    @staticmethod
    def _decimal_value(edit: QLineEdit, label: str) -> float:
        return MagneticComponentDialog._decimal_value(edit, label)

    @staticmethod
    def _set_direction_inputs(
        inputs: tuple[QLineEdit, QLineEdit, QLineEdit], values: Sequence[float]
    ) -> None:
        MagneticComponentDialog._set_direction_inputs(inputs, values)


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


class MagneticStructureBuilderDialog(QDialog):
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

        self._q_basis_table = _create_q_basis_table(editable=True)
        self._q_basis_table.itemChanged.connect(self._on_q_basis_item_changed)
        self._q_basis_hint = QLabel(
            "Each component stores its q as a combination of these rows. Double-click a q vector to edit it; "
            "every component that uses the row follows it. Rows used by a texture change through Edit...."
        )
        self._q_basis_hint.setWordWrap(True)
        style_hint_text(self._q_basis_hint)

        self._add_button = QPushButton("Add Component")
        self._add_button.clicked.connect(self._add_component)
        self._add_preset_button = QPushButton("Add Preset...")
        self._add_preset_button.setToolTip(
            "Build a helix, conical, cycloid or spin-density wave from a propagation direction and crystal axes"
        )
        self._add_preset_button.clicked.connect(self._add_preset)
        self._add_texture_button = QPushButton("Add Texture...")
        self._add_texture_button.clicked.connect(self._add_texture)
        self._edit_button = QPushButton("Edit...")
        self._edit_button.clicked.connect(self._edit_selected_component)
        self._duplicate_button = QPushButton("Duplicate")
        self._duplicate_button.setToolTip(
            "Add a copy of each selected component below its original"
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

        self._viewer = AtomicStructureViewer()
        self._viewer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self._style_viewer_toggles()
        self._viewer_shutdown = False

        self._build_layout()
        self._sync_enabled_state()

    def _style_viewer_toggles(self) -> None:
        for checkbox in (
            self._viewer._show_cell,
            self._viewer._show_bonds,
            self._viewer._show_spins,
        ):
            checkbox.setObjectName("magneticBuilderViewerCheckBox")
            checkbox.setStyleSheet(
                "QCheckBox#magneticBuilderViewerCheckBox {"
                "background: transparent;"
                "spacing: 9px;"
                "padding-left: 2px;"
                "padding-right: 6px;"
                "}"
                "QCheckBox#magneticBuilderViewerCheckBox::indicator {"
                "width: 18px;"
                "height: 18px;"
                "}"
            )

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

    def keyPressEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        # Enter reaches the dialog after a preview-repeat spin box commits its value; it must not
        # press the default "Use Structure" button and close the builder.
        if event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter):
            event.accept()
            return
        super().keyPressEvent(event)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        self._shutdown_viewer()
        super().closeEvent(event)

    def set_document(self, document: ProjectDocument | None) -> None:
        self._document = document
        self._host_structure = _active_host_structure(document)
        self._preview_atom_indices = np.empty((0,), dtype=int)
        self._model = self._model_from_document(document, self._host_structure)
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

    def _model_from_document(
        self,
        document: ProjectDocument | None,
        host: Structure | None,
    ) -> MagneticStructureModel:
        if document is None or host is None:
            return MagneticStructureModel()
        payload = document.project.magnetic_structure
        if not isinstance(payload, dict):
            payload = host.properties.get("magnetic_structure")
        if isinstance(payload, dict) and "components" in payload:
            return MagneticStructureModel.from_json(payload)
        return magnetic_model_from_legacy_structure(host)

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
            self._set_item(row, 3, _selector_label(component.selector, self._host_structure))
            self._set_item(row, 4, _component_q_label(self._model, component))
            self._set_item(row, 5, _component_moment_label(component))
            self._style_component_row(row, component.enabled)
        self._component_table.resizeRowsToContents()
        self._populate_q_basis()
        self._updating = False

    def _populate_q_basis(self) -> None:
        """Show each stored q-basis row, the components that use it, and whether it can be edited here."""

        table = self._q_basis_table
        _fill_q_basis_table(table, self._model)
        for slot in q_basis_slots(self._model):
            vector_item = table.item(slot.index, self._Q_BASIS_VECTOR_COLUMN)
            if not slot.texture_component_ids:
                vector_item.setFlags(vector_item.flags() | Qt.ItemFlag.ItemIsEditable)
            if slot.texture_component_ids:
                vector_item.setToolTip(
                    "Used by a texture, whose moment directions follow its q vectors: edit the texture to change it"
                )
            elif slot.direction_component_ids:
                vector_item.setToolTip(
                    "Double-click to change the wavelength: a helix, cycloid or conical component uses this row, "
                    "so only edits that keep its wavevector directions are accepted"
                )
            elif slot.in_use:
                vector_item.setToolTip(
                    "Double-click to edit; every component that uses this row changes wavevector with it"
                )
            else:
                vector_item.setToolTip(
                    "Unused: double-click to edit. Adding a component with a new q fills an unused row"
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
                    row_item.setToolTip("Used by the selected component")
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
        item.setText(_q_vector_text(self._model.q_basis_rlu[index]))
        for row, component in enumerate(self._model.components):
            q_item = self._component_table.item(row, 4)
            if q_item is not None:
                q_item.setText(_component_q_label(self._model, component))
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
            q_vector=_component_edit_q_vector(self._model, component),
            model=self._model,
            q_coeff=_component_edit_q_coeff(component),
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
        q = np.asarray(q_vector, dtype=float)
        basis = list(self._model.q_basis_rlu)
        if np.allclose(q, 0.0, atol=1e-12):
            return tuple(basis), (0.0, 0.0, 0.0)
        used_indices = self._used_q_basis_indices(ignored_component_ids=ignored_component_ids)
        for index, row in enumerate(basis):
            if index in used_indices and np.allclose(row, q, atol=1e-9):
                return tuple(basis), _unit_q_coeff(index)
        for index, row in enumerate(basis):
            if index not in used_indices and np.allclose(row, q, atol=1e-9):
                return tuple(basis), _unit_q_coeff(index)
        for index in range(len(basis)):
            if index not in used_indices:
                basis[index] = tuple(float(value) for value in q)
                return tuple(basis), _unit_q_coeff(index)
        return tuple(basis), None

    def _q_basis_for_texture(
        self,
        q_vectors: Sequence[Sequence[float]],
        *,
        ignored_component_ids: set[str] | None = None,
    ) -> tuple[
        tuple[tuple[float, float, float], ...], tuple[tuple[float, float, float], ...] | None
    ]:
        basis = [tuple(float(value) for value in row) for row in self._model.q_basis_rlu]
        used_indices = self._used_q_basis_indices(ignored_component_ids=ignored_component_ids)
        transform_rows: list[tuple[float, float, float]] = []
        allocated: set[int] = set()
        for q_vector in q_vectors[:3]:
            q = np.asarray(q_vector, dtype=float)
            if np.allclose(q, 0.0, atol=1e-12):
                transform_rows.append((0.0, 0.0, 0.0))
                continue
            matched = None
            for index, row in enumerate(basis):
                if np.allclose(row, q, atol=1e-9, rtol=0.0):
                    matched = index
                    break
            if matched is None:
                for index in range(3):
                    if index not in used_indices and index not in allocated:
                        matched = index
                        basis[index] = tuple(float(value) for value in q)
                        break
            if matched is None:
                return tuple(basis), None
            allocated.add(matched)
            transform_rows.append(_unit_q_coeff(matched))
        while len(transform_rows) < 3:
            transform_rows.append((0.0, 0.0, 0.0))
        return tuple(basis), tuple(transform_rows)

    def _used_q_basis_indices(self, *, ignored_component_ids: set[str] | None = None) -> set[int]:
        used: set[int] = set()
        for component in self._model.components:
            if (
                ignored_component_ids is not None
                and component.component_id in ignored_component_ids
            ):
                continue
            used.update(component_q_basis_indices(component))
        return used

    def _build_from_host_spins(self) -> None:
        if self._host_structure is None:
            return
        count = len(self._model.components)
        if count and not workflow_table_utils.confirm_destructive(
            self,
            "Replace magnetic structure",
            f"Replace the {workflow_table_utils.count_phrase(count, ('component', 'components'))} "
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
            preview_structure, preview_atom_indices = _preview_supercell_structure(
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
        except Exception as exc:  # pragma: no cover - defensive UI guard
            self._viewer.set_moment_vectors(None)
            self._preview_status.setText(f"Preview unavailable: {exc}")
            self._classification.setText("Structure type: -")


class MuonSiteSelectionDialog(QDialog):
    """Dialog for choosing the muon site used by magnetic-field calculations.

    Structural-relaxation clusters list the sites pinned to the Muon Sites
    summary unless "Show every relaxed site" is on, which lists every analysis
    workspace's cluster representatives.
    """

    open_muon_sites_requested = Signal()

    def __init__(
        self,
        document: ProjectDocument | None,
        host_structure: Structure | None,
        current_site: MagneticMuonSiteOption | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Select Muon Site")
        self.resize(820, 420)
        self._document = document
        self._host_structure = host_structure
        self._site_options: list[MagneticMuonSiteOption] = []
        self._accepted_site: MagneticMuonSiteOption | None = None
        self._current_site = current_site

        self._site_source_combo = QComboBox()
        self._site_source_combo.addItem(_site_source_label(UEP_SITE_SOURCE), UEP_SITE_SOURCE)
        self._site_source_combo.addItem("Structural relaxation clustering", "relaxation_clustering")
        self._site_source_combo.addItem("Custom fractional position", "custom")
        self._site_source_combo.currentIndexChanged.connect(self._refresh_muon_site_selection)
        self._estimate_note = QLabel(_UNRELAXED_ESTIMATE_NOTE)
        self._estimate_note.setWordWrap(True)
        style_hint_text(self._estimate_note)

        self._site_table = create_workflow_table(
            ("Site", "Source", "Cluster", "fx", "fy", "fz"),
            selection_mode=QAbstractItemView.SelectionMode.SingleSelection,
            stretch_last_section=False,
        )
        self._site_table.itemSelectionChanged.connect(self._on_site_selection_changed)
        self._configure_site_table_columns()
        self._site_scope = PinnedSitesScopeBar(self)
        self._site_scope.show_every_site_toggled.connect(self._on_show_every_site_toggled)
        self._site_scope.open_muon_sites_requested.connect(self._on_open_muon_sites_requested)

        self._custom_x = self._fractional_position_input("0.0")
        self._custom_y = self._fractional_position_input("0.0")
        self._custom_z = self._fractional_position_input("0.0")
        for edit in (self._custom_x, self._custom_y, self._custom_z):
            edit.textChanged.connect(lambda *_: self._refresh_site_status())
        self._site_stack = QStackedWidget()
        self._site_status = QLabel("Muon site: -")
        self._site_status.setWordWrap(True)
        style_hint_text(self._site_status)
        self._distortion_checkbox = QCheckBox(
            "Correct the local field for muon-induced displacements"
        )
        self._distortion_checkbox.setToolTip(
            "Adds the finite-cell change in the dipolar field, and in the contact field when a hyperfine "
            "coupling is set, from the host-ion displacements in the representative relaxed structure. "
            "Available for structural-relaxation clusters whose relaxed structure is present."
        )
        self._distortion_requested = bool(
            current_site is not None and current_site.include_distortion
        )
        self._distortion_checkbox.toggled.connect(self._on_distortion_toggled)

        self._apply_dialog_styles()
        self._build_layout()
        self._set_current_site(current_site)
        self._refresh_muon_site_selection()

    def selected_muon_site(self) -> MagneticMuonSiteOption | None:
        source = self._site_source()
        if source == "custom":
            if self._host_structure is None:
                return None
            fractional = tuple(
                _decimal_edit_value(edit, label)
                for edit, label in (
                    (self._custom_x, "custom fx"),
                    (self._custom_y, "custom fy"),
                    (self._custom_z, "custom fz"),
                )
            )
            cartesian = _fractional_to_cartesian(fractional, self._host_structure)
            return MagneticMuonSiteOption(
                source="custom",
                label="Custom site",
                fractional_position=fractional,  # type: ignore[arg-type]
                cartesian_position=cartesian,
            )
        option = self._selected_site_option()
        if option is None:
            return None
        include = self._distortion_checkbox.isEnabled() and self._distortion_checkbox.isChecked()
        return replace(option, include_distortion=include)

    def accept(self) -> None:
        try:
            site = self.selected_muon_site()
        except ValueError as exc:
            QMessageBox.warning(self, "Invalid Muon Site", str(exc))
            return
        if site is None:
            QMessageBox.warning(self, "Select Muon Site", "Select a muon site before continuing.")
            return
        if self._host_structure is not None:
            clearance_error = muon_site_clearance_error(
                self._host_structure, site.fractional_position, label=site.label
            )
            if clearance_error is not None:
                QMessageBox.warning(self, "Invalid Muon Site", clearance_error)
                return
        self._accepted_site = site
        super().accept()

    def accepted_site(self) -> MagneticMuonSiteOption | None:
        return self._accepted_site

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        source_row = QHBoxLayout()
        source_row.addWidget(QLabel("Source"))
        source_row.addWidget(self._site_source_combo, 1)
        layout.addLayout(source_row)
        layout.addWidget(self._estimate_note)
        layout.addWidget(self._site_scope)

        table_page = QWidget()
        table_layout = QVBoxLayout(table_page)
        table_layout.setContentsMargins(0, 0, 0, 0)
        table_layout.addWidget(self._site_table)
        self._site_stack.addWidget(table_page)

        custom_page = QWidget()
        custom_page.setObjectName("magneticMuonSiteInlinePanel")
        custom_layout = QHBoxLayout(custom_page)
        custom_layout.setContentsMargins(0, 0, 0, 0)
        custom_layout.addWidget(QLabel("fx"))
        custom_layout.addWidget(self._custom_x)
        custom_layout.addWidget(QLabel("fy"))
        custom_layout.addWidget(self._custom_y)
        custom_layout.addWidget(QLabel("fz"))
        custom_layout.addWidget(self._custom_z)
        custom_layout.addStretch(1)
        self._site_stack.addWidget(custom_page)

        layout.addWidget(self._site_stack, 1)
        layout.addWidget(self._distortion_checkbox)
        layout.addWidget(self._site_status)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _apply_dialog_styles(self) -> None:
        palette = current_palette()
        self.setStyleSheet(
            "QRadioButton, QCheckBox {"
            "background: transparent;"
            "spacing: 10px;"
            "padding-top: 3px;"
            "padding-bottom: 3px;"
            "}"
            "QWidget#magneticMuonSiteInlinePanel {"
            "background: transparent;"
            "}"
            "QStackedWidget, QStackedWidget > QWidget, QLabel {"
            "background: transparent;"
            "}"
            "QCheckBox:disabled {"
            f"color: {palette.text_muted};"
            "}"
        )

    def _set_current_site(self, site: MagneticMuonSiteOption | None) -> None:
        if site is None:
            # Relaxed sites come before unrelaxed UEP estimates once the relaxation list has any.
            source = (
                "relaxation_clustering"
                if self._site_options_for_source("relaxation_clustering")
                else UEP_SITE_SOURCE
            )
            self._site_source_combo.setCurrentIndex(self._site_source_combo.findData(source))
            return
        index = self._site_source_combo.findData(site.source)
        if index >= 0:
            self._site_source_combo.setCurrentIndex(index)
        if site.source == "custom":
            self._custom_x.setText(f"{site.fractional_position[0]:.8g}")
            self._custom_y.setText(f"{site.fractional_position[1]:.8g}")
            self._custom_z.setText(f"{site.fractional_position[2]:.8g}")

    def _site_source(self) -> str:
        return str(self._site_source_combo.currentData() or UEP_SITE_SOURCE)

    def _refresh_muon_site_selection(self) -> None:
        source = self._site_source()
        self._site_stack.setCurrentIndex(1 if source == "custom" else 0)
        self._estimate_note.setVisible(is_unrelaxed_site_source(source))
        self._site_options = self._site_options_for_source(source)
        self._refresh_site_scope(source)
        self._populate_site_table()
        self._sync_distortion_controls()
        self._refresh_site_status()

    def _site_options_for_source(self, source: str) -> list[MagneticMuonSiteOption]:
        if self._document is None or self._host_structure is None:
            return []
        if source == UEP_SITE_SOURCE:
            return _uep_cluster_muon_site_options(self._document, self._host_structure)
        if source == "relaxation_clustering":
            return _relaxation_cluster_muon_site_options(
                self._document,
                self._host_structure,
                pinned_only=not show_every_relaxed_site(self._document, MAGNETIC_FIELDS_SCOPE),
            )
        return []

    def _refresh_site_scope(self, source: str) -> None:
        self._site_scope.setVisible(source == "relaxation_clustering")
        show_every = show_every_relaxed_site(self._document, MAGNETIC_FIELDS_SCOPE)
        notice = "" if show_every or has_pinned_sites(self._document) else NO_PINNED_SITES_NOTICE
        self._site_scope.set_state(show_every_site=show_every, notice=notice)

    def _on_show_every_site_toggled(self, enabled: bool) -> None:
        set_show_every_relaxed_site(self._document, MAGNETIC_FIELDS_SCOPE, enabled)
        self._refresh_muon_site_selection()

    def _on_open_muon_sites_requested(self) -> None:
        self.open_muon_sites_requested.emit()
        self.reject()

    def _populate_site_table(self) -> None:
        self._site_table.setRowCount(0)
        for option in self._site_options:
            row = self._site_table.rowCount()
            self._site_table.insertRow(row)
            values = [
                option.label,
                option.source_label or _site_source_label(option.source),
                option.cluster_label or "-",
                f"{option.fractional_position[0]:.5f}",
                f"{option.fractional_position[1]:.5f}",
                f"{option.fractional_position[2]:.5f}",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(_SITE_OPTION_ROLE, option)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._site_table.setItem(row, column, item)
        if self._site_table.rowCount():
            selected_row = 0
            if self._current_site is not None:
                for row, option in enumerate(self._site_options):
                    if _same_muon_site(option, self._current_site):
                        selected_row = row
                        break
            self._site_table.selectRow(selected_row)
        self._site_table.resizeRowsToContents()

    def _selected_site_option(self) -> MagneticMuonSiteOption | None:
        selected = self._site_table.selectionModel().selectedRows()
        if not selected:
            return None
        row = selected[0].row()
        item = self._site_table.item(row, 0)
        option = item.data(_SITE_OPTION_ROLE) if item is not None else None
        return option if isinstance(option, MagneticMuonSiteOption) else None

    def _sync_distortion_controls(self) -> None:
        option = self._selected_site_option()
        enabled = (
            self._site_source() == "relaxation_clustering"
            and option is not None
            and option.distortion_structure_id is not None
        )
        self._distortion_checkbox.setEnabled(enabled)
        blocked = self._distortion_checkbox.blockSignals(True)
        self._distortion_checkbox.setChecked(enabled and self._distortion_requested)
        self._distortion_checkbox.blockSignals(blocked)

    def _on_distortion_toggled(self, checked: bool) -> None:
        self._distortion_requested = bool(checked)

    def _on_site_selection_changed(self) -> None:
        self._sync_distortion_controls()
        self._refresh_site_status()

    def _refresh_site_status(self) -> None:
        if self._host_structure is None:
            self._site_status.setText("Muon site: load a host structure first.")
            return
        source = self._site_source()
        if source == "custom":
            self._site_status.setText(
                "Muon site: custom fractional coordinate in the active host cell."
            )
            return
        option = self._selected_site_option()
        if option is None:
            self._site_status.setText(
                f"Muon site: no {_site_source_label(source).lower()} sites available."
            )
            return
        distortion_text = (
            " Distortion corrections are available for this representative."
            if option.distortion_structure_id is not None
            else ""
        )
        self._site_status.setText(
            f"Muon site: {option.label} at ({option.fractional_position[0]:.4f}, "
            f"{option.fractional_position[1]:.4f}, {option.fractional_position[2]:.4f}).{distortion_text}"
        )

    @staticmethod
    def _fractional_position_input(value: str) -> QLineEdit:
        edit = QLineEdit(value)
        edit.setValidator(QDoubleValidator(0.0, 1.0, 8, edit))
        edit.setPlaceholderText("0.0")
        return edit

    def _configure_site_table_columns(self) -> None:
        header = self._site_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        for column in (3, 4, 5):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)


class MagneticFieldDistributionDialog(QDialog):
    """Plot an incommensurate magnetic-field magnitude distribution."""

    def __init__(
        self,
        results: object | Sequence[object],
        parent: QWidget | None = None,
        *,
        estimate: bool = False,
    ) -> None:
        super().__init__(parent)
        self._results = self._coerce_results(results)
        self._estimate = estimate
        label = self._title_label()
        self.setWindowTitle(f"Field Distribution - {label}")
        self.resize(760, 520)

        palette = current_palette()
        self._figure, self._canvas = create_plot_canvas(
            figsize=(7.2, 4.6),
            facecolor=palette.bg_surface,
            minimum_size=(400, 280),
        )

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        self._export_button = buttons.addButton(
            "Export Data...", QDialogButtonBox.ButtonRole.ActionRole
        )
        self._export_button.clicked.connect(self._export_data)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._canvas, 1)
        layout.addWidget(buttons)

        self._draw_plot()

    def _draw_plot(self) -> None:
        palette = current_palette()
        self._figure.clear()
        axis = self._figure.add_subplot(111)
        style_plot_axes(axis)
        axis.set_xlabel(r"$B$ (T)")
        axis.set_ylabel(r"$p(B)$")
        axis.set_title(self._title_label())
        x_formatter = ScalarFormatter(useOffset=False)
        x_formatter.set_scientific(False)
        axis.xaxis.set_major_formatter(x_formatter)

        series = self._distribution_series()
        common_edges: np.ndarray | None = None
        if len(series) > 1:
            common = self._common_binned_series(series)
            if common is not None:
                series, common_edges = common
        x_limits: list[tuple[float, float]] = []
        y_max = 0.0
        for centres, probabilities, magnitudes, _label in series:
            x_limits.append(self._field_axis_limits(centres, magnitudes))
            y_max = max(y_max, float(np.max(probabilities)))

        if series:
            low = min(limit[0] for limit in x_limits)
            high = max(limit[1] for limit in x_limits)
            for centres, probabilities, magnitudes, label in series:
                if common_edges is None:
                    plot_x, plot_y = self._probability_steps_with_zero_baseline(
                        centres, probabilities, low, high
                    )
                else:
                    plot_x, plot_y = self._interval_steps_with_zero_baseline(
                        common_edges, probabilities, low, high
                    )
                line = axis.plot(plot_x, plot_y, linewidth=1.8, label=label)[0]
                fill_alpha = 0.24 if len(series) == 1 else 0.06
                axis.fill_between(plot_x, plot_y, color=line.get_color(), alpha=fill_alpha)
            if len(series) > 1:
                if common_edges is None:
                    sum_x, sum_y, sum_values = self._summed_probability_steps(series, low, high)
                else:
                    sum_values = np.sum(
                        [probabilities for _centres, probabilities, _magnitudes, _label in series],
                        axis=0,
                    )
                    sum_x, sum_y = self._interval_steps_with_zero_baseline(
                        common_edges, sum_values, low, high
                    )
                sum_line = axis.plot(
                    sum_x, sum_y, linewidth=2.4, color=palette.text_primary, label="sum"
                )[0]
                axis.fill_between(sum_x, sum_y, color=sum_line.get_color(), alpha=0.16)
                if sum_values.size:
                    y_max = max(y_max, float(np.max(sum_values)))
            axis.set_xlim(low, high)
            axis.set_ylim(0.0, max(1.05, y_max * 1.08))
            if len(self._results) > 1:
                style_plot_legend(axis.legend(loc="best", frameon=True))
        else:
            axis.text(
                0.5,
                0.5,
                "No distribution samples available.",
                ha="center",
                va="center",
                color=palette.text_muted,
                transform=axis.transAxes,
            )
        self._canvas.draw_idle()

    @staticmethod
    def _coerce_results(results: object | Sequence[object]) -> tuple[object, ...]:
        if isinstance(results, Sequence) and not isinstance(results, (str, bytes, bytearray)):
            return tuple(results)
        return (results,)

    def _title_label(self) -> str:
        if len(self._results) == 1:
            label = str(getattr(self._results[0], "site_label", "Field distribution"))
        else:
            label = f"{len(self._results)} field distributions"
        return f"{label} (unrelaxed estimate)" if self._estimate else label

    def _distribution_series(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        series: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
        for index, result in enumerate(self._results):
            centres = np.asarray(getattr(result, "histogram_centres_T", []), dtype=float)
            counts = np.asarray(getattr(result, "histogram_counts", []), dtype=float)
            magnitudes = self._magnitudes(result)
            if centres.size == 0 or counts.size != centres.size:
                centres, counts = self._histogram_from_magnitudes(magnitudes)
            if not centres.size or not counts.size:
                continue
            probabilities = self._normalized_probabilities(counts)
            label = str(getattr(result, "site_label", f"site {index + 1}"))
            series.append((centres, probabilities, magnitudes, label))
        return series

    def _common_binned_series(
        self,
        series: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
    ) -> tuple[list[tuple[np.ndarray, np.ndarray, np.ndarray, str]], np.ndarray] | None:
        finite_magnitudes = [
            magnitudes[np.isfinite(magnitudes)]
            for _centres, _probabilities, magnitudes, _label in series
            if magnitudes.size
        ]
        if len(finite_magnitudes) != len(series):
            return None
        all_magnitudes = np.concatenate(finite_magnitudes)
        if all_magnitudes.size == 0:
            return None
        low, high = self._field_axis_limits(np.asarray([], dtype=float), all_magnitudes)
        bins = max(10, max(len(centres) for centres, _probabilities, _magnitudes, _label in series))
        common_edges = np.linspace(low, high, int(bins) + 1, dtype=float)
        common_centres = common_edges[:-1] + 0.5 * np.diff(common_edges)
        out: list[tuple[np.ndarray, np.ndarray, np.ndarray, str]] = []
        for _centres, _probabilities, magnitudes, label in series:
            counts, _edges = np.histogram(magnitudes[np.isfinite(magnitudes)], bins=common_edges)
            out.append((common_centres, self._normalized_probabilities(counts), magnitudes, label))
        return out, common_edges

    def _export_data(self) -> None:
        path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export Field Distribution",
            "field_distribution.dat",
            "DAT files (*.dat);;Text files (*.txt);;All files (*)",
        )
        if not path:
            return
        if "." not in path.rsplit("/", 1)[-1]:
            path = f"{path}.dat"
        try:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self._export_text())
        except OSError as exc:
            QMessageBox.warning(self, "Export Field Distribution", f"Could not write file: {exc}")

    def _export_text(self) -> str:
        series = self._distribution_series()
        common_edges: np.ndarray | None = None
        if len(series) > 1:
            common = self._common_binned_series(series)
            if common is not None:
                series, common_edges = common
        lines = self._export_header(series)
        if not series:
            return "\n".join(lines) + "\n"
        low = min(
            self._field_axis_limits(centres, magnitudes)[0]
            for centres, _probabilities, magnitudes, _label in series
        )
        high = max(
            self._field_axis_limits(centres, magnitudes)[1]
            for centres, _probabilities, magnitudes, _label in series
        )
        if common_edges is None:
            interval_edges, values_by_label, sum_values = self._summed_probability_intervals(
                series, low, high
            )
        else:
            interval_edges = common_edges
            values_by_label = {
                label: probabilities for _centres, probabilities, _magnitudes, label in series
            }
            sum_values = np.sum(tuple(values_by_label.values()), axis=0)
        centres = 0.5 * (interval_edges[:-1] + interval_edges[1:])
        labels = [label for _centres, _probabilities, _magnitudes, label in series]
        column_labels = ["B_T", *(f"p_{label}" for label in labels)]
        if len(series) > 1:
            column_labels.append("p_sum")
        lines.append(
            "# Columns: " + " ".join(self._safe_column_label(label) for label in column_labels)
        )
        for row_index, centre in enumerate(centres):
            row = [float(centre)]
            row.extend(float(values_by_label[label][row_index]) for label in labels)
            if len(series) > 1:
                row.append(float(sum_values[row_index]))
            lines.append(" ".join(f"{value:.12g}" for value in row))
        return "\n".join(lines) + "\n"

    def _export_header(
        self, series: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, str]]
    ) -> list[str]:
        lines = [
            "# MuFinder field distribution export",
            f"# Distributions: {len(series)}",
        ]
        if self._estimate:
            lines.append(f"# Estimate: {_UNRELAXED_ESTIMATE_NOTE}")
        for index, result in enumerate(self._results, start=1):
            label = str(getattr(result, "site_label", f"site {index}"))
            position = getattr(result, "muon_position", None)
            if position is not None:
                try:
                    x, y, z = _tuple3(position)
                    lines.append(f"# {label} muon_position_fractional: {x:.12g} {y:.12g} {z:.12g}")
                except (TypeError, ValueError):
                    pass
            params = getattr(result, "calculation_parameters", None)
            if isinstance(params, Mapping) and params:
                lines.append(f"# {label} calculation_parameters:")
                for key in sorted(params):
                    lines.append(f"#   {key}: {params[key]}")
        if not any("calculation_parameters" in line for line in lines):
            lines.append("# Calculation parameters: unavailable for this result")
        return lines

    @staticmethod
    def _safe_column_label(label: str) -> str:
        return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in label)

    @staticmethod
    def _magnitudes(result: object) -> np.ndarray:
        fields = np.asarray(getattr(result, "fields_T", []), dtype=float)
        if fields.ndim != 2 or fields.shape[1] != 3:
            return np.asarray([], dtype=float)
        return np.linalg.norm(fields, axis=1)

    @staticmethod
    def _histogram_from_magnitudes(magnitudes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if magnitudes.size == 0:
            return np.asarray([], dtype=float), np.asarray([], dtype=float)
        bins = min(80, max(10, int(np.sqrt(int(magnitudes.size))) * 2))
        counts, edges = np.histogram(magnitudes, bins=bins)
        centres = edges[:-1] + 0.5 * np.diff(edges)
        return centres, counts.astype(float)

    @staticmethod
    def _normalized_probabilities(counts: np.ndarray) -> np.ndarray:
        counts = np.asarray(counts, dtype=float)
        if counts.size == 0 or not np.any(counts > 0.0):
            return np.zeros_like(counts, dtype=float)
        return counts / float(np.max(counts))

    @staticmethod
    def _probability_steps_with_zero_baseline(
        centres: np.ndarray,
        probabilities: np.ndarray,
        low: float,
        high: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        centres = np.asarray(centres, dtype=float)
        probabilities = np.asarray(probabilities, dtype=float)
        if centres.size == 0 or probabilities.size != centres.size or high <= low:
            return centres, probabilities
        edges = MagneticFieldDistributionDialog._histogram_edges_from_centres(centres, low, high)
        return MagneticFieldDistributionDialog._interval_steps_with_zero_baseline(
            edges, probabilities, low, high
        )

    @staticmethod
    def _interval_steps_with_zero_baseline(
        edges: np.ndarray,
        values: np.ndarray,
        low: float,
        high: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        x_values: list[float] = []
        y_values: list[float] = []
        if low < edges[0]:
            x_values.extend([float(low), float(min(edges[0], high))])
            y_values.extend([0.0, 0.0])
        for index, probability in enumerate(values):
            left = max(float(edges[index]), float(low))
            right = min(float(edges[index + 1]), float(high))
            if right <= left:
                continue
            if x_values and not np.isclose(x_values[-1], left):
                x_values.append(left)
                y_values.append(y_values[-1])
            x_values.extend([left, right])
            y_values.extend([float(probability), float(probability)])
        if not x_values:
            return np.asarray([low, high], dtype=float), np.asarray([0.0, 0.0], dtype=float)
        if high > edges[-1]:
            edge = max(float(edges[-1]), float(low))
            if not np.isclose(x_values[-1], edge):
                x_values.append(edge)
                y_values.append(y_values[-1])
            x_values.extend([edge, float(high)])
            y_values.extend([0.0, 0.0])
        return np.asarray(x_values, dtype=float), np.asarray(y_values, dtype=float)

    @staticmethod
    def _histogram_edges_from_centres(centres: np.ndarray, low: float, high: float) -> np.ndarray:
        centres = np.asarray(centres, dtype=float)
        if centres.size == 1:
            half_width = 0.5 * max(abs(float(centres[0])) * 1e-6, (high - low) * 1e-4, 1e-9)
            return np.asarray([centres[0] - half_width, centres[0] + half_width], dtype=float)
        midpoints = 0.5 * (centres[:-1] + centres[1:])
        first_edge = centres[0] - (midpoints[0] - centres[0])
        last_edge = centres[-1] + (centres[-1] - midpoints[-1])
        return np.concatenate(([first_edge], midpoints, [last_edge])).astype(float)

    @staticmethod
    def _summed_probability_steps(
        series: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
        low: float,
        high: float,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x_values = np.unique(
            np.concatenate(
                (
                    [low, high],
                    *(
                        np.asarray(centres, dtype=float)
                        for centres, _probabilities, _magnitudes, _label in series
                    ),
                )
            )
        )
        x_values = x_values[np.isfinite(x_values)]
        x_values.sort()
        if x_values.size < 2:
            x_values = np.asarray([low, high], dtype=float)
        sum_values = np.zeros_like(x_values, dtype=float)
        for centres, probabilities, _magnitudes, _label in series:
            edges = MagneticFieldDistributionDialog._histogram_edges_from_centres(
                np.asarray(centres, dtype=float), low, high
            )
            for index, probability in enumerate(probabilities):
                left = edges[index]
                right = edges[index + 1]
                mask = (x_values >= left) & (x_values < right)
                if index == len(probabilities) - 1:
                    mask = mask | np.isclose(x_values, right)
                sum_values[mask] += float(probability)
        sum_values[np.isclose(x_values, low) | np.isclose(x_values, high)] = 0.0
        return x_values, sum_values, sum_values

    @staticmethod
    def _summed_probability_intervals(
        series: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, str]],
        low: float,
        high: float,
    ) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
        source_edges = [
            MagneticFieldDistributionDialog._histogram_edges_from_centres(centres, low, high)
            for centres, _probabilities, _magnitudes, _label in series
        ]
        raw_edges = np.concatenate(
            ([low, high], *(np.clip(edges, low, high) for edges in source_edges))
        )
        edges = np.unique(raw_edges[np.isfinite(raw_edges)])
        edges.sort()
        if edges.size < 2:
            edges = np.asarray([low, high], dtype=float)
        midpoints = 0.5 * (edges[:-1] + edges[1:])
        values_by_label: dict[str, np.ndarray] = {}
        for (centres, probabilities, _magnitudes, label), edges_for_series in zip(
            series, source_edges, strict=True
        ):
            values = np.zeros_like(midpoints, dtype=float)
            for index, probability in enumerate(probabilities):
                left = edges_for_series[index]
                right = edges_for_series[index + 1]
                mask = (midpoints >= left) & (midpoints < right)
                if index == len(probabilities) - 1:
                    mask = mask | np.isclose(midpoints, right)
                values[mask] = float(probability)
            values_by_label[label] = values
        sum_values = (
            np.sum(tuple(values_by_label.values()), axis=0)
            if values_by_label
            else np.zeros_like(midpoints)
        )
        return edges, values_by_label, sum_values

    @staticmethod
    def _field_axis_limits(centres: np.ndarray, magnitudes: np.ndarray) -> tuple[float, float]:
        source = magnitudes if magnitudes.size else centres
        finite = np.asarray(source[np.isfinite(source)], dtype=float)
        if finite.size == 0:
            return (0.0, 1.0)
        data_min = float(np.min(finite))
        data_max = float(np.max(finite))
        centre = 0.5 * (data_min + data_max)
        span = data_max - data_min
        scale = max(abs(centre), 1.0)
        min_window = max(abs(centre) * 0.04, scale * 1e-4)
        if span < min_window:
            half_window = 0.5 * min_window
            low = centre - half_window
            high = centre + half_window
        else:
            padding = max(0.08 * span, scale * 1e-5)
            low = data_min - padding
            high = data_max + padding
        if data_min >= 0.0:
            low = max(0.0, low)
        if high <= low:
            high = low + max(scale * 1e-4, 1e-6)
        return (low, high)


class MagneticFieldCalculationSettingsDialog(QDialog):
    """Dialog for numerical magnetic-field calculation settings."""

    def __init__(
        self,
        *,
        supercell: tuple[int, int, int],
        nangles: int,
        phase_sample_count: int,
        histogram_bins: int,
        phase_sampling: str = "auto",
        use_approximants: bool = False,
        approximant_tolerance: float = MagneticFieldOptions().approximant_tolerance,
        approximant_max_period: int = MagneticFieldOptions().approximant_max_period,
        contact_nnn: int = MagneticFieldOptions().nnn,
        contact_rcont: float = MagneticFieldOptions().rcont,
        contact_tie_check: Callable[[int, float], Sequence[str]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """``contact_tie_check`` returns the contact-shell tie warnings for a
        Contact neighbours and radius pair; without it no warning is shown."""

        super().__init__(parent)
        self.setWindowTitle("Calculation Settings")
        self._contact_tie_check = contact_tie_check
        self.resize(460, 360)

        self._calc_cells_a = self._calculation_cell_input(supercell[0])
        self._calc_cells_b = self._calculation_cell_input(supercell[1])
        self._calc_cells_c = self._calculation_cell_input(supercell[2])
        self._lookup_angles = self._lookup_angle_input(nangles)
        self._phase_samples = self._incommensurate_sampling_input(phase_sample_count)
        self._phase_sampling = self._phase_sampling_input(phase_sampling)
        self._histogram_bins = self._histogram_bin_input(histogram_bins)

        layout = QVBoxLayout(self)
        form_group = QGroupBox("Numerical Parameters", self)
        form = QFormLayout(form_group)

        supercell_group = QWidget(form_group)
        supercell_layout = QHBoxLayout(supercell_group)
        supercell_layout.setContentsMargins(0, 0, 0, 0)
        supercell_layout.addWidget(QLabel("a"))
        supercell_layout.addWidget(self._calc_cells_a)
        supercell_layout.addWidget(QLabel("b"))
        supercell_layout.addWidget(self._calc_cells_b)
        supercell_layout.addWidget(QLabel("c"))
        supercell_layout.addWidget(self._calc_cells_c)
        supercell_layout.addStretch(1)
        form.addRow("Cells along", supercell_group)
        form.addRow("Lookup angles per q", self._lookup_angles)
        form.addRow("Distribution samples", self._phase_samples)
        form.addRow("Sampling method", self._phase_sampling)
        form.addRow("Histogram bins", self._histogram_bins)
        layout.addWidget(form_group)
        guidance = QLabel(CONVERGENCE_GUIDANCE_TEXT, self)
        guidance.setWordWrap(True)
        style_hint_text(guidance)
        layout.addWidget(guidance)

        approximant_group = QGroupBox("Commensurate Approximants", self)
        approximant_form = QFormLayout(approximant_group)
        self._use_approximants = QCheckBox(
            "Report finite fields for near-commensurate q", approximant_group
        )
        self._use_approximants.setChecked(bool(use_approximants))
        self._use_approximants.setToolTip(
            "Replace each incommensurate wavevector by its nearest short-period fraction and list the fields "
            "at the translation-equivalent sites of that magnetic supercell instead of sampling a distribution."
        )
        self._approximant_tolerance = QDoubleSpinBox(approximant_group)
        self._approximant_tolerance.setDecimals(6)
        self._approximant_tolerance.setRange(1e-6, 0.05)
        self._approximant_tolerance.setSingleStep(1e-4)
        self._approximant_tolerance.setValue(float(approximant_tolerance))
        self._approximant_tolerance.setSuffix(" r.l.u.")
        self._approximant_tolerance.setToolTip(
            "Largest change of any wavevector component when it is replaced by a fraction p/n."
        )
        self._approximant_max_period = QSpinBox(approximant_group)
        self._approximant_max_period.setRange(1, MAX_APPROXIMANT_PERIOD)
        self._approximant_max_period.setValue(int(approximant_max_period))
        self._approximant_max_period.setToolTip(
            "Largest denominator n allowed for each wavevector component."
        )
        approximant_form.addRow(self._use_approximants)
        approximant_form.addRow("Tolerance", self._approximant_tolerance)
        approximant_form.addRow("Longest period", self._approximant_max_period)
        layout.addWidget(approximant_group)

        contact_group = QGroupBox("Contact Field", self)
        contact_form = QFormLayout(contact_group)
        self._contact_nnn = QSpinBox(contact_group)
        self._contact_nnn.setRange(1, MAX_CONTACT_NEIGHBOURS)
        self._contact_nnn.setValue(int(contact_nnn))
        self._contact_nnn.setToolTip(
            "Number of nearest magnetic ions whose 1/r³-weighted mean moment sets the contact "
            "field. Used only when a hyperfine coupling is set."
        )
        self._contact_rcont = QDoubleSpinBox(contact_group)
        self._contact_rcont.setDecimals(CONTACT_RADIUS_DECIMALS)
        self._contact_rcont.setRange(MIN_CONTACT_RADIUS_ANG, MAX_CONTACT_RADIUS_ANG)
        self._contact_rcont.setSingleStep(0.5)
        self._contact_rcont.setValue(float(contact_rcont))
        self._contact_rcont.setToolTip(
            "Only magnetic ions closer than this to the muon enter the contact shell. "
            "Used only when a hyperfine coupling is set."
        )
        contact_form.addRow("Contact neighbours (nnn)", self._contact_nnn)
        contact_form.addRow("Contact radius (Å)", self._contact_rcont)
        layout.addWidget(contact_group)
        self._contact_tie_warning = QLabel(self)
        self._contact_tie_warning.setWordWrap(True)
        style_warning_text(self._contact_tie_warning)
        # Inside the group box the wrapped warning would not get the height its text needs.
        layout.addWidget(self._contact_tie_warning)
        # Each check searches the shell of every site, so a typed value is
        # checked once it is finished, not at every keystroke.
        for spinbox in (self._contact_nnn, self._contact_rcont):
            spinbox.setKeyboardTracking(False)
            spinbox.valueChanged.connect(self._refresh_contact_tie_warning)
        self._refresh_contact_tie_warning()

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _refresh_contact_tie_warning(self) -> None:
        """Show whether the Contact neighbours split a tie of the nearest magnetic ions."""

        warnings = (
            tuple(
                self._contact_tie_check(
                    int(self._contact_nnn.value()), float(self._contact_rcont.value())
                )
            )
            if self._contact_tie_check is not None
            else ()
        )
        text = ""
        if warnings:
            text = warnings[0]
            if len(warnings) > 1:
                others = len(warnings) - 1
                text += f" The same holds at {others} more {'site' if others == 1 else 'sites'}."
        self._contact_tie_warning.setText(text)
        self._contact_tie_warning.setVisible(bool(text))

    def values(self) -> dict[str, object]:
        return {
            "supercell": (
                int(self._calc_cells_a.value()),
                int(self._calc_cells_b.value()),
                int(self._calc_cells_c.value()),
            ),
            "nangles": int(self._lookup_angles.value()),
            "phase_sample_count": int(self._phase_samples.value()),
            "phase_sampling": str(self._phase_sampling.currentData() or "auto"),
            "histogram_bins": int(self._histogram_bins.value()),
            "use_approximants": self._use_approximants.isChecked(),
            "approximant_tolerance": float(self._approximant_tolerance.value()),
            "approximant_max_period": int(self._approximant_max_period.value()),
            "contact_nnn": int(self._contact_nnn.value()),
            "contact_rcont": float(self._contact_rcont.value()),
        }

    @staticmethod
    def _calculation_cell_input(value: int) -> QSpinBox:
        spinbox = QSpinBox()
        spinbox.setRange(1, 500)
        spinbox.setValue(int(value))
        spinbox.setToolTip(
            "Unit-cell repeats used for the muesr lattice summation. Larger values are slower but reduce cutoff effects."
        )
        spinbox.setMinimumWidth(104)
        return spinbox

    @staticmethod
    def _incommensurate_sampling_input(value: int) -> QSpinBox:
        spinbox = QSpinBox()
        spinbox.setRange(12, MAX_INCOMMENSURATE_PHASE_SAMPLES)
        spinbox.setSingleStep(1_024)
        spinbox.setValue(int(value))
        spinbox.setToolTip(
            "Target number of phase points in the final incommensurate field distribution. "
            "Grid sampling may use a nearby lower value so each active phase axis has the same resolution."
        )
        spinbox.setMinimumWidth(112)
        return spinbox

    @staticmethod
    def _lookup_angle_input(value: int) -> QSpinBox:
        spinbox = QSpinBox()
        spinbox.setRange(12, MAX_INCOMMENSURATE_LOOKUP_ANGLES)
        spinbox.setSingleStep(512)
        spinbox.setValue(int(value))
        spinbox.setToolTip(
            "Phase lookup resolution passed to muesr for each Fourier component. "
            "This is separate from the number of distribution samples."
        )
        spinbox.setMinimumWidth(112)
        return spinbox

    @staticmethod
    def _phase_sampling_input(value: str) -> QComboBox:
        combo = QComboBox()
        for key, label in PHASE_SAMPLING_METHOD_OPTIONS:
            combo.addItem(label, key)
        index = combo.findData(str(value).strip().lower())
        combo.setCurrentIndex(index if index >= 0 else combo.findData("auto"))
        combo.setToolTip(
            "Grid gives deterministic multi-q phase distributions. Monte Carlo keeps the sample count bounded for larger phase spaces."
        )
        return combo

    @staticmethod
    def _histogram_bin_input(value: int) -> QSpinBox:
        spinbox = QSpinBox()
        spinbox.setRange(10, 10_000)
        spinbox.setSingleStep(64)
        spinbox.setValue(int(value))
        spinbox.setToolTip(
            "Number of bins used for incommensurate p(B) histograms. More bins narrow the bin width."
        )
        spinbox.setMinimumWidth(104)
        return spinbox


def demag_sample_summary(sample: DemagnetizingSample) -> str:
    """Return a one-line description of an advanced sample's shape, without its orientation."""

    shape = str(sample.shape).strip().lower()
    if shape == "ellipsoid":
        a, b, c = sample.semi_axes
        return f"Ellipsoid a={a:g}, b={b:g}, c={c:g}"
    if shape == "rectangular_prism":
        a, b, c = sample.prism_lengths
        return f"Rectangular prism {a:g} x {b:g} x {c:g}"
    if shape == "image_plate":
        if sample.plate_mask is None:
            return "No outline yet. Click From photo... to outline the sample in a photo."
        width, height = sample.plate_size
        text = f"Plate from photo {width:g} x {height:g} x {sample.plate_thickness:g}"
        if sample.photo is not None and sample.photo.path:
            text += f" ({Path(sample.photo.path).name})"
        return text
    if shape == DEMAG_VOXEL_GRID_OPTION[0]:
        counts = "unset" if sample.voxels is None else " x ".join(map(str, sample.voxels.shape))
        return f"Voxel grid {counts} voxels"
    return "No advanced shape selected."


class _DemagnetizingPyVistaCanvas(QWidget):
    """PyVista preview canvas for demagnetizing sample shapes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._pv, qt_interactor = _pyvista_modules()
        if self._pv is None or qt_interactor is None:
            raise RuntimeError(
                "PyVista and pyvistaqt are required for demagnetizing-field visualisation."
            )
        self._sample: DemagnetizingSample | None = None
        self._lattice_vectors: np.ndarray | None = None
        # The last voxel surface and the shape it was built for: editing the
        # orientation only rotates it.
        self._voxel_mesh_cache: tuple[DemagnetizingSample, object] | None = None
        self._refreshing_theme = False
        self._style_signature = self._style_signature_for_current_palette()
        # auto_update=False disables pyvistaqt's always-on render_timer; we
        # render explicitly (draw_idle/plotter.render), so the ~5Hz timer is
        # pure CPU waste that runs even when this widget is not visible.
        self._plotter = qt_interactor(self, auto_update=False)
        _configure_transparent_pyvista_background(self._plotter)
        self._plotter.enable_parallel_projection()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plotter)
        self.setMinimumSize(360, 280)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    @staticmethod
    def _style_signature_for_current_palette() -> tuple[str, ...]:
        palette = current_palette()
        return (
            palette.bg_primary,
            palette.status_running,
            palette.accent,
            palette.success,
            palette.status_submitting,
            palette.text_primary,
        )

    def set_sample(
        self, sample: DemagnetizingSample, lattice_vectors: Sequence[Sequence[float]] | None = None
    ) -> None:
        """Draw the sample rotated into the crystal frame, with arrows along the crystal axes.

        Without ``lattice_vectors`` the axis arrows follow Cartesian x, y and z.
        """

        self._sample = sample
        self._lattice_vectors = (
            None if lattice_vectors is None else np.asarray(lattice_vectors, dtype=float)
        )
        rotation = sample.rotation_matrix() if sample.rotation is not None else None
        plotter = self._plotter
        pv = self._pv
        plotter.clear()
        _configure_transparent_pyvista_background(plotter)
        shape = str(sample.shape).strip().lower()
        voxel_mesh = self._voxel_mesh(sample) if sample.uses_fourier_method else None
        if sample.uses_fourier_method:
            mesh, extent = voxel_mesh if voxel_mesh is not None else (None, 1.0)
            if mesh is not None and rotation is not None:
                transform = np.eye(4, dtype=float)
                transform[:3, :3] = rotation
                mesh = mesh.transform(transform, inplace=False)
        elif shape == "rectangular_prism":
            lengths = np.asarray(sample.prism_lengths, dtype=float)
            mesh = pv.Cube(
                bounds=(
                    -0.5 * lengths[0],
                    0.5 * lengths[0],
                    -0.5 * lengths[1],
                    0.5 * lengths[1],
                    -0.5 * lengths[2],
                    0.5 * lengths[2],
                )
            )
            extent = float(np.max(lengths))
            if rotation is not None:
                transform = np.eye(4, dtype=float)
                transform[:3, :3] = rotation
                mesh = mesh.transform(transform, inplace=False)
        else:
            axes = np.asarray(sample.semi_axes, dtype=float)
            mesh = self._ellipsoid_mesh(axes, rotation)
            extent = float(np.max(axes))

        palette = current_palette()
        if mesh is not None:
            plotter.add_mesh(
                mesh,
                color=palette.status_running,
                opacity=0.42,
                smooth_shading=not sample.uses_fourier_method,
            )
        axis_colors = (palette.accent, palette.success, palette.status_submitting)
        directions = np.eye(3, dtype=float)
        if self._lattice_vectors is not None and self._lattice_vectors.shape == (3, 3):
            lengths = np.linalg.norm(self._lattice_vectors, axis=1)
            if np.all(lengths > 0.0):
                directions = self._lattice_vectors / lengths[:, None]
        for direction, color, label in zip(directions, axis_colors, ("a", "b", "c"), strict=True):
            self._add_arrow((0.0, 0.0, 0.0), 1.25 * extent * direction, color, label)
        self._add_arrow(
            (0.0, 0.0, -0.18 * extent), (0.0, 0.0, 1.65 * extent), palette.text_primary, "M"
        )
        plotter.reset_camera()
        plotter.render()

    def changeEvent(self, event) -> None:  # noqa: N802
        super().changeEvent(event)
        plotter = getattr(self, "_plotter", None)
        signature = self._style_signature_for_current_palette()
        if (
            self._refreshing_theme
            or plotter is None
            or signature == getattr(self, "_style_signature", None)
            or event.type()
            not in {
                QEvent.Type.StyleChange,
                QEvent.Type.PaletteChange,
                QEvent.Type.ApplicationPaletteChange,
            }
        ):
            return
        self._refreshing_theme = True
        try:
            self._style_signature = signature
            _configure_transparent_pyvista_background(plotter)
            if self._sample is not None:
                self.set_sample(self._sample, self._lattice_vectors)
            else:
                plotter.render()
        finally:
            self._refreshing_theme = False

    def shutdown(self) -> None:
        _shutdown_pyvista_widget(self._plotter)

    def _voxel_mesh(self, sample: DemagnetizingSample):
        """Return the surface of the sample's voxels, centred on the origin, and its half-extent.

        A plate is drawn one voxel thick, since its layers are identical. A
        sample without voxels yet (a plate with no photo) draws nothing.
        """

        shape_only = replace(sample, rotation=None, normal_hkl=None, in_plane_uvw=None, photo=None)
        cached = self._voxel_mesh_cache
        if cached is not None and cached[0] == shape_only:
            return cached[1]
        mesh = self._build_voxel_mesh(shape_only)
        self._voxel_mesh_cache = (shape_only, mesh)
        return mesh

    def _build_voxel_mesh(self, sample: DemagnetizingSample):
        try:
            if sample.shape == "image_plate":
                sample = replace(sample, thickness_voxels=1)
            model = sample.voxel_model()
        except DemagnetizationError:
            return None
        mask, spacing = model.mask, np.asarray(model.spacing, dtype=float)
        step = max(1, int(np.ceil((mask.size / _PREVIEW_MAX_VOXELS) ** (1.0 / 3.0))))
        if step > 1:
            mask = mask[::step, ::step, ::step]
            spacing = spacing * step
        grid = self._pv.ImageData(
            dimensions=tuple(int(n) + 1 for n in mask.shape),
            spacing=tuple(float(s) for s in spacing),
            origin=tuple(-0.5 * float(n) * float(s) for n, s in zip(mask.shape, spacing)),
        )
        grid.cell_data["inside"] = mask.ravel(order="F").astype(np.uint8)
        surface = grid.threshold(0.5, scalars="inside").extract_surface(algorithm="dataset_surface")
        return surface, 0.5 * float(np.max(np.asarray(mask.shape) * spacing))

    def _ellipsoid_mesh(self, axes: np.ndarray, rotation: np.ndarray | None = None):
        theta = np.linspace(0.0, 2.0 * np.pi, 72, endpoint=False)
        phi = np.linspace(0.0, np.pi, 36)
        tt, pp = np.meshgrid(theta, phi)
        points = np.column_stack(
            [
                (axes[0] * np.sin(pp) * np.cos(tt)).ravel(),
                (axes[1] * np.sin(pp) * np.sin(tt)).ravel(),
                (axes[2] * np.cos(pp)).ravel(),
            ]
        )
        if rotation is not None:
            points = points @ rotation.T
        faces: list[int] = []
        n_phi, n_theta = tt.shape
        for i in range(n_phi - 1):
            for j in range(n_theta):
                a = i * n_theta + j
                b = i * n_theta + ((j + 1) % n_theta)
                c = (i + 1) * n_theta + ((j + 1) % n_theta)
                d = (i + 1) * n_theta + j
                faces.extend([4, a, b, c, d])
        return self._pv.PolyData(points, np.asarray(faces, dtype=int))

    def _add_arrow(
        self, start: Sequence[float], vector: Sequence[float], color: str, label: str
    ) -> None:
        pv = self._pv
        start_arr = np.asarray(start, dtype=float)
        vector_arr = np.asarray(vector, dtype=float)
        length = float(np.linalg.norm(vector_arr))
        if length <= 1e-12:
            return
        arrow = pv.Arrow(
            start=tuple(start_arr),
            direction=tuple(vector_arr / length),
            # pv.Arrow builds a unit arrow and scales all of it by ``scale``.
            scale=length,
            shaft_radius=0.018,
            tip_radius=0.055,
            tip_length=0.18,
        )
        self._plotter.add_mesh(arrow, color=color)
        self._plotter.add_point_labels(
            np.asarray([start_arr + vector_arr], dtype=float),
            [label],
            text_color=color,
            font_size=14,
            point_size=0,
            shape_opacity=0.0,
        )


class DemagnetizingFieldAdvancedDialog(QDialog):
    """Advanced demagnetizing-field shape and tensor settings."""

    def __init__(
        self,
        sample: DemagnetizingSample | None = None,
        parent: QWidget | None = None,
        *,
        lattice_vectors: Sequence[Sequence[float]] | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Advanced Demagnetizing Field")
        self.resize(860, 560)
        self._preview: _DemagnetizingPyVistaCanvas | QLabel
        self._lattice_vectors = (
            None if lattice_vectors is None else [list(row) for row in lattice_vectors]
        )

        sample = sample or DemagnetizingSample(shape="ellipsoid")
        if str(sample.shape).strip().lower() not in ADVANCED_DEMAG_SHAPES:
            sample = DemagnetizingSample(shape="ellipsoid")

        self._shape = QComboBox(self)
        for key, label in ADVANCED_DEMAG_SHAPE_OPTIONS:
            self._shape.addItem(label, key)
        if sample.shape == DEMAG_VOXEL_GRID_OPTION[0]:
            self._shape.addItem(DEMAG_VOXEL_GRID_OPTION[1], DEMAG_VOXEL_GRID_OPTION[0])
        index = self._shape.findData(sample.shape)
        self._shape.setCurrentIndex(index if index >= 0 else 0)

        self._ellipsoid_a = demag_length_input(sample.semi_axes[0])
        self._ellipsoid_b = demag_length_input(sample.semi_axes[1])
        self._ellipsoid_c = demag_length_input(sample.semi_axes[2])
        self._prism_a = demag_length_input(sample.prism_lengths[0])
        self._prism_b = demag_length_input(sample.prism_lengths[1])
        self._prism_c = demag_length_input(sample.prism_lengths[2])
        # The plate and voxel grid are edited as a whole: through the photo
        # dialog, or not at all.
        self._plate_sample = (
            sample if sample.shape == "image_plate" else DemagnetizingSample(shape="image_plate")
        )
        self._voxel_sample = sample if sample.shape == DEMAG_VOXEL_GRID_OPTION[0] else None
        self._plate_summary = QLabel("", self)
        self._plate_summary.setWordWrap(True)
        style_hint_text(self._plate_summary)
        self._photo_button = QPushButton("From photo...", self)
        self._photo_button.clicked.connect(self._open_photo_dialog)
        self._voxel_summary = QLabel("", self)
        self._voxel_summary.setWordWrap(True)
        style_hint_text(self._voxel_summary)

        self._stack = QStackedWidget(self)
        self._stack.addWidget(self._ellipsoid_page())
        self._stack.addWidget(self._prism_page())
        self._stack.addWidget(self._plate_page())
        self._stack.addWidget(self._voxel_page())

        # The seed's orientation is kept whenever the dialog cannot edit it: a
        # rotation saved without crystal directions, or directions with no host.
        self._seed = sample
        self._principal: tuple[DemagnetizingSample, np.ndarray] | None = None
        self._orientation_mode = QComboBox(self)
        for key, label in DEMAG_ORIENTATION_OPTIONS:
            self._orientation_mode.addItem(label, key)
        if sample.rotation is not None and sample.normal_hkl is None:
            self._orientation_mode.addItem(
                DEMAG_SAVED_ROTATION_OPTION[1], DEMAG_SAVED_ROTATION_OPTION[0]
            )
            initial_mode = DEMAG_SAVED_ROTATION_OPTION[0]
        else:
            initial_mode = "directions" if sample.normal_hkl is not None else "aligned"
        self._orientation_mode.setCurrentIndex(self._orientation_mode.findData(initial_mode))
        self._orientation_mode.setEnabled(self._lattice_vectors is not None)
        self._normal_inputs = MagneticComponentDialog._direction_inputs(
            sample.normal_hkl or (0.0, 0.0, 1.0), placeholders="hkl"
        )
        self._in_plane_inputs = MagneticComponentDialog._direction_inputs(
            sample.in_plane_uvw or (1.0, 0.0, 0.0), placeholders="uvw"
        )
        self._orientation_hint = QLabel("", self)
        self._orientation_hint.setWordWrap(True)
        style_hint_text(self._orientation_hint)

        self._tensor_label = QLabel("", self)
        self._tensor_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        style_hint_text(self._tensor_label)

        try:
            self._preview = _DemagnetizingPyVistaCanvas(self)
        except RuntimeError as exc:
            self._preview = QLabel(str(exc), self)
            self._preview.setWordWrap(True)
            self._preview.setMinimumSize(360, 280)
            style_hint_text(self._preview)

        title = QLabel("Advanced Demagnetizing Field", self)
        style_section_title(title)
        subtitle = QLabel(
            "Choose a sample shape, inspect the demagnetizing tensor, and preview the geometry used for the field correction.",
            self,
        )
        subtitle.setWordWrap(True)
        style_section_subtitle(subtitle)

        form = QFormLayout()
        form.addRow("Shape", self._shape)
        form.addRow("Parameters", self._stack)

        orientation_group = QGroupBox("Orientation", self)
        orientation_form = QFormLayout(orientation_group)
        orientation_form.addRow("Sample axes", self._orientation_mode)
        self._index_rows: list[QWidget] = []
        for label, inputs in (
            ("Normal (h k l)", self._normal_inputs),
            ("In-plane axis [u v w]", self._in_plane_inputs),
        ):
            label_widget = QLabel(label, orientation_group)
            group = MagneticComponentDialog._direction_group(inputs)
            orientation_form.addRow(label_widget, group)
            self._index_rows.extend((label_widget, group))
        orientation_form.addRow(self._orientation_hint)

        left = QVBoxLayout()
        left.addWidget(title)
        left.addWidget(subtitle)
        left.addLayout(form)
        left.addWidget(orientation_group)
        left.addWidget(self._tensor_label)
        left.addStretch(1)

        body = QHBoxLayout()
        body.addLayout(left, 0)
        body.addWidget(self._preview, 1)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel, self
        )
        self._use_button = buttons.button(QDialogButtonBox.StandardButton.Ok)
        self._use_button.setText("Use Shape")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addLayout(body)
        layout.addWidget(buttons)

        self._shape.currentIndexChanged.connect(self._sync)
        for widget in (
            self._ellipsoid_a,
            self._ellipsoid_b,
            self._ellipsoid_c,
            self._prism_a,
            self._prism_b,
            self._prism_c,
        ):
            widget.valueChanged.connect(self._sync)
        self._orientation_mode.currentIndexChanged.connect(self._sync)
        for edit in (*self._normal_inputs, *self._in_plane_inputs):
            edit.textChanged.connect(self._sync)
        self._sync()

    def selected_sample(self) -> DemagnetizingSample:
        """Return the configured sample, oriented when ``Crystal directions`` is chosen.

        Raises :class:`DemagnetizationError` for unusable crystal directions.
        """

        return self._resolve()[0]

    def _resolve(self) -> tuple[DemagnetizingSample, SampleOrientation | None]:
        """Return the sample and, for directions set here, their resolved orientation."""

        sample = self._shape_sample()
        mode = self._orientation_key()
        if mode == "aligned":
            return sample, None
        seed = self._seed
        if mode == DEMAG_SAVED_ROTATION_OPTION[0] or self._lattice_vectors is None:
            return (
                replace(
                    sample,
                    rotation=seed.rotation,
                    normal_hkl=seed.normal_hkl,
                    in_plane_uvw=seed.in_plane_uvw,
                ),
                None,
            )
        normal = self._index_values(self._normal_inputs, "normal (h k l)")
        in_plane = self._index_values(self._in_plane_inputs, "in-plane axis [u v w]")
        orientation = sample_orientation(self._lattice_vectors, normal, in_plane)
        return (
            replace(
                sample, rotation=orientation.rotation, normal_hkl=normal, in_plane_uvw=in_plane
            ),
            orientation,
        )

    def _shape_sample(self) -> DemagnetizingSample:
        shape = str(self._shape.currentData() or "ellipsoid")
        if shape == "image_plate":
            return replace(self._plate_sample, rotation=None, normal_hkl=None, in_plane_uvw=None)
        if shape == DEMAG_VOXEL_GRID_OPTION[0] and self._voxel_sample is not None:
            return replace(self._voxel_sample, rotation=None, normal_hkl=None, in_plane_uvw=None)
        return DemagnetizingSample(
            shape=shape,
            semi_axes=(
                float(self._ellipsoid_a.value()),
                float(self._ellipsoid_b.value()),
                float(self._ellipsoid_c.value()),
            ),
            prism_lengths=(
                float(self._prism_a.value()),
                float(self._prism_b.value()),
                float(self._prism_c.value()),
            ),
        )

    def _open_photo_dialog(self) -> None:
        dialog = SamplePhotoDialog(self._plate_sample, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._plate_sample = dialog.selected_sample()
        self._sync()

    def closeEvent(self, event) -> None:  # noqa: N802
        if isinstance(self._preview, _DemagnetizingPyVistaCanvas):
            self._preview.shutdown()
        super().closeEvent(event)

    def _ellipsoid_page(self) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)
        form.addRow("Semi-axis a", self._ellipsoid_a)
        form.addRow("Semi-axis b", self._ellipsoid_b)
        form.addRow("Semi-axis c", self._ellipsoid_c)
        return page

    def _prism_page(self) -> QWidget:
        page = QWidget(self)
        form = QFormLayout(page)
        form.addRow("Length a", self._prism_a)
        form.addRow("Length b", self._prism_b)
        form.addRow("Length c", self._prism_c)
        return page

    def _plate_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._plate_summary)
        row = QHBoxLayout()
        row.addWidget(self._photo_button)
        row.addStretch(1)
        layout.addLayout(row)
        layout.addStretch(1)
        return page

    def _voxel_page(self) -> QWidget:
        page = QWidget(self)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(self._voxel_summary)
        layout.addStretch(1)
        return page

    def _sync_voxel_summaries(self) -> None:
        self._plate_summary.setText(demag_sample_summary(self._plate_sample))
        self._photo_button.setText(
            "From photo..." if self._plate_sample.plate_mask is None else "Edit outline..."
        )
        if self._voxel_sample is not None:
            self._voxel_summary.setText(
                demag_sample_summary(self._voxel_sample)
                + ". A voxel grid is set through the Python API and cannot be edited here."
            )

    def _orientation_key(self) -> str:
        return str(self._orientation_mode.currentData() or "aligned")

    @staticmethod
    def _index_values(
        inputs: tuple[QLineEdit, QLineEdit, QLineEdit], name: str
    ) -> tuple[float, float, float]:
        try:
            h, k, l_ = (float(edit.text().strip()) for edit in inputs)
        except ValueError as exc:
            raise DemagnetizationError(f"enter three numbers for the {name}") from exc
        return (h, k, l_)

    def _orientation_text(self, orientation: SampleOrientation | None) -> str:
        mode = self._orientation_key()
        if mode == "aligned":
            text = "The sample axes a, b and c lie along Cartesian x, y and z."
            if self._lattice_vectors is None:
                text += " Orienting the sample by crystal directions needs a host structure."
            return text
        if mode == DEMAG_SAVED_ROTATION_OPTION[0]:
            return (
                "The sample axes follow a rotation saved without crystal directions. Choose "
                "Crystal directions to set them by (h k l) and [u v w]."
            )
        if self._lattice_vectors is None:
            return "The saved crystal directions are kept; editing them needs a host structure."
        text = "The sample c axis is normal to (h k l) and its a axis lies along [u v w]."
        if orientation is not None and not orientation.orthogonal:
            text += (
                f" [u v w] is {orientation.angle_deg:.2f}° from the normal, so the a axis "
                "follows its component in the plane."
            )
        return text

    def _principal_tensor(self, sample: DemagnetizingSample) -> np.ndarray:
        """Return the shape's principal tensor, reused while only the orientation changes."""

        shape = replace(sample, rotation=None, normal_hkl=None, in_plane_uvw=None)
        if self._principal is None or self._principal[0] != shape:
            self._principal = (shape, principal_demagnetizing_tensor(shape))
        return self._principal[1]

    def _sync(self, *_args: object) -> None:
        shape = str(self._shape.currentData() or "ellipsoid")
        pages = ("ellipsoid", "rectangular_prism", "image_plate", DEMAG_VOXEL_GRID_OPTION[0])
        self._stack.setCurrentIndex(pages.index(shape) if shape in pages else 0)
        self._sync_voxel_summaries()
        editable = self._orientation_key() == "directions" and self._lattice_vectors is not None
        for row in self._index_rows:
            row.setEnabled(editable)
        sample = self._shape_sample()
        try:
            sample, orientation = self._resolve()
            self._orientation_hint.setText(self._orientation_text(orientation))
        except DemagnetizationError as exc:
            self._orientation_hint.setText(f"Orientation unavailable: {exc}")
            self._tensor_label.setText("Tensor unavailable until the orientation is valid.")
            self._use_button.setEnabled(False)
            if isinstance(self._preview, _DemagnetizingPyVistaCanvas):
                self._preview.set_sample(sample, self._lattice_vectors)
            return
        try:
            tensor = sample.rotate(self._principal_tensor(sample))
        except DemagnetizationError as exc:
            self._tensor_label.setText(f"Tensor unavailable: {exc}")
            self._use_button.setEnabled(False)
        else:
            # Elements that round to zero print as 0.00000, not -0.00000.
            tensor = np.where(np.abs(tensor) < 5e-6, 0.0, tensor)
            text = (
                "N = "
                f"[[{tensor[0, 0]:.5f}, {tensor[0, 1]:.5f}, {tensor[0, 2]:.5f}], "
                f"[{tensor[1, 0]:.5f}, {tensor[1, 1]:.5f}, {tensor[1, 2]:.5f}], "
                f"[{tensor[2, 0]:.5f}, {tensor[2, 1]:.5f}, {tensor[2, 2]:.5f}]]"
            )
            if sample.uses_fourier_method:
                result = fourier_demagnetizing_result(sample)
                box = " x ".join(str(n) for n in result.box_shape)
                text += (
                    f"\nTr N = {result.trace:.5f} from {result.voxel_count} voxels in a "
                    f"{box} FFT box."
                )
            self._tensor_label.setText(text)
            self._use_button.setEnabled(True)
        if isinstance(self._preview, _DemagnetizingPyVistaCanvas):
            self._preview.set_sample(sample, self._lattice_vectors)


class NuclearRelaxationDialog(QDialog):
    """Configure and display nuclear dipolar relaxation calculations."""

    _MODE_LABELS = {
        "tf_single_crystal": "TF single crystal",
        "zf_single_crystal": "ZF single crystal",
        "tf_polycrystal": "TF polycrystal",
        "zf_polycrystal": "ZF polycrystal",
    }

    def __init__(
        self,
        *,
        muon_site: MagneticMuonSiteOption | None,
        distortion_available: bool,
        elements: Sequence[str] = (),
        options_state: Mapping[str, object] | None = None,
        results_state: Mapping[str, object] | None = None,
        calculate_callback: Callable[["NuclearRelaxationDialog"], object] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Nuclear Relaxation")
        self.resize(720, 600)
        self._muon_site = muon_site
        self._distortion_available = bool(distortion_available)
        self._calculate_callback = calculate_callback
        self._results_state: dict[str, object] = (
            dict(results_state) if isinstance(results_state, Mapping) else {}
        )

        self._regime_combo = QComboBox(self)
        self._regime_combo.addItem("Zero field", "zf")
        self._regime_combo.addItem("Transverse field", "tf")
        self._sample_combo = QComboBox(self)
        self._sample_combo.addItem("Polycrystal", "polycrystal")
        self._sample_combo.addItem("Single crystal", "single_crystal")
        self._direction_x = self._decimal_input("0.0")
        self._direction_y = self._decimal_input("0.0")
        self._direction_z = self._decimal_input("1.0")
        for edit in (self._direction_x, self._direction_y, self._direction_z):
            edit.setObjectName("nuclearDirectionInput")
        self._radius = self._decimal_input("", minimum=0.0)
        self._radius.setPlaceholderText("auto")
        self._rtol = self._decimal_input("1e-6", minimum=0.0)
        self._element_checks = {
            str(element): QCheckBox(str(element), self) for element in dict.fromkeys(elements)
        }
        # Without checkboxes the saved filter is carried through unchanged.
        self._saved_elements: tuple[str, ...] | None = None
        for check in self._element_checks.values():
            check.setChecked(True)
            check.setToolTip("Include this element's nuclei in the second-moment sum.")
        self._quadrupolar = QCheckBox("Include ZF quadrupolar correction", self)
        self._distortion = QCheckBox("Apply muon-induced distortion correction", self)
        self._distortion.setEnabled(self._distortion_available)
        self._distortion.setToolTip(
            "Available when the selected relaxation-cluster site has a derived distorted structure."
        )
        self._status = QLabel(
            "Ready." if muon_site is not None else "Select a muon site before calculating.", self
        )
        self._status.setWordWrap(True)
        style_hint_text(self._status)
        self._calculate_button = QPushButton("Calculate", self)
        self._calculate_button.clicked.connect(self._calculate)
        self._calculate_button.setEnabled(muon_site is not None)

        self._summary_table = create_workflow_table(
            ("Quantity", "Value"),
            resize_to_contents_columns=(0,),
            stretch_last_section=False,
        )
        self._element_table = create_workflow_table(
            ("Element", "σ² contribution (μs⁻²)"),
            resize_to_contents_columns=(0,),
            stretch_last_section=False,
        )
        self._estimate_note = unrelaxed_results_note(self)

        self._build_layout()
        self._apply_styles()
        self._restore_options(options_state)
        self._sync_mode_controls()
        self._regime_combo.currentIndexChanged.connect(self._sync_mode_controls)
        self._sample_combo.currentIndexChanged.connect(self._sync_mode_controls)
        self._populate_results(self._results_state)

    def options_state(self) -> dict[str, object]:
        return self.options().to_json()

    def results_state(self) -> dict[str, object]:
        return dict(self._results_state)

    def options(self) -> NuclearRelaxationOptions:
        mode = f"{self._regime_combo.currentData()}_{self._sample_combo.currentData()}"
        radius_text = self._radius.text().strip()
        radius = None if not radius_text else float(radius_text)
        return NuclearRelaxationOptions(
            mode=mode,  # type: ignore[arg-type]
            elements=self._selected_elements(),
            direction=(
                float(self._direction_x.text() or "0"),
                float(self._direction_y.text() or "0"),
                float(self._direction_z.text() or "1"),
            ),
            include_quadrupolar=self._quadrupolar.isChecked(),
            radius=radius,
            convergence_rtol=float(self._rtol.text() or "1e-6"),
            use_distortion_correction=self._distortion.isChecked(),
        )

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        title = QLabel("Nuclear Relaxation", self)
        style_section_title(title)
        layout.addWidget(title)

        site_group = QGroupBox("Muon Site", self)
        site_layout = QVBoxLayout(site_group)
        site_layout.addWidget(QLabel(self._muon_site_summary(), site_group))
        layout.addWidget(site_group)

        mode_group = QGroupBox("Mode", self)
        mode_form = QFormLayout(mode_group)
        mode_form.addRow("Field regime", self._regime_combo)
        mode_form.addRow("Sample", self._sample_combo)
        direction_group = QWidget(mode_group)
        style_inline_group(direction_group)
        direction_layout = QHBoxLayout(direction_group)
        direction_layout.setContentsMargins(0, 0, 0, 0)
        direction_layout.addWidget(QLabel("x"))
        direction_layout.addWidget(self._direction_x)
        direction_layout.addWidget(QLabel("y"))
        direction_layout.addWidget(self._direction_y)
        direction_layout.addWidget(QLabel("z"))
        direction_layout.addWidget(self._direction_z)
        mode_form.addRow("Crystal-axis direction", direction_group)
        mode_form.addRow("Radius (Å)", self._radius)
        mode_form.addRow("Convergence rtol", self._rtol)
        if self._element_checks:
            elements_group = QWidget(mode_group)
            style_inline_group(elements_group)
            elements_layout = QHBoxLayout(elements_group)
            elements_layout.setContentsMargins(0, 0, 0, 0)
            elements_layout.setSpacing(24)
            for check in self._element_checks.values():
                elements_layout.addWidget(check)
            elements_layout.addStretch(1)
            mode_form.addRow("Elements", elements_group)
        mode_form.addRow("", self._quadrupolar)
        mode_form.addRow("", self._distortion)
        layout.addWidget(mode_group)

        action_row = QHBoxLayout()
        action_row.addWidget(self._calculate_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)
        layout.addWidget(self._status)

        results_group = QGroupBox("Results", self)
        results_layout = QVBoxLayout(results_group)
        results_layout.addWidget(self._estimate_note)
        results_layout.addWidget(self._summary_table)
        results_layout.addWidget(self._element_table)
        layout.addWidget(results_group, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _apply_styles(self) -> None:
        palette = current_palette()
        self.setStyleSheet(
            "QLineEdit#nuclearDirectionInput {"
            f"background: {palette.bg_surface};"
            f"border: 1px solid {palette.border};"
            f"color: {palette.text_primary};"
            "border-radius: 5px;"
            "padding: 5px 8px;"
            "}"
            "QLineEdit#nuclearDirectionInput:disabled {"
            f"background: {palette.bg_surface};"
            f"color: {palette.text_muted};"
            "}"
        )

    def _selected_elements(self) -> tuple[str, ...] | None:
        """Return the ticked elements, or ``None`` (every element) when all are ticked."""

        if not self._element_checks:
            return self._saved_elements
        ticked = tuple(
            element for element, check in self._element_checks.items() if check.isChecked()
        )
        return None if len(ticked) == len(self._element_checks) else ticked

    def _calculate(self) -> None:
        if self._calculate_callback is None:
            return
        if self._selected_elements() == ():
            self._status.setText("Tick at least one element to include in the sum.")
            return
        try:
            result = self._calculate_callback(self)
        except Exception as exc:  # noqa: BLE001 - UI boundary
            self._status.setText(f"Calculation failed: {exc}")
            return
        payload = result.to_json() if hasattr(result, "to_json") else dict(result)
        jsonable = to_jsonable(payload)
        self._results_state = dict(jsonable) if isinstance(jsonable, Mapping) else {}
        if self._results_state and self._muon_site is not None:
            self._results_state["site_source"] = self._muon_site.source
        elements = self._selected_elements()
        if self._results_state and elements is not None:
            self._results_state["elements"] = list(elements)
        self._populate_results(self._results_state)
        mode = str(self._results_state.get("mode", self.options().mode))
        suffix = _UNRELAXED_ESTIMATE_SUFFIX if not self._estimate_note.isHidden() else ""
        self._status.setText(
            f"Calculated {self._MODE_LABELS.get(mode, mode)} nuclear relaxation.{suffix}"
        )

    def _restore_options(self, state: Mapping[str, object] | None) -> None:
        if not isinstance(state, Mapping):
            return
        mode = str(state.get("mode", "zf_polycrystal"))
        if mode.startswith("tf_"):
            self._regime_combo.setCurrentIndex(self._regime_combo.findData("tf"))
        else:
            self._regime_combo.setCurrentIndex(self._regime_combo.findData("zf"))
        self._sample_combo.setCurrentIndex(
            self._sample_combo.findData(
                "single_crystal" if mode.endswith("single_crystal") else "polycrystal"
            )
        )
        direction = MagneticFieldWidget._state_float_triplet(state.get("direction"))
        if direction is not None:
            self._direction_x.setText(f"{direction[0]:.8g}")
            self._direction_y.setText(f"{direction[1]:.8g}")
            self._direction_z.setText(f"{direction[2]:.8g}")
        if state.get("radius") not in {None, ""}:
            self._radius.setText(str(state.get("radius")))
        if state.get("convergence_rtol") not in {None, ""}:
            self._rtol.setText(str(state.get("convergence_rtol")))
        saved_elements = state.get("elements")
        if isinstance(saved_elements, (list, tuple)):
            saved = tuple(str(value) for value in saved_elements)
            self._saved_elements = saved
            for element, check in self._element_checks.items():
                check.setChecked(element in saved)
        self._quadrupolar.setChecked(bool(state.get("include_quadrupolar", False)))
        self._distortion.setChecked(
            bool(state.get("use_distortion_correction", False)) and self._distortion_available
        )

    def _sync_mode_controls(self) -> None:
        single = str(self._sample_combo.currentData()) == "single_crystal"
        zf = str(self._regime_combo.currentData()) == "zf"
        for edit in (self._direction_x, self._direction_y, self._direction_z):
            edit.setEnabled(single)
        self._quadrupolar.setEnabled(zf)
        if not zf:
            self._quadrupolar.setChecked(False)

    def _populate_results(self, result: Mapping[str, object]) -> None:
        self._summary_table.setRowCount(0)
        self._element_table.setRowCount(0)
        self._estimate_note.setVisible(is_unrelaxed_site_source(result.get("site_source")))
        if not result:
            return
        rows = [
            (
                "Mode",
                self._MODE_LABELS.get(str(result.get("mode", "")), str(result.get("mode", ""))),
            ),
            ("σ (μs⁻¹)", _format_float(result.get("sigma"))),
            ("σ² (μs⁻²)", _format_float(result.get("sigma_squared"))),
            ("M₂ (μs⁻²)", _format_float(result.get("m2"))),
            ("Δ field (T)", _format_float(result.get("delta_field_t"))),
            ("Convergence radius (Å)", _format_float(result.get("convergence_radius_a"))),
            ("Nuclei summed", str(result.get("n_nuclei_summed", ""))),
            ("Elements", _nuclear_elements_text(result.get("elements"))),
        ]
        if result.get("sigma_perp_squared") is not None:
            rows.append(("σ⊥² (μs⁻²)", _format_float(result.get("sigma_perp_squared"))))
        if result.get("sigma_parallel_squared") is not None:
            rows.append(("σ∥² (μs⁻²)", _format_float(result.get("sigma_parallel_squared"))))
        correction = result.get("distortion_correction")
        if isinstance(correction, Mapping):
            rows.extend(
                [
                    (
                        "Correction Δσ²",
                        _format_float(correction.get("distortion_correction_sigma_squared")),
                    ),
                    ("Atoms unwrapped", str(correction.get("n_atoms_unwrapped", ""))),
                ]
            )
        for label, value in rows:
            row = self._summary_table.rowCount()
            self._summary_table.insertRow(row)
            self._summary_table.setItem(row, 0, QTableWidgetItem(label))
            self._summary_table.setItem(row, 1, QTableWidgetItem(value))
        per_element = result.get("per_element_sigma_squared")
        if isinstance(per_element, Mapping):
            for element, value in sorted(per_element.items()):
                row = self._element_table.rowCount()
                self._element_table.insertRow(row)
                self._element_table.setItem(row, 0, QTableWidgetItem(str(element)))
                self._element_table.setItem(row, 1, QTableWidgetItem(_format_float(value)))
        self._summary_table.resizeRowsToContents()
        self._element_table.resizeRowsToContents()

    def _muon_site_summary(self) -> str:
        if self._muon_site is None:
            return "Muon site: no site selected."
        x, y, z = self._muon_site.fractional_position
        return f"Muon site: {self._muon_site.label} at ({x:.4f}, {y:.4f}, {z:.4f})."

    @staticmethod
    def _decimal_input(value: str, *, minimum: float | None = None) -> QLineEdit:
        edit = QLineEdit(value)
        bottom = -1_000_000.0 if minimum is None else float(minimum)
        edit.setValidator(QDoubleValidator(bottom, 1_000_000.0, 12, edit))
        edit.setMinimumWidth(92)
        return edit


class DipolarTensorResultsDialog(QDialog):
    """Display one or more dipolar tensor matrices."""

    def __init__(self, results: Sequence[object], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._results = tuple(results)
        self._matrix_tables: list[QTableWidget] = []
        self._raw_tensors: list[np.ndarray] = []
        self.setWindowTitle("Dipolar Tensor Results")
        self.resize(560, 440)

        layout = QVBoxLayout(self)
        title = QLabel("Dipolar tensors")
        style_section_title(title)
        layout.addWidget(title)

        unit_row = QHBoxLayout()
        unit_row.addWidget(QLabel("Units"))
        self._unit_combo = QComboBox()
        for unit_key, label in CONTACT_COUPLING_UNIT_OPTIONS:
            self._unit_combo.addItem(label, unit_key)
        t_index = self._unit_combo.findData("T_per_muB")
        if t_index >= 0:
            self._unit_combo.setCurrentIndex(t_index)
        self._unit_combo.currentIndexChanged.connect(self._refresh_tensor_units)
        unit_row.addWidget(self._unit_combo)
        unit_row.addStretch(1)
        layout.addLayout(unit_row)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        content = QWidget(scroll)
        content_layout = QVBoxLayout(content)
        content_layout.setSpacing(14)
        for result in self._results:
            content_layout.addWidget(self._result_card(result))
        content_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _result_card(self, result: object) -> QWidget:
        group = QGroupBox(str(getattr(result, "site_label", "site")))
        layout = QVBoxLayout(group)
        position = getattr(result, "muon_position", None)
        if position is not None:
            try:
                x, y, z = _tuple3(position)
                summary = QLabel(f"Muon position: ({x:.6g}, {y:.6g}, {z:.6g})")
                style_hint_text(summary)
                layout.addWidget(summary)
            except (TypeError, ValueError):
                pass

        table = QTableWidget(3, 3, group)
        track_item_view_corners(table)
        table.setHorizontalHeaderLabels(["x", "y", "z"])
        table.setVerticalHeaderLabels(["x", "y", "z"])
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        tensor = self._tensor(result)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.setMinimumHeight(136)
        layout.addWidget(table)
        self._raw_tensors.append(tensor)
        self._matrix_tables.append(table)
        self._set_table_tensor(table, self._display_tensor(tensor))
        return group

    def _refresh_tensor_units(self) -> None:
        label = self._unit_combo.currentText()
        for table in self._matrix_tables:
            table.setToolTip(f"Dipolar tensor in {label}")
        for table, tensor in zip(self._matrix_tables, self._raw_tensors, strict=True):
            self._set_table_tensor(table, self._display_tensor(tensor))

    def _display_tensor(self, tensor_T_per_muB: np.ndarray) -> np.ndarray:
        return np.asarray(tensor_T_per_muB, dtype=float) * self._display_factor()

    def _display_factor(self) -> float:
        unit = str(self._unit_combo.currentData() or "T_per_muB")
        if unit == "MHz_per_muB":
            return MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
        if unit in {"angstrom^-3", "a^-3", "ang^-3"}:
            return 1.0 / CONTACT_FIELD_T_PER_ANGSTROM_MINUS3_PER_MUB
        return 1.0

    @staticmethod
    def _set_table_tensor(table: QTableWidget, tensor: np.ndarray) -> None:
        for row in range(3):
            for column in range(3):
                item = table.item(row, column)
                if item is None:
                    item = QTableWidgetItem()
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    table.setItem(row, column, item)
                item.setText(f"{float(tensor[row, column]):.8g}")

    @staticmethod
    def _tensor(result: object) -> np.ndarray:
        tensor = getattr(result, "tensor_T_per_muB", None)
        if tensor is None and isinstance(result, Mapping):
            tensor = result.get("dipolar_tensor_T_per_muB")
        arr = np.asarray(tensor, dtype=float)
        if arr.shape != (3, 3):
            arr = np.zeros((3, 3), dtype=float)
        return arr


class DipolarTensorDialog(QDialog):
    """Configure magnetic-ion selection for a dipolar tensor calculation."""

    def __init__(
        self,
        structure: Structure,
        model: MagneticStructureModel,
        muon_site: MagneticMuonSiteOption | None,
        *,
        include_equivalent_sites: bool,
        selector_state: Mapping[str, object] | None = None,
        supercell: tuple[int, int, int] = MagneticFieldOptions().supercell,
        results_state: Sequence[object] = (),
        calculate_callback: Callable[["DipolarTensorDialog"], Sequence[object]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle("Dipolar Tensor")
        self.resize(640, 620)
        self._structure = structure
        self._model = model
        self._muon_site = muon_site
        self._supercell = tuple(int(value) for value in supercell)
        self._calculate_callback = calculate_callback
        self._results_state: list[dict[str, object]] = [
            dict(result) for result in results_state if isinstance(result, Mapping)
        ]
        self._result_tables: list[QTableWidget] = []
        self._result_tensors: list[np.ndarray] = []
        self._assigned_indices = assigned_magnetic_atom_indices(structure, model)

        self._target_group = QButtonGroup(self)
        self._assigned_radio = QRadioButton("Atoms with assigned moments")
        self._all_atoms_radio = QRadioButton("All atoms")
        self._species_radio = QRadioButton("Element")
        self._equivalent_radio = QRadioButton("Equivalent site")
        self._indices_radio = QRadioButton("Atom indices")
        for button in (
            self._assigned_radio,
            self._all_atoms_radio,
            self._species_radio,
            self._equivalent_radio,
            self._indices_radio,
        ):
            self._target_group.addButton(button)
            button.toggled.connect(self._refresh_enabled_state)

        self._assigned_radio.setEnabled(bool(self._assigned_indices))
        self._species_combo = QComboBox()
        for species in sorted(set(str(symbol) for symbol in structure.species)):
            self._species_combo.addItem(species)
        self._species_combo.currentIndexChanged.connect(self._refresh_enabled_state)

        self._equivalent_combo = QComboBox()
        for selector in viable_atom_selectors(structure):
            if selector.kind == "symmetry_equivalent" and selector.values:
                self._equivalent_combo.addItem(_selector_label(selector, structure), selector)
        self._equivalent_combo.currentIndexChanged.connect(self._refresh_enabled_state)

        self._indices_edit = QLineEdit()
        self._indices_edit.setPlaceholderText("1, 3, 6")
        self._indices_edit.setToolTip("Atom numbers counted from 1 in the host's atom order")
        self._indices_edit.textChanged.connect(self._refresh_enabled_state)

        self._include_equivalent_sites = QCheckBox("Include equivalent muon sites")
        self._include_equivalent_sites.setChecked(bool(include_equivalent_sites))

        self._settings_summary = QLabel(
            f"Lattice sum: cells {self._supercell[0]} x {self._supercell[1]} x {self._supercell[2]}."
        )
        self._settings_summary.setWordWrap(True)
        style_hint_text(self._settings_summary)

        self._status = QLabel("")
        self._status.setWordWrap(True)
        style_hint_text(self._status)

        self._result_unit_combo = QComboBox()
        for unit_key, label in CONTACT_COUPLING_UNIT_OPTIONS:
            self._result_unit_combo.addItem(label, unit_key)
        t_index = self._result_unit_combo.findData("T_per_muB")
        if t_index >= 0:
            self._result_unit_combo.setCurrentIndex(t_index)
        self._result_unit_combo.currentIndexChanged.connect(self._refresh_result_units)

        self._estimate_note = unrelaxed_results_note(self)
        self._result_scroll = QScrollArea(self)
        self._result_scroll.setWidgetResizable(True)
        self._result_content = QWidget(self._result_scroll)
        self._result_content_layout = QVBoxLayout(self._result_content)
        self._result_content_layout.setSpacing(14)
        self._result_scroll.setWidget(self._result_content)

        self._calculate_button = QPushButton("Calculate")
        self._calculate_button.clicked.connect(self._calculate)
        self._buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        self._buttons.rejected.connect(self.reject)

        self._build_layout()
        self._restore_selector_state(selector_state)
        # What the dialog opened with, kept so closing it on invalid input
        # (an atom number below 1 or a non-number) does not lose the last
        # valid selection.
        self._opening_selector_state = self.selector_state()
        self._populate_results(self._results_state)
        self._refresh_enabled_state()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        title = QLabel("Dipolar Tensor")
        style_section_title(title)
        layout.addWidget(title)

        site_group = QGroupBox("Muon Site")
        site_layout = QVBoxLayout(site_group)
        site_summary = QLabel(self._muon_site_summary())
        site_summary.setWordWrap(True)
        style_hint_text(site_summary)
        site_layout.addWidget(site_summary)
        site_layout.addWidget(self._include_equivalent_sites)
        layout.addWidget(site_group)

        target_group = QGroupBox("Magnetic Ions")
        form = QFormLayout(target_group)
        assigned_text = (
            f"{len(self._assigned_indices)} atom(s) targeted by enabled magnetic components"
            if self._assigned_indices
            else "No atoms currently have assigned moments"
        )
        assigned_row = QWidget(target_group)
        assigned_layout = QHBoxLayout(assigned_row)
        assigned_layout.setContentsMargins(0, 0, 0, 0)
        assigned_layout.addWidget(self._assigned_radio)
        assigned_layout.addWidget(QLabel(assigned_text), 1)
        form.addRow(assigned_row)
        form.addRow(self._all_atoms_radio)
        form.addRow(self._species_radio, self._species_combo)
        form.addRow(self._equivalent_radio, self._equivalent_combo)
        form.addRow(self._indices_radio, self._indices_edit)
        layout.addWidget(target_group)

        settings_row = QHBoxLayout()
        settings_row.addWidget(self._settings_summary, 1)
        layout.addLayout(settings_row)
        layout.addWidget(self._status)
        action_row = QHBoxLayout()
        action_row.addWidget(self._calculate_button)
        action_row.addStretch(1)
        layout.addLayout(action_row)

        result_group = QGroupBox("Most Recent Result")
        result_layout = QVBoxLayout(result_group)
        unit_row = QHBoxLayout()
        unit_row.addWidget(QLabel("Units"))
        unit_row.addWidget(self._result_unit_combo)
        unit_row.addStretch(1)
        result_layout.addLayout(unit_row)
        result_layout.addWidget(self._estimate_note)
        result_layout.addWidget(self._result_scroll, 1)
        layout.addWidget(result_group, 1)
        layout.addWidget(self._buttons)

    def selector_state(self) -> dict[str, object]:
        state: dict[str, object] = {
            "include_equivalent_sites": self._include_equivalent_sites.isChecked(),
        }
        if self._assigned_radio.isChecked():
            state["mode"] = "assigned"
            return state
        try:
            selector = self.selector()
        except ValueError:
            opening = getattr(self, "_opening_selector_state", {"mode": "assigned"})
            return {**opening, **state}
        state["mode"] = "selector"
        state["selector"] = selector.to_json()
        return state

    def results_state(self) -> list[dict[str, object]]:
        return [dict(result) for result in self._results_state]

    def include_equivalent_sites(self) -> bool:
        return self._include_equivalent_sites.isChecked()

    def selector(self) -> AtomSelector:
        if self._species_radio.isChecked():
            return AtomSelector.species(self._species_combo.currentText())
        if self._equivalent_radio.isChecked():
            selector = self._equivalent_combo.currentData()
            if isinstance(selector, AtomSelector):
                return selector
        if self._indices_radio.isChecked():
            return AtomSelector.indices(self._parse_indices())
        if self._all_atoms_radio.isChecked():
            return AtomSelector.all()
        return AtomSelector.indices(self._assigned_indices)

    def selected_ion_indices(self) -> tuple[int, ...]:
        if self._assigned_radio.isChecked():
            return self._assigned_indices
        if not any(
            button.isChecked()
            for button in (
                self._all_atoms_radio,
                self._species_radio,
                self._equivalent_radio,
                self._indices_radio,
            )
        ):
            return ()
        return resolve_atom_selector(self._structure, self.selector())

    def _calculate(self) -> None:
        if self._calculate_callback is None:
            return
        try:
            results = tuple(self._calculate_callback(self))
        except MagneticFieldDependencyError as exc:
            self._status.setText(f"Result unavailable: {exc}")
            return
        except (MagneticFieldError, ValueError) as exc:
            self._status.setText(f"Calculation failed: {exc}")
            return
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self._status.setText(f"Calculation failed: {exc}")
            return
        self._results_state = [self._result_to_state(result) for result in results]
        if self._muon_site is not None:
            for state in self._results_state:
                state["site_source"] = self._muon_site.source
        self._populate_results(self._results_state)
        count = len(self._results_state)
        plural = "" if count == 1 else "s"
        suffix = _UNRELAXED_ESTIMATE_SUFFIX if not self._estimate_note.isHidden() else ""
        self._status.setText(f"Calculated {count} dipolar tensor{plural}.{suffix}")

    def _restore_selector_state(self, state: Mapping[str, object] | None) -> None:
        if isinstance(state, Mapping):
            self._include_equivalent_sites.setChecked(
                bool(
                    state.get(
                        "include_equivalent_sites", self._include_equivalent_sites.isChecked()
                    )
                )
            )
            if str(state.get("mode", "")).strip() == "assigned" and self._assigned_indices:
                self._assigned_radio.setChecked(True)
                return
            if "selector" in state:
                selector = AtomSelector.from_json(state.get("selector"))
                if self._set_selector(selector):
                    return
        if self._assigned_indices:
            self._assigned_radio.setChecked(True)

    def _set_selector(self, selector: AtomSelector) -> bool:
        if selector.kind == "species" and selector.values:
            index = self._species_combo.findText(str(selector.values[0]))
            if index >= 0:
                self._species_combo.setCurrentIndex(index)
                self._species_radio.setChecked(True)
                return True
        if selector.kind == "symmetry_equivalent":
            for index in range(self._equivalent_combo.count()):
                if self._equivalent_combo.itemData(index) == selector:
                    self._equivalent_combo.setCurrentIndex(index)
                    self._equivalent_radio.setChecked(True)
                    return True
        if selector.kind == "indices":
            self._indices_edit.setText(", ".join(str(int(value) + 1) for value in selector.values))
            self._indices_radio.setChecked(True)
            return True
        if selector.kind == "all":
            self._all_atoms_radio.setChecked(True)
            return True
        return False

    def _refresh_enabled_state(self, *_args: object) -> None:
        self._species_combo.setEnabled(self._species_radio.isChecked())
        self._equivalent_combo.setEnabled(self._equivalent_radio.isChecked())
        self._indices_edit.setEnabled(self._indices_radio.isChecked())
        try:
            indices = self.selected_ion_indices()
        except Exception:
            indices = ()
        has_site = self._muon_site is not None
        self._calculate_button.setEnabled(has_site and bool(indices))
        if not has_site:
            self._status.setText("Select a muon site before calculating a tensor.")
        elif not indices:
            self._status.setText("Choose at least one magnetic ion.")
        else:
            self._status.setText(f"Ready to calculate tensor using {len(indices)} magnetic ion(s).")

    def _muon_site_summary(self) -> str:
        if self._muon_site is None:
            return "Muon site: no site selected."
        x, y, z = self._muon_site.fractional_position
        return f"Muon site: {self._muon_site.label} at ({x:.4f}, {y:.4f}, {z:.4f})."

    def _parse_indices(self) -> tuple[int, ...]:
        """Return the zero-based atom indices for the one-based numbers typed in the box.

        The box counts from 1, like the magnetic-structure builder and the
        ``Equivalent site`` labels; the selector stores zero-based indices.
        """

        raw = self._indices_edit.text().replace(",", " ").split()
        numbers = tuple(int(value) for value in raw)
        if any(number < 1 for number in numbers):
            raise ValueError("Atom indices count from 1.")
        return tuple(number - 1 for number in numbers)

    def _populate_results(self, results: Sequence[Mapping[str, object]]) -> None:
        while self._result_content_layout.count():
            item = self._result_content_layout.takeAt(0)
            widget = item.widget()
            if widget is not None:
                widget.deleteLater()
        self._result_tables = []
        self._result_tensors = []
        self._estimate_note.setVisible(
            any(is_unrelaxed_site_source(result.get("site_source")) for result in results)
        )
        if not results:
            empty = QLabel("No tensor calculated yet.")
            style_hint_text(empty)
            self._result_content_layout.addWidget(empty)
            self._result_content_layout.addStretch(1)
            return
        for result in results:
            self._result_content_layout.addWidget(self._result_card(result))
        self._result_content_layout.addStretch(1)

    def _result_card(self, result: Mapping[str, object]) -> QWidget:
        title = str(result.get("site_label", "site"))
        if is_unrelaxed_site_source(result.get("site_source")):
            title = f"{title} (unrelaxed estimate)"
        group = QGroupBox(title)
        layout = QVBoxLayout(group)
        position = result.get("muon_position")
        if position is not None:
            try:
                x, y, z = _tuple3(position)
                summary = QLabel(f"Muon position: ({x:.6g}, {y:.6g}, {z:.6g})")
                style_hint_text(summary)
                layout.addWidget(summary)
            except (TypeError, ValueError):
                pass
        table = QTableWidget(3, 3, group)
        track_item_view_corners(table)
        table.setHorizontalHeaderLabels(["x", "y", "z"])
        table.setVerticalHeaderLabels(["x", "y", "z"])
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.setMinimumHeight(136)
        tensor = DipolarTensorResultsDialog._tensor(result)
        self._result_tables.append(table)
        self._result_tensors.append(tensor)
        self._set_result_table_tensor(table, self._display_result_tensor(tensor))
        layout.addWidget(table)
        return group

    def _refresh_result_units(self) -> None:
        for table, tensor in zip(self._result_tables, self._result_tensors, strict=True):
            self._set_result_table_tensor(table, self._display_result_tensor(tensor))

    def _display_result_tensor(self, tensor_T_per_muB: np.ndarray) -> np.ndarray:
        return np.asarray(tensor_T_per_muB, dtype=float) * self._result_display_factor()

    def _result_display_factor(self) -> float:
        unit = str(self._result_unit_combo.currentData() or "T_per_muB")
        if unit == "MHz_per_muB":
            return MUON_GYROMAGNETIC_RATIO_MHZ_PER_T
        if unit in {"angstrom^-3", "a^-3", "ang^-3"}:
            return 1.0 / CONTACT_FIELD_T_PER_ANGSTROM_MINUS3_PER_MUB
        return 1.0

    @staticmethod
    def _set_result_table_tensor(table: QTableWidget, tensor: np.ndarray) -> None:
        for row in range(3):
            for column in range(3):
                item = table.item(row, column)
                if item is None:
                    item = QTableWidgetItem()
                    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                    table.setItem(row, column, item)
                item.setText(f"{float(tensor[row, column]):.8g}")

    @staticmethod
    def _result_to_state(result: object) -> dict[str, object]:
        if hasattr(result, "to_json"):
            payload = result.to_json()
        elif isinstance(result, Mapping):
            payload = dict(result)
        else:
            payload = {
                "site_label": str(getattr(result, "site_label", "site")),
                "muon_position": list(getattr(result, "muon_position", (0.0, 0.0, 0.0))),
                "dipolar_tensor_T_per_muB": np.asarray(
                    getattr(result, "tensor_T_per_muB", np.zeros((3, 3))),
                    dtype=float,
                ).tolist(),
            }
        jsonable = to_jsonable(payload)
        return dict(jsonable) if isinstance(jsonable, Mapping) else {}


class MagneticFieldWidget(QWidget):
    """Magnetic-field workflow page."""

    magnetic_structure_changed = Signal(object)
    form_state_changed = Signal(dict)
    pin_requested = Signal(dict)
    open_muon_sites_requested = Signal()
    open_polarisation_requested = Signal()
    workspace_title_changed = Signal(str)
    delete_requested = Signal()

    _COMPONENT_SUMMARY_HEADERS = ("Component", "Type", "Apply to", "q", "Moment")
    _BREAKDOWN_HEADERS = ("Term", "Bx (T)", "By (T)", "Bz (T)", "|B| (T)")
    _DISTRIBUTION_BREAKDOWN_HEADERS = (
        "Term",
        "Mean Bx (T)",
        "Mean By (T)",
        "Mean Bz (T)",
        "RMS |B| (T)",
    )
    # Summary tables grow to this many rows, then scroll, so long models cannot push the controls off the page.
    _SUMMARY_MAX_VISIBLE_ROWS = 8
    # Commensurate result payload keys, in the order the breakdown lists them.
    _BREAKDOWN_TERMS = (
        ("Dipolar", "dipolar_T"),
        ("Lorentz", "lorentz_T"),
        ("Contact", "contact_T"),
        ("Demagnetizing", "demag_T"),
        ("Distortion correction", "distortion_T"),
        ("Applied", "applied_T"),
    )
    # For a modulated structure only the sum of the lattice sum and the continuum
    # outside the sphere is physical, so the breakdown shows them as one row.
    _MERGED_DIPOLAR_LORENTZ_LABEL = "Dipolar (incl. Lorentz)"
    # Distribution payload ``terms`` keys, in the order the breakdown lists them.
    _DISTRIBUTION_BREAKDOWN_TERMS = (
        ("Dipolar", "dipolar"),
        ("Lorentz", "lorentz"),
        ("Contact", "contact"),
        ("Demagnetizing", "demag"),
        ("Applied", "applied"),
    )

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._document: ProjectDocument | None = None
        self._host_structure: Structure | None = None
        self._model = MagneticStructureModel()
        self._selected_muon_site: MagneticMuonSiteOption | None = None
        self._distribution_results: list[object] = []
        self._field_results_state: dict[str, object] | None = None
        self._dipolar_tensor_selector_state: dict[str, object] = {"mode": "assigned"}
        self._dipolar_tensor_results_state: list[dict[str, object]] = []
        self._nuclear_relaxation_options_state: dict[str, object] = (
            NuclearRelaxationOptions().to_json()
        )
        self._nuclear_relaxation_results_state: dict[str, object] = {}
        self._restoring_form_state = False
        self._restoring_title = False

        self._title = QLabel("Magnetic Fields")
        style_section_title(self._title)
        self._title_edit = QLineEdit("Magnetic Fields")
        self._title_edit.setVisible(False)
        self._title_edit.setMaxLength(80)
        self._title_edit.editingFinished.connect(self._on_title_edit_finished)
        self._subtitle = QLabel("Configure the magnetic model used for local field calculations.")
        self._subtitle.setWordWrap(True)
        style_section_subtitle(self._subtitle)
        self._pin_button = QToolButton()
        self._pin_button.setIcon(icon("save_workspace"))
        self._pin_button.setToolTip("Save this magnetic-field workspace")
        self._pin_button.clicked.connect(self._emit_pin_requested)
        self._delete_workspace_button = QToolButton()
        self._delete_workspace_button.setIcon(icon("delete"))
        self._delete_workspace_button.setToolTip("Delete this saved magnetic-field workspace")
        self._delete_workspace_button.setVisible(False)
        self._delete_workspace_button.clicked.connect(self.delete_requested.emit)

        self._structure_summary = QLabel("Magnetic structure: -")
        self._structure_summary.setWordWrap(True)
        self._classification = QLabel("Structure type: -")
        self._component_summary_table = self._summary_table(
            self._COMPONENT_SUMMARY_HEADERS, stretch_column=4
        )
        self._component_summary_table.setVisible(False)
        self._edit_structure_button = QPushButton("Edit Magnetic Structure...")
        self._edit_structure_button.clicked.connect(self._open_structure_builder)

        self._site_summary = QLabel("Muon site: -")
        self._site_summary.setWordWrap(True)
        style_hint_text(self._site_summary)
        self._select_site_button = QPushButton("Select Muon Site...")
        self._select_site_button.clicked.connect(self._open_muon_site_dialog)
        self._include_equivalent_sites = QCheckBox("Include equivalent sites")
        self._include_equivalent_sites.setToolTip(
            "Calculate fields for all symmetry-equivalent positions of the selected muon site."
        )
        self._include_equivalent_sites.toggled.connect(self._on_form_setting_changed)

        self._sphere_radio = QRadioButton("Sphere")
        self._plate_radio = QRadioButton("Infinite plate")
        self._none_radio = QRadioButton("None")
        self._advanced_demag_radio = QRadioButton("Advanced shape")
        self._none_radio.setChecked(True)
        self._demag_sample = DemagnetizingSample()
        self._advanced_demag_summary = QLabel("No advanced shape selected.")
        self._advanced_demag_summary.setWordWrap(True)
        self._advanced_demag_summary.setVisible(False)
        style_hint_text(self._advanced_demag_summary)
        self._advanced_demag_button = QPushButton("Advanced...")
        self._advanced_demag_button.clicked.connect(self._open_advanced_demag_dialog)
        self._sphere_radio.toggled.connect(self._on_demag_shape_toggled)
        self._plate_radio.toggled.connect(self._on_demag_shape_toggled)
        self._none_radio.toggled.connect(self._on_demag_shape_toggled)
        self._advanced_demag_radio.toggled.connect(self._on_demag_shape_toggled)
        self._measured_magnetization_check = QCheckBox("Measured bulk magnetisation")
        self._measured_magnetization_check.setToolTip(
            "Use a measured bulk magnetisation, for example from magnetometry, for the demagnetizing field "
            "instead of the model's uniform moment, which assumes a single domain."
        )
        self._measured_magnetization_check.toggled.connect(self._on_measured_magnetization_toggled)
        self._measured_magnetization_inputs = tuple(
            self._measured_magnetization_input(axis) for axis in ("x", "y", "z")
        )
        for edit in self._measured_magnetization_inputs:
            edit.textChanged.connect(self._on_form_setting_changed)
        self._measured_magnetization_unit = QLabel("μB per formula unit")
        style_hint_text(self._measured_magnetization_unit)
        self._applied_field_inputs = tuple(
            self._applied_field_input(axis) for axis in ("x", "y", "z")
        )
        for edit in self._applied_field_inputs:
            edit.textChanged.connect(self._on_form_setting_changed)
        self._applied_field_unit = QLabel("T (Cartesian x, y, z)")
        style_hint_text(self._applied_field_unit)

        default_supercell = MagneticFieldOptions().supercell
        self._calc_cells_a = self._calculation_cell_input(default_supercell[0])
        self._calc_cells_b = self._calculation_cell_input(default_supercell[1])
        self._calc_cells_c = self._calculation_cell_input(default_supercell[2])
        self._lookup_angles = self._lookup_angle_input(MagneticFieldOptions().nangles)
        self._phase_samples = self._incommensurate_sampling_input(
            MagneticFieldOptions().phase_sample_count
        )
        self._phase_sampling_method = MagneticFieldOptions().phase_sampling
        self._histogram_bins = self._histogram_bin_input(MagneticFieldOptions().histogram_bins)
        self._use_approximants = False
        self._approximant_tolerance = MagneticFieldOptions().approximant_tolerance
        self._approximant_max_period = MagneticFieldOptions().approximant_max_period
        self._contact_nnn = MagneticFieldOptions().nnn
        self._contact_rcont = MagneticFieldOptions().rcont
        self._hyperfine_coupling = self._hyperfine_coupling_input()
        self._hyperfine_units = QComboBox()
        for unit_key, label in CONTACT_COUPLING_UNIT_OPTIONS:
            self._hyperfine_units.addItem(label, unit_key)
        for spinbox in (
            self._calc_cells_a,
            self._calc_cells_b,
            self._calc_cells_c,
            self._lookup_angles,
            self._phase_samples,
            self._histogram_bins,
        ):
            spinbox.valueChanged.connect(self._on_form_setting_changed)
        self._hyperfine_coupling.textChanged.connect(self._on_form_setting_changed)
        self._hyperfine_units.currentIndexChanged.connect(self._on_form_setting_changed)

        self._calculate_button = QPushButton("Calculate Dipolar Field")
        self._calculate_button.clicked.connect(self._calculate_fields)
        self._dipolar_tensor_button = QPushButton("Dipolar Tensor...")
        self._dipolar_tensor_button.clicked.connect(self._open_dipolar_tensor_dialog)
        self._nuclear_relaxation_button = QPushButton("Nuclear Relaxation...")
        self._nuclear_relaxation_button.clicked.connect(self._open_nuclear_relaxation_dialog)
        self._calculation_settings_summary = QLabel("")
        self._calculation_settings_summary.setWordWrap(True)
        style_hint_text(self._calculation_settings_summary)
        self._calculation_settings_button = QPushButton("Calculation Settings...")
        self._calculation_settings_button.clicked.connect(self._open_calculation_settings_dialog)
        self._open_distribution_button = QPushButton("Open Distribution Plot")
        self._open_distribution_button.setEnabled(False)
        self._open_distribution_button.clicked.connect(self._open_distribution_plot)
        self._polarisation_button = QPushButton("Polarisation →")
        self._polarisation_button.setToolTip(
            "Simulate the muon polarisation from these field results on the Polarisation page."
        )
        self._polarisation_button.setEnabled(False)
        self._polarisation_button.clicked.connect(self.open_polarisation_requested.emit)
        self._calculation_status = QLabel("Results: -")
        self._calculation_status.setWordWrap(True)
        style_hint_text(self._calculation_status)
        self._result_table = create_workflow_table(
            ("Site", "Bx (T)", "By (T)", "Bz (T)", "|B| (T)", "Result"),
            stretch_last_section=False,
        )
        self._result_table.itemSelectionChanged.connect(self._sync_distribution_plot_button)
        self._result_table.itemSelectionChanged.connect(self._refresh_field_breakdown)
        self._configure_result_table_columns()
        self._breakdown_label = QLabel("Field breakdown")
        self._breakdown_table = self._summary_table(self._BREAKDOWN_HEADERS, stretch_column=0)
        self._breakdown_hint = QLabel("")
        self._breakdown_hint.setWordWrap(True)
        style_hint_text(self._breakdown_hint)
        self._breakdown_contact_warning = QLabel("")
        self._breakdown_contact_warning.setWordWrap(True)
        style_warning_text(self._breakdown_contact_warning)
        self._breakdown_contact_warning.setVisible(False)

        self._apply_page_styles()
        self._build_layout()
        self._refresh_demag_summary()
        self._refresh_calculation_settings_summary()
        self._refresh_field_breakdown()
        self._sync_enabled_state()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        header_row = QHBoxLayout()
        header_row.addWidget(self._title)
        header_row.addWidget(self._title_edit)
        header_row.addStretch(1)
        header_row.addWidget(self._pin_button)
        header_row.addWidget(self._delete_workspace_button)
        layout.addLayout(header_row)
        layout.addWidget(self._subtitle)

        structure_group = QGroupBox("Magnetic Structure")
        structure_layout = QVBoxLayout(structure_group)
        structure_header_layout = QHBoxLayout()
        structure_layout.addLayout(structure_header_layout)
        structure_left_layout = QVBoxLayout()
        structure_left_layout.addWidget(self._structure_summary)
        structure_left_layout.addWidget(self._classification)
        action_row = QHBoxLayout()
        action_row.addWidget(self._edit_structure_button)
        action_row.addStretch(1)
        structure_left_layout.addLayout(action_row)
        structure_right_layout = QVBoxLayout()
        hyperfine_label = QLabel("Hyperfine coupling")
        hyperfine_label.setToolTip(
            "Contact hyperfine coupling for the muesr ACont term. Internally converted to Å⁻³."
        )
        hyperfine_row = QHBoxLayout()
        hyperfine_row.addWidget(self._hyperfine_coupling)
        hyperfine_row.addWidget(self._hyperfine_units)
        structure_right_layout.addWidget(hyperfine_label)
        structure_right_layout.addLayout(hyperfine_row)
        structure_right_layout.addStretch(1)
        structure_header_layout.addLayout(structure_left_layout, 1)
        structure_header_layout.addLayout(structure_right_layout)
        structure_layout.addWidget(self._component_summary_table)
        layout.addWidget(structure_group)

        site_group = QGroupBox("Muon Site")
        site_layout = QVBoxLayout(site_group)
        site_action_row = QHBoxLayout()
        site_action_row.addWidget(self._site_summary, 1)
        site_action_row.addWidget(self._select_site_button)
        site_layout.addLayout(site_action_row)
        site_layout.addWidget(self._include_equivalent_sites)
        layout.addWidget(site_group)

        results_group = QGroupBox("Calculation")
        results_layout = QVBoxLayout(results_group)
        demag_row = QHBoxLayout()
        demag_row.addWidget(QLabel("Demagnetizing field"))
        for button in (
            self._sphere_radio,
            self._plate_radio,
            self._none_radio,
            self._advanced_demag_radio,
        ):
            demag_row.addWidget(button)
        demag_row.addWidget(self._advanced_demag_button)
        demag_row.addStretch(1)
        results_layout.addLayout(demag_row)
        results_layout.addWidget(self._advanced_demag_summary)
        magnetization_row = QHBoxLayout()
        magnetization_row.addWidget(self._measured_magnetization_check)
        for edit in self._measured_magnetization_inputs:
            magnetization_row.addWidget(edit)
        magnetization_row.addWidget(self._measured_magnetization_unit)
        magnetization_row.addStretch(1)
        results_layout.addLayout(magnetization_row)
        applied_row = QHBoxLayout()
        applied_label = QLabel("Applied field")
        applied_label.setToolTip(
            "Applied field added to every muon site, in the crystal Cartesian frame. "
            "It appears as the Applied row of the field breakdown."
        )
        applied_row.addWidget(applied_label)
        for edit in self._applied_field_inputs:
            applied_row.addWidget(edit)
        applied_row.addWidget(self._applied_field_unit)
        applied_row.addStretch(1)
        results_layout.addLayout(applied_row)
        settings_row = QHBoxLayout()
        settings_row.addWidget(self._calculation_settings_summary, 1)
        settings_row.addWidget(self._calculation_settings_button)
        results_layout.addLayout(settings_row)
        action_row = QHBoxLayout()
        action_row.addWidget(self._calculate_button)
        action_row.addWidget(self._dipolar_tensor_button)
        action_row.addWidget(self._nuclear_relaxation_button)
        action_row.addWidget(self._open_distribution_button)
        action_row.addStretch(1)
        action_row.addWidget(self._polarisation_button)
        results_layout.addLayout(action_row)
        results_layout.addWidget(self._calculation_status)
        results_layout.addWidget(self._result_table)
        results_layout.addWidget(self._breakdown_label)
        results_layout.addWidget(self._breakdown_table)
        results_layout.addWidget(self._breakdown_contact_warning)
        results_layout.addWidget(self._breakdown_hint)
        layout.addWidget(results_group, 1)
        layout.addStretch(1)

    def _apply_page_styles(self) -> None:
        palette = current_palette()
        self.setStyleSheet(
            "QRadioButton, QCheckBox {"
            "background: transparent;"
            "spacing: 10px;"
            "padding-top: 3px;"
            "padding-bottom: 3px;"
            "}"
            "QWidget#magneticMuonSiteInlinePanel {"
            "background: transparent;"
            "}"
            "QStackedWidget {"
            "background: transparent;"
            "}"
            "QStackedWidget > QWidget {"
            "background: transparent;"
            "}"
            "QLabel {"
            "background: transparent;"
            "}"
            "QCheckBox:disabled {"
            f"color: {palette.text_muted};"
            "}"
        )

    def set_document(
        self,
        document: ProjectDocument | None,
        *,
        magnetic_structure: Mapping[str, object] | None = None,
        title: str = "Magnetic Fields",
        title_editable: bool = False,
        deletable: bool = False,
    ) -> None:
        self._document = document
        self._host_structure = _active_host_structure(document)
        self.set_workspace_title(title, editable=title_editable, deletable=deletable)
        self._model = self._model_from_document(
            document, self._host_structure, magnetic_structure=magnetic_structure
        )
        self._selected_muon_site = self._default_muon_site()
        self._refresh_measured_magnetization_unit()
        self._sync_enabled_state()
        self._refresh_summary()
        self._refresh_site_summary()

    def collect_state(self) -> dict[str, Any]:
        """Collect magnetic-field tab settings for project persistence."""
        return {
            "demag_shape": self._demag_shape(),
            "demag_sample": self._demag_sample.to_json(),
            "use_measured_magnetization": self._measured_magnetization_check.isChecked(),
            "measured_magnetization": [
                self._vector_component_or_zero(edit) for edit in self._measured_magnetization_inputs
            ],
            "applied_field_T": [
                self._vector_component_or_zero(edit) for edit in self._applied_field_inputs
            ],
            "supercell": [
                int(self._calc_cells_a.value()),
                int(self._calc_cells_b.value()),
                int(self._calc_cells_c.value()),
            ],
            "include_equivalent_sites": self._include_equivalent_sites.isChecked(),
            "phase_lookup_angles": int(self._lookup_angles.value()),
            "incommensurate_samples": int(self._phase_samples.value()),
            "phase_sampling": str(self._phase_sampling_method),
            "histogram_bins": int(self._histogram_bins.value()),
            "use_commensurate_approximants": bool(self._use_approximants),
            "approximant_tolerance": float(self._approximant_tolerance),
            "approximant_max_period": int(self._approximant_max_period),
            "contact_nnn": int(self._contact_nnn),
            "contact_rcont": float(self._contact_rcont),
            "hyperfine_coupling": self._hyperfine_coupling.text().strip(),
            "hyperfine_coupling_unit": self._current_hyperfine_unit(),
            "selected_muon_site": self._muon_site_to_state(self._selected_muon_site),
            "magnetic_field_results": (
                dict(self._field_results_state)
                if isinstance(self._field_results_state, dict)
                else None
            ),
            "dipolar_tensor_selector": dict(self._dipolar_tensor_selector_state),
            "dipolar_tensor_results": [
                dict(result) for result in self._dipolar_tensor_results_state
            ],
            "nuclear_relaxation_options": dict(self._nuclear_relaxation_options_state),
            "nuclear_relaxation_results": dict(self._nuclear_relaxation_results_state),
        }

    def collect_workspace_state(self) -> dict[str, Any]:
        """Collect the complete workspace snapshot, including magnetic model."""
        return {
            "title": self.workspace_title(),
            "magnetic_structure": to_jsonable(self._model.to_json()),
            "form": self.collect_state(),
        }

    def workspace_title(self) -> str:
        title = self._title_edit.text() if self._title_edit.isVisible() else self._title.text()
        return title.strip() or "Magnetic Fields"

    def set_workspace_title(
        self,
        title: str,
        *,
        editable: bool = False,
        deletable: bool = False,
    ) -> None:
        normalized = str(title).strip() or "Magnetic Fields"
        self._restoring_title = True
        try:
            self._title.setText(normalized)
            self._title.setVisible(not editable)
            self._title_edit.setText(normalized)
            self._title_edit.setVisible(editable)
            self._delete_workspace_button.setVisible(deletable)
        finally:
            self._restoring_title = False

    def set_form_state(self, state: Mapping[str, object] | None) -> None:
        """Restore magnetic-field tab settings from project UI state."""
        if not isinstance(state, Mapping) or not state:
            self._reset_form_state()
            return

        selected_site = self._muon_site_from_state(state.get("selected_muon_site"))
        self._restoring_form_state = True
        try:
            demag_shape = str(state.get("demag_shape", "")).strip()
            raw_demag_sample = state.get("demag_sample")
            self._demag_sample = DemagnetizingSample.from_json(raw_demag_sample)
            if demag_shape in {"none", "no_demag", "off"}:
                self._none_radio.setChecked(True)
            elif demag_shape in {"infinite_plate", "plate"}:
                self._plate_radio.setChecked(True)
            elif demag_shape == "sphere":
                self._sphere_radio.setChecked(True)
            elif demag_shape in ADVANCED_DEMAG_SHAPES:
                if self._demag_sample.shape == "none":
                    self._demag_sample = DemagnetizingSample(shape=demag_shape)
                self._advanced_demag_radio.setChecked(True)
            else:
                self._none_radio.setChecked(True)
            self._set_measured_magnetization_from_state(
                state.get("use_measured_magnetization"), state.get("measured_magnetization")
            )
            self._set_vector_inputs(self._applied_field_inputs, state.get("applied_field_T"))

            supercell = self._state_int_triplet(state.get("supercell"))
            if supercell is not None:
                self._calc_cells_a.setValue(supercell[0])
                self._calc_cells_b.setValue(supercell[1])
                self._calc_cells_c.setValue(supercell[2])

            self._include_equivalent_sites.setChecked(
                bool(state.get("include_equivalent_sites", False))
            )
            self._set_spinbox_from_state(
                self._lookup_angles,
                state.get("phase_lookup_angles", state.get("nangles")),
            )
            self._set_spinbox_from_state(
                self._phase_samples,
                state.get(
                    "incommensurate_samples", state.get("phase_sample_count", state.get("nangles"))
                ),
            )
            self._phase_sampling_method = self._normalize_phase_sampling_method(
                state.get("phase_sampling", self._phase_sampling_method)
            )
            self._set_spinbox_from_state(self._histogram_bins, state.get("histogram_bins"))
            self._use_approximants = bool(state.get("use_commensurate_approximants", False))
            self._approximant_tolerance = self._state_approximant_tolerance(
                state.get("approximant_tolerance")
            )
            self._approximant_max_period = self._state_approximant_max_period(
                state.get("approximant_max_period")
            )
            self._contact_nnn = self._state_contact_nnn(state.get("contact_nnn"))
            self._contact_rcont = self._state_contact_rcont(state.get("contact_rcont"))
            raw_hyperfine = state.get("hyperfine_coupling")
            if raw_hyperfine is not None:
                self._hyperfine_coupling.setText(str(raw_hyperfine))
            self._set_hyperfine_unit(
                str(state.get("hyperfine_coupling_unit", self._current_hyperfine_unit()))
            )
            tensor_selector = state.get("dipolar_tensor_selector")
            if isinstance(tensor_selector, Mapping):
                self._dipolar_tensor_selector_state = dict(tensor_selector)
            tensor_results = state.get("dipolar_tensor_results")
            if isinstance(tensor_results, Sequence) and not isinstance(
                tensor_results, (str, bytes)
            ):
                self._dipolar_tensor_results_state = [
                    dict(result) for result in tensor_results if isinstance(result, Mapping)
                ]
            nuclear_options = state.get("nuclear_relaxation_options")
            if isinstance(nuclear_options, Mapping):
                self._nuclear_relaxation_options_state = dict(nuclear_options)
            nuclear_results = state.get("nuclear_relaxation_results")
            if isinstance(nuclear_results, Mapping):
                self._nuclear_relaxation_results_state = dict(nuclear_results)
            field_results = state.get("magnetic_field_results")
            self._field_results_state = (
                dict(field_results) if isinstance(field_results, Mapping) else None
            )

            if selected_site is not None:
                self._selected_muon_site = selected_site
        finally:
            self._restoring_form_state = False

        self._sync_enabled_state()
        self._refresh_site_summary()
        self._refresh_demag_summary()
        self._refresh_calculation_settings_summary()
        self._restore_field_results()

    def _reset_form_state(self) -> None:
        defaults = MagneticFieldOptions()
        self._restoring_form_state = True
        try:
            self._none_radio.setChecked(True)
            self._demag_sample = DemagnetizingSample()
            self._set_measured_magnetization_from_state(False, None)
            self._set_vector_inputs(self._applied_field_inputs, None)
            self._calc_cells_a.setValue(defaults.supercell[0])
            self._calc_cells_b.setValue(defaults.supercell[1])
            self._calc_cells_c.setValue(defaults.supercell[2])
            self._include_equivalent_sites.setChecked(False)
            self._lookup_angles.setValue(defaults.nangles)
            self._phase_samples.setValue(defaults.phase_sample_count)
            self._phase_sampling_method = defaults.phase_sampling
            self._histogram_bins.setValue(defaults.histogram_bins)
            self._use_approximants = False
            self._approximant_tolerance = defaults.approximant_tolerance
            self._approximant_max_period = defaults.approximant_max_period
            self._contact_nnn = defaults.nnn
            self._contact_rcont = defaults.rcont
            self._hyperfine_coupling.setText("0.0")
            self._set_hyperfine_unit("angstrom^-3")
            self._selected_muon_site = self._default_muon_site()
            self._dipolar_tensor_selector_state = {"mode": "assigned"}
            self._dipolar_tensor_results_state = []
            self._nuclear_relaxation_options_state = NuclearRelaxationOptions().to_json()
            self._nuclear_relaxation_results_state = {}
        finally:
            self._restoring_form_state = False

        self._sync_enabled_state()
        self._refresh_site_summary()
        self._refresh_demag_summary()
        self._refresh_calculation_settings_summary()
        self._clear_results(clear_state=True)

    def _model_from_document(
        self,
        document: ProjectDocument | None,
        host: Structure | None,
        *,
        magnetic_structure: Mapping[str, object] | None = None,
    ) -> MagneticStructureModel:
        if document is None or host is None:
            return MagneticStructureModel()
        payload = (
            magnetic_structure
            if magnetic_structure is not None
            else document.project.magnetic_structure
        )
        if not isinstance(payload, dict):
            payload = host.properties.get("magnetic_structure")
        if isinstance(payload, dict) and "components" in payload:
            return MagneticStructureModel.from_json(payload)
        return magnetic_model_from_legacy_structure(host)

    def _sync_enabled_state(self) -> None:
        has_host = self._host_structure is not None
        self._edit_structure_button.setEnabled(has_host)
        self._select_site_button.setEnabled(has_host)
        self._include_equivalent_sites.setEnabled(has_host)
        site_reason = self._muon_site_unavailable_reason() if has_host else None
        has_components = bool(enabled_magnetic_components(self._model))
        self._calculate_button.setEnabled(has_host and has_components and site_reason is None)
        self._dipolar_tensor_button.setEnabled(has_host and site_reason is None)
        self._nuclear_relaxation_button.setEnabled(has_host and site_reason is None)
        for button in (self._dipolar_tensor_button, self._nuclear_relaxation_button):
            button.setToolTip(site_reason or "")
        if site_reason is None and has_host and not has_components:
            site_reason = (
                "Enable at least one magnetic-structure component first."
                if self._model.components
                else "Build or import a magnetic structure first."
            )
        self._calculate_button.setToolTip(site_reason or "")
        self._calculation_settings_button.setEnabled(has_host)
        for spinbox in (
            self._calc_cells_a,
            self._calc_cells_b,
            self._calc_cells_c,
            self._phase_samples,
            self._histogram_bins,
        ):
            spinbox.setEnabled(has_host)
        self._hyperfine_coupling.setEnabled(has_host)
        self._hyperfine_units.setEnabled(has_host)
        self._measured_magnetization_check.setEnabled(has_host)
        measured = has_host and self._measured_magnetization_check.isChecked()
        for edit in self._measured_magnetization_inputs:
            edit.setEnabled(measured)
        for edit in self._applied_field_inputs:
            edit.setEnabled(has_host)

    def _refresh_summary(self) -> None:
        if self._host_structure is None:
            self._structure_summary.setText("Magnetic structure: no host structure")
            self._classification.setText("Structure type: -")
            self._set_summary_rows(self._component_summary_table, ())
            self._component_summary_table.setVisible(False)
            return
        count = len(self._model.components)
        enabled = enabled_magnetic_components(self._model)
        disabled = count - len(enabled)
        plural = "" if count == 1 else "s"
        suffix = f" ({disabled} disabled)" if disabled else ""
        self._structure_summary.setText(f"Magnetic structure: {count} component{plural}{suffix}")
        self._set_summary_rows(
            self._component_summary_table,
            [
                (
                    component.label,
                    component.kind.replace("_", " ").title(),
                    _selector_label(component.selector, self._host_structure),
                    _component_q_label(self._model, component),
                    _component_moment_label(component),
                )
                for component in enabled
            ],
        )
        self._component_summary_table.setVisible(bool(enabled))
        if count == disabled:
            self._classification.setText("Structure type: -")
            return
        try:
            classification = classify_magnetic_model(self._host_structure, self._model)
        except Exception:
            self._classification.setText("Structure type: unavailable")
            return
        self._classification.setText(f"Structure type: {classification.replace('_', ' ')}")

    def _open_structure_builder(self) -> None:
        if self._host_structure is None:
            return
        dialog = MagneticStructureBuilderDialog(self)
        dialog.set_context(self._host_structure, self._model)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._set_model(dialog.model())

    def _set_model(self, model: MagneticStructureModel) -> None:
        self._model = model
        self._refresh_summary()
        self._sync_enabled_state()
        self._clear_results(clear_state=True)
        self.magnetic_structure_changed.emit(model.to_json())

    def selected_muon_site(self) -> MagneticMuonSiteOption | None:
        return self._selected_muon_site

    def _open_muon_site_dialog(self) -> None:
        dialog = MuonSiteSelectionDialog(
            self._document, self._host_structure, self._selected_muon_site, self
        )
        dialog.open_muon_sites_requested.connect(self.open_muon_sites_requested)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._selected_muon_site = dialog.accepted_site()
        self._sync_enabled_state()
        self._refresh_site_summary()
        self._clear_results(clear_state=True)
        self._emit_form_state_changed()

    def _default_muon_site(self) -> MagneticMuonSiteOption | None:
        if self._document is None or self._host_structure is None:
            return None
        # The lowest-energy site pinned to the Muon Sites summary comes first.
        for options in (
            _relaxation_cluster_muon_site_options(
                self._document, self._host_structure, pinned_only=True
            ),
            _relaxation_cluster_muon_site_options(self._document, self._host_structure),
            _uep_cluster_muon_site_options(self._document, self._host_structure),
        ):
            if options:
                return options[0]
        return None

    def _muon_site_unavailable_reason(self) -> str | None:
        """Return why the selected muon site cannot be calculated, or ``None`` when it can."""

        site = self._selected_muon_site
        if site is None:
            return "Choose a muon site first."
        if self._host_structure is None:
            return None
        return muon_site_clearance_error(
            self._host_structure, site.fractional_position, label=site.label
        )

    def _refresh_site_summary(self) -> None:
        if self._host_structure is None:
            self._site_summary.setText("Muon site: load a host structure first.")
            return
        if self._selected_muon_site is None:
            self._site_summary.setText("Choose a muon site.")
            return
        site = self._selected_muon_site
        source = site.source_label or _site_source_label(site.source)
        cluster = f" ({site.cluster_label})" if site.cluster_label else ""
        distortion = (
            " Local field corrected for muon-induced displacements."
            if site.include_distortion
            else ""
        )
        estimate = f" {_UNRELAXED_ESTIMATE_NOTE}" if is_unrelaxed_site_source(site.source) else ""
        clearance_error = self._muon_site_unavailable_reason()
        refused = f" {clearance_error}" if clearance_error else ""
        self._site_summary.setText(
            f"Muon site: {site.label}{cluster} from {source} at "
            f"({site.fractional_position[0]:.4f}, {site.fractional_position[1]:.4f}, "
            f"{site.fractional_position[2]:.4f}).{distortion}{estimate}{refused}"
        )

    @staticmethod
    def _fractional_position_input(value: str) -> QLineEdit:
        edit = QLineEdit(value)
        edit.setValidator(QDoubleValidator(0.0, 1.0, 8, edit))
        edit.setPlaceholderText("0.0")
        return edit

    @staticmethod
    def _calculation_cell_input(value: int) -> QSpinBox:
        spinbox = QSpinBox()
        spinbox.setRange(1, 500)
        spinbox.setValue(int(value))
        spinbox.setToolTip(
            "Unit-cell repeats used for the muesr lattice summation. Larger values are slower but reduce cutoff effects."
        )
        spinbox.setMinimumWidth(104)
        return spinbox

    @staticmethod
    def _incommensurate_sampling_input(value: int) -> QSpinBox:
        spinbox = QSpinBox()
        spinbox.setRange(12, MAX_INCOMMENSURATE_PHASE_SAMPLES)
        spinbox.setSingleStep(1_024)
        spinbox.setValue(int(value))
        spinbox.setToolTip(
            "Number of phase angles sampled for incommensurate field distributions. "
            "Very high values are slower and use more memory, but can resolve sharper features."
        )
        spinbox.setMinimumWidth(112)
        return spinbox

    @staticmethod
    def _lookup_angle_input(value: int) -> QSpinBox:
        spinbox = QSpinBox()
        spinbox.setRange(12, MAX_INCOMMENSURATE_LOOKUP_ANGLES)
        spinbox.setSingleStep(512)
        spinbox.setValue(int(value))
        spinbox.setToolTip(
            "Phase lookup resolution passed to muesr for each Fourier component. "
            "This is separate from the number of distribution samples."
        )
        spinbox.setMinimumWidth(112)
        return spinbox

    @staticmethod
    def _histogram_bin_input(value: int) -> QSpinBox:
        spinbox = QSpinBox()
        spinbox.setRange(10, 10_000)
        spinbox.setSingleStep(64)
        spinbox.setValue(int(value))
        spinbox.setToolTip(
            "Number of bins used for incommensurate p(B) histograms. More bins narrow the bin width."
        )
        spinbox.setMinimumWidth(104)
        return spinbox

    @staticmethod
    def _measured_magnetization_input(axis: str) -> QLineEdit:
        edit = QLineEdit("0.0")
        edit.setValidator(QDoubleValidator(-1_000.0, 1_000.0, 6, edit))
        edit.setToolTip(
            f"Cartesian {axis} component of the measured bulk magnetisation, in μB per formula unit."
        )
        edit.setMaximumWidth(80)
        edit.setEnabled(False)
        return edit

    @staticmethod
    def _applied_field_input(axis: str) -> QLineEdit:
        edit = QLineEdit("0.0")
        edit.setValidator(QDoubleValidator(-1_000.0, 1_000.0, 6, edit))
        edit.setToolTip(f"Cartesian {axis} component of the applied field, in T.")
        edit.setMaximumWidth(80)
        return edit

    @staticmethod
    def _hyperfine_coupling_input() -> QLineEdit:
        edit = QLineEdit("0.0")
        edit.setValidator(QDoubleValidator(-1_000_000.0, 1_000_000.0, 8, edit))
        edit.setToolTip("Set to zero to omit the contact hyperfine contribution.")
        edit.setMinimumWidth(104)
        return edit

    @staticmethod
    def _summary_table(headers: tuple[str, ...], *, stretch_column: int) -> QTableWidget:
        """Build a read-only table that grows to fit its first rows and scrolls beyond them."""

        table = create_workflow_table(
            headers,
            resize_to_contents_columns=tuple(
                column for column in range(len(headers)) if column != stretch_column
            ),
            selection_mode=QAbstractItemView.SelectionMode.NoSelection,
            stretch_last_section=stretch_column == len(headers) - 1,
        )
        table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        table.setSizeAdjustPolicy(QAbstractItemView.SizeAdjustPolicy.AdjustToContents)
        table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        return table

    @classmethod
    def _set_summary_rows(cls, table: QTableWidget, rows: Sequence[Sequence[str]]) -> None:
        table.setRowCount(0)
        for values in rows:
            row = table.rowCount()
            table.insertRow(row)
            for column, value in enumerate(values):
                table.setItem(row, column, QTableWidgetItem(str(value)))
        table.resizeRowsToContents()
        visible_rows = min(table.rowCount(), cls._SUMMARY_MAX_VISIBLE_ROWS)
        table.setMaximumHeight(
            table.horizontalHeader().sizeHint().height()
            + sum(table.rowHeight(row) for row in range(visible_rows))
            + 2 * table.frameWidth()
            + 2
        )

    def _configure_result_table_columns(self) -> None:
        header = self._result_table.horizontalHeader()
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        for column in (1, 2, 3, 4):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(5, QHeaderView.ResizeMode.ResizeToContents)

    def _calculate_fields(self) -> None:
        self._clear_results(clear_state=True)
        if self._host_structure is None:
            self._calculation_status.setText("Results: load a host structure first.")
            return
        if not self._model.components:
            self._calculation_status.setText("Results: build or import a magnetic structure first.")
            return
        if not enabled_magnetic_components(self._model):
            self._calculation_status.setText(
                "Results: enable at least one magnetic-structure component first."
            )
            return
        try:
            site = self.selected_muon_site()
        except ValueError as exc:
            self._calculation_status.setText(f"Results: invalid muon site - {exc}")
            return
        if site is None:
            self._calculation_status.setText("Results: select a muon site first.")
            return
        site_reason = self._muon_site_unavailable_reason()
        if site_reason is not None:
            self._calculation_status.setText(f"Results: {site_reason}")
            return

        try:
            options = self._field_options()
        except ValueError as exc:
            self._calculation_status.setText(f"Results: invalid calculation input - {exc}")
            return
        try:
            distorted_structures = [self._distorted_structure_for_site(site)]
            classification = classify_magnetic_model(self._host_structure, self._model)
            corrected = (
                " with muon-induced distortion corrections" if site.include_distortion else ""
            )
            if classification == "commensurate":
                results = calculate_commensurate_fields(
                    self._host_structure,
                    self._model,
                    [site.fractional_position],
                    options=options,
                    distorted_structures=distorted_structures,
                )
                _raise_for_non_finite_fields(results)
                self._populate_commensurate_results(results, site_source=site.source)
                self._calculation_status.setText(
                    f"Results: calculated {len(results)} commensurate field value(s){corrected}."
                    + self._estimate_status_suffix()
                )
                self._emit_form_state_changed()
            else:
                approximant = find_commensurate_approximant(
                    self._host_structure,
                    self._model,
                    tolerance=options.approximant_tolerance,
                    max_period=options.approximant_max_period,
                )
                if approximant is not None and self._use_approximants:
                    results = calculate_commensurate_approximant_fields(
                        self._host_structure,
                        self._model,
                        [site.fractional_position],
                        options=options,
                        distorted_structures=distorted_structures,
                    )
                    _raise_for_non_finite_fields(results)
                    self._populate_commensurate_results(results, site_source=site.source)
                    self._calculation_status.setText(
                        f"Results: calculated {len(results)} field value(s) of the commensurate approximant "
                        f"{approximant.describe()} (q moved by at most {approximant.max_deviation_rlu:.2g} r.l.u.)"
                        f"{corrected}." + self._estimate_status_suffix()
                    )
                else:
                    results = calculate_incommensurate_field_distributions(
                        self._host_structure,
                        self._model,
                        [site.fractional_position],
                        options=options,
                        distorted_structures=distorted_structures,
                    )
                    _raise_for_non_finite_fields(results)
                    self._populate_distribution_results(results, site_source=site.source)
                    status = f"Results: sampled {len(results)} incommensurate field distribution(s){corrected}"
                    flagged = sum(
                        1
                        for result in results
                        for check in getattr(result, "convergence", ())
                        if not check.converged
                    )
                    if flagged:
                        status += f"; {flagged} convergence check(s) need attention, select a row for advice"
                    status += "."
                    if approximant is not None:
                        status += (
                            f" q is within {approximant.max_deviation_rlu:.2g} r.l.u. of the commensurate "
                            f"approximant {approximant.describe()}; turn on commensurate approximants in "
                            "Calculation Settings to list its finite fields."
                        )
                    self._calculation_status.setText(status + self._estimate_status_suffix())
                self._emit_form_state_changed()
        except MagneticFieldDependencyError as exc:
            self._calculation_status.setText(f"Results unavailable: {exc}")
        except _NonFiniteFieldError as exc:
            self._calculation_status.setText(f"Calculation failed: {exc}")
            # The earlier results were cleared on entry; save that so they do not come back.
            self._emit_form_state_changed()
        except MagneticFieldError as exc:
            self._calculation_status.setText(f"Calculation failed: {exc}")
        except Exception as exc:  # pragma: no cover - defensive UI boundary
            self._calculation_status.setText(f"Calculation failed: {exc}")

    def _distorted_structure_for_site(self, site: MagneticMuonSiteOption) -> Structure | None:
        if not site.include_distortion:
            return None
        structure_id = site.distortion_structure_id
        distorted = None
        if self._document is not None and structure_id is not None:
            distorted = self._document.structures.get(structure_id)
        if distorted is None:
            raise MagneticFieldError(
                "the relaxed structure for the muon-induced distortion correction is no longer available"
            )
        return distorted

    def _open_calculation_settings_dialog(self) -> None:
        dialog = MagneticFieldCalculationSettingsDialog(
            supercell=(
                int(self._calc_cells_a.value()),
                int(self._calc_cells_b.value()),
                int(self._calc_cells_c.value()),
            ),
            nangles=int(self._lookup_angles.value()),
            phase_sample_count=int(self._phase_samples.value()),
            histogram_bins=int(self._histogram_bins.value()),
            phase_sampling=str(self._phase_sampling_method),
            use_approximants=self._use_approximants,
            approximant_tolerance=self._approximant_tolerance,
            approximant_max_period=self._approximant_max_period,
            contact_nnn=self._contact_nnn,
            contact_rcont=self._contact_rcont,
            contact_tie_check=self._contact_tie_checker(),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        values = dialog.values()
        self._restoring_form_state = True
        try:
            supercell = values.get("supercell")
            if isinstance(supercell, (tuple, list)) and len(supercell) == 3:
                self._calc_cells_a.setValue(int(supercell[0]))
                self._calc_cells_b.setValue(int(supercell[1]))
                self._calc_cells_c.setValue(int(supercell[2]))
            self._lookup_angles.setValue(int(values.get("nangles", self._lookup_angles.value())))
            self._phase_samples.setValue(
                int(values.get("phase_sample_count", self._phase_samples.value()))
            )
            self._phase_sampling_method = self._normalize_phase_sampling_method(
                values.get("phase_sampling", self._phase_sampling_method)
            )
            self._histogram_bins.setValue(
                int(values.get("histogram_bins", self._histogram_bins.value()))
            )
            self._use_approximants = bool(values.get("use_approximants", self._use_approximants))
            self._approximant_tolerance = self._state_approximant_tolerance(
                values.get("approximant_tolerance", self._approximant_tolerance)
            )
            self._approximant_max_period = self._state_approximant_max_period(
                values.get("approximant_max_period", self._approximant_max_period)
            )
            self._contact_nnn = self._state_contact_nnn(
                values.get("contact_nnn", self._contact_nnn)
            )
            self._contact_rcont = self._state_contact_rcont(
                values.get("contact_rcont", self._contact_rcont)
            )
        finally:
            self._restoring_form_state = False
        self._on_form_setting_changed()

    def _open_dipolar_tensor_dialog(self) -> None:
        if self._host_structure is None:
            QMessageBox.information(self, "Dipolar Tensor", "Load a host structure first.")
            return
        site = self.selected_muon_site()
        if site is None:
            QMessageBox.information(self, "Dipolar Tensor", "Select a muon site first.")
            return
        site_reason = self._muon_site_unavailable_reason()
        if site_reason is not None:
            QMessageBox.information(self, "Dipolar Tensor", site_reason)
            return
        include_equivalent = bool(
            self._dipolar_tensor_selector_state.get(
                "include_equivalent_sites",
                self._include_equivalent_sites.isChecked(),
            )
        )
        dialog = DipolarTensorDialog(
            self._host_structure,
            self._model,
            site,
            include_equivalent_sites=include_equivalent,
            selector_state=self._dipolar_tensor_selector_state,
            supercell=(
                int(self._calc_cells_a.value()),
                int(self._calc_cells_b.value()),
                int(self._calc_cells_c.value()),
            ),
            results_state=self._dipolar_tensor_results_state,
            calculate_callback=self._calculate_dipolar_tensors_from_dialog,
            parent=self,
        )
        dialog.exec()
        self._dipolar_tensor_selector_state = dialog.selector_state()
        self._dipolar_tensor_results_state = dialog.results_state()
        self._emit_form_state_changed()

    def _calculate_dipolar_tensors_from_dialog(
        self, dialog: DipolarTensorDialog
    ) -> Sequence[object]:
        if self._host_structure is None:
            raise MagneticFieldError("load a host structure first")
        site = self.selected_muon_site()
        if site is None:
            raise MagneticFieldError("select a muon site first")
        ion_indices = dialog.selected_ion_indices()
        options = MagneticFieldOptions(
            supercell=(
                int(self._calc_cells_a.value()),
                int(self._calc_cells_b.value()),
                int(self._calc_cells_c.value()),
            ),
            find_equivalent_sites=dialog.include_equivalent_sites(),
            include_lorentz=False,
        )
        return calculate_dipolar_tensors_for_atoms(
            self._host_structure,
            [site.fractional_position],
            magnetic_ion_indices=ion_indices,
            options=options,
        )

    def _open_nuclear_relaxation_dialog(self) -> None:
        if self._host_structure is None:
            QMessageBox.information(self, "Nuclear Relaxation", "Load a host structure first.")
            return
        site = self.selected_muon_site()
        if site is None:
            QMessageBox.information(self, "Nuclear Relaxation", "Select a muon site first.")
            return
        site_reason = self._muon_site_unavailable_reason()
        if site_reason is not None:
            QMessageBox.information(self, "Nuclear Relaxation", site_reason)
            return
        distortion_available = (
            self._document is not None
            and site.distortion_structure_id is not None
            and site.distortion_structure_id in self._document.structures
        )
        dialog = NuclearRelaxationDialog(
            muon_site=site,
            distortion_available=distortion_available,
            elements=host_elements(self._host_structure.species),
            options_state=self._nuclear_relaxation_options_state,
            results_state=self._nuclear_relaxation_results_state,
            calculate_callback=self._calculate_nuclear_relaxation_from_dialog,
            parent=self,
        )
        dialog.exec()
        self._nuclear_relaxation_options_state = dialog.options_state()
        self._nuclear_relaxation_results_state = dialog.results_state()
        self._emit_form_state_changed()

    def _calculate_nuclear_relaxation_from_dialog(self, dialog: NuclearRelaxationDialog) -> object:
        if self._host_structure is None:
            raise NuclearRelaxationError("load a host structure first")
        site = self.selected_muon_site()
        if site is None:
            raise NuclearRelaxationError("select a muon site first")
        site_reason = self._muon_site_unavailable_reason()
        if site_reason is not None:
            raise NuclearRelaxationError(site_reason)
        options = dialog.options()
        if options.use_distortion_correction:
            if self._document is None or site.distortion_structure_id is None:
                raise NuclearRelaxationError("selected muon site has no distortion structure")
            distorted = self._document.structures.get(site.distortion_structure_id)
            if distorted is None:
                raise NuclearRelaxationError("selected distortion structure is no longer available")
            return calculate_nuclear_relaxation_with_distortion(
                self._host_structure,
                distorted,
                site.fractional_position,
                site_label=site.label,
                options=options,
            )
        return calculate_nuclear_relaxation(
            self._host_structure,
            site.fractional_position,
            site_label=site.label,
            options=options,
        )

    def _refresh_calculation_settings_summary(self) -> None:
        self._calculation_settings_summary.setText(
            "Settings: "
            f"cells {int(self._calc_cells_a.value())} x {int(self._calc_cells_b.value())} x "
            f"{int(self._calc_cells_c.value())}; "
            f"{int(self._lookup_angles.value())} lookup angles per q; "
            f"{int(self._phase_samples.value())} distribution samples; "
            f"{self._phase_sampling_label()} sampling; "
            f"{int(self._histogram_bins.value())} histogram bins"
            + (
                f"; commensurate approximants within {self._approximant_tolerance:g} r.l.u., "
                f"period up to {self._approximant_max_period}"
                if self._use_approximants
                else ""
            )
            + (
                f"; contact shell of {self._contact_nnn} nearest magnetic "
                f"{'ion' if self._contact_nnn == 1 else 'ions'} within {self._contact_rcont:g} Å"
                if self._has_contact_coupling()
                else ""
            )
            + "."
        )

    def _contact_tie_checker(self) -> Callable[[int, float], tuple[str, ...]] | None:
        """Return the contact-shell tie check of the selected site, or ``None`` when there is nothing to check.

        There is nothing to check without a host, a contact coupling or a usable muon site.
        """

        if self._host_structure is None or not self._has_contact_coupling():
            return None
        try:
            site = self.selected_muon_site()
        except ValueError:
            return None
        if site is None or self._muon_site_unavailable_reason() is not None:
            return None
        try:
            return contact_shell_tie_checker(
                self._host_structure,
                self._model,
                [site.fractional_position],
                find_equivalent_sites=self._include_equivalent_sites.isChecked(),
            )
        except ValueError:
            return None

    def _has_contact_coupling(self) -> bool:
        try:
            return self._contact_coupling_angstrom_minus3() is not None
        except ValueError:
            return False

    def _field_options(self) -> MagneticFieldOptions:
        return MagneticFieldOptions(
            demag_shape=self._demag_shape(),
            demag_sample=self._active_demag_sample(),
            measured_magnetization=self._measured_magnetization(),
            applied_field=self._applied_field(),
            supercell=(
                int(self._calc_cells_a.value()),
                int(self._calc_cells_b.value()),
                int(self._calc_cells_c.value()),
            ),
            find_equivalent_sites=self._include_equivalent_sites.isChecked(),
            contact_coupling=self._contact_coupling_angstrom_minus3(),
            nnn=int(self._contact_nnn),
            rcont=float(self._contact_rcont),
            nangles=int(self._lookup_angles.value()),
            phase_sample_count=int(self._phase_samples.value()),
            phase_sampling=str(self._phase_sampling_method),
            histogram_bins=int(self._histogram_bins.value()),
            approximant_tolerance=float(self._approximant_tolerance),
            approximant_max_period=int(self._approximant_max_period),
        )

    @staticmethod
    def _state_approximant_tolerance(value: object) -> float:
        default = MagneticFieldOptions().approximant_tolerance
        try:
            tolerance = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default
        return tolerance if np.isfinite(tolerance) and 0.0 < tolerance <= 0.05 else default

    @staticmethod
    def _state_approximant_max_period(value: object) -> int:
        default = MagneticFieldOptions().approximant_max_period
        try:
            period = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default
        return period if 1 <= period <= MAX_APPROXIMANT_PERIOD else default

    @staticmethod
    def _state_contact_nnn(value: object) -> int:
        default = MagneticFieldOptions().nnn
        try:
            nnn = int(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default
        return nnn if 1 <= nnn <= MAX_CONTACT_NEIGHBOURS else default

    @staticmethod
    def _state_contact_rcont(value: object) -> float:
        default = MagneticFieldOptions().rcont
        try:
            rcont = float(value)  # type: ignore[arg-type]
        except (TypeError, ValueError):
            return default
        in_range = MIN_CONTACT_RADIUS_ANG <= rcont <= MAX_CONTACT_RADIUS_ANG
        # The settings dialog shows the radius to CONTACT_RADIUS_DECIMALS places.
        return round(rcont, CONTACT_RADIUS_DECIMALS) if np.isfinite(rcont) and in_range else default

    def _phase_sampling_label(self) -> str:
        method = self._normalize_phase_sampling_method(self._phase_sampling_method)
        for key, label in PHASE_SAMPLING_METHOD_OPTIONS:
            if key == method:
                return label
        return "Auto"

    @staticmethod
    def _normalize_phase_sampling_method(value: object) -> str:
        text = str(value).strip().lower()
        if text in {"monte_carlo", "random"}:
            text = "montecarlo"
        allowed = {key for key, _label in PHASE_SAMPLING_METHOD_OPTIONS}
        return text if text in allowed else "auto"

    def _demag_shape(self) -> str:
        if self._advanced_demag_radio.isChecked():
            return str(self._demag_sample.shape)
        if self._sphere_radio.isChecked():
            return "sphere"
        if self._plate_radio.isChecked():
            return "infinite_plate"
        return "none"

    def _active_demag_sample(self) -> DemagnetizingSample | None:
        """Return the advanced sample, its orientation resolved against the host lattice."""

        if not self._advanced_demag_radio.isChecked():
            return None
        if self._host_structure is None:
            return self._demag_sample
        return self._demag_sample.resolve_orientation(self._host_structure.lattice_vectors)

    def _open_advanced_demag_dialog(self) -> None:
        seed = self._demag_sample
        if seed.shape == "none":
            seed = DemagnetizingSample(shape="ellipsoid")
        dialog = DemagnetizingFieldAdvancedDialog(
            seed,
            self,
            lattice_vectors=(
                None if self._host_structure is None else self._host_structure.lattice_vectors
            ),
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self._demag_sample = dialog.selected_sample()
        self._advanced_demag_radio.setChecked(True)
        self._refresh_demag_summary()
        self._on_form_setting_changed()

    def _refresh_demag_summary(self) -> None:
        sample = self._demag_sample
        shape = str(sample.shape).strip().lower()
        if shape not in ADVANCED_DEMAG_SHAPES:
            self._advanced_demag_summary.setText("No advanced shape selected.")
            return
        text = demag_sample_summary(sample)
        if sample.normal_hkl is not None:
            normal = " ".join(f"{value:g}" for value in sample.normal_hkl)
            in_plane = " ".join(f"{value:g}" for value in sample.in_plane_uvw or ())
            text += f"; normal ({normal}), in-plane [{in_plane}]"
        elif sample.rotation is not None:
            text += "; rotated sample axes"
        self._advanced_demag_summary.setText(text)

    def _measured_magnetization(self) -> tuple[float, float, float] | None:
        if not self._measured_magnetization_check.isChecked():
            return None
        x, y, z = (
            self._parse_vector_component(edit.text(), "measured magnetisation")
            for edit in self._measured_magnetization_inputs
        )
        return (x, y, z)

    def _applied_field(self) -> tuple[float, float, float]:
        x, y, z = (
            self._parse_vector_component(edit.text(), "applied field")
            for edit in self._applied_field_inputs
        )
        return (x, y, z)

    @staticmethod
    def _parse_vector_component(text: str, quantity: str) -> float:
        stripped = text.strip()
        if not stripped:
            return 0.0
        try:
            value = float(stripped)
        except ValueError as exc:
            raise ValueError(f"{quantity} components must be decimal numbers") from exc
        if not np.isfinite(value):
            raise ValueError(f"{quantity} components must be finite")
        return value

    def _vector_component_or_zero(self, edit: QLineEdit) -> float:
        try:
            return self._parse_vector_component(edit.text(), "vector")
        except ValueError:
            return 0.0

    @staticmethod
    def _set_vector_inputs(edits: Sequence[QLineEdit], values: object) -> None:
        """Show the three components of a saved vector, or zeros when it is malformed."""

        components = (0.0, 0.0, 0.0)
        if (
            isinstance(values, Sequence)
            and not isinstance(values, (str, bytes))
            and len(values) == 3
        ):
            try:
                parsed = tuple(float(value) for value in values)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                parsed = components
            if all(np.isfinite(parsed)):
                components = parsed
        for edit, value in zip(edits, components):
            edit.setText(f"{value:g}")

    def _set_measured_magnetization_from_state(self, enabled: object, values: object) -> None:
        self._set_vector_inputs(self._measured_magnetization_inputs, values)
        self._measured_magnetization_check.setChecked(bool(enabled))

    def _refresh_measured_magnetization_unit(self) -> None:
        formula = (
            "" if self._host_structure is None else reduced_formula(self._host_structure.species)[0]
        )
        self._measured_magnetization_unit.setText(
            f"μB per {formula} (Cartesian x, y, z)"
            if formula
            else "μB per formula unit (Cartesian x, y, z)"
        )

    def _on_measured_magnetization_toggled(self, _checked: bool) -> None:
        self._sync_enabled_state()
        self._on_form_setting_changed()

    def _current_hyperfine_unit(self) -> str:
        value = self._hyperfine_units.currentData()
        return str(value or "angstrom^-3")

    def _set_hyperfine_unit(self, unit: str) -> None:
        index = self._hyperfine_units.findData(unit)
        if index < 0:
            index = self._hyperfine_units.findData("angstrom^-3")
        if index >= 0:
            self._hyperfine_units.setCurrentIndex(index)

    def _contact_coupling_angstrom_minus3(self) -> float | None:
        text = self._hyperfine_coupling.text().strip()
        if not text:
            return None
        try:
            value = float(text)
        except ValueError as exc:
            raise ValueError("hyperfine coupling must be a decimal number") from exc
        if abs(value) <= 1e-15:
            return None
        return contact_coupling_to_angstrom_minus3(value, self._current_hyperfine_unit())

    def _on_demag_shape_toggled(self, checked: bool) -> None:
        self._advanced_demag_summary.setVisible(self._advanced_demag_radio.isChecked())
        if checked:
            self._on_form_setting_changed()

    def _on_form_setting_changed(self, *_args: object) -> None:
        self._refresh_calculation_settings_summary()
        self._clear_results(clear_state=True)
        self._emit_form_state_changed()

    def _emit_form_state_changed(self) -> None:
        if self._restoring_form_state:
            return
        self.form_state_changed.emit(self.collect_state())

    def _emit_pin_requested(self) -> None:
        self.pin_requested.emit(self.collect_workspace_state())

    def _on_title_edit_finished(self) -> None:
        title = self._title_edit.text().strip() or "Magnetic Fields"
        if self._title_edit.text() != title:
            self._title_edit.setText(title)
        self._title.setText(title)
        if self._restoring_title:
            return
        self.workspace_title_changed.emit(title)

    @staticmethod
    def _set_spinbox_from_state(spinbox: QSpinBox, value: object) -> None:
        if value is None:
            return
        try:
            spinbox.setValue(int(value))
        except (TypeError, ValueError):
            return

    @staticmethod
    def _state_float_triplet(value: object) -> tuple[float, float, float] | None:
        if not isinstance(value, list | tuple) or len(value) != 3:
            return None
        try:
            return (float(value[0]), float(value[1]), float(value[2]))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _state_int_triplet(value: object) -> tuple[int, int, int] | None:
        if not isinstance(value, list | tuple) or len(value) != 3:
            return None
        try:
            return (int(value[0]), int(value[1]), int(value[2]))
        except (TypeError, ValueError):
            return None

    def _muon_site_to_state(self, site: MagneticMuonSiteOption | None) -> dict[str, Any] | None:
        if site is None:
            return None
        payload = to_jsonable(
            {
                "source": site.source,
                "label": site.label,
                "fractional_position": site.fractional_position,
                "cartesian_position": site.cartesian_position,
                "cluster_label": site.cluster_label,
                "source_label": site.source_label,
                "distortion_structure_id": site.distortion_structure_id,
                "include_distortion": site.include_distortion,
                "properties": site.properties,
            }
        )
        return payload if isinstance(payload, dict) else None

    def _muon_site_from_state(self, value: object) -> MagneticMuonSiteOption | None:
        if not isinstance(value, Mapping):
            return None
        fractional_position = self._state_float_triplet(value.get("fractional_position"))
        if fractional_position is None:
            return None
        cartesian_position = self._state_float_triplet(value.get("cartesian_position"))
        if cartesian_position is None and self._host_structure is not None:
            cartesian_position = _fractional_to_cartesian(fractional_position, self._host_structure)

        distortion_structure_id = None
        raw_distortion_id = value.get("distortion_structure_id")
        if raw_distortion_id not in {None, ""}:
            try:
                distortion_structure_id = UUID(str(raw_distortion_id))
            except ValueError:
                distortion_structure_id = None

        properties = value.get("properties")
        return MagneticMuonSiteOption(
            source=str(value.get("source", "custom")).strip() or "custom",
            label=str(value.get("label", "Custom site")).strip() or "Custom site",
            fractional_position=fractional_position,
            cartesian_position=cartesian_position,
            cluster_label=str(value["cluster_label"]) if value.get("cluster_label") else None,
            source_label=str(value["source_label"]) if value.get("source_label") else None,
            distortion_structure_id=distortion_structure_id,
            properties=dict(properties) if isinstance(properties, Mapping) else None,
            include_distortion=bool(value.get("include_distortion", False))
            and distortion_structure_id is not None,
        )

    def _clear_results(self, *, clear_state: bool = False) -> None:
        self._distribution_results = []
        if clear_state:
            self._field_results_state = None
        self._result_table.setRowCount(0)
        self._calculation_status.setText("Results: -")
        self._sync_distribution_plot_button()
        self._refresh_field_breakdown()

    def _refresh_field_breakdown(self) -> None:
        """List the field terms that sum to the selected result row.

        A commensurate row lists each term's vector; a distribution row lists
        each term's mean vector and RMS magnitude over the phase samples.
        """

        rows: list[tuple[str, ...]] = []
        hint = ""
        contact_warnings: list[str] = []
        headers = self._BREAKDOWN_HEADERS
        selected_rows = sorted({item.row() for item in self._result_table.selectedItems()})
        state = self._field_results_state
        results = state.get("results") if isinstance(state, Mapping) else None
        if (
            not selected_rows
            or not isinstance(results, Sequence)
            or isinstance(results, (str, bytes))
        ):
            hint = (
                "Select a result row to see its field terms."
                if self._result_table.rowCount()
                else ""
            )
        elif state.get("kind") != "commensurate":
            headers = self._DISTRIBUTION_BREAKDOWN_HEADERS
            if selected_rows[0] >= len(results) or not isinstance(
                results[selected_rows[0]], Mapping
            ):
                hint = "Select a result row to see its field terms."
            else:
                payload = results[selected_rows[0]]
                rows = self._distribution_breakdown_rows(payload)
                contact_warnings = self._contact_shell_warning_lines(payload)
                lines = []
                checks = self._convergence_lines(payload.get("convergence"))
                if checks:
                    lines = ["Convergence checks:", *checks]
                if not rows:
                    lines.append(
                        "This saved distribution has no term breakdown; recalculate to see it."
                    )
                elif len(selected_rows) > 1:
                    lines.append(f"Showing {payload.get('site_label', 'the first selected site')}.")
                hint = "\n".join(lines)
        elif selected_rows[0] >= len(results) or not isinstance(results[selected_rows[0]], Mapping):
            hint = "Select a result row to see its field terms."
        else:
            payload = results[selected_rows[0]]
            contact_warnings = self._contact_shell_warning_lines(payload)
            merge = self._merges_dipolar_and_lorentz(payload)
            for label, key in self._BREAKDOWN_TERMS:
                if merge and key == "lorentz_T":
                    continue
                vector = np.asarray(payload.get(key, ()), dtype=float)
                if merge and key == "dipolar_T":
                    label = self._MERGED_DIPOLAR_LORENTZ_LABEL
                    lorentz = np.asarray(payload.get("lorentz_T", ()), dtype=float)
                    # A malformed saved payload leaves the merged row out, like any other term.
                    vector = (
                        vector + lorentz if lorentz.shape == vector.shape == (3,) else np.empty(0)
                    )
                if vector.shape == (3,):
                    rows.append(
                        (
                            label,
                            *format_vector_components(vector),
                            format_table_number(np.linalg.norm(vector)),
                        )
                    )
            if not rows:
                hint = "This saved result has no term breakdown; recalculate to see it."
            else:
                total = np.asarray(payload.get("total_T", (0.0, 0.0, 0.0)), dtype=float)
                rows.append(
                    (
                        "Total",
                        *format_vector_components(total),
                        format_table_number(np.linalg.norm(total)),
                    )
                )
                if len(selected_rows) > 1:
                    hint = f"Showing {payload.get('site_label', 'the first selected site')}."
        self._breakdown_table.setHorizontalHeaderLabels(list(headers))
        self._set_summary_rows(self._breakdown_table, rows)
        self._breakdown_label.setVisible(bool(rows))
        self._breakdown_table.setVisible(bool(rows))
        self._breakdown_hint.setText(hint)
        self._breakdown_hint.setVisible(bool(hint))
        self._breakdown_contact_warning.setText("\n".join(contact_warnings))
        self._breakdown_contact_warning.setVisible(bool(contact_warnings))

    @classmethod
    def _distribution_breakdown_rows(cls, payload: Mapping[str, object]) -> list[tuple[str, ...]]:
        """Return the mean and RMS rows of a distribution payload, or none for an old record."""

        terms = payload.get("terms")
        if not isinstance(terms, Mapping):
            return []
        merged = payload.get("dipolar_lorentz")
        merge = cls._merges_dipolar_and_lorentz(payload) and isinstance(merged, Mapping)
        entries = []
        for label, key in cls._DISTRIBUTION_BREAKDOWN_TERMS:
            if merge and key == "lorentz":
                continue
            if merge and key == "dipolar":
                entries.append((cls._MERGED_DIPOLAR_LORENTZ_LABEL, merged))
            else:
                entries.append((label, terms.get(key)))
        entries.append(
            ("Total", {"mean_T": payload.get("mean_T"), "rms_T": payload.get("total_rms_T")})
        )
        rows: list[tuple[str, ...]] = []
        for label, term in entries:
            if not isinstance(term, Mapping):
                return []
            try:
                mean = np.asarray(term.get("mean_T"), dtype=float)
                rms = float(term.get("rms_T"))
            except (TypeError, ValueError):
                return []
            if mean.shape != (3,):
                return []
            rows.append(
                (label, *format_vector_components(mean, scale=rms), format_table_number(rms))
            )
        return rows

    @staticmethod
    def _merges_dipolar_and_lorentz(payload: Mapping[str, object]) -> bool:
        """Return whether a result was calculated for a structure with a component at q != 0."""

        parameters = payload.get("calculation_parameters")
        return isinstance(parameters, Mapping) and bool(parameters.get("modulated", False))

    @staticmethod
    def _contact_shell_warning_lines(payload: Mapping[str, object]) -> list[str]:
        """Return the saved contact-shell tie warnings of a result, each as one line."""

        parameters = payload.get("calculation_parameters")
        warnings = (
            parameters.get("contact_shell_warnings") if isinstance(parameters, Mapping) else None
        )
        if not isinstance(warnings, Sequence) or isinstance(warnings, (str, bytes)):
            return []
        return [f"Contact shell: {warning}" for warning in warnings if isinstance(warning, str)]

    @staticmethod
    def _convergence_lines(raw_checks: object) -> list[str]:
        """Return one line per saved convergence check of a distribution result."""

        if not isinstance(raw_checks, Sequence) or isinstance(raw_checks, (str, bytes)):
            return []
        lines: list[str] = []
        for check in raw_checks:
            if not isinstance(check, Mapping):
                continue
            verdict = "OK" if check.get("converged") else "Needs attention"
            lines.append(
                f"{check.get('label', check.get('setting', 'Check'))}: {verdict}. {check.get('message', '')}".strip()
            )
        return lines

    def _populate_commensurate_results(self, results: object, *, site_source: str) -> None:
        self._distribution_results = []
        self._field_results_state = self._field_results_to_state(
            "commensurate", results, site_source=site_source
        )
        self._result_table.setRowCount(0)
        for result in results:
            total = np.asarray(result.total, dtype=float)
            self._append_result_row(
                result.site_label,
                total,
                float(result.magnitude),
                self._result_type_label(
                    "field" if result.distortion is None else _DISTORTION_CORRECTED_FIELD
                ),
            )
        if self._result_table.rowCount() > 0:
            self._result_table.selectRow(0)

    def _populate_distribution_results(self, results: object, *, site_source: str) -> None:
        self._distribution_results = list(results)
        self._field_results_state = self._field_results_to_state(
            "distribution", self._distribution_results, site_source=site_source
        )
        self._result_table.setRowCount(0)
        for index, result in enumerate(self._distribution_results):
            fields = np.asarray(result.fields_T, dtype=float)
            mean = np.mean(fields, axis=0)
            magnitudes = np.linalg.norm(fields, axis=1)
            self._append_result_row(
                result.site_label,
                mean,
                float(np.mean(magnitudes)),
                self._result_type_label(
                    _distribution_label(getattr(result, "calculation_parameters", None))
                ),
                distribution_index=index,
            )
        if self._result_table.rowCount() > 0:
            self._result_table.selectRow(0)
        self._sync_distribution_plot_button()

    def _restore_field_results(self) -> None:
        state = self._field_results_state
        self._distribution_results = []
        self._result_table.setRowCount(0)
        self._calculation_status.setText("Results: -")
        if not isinstance(state, Mapping):
            self._sync_distribution_plot_button()
            return
        kind = str(state.get("kind", "")).strip()
        results = state.get("results")
        if not isinstance(results, Sequence) or isinstance(results, (str, bytes)):
            self._sync_distribution_plot_button()
            return
        result_payloads = [dict(result) for result in results if isinstance(result, Mapping)]
        non_finite_label = _non_finite_field_label(result_payloads)
        if non_finite_label is not None:
            self._calculation_status.setText(
                f"Results: the saved field at {non_finite_label} is not a finite number "
                "(NaN or infinity), so it is not shown."
            )
            self._sync_distribution_plot_button()
            return
        if kind == "commensurate":
            for result in result_payloads:
                total = np.asarray(result.get("total_T", (0.0, 0.0, 0.0)), dtype=float)
                if total.shape != (3,):
                    total = np.zeros(3, dtype=float)
                magnitude = float(result.get("magnitude_T", np.linalg.norm(total)))
                result_type = self._result_type_label(
                    _DISTORTION_CORRECTED_FIELD if "distortion_T" in result else "field"
                )
                self._append_result_row(
                    str(result.get("site_label", "site")), total, magnitude, result_type
                )
            if self._result_table.rowCount() > 0:
                self._result_table.selectRow(0)
            self._calculation_status.setText(
                f"Results: restored {len(result_payloads)} commensurate field value(s)."
                + self._estimate_status_suffix()
            )
        elif kind == "distribution":
            self._distribution_results = [
                self._distribution_result_from_payload(result) for result in result_payloads
            ]
            for index, result in enumerate(result_payloads):
                fields = np.asarray(result.get("fields_T", []), dtype=float)
                if fields.ndim == 2 and fields.shape[1] == 3 and fields.size:
                    mean = np.mean(fields, axis=0)
                    magnitude = float(np.mean(np.linalg.norm(fields, axis=1)))
                else:
                    mean = np.zeros(3, dtype=float)
                    magnitude = 0.0
                self._append_result_row(
                    str(result.get("site_label", f"site {index + 1}")),
                    mean,
                    magnitude,
                    self._result_type_label(
                        _distribution_label(result.get("calculation_parameters"))
                    ),
                    distribution_index=index,
                )
            if self._result_table.rowCount() > 0:
                self._result_table.selectRow(0)
            self._calculation_status.setText(
                f"Results: restored {len(result_payloads)} incommensurate distribution(s)."
                + self._estimate_status_suffix()
            )
        self._sync_distribution_plot_button()

    @staticmethod
    def _distribution_result_from_payload(payload: dict[str, object]) -> object:
        """Return the saved distribution as a result object, or the payload when it is malformed.

        The distribution plot reads result attributes, so a restored row must
        hand it the same kind of object a fresh calculation does.
        """

        try:
            return MagneticFieldDistributionResult.from_json(payload)
        except (TypeError, ValueError, KeyError):
            return payload

    def _results_are_estimates(self) -> bool:
        """Return whether the shown field results were computed at an unrelaxed UEP site."""

        state = self._field_results_state
        return isinstance(state, Mapping) and is_unrelaxed_site_source(state.get("site_source"))

    def _result_type_label(self, result_type: str) -> str:
        return (
            f"{result_type}, unrelaxed estimate" if self._results_are_estimates() else result_type
        )

    def _estimate_status_suffix(self) -> str:
        return _UNRELAXED_ESTIMATE_SUFFIX if self._results_are_estimates() else ""

    def _field_results_to_state(
        self, kind: str, results: object, *, site_source: str
    ) -> dict[str, object]:
        payloads: list[dict[str, object]] = []
        for result in results:
            if hasattr(result, "to_json"):
                payload = result.to_json()
            elif isinstance(result, Mapping):
                payload = dict(result)
            else:
                payload = self._field_result_object_to_state(result)
            jsonable = to_jsonable(payload)
            if isinstance(jsonable, Mapping):
                payloads.append(dict(jsonable))
        return {"kind": kind, "results": payloads, "site_source": site_source}

    @staticmethod
    def _field_result_object_to_state(result: object) -> dict[str, object]:
        if hasattr(result, "fields_T"):
            fields = np.asarray(getattr(result, "fields_T", []), dtype=float)
            return {
                "site_label": str(getattr(result, "site_label", "site")),
                "muon_position": list(getattr(result, "muon_position", (0.0, 0.0, 0.0))),
                "fields_T": fields.tolist(),
                "histogram_centres_T": np.asarray(
                    getattr(result, "histogram_centres_T", []), dtype=float
                ).tolist(),
                "histogram_counts": np.asarray(
                    getattr(result, "histogram_counts", []), dtype=float
                ).tolist(),
                "calculation_parameters": dict(getattr(result, "calculation_parameters", {})),
            }
        total = np.asarray(getattr(result, "total", (0.0, 0.0, 0.0)), dtype=float)
        return {
            "site_label": str(getattr(result, "site_label", "site")),
            "muon_position": list(getattr(result, "muon_position", (0.0, 0.0, 0.0))),
            "total_T": total.tolist(),
            "magnitude_T": float(getattr(result, "magnitude", np.linalg.norm(total))),
        }

    def _append_result_row(
        self,
        label: str,
        field: np.ndarray,
        magnitude: float,
        result_type: str,
        *,
        distribution_index: int | None = None,
    ) -> None:
        row = self._result_table.rowCount()
        self._result_table.insertRow(row)
        values = [
            label,
            *format_vector_components(field[:3], scale=magnitude),
            format_table_number(magnitude),
            result_type,
        ]
        for column, value in enumerate(values):
            item = QTableWidgetItem(str(value))
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            if distribution_index is not None:
                item.setData(_DISTRIBUTION_RESULT_ROLE, int(distribution_index))
            self._result_table.setItem(row, column, item)
        self._result_table.resizeRowsToContents()

    def _selected_distribution_results(self) -> tuple[object, ...]:
        selected = self._result_table.selectedItems()
        if not selected:
            return ()
        results: list[object] = []
        for row in sorted({item.row() for item in selected}):
            item = self._result_table.item(row, 0)
            if item is None:
                continue
            raw_index = item.data(_DISTRIBUTION_RESULT_ROLE)
            if raw_index is None:
                continue
            try:
                index = int(raw_index)
            except (TypeError, ValueError):
                continue
            if 0 <= index < len(self._distribution_results):
                results.append(self._distribution_results[index])
        return tuple(results)

    def _sync_distribution_plot_button(self) -> None:
        self._open_distribution_button.setEnabled(bool(self._selected_distribution_results()))
        state = self._field_results_state
        results = state.get("results") if isinstance(state, Mapping) else None
        self._polarisation_button.setEnabled(
            isinstance(results, Sequence)
            and not isinstance(results, (str, bytes))
            and bool(results)
            and _non_finite_field_label(
                [result for result in results if isinstance(result, Mapping)]
            )
            is None
        )

    def _open_distribution_plot(self) -> None:
        results = self._selected_distribution_results()
        if not results:
            QMessageBox.information(
                self, "Field Distribution", "Select an incommensurate distribution result first."
            )
            return
        dialog = MagneticFieldDistributionDialog(
            results, self, estimate=self._results_are_estimates()
        )
        dialog.exec()


def _active_host_structure(document: ProjectDocument | None) -> Structure | None:
    if document is None or document.project.host_structure_id is None:
        return None
    return document.structures.get(document.project.host_structure_id)


def _uep_cluster_muon_site_options(
    document: ProjectDocument,
    host_structure: Structure,
) -> list[MagneticMuonSiteOption]:
    options: list[MagneticMuonSiteOption] = []
    seen: set[str] = set()

    for site in document.sites.values():
        if site.generation_method not in {"uep_cluster", "uep_cluster_member", "uep_batch_cluster"}:
            continue
        key = f"site:{site.id}"
        seen.add(key)
        options.append(
            MagneticMuonSiteOption(
                source=UEP_SITE_SOURCE,
                label=site.label or "UEP cluster site",
                source_label="Generated UEP site",
                cluster_label=_cluster_label_from_properties(site.properties),
                fractional_position=_tuple3(site.initial_fractional_position),
                cartesian_position=_tuple3(site.initial_cartesian_position),
                properties=dict(site.properties),
            )
        )

    states = document.ui_state.get(_UEP_ANALYSIS_STATES_KEY, {})
    if isinstance(states, dict):
        for raw_job_id, state in states.items():
            if not isinstance(state, dict):
                continue
            source_job_label = _job_label(document, raw_job_id, fallback="UEP source")
            batch_results = state.get("batch_results")
            summary = state.get("batch_summary_data")
            if not isinstance(batch_results, list) or not isinstance(summary, dict):
                continue
            clusters = summary.get("clusters")
            if not isinstance(clusters, (list, tuple)):
                continue
            for cluster in clusters:
                if not isinstance(cluster, dict):
                    continue
                representative_index = _optional_int(cluster.get("representative_index"))
                cluster_index = _optional_int(cluster.get("cluster_index"))
                if representative_index is None or not (
                    0 <= representative_index < len(batch_results)
                ):
                    continue
                result = batch_results[representative_index]
                if not isinstance(result, dict):
                    continue
                fractional = _triplet_or_none(
                    result.get("optimized_fractional_position")
                    or cluster.get("representative_fractional_position")
                )
                if fractional is None:
                    continue
                cartesian = _triplet_or_none(
                    result.get("optimized_position_ang")
                ) or _fractional_to_cartesian(fractional, host_structure)
                key = f"uep:{raw_job_id}:{cluster_index}:{representative_index}"
                if key in seen:
                    continue
                seen.add(key)
                options.append(
                    MagneticMuonSiteOption(
                        source=UEP_SITE_SOURCE,
                        label=f"UEP cluster {cluster_index if cluster_index is not None else len(options) + 1}",
                        source_label=str(source_job_label),
                        cluster_label="-" if cluster_index is None else f"Cluster {cluster_index}",
                        fractional_position=fractional,
                        cartesian_position=cartesian,
                        properties={
                            "source_job_id": str(raw_job_id),
                            "cluster": dict(cluster),
                            "representative": dict(result),
                        },
                    )
                )
    return sorted(
        options,
        key=lambda option: (option.source_label or "", option.cluster_label or "", option.label),
    )


def _relaxation_cluster_muon_site_options(
    document: ProjectDocument,
    host_structure: Structure,
    *,
    pinned_only: bool = False,
) -> list[MagneticMuonSiteOption]:
    """Return the relaxation-analysis cluster representatives as muon-site options.

    With ``pinned_only`` only the clusters pinned to the Muon Sites summary are
    returned, labelled by Wyckoff label and cluster, in charge-state then
    energy-rank order; a live cluster matches a pin by workspace and
    representative relaxation job. Otherwise every workspace's clusters, sorted
    by name.
    """

    workspaces = document.ui_state.get(_RELAXATION_ANALYSIS_STATE_KEY, [])
    if not isinstance(workspaces, list):
        return []
    pinned = pinned_sites_by_representative(document) if pinned_only else {}
    ranks = {key: order for order, key in enumerate(pinned)}
    pinned_options: list[tuple[int, MagneticMuonSiteOption]] = []
    options: list[MagneticMuonSiteOption] = []
    for workspace in workspaces:
        if not isinstance(workspace, dict):
            continue
        workspace_name = (
            str(workspace.get("name", "Relaxation analysis")).strip() or "Relaxation analysis"
        )
        clustering = workspace.get("clustering")
        result = clustering.get("result") if isinstance(clustering, dict) else None
        if not isinstance(result, dict):
            continue
        clusters = result.get("clusters")
        if not isinstance(clusters, (list, tuple)):
            continue
        for cluster in clusters:
            if not isinstance(cluster, dict):
                continue
            fractional = _triplet_or_none(cluster.get("representative_fractional_position"))
            if fractional is None:
                continue
            cluster_index = _optional_int(cluster.get("cluster_index"))
            job_id = _uuid_or_none(cluster.get("representative_job_id"))
            # A pin names its cluster's representative relaxation, which survives
            # re-clustering; the cluster index does not.
            pinned_key = (str(workspace.get("id", "")), job_id)
            if pinned_only and pinned_key not in pinned:
                continue
            structure_id = _distorted_structure_id_for_job(document, job_id)
            cartesian = _triplet_or_none(
                cluster.get("representative_cartesian_position_ang")
            ) or _fractional_to_cartesian(
                fractional,
                host_structure,
            )
            option = MagneticMuonSiteOption(
                source="relaxation_clustering",
                label=f"{workspace_name} cluster {cluster_index if cluster_index is not None else len(options) + 1}",
                source_label=workspace_name,
                cluster_label="-" if cluster_index is None else f"Cluster {cluster_index}",
                fractional_position=fractional,
                cartesian_position=cartesian,
                distortion_structure_id=structure_id,
                properties={"workspace": dict(workspace), "cluster": dict(cluster)},
            )
            if pinned_only:
                # The label stays the workspace/cluster name so a saved selection
                # still matches; the Cluster column carries the summary label.
                option = replace(option, cluster_label=pinned[pinned_key].label)
                pinned_options.append((ranks[pinned_key], option))
            options.append(option)
    if pinned_only:
        return [option for _rank, option in sorted(pinned_options, key=lambda item: item[0])]
    return sorted(
        options,
        key=lambda option: (option.source_label or "", option.cluster_label or "", option.label),
    )


def unrelaxed_results_note(parent: QWidget) -> QLabel:
    """Return a hidden note marking results computed at an unrelaxed UEP site as estimates."""

    note = QLabel(UNRELAXED_RESULTS_NOTE, parent)
    note.setWordWrap(True)
    style_hint_text(note)
    note.setVisible(False)
    return note


def _site_source_label(source: str) -> str:
    labels = {
        UEP_SITE_SOURCE: "UEP clustering (unrelaxed estimate)",
        "relaxation_clustering": "Structural relaxation clustering",
        "custom": "Custom",
    }
    return labels.get(source, source.replace("_", " ").title())


def _same_muon_site(left: MagneticMuonSiteOption, right: MagneticMuonSiteOption) -> bool:
    return (
        left.source == right.source
        and left.label == right.label
        and np.allclose(left.fractional_position, right.fractional_position, atol=1e-8)
    )


def _cluster_label_from_properties(properties: dict[str, object]) -> str | None:
    cluster = properties.get("cluster_index")
    if cluster is None:
        return None
    return f"Cluster {cluster}"


def _distorted_structure_id_for_job(document: ProjectDocument, job_id: UUID | None) -> UUID | None:
    if job_id is None:
        return None
    for run in document.runs.values():
        if run.job_id != job_id or run.result_id is None:
            continue
        result = document.results.get(run.result_id)
        if result is None:
            continue
        for structure_id in result.derived_structure_ids:
            if structure_id in document.structures:
                return structure_id
    return None


def _job_label(document: ProjectDocument, raw_job_id: object, *, fallback: str) -> str:
    job_id = _uuid_or_none(raw_job_id)
    if job_id is None:
        return fallback
    job = document.jobs.get(job_id)
    return job.label if job is not None and job.label else fallback


def _uuid_or_none(value: object) -> UUID | None:
    try:
        return UUID(str(value))
    except (TypeError, ValueError):
        return None


def _optional_int(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _triplet_or_none(value: object) -> tuple[float, float, float] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        return None
    try:
        return _tuple3(value)
    except (TypeError, ValueError):
        return None


def _tuple3(value: object) -> tuple[float, float, float]:
    values = list(value)  # type: ignore[arg-type]
    if len(values) != 3:
        raise ValueError("expected a three-vector")
    return (float(values[0]), float(values[1]), float(values[2]))


def _non_finite_field_label(results: Sequence[object]) -> str | None:
    """Return the site label of the first result whose field is not finite, or ``None``.

    A result is a calculated field object or a saved payload: its total field
    and magnitude, or its sampled distribution fields, must all be finite.
    """

    for index, result in enumerate(results, start=1):
        if isinstance(result, Mapping):
            label = result.get("site_label")
            arrays = [result.get(key) for key in ("total_T", "magnitude_T", "fields_T")]
        else:
            label = getattr(result, "site_label", None)
            arrays = [getattr(result, key, None) for key in ("total", "magnitude", "fields_T")]
        for values in arrays:
            if values is None:
                continue
            try:
                finite = bool(np.all(np.isfinite(np.asarray(values, dtype=float))))
            except (TypeError, ValueError):
                continue
            if not finite:
                return str(label or f"site {index}")
    return None


class _NonFiniteFieldError(MagneticFieldError):
    """Raised when a calculated field contains a NaN or an infinity."""


def _raise_for_non_finite_fields(results: Sequence[object]) -> None:
    """Refuse calculated fields that contain a NaN or an infinity, naming the site."""

    label = _non_finite_field_label(results)
    if label is not None:
        raise _NonFiniteFieldError(
            f"the field at {label} is not a finite number (NaN or infinity), so it is not shown"
        )


def _fractional_to_cartesian(
    fractional: tuple[float, float, float],
    structure: Structure,
) -> tuple[float, float, float]:
    cart = np.asarray(fractional, dtype=float) @ np.asarray(structure.lattice_vectors, dtype=float)
    return (float(cart[0]), float(cart[1]), float(cart[2]))


def _decimal_edit_value(edit: QLineEdit, label: str) -> float:
    text = edit.text().strip()
    if not text:
        raise ValueError(f"{label} is required")
    return float(text)


def _preview_supercell_structure(
    structure: Structure,
    repeats: tuple[int, int, int],
) -> tuple[Structure, np.ndarray]:
    repeat_array = np.asarray(repeats, dtype=float)
    base_frac = np.asarray(structure.fractional_positions, dtype=float)
    offsets = np.array(
        [
            (i, j, k)
            for i in range(repeats[0])
            for j in range(repeats[1])
            for k in range(repeats[2])
        ],
        dtype=float,
    )
    atom_count = len(structure.species)
    positions_frac = (
        (base_frac[np.newaxis, :, :] + offsets[:, np.newaxis, :]) / repeat_array
    ).reshape(-1, 3)
    atom_indices = np.tile(np.arange(atom_count, dtype=int), len(offsets))
    lattice = np.asarray(structure.lattice_vectors, dtype=float).copy()
    lattice[0] *= repeats[0]
    lattice[1] *= repeats[1]
    lattice[2] *= repeats[2]
    preview = Structure(
        label=f"{structure.label or 'structure'} preview {repeats[0]}x{repeats[1]}x{repeats[2]}",
        origin=structure.origin,
        lattice_vectors=lattice.tolist(),
        species=[structure.species[int(index)] for index in atom_indices],
        fractional_positions=positions_frac.tolist(),
        source_file=structure.source_file,
        parent_structure_id=structure.id,
        properties={
            **structure.properties,
            "pbc": structure.properties.get("pbc", [True, True, True]),
        },
    )
    return preview, atom_indices


def _nuclear_elements_text(elements: object) -> str:
    """Name the elements a nuclear sum included, or ``All`` when it was not filtered."""

    if isinstance(elements, (list, tuple)):
        return ", ".join(str(element) for element in elements)
    return "All"


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


def _selector_label(selector: AtomSelector, structure: Structure | None = None) -> str:
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
    except Exception:
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


def _component_q_label(model: MagneticStructureModel, component: MagneticComponent) -> str:
    """List each distinct q a component uses as its q-basis combination and its r.l.u. vector."""

    values: list[str] = []
    for term in component.terms:
        vector = _q_vector_text(model.q_coeff_to_rlu(term.q_coeff))
        expression = q_coeff_expression(term.q_coeff)
        value = f"({vector})" if expression == "0" else f"{expression} = ({vector})"
        if value not in values:
            values.append(value)
    return "; ".join(values) or "-"


def _create_q_basis_table(*, editable: bool) -> QTableWidget:
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


def _fill_q_basis_table(table: QTableWidget, model: MagneticStructureModel) -> None:
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
            (QTableWidgetItem(slot.label), QTableWidgetItem(_q_vector_text(slot.q_rlu)), users_item)
        ):
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
            table.setItem(slot.index, column, item)


def _q_vector_text(values: Sequence[float]) -> str:
    return ", ".join(
        format_table_number(value, absolute_tolerance=5e-13, relative_tolerance=0.0)
        for value in values
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


def _component_moment_label(component: MagneticComponent) -> str:
    if not component.terms:
        return "-"
    if is_texture_component(component):
        magnitudes = [
            float(np.linalg.norm(term.coefficient.to_array())) for term in component.terms
        ]
        mode = str(component.metadata.get("texture_mode", "texture")).replace("_", " ")
        return f"{mode.title()}\n{len(component.terms)} Fourier term(s)\nmax {_format_moment_magnitude(max(magnitudes))}"
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


def _format_float(value: object) -> str:
    try:
        return f"{float(value):.8g}"
    except (TypeError, ValueError):
        return ""


def _component_edit_q_vector(
    model: MagneticStructureModel, component: MagneticComponent
) -> tuple[float, float, float]:
    for term in component.terms:
        if any(abs(float(value)) > 1e-12 for value in term.q_coeff):
            return _tuple3_float(model.q_coeff_to_rlu(term.q_coeff))
    if component.terms:
        return _tuple3_float(model.q_coeff_to_rlu(component.terms[0].q_coeff))
    return (0.0, 0.0, 0.0)


def _component_edit_q_coeff(component: MagneticComponent) -> tuple[float, float, float] | None:
    """Return the q-basis coefficients of a component's first finite-q term, or ``None`` when it has none."""

    for term in component.terms:
        if any(abs(float(value)) > 1e-12 for value in term.q_coeff):
            return _tuple3_float(term.q_coeff)
    return None


def _tuple3_float(values: Sequence[float]) -> tuple[float, float, float]:
    arr = np.asarray(values, dtype=float)
    return (float(arr[0]), float(arr[1]), float(arr[2]))


def _fallback_direction(
    vector: Sequence[float], fallback: tuple[float, float, float]
) -> tuple[float, float, float]:
    arr = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(arr))
    if norm <= _DIRECTION_TOLERANCE:
        return fallback
    return tuple(float(value) for value in arr / norm)


def _orthogonal_unit(vector: Sequence[float]) -> np.ndarray:
    axis = np.asarray(vector, dtype=float)
    norm = float(np.linalg.norm(axis))
    if norm <= _DIRECTION_TOLERANCE:
        axis = np.asarray((0.0, 0.0, 1.0), dtype=float)
    else:
        axis = axis / norm
    trial = np.asarray((1.0, 0.0, 0.0), dtype=float)
    if abs(float(axis @ trial)) > 0.9:
        trial = np.asarray((0.0, 1.0, 0.0), dtype=float)
    projected = trial - float(trial @ axis) * axis
    projected_norm = float(np.linalg.norm(projected))
    if projected_norm <= _DIRECTION_TOLERANCE:
        return np.asarray((0.0, 0.0, 1.0), dtype=float)
    return projected / projected_norm


def _unit_q_coeff(index: int) -> tuple[float, float, float]:
    rows = [(1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)]
    return rows[int(index)]
