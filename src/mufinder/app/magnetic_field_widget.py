"""Magnetic Fields workflow page and its calculation settings dialog."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import replace
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.app.demagnetizing_dialogs import (
    ADVANCED_DEMAG_SHAPES,
    DemagnetizingFieldAdvancedDialog,
    demag_sample_summary,
)
from mufinder.app.icons import icon
from mufinder.app.magnetic_analysis_dialogs import DipolarTensorDialog, NuclearRelaxationDialog
from mufinder.app.magnetic_component_dialogs import (
    component_moment_label,
    component_q_label,
    selector_label,
)
from mufinder.app.magnetic_field_distribution_dialog import MagneticFieldDistributionDialog
from mufinder.app.magnetic_site_dialog import MuonSiteSelectionDialog
from mufinder.app.magnetic_structure_builder import MagneticStructureBuilderDialog
from mufinder.app.theme import (
    magnetic_fields_stylesheet,
    style_hint_text,
    style_section_subtitle,
    style_section_title,
    style_warning_text,
)
from mufinder.app.workflow_table_utils import (
    create_workflow_table,
    format_table_number,
    format_vector_components,
)
from mufinder.domain.models import Structure
from mufinder.domain.text import plural
from mufinder.persistence.codec import float_triplet_or_none, to_jsonable, uuid_or_none
from mufinder.science.ase_bridge import host_elements, reduced_formula
from mufinder.science.demagnetization import DemagnetizingSample
from mufinder.science.magnetic_fields import (
    CONTACT_COUPLING_UNIT_OPTIONS,
    MagneticFieldDependencyError,
    MagneticFieldDistributionResult,
    MagneticFieldError,
    MagneticFieldOptions,
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
    MagneticStructureModel,
    enabled_magnetic_components,
    magnetic_model_for_host,
)
from mufinder.science.nuclear_relaxation import (
    NuclearRelaxationError,
    NuclearRelaxationOptions,
    calculate_nuclear_relaxation,
    calculate_nuclear_relaxation_with_distortion,
)
from mufinder.services.magnetic_field_results import (
    UNRELAXED_ESTIMATE_SUFFIX,
    UNRELAXED_SITE_NOTE,
    is_unrelaxed_site_source,
)
from mufinder.services.magnetic_field_settings import (
    CONTACT_RADIUS_DECIMALS,
    MAX_APPROXIMANT_PERIOD,
    MAX_APPROXIMANT_TOLERANCE_RLU,
    MAX_CALCULATION_CELLS,
    MAX_CONTACT_NEIGHBOURS,
    MAX_CONTACT_RADIUS_ANG,
    MAX_HISTOGRAM_BINS,
    MAX_INCOMMENSURATE_LOOKUP_ANGLES,
    MAX_INCOMMENSURATE_PHASE_SAMPLES,
    MIN_CALCULATION_CELLS,
    MIN_CONTACT_RADIUS_ANG,
    MIN_HISTOGRAM_BINS,
    MIN_INCOMMENSURATE_LOOKUP_ANGLES,
    MIN_INCOMMENSURATE_PHASE_SAMPLES,
    PHASE_SAMPLING_METHOD_OPTIONS,
    MagneticFieldCalculationSettings,
    contact_rcont_or_default,
    normalize_phase_sampling_method,
)
from mufinder.services.magnetic_field_sites import (
    CUSTOM_SITE_SOURCE,
    MagneticMuonSiteOption,
    default_muon_site,
    distortion_structure,
    site_cartesian_position,
    site_source_label,
)
from mufinder.services.project_document import active_host_structure
from mufinder.services.project_service import ProjectDocument

_DISTORTION_CORRECTED_FIELD = "field (distortion-corrected)"


_DISTORTION_CORRECTED_DISTRIBUTION = "distribution mean (distortion-corrected)"


def _distribution_label(calculation_parameters: object) -> str:
    """The Result column text of a distribution row, marking a distortion-corrected one."""

    corrected = isinstance(calculation_parameters, Mapping) and bool(
        calculation_parameters.get("distortion_corrected", False)
    )
    return _DISTORTION_CORRECTED_DISTRIBUTION if corrected else "distribution mean"


_DISTRIBUTION_RESULT_ROLE = int(Qt.ItemDataRole.UserRole) + 42


CONVERGENCE_GUIDANCE_TEXT = (
    "Each distribution is checked after it is calculated: halving the lookup angles or the samples "
    "must shift the field distribution by at most 1%, and every occupied histogram bin needs at "
    "least 20 samples. Select a distribution row to read the checks, then double any flagged "
    "setting and recalculate."
)


class MagneticFieldCalculationSettingsDialog(QDialog):
    """Dialog for numerical magnetic-field calculation settings."""

    def __init__(
        self,
        settings: MagneticFieldCalculationSettings = MagneticFieldCalculationSettings(),
        *,
        contact_tie_check: Callable[[int, float], Sequence[str]] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        """``contact_tie_check`` returns the contact-shell tie warnings for a
        Contact neighbours and radius pair; without it no warning is shown."""

        super().__init__(parent)
        self.setWindowTitle("Calculation Settings")
        self._contact_tie_check = contact_tie_check
        self.resize(460, 360)

        self._settings = settings
        self._calc_cells_a = self._calculation_cell_input(settings.supercell[0])
        self._calc_cells_b = self._calculation_cell_input(settings.supercell[1])
        self._calc_cells_c = self._calculation_cell_input(settings.supercell[2])
        self._lookup_angles = self._lookup_angle_input(settings.nangles)
        self._phase_samples = self._incommensurate_sampling_input(settings.phase_sample_count)
        self._phase_sampling = self._phase_sampling_input(settings.phase_sampling)
        self._histogram_bins = self._histogram_bin_input(settings.histogram_bins)

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
        self._use_approximants.setChecked(bool(settings.use_approximants))
        self._use_approximants.setToolTip(
            "Replace each incommensurate wavevector by its nearest short-period fraction and list the fields "
            "at the translation-equivalent sites of that magnetic supercell instead of sampling a distribution."
        )
        self._approximant_tolerance = QDoubleSpinBox(approximant_group)
        self._approximant_tolerance.setDecimals(6)
        self._approximant_tolerance.setRange(1e-6, MAX_APPROXIMANT_TOLERANCE_RLU)
        self._approximant_tolerance.setSingleStep(1e-4)
        self._approximant_tolerance.setValue(float(settings.approximant_tolerance))
        self._approximant_tolerance.setSuffix(" r.l.u.")
        self._approximant_tolerance.setToolTip(
            "Largest change of any wavevector component when it is replaced by a fraction p/n."
        )
        self._approximant_max_period = QSpinBox(approximant_group)
        self._approximant_max_period.setRange(1, MAX_APPROXIMANT_PERIOD)
        self._approximant_max_period.setValue(int(settings.approximant_max_period))
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
        self._contact_nnn.setValue(int(settings.contact_nnn))
        self._contact_nnn.setToolTip(
            "Number of nearest magnetic ions whose 1/r³-weighted mean moment sets the contact "
            "field. Used only when a hyperfine coupling is set."
        )
        self._contact_rcont = QDoubleSpinBox(contact_group)
        self._contact_rcont.setDecimals(CONTACT_RADIUS_DECIMALS)
        self._contact_rcont.setRange(MIN_CONTACT_RADIUS_ANG, MAX_CONTACT_RADIUS_ANG)
        self._contact_rcont.setSingleStep(0.5)
        self._contact_rcont.setValue(float(settings.contact_rcont))
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

    def settings(self) -> MagneticFieldCalculationSettings:
        """The settings as the dialog shows them."""

        return replace(
            self._settings,
            supercell=(
                int(self._calc_cells_a.value()),
                int(self._calc_cells_b.value()),
                int(self._calc_cells_c.value()),
            ),
            nangles=int(self._lookup_angles.value()),
            phase_sample_count=int(self._phase_samples.value()),
            phase_sampling=normalize_phase_sampling_method(
                self._phase_sampling.currentData() or "auto"
            ),
            histogram_bins=int(self._histogram_bins.value()),
            use_approximants=self._use_approximants.isChecked(),
            approximant_tolerance=float(self._approximant_tolerance.value()),
            approximant_max_period=int(self._approximant_max_period.value()),
            contact_nnn=int(self._contact_nnn.value()),
            contact_rcont=contact_rcont_or_default(self._contact_rcont.value()),
        )

    @staticmethod
    def _calculation_cell_input(value: int) -> QSpinBox:
        spinbox = QSpinBox()
        spinbox.setRange(MIN_CALCULATION_CELLS, MAX_CALCULATION_CELLS)
        spinbox.setValue(int(value))
        spinbox.setToolTip(
            "Unit-cell repeats used for the dipolar lattice summation. Larger values are slower but reduce cutoff effects."
        )
        spinbox.setMinimumWidth(104)
        return spinbox

    @staticmethod
    def _incommensurate_sampling_input(value: int) -> QSpinBox:
        spinbox = QSpinBox()
        spinbox.setRange(MIN_INCOMMENSURATE_PHASE_SAMPLES, MAX_INCOMMENSURATE_PHASE_SAMPLES)
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
        spinbox.setRange(MIN_INCOMMENSURATE_LOOKUP_ANGLES, MAX_INCOMMENSURATE_LOOKUP_ANGLES)
        spinbox.setSingleStep(512)
        spinbox.setValue(int(value))
        spinbox.setToolTip(
            "Phase lookup resolution of the field calculation for each Fourier component. "
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
        spinbox.setRange(MIN_HISTOGRAM_BINS, MAX_HISTOGRAM_BINS)
        spinbox.setSingleStep(64)
        spinbox.setValue(int(value))
        spinbox.setToolTip(
            "Number of bins used for incommensurate p(B) histograms. More bins narrow the bin width."
        )
        spinbox.setMinimumWidth(104)
        return spinbox


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
        self._pin_button.setToolTip("Save this magnetic-field workspace.")
        self._pin_button.setAccessibleName("Save Workspace")
        self._pin_button.clicked.connect(self._emit_pin_requested)
        self._delete_workspace_button = QToolButton()
        self._delete_workspace_button.setIcon(icon("delete"))
        self._delete_workspace_button.setToolTip("Delete this saved magnetic-field workspace.")
        self._delete_workspace_button.setAccessibleName("Delete Workspace")
        self._delete_workspace_button.setVisible(False)
        self._delete_workspace_button.clicked.connect(self.delete_requested.emit)

        self._structure_summary = QLabel("Magnetic structure: -")
        self._structure_summary.setWordWrap(True)
        self._classification = QLabel("Structure type: -")
        self._component_summary_table = self._summary_table(
            self._COMPONENT_SUMMARY_HEADERS,
            stretch_column=4,
            accessible_name="Magnetic Structure Components",
        )
        self._component_summary_table.setVisible(False)
        self._edit_structure_button = QPushButton("Edit Magnetic Structure…")
        self._edit_structure_button.clicked.connect(self._open_structure_builder)

        self._site_summary = QLabel("Muon site: -")
        self._site_summary.setWordWrap(True)
        style_hint_text(self._site_summary)
        self._select_site_button = QPushButton("Select Muon Site…")
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
        self._advanced_demag_button = QPushButton("Advanced…")
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

        self._settings = MagneticFieldCalculationSettings()
        self._hyperfine_coupling = self._hyperfine_coupling_input()
        self._hyperfine_units = QComboBox()
        self._hyperfine_units.setAccessibleName("Hyperfine Coupling Units")
        for unit_key, label in CONTACT_COUPLING_UNIT_OPTIONS:
            self._hyperfine_units.addItem(label, unit_key)
        self._hyperfine_coupling.textChanged.connect(self._on_form_setting_changed)
        self._hyperfine_units.currentIndexChanged.connect(self._on_form_setting_changed)

        self._calculate_button = QPushButton("Calculate Dipolar Field")
        self._calculate_button.clicked.connect(self.calculate_fields)
        self._dipolar_tensor_button = QPushButton("Dipolar Tensor…")
        self._dipolar_tensor_button.clicked.connect(self._open_dipolar_tensor_dialog)
        self._nuclear_relaxation_button = QPushButton("Nuclear Relaxation…")
        self._nuclear_relaxation_button.clicked.connect(self._open_nuclear_relaxation_dialog)
        self._calculation_settings_summary = QLabel("")
        self._calculation_settings_summary.setWordWrap(True)
        style_hint_text(self._calculation_settings_summary)
        self._calculation_settings_button = QPushButton("Calculation Settings…")
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
            accessible_name="Field Results",
        )
        self._result_table.itemSelectionChanged.connect(self._sync_distribution_plot_button)
        self._result_table.itemSelectionChanged.connect(self._refresh_field_breakdown)
        self._configure_result_table_columns()
        self._breakdown_label = QLabel("Field breakdown")
        self._breakdown_table = self._summary_table(
            self._BREAKDOWN_HEADERS, stretch_column=0, accessible_name="Field Breakdown"
        )
        self._breakdown_hint = QLabel("")
        self._breakdown_hint.setWordWrap(True)
        style_hint_text(self._breakdown_hint)
        self._breakdown_contact_warning = QLabel("")
        self._breakdown_contact_warning.setWordWrap(True)
        style_warning_text(self._breakdown_contact_warning)
        self._breakdown_contact_warning.setVisible(False)

        self.setStyleSheet(magnetic_fields_stylesheet())
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
        hyperfine_label.setBuddy(self._hyperfine_coupling)
        hyperfine_label.setToolTip(
            "Contact hyperfine coupling that sets the Fermi contact field. Internally converted to Å⁻³."
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
        self._host_structure = None if document is None else active_host_structure(document)
        self.set_workspace_title(title, editable=title_editable, deletable=deletable)
        if magnetic_structure is None and document is not None:
            magnetic_structure = document.project.magnetic_structure
        self._model = magnetic_model_for_host(magnetic_structure, self._host_structure)
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
            "include_equivalent_sites": self._include_equivalent_sites.isChecked(),
            **self._settings.state_items(),
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

            self._settings = self._settings.with_state(state)
            self._include_equivalent_sites.setChecked(
                bool(state.get("include_equivalent_sites", False))
            )
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
        self._restoring_form_state = True
        try:
            self._none_radio.setChecked(True)
            self._demag_sample = DemagnetizingSample()
            self._set_measured_magnetization_from_state(False, None)
            self._set_vector_inputs(self._applied_field_inputs, None)
            self._settings = MagneticFieldCalculationSettings()
            self._include_equivalent_sites.setChecked(False)
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
        suffix = f" ({disabled} disabled)" if disabled else ""
        self._structure_summary.setText(f"Magnetic structure: {plural(count, 'component')}{suffix}")
        self._set_summary_rows(
            self._component_summary_table,
            [
                (
                    component.label,
                    component.kind.replace("_", " ").title(),
                    selector_label(component.selector, self._host_structure),
                    component_q_label(self._model, component),
                    component_moment_label(component),
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
        except ValueError:
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
        self.set_model(dialog.model())

    def set_model(self, model: MagneticStructureModel) -> None:
        """Use ``model`` as the page's magnetic structure, as the Builder's OK does."""

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
        return default_muon_site(self._document, self._host_structure)

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
        source = site.source_label or site_source_label(site.source)
        cluster = f" ({site.cluster_label})" if site.cluster_label else ""
        distortion = (
            " Local field corrected for muon-induced displacements."
            if site.include_distortion
            else ""
        )
        estimate = f" {UNRELAXED_SITE_NOTE}" if is_unrelaxed_site_source(site.source) else ""
        clearance_error = self._muon_site_unavailable_reason()
        refused = f" {clearance_error}" if clearance_error else ""
        self._site_summary.setText(
            f"Muon site: {site.label}{cluster} from {source} at "
            f"({site.fractional_position[0]:.4f}, {site.fractional_position[1]:.4f}, "
            f"{site.fractional_position[2]:.4f}).{distortion}{estimate}{refused}"
        )

    @staticmethod
    def _measured_magnetization_input(axis: str) -> DecimalLineEdit:
        edit = DecimalLineEdit(text="0.0", minimum=-1_000.0, maximum=1_000.0, decimals=6)
        edit.setToolTip(
            f"Cartesian {axis} component of the measured bulk magnetisation, in μB per formula unit."
        )
        edit.setMaximumWidth(80)
        edit.setEnabled(False)
        edit.setAccessibleName(f"Measured Bulk Magnetisation {axis}")
        return edit

    @staticmethod
    def _applied_field_input(axis: str) -> DecimalLineEdit:
        edit = DecimalLineEdit(text="0.0", minimum=-1_000.0, maximum=1_000.0, decimals=6)
        edit.setToolTip(f"Cartesian {axis} component of the applied field, in T.")
        edit.setMaximumWidth(80)
        edit.setAccessibleName(f"Applied Field {axis}")
        return edit

    @staticmethod
    def _hyperfine_coupling_input() -> DecimalLineEdit:
        edit = DecimalLineEdit(text="0.0", minimum=-1_000_000.0, maximum=1_000_000.0)
        edit.setToolTip("Set to zero to omit the contact hyperfine contribution.")
        edit.setMinimumWidth(104)
        return edit

    @staticmethod
    def _summary_table(
        headers: tuple[str, ...], *, stretch_column: int, accessible_name: str
    ) -> QTableWidget:
        """Build a read-only table that grows to fit its first rows and scrolls beyond them."""

        table = create_workflow_table(
            headers,
            accessible_name=accessible_name,
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

    def calculate_fields(self) -> None:
        """Run the field calculation for the selected site, as the Calculate button does."""

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
            options = self.field_options()
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
                    f"Results: calculated {plural(len(results), 'commensurate field value')}{corrected}."
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
                if approximant is not None and self._settings.use_approximants:
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
                        f"Results: calculated {plural(len(results), 'field value')} of the commensurate approximant "
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
                    status = f"Results: sampled {plural(len(results), 'incommensurate field distribution')}{corrected}"
                    flagged = sum(
                        1
                        for result in results
                        for check in getattr(result, "convergence", ())
                        if not check.converged
                    )
                    if flagged:
                        status += f"; {plural(flagged, 'convergence check needs', 'convergence checks need')} attention, select a row for advice"
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
        except Exception as exc:  # noqa: BLE001 - muesr and muLFC raise untyped errors
            self._calculation_status.setText(f"Calculation failed: {exc}")

    def _distorted_structure_for_site(self, site: MagneticMuonSiteOption) -> Structure | None:
        if not site.include_distortion:
            return None
        distorted = distortion_structure(self._document, site)
        if distorted is None:
            raise MagneticFieldError(
                "the relaxed structure for the muon-induced distortion correction is no longer available"
            )
        return distorted

    def calculation_settings(self) -> MagneticFieldCalculationSettings:
        """The numerical settings the next calculation uses."""

        return self._settings

    def set_calculation_settings(self, settings: MagneticFieldCalculationSettings) -> None:
        """Use ``settings`` from now on; results calculated with the old ones are cleared."""

        self._settings = settings
        self._on_form_setting_changed()

    def _open_calculation_settings_dialog(self) -> None:
        dialog = MagneticFieldCalculationSettingsDialog(
            self._settings,
            contact_tie_check=self._contact_tie_checker(),
            parent=self,
        )
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.set_calculation_settings(dialog.settings())

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
            supercell=self._settings.supercell,
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
            supercell=self._settings.supercell,
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
        distortion_available = distortion_structure(self._document, site) is not None
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
            distorted = distortion_structure(self._document, site)
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
            self._settings.summary(contact_shell=self._has_contact_coupling())
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

    def field_options(self) -> MagneticFieldOptions:
        """The calculation options the page's form and settings give the next calculation."""

        return MagneticFieldOptions(
            demag_shape=self._demag_shape(),
            demag_sample=self._active_demag_sample(),
            measured_magnetization=self._measured_magnetization(),
            applied_field=self._applied_field(),
            find_equivalent_sites=self._include_equivalent_sites.isChecked(),
            contact_coupling=self._contact_coupling_angstrom_minus3(),
            **self._settings.option_values(),
        )

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
        fractional_position = float_triplet_or_none(value.get("fractional_position"))
        if fractional_position is None:
            return None
        cartesian_position = float_triplet_or_none(value.get("cartesian_position"))
        if cartesian_position is None and self._host_structure is not None:
            cartesian_position = site_cartesian_position(fractional_position, self._host_structure)

        distortion_structure_id = uuid_or_none(value.get("distortion_structure_id"))

        properties = value.get("properties")
        return MagneticMuonSiteOption(
            source=str(value.get("source", CUSTOM_SITE_SOURCE)).strip() or CUSTOM_SITE_SOURCE,
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
                f"Results: restored {plural(len(result_payloads), 'commensurate field value')}."
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
                f"Results: restored {plural(len(result_payloads), 'incommensurate distribution')}."
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
        return UNRELAXED_ESTIMATE_SUFFIX if self._results_are_estimates() else ""

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
