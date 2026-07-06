"""Interactive analysis window for completed Quantum Corrections phonon jobs."""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from time import perf_counter

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGridLayout,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QScrollArea,
    QTableWidgetSelectionRange,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.math_label import MathLabel
from mufinder.app.screen_utils import clamped_size
from mufinder.app.theme import (
    body_font_family,
    current_palette,
    style_hint_text,
    style_section_title,
)
from mufinder.app.workflow_table_utils import create_workflow_table
from mufinder.domain.models import Structure
from mufinder.science.quantum_adiabatic import DirectionalPotential, SingleParticleResult, solve_single_particle_ground_state
from mufinder.science.quantum_phonons import (
    ATOMIC_MASS_KG,
    ELEMENTARY_CHARGE_C,
    LIGHT_SPEED_M_S,
    MUON_MASS_AMU,
    PhononCalculation,
    correlation_matrix,
    covariance_matrix,
    muon_mode_participations,
    parse_phonon_path,
    real_mode_displacements,
    scaled_mode_displacements,
    select_muon_modes,
    structure_from_phonon_calculation,
    w2_entanglement_witness,
)


@dataclass(frozen=True, slots=True)
class SingleParticleAnalysisRow:
    """Displayed summary for one solved single-particle direction."""

    label: str
    mode_index: int
    frequency_cm: float
    zpe_eV: float
    width_A: float


@dataclass(frozen=True, slots=True)
class PhononModeAnalysisRow:
    """Displayed summary for one Gamma-point phonon mode."""

    mode_index: int
    frequency_cm: float
    participation: float
    direction_label: str
    muon_dominant: bool = False


class QuantumPhononAnalysisDialog(QDialog):
    """Analyse one completed CASTEP phonon calculation."""

    def __init__(
        self,
        *,
        phonon_path: str,
        job_label: str,
        result_data: dict[str, object] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("quantumPhononAnalysisDialog")
        self.setWindowTitle(f"Quantum Analysis - {job_label}")
        self.resize(clamped_size(self, 1320, 760))

        self._phonon_path = phonon_path
        self._job_label = job_label
        self._result_data = dict(result_data or {})
        self._calculation = parse_phonon_path(phonon_path)
        self._selected_muon_modes = select_muon_modes(self._calculation)
        self._selected_muon_mode_indices = {mode.mode_index for mode in self._selected_muon_modes}
        self._phonon_mode_rows: list[PhononModeAnalysisRow] = []
        self._single_particle_results: list[SingleParticleResult] = []
        self._phonon_structure = structure_from_phonon_calculation(
            self._calculation,
            label=f"{job_label} phonon structure",
        )
        self._phonon_lattice = np.asarray(self._phonon_structure.lattice_vectors, dtype=float)
        self._phonon_fractional_positions = np.asarray(self._phonon_structure.fractional_positions, dtype=float)
        self._phonon_equilibrium_cartesian_positions = self._phonon_fractional_positions @ self._phonon_lattice
        self._muon_atom_index = self._calculation.atom_index_for_species("H:mu") - 1
        self._current_phonon_mode_row: int | None = None
        self._phonon_view_initialized = False
        self._phonon_phase_radians = 0.0
        self._phonon_last_tick_seconds: float | None = None
        self._phonon_animation_timer = QTimer(self)
        self._phonon_animation_timer.setInterval(20)
        self._phonon_animation_timer.timeout.connect(self._advance_phonon_animation)
        self._phonon_viewer_shutdown = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel(job_label)
        style_section_title(title)
        layout.addWidget(title)

        tabs = QTabWidget()
        tabs.setObjectName("quantumAnalysisTabs")
        tabs.addTab(self._phonon_modes_tab(), "Phonon Modes")
        tabs.addTab(self._single_particle_tab(), "Adiabatic Solver")
        tabs.addTab(self._correlation_tab(), "Muon Coupling")
        layout.addWidget(tabs, 1)

        self._populate_phonon_mode_table()
        self._select_default_phonon_mode_row()
        self._solve_single_particle()
        self._calculate_pair_matrices()

    @property
    def calculation(self) -> PhononCalculation:
        """Return the parsed phonon calculation."""

        return self._calculation

    @property
    def single_particle_results(self) -> tuple[SingleParticleResult, ...]:
        """Return solved single-particle directions."""

        return tuple(self._single_particle_results)

    def _phonon_modes_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("quantumAnalysisTabPage")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        title = QLabel("Phonon mode viewer")
        style_section_title(title)
        layout.addWidget(title)
        layout.addWidget(
            _guidance_label(
                "Inspect all Gamma-point phonon modes for the muonated structure. "
                "Selecting a row updates the embedded viewer; animation oscillates the structure while keeping "
                "the full-amplitude mode vectors as a fixed reference.",
                tab,
            )
        )

        controls = QHBoxLayout()
        controls.setContentsMargins(0, 0, 0, 0)
        controls.setSpacing(8)
        controls.addWidget(QLabel("Amplitude (Å)"))
        self._phonon_amplitude_spin = QDoubleSpinBox(tab)
        self._phonon_amplitude_spin.setRange(0.01, 2.0)
        self._phonon_amplitude_spin.setDecimals(2)
        self._phonon_amplitude_spin.setSingleStep(0.05)
        self._phonon_amplitude_spin.setValue(1.00)
        self._phonon_amplitude_spin.valueChanged.connect(lambda _value: self._refresh_phonon_mode_view())
        controls.addWidget(self._phonon_amplitude_spin)
        controls.addWidget(QLabel("Cycle time (s)"))
        self._phonon_cycle_spin = QDoubleSpinBox(tab)
        self._phonon_cycle_spin.setRange(0.25, 10.0)
        self._phonon_cycle_spin.setDecimals(2)
        self._phonon_cycle_spin.setSingleStep(0.25)
        self._phonon_cycle_spin.setValue(2.0)
        self._phonon_cycle_spin.valueChanged.connect(lambda _value: self._refresh_phonon_mode_view())
        controls.addWidget(self._phonon_cycle_spin)
        self._phonon_show_vectors_checkbox = QCheckBox("Show vectors", tab)
        self._phonon_show_vectors_checkbox.setChecked(False)
        self._phonon_show_vectors_checkbox.toggled.connect(lambda _checked: self._refresh_phonon_mode_view())
        controls.addWidget(self._phonon_show_vectors_checkbox)
        self._phonon_play_button = QPushButton("Play animation", tab)
        self._phonon_play_button.setCheckable(True)
        play_button_min_width = (
            max(
                self._phonon_play_button.fontMetrics().horizontalAdvance("Play animation"),
                self._phonon_play_button.fontMetrics().horizontalAdvance("Pause animation"),
            )
            + 36
        )
        self._phonon_play_button.setMinimumWidth(play_button_min_width)
        self._phonon_play_button.toggled.connect(self._toggle_phonon_animation)
        controls.addWidget(self._phonon_play_button)
        self._phonon_reset_button = QPushButton("Reset", tab)
        self._phonon_reset_button.clicked.connect(self._reset_phonon_animation)
        play_button_fixed_height = max(self._phonon_play_button.sizeHint().height(), self._phonon_reset_button.sizeHint().height())
        self._phonon_play_button.setFixedHeight(play_button_fixed_height)
        controls.addWidget(self._phonon_reset_button)
        self._phonon_animation_status_label = QLabel(tab)
        self._phonon_animation_status_label.setObjectName("phononAnimationStatusLabel")
        controls.addWidget(self._phonon_animation_status_label)
        self._update_phonon_animation_controls(False)
        controls.addStretch(1)
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal, tab)
        self._phonon_mode_table = _quantum_analysis_table(
            ("Mode", "Frequency (cm⁻¹)", "Muon participation", "Muon direction")
        )
        self._phonon_mode_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._phonon_mode_table.setMinimumWidth(320)
        phonon_header = self._phonon_mode_table.horizontalHeader()
        phonon_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        phonon_header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        phonon_header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        phonon_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        self._phonon_mode_table.itemSelectionChanged.connect(self._sync_selected_phonon_mode)
        splitter.addWidget(self._phonon_mode_table)

        self._phonon_viewer = AtomicStructureViewer(splitter)
        splitter.addWidget(self._phonon_viewer)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([580, 1240])
        layout.addWidget(splitter, 1)
        return tab

    def _single_particle_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("quantumAnalysisTabPage")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        layout.addLayout(
            _section_header(
                "Single-particle adiabatic approximation",
                "Use this view to compare confinement along the selected muon-dominated Gamma modes.",
                self._show_single_particle_help,
            )
        )

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Range (Å)"))
        self._range_spin = QDoubleSpinBox()
        self._range_spin.setRange(0.05, 5.0)
        self._range_spin.setDecimals(3)
        self._range_spin.setSingleStep(0.05)
        self._range_spin.setValue(2.0)
        self._range_spin.valueChanged.connect(lambda _value: self._solve_single_particle())
        controls.addWidget(self._range_spin)
        controls.addWidget(QLabel("Energy scale"))
        self._scale_combo = QComboBox()
        self._scale_combo.addItems(("Energy", "Sqrt energy"))
        self._scale_combo.currentTextChanged.connect(lambda _value: self._plot_single_particle())
        controls.addWidget(self._scale_combo)
        controls.addStretch(1)
        layout.addLayout(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self._single_particle_table = _quantum_analysis_table(
            ("Mode no.", "Direction", "Frequency (cm⁻¹)", "ZPE (eV)", "Width (Å)")
        )
        self._single_particle_table.setMinimumWidth(320)
        single_particle_header = self._single_particle_table.horizontalHeader()
        single_particle_header.setSectionResizeMode(0, QHeaderView.ResizeMode.ResizeToContents)
        single_particle_header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        single_particle_header.setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        single_particle_header.setSectionResizeMode(3, QHeaderView.ResizeMode.Stretch)
        single_particle_header.setSectionResizeMode(4, QHeaderView.ResizeMode.Stretch)
        self._single_particle_table.itemSelectionChanged.connect(self._plot_single_particle)
        splitter.addWidget(self._single_particle_table)
        self._figure = Figure(figsize=(6.0, 4.0), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        splitter.addWidget(self._canvas)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([480, 760])
        layout.addWidget(splitter, 1)
        return tab

    def _correlation_tab(self) -> QWidget:
        tab = QWidget()
        tab.setObjectName("quantumAnalysisTabPage")
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        layout.addLayout(
            _section_header(
                "Phonon covariance and correlation",
                "Use this view to inspect how zero-point motion couples the selected atom pair.",
                self._show_correlation_help,
            )
        )

        controls = QHBoxLayout()
        controls.addWidget(QLabel("Atom A"))
        self._atom_a_combo = QComboBox()
        controls.addWidget(self._atom_a_combo)
        controls.addWidget(QLabel("Atom B"))
        self._atom_b_combo = QComboBox()
        controls.addWidget(self._atom_b_combo)
        self._calculate_button = QPushButton("Calculate")
        self._calculate_button.clicked.connect(self._calculate_pair_matrices)
        controls.addWidget(self._calculate_button)
        controls.addStretch(1)
        layout.addLayout(controls)

        for atom in self._calculation.atoms:
            label = f"{atom.index}: {atom.species}"
            self._atom_a_combo.addItem(label, atom.index)
            self._atom_b_combo.addItem(label, atom.index)
        muon_index = self._calculation.atom_index_for_species("H:mu")
        self._atom_a_combo.setCurrentIndex(max(0, muon_index - 1))
        self._atom_b_combo.setCurrentIndex(0 if muon_index != 1 else min(1, len(self._calculation.atoms) - 1))

        summary = QHBoxLayout()
        self._w2_label = QLabel("w2")
        style_section_title(self._w2_label)
        summary.addWidget(self._w2_label)
        summary.addStretch(1)
        layout.addLayout(summary)

        matrices = QGridLayout()
        matrices.setContentsMargins(0, 0, 0, 0)
        matrices.setHorizontalSpacing(8)
        matrices.setVerticalSpacing(8)
        matrices.setColumnStretch(0, 1)
        matrices.setColumnStretch(1, 1)
        covariance_box = QGroupBox("Covariance")
        covariance_layout = QVBoxLayout(covariance_box)
        covariance_layout.setContentsMargins(12, 12, 12, 12)
        covariance_layout.setSpacing(10)
        covariance_layout.addStretch(1)
        covariance_layout.addWidget(
            MathLabel(r"C_{ab}\;(\AA^2)", display_mode=False, parent=covariance_box),
            0,
            Qt.AlignmentFlag.AlignHCenter,
        )
        covariance_layout.addSpacing(10)
        self._covariance_table = _matrix_table()
        covariance_layout.addWidget(self._covariance_table, 0, Qt.AlignmentFlag.AlignHCenter)
        covariance_layout.addStretch(1)
        matrices.addWidget(covariance_box, 0, 0)

        correlation_box = QGroupBox("Correlation")
        correlation_layout = QVBoxLayout(correlation_box)
        correlation_layout.setContentsMargins(12, 12, 12, 12)
        correlation_layout.setSpacing(10)
        correlation_layout.addStretch(1)
        correlation_layout.addWidget(
            MathLabel(r"\rho_{ab}", display_mode=False, parent=correlation_box),
            0,
            Qt.AlignmentFlag.AlignHCenter,
        )
        correlation_layout.addSpacing(10)
        self._correlation_table = _matrix_table()
        correlation_layout.addWidget(self._correlation_table, 0, Qt.AlignmentFlag.AlignHCenter)
        correlation_layout.addStretch(1)
        matrices.addWidget(correlation_box, 0, 1)
        layout.addLayout(matrices, 1)
        return tab

    def _solve_single_particle(self) -> None:
        half_range = float(self._range_spin.value()) if hasattr(self, "_range_spin") else 2.0
        x = np.linspace(-half_range, half_range, 161)
        results: list[SingleParticleResult] = []
        rows: list[SingleParticleAnalysisRow] = []
        for mode in self._selected_muon_modes:
            omega = 2.0 * pi * LIGHT_SPEED_M_S * 100.0 * mode.frequency_cm
            mass_kg = MUON_MASS_AMU * ATOMIC_MASS_KG
            x_m = x * 1e-10
            potential_eV = 0.5 * mass_kg * omega**2 * x_m**2 / ELEMENTARY_CHARGE_C
            direction = _muon_mode_direction(self._calculation, mode.mode_index)
            direction_label = _direction_label(direction)
            result = solve_single_particle_ground_state(
                DirectionalPotential(
                    label=direction_label,
                    displacement_A=tuple(float(value) for value in x),
                    energy_eV=tuple(float(value) for value in potential_eV),
                    effective_mass_amu=MUON_MASS_AMU,
                )
            )
            results.append(result)
            rows.append(
                SingleParticleAnalysisRow(
                    label=direction_label,
                    mode_index=mode.mode_index,
                    frequency_cm=mode.frequency_cm,
                    zpe_eV=result.ground_energy_eV,
                    width_A=result.width_A,
                )
            )
        self._single_particle_results = results
        if hasattr(self, "_single_particle_table"):
            _fill_single_particle_table(self._single_particle_table, rows)
            if rows:
                self._single_particle_table.setRangeSelected(
                    QTableWidgetSelectionRange(0, 0, len(rows) - 1, self._single_particle_table.columnCount() - 1),
                    True,
                )
            self._plot_single_particle()

    def _plot_single_particle(self) -> None:
        if not hasattr(self, "_figure"):
            return
        self._figure.clear()
        axis = self._figure.add_subplot(111)
        _style_energy_axis(self._figure, axis)
        sqrt_scale = hasattr(self, "_scale_combo") and self._scale_combo.currentText() == "Sqrt energy"
        palette = current_palette()
        colours = _analysis_line_colours()
        selected_indices = self._selected_single_particle_indices()
        for plot_index, result_index in enumerate(selected_indices):
            result = self._single_particle_results[result_index]
            x = np.asarray(result.grid_A, dtype=float)
            potential_meV = 1000.0 * np.asarray(result.potential_eV, dtype=float)
            energy_meV = 1000.0 * result.ground_energy_eV
            density = np.asarray(result.probability_density, dtype=float)
            colour = colours[plot_index % len(colours)]
            if sqrt_scale:
                y_potential = np.sqrt(np.clip(potential_meV, 0.0, None))
                y_energy = float(np.sqrt(max(energy_meV, 0.0)))
                y_label = r"$\sqrt{E}\ (\sqrt{\mathrm{meV}})$"
            else:
                y_potential = potential_meV
                y_energy = energy_meV
                y_label = r"$E\ (\mathrm{meV})$"
            axis.plot(x, y_potential, color=colour, linewidth=1.7, label=rf"{result.label}: $V(q)$")
            axis.axhline(y_energy, color=colour, linestyle="--", linewidth=1.0, alpha=0.82, label=rf"{result.label}: $E_0$")
            if density.max() > 0.0:
                density_scaled = density / density.max()
                span = max(float(np.max(y_potential)), y_energy, 1.0)
                axis.plot(
                    x,
                    y_energy + density_scaled * 0.25 * span,
                    color=colour,
                    alpha=0.58,
                    linewidth=1.0,
                    label=rf"{result.label}: $|\psi(q)|^2$",
                )
        axis.set_xlabel(r"Muon displacement, $q$ ($\AA$)")
        axis.set_ylabel(y_label)
        legend = axis.legend(loc="best", fontsize="small", frameon=True)
        legend.get_frame().set_facecolor(palette.bg_surface)
        legend.get_frame().set_edgecolor(palette.border)
        for text in legend.get_texts():
            text.set_color(palette.text_primary)
        self._canvas.draw_idle()

    def _selected_single_particle_indices(self) -> list[int]:
        if not hasattr(self, "_single_particle_table"):
            return list(range(len(self._single_particle_results)))
        selected = sorted({index.row() for index in self._single_particle_table.selectedIndexes()})
        if not selected:
            return list(range(len(self._single_particle_results)))
        return [row for row in selected if 0 <= row < len(self._single_particle_results)]

    def _calculate_pair_matrices(self) -> None:
        if not hasattr(self, "_atom_a_combo"):
            return
        atom_a = int(self._atom_a_combo.currentData())
        atom_b = int(self._atom_b_combo.currentData())
        cov = covariance_matrix(self._calculation, atom_a, atom_b)
        corr = correlation_matrix(self._calculation, atom_a, atom_b)
        _fill_matrix_table(self._covariance_table, cov)
        _fill_matrix_table(self._correlation_table, corr)
        self._w2_label.setText(f"w₂ = {w2_entanglement_witness(self._calculation, atom_a, atom_b):.6g}")

    def _show_single_particle_help(self) -> None:
        dialog = QuantumAnalysisHelpDialog(_single_particle_help_definition(), parent=self)
        dialog.exec()

    def _show_correlation_help(self) -> None:
        dialog = QuantumAnalysisHelpDialog(_correlation_help_definition(), parent=self)
        dialog.exec()

    def _populate_phonon_mode_table(self) -> None:
        gamma = self._calculation.gamma_point()
        participation = muon_mode_participations(self._calculation)
        rows: list[PhononModeAnalysisRow] = []
        for zero_based_index, frequency_cm in enumerate(gamma.frequencies_cm):
            mode_index = zero_based_index + 1
            mode_shape = real_mode_displacements(self._calculation, mode_index)
            direction_label = _direction_label(_normalized_mode_direction(mode_shape[self._muon_atom_index]))
            rows.append(
                PhononModeAnalysisRow(
                    mode_index=mode_index,
                    frequency_cm=float(frequency_cm),
                    participation=float(participation[zero_based_index]),
                    direction_label=direction_label,
                    muon_dominant=mode_index in self._selected_muon_mode_indices,
                )
            )
        self._phonon_mode_rows = rows
        _fill_phonon_mode_table(self._phonon_mode_table, rows)
        _resize_table_width_to_contents(self._phonon_mode_table, minimum=420, maximum=620)

    def _default_phonon_mode_row(self) -> int:
        if not self._phonon_mode_rows:
            return -1
        if self._selected_muon_modes:
            target_mode_index = self._selected_muon_modes[0].mode_index
            for row_index, row in enumerate(self._phonon_mode_rows):
                if row.mode_index == target_mode_index:
                    return row_index
        return 0

    def _select_default_phonon_mode_row(self) -> None:
        row = self._default_phonon_mode_row()
        if row < 0:
            return
        self._phonon_mode_table.selectRow(row)
        self._sync_selected_phonon_mode()

    def _selected_phonon_mode_row(self) -> int | None:
        selected = sorted({index.row() for index in self._phonon_mode_table.selectedIndexes()})
        if not selected:
            return None
        row = selected[0]
        if not (0 <= row < len(self._phonon_mode_rows)):
            return None
        return row

    def _selected_phonon_mode(self) -> PhononModeAnalysisRow | None:
        row = self._selected_phonon_mode_row()
        if row is None:
            return None
        return self._phonon_mode_rows[row]

    def _sync_selected_phonon_mode(self) -> None:
        self._current_phonon_mode_row = self._selected_phonon_mode_row()
        self._refresh_phonon_mode_view()

    def _update_phonon_animation_controls(self, playing: bool) -> None:
        palette = current_palette()
        if playing:
            self._phonon_play_button.setText("Pause animation")
            self._phonon_play_button.setToolTip("Animation is running. Click to pause the phonon mode.")
            self._phonon_play_button.setStyleSheet(
                f"QPushButton {{ background-color: {palette.success}; color: white; border: 2px solid {palette.success}; font-weight: 700; }}"
                f"QPushButton:hover {{ background-color: {palette.success}; }}"
            )
            self._phonon_animation_status_label.setText("Animation running")
            self._phonon_animation_status_label.setStyleSheet(
                f"color: {palette.success}; font-weight: 700; padding: 0 6px;"
            )
        else:
            self._phonon_play_button.setText("Play animation")
            self._phonon_play_button.setToolTip("Start the phonon mode animation.")
            self._phonon_play_button.setStyleSheet("")
            self._phonon_animation_status_label.setText("Animation paused")
            self._phonon_animation_status_label.setStyleSheet(
                f"color: {palette.text_muted}; font-weight: 600; padding: 0 6px;"
            )

    def _toggle_phonon_animation(self, playing: bool) -> None:
        self._update_phonon_animation_controls(playing)
        if playing:
            self._phonon_phase_radians = 0.0
            self._phonon_last_tick_seconds = perf_counter()
            self._phonon_animation_timer.start()
        else:
            self._phonon_animation_timer.stop()
            self._phonon_last_tick_seconds = None
        self._refresh_phonon_mode_view()

    def _reset_phonon_animation(self) -> None:
        self._phonon_phase_radians = 0.0
        self._phonon_last_tick_seconds = perf_counter() if self._phonon_animation_timer.isActive() else None
        self._refresh_phonon_mode_view()

    def _advance_phonon_animation(self) -> None:
        cycle_seconds = max(0.25, float(self._phonon_cycle_spin.value()))
        now = perf_counter()
        previous = self._phonon_last_tick_seconds
        elapsed_seconds = max(0.0, now - previous) if previous is not None else (self._phonon_animation_timer.interval() / 1000.0)
        self._phonon_last_tick_seconds = now
        phase_step = 2.0 * pi * elapsed_seconds / cycle_seconds
        self._phonon_phase_radians = (self._phonon_phase_radians + phase_step) % (2.0 * pi)
        self._refresh_phonon_mode_view()

    def _refresh_phonon_mode_view(self) -> None:
        viewer = getattr(self, "_phonon_viewer", None)
        if viewer is None:
            return
        mode = self._selected_phonon_mode()
        if mode is None:
            viewer.set_structure(self._phonon_structure) if not self._phonon_view_initialized else viewer.update_structure_geometry_preserving_view(self._phonon_structure)
            viewer.set_line_overlays(None)
            self._phonon_view_initialized = True
            return

        amplitude = float(self._phonon_amplitude_spin.value())
        full_displacements = scaled_mode_displacements(self._calculation, mode.mode_index, amplitude)
        phase_scale = np.sin(self._phonon_phase_radians) if self._phonon_play_button.isChecked() else 0.0
        animated_displacements = full_displacements * phase_scale
        structure = self._structure_with_cartesian_displacements(animated_displacements)
        if not self._phonon_view_initialized or getattr(viewer, "_structure", None) is None:
            viewer.set_structure(structure)
            self._phonon_view_initialized = True
        else:
            viewer.update_structure_geometry_preserving_view(structure)
        if self._phonon_show_vectors_checkbox.isChecked():
            viewer.set_line_overlays(self._phonon_vector_overlays(full_displacements))
        else:
            viewer.set_line_overlays(None)
        if viewer.selected_atom_indices() != (self._muon_atom_index,):
            viewer.select_atoms([self._muon_atom_index])

    def _structure_with_cartesian_displacements(self, displacements_ang: np.ndarray) -> Structure:
        cartesian = self._phonon_equilibrium_cartesian_positions + np.asarray(displacements_ang, dtype=float)
        fractional = np.linalg.solve(self._phonon_lattice.T, cartesian.T).T
        return type(self._phonon_structure)(
            label=self._phonon_structure.label,
            origin=self._phonon_structure.origin,
            lattice_vectors=[list(vector) for vector in self._phonon_structure.lattice_vectors],
            species=list(self._phonon_structure.species),
            fractional_positions=[[float(value) for value in row] for row in fractional.tolist()],
            source_file=self._phonon_structure.source_file,
            parent_structure_id=self._phonon_structure.parent_structure_id,
            properties=dict(self._phonon_structure.properties),
        )

    def _phonon_vector_overlays(self, displacements_ang: np.ndarray) -> list[dict[str, object]]:
        overlays: list[dict[str, object]] = []
        vectors = np.asarray(displacements_ang, dtype=float)
        palette = current_palette()
        for atom_index, displacement in enumerate(vectors):
            norm = float(np.linalg.norm(displacement))
            if norm <= 1e-12:
                continue
            start = self._phonon_equilibrium_cartesian_positions[atom_index]
            end = start + displacement
            is_muon = atom_index == self._muon_atom_index
            overlays.append(
                {
                    "start": [float(value) for value in start.tolist()],
                    "end": [float(value) for value in end.tolist()],
                    "color": "#e11d48" if is_muon else palette.accent,
                    "width": 6.0 if is_muon else 3.5,
                    "alpha": 0.95 if is_muon else 0.72,
                    "arrow": True,
                }
            )
        return overlays

    def _shutdown_phonon_viewer(self) -> None:
        timer = getattr(self, "_phonon_animation_timer", None)
        if timer is not None:
            timer.stop()
        if self._phonon_viewer_shutdown:
            return
        self._phonon_viewer_shutdown = True
        viewer = getattr(self, "_phonon_viewer", None)
        if viewer is None:
            return
        try:
            viewer.shutdown()
        except RuntimeError:
            pass

    def done(self, result: int) -> None:  # noqa: D401 - Qt API override.
        """Finish the dialog after releasing viewer resources."""

        self._shutdown_phonon_viewer()
        super().done(result)

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        self._shutdown_phonon_viewer()
        super().closeEvent(event)


class QuantumAnalysisHelpDialog(QDialog):
    """Read-only documentation for one quantum-analysis panel."""

    def __init__(self, definition: dict[str, object], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        title = str(definition.get("title", "Quantum analysis"))
        self.setObjectName("quantumAnalysisHelpDialog")
        self.setWindowTitle(title)
        self.resize(660, 560)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)

        title_label = QLabel(title)
        style_section_title(title_label)
        layout.addWidget(title_label)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)

        content = QWidget(scroll)
        content.setObjectName("quantumAnalysisHelpContent")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(0, 0, 0, 0)
        content_layout.setSpacing(12)

        for paragraph in definition.get("paragraphs", ()):
            content_layout.addWidget(_guidance_label(str(paragraph), content))

        equations = definition.get("equations", ())
        if equations:
            equation_title = QLabel("Equations", content)
            style_section_title(equation_title)
            content_layout.addWidget(equation_title)
            for equation in equations:
                math = MathLabel(str(equation), display_mode=True, parent=content)
                content_layout.addWidget(math, 0, Qt.AlignmentFlag.AlignLeft)

        symbols = definition.get("symbols", ())
        if symbols:
            symbols_title = QLabel("Symbols", content)
            style_section_title(symbols_title)
            content_layout.addWidget(symbols_title)
            for symbol, meaning in symbols:
                content_layout.addLayout(_symbol_row(str(symbol), str(meaning), content))

        content_layout.addStretch(1)
        scroll.setWidget(content)
        layout.addWidget(scroll, 1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)


def _quantum_analysis_table(headers: tuple[str, ...]) -> QTableWidget:
    """Shared chrome for this dialog's phonon-mode/single-particle/matrix tables.

    All columns size to content (not stretch-to-fill): these tables show
    compact numeric columns that should not stretch to the dialog width.
    """
    table = create_workflow_table(
        headers,
        default_resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        stretch_last_section=False,
    )
    table.horizontalHeader().setMinimumSectionSize(48)
    return table


def _guidance_label(text: str, parent: QWidget | None = None) -> QLabel:
    label = QLabel(text, parent)
    label.setWordWrap(True)
    style_hint_text(label)
    return label


def _section_header(title: str, guidance: str, callback) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(8)

    text_column = QVBoxLayout()
    text_column.setContentsMargins(0, 0, 0, 0)
    text_column.setSpacing(2)
    title_label = QLabel(title)
    style_section_title(title_label)
    text_column.addWidget(title_label)
    text_column.addWidget(_guidance_label(guidance))
    row.addLayout(text_column, 1)

    help_button = QToolButton()
    help_button.setObjectName("quantumAnalysisHelpButton")
    help_button.setText("?")
    help_button.setToolTip("Show calculation notes and equations")
    help_button.setAccessibleName(f"{title} help")
    help_button.clicked.connect(callback)
    row.addWidget(help_button, 0, Qt.AlignmentFlag.AlignTop)
    return row


def _symbol_row(symbol: str, meaning: str, parent: QWidget) -> QHBoxLayout:
    row = QHBoxLayout()
    row.setContentsMargins(0, 0, 0, 0)
    row.setSpacing(8)
    symbol_widget = MathLabel(symbol, display_mode=False, parent=parent)
    symbol_widget.setMinimumWidth(80)
    row.addWidget(symbol_widget, 0, Qt.AlignmentFlag.AlignTop)
    row.addWidget(_guidance_label(meaning, parent), 1)
    return row


def _style_energy_axis(figure: Figure, axis) -> None:
    palette = current_palette()
    figure.set_facecolor(palette.bg_primary)
    axis.set_facecolor(palette.bg_primary)
    axis.tick_params(colors=palette.text_muted, labelsize=9)
    axis.xaxis.label.set_color(palette.text_primary)
    axis.yaxis.label.set_color(palette.text_primary)
    axis.title.set_color(palette.text_primary)
    for spine in axis.spines.values():
        spine.set_color(palette.border)
    axis.grid(True, color=palette.border, linewidth=0.7, alpha=0.45)
    axis.set_axisbelow(True)


def _analysis_line_colours() -> tuple[str, ...]:
    palette = current_palette()
    if palette.mode == "dark":
        return (
            "#E8A838",
            "#4ADE80",
            "#7DD3FC",
            "#F87171",
            "#C4B5FD",
            "#FBBF24",
        )
    return (
        "#E02020",
        "#146C43",
        "#0369A1",
        "#A16207",
        "#7C3AED",
        "#B45309",
    )


def _muon_mode_direction(
    calculation: PhononCalculation,
    mode_index: int,
    *,
    muon_species: str = "H:mu",
) -> tuple[float, float, float]:
    muon_index = calculation.atom_index_for_species(muon_species) - 1
    vector = real_mode_displacements(calculation, mode_index, muon_species=muon_species)[muon_index]
    if vector.size != 3:
        return (0.0, 0.0, 0.0)
    return _normalized_mode_direction(np.asarray(vector, dtype=float))


def _normalized_mode_direction(vector: np.ndarray) -> tuple[float, float, float]:
    values = np.asarray(vector, dtype=float).reshape(-1)
    if values.size != 3:
        return (0.0, 0.0, 0.0)
    norm = float(np.linalg.norm(values))
    if norm <= 0.0:
        return (0.0, 0.0, 0.0)
    direction = values / norm
    return tuple(float(component) for component in direction)


def _direction_label(direction: tuple[float, float, float]) -> str:
    return "[" + " ".join(_format_direction_component(component) for component in direction) + "]"


def _format_direction_component(value: float) -> str:
    rounded = round(float(value), 2)
    if abs(rounded) < 5e-3:
        rounded = 0.0
    return f"{rounded:.2f}"


def _single_particle_help_definition() -> dict[str, object]:
    return {
        "title": "Single-particle adiabatic solver",
        "paragraphs": (
            "The solver builds a one-dimensional effective potential along each selected muon-dominated Gamma-mode direction. In the current phonon-derived approximation the host atoms remain fixed and the effective mass is the muon mass.",
            "The ZPE column reports the numerical ground-state energy of the finite-difference Schrodinger problem. The width column reports the RMS spread of the normalized muon probability density along the displacement coordinate.",
            "The square-root energy scale is a plotting aid for shallow wells: it keeps the minimum visible while compressing the high-energy part of the harmonic potential.",
        ),
        "equations": (
            r"V(q)=\frac{1}{2}m_\mu\omega^2q^2",
            r"\left[-\frac{\hbar^2}{2m_{\mathrm{eff}}}\frac{d^2}{dq^2}+V(q)\right]\psi(q)=E_0\psi(q)",
            r"\sigma_\mu=\sqrt{\left<q^2\right>-\left<q\right>^2}",
        ),
        "symbols": (
            (r"q", "Muon displacement along the selected phonon-mode direction."),
            (r"\omega=2\pi c\nu", "Angular frequency converted from the CASTEP phonon frequency."),
            (r"E_0", "Single-particle ground-state energy; reported as ZPE in the table."),
            (r"\sigma_\mu", "Muon wavefunction width along this one-dimensional coordinate."),
        ),
    }


def _correlation_help_definition() -> dict[str, object]:
    return {
        "title": "Muon coupling tensors",
        "paragraphs": (
            "The covariance matrix estimates how zero-point displacements of two atoms are coupled by the parsed Gamma-point phonon modes. MuFinder reports the tensor for the selected one-based CASTEP atom indices.",
            "The correlation matrix normalizes each covariance component by the corresponding auto-covariances, so values are dimensionless and easier to compare across atom pairs.",
            "The w2 witness follows the legacy MuFinder pairwise witness calculation and is intended as a comparative diagnostic rather than a standalone proof of many-body quantum behavior.",
        ),
        "equations": (
            r"C_{ab}^{ij}=\sum_n \frac{\hbar}{2\omega_n\sqrt{m_am_b}} e_{an}^i e_{bn}^{j*}",
            r"\rho_{ab}^{ij}=\frac{C_{ab}^{ij}}{\sqrt{C_{aa}^{ii}C_{bb}^{jj}}}",
            r"w_2=\lambda_{\min}\left(C_{aa}C_{bb}-C_{ab}C_{ab}^{T}\right)",
        ),
        "symbols": (
            (r"C_{ab}", "Covariance tensor between atoms a and b, reported in Å squared."),
            (r"\rho_{ab}", "Normalized correlation tensor between atoms a and b."),
            (r"e_{an}^i", "Cartesian phonon eigenvector component for atom a in mode n."),
            (r"w_2", "Pairwise entanglement-witness diagnostic from the legacy MuFinder analysis."),
        ),
    }


def _matrix_table() -> QTableWidget:
    table = _quantum_analysis_table(("x", "y", "z"))
    table.setRowCount(3)
    table.setVerticalHeaderLabels(("x", "y", "z"))
    table.verticalHeader().setVisible(True)
    table.setSizePolicy(QSizePolicy.Policy.Preferred, QSizePolicy.Policy.Preferred)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    return table


def _fill_single_particle_table(table: QTableWidget, rows: list[SingleParticleAnalysisRow]) -> None:
    previous = table.blockSignals(True)
    table.setRowCount(len(rows))
    for row_index, row in enumerate(rows):
        values = (
            str(row.mode_index),
            row.label,
            f"{row.frequency_cm:.3f}",
            f"{row.zpe_eV:.6g}",
            f"{row.width_A:.6g}",
        )
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            table.setItem(row_index, column, item)
    table.resizeRowsToContents()
    table.resizeColumnsToContents()
    if table.columnCount() >= 5:
        table.setColumnWidth(0, max(table.columnWidth(0), 78))
        table.setColumnWidth(3, max(table.columnWidth(3), 66))
        table.setColumnWidth(4, max(table.columnWidth(4), 62))
    table.blockSignals(previous)


def _fill_phonon_mode_table(table: QTableWidget, rows: list[PhononModeAnalysisRow]) -> None:
    previous = table.blockSignals(True)
    table.setRowCount(len(rows))
    success_color = QColor(current_palette().success)
    for row_index, row in enumerate(rows):
        values = (
            str(row.mode_index),
            f"{row.frequency_cm:.3f}",
            f"{row.participation:.6g}",
            row.direction_label,
        )
        for column, value in enumerate(values):
            item = QTableWidgetItem(value)
            if column == 2 and row.muon_dominant and success_color.isValid():
                item.setForeground(success_color)
            table.setItem(row_index, column, item)
    table.resizeRowsToContents()
    table.resizeColumnsToContents()
    if table.columnCount() >= 4:
        table.setColumnWidth(0, max(table.columnWidth(0), 72))
        table.setColumnWidth(1, max(table.columnWidth(1), 122))
        table.setColumnWidth(2, max(table.columnWidth(2), 136))
        table.setColumnWidth(3, min(max(table.columnWidth(3), 112), 132))
    table.blockSignals(previous)


def _resize_table_width_to_contents(table: QTableWidget, *, minimum: int, maximum: int) -> None:
    width = table.verticalHeader().width() + table.frameWidth() * 2
    for column in range(table.columnCount()):
        width += table.columnWidth(column)
    if table.verticalScrollBar().isVisible():
        width += table.verticalScrollBar().sizeHint().width()
    # A splitter (not a fixed width) governs how much horizontal space this
    # table actually gets, so only cap the content-driven width from above;
    # never lock it, or the splitter cannot shrink the table on small screens.
    table.setMaximumWidth(max(int(minimum), min(int(width), int(maximum))))


def _fill_matrix_table(table: QTableWidget, matrix: np.ndarray) -> None:
    table.setRowCount(3)
    for row in range(3):
        for column in range(3):
            table.setItem(row, column, QTableWidgetItem(f"{float(matrix[row, column]):.6g}"))
    table.resizeColumnsToContents()
    table.resizeRowsToContents()
    _resize_matrix_table_to_contents(table)


def _resize_matrix_table_to_contents(table: QTableWidget) -> None:
    width = table.verticalHeader().width() + table.frameWidth() * 2
    for column in range(table.columnCount()):
        width += table.columnWidth(column)
    if table.verticalScrollBar().isVisible():
        width += table.verticalScrollBar().sizeHint().width()

    height = table.horizontalHeader().height() + table.frameWidth() * 2
    for row in range(table.rowCount()):
        height += table.rowHeight(row)
    if table.horizontalScrollBar().isVisible():
        height += table.horizontalScrollBar().sizeHint().height()

    # Cap rather than lock the size: narrowing the dialog should let the
    # table scroll instead of refusing to shrink and starving its neighbours.
    table.setMaximumSize(width, height)
