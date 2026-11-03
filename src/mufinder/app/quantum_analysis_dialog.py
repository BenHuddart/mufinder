"""Interactive analysis window for completed Quantum Corrections phonon jobs."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from math import pi
from time import perf_counter

import numpy as np
from PySide6.QtCore import QSignalBlocker, Qt, QTimer
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QAbstractScrollArea,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QDoubleSpinBox,
    QGroupBox,
    QHeaderView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSizePolicy,
    QSpinBox,
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
    SPACING,
    STRUCTURE_OVERLAY_COLOURS,
    card_margins,
    current_palette,
    page_margins,
    reading_scroll_area,
    style_hint_text,
    style_section_title,
    style_warning_text,
)
from mufinder.app.viewer_dialog import StructureViewerDialog
from mufinder.app.workflow_table_utils import (
    create_workflow_table,
    fill_workflow_table,
    format_three_figures,
    selected_table_rows,
    set_header_tooltips,
    set_workflow_table_item,
    table_natural_width,
)
from mufinder.domain.models import Structure
from mufinder.science.quantum_phonons import (
    MuonNeighbour,
    PhononCalculation,
    correlation_matrix,
    covariance_matrix,
    muon_mass_warning,
    muon_mode_participations,
    muon_nearest_atoms,
    parse_phonon_path,
    real_mode_displacements,
    scaled_mode_displacements,
    select_muon_modes,
    structure_from_phonon_calculation,
    w2_entanglement_witness,
)


#: How many of the muon's nearest atoms ``Muon Coupling`` lists at first.
NEAREST_ATOMS_SHOWN = 12


@dataclass(frozen=True, slots=True)
class PhononModeAnalysisRow:
    """Displayed summary for one Gamma-point phonon mode."""

    mode_index: int
    frequency_cm: float
    participation: float
    direction_label: str
    muon_dominant: bool = False


class QuantumPhononAnalysisDialog(StructureViewerDialog):
    """Analyse one completed CASTEP phonon calculation."""

    def __init__(
        self,
        *,
        phonon_path: str,
        site_label: str,
        result_data: dict[str, object] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("quantumPhononAnalysisDialog")
        self.setWindowTitle(site_label)
        self.resize(clamped_size(self, 1320, 760))

        self._phonon_path = phonon_path
        self._result_data = dict(result_data or {})
        self._calculation = parse_phonon_path(phonon_path)
        self._selected_muon_modes = select_muon_modes(self._calculation)
        self._selected_muon_mode_indices = {mode.mode_index for mode in self._selected_muon_modes}
        self._phonon_mode_rows: list[PhononModeAnalysisRow] = []
        self._phonon_structure = structure_from_phonon_calculation(
            self._calculation,
            label=f"{site_label} phonon structure",
        )
        self._phonon_lattice = np.asarray(self._phonon_structure.lattice_vectors, dtype=float)
        self._phonon_fractional_positions = np.asarray(
            self._phonon_structure.fractional_positions, dtype=float
        )
        self._phonon_equilibrium_cartesian_positions = (
            self._phonon_fractional_positions @ self._phonon_lattice
        )
        self._muon_atom_index = self._calculation.atom_index_for_species("H:mu") - 1
        self._current_phonon_mode_row: int | None = None
        self._phonon_view_initialized = False
        self._phonon_vectors_shown = False
        self._phonon_phase_radians = 0.0
        self._phonon_last_tick_seconds: float | None = None
        self._phonon_animation_timer = QTimer(self)
        self._phonon_animation_timer.setInterval(20)
        self._phonon_animation_timer.timeout.connect(self._advance_phonon_animation)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel(site_label)
        style_section_title(title)
        layout.addWidget(title)

        # A phonon run CASTEP performed with the wrong mass for H:mu produces
        # entirely plausible numbers on every tab below, so say so up front rather
        # than letting the analysis be read at face value.
        self._muon_mass_warning = muon_mass_warning(self._calculation)
        self._muon_mass_warning_label: QLabel | None = None
        if self._muon_mass_warning is not None:
            # The theme keys warning styling off the object name, which
            # ``style_warning_text`` sets, so the label is identified by attribute
            # rather than by a custom object name.
            warning_label = QLabel(self._muon_mass_warning)
            warning_label.setWordWrap(True)
            style_warning_text(warning_label)
            layout.addWidget(warning_label)
            self._muon_mass_warning_label = warning_label

        # Only the first tab is built now; the others are built the first time
        # they are shown, so opening the window does not pay for hidden tabs.
        tabs = QTabWidget()
        tabs.setObjectName("quantumAnalysisTabs")
        self._unbuilt_tabs: dict[QWidget, Callable[[QWidget], None]] = {}
        tabs.addTab(self._phonon_modes_tab(), "Phonon Modes")
        self._add_lazy_tab(tabs, "Muon Coupling", self._build_correlation_tab)
        tabs.currentChanged.connect(lambda index: self._build_tab_if_needed(tabs.widget(index)))
        layout.addWidget(tabs, 1)

        self._populate_phonon_mode_table()
        self._select_default_phonon_mode_row()

    @property
    def calculation(self) -> PhononCalculation:
        """Return the parsed phonon calculation."""

        return self._calculation

    @property
    def muon_mass_warning(self) -> str | None:
        """Return the muon-mass mismatch warning shown for this run, if any."""

        return self._muon_mass_warning

    def _add_lazy_tab(self, tabs: QTabWidget, title: str, build: Callable[[QWidget], None]) -> None:
        page = QWidget()
        page.setObjectName("quantumAnalysisTabPage")
        page_layout = QVBoxLayout(page)
        page_layout.setContentsMargins(0, 0, 0, 0)
        self._unbuilt_tabs[page] = build
        tabs.addTab(page, title)

    def _build_tab_if_needed(self, page: QWidget | None) -> None:
        build = self._unbuilt_tabs.pop(page, None)
        if build is None:
            return
        content = QWidget(page)
        try:
            build(content)
        except Exception as exc:  # noqa: BLE001 - shown on the tab instead of opening.
            # Built on open, these tabs failed the whole window; built on first
            # show, the failure stays on the tab and the rest of the window works.
            content.setParent(None)
            content.deleteLater()
            content = QLabel(f"Could not compute this tab: {exc}", page)
            content.setWordWrap(True)
            content.setAlignment(Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignLeft)
            style_warning_text(content)
        page.layout().addWidget(content)

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
        self._phonon_amplitude_spin.valueChanged.connect(
            lambda _value: self._refresh_phonon_mode_view()
        )
        controls.addWidget(self._phonon_amplitude_spin)
        controls.addWidget(QLabel("Cycle time (s)"))
        self._phonon_cycle_spin = QDoubleSpinBox(tab)
        self._phonon_cycle_spin.setRange(0.25, 10.0)
        self._phonon_cycle_spin.setDecimals(2)
        self._phonon_cycle_spin.setSingleStep(0.25)
        self._phonon_cycle_spin.setValue(2.0)
        self._phonon_cycle_spin.valueChanged.connect(
            lambda _value: self._refresh_phonon_mode_view()
        )
        controls.addWidget(self._phonon_cycle_spin)
        self._phonon_show_vectors_checkbox = QCheckBox("Show vectors", tab)
        self._phonon_show_vectors_checkbox.setChecked(False)
        self._phonon_show_vectors_checkbox.toggled.connect(
            lambda _checked: self._refresh_phonon_mode_view()
        )
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
        play_button_fixed_height = max(
            self._phonon_play_button.sizeHint().height(),
            self._phonon_reset_button.sizeHint().height(),
        )
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

        self._phonon_viewer = self.hold_viewer(AtomicStructureViewer(splitter))
        splitter.addWidget(self._phonon_viewer)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([580, 1240])
        layout.addWidget(splitter, 1)
        return tab

    def _build_correlation_tab(self, tab: QWidget) -> None:
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)

        layout.addLayout(
            _section_header(
                "Muon's nearest atoms",
                "How strongly each nearby atom's zero-point motion follows the muon's. "
                "Select an atom to see its tensors.",
                self._show_correlation_help,
            )
        )

        controls = QHBoxLayout()
        controls.addWidget(QLabel(f"Muon: atom {self._muon_atom_index + 1}"))
        controls.addSpacing(SPACING.md)
        count_label = QLabel("Atoms shown")
        controls.addWidget(count_label)
        other_atoms = len(self._calculation.atoms) - 1
        self._nearest_count_spin = QSpinBox()
        self._nearest_count_spin.setRange(1, max(1, other_atoms))
        self._nearest_count_spin.setValue(min(NEAREST_ATOMS_SHOWN, max(1, other_atoms)))
        self._nearest_count_spin.setAccessibleName("Atoms shown")
        # A half-typed count must not shrink the list and lose the selected atom.
        self._nearest_count_spin.setKeyboardTracking(False)
        self._nearest_count_spin.setToolTip("How many of the muon's nearest atoms to list.")
        count_label.setBuddy(self._nearest_count_spin)
        controls.addWidget(self._nearest_count_spin)
        controls.addStretch(1)
        self._other_pair_button = QPushButton("Other pair…")
        self._other_pair_button.setToolTip("Show the tensors for any two atoms.")
        self._other_pair_button.clicked.connect(self._open_other_pair)
        controls.addWidget(self._other_pair_button)
        layout.addLayout(controls)

        content = QHBoxLayout()
        content.setSpacing(SPACING.sm)
        self._neighbour_table = _quantum_analysis_table(
            ("Atom", "Species", "Distance (Å)", "|c|"), accessible_name="Muon's nearest atoms"
        )
        self._neighbour_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._neighbour_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.ResizeMode.Stretch
        )
        set_header_tooltips(
            self._neighbour_table,
            {
                "Atom": "The atom's one-based CASTEP index.",
                "Distance (Å)": "Distance from the muon to the nearest periodic image of the atom.",
                "|c|": "Correlation of the atom's zero-point motion with the muon's, from 0 to 1. "
                "It equals −w2.",
            },
        )
        content.addWidget(self._neighbour_table, 1)

        tensor_column = QVBoxLayout()
        tensor_column.setSpacing(SPACING.sm)
        self._selected_atom_label = QLabel()
        style_section_title(self._selected_atom_label)
        tensor_column.addWidget(self._selected_atom_label)
        self._neighbour_tensors = _CouplingTensors()
        tensor_column.addWidget(self._neighbour_tensors)
        tensor_column.addStretch(1)
        content.addLayout(tensor_column)
        layout.addLayout(content, 1)

        self._neighbours: tuple[MuonNeighbour, ...] = ()
        self._refresh_nearest_atoms()
        self._neighbour_table.setMinimumWidth(table_natural_width(self._neighbour_table))
        self._neighbour_table.itemSelectionChanged.connect(self._show_selected_neighbour)
        self._nearest_count_spin.valueChanged.connect(lambda _value: self._refresh_nearest_atoms())

    def _selected_neighbour(self) -> MuonNeighbour | None:
        rows = selected_table_rows(self._neighbour_table)
        return self._neighbours[rows[0]] if rows else None

    def _refresh_nearest_atoms(self) -> None:
        """List the nearest atoms, keeping the selected atom if it is still listed."""

        previous = self._selected_neighbour()
        self._neighbours = muon_nearest_atoms(self._calculation, self._nearest_count_spin.value())
        with QSignalBlocker(self._neighbour_table):
            fill_workflow_table(
                self._neighbour_table,
                [
                    (
                        str(neighbour.atom_index),
                        neighbour.species,
                        f"{neighbour.distance_ang:.3f}",
                        format_three_figures(neighbour.correlation),
                    )
                    for neighbour in self._neighbours
                ],
            )
            kept = next(
                (
                    row
                    for row, neighbour in enumerate(self._neighbours)
                    if previous is not None and neighbour.atom_index == previous.atom_index
                ),
                0,
            )
            self._neighbour_table.selectRow(kept)
        selected = self._selected_neighbour()
        if selected is None or selected != previous:
            self._show_selected_neighbour()

    def _show_selected_neighbour(self) -> None:
        neighbour = self._selected_neighbour()
        if neighbour is None:
            self._selected_atom_label.setText("No atom selected")
            self._neighbour_tensors.clear()
            return
        self._selected_atom_label.setText(
            f"Muon and atom {neighbour.atom_index}: {neighbour.species}"
        )
        self._neighbour_tensors.show_pair(
            self._calculation, self._muon_atom_index + 1, neighbour.atom_index
        )

    def _open_other_pair(self) -> None:
        neighbour = self._selected_neighbour()
        dialog = MuonCouplingPairDialog(
            self._calculation,
            atom_b_index=neighbour.atom_index if neighbour is not None else None,
            parent=self,
        )
        dialog.exec()
        dialog.deleteLater()

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
            direction_label = _direction_label(
                _normalized_mode_direction(mode_shape[self._muon_atom_index])
            )
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
        # Select without the table's signal, then refresh the viewer once.
        with QSignalBlocker(self._phonon_mode_table):
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
            self._phonon_play_button.setToolTip(
                "Animation is running. Click to pause the phonon mode."
            )
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
        self._phonon_last_tick_seconds = (
            perf_counter() if self._phonon_animation_timer.isActive() else None
        )
        self._refresh_phonon_mode_view()

    def _advance_phonon_animation(self) -> None:
        cycle_seconds = max(0.25, float(self._phonon_cycle_spin.value()))
        now = perf_counter()
        previous = self._phonon_last_tick_seconds
        elapsed_seconds = (
            max(0.0, now - previous)
            if previous is not None
            else (self._phonon_animation_timer.interval() / 1000.0)
        )
        self._phonon_last_tick_seconds = now
        phase_step = 2.0 * pi * elapsed_seconds / cycle_seconds
        self._phonon_phase_radians = (self._phonon_phase_radians + phase_step) % (2.0 * pi)
        self._refresh_phonon_mode_view()

    def _refresh_phonon_mode_view(self) -> None:
        viewer = getattr(self, "_phonon_viewer", None)
        if viewer is None:
            return
        # Geometry, vectors and the muon highlight draw once per frame.
        with viewer.batch_update():
            self._show_phonon_mode_frame(viewer)

    def _show_phonon_mode_frame(self, viewer: AtomicStructureViewer) -> None:
        mode = self._selected_phonon_mode()
        if mode is None:
            if self._phonon_view_initialized:
                viewer.update_structure_geometry_preserving_view(self._phonon_structure)
            else:
                viewer.set_structure(self._phonon_structure)
                self._phonon_vectors_shown = False
            self._show_phonon_vectors(viewer, None)
            self._phonon_view_initialized = True
            return

        amplitude = float(self._phonon_amplitude_spin.value())
        full_displacements = scaled_mode_displacements(
            self._calculation, mode.mode_index, amplitude
        )
        phase_scale = (
            np.sin(self._phonon_phase_radians) if self._phonon_play_button.isChecked() else 0.0
        )
        animated_displacements = full_displacements * phase_scale
        structure = self._structure_with_cartesian_displacements(animated_displacements)
        if not self._phonon_view_initialized or getattr(viewer, "_structure", None) is None:
            viewer.set_structure(structure)
            # A new structure starts without overlays.
            self._phonon_vectors_shown = False
            self._phonon_view_initialized = True
        else:
            viewer.update_structure_geometry_preserving_view(structure)
        if self._phonon_show_vectors_checkbox.isChecked():
            self._show_phonon_vectors(viewer, self._phonon_vector_overlays(full_displacements))
        else:
            self._show_phonon_vectors(viewer, None)
        if viewer.selected_atom_indices() != (self._muon_atom_index,):
            viewer.select_atoms([self._muon_atom_index])

    def _show_phonon_vectors(
        self, viewer: AtomicStructureViewer, overlays: list[dict[str, object]] | None
    ) -> None:
        # Clearing vectors that are not drawn would only cost the viewer a render.
        if overlays is None and not self._phonon_vectors_shown:
            return
        viewer.set_line_overlays(overlays)
        self._phonon_vectors_shown = bool(overlays)

    def _structure_with_cartesian_displacements(self, displacements_ang: np.ndarray) -> Structure:
        cartesian = self._phonon_equilibrium_cartesian_positions + np.asarray(
            displacements_ang, dtype=float
        )
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
                    "color": STRUCTURE_OVERLAY_COLOURS.muon if is_muon else palette.accent,
                    "width": 6.0 if is_muon else 3.5,
                    "alpha": 0.95 if is_muon else 0.72,
                    "arrow": True,
                }
            )
        return overlays

    def shutdown_viewers(self) -> None:
        """Stop the mode animation, then shut the phonon viewer down."""

        timer = getattr(self, "_phonon_animation_timer", None)
        if timer is not None:
            timer.stop()
        super().shutdown_viewers()


class _CouplingTensors(QWidget):
    """The covariance and correlation tensors of one atom pair, sized to fit."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(SPACING.sm)
        self.covariance_table = _matrix_table("Covariance tensor")
        layout.addWidget(_tensor_box("Covariance", r"C_{ab}\;(\AA^2)", self.covariance_table))
        self.correlation_table = _matrix_table("Correlation tensor")
        layout.addWidget(_tensor_box("Correlation", r"\rho_{ab}", self.correlation_table))

    def show_pair(self, calculation: PhononCalculation, atom_a: int, atom_b: int) -> None:
        """Show the tensors of two one-based atoms."""

        _fill_matrix_table(self.covariance_table, covariance_matrix(calculation, atom_a, atom_b))
        _fill_matrix_table(self.correlation_table, correlation_matrix(calculation, atom_a, atom_b))

    def clear(self) -> None:
        """Show no pair."""

        self.covariance_table.clearContents()
        self.correlation_table.clearContents()


class MuonCouplingPairDialog(QDialog):
    """The coupling of any two atoms, chosen by their CASTEP index."""

    def __init__(
        self,
        calculation: PhononCalculation,
        *,
        atom_b_index: int | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setObjectName("muonCouplingPairDialog")
        self.setWindowTitle("Other pair")
        self._calculation = calculation

        layout = QVBoxLayout(self)
        layout.setContentsMargins(page_margins())
        layout.setSpacing(SPACING.sm)

        controls = QHBoxLayout()
        muon_index = calculation.atom_index_for_species("H:mu")
        self._atom_a_combo = self._atom_combo("Atom A", muon_index, controls)
        fallback = next((a.index for a in calculation.atoms if a.index != muon_index), muon_index)
        self._atom_b_combo = self._atom_combo("Atom B", atom_b_index or fallback, controls)
        controls.addStretch(1)
        layout.addLayout(controls)

        self._correlation_label = QLabel()
        style_section_title(self._correlation_label)
        layout.addWidget(self._correlation_label)
        self._tensors = _CouplingTensors(self)
        layout.addWidget(self._tensors)
        layout.addStretch(1)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, parent=self)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._show_pair()
        self._atom_a_combo.currentIndexChanged.connect(lambda _index: self._show_pair())
        self._atom_b_combo.currentIndexChanged.connect(lambda _index: self._show_pair())

    def _atom_combo(self, title: str, atom_index: int, row: QHBoxLayout) -> QComboBox:
        label = QLabel(title)
        combo = QComboBox()
        combo.setAccessibleName(title)
        label.setBuddy(combo)
        for atom in self._calculation.atoms:
            combo.addItem(f"{atom.index}: {atom.species}", atom.index)
        combo.setCurrentIndex(combo.findData(atom_index))
        row.addWidget(label)
        row.addWidget(combo)
        return combo

    def _show_pair(self) -> None:
        atom_a = int(self._atom_a_combo.currentData())
        atom_b = int(self._atom_b_combo.currentData())
        self._tensors.show_pair(self._calculation, atom_a, atom_b)
        correlation = abs(w2_entanglement_witness(self._calculation, atom_a, atom_b))
        self._correlation_label.setText(f"|c| = {format_three_figures(correlation)}")


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

        scroll = reading_scroll_area(self)

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


def _quantum_analysis_table(headers: tuple[str, ...], *, accessible_name: str = "") -> QTableWidget:
    """Shared chrome for this dialog's tables.

    Columns size to content (not stretch-to-fill) unless the caller says
    otherwise: most of these tables show compact numeric columns that should
    not stretch to the dialog width.
    """
    table = create_workflow_table(
        headers,
        default_resize_mode=QHeaderView.ResizeMode.ResizeToContents,
        stretch_last_section=False,
        accessible_name=accessible_name,
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
    help_button.setToolTip("Show calculation notes and equations.")
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


def _correlation_help_definition() -> dict[str, object]:
    return {
        "title": "Muon coupling tensors",
        "paragraphs": (
            "The covariance matrix estimates how zero-point displacements of two atoms are coupled by the parsed Gamma-point phonon modes. MuFinder reports the tensor for the muon and the selected atom, or for any two atoms through 'Other pair…'.",
            "The correlation matrix normalizes each covariance component by the corresponding auto-covariances, so values are dimensionless and easier to compare across atom pairs.",
            "The w2 witness is minus the magnitude of the pair-covariance trace, normalized by the geometric mean of the two auto-covariance traces. It lies between -1 and 0, with values closer to -1 indicating more strongly correlated zero-point motion, and is intended as a comparative diagnostic rather than a standalone proof of many-body quantum behavior. The table lists its size, |c| = −w2, from 0 to 1.",
            "Here w2 is the harmonic (phonon-covariance) estimate of the positional entanglement witness w2 = −|c| of Gomilšek et al., where c is the Pearson correlation of the muon and nuclear displacements. The sum runs over every stable mode; modes with imaginary frequencies are left out. On the α-N₂ phonon run of that paper this gives |c| = 0.30 and 0.31 for the two nitrogens nearest the muon, against 0.30(4) from path-integral molecular dynamics. The paper's harmonic ≈ 0.26 sums only the modes that carry some muon motion (a participation of at least 1e-4), so it leaves out the soft host modes that widen each nitrogen's motion.",
        ),
        "equations": (
            r"C_{ab}^{ij}=\sum_n \frac{\hbar}{2\omega_n\sqrt{m_am_b}} e_{an}^i e_{bn}^{j*}",
            r"\rho_{ab}^{ij}=\frac{C_{ab}^{ij}}{\sqrt{C_{aa}^{ii}C_{bb}^{jj}}}",
            r"w_2=-\frac{\left|\mathrm{Tr}\,C_{ab}\right|}{\sqrt{\mathrm{Tr}\,C_{aa}\,\mathrm{Tr}\,C_{bb}}}",
        ),
        "symbols": (
            (r"C_{ab}", "Covariance tensor between atoms a and b, reported in Å squared."),
            (r"\rho_{ab}", "Normalized correlation tensor between atoms a and b."),
            (r"e_{an}^i", "Cartesian phonon eigenvector component for atom a in mode n."),
            (
                r"w_2",
                "Pairwise witness: the normalized covariance trace, from -1 (strongly correlated) to 0.",
            ),
        ),
    }


def _tensor_box(title: str, symbol: str, table: QTableWidget) -> QGroupBox:
    box = QGroupBox(title)
    layout = QVBoxLayout(box)
    layout.setContentsMargins(card_margins())
    layout.setSpacing(SPACING.sm)
    layout.addWidget(
        MathLabel(symbol, display_mode=False, parent=box), 0, Qt.AlignmentFlag.AlignHCenter
    )
    layout.addWidget(table, 0, Qt.AlignmentFlag.AlignHCenter)
    return box


def _matrix_table(accessible_name: str) -> QTableWidget:
    """A 3 × 3 table that takes the size of its contents and never scrolls."""

    table = _quantum_analysis_table(("x", "y", "z"), accessible_name=accessible_name)
    table.setRowCount(3)
    table.setVerticalHeaderLabels(("x", "y", "z"))
    table.verticalHeader().setVisible(True)
    # Wide enough for any three-figure value, so moving between atoms moves nothing.
    table.horizontalHeader().setMinimumSectionSize(
        table.fontMetrics().horizontalAdvance("-0.000000") + SPACING.lg
    )
    table.verticalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
    table.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
    table.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustToContents)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    return table


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
    scale = float(np.abs(matrix).max())
    for row in range(3):
        for column in range(3):
            set_workflow_table_item(
                table, row, column, format_three_figures(matrix[row, column], scale=scale)
            )
