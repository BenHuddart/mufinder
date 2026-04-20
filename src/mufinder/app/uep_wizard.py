"""Interactive UEP analysis dialog for one completed source calculation."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from dataclasses import asdict
from pathlib import Path
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from PySide6.QtCore import Qt, QThread, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QFileDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QRadioButton,
    QSlider,
    QSplitter,
    QSpinBox,
    QTabWidget,
    QTableWidget,
    QTableWidgetItem,
    QTextEdit,
    QVBoxLayout,
    QWidget,
    QSizePolicy,
)

from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.app.theme import current_palette, style_hint_text, style_section_title
from mufinder.domain.models import Structure
from mufinder.science.site_generation import generate_random_sites, generate_random_sites_by_distance, lattice_frac_to_cart
from mufinder.science.uep import (
    StationaryPointClassification,
    UepBatchSummary,
    UepClusterSummary,
    UepError,
    UepField,
    UepIsosurfaceMesh,
    UepLineSample,
    UepOptimizationResult,
    UepPlaneSample,
    UepPotentialGrid,
    UepProbeResult,
    UepSourceData,
    cluster_optimized_positions,
    deserialize_potential_grid,
    extract_isosurface,
    load_uep_source,
    optimize_muon_position,
    potential_grid,
    probe_point,
    sample_line,
    sample_plane,
    serialize_potential_grid,
    write_potential_grid_xsf,
)


_THEORY_TEXT = (
    "UEP in MuFinder follows the method of Simone Sturniolo and Leandro Liborio, "
    "\"Computational prediction of muon stopping sites: A novel take on the unperturbed "
    "electrostatic potential method,\" J. Chem. Phys. 153, 044111 (2020), "
    "DOI: 10.1063/5.0012381.\n\n"
    "The method assumes a neutral unperturbed host crystal and a positively charged "
    "muon rather than muonium. MuFinder combines the CASTEP electronic density with a "
    "Gaussian ionic model, evaluates the electrostatic potential in reciprocal space, "
    "and uses gradients and Hessians to identify stationary points.\n\n"
    "The paper also highlights that special Wyckoff positions without free coordinates "
    "are stationary points of symmetry-preserving scalar fields, so high-symmetry empty "
    "interstitial sites are useful UEP candidates. Symmetry alone is not enough to "
    "guarantee a minimum, which is why MuFinder also exposes Hessian classification, "
    "zero-point estimates, and batch-search clustering.\n\n"
    "UEP tends to be more reliable in metallic or weakly perturbative hosts. It is much "
    "less reliable when the muon forms a strong local bond or reconstructs the lattice, "
    "as can happen in some insulating or semiconducting systems."
)


class _TripleInput(QWidget):
    """Compact row of three decimal controls for fractional coordinates."""

    def __init__(self, *, label_x: str = "fX", label_y: str = "fY", label_z: str = "fZ", parent: QWidget | None = None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)
        self._x = DecimalLineEdit(self)
        self._y = DecimalLineEdit(self)
        self._z = DecimalLineEdit(self)
        for control in (self._x, self._y, self._z):
            control.setRange(-10.0, 10.0)
            control.setDecimals(6)
        for label, control in ((label_x, self._x), (label_y, self._y), (label_z, self._z)):
            layout.addWidget(QLabel(label, self))
            layout.addWidget(control)

    def set_values(self, values: tuple[float, float, float]) -> None:
        self._x.setValue(float(values[0]))
        self._y.setValue(float(values[1]))
        self._z.setValue(float(values[2]))

    def values(self) -> tuple[float, float, float]:
        return (self._x.value(), self._y.value(), self._z.value())


class _IsosurfaceGridThread(QThread):
    """Background thread that computes the full-cell UEP potential grid."""

    grid_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, field: UepField) -> None:
        super().__init__()
        self._field = field

    def run(self) -> None:
        try:
            self.grid_ready.emit(potential_grid(self._field))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class _IsosurfaceMeshThread(QThread):
    """Background thread that extracts one isosurface mesh from a cached grid."""

    mesh_ready = Signal(int, object)
    failed = Signal(int, str)

    def __init__(self, grid: UepPotentialGrid, *, level_ev: float, request_id: int) -> None:
        super().__init__()
        self._grid = grid
        self._level_ev = float(level_ev)
        self._request_id = int(request_id)

    def run(self) -> None:
        try:
            self.mesh_ready.emit(self._request_id, extract_isosurface(self._grid, level_ev=self._level_ev))
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(self._request_id, str(exc))


class UepAnalysisDialog(QDialog):
    """Wizard-like dialog for UEP probing, plotting, batch search, and promotion."""

    promote_requested = Signal(object)
    state_changed = Signal(dict)
    potential_grid_ready = Signal(object)

    def __init__(
        self,
        *,
        source_job_label: str,
        source_structure_id,
        source_structure: Structure | None = None,
        den_fmt_path: str | Path,
        castep_path: str | Path,
        cell_path: str | Path | None = None,
        initial_potential_grid: UepPotentialGrid | None = None,
        initial_state: Mapping[str, object] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"UEP Analysis - {source_job_label}")
        self.resize(1180, 920)
        self.setMinimumSize(980, 760)

        self._source_job_label = source_job_label
        self._source_structure_id = source_structure_id
        self._source_structure = source_structure
        self._source: UepSourceData = load_uep_source(
            den_fmt_path=den_fmt_path,
            castep_path=castep_path,
            cell_path=cell_path,
        )
        self._field = UepField(self._source)
        self._latest_probe: UepProbeResult | None = None
        self._latest_line_sample = None
        self._latest_plane_sample = None
        self._latest_batch_results: list[UepOptimizationResult] = []
        self._latest_batch_summary: UepBatchSummary | None = None
        self._potential_grid_cache: UepPotentialGrid | None = None
        self._latest_isosurface_mesh: UepIsosurfaceMesh | None = None
        self._isosurface_status_kind = "info"
        self._restoring_state = False
        self._isosurface_grid_thread: _IsosurfaceGridThread | None = None
        self._isosurface_mesh_thread: _IsosurfaceMeshThread | None = None
        self._isosurface_request_id = 0
        self._viewer_host_structure = self._build_viewer_host_structure()

        palette = current_palette()
        self._line_figure = Figure(figsize=(6.0, 4.0), facecolor=palette.bg_surface, constrained_layout=True)
        self._line_canvas = FigureCanvasQTAgg(self._line_figure)
        self._plane_figure = Figure(figsize=(6.0, 4.0), facecolor=palette.bg_surface, constrained_layout=True)
        self._plane_canvas = FigureCanvasQTAgg(self._plane_figure)

        self._tabs = QTabWidget(self)
        self._summary_tab = self._build_summary_tab()
        self._probe_tab = self._build_probe_tab()
        self._line_tab = self._build_line_tab()
        self._plane_tab = self._build_plane_tab()
        self._isosurface_tab = self._build_isosurface_tab()
        self._batch_tab = self._build_batch_tab()
        self._tabs.addTab(self._summary_tab, "Summary")
        self._tabs.addTab(self._probe_tab, "Probe")
        self._tabs.addTab(self._line_tab, "Line")
        self._tabs.addTab(self._plane_tab, "Plane")
        self._tabs.addTab(self._isosurface_tab, "Isosurface")
        self._tabs.addTab(self._batch_tab, "Batch Search")
        self._tabs.currentChanged.connect(lambda _index: self._emit_state_changed())

        layout = QVBoxLayout(self)
        layout.addWidget(self._tabs)

        self._populate_summary()
        if initial_potential_grid is not None:
            self._apply_potential_grid_cache(initial_potential_grid, status_text="Cached full-cell potential grid is ready.")
        self._restore_initial_state(initial_state)

    def _build_summary_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        source_group = QGroupBox("Source Validation", tab)
        source_layout = QVBoxLayout(source_group)
        self._summary_text = QTextEdit(source_group)
        self._summary_text.setReadOnly(True)
        source_layout.addWidget(self._summary_text)

        theory_group = QGroupBox("Theory and Caveats", tab)
        theory_layout = QVBoxLayout(theory_group)
        self._theory_text = QTextEdit(theory_group)
        self._theory_text.setReadOnly(True)
        self._theory_text.setPlainText(_THEORY_TEXT)
        theory_layout.addWidget(self._theory_text)

        layout.addWidget(source_group)
        layout.addWidget(theory_group)
        return tab

    def _build_probe_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        controls = QGroupBox("Point Probe (fractional coordinates)", tab)
        form = QFormLayout(controls)
        self._probe_point = _TripleInput(parent=controls)
        self._probe_point.set_values((0.25, 0.25, 0.25))
        self._probe_run = QPushButton("Evaluate Point", controls)
        self._probe_promote = QPushButton("Promote Current Point", controls)
        self._probe_promote.setEnabled(False)
        self._probe_run.clicked.connect(self._run_probe)
        self._probe_promote.clicked.connect(self._promote_current_probe)
        actions = QHBoxLayout()
        actions.addWidget(self._probe_run)
        actions.addWidget(self._probe_promote)
        actions.addStretch(1)
        form.addRow("Fractional position", self._probe_point)
        form.addRow("", actions)

        self._probe_output = QTextEdit(tab)
        self._probe_output.setReadOnly(True)

        layout.addWidget(controls)
        layout.addWidget(self._probe_output, 1)
        return tab

    def _build_line_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        controls = QGroupBox("Line Sampling (fractional coordinates)", tab)
        form = QFormLayout(controls)
        self._line_start = _TripleInput(parent=controls)
        self._line_end = _TripleInput(parent=controls)
        self._line_start.set_values((0.0, 0.0, 0.0))
        self._line_end.set_values((0.5, 0.5, 0.5))
        self._line_points = QSpinBox(controls)
        self._line_points.setRange(2, 500)
        self._line_points.setValue(101)
        self._line_run = QPushButton("Sample Line", controls)
        self._line_run.clicked.connect(self._run_line_sample)
        form.addRow("Start", self._line_start)
        form.addRow("End", self._line_end)
        form.addRow("Points", self._line_points)
        form.addRow("", self._line_run)

        self._line_data = QTextEdit(tab)
        self._line_data.setReadOnly(True)
        layout.addWidget(controls)
        layout.addWidget(self._line_canvas, 1)
        layout.addWidget(self._line_data, 1)
        return tab

    def _build_plane_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        controls = QGroupBox("Plane Sampling (fractional coordinates)", tab)
        form = QFormLayout(controls)
        self._plane_origin = _TripleInput(parent=controls)
        self._plane_x = _TripleInput(parent=controls)
        self._plane_y = _TripleInput(parent=controls)
        self._plane_origin.set_values((0.0, 0.0, 0.0))
        self._plane_x.set_values((0.5, 0.0, 0.0))
        self._plane_y.set_values((0.0, 0.5, 0.0))
        self._plane_points_x = QSpinBox(controls)
        self._plane_points_x.setRange(2, 200)
        self._plane_points_x.setValue(61)
        self._plane_points_y = QSpinBox(controls)
        self._plane_points_y.setRange(2, 200)
        self._plane_points_y.setValue(61)
        self._plane_run = QPushButton("Sample Plane", controls)
        self._plane_run.clicked.connect(self._run_plane_sample)
        form.addRow("Origin", self._plane_origin)
        form.addRow("X corner", self._plane_x)
        form.addRow("Y corner", self._plane_y)
        form.addRow("Points along X", self._plane_points_x)
        form.addRow("Points along Y", self._plane_points_y)
        form.addRow("", self._plane_run)

        self._plane_data = QTextEdit(tab)
        self._plane_data.setReadOnly(True)
        layout.addWidget(controls)
        layout.addWidget(self._plane_canvas, 1)
        layout.addWidget(self._plane_data, 1)
        return tab

    def _build_isosurface_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(14)

        controls = QWidget(tab)
        controls.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Maximum)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(12)

        title = QLabel("Electrostatic-Potential Isosurface", controls)
        style_section_title(title)
        subtitle = QLabel(
            "Explore the host electrostatic-potential landscape with a continuous level control and export-ready grid data.",
            controls,
        )
        style_hint_text(subtitle)
        subtitle.setWordWrap(True)
        controls_layout.addWidget(title)
        controls_layout.addWidget(subtitle)

        self._isosurface_compute_button = QPushButton("Calculate Cell Potential Grid", controls)
        self._isosurface_compute_button.clicked.connect(self._start_isosurface_grid_calculation)
        self._isosurface_export_button = QPushButton("Export XSF...", controls)
        self._isosurface_export_button.setEnabled(False)
        self._isosurface_export_button.clicked.connect(self._export_potential_grid_xsf)
        button_row = QHBoxLayout()
        button_row.setContentsMargins(0, 0, 0, 0)
        button_row.setSpacing(10)
        button_row.addWidget(self._isosurface_compute_button)
        button_row.addWidget(self._isosurface_export_button)
        button_row.addStretch(1)
        controls_layout.addLayout(button_row)

        slider_panel = QFrame(controls)
        slider_panel.setObjectName("uepIsosurfaceSliderPanel")
        slider_layout = QVBoxLayout(slider_panel)
        slider_layout.setContentsMargins(18, 16, 18, 16)
        slider_layout.setSpacing(12)

        slider_title = QLabel("Isosurface level", slider_panel)
        slider_title.setObjectName("uepIsosurfacePanelTitle")
        slider_layout.addWidget(slider_title)

        slider_intro = QLabel(
            "Sweep through the cached potential range to inspect filled isosurfaces across the host cell.",
            slider_panel,
        )
        style_hint_text(slider_intro)
        slider_intro.setWordWrap(True)
        slider_layout.addWidget(slider_intro)

        self._isosurface_slider = QSlider(Qt.Orientation.Horizontal, controls)
        self._isosurface_slider.setObjectName("uepIsosurfaceSlider")
        self._isosurface_slider.setRange(0, 10000)
        self._isosurface_slider.setValue(500)
        self._isosurface_slider.setMinimumWidth(620)
        self._isosurface_slider.setMinimumHeight(42)
        self._isosurface_slider.setSingleStep(10)
        self._isosurface_slider.setPageStep(250)
        self._isosurface_slider.setEnabled(False)
        self._isosurface_slider.valueChanged.connect(self._sync_isosurface_level_editor_from_slider)
        self._isosurface_slider.sliderReleased.connect(self._request_isosurface_from_slider)

        self._isosurface_min_label = QLabel("Min", slider_panel)
        self._isosurface_min_label.setObjectName("uepIsosurfaceBound")
        self._isosurface_max_label = QLabel("Max", slider_panel)
        self._isosurface_max_label.setObjectName("uepIsosurfaceBound")
        slider_layout.addWidget(self._isosurface_slider)

        slider_bounds_row = QHBoxLayout()
        slider_bounds_row.setContentsMargins(0, 0, 0, 0)
        slider_bounds_row.setSpacing(12)
        slider_bounds_row.addWidget(self._isosurface_min_label)
        slider_bounds_row.addStretch(1)
        slider_bounds_row.addWidget(self._isosurface_max_label)
        slider_layout.addLayout(slider_bounds_row)

        slider_hint = QLabel("Drag to a new threshold, then release to update the filled isosurface.", slider_panel)
        style_hint_text(slider_hint)
        slider_hint.setWordWrap(True)
        slider_layout.addWidget(slider_hint)

        self._isosurface_level_input = DecimalLineEdit(controls)
        self._isosurface_level_input.setRange(-1_000_000.0, 1_000_000.0)
        self._isosurface_level_input.setDecimals(6)
        self._isosurface_level_input.setEnabled(False)
        self._isosurface_level_input.editingFinished.connect(self._request_isosurface_from_editor)

        value_grid = QGridLayout()
        value_grid.setContentsMargins(0, 0, 0, 0)
        value_grid.setHorizontalSpacing(12)
        value_grid.setVerticalSpacing(8)
        level_value_label = QLabel("Level value (eV)", slider_panel)
        level_value_label.setObjectName("uepIsosurfaceMinorLabel")
        self._isosurface_level_input.setMinimumWidth(170)
        self._isosurface_level_input.setMinimumHeight(38)
        value_grid.addWidget(level_value_label, 0, 0)
        value_grid.addWidget(self._isosurface_level_input, 0, 1)
        value_grid.setColumnStretch(2, 1)
        slider_layout.addLayout(value_grid)
        controls_layout.addWidget(slider_panel)

        self._isosurface_status_card = QFrame(controls)
        self._isosurface_status_card.setObjectName("uepIsosurfaceStatusCard")
        status_layout = QVBoxLayout(self._isosurface_status_card)
        status_layout.setContentsMargins(16, 12, 16, 12)
        status_layout.setSpacing(6)
        status_title = QLabel("Status", self._isosurface_status_card)
        status_title.setObjectName("uepIsosurfaceMinorLabel")
        self._isosurface_status_label = QLabel(self._isosurface_status_card)
        self._isosurface_status_label.setObjectName("uepIsosurfaceStatusLabel")
        self._isosurface_status_label.setWordWrap(True)
        self._isosurface_status_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        self._isosurface_status_label.setMinimumHeight(0)
        status_layout.addWidget(status_title)
        status_layout.addWidget(self._isosurface_status_label)
        controls_layout.addWidget(self._isosurface_status_card)

        viewer_panel = QWidget(tab)
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        self._isosurface_viewer = AtomicStructureViewer(viewer_panel)
        self._isosurface_viewer.setMinimumHeight(460)
        self._isosurface_viewer.set_structure(self._viewer_host_structure)
        viewer_layout.addWidget(self._isosurface_viewer, 1)

        layout.addWidget(controls)
        layout.addWidget(viewer_panel, 1)
        self._set_isosurface_status(
            "Calculate the full-cell electrostatic-potential grid to enable isosurface viewing.",
            kind="info",
        )
        return tab

    def _build_batch_tab(self) -> QWidget:
        tab = QWidget(self)
        layout = QVBoxLayout(tab)

        controls = QGroupBox("Random-Start Batch Search", tab)
        form = QFormLayout(controls)
        self._batch_mode_count = QRadioButton("Number of starts", controls)
        self._batch_mode_spacing = QRadioButton("Muon-muon spacing", controls)
        self._batch_mode_count.setChecked(True)
        self._batch_mode_count.toggled.connect(self._on_batch_mode_toggled)
        self._batch_mode_spacing.toggled.connect(self._on_batch_mode_toggled)
        mode_row = QHBoxLayout()
        mode_row.addWidget(self._batch_mode_count)
        mode_row.addWidget(self._batch_mode_spacing)
        mode_row.addStretch(1)
        self._batch_count = QSpinBox(controls)
        self._batch_count.setRange(1, 200)
        self._batch_count.setValue(12)
        self._batch_min_muon_muon = DecimalLineEdit(controls)
        self._batch_min_muon_muon.setRange(0.0, 10.0)
        self._batch_min_muon_muon.setValue(0.8)
        self._batch_min_muon_atom = DecimalLineEdit(controls)
        self._batch_min_muon_atom.setRange(0.0, 10.0)
        self._batch_min_muon_atom.setValue(1.0)
        self._batch_max_attempts = QSpinBox(controls)
        self._batch_max_attempts.setRange(10, 500000)
        self._batch_max_attempts.setValue(5000)
        self._batch_cluster_threshold = DecimalLineEdit(controls)
        self._batch_cluster_threshold.setRange(0.01, 5.0)
        self._batch_cluster_threshold.setValue(0.3)
        self._batch_max_steps = QSpinBox(controls)
        self._batch_max_steps.setRange(1, 1000)
        self._batch_max_steps.setValue(80)
        self._batch_run = QPushButton("Run Batch Search", controls)
        self._batch_run.clicked.connect(self._run_batch_search)
        self._batch_promote = QPushButton("Promote Selected Cluster", controls)
        self._batch_promote.setEnabled(False)
        self._batch_promote.clicked.connect(self._promote_selected_cluster)
        button_row = QHBoxLayout()
        button_row.addWidget(self._batch_run)
        button_row.addWidget(self._batch_promote)
        button_row.addStretch(1)
        form.addRow("Sampling mode", mode_row)
        form.addRow("Random starts", self._batch_count)
        form.addRow("Min muon-muon distance (A)", self._batch_min_muon_muon)
        form.addRow("Min muon-atom distance (A)", self._batch_min_muon_atom)
        form.addRow("Max attempts", self._batch_max_attempts)
        form.addRow("Optimizer max steps", self._batch_max_steps)
        form.addRow("Cluster threshold (A)", self._batch_cluster_threshold)
        form.addRow("", button_row)

        results_panel = QWidget(tab)
        results_layout = QVBoxLayout(results_panel)
        results_layout.setContentsMargins(0, 0, 0, 0)
        self._batch_summary = QTextEdit(tab)
        self._batch_summary.setReadOnly(True)
        self._batch_table = QTableWidget(0, 7, tab)
        self._batch_table.setHorizontalHeaderLabels(
            ["Cluster", "Size", "Energy (eV)", "Class", "fX", "fY", "fZ"]
        )
        self._batch_table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._batch_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self._batch_table.itemSelectionChanged.connect(self._refresh_batch_promote_button)
        self._batch_table.itemSelectionChanged.connect(self._sync_batch_structure_viewer)
        results_layout.addWidget(self._batch_summary)
        results_layout.addWidget(self._batch_table, 1)

        viewer_panel = QWidget(tab)
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        self._batch_viewer_summary = QLabel("Showing the host structure. Select a cluster to overlay its muon positions.")
        self._batch_viewer_summary.setWordWrap(True)
        self._batch_viewer = AtomicStructureViewer(viewer_panel)
        self._batch_viewer.setMinimumWidth(560)
        self._batch_viewer.set_structure(self._viewer_host_structure)
        viewer_layout.addWidget(self._batch_viewer_summary)
        viewer_layout.addWidget(self._batch_viewer, 1)

        splitter = QSplitter(Qt.Orientation.Horizontal, tab)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(results_panel)
        splitter.addWidget(viewer_panel)
        splitter.setStretchFactor(0, 3)
        splitter.setStretchFactor(1, 5)
        splitter.setSizes([420, 760])

        layout.addWidget(controls)
        layout.addWidget(splitter, 1)
        self._update_batch_mode_controls()
        return tab

    def _build_viewer_host_structure(self) -> Structure:
        source_structure = self._source.structure
        properties: dict[str, Any] = {"pbc": [True, True, True]}
        if (
            source_structure.atomic_spins is not None
            and len(source_structure.atomic_spins) == len(source_structure.species)
        ):
            properties["atomic_spins"] = [float(value) for value in source_structure.atomic_spins]
        linked_source = self._source_structure
        if linked_source is not None and len(linked_source.species) == len(source_structure.species):
            for key in ("atomic_spins", "initial_magnetic_moments", "magnetic_structure", "propagation_vector"):
                if key == "atomic_spins" and "atomic_spins" in properties:
                    continue
                value = linked_source.properties.get(key)
                if isinstance(value, (list, dict)):
                    properties[key] = deepcopy(value)
        return Structure(
            label=f"{self._source_job_label} host",
            origin="uep_source",
            lattice_vectors=[list(vector) for vector in source_structure.lattice_vectors],
            species=list(source_structure.species),
            fractional_positions=[list(position) for position in source_structure.fractional_positions],
            properties=properties,
        )

    def _structure_for_batch_payload(self, payload: Mapping[str, Any] | None) -> Structure:
        if payload is None:
            return self._viewer_host_structure
        cluster_payload = payload.get("cluster")
        if not isinstance(cluster_payload, Mapping):
            return self._viewer_host_structure
        cluster = self._deserialize_cluster_summary(cluster_payload)
        if cluster is None:
            return self._viewer_host_structure

        fractional_positions = [list(position) for position in self._source.structure.fractional_positions]
        species = list(self._source.structure.species)
        for result_index in cluster.member_indexes:
            if result_index < 0 or result_index >= len(self._latest_batch_results):
                continue
            muon_position = self._latest_batch_results[result_index].optimized_fractional_position
            fractional_positions.append([float(muon_position[0]), float(muon_position[1]), float(muon_position[2])])
            species.append("X")

        return Structure(
            label=f"{self._source_job_label} cluster {cluster.cluster_index}",
            origin="uep_batch_cluster",
            lattice_vectors=[list(vector) for vector in self._source.structure.lattice_vectors],
            species=species,
            fractional_positions=fractional_positions,
            properties={"pbc": [True, True, True]},
        )

    def _sync_batch_structure_viewer(self) -> None:
        payload = self._selected_batch_payload()
        structure = self._structure_for_batch_payload(payload)
        self._batch_viewer.set_structure(structure)
        if payload is None:
            self._batch_viewer_summary.setText(
                "Showing the host structure. Select a cluster to overlay its muon positions."
            )
            return

        cluster_payload = payload.get("cluster")
        cluster = self._deserialize_cluster_summary(cluster_payload) if isinstance(cluster_payload, Mapping) else None
        if cluster is None:
            self._batch_viewer_summary.setText(
                "Showing the host structure. Select a cluster to overlay its muon positions."
            )
            return
        self._batch_viewer_summary.setText(
            f"Showing cluster {cluster.cluster_index} with {cluster.size} optimized muon positions overlaid on the host structure."
        )

    @classmethod
    def _deserialize_isosurface_mesh(cls, payload: object) -> UepIsosurfaceMesh | None:
        if not isinstance(payload, Mapping):
            return None
        vertices = cls._matrix_of_floats(payload.get("vertices_ang"))
        faces_raw = payload.get("faces")
        if vertices is None or not isinstance(faces_raw, (list, tuple)):
            return None
        faces: list[tuple[int, int, int]] = []
        for row in faces_raw:
            if not isinstance(row, (list, tuple)) or len(row) != 3:
                return None
            try:
                faces.append((int(row[0]), int(row[1]), int(row[2])))
            except (TypeError, ValueError):
                return None
        try:
            return UepIsosurfaceMesh(
                level_ev=float(payload.get("level_ev", 0.0)),
                minimum_potential_ev=float(payload.get("minimum_potential_ev", 0.0)),
                maximum_potential_ev=float(payload.get("maximum_potential_ev", 0.0)),
                vertices_ang=vertices,
                faces=tuple(faces),
            )
        except (TypeError, ValueError):
            return None

    def _apply_potential_grid_cache(self, grid: UepPotentialGrid, *, status_text: str | None = None) -> None:
        self._potential_grid_cache = grid
        self._isosurface_export_button.setEnabled(not self._is_isosurface_busy())
        bounds = self._isosurface_level_bounds()
        if bounds is not None:
            self._isosurface_level_input.setRange(bounds[0], bounds[1])
        self._isosurface_slider.setEnabled(self._current_isosurface_level() is not None)
        self._isosurface_level_input.setEnabled(self._current_isosurface_level() is not None)
        self._update_isosurface_bounds()
        self._update_isosurface_level_label()
        if status_text is not None:
            self._set_isosurface_status(status_text, kind="success")

    def _set_isosurface_busy(self, busy: bool, *, status_text: str | None = None) -> None:
        self._isosurface_compute_button.setEnabled(not busy)
        self._isosurface_export_button.setEnabled((not busy) and self._potential_grid_cache is not None)
        self._isosurface_slider.setEnabled((not busy) and self._current_isosurface_level() is not None)
        self._isosurface_level_input.setEnabled((not busy) and self._current_isosurface_level() is not None)
        if status_text is not None:
            self._set_isosurface_status(status_text, kind="busy" if busy else "info")

    def _is_isosurface_busy(self) -> bool:
        return self._isosurface_grid_thread is not None or self._isosurface_mesh_thread is not None

    def _current_isosurface_level(self) -> float | None:
        bounds = self._isosurface_level_bounds()
        if bounds is None:
            mesh = self._latest_isosurface_mesh
            return None if mesh is None else float(mesh.level_ev)
        editor_level = float(self._isosurface_level_input.value())
        return min(max(editor_level, bounds[0]), bounds[1])

    def _update_isosurface_level_label(self) -> None:
        return

    def _isosurface_level_bounds(self) -> tuple[float, float] | None:
        grid = self._potential_grid_cache
        if grid is None:
            return None
        minimum = float(np.min(grid.potential_total_ev))
        maximum = float(np.max(grid.potential_total_ev))
        return (minimum, maximum)

    def _update_isosurface_bounds(self) -> None:
        bounds = self._isosurface_level_bounds()
        if bounds is None and self._latest_isosurface_mesh is not None:
            minimum = float(self._latest_isosurface_mesh.minimum_potential_ev)
            maximum = float(self._latest_isosurface_mesh.maximum_potential_ev)
            bounds = (minimum, maximum)
        if bounds is None:
            self._isosurface_min_label.setText("Min")
            self._isosurface_max_label.setText("Max")
            return
        self._isosurface_min_label.setText(f"{bounds[0]:.2f} eV")
        self._isosurface_max_label.setText(f"{bounds[1]:.2f} eV")

    def _isosurface_mesh_color(self) -> str:
        return "#dc2626"

    def _set_isosurface_status(self, text: str, *, kind: str) -> None:
        normalized = str(kind).strip().lower()
        self._isosurface_status_label.setText(str(text))
        self._isosurface_status_kind = normalized
        self._isosurface_status_card.setProperty("statusKind", normalized)
        self._isosurface_status_label.setProperty("statusKind", normalized)
        for widget in (self._isosurface_status_card, self._isosurface_status_label):
            style = widget.style()
            style.unpolish(widget)
            style.polish(widget)
            widget.update()

    def _set_isosurface_level(self, level: float) -> None:
        bounds = self._isosurface_level_bounds()
        numeric = float(level)
        if bounds is not None:
            numeric = min(max(numeric, bounds[0]), bounds[1])
            fraction = 0.0 if np.isclose(bounds[0], bounds[1]) else (numeric - bounds[0]) / (bounds[1] - bounds[0])
            slider_value = int(round(fraction * float(self._isosurface_slider.maximum())))
            previous = self._isosurface_slider.blockSignals(True)
            self._isosurface_slider.setValue(slider_value)
            self._isosurface_slider.blockSignals(previous)
            self._isosurface_level_input.setRange(bounds[0], bounds[1])
        previous = self._isosurface_level_input.blockSignals(True)
        self._isosurface_level_input.setValue(numeric)
        self._isosurface_level_input.blockSignals(previous)
        self._update_isosurface_level_label()

    def _sync_isosurface_level_editor_from_slider(self) -> None:
        bounds = self._isosurface_level_bounds()
        if bounds is None:
            self._update_isosurface_level_label()
            return
        minimum, maximum = bounds
        fraction = float(self._isosurface_slider.value()) / float(max(1, self._isosurface_slider.maximum()))
        level = minimum + fraction * (maximum - minimum)
        previous = self._isosurface_level_input.blockSignals(True)
        self._isosurface_level_input.setValue(level)
        self._isosurface_level_input.blockSignals(previous)
        self._update_isosurface_level_label()

    def _request_isosurface_from_editor(self) -> None:
        if self._potential_grid_cache is None or self._is_isosurface_busy():
            return
        self._set_isosurface_level(self._isosurface_level_input.value())
        self._start_isosurface_mesh_calculation(self._isosurface_level_input.value())
        if not self._restoring_state:
            self._emit_state_changed()

    def _start_isosurface_grid_calculation(self) -> None:
        if self._is_isosurface_busy():
            return
        self._isosurface_viewer.set_structure(self._viewer_host_structure)
        self._isosurface_viewer.set_isosurface(None, None)
        self._set_isosurface_busy(True, status_text="Calculating the full-cell electrostatic-potential grid...")

        thread = _IsosurfaceGridThread(self._field)
        thread.setParent(self)
        thread.grid_ready.connect(self._on_isosurface_grid_ready)
        thread.failed.connect(self._on_isosurface_grid_failed)
        thread.finished.connect(self._on_isosurface_grid_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._isosurface_grid_thread = thread
        thread.start()

    def _on_isosurface_grid_thread_finished(self) -> None:
        self._isosurface_grid_thread = None

    def _on_isosurface_mesh_thread_finished(self) -> None:
        self._isosurface_mesh_thread = None

    def _on_isosurface_grid_ready(self, grid: object) -> None:
        if not isinstance(grid, UepPotentialGrid):
            self._on_isosurface_grid_failed("Unexpected isosurface grid payload.")
            return
        self._apply_potential_grid_cache(grid, status_text="Cached full-cell potential grid is ready.")
        self.potential_grid_ready.emit(serialize_potential_grid(grid))
        bounds = self._isosurface_level_bounds()
        if self._latest_isosurface_mesh is None and bounds is not None:
            level = 0.5 * (bounds[0] + bounds[1])
        else:
            level = self._current_isosurface_level()
        if level is None and bounds is not None:
            level = 0.5 * (bounds[0] + bounds[1])
        if level is not None:
            self._set_isosurface_level(level)
            self._start_isosurface_mesh_calculation(level)
        else:
            self._set_isosurface_busy(False)
            self._set_isosurface_status(
                "The cached potential grid does not contain a usable isosurface level.",
                kind="info",
            )

    def _on_isosurface_grid_failed(self, message: str) -> None:
        self._set_isosurface_busy(False)
        self._set_isosurface_status(f"Isosurface calculation failed: {message}", kind="error")

    def _start_isosurface_mesh_calculation(self, level_ev: float) -> None:
        if self._potential_grid_cache is None:
            return
        if self._isosurface_mesh_thread is not None:
            return
        self._isosurface_request_id += 1
        request_id = self._isosurface_request_id
        self._set_isosurface_busy(True, status_text=f"Extracting isosurface at {level_ev:.6f} eV...")
        thread = _IsosurfaceMeshThread(self._potential_grid_cache, level_ev=level_ev, request_id=request_id)
        thread.setParent(self)
        thread.mesh_ready.connect(self._on_isosurface_mesh_ready)
        thread.failed.connect(self._on_isosurface_mesh_failed)
        thread.finished.connect(self._on_isosurface_mesh_thread_finished)
        thread.finished.connect(thread.deleteLater)
        self._isosurface_mesh_thread = thread
        thread.start()

    def _request_isosurface_from_slider(self) -> None:
        if self._potential_grid_cache is None or self._is_isosurface_busy():
            return
        level = self._current_isosurface_level()
        if level is None:
            return
        self._set_isosurface_level(level)
        self._start_isosurface_mesh_calculation(level)

    def _export_potential_grid_xsf(self) -> None:
        grid = self._potential_grid_cache
        if grid is None:
            QMessageBox.information(self, "UEP Isosurface", "Calculate the full-cell potential grid before exporting XSF.")
            return
        default_name = f"{self._source.den_fmt_path.stem}_potential.xsf"
        default_path = self._source.den_fmt_path.with_name(default_name)
        selected_path, _selected_filter = QFileDialog.getSaveFileName(
            self,
            "Export UEP Potential Grid",
            str(default_path),
            "XSF Files (*.xsf)",
        )
        if not selected_path:
            return
        try:
            exported_path = write_potential_grid_xsf(
                selected_path,
                structure=self._source.structure,
                potential=grid,
            )
        except (OSError, UepError) as exc:
            QMessageBox.critical(self, "UEP Isosurface", f"Failed to export XSF: {exc}")
            return
        self._set_isosurface_status(f"Cached full-cell potential grid exported to {exported_path.name}.", kind="exported")
        self._emit_state_changed()

    def _render_isosurface_mesh(self, mesh: UepIsosurfaceMesh) -> None:
        self._latest_isosurface_mesh = mesh
        self._set_isosurface_level(mesh.level_ev)
        self._isosurface_viewer.set_structure(self._viewer_host_structure)
        self._isosurface_viewer.set_isosurface(
            mesh.vertices_ang,
            mesh.faces,
            color=self._isosurface_mesh_color(),
            alpha=0.3,
        )
        self._update_isosurface_bounds()
        self._set_isosurface_status(
            f"Cached full-cell potential grid is ready. Displaying the filled isosurface with {len(mesh.faces)} triangles at {mesh.level_ev:.6f} eV.",
            kind="success",
        )

    def _on_isosurface_mesh_ready(self, request_id: int, mesh: object) -> None:
        if int(request_id) != self._isosurface_request_id or not isinstance(mesh, UepIsosurfaceMesh):
            return
        self._render_isosurface_mesh(mesh)
        self._set_isosurface_busy(False)
        self._emit_state_changed()

    def _on_isosurface_mesh_failed(self, request_id: int, message: str) -> None:
        if int(request_id) != self._isosurface_request_id:
            return
        self._set_isosurface_busy(False)
        self._set_isosurface_status(f"Isosurface extraction failed: {message}", kind="error")
        self._emit_state_changed()

    def _batch_mode(self) -> str:
        return "spacing" if self._batch_mode_spacing.isChecked() else "count"

    def _set_batch_mode(self, mode: str) -> None:
        normalized = "spacing" if str(mode).strip().lower() == "spacing" else "count"
        self._batch_mode_spacing.setChecked(normalized == "spacing")
        self._batch_mode_count.setChecked(normalized == "count")
        self._update_batch_mode_controls()

    def _update_batch_mode_controls(self) -> None:
        count_mode = self._batch_mode() == "count"
        self._batch_count.setEnabled(count_mode)
        self._batch_min_muon_muon.setEnabled(not count_mode)

    def _on_batch_mode_toggled(self, _checked: bool) -> None:
        self._update_batch_mode_controls()
        self._emit_state_changed()

    def _populate_summary(self) -> None:
        pseudo_species = sorted({entry.species for entry in self._source.pseudopotentials})
        lines = [
            f"Source job: {self._source_job_label}",
            f"Density file: {self._source.den_fmt_path}",
            f"CASTEP file: {self._source.castep_path}",
            f"CELL fallback: {self._source.cell_path if self._source.cell_path is not None else 'not used'}",
            f"Atoms in source structure: {len(self._source.structure.species)}",
            f"Pseudopotential species recovered: {', '.join(pseudo_species)}",
            f"Gaussian width factor (gw_factor): {self._source.gw_factor:.3f}",
            (
                "Mulliken spin moments recovered from CASTEP output: yes"
                if self._source.structure.atomic_spins is not None
                else "Mulliken spin moments recovered from CASTEP output: no"
            ),
        ]
        if self._source.neutrality_warning:
            lines.append(f"Neutrality note: {self._source.neutrality_warning}")
        lines.append("Required UEP source files are present: `.den_fmt` and `.castep`.")
        self._summary_text.setPlainText("\n".join(lines))

    def _fractional_to_cartesian(self, fractional: tuple[float, float, float]) -> tuple[float, float, float]:
        cart = lattice_frac_to_cart(fractional, self._source.structure.lattice_vectors)
        return (float(cart[0]), float(cart[1]), float(cart[2]))

    @staticmethod
    def _tuple3(values: object) -> tuple[float, float, float] | None:
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            return None
        try:
            return (float(values[0]), float(values[1]), float(values[2]))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _matrix3(values: object) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] | None:
        if not isinstance(values, (list, tuple)) or len(values) != 3:
            return None
        rows: list[tuple[float, float, float]] = []
        for row in values:
            parsed = UepAnalysisDialog._tuple3(row)
            if parsed is None:
                return None
            rows.append(parsed)
        return (rows[0], rows[1], rows[2])

    @classmethod
    def _deserialize_classification(cls, payload: object) -> Any | None:
        if not isinstance(payload, Mapping):
            return None
        eigenvalues = cls._tuple3(payload.get("eigenvalues"))
        if eigenvalues is None:
            return None
        try:
            return StationaryPointClassification(
                label=str(payload.get("label", "")).strip(),
                eigenvalues=eigenvalues,
                positive_count=int(payload.get("positive_count", 0)),
                negative_count=int(payload.get("negative_count", 0)),
                near_zero_count=int(payload.get("near_zero_count", 0)),
            )
        except (TypeError, ValueError):
            return None

    @classmethod
    def _deserialize_probe(cls, payload: object) -> UepProbeResult | None:
        if not isinstance(payload, Mapping):
            return None
        position = cls._tuple3(payload.get("position_ang"))
        fractional = cls._tuple3(payload.get("fractional_position"))
        gradient_total = cls._tuple3(payload.get("gradient_total_ev_per_ang"))
        gradient_electronic = cls._tuple3(payload.get("gradient_electronic_ev_per_ang"))
        gradient_ionic = cls._tuple3(payload.get("gradient_ionic_ev_per_ang"))
        hessian_total = cls._matrix3(payload.get("hessian_total_ev_per_ang2"))
        hessian_electronic = cls._matrix3(payload.get("hessian_electronic_ev_per_ang2"))
        hessian_ionic = cls._matrix3(payload.get("hessian_ionic_ev_per_ang2"))
        classification = cls._deserialize_classification(payload.get("classification"))
        if None in (
            position,
            fractional,
            gradient_total,
            gradient_electronic,
            gradient_ionic,
            hessian_total,
            hessian_electronic,
            hessian_ionic,
            classification,
        ):
            return None
        try:
            return UepProbeResult(
                position_ang=position,
                fractional_position=fractional,
                potential_total_ev=float(payload.get("potential_total_ev", 0.0)),
                potential_electronic_ev=float(payload.get("potential_electronic_ev", 0.0)),
                potential_ionic_ev=float(payload.get("potential_ionic_ev", 0.0)),
                gradient_total_ev_per_ang=gradient_total,
                gradient_electronic_ev_per_ang=gradient_electronic,
                gradient_ionic_ev_per_ang=gradient_ionic,
                hessian_total_ev_per_ang2=hessian_total,
                hessian_electronic_ev_per_ang2=hessian_electronic,
                hessian_ionic_ev_per_ang2=hessian_ionic,
                classification=classification,
                zero_point_energy_ev=(
                    None if payload.get("zero_point_energy_ev") is None else float(payload.get("zero_point_energy_ev"))
                ),
                total_quantum_energy_ev=(
                    None
                    if payload.get("total_quantum_energy_ev") is None
                    else float(payload.get("total_quantum_energy_ev"))
                ),
            )
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _tuple_of_floats(values: object) -> tuple[float, ...] | None:
        if not isinstance(values, (list, tuple)):
            return None
        try:
            return tuple(float(value) for value in values)
        except (TypeError, ValueError):
            return None

    @classmethod
    def _matrix_of_floats(cls, values: object) -> tuple[tuple[float, ...], ...] | None:
        if not isinstance(values, (list, tuple)):
            return None
        rows: list[tuple[float, ...]] = []
        for row in values:
            parsed = cls._tuple_of_floats(row)
            if parsed is None:
                return None
            rows.append(parsed)
        return tuple(rows)

    @classmethod
    def _deserialize_line_sample(cls, payload: object):
        if not isinstance(payload, Mapping):
            return None
        fields = (
            cls._tuple_of_floats(payload.get("distance_ang")),
            cls._tuple_of_floats(payload.get("potential_total_ev")),
            cls._tuple_of_floats(payload.get("potential_electronic_ev")),
            cls._tuple_of_floats(payload.get("potential_ionic_ev")),
            cls._tuple_of_floats(payload.get("charge_total_e_per_ang3")),
            cls._tuple_of_floats(payload.get("charge_electronic_e_per_ang3")),
            cls._tuple_of_floats(payload.get("charge_ionic_e_per_ang3")),
        )
        if any(field is None for field in fields):
            return None
        return UepLineSample(
            distance_ang=fields[0],
            potential_total_ev=fields[1],
            potential_electronic_ev=fields[2],
            potential_ionic_ev=fields[3],
            charge_total_e_per_ang3=fields[4],
            charge_electronic_e_per_ang3=fields[5],
            charge_ionic_e_per_ang3=fields[6],
        )

    @classmethod
    def _deserialize_plane_sample(cls, payload: object):
        if not isinstance(payload, Mapping):
            return None
        fields = (
            cls._matrix_of_floats(payload.get("x_coordinates_ang")),
            cls._matrix_of_floats(payload.get("y_coordinates_ang")),
            cls._matrix_of_floats(payload.get("potential_total_ev")),
            cls._matrix_of_floats(payload.get("potential_electronic_ev")),
            cls._matrix_of_floats(payload.get("potential_ionic_ev")),
            cls._matrix_of_floats(payload.get("charge_total_e_per_ang3")),
            cls._matrix_of_floats(payload.get("charge_electronic_e_per_ang3")),
            cls._matrix_of_floats(payload.get("charge_ionic_e_per_ang3")),
        )
        if any(field is None for field in fields):
            return None
        return UepPlaneSample(
            x_coordinates_ang=fields[0],
            y_coordinates_ang=fields[1],
            potential_total_ev=fields[2],
            potential_electronic_ev=fields[3],
            potential_ionic_ev=fields[4],
            charge_total_e_per_ang3=fields[5],
            charge_electronic_e_per_ang3=fields[6],
            charge_ionic_e_per_ang3=fields[7],
        )

    @classmethod
    def _deserialize_optimization_result(cls, payload: object) -> UepOptimizationResult | None:
        if not isinstance(payload, Mapping):
            return None
        start = cls._tuple3(payload.get("start_position_ang"))
        optimized = cls._tuple3(payload.get("optimized_position_ang"))
        fractional = cls._tuple3(payload.get("optimized_fractional_position"))
        classification = cls._deserialize_classification(payload.get("classification"))
        if None in (start, optimized, fractional, classification):
            return None
        try:
            return UepOptimizationResult(
                start_position_ang=start,
                optimized_position_ang=optimized,
                optimized_fractional_position=fractional,
                classical_energy_ev=float(payload.get("classical_energy_ev", 0.0)),
                zero_point_energy_ev=(
                    None if payload.get("zero_point_energy_ev") is None else float(payload.get("zero_point_energy_ev"))
                ),
                total_energy_ev=None if payload.get("total_energy_ev") is None else float(payload.get("total_energy_ev")),
                success=bool(payload.get("success", False)),
                message=str(payload.get("message", "")),
                iterations=int(payload.get("iterations", 0)),
                classification=classification,
            )
        except (TypeError, ValueError):
            return None

    @classmethod
    def _deserialize_cluster_summary(cls, payload: object) -> Any | None:
        if not isinstance(payload, Mapping):
            return None
        position = cls._tuple3(payload.get("representative_position_ang"))
        fractional = cls._tuple3(payload.get("representative_fractional_position"))
        members = payload.get("member_indexes")
        if position is None or fractional is None or not isinstance(members, (list, tuple)):
            return None
        try:
            return UepClusterSummary(
                cluster_index=int(payload.get("cluster_index", 0)),
                size=int(payload.get("size", 0)),
                representative_index=int(payload.get("representative_index", 0)),
                representative_position_ang=position,
                representative_fractional_position=fractional,
                minimum_energy_ev=(
                    None if payload.get("minimum_energy_ev") is None else float(payload.get("minimum_energy_ev"))
                ),
                average_energy_ev=(
                    None if payload.get("average_energy_ev") is None else float(payload.get("average_energy_ev"))
                ),
                standard_deviation_ev=(
                    None if payload.get("standard_deviation_ev") is None else float(payload.get("standard_deviation_ev"))
                ),
                member_indexes=tuple(int(value) for value in members),
            )
        except (TypeError, ValueError):
            return None

    @classmethod
    def _deserialize_batch_summary(cls, payload: object) -> UepBatchSummary | None:
        if not isinstance(payload, Mapping):
            return None
        raw_clusters = payload.get("clusters")
        if not isinstance(raw_clusters, (list, tuple)):
            return None
        clusters = [cls._deserialize_cluster_summary(entry) for entry in raw_clusters]
        if any(cluster is None for cluster in clusters):
            return None
        try:
            return UepBatchSummary(
                threshold_ang=float(payload.get("threshold_ang", 0.0)),
                clusters=tuple(clusters),
            )
        except (TypeError, ValueError):
            return None

    def _render_probe_result(self, probe: UepProbeResult) -> None:
        self._latest_probe = probe
        self._probe_promote.setEnabled(True)
        self._probe_output.setPlainText(
            "\n".join(
                [
                    f"Cartesian position (A): {probe.position_ang}",
                    f"Fractional position: {probe.fractional_position}",
                    f"Total potential (eV): {probe.potential_total_ev:.8f}",
                    f"Electronic contribution (eV): {probe.potential_electronic_ev:.8f}",
                    f"Ionic contribution (eV): {probe.potential_ionic_ev:.8f}",
                    f"Gradient (eV/A): {probe.gradient_total_ev_per_ang}",
                    f"Hessian (eV/A^2): {probe.hessian_total_ev_per_ang2}",
                    f"Classification: {probe.classification.label}",
                    f"Hessian eigenvalues: {probe.classification.eigenvalues}",
                    "Zero-point energy (eV): "
                    + ("n/a" if probe.zero_point_energy_ev is None else f"{probe.zero_point_energy_ev:.8f}"),
                    "Quantum-corrected total (eV): "
                    + (
                        "n/a"
                        if probe.total_quantum_energy_ev is None
                        else f"{probe.total_quantum_energy_ev:.8f}"
                    ),
                ]
            )
        )

    def _render_line_sample(self, sample) -> None:
        self._latest_line_sample = sample
        self._line_figure.clear()
        axis = self._line_figure.add_subplot(1, 1, 1)
        axis.plot(sample.distance_ang, sample.potential_total_ev, label="Total", color="#1d4ed8")
        axis.plot(sample.distance_ang, sample.potential_electronic_ev, label="Electronic", color="#0f766e")
        axis.plot(sample.distance_ang, sample.potential_ionic_ev, label="Ionic", color="#b91c1c")
        axis.set_xlabel("Distance along line (A)")
        axis.set_ylabel("Potential (eV)")
        axis.legend()
        axis.grid(True, alpha=0.3)
        self._line_canvas.draw_idle()

        rows = ["distance_A,total_eV,electronic_eV,ionic_eV"]
        for index in range(min(25, len(sample.distance_ang))):
            rows.append(
                f"{sample.distance_ang[index]:.6f},{sample.potential_total_ev[index]:.8f},"
                f"{sample.potential_electronic_ev[index]:.8f},{sample.potential_ionic_ev[index]:.8f}"
            )
        self._line_data.setPlainText("\n".join(rows))

    def _render_plane_sample(self, sample: UepPlaneSample) -> None:
        self._latest_plane_sample = sample
        self._draw_plane(sample)
        grid = np.array(sample.potential_total_ev, dtype=float)
        self._plane_data.setPlainText(
            "\n".join(
                [
                    f"Grid shape: {grid.shape[0]} x {grid.shape[1]}",
                    f"Minimum potential (eV): {float(np.min(grid)):.8f}",
                    f"Maximum potential (eV): {float(np.max(grid)):.8f}",
                    f"Mean potential (eV): {float(np.mean(grid)):.8f}",
                ]
            )
        )

    def _populate_batch_results(
        self,
        results: list[UepOptimizationResult],
        summary: UepBatchSummary,
        *,
        summary_text: str | None = None,
        selected_row: int | None = None,
    ) -> None:
        self._latest_batch_results = list(results)
        self._latest_batch_summary = summary
        self._batch_table.setRowCount(0)
        for cluster in summary.clusters:
            representative = results[cluster.representative_index]
            row = self._batch_table.rowCount()
            self._batch_table.insertRow(row)
            payload = {
                "cluster": asdict(cluster),
                "representative": asdict(representative),
            }
            items = [
                QTableWidgetItem(str(cluster.cluster_index)),
                QTableWidgetItem(str(cluster.size)),
                QTableWidgetItem("" if cluster.minimum_energy_ev is None else f"{cluster.minimum_energy_ev:.6f}"),
                QTableWidgetItem(representative.classification.label),
                QTableWidgetItem(f"{representative.optimized_fractional_position[0]:.6f}"),
                QTableWidgetItem(f"{representative.optimized_fractional_position[1]:.6f}"),
                QTableWidgetItem(f"{representative.optimized_fractional_position[2]:.6f}"),
            ]
            items[0].setData(Qt.ItemDataRole.UserRole, payload)
            for column, item in enumerate(items):
                self._batch_table.setItem(row, column, item)
        if summary_text is not None:
            self._batch_summary.setPlainText(summary_text)
        if selected_row is not None and 0 <= selected_row < self._batch_table.rowCount():
            self._batch_table.selectRow(selected_row)
        else:
            self._batch_viewer.set_structure(self._viewer_host_structure)
            self._batch_viewer_summary.setText(
                "Showing the host structure. Select a cluster to overlay its muon positions."
            )
        self._refresh_batch_promote_button()

    def state_snapshot(self) -> dict[str, object]:
        selected = self._selected_batch_payload()
        selected_row = None
        if selected is not None:
            current = self._batch_table.currentRow()
            selected_row = current if current >= 0 else None
        return {
            "current_tab": int(self._tabs.currentIndex()),
            "probe_point": list(self._probe_point.values()),
            "latest_probe": None if self._latest_probe is None else asdict(self._latest_probe),
            "probe_output": self._probe_output.toPlainText(),
            "line_start": list(self._line_start.values()),
            "line_end": list(self._line_end.values()),
            "line_points": int(self._line_points.value()),
            "line_sample": None if self._latest_line_sample is None else asdict(self._latest_line_sample),
            "line_data": self._line_data.toPlainText(),
            "plane_origin": list(self._plane_origin.values()),
            "plane_x": list(self._plane_x.values()),
            "plane_y": list(self._plane_y.values()),
            "plane_points_x": int(self._plane_points_x.value()),
            "plane_points_y": int(self._plane_points_y.value()),
            "plane_sample": None if self._latest_plane_sample is None else asdict(self._latest_plane_sample),
            "plane_data": self._plane_data.toPlainText(),
            "isosurface_slider_value": int(self._isosurface_slider.value()),
            "isosurface_level_value": float(self._isosurface_level_input.value()),
            "isosurface_status_text": self._isosurface_status_label.text(),
            "isosurface_status_kind": self._isosurface_status_kind,
            "isosurface_mesh": (
                None
                if self._latest_isosurface_mesh is None
                else {
                    "level_ev": float(self._latest_isosurface_mesh.level_ev),
                    "minimum_potential_ev": float(self._latest_isosurface_mesh.minimum_potential_ev),
                    "maximum_potential_ev": float(self._latest_isosurface_mesh.maximum_potential_ev),
                    "vertices_ang": [list(vertex) for vertex in self._latest_isosurface_mesh.vertices_ang],
                    "faces": [list(face) for face in self._latest_isosurface_mesh.faces],
                }
            ),
            "batch_mode": self._batch_mode(),
            "batch_count": int(self._batch_count.value()),
            "batch_min_muon_muon": float(self._batch_min_muon_muon.value()),
            "batch_min_muon_atom": float(self._batch_min_muon_atom.value()),
            "batch_max_attempts": int(self._batch_max_attempts.value()),
            "batch_cluster_threshold": float(self._batch_cluster_threshold.value()),
            "batch_max_steps": int(self._batch_max_steps.value()),
            "batch_results": [asdict(result) for result in self._latest_batch_results],
            "batch_summary_data": None if self._latest_batch_summary is None else asdict(self._latest_batch_summary),
            "batch_summary_text": self._batch_summary.toPlainText(),
            "batch_selected_row": selected_row,
        }

    def _emit_state_changed(self) -> None:
        self.state_changed.emit(self.state_snapshot())

    def _restore_initial_state(self, state: Mapping[str, object] | None) -> None:
        if not isinstance(state, Mapping):
            return

        self._restoring_state = True
        try:
            self._set_batch_mode(str(state.get("batch_mode", "count")))

            for control, key in (
                (self._probe_point, "probe_point"),
                (self._line_start, "line_start"),
                (self._line_end, "line_end"),
                (self._plane_origin, "plane_origin"),
                (self._plane_x, "plane_x"),
                (self._plane_y, "plane_y"),
            ):
                values = self._tuple3(state.get(key))
                if values is not None:
                    control.set_values(values)

            for control, key in (
                (self._line_points, "line_points"),
                (self._plane_points_x, "plane_points_x"),
                (self._plane_points_y, "plane_points_y"),
                (self._isosurface_slider, "isosurface_slider_value"),
                (self._batch_count, "batch_count"),
                (self._batch_max_attempts, "batch_max_attempts"),
                (self._batch_max_steps, "batch_max_steps"),
            ):
                raw = state.get(key)
                if raw is not None:
                    try:
                        control.setValue(int(raw))
                    except (TypeError, ValueError):
                        pass

            for control, key in (
                (self._isosurface_level_input, "isosurface_level_value"),
                (self._batch_min_muon_muon, "batch_min_muon_muon"),
                (self._batch_min_muon_atom, "batch_min_muon_atom"),
                (self._batch_cluster_threshold, "batch_cluster_threshold"),
            ):
                raw = state.get(key)
                if raw is not None:
                    try:
                        control.setValue(float(raw))
                    except (TypeError, ValueError):
                        pass

            probe = self._deserialize_probe(state.get("latest_probe"))
            if probe is not None:
                self._render_probe_result(probe)
            else:
                text = state.get("probe_output")
                if isinstance(text, str):
                    self._probe_output.setPlainText(text)

            line_sample = self._deserialize_line_sample(state.get("line_sample"))
            if line_sample is not None:
                self._render_line_sample(line_sample)
            else:
                text = state.get("line_data")
                if isinstance(text, str):
                    self._line_data.setPlainText(text)

            plane_sample = self._deserialize_plane_sample(state.get("plane_sample"))
            if plane_sample is not None:
                self._render_plane_sample(plane_sample)
            else:
                text = state.get("plane_data")
                if isinstance(text, str):
                    self._plane_data.setPlainText(text)

            if self._potential_grid_cache is None:
                restored_grid = deserialize_potential_grid(state.get("potential_grid"))
                if restored_grid is not None:
                    self._apply_potential_grid_cache(
                        restored_grid,
                        status_text="Cached full-cell potential grid is ready.",
                    )

            isosurface_mesh = self._deserialize_isosurface_mesh(state.get("isosurface_mesh"))
            if isosurface_mesh is not None:
                self._render_isosurface_mesh(isosurface_mesh)
                self._isosurface_slider.setEnabled(False)
            else:
                text = state.get("isosurface_status_text")
                if isinstance(text, str):
                    self._set_isosurface_status(
                        text,
                        kind=str(state.get("isosurface_status_kind", "info")),
                    )
            self._update_isosurface_level_label()

            raw_results = state.get("batch_results")
            batch_results = (
                [result for entry in raw_results if (result := self._deserialize_optimization_result(entry)) is not None]
                if isinstance(raw_results, list)
                else []
            )
            batch_summary = self._deserialize_batch_summary(state.get("batch_summary_data"))
            if batch_results and batch_summary is not None:
                selected_row = state.get("batch_selected_row")
                selected_index = int(selected_row) if isinstance(selected_row, int) else None
                summary_text = state.get("batch_summary_text")
                self._populate_batch_results(
                    batch_results,
                    batch_summary,
                    summary_text=str(summary_text) if isinstance(summary_text, str) else None,
                    selected_row=selected_index,
                )
            else:
                text = state.get("batch_summary_text")
                if isinstance(text, str):
                    self._batch_summary.setPlainText(text)

            current_tab = state.get("current_tab")
            if isinstance(current_tab, int) and 0 <= current_tab < self._tabs.count():
                self._tabs.setCurrentIndex(current_tab)
        finally:
            self._restoring_state = False

    def _run_probe(self) -> None:
        try:
            fractional = self._probe_point.values()
            probe = probe_point(self._field, self._fractional_to_cartesian(fractional))
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "UEP probe failed", str(exc))
            return

        self._render_probe_result(probe)
        self._emit_state_changed()

    def _run_line_sample(self) -> None:
        try:
            sample = sample_line(
                self._field,
                start_ang=self._fractional_to_cartesian(self._line_start.values()),
                end_ang=self._fractional_to_cartesian(self._line_end.values()),
                num_points=self._line_points.value(),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "UEP line sampling failed", str(exc))
            return

        self._render_line_sample(sample)
        self._emit_state_changed()

    def _draw_plane(self, sample: UepPlaneSample) -> None:
        self._plane_figure.clear()
        axis = self._plane_figure.add_subplot(1, 1, 1)
        image = axis.imshow(
            np.array(sample.potential_total_ev, dtype=float).T,
            origin="lower",
            aspect="auto",
            extent=(
                float(np.min(sample.x_coordinates_ang)),
                float(np.max(sample.x_coordinates_ang)),
                float(np.min(sample.y_coordinates_ang)),
                float(np.max(sample.y_coordinates_ang)),
            ),
            cmap="viridis",
        )
        axis.set_xlabel("Plane X (A)")
        axis.set_ylabel("Plane Y (A)")
        axis.set_title("Total UEP (eV)")
        self._plane_figure.colorbar(image, ax=axis, label="Potential (eV)")
        self._plane_canvas.draw_idle()

    def _run_plane_sample(self) -> None:
        try:
            sample = sample_plane(
                self._field,
                origin_ang=self._fractional_to_cartesian(self._plane_origin.values()),
                corner_x_ang=self._fractional_to_cartesian(self._plane_x.values()),
                corner_y_ang=self._fractional_to_cartesian(self._plane_y.values()),
                points_x=self._plane_points_x.value(),
                points_y=self._plane_points_y.value(),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "UEP plane sampling failed", str(exc))
            return

        self._render_plane_sample(sample)
        self._emit_state_changed()

    def _run_batch_search(self) -> None:
        try:
            if self._batch_mode() == "spacing":
                starts, attempts_used = generate_random_sites_by_distance(
                    lattice_vectors=self._source.structure.lattice_vectors,
                    host_positions_frac=self._source.structure.fractional_positions,
                    min_muon_muon=self._batch_min_muon_muon.value(),
                    min_muon_atom=self._batch_min_muon_atom.value(),
                    max_attempts=self._batch_max_attempts.value(),
                )
                generation_lines = [
                    f"Spacing-constrained starts generated: {len(starts)}",
                    f"Min muon-muon distance (A): {self._batch_min_muon_muon.value():.3f}",
                ]
            else:
                starts, attempts_used = generate_random_sites(
                    lattice_vectors=self._source.structure.lattice_vectors,
                    host_positions_frac=self._source.structure.fractional_positions,
                    count=self._batch_count.value(),
                    min_muon_muon=0.0,
                    min_muon_atom=self._batch_min_muon_atom.value(),
                    max_attempts=self._batch_max_attempts.value(),
                )
                generation_lines = [f"Random starts generated: {len(starts)}"]
            results = [
                optimize_muon_position(
                    self._field,
                    start_position_ang=tuple(point.cartesian),
                    max_steps=self._batch_max_steps.value(),
                )
                for point in starts
            ]
            summary = cluster_optimized_positions(
                results,
                threshold_ang=self._batch_cluster_threshold.value(),
            )
        except Exception as exc:  # noqa: BLE001
            QMessageBox.warning(self, "UEP batch search failed", str(exc))
            return

        self._populate_batch_results(
            results,
            summary,
            summary_text="\n".join(
                generation_lines
                + [
                    f"Generation attempts used: {attempts_used}",
                    f"Clusters found: {len(summary.clusters)}",
                    f"Cluster threshold (A): {summary.threshold_ang:.3f}",
                ]
            ),
        )
        self._emit_state_changed()

    def closeEvent(self, event) -> None:
        for thread in (self._isosurface_grid_thread, self._isosurface_mesh_thread):
            if thread is None:
                continue
            thread.quit()
            thread.wait(100)
        super().closeEvent(event)

    def _refresh_batch_promote_button(self) -> None:
        self._batch_promote.setEnabled(bool(self._selected_batch_payload()))

    def _selected_batch_payload(self) -> dict[str, Any] | None:
        selected = self._batch_table.selectedItems()
        if not selected:
            return None
        item = self._batch_table.item(selected[0].row(), 0)
        payload = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
        return payload if isinstance(payload, dict) else None

    def _emit_promote_payload(
        self,
        *,
        label_hint: str,
        fractional_position: tuple[float, float, float],
        cartesian_position: tuple[float, float, float],
        generation_method: str,
        properties: dict[str, Any],
    ) -> None:
        self.promote_requested.emit(
            {
                "label_hint": label_hint,
                "fractional_position": list(fractional_position),
                "cartesian_position": list(cartesian_position),
                "generation_method": generation_method,
                "source_structure_id": str(self._source_structure_id),
                "source_job_label": self._source_job_label,
                "properties": properties,
            }
        )

    def _promote_current_probe(self) -> None:
        if self._latest_probe is None:
            return
        self._emit_promote_payload(
            label_hint=f"UEP Probe {self._source_job_label}",
            fractional_position=self._latest_probe.fractional_position,
            cartesian_position=self._latest_probe.position_ang,
            generation_method="uep_probe",
            properties={
                "classification": self._latest_probe.classification.label,
                "total_potential_ev": self._latest_probe.potential_total_ev,
                "zero_point_energy_ev": self._latest_probe.zero_point_energy_ev,
                "total_quantum_energy_ev": self._latest_probe.total_quantum_energy_ev,
            },
        )

    def _promote_selected_cluster(self) -> None:
        payload = self._selected_batch_payload()
        if payload is None:
            return
        representative = payload.get("representative", {})
        try:
            fractional = tuple(float(value) for value in representative["optimized_fractional_position"])
            cartesian = tuple(float(value) for value in representative["optimized_position_ang"])
        except Exception:  # noqa: BLE001
            QMessageBox.warning(self, "Promotion failed", "Selected cluster data is incomplete.")
            return

        self._emit_promote_payload(
            label_hint=f"UEP Cluster {payload.get('cluster', {}).get('cluster_index', '?')}",
            fractional_position=fractional,
            cartesian_position=cartesian,
            generation_method="uep_batch_cluster",
            properties={
                "cluster_index": payload.get("cluster", {}).get("cluster_index"),
                "cluster_size": payload.get("cluster", {}).get("size"),
                "classification": representative.get("classification", {}).get("label"),
                "classical_energy_ev": representative.get("classical_energy_ev"),
                "zero_point_energy_ev": representative.get("zero_point_energy_ev"),
                "total_energy_ev": representative.get("total_energy_ev"),
            },
        )
