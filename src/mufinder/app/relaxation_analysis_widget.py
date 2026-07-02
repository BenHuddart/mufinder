"""Workspace UI for analysing completed structural relaxations."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from uuid import UUID, uuid4

from ase.data import atomic_numbers
from ase.data.colors import jmol_colors
from matplotlib import colormaps, colors as mcolors
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
import numpy as np
from PySide6.QtCore import Qt, QTimer, Signal
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSizePolicy,
    QSpinBox,
    QSplitter,
    QTableWidget,
    QTableWidgetItem,
    QTabWidget,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.atomic_structure_viewer import (
    AtomicStructureViewer,
    MUON_VIEWER_RADIUS_ANG,
    viewer_atom_style_for_species,
)
from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.app.theme import (
    current_palette,
    status_brush,
    style_hint_text,
    style_section_subtitle,
    style_section_title,
    use_tabbed_table_corners,
)
from mufinder.app.uep_clustering_dialog import _DendrogramNavigationToolbar
from mufinder.domain.models import Structure
from mufinder.science.relaxation_analysis import (
    AnchoredRelaxedSite,
    ConnectedComponentsResult,
    DisplacementAnalysisResult,
    RelaxationAnalysisError,
    RelaxationClusterSummary,
    RelaxedMuonSite,
    analyze_displacements,
    anchored_sites,
    cluster_connected_components,
    cluster_soprano_style,
    dataclass_to_plain,
    normalize_site_energies,
    parse_tolerance_scan,
    scan_connected_component_tolerances,
    uep_summary_to_relaxation_clusters,
)
from mufinder.science.wyckoff_analysis import (
    ClusterWyckoffAssignment,
    assign_cluster_wyckoff_sites,
)


_WORKSPACE_SCHEMA_VERSION = 1
_METHOD_CONNECTED = "connected"
_METHOD_HIERARCHICAL = "hier"
_METHOD_KMEANS = "kmeans"


class _EqualInitialHorizontalSplitter(QSplitter):
    """Horizontal splitter that applies an equal first split after layout."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(Qt.Orientation.Horizontal, parent)
        self._equal_initial_sizes_pending = True
        self._equal_initial_sizes_scheduled = False
        self._last_equal_width = 0
        self._stable_equal_width_passes = 0

    def showEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().showEvent(event)
        self._schedule_equal_initial_sizes()

    def resizeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().resizeEvent(event)
        self._schedule_equal_initial_sizes()

    def _schedule_equal_initial_sizes(self) -> None:
        if self._equal_initial_sizes_pending and self.isVisible() and not self._equal_initial_sizes_scheduled:
            self._equal_initial_sizes_scheduled = True
            QTimer.singleShot(0, self._apply_equal_initial_sizes)

    def _apply_equal_initial_sizes(self) -> None:
        self._equal_initial_sizes_scheduled = False
        if not self._equal_initial_sizes_pending or self.count() < 2:
            return
        width = self.width()
        if width <= 2:
            self._schedule_equal_initial_sizes()
            return
        self.setSizes([width // 2, width - width // 2])
        if width == self._last_equal_width:
            self._stable_equal_width_passes += 1
        else:
            self._last_equal_width = width
            self._stable_equal_width_passes = 0
        self._equal_initial_sizes_pending = self._stable_equal_width_passes < 2
        self._schedule_equal_initial_sizes()


@dataclass(frozen=True, slots=True)
class CompletedRelaxationAnalysisSource:
    """UI source row for a completed muon structural relaxation."""

    job_id: UUID
    job_label: str
    charge_state: str
    structure_id: UUID
    site_id: UUID | None
    folded_fractional_position: tuple[float, float, float]
    total_energy_ev: float | None

    def to_relaxed_site(self, host_structure: Structure) -> RelaxedMuonSite:
        cart = tuple(
            float(value)
            for value in (
                np.asarray(self.folded_fractional_position, dtype=float)
                @ np.asarray(host_structure.lattice_vectors, dtype=float)
            ).tolist()
        )
        return RelaxedMuonSite(
            job_id=self.job_id,
            job_label=self.job_label,
            charge_state=self.charge_state,
            structure_id=self.structure_id,
            site_id=self.site_id,
            folded_fractional_position=self.folded_fractional_position,
            cartesian_position_ang=cart,
            total_energy_ev=self.total_energy_ev,
        )


class RelaxationAnalysisJobDialog(QDialog):
    """Select completed relaxations to add to one analysis workspace."""

    _CHECK_COLUMN = 0
    _LABEL_COLUMN = 1
    _ENERGY_COLUMN = 2
    _FX_COLUMN = 3
    _FY_COLUMN = 4
    _FZ_COLUMN = 5

    def __init__(
        self,
        *,
        title: str,
        sources: Sequence[CompletedRelaxationAnalysisSource],
        excluded_job_ids: set[UUID] | None = None,
        reference_job_ids: set[UUID] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(860, 420)
        self._sources = list(sources)
        self._excluded_job_ids = set(excluded_job_ids or set())
        self._reference_job_ids = set(reference_job_ids or set())
        self._reference_minimum_ev = self._minimum_reference_energy()
        self._table = QTableWidget(0, 6, self)
        self._table.setHorizontalHeaderLabels(["Add", "Relaxation", "dE vs workspace min (eV)", "fX", "fY", "fZ"])
        self._table.verticalHeader().setVisible(False)
        self._table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._build_layout()
        self._populate()

    def _minimum_reference_energy(self) -> float | None:
        referenced = [
            float(source.total_energy_ev)
            for source in self._sources
            if source.total_energy_ev is not None
            and (not self._reference_job_ids or source.job_id in self._reference_job_ids)
        ]
        if not referenced:
            referenced = [
                float(source.total_energy_ev)
                for source in self._sources
                if source.total_energy_ev is not None
            ]
        return min(referenced) if referenced else None

    def _relative_energy_text(self, source: CompletedRelaxationAnalysisSource) -> str:
        if source.total_energy_ev is None or self._reference_minimum_ev is None:
            return ""
        return f"{float(source.total_energy_ev) - float(self._reference_minimum_ev):.5f}"

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        intro = QLabel("Select completed structural relaxations to include in this analysis workspace.", self)
        intro.setWordWrap(True)
        style_hint_text(intro)
        layout.addWidget(intro)
        layout.addWidget(self._table, 1)
        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.accepted.connect(self._accept_if_valid)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate(self) -> None:
        self._table.setRowCount(0)
        for source in self._sources:
            row = self._table.rowCount()
            self._table.insertRow(row)
            check = QTableWidgetItem("")
            check.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsSelectable)
            already_present = source.job_id in self._excluded_job_ids
            check.setCheckState(Qt.CheckState.Unchecked)
            check.setData(Qt.ItemDataRole.UserRole, str(source.job_id))
            if already_present:
                check.setFlags(Qt.ItemFlag.ItemIsSelectable)
                check.setToolTip("Already included in this workspace.")
            self._table.setItem(row, self._CHECK_COLUMN, check)
            label = QTableWidgetItem(source.job_label)
            if already_present:
                label.setToolTip("Already included in this workspace.")
            self._table.setItem(row, self._LABEL_COLUMN, label)
            self._table.setItem(
                row,
                self._ENERGY_COLUMN,
                QTableWidgetItem(self._relative_energy_text(source)),
            )
            for column, value in zip(
                (self._FX_COLUMN, self._FY_COLUMN, self._FZ_COLUMN),
                source.folded_fractional_position,
                strict=True,
            ):
                self._table.setItem(row, column, QTableWidgetItem(f"{float(value):.4f}"))
        self._table.horizontalHeader().setSectionResizeMode(self._LABEL_COLUMN, QHeaderView.ResizeMode.Stretch)
        self._table.resizeColumnsToContents()

    def _accept_if_valid(self) -> None:
        if not self.selected_job_ids():
            QMessageBox.warning(self, "No relaxations selected", "Select at least one completed relaxation.")
            return
        self.accept()

    def selected_job_ids(self) -> list[UUID]:
        selected: list[UUID] = []
        for row in range(self._table.rowCount()):
            item = self._table.item(row, self._CHECK_COLUMN)
            if item is None or item.checkState() != Qt.CheckState.Checked:
                continue
            try:
                selected.append(UUID(str(item.data(Qt.ItemDataRole.UserRole))))
            except (TypeError, ValueError):
                continue
        return selected


def default_workspace_state(
    *,
    name: str,
    charge_state: str,
    host_structure_id: UUID,
    job_ids: Sequence[UUID],
) -> dict[str, object]:
    """Return a persisted workspace state dict."""

    return {
        "schema_version": _WORKSPACE_SCHEMA_VERSION,
        "id": str(uuid4()),
        "name": str(name).strip() or "Relaxation analysis",
        "charge_state": str(charge_state),
        "host_structure_id": str(host_structure_id),
        "job_ids": [str(job_id) for job_id in job_ids],
        "show_shifted_positions": False,
        "clustering": {
            "method": _METHOD_CONNECTED,
            "connected_tolerance_ang": 0.25,
            "hierarchical_t": 0.3,
            "kmeans_k": 4,
            "wyckoff_match_tolerance_ang": 0.25,
            "tolerance_scan": [],
            "result": None,
        },
        "viewer_state": {},
        "displacement_state": {},
    }


def _workspace_id(workspace: Mapping[str, object]) -> str:
    return str(workspace.get("id", "")).strip()


def _workspace_name(workspace: Mapping[str, object]) -> str:
    return str(workspace.get("name", "Relaxation analysis")).strip() or "Relaxation analysis"


def _workspace_charge_state(workspace: Mapping[str, object]) -> str:
    return str(workspace.get("charge_state", "positive_muon")).strip() or "positive_muon"


def _workspace_job_ids(workspace: Mapping[str, object]) -> list[UUID]:
    ids: list[UUID] = []
    raw = workspace.get("job_ids", [])
    if not isinstance(raw, list):
        return ids
    for value in raw:
        try:
            ids.append(UUID(str(value)))
        except (TypeError, ValueError):
            continue
    return ids


def _cluster_color_map(labels: Sequence[int]) -> dict[int, str]:
    cmap = colormaps["tab20"]
    colors: dict[int, str] = {}
    for index, label in enumerate(sorted(set(int(value) for value in labels))):
        colors[int(label)] = mcolors.to_hex(cmap(index % cmap.N))
    return colors


def _lightened_cluster_color(color: str) -> str:
    rgb = np.asarray(mcolors.to_rgb(color), dtype=float)
    return mcolors.to_hex((0.28 * rgb) + 0.72)


def _viewer_species_color(species: str) -> str:
    key = str(species).strip()
    number = atomic_numbers.get(key)
    if number is None:
        number = atomic_numbers.get(key.capitalize())
    if number is None:
        return current_palette().accent
    return mcolors.to_hex(np.asarray(jmol_colors[int(number)], dtype=float))


def _charge_state_label(charge_state: str) -> str:
    return "Muonium" if charge_state == "muonium" else "Positive muon"


def _format_energy(value: float | None) -> str:
    return "" if value is None else f"{float(value):.5f}"


def _format_fractional(position: Sequence[float] | None, *, precision: int = 4) -> str:
    if position is None or len(position) != 3:
        return ""
    return f"({float(position[0]):.{precision}f}, {float(position[1]):.{precision}f}, {float(position[2]):.{precision}f})"


def _wyckoff_assignments_from_payload(payload: object) -> dict[int, Mapping[str, object]]:
    assignments: dict[int, Mapping[str, object]] = {}
    if not isinstance(payload, list):
        return assignments
    for entry in payload:
        if not isinstance(entry, Mapping):
            continue
        try:
            cluster_index = int(entry.get("cluster_index", 0))
        except (TypeError, ValueError):
            continue
        match = entry.get("match")
        if cluster_index > 0 and isinstance(match, Mapping):
            assignments[cluster_index] = match
    return assignments


def _wyckoff_assignment_text(match: Mapping[str, object] | None) -> str:
    if match is None:
        return "Wyckoff unavailable"
    label = str(match.get("label", "")).strip()
    position = match.get("fractional_position")
    fractional = None
    if isinstance(position, (list, tuple)) and len(position) == 3:
        try:
            fractional = tuple(float(value) for value in position)
        except (TypeError, ValueError):
            fractional = None
    if label and fractional is not None:
        return f"{_format_fractional(fractional)} {label}"
    if label:
        return label
    return "Wyckoff unavailable"


def _wyckoff_assignment_tooltip(match: Mapping[str, object] | None) -> str:
    if match is None:
        return ""
    rows: list[str] = []
    group = match.get("space_group_symbol")
    number = match.get("space_group_number")
    if group:
        rows.append(f"Space group: {group}" + (f" ({number})" if number else ""))
    site_symmetry = match.get("site_symmetry_symbol")
    if site_symmetry:
        rows.append(f"Site symmetry: {site_symmetry}")
    max_distance = match.get("max_distance_ang")
    mean_distance = match.get("mean_distance_ang")
    try:
        rows.append(f"Max distance: {float(max_distance):.4f} Å")
    except (TypeError, ValueError):
        pass
    try:
        rows.append(f"Mean distance: {float(mean_distance):.4f} Å")
    except (TypeError, ValueError):
        pass
    warnings = match.get("warnings")
    if isinstance(warnings, list) and warnings:
        rows.extend(str(warning) for warning in warnings)
    return "\n".join(rows)


class RelaxationClusteringDialog(QDialog):
    """Dialog for clustering one structural-relaxation analysis workspace."""

    clustering_applied = Signal(dict)

    def __init__(
        self,
        *,
        workspace: Mapping[str, object],
        host_structure: Structure,
        sites: Sequence[RelaxedMuonSite],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Clustering - {_workspace_name(workspace)}")
        self.resize(1240, 820)
        self.setMinimumSize(900, 620)
        self._workspace = dict(workspace)
        self._host_structure = host_structure
        self._sites = normalize_site_energies(sites)
        self._viewer_shutdown = False
        self._current_labels: tuple[int, ...] = ()
        self._current_clusters: tuple[RelaxationClusterSummary, ...] = ()
        self._current_wyckoff_assignments: tuple[ClusterWyckoffAssignment, ...] = ()
        self._tolerance_scan = parse_tolerance_scan(
            (self._clustering_state()).get("tolerance_scan", [])
        )

        self._graph_figure = Figure(figsize=(5.6, 3.0), facecolor="none")
        self._graph_figure.patch.set_alpha(0.0)
        self._graph_figure.subplots_adjust(left=0.02, right=0.98, top=0.98, bottom=0.02)
        self._graph_canvas = FigureCanvasQTAgg(self._graph_figure)
        self._graph_canvas.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._graph_canvas.setStyleSheet("background: transparent;")
        self._scan_figure = Figure(figsize=(5.6, 2.4), facecolor=current_palette().bg_surface)
        self._scan_figure.subplots_adjust(left=0.14, right=0.98, top=0.90, bottom=0.24)
        self._scan_canvas = FigureCanvasQTAgg(self._scan_figure)
        self._left_figure = Figure(figsize=(6.2, 5.2), facecolor=current_palette().bg_surface)
        self._left_figure.subplots_adjust(left=0.12, right=0.98, top=0.96, bottom=0.12)
        self._left_canvas = FigureCanvasQTAgg(self._left_figure)
        self._viewer = AtomicStructureViewer(self)
        self._summary_table = QTableWidget(0, 6, self)
        self._summary_table.setHorizontalHeaderLabels(
            ["Cluster", "Wyckoff", "Size", "Representative", "Min dE (eV)", "Std. dev."]
        )
        self._summary_table.verticalHeader().setVisible(False)
        self._summary_table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        header = self._summary_table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        for column in (0, 2, 3, 4, 5):
            item = self._summary_table.horizontalHeaderItem(column)
            if item is None:
                continue
            width = self._summary_table.fontMetrics().horizontalAdvance(item.text()) + 28
            self._summary_table.setColumnWidth(column, max(self._summary_table.columnWidth(column), width))

        self._method_combo = QComboBox(self)
        self._method_combo.addItem("Connected Components", _METHOD_CONNECTED)
        self._method_combo.addItem("Hierarchical", _METHOD_HIERARCHICAL)
        self._method_combo.addItem("K-means", _METHOD_KMEANS)
        self._tolerance_input = DecimalLineEdit(self)
        self._tolerance_input.setRange(0.0, 99.0)
        self._tolerance_input.setDecimals(4)
        self._hierarchical_input = DecimalLineEdit(self)
        self._hierarchical_input.setRange(1e-9, 99.0)
        self._hierarchical_input.setDecimals(4)
        self._kmeans_spin = QSpinBox(self)
        self._kmeans_spin.setRange(1, max(len(self._sites), 1))
        self._wyckoff_tolerance_input = DecimalLineEdit(self)
        self._wyckoff_tolerance_input.setRange(0.0, 99.0)
        self._wyckoff_tolerance_input.setDecimals(4)
        self._scan_button = QPushButton("Scan Tolerances", self)
        self._scan_button.setObjectName("relaxationToleranceScanButton")
        self._apply_button = QPushButton("Apply Clustering", self)

        self._build_ui()
        self._restore_controls()
        self._connect_signals()
        self._refresh()

    def _clustering_state(self) -> dict[str, object]:
        raw = self._workspace.get("clustering", {})
        return dict(raw) if isinstance(raw, Mapping) else {}

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)
        title = QLabel("Clustering", self)
        style_section_title(title)
        subtitle = QLabel("Explore relaxed muon endpoints with graph, hierarchical, or k-means clustering.", self)
        subtitle.setWordWrap(True)
        style_section_subtitle(subtitle)
        layout.addWidget(title)
        layout.addWidget(subtitle)

        controls = QFrame(self)
        controls.setObjectName("uepIsosurfaceSliderPanel")
        controls_layout = QGridLayout(controls)
        controls_layout.setContentsMargins(14, 12, 14, 12)
        controls_layout.setHorizontalSpacing(12)
        controls_layout.setVerticalSpacing(8)
        controls_layout.addWidget(QLabel("Method", controls), 0, 0)
        controls_layout.addWidget(self._method_combo, 0, 1)
        self._tolerance_label = QLabel("Distance tolerance (Å)", controls)
        self._hierarchical_label = QLabel("Hierarchical t", controls)
        self._kmeans_label = QLabel("Clusters k", controls)
        self._wyckoff_tolerance_label = QLabel("Wyckoff tolerance (Å)", controls)
        controls_layout.addWidget(self._tolerance_label, 0, 2)
        controls_layout.addWidget(self._tolerance_input, 0, 3)
        controls_layout.addWidget(self._hierarchical_label, 0, 4)
        controls_layout.addWidget(self._hierarchical_input, 0, 5)
        controls_layout.addWidget(self._kmeans_label, 0, 6)
        controls_layout.addWidget(self._kmeans_spin, 0, 7)
        controls_layout.addWidget(self._wyckoff_tolerance_label, 0, 9)
        controls_layout.addWidget(self._wyckoff_tolerance_input, 0, 10)
        controls_layout.setColumnStretch(8, 1)
        layout.addWidget(controls)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)

        left = QWidget(splitter)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(8)
        self._connected_panel = QWidget(left)
        connected_layout = QVBoxLayout(self._connected_panel)
        connected_layout.setContentsMargins(0, 0, 0, 0)
        connected_layout.setSpacing(8)
        scan_panel = QFrame(self._connected_panel)
        scan_panel.setObjectName("connectedToleranceScanPanel")
        scan_layout = QGridLayout(scan_panel)
        scan_layout.setContentsMargins(12, 12, 12, 12)
        scan_layout.setSpacing(0)
        scan_layout.addWidget(self._scan_canvas, 0, 0)
        scan_layout.addWidget(self._scan_button, 0, 0, Qt.AlignmentFlag.AlignTop | Qt.AlignmentFlag.AlignRight)
        connected_layout.addWidget(self._graph_canvas, 3)
        connected_layout.addWidget(scan_panel, 2)
        self._soprano_panel = QWidget(left)
        soprano_layout = QVBoxLayout(self._soprano_panel)
        soprano_layout.setContentsMargins(0, 0, 0, 0)
        self._hier_toolbar = _DendrogramNavigationToolbar(self._left_canvas, self._soprano_panel)
        self._hier_toolbar.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._hier_toolbar.setMovable(False)
        self._hier_toolbar.setContextMenuPolicy(Qt.ContextMenuPolicy.PreventContextMenu)
        self._hier_toolbar.setObjectName("uepClusteringDendrogramToolbar")
        soprano_layout.addWidget(self._hier_toolbar)
        soprano_layout.addWidget(self._left_canvas, 1)
        left_layout.addWidget(self._connected_panel, 1)
        left_layout.addWidget(self._soprano_panel, 1)

        right = QWidget(splitter)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(8)
        self._viewer.setMinimumHeight(300)
        right_layout.addWidget(self._viewer, 2)
        right_layout.addWidget(self._summary_table, 1)

        splitter.addWidget(left)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

        button_row = QHBoxLayout()
        button_row.addStretch(1)
        button_row.addWidget(self._apply_button)
        close_button = QPushButton("Close", self)
        close_button.clicked.connect(self.reject)
        button_row.addWidget(close_button)
        layout.addLayout(button_row)

    def _restore_controls(self) -> None:
        state = self._clustering_state()
        method = str(state.get("method", _METHOD_CONNECTED))
        index = self._method_combo.findData(method)
        self._method_combo.setCurrentIndex(max(index, 0))
        try:
            self._tolerance_input.setValue(float(state.get("connected_tolerance_ang", 0.25)))
        except (TypeError, ValueError):
            self._tolerance_input.setValue(0.25)
        try:
            self._hierarchical_input.setValue(float(state.get("hierarchical_t", 0.3)))
        except (TypeError, ValueError):
            self._hierarchical_input.setValue(0.3)
        try:
            self._kmeans_spin.setValue(min(max(int(state.get("kmeans_k", 4)), 1), max(len(self._sites), 1)))
        except (TypeError, ValueError):
            self._kmeans_spin.setValue(min(4, max(len(self._sites), 1)))
        try:
            self._wyckoff_tolerance_input.setValue(float(state.get("wyckoff_match_tolerance_ang", 0.25)))
        except (TypeError, ValueError):
            self._wyckoff_tolerance_input.setValue(0.25)

    def _connect_signals(self) -> None:
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        self._tolerance_input.editingFinished.connect(self._refresh)
        self._hierarchical_input.editingFinished.connect(self._refresh)
        self._kmeans_spin.valueChanged.connect(lambda *_: self._refresh())
        self._wyckoff_tolerance_input.editingFinished.connect(self._refresh)
        self._scan_button.clicked.connect(self._scan_tolerances)
        self._apply_button.clicked.connect(self._apply)

    def _method(self) -> str:
        data = self._method_combo.currentData()
        return str(data) if isinstance(data, str) else _METHOD_CONNECTED

    def _sync_controls(self) -> None:
        method = self._method()
        connected = method == _METHOD_CONNECTED
        hierarchical = method == _METHOD_HIERARCHICAL
        kmeans = method == _METHOD_KMEANS
        self._tolerance_label.setVisible(connected)
        self._tolerance_input.setVisible(connected)
        self._scan_button.setVisible(connected)
        self._hierarchical_label.setVisible(hierarchical)
        self._hierarchical_input.setVisible(hierarchical)
        self._kmeans_label.setVisible(kmeans)
        self._kmeans_spin.setVisible(kmeans)
        self._hier_toolbar.setVisible(hierarchical)
        self._connected_panel.setVisible(connected)
        self._soprano_panel.setVisible(not connected)

    def _on_method_changed(self, _index: int) -> None:
        self._sync_controls()
        self._refresh()

    def _scan_tolerances(self) -> None:
        try:
            self._tolerance_scan = scan_connected_component_tolerances(
                self._sites,
                self._host_structure,
                start_ang=0.0,
                stop_ang=1.0,
                points=51,
            )
        except RelaxationAnalysisError as exc:
            QMessageBox.warning(self, "Tolerance scan failed", str(exc))
            return
        self._refresh()

    def _refresh(self) -> None:
        self._sync_controls()
        if not self._sites:
            return
        try:
            if self._method() == _METHOD_CONNECTED:
                result = cluster_connected_components(
                    self._sites,
                    self._host_structure,
                    tolerance_ang=self._tolerance_input.value(),
                )
                self._current_labels = result.cluster_labels
                self._current_clusters = result.clusters
                self._append_scan_point(result)
                self._render_connected(result)
            else:
                result = cluster_soprano_style(
                    self._sites,
                    self._host_structure,
                    method=self._method(),
                    hierarchical_t=self._hierarchical_input.value(),
                    kmeans_k=self._kmeans_spin.value(),
                )
                self._current_labels = result.cluster_labels
                self._current_clusters = uep_summary_to_relaxation_clusters(self._sites, result.summary)
                self._render_soprano(result)
            self._current_wyckoff_assignments = assign_cluster_wyckoff_sites(
                self._sites,
                self._current_clusters,
                self._host_structure,
                match_tolerance_ang=self._wyckoff_tolerance_input.value(),
            )
        except Exception as exc:  # noqa: BLE001
            self._show_plot_error(str(exc))
            return
        self._render_viewer()
        self._populate_summary_table()

    def _append_scan_point(self, result: ConnectedComponentsResult) -> None:
        rows = dict(parse_tolerance_scan(self._tolerance_scan))
        rows[round(float(result.tolerance_ang), 10)] = len(result.clusters)
        self._tolerance_scan = tuple((float(key), int(value)) for key, value in sorted(rows.items()))

    def _show_plot_error(self, text: str) -> None:
        figure = self._graph_figure if self._method() == _METHOD_CONNECTED else self._left_figure
        canvas = self._graph_canvas if self._method() == _METHOD_CONNECTED else self._left_canvas
        figure.clear()
        axis = figure.add_subplot(111)
        axis.axis("off")
        axis.text(0.5, 0.5, text, ha="center", va="center")
        canvas.draw_idle()

    @staticmethod
    def _spring_component_layout(members: Sequence[int], adjacency: np.ndarray) -> np.ndarray:
        count = len(members)
        if count == 0:
            return np.empty((0, 2), dtype=float)
        if count == 1:
            return np.zeros((1, 2), dtype=float)
        if count == 2:
            return np.asarray([[-0.75, 0.0], [0.75, 0.0]], dtype=float)

        angles = np.linspace(0.0, 2.0 * np.pi, count, endpoint=False)
        positions = np.column_stack((np.cos(angles), np.sin(angles))).astype(float)
        local_adjacency = np.asarray(adjacency[np.ix_(members, members)], dtype=bool)
        ideal = 1.0 / np.sqrt(float(count))
        temperature = 0.22
        for _ in range(140):
            delta = positions[:, np.newaxis, :] - positions[np.newaxis, :, :]
            distances = np.linalg.norm(delta, axis=2)
            distances = np.where(distances < 1e-6, 1e-6, distances)
            direction = delta / distances[:, :, np.newaxis]
            repulsive = (ideal * ideal / distances)[:, :, np.newaxis] * direction
            displacement = np.sum(repulsive, axis=1)
            for left in range(count):
                for right in range(left + 1, count):
                    if not local_adjacency[left, right]:
                        continue
                    vector = positions[left] - positions[right]
                    distance = max(float(np.linalg.norm(vector)), 1e-6)
                    attractive = (distance * distance / ideal) * vector / distance
                    displacement[left] -= attractive
                    displacement[right] += attractive
            lengths = np.linalg.norm(displacement, axis=1)
            scale = np.minimum(lengths, temperature) / np.where(lengths < 1e-6, 1.0, lengths)
            positions += displacement * scale[:, np.newaxis]
            positions -= np.mean(positions, axis=0)
            temperature *= 0.96

        positions -= np.mean(positions, axis=0)
        if count > 2:
            _, _, vh = np.linalg.svd(positions, full_matrices=False)
            positions = positions @ vh.T
        span = np.ptp(positions, axis=0)
        largest_span = max(float(np.max(span)), 1e-6)
        return positions * (1.95 / largest_span)

    @classmethod
    def _connected_graph_positions(cls, result: ConnectedComponentsResult) -> dict[int, tuple[float, float]]:
        adjacency = np.asarray(result.adjacency_matrix, dtype=int)
        component_layouts: list[tuple[list[int], np.ndarray, float, float]] = []
        for cluster in result.clusters:
            members = list(cluster.member_indexes)
            layout = cls._spring_component_layout(members, adjacency)
            if layout.size:
                layout -= np.mean(layout, axis=0)
            span = np.ptp(layout, axis=0) if len(layout) else np.zeros(2, dtype=float)
            width = max(float(span[0]) + 1.55, 1.55)
            height = max(float(span[1]) + 1.55, 1.55)
            component_layouts.append((members, layout, width, height))

        if not component_layouts:
            return {}

        target_width = max(4.8, min(12.0, np.sqrt(sum(width * height for _, _, width, height in component_layouts)) * 1.85))
        rows: list[list[tuple[list[int], np.ndarray, float, float]]] = []
        current: list[tuple[list[int], np.ndarray, float, float]] = []
        current_width = 0.0
        gap = 1.25
        for component in component_layouts:
            width = component[2]
            projected = width if not current else current_width + gap + width
            if current and projected > target_width:
                rows.append(current)
                current = [component]
                current_width = width
            else:
                current.append(component)
                current_width = projected
        if current:
            rows.append(current)

        positions: dict[int, tuple[float, float]] = {}
        y_offset = 0.0
        row_gap = 1.65
        for row in rows:
            row_width = sum(component[2] for component in row) + gap * max(len(row) - 1, 0)
            row_height = max(component[3] for component in row)
            x_offset = -0.5 * row_width
            for members, layout, width, _height in row:
                center_x = x_offset + 0.5 * width
                for member, point in zip(members, layout, strict=False):
                    positions[int(member)] = (float(center_x + point[0]), float(y_offset + point[1]))
                x_offset += width + gap
            y_offset -= row_height + row_gap
        if positions:
            all_points = np.asarray(list(positions.values()), dtype=float)
            centre = np.mean(all_points, axis=0)
            positions = {index: (float(x - centre[0]), float(y - centre[1])) for index, (x, y) in positions.items()}
        return positions

    @staticmethod
    def _apply_convergence_axis_theme(axis) -> None:
        palette = current_palette()
        axis.set_facecolor(palette.bg_surface_alt)
        axis.tick_params(axis="both", colors=palette.text_primary, labelcolor=palette.text_primary)
        axis.xaxis.label.set_color(palette.text_primary)
        axis.yaxis.label.set_color(palette.text_primary)
        axis.title.set_color(palette.text_primary)
        for spine in axis.spines.values():
            spine.set_color(palette.border)
        axis.grid(True, color=palette.border, alpha=0.35)

    def _render_connected(self, result: ConnectedComponentsResult) -> None:
        colors = _cluster_color_map(result.cluster_labels)
        self._graph_figure.clear()
        self._graph_figure.patch.set_alpha(0.0)
        axis = self._graph_figure.add_subplot(111)
        axis.set_facecolor("none")
        labels = np.asarray(result.cluster_labels, dtype=int)
        adjacency = np.asarray(result.adjacency_matrix, dtype=int)
        positions = self._connected_graph_positions(result)
        for i in range(len(labels)):
            for j in range(i + 1, len(labels)):
                if adjacency[i, j] <= 0:
                    continue
                x1, y1 = positions.get(i, (0.0, 0.0))
                x2, y2 = positions.get(j, (0.0, 0.0))
                edge_color = colors.get(int(labels[i]), current_palette().accent)
                axis.plot(
                    [x1, x2],
                    [y1, y2],
                    color=edge_color,
                    linewidth=3.2,
                    solid_capstyle="round",
                    zorder=1,
                )
        for index, label in enumerate(labels):
            x, y = positions.get(index, (float(index), 0.0))
            color = colors.get(int(label), current_palette().accent)
            axis.scatter(
                [x],
                [y],
                s=980,
                facecolors=_lightened_cluster_color(color),
                edgecolors=color,
                linewidths=3.0,
                zorder=2,
            )
            axis.text(
                x,
                y,
                str(index + 1),
                color="#111111",
                ha="center",
                va="center",
                fontsize=18,
                fontfamily="serif",
                zorder=3,
            )
        axis.set_aspect("equal", adjustable="datalim")
        axis.margins(0.18)
        axis.axis("off")
        self._graph_canvas.draw_idle()

        self._scan_figure.clear()
        self._scan_figure.set_facecolor(current_palette().bg_surface)
        scan_axis = self._scan_figure.add_subplot(111)
        self._apply_convergence_axis_theme(scan_axis)
        scan_rows = parse_tolerance_scan(self._tolerance_scan)
        if scan_rows:
            x_values, y_values = zip(*scan_rows, strict=False)
            scan_axis.plot(x_values, y_values, marker="o", linestyle="-", color="#b91c1c", linewidth=1.5)
            scan_axis.axvline(result.tolerance_ang, color=current_palette().text_muted, linestyle="-", linewidth=1.0, alpha=0.55)
        scan_axis.set_xlabel("Tolerance (Å)")
        scan_axis.set_ylabel("Clusters")
        self._scan_canvas.draw_idle()

    def _render_soprano(self, result) -> None:
        colors = _cluster_color_map(result.cluster_labels)
        self._left_figure.clear()
        axis = self._left_figure.add_subplot(111)
        axis.set_facecolor(current_palette().bg_surface)
        if self._method() == _METHOD_HIERARCHICAL:
            from scipy.cluster import hierarchy

            linkage_matrix = np.asarray(result.linkage_matrix, dtype=float)
            if linkage_matrix.size == 0:
                axis.axis("off")
                axis.text(0.5, 0.5, "Only one relaxed site is available.", ha="center", va="center")
                self._left_canvas.draw_idle()
                return
            hierarchy.dendrogram(
                linkage_matrix,
                ax=axis,
                color_threshold=0.0,
                above_threshold_color=current_palette().text_muted,
                labels=[str(index + 1) for index in range(len(self._sites))],
            )
            axis.axhline(self._hierarchical_input.value(), color=current_palette().accent, linestyle="--", linewidth=1.2)
            axis.set_ylabel("Distance")
            axis.set_xlabel("Relaxed site")
        else:
            vectors = np.asarray(result.normalized_genome_vectors, dtype=float)
            if len(vectors) == 0:
                axis.text(0.5, 0.5, "No sites available.", ha="center", va="center")
            else:
                origin = np.mean(vectors, axis=0)
                centred = vectors - origin
                if len(vectors) == 1:
                    points = np.zeros((1, 2), dtype=float)
                else:
                    _, _, vh = np.linalg.svd(centred, full_matrices=False)
                    components = vh[:2]
                    if components.shape[0] == 1:
                        components = np.vstack([components, np.zeros((1, components.shape[1]), dtype=float)])
                    points = centred @ components.T
                axis.scatter(
                    points[:, 0],
                    points[:, 1],
                    c=[colors.get(int(label), current_palette().accent) for label in result.cluster_labels],
                    s=96,
                    edgecolors=current_palette().bg_surface,
                )
                for index, point in enumerate(points, start=1):
                    axis.text(point[0], point[1], f" {index}", color=current_palette().text_primary, va="center")
                axis.set_xlabel("PC1")
                axis.set_ylabel("PC2")
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(current_palette().border)
        axis.spines["bottom"].set_color(current_palette().border)
        axis.tick_params(colors=current_palette().text_primary)
        self._left_canvas.draw_idle()

    def _render_viewer(self) -> None:
        colors = _cluster_color_map(self._current_labels)
        positions = [
            site.cartesian_position_ang
            for site in normalize_site_energies(self._sites)
        ]
        overlays = [
            {
                "positions": positions,
                "colors": [colors.get(int(label), current_palette().accent) for label in self._current_labels],
                "radii": [MUON_VIEWER_RADIUS_ANG] * len(positions),
                "alpha": 0.96,
            }
        ]
        self._viewer.set_structure(self._host_structure)
        self._viewer.set_point_overlays(overlays)

    def _populate_summary_table(self) -> None:
        self._summary_table.setRowCount(0)
        assignments = {
            int(assignment.cluster_index): dataclass_to_plain(assignment.match)
            for assignment in self._current_wyckoff_assignments
        }
        for cluster in self._current_clusters:
            row = self._summary_table.rowCount()
            self._summary_table.insertRow(row)
            wyckoff_match = assignments.get(int(cluster.cluster_index))
            values = [
                str(cluster.cluster_index),
                _wyckoff_assignment_text(wyckoff_match),
                str(cluster.size),
                str(cluster.representative_index + 1),
                _format_energy(cluster.minimum_relative_energy_ev),
                _format_energy(cluster.standard_deviation_ev),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 1:
                    item.setToolTip(_wyckoff_assignment_tooltip(wyckoff_match))
                self._summary_table.setItem(row, column, item)
        self._summary_table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)

    def _apply(self) -> None:
        if not self._current_labels:
            QMessageBox.warning(self, "No clustering", "Run clustering before applying it to the workspace.")
            return
        result_payload = {
            "cluster_labels": list(self._current_labels),
            "clusters": [dataclass_to_plain(cluster) for cluster in self._current_clusters],
            "wyckoff_assignments": [
                dataclass_to_plain(assignment)
                for assignment in self._current_wyckoff_assignments
            ],
        }
        clustering = {
            "method": self._method(),
            "connected_tolerance_ang": float(self._tolerance_input.value()),
            "hierarchical_t": float(self._hierarchical_input.value()),
            "kmeans_k": int(self._kmeans_spin.value()),
            "wyckoff_match_tolerance_ang": float(self._wyckoff_tolerance_input.value()),
            "tolerance_scan": [[float(t), int(count)] for t, count in parse_tolerance_scan(self._tolerance_scan)],
            "result": result_payload,
        }
        self.clustering_applied.emit(clustering)
        self.accept()

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


class DisplacementAnalysisDialog(QDialog):
    """Inspect host-ion displacements around a relaxed muon site."""

    def __init__(
        self,
        *,
        workspace_name: str,
        sites: Sequence[RelaxedMuonSite],
        structures_by_id: Mapping[UUID, Structure],
        comparison_options: Sequence[tuple[UUID, str, bool]],
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Analyse Displacements - {workspace_name}")
        self.resize(1180, 780)
        self.setMinimumSize(920, 620)
        self._sites = normalize_site_energies(sites)
        self._structures = dict(structures_by_id)
        self._comparison_options = list(comparison_options)
        self._result: DisplacementAnalysisResult | None = None
        self._viewer_shutdown = False
        self._site_combo = QComboBox(self)
        self._comparison_combo = QComboBox(self)
        self._message = QLabel("", self)
        self._message.setWordWrap(True)
        style_hint_text(self._message)
        self._viewer = AtomicStructureViewer(self)
        self._figure = Figure(figsize=(5.6, 4.0), facecolor="none")
        self._figure.patch.set_alpha(0.0)
        self._figure.subplots_adjust(left=0.14, right=0.82, top=0.94, bottom=0.16)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        self._canvas.setStyleSheet("background: transparent;")
        self._build_ui()
        self._populate()
        self._connect()
        self._refresh()

    def _build_ui(self) -> None:
        layout = QVBoxLayout(self)
        title = QLabel("Displacements", self)
        style_section_title(title)
        layout.addWidget(title)
        controls = QFrame(self)
        controls.setObjectName("uepIsosurfaceSliderPanel")
        form = QGridLayout(controls)
        form.setContentsMargins(14, 12, 14, 12)
        form.setHorizontalSpacing(12)
        form.addWidget(QLabel("Muon site", controls), 0, 0)
        form.addWidget(self._site_combo, 0, 1)
        form.addWidget(QLabel("Comparison host", controls), 0, 2)
        form.addWidget(self._comparison_combo, 0, 3)
        form.setColumnStretch(1, 1)
        form.setColumnStretch(3, 1)
        layout.addWidget(controls)
        layout.addWidget(self._message)
        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        self._viewer.setMinimumHeight(320)
        splitter.addWidget(self._viewer)
        right = QWidget(splitter)
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.addWidget(self._canvas, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)
        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _populate(self) -> None:
        self._site_combo.clear()
        for index, site in enumerate(self._sites, start=1):
            self._site_combo.addItem(f"Site {index} - {site.job_label}", index - 1)
        self._comparison_combo.clear()
        for structure_id, label, blocked in self._comparison_options:
            self._comparison_combo.addItem(label, str(structure_id))
            item_index = self._comparison_combo.count() - 1
            if blocked:
                model_item = self._comparison_combo.model().item(item_index)
                if model_item is not None:
                    model_item.setEnabled(False)
                    model_item.setToolTip("Free-cell pristine relaxations cannot be used for displacement analysis.")

    def _connect(self) -> None:
        self._site_combo.currentIndexChanged.connect(lambda *_: self._refresh())
        self._comparison_combo.currentIndexChanged.connect(lambda *_: self._refresh())

    def _selected_site(self) -> RelaxedMuonSite | None:
        index = self._site_combo.currentData()
        try:
            return self._sites[int(index)]
        except (TypeError, ValueError, IndexError):
            return None

    def _selected_comparison(self) -> Structure | None:
        raw = self._comparison_combo.currentData()
        try:
            return self._structures.get(UUID(str(raw)))
        except (TypeError, ValueError):
            return None

    def _refresh(self) -> None:
        site = self._selected_site()
        comparison = self._selected_comparison()
        if site is None or comparison is None:
            self._message.setText("Choose a muon site and a comparison structure.")
            return
        relaxed = self._structures.get(site.structure_id)
        if relaxed is None:
            self._message.setText("The relaxed structure for this site is no longer available.")
            return
        try:
            result = analyze_displacements(
                relaxed_structure=relaxed,
                comparison_structure=comparison,
                muon_fractional_position=site.folded_fractional_position,
            )
        except Exception as exc:  # noqa: BLE001
            self._message.setText(str(exc))
            return
        self._result = result
        self._message.setText(" ".join(result.warnings) if result.warnings else "")
        self._render_viewer(relaxed, result.comparison_structure, site)
        self._render_plot(result)

    @staticmethod
    def _structure_contains_muon(structure: Structure) -> bool:
        return any(str(species).strip().casefold().endswith(":mu") for species in structure.species)

    def _render_viewer(self, relaxed: Structure, comparison: Structure, site: RelaxedMuonSite) -> None:
        self._viewer.set_structure(relaxed)
        comp_lattice = np.asarray(comparison.lattice_vectors, dtype=float)
        comparison_positions = (np.asarray(comparison.fractional_positions, dtype=float) @ comp_lattice).tolist()
        comparison_colors, comparison_radii = viewer_atom_style_for_species(comparison.species)
        overlays = [
            {
                "positions": comparison_positions,
                "colors": comparison_colors.tolist(),
                "radii": comparison_radii.tolist(),
                "alpha": 0.26,
            }
        ]
        if not self._structure_contains_muon(relaxed):
            overlays.append(
                {
                    "positions": [site.cartesian_position_ang],
                    "colors": [current_palette().accent],
                    "radii": [MUON_VIEWER_RADIUS_ANG * 1.25],
                    "alpha": 0.98,
                }
            )
        self._viewer.set_point_overlays(overlays)

    def _render_plot(self, result: DisplacementAnalysisResult) -> None:
        self._figure.clear()
        self._figure.patch.set_alpha(0.0)
        axis = self._figure.add_subplot(111)
        axis.set_facecolor(current_palette().bg_surface)
        species = sorted(set(point.species for point in result.points))
        for symbol in species:
            points = [point for point in result.points if point.species == symbol]
            axis.scatter(
                [point.distance_from_muon_ang for point in points],
                [point.radial_displacement_ang for point in points],
                label=symbol,
                color=_viewer_species_color(symbol),
                s=42,
                alpha=0.9,
            )
        axis.axhline(0.0, color=current_palette().text_muted, linewidth=1.0)
        axis.set_xlabel("Distance from muon (Å)")
        axis.set_ylabel("Radial displacement (Å)")
        if species:
            legend = axis.legend(
                loc="upper left",
                bbox_to_anchor=(1.02, 1.0),
                borderaxespad=0.0,
                frameon=True,
            )
            frame = legend.get_frame()
            frame.set_facecolor(current_palette().bg_surface)
            frame.set_edgecolor(current_palette().border)
            frame.set_alpha(0.96)
            for label in legend.get_texts():
                label.set_color(current_palette().text_primary)
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.spines["left"].set_color(current_palette().border)
        axis.spines["bottom"].set_color(current_palette().border)
        axis.tick_params(colors=current_palette().text_primary)
        self._canvas.draw_idle()

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


class RelaxationAnalysisWidget(QWidget):
    """Container for persistent structural-relaxation analysis workspaces."""

    state_changed = Signal(list)
    add_sites_requested = Signal(str)
    pin_summary_requested = Signal(str)

    _SITE_COLUMN = 0
    _CLUSTER_COLUMN = 1
    _LABEL_COLUMN = 2
    _ENERGY_COLUMN = 3
    _FX_COLUMN = 4
    _FY_COLUMN = 5
    _FZ_COLUMN = 6

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._restoring = False
        self._workspaces: list[dict[str, object]] = []
        self._sources_by_job_id: dict[UUID, CompletedRelaxationAnalysisSource] = {}
        self._structures_by_id: dict[UUID, Structure] = {}
        self._host_structure: Structure | None = None
        self._comparison_options: list[tuple[UUID, str, bool]] = []
        self._workspace_tabs = QTabWidget(self)
        self._workspace_tabs.setObjectName("analysisWorkspaceTabs")
        self._workspace_tabs.setTabsClosable(False)
        self._workspace_tabs.currentChanged.connect(self._on_current_tab_changed)
        self._empty_label = QLabel("Create an analysis workspace from selected completed structural relaxations.", self)
        self._empty_label.setWordWrap(True)
        style_hint_text(self._empty_label)
        self._build_layout()

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(8)
        title = QLabel("Structural Relaxation Analysis", self)
        style_section_title(title)
        subtitle = QLabel("Cluster completed relaxation endpoints and inspect muon-induced distortions.", self)
        subtitle.setWordWrap(True)
        style_section_subtitle(subtitle)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addWidget(self._empty_label)
        layout.addWidget(self._workspace_tabs, 1)

    def set_context(
        self,
        *,
        workspaces: Sequence[Mapping[str, object]],
        sources: Sequence[CompletedRelaxationAnalysisSource],
        structures: Mapping[UUID, Structure],
        host_structure: Structure | None,
        comparison_options: Sequence[tuple[UUID, str, bool]],
    ) -> None:
        self._restoring = True
        try:
            self._workspaces = [dict(workspace) for workspace in workspaces]
            self._sources_by_job_id = {source.job_id: source for source in sources}
            self._structures_by_id = dict(structures)
            self._host_structure = host_structure
            self._comparison_options = list(comparison_options)
            self._rebuild_tabs()
        finally:
            self._restoring = False

    def focus_workspace(self, workspace_id: str, *, cluster_index: int | None = None) -> bool:
        """Switch to ``workspace_id``'s tab, optionally revealing one cluster.

        Returns ``True`` when the workspace (and, when requested, the cluster)
        was found. A ``False`` result lets callers surface a "the pinned
        analysis has changed" hint when a decoupled snapshot points at a
        workspace or cluster that no longer exists.
        """

        page: QWidget | None = None
        for index in range(self._workspace_tabs.count()):
            candidate = self._workspace_tabs.widget(index)
            if candidate.property("workspace_id") == workspace_id:
                self._workspace_tabs.setCurrentIndex(index)
                page = candidate
                break
        if page is None:
            return False
        # cluster_index <= 0 means "no specific cluster" (unclustered sites carry
        # label 0 and get no header); treat it as a plain workspace focus rather
        # than a failed reveal, so callers don't report a false "analysis changed".
        if cluster_index is None or int(cluster_index) <= 0:
            return True
        return self._reveal_cluster(page, int(cluster_index))

    @staticmethod
    def _reveal_cluster(page: QWidget, cluster_index: int) -> bool:
        table = page.findChild(QTreeWidget)
        if table is None:
            return False
        for row in range(table.topLevelItemCount()):
            item = table.topLevelItem(row)
            if item is None:
                continue
            if item.data(0, Qt.ItemDataRole.UserRole) != cluster_index:
                continue
            item.setExpanded(True)
            # Cluster headers are non-selectable; select the first member site
            # so the reveal has a visible highlight, falling back to the header.
            target = item.child(0) if item.childCount() else item
            table.setCurrentItem(target)
            table.scrollToItem(target, QAbstractItemView.ScrollHint.PositionAtCenter)
            return True
        return False

    def _rebuild_tabs(self) -> None:
        current_id = None
        current = self._workspace_tabs.currentWidget()
        if current is not None:
            current_id = current.property("workspace_id")
        self._workspace_tabs.clear()
        for workspace in self._workspaces:
            tab = self._build_workspace_tab(workspace)
            workspace_id = _workspace_id(workspace)
            tab.setProperty("workspace_id", workspace_id)
            self._workspace_tabs.addTab(tab, _workspace_name(workspace))
        self._empty_label.setVisible(not self._workspaces)
        self._workspace_tabs.setVisible(bool(self._workspaces))
        if current_id:
            self.focus_workspace(str(current_id))

    def _workspace_sites(self, workspace: Mapping[str, object]) -> tuple[RelaxedMuonSite, ...]:
        host = self._host_structure
        if host is None:
            return ()
        sites: list[RelaxedMuonSite] = []
        for job_id in _workspace_job_ids(workspace):
            source = self._sources_by_job_id.get(job_id)
            if source is not None:
                sites.append(source.to_relaxed_site(host))
        return normalize_site_energies(sites)

    def _build_workspace_tab(self, workspace: Mapping[str, object]) -> QWidget:
        page = QWidget(self._workspace_tabs)
        layout = QVBoxLayout(page)
        layout.setContentsMargins(8, 8, 8, 8)
        layout.setSpacing(8)

        header = QFrame(page)
        header.setObjectName("uepIsosurfaceSliderPanel")
        header_layout = QGridLayout(header)
        header_layout.setContentsMargins(14, 12, 14, 12)
        header_layout.setHorizontalSpacing(10)
        name_edit = QLineEdit(_workspace_name(workspace), header)
        name_edit.editingFinished.connect(lambda wid=_workspace_id(workspace), edit=name_edit: self._rename_workspace(wid, edit.text()))
        charge = QLabel(_charge_state_label(_workspace_charge_state(workspace)), header)
        add_button = QPushButton("Add structure to analysis", header)
        add_button.clicked.connect(lambda _checked=False, wid=_workspace_id(workspace): self.add_sites_requested.emit(wid))
        cluster_button = QPushButton("Clustering", header)
        cluster_button.clicked.connect(lambda _checked=False, wid=_workspace_id(workspace): self._open_clustering(wid))
        displacement_button = QPushButton("Analyse displacements", header)
        displacement_button.clicked.connect(lambda _checked=False, wid=_workspace_id(workspace): self._open_displacements(wid))
        pin_button = QPushButton("Pin to summary", header)
        pin_button.setToolTip(
            "Pin this workspace's distinct muon sites to the Muon Sites summary page."
        )
        pin_button.clicked.connect(lambda _checked=False, wid=_workspace_id(workspace): self.pin_summary_requested.emit(wid))
        show_shifted = QCheckBox("Show shifted positions", header)
        show_shifted.setStyleSheet("background: transparent;")
        show_shifted.setChecked(bool(workspace.get("show_shifted_positions", False)))
        show_shifted.toggled.connect(lambda checked, wid=_workspace_id(workspace): self._set_show_shifted(wid, checked))
        header_layout.addWidget(QLabel("Workspace", header), 0, 0)
        header_layout.addWidget(name_edit, 0, 1)
        header_layout.addWidget(QLabel("Charge state", header), 0, 2)
        header_layout.addWidget(charge, 0, 3)
        header_layout.addWidget(show_shifted, 0, 4)
        header_layout.addWidget(add_button, 1, 1)
        header_layout.addWidget(cluster_button, 1, 2)
        header_layout.addWidget(displacement_button, 1, 3)
        header_layout.addWidget(pin_button, 1, 4)
        header_layout.setColumnStretch(1, 1)
        layout.addWidget(header)

        splitter = _EqualInitialHorizontalSplitter(page)
        splitter.setObjectName("analysisWorkspaceContentSplitter")
        splitter.setChildrenCollapsible(False)
        table = QTreeWidget(splitter)
        table.setHeaderLabels(["Site", "Cluster", "Relaxation", "dE (eV)", "fX", "fY", "fZ"])
        table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        table.header().setSectionResizeMode(self._LABEL_COLUMN, QHeaderView.ResizeMode.Stretch)
        use_tabbed_table_corners(table)
        viewer = AtomicStructureViewer(splitter)
        viewer.setMinimumWidth(0)
        viewer.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Expanding)
        splitter.addWidget(table)
        splitter.addWidget(viewer)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([600, 600])
        layout.addWidget(splitter, 1)
        self._populate_workspace_table(table, viewer, workspace)
        return page

    def _workspace_by_id(self, workspace_id: str) -> dict[str, object] | None:
        for workspace in self._workspaces:
            if _workspace_id(workspace) == workspace_id:
                return workspace
        return None

    def _clustering_result_state(self, workspace: Mapping[str, object]) -> Mapping[str, object]:
        raw = workspace.get("clustering", {})
        if not isinstance(raw, Mapping):
            return {}
        result = raw.get("result")
        return result if isinstance(result, Mapping) else {}

    def _cluster_labels(self, workspace: Mapping[str, object], sites: Sequence[RelaxedMuonSite]) -> tuple[int, ...]:
        result = self._clustering_result_state(workspace)
        raw = result.get("cluster_labels")
        if isinstance(raw, list) and len(raw) == len(sites):
            try:
                return tuple(int(value) for value in raw)
            except (TypeError, ValueError):
                pass
        return tuple(0 for _ in sites)

    def _anchored(self, workspace: Mapping[str, object], sites: Sequence[RelaxedMuonSite]) -> tuple[AnchoredRelaxedSite, ...]:
        host = self._host_structure
        labels = self._cluster_labels(workspace, sites)
        if host is None or not sites or not any(labels):
            return tuple(
                AnchoredRelaxedSite(
                    site_index=index,
                    cluster_index=int(labels[index]) if index < len(labels) else 0,
                    original_fractional_position=site.folded_fractional_position,
                    shifted_fractional_position=site.folded_fractional_position,
                    shifted=False,
                )
                for index, site in enumerate(sites)
            )
        try:
            return anchored_sites(sites, host, labels)
        except RelaxationAnalysisError:
            return ()

    def _populate_workspace_table(self, table: QTreeWidget, viewer: AtomicStructureViewer, workspace: Mapping[str, object]) -> None:
        sites = self._workspace_sites(workspace)
        anchored = self._anchored(workspace, sites)
        labels = self._cluster_labels(workspace, sites)
        show_shifted = bool(workspace.get("show_shifted_positions", False))
        colors = _cluster_color_map(labels)
        table.clear()
        lattice = np.asarray(self._host_structure.lattice_vectors, dtype=float) if self._host_structure is not None else np.eye(3)
        viewer_positions: list[list[float]] = []
        viewer_colors: list[str] = []
        result = self._clustering_result_state(workspace)
        assignments = _wyckoff_assignments_from_payload(result.get("wyckoff_assignments"))
        cluster_items: dict[int, QTreeWidgetItem] = {}

        def cluster_item_for(label: int) -> QTreeWidgetItem | None:
            if not label:
                return None
            existing = cluster_items.get(int(label))
            if existing is not None:
                return existing
            match = assignments.get(int(label))
            title = f"Cluster {int(label)}: {_wyckoff_assignment_text(match)}"
            item = QTreeWidgetItem([title, "", "", "", "", "", ""])
            item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            item.setData(0, Qt.ItemDataRole.UserRole, int(label))
            item.setToolTip(0, _wyckoff_assignment_tooltip(match))
            table.addTopLevelItem(item)
            item.setFirstColumnSpanned(True)
            item.setExpanded(True)
            cluster_items[int(label)] = item
            return item

        for index, site in enumerate(sites):
            shifted = anchored[index] if index < len(anchored) else None
            fractional = (
                shifted.shifted_fractional_position
                if show_shifted and shifted is not None
                else site.folded_fractional_position
            )
            if shifted is not None and show_shifted:
                viewer_positions.append((np.asarray(fractional, dtype=float) @ lattice).tolist())
            else:
                viewer_positions.append(list(site.cartesian_position_ang))
            label = labels[index] if index < len(labels) else 0
            viewer_colors.append(colors.get(int(label), current_palette().accent))
            values = [
                str(index + 1),
                "" if not label else str(label),
                site.job_label,
                _format_energy(site.relative_energy_ev),
                f"{fractional[0]:.4f}",
                f"{fractional[1]:.4f}",
                f"{fractional[2]:.4f}",
            ]
            row_item = QTreeWidgetItem(values)
            for column, value in enumerate(values):
                if shifted is not None and shifted.shifted and show_shifted and column in (self._FX_COLUMN, self._FY_COLUMN, self._FZ_COLUMN):
                    row_item.setForeground(column, status_brush("submitting"))
                if column == self._CLUSTER_COLUMN and label:
                    row_item.setForeground(column, status_brush("running"))
            parent_item = cluster_item_for(int(label))
            if parent_item is None:
                table.addTopLevelItem(row_item)
            else:
                parent_item.addChild(row_item)
        for column in range(table.columnCount()):
            if column != self._LABEL_COLUMN:
                table.resizeColumnToContents(column)
        table.header().setSectionResizeMode(self._LABEL_COLUMN, QHeaderView.ResizeMode.Stretch)
        if self._host_structure is not None:
            viewer.set_structure(self._host_structure)
            viewer.set_point_overlays(
                [
                    {
                        "positions": viewer_positions,
                        "colors": viewer_colors,
                        "radii": [MUON_VIEWER_RADIUS_ANG] * len(viewer_positions),
                        "alpha": 0.96,
                    }
                ]
                if viewer_positions
                else []
            )

    def _rename_workspace(self, workspace_id: str, name: str) -> None:
        workspace = self._workspace_by_id(workspace_id)
        if workspace is None:
            return
        cleaned = str(name).strip() or "Relaxation analysis"
        if workspace.get("name") == cleaned:
            return
        workspace["name"] = cleaned
        self._emit_state()
        self._rebuild_tabs()
        self.focus_workspace(workspace_id)

    def _set_show_shifted(self, workspace_id: str, checked: bool) -> None:
        workspace = self._workspace_by_id(workspace_id)
        if workspace is None:
            return
        workspace["show_shifted_positions"] = bool(checked)
        self._emit_state()
        self._rebuild_tabs()
        self.focus_workspace(workspace_id)

    def add_job_ids_to_workspace(self, workspace_id: str, job_ids: Sequence[UUID]) -> None:
        workspace = self._workspace_by_id(workspace_id)
        if workspace is None:
            return
        current = _workspace_job_ids(workspace)
        seen = set(current)
        for job_id in job_ids:
            if job_id not in seen:
                current.append(job_id)
                seen.add(job_id)
        workspace["job_ids"] = [str(job_id) for job_id in current]
        self._emit_state()
        self._rebuild_tabs()
        self.focus_workspace(workspace_id)

    def _open_clustering(self, workspace_id: str) -> None:
        workspace = self._workspace_by_id(workspace_id)
        if workspace is None or self._host_structure is None:
            return
        sites = self._workspace_sites(workspace)
        if not sites:
            QMessageBox.warning(self, "No sites", "This workspace does not contain any completed relaxations.")
            return
        dialog = RelaxationClusteringDialog(
            workspace=workspace,
            host_structure=self._host_structure,
            sites=sites,
            parent=self,
        )
        dialog.clustering_applied.connect(lambda clustering, wid=workspace_id: self._apply_clustering(wid, clustering))
        dialog.exec()

    def _apply_clustering(self, workspace_id: str, clustering: dict) -> None:
        workspace = self._workspace_by_id(workspace_id)
        if workspace is None:
            return
        workspace["clustering"] = dict(clustering)
        self._emit_state()
        self._rebuild_tabs()
        self.focus_workspace(workspace_id)

    def _open_displacements(self, workspace_id: str) -> None:
        workspace = self._workspace_by_id(workspace_id)
        if workspace is None:
            return
        sites = self._workspace_sites(workspace)
        if not sites:
            QMessageBox.information(
                self,
                "No sites",
                "This workspace does not contain any completed relaxations.",
            )
            return
        dialog = DisplacementAnalysisDialog(
            workspace_name=_workspace_name(workspace),
            sites=sites,
            structures_by_id=self._structures_by_id,
            comparison_options=self._comparison_options,
            parent=self,
        )
        dialog.exec()

    def _on_current_tab_changed(self, _index: int) -> None:
        pass

    def _emit_state(self) -> None:
        if self._restoring:
            return
        self.state_changed.emit([dict(workspace) for workspace in self._workspaces])

    def shutdown(self) -> None:
        for index in range(self._workspace_tabs.count()):
            tab = self._workspace_tabs.widget(index)
            if tab is None:
                continue
            for viewer in tab.findChildren(AtomicStructureViewer):
                viewer.shutdown()
