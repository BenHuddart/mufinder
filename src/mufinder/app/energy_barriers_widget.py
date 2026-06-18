"""Energy Barriers workflow surface."""

from __future__ import annotations

from dataclasses import dataclass

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QKeySequence, QMouseEvent, QPainter, QPainterPath, QPen, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QComboBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QPushButton,
    QSlider,
    QSplitter,
    QTabWidget,
    QTableWidget,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.atomic_structure_viewer import AtomicStructureViewer
from mufinder.app.selection_snapshot import SelectionSnapshotPushButton, resolve_target_keys
from mufinder.app.theme import current_palette, style_hint_text, style_section_subtitle, style_section_title
from mufinder.app.workflow_table_utils import (
    create_workflow_table,
    set_workflow_table_item,
    style_workflow_configure_button,
    workflow_row_is_locked,
)
from mufinder.domain.models import Structure


@dataclass(frozen=True, slots=True)
class EnergyBarrierEndpointRow:
    """One completed relaxed structure available as an energy-barrier endpoint."""

    key: str
    label: str
    charge_state: str
    source_job: str
    relative_energy: str


@dataclass(frozen=True, slots=True)
class EnergyBarrierSourceRow:
    """One diffusion path eligible for NEB, shown in the Energy Barriers view."""

    key: str
    reactant: str
    product: str
    charge_state: str
    path_length: str
    neb_status: str
    configured: bool = False


@dataclass(frozen=True, slots=True)
class EnergyBarrierNebResultRow:
    """One completed NEB result row."""

    key: str
    label: str
    barrier_height: str
    images: str
    transition_state_file: str


@dataclass(frozen=True, slots=True)
class EnergyBarrierPreviewImageRow:
    """One image entry in the NEB preview panel."""

    label: str
    reaction_coordinate: str
    energy: str
    reaction_coordinate_value: float
    energy_value: float


class _ReactionCoordinateProfile(QWidget):
    """Simple reaction-coordinate profile plot used by the NEB inspector."""

    point_selected = Signal(int)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._rows: list[EnergyBarrierPreviewImageRow] = []
        self._selected_index = 0
        self.setMinimumHeight(180)

    def clear_profile(self) -> None:
        self._rows = []
        self._selected_index = 0
        self.update()

    def set_profile(self, rows: list[EnergyBarrierPreviewImageRow], *, selected_index: int = 0) -> None:
        self._rows = list(rows)
        self._selected_index = max(0, min(selected_index, len(self._rows) - 1)) if self._rows else 0
        self.update()

    def point_count(self) -> int:
        return len(self._rows)

    def set_selected_index(self, index: int) -> None:
        if not self._rows:
            self._selected_index = 0
        else:
            self._selected_index = max(0, min(index, len(self._rows) - 1))
        self.update()

    def mousePressEvent(self, event: QMouseEvent) -> None:  # noqa: N802 - Qt API naming.
        if event.button() != Qt.MouseButton.LeftButton or not self._rows:
            super().mousePressEvent(event)
            return
        points = self._plot_points()
        if not points:
            super().mousePressEvent(event)
            return
        target = event.position()
        nearest_index = min(
            range(len(points)),
            key=lambda index: (points[index][0] - target.x()) ** 2 + (points[index][1] - target.y()) ** 2,
        )
        self.point_selected.emit(nearest_index)
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        super().paintEvent(event)
        palette = current_palette()
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, True)
        painter.fillRect(self.rect(), QColor(palette.bg_primary))

        bounds = self.rect().adjusted(18, 12, -12, -28)
        axis_color = QColor(palette.border)
        text_color = QColor(palette.text_muted)
        series_color = QColor(palette.accent)
        highlight_color = QColor(palette.success)

        painter.setPen(QPen(axis_color, 1.0))
        painter.drawLine(bounds.bottomLeft(), bounds.bottomRight())
        painter.drawLine(bounds.bottomLeft(), bounds.topLeft())

        if not self._rows:
            painter.setPen(text_color)
            painter.drawText(bounds, int(Qt.AlignmentFlag.AlignCenter), "No profile loaded")
            painter.end()
            return

        points = self._plot_points(bounds)
        if len(points) >= 2:
            path = QPainterPath()
            path.moveTo(*points[0])
            for point in points[1:]:
                path.lineTo(*point)
            painter.setPen(QPen(series_color, 2.0))
            painter.drawPath(path)

        for index, point in enumerate(points):
            color = highlight_color if index == self._selected_index else series_color
            painter.setBrush(color)
            painter.setPen(QPen(QColor(palette.bg_primary), 1.5))
            painter.drawEllipse(int(point[0] - 4), int(point[1] - 4), 8, 8)

        painter.setPen(text_color)
        painter.drawText(bounds.adjusted(0, bounds.height() + 4, 0, 22), int(Qt.AlignmentFlag.AlignLeft), "Reaction coordinate")
        painter.save()
        painter.translate(6, bounds.center().y())
        painter.rotate(-90)
        painter.drawText(0, 0, "Energy (eV)")
        painter.restore()
        painter.end()

    def _plot_points(self, bounds=None) -> list[tuple[float, float]]:
        if not self._rows:
            return []
        plot_bounds = self.rect().adjusted(18, 12, -12, -28) if bounds is None else bounds
        coordinates = [row.reaction_coordinate_value for row in self._rows]
        energies = [row.energy_value for row in self._rows]
        min_x = min(coordinates)
        max_x = max(coordinates)
        min_y = min(energies)
        max_y = max(energies)
        if abs(max_x - min_x) < 1e-12:
            min_x -= 0.5
            max_x += 0.5
        if abs(max_y - min_y) < 1e-12:
            min_y -= 0.5
            max_y += 0.5
        points: list[tuple[float, float]] = []
        for row in self._rows:
            x_fraction = (row.reaction_coordinate_value - min_x) / (max_x - min_x)
            y_fraction = (row.energy_value - min_y) / (max_y - min_y)
            x = plot_bounds.left() + x_fraction * plot_bounds.width()
            y = plot_bounds.bottom() - y_fraction * plot_bounds.height()
            points.append((x, y))
        return points


class EnergyBarriersWidget(QWidget):
    """Display eligible path sources and completed transition-state results."""

    endpoint_selection_changed = Signal(str, str)
    generate_paths_requested = Signal(str, str, str)
    remove_sources_requested = Signal(list)
    view_source_requested = Signal(str)
    configure_result_requested = Signal(object)
    submit_neb_local_requested = Signal(list)
    submit_neb_cluster_requested = Signal(list)
    neb_result_selected = Signal(str)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setAccessibleName("Energy Barriers workflow")
        self._endpoint_rows: list[EnergyBarrierEndpointRow] = []
        self._source_rows: list[EnergyBarrierSourceRow] = []
        self._neb_result_rows: list[EnergyBarrierNebResultRow] = []
        self._preview_structures: list[Structure] = []
        self._preview_rows: list[EnergyBarrierPreviewImageRow] = []
        self._preview_viewer: AtomicStructureViewer | None = None

        layout = QVBoxLayout(self)
        layout.setContentsMargins(6, 6, 6, 6)
        layout.setSpacing(6)

        header = QVBoxLayout()
        header.setContentsMargins(0, 0, 0, 0)
        header.setSpacing(3)

        title = QLabel("Energy Barriers")
        style_section_title(title)
        header.addWidget(title)

        subtitle = QLabel(
            "Build NEB jobs from two relaxed muon endpoints, then inspect the minimum-energy path inside MuFinder."
        )
        subtitle.setWordWrap(True)
        style_section_subtitle(subtitle)
        header.addWidget(subtitle)
        layout.addLayout(header)

        self._tabs = QTabWidget(self)
        self._tabs.setObjectName("energyBarriersTabs")
        self._tabs.setDocumentMode(False)
        self._tabs.setUsesScrollButtons(True)
        self._tabs.tabBar().setExpanding(False)
        self._tabs.setElideMode(Qt.TextElideMode.ElideRight)
        layout.addWidget(self._tabs, 1)

        paths_tab = QWidget(self._tabs)
        paths_layout = QVBoxLayout(paths_tab)
        paths_layout.setContentsMargins(10, 10, 10, 10)
        paths_layout.setSpacing(8)

        path_intro = QLabel(
            "Select two relaxed endpoints with the same charge state, generate symmetry-distinct paths, then configure or preview each NEB path."
        )
        path_intro.setWordWrap(True)
        style_hint_text(path_intro)
        paths_layout.addWidget(path_intro)

        endpoint_group = QGroupBox("Path Endpoints")
        endpoint_layout = QVBoxLayout(endpoint_group)
        endpoint_layout.setContentsMargins(10, 8, 10, 8)
        endpoint_layout.setSpacing(6)

        self._endpoint_label = QLabel("Relaxed structures")
        style_section_title(self._endpoint_label)
        endpoint_layout.addWidget(self._endpoint_label)
        endpoint_hint = QLabel(
            "MuFinder only generates paths after you choose both endpoints explicitly. This keeps path generation pair-scoped and traceable."
        )
        endpoint_hint.setWordWrap(True)
        style_hint_text(endpoint_hint)
        endpoint_layout.addWidget(endpoint_hint)

        endpoint_form = QFormLayout()
        endpoint_form.setHorizontalSpacing(10)
        endpoint_form.setVerticalSpacing(8)
        self._reactant_combo = QComboBox(self)
        self._reactant_combo.currentIndexChanged.connect(self._emit_endpoint_selection_changed)
        endpoint_form.addRow("Initial", self._reactant_combo)
        self._product_combo = QComboBox(self)
        self._product_combo.currentIndexChanged.connect(self._emit_endpoint_selection_changed)
        endpoint_form.addRow("Final", self._product_combo)
        self._max_path_length_input = QLineEdit(self)
        self._max_path_length_input.setPlaceholderText("Auto")
        self._max_path_length_input.setClearButtonEnabled(True)
        self._max_path_length_input.setToolTip(
            "Optional maximum path length in angstrom. Leave blank to use MuFinder's automatic cutoff."
        )
        endpoint_form.addRow("Max path length (ang)", self._max_path_length_input)
        endpoint_layout.addLayout(endpoint_form)

        endpoint_actions = QHBoxLayout()
        endpoint_actions.setSpacing(8)
        self._generate_paths_button = QPushButton("Generate Paths")
        self._generate_paths_button.clicked.connect(self._emit_generate_paths)
        endpoint_actions.addWidget(self._generate_paths_button)
        endpoint_actions.addStretch(1)
        endpoint_layout.addLayout(endpoint_actions)
        paths_layout.addWidget(endpoint_group)

        sources_group = QGroupBox("Diffusion Paths")
        sources_layout = QVBoxLayout(sources_group)
        sources_layout.setContentsMargins(10, 8, 10, 8)
        sources_layout.setSpacing(6)

        self._source_label = QLabel("Generated paths")
        style_section_title(self._source_label)
        sources_layout.addWidget(self._source_label)
        source_hint = QLabel(
            "Each row is a candidate hop between two relaxed endpoints. Configure the climbing-image "
            "NEB controls (image count, spring constant, tolerances) per path, then build the NEB job "
            "locally or on the cluster. Use View to preview a path in the undistorted supercell."
        )
        source_hint.setWordWrap(True)
        style_hint_text(source_hint)
        sources_layout.addWidget(source_hint)
        self._sources_table = create_workflow_table(
            ("Reactant", "Product", "Charge", "Path Length", "NEB Status", "Parameters", "View"),
            resize_to_contents_columns=(5, 6),
        )
        self._sources_table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self._sources_table.itemSelectionChanged.connect(self._update_actions)
        sources_layout.addWidget(self._sources_table, 1)

        actions = QHBoxLayout()
        actions.setSpacing(8)
        self._batch_configure_results_button = QPushButton("Configure Selected Parameters...")
        self._remove_sources_button = QPushButton("Delete Selected Paths")
        self._submit_neb_local_button = QPushButton("Build NEB Locally")
        self._submit_neb_cluster_button = QPushButton("Build NEB on Cluster")
        self._batch_configure_results_button.clicked.connect(self._emit_batch_configure_results)
        self._remove_sources_button.clicked.connect(self._delete_selected_sources)
        self._submit_neb_local_button.clicked.connect(self._emit_submit_neb_local)
        self._submit_neb_cluster_button.clicked.connect(self._emit_submit_neb_cluster)
        actions.addWidget(self._remove_sources_button)
        actions.addWidget(self._submit_neb_local_button)
        actions.addWidget(self._submit_neb_cluster_button)
        actions.addStretch(1)
        actions.addWidget(self._batch_configure_results_button)
        sources_layout.addLayout(actions)
        paths_layout.addWidget(sources_group, 1)
        self._tabs.addTab(paths_tab, "Paths")

        neb_tab = QWidget(self._tabs)
        neb_layout = QVBoxLayout(neb_tab)
        neb_layout.setContentsMargins(10, 10, 10, 10)
        neb_layout.setSpacing(8)

        neb_intro = QLabel(
            "Inspect completed NEB profiles by stepping through images, the synchronized reaction-coordinate plot, and the shared structure viewer."
        )
        neb_intro.setWordWrap(True)
        style_hint_text(neb_intro)
        neb_layout.addWidget(neb_intro)

        neb_results_group = QGroupBox("Completed NEB Calculations")
        neb_results_layout = QVBoxLayout(neb_results_group)
        neb_results_layout.setContentsMargins(10, 8, 10, 8)
        neb_results_layout.setSpacing(6)

        self._neb_result_label = QLabel("Completed NEB calculations")
        style_section_title(self._neb_result_label)
        neb_results_layout.addWidget(self._neb_result_label)
        self._neb_results_table = create_workflow_table(("Job", "Barrier", "Images", "TS File"))
        self._neb_results_table.itemSelectionChanged.connect(self._emit_neb_result_selected)
        neb_results_layout.addWidget(self._neb_results_table)
        neb_layout.addWidget(neb_results_group)

        preview_group = QGroupBox("NEB Path Inspector")
        preview_layout = QVBoxLayout(preview_group)
        preview_layout.setContentsMargins(10, 8, 10, 8)
        preview_layout.setSpacing(6)

        self._preview_label = QLabel("NEB path inspector")
        style_section_title(self._preview_label)
        preview_layout.addWidget(self._preview_label)

        self._preview_splitter = QSplitter(Qt.Orientation.Horizontal, self)
        self._preview_splitter.setChildrenCollapsible(False)
        controls = QWidget(self._preview_splitter)
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 0, 0)
        controls_layout.setSpacing(10)

        self._preview_title = QLabel("Select a completed NEB result to inspect the minimum-energy path.")
        self._preview_title.setWordWrap(True)
        controls_layout.addWidget(self._preview_title)

        self._preview_summary = QLabel("")
        self._preview_summary.setWordWrap(True)
        style_hint_text(self._preview_summary)
        controls_layout.addWidget(self._preview_summary)

        self._profile_plot = _ReactionCoordinateProfile(controls)
        self._profile_plot.point_selected.connect(self._apply_preview_index)
        controls_layout.addWidget(self._profile_plot)

        slider_row = QHBoxLayout()
        slider_row.setContentsMargins(18, 0, 12, 0)
        slider_row.setSpacing(0)
        self._preview_slider = QSlider(Qt.Orientation.Horizontal, controls)
        self._preview_slider.setSingleStep(1)
        self._preview_slider.setPageStep(1)
        self._preview_slider.valueChanged.connect(self._on_preview_slider_changed)
        slider_row.addWidget(self._preview_slider, 1)
        controls_layout.addLayout(slider_row)

        slider_labels = QHBoxLayout()
        slider_labels.setContentsMargins(18, 0, 12, 0)
        slider_labels.setSpacing(8)
        self._preview_slider_start_label = QLabel("", controls)
        self._preview_slider_end_label = QLabel("", controls)
        style_hint_text(self._preview_slider_start_label)
        style_hint_text(self._preview_slider_end_label)
        slider_labels.addWidget(self._preview_slider_start_label)
        slider_labels.addStretch(1)
        slider_labels.addWidget(self._preview_slider_end_label)
        controls_layout.addLayout(slider_labels)

        self._preview_viewer_placeholder = QLabel(
            "Select a completed NEB result to load the 3D structure preview.",
            self._preview_splitter,
        )
        self._preview_viewer_placeholder.setWordWrap(True)
        self._preview_viewer_placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        style_hint_text(self._preview_viewer_placeholder)
        self._preview_splitter.addWidget(controls)
        self._preview_splitter.addWidget(self._preview_viewer_placeholder)
        self._preview_splitter.setStretchFactor(0, 0)
        self._preview_splitter.setStretchFactor(1, 1)
        self._preview_splitter.setSizes([420, 820])
        preview_layout.addWidget(self._preview_splitter, 1)
        neb_layout.addWidget(preview_group, 1)
        self._tabs.addTab(neb_tab, "NEB Analysis")

        self.clear_neb_preview()
        self._apply_responsive_layout()
        self._delete_shortcut = QShortcut(QKeySequence.StandardKey.Delete, self)
        self._delete_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._delete_shortcut.activated.connect(self._delete_selected_sources)
        self._backspace_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Backspace), self)
        self._backspace_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._backspace_shortcut.activated.connect(self._delete_selected_sources)

    def _apply_responsive_layout(self) -> None:
        if not hasattr(self, "_preview_splitter"):
            return
        if self._preview_splitter.orientation() != Qt.Orientation.Horizontal:
            self._preview_splitter.setOrientation(Qt.Orientation.Horizontal)
        self._preview_splitter.setSizes([420, 820])

    def set_endpoint_options(
        self,
        rows: list[EnergyBarrierEndpointRow],
        *,
        reactant_key: str = "",
        product_key: str = "",
        max_path_length_text: str = "",
    ) -> None:
        self._endpoint_rows = list(rows)
        reactant_key = reactant_key.strip()
        product_key = product_key.strip()
        current_reactant = reactant_key or self.selected_endpoint_keys()[0]
        current_product = product_key or self.selected_endpoint_keys()[1]
        if self._max_path_length_input.text().strip() != max_path_length_text.strip():
            self._max_path_length_input.setText(max_path_length_text.strip())

        self._reactant_combo.blockSignals(True)
        self._product_combo.blockSignals(True)
        self._reactant_combo.clear()
        self._product_combo.clear()
        for row in rows:
            label = f"{row.label} | {row.charge_state} | {row.source_job} | {row.relative_energy}"
            self._reactant_combo.addItem(label, row.key)
            self._product_combo.addItem(label, row.key)

        self._set_combo_key(self._reactant_combo, current_reactant, fallback_index=0)
        if len(rows) >= 2:
            fallback_index = 1 if self._reactant_combo.currentIndex() == 0 else 0
        else:
            fallback_index = 0
        self._set_combo_key(self._product_combo, current_product, fallback_index=fallback_index)
        self._reactant_combo.blockSignals(False)
        self._product_combo.blockSignals(False)
        self._update_actions()

    def set_sources(self, rows: list[EnergyBarrierSourceRow]) -> None:
        self._source_rows = list(rows)
        self._source_label.setText(f"Generated paths ({len(rows)})")
        _fill_source_table(
            self._sources_table,
            rows,
            configure_requested=self.configure_result_requested.emit,
            view_requested=self.view_source_requested.emit,
        )
        self._update_actions()

    def set_neb_results(self, rows: list[EnergyBarrierNebResultRow]) -> None:
        previous_key = next(iter(self.selected_neb_result_keys()), "")
        self._neb_result_rows = list(rows)
        self._neb_result_label.setText(f"Completed NEB calculations ({len(rows)})")
        _fill_table(
            self._neb_results_table,
            [
                (row.label, row.barrier_height, row.images, row.transition_state_file)
                for row in rows
            ],
        )
        target_row = 0 if rows else -1
        if previous_key:
            for index, row in enumerate(rows):
                if row.key == previous_key:
                    target_row = index
                    break
        if target_row >= 0:
            self._neb_results_table.selectRow(target_row)
        else:
            self.clear_neb_preview()

    def set_neb_preview(
        self,
        *,
        title: str,
        image_rows: list[EnergyBarrierPreviewImageRow],
        structures: list[Structure],
        selected_index: int = 0,
    ) -> None:
        self._preview_title.setText(title)
        self._preview_rows = list(image_rows)
        self._preview_structures = list(structures)
        if not image_rows or not structures:
            self.clear_neb_preview(title=title)
            return
        self._preview_slider.setRange(0, len(image_rows) - 1)
        self._preview_slider_start_label.setText(image_rows[0].reaction_coordinate)
        self._preview_slider_end_label.setText(image_rows[-1].reaction_coordinate)
        self._apply_preview_index(selected_index)

    def clear_neb_preview(self, *, title: str | None = None) -> None:
        self._preview_rows = []
        self._preview_structures = []
        self._preview_title.setText(title or "Select a completed NEB result to inspect the minimum-energy path.")
        self._preview_summary.setText("")
        self._preview_slider.blockSignals(True)
        self._preview_slider.setRange(0, 0)
        self._preview_slider.setValue(0)
        self._preview_slider.blockSignals(False)
        self._preview_slider_start_label.setText("")
        self._preview_slider_end_label.setText("")
        self._profile_plot.clear_profile()
        if self._preview_viewer is not None:
            self._preview_viewer.set_structure(None)

    def selected_endpoint_keys(self) -> tuple[str, str]:
        reactant = self._reactant_combo.currentData()
        product = self._product_combo.currentData()
        return (
            str(reactant) if isinstance(reactant, str) else "",
            str(product) if isinstance(product, str) else "",
        )

    def selected_source_keys(self) -> list[str]:
        indexes = sorted({index.row() for index in self._sources_table.selectedIndexes()})
        keys: list[str] = []
        for row in indexes:
            if 0 <= row < len(self._source_rows):
                keys.append(self._source_rows[row].key)
        return keys

    def _emit_endpoint_selection_changed(self) -> None:
        reactant_key, product_key = self.selected_endpoint_keys()
        self.endpoint_selection_changed.emit(reactant_key, product_key)
        self._update_actions()

    def _emit_generate_paths(self) -> None:
        reactant_key, product_key = self.selected_endpoint_keys()
        self.generate_paths_requested.emit(reactant_key, product_key, self._max_path_length_input.text().strip())

    def _delete_selected_sources(self) -> None:
        keys = self.selected_source_keys()
        if not keys:
            return
        removed = set(keys)
        self.set_sources([row for row in self._source_rows if row.key not in removed])
        self.remove_sources_requested.emit(keys)

    def _emit_submit_neb_local(self) -> None:
        self.submit_neb_local_requested.emit(self.selected_source_keys())

    def _emit_batch_configure_results(self) -> None:
        keys = self.selected_source_keys()
        if keys:
            self.configure_result_requested.emit(keys)

    def _emit_submit_neb_cluster(self) -> None:
        self.submit_neb_cluster_requested.emit(self.selected_source_keys())

    def selected_neb_result_keys(self) -> list[str]:
        indexes = sorted({index.row() for index in self._neb_results_table.selectedIndexes()})
        keys: list[str] = []
        for row in indexes:
            if 0 <= row < len(self._neb_result_rows):
                keys.append(self._neb_result_rows[row].key)
        return keys

    def _emit_neb_result_selected(self) -> None:
        self._update_actions()
        keys = self.selected_neb_result_keys()
        self.neb_result_selected.emit(keys[0] if keys else "")

    def _on_preview_slider_changed(self, value: int) -> None:
        self._apply_preview_index(value)

    def _apply_preview_index(self, index: int) -> None:
        if not self._preview_rows or not self._preview_structures:
            self.clear_neb_preview(title=self._preview_title.text())
            return
        clamped = max(0, min(index, len(self._preview_rows) - 1, len(self._preview_structures) - 1))
        self._preview_slider.blockSignals(True)
        self._preview_slider.setValue(clamped)
        self._preview_slider.blockSignals(False)

        row = self._preview_rows[clamped]
        structure = self._preview_structures[clamped]
        self._preview_summary.setText(
            f"{row.label} | Reaction coordinate {row.reaction_coordinate} | Energy {row.energy}"
        )
        self._profile_plot.set_profile(self._preview_rows, selected_index=clamped)
        preview_viewer = self._ensure_preview_viewer()
        preview_viewer.set_structure(structure)
        preview_viewer.select_atoms(_muon_atom_indexes(structure))

    def closeEvent(self, event) -> None:  # noqa: N802 - Qt API naming.
        self._shutdown_preview_viewer()
        super().closeEvent(event)

    def _ensure_preview_viewer(self) -> AtomicStructureViewer:
        if self._preview_viewer is not None:
            return self._preview_viewer

        viewer = AtomicStructureViewer(self._preview_splitter)
        self._preview_splitter.replaceWidget(1, viewer)
        self._preview_viewer_placeholder.deleteLater()
        self._preview_viewer = viewer
        self._preview_splitter.setStretchFactor(1, 1)
        self._preview_splitter.setSizes([420, 820])
        return viewer

    def _shutdown_preview_viewer(self) -> None:
        if self._preview_viewer is None:
            return
        try:
            self._preview_viewer.shutdown()
        except RuntimeError:
            pass
        self._preview_viewer = None

    def _update_actions(self) -> None:
        reactant_key, product_key = self.selected_endpoint_keys()
        source_enabled = bool(self.selected_source_keys())
        can_generate = bool(reactant_key and product_key and len(self._endpoint_rows) >= 1)
        self._generate_paths_button.setEnabled(can_generate)
        self._remove_sources_button.setEnabled(source_enabled)
        self._batch_configure_results_button.setEnabled(source_enabled)
        self._submit_neb_local_button.setEnabled(source_enabled)
        self._submit_neb_cluster_button.setEnabled(source_enabled)

    @staticmethod
    def _set_combo_key(combo: QComboBox, key: str, *, fallback_index: int) -> None:
        if key:
            for index in range(combo.count()):
                data = combo.itemData(index)
                if isinstance(data, str) and data == key:
                    combo.setCurrentIndex(index)
                    return
        if combo.count():
            combo.setCurrentIndex(max(0, min(fallback_index, combo.count() - 1)))

def _fill_source_table(
    table: QTableWidget,
    rows: list[EnergyBarrierSourceRow],
    *,
    configure_requested,
    view_requested,
) -> None:
    table.setRowCount(len(rows))
    key_by_row: dict[int, str] = {i: row.key for i, row in enumerate(rows)}
    for row_index, row in enumerate(rows):
        for column, value in enumerate((row.reactant, row.product, row.charge_state, row.path_length, row.neb_status)):
            set_workflow_table_item(table, row_index, column, value)
        configure_button = SelectionSnapshotPushButton(table)
        configure_button.selection_snapshot_key = row.key
        style_workflow_configure_button(
            configure_button,
            workflow_label="Energy Barriers",
            configured=row.configured,
            locked=workflow_row_is_locked(row.neb_status),
        )
        configure_button.clicked.connect(
            lambda _checked=False, key=row.key, btn=configure_button: configure_requested(
                list(resolve_target_keys(
                    btn,
                    clicked_key=key,
                    snapshot_keys=[key_by_row[r] for r in btn.selection_snapshot_rows if r in key_by_row],
                    eligibility=lambda k: True,
                    parent=table,
                    title="Configure Energy Barriers",
                    message=None,
                ))
            )
        )
        table.setCellWidget(row_index, 5, configure_button)

        view_button = SelectionSnapshotPushButton("View...", table)
        view_button.selection_snapshot_key = row.key
        view_button.setToolTip("Open the path viewer for this diffusion path.")
        view_button.clicked.connect(
            lambda _checked=False, key=row.key, btn=view_button: _open_view_single_row(
                btn, key, key_by_row, view_requested, table
            )
        )
        table.setCellWidget(row_index, 6, view_button)
    table.resizeRowsToContents()


def _open_view_single_row(
    btn: SelectionSnapshotPushButton,
    key: str,
    key_by_row: dict[int, str],
    view_requested,
    table: QTableWidget,
) -> None:
    snapshot_keys = [key_by_row[r] for r in btn.selection_snapshot_rows if r in key_by_row]
    btn.selection_snapshot_rows = ()
    btn.selection_snapshot_key = None
    if key in snapshot_keys and len(snapshot_keys) > 1:
        response = QMessageBox.question(
            table,
            "View path",
            "The path viewer opens one row at a time. Open the viewer for this row?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No,
        )
        if response != QMessageBox.StandardButton.Yes:
            return
    view_requested(key)


def _fill_table(table: QTableWidget, rows: list[tuple[str, ...]]) -> None:
    table.setRowCount(len(rows))
    for row_index, values in enumerate(rows):
        for column, value in enumerate(values):
            set_workflow_table_item(table, row_index, column, value)
    table.resizeRowsToContents()


def _muon_atom_indexes(structure: Structure) -> list[int]:
    return [
        index
        for index, species in enumerate(structure.species)
        if str(species).strip().casefold() == "h:mu" or str(species).strip().casefold().endswith(":mu")
    ]
