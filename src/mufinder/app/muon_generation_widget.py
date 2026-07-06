"""Muon site generation panel."""

from __future__ import annotations

from typing import Any

from PySide6.QtCore import QEvent, QItemSelectionModel, Qt, Signal
from PySide6.QtGui import QKeySequence, QShortcut
from PySide6.QtWidgets import (
    QButtonGroup,
    QComboBox,
    QDialog,
    QDialogButtonBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QRadioButton,
    QSizePolicy,
    QSplitter,
    QSpinBox,
    QStackedWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.active_host_badge import ActiveHostBadge
from mufinder.app.atomic_structure_viewer import AtomicStructureViewer, MUON_VIEWER_RADIUS_ANG
from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.app.structure_render_guard import (
    NO_STRUCTURE,
    structure_geometry_fingerprint,
)
from mufinder.app.theme import refresh_widget_style, style_section_subtitle, style_section_title
from mufinder.app.workflow_table_utils import create_workflow_table
from mufinder.domain.models import Structure
from mufinder.science.site_generation import (
    SiteGenerationError,
    estimate_muon_muon_separation_for_count,
    lattice_frac_to_cart,
)


class _ClusterPositionsDialog(QDialog):
    """Collect the fractional anchor used to cluster selected muon positions."""

    def __init__(self, site_rows: list[dict[str, Any]], parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("clusterPositionsDialog")
        self.setWindowTitle("Cluster Positions")
        self.resize(420, 250)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 14)
        layout.setSpacing(10)

        title = QLabel("Cluster Positions", self)
        style_section_title(title)
        layout.addWidget(title)

        subtitle = QLabel(
            "Specify an anchor in fractional coordinates. Selected muon sites will be replaced by their "
            "symmetry-equivalent site that is closest to the anchor.",
            self,
        )
        style_section_subtitle(subtitle)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        card = QWidget(self)
        card.setObjectName("clusterPositionsCard")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(12, 12, 12, 12)
        card_layout.setSpacing(8)

        form = QFormLayout()
        form.setContentsMargins(0, 0, 0, 0)
        form.setHorizontalSpacing(10)
        form.setVerticalSpacing(6)
        self._anchor_source = QComboBox(self)
        self._anchor_source.addItem("Manual anchor", None)
        for row in site_rows:
            fractional = row.get("fractional_position", row.get("fractional"))
            if not isinstance(fractional, (list, tuple)) or len(fractional) != 3:
                continue
            label = str(row.get("label", "Muon position")).strip() or "Muon position"
            self._anchor_source.addItem(label, [float(value) for value in fractional[:3]])
        form.addRow("Anchor", self._anchor_source)

        self._anchor_x = DecimalLineEdit()
        self._anchor_y = DecimalLineEdit()
        self._anchor_z = DecimalLineEdit()
        for control in (self._anchor_x, self._anchor_y, self._anchor_z):
            control.setRange(-10000.0, 10000.0)
            control.setDecimals(6)
        form.addRow("fX", self._anchor_x)
        form.addRow("fY", self._anchor_y)
        form.addRow("fZ", self._anchor_z)
        card_layout.addLayout(form)
        layout.addWidget(card)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel,
            self,
        )
        buttons.button(QDialogButtonBox.StandardButton.Ok).setText("Cluster Positions")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

        self._anchor_source.currentIndexChanged.connect(self._on_anchor_source_changed)

    def _on_anchor_source_changed(self) -> None:
        fractional = self._anchor_source.currentData()
        if not isinstance(fractional, (list, tuple)) or len(fractional) != 3:
            return
        for control, value in zip(
            (self._anchor_x, self._anchor_y, self._anchor_z),
            fractional,
            strict=True,
        ):
            control.setValue(float(value))

    def anchor_fractional(self) -> list[float]:
        return [self._anchor_x.value(), self._anchor_y.value(), self._anchor_z.value()]


class MuonGenerationWidget(QWidget):
    """Form-based UI for generated muon sites."""

    generate_requested = Signal(str, dict)
    cluster_positions_requested = Signal(object)
    delete_candidates_requested = Signal(object)
    create_jobs_requested = Signal(object)
    form_state_changed = Signal(dict)

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)

        self._method_combo = QComboBox()
        self._method_combo.addItem("Random generation", "random")
        self._method_combo.addItem("Attach muon to element", "near_element")
        self._method_combo.addItem("Custom coordinates", "manual")
        self._method_combo.addItem("UEP clusters", "uep_clusters")
        self._host_structure: Structure | None = None
        self._active_host: Structure | None = None
        self._site_rows: list[dict[str, Any]] = []
        self._uep_cluster_rows: list[dict[str, Any]] = []
        # Flush-path guards: skip the expensive viewer/table rebuild when the
        # inputs are byte-for-byte unchanged (see structure_render_guard).
        self._host_fingerprint: tuple[object, ...] = NO_STRUCTURE
        self._host_fingerprint_valid = False
        self._restoring_form_state = False
        self._active_host_badge = ActiveHostBadge()

        self._param_stack = QStackedWidget()
        self._param_stack.setObjectName("muonGenerationParamStack")

        self._random_by_separation = QRadioButton("Muon-muon separation (Å)")
        self._random_by_count = QRadioButton("Number of initial positions")
        self._random_mode_group = QButtonGroup(self)
        self._random_mode_group.setExclusive(True)
        self._random_mode_group.addButton(self._random_by_separation)
        self._random_mode_group.addButton(self._random_by_count)
        self._random_by_separation.setChecked(True)

        self._random_count = QSpinBox()
        self._random_count.setRange(1, 2000)
        self._random_count.setValue(24)
        self._random_count.setEnabled(False)

        self._random_min_muon_muon = DecimalLineEdit()
        self._random_min_muon_muon.setRange(0.1, 10.0)
        self._random_min_muon_muon.setSingleStep(0.1)
        self._random_min_muon_muon.setValue(0.8)

        self._random_min_muon_atom = DecimalLineEdit()
        self._random_min_muon_atom.setRange(0.1, 10.0)
        self._random_min_muon_atom.setSingleStep(0.1)
        self._random_min_muon_atom.setValue(1.0)

        self._random_max_attempts = QSpinBox()
        self._random_max_attempts.setRange(100, 1_000_000)
        self._random_max_attempts.setSingleStep(100)
        self._random_max_attempts.setValue(20000)

        self._random_seed = ""

        self._near_element = QComboBox()
        self._near_element.setEditable(False)
        self._near_muons_per_atom = QSpinBox()
        self._near_muons_per_atom.setRange(1, 24)
        self._near_muons_per_atom.setValue(2)

        self._near_bond_distance = DecimalLineEdit()
        self._near_bond_distance.setRange(0.1, 5.0)
        self._near_bond_distance.setSingleStep(0.1)
        self._near_bond_distance.setValue(1.0)

        self._near_min_muon_atom = DecimalLineEdit()
        self._near_min_muon_atom.setRange(0.1, 10.0)
        self._near_min_muon_atom.setSingleStep(0.1)
        self._near_min_muon_atom.setValue(1.0)

        self._manual_coord_system = QComboBox()
        self._manual_coord_system.addItems(["fractional", "cartesian"])
        self._manual_x = DecimalLineEdit()
        self._manual_y = DecimalLineEdit()
        self._manual_z = DecimalLineEdit()
        for control in (self._manual_x, self._manual_y, self._manual_z):
            control.setRange(-10000.0, 10000.0)
            control.setDecimals(6)
        self._manual_x_label = QLabel("fX")
        self._manual_y_label = QLabel("fY")
        self._manual_z_label = QLabel("fZ")

        self._site_table = create_workflow_table(("Site", "Method", "State", "fX", "fY", "fZ"))
        self._site_table.horizontalHeaderItem(0).setToolTip("Site label")
        self._site_table.horizontalHeaderItem(1).setToolTip("Generation method")
        self._site_table.horizontalHeaderItem(2).setToolTip("Workflow state")
        self._configure_site_table_columns()
        self._delete_shortcut = QShortcut(QKeySequence.StandardKey.Delete, self)
        self._delete_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._delete_shortcut.activated.connect(self._emit_delete_selected)
        self._backspace_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Backspace), self)
        self._backspace_shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._backspace_shortcut.activated.connect(self._emit_delete_selected)
        self._clear_selection_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Escape), self._site_table)
        self._clear_selection_shortcut.activated.connect(self._clear_site_selection)
        self._site_table.viewport().installEventFilter(self)

        # Columns are sized by resizeColumnsToContents() after each repopulation
        # (see _set_uep_cluster_rows), so this table opts out of the shared
        # stretch-to-fill default in favor of Qt's interactive resize mode.
        self._uep_cluster_table = create_workflow_table(
            ("Source", "Cluster", "Size", "Class", "E (eV)", "fX", "fY", "fZ"),
            default_resize_mode=QHeaderView.ResizeMode.Interactive,
            stretch_last_section=False,
        )
        self._uep_cluster_table.setFixedHeight(128)
        self._uep_cluster_table.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._uep_cluster_table.viewport().installEventFilter(self)

        self._viewer = AtomicStructureViewer()

        self._build_layout()
        self._connect_signals()
        self._style_random_mode_rows()
        self._apply_generation_progress_style(completed=False)

    def _build_layout(self) -> None:
        layout = QVBoxLayout(self)

        header = QLabel("Initial muon position generation")
        self._header_label = header
        style_section_title(header)
        layout.addWidget(header)

        subtitle = QLabel("Generate initial muon site positions using a suite of generation methods.")
        style_section_subtitle(subtitle)
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)
        layout.addWidget(self._active_host_badge)

        splitter = QSplitter(Qt.Orientation.Horizontal, self)
        splitter.setChildrenCollapsible(False)
        left_panel = QWidget(splitter)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(6)

        viewer_panel = QWidget(splitter)
        viewer_layout = QVBoxLayout(viewer_panel)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_label = QLabel("Preview Muon Sites")
        style_section_subtitle(viewer_label)
        viewer_layout.addWidget(viewer_label)
        viewer_layout.addWidget(self._viewer, 1)

        form_box = QGroupBox("Generation Setup")
        form_box.setObjectName("muonGenerationSetupCard")
        self._form_box = form_box
        form_layout = QFormLayout(form_box)
        form_layout.setContentsMargins(10, 10, 10, 8)
        form_layout.setHorizontalSpacing(10)
        form_layout.setVerticalSpacing(4)
        form_layout.addRow("Method", self._method_combo)
        form_layout.addRow(self._param_stack)

        random_page = QWidget()
        random_page.setObjectName("muonGenerationRandomPage")
        random_layout = QFormLayout(random_page)
        self._compact_form_layout(random_layout)
        self._random_separation_row = self._build_random_choice_row(
            "randomSeparationChoice",
            self._random_by_separation,
            self._random_min_muon_muon,
        )
        self._random_count_row = self._build_random_choice_row(
            "randomCountChoice",
            self._random_by_count,
            self._random_count,
        )
        random_layout.addRow(self._random_separation_row)
        random_layout.addRow(self._random_count_row)
        random_layout.addRow("Min muon-atom distance (Å)", self._random_min_muon_atom)
        random_layout.addRow("Max attempts", self._random_max_attempts)

        near_page = QWidget()
        near_page.setObjectName("muonGenerationNearPage")
        near_layout = QFormLayout(near_page)
        self._compact_form_layout(near_layout)
        near_layout.addRow("Element", self._near_element)
        near_layout.addRow("Muons per atom", self._near_muons_per_atom)
        near_layout.addRow("Muon-element distance (Å)", self._near_bond_distance)
        near_layout.addRow("Min muon-atom distance (Å)", self._near_min_muon_atom)

        manual_page = QWidget()
        manual_page.setObjectName("muonGenerationManualPage")
        manual_layout = QFormLayout(manual_page)
        self._compact_form_layout(manual_layout)
        manual_layout.addRow("Coordinate system", self._manual_coord_system)
        manual_layout.addRow(self._manual_x_label, self._manual_x)
        manual_layout.addRow(self._manual_y_label, self._manual_y)
        manual_layout.addRow(self._manual_z_label, self._manual_z)

        uep_page = QWidget()
        uep_page.setObjectName("muonGenerationUepPage")
        uep_layout = QVBoxLayout(uep_page)
        uep_layout.setContentsMargins(0, 0, 0, 0)
        uep_layout.setSpacing(4)
        uep_hint = QLabel("Select clustered UEP rows to add as generated sites.")
        uep_hint.setWordWrap(True)
        style_section_subtitle(uep_hint)
        uep_layout.addWidget(uep_hint)
        uep_layout.addWidget(self._uep_cluster_table, 1)

        self._param_stack.addWidget(random_page)
        self._param_stack.addWidget(near_page)
        self._param_stack.addWidget(manual_page)
        self._param_stack.addWidget(uep_page)
        param_stack_height = max(
            page.sizeHint().height()
            for page in (random_page, near_page, manual_page, uep_page)
        )
        self._param_stack.setFixedHeight(param_stack_height)

        action_row = QHBoxLayout()
        action_row.setContentsMargins(0, 0, 0, 0)
        action_row.setSpacing(6)
        self._generate_btn = QPushButton("Generate Sites")
        self._generate_btn.clicked.connect(self._emit_generate)
        action_row.addWidget(self._generate_btn)
        self._cluster_btn = QPushButton("Cluster Positions")
        self._cluster_btn.setEnabled(False)
        self._cluster_btn.clicked.connect(self._open_cluster_positions_dialog)
        action_row.addWidget(self._cluster_btn)
        self._delete_btn = QPushButton("Remove Selected")
        self._delete_btn.setEnabled(False)
        self._delete_btn.clicked.connect(self._emit_delete_selected)
        action_row.addWidget(self._delete_btn)
        self._create_jobs_btn = QPushButton("Create job(s) from selected")
        self._create_jobs_btn.setEnabled(False)
        action_row.addWidget(self._create_jobs_btn)
        action_row.addStretch(1)

        self._generation_progress = QProgressBar()
        self._generation_progress.setObjectName("muonGenerationProgress")
        self._generation_progress.setRange(0, 1)
        self._generation_progress.setValue(0)
        self._generation_progress.setFormat("Ready")
        self._generation_progress.setTextVisible(True)

        left_layout.addWidget(form_box)
        left_layout.addLayout(action_row)
        left_layout.addWidget(self._generation_progress)
        left_layout.addWidget(self._site_table)
        splitter.addWidget(left_panel)
        splitter.addWidget(viewer_panel)
        splitter.setStretchFactor(0, 1)
        splitter.setStretchFactor(1, 1)
        layout.addWidget(splitter, 1)

    @staticmethod
    def _compact_form_layout(layout: QFormLayout) -> None:
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(8)
        layout.setVerticalSpacing(3)

    def _configure_site_table_columns(self) -> None:
        header = self._site_table.horizontalHeader()
        header.setStretchLastSection(False)
        header.setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        for column in range(3, 6):
            header.setSectionResizeMode(column, QHeaderView.ResizeMode.Fixed)
            self._site_table.setColumnWidth(column, 58)

    @staticmethod
    def _build_random_choice_row(row_name: str, radio: QRadioButton, editor: QWidget) -> QWidget:
        row = QWidget()
        row.setObjectName(row_name)
        row.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        radio.setMinimumWidth(170)
        radio.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        editor.setMinimumWidth(110)
        editor.setMaximumWidth(135)
        editor.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Fixed)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(6, 3, 6, 3)
        row_layout.setSpacing(8)
        row_layout.addWidget(radio, 1)
        row_layout.addWidget(editor)
        return row

    def _connect_signals(self) -> None:
        self._method_combo.currentIndexChanged.connect(self._on_method_changed)
        self._method_combo.currentIndexChanged.connect(self._emit_state)

        self._random_by_separation.toggled.connect(self._on_random_mode_changed)
        self._random_by_count.toggled.connect(self._emit_state)
        self._random_count.valueChanged.connect(self._emit_state)
        self._random_count.valueChanged.connect(self._sync_random_count_spacing)
        self._random_min_muon_muon.valueChanged.connect(self._emit_state)
        self._random_min_muon_atom.valueChanged.connect(self._emit_state)
        self._random_min_muon_atom.valueChanged.connect(self._sync_random_count_spacing)
        self._random_max_attempts.valueChanged.connect(self._emit_state)
        self._near_element.currentTextChanged.connect(self._emit_state)
        self._near_muons_per_atom.valueChanged.connect(self._emit_state)
        self._near_bond_distance.valueChanged.connect(self._emit_state)
        self._near_min_muon_atom.valueChanged.connect(self._emit_state)

        self._manual_coord_system.currentIndexChanged.connect(self._emit_state)
        self._manual_coord_system.currentIndexChanged.connect(self._sync_manual_coordinate_labels)
        self._manual_x.valueChanged.connect(self._emit_state)
        self._manual_y.valueChanged.connect(self._emit_state)
        self._manual_z.valueChanged.connect(self._emit_state)
        self._manual_coord_system.currentIndexChanged.connect(self._sync_viewer_overlays)
        self._manual_x.valueChanged.connect(self._sync_viewer_overlays)
        self._manual_y.valueChanged.connect(self._sync_viewer_overlays)
        self._manual_z.valueChanged.connect(self._sync_viewer_overlays)
        self._site_table.itemSelectionChanged.connect(self._sync_viewer_overlays)
        self._site_table.itemSelectionChanged.connect(self._update_cluster_button_state)
        self._site_table.itemSelectionChanged.connect(self._update_delete_button_state)
        self._site_table.itemSelectionChanged.connect(self._update_create_jobs_button_state)
        self._uep_cluster_table.itemSelectionChanged.connect(self._sync_viewer_overlays)
        self._create_jobs_btn.clicked.connect(self._emit_create_jobs_selected)

    def _on_method_changed(self) -> None:
        self._param_stack.setCurrentIndex(self._method_combo.currentIndex())
        self._sync_viewer_overlays()

    def eventFilter(self, watched: object, event: QEvent) -> bool:
        if watched is self._site_table.viewport() and event.type() == QEvent.Type.MouseButtonPress:
            position_getter = getattr(event, "position", None)
            position = position_getter().toPoint() if callable(position_getter) else event.pos()
            item = self._site_table.itemAt(position)
            if item is None:
                self._clear_site_selection()
            elif self._site_table.selectionModel().isRowSelected(item.row(), self._site_table.rootIndex()):
                self._deselect_site_row(item.row())
                return True
        if watched is self._uep_cluster_table.viewport() and event.type() == QEvent.Type.MouseButtonPress:
            position_getter = getattr(event, "position", None)
            position = position_getter().toPoint() if callable(position_getter) else event.pos()
            item = self._uep_cluster_table.itemAt(position)
            modifiers = getattr(event, "modifiers", lambda: Qt.KeyboardModifier.NoModifier)()
            if item is not None and modifiers == Qt.KeyboardModifier.NoModifier:
                self._toggle_uep_cluster_row(item.row())
                return True
        return super().eventFilter(watched, event)

    def _sync_manual_coordinate_labels(self) -> None:
        labels = ("X (Å)", "Y (Å)", "Z (Å)") if self._manual_coord_system.currentText() == "cartesian" else ("fX", "fY", "fZ")
        for label, text in zip(
            (self._manual_x_label, self._manual_y_label, self._manual_z_label),
            labels,
            strict=True,
        ):
            label.setText(text)

    def _current_method(self) -> str:
        method = self._method_combo.currentData()
        return str(method if method is not None else self._method_combo.currentText())

    def _set_current_method(self, method: str) -> None:
        index = self._method_combo.findData(self._normalize_method_key(method))
        if index >= 0:
            self._method_combo.setCurrentIndex(index)

    @staticmethod
    def _normalize_method_key(method: str) -> str:
        key = str(method).strip().lower().replace("-", "_").replace(" ", "_")
        aliases = {
            "random_generation": "random",
            "attach_muon_to_element": "near_element",
            "attach_to_element": "near_element",
            "custom_coordinates": "manual",
            "custom": "manual",
            "uep_cluster": "uep_clusters",
        }
        return aliases.get(key, key)

    def _on_random_mode_changed(self) -> None:
        by_separation = self._random_by_separation.isChecked()
        self._random_min_muon_muon.setEnabled(by_separation)
        self._random_count.setEnabled(not by_separation)
        self._style_random_mode_rows()
        self._sync_random_count_spacing()
        self._emit_state()

    def _style_random_mode_rows(self) -> None:
        for row, active in (
            (self._random_separation_row, self._random_by_separation.isChecked()),
            (self._random_count_row, self._random_by_count.isChecked()),
        ):
            row.setProperty("active", "true" if active else "false")
            refresh_widget_style(row)
            for child in row.findChildren(QWidget):
                refresh_widget_style(child)

    def _sync_random_count_spacing(self) -> None:
        if self._restoring_form_state or not self._random_by_count.isChecked() or self._host_structure is None:
            return
        try:
            spacing = estimate_muon_muon_separation_for_count(
                lattice_vectors=self._host_structure.lattice_vectors,
                host_positions_frac=self._host_structure.fractional_positions,
                host_species=self._host_structure.species,
                target_count=self._random_count.value(),
                min_muon_atom=self._random_min_muon_atom.value(),
                include_symmetry_equivalents=True,
            )
        except SiteGenerationError:
            return
        self.set_random_muon_muon_distance(spacing)

    def _apply_generation_progress_style(self, *, completed: bool) -> None:
        self._generation_progress.setProperty("generationState", "complete" if completed else "active")
        refresh_widget_style(self._generation_progress)

    def changeEvent(self, event: QEvent) -> None:
        super().changeEvent(event)
        if (
            event.type() in {QEvent.Type.PaletteChange, QEvent.Type.StyleChange}
            and hasattr(self, "_random_separation_row")
            and hasattr(self, "_generation_progress")
        ):
            self._style_random_mode_rows()
            self._apply_generation_progress_style(
                completed=self._generation_progress.value() >= self._generation_progress.maximum()
                and self._generation_progress.maximum() > 0
                and self._generation_progress.format() != "Ready"
            )
            # Stale-row foreground colours are palette-derived. set_sites now
            # skips rebuilds when rows are unchanged, so re-apply the styling on
            # a theme flip rather than relying on a table rebuild to refresh it.
            if hasattr(self, "_site_table"):
                self._apply_stale_row_styling()

    def set_generation_busy(self, busy: bool, message: str = "Generating sites...") -> None:
        """Show or clear the generation progress indicator."""
        if busy:
            self._apply_generation_progress_style(completed=False)
            self._generation_progress.setRange(0, 0)
            self._generation_progress.setFormat(message)
        else:
            self._apply_generation_progress_style(completed=False)
            self._generation_progress.setRange(0, 1)
            self._generation_progress.setValue(0)
            self._generation_progress.setFormat(message or "Ready")
        self.set_generation_controls_enabled(not busy)

    def set_generation_controls_enabled(self, enabled: bool) -> None:
        self._generate_btn.setEnabled(enabled)
        self._cluster_btn.setEnabled(enabled and bool(self._selected_site_ids()))
        self._delete_btn.setEnabled(enabled and bool(self._selected_site_ids()))
        self._create_jobs_btn.setEnabled(enabled and bool(self._selected_site_ids()))

    def set_generation_progress(self, generated: int, expected: int, message: str = "Generated %v/%m sites") -> None:
        """Show determinate generation progress as accepted muon-site count."""
        bounded_expected = max(int(expected), 1)
        bounded_generated = max(0, int(generated))
        if bounded_generated > bounded_expected:
            bounded_expected = bounded_generated
        self._apply_generation_progress_style(completed=bounded_generated >= bounded_expected)
        self._generation_progress.setRange(0, bounded_expected)
        self._generation_progress.setValue(min(bounded_generated, bounded_expected))
        self._generation_progress.setFormat(message)

    def set_random_muon_muon_distance(self, distance: float) -> None:
        """Reflect the random-generation spacing used by the service in the UI."""
        previous = self._random_min_muon_muon.blockSignals(True)
        self._random_min_muon_muon.setValue(float(distance))
        self._random_min_muon_muon.blockSignals(previous)
        self._emit_state()

    def _emit_generate(self) -> None:
        method = self._current_method()
        payload = self._collect_method_parameters(method)
        self.generate_requested.emit(method, payload)

    def _open_cluster_positions_dialog(self) -> None:
        site_rows = self._selected_site_rows()
        site_ids = self._selected_site_ids()
        if not site_ids:
            QMessageBox.information(self, "No positions selected", "Select one or more generated muon positions first.")
            return

        dialog = _ClusterPositionsDialog(site_rows, self)
        if dialog.exec() != QDialog.DialogCode.Accepted:
            return
        self.cluster_positions_requested.emit(
            {
                "site_ids": site_ids,
                "anchor_fractional": dialog.anchor_fractional(),
            }
        )

    def _emit_delete_selected(self) -> None:
        site_ids = self._selected_site_ids()
        if site_ids:
            self.delete_candidates_requested.emit(site_ids)

    def _emit_create_jobs_selected(self) -> None:
        site_ids = self._selected_site_ids()
        if site_ids:
            self.create_jobs_requested.emit(site_ids)

    def _clear_site_selection(self) -> None:
        self._site_table.clearSelection()
        self._sync_viewer_overlays()
        self._update_cluster_button_state()
        self._update_delete_button_state()

    def _deselect_site_row(self, row: int) -> None:
        for column in range(self._site_table.columnCount()):
            item = self._site_table.item(row, column)
            if item is not None:
                item.setSelected(False)
        self._sync_viewer_overlays()
        self._update_cluster_button_state()
        self._update_delete_button_state()

    def _toggle_uep_cluster_row(self, row: int) -> None:
        index = self._uep_cluster_table.model().index(row, 0)
        self._uep_cluster_table.selectionModel().select(
            index,
            QItemSelectionModel.SelectionFlag.Toggle | QItemSelectionModel.SelectionFlag.Rows,
        )
        self._sync_viewer_overlays()

    def _collect_method_parameters(self, method: str) -> dict[str, Any]:
        if method == "random":
            mode = "separation" if self._random_by_separation.isChecked() else "count"
            payload = {
                "selection_mode": mode,
                "min_muon_muon": self._random_min_muon_muon.value(),
                "min_muon_atom": self._random_min_muon_atom.value(),
                "max_attempts": self._random_max_attempts.value(),
                "include_symmetry_equivalents": True,
            }
            if mode == "count":
                payload["count"] = self._random_count.value()
            return payload
        if method == "near_element":
            return {
                "element": self._near_element.currentText().strip(),
                "strategy": "attached",
                "muons_per_atom": self._near_muons_per_atom.value(),
                "bond_distance": self._near_bond_distance.value(),
                "min_muon_atom": self._near_min_muon_atom.value(),
            }
        if method == "uep_clusters":
            return {"selections": self._selected_uep_cluster_rows()}
        return {
            "coord_system": self._manual_coord_system.currentText(),
            "x": self._manual_x.value(),
            "y": self._manual_y.value(),
            "z": self._manual_z.value(),
        }

    def _emit_state(self) -> None:
        self.form_state_changed.emit(self.collect_state())

    def collect_state(self) -> dict[str, Any]:
        """Collect all form values for project persistence."""
        return {
            "method": self._current_method(),
            "random": {
                "selection_mode": "separation" if self._random_by_separation.isChecked() else "count",
                "count": self._random_count.value(),
                "min_muon_muon": self._random_min_muon_muon.value(),
                "min_muon_atom": self._random_min_muon_atom.value(),
                "max_attempts": self._random_max_attempts.value(),
            },
            "near_element": {
                "element": self._near_element.currentText(),
                "muons_per_atom": self._near_muons_per_atom.value(),
                "bond_distance": self._near_bond_distance.value(),
                "min_muon_atom": self._near_min_muon_atom.value(),
            },
            "manual": {
                "coord_system": self._manual_coord_system.currentText(),
                "x": self._manual_x.value(),
                "y": self._manual_y.value(),
                "z": self._manual_z.value(),
            },
            "uep_clusters": {},
        }

    def set_form_state(self, state: dict[str, Any]) -> None:
        """Apply saved form values when a project is loaded."""
        if not state:
            return

        random_state = state.get("random", {})
        near_state = state.get("near_element", {})
        manual_state = state.get("manual", {})

        self._restoring_form_state = True
        try:
            self._random_count.setValue(int(random_state.get("count", self._random_count.value())))
            self._random_min_muon_muon.setValue(
                float(random_state.get("min_muon_muon", self._random_min_muon_muon.value()))
            )
            self._random_min_muon_atom.setValue(
                float(random_state.get("min_muon_atom", self._random_min_muon_atom.value()))
            )
            self._random_max_attempts.setValue(
                int(random_state.get("max_attempts", self._random_max_attempts.value()))
            )
            self._random_seed = str(random_state.get("seed", self._random_seed))
            random_mode = str(random_state.get("selection_mode", random_state.get("mode", ""))).strip().lower()
            if random_mode in {"count", "number", "target_count"}:
                self._random_by_count.setChecked(True)
            else:
                self._random_by_separation.setChecked(True)
            self._on_random_mode_changed()
        finally:
            self._restoring_form_state = False

        saved_element = str(near_state.get("element", self._near_element.currentText()))
        if saved_element and self._near_element.findText(saved_element) < 0:
            self._near_element.addItem(saved_element)
        self._near_element.setCurrentText(saved_element)
        self._near_muons_per_atom.setValue(
            int(
                near_state.get(
                    "muons_per_atom",
                    near_state.get(
                        "max_candidates_per_atom",
                        near_state.get("count", self._near_muons_per_atom.value()),
                    ),
                )
            )
        )
        self._near_bond_distance.setValue(
            float(near_state.get("bond_distance", near_state.get("shell_min", self._near_bond_distance.value())))
        )
        self._near_min_muon_atom.setValue(
            float(near_state.get("min_muon_atom", self._near_min_muon_atom.value()))
        )

        self._manual_coord_system.setCurrentText(
            str(manual_state.get("coord_system", self._manual_coord_system.currentText()))
        )
        self._manual_x.setValue(float(manual_state.get("x", self._manual_x.value())))
        self._manual_y.setValue(float(manual_state.get("y", self._manual_y.value())))
        self._manual_z.setValue(float(manual_state.get("z", self._manual_z.value())))

        method = str(state.get("method", "random"))
        self._set_current_method(method)
        self._on_method_changed()

    def set_sites(self, rows: list[dict[str, Any]]) -> None:
        """Render generated sites in the table.

        Skips the full table rebuild (and the overlay resync it triggers) when
        the row payloads are unchanged. The rows are plain dicts, so an ``==``
        compare against the stored copy is exact and cheap. Skipping the rebuild
        also preserves the user's current row selection across dashboard
        flushes, which the rebuild would otherwise clear. Stale-row styling
        depends on the active host, not on the rows, and is re-applied by
        :meth:`set_active_host` whenever the host changes, so it is safe to skip
        here when the rows themselves have not changed.
        """
        if rows == self._site_rows:
            return
        self._site_rows = [dict(row) for row in rows]
        self._site_table.setRowCount(0)
        for row_data in self._site_rows:
            row = self._site_table.rowCount()
            self._site_table.insertRow(row)
            for column, value in enumerate(
                [
                    str(row_data.get("label", "")),
                    self._display_generation_method(row_data.get("method", "")),
                    self._display_site_status(row_data.get("status", "")),
                ]
            ):
                item = QTableWidgetItem(value)
                item.setData(Qt.ItemDataRole.UserRole, row_data)
                self._site_table.setItem(row, column, item)
            frac = row_data.get("fractional", [0.0, 0.0, 0.0])
            for offset, value in enumerate(frac[:3], start=3):
                item = QTableWidgetItem(f"{float(value):.4f}")
                item.setData(Qt.ItemDataRole.UserRole, row_data)
                self._site_table.setItem(row, offset, item)
        self._apply_stale_row_styling()
        self._update_delete_button_state()
        self._update_cluster_button_state()
        self._update_create_jobs_button_state()
        self._sync_viewer_overlays()

    def _apply_stale_row_styling(self) -> None:
        """Dim rows whose source_structure_id doesn't match the active host."""
        active_host_id = str(self._active_host.id) if self._active_host is not None else None
        muted_color = self.palette().color(self.palette().ColorRole.PlaceholderText)
        default_color = self.palette().color(self.palette().ColorRole.Text)
        for row in range(self._site_table.rowCount()):
            item0 = self._site_table.item(row, 0)
            if item0 is None:
                continue
            row_data = item0.data(Qt.ItemDataRole.UserRole)
            source_id = str(row_data.get("source_structure_id", "")) if isinstance(row_data, dict) else ""
            stale = active_host_id is not None and source_id and source_id != active_host_id
            for col in range(self._site_table.columnCount()):
                item = self._site_table.item(row, col)
                if item is None:
                    continue
                item.setForeground(muted_color if stale else default_color)
                if col == 0 and stale:
                    original_label = str(row_data.get("label", ""))
                    if not original_label.endswith(" (stale)"):
                        item.setText(original_label + " (stale)")
                    item.setToolTip("Generated for a different host structure. Regenerate or switch active host.")
                elif col == 0:
                    item.setText(str(row_data.get("label", "")))
                    item.setToolTip("")

    def select_site_ids(self, site_ids: list[str]) -> None:
        """Restore selection for generated-site table rows by id."""
        wanted = {str(site_id).strip() for site_id in site_ids if str(site_id).strip()}
        if not wanted:
            self._clear_site_selection()
            return

        selection_model = self._site_table.selectionModel()
        previous = self._site_table.blockSignals(True)
        self._site_table.clearSelection()
        for row in range(self._site_table.rowCount()):
            item = self._site_table.item(row, 0)
            payload = item.data(Qt.ItemDataRole.UserRole) if item is not None else None
            if not isinstance(payload, dict):
                continue
            raw_id = str(payload.get("id", "")).strip()
            if raw_id not in wanted:
                continue
            index = self._site_table.model().index(row, 0)
            selection_model.select(
                index,
                QItemSelectionModel.SelectionFlag.Select | QItemSelectionModel.SelectionFlag.Rows,
            )
        self._site_table.blockSignals(previous)
        self._sync_viewer_overlays()
        self._update_cluster_button_state()
        self._update_delete_button_state()
        self._update_create_jobs_button_state()

    def set_active_host(self, structure: Structure | None) -> None:
        """Update the active-host badge and re-apply stale-row styling."""
        self._active_host = structure
        self._active_host_badge.set_active_host(structure)
        self._apply_stale_row_styling()

    def set_host_structure(self, structure: Structure | None) -> None:
        """Render the active host structure in the preview viewer.

        No-op (beyond storing the reference) when the structure is unchanged
        since the last render: the combo-element list, viewer mesh and overlay
        sync all derive purely from the host geometry, so an unchanged
        fingerprint means an unchanged result. This avoids the multi-second
        full 3D re-render that otherwise fires on every dashboard flush.
        """
        fingerprint = structure_geometry_fingerprint(structure)
        if self._host_fingerprint_valid and fingerprint == self._host_fingerprint:
            self._host_structure = structure
            return
        self._host_structure = structure
        self._host_fingerprint = fingerprint
        self._host_fingerprint_valid = True
        current = self._near_element.currentText()
        elements = sorted({str(symbol).strip() for symbol in (structure.species if structure else []) if str(symbol).strip()})
        previous = self._near_element.blockSignals(True)
        self._near_element.clear()
        self._near_element.addItems(elements)
        if current and self._near_element.findText(current) >= 0:
            self._near_element.setCurrentText(current)
        elif elements:
            self._near_element.setCurrentIndex(0)
        self._near_element.blockSignals(previous)
        self._viewer.set_structure(structure)
        self._sync_random_count_spacing()
        self._sync_viewer_overlays()

    def set_uep_cluster_sources(self, rows: list[dict[str, Any]]) -> None:
        """Render available clustered UEP results that can become generated sites.

        Skips the table rebuild and overlay resync when the row payloads are
        unchanged (see :meth:`set_sites` for the rationale); the rows are plain
        dicts compared by value.
        """
        if rows == self._uep_cluster_rows:
            return
        self._uep_cluster_rows = [dict(row) for row in rows]
        self._uep_cluster_table.setRowCount(0)
        for row_data in self._uep_cluster_rows:
            row = self._uep_cluster_table.rowCount()
            self._uep_cluster_table.insertRow(row)
            frac = row_data.get("fractional_position", row_data.get("fractional", [0.0, 0.0, 0.0]))
            values = [
                row_data.get("source_job_label", ""),
                row_data.get("cluster_index", ""),
                row_data.get("cluster_size", ""),
                row_data.get("classification", ""),
                self._format_optional_float(row_data.get("classical_energy_ev")),
                f"{float(frac[0]):.4f}",
                f"{float(frac[1]):.4f}",
                f"{float(frac[2]):.4f}",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(str(value))
                item.setData(Qt.ItemDataRole.UserRole, row_data)
                self._uep_cluster_table.setItem(row, column, item)
        self._uep_cluster_table.resizeColumnsToContents()
        self._sync_viewer_overlays()

    def _selected_uep_cluster_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[int] = set()
        for item in self._uep_cluster_table.selectedItems():
            row_index = item.row()
            if row_index in seen:
                continue
            seen.add(row_index)
            payload = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(payload, dict):
                rows.append(dict(payload))
        return rows

    @staticmethod
    def _format_optional_float(value: object) -> str:
        if value is None or value == "":
            return ""
        try:
            return f"{float(value):.6g}"
        except (TypeError, ValueError):
            return ""

    @staticmethod
    def _display_site_status(status: object) -> str:
        value = str(status or "")
        if value == "candidate":
            return "generated"
        return value

    @staticmethod
    def _display_generation_method(method: object) -> str:
        value = str(method or "")
        labels = {
            "near_element": "attach",
            "uep_cluster": "UEP",
            "uep_cluster_member": "UEP mem.",
            "uep_batch_cluster": "UEP",
        }
        return labels.get(value, value)

    def _cartesian_from_row(self, row_data: dict[str, Any]) -> list[float] | None:
        cartesian = row_data.get("cartesian_position", row_data.get("cartesian"))
        if isinstance(cartesian, (list, tuple)) and len(cartesian) == 3:
            return [float(value) for value in cartesian]
        fractional = row_data.get("fractional_position", row_data.get("fractional"))
        if self._host_structure is not None and isinstance(fractional, (list, tuple)) and len(fractional) == 3:
            return lattice_frac_to_cart([float(value) for value in fractional], self._host_structure.lattice_vectors)
        return None

    def _selected_site_rows(self) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        seen: set[int] = set()
        for item in self._site_table.selectedItems():
            row_index = item.row()
            if row_index in seen:
                continue
            seen.add(row_index)
            payload = item.data(Qt.ItemDataRole.UserRole)
            if isinstance(payload, dict):
                rows.append(dict(payload))
        return rows

    def _selected_site_ids(self) -> list[str]:
        ids: list[str] = []
        seen: set[str] = set()
        for row_data in self._selected_site_rows():
            raw_id = str(row_data.get("id", "")).strip()
            if raw_id and raw_id not in seen:
                seen.add(raw_id)
                ids.append(raw_id)
        return ids

    def _update_delete_button_state(self) -> None:
        self._delete_btn.setEnabled(self._generate_btn.isEnabled() and bool(self._selected_site_ids()))

    def _update_cluster_button_state(self) -> None:
        self._cluster_btn.setEnabled(self._generate_btn.isEnabled() and bool(self._selected_site_ids()))

    def _update_create_jobs_button_state(self) -> None:
        self._create_jobs_btn.setEnabled(self._generate_btn.isEnabled() and bool(self._selected_site_ids()))

    def _manual_preview_row(self) -> dict[str, Any] | None:
        if self._host_structure is None:
            return None
        values = [self._manual_x.value(), self._manual_y.value(), self._manual_z.value()]
        if self._manual_coord_system.currentText() == "cartesian":
            return {"cartesian": values}
        return {"fractional": values}

    def _sync_viewer_overlays(self) -> None:
        if self._host_structure is None:
            return
        method = self._current_method()
        rows = self._selected_site_rows()
        if method == "uep_clusters":
            rows.extend(self._selected_uep_cluster_rows())
        elif method == "manual":
            manual_row = self._manual_preview_row()
            if manual_row is not None:
                rows.append(manual_row)
        positions = [cart for row in rows if (cart := self._cartesian_from_row(row)) is not None]
        if not positions:
            self._viewer.set_point_overlays([])
            return
        self._viewer.set_point_overlays(
            [
                {
                    "positions": positions,
                    "colors": "#e11d48",
                    "radii": [MUON_VIEWER_RADIUS_ANG] * len(positions),
                    "alpha": 0.96,
                }
            ]
        )
