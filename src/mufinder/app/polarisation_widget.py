"""The Polarisation page: simulate muon polarisation from stored field results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import (
    QAbstractItemView,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QHeaderView,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.app import workflow_table_utils
from mufinder.app.blocking_call import BlockingCallCancelled, run_blocking_call
from mufinder.app.export_dialogs import (
    ExportFormat,
    ask_save_path,
    report_export_failed,
    report_export_saved,
)
from mufinder.app.magnetic_field_widget import unrelaxed_results_note
from mufinder.app.plot_canvas import PlotFrame, create_plot_canvas
from mufinder.app.theme import (
    current_palette,
    style_hint_text,
    style_inline_group,
    style_plot_axes,
    style_plot_legend,
    style_section_subtitle,
    style_section_title,
    style_warning_text,
)
from mufinder.app.workflow_table_utils import (
    bind_delete_shortcuts,
    create_workflow_table,
    format_vector_components,
    name_row_control,
)
from mufinder.domain.constants import MUON_GAMMA_OVER_2PI_MHZ_T
from mufinder.domain.text import plural
from mufinder.science.polarisation import PolarisationError
from mufinder.science.polarisation_simulation import (
    MAX_PADDING_FACTOR,
    MAX_SPHERE_POINTS,
    MAX_TIME_POINTS,
    PolarisationSettings,
    dynamics_supported,
)
from mufinder.services.magnetic_field_results import UNRELAXED_RESULTS_NOTE
from mufinder.services.polarisation_simulations import (
    FieldSource,
    PolarisationSimulationRecord,
    SimulationSource,
    field_sources,
    polarisation_simulation,
    polarisation_simulation_summaries,
    remove_polarisation_simulation,
    run_polarisation_simulation,
    stale_sources,
    store_polarisation_simulation,
)
from mufinder.services.project_document import ProjectDocument

GEOMETRY_OPTIONS: tuple[tuple[str, str], ...] = (
    ("zero_field", "Zero field"),
    ("longitudinal", "Longitudinal field"),
    ("transverse", "Transverse field"),
)
SAMPLE_OPTIONS: tuple[tuple[bool, str], ...] = (
    (False, "Single crystal"),
    (True, "Polycrystal"),
)
DYNAMICS_OPTIONS: tuple[tuple[str, str], ...] = (
    ("exact", "Exact recursion"),
    ("fft", "FFT (long grids)"),
)
EXPORT_FORMATS = (
    ExportFormat("DAT", "DAT files", ".dat"),
    ExportFormat("Text", "Text files", ".txt"),
)
# Above this many sites the per-site traces are drawn without a legend entry each.
_MAX_LEGEND_SITES = 8
# Sources-table columns.
_USE_COLUMN, _SOURCE_COLUMN, _APPLIED_COLUMN, _WEIGHT_COLUMN = range(4)
_OWNER_ROLE = int(Qt.ItemDataRole.UserRole) + 1
# The source and simulation tables stop growing at about five rows, then scroll.
_TABLE_MAX_HEIGHT = 150
_SIMULATION_ROLE = int(Qt.ItemDataRole.UserRole) + 2


class VectorInput(QWidget):
    """Three decimal boxes for a vector; empty boxes count as zero.

    ``name`` names each box for screen readers as ``"<name> x"`` and so on.
    """

    def __init__(
        self,
        values: Sequence[float] = (0.0, 0.0, 0.0),
        *,
        name: str,
        tooltip: str = "",
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        style_inline_group(self)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._edits = []
        for axis in ("x", "y", "z"):
            edit = QLineEdit(self)
            edit.setValidator(QDoubleValidator(edit))
            edit.setMaximumWidth(64)
            edit.setToolTip(f"{tooltip} ({axis})" if tooltip else axis)
            edit.setAccessibleName(f"{name} {axis}")
            layout.addWidget(edit)
            self._edits.append(edit)
        layout.addStretch(1)
        self.set_value(values)

    @property
    def edits(self) -> tuple[QLineEdit, ...]:
        return tuple(self._edits)

    def value(self, name: str) -> tuple[float, float, float]:
        """Return the vector; raises ``ValueError`` naming ``name`` for a non-number."""

        components = []
        for edit in self._edits:
            text = edit.text().strip()
            try:
                value = float(text) if text else 0.0
            except ValueError as exc:
                raise ValueError(f"{name} components must be decimal numbers") from exc
            if not np.isfinite(value):
                raise ValueError(f"{name} components must be finite")
            components.append(value)
        x, y, z = components
        return (x, y, z)

    def set_value(self, values: Sequence[float]) -> None:
        for edit, value in zip(self._edits, values, strict=True):
            edit.setText(f"{float(value):g}")


class PolarisationWidget(QWidget):
    """Simulate the muon polarisation and its spectrum from stored field results.

    The page lists the field results of the Magnetic Fields page and its
    saved workspaces as sources, runs simulations of a weighted selection of
    them and keeps every simulation in the project. Selecting a simulation
    shows its traces and loads its settings and sources into the form.
    """

    form_state_changed = Signal(dict)
    simulations_changed = Signal()
    open_magnetic_fields_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._document: ProjectDocument | None = None
        self._sources: list[FieldSource] = []
        self._restoring = False
        # Decoded simulations by id, cleared whenever the list is re-read.
        self._record_cache: dict[str, PolarisationSimulationRecord] = {}
        # The last settings the form held that were valid, saved while a box is not.
        self._last_valid_settings = PolarisationSettings()

        title = QLabel("Polarisation", self)
        style_section_title(title)
        subtitle = QLabel(
            "Simulate the muon polarisation P(t) and its spectrum from the field results of "
            "Magnetic Fields and its saved workspaces.",
            self,
        )
        subtitle.setWordWrap(True)
        style_section_subtitle(subtitle)

        self._sources_table = create_workflow_table(
            ("Use", "Field result", "Applied field (T)", "Weight"),
            resize_to_contents_columns=(_USE_COLUMN, _APPLIED_COLUMN, _WEIGHT_COLUMN),
            stretch_last_section=False,
            accessible_name="Field Results",
        )
        self._sources_table.horizontalHeader().setSectionResizeMode(
            _SOURCE_COLUMN, QHeaderView.ResizeMode.Stretch
        )
        self._sources_table.itemChanged.connect(self._on_source_item_changed)
        self._sources_hint = QLabel(self)
        self._sources_hint.setWordWrap(True)
        style_hint_text(self._sources_hint)
        self._open_fields_button = QPushButton("Go to Magnetic Fields →", self)
        self._open_fields_button.clicked.connect(self.open_magnetic_fields_requested.emit)

        self._build_settings_controls()

        self._simulations_table = create_workflow_table(
            ("Simulation", "Geometry", "Sample", "Sources", "Status"),
            resize_to_contents_columns=(1, 2, 4),
            stretch_last_section=False,
            accessible_name="Simulations",
        )
        self._simulations_table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        for table in (self._sources_table, self._simulations_table):
            table.setMaximumHeight(_TABLE_MAX_HEIGHT)
        for column in (0, 3):
            self._simulations_table.horizontalHeader().setSectionResizeMode(
                column, QHeaderView.ResizeMode.Stretch
            )
        self._simulations_table.itemSelectionChanged.connect(self._on_simulation_selected)
        self._delete_shortcuts = bind_delete_shortcuts(
            self._simulations_table, self._remove_selected_simulation
        )

        self._add_button = QPushButton("Add Simulation", self)
        self._add_button.setToolTip("Simulate the selected field results with these settings.")
        self._add_button.clicked.connect(self._add_simulation)
        self._recalculate_button = QPushButton("Recalculate", self)
        self._recalculate_button.setToolTip(
            "Run the selected simulation again with these settings and the current fields."
        )
        self._recalculate_button.clicked.connect(self._recalculate_simulation)
        self._remove_button = QPushButton("Remove…", self)
        self._remove_button.clicked.connect(self._remove_selected_simulation)
        self._export_button = QPushButton("Export Data…", self)
        self._export_button.clicked.connect(self._export_data)

        self._status = QLabel("", self)
        self._status.setWordWrap(True)
        style_hint_text(self._status)
        self._stale_note = QLabel("", self)
        self._stale_note.setWordWrap(True)
        style_warning_text(self._stale_note)
        self._stale_note.setVisible(False)
        self._warning_note = QLabel("", self)
        self._warning_note.setWordWrap(True)
        style_warning_text(self._warning_note)
        self._warning_note.setVisible(False)
        self._estimate_note = unrelaxed_results_note(self)

        palette = current_palette()
        self._figure, self._canvas = create_plot_canvas(
            figsize=(7.0, 6.4), facecolor=palette.bg_surface, minimum_size=(420, 420)
        )
        self._plot_frame = PlotFrame(self._canvas, title="Polarisation and spectrum", parent=self)

        self._build_layout(title, subtitle)
        self._show_settings(PolarisationSettings())
        self._draw(None)

    # --- Public API -------------------------------------------------------------

    def set_document(self, document: ProjectDocument | None) -> None:
        """Show ``document``'s field results and simulations."""

        self._document = document
        self._refresh_sources()
        self._refresh_simulations()

    def refresh(self) -> None:
        """Re-read the field results and simulations, keeping the selection and checks."""

        self._refresh_sources()
        self._refresh_simulations(keep=self.selected_simulation_id())

    def select_sources(self, owners: Sequence[str]) -> None:
        """Use exactly the field results of ``owners`` and clear the simulation selection."""

        self._simulations_table.clearSelection()
        self._set_checked_owners(set(owners))
        self._emit_form_state()

    def selected_simulation_id(self) -> str | None:
        rows = workflow_table_utils.selected_table_rows(self._simulations_table)
        if not rows:
            return None
        item = self._simulations_table.item(rows[0], 0)
        value = item.data(_SIMULATION_ROLE) if item is not None else None
        return str(value) if value else None

    def collect_state(self) -> dict[str, Any]:
        """The form: settings text, the checked sources and weights, the selection."""

        try:
            # The form keeps the rate in the box even where the geometry runs static,
            # so returning to a geometry with dynamics brings it back.
            self._last_valid_settings = replace(
                self.settings(), collision_rate_per_us=float(self._collision_rate.value())
            )
        except (ValueError, PolarisationError):
            pass
        return {
            "title": self._title_edit.text(),
            "settings": self._last_valid_settings.to_json(),
            "sources": [source.to_json() for source in self._checked_sources()],
            "selected_simulation": self.selected_simulation_id(),
        }

    def set_form_state(self, state: Mapping[str, object] | None) -> None:
        """Restore a form saved by :meth:`collect_state`; malformed parts take defaults."""

        state = state if isinstance(state, Mapping) else {}
        self._restoring = True
        try:
            raw_settings = state.get("settings")
            try:
                settings = (
                    PolarisationSettings.from_json(raw_settings)
                    if isinstance(raw_settings, Mapping)
                    else PolarisationSettings()
                )
            except PolarisationError:
                settings = PolarisationSettings()
            self._show_settings(settings)
            self._last_valid_settings = settings
            self._title_edit.setText(str(state.get("title") or ""))
            raw_sources = state.get("sources")
            if isinstance(raw_sources, list):
                sources = []
                for item in raw_sources:
                    try:
                        sources.append(SimulationSource.from_json(item))
                    except (KeyError, TypeError, ValueError):
                        continue
            else:
                # A form never saved starts with the first field result ticked.
                sources = [
                    SimulationSource(source.owner, source.record_id, source.title)
                    for source in self._sources[:1]
                ]
            self._show_sources(sources)
            selected = state.get("selected_simulation")
            self._simulations_table.clearSelection()
            if selected:
                self._select_simulation(str(selected))
        finally:
            self._restoring = False
        self._show_selected_simulation()

    def settings(self) -> PolarisationSettings:
        """The settings in the form; raises ``ValueError`` or ``PolarisationError`` when invalid."""

        zero_field = self._geometry.currentData() == "zero_field"
        spin = self._direction("initial polarisation", self._spin, strict=zero_field)
        detector = None
        if not self._detector_along_spin.isChecked():
            detector = self._direction("detector direction", self._detector, strict=zero_field)
        added = None
        if self._added_field.isEnabled():
            added = self._added_field.value("added applied field")
        return PolarisationSettings(
            geometry=self._geometry.currentData(),
            polarisation_direction=spin,
            detector_direction=detector,
            crystal_along_z_uvw=self._along_z.value("crystal direction along z"),
            crystal_along_x_uvw=self._along_x.value("crystal direction along x"),
            added_applied_field_T=added,
            polycrystalline=bool(self._sample.currentData()),
            sphere_points=int(self._sphere_points.value()),
            lambda_z_per_us=float(self._lambda_z.value()),
            sigma_per_us=float(self._sigma.value()),
            sigma_from_nuclear_widths=self._sigma_from_widths.isChecked(),
            end_time_us=float(self._end_time.value()),
            time_points=int(self._time_points.value()),
            # Dynamics apply only where the geometry allows them (see _dynamics_allowed).
            collision_rate_per_us=float(self._collision_rate.value())
            if self._dynamics_allowed(spin, detector)
            else 0.0,
            dynamics_method=self._dynamics_method.currentData(),
            subtract_baseline=self._subtract_baseline.isChecked(),
            apodisation_sigma_us=_optional_positive(self._apodisation_sigma, "apodisation σ"),
            apodisation_lambda_us=_optional_positive(self._apodisation_lambda, "apodisation λ"),
            padding_factor=int(self._padding.value()),
        )

    @staticmethod
    def _direction(name: str, box: VectorInput, *, strict: bool) -> tuple[float, float, float]:
        """The direction in ``box``; when the geometry ignores it, a bad one reads as z."""

        try:
            value = box.value(name)
        except ValueError:
            if strict:
                raise
            return (0.0, 0.0, 1.0)
        if not strict and not any(value):
            return (0.0, 0.0, 1.0)
        return value

    # --- Layout -------------------------------------------------------------------

    def _build_settings_controls(self) -> None:
        self._title_edit = QLineEdit(self)
        self._title_edit.setPlaceholderText("Simulation")
        self._title_edit.setMaxLength(80)

        self._geometry = QComboBox(self)
        for key, label in GEOMETRY_OPTIONS:
            self._geometry.addItem(label, key)
        self._geometry.setToolTip(
            "Zero field: choose the spin and detector directions. Longitudinal and transverse: "
            "the spin lies along or across the applied field."
        )
        self._spin = VectorInput(
            (0.0, 0.0, 1.0), name="Initial Polarisation", tooltip="Initial muon spin, lab frame"
        )
        self._detector_along_spin = QCheckBox("Along the spin", self)
        self._detector_along_spin.setChecked(True)
        self._detector = VectorInput(
            (0.0, 0.0, 1.0), name="Detector", tooltip="Detector axis, lab frame"
        )
        self._along_z = VectorInput(
            (0.0, 0.0, 1.0),
            name="Crystal [uvw] Along z",
            tooltip="Lattice direction [uvw] along lab z",
        )
        self._along_x = VectorInput(
            (1.0, 0.0, 0.0),
            name="Crystal [uvw] Along x",
            tooltip="Lattice direction [uvw] along lab x",
        )
        self._applied_summary = QLabel("", self)
        self._applied_summary.setWordWrap(True)
        style_hint_text(self._applied_summary)
        self._added_field = VectorInput(
            (0.0, 0.0, 0.0),
            name="Add Applied Field (T)",
            tooltip="Applied field added to zero-field results, T, crystal frame",
        )

        self._sample = QComboBox(self)
        self._sample.setAccessibleName("Sample")
        for polycrystalline, label in SAMPLE_OPTIONS:
            self._sample.addItem(label, polycrystalline)
        self._sphere_points = QSpinBox(self)
        self._sphere_points.setRange(1, MAX_SPHERE_POINTS)
        self._sphere_points.setMaximumWidth(100)
        self._sphere_points.setToolTip(
            "Orientations of the applied field in a polycrystal; the zero-field average is exact."
        )

        self._lambda_z = _rate_input("Longitudinal relaxation rate λ_z")
        # The form label is rich text (λ<sub>z</sub>), which a buddy would read out as markup.
        self._lambda_z.setAccessibleName("λz")
        self._sigma = _rate_input("Gaussian width σ of the oscillating part")
        self._sigma.setAccessibleName("σ")
        self._sigma_from_widths = QCheckBox("From nuclear widths", self)
        self._sigma_from_widths.setToolTip(
            "Sum the nuclear dipolar widths of the host at each muon position and damp each "
            "field with the transverse width along it."
        )

        self._end_time = QDoubleSpinBox(self)
        self._end_time.setRange(0.001, 1_000.0)
        self._end_time.setDecimals(3)
        self._end_time.setSuffix(" μs")
        self._end_time.setMaximumWidth(130)
        self._end_time.setAccessibleName("End Time")
        self._time_points = QSpinBox(self)
        self._time_points.setRange(2, MAX_TIME_POINTS)
        self._time_points.setMaximumWidth(100)

        self._collision_rate = _rate_input("Strong-collision rate ν; 0 is static")
        self._collision_rate.setAccessibleName("Collision Rate ν")
        self._dynamics_method = QComboBox(self)
        self._dynamics_method.setAccessibleName("Dynamics Method")
        for key, label in DYNAMICS_OPTIONS:
            self._dynamics_method.addItem(label, key)

        self._subtract_baseline = QCheckBox("Subtract baseline", self)
        self._apodisation_sigma = _optional_time_input("Gaussian apodisation time")
        self._apodisation_lambda = _optional_time_input("Exponential apodisation time")
        self._apodisation_sigma.setAccessibleName("Apodisation σ")
        self._apodisation_lambda.setAccessibleName("Apodisation λ")
        self._padding = QSpinBox(self)
        self._padding.setRange(1, MAX_PADDING_FACTOR)
        self._padding.setPrefix("× ")
        self._padding.setMaximumWidth(90)

        for combo in (self._geometry, self._sample, self._dynamics_method):
            combo.currentIndexChanged.connect(self._on_setting_changed)
        for check in (self._detector_along_spin, self._sigma_from_widths, self._subtract_baseline):
            check.toggled.connect(self._on_setting_changed)
        for spin in (
            self._sphere_points,
            self._lambda_z,
            self._sigma,
            self._end_time,
            self._time_points,
            self._collision_rate,
            self._padding,
        ):
            spin.valueChanged.connect(self._on_setting_changed)
        for edit in (
            self._title_edit,
            self._apodisation_sigma,
            self._apodisation_lambda,
            *self._spin.edits,
            *self._detector.edits,
            *self._along_z.edits,
            *self._along_x.edits,
            *self._added_field.edits,
        ):
            edit.textChanged.connect(self._on_setting_changed)

    def _build_layout(self, title: QLabel, subtitle: QLabel) -> None:
        sources_group = QGroupBox("Field Results", self)
        sources_layout = QVBoxLayout(sources_group)
        sources_layout.addWidget(self._sources_table)
        hint_row = QHBoxLayout()
        hint_row.addWidget(self._sources_hint, 1)
        hint_row.addWidget(self._open_fields_button)
        sources_layout.addLayout(hint_row)

        settings_group = QGroupBox("Settings", self)
        form = QFormLayout(settings_group)
        form.addRow("Name", self._title_edit)
        form.addRow("Geometry", self._geometry)
        form.addRow("Initial polarisation", self._spin)
        detector_row = QWidget(settings_group)
        style_inline_group(detector_row)
        detector_layout = QHBoxLayout(detector_row)
        detector_layout.setContentsMargins(0, 0, 0, 0)
        detector_layout.addWidget(self._detector_along_spin)
        detector_layout.addWidget(self._detector, 1)
        form.addRow("Detector", detector_row)
        form.addRow("Crystal [uvw] along z", self._along_z)
        form.addRow("Crystal [uvw] along x", self._along_x)
        form.addRow("Applied field", self._applied_summary)
        form.addRow("Add applied field (T)", self._added_field)
        sample_row = QWidget(settings_group)
        style_inline_group(sample_row)
        sample_layout = QHBoxLayout(sample_row)
        sample_layout.setContentsMargins(0, 0, 0, 0)
        sample_layout.addWidget(self._sample)
        sphere_points_label = QLabel("Sphere points", sample_row)
        sphere_points_label.setBuddy(self._sphere_points)
        sample_layout.addWidget(sphere_points_label)
        sample_layout.addWidget(self._sphere_points)
        sample_layout.addStretch(1)
        form.addRow("Sample", sample_row)
        form.addRow("λ<sub>z</sub>", self._lambda_z)
        sigma_row = QWidget(settings_group)
        style_inline_group(sigma_row)
        sigma_layout = QHBoxLayout(sigma_row)
        sigma_layout.setContentsMargins(0, 0, 0, 0)
        sigma_layout.addWidget(self._sigma)
        sigma_layout.addWidget(self._sigma_from_widths)
        sigma_layout.addStretch(1)
        form.addRow("σ", sigma_row)
        time_row = QWidget(settings_group)
        style_inline_group(time_row)
        time_layout = QHBoxLayout(time_row)
        time_layout.setContentsMargins(0, 0, 0, 0)
        time_layout.addWidget(self._end_time)
        time_points_label = QLabel("Points", time_row)
        time_points_label.setBuddy(self._time_points)
        time_layout.addWidget(time_points_label)
        time_layout.addWidget(self._time_points)
        time_layout.addStretch(1)
        form.addRow("End time", time_row)
        dynamics_row = QWidget(settings_group)
        style_inline_group(dynamics_row)
        dynamics_layout = QHBoxLayout(dynamics_row)
        dynamics_layout.setContentsMargins(0, 0, 0, 0)
        dynamics_layout.addWidget(self._collision_rate)
        dynamics_layout.addWidget(self._dynamics_method)
        dynamics_layout.addStretch(1)
        form.addRow("Collision rate ν", dynamics_row)
        form.addRow("Spectrum", self._subtract_baseline)
        apodisation_row = QWidget(settings_group)
        style_inline_group(apodisation_row)
        apodisation_layout = QHBoxLayout(apodisation_row)
        apodisation_layout.setContentsMargins(0, 0, 0, 0)
        apodisation_layout.addWidget(QLabel("σ", apodisation_row))
        apodisation_layout.addWidget(self._apodisation_sigma)
        apodisation_layout.addWidget(QLabel("λ", apodisation_row))
        apodisation_layout.addWidget(self._apodisation_lambda)
        apodisation_layout.addStretch(1)
        form.addRow("Apodisation (μs)", apodisation_row)
        form.addRow("Zero padding", self._padding)

        simulations_group = QGroupBox("Simulations", self)
        simulations_layout = QVBoxLayout(simulations_group)
        simulations_layout.addWidget(self._simulations_table)
        button_row = QHBoxLayout()
        for button in (
            self._add_button,
            self._recalculate_button,
            self._remove_button,
            self._export_button,
        ):
            button_row.addWidget(button)
        button_row.addStretch(1)
        simulations_layout.addLayout(button_row)
        simulations_layout.addWidget(self._status)
        simulations_layout.addWidget(self._stale_note)
        simulations_layout.addWidget(self._warning_note)
        simulations_layout.addWidget(self._estimate_note)
        simulations_layout.addWidget(self._plot_frame, 1)

        left = QWidget(self)
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.addWidget(sources_group)
        left_layout.addWidget(settings_group)
        left_layout.addStretch(1)
        left_scroll = QScrollArea(self)
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        left_scroll.setWidget(left)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        left_scroll.setMinimumWidth(left.minimumSizeHint().width() + 4)

        body = QHBoxLayout()
        body.addWidget(left_scroll, 1)
        body.addWidget(simulations_group, 1)

        layout = QVBoxLayout(self)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addLayout(body, 1)

    # --- Sources --------------------------------------------------------------------

    def _refresh_sources(self) -> None:
        checked = {source.owner: source.weight for source in self._checked_sources()}
        first_fill = self._sources_table.rowCount() == 0 and not checked
        self._sources = field_sources(self._document) if self._document is not None else []
        self._fill_sources(
            [
                SimulationSource(source.owner, source.record_id, source.title, weight)
                for source in self._sources
                if (weight := checked.get(source.owner)) is not None
            ]
            if not first_fill
            else [
                SimulationSource(source.owner, source.record_id, source.title)
                for source in self._sources[:1]
            ]
        )
        if self._document is None:
            text = "Open a project to simulate its polarisation."
        elif not self._sources:
            text = "No field results yet. Calculate fields on the Magnetic Fields page or in a saved workspace."
        else:
            text = "Tick the field results to simulate together; weights are shared among each result's equivalent sites."
        self._sources_hint.setText(text)
        self._open_fields_button.setVisible(self._document is not None and not self._sources)
        self._sync_controls()

    def _fill_sources(self, checked: Sequence[SimulationSource]) -> None:
        weights = {source.owner: source.weight for source in checked}
        table = self._sources_table
        table.blockSignals(True)
        try:
            table.setRowCount(0)
            for row, source in enumerate(self._sources):
                table.insertRow(row)
                use = QTableWidgetItem()
                use.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsUserCheckable)
                use.setCheckState(
                    Qt.CheckState.Checked if source.owner in weights else Qt.CheckState.Unchecked
                )
                use.setData(_OWNER_ROLE, source.owner)
                table.setItem(row, _USE_COLUMN, use)
                label = QTableWidgetItem(source.label)
                label.setFlags(Qt.ItemFlag.ItemIsEnabled)
                label.setToolTip(", ".join(source.site_labels))
                table.setItem(row, _SOURCE_COLUMN, label)
                applied = QTableWidgetItem(
                    "differs between sites"
                    if source.applied_T is None
                    else ", ".join(format_vector_components(np.asarray(source.applied_T)))
                )
                applied.setFlags(Qt.ItemFlag.ItemIsEnabled)
                table.setItem(row, _APPLIED_COLUMN, applied)
                weight = QDoubleSpinBox(table)
                weight.setRange(0.0, 1_000.0)
                weight.setDecimals(2)
                weight.setMaximumWidth(90)
                weight.setValue(float(weights.get(source.owner, 1.0)))
                weight.valueChanged.connect(self._on_setting_changed)
                name_row_control(weight, "Weight for", source.label)
                table.setCellWidget(row, _WEIGHT_COLUMN, weight)
        finally:
            table.blockSignals(False)
        self._sync_applied_field()

    def _show_sources(self, sources: Sequence[SimulationSource]) -> None:
        self._fill_sources(list(sources))

    def _set_checked_owners(self, owners: set[str]) -> None:
        self._fill_sources(
            [
                SimulationSource(source.owner, source.record_id, source.title)
                for source in self._sources
                if source.owner in owners
            ]
        )

    def _checked_sources(self) -> list[SimulationSource]:
        sources = []
        by_owner = {source.owner: source for source in self._sources}
        for row in range(self._sources_table.rowCount()):
            item = self._sources_table.item(row, _USE_COLUMN)
            if item is None or item.checkState() != Qt.CheckState.Checked:
                continue
            owner = str(item.data(_OWNER_ROLE))
            source = by_owner.get(owner)
            if source is None:
                continue
            widget = self._sources_table.cellWidget(row, _WEIGHT_COLUMN)
            weight = float(widget.value()) if isinstance(widget, QDoubleSpinBox) else 1.0
            sources.append(SimulationSource(owner, source.record_id, source.title, weight))
        return sources

    def _on_source_item_changed(self, item: QTableWidgetItem) -> None:
        if item.column() == _USE_COLUMN:
            self._sync_applied_field()
            self._on_setting_changed()

    def _sync_applied_field(self) -> None:
        """Offer an added field only when every checked result is a zero-field one."""

        by_owner = {source.owner: source for source in self._sources}
        checked = [by_owner[source.owner] for source in self._checked_sources()]
        applied = {source.applied_T for source in checked}
        if not checked:
            text, can_add = "Select field results.", False
        elif None in applied or len(applied) > 1:
            text, can_add = "The selected field results carry different applied fields.", False
        else:
            vector = applied.pop()
            assert vector is not None
            if any(value != 0.0 for value in vector):
                components = ", ".join(format_vector_components(np.asarray(vector)))
                text, can_add = f"From the field results: ({components}) T.", False
            else:
                text, can_add = "None in the field results; one can be added below.", True
        self._applied_summary.setText(text)
        self._added_field.setEnabled(can_add)

    # --- Simulations --------------------------------------------------------------

    def _refresh_simulations(self, keep: str | None = None) -> None:
        self._record_cache.clear()
        records = (
            polarisation_simulation_summaries(self._document) if self._document is not None else []
        )
        table = self._simulations_table
        table.blockSignals(True)
        try:
            table.setRowCount(0)
            for row, record in enumerate(records):
                table.insertRow(row)
                stale = self._document is not None and bool(stale_sources(self._document, record))
                status = "Stale" if stale else "Current"
                if record.estimate:
                    status += ", estimate"
                cells = (
                    record.title,
                    dict(GEOMETRY_OPTIONS)[record.settings.geometry],
                    dict(SAMPLE_OPTIONS)[record.settings.polycrystalline],
                    ", ".join(source.title for source in record.sources),
                    status,
                )
                for column, text in enumerate(cells):
                    item = QTableWidgetItem(text)
                    item.setFlags(Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable)
                    if column == 0:
                        item.setData(_SIMULATION_ROLE, record.simulation_id)
                    table.setItem(row, column, item)
            if keep:
                self._select_simulation(keep)
        finally:
            table.blockSignals(False)
        self._show_selected_simulation()

    def _select_simulation(self, simulation_id: str) -> None:
        for row in range(self._simulations_table.rowCount()):
            item = self._simulations_table.item(row, 0)
            if item is not None and item.data(_SIMULATION_ROLE) == simulation_id:
                self._simulations_table.selectRow(row)
                return

    def _selected_record(self) -> PolarisationSimulationRecord | None:
        simulation_id = self.selected_simulation_id()
        if self._document is None or simulation_id is None:
            return None
        record = self._record_cache.get(simulation_id)
        if record is None:
            record = polarisation_simulation(self._document, simulation_id)
            if record is not None:
                self._record_cache[simulation_id] = record
        return record

    def _on_simulation_selected(self) -> None:
        record = self._selected_record()
        if record is not None and not self._restoring:
            self._restoring = True
            try:
                self._show_settings(record.settings)
                self._title_edit.setText(record.title)
                self._show_sources(record.sources)
            finally:
                self._restoring = False
        self._show_selected_simulation()
        self._emit_form_state()

    def _show_selected_simulation(self) -> None:
        record = self._selected_record()
        self._draw(record)
        stale = stale_sources(self._document, record) if record and self._document else []
        if stale:
            names = ", ".join(source.title for source in stale)
            self._stale_note.setText(
                f"Stale: the fields of {names} were recalculated or removed since this "
                "simulation ran. Recalculate it to use the current fields."
            )
        self._stale_note.setVisible(bool(stale))
        warnings = record.simulation.warnings if record is not None else ()
        self._warning_note.setText("\n".join(warnings))
        self._warning_note.setVisible(bool(warnings))
        self._estimate_note.setVisible(record is not None and record.estimate)
        self._sync_controls()

    def _add_simulation(self) -> None:
        self._run(simulation_id=None)

    def _recalculate_simulation(self) -> None:
        simulation_id = self.selected_simulation_id()
        if simulation_id is not None:
            self._run(simulation_id=simulation_id)

    def _run(self, *, simulation_id: str | None) -> None:
        document = self._document
        if document is None:
            return
        try:
            settings = self.settings()
        except (ValueError, PolarisationError) as exc:
            self._status.setText(f"Invalid settings: {exc}")
            return
        sources = self._checked_sources()
        title = self._title_edit.text().strip() or self._unused_title()
        try:
            record = run_blocking_call(
                self,
                lambda: run_polarisation_simulation(
                    document,
                    title=title,
                    settings=settings,
                    sources=sources,
                    simulation_id=simulation_id,
                ),
                title="Polarisation",
                description="Simulating the muon polarisation…",
                cancellable=True,
            )
        except BlockingCallCancelled:
            self._status.setText("Simulation cancelled.")
            return
        except (PolarisationError, ValueError) as exc:
            self._status.setText(f"Simulation failed: {exc}")
            return
        store_polarisation_simulation(document, record)
        self._status.setText(
            f"{record.title}: simulated {plural(len(record.simulation.static.site_labels), 'site')}."
        )
        self._refresh_simulations(keep=record.simulation_id)
        # The form first, so the save that follows carries it.
        self._emit_form_state()
        self.simulations_changed.emit()

    def _unused_title(self) -> str:
        """``Simulation N`` with the smallest N no simulation is called."""

        titles = {
            self._simulations_table.item(row, 0).text()
            for row in range(self._simulations_table.rowCount())
        }
        number = 1
        while f"Simulation {number}" in titles:
            number += 1
        return f"Simulation {number}"

    def _remove_selected_simulation(self) -> None:
        record = self._selected_record()
        if self._document is None or record is None:
            return
        if not workflow_table_utils.confirm_destructive(
            self,
            "Remove simulation",
            f"Remove the simulation {record.title!r}?",
            "Its traces and settings are deleted from the project.",
        ):
            return
        remove_polarisation_simulation(self._document, record.simulation_id)
        self._status.setText(f"Removed {record.title}.")
        self._refresh_simulations()
        # The form first, so the save that follows carries it.
        self._emit_form_state()
        self.simulations_changed.emit()

    # --- Settings -------------------------------------------------------------------

    def _show_settings(self, settings: PolarisationSettings) -> None:
        widgets: list[QWidget] = [
            self._geometry,
            self._sample,
            self._dynamics_method,
            self._detector_along_spin,
            self._sigma_from_widths,
            self._subtract_baseline,
            self._sphere_points,
            self._lambda_z,
            self._sigma,
            self._end_time,
            self._time_points,
            self._collision_rate,
            self._padding,
        ]
        previous = [widget.blockSignals(True) for widget in widgets]
        try:
            self._geometry.setCurrentIndex(self._geometry.findData(settings.geometry))
            self._spin.set_value(settings.polarisation_direction)
            self._detector_along_spin.setChecked(settings.detector_direction is None)
            self._detector.set_value(settings.detector_direction or settings.polarisation_direction)
            self._along_z.set_value(settings.crystal_along_z_uvw)
            self._along_x.set_value(settings.crystal_along_x_uvw)
            self._added_field.set_value(settings.added_applied_field_T or (0.0, 0.0, 0.0))
            self._sample.setCurrentIndex(self._sample.findData(settings.polycrystalline))
            self._sphere_points.setValue(settings.sphere_points)
            self._lambda_z.setValue(settings.lambda_z_per_us)
            self._sigma.setValue(settings.sigma_per_us)
            self._sigma_from_widths.setChecked(settings.sigma_from_nuclear_widths)
            self._end_time.setValue(settings.end_time_us)
            self._time_points.setValue(settings.time_points)
            self._collision_rate.setValue(settings.collision_rate_per_us)
            self._dynamics_method.setCurrentIndex(
                self._dynamics_method.findData(settings.dynamics_method)
            )
            self._subtract_baseline.setChecked(settings.subtract_baseline)
            self._apodisation_sigma.setText(_optional_text(settings.apodisation_sigma_us))
            self._apodisation_lambda.setText(_optional_text(settings.apodisation_lambda_us))
            self._padding.setValue(settings.padding_factor)
        finally:
            for widget, blocked in zip(widgets, previous, strict=True):
                widget.blockSignals(blocked)
        self._sync_controls()

    def _on_setting_changed(self, *_args: object) -> None:
        self._sync_controls()
        self._emit_form_state()

    def _sync_controls(self) -> None:
        zero_field = self._geometry.currentData() == "zero_field"
        for widget in (self._spin, self._detector_along_spin):
            widget.setEnabled(zero_field)
        self._detector.setEnabled(zero_field and not self._detector_along_spin.isChecked())
        self._sphere_points.setEnabled(bool(self._sample.currentData()))
        self._sigma.setEnabled(not self._sigma_from_widths.isChecked())
        dynamics = self._dynamics_allowed()
        self._collision_rate.setEnabled(dynamics)
        self._collision_rate.setToolTip(
            "Strong-collision rate ν; 0 is static."
            if dynamics
            else "Dynamics need a polycrystal in zero field with the detector along the spin, "
            "or a polycrystal in a longitudinal field."
        )
        self._dynamics_method.setEnabled(dynamics and self._collision_rate.value() > 0.0)
        has_sources = bool(self._sources)
        selected = self.selected_simulation_id() is not None
        self._add_button.setEnabled(self._document is not None and has_sources)
        self._recalculate_button.setEnabled(self._document is not None and selected)
        self._remove_button.setEnabled(selected)
        self._export_button.setEnabled(selected)

    def _dynamics_allowed(
        self,
        spin: tuple[float, float, float] | None = None,
        detector: tuple[float, float, float] | None = None,
    ) -> bool:
        """Whether the form's geometry allows dynamics (:func:`dynamics_supported`).

        Without ``spin`` the rule reads the form; a detector box that does not
        parse counts as not along the spin.
        """

        if spin is None:
            try:
                spin = self._spin.value("initial polarisation")
                if not self._detector_along_spin.isChecked():
                    detector = self._detector.value("detector direction")
            except ValueError:
                return False
            if not any(spin) or (detector is not None and not any(detector)):
                return False
        return dynamics_supported(
            self._geometry.currentData(), bool(self._sample.currentData()), spin, detector
        )

    def _emit_form_state(self) -> None:
        if not self._restoring:
            self.form_state_changed.emit(self.collect_state())

    # --- Plot -------------------------------------------------------------------------

    def _draw(self, record: PolarisationSimulationRecord | None) -> None:
        palette = current_palette()
        self._figure.clear()
        time_axis, spectrum_axis = self._figure.subplots(2, 1)
        for axis in (time_axis, spectrum_axis):
            style_plot_axes(axis)
        time_axis.set_xlabel(r"$t$ ($\mu$s)")
        time_axis.set_ylabel(r"$P(t)$")
        spectrum_axis.set_xlabel(r"$f$ (MHz)")
        spectrum_axis.set_ylabel("Amplitude (MHz⁻¹)")
        field_axis = spectrum_axis.secondary_xaxis(
            "top",
            functions=(
                lambda f: f / MUON_GAMMA_OVER_2PI_MHZ_T * 1e3,
                lambda b: b * MUON_GAMMA_OVER_2PI_MHZ_T / 1e3,
            ),
        )
        field_axis.set_xlabel(r"$B$ (mT)", color=palette.text_primary)
        field_axis.tick_params(colors=palette.text_primary)
        if record is None:
            time_axis.text(
                0.5,
                0.5,
                "Select or add a simulation.",
                ha="center",
                va="center",
                color=palette.text_muted,
                transform=time_axis.transAxes,
            )
            self._canvas.draw_idle()
            return

        static = record.simulation.static
        times = static.times_us
        sites = np.atleast_2d(static.site_polarisations)
        many = sites.shape[0] > _MAX_LEGEND_SITES
        if sites.shape[0] > 1:
            for label, trace in zip(static.site_labels, sites, strict=True):
                time_axis.plot(
                    times,
                    trace,
                    linewidth=0.6,
                    alpha=0.18 if many else 0.3,
                    label=None if many else label,
                )
        dynamic = record.simulation.dynamic
        if dynamic is not None:
            time_axis.plot(
                times,
                static.polarisation,
                linestyle="--",
                linewidth=1.0,
                alpha=0.7,
                color=palette.text_muted,
                label="static average",
            )
            time_axis.plot(
                times,
                dynamic.polarisation,
                linewidth=2.2,
                color=palette.text_primary,
                label=f"dynamic, ν = {dynamic.collision_rate_per_us:g} μs⁻¹",
            )
        else:
            time_axis.plot(
                times,
                static.polarisation,
                linewidth=2.2,
                color=palette.text_primary,
                label="weighted average" if sites.shape[0] > 1 else static.site_labels[0],
            )
        time_axis.set_xlim(float(times[0]), float(times[-1]))
        time_axis.set_ylim(-1.05, 1.05)
        style_plot_legend(time_axis.legend(loc="upper right", fontsize="small"))

        spectrum = record.simulation.spectrum
        nyquist = float(spectrum.frequencies_MHz[-1]) if spectrum.frequencies_MHz.size else 1.0
        spectrum_axis.plot(
            spectrum.frequencies_MHz, spectrum.amplitudes, linewidth=1.6, color=palette.text_primary
        )
        spectrum_axis.fill_between(
            spectrum.frequencies_MHz, spectrum.amplitudes, color=palette.text_primary, alpha=0.12
        )
        spectrum_axis.set_xlim(
            0.0, _spectrum_upper_limit(spectrum.frequencies_MHz, spectrum.amplitudes, nyquist)
        )
        spectrum_axis.set_ylim(bottom=0.0)
        self._canvas.draw_idle()

    # --- Export -----------------------------------------------------------------------

    def _export_data(self) -> None:
        record = self._selected_record()
        if record is None:
            return
        path = ask_save_path(
            self,
            "Export Polarisation",
            f"{_safe_name(record.title)}.dat",
            EXPORT_FORMATS,
        )
        if path is None:
            return
        try:
            Path(path).write_text(export_text(record), encoding="utf-8")
        except OSError as exc:
            report_export_failed(self, exc)
            return
        report_export_saved(self, path)


def export_text(record: PolarisationSimulationRecord) -> str:
    """The simulation as text: a header, the time block, a blank line, the spectrum block."""

    static = record.simulation.static
    lines = [
        "# MuFinder polarisation export",
        f"# Simulation: {record.title}",
        f"# Sources: {', '.join(f'{s.title} (weight {s.weight:g})' for s in record.sources)}",
    ]
    if record.estimate:
        lines.append(f"# Estimate: {UNRELAXED_RESULTS_NOTE}")
    for key, value in record.settings.to_json().items():
        lines.append(f"# {key}: {value}")
    for warning in record.simulation.warnings:
        lines.append(f"# Warning: {warning}")
    labels = _unique_column_labels(static.site_labels)
    columns = ["t_us", *(f"P_{label}" for label in labels), "P_average"]
    rows = [static.times_us, *np.atleast_2d(static.site_polarisations), static.polarisation]
    if record.simulation.dynamic is not None:
        columns.append("P_dynamic")
        rows.append(record.simulation.dynamic.polarisation)
    lines.append("# Time domain. Columns: " + " ".join(columns))
    for values in np.column_stack(rows):
        lines.append(" ".join(f"{value:.12g}" for value in values))
    spectrum = record.simulation.spectrum
    lines.append("")
    lines.append("# Spectrum. Columns: f_MHz B_T amplitude_per_MHz")
    for values in np.column_stack(
        (spectrum.frequencies_MHz, spectrum.fields_T, spectrum.amplitudes)
    ):
        lines.append(" ".join(f"{value:.12g}" for value in values))
    return "\n".join(lines) + "\n"


def _spectrum_upper_limit(frequencies: np.ndarray, amplitudes: np.ndarray, nyquist: float) -> float:
    """Show the spectrum up to a little past its last significant amplitude."""

    if amplitudes.size == 0 or not np.any(amplitudes > 0.0):
        return max(nyquist, 1e-6)
    significant = np.nonzero(amplitudes >= 0.03 * float(np.max(amplitudes)))[0]
    upper = float(frequencies[significant[-1]]) * 1.15
    return min(max(upper, 1e-6), nyquist) if nyquist > 0.0 else upper


def _rate_input(tooltip: str) -> QDoubleSpinBox:
    spin = QDoubleSpinBox()
    spin.setRange(0.0, 1_000.0)
    spin.setDecimals(4)
    spin.setSuffix(" μs⁻¹")
    spin.setToolTip(tooltip)
    spin.setMaximumWidth(140)
    return spin


def _optional_time_input(tooltip: str) -> QLineEdit:
    edit = QLineEdit()
    edit.setValidator(QDoubleValidator(0.0, 1_000_000.0, 6, edit))
    edit.setPlaceholderText("none")
    edit.setMaximumWidth(90)
    edit.setToolTip(f"{tooltip} in μs; leave empty for none.")
    return edit


def _optional_positive(edit: QLineEdit, name: str) -> float | None:
    text = edit.text().strip()
    if not text:
        return None
    try:
        value = float(text)
    except ValueError as exc:
        raise ValueError(f"{name} must be a decimal number of μs") from exc
    if not value > 0.0:
        raise ValueError(f"{name} must be > 0, or empty for none")
    return value


def _optional_text(value: float | None) -> str:
    return "" if value is None else f"{value:g}"


def _unique_column_labels(labels: Sequence[str]) -> list[str]:
    """Column-safe site labels, numbered ``_2``, ``_3``… where two would read the same."""

    unique: list[str] = []
    for index, label in enumerate(labels, start=1):
        base = _column_label(label) or f"site{index}"
        name, count = base, 1
        while name in unique:
            count += 1
            name = f"{base}_{count}"
        unique.append(name)
    return unique


def _column_label(label: str) -> str:
    return "".join(character if character.isalnum() else "_" for character in label).strip("_")


def _safe_name(title: str) -> str:
    return _column_label(title) or "polarisation"
