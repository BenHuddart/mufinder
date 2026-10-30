"""Dialog for analyzing completed convergence results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import replace
from html import escape
import math
from typing import Any

import numpy as np
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.convergence_plot import ConvergencePlotData, draw_convergence_plot
from mufinder.app.convergence_types import KPointGrid, SweepPointRunState
from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.app.plot_canvas import create_plot_canvas
from mufinder.app.screen_utils import clamped_size
from mufinder.app.theme import current_palette, plot_badge_box
from mufinder.castep_io.output_parser import (
    CASTEP_FORCE_PRINT_RESOLUTION_EV_ANG,
    CASTEP_STRESS_PRINT_RESOLUTION_GPA,
)
from mufinder.domain.models import CONVERGED_DEFAULT_METHOD
from mufinder.science.convergence_analysis import (
    ConvergenceAnalysis,
    ConvergenceResult,
    analyze_convergence,
    analyze_kpoint_convergence,
)
from mufinder.science.kpoint_grids import (
    distinct_grid_spacings,
    format_kpoint_grid,
)
from mufinder.services.convergence_defaults import kpoint_grid_label


#: Smallest thresholds the fields accept. Forces and stresses cannot be compared
#: more finely than CASTEP prints them, so their floors are that printed
#: resolution; a finer threshold could only be met by values that print
#: identically. Energies are printed to 13 significant figures, so the energy
#: threshold only has to be positive.
_THRESHOLD_DECIMALS = 12
_MIN_ENERGY_THRESHOLD = 10.0**-_THRESHOLD_DECIMALS
_MIN_FORCE_THRESHOLD = CASTEP_FORCE_PRINT_RESOLUTION_EV_ANG
_MIN_STRESS_THRESHOLD = CASTEP_STRESS_PRINT_RESOLUTION_GPA


_NO_RECOMMENDED_GRID_TOOLTIP = (
    "None of the examined grids has the recommended k-point count. "
    "Run the analysis again to recommend one of them."
)
_STRESS_DISABLED_TOOLTIP = (
    "All completed sweep jobs must have CALCULATE_STRESS=true (and report a max stress) "
    "to enable stress convergence analysis."
)


#: One row of the results box: a label and its lines. Rows are grouped, and a
#: blank line separates the groups.
ResultRow = tuple[str, tuple[str, ...]]


class ConvergenceExaminerDialog(QDialog):
    """Analyze completed runs and recommend a converged sweep value."""

    #: The adopted sweep key, the method and the energy, force and stress
    #: thresholds. The key is the cutoff in eV (``float``) or the k-point grid
    #: (``KPointGrid``) of the recommended point.
    default_adopted = Signal(object, str, float, float, object)
    state_changed = Signal(dict)

    def __init__(
        self,
        sweep_data: Mapping[float, SweepPointRunState],
        *,
        num_atoms: int,
        sweep_kind: str = "cutoff",
        point_grids: Mapping[float, KPointGrid] | None = None,
        lattice_vectors: Sequence[Sequence[float]] | None = None,
        initial_state: Mapping[str, object] | None = None,
        notes: Sequence[str] = (),
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        # Caller-supplied caveats about the sweep data (for example grids left out
        # because they share a k-point count), repeated in every results summary.
        self._notes = tuple(str(note) for note in notes if str(note).strip())
        if sweep_kind not in {"cutoff", "kpoint"}:
            raise ValueError("sweep_kind must be 'cutoff' or 'kpoint'.")

        self._sweep_kind = sweep_kind
        self._lattice_vectors: tuple[tuple[float, float, float], ...] | None = (
            tuple(tuple(float(component) for component in vector) for vector in lattice_vectors)
            if lattice_vectors is not None
            else None
        )
        # The grid of each k-point count on the plot axis, as the caller passed it.
        self._point_grids: dict[float, KPointGrid] = {}
        self._point_spacings: dict[float, float] = {}
        if self._sweep_kind == "kpoint":
            self._point_grids = {
                float(value): (int(grid[0]), int(grid[1]), int(grid[2]))
                for value, grid in (point_grids or {}).items()
            }
            if self._lattice_vectors is not None:
                self._point_spacings = distinct_grid_spacings(
                    self._lattice_vectors, self._point_grids
                )
        self._hover_annotation = None
        # Axis references are populated in _draw_plot. Twin axes sit on
        # top of their primary axis and capture mouse events, so the hover
        # handler needs to route those back to the logical subplot.
        self._energy_ax = None
        self._force_ax = None
        self._spacing_ax = None
        self._stress_ax = None

        self.setWindowTitle(
            "Examine Cutoff Convergence"
            if self._sweep_kind == "cutoff"
            else "Examine K-Point Convergence"
        )
        self.resize(clamped_size(self, 1080, 900))
        self.setMinimumSize(880, 640)

        self._num_atoms = int(num_atoms)
        self._completed = self._collect_completed(sweep_data)
        self._energy_result: ConvergenceResult | None = None
        self._force_result: ConvergenceResult | None = None
        self._stress_result: ConvergenceResult | None = None
        self._recommended_value: float | None = None
        self._result_groups: list[list[ResultRow]] = []
        self._energy_threshold_for_recommendation: float | None = None
        self._force_threshold_for_recommendation: float | None = None
        self._stress_threshold_for_recommendation: float | None = None
        self._last_plot_data: dict[str, Any] | None = None

        self._energy_threshold = DecimalLineEdit(self)
        self._energy_threshold.setRange(_MIN_ENERGY_THRESHOLD, 1.0)
        self._energy_threshold.setDecimals(_THRESHOLD_DECIMALS)
        self._energy_threshold.setSingleStep(0.001)
        self._energy_threshold.setValue(0.001)

        self._force_threshold = DecimalLineEdit(self)
        self._force_threshold.setRange(_MIN_FORCE_THRESHOLD, 1.0)
        self._force_threshold.setDecimals(_THRESHOLD_DECIMALS)
        self._force_threshold.setSingleStep(0.01)
        self._force_threshold.setValue(0.05)

        self._include_stress = QCheckBox("Stress threshold (GPa)", self)
        self._include_stress.setChecked(False)
        self._stress_threshold = DecimalLineEdit(self)
        self._stress_threshold.setRange(_MIN_STRESS_THRESHOLD, 100.0)
        self._stress_threshold.setDecimals(_THRESHOLD_DECIMALS)
        self._stress_threshold.setSingleStep(0.05)
        self._stress_threshold.setValue(0.1)
        self._stress_threshold.setEnabled(False)

        self._run_button = QPushButton("Analyse Convergence", self)
        self._results = QTextEdit(self)
        self._results.setReadOnly(True)

        self._figure, self._canvas = create_plot_canvas(
            figsize=(8.0, 5.5), facecolor=current_palette().bg_surface
        )
        layout_engine = self._figure.get_layout_engine()
        if layout_engine is not None:
            layout_engine.set(w_pad=0.04, h_pad=0.04, hspace=0.08, wspace=0.04)
        self._canvas.setMinimumHeight(520)

        self._adopt_button = QPushButton(
            "Set as Project Default Cutoff"
            if self._sweep_kind == "cutoff"
            else "Set as Project Default K-Point Grid",
            self,
        )
        self._adopt_button.setEnabled(False)

        self._build_layout()
        self._connect_signals()
        self._update_availability()
        self._write_initial_summary()
        self._draw_initial_plot()
        self._restore_initial_state(initial_state)

    def _build_layout(self) -> None:
        outer_layout = QVBoxLayout(self)
        outer_layout.setContentsMargins(0, 0, 0, 0)

        scroll = QScrollArea(self)
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        outer_layout.addWidget(scroll)

        content = QWidget(scroll)
        scroll.setWidget(content)
        layout = QVBoxLayout(content)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(12)

        criteria_group = QGroupBox("Convergence Criteria", self)
        criteria_form = QFormLayout(criteria_group)
        criteria_form.setHorizontalSpacing(16)
        criteria_form.setVerticalSpacing(12)
        criteria_form.addRow("Energy threshold (eV/atom)", self._energy_threshold)
        criteria_form.addRow("Force threshold (eV/Ang)", self._force_threshold)
        criteria_form.addRow(self._include_stress, self._stress_threshold)
        self._force_threshold.setToolTip(
            "Force convergence threshold (eV/Ang). The minimum is "
            f"{_MIN_FORCE_THRESHOLD:.5f} eV/Ang, the precision CASTEP prints forces to."
        )
        self._stress_threshold.setToolTip(
            "Stress convergence threshold (GPa). Mirrors GEOM_STRESS_TOL. The minimum is "
            f"{_MIN_STRESS_THRESHOLD:.6f} GPa, the precision CASTEP prints stresses to."
        )
        layout.addWidget(criteria_group)

        action_row = QHBoxLayout()
        action_row.setSpacing(12)
        action_row.addStretch(1)
        action_row.addWidget(self._run_button)
        action_row.addWidget(self._adopt_button)
        layout.addLayout(action_row)

        self._results.setMinimumHeight(220)
        self._results.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding
        )
        layout.addWidget(self._results)
        layout.addWidget(self._canvas)
        layout.addStretch(1)

    def _connect_signals(self) -> None:
        self._run_button.clicked.connect(self._run_analysis)
        self._adopt_button.clicked.connect(self._emit_adopted_default)
        self._canvas.mpl_connect("motion_notify_event", self._on_plot_hover)
        self._canvas.mpl_connect("figure_leave_event", self._on_plot_leave)
        self._include_stress.toggled.connect(self._on_include_stress_toggled)

    def _on_include_stress_toggled(self, checked: bool) -> None:
        self._stress_threshold.setEnabled(bool(checked))
        self._emit_state_changed()

    def _sweep_point_tooltip(self, x_value: float) -> str:
        if self._sweep_kind == "cutoff":
            return f"Cutoff: {x_value:.8g} eV"
        grid = self._point_grids.get(float(x_value))
        if grid is not None:
            return kpoint_grid_label(grid)
        return f"Grid: {x_value:.8g} k-points"

    @staticmethod
    def _annotation_box_style() -> dict[str, object]:
        return plot_badge_box(pad=0.3)

    def _annotation_text_color(self) -> str:
        return current_palette().text_primary

    def _on_plot_leave(self, _event) -> None:
        if self._hover_annotation is not None:
            self._hover_annotation.set_visible(False)
            self._canvas.draw_idle()

    def _on_plot_hover(self, event) -> None:
        if self._last_plot_data is None or self._energy_ax is None or self._force_ax is None:
            return
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            self._on_plot_leave(event)
            return

        cutoffs = list(self._last_plot_data.get("cutoffs", []))
        energies = list(self._last_plot_data.get("energies", []))
        forces = list(self._last_plot_data.get("forces", []))
        stresses = list(self._last_plot_data.get("stresses", []) or [])
        if not cutoffs:
            return

        # Twin axes (spacing on energy, stress on force) sit on top of their
        # primary axis and capture the mouse event. Route them back to the
        # logical subplot the user is hovering over, and anchor the annotation
        # on the primary axis so the y-coordinate is in the right data frame.
        energy_axes = {self._energy_ax}
        if self._spacing_ax is not None:
            energy_axes.add(self._spacing_ax)
        force_axes = {self._force_ax}
        if self._stress_ax is not None:
            force_axes.add(self._stress_ax)

        has_stress = bool(stresses) and len(stresses) == len(cutoffs)

        points: list[tuple[float, float, str]] = []
        if event.inaxes in energy_axes:
            anchor_ax = self._energy_ax
            if self._sweep_kind == "kpoint":
                points = [
                    (x_value, y_value, self._sweep_point_tooltip(x_value))
                    for x_value, y_value in zip(cutoffs, energies, strict=False)
                ]
            else:
                points = [
                    (
                        x_value,
                        y_value,
                        f"{self._sweep_point_tooltip(x_value)}\nEnergy: {y_value:.12g} eV",
                    )
                    for x_value, y_value in zip(cutoffs, energies, strict=False)
                ]
        elif event.inaxes in force_axes:
            # In cutoff mode, hovering the stress twin should anchor the
            # annotation on the stress axis so the arrow points at the
            # stress data point (not at the force line at that x).
            on_stress_twin = (
                self._sweep_kind == "cutoff"
                and event.inaxes is self._stress_ax
                and self._stress_ax is not None
                and has_stress
            )
            if on_stress_twin:
                anchor_ax = self._stress_ax
                for idx, (x_value, stress_y) in enumerate(zip(cutoffs, stresses, strict=False)):
                    text = f"{self._sweep_point_tooltip(x_value)}\nMax stress: {stress_y:.6g} GPa"
                    if idx < len(forces):
                        text += f"\nMax force: {forces[idx]:.12g} eV/Ang"
                    points.append((x_value, stress_y, text))
            else:
                anchor_ax = self._force_ax
                if self._sweep_kind == "kpoint":
                    points = [
                        (x_value, y_value, self._sweep_point_tooltip(x_value))
                        for x_value, y_value in zip(cutoffs, forces, strict=False)
                    ]
                else:
                    # Cutoff mode on primary force axis: surface both metrics
                    # in one tooltip.
                    for idx, (x_value, y_value) in enumerate(zip(cutoffs, forces, strict=False)):
                        text = (
                            f"{self._sweep_point_tooltip(x_value)}"
                            f"\nMax force: {y_value:.12g} eV/Ang"
                        )
                        if has_stress and idx < len(stresses):
                            text += f"\nMax stress: {stresses[idx]:.6g} GPa"
                        points.append((x_value, y_value, text))
        else:
            return

        x_range = max(cutoffs) - min(cutoffs) if len(cutoffs) > 1 else max(abs(cutoffs[0]), 1.0)
        x_tol = max(x_range * 0.03, 0.5)

        nearest = min(points, key=lambda point: abs(point[0] - float(event.xdata)))
        if abs(nearest[0] - float(event.xdata)) > x_tol:
            self._on_plot_leave(event)
            return

        if self._hover_annotation is None:
            self._hover_annotation = anchor_ax.annotate(
                nearest[2],
                xy=(nearest[0], nearest[1]),
                xytext=(10, 10),
                textcoords="offset points",
                bbox=self._annotation_box_style(),
                fontsize=9,
                color=self._annotation_text_color(),
            )
        else:
            self._hover_annotation.set_visible(True)
            self._hover_annotation.xy = (nearest[0], nearest[1])
            self._hover_annotation.set_text(nearest[2])
            self._hover_annotation.set_figure(self._figure)
            self._hover_annotation.set_bbox(self._annotation_box_style())
            self._hover_annotation.set_color(self._annotation_text_color())
            if self._hover_annotation.axes is not anchor_ax:
                try:
                    self._hover_annotation.remove()
                except (NotImplementedError, ValueError):
                    # Already detached from the figure; nothing to remove.
                    pass
                self._hover_annotation = anchor_ax.annotate(
                    nearest[2],
                    xy=(nearest[0], nearest[1]),
                    xytext=(10, 10),
                    textcoords="offset points",
                    bbox=self._annotation_box_style(),
                    fontsize=9,
                    color=self._annotation_text_color(),
                )

        self._canvas.draw_idle()

    def _collect_completed(
        self, sweep_data: Mapping[float, SweepPointRunState]
    ) -> list[tuple[float, SweepPointRunState]]:
        completed: list[tuple[float, SweepPointRunState]] = []
        for cutoff, state in sweep_data.items():
            if state.status.strip().lower() != "completed":
                continue
            completed.append((float(cutoff), state))
        completed.sort(key=lambda item: item[0])
        return completed

    def _energy_series(self) -> tuple[list[float], list[float]]:
        cutoffs: list[float] = []
        energies: list[float] = []
        for cutoff, state in self._completed:
            if state.total_energy is None:
                continue
            cutoffs.append(cutoff)
            energies.append(float(state.total_energy))
        return cutoffs, energies

    def _force_series(self) -> tuple[list[float], list[float]]:
        cutoffs: list[float] = []
        forces: list[float] = []
        for cutoff, state in self._completed:
            if state.max_force is None:
                continue
            cutoffs.append(cutoff)
            forces.append(float(state.max_force))
        return cutoffs, forces

    def _stress_series(self) -> tuple[list[float], list[float]]:
        cutoffs: list[float] = []
        stresses: list[float] = []
        for cutoff, state in self._completed:
            if state.max_stress is None:
                continue
            cutoffs.append(cutoff)
            stresses.append(float(state.max_stress))
        return cutoffs, stresses

    def _stress_available_for_all_completed(self) -> bool:
        if not self._completed:
            return False
        return all(state.max_stress is not None for _cutoff, state in self._completed)

    def _update_availability(self) -> None:
        cutoffs, _energies = self._energy_series()
        force_cutoffs, _forces = self._force_series()
        can_run = len(cutoffs) >= 2 and len(force_cutoffs) >= 2 and self._num_atoms > 0
        self._run_button.setEnabled(can_run)

        stress_enabled = self._stress_available_for_all_completed() and len(cutoffs) >= 2
        if not stress_enabled and self._include_stress.isChecked():
            self._include_stress.setChecked(False)
        self._include_stress.setEnabled(stress_enabled)
        self._include_stress.setToolTip("" if stress_enabled else _STRESS_DISABLED_TOOLTIP)
        self._stress_threshold.setEnabled(stress_enabled and self._include_stress.isChecked())

        if not can_run:
            self._reset_adopt_button()

    def _write_initial_summary(self, extra_rows: Sequence[ResultRow] = ()) -> None:
        energy_cutoffs, _energies = self._energy_series()
        force_cutoffs, _forces = self._force_series()
        stress_cutoffs, _stresses = self._stress_series()

        summary: list[ResultRow] = [
            ("Completed points", (str(len(self._completed)),)),
            (
                "With values",
                (
                    f"{len(energy_cutoffs)} energy, {len(force_cutoffs)} force, "
                    f"{len(stress_cutoffs)} stress",
                ),
            ),
            ("Atoms", (f"{self._num_atoms}, for the energy per atom",)),
        ]
        if len(energy_cutoffs) < 2 or len(force_cutoffs) < 2:
            summary.append(
                (
                    "Analysis",
                    (
                        "Need at least two completed rows with both energy and max-force "
                        "values to run analysis.",
                    ),
                )
            )
        elif self._num_atoms <= 0:
            summary.append(
                (
                    "Analysis",
                    ("Simulation-cell atom count must be positive before analysis can run.",),
                )
            )
        summary.extend(extra_rows)
        self._show_results([summary, *self._note_groups()])

    def _aligned_points(self) -> tuple[list[float], list[float], list[float]]:
        """Return the sweep values with both an energy and a force, and those series."""

        energy_by_value = dict(zip(*self._energy_series(), strict=True))
        force_by_value = dict(zip(*self._force_series(), strict=True))
        values = sorted(set(energy_by_value) & set(force_by_value))
        return (
            values,
            [energy_by_value[value] for value in values],
            [force_by_value[value] for value in values],
        )

    def _draw_initial_plot(self) -> None:
        """Plot the completed points before any analysis: no thresholds or marks.

        A point with only an energy or only a force is shown in its panel; the
        missing value is left as a gap.
        """

        energy_by_value = dict(zip(*self._energy_series(), strict=True))
        force_by_value = dict(zip(*self._force_series(), strict=True))
        values = sorted(set(energy_by_value) | set(force_by_value))
        self._draw_plot(
            ConvergencePlotData(
                sweep_values=tuple(values),
                energies=tuple(energy_by_value.get(value, math.nan) for value in values),
                forces=tuple(force_by_value.get(value, math.nan) for value in values),
            ),
            analysed=False,
        )

    @staticmethod
    def _serialize_result(result: ConvergenceResult | None) -> dict[str, object] | None:
        if result is None:
            return None
        return {
            "converged": bool(result.converged),
            "quantity": result.quantity,
            "summary": result.summary,
            "explanation": result.explanation,
            "converged_value": result.converged_value,
            "converged_cutoff": result.converged_value,
            "reference_value": result.reference_value,
            "deviations": list(result.deviations) if result.deviations is not None else None,
            "fitted_curve": list(result.fitted_curve) if result.fitted_curve is not None else None,
            "extrapolated_energy": result.extrapolated_energy,
        }

    @staticmethod
    def _deserialize_result(payload: object) -> ConvergenceResult | None:
        if not isinstance(payload, dict):
            return None

        def optional_float(raw: object) -> float | None:
            if raw is None:
                return None
            try:
                return float(raw)  # type: ignore[arg-type]
            except (TypeError, ValueError):
                return None

        def parse_series(raw: object) -> tuple[tuple[float, float], ...] | None:
            if not isinstance(raw, list):
                return None
            values: list[tuple[float, float]] = []
            for item in raw:
                if not isinstance(item, (list, tuple)) or len(item) != 2:
                    continue
                try:
                    values.append((float(item[0]), float(item[1])))
                except (TypeError, ValueError):
                    continue
            return tuple(values)

        return ConvergenceResult(
            converged=bool(payload.get("converged", False)),
            quantity=str(payload.get("quantity", "")),
            summary=str(payload.get("summary", "")),
            explanation=str(payload.get("explanation", "")),
            converged_value=optional_float(
                payload.get("converged_value", payload.get("converged_cutoff"))
            ),
            reference_value=optional_float(payload.get("reference_value")),
            deviations=parse_series(payload.get("deviations")),
            fitted_curve=parse_series(payload.get("fitted_curve")),
            extrapolated_energy=optional_float(payload.get("extrapolated_energy")),
        )

    def state_snapshot(self) -> dict[str, object]:
        """Return a persistable snapshot of controls, output text, and plot inputs."""
        return {
            "energy_threshold": float(self._energy_threshold.value()),
            "force_threshold": float(self._force_threshold.value()),
            "stress_threshold": float(self._stress_threshold.value()),
            "include_stress": bool(self._include_stress.isChecked()),
            "sweep_kind": self._sweep_kind,
            "results_rows": [
                [[label, list(lines)] for label, lines in group] for group in self._result_groups
            ],
            "recommended_value": self._recommended_value,
            "recommended_cutoff": self._recommended_value,
            "energy_threshold_for_recommendation": self._energy_threshold_for_recommendation,
            "force_threshold_for_recommendation": self._force_threshold_for_recommendation,
            "stress_threshold_for_recommendation": self._stress_threshold_for_recommendation,
            "energy_result": self._serialize_result(self._energy_result),
            "force_result": self._serialize_result(self._force_result),
            "stress_result": self._serialize_result(self._stress_result),
            "plot_data": dict(self._last_plot_data or {}),
        }

    def _emit_state_changed(self) -> None:
        self.state_changed.emit(self.state_snapshot())

    def _restore_initial_state(self, state: Mapping[str, object] | None) -> None:
        if not isinstance(state, Mapping):
            return

        energy_threshold_raw = state.get("energy_threshold")
        force_threshold_raw = state.get("force_threshold")
        stress_threshold_raw = state.get("stress_threshold")
        if energy_threshold_raw is not None:
            try:
                self._energy_threshold.setValue(float(energy_threshold_raw))
            except (TypeError, ValueError):
                pass
        if force_threshold_raw is not None:
            try:
                self._force_threshold.setValue(float(force_threshold_raw))
            except (TypeError, ValueError):
                pass
        if stress_threshold_raw is not None:
            try:
                self._stress_threshold.setValue(float(stress_threshold_raw))
            except (TypeError, ValueError):
                pass

        include_stress_raw = state.get("include_stress")
        if isinstance(include_stress_raw, bool) and self._include_stress.isEnabled():
            self._include_stress.setChecked(include_stress_raw)
            self._stress_threshold.setEnabled(include_stress_raw)

        # An analysis saved before the single reference criterion carries the
        # name of the method that produced it. Its results and recommendation
        # are discarded; the thresholds and the plotted points are kept.
        legacy_analysis = "method" in state

        self._energy_result = self._deserialize_result(state.get("energy_result"))
        self._force_result = self._deserialize_result(state.get("force_result"))
        self._stress_result = self._deserialize_result(state.get("stress_result"))

        recommended_raw = state.get("recommended_value", state.get("recommended_cutoff"))
        if recommended_raw is not None:
            try:
                self._recommended_value = float(recommended_raw)
            except (TypeError, ValueError):
                self._recommended_value = None
        else:
            self._recommended_value = None

        energy_threshold_raw = state.get("energy_threshold_for_recommendation")
        if energy_threshold_raw is not None:
            try:
                self._energy_threshold_for_recommendation = float(energy_threshold_raw)
            except (TypeError, ValueError):
                self._energy_threshold_for_recommendation = None
        else:
            self._energy_threshold_for_recommendation = None

        force_threshold_raw = state.get("force_threshold_for_recommendation")
        if force_threshold_raw is not None:
            try:
                self._force_threshold_for_recommendation = float(force_threshold_raw)
            except (TypeError, ValueError):
                self._force_threshold_for_recommendation = None
        else:
            self._force_threshold_for_recommendation = None

        stress_threshold_for_rec_raw = state.get("stress_threshold_for_recommendation")
        if stress_threshold_for_rec_raw is not None:
            try:
                self._stress_threshold_for_recommendation = float(stress_threshold_for_rec_raw)
            except (TypeError, ValueError):
                self._stress_threshold_for_recommendation = None
        else:
            self._stress_threshold_for_recommendation = None

        if legacy_analysis:
            self._energy_result = None
            self._force_result = None
            self._stress_result = None
            self._recommended_value = None
            self._energy_threshold_for_recommendation = None
            self._force_threshold_for_recommendation = None
            self._stress_threshold_for_recommendation = None
            self._write_initial_summary(
                (
                    (
                        "Earlier analysis",
                        (
                            "set aside: it used an energy method this version no longer offers",
                            "click Analyse Convergence to judge the sweep again",
                        ),
                    ),
                )
            )
        else:
            groups = self._parse_result_groups(state.get("results_rows"))
            if groups:
                self._show_results(groups)

        plot_data = ConvergencePlotData.from_mapping(
            state.get("plot_data"),
            default_energy_threshold=self._energy_threshold.value(),
            default_force_threshold=self._force_threshold.value(),
        )
        if plot_data is not None:
            plot_data = replace(plot_data, point_spacings=dict(self._point_spacings))
            self._last_plot_data = plot_data.to_mapping()
            self._draw_plot(plot_data)

        if self._recommended_value is not None:
            self._offer_adoption()
            self._emit_state_changed()
        elif legacy_analysis:
            self._emit_state_changed()

    @staticmethod
    def _parse_result_groups(raw: object) -> list[list[ResultRow]]:
        groups: list[list[ResultRow]] = []
        if not isinstance(raw, list):
            return groups
        for raw_group in raw:
            if not isinstance(raw_group, list):
                continue
            group: list[ResultRow] = []
            for row in raw_group:
                if (
                    isinstance(row, (list, tuple))
                    and len(row) == 2
                    and isinstance(row[0], str)
                    and isinstance(row[1], (list, tuple))
                ):
                    group.append((row[0], tuple(str(line) for line in row[1])))
            if group:
                groups.append(group)
        return groups

    def _note_groups(self) -> list[list[ResultRow]]:
        if not self._notes:
            return []
        return [[("Notes", self._notes)]]

    def _show_results(self, groups: Sequence[Sequence[ResultRow]]) -> None:
        """Show labelled rows: the label in bold on the left, each line beside it.

        The first line of a row is the result; any further lines explain it in
        muted text. A blank row separates the groups.
        """

        self._result_groups = [list(group) for group in groups if group]
        palette = current_palette()
        cells: list[str] = []
        for index, group in enumerate(self._result_groups):
            if index:
                cells.append('<tr><td colspan="2" style="font-size:6px">&nbsp;</td></tr>')
            for label, lines in group:
                for line_index, line in enumerate(lines or ("",)):
                    label_cell = f"<b>{escape(label)}</b>" if line_index == 0 else ""
                    color = palette.text_primary if line_index == 0 else palette.text_muted
                    if line_index == 0 and self._is_warning_row(label, line):
                        color = palette.warning
                    cells.append(
                        f'<tr><td style="padding-right:16px">{label_cell}</td>'
                        f'<td style="color:{color}">{escape(line)}</td></tr>'
                    )
        self._results.setHtml(f'<table cellspacing="0" cellpadding="1">{"".join(cells)}</table>')

    @staticmethod
    def _is_warning_row(label: str, line: str) -> bool:
        return (label == "Sweep length" and line == "too short") or (
            label.startswith("Recommended") and line == "none"
        )

    def _recommended_point_text(self) -> str:
        """Return the recommended k-point grid as ``4 x 4 x 4`` (or a count fallback)."""

        value = float(self._recommended_value)  # type: ignore[arg-type]
        grid = self._point_grids.get(value)
        if grid is not None:
            return format_kpoint_grid(grid)
        return f"{value:.8g} k-points"

    def _adopted_key(self) -> float | KPointGrid | None:
        """Return the sweep key the adopt button sets: the cutoff, or the point's grid.

        ``None`` when nothing is recommended, or when a recommended k-point
        count (restored from saved state) has no grid among the examined points.
        """

        if self._recommended_value is None:
            return None
        if self._sweep_kind == "cutoff":
            return float(self._recommended_value)
        return self._point_grids.get(float(self._recommended_value))

    def _reset_adopt_button(self) -> None:
        """Disable the adopt button with its neutral text: nothing to adopt."""

        self._adopt_button.setEnabled(False)
        self._adopt_button.setText(
            "Set as Project Default Cutoff"
            if self._sweep_kind == "cutoff"
            else "Set as Project Default K-Point Grid"
        )
        self._adopt_button.setToolTip("")

    def _offer_adoption(self) -> None:
        """Name the recommendation on the adopt button, enabled when it can be set."""

        settable = self._adopted_key() is not None
        self._adopt_button.setEnabled(settable)
        self._adopt_button.setText(self._adopt_button_text())
        self._adopt_button.setToolTip("" if settable else _NO_RECOMMENDED_GRID_TOOLTIP)

    def _adopt_button_text(self) -> str:
        if self._sweep_kind == "cutoff":
            return f"Set {self._recommended_value:.8g} eV as Project Default Cutoff"
        return f"Set {self._recommended_point_text()} as Project Default K-Point Grid"

    def _analysis_groups(
        self, analysis: ConvergenceAnalysis, stress_threshold: float | None
    ) -> list[list[ResultRow]]:
        def number(value: float) -> str:
            return np.format_float_positional(float(value), trim="-")

        thresholds = [
            f"energy {number(self._energy_threshold.value())} eV/atom",
            f"force {number(self._force_threshold.value())} eV/Ang",
        ]
        if analysis.stress is not None and stress_threshold is not None:
            thresholds.append(f"stress {number(stress_threshold)} GPa")
        reference_lines: tuple[str, ...] = (analysis.reference_text,)
        if self._sweep_kind == "kpoint":
            reference_lines += ("a sweep should extend beyond the grid it is meant to certify",)
        groups: list[list[ResultRow]] = [
            [("Thresholds", (", ".join(thresholds),)), ("Reference", reference_lines)],
            [
                (result.quantity, (result.summary, result.explanation))
                for result in (analysis.energy, analysis.force, analysis.stress)
                if result is not None
            ],
        ]
        if analysis.sweep_length is not None:
            groups.append(
                [
                    (
                        "Sweep length",
                        (analysis.sweep_length.summary, analysis.sweep_length.explanation),
                    )
                ]
            )

        label = "Recommended cutoff" if self._sweep_kind == "cutoff" else "Recommended k-point grid"
        if self._recommended_value is not None:
            if self._sweep_kind == "cutoff":
                value = f"{self._recommended_value:.8g} eV"
            else:
                value = self._recommended_point_text()
                grid = self._point_grids.get(float(self._recommended_value))
                spacing = self._point_spacings.get(float(self._recommended_value))
                if spacing is not None and grid is not None and grid != (1, 1, 1):
                    value += f" (spacing {spacing:.4g} 1/Ang)"
            groups.append([(label, (value,))])
        else:
            unconverged = analysis.unconverged
            setting = "cutoffs" if self._sweep_kind == "cutoff" else "denser k-point grids"
            criteria = " and ".join(unconverged) + (
                " criterion" if len(unconverged) == 1 else " criteria"
            )
            groups.append(
                [
                    (
                        label,
                        (
                            "none",
                            f"the {criteria} did not converge within the sweep; extend the "
                            f"sweep to larger {setting}, or relax the threshold, and analyse "
                            "again",
                        ),
                    )
                ]
            )
        return [*groups, *self._note_groups()]

    def _run_analysis(self) -> None:
        # Show the thresholds actually used: a typed value outside a field's range
        # is clamped when read, and clicking the button does not always end the
        # edit (macOS buttons take no focus), so snap the text here as well.
        for field in (self._energy_threshold, self._force_threshold, self._stress_threshold):
            field.setValue(field.value())
        self._update_availability()
        if not self._run_button.isEnabled():
            self._write_initial_summary()
            return

        cutoffs, energies, forces = self._aligned_points()

        if len(cutoffs) < 2:
            self._show_results(
                [
                    [
                        (
                            "Analysis",
                            (
                                "Need at least two matching completed rows containing both "
                                "total energy and max force values.",
                            ),
                        )
                    ],
                    *self._note_groups(),
                ]
            )
            self._reset_adopt_button()
            self._draw_initial_plot()
            return

        energy_threshold = float(self._energy_threshold.value())
        force_threshold = float(self._force_threshold.value())

        include_stress = self._include_stress.isEnabled() and self._include_stress.isChecked()
        stresses: list[float] = []
        stress_threshold: float | None = None
        if include_stress:
            stress_by_cutoff = {
                cutoff: stress for cutoff, stress in zip(*self._stress_series(), strict=False)
            }
            try:
                stresses = [stress_by_cutoff[cutoff] for cutoff in cutoffs]
            except KeyError:
                # A completed row used for energy/force does not have stress; fall back to no-stress run.
                include_stress = False
                stresses = []
            else:
                stress_threshold = float(self._stress_threshold.value())

        series = {
            "energies": energies,
            "max_forces": forces,
            "num_atoms": self._num_atoms,
            "energy_threshold": energy_threshold,
            "force_threshold": force_threshold,
            "max_stresses": stresses if include_stress else None,
            "stress_threshold": stress_threshold if include_stress else None,
        }
        try:
            if self._sweep_kind == "kpoint":
                analysis = analyze_kpoint_convergence(
                    kpoint_counts=cutoffs,
                    point_grids=self._point_grids,
                    point_spacings=self._point_spacings or None,
                    **series,
                )
            else:
                analysis = analyze_convergence(cutoffs=cutoffs, **series)
        except ValueError as exc:
            QMessageBox.warning(self, "Convergence analysis failed", str(exc))
            return

        self._energy_result = analysis.energy
        self._force_result = analysis.force
        self._stress_result = analysis.stress
        self._recommended_value = analysis.recommended
        self._energy_threshold_for_recommendation = energy_threshold
        self._force_threshold_for_recommendation = force_threshold
        self._stress_threshold_for_recommendation = stress_threshold if include_stress else None
        self._show_results(self._analysis_groups(analysis, stress_threshold))
        plot_data = ConvergencePlotData(
            sweep_values=tuple(cutoffs),
            energies=tuple(energies),
            forces=tuple(forces),
            energy_threshold=energy_threshold,
            force_threshold=force_threshold,
            stresses=tuple(stresses) if include_stress else (),
            stress_threshold=stress_threshold if include_stress else None,
            point_spacings=dict(self._point_spacings),
        )
        self._last_plot_data = plot_data.to_mapping()
        self._draw_plot(plot_data)

        if self._recommended_value is None:
            self._reset_adopt_button()
        else:
            self._offer_adoption()

        self._emit_state_changed()

    def _draw_plot(self, data: ConvergencePlotData, *, analysed: bool = True) -> None:
        """Draw ``data`` with the shared convergence plot, and the analysis marks when ``analysed``."""

        recommended = self._recommended_value if analysed else None
        recommended_grid = (
            self._point_grids.get(float(recommended)) if recommended is not None else None
        )
        self._hover_annotation = None
        axes = draw_convergence_plot(
            self._figure,
            data,
            kind=self._sweep_kind,
            compact=False,
            recommended=recommended,
            recommended_label=(
                format_kpoint_grid(recommended_grid) if recommended_grid is not None else None
            ),
            lattice_vectors=self._lattice_vectors,
            energy_result=self._energy_result if analysed else None,
            force_result=self._force_result if analysed else None,
            stress_result=self._stress_result if analysed else None,
        )
        self._energy_ax = axes.energy
        self._force_ax = axes.force
        self._spacing_ax = axes.spacing
        self._stress_ax = axes.stress
        self._canvas.draw_idle()

    def _emit_adopted_default(self) -> None:
        key = self._adopted_key()
        if key is None:
            return
        method = CONVERGED_DEFAULT_METHOD
        energy_threshold = self._energy_threshold_for_recommendation
        if energy_threshold is None:
            energy_threshold = float(self._energy_threshold.value())
        force_threshold = self._force_threshold_for_recommendation
        if force_threshold is None:
            force_threshold = float(self._force_threshold.value())
        stress_threshold = self._stress_threshold_for_recommendation
        self._emit_state_changed()
        self.default_adopted.emit(
            key,
            method,
            energy_threshold,
            force_threshold,
            stress_threshold,
        )
