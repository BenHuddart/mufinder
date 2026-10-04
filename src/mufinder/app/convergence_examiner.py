"""Dialog for analyzing completed convergence results."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
import math
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDialog,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.convergence_types import SweepPointRunState
from mufinder.app.decimal_input import DecimalLineEdit
from mufinder.app.screen_utils import clamped_size
from mufinder.app.theme import current_palette
from mufinder.castep_io.output_parser import (
    CASTEP_FORCE_PRINT_RESOLUTION_EV_ANG,
    CASTEP_STRESS_PRINT_RESOLUTION_GPA,
)
from mufinder.science.convergence_analysis import (
    ConvergenceResult,
    analyze_convergence,
    analyze_kpoint_convergence,
    apply_kpoint_label_to_stress_result,
    delta_threshold_stress,
    recommended_value,
    unconverged_criteria,
)
from mufinder.science.kpoint_grids import format_kpoint_grid, kpoint_count_for_spacing, spacing_from_grid


#: Thresholds must be positive; this is the smallest value the fields accept,
#: with enough decimals to type it.
_THRESHOLD_DECIMALS = 12
_MIN_THRESHOLD = 10.0 ** -_THRESHOLD_DECIMALS


def _precision_notes(force_threshold: float, stress_threshold: float | None) -> list[str]:
    """Explain thresholds finer than the precision CASTEP prints forces and stresses.

    Such a threshold is accepted, but a criterion can then only be met by two
    consecutive values that print identically.
    """

    notes: list[str] = []
    if force_threshold < CASTEP_FORCE_PRINT_RESOLUTION_EV_ANG:
        notes.append(
            f"The force threshold ({force_threshold:.6g} eV/Ang) is finer than the "
            f"{CASTEP_FORCE_PRINT_RESOLUTION_EV_ANG:g} eV/Ang to which CASTEP prints forces, "
            "so it is met only when consecutive maximum forces print identically."
        )
    if stress_threshold is not None and stress_threshold < CASTEP_STRESS_PRINT_RESOLUTION_GPA:
        notes.append(
            f"The stress threshold ({stress_threshold:.6g} GPa) is finer than the "
            f"{CASTEP_STRESS_PRINT_RESOLUTION_GPA:g} GPa to which CASTEP prints stresses, "
            "so it is met only when consecutive maximum stresses print identically."
        )
    return notes


_STRESS_DISABLED_TOOLTIP = (
    "All completed sweep jobs must have CALCULATE_STRESS=true (and report a max stress) "
    "to enable stress convergence analysis."
)
_STRESS_COLOR = "#d97706"

# Mantissa values combined with each decade form the "nice" candidate
# spacings rendered on the secondary k-point spacing axis. The axis filters
# these down to the ones whose CASTEP-equivalent k-point count falls inside
# the current plot's x-range.
_SPACING_TICK_MANTISSAS = (0.1, 0.15, 0.2, 0.3, 0.5, 0.7, 1.0)
_SPACING_TICK_DECADES = (-3, -2, -1, 0)


_METHOD_OPTIONS = {
    "Delta-Threshold": "delta-threshold",
    "Derivative (dE/dEcut)": "derivative",
    "Asymptotic Extrapolation": "asymptotic",
}

_METHOD_DOCS = {
    "delta-threshold": (
        "Delta-Threshold compares total energy between successive cutoff values. "
        "Convergence is reached when the energy difference per atom falls below the selected threshold."
    ),
    "derivative": (
        "Derivative analysis estimates dE/dEcut between successive cutoffs. "
        "Convergence is reached when the absolute derivative per atom is sufficiently small."
    ),
    "asymptotic": (
        "Asymptotic extrapolation fits E(Ecut) = E_inf + A*exp(-B*Ecut) to estimate infinite-cutoff energy. "
        "Convergence is reached when the predicted gap to E_inf per atom falls below threshold. "
        "This method needs at least four completed points."
    ),
}


class ConvergenceExaminerDialog(QDialog):
    """Analyze completed runs and recommend a converged sweep value."""

    default_adopted = Signal(object, str, float, float, object)
    cutoff_adopted = Signal(float, str, float, float, object)
    state_changed = Signal(dict)

    @property
    def _recommended_cutoff(self) -> float | None:  # pragma: no cover - compatibility alias.
        return self._recommended_value

    @_recommended_cutoff.setter
    def _recommended_cutoff(self, value: float | None) -> None:  # pragma: no cover
        self._recommended_value = value

    def __init__(
        self,
        sweep_data: Mapping[float, SweepPointRunState],
        *,
        num_atoms: int,
        sweep_kind: str = "cutoff",
        point_labels: Mapping[float, str] | None = None,
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
        self._point_labels = {float(key): str(value) for key, value in (point_labels or {}).items()}
        self._lattice_vectors: tuple[tuple[float, float, float], ...] | None = (
            tuple(tuple(float(component) for component in vector) for vector in lattice_vectors)
            if lattice_vectors is not None
            else None
        )
        self._point_grids: dict[float, tuple[int, int, int]] = {}
        self._point_spacings: dict[float, float] = {}
        if self._sweep_kind == "kpoint":
            self._point_grids = self._parse_grids_from_labels(self._point_labels)
            if self._lattice_vectors is not None:
                self._point_spacings = self._compute_spacings(
                    self._lattice_vectors, self._point_grids
                )
        self._hover_annotation = None
        # Axis references are populated in _update_plot. Twin axes sit on
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
        self._selected_method_for_recommendation: str | None = None
        self._energy_threshold_for_recommendation: float | None = None
        self._force_threshold_for_recommendation: float | None = None
        self._stress_threshold_for_recommendation: float | None = None
        self._last_plot_data: dict[str, Any] | None = None

        # Thresholds only need to be positive; the smallest representable step
        # stands in for "greater than zero". See _precision_notes for when a
        # threshold is finer than CASTEP prints the quantity.
        self._energy_threshold = DecimalLineEdit(self)
        self._energy_threshold.setRange(_MIN_THRESHOLD, 1.0)
        self._energy_threshold.setDecimals(_THRESHOLD_DECIMALS)
        self._energy_threshold.setSingleStep(0.001)
        self._energy_threshold.setValue(0.001)

        self._force_threshold = DecimalLineEdit(self)
        self._force_threshold.setRange(_MIN_THRESHOLD, 1.0)
        self._force_threshold.setDecimals(_THRESHOLD_DECIMALS)
        self._force_threshold.setSingleStep(0.01)
        self._force_threshold.setValue(0.05)

        self._include_stress = QCheckBox("Stress threshold (GPa)", self)
        self._include_stress.setChecked(False)
        # Match the surrounding QGroupBox frame so the checkbox label does not
        # render with a different (default widget) background colour.
        self._include_stress.setAttribute(Qt.WA_TranslucentBackground, True)
        self._include_stress.setAutoFillBackground(False)
        self._include_stress.setStyleSheet("QCheckBox { background: transparent; }")
        self._stress_threshold = DecimalLineEdit(self)
        self._stress_threshold.setRange(_MIN_THRESHOLD, 100.0)
        self._stress_threshold.setDecimals(_THRESHOLD_DECIMALS)
        self._stress_threshold.setSingleStep(0.05)
        self._stress_threshold.setValue(0.1)
        self._stress_threshold.setEnabled(False)

        self._method_combo = QComboBox(self)
        for label in _METHOD_OPTIONS:
            self._method_combo.addItem(label)

        self._method_info_button = QPushButton("Info", self)
        self._run_button = QPushButton("Analyse Convergence", self)
        self._results = QTextEdit(self)
        self._results.setReadOnly(True)

        palette = current_palette()
        self._figure = Figure(
            figsize=(8.0, 5.5),
            facecolor=palette.bg_surface,
            constrained_layout=True,
        )
        layout_engine = self._figure.get_layout_engine()
        if layout_engine is not None:
            layout_engine.set(w_pad=0.04, h_pad=0.04, hspace=0.08, wspace=0.04)
        self._canvas = FigureCanvasQTAgg(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
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
        self._update_method_availability()
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
        self._stress_threshold.setToolTip("Stress convergence threshold (GPa). Mirrors GEOM_STRESS_TOL.")
        layout.addWidget(criteria_group)

        method_row = QHBoxLayout()
        method_row.setSpacing(12)
        method_row.addWidget(QLabel("Energy method"))
        method_row.addWidget(self._method_combo, stretch=1)
        method_row.addWidget(self._method_info_button)
        method_row.addWidget(self._run_button)
        method_row.addWidget(self._adopt_button)
        layout.addLayout(method_row)

        self._results.setMinimumHeight(220)
        self._results.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.MinimumExpanding)
        layout.addWidget(self._results)
        layout.addWidget(self._canvas)
        layout.addStretch(1)

    def _connect_signals(self) -> None:
        self._method_info_button.clicked.connect(self._show_method_info)
        self._run_button.clicked.connect(self._run_analysis)
        self._adopt_button.clicked.connect(self._emit_adopted_cutoff)
        self._canvas.mpl_connect("motion_notify_event", self._on_plot_hover)
        self._canvas.mpl_connect("figure_leave_event", self._on_plot_leave)
        self._include_stress.toggled.connect(self._on_include_stress_toggled)

    def _on_include_stress_toggled(self, checked: bool) -> None:
        self._stress_threshold.setEnabled(bool(checked))
        self._emit_state_changed()

    def _sweep_point_tooltip(self, x_value: float) -> str:
        if self._sweep_kind == "cutoff":
            return f"Cutoff: {x_value:.8g} eV"
        return self._point_labels.get(float(x_value), f"Grid: {x_value:.8g} k-points")

    def _annotation_box_style(self) -> dict[str, object]:
        palette = current_palette()
        return {
            "boxstyle": "round,pad=0.3",
            "fc": palette.bg_surface,
            "ec": palette.border,
            "alpha": 0.95,
        }

    def _annotation_text_color(self) -> str:
        return current_palette().text_primary

    def _apply_axis_theme(self, axis) -> None:
        palette = current_palette()
        axis.set_facecolor(palette.bg_surface_alt)
        axis.tick_params(axis="both", colors=palette.text_primary, labelcolor=palette.text_primary)
        axis.xaxis.label.set_color(palette.text_primary)
        axis.yaxis.label.set_color(palette.text_primary)
        axis.title.set_color(palette.text_primary)
        for spine in axis.spines.values():
            spine.set_color(palette.border)
        axis.grid(True, color=palette.border, alpha=0.35)

    def _apply_legend_theme(self, axis) -> None:
        legend = axis.get_legend()
        if legend is None:
            return

        palette = current_palette()
        frame = legend.get_frame()
        frame.set_facecolor(palette.bg_surface)
        frame.set_edgecolor(palette.border)
        frame.set_alpha(0.95)
        for label in legend.get_texts():
            label.set_color(palette.text_primary)

    def _clear_figure(self) -> None:
        """Clear the figure and forget the hover annotation that lived on it.

        ``Figure.clear`` detaches every artist, including a cached hover
        annotation, which then can no longer be removed or redrawn; the next
        hover creates a fresh one on the new axes.
        """

        self._figure.clear()
        self._hover_annotation = None

    def _on_plot_leave(self, _event) -> None:
        if self._hover_annotation is not None:
            self._hover_annotation.set_visible(False)
            self._canvas.draw_idle()

    def _on_plot_hover(self, event) -> None:
        if (
            self._last_plot_data is None
            or self._energy_ax is None
            or self._force_ax is None
        ):
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
                    (x_value, y_value, f"{self._sweep_point_tooltip(x_value)}\nEnergy: {y_value:.12g} eV")
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
                for idx, (x_value, stress_y) in enumerate(
                    zip(cutoffs, stresses, strict=False)
                ):
                    text = (
                        f"{self._sweep_point_tooltip(x_value)}"
                        f"\nMax stress: {stress_y:.6g} GPa"
                    )
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
                    for idx, (x_value, y_value) in enumerate(
                        zip(cutoffs, forces, strict=False)
                    ):
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

    def _collect_completed(self, sweep_data: Mapping[float, SweepPointRunState]) -> list[tuple[float, SweepPointRunState]]:
        completed: list[tuple[float, SweepPointRunState]] = []
        for cutoff, state in sweep_data.items():
            if state.status.strip().lower() != "completed":
                continue
            completed.append((float(cutoff), state))
        completed.sort(key=lambda item: item[0])
        return completed

    @staticmethod
    def _parse_grids_from_labels(
        point_labels: Mapping[float, str],
    ) -> dict[float, tuple[int, int, int]]:
        grids: dict[float, tuple[int, int, int]] = {}
        for x_value, label in point_labels.items():
            if not label.startswith("Grid:"):
                continue
            payload = label.split(":", maxsplit=1)[1].strip().replace(" ", "")
            parts = payload.split("x")
            if len(parts) != 3:
                continue
            try:
                grid = (int(parts[0]), int(parts[1]), int(parts[2]))
            except ValueError:
                continue
            if grid[0] <= 0 or grid[1] <= 0 or grid[2] <= 0:
                continue
            grids[float(x_value)] = grid
        return grids

    @staticmethod
    def _compute_spacings(
        lattice_vectors: Sequence[Sequence[float]],
        grids: Mapping[float, tuple[int, int, int]],
    ) -> dict[float, float]:
        spacings: dict[float, float] = {}
        last_spacing: float | None = None
        for x_value in sorted(grids):
            grid = grids[x_value]
            # (1,1,1) is Gamma-only sampling; no single spacing characterises
            # it (every spacing above the cell's coarsest reciprocal transition
            # produces it) so spacing_from_grid raises. Skip explicitly so the
            # annotation machinery never exposes it to the UI.
            if grid == (1, 1, 1):
                continue
            try:
                spacing = spacing_from_grid(lattice_vectors, grid)
            except ValueError:
                continue
            # Skip duplicate spacings: when one axis dominates the round-trip
            # bound, successive grids can report the same value even as other
            # axes grow. Once a plateau starts, every subsequent repeat is
            # suppressed so the annotation series stays strictly decreasing
            # and does not mislead the user about convergence status.
            if last_spacing is not None and math.isclose(
                spacing, last_spacing, rel_tol=1e-9
            ):
                continue
            spacings[float(x_value)] = spacing
            last_spacing = spacing
        return spacings

    @staticmethod
    def recompute_spacings_in_snapshot(
        snapshot: Mapping[str, object],
        lattice_vectors: Sequence[Sequence[float]],
    ) -> dict[str, object]:
        """Return a copy of *snapshot* with ``point_spacings`` recomputed.

        Re-derives spacing values from *lattice_vectors* and the grid labels
        stored in the snapshot, so that stale values persisted with an older
        formula are replaced without mutating the caller's dict.

        Returns the original dict (as a shallow copy) unchanged when the
        snapshot lacks ``point_labels`` or ``plot_data``, or if recomputation
        raises an unexpected error.
        """
        raw_labels = snapshot.get("point_labels")
        plot_data = snapshot.get("plot_data")
        if not isinstance(raw_labels, Mapping) or not isinstance(plot_data, Mapping):
            return dict(snapshot)

        float_labels: dict[float, str] = {}
        for k, v in raw_labels.items():
            try:
                float_labels[float(k)] = str(v)
            except (TypeError, ValueError):
                continue

        grids = ConvergenceExaminerDialog._parse_grids_from_labels(float_labels)
        try:
            fresh_spacings = ConvergenceExaminerDialog._compute_spacings(
                lattice_vectors, grids
            )
        except Exception:
            return dict(snapshot)

        return {
            **dict(snapshot),
            "plot_data": {**dict(plot_data), "point_spacings": fresh_spacings},
        }

    @staticmethod
    def _format_spacing_label(spacing: float) -> str:
        # Trim trailing zeros from a fixed-precision format so 0.1, 0.05 etc.
        # render naturally without leaking float imprecision (0.0500000…).
        text = f"{spacing:.4f}".rstrip("0").rstrip(".")
        return text or "0"

    @classmethod
    def _candidate_spacings(cls) -> list[float]:
        seen: set[float] = set()
        candidates: list[float] = []
        for decade in _SPACING_TICK_DECADES:
            scale = 10.0 ** decade
            for mantissa in _SPACING_TICK_MANTISSAS:
                value = round(mantissa * scale, 12)
                if value <= 0.0 or value in seen:
                    continue
                seen.add(value)
                candidates.append(value)
        candidates.sort()
        return candidates

    def _regular_spacing_ticks(
        self,
        x_min: float,
        x_max: float,
    ) -> tuple[list[float], list[str]]:
        """Build the spacing-axis (position, label) pairs for the current xlim.

        Positions come from :func:`kpoint_count_for_spacing` so each labelled
        spacing lands at the x-coordinate where CASTEP would place the
        corresponding grid. Larger spacings -> smaller n_kpoints, so on the
        log x-axis they fall to the left, mirroring the cost-ordered reading.
        """
        if self._lattice_vectors is None or x_max <= x_min:
            return [], []

        positions: list[float] = []
        labels: list[str] = []
        last_label: str | None = None
        for spacing in self._candidate_spacings():
            try:
                position = kpoint_count_for_spacing(
                    self._lattice_vectors, spacing
                )
            except ValueError:
                continue
            if not (x_min <= position <= x_max):
                continue
            label = self._format_spacing_label(spacing)
            if label == last_label:
                # Avoid emitting two ticks that would render with identical
                # text (can happen at extreme decades where the format
                # collapses two values onto the same string).
                continue
            positions.append(position)
            labels.append(label)
            last_label = label

        # _candidate_spacings is ascending in spacing -> n_kpoints is
        # descending, so reverse to get left-to-right (small n -> large n)
        # ordering that matches the underlying axis.
        positions.reverse()
        labels.reverse()

        # A second consecutive-duplicate pass is needed after the reversal:
        # the range filter above can remove spacings that previously separated
        # two identical-label candidates, making them adjacent only after
        # the list is flipped.
        deduped_positions: list[float] = []
        deduped_labels: list[str] = []
        last_after_reverse: str | None = None
        for pos, lbl in zip(positions, labels, strict=True):
            if lbl == last_after_reverse:
                continue
            deduped_positions.append(pos)
            deduped_labels.append(lbl)
            last_after_reverse = lbl
        return deduped_positions, deduped_labels

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

    def _update_method_availability(self) -> None:
        cutoffs, energies = self._energy_series()
        count = len(cutoffs)
        derivative_index = self._method_combo.findText("Derivative (dE/dEcut)")
        asymptotic_index = self._method_combo.findText("Asymptotic Extrapolation")
        if self._sweep_kind == "kpoint":
            if derivative_index >= 0:
                self._method_combo.model().item(derivative_index).setEnabled(False)
                self._method_combo.setItemData(
                    derivative_index,
                    "K-point convergence only uses Delta-Threshold.",
                    Qt.ItemDataRole.ToolTipRole,
                )
            if asymptotic_index >= 0:
                self._method_combo.model().item(asymptotic_index).setEnabled(False)
                self._method_combo.setItemData(
                    asymptotic_index,
                    "K-point convergence only uses Delta-Threshold.",
                    Qt.ItemDataRole.ToolTipRole,
                )
            if self._selected_method() != "delta-threshold":
                self._method_combo.setCurrentIndex(0)
        elif asymptotic_index >= 0:
            enabled = count >= 4
            self._method_combo.model().item(asymptotic_index).setEnabled(enabled)
            self._method_combo.setItemData(
                asymptotic_index,
                "Requires at least 4 completed energy points." if not enabled else "",
                Qt.ItemDataRole.ToolTipRole,
            )
            if not enabled and self._method_combo.currentIndex() == asymptotic_index:
                self._method_combo.setCurrentIndex(0)

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
            self._adopt_button.setEnabled(False)
            self._adopt_button.setText(
                "Set as Project Default Cutoff"
                if self._sweep_kind == "cutoff"
                else "Set as Project Default K-Point Grid"
            )

    def _write_initial_summary(self) -> None:
        energy_cutoffs, _energies = self._energy_series()
        force_cutoffs, _forces = self._force_series()
        stress_cutoffs, _stresses = self._stress_series()

        lines = [
            f"Completed points: {len(self._completed)}",
            f"Energy datapoints available: {len(energy_cutoffs)}",
            f"Force datapoints available: {len(force_cutoffs)}",
            f"Stress datapoints available: {len(stress_cutoffs)}",
            f"Simulation-cell atoms used for per-atom normalization: {self._num_atoms}",
        ]

        if len(energy_cutoffs) < 2 or len(force_cutoffs) < 2:
            lines.append("Need at least two completed rows with both energy and max-force values to run analysis.")
        elif self._num_atoms <= 0:
            lines.append("Simulation-cell atom count must be positive before analysis can run.")
        lines.extend(self._note_lines())

        self._results.setPlainText("\n".join(lines))

    def _draw_initial_plot(self) -> None:
        energy_cutoffs, energies = self._energy_series()
        force_cutoffs, forces = self._force_series()

        palette = current_palette()
        self._figure.set_facecolor(palette.bg_surface)
        self._clear_figure()
        energy_ax = self._figure.add_subplot(2, 1, 1)
        force_ax = self._figure.add_subplot(2, 1, 2, sharex=energy_ax)
        self._energy_ax = energy_ax
        self._force_ax = force_ax
        self._spacing_ax = None
        self._stress_ax = None

        if energy_cutoffs:
            energy_ax.plot(energy_cutoffs, energies, marker="o", linestyle="-", color="#1d4ed8", label="Total energy")
        energy_ax.set_ylabel("Energy (eV)")
        energy_ax.tick_params(axis="x", labelbottom=False)
        energy_ax.legend(loc="best") if energy_cutoffs else None
        self._apply_axis_theme(energy_ax)
        self._apply_legend_theme(energy_ax)

        if force_cutoffs:
            force_ax.plot(force_cutoffs, forces, marker="o", linestyle="-", color="#b91c1c", label="Max force")
        force_ax.set_xlabel(
            "Cutoff energy (eV)" if self._sweep_kind == "cutoff" else "Number of k-points",
            labelpad=8,
        )
        force_ax.set_ylabel("Max force (eV/Ang)")
        force_ax.legend(loc="best") if force_cutoffs else None
        self._apply_axis_theme(force_ax)
        self._apply_legend_theme(force_ax)

        self._canvas.draw_idle()

    def _selected_method(self) -> str:
        label = self._method_combo.currentText()
        return _METHOD_OPTIONS.get(label, "delta-threshold")

    def _set_method(self, method: str) -> None:
        normalized = method.strip().lower()
        for label, key in _METHOD_OPTIONS.items():
            if key != normalized:
                continue
            index = self._method_combo.findText(label)
            if index >= 0:
                self._method_combo.setCurrentIndex(index)
            return

    @staticmethod
    def _serialize_result(result: ConvergenceResult | None) -> dict[str, object] | None:
        if result is None:
            return None
        return {
            "converged": bool(result.converged),
            "method": result.method,
            "details": result.details,
            "converged_value": result.converged_value,
            "converged_cutoff": result.converged_value,
            "deltas": list(result.deltas) if result.deltas is not None else None,
            "fitted_curve": list(result.fitted_curve) if result.fitted_curve is not None else None,
            "extrapolated_energy": result.extrapolated_energy,
            "fallback_suggestion": result.fallback_suggestion,
        }

    @staticmethod
    def _deserialize_result(payload: object) -> ConvergenceResult | None:
        if not isinstance(payload, dict):
            return None

        try:
            converged = bool(payload.get("converged", False))
            method = str(payload.get("method", ""))
            details = str(payload.get("details", ""))
        except (TypeError, ValueError):
            return None

        converged_value_raw = payload.get("converged_value", payload.get("converged_cutoff"))
        converged_value: float | None = None
        if converged_value_raw is not None:
            try:
                converged_value = float(converged_value_raw)
            except (TypeError, ValueError):
                converged_value = None

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

        extrapolated_raw = payload.get("extrapolated_energy")
        extrapolated_energy: float | None = None
        if extrapolated_raw is not None:
            try:
                extrapolated_energy = float(extrapolated_raw)
            except (TypeError, ValueError):
                extrapolated_energy = None

        fallback_raw = payload.get("fallback_suggestion")
        fallback_suggestion = str(fallback_raw) if isinstance(fallback_raw, str) else None

        return ConvergenceResult(
            converged=converged,
            method=method,
            details=details,
            converged_value=converged_value,
            deltas=parse_series(payload.get("deltas")),
            fitted_curve=parse_series(payload.get("fitted_curve")),
            extrapolated_energy=extrapolated_energy,
            fallback_suggestion=fallback_suggestion,
        )

    def state_snapshot(self) -> dict[str, object]:
        """Return a persistable snapshot of controls, output text, and plot inputs."""
        return {
            "energy_threshold": float(self._energy_threshold.value()),
            "force_threshold": float(self._force_threshold.value()),
            "stress_threshold": float(self._stress_threshold.value()),
            "include_stress": bool(self._include_stress.isChecked()),
            "method": self._selected_method(),
            "sweep_kind": self._sweep_kind,
            "results_text": self._results.toPlainText(),
            "recommended_value": self._recommended_value,
            "recommended_cutoff": self._recommended_value,
            "selected_method_for_recommendation": self._selected_method_for_recommendation,
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

        method_raw = state.get("method")
        if isinstance(method_raw, str) and method_raw.strip():
            self._set_method(method_raw)

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

        selected_method_raw = state.get("selected_method_for_recommendation")
        self._selected_method_for_recommendation = (
            str(selected_method_raw).strip()
            if isinstance(selected_method_raw, str) and selected_method_raw.strip()
            else None
        )

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

        text_raw = state.get("results_text")
        if isinstance(text_raw, str) and text_raw.strip():
            self._results.setPlainText(text_raw)

        plot_data = state.get("plot_data")
        if isinstance(plot_data, Mapping):
            try:
                cutoffs = [float(value) for value in plot_data.get("cutoffs", [])]
                energies = [float(value) for value in plot_data.get("energies", [])]
                forces = [float(value) for value in plot_data.get("forces", [])]
                stresses_raw = plot_data.get("stresses", []) or []
                stresses = [float(value) for value in stresses_raw]
                energy_threshold = float(plot_data.get("energy_threshold", self._energy_threshold.value()))
                force_threshold = float(plot_data.get("force_threshold", self._force_threshold.value()))
                stress_threshold_raw = plot_data.get("stress_threshold")
                stress_threshold = (
                    float(stress_threshold_raw) if stress_threshold_raw is not None else None
                )
            except (TypeError, ValueError):
                cutoffs = []
                energies = []
                forces = []
                stresses = []
                stress_threshold = None

            if cutoffs and len(cutoffs) == len(energies) == len(forces):
                use_stress = bool(stresses) and len(stresses) == len(cutoffs) and stress_threshold is not None
                self._last_plot_data = {
                    "cutoffs": list(cutoffs),
                    "energies": list(energies),
                    "forces": list(forces),
                    "stresses": list(stresses) if use_stress else [],
                    "energy_threshold": energy_threshold,
                    "force_threshold": force_threshold,
                    "stress_threshold": stress_threshold if use_stress else None,
                    "point_spacings": dict(self._point_spacings) if self._point_spacings else {},
                }
                self._update_plot(
                    cutoffs,
                    energies,
                    forces,
                    energy_threshold,
                    force_threshold,
                    stresses=stresses if use_stress else None,
                    stress_threshold=stress_threshold if use_stress else None,
                )

        if self._recommended_value is not None:
            self._adopt_button.setEnabled(True)
            self._adopt_button.setText(self._adopt_button_text())
            self._emit_state_changed()

    def _note_lines(self, extra: Sequence[str] = ()) -> list[str]:
        notes = [*self._notes, *extra]
        if not notes:
            return []
        return ["", "Notes:", *(f"- {note}" for note in notes)]

    def _recommended_point_text(self) -> str:
        """Return the recommended k-point grid as ``4 x 4 x 4`` (or a count fallback)."""

        value = float(self._recommended_value)  # type: ignore[arg-type]
        grid = self._point_grids.get(value)
        if grid is not None:
            return format_kpoint_grid(grid)
        return self._point_labels.get(value, f"{value:.8g} k-points")

    def _adopt_button_text(self) -> str:
        if self._sweep_kind == "cutoff":
            return f"Set {self._recommended_value:.8g} eV as Project Default Cutoff"
        return f"Set {self._recommended_point_text()} as Project Default K-Point Grid"

    def _show_method_info(self) -> None:
        method = self._selected_method()
        text = _METHOD_DOCS.get(method, "No help text available for this method.")
        QMessageBox.information(self, "Energy method info", text)

    def _run_analysis(self) -> None:
        # Show the thresholds actually used: a typed value outside a field's range
        # is clamped when read, and clicking the button does not always end the
        # edit (macOS buttons take no focus), so snap the text here as well.
        for field in (self._energy_threshold, self._force_threshold, self._stress_threshold):
            field.setValue(field.value())
        self._update_method_availability()
        if not self._run_button.isEnabled():
            self._write_initial_summary()
            return

        cutoffs, energies = self._energy_series()
        force_cutoffs, forces = self._force_series()
        if cutoffs != force_cutoffs:
            energy_by_cutoff = dict(zip(cutoffs, energies, strict=False))
            force_by_cutoff = dict(zip(force_cutoffs, forces, strict=False))
            aligned_cutoffs = sorted(set(energy_by_cutoff) & set(force_by_cutoff))
            cutoffs = aligned_cutoffs
            energies = [energy_by_cutoff[cutoff] for cutoff in aligned_cutoffs]
            forces = [force_by_cutoff[cutoff] for cutoff in aligned_cutoffs]

        if len(cutoffs) < 2:
            self._results.setPlainText(
                "Need at least two matching completed rows containing both total energy and max force values."
            )
            self._adopt_button.setEnabled(False)
            self._adopt_button.setText(
                "Set as Project Default Cutoff"
                if self._sweep_kind == "cutoff"
                else "Set as Project Default K-Point Grid"
            )
            self._draw_initial_plot()
            return

        method = self._selected_method()
        energy_threshold = float(self._energy_threshold.value())
        force_threshold = float(self._force_threshold.value())

        include_stress = self._include_stress.isEnabled() and self._include_stress.isChecked()
        stresses: list[float] = []
        stress_threshold: float | None = None
        if include_stress:
            stress_by_cutoff = {cutoff: stress for cutoff, stress in zip(*self._stress_series(), strict=False)}
            try:
                stresses = [stress_by_cutoff[cutoff] for cutoff in cutoffs]
            except KeyError:
                # A completed row used for energy/force does not have stress; fall back to no-stress run.
                include_stress = False
                stresses = []
            else:
                stress_threshold = float(self._stress_threshold.value())

        try:
            if self._sweep_kind == "kpoint":
                energy_result, force_result = analyze_kpoint_convergence(
                    kpoint_counts=cutoffs,
                    energies=energies,
                    max_forces=forces,
                    num_atoms=self._num_atoms,
                    energy_threshold=energy_threshold,
                    force_threshold=force_threshold,
                    point_labels=self._point_labels,
                    point_spacings=self._point_spacings or None,
                )
            else:
                energy_result, force_result = analyze_convergence(
                    method=method,
                    cutoffs=cutoffs,
                    energies=energies,
                    max_forces=forces,
                    num_atoms=self._num_atoms,
                    energy_threshold=energy_threshold,
                    force_threshold=force_threshold,
                )
        except ValueError as exc:
            QMessageBox.warning(self, "Convergence analysis failed", str(exc))
            return

        stress_result: ConvergenceResult | None = None
        if include_stress and stress_threshold is not None:
            stress_result = delta_threshold_stress(cutoffs, stresses, threshold=stress_threshold)
            if self._sweep_kind == "kpoint":
                stress_result = apply_kpoint_label_to_stress_result(
                    stress_result,
                    self._point_labels,
                    point_spacings=self._point_spacings or None,
                )

        self._energy_result = energy_result
        self._force_result = force_result
        self._stress_result = stress_result
        self._recommended_value = recommended_value(energy_result, force_result, stress_result)
        self._selected_method_for_recommendation = method
        self._energy_threshold_for_recommendation = energy_threshold
        self._force_threshold_for_recommendation = force_threshold
        self._stress_threshold_for_recommendation = stress_threshold if include_stress else None

        lines = [
            f"Energy method: {method}",
            f"Energy threshold: {energy_threshold:.6g} eV/atom",
            f"Force threshold: {force_threshold:.6g} eV/Ang",
        ]
        if include_stress and stress_threshold is not None:
            lines.append(f"Stress threshold: {stress_threshold:.6g} GPa")
        lines.extend(
            [
                "",
                "Energy result:",
                energy_result.details,
                "",
                "Force result:",
                force_result.details,
            ]
        )
        if stress_result is not None:
            lines.extend(["", "Stress result:", stress_result.details])

        if energy_result.fallback_suggestion:
            lines.extend(["", f"Energy method fallback suggestion: {energy_result.fallback_suggestion}"])

        if self._recommended_value is not None:
            if self._sweep_kind == "cutoff":
                lines.extend(["", f"Recommended cutoff: {self._recommended_value:.8g} eV"])
            else:
                label = self._recommended_point_text()
                recommended_line = f"Recommended k-point grid: {label}"
                grid = self._point_grids.get(float(self._recommended_value))
                spacing = self._point_spacings.get(float(self._recommended_value))
                if spacing is not None and grid is not None and grid != (1, 1, 1):
                    recommended_line += f" (spacing {spacing:.4g} 1/Ang)"
                lines.extend(["", recommended_line])
        else:
            unconverged = unconverged_criteria(energy_result, force_result, stress_result)
            if unconverged:
                setting = "cutoffs" if self._sweep_kind == "cutoff" else "denser k-point grids"
                lines.extend(
                    [
                        "",
                        "No recommendation: the "
                        + " and ".join(unconverged)
                        + (" criterion" if len(unconverged) == 1 else " criteria")
                        + " did not converge within the sweep. Extend the sweep to larger "
                        + f"{setting}, or relax the threshold, and analyse again.",
                    ]
                )
            else:
                lines.extend(["", "No combined convergence recommendation available yet."])
        lines.extend(
            self._note_lines(
                _precision_notes(force_threshold, stress_threshold if include_stress else None)
            )
        )

        self._results.setPlainText("\n".join(lines))
        self._last_plot_data = {
            "cutoffs": list(cutoffs),
            "energies": list(energies),
            "forces": list(forces),
            "stresses": list(stresses) if include_stress else [],
            "energy_threshold": energy_threshold,
            "force_threshold": force_threshold,
            "stress_threshold": stress_threshold,
            "point_spacings": dict(self._point_spacings) if self._point_spacings else {},
        }
        self._update_plot(
            cutoffs,
            energies,
            forces,
            energy_threshold,
            force_threshold,
            stresses=stresses if include_stress else None,
            stress_threshold=stress_threshold if include_stress else None,
        )

        if self._recommended_value is None:
            self._adopt_button.setEnabled(False)
            self._adopt_button.setText(
                "Set as Project Default Cutoff"
                if self._sweep_kind == "cutoff"
                else "Set as Project Default K-Point Grid"
            )
        else:
            self._adopt_button.setEnabled(True)
            self._adopt_button.setText(self._adopt_button_text())

        self._emit_state_changed()

    def _update_plot(
        self,
        cutoffs: list[float],
        energies: list[float],
        forces: list[float],
        energy_threshold: float,
        force_threshold: float,
        *,
        stresses: list[float] | None = None,
        stress_threshold: float | None = None,
    ) -> None:
        palette = current_palette()
        recommended_line = palette.text_muted
        threshold_line = palette.warning

        self._figure.set_facecolor(palette.bg_surface)
        # matplotlib's Axes.clear() resets the xlim to (0, 1), which is
        # invalid on a log axis. When we re-render after a previous kpoint
        # plot, the existing axes (including the twiny spacing axis) are
        # log-scaled, so the upcoming figure.clear() would emit a
        # "non-positive xlim on log axis" warning. Reset every axis back to
        # linear first so the internal teardown reset is well-defined.
        for axis in list(self._figure.axes):
            try:
                if axis.get_xscale() != "linear":
                    axis.set_xscale("linear")
            except Exception:
                pass
        self._clear_figure()
        energy_ax = self._figure.add_subplot(2, 1, 1)
        force_ax = self._figure.add_subplot(2, 1, 2, sharex=energy_ax)
        self._energy_ax = energy_ax
        self._force_ax = force_ax
        self._spacing_ax = None
        self._stress_ax = None

        energy_ax.plot(cutoffs, energies, marker="o", linestyle="-", color="#1d4ed8", label="Total energy")
        if self._energy_result and self._energy_result.fitted_curve:
            fit_x = [point[0] for point in self._energy_result.fitted_curve]
            fit_y = [point[1] for point in self._energy_result.fitted_curve]
            energy_ax.plot(fit_x, fit_y, linestyle="--", color="#0f766e", label="Asymptotic fit")
        if self._energy_result and self._energy_result.extrapolated_energy is not None:
            energy_ax.axhline(
                self._energy_result.extrapolated_energy,
                color="#0f766e",
                linestyle=":",
                label="E_inf",
            )
        if self._energy_result and self._energy_result.converged_value is not None:
            energy_ax.axvline(
                self._energy_result.converged_value,
                color="#059669",
                linestyle="--",
                label="Energy convergence",
            )
        if self._recommended_value is not None:
            energy_ax.axvline(self._recommended_value, color=recommended_line, linestyle="-", alpha=0.55)

        energy_ax.set_ylabel("Energy (eV)")
        energy_ax.tick_params(axis="x", labelbottom=False)
        energy_ax.legend(loc="best")
        self._apply_axis_theme(energy_ax)
        self._apply_legend_theme(energy_ax)

        if self._sweep_kind == "kpoint":
            # n_kpoints grows as ~n^3 across a typical grid sweep, so small
            # grids collapse into the leftmost sliver of a linear axis. Log
            # scale spreads the cheap-grid region out so the spacing labels
            # (and the data points themselves) stop overlapping.
            energy_ax.set_xscale("log")
            # set_xscale only *requests* a view-limit rescale; without
            # autoscale_view the existing (linear) xlim can include a
            # non-positive lower margin, which then triggers a matplotlib
            # UserWarning when handed to set_xlim on the log-scaled twin.
            energy_ax.relim()
            energy_ax.autoscale_view()
            if self._lattice_vectors is not None:
                x_min, x_max = energy_ax.get_xlim()
                positions, labels = self._regular_spacing_ticks(x_min, x_max)
                if positions:
                    spacing_ax = energy_ax.twiny()
                    self._spacing_ax = spacing_ax
                    spacing_ax.set_xscale("log")
                    spacing_ax.set_xlim(x_min, x_max)
                    spacing_ax.set_xticks(positions)
                    spacing_ax.set_xticklabels(labels)
                    spacing_ax.minorticks_off()
                    spacing_ax.set_xlabel("K-point spacing (1/Ang)", labelpad=8)
                    self._apply_axis_theme(spacing_ax)
                    # twiny axes inherit the energy axis's grid; suppress to avoid
                    # double-drawn vertical gridlines from the spacing tick positions.
                    spacing_ax.grid(False)

        force_line, = force_ax.plot(
            cutoffs, forces, marker="o", linestyle="-", color="#b91c1c", label="Max force"
        )
        force_handles = [force_line]
        if self._force_result and self._force_result.converged_value is not None:
            line = force_ax.axvline(
                self._force_result.converged_value,
                color="#059669",
                linestyle="--",
                label="Force convergence",
            )
            force_handles.append(line)
        threshold_handle = force_ax.axhline(
            force_threshold, color=threshold_line, linestyle=":", label="Force threshold"
        )
        force_handles.append(threshold_handle)
        if self._recommended_value is not None:
            force_ax.axvline(self._recommended_value, color=recommended_line, linestyle="-", alpha=0.55)

        force_ax.set_xlabel(
            "Cutoff energy (eV)" if self._sweep_kind == "cutoff" else "Number of k-points",
            labelpad=8,
        )
        force_ax.set_ylabel("Max force (eV/Ang)")
        force_ax.yaxis.label.set_color("#b91c1c")
        force_ax.tick_params(axis="y", colors="#b91c1c", labelcolor="#b91c1c")
        self._apply_axis_theme(force_ax)
        # The shared theme repaints the y-axis label; re-apply force color afterwards.
        force_ax.yaxis.label.set_color("#b91c1c")
        force_ax.tick_params(axis="y", colors="#b91c1c", labelcolor="#b91c1c")

        stress_handles: list[Any] = []
        if stresses:
            stress_ax = force_ax.twinx()
            self._stress_ax = stress_ax
            stress_line, = stress_ax.plot(
                cutoffs,
                stresses,
                marker="s",
                linestyle="-",
                color=_STRESS_COLOR,
                label="Max stress",
            )
            stress_handles.append(stress_line)
            if self._stress_result and self._stress_result.converged_value is not None:
                stress_handles.append(
                    stress_ax.axvline(
                        self._stress_result.converged_value,
                        color="#15803d",
                        linestyle="--",
                        alpha=0.85,
                        label="Stress convergence",
                    )
                )
            if stress_threshold is not None:
                stress_handles.append(
                    stress_ax.axhline(
                        stress_threshold,
                        color=_STRESS_COLOR,
                        linestyle=":",
                        alpha=0.85,
                        label="Stress threshold",
                    )
                )
            stress_ax.set_ylabel("Max stress (GPa)")
            stress_ax.yaxis.label.set_color(_STRESS_COLOR)
            stress_ax.tick_params(axis="y", colors=_STRESS_COLOR, labelcolor=_STRESS_COLOR)
            for spine in stress_ax.spines.values():
                spine.set_color(current_palette().border)
            stress_ax.spines["right"].set_color(_STRESS_COLOR)

        handles = force_handles + stress_handles
        labels = [handle.get_label() for handle in handles]
        force_ax.legend(handles, labels, loc="best")
        self._apply_legend_theme(force_ax)

        # Mention the active thresholds in the axes titles for quick scanning.
        energy_ax.set_title(f"Energy convergence (threshold {energy_threshold:.3g} eV/atom)", pad=10)
        if stresses and stress_threshold is not None:
            force_ax.set_title(
                "Force / stress convergence "
                f"(dF ≤ {force_threshold:.3g} eV/Ang, dS ≤ {stress_threshold:.3g} GPa)",
                pad=10,
            )
        else:
            force_ax.set_title(
                f"Force convergence (delta threshold {force_threshold:.3g} eV/Ang)", pad=10
            )

        self._canvas.draw_idle()

    def _emit_adopted_cutoff(self) -> None:
        if self._recommended_value is None:
            return
        method = self._selected_method_for_recommendation or self._selected_method()
        energy_threshold = self._energy_threshold_for_recommendation
        if energy_threshold is None:
            energy_threshold = float(self._energy_threshold.value())
        force_threshold = self._force_threshold_for_recommendation
        if force_threshold is None:
            force_threshold = float(self._force_threshold.value())
        stress_threshold = self._stress_threshold_for_recommendation
        self._emit_state_changed()
        self.default_adopted.emit(
            float(self._recommended_value), method, energy_threshold, force_threshold, stress_threshold
        )
        if self._sweep_kind == "cutoff":
            self.cutoff_adopted.emit(
                float(self._recommended_value), method, energy_threshold, force_threshold, stress_threshold
            )
