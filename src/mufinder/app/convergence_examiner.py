"""Dialog for analyzing completed convergence results."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
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
from mufinder.app.theme import current_palette
from mufinder.science.convergence_analysis import (
    ConvergenceResult,
    analyze_convergence,
    analyze_kpoint_convergence,
    recommended_value,
)


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

    default_adopted = Signal(object, str, float, float)
    cutoff_adopted = Signal(float, str, float, float)
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
        initial_state: Mapping[str, object] | None = None,
        parent: QWidget | None = None,
    ) -> None:
        super().__init__(parent)
        if sweep_kind not in {"cutoff", "kpoint"}:
            raise ValueError("sweep_kind must be 'cutoff' or 'kpoint'.")

        self._sweep_kind = sweep_kind
        self._point_labels = {float(key): str(value) for key, value in (point_labels or {}).items()}
        self._hover_annotation = None

        self.setWindowTitle(
            "Examine Cutoff Convergence"
            if self._sweep_kind == "cutoff"
            else "Examine K-Point Convergence"
        )
        self.resize(1080, 900)
        self.setMinimumSize(920, 760)

        self._num_atoms = int(num_atoms)
        self._completed = self._collect_completed(sweep_data)
        self._energy_result: ConvergenceResult | None = None
        self._force_result: ConvergenceResult | None = None
        self._recommended_value: float | None = None
        self._selected_method_for_recommendation: str | None = None
        self._energy_threshold_for_recommendation: float | None = None
        self._force_threshold_for_recommendation: float | None = None
        self._last_plot_data: dict[str, Any] | None = None

        self._energy_threshold = DecimalLineEdit(self)
        self._energy_threshold.setRange(0.0001, 1.0)
        self._energy_threshold.setDecimals(6)
        self._energy_threshold.setSingleStep(0.001)
        self._energy_threshold.setValue(0.001)

        self._force_threshold = DecimalLineEdit(self)
        self._force_threshold.setRange(0.001, 1.0)
        self._force_threshold.setDecimals(6)
        self._force_threshold.setSingleStep(0.01)
        self._force_threshold.setValue(0.05)

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

    def _on_plot_leave(self, _event) -> None:
        if self._hover_annotation is not None:
            self._hover_annotation.set_visible(False)
            self._canvas.draw_idle()

    def _on_plot_hover(self, event) -> None:
        if self._last_plot_data is None:
            return
        if event.inaxes is None or event.xdata is None or event.ydata is None:
            self._on_plot_leave(event)
            return

        cutoffs = list(self._last_plot_data.get("cutoffs", []))
        energies = list(self._last_plot_data.get("energies", []))
        forces = list(self._last_plot_data.get("forces", []))
        if not cutoffs:
            return

        x_range = max(cutoffs) - min(cutoffs) if len(cutoffs) > 1 else max(abs(cutoffs[0]), 1.0)
        x_tol = max(x_range * 0.03, 0.5)

        points: list[tuple[float, float, str]] = []
        if event.inaxes.get_ylabel() == "Energy (eV)":
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
        else:
            if self._sweep_kind == "kpoint":
                points = [
                    (x_value, y_value, self._sweep_point_tooltip(x_value))
                    for x_value, y_value in zip(cutoffs, forces, strict=False)
                ]
            else:
                points = [
                    (x_value, y_value, f"{self._sweep_point_tooltip(x_value)}\nMax force: {y_value:.12g} eV/Ang")
                    for x_value, y_value in zip(cutoffs, forces, strict=False)
                ]

        nearest = min(points, key=lambda point: abs(point[0] - float(event.xdata)))
        if abs(nearest[0] - float(event.xdata)) > x_tol:
            self._on_plot_leave(event)
            return

        if self._hover_annotation is None:
            self._hover_annotation = event.inaxes.annotate(
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
            if self._hover_annotation.axes is not event.inaxes:
                self._hover_annotation.remove()
                self._hover_annotation = event.inaxes.annotate(
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

        lines = [
            f"Completed points: {len(self._completed)}",
            f"Energy datapoints available: {len(energy_cutoffs)}",
            f"Force datapoints available: {len(force_cutoffs)}",
            f"Host atoms used for per-atom normalization: {self._num_atoms}",
        ]

        if len(energy_cutoffs) < 2 or len(force_cutoffs) < 2:
            lines.append("Need at least two completed rows with both energy and max-force values to run analysis.")
        elif self._num_atoms <= 0:
            lines.append("Host structure atom count must be positive before analysis can run.")

        self._results.setPlainText("\n".join(lines))

    def _draw_initial_plot(self) -> None:
        energy_cutoffs, energies = self._energy_series()
        force_cutoffs, forces = self._force_series()

        palette = current_palette()
        self._figure.set_facecolor(palette.bg_surface)
        self._figure.clear()
        energy_ax = self._figure.add_subplot(2, 1, 1)
        force_ax = self._figure.add_subplot(2, 1, 2, sharex=energy_ax)

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
            "method": self._selected_method(),
            "sweep_kind": self._sweep_kind,
            "results_text": self._results.toPlainText(),
            "recommended_value": self._recommended_value,
            "recommended_cutoff": self._recommended_value,
            "selected_method_for_recommendation": self._selected_method_for_recommendation,
            "energy_threshold_for_recommendation": self._energy_threshold_for_recommendation,
            "force_threshold_for_recommendation": self._force_threshold_for_recommendation,
            "energy_result": self._serialize_result(self._energy_result),
            "force_result": self._serialize_result(self._force_result),
            "plot_data": dict(self._last_plot_data or {}),
        }

    def _emit_state_changed(self) -> None:
        self.state_changed.emit(self.state_snapshot())

    def _restore_initial_state(self, state: Mapping[str, object] | None) -> None:
        if not isinstance(state, Mapping):
            return

        energy_threshold_raw = state.get("energy_threshold")
        force_threshold_raw = state.get("force_threshold")
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

        method_raw = state.get("method")
        if isinstance(method_raw, str) and method_raw.strip():
            self._set_method(method_raw)

        self._energy_result = self._deserialize_result(state.get("energy_result"))
        self._force_result = self._deserialize_result(state.get("force_result"))

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

        text_raw = state.get("results_text")
        if isinstance(text_raw, str) and text_raw.strip():
            self._results.setPlainText(text_raw)

        plot_data = state.get("plot_data")
        if isinstance(plot_data, Mapping):
            try:
                cutoffs = [float(value) for value in plot_data.get("cutoffs", [])]
                energies = [float(value) for value in plot_data.get("energies", [])]
                forces = [float(value) for value in plot_data.get("forces", [])]
                energy_threshold = float(plot_data.get("energy_threshold", self._energy_threshold.value()))
                force_threshold = float(plot_data.get("force_threshold", self._force_threshold.value()))
            except (TypeError, ValueError):
                cutoffs = []
                energies = []
                forces = []

            if cutoffs and len(cutoffs) == len(energies) == len(forces):
                self._last_plot_data = {
                    "cutoffs": list(cutoffs),
                    "energies": list(energies),
                    "forces": list(forces),
                    "energy_threshold": energy_threshold,
                    "force_threshold": force_threshold,
                }
                self._update_plot(cutoffs, energies, forces, energy_threshold, force_threshold)

        if self._recommended_value is not None:
            self._adopt_button.setEnabled(True)
            if self._sweep_kind == "cutoff":
                self._adopt_button.setText(
                    f"Set {self._recommended_value:.8g} eV as Project Default Cutoff"
                )
            else:
                label = self._point_labels.get(
                    float(self._recommended_value),
                    f"{self._recommended_value:.8g} k-points",
                )
                self._adopt_button.setText(f"Set {label} as Project Default K-Point Grid")
            self._emit_state_changed()

    def _show_method_info(self) -> None:
        method = self._selected_method()
        text = _METHOD_DOCS.get(method, "No help text available for this method.")
        QMessageBox.information(self, "Energy method info", text)

    def _run_analysis(self) -> None:
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

        self._energy_result = energy_result
        self._force_result = force_result
        self._recommended_value = recommended_value(energy_result, force_result)
        self._selected_method_for_recommendation = method
        self._energy_threshold_for_recommendation = energy_threshold
        self._force_threshold_for_recommendation = force_threshold

        lines = [
            f"Energy method: {method}",
            f"Energy threshold: {energy_threshold:.6g} eV/atom",
            f"Force threshold: {force_threshold:.6g} eV/Ang",
            "",
            "Energy result:",
            energy_result.details,
            "",
            "Force result:",
            force_result.details,
        ]

        if energy_result.fallback_suggestion:
            lines.extend(["", f"Energy method fallback suggestion: {energy_result.fallback_suggestion}"])

        if self._recommended_value is not None:
            if self._sweep_kind == "cutoff":
                lines.extend(["", f"Recommended cutoff: {self._recommended_value:.8g} eV"])
            else:
                label = self._point_labels.get(
                    float(self._recommended_value),
                    f"{self._recommended_value:.8g} k-points",
                )
                lines.extend(["", f"Recommended k-point grid: {label}"])
        else:
            lines.extend(["", "No combined convergence recommendation available yet."])

        self._results.setPlainText("\n".join(lines))
        self._last_plot_data = {
            "cutoffs": list(cutoffs),
            "energies": list(energies),
            "forces": list(forces),
            "energy_threshold": energy_threshold,
            "force_threshold": force_threshold,
        }
        self._update_plot(cutoffs, energies, forces, energy_threshold, force_threshold)

        if self._recommended_value is None:
            self._adopt_button.setEnabled(False)
            self._adopt_button.setText(
                "Set as Project Default Cutoff"
                if self._sweep_kind == "cutoff"
                else "Set as Project Default K-Point Grid"
            )
        else:
            self._adopt_button.setEnabled(True)
            if self._sweep_kind == "cutoff":
                self._adopt_button.setText(
                    f"Set {self._recommended_value:.8g} eV as Project Default Cutoff"
                )
            else:
                label = self._point_labels.get(
                    float(self._recommended_value),
                    f"{self._recommended_value:.8g} k-points",
                )
                self._adopt_button.setText(f"Set {label} as Project Default K-Point Grid")

        self._emit_state_changed()

    def _update_plot(
        self,
        cutoffs: list[float],
        energies: list[float],
        forces: list[float],
        energy_threshold: float,
        force_threshold: float,
    ) -> None:
        palette = current_palette()
        recommended_line = palette.text_muted
        threshold_line = palette.warning

        self._figure.set_facecolor(palette.bg_surface)
        self._figure.clear()
        energy_ax = self._figure.add_subplot(2, 1, 1)
        force_ax = self._figure.add_subplot(2, 1, 2, sharex=energy_ax)

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

        force_ax.plot(cutoffs, forces, marker="o", linestyle="-", color="#b91c1c", label="Max force")
        if self._force_result and self._force_result.converged_value is not None:
            force_ax.axvline(
                self._force_result.converged_value,
                color="#059669",
                linestyle="--",
                label="Force convergence",
            )
        force_ax.axhline(force_threshold, color=threshold_line, linestyle=":", label="Force threshold")
        if self._recommended_value is not None:
            force_ax.axvline(self._recommended_value, color=recommended_line, linestyle="-", alpha=0.55)

        force_ax.set_xlabel(
            "Cutoff energy (eV)" if self._sweep_kind == "cutoff" else "Number of k-points",
            labelpad=8,
        )
        force_ax.set_ylabel("Max force (eV/Ang)")
        force_ax.legend(loc="best")
        self._apply_axis_theme(force_ax)
        self._apply_legend_theme(force_ax)

        # Mention the active thresholds in the axes titles for quick scanning.
        energy_ax.set_title(f"Energy convergence (threshold {energy_threshold:.3g} eV/atom)", pad=10)
        force_ax.set_title(f"Force convergence (delta threshold {force_threshold:.3g} eV/Ang)", pad=10)

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
        self._emit_state_changed()
        self.default_adopted.emit(float(self._recommended_value), method, energy_threshold, force_threshold)
        if self._sweep_kind == "cutoff":
            self.cutoff_adopted.emit(float(self._recommended_value), method, energy_threshold, force_threshold)
