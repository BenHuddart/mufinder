"""Magnetic Fields field-distribution plot and its data export."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
from matplotlib.ticker import ScalarFormatter
from PySide6.QtWidgets import QDialog, QDialogButtonBox, QVBoxLayout, QWidget

from mufinder.app.export_dialogs import (
    DATA_TEXT_FORMATS,
    ask_save_path,
    report_export_failed,
    report_export_saved,
)
from mufinder.app.plot_canvas import create_plot_canvas
from mufinder.app.theme import (
    categorical_colour,
    current_palette,
    plot_series_colours,
    style_plot_axes,
    style_plot_legend,
)
from mufinder.persistence.codec import float_triplet
from mufinder.science.field_histograms import (
    common_binned_series,
    distribution_series,
    field_axis_limits,
    interval_steps_with_zero_baseline,
    probability_steps_with_zero_baseline,
    summed_probability_intervals,
    summed_probability_steps,
)
from mufinder.science.magnetic_fields import MagneticFieldError
from mufinder.services.magnetic_field_results import UNRELAXED_SITE_NOTE


class MagneticFieldDistributionDialog(QDialog):
    """Plot an incommensurate magnetic-field magnitude distribution."""

    def __init__(
        self,
        results: object | Sequence[object],
        parent: QWidget | None = None,
        *,
        estimate: bool = False,
    ) -> None:
        super().__init__(parent)
        self._results = self._coerce_results(results)
        self._estimate = estimate
        self._series_error: str | None = None
        label = self._title_label()
        self.setWindowTitle(f"Field Distribution - {label}")
        self.resize(760, 520)

        palette = current_palette()
        self._figure, self._canvas = create_plot_canvas(
            figsize=(7.2, 4.6),
            facecolor=palette.bg_surface,
            minimum_size=(400, 280),
        )

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        self._export_button = buttons.addButton(
            "Export Data…", QDialogButtonBox.ButtonRole.ActionRole
        )
        self._export_button.clicked.connect(self._export_data)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._canvas, 1)
        layout.addWidget(buttons)

        self._draw_plot()
        self._export_button.setEnabled(self._series_error is None)

    def _draw_plot(self) -> None:
        palette = current_palette()
        self._figure.clear()
        axis = self._figure.add_subplot(111)
        style_plot_axes(axis)
        axis.set_xlabel(r"$B$ (T)")
        axis.set_ylabel(r"$p(B)$")
        axis.set_title(self._title_label())
        x_formatter = ScalarFormatter(useOffset=False)
        x_formatter.set_scientific(False)
        axis.xaxis.set_major_formatter(x_formatter)

        try:
            series = self._distribution_series()
        except MagneticFieldError as exc:
            # Samples binned here (no stored histogram) may hold a non-finite field.
            series = []
            self._series_error = str(exc)
        common_edges: np.ndarray | None = None
        if len(series) > 1:
            common = common_binned_series(series)
            if common is not None:
                series, common_edges = common
        x_limits: list[tuple[float, float]] = []
        y_max = 0.0
        for centres, probabilities, magnitudes, _label in series:
            x_limits.append(field_axis_limits(centres, magnitudes))
            y_max = max(y_max, float(np.max(probabilities)))

        if series:
            low = min(limit[0] for limit in x_limits)
            high = max(limit[1] for limit in x_limits)
            for index, (centres, probabilities, magnitudes, label) in enumerate(series):
                if common_edges is None:
                    plot_x, plot_y = probability_steps_with_zero_baseline(
                        centres, probabilities, low, high
                    )
                else:
                    plot_x, plot_y = interval_steps_with_zero_baseline(
                        common_edges, probabilities, low, high
                    )
                line = axis.plot(
                    plot_x, plot_y, linewidth=1.8, color=categorical_colour(index), label=label
                )[0]
                fill_alpha = 0.24 if len(series) == 1 else 0.06
                axis.fill_between(plot_x, plot_y, color=line.get_color(), alpha=fill_alpha)
            if len(series) > 1:
                if common_edges is None:
                    sum_x, sum_y, sum_values = summed_probability_steps(series, low, high)
                else:
                    sum_values = np.sum(
                        [probabilities for _centres, probabilities, _magnitudes, _label in series],
                        axis=0,
                    )
                    sum_x, sum_y = interval_steps_with_zero_baseline(
                        common_edges, sum_values, low, high
                    )
                sum_line = axis.plot(
                    sum_x, sum_y, linewidth=2.4, color=plot_series_colours().total, label="sum"
                )[0]
                axis.fill_between(sum_x, sum_y, color=sum_line.get_color(), alpha=0.16)
                if sum_values.size:
                    y_max = max(y_max, float(np.max(sum_values)))
            axis.set_xlim(low, high)
            axis.set_ylim(0.0, max(1.05, y_max * 1.08))
            if len(self._results) > 1:
                style_plot_legend(axis.legend(loc="best", frameon=True))
        else:
            axis.text(
                0.5,
                0.5,
                "No distribution samples available."
                if self._series_error is None
                else f"Distribution unavailable: {self._series_error}.",
                ha="center",
                va="center",
                color=palette.text_muted,
                transform=axis.transAxes,
            )
        self._canvas.draw_idle()

    @staticmethod
    def _coerce_results(results: object | Sequence[object]) -> tuple[object, ...]:
        if isinstance(results, Sequence) and not isinstance(results, (str, bytes, bytearray)):
            return tuple(results)
        return (results,)

    def _title_label(self) -> str:
        if len(self._results) == 1:
            label = str(getattr(self._results[0], "site_label", "Field distribution"))
        else:
            label = f"{len(self._results)} field distributions"
        return f"{label} (unrelaxed estimate)" if self._estimate else label

    def _distribution_series(self) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, str]]:
        return distribution_series(self._results)

    def _export_data(self) -> None:
        path = ask_save_path(
            self, "Export Field Distribution", "field_distribution.dat", DATA_TEXT_FORMATS
        )
        if path is None:
            return
        try:
            path.write_text(self._export_text(), encoding="utf-8")
        except OSError as exc:
            report_export_failed(self, exc)
            return
        report_export_saved(self, path)

    def _export_text(self) -> str:
        series = self._distribution_series()
        common_edges: np.ndarray | None = None
        if len(series) > 1:
            common = common_binned_series(series)
            if common is not None:
                series, common_edges = common
        lines = self._export_header(series)
        if not series:
            return "\n".join(lines) + "\n"
        low = min(
            field_axis_limits(centres, magnitudes)[0]
            for centres, _probabilities, magnitudes, _label in series
        )
        high = max(
            field_axis_limits(centres, magnitudes)[1]
            for centres, _probabilities, magnitudes, _label in series
        )
        if common_edges is None:
            interval_edges, values_by_label, sum_values = summed_probability_intervals(
                series, low, high
            )
        else:
            interval_edges = common_edges
            values_by_label = {
                label: probabilities for _centres, probabilities, _magnitudes, label in series
            }
            sum_values = np.sum(tuple(values_by_label.values()), axis=0)
        centres = 0.5 * (interval_edges[:-1] + interval_edges[1:])
        labels = [label for _centres, _probabilities, _magnitudes, label in series]
        column_labels = ["B_T", *(f"p_{label}" for label in labels)]
        if len(series) > 1:
            column_labels.append("p_sum")
        lines.append(
            "# Columns: " + " ".join(self._safe_column_label(label) for label in column_labels)
        )
        for row_index, centre in enumerate(centres):
            row = [float(centre)]
            row.extend(float(values_by_label[label][row_index]) for label in labels)
            if len(series) > 1:
                row.append(float(sum_values[row_index]))
            lines.append(" ".join(f"{value:.12g}" for value in row))
        return "\n".join(lines) + "\n"

    def _export_header(
        self, series: Sequence[tuple[np.ndarray, np.ndarray, np.ndarray, str]]
    ) -> list[str]:
        lines = [
            "# MuFinder field distribution export",
            f"# Distributions: {len(series)}",
        ]
        if self._estimate:
            lines.append(f"# Estimate: {UNRELAXED_SITE_NOTE}")
        for index, result in enumerate(self._results, start=1):
            label = str(getattr(result, "site_label", f"site {index}"))
            position = getattr(result, "muon_position", None)
            if position is not None:
                try:
                    x, y, z = float_triplet(position)
                    lines.append(f"# {label} muon_position_fractional: {x:.12g} {y:.12g} {z:.12g}")
                except (TypeError, ValueError):
                    pass
            params = getattr(result, "calculation_parameters", None)
            if isinstance(params, Mapping) and params:
                lines.append(f"# {label} calculation_parameters:")
                for key in sorted(params):
                    lines.append(f"#   {key}: {params[key]}")
        if not any("calculation_parameters" in line for line in lines):
            lines.append("# Calculation parameters: unavailable for this result")
        return lines

    @staticmethod
    def _safe_column_label(label: str) -> str:
        return "".join(char if char.isalnum() or char in {"_", "-"} else "_" for char in label)
