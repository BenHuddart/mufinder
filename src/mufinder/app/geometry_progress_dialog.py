"""Geometry-optimisation progress on the Job Dashboard: its cell, tooltip and dialog.

The dashboard's Progress column stays narrow for small screens: it reads
``14 · 3/4`` (latest iteration · convergence criteria met / criteria tested).
Its tooltip spells out each criterion against its tolerance, and Inspect on a
running geometry optimisation opens :class:`GeometryProgressDialog`, which plots
each criterion over the iterations against its tolerance.
"""

from __future__ import annotations

from html import escape
import math

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from mufinder.app.plot_canvas import PlotFrame, create_plot_canvas
from mufinder.app.screen_utils import clamped_size
from mufinder.app.theme import current_palette, style_hint_text, use_square_top_table_corners
from mufinder.castep_io.geometry_progress import GeometryIteration, GeometryProgress

PROGRESS_COLUMN_TOOLTIP = (
    "Geometry optimisation progress: latest iteration · convergence criteria met.\n"
    "Hover a cell for each criterion; Inspect plots the history."
)


def progress_cell_text(progress: GeometryProgress | None) -> str:
    """Return the dashboard's compact Progress text, e.g. ``14 · 3/4``."""

    if progress is None:
        return ""
    latest = progress.latest
    if latest is None:
        return "starting"
    return f"{latest.iteration} · {latest.criteria_met}/{len(latest.criteria)}"


def _format_value(value: float) -> str:
    return f"{value:.2e}" if value and not 1e-2 <= abs(value) < 1e3 else f"{value:.3g}"


def progress_heading(progress: GeometryProgress) -> str:
    """Return one line naming the method, iteration, step limit and restart."""

    latest = progress.latest
    parts = [progress.method or "Geometry optimisation"]
    if latest is None:
        parts.append("waiting for the first iteration")
    elif progress.max_iterations is not None:
        parts.append(f"iteration {latest.iteration} of at most {progress.max_iterations}")
    else:
        parts.append(f"iteration {latest.iteration}")
    if progress.restart:
        parts.append(f"restart {progress.restart}")
    if progress.converged is True:
        parts.append("converged")
    elif progress.converged is False:
        parts.append("stopped without converging")
    return " · ".join(parts)


def progress_tooltip(progress: GeometryProgress | None) -> str:
    """Return the Progress cell's tooltip: each criterion against its tolerance."""

    if progress is None:
        return ""
    rows = "".join(
        "<tr>"
        f"<td>{escape(criterion.name)}</td>"
        f"<td align='right'>{_format_value(criterion.value)}</td>"
        f"<td>/ {_format_value(criterion.tolerance)} {escape(criterion.unit)}</td>"
        f"<td>{'✓' if criterion.converged else '✗'}</td>"
        "</tr>"
        for criterion in (progress.latest.criteria if progress.latest is not None else ())
    )
    table = f"<table cellspacing='4'>{rows}</table>" if rows else ""
    return f"<b>{escape(progress_heading(progress))}</b>{table}"


class GeometryProgressDialog(QDialog):
    """Plot a running geometry optimisation's convergence criteria over its iterations."""

    #: Asks the window to read the job's output again now.
    refresh_requested = Signal()

    def __init__(
        self, job_label: str, progress: GeometryProgress | None, parent: QWidget | None = None
    ) -> None:
        super().__init__(parent)
        self.setWindowTitle(f"Geometry Progress - {job_label}")
        self.resize(clamped_size(self, 820, 640))

        self._heading = QLabel(self)
        self._heading.setWordWrap(True)
        hint = QLabel(
            "Each plot follows one convergence criterion over the iterations; the dashed line is its "
            "tolerance. CASTEP stops once every criterion is at or below its tolerance.",
            self,
        )
        style_hint_text(hint)
        hint.setWordWrap(True)

        self._table = QTableWidget(0, 4, self)
        self._table.setHorizontalHeaderLabels(["Criterion", "Value", "Tolerance", "Met"])
        self._table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self._table.setSelectionMode(QAbstractItemView.SelectionMode.NoSelection)
        self._table.setFocusPolicy(Qt.FocusPolicy.NoFocus)
        self._table.verticalHeader().setVisible(False)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        use_square_top_table_corners(self._table)

        palette = current_palette()
        self._figure, self._canvas = create_plot_canvas(
            figsize=(7.6, 4.4),
            facecolor=palette.bg_surface,
            minimum_size=(420, 300),
        )
        self._plot_frame = PlotFrame(self._canvas, title="Convergence history", parent=self)

        buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close, self)
        self._refresh_button = buttons.addButton("Refresh", QDialogButtonBox.ButtonRole.ActionRole)
        self._refresh_button.clicked.connect(self.refresh_requested)
        buttons.rejected.connect(self.reject)

        layout = QVBoxLayout(self)
        layout.addWidget(self._heading)
        layout.addWidget(hint)
        layout.addWidget(self._table)
        layout.addWidget(self._plot_frame, 1)
        layout.addWidget(buttons)

        self._progress: GeometryProgress | None = None
        self.set_progress(progress)

    @property
    def progress(self) -> GeometryProgress | None:
        return self._progress

    def set_progress(self, progress: GeometryProgress | None) -> None:
        self._progress = progress
        if progress is None:
            self._heading.setText(
                "No geometry-optimisation progress has been read from this job's output yet."
            )
        else:
            self._heading.setText(progress_heading(progress))
        self._fill_table(progress.latest if progress is not None else None)
        self._draw(progress)

    def _fill_table(self, latest: GeometryIteration | None) -> None:
        criteria = latest.criteria if latest is not None else ()
        palette = current_palette()
        self._table.setRowCount(len(criteria))
        for row, criterion in enumerate(criteria):
            met = QTableWidgetItem("Yes" if criterion.converged else "No")
            met.setForeground(QColor(palette.success if criterion.converged else palette.danger))
            cells = (
                QTableWidgetItem(criterion.name),
                QTableWidgetItem(f"{_format_value(criterion.value)} {criterion.unit}"),
                QTableWidgetItem(f"{_format_value(criterion.tolerance)} {criterion.unit}"),
                met,
            )
            for column, item in enumerate(cells):
                item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self._table.setItem(row, column, item)
        header_height = self._table.horizontalHeader().sizeHint().height()
        row_height = self._table.verticalHeader().defaultSectionSize()
        self._table.setFixedHeight(header_height + row_height * max(1, len(criteria)) + 4)

    def _draw(self, progress: GeometryProgress | None) -> None:
        palette = current_palette()
        self._figure.clear()
        iterations = progress.iterations if progress is not None else ()
        names: list[str] = []
        for iteration in iterations:
            names.extend(
                criterion.name for criterion in iteration.criteria if criterion.name not in names
            )
        if not names:
            axis = self._figure.add_subplot(111)
            axis.set_axis_off()
            axis.text(
                0.5, 0.5, "No iterations yet", ha="center", va="center", color=palette.text_muted
            )
            self._canvas.draw_idle()
            return

        columns = 2 if len(names) > 1 else 1
        rows = math.ceil(len(names) / columns)
        for index, name in enumerate(names):
            axis = self._figure.add_subplot(rows, columns, index + 1)
            points = [
                (iteration.iteration, criterion)
                for iteration in iterations
                for criterion in iteration.criteria
                if criterion.name == name
            ]
            steps = [step for step, _criterion in points]
            unit = points[-1][1].unit
            axis.set_facecolor(palette.bg_surface_alt)
            axis.tick_params(colors=palette.text_primary, labelsize=8)
            for spine in axis.spines.values():
                spine.set_color(palette.border)
            axis.grid(True, color=palette.border, alpha=0.35)
            # dE/ion is 0 at iteration 0 and a log axis cannot show it, so
            # non-positive values are masked rather than drawn at the axis edge.
            axis.set_yscale("log", nonpositive="mask")
            axis.plot(
                steps,
                [criterion.value for _step, criterion in points],
                marker="o",
                markersize=3,
                linewidth=1.4,
                color=palette.status_running,
            )
            axis.plot(
                steps,
                [criterion.tolerance for _step, criterion in points],
                linestyle="--",
                linewidth=1.2,
                color=palette.success,
            )
            axis.set_title(
                f"{name} ({unit})" if unit else name, color=palette.text_primary, fontsize=9
            )
            if index >= len(names) - columns:
                axis.set_xlabel("Iteration", color=palette.text_primary, fontsize=8)
            axis.xaxis.get_major_locator().set_params(integer=True)
        self._canvas.draw_idle()
