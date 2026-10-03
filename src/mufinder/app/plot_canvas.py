"""Matplotlib canvases that keep axis labels, ticks and legends visible at any size.

Margins given as figure fractions (``Figure.subplots_adjust``) reserve a fixed
*share* of the canvas, but tick labels and axis labels need a fixed number of
*pixels*. When a dialog opens small (a 13" laptop, or Windows at 125-150 %
display scaling) that share shrinks below what the labels need and they are
clipped, including the minus signs of negative tick values, which silently
flips the sign the reader sees. Constrained layout instead recomputes the
margins from the rendered label extents on every resize.

Every embedded MuFinder plot is created through :func:`create_plot_canvas`, so
each figure uses constrained layout and each canvas has a minimum size below
which a splitter or box layout cannot squeeze it into an unreadable strip.
"""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QSizePolicy

DEFAULT_MINIMUM_CANVAS_SIZE: tuple[int, int] = (220, 160)
"""Smallest canvas (logical pixels) that still fits ticks, labels and a plot area."""


def create_plot_canvas(
    *,
    figsize: tuple[float, float],
    facecolor: str | None = None,
    transparent: bool = False,
    minimum_size: tuple[int, int] = DEFAULT_MINIMUM_CANVAS_SIZE,
) -> tuple[Figure, FigureCanvasQTAgg]:
    """Return a constrained-layout figure and the Qt canvas that displays it.

    ``figsize`` only seeds the canvas size hint; the rendered size follows the
    widget. ``transparent`` lets the surrounding Qt surface show through the
    figure background (the figure patch and the canvas widget are both made
    transparent), in which case ``facecolor`` is ignored. ``minimum_size`` is
    the ``(width, height)`` the canvas refuses to shrink below; pass a larger
    value for plots that need more room, never ``(0, 0)``.
    """

    figure = Figure(figsize=figsize, facecolor="none" if transparent else facecolor, layout="constrained")
    if transparent:
        figure.patch.set_alpha(0.0)
    canvas = FigureCanvasQTAgg(figure)
    canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
    canvas.setMinimumSize(*minimum_size)
    if transparent:
        canvas.setAttribute(Qt.WidgetAttribute.WA_TranslucentBackground, True)
        canvas.setStyleSheet("background: transparent;")
    return figure, canvas
