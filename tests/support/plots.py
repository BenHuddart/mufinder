"""Measuring hover notes on a drawn matplotlib figure, for the plot tests."""

from __future__ import annotations

import math

from matplotlib.text import Annotation
from matplotlib.transforms import Bbox


def hover_note_box(note: Annotation, renderer) -> Bbox:
    """Where a drawn hover note is on screen, its box and border included, in display pixels."""

    boxes = [note.get_window_extent(renderer)]
    patch = note.get_bbox_patch()
    if patch is not None:
        border = patch.get_linewidth() * note.figure.dpi / 72.0 / 2.0
        boxes.append(patch.get_window_extent(renderer).padded(border))
    return Bbox.union(boxes)


def panel_positions(figure) -> list[float]:
    """Every panel's position in ``figure``, as one flat list of figure fractions."""

    return [value for axes in figure.axes for value in axes.get_position().bounds]


def box_inside(box: Bbox, bounds: Bbox, *, tolerance: float = 0.5) -> bool:
    """Whether ``box`` lies wholly within ``bounds``, to ``tolerance`` pixels."""

    return (
        box.x0 >= bounds.x0 - tolerance
        and box.x1 <= bounds.x1 + tolerance
        and box.y0 >= bounds.y0 - tolerance
        and box.y1 <= bounds.y1 + tolerance
    )


def distance_to_box(box: Bbox, x: float, y: float) -> float:
    """The distance from the pixel ``(x, y)`` to the nearest edge of ``box``; 0 inside it."""

    return math.hypot(max(box.x0 - x, 0.0, x - box.x1), max(box.y0 - y, 0.0, y - box.y1))
