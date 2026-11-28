"""Tests for cutoff convergence examiner dialog."""

from __future__ import annotations

import itertools
import math

from matplotlib.backend_bases import MouseEvent
from matplotlib.colors import to_hex
from matplotlib.transforms import Bbox
import pytest
from PySide6.QtCore import QPoint, QRect, QRectF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QComboBox, QPushButton, QScrollArea, QWidget

import mufinder.app.convergence_examiner as convergence_examiner_module
from mufinder.app import theme as theme_module
from mufinder.app.convergence_examiner import ConvergenceExaminerDialog
from mufinder.app.convergence_types import SweepPointRunState
from mufinder.app.plot_canvas import HOVER_NOTE_OFFSET
from mufinder.persistence.codec import to_jsonable
from tests.support.plots import box_inside, distance_to_box, hover_note_box, panel_positions


def _completed_state(energy: float, force: float) -> SweepPointRunState:
    return SweepPointRunState(status="completed", total_energy=energy, max_force=force)


def _move_mouse(dialog: ConvergenceExaminerDialog, axis, x: float) -> MouseEvent:
    """Move the mouse over the middle of ``axis``'s panel at its data x ``x``."""

    dialog._canvas.draw()
    pixel_x = axis.transData.transform((x, 0.0))[0]
    pixel_y = axis.bbox.y0 + axis.bbox.height / 2
    event = MouseEvent("motion_notify_event", dialog._canvas, pixel_x, pixel_y)
    dialog._canvas.callbacks.process("motion_notify_event", event)
    return event


def test_convergence_examiner_disables_run_with_insufficient_points(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.3),
        },
        num_atoms=2,
    )

    assert dialog._run_button.isEnabled() is False
    assert "Need at least two" in dialog._results.toPlainText()


def _settle(qapp, passes: int = 4) -> None:
    for _ in range(passes):
        qapp.processEvents()


def _analysed_dialog(qapp, size: tuple[int, int]) -> ConvergenceExaminerDialog:
    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.30),
            400.0: _completed_state(-10.01, 0.20),
            500.0: _completed_state(-10.0105, 0.19),
        },
        num_atoms=2,
    )
    dialog.show()
    dialog.resize(*size)
    _settle(qapp)
    dialog._run_button.click()
    _settle(qapp)
    return dialog


def _kpoint_dialog() -> ConvergenceExaminerDialog:
    return ConvergenceExaminerDialog(
        {
            4.0: _completed_state(-10.0, 0.30),
            32.0: _completed_state(-10.01, 0.20),
            108.0: _completed_state(-10.0105, 0.19),
        },
        num_atoms=2,
        sweep_kind="kpoint",
        point_grids={4.0: (1, 1, 4), 32.0: (2, 2, 8), 108.0: (3, 3, 12)},
    )


def _bounds_in_dialog(dialog: ConvergenceExaminerDialog, widget: QWidget) -> QRect:
    return QRect(widget.mapTo(dialog, QPoint(0, 0)), widget.size())


def _action_buttons(dialog: ConvergenceExaminerDialog) -> list[QRect]:
    return [
        _bounds_in_dialog(dialog, button)
        for button in (dialog._run_button, dialog._adopt_button, dialog._close_button)
    ]


@pytest.mark.parametrize("size", [(1080, 900), (880, 646)], ids=["default", "minimum"])
def test_convergence_examiner_shows_the_whole_plot_without_scrolling(qapp, size) -> None:
    dialog = _analysed_dialog(qapp, size)
    try:
        assert (dialog.minimumWidth(), dialog.minimumHeight()) == (880, 646)
        assert (dialog.width(), dialog.height()) == size
        assert not dialog.findChildren(QScrollArea)
        for widget in (
            dialog._energy_threshold,
            dialog._force_threshold,
            dialog._stress_threshold,
            dialog._run_button,
            dialog._adopt_button,
            dialog._close_button,
            dialog._results,
            dialog._canvas,
        ):
            assert dialog.rect().contains(_bounds_in_dialog(dialog, widget)), widget

        # Everything under the results is the plot, down to the dialog's margin.
        results = _bounds_in_dialog(dialog, dialog._results)
        canvas = _bounds_in_dialog(dialog, dialog._canvas)
        assert canvas.top() > results.bottom()
        assert canvas.bottom() >= dialog.rect().bottom() - 16
        assert canvas.height() > results.height()

        # The force panel's axis and legend are on the canvas.
        force_bounds = dialog._force_ax.get_tightbbox(dialog._canvas.get_renderer())
        assert force_bounds.y0 >= 0
        assert force_bounds.y1 <= dialog._figure.bbox.height
        assert dialog._force_ax.get_legend() is not None
        # No axis label is taller than its panel.
        for axes in (dialog._energy_ax, dialog._force_ax):
            label = axes.yaxis.label.get_window_extent(dialog._canvas.get_renderer())
            assert label.height <= axes.get_window_extent().height
    finally:
        dialog.close()


# The CoF2 docs example (two formula units): energies above the finest point's, max forces.
_COF2_ENERGY_EV = -5621.8743
_COF2_CUTOFF_SWEEP = {
    300.0: (1.84, 0.62),
    400.0: (0.37, 0.21),
    500.0: (0.061, 0.083),
    600.0: (0.0049, 0.034),
    700.0: (0.0007, 0.013),
    800.0: (0.0, 0.011),
}
_COF2_KPOINT_SWEEP = {
    (2, 2, 3): (0.214, 0.046),
    (3, 3, 4): (0.0112, 0.012),
    (4, 4, 6): (0.0006, 0.004),
    (5, 5, 7): (0.0001, 0.003),
    (6, 6, 8): (0.0, 0.003),
}
_COF2_LATTICE = ((4.6956, 0.0, 0.0), (0.0, 4.6956, 0.0), (0.0, 0.0, 3.1817))
_COF2_STRESSES = (2.1, 0.9, 0.35, 0.12, 0.07, 0.05)


def _cof2_examiner(kind: str, *, stress: bool = False) -> ConvergenceExaminerDialog:
    if kind == "cutoff":
        points = {
            cutoff: SweepPointRunState(
                status="completed",
                total_energy=_COF2_ENERGY_EV + energy,
                max_force=force,
                max_stress=_COF2_STRESSES[index] if stress else None,
            )
            for index, (cutoff, (energy, force)) in enumerate(_COF2_CUTOFF_SWEEP.items())
        }
        return ConvergenceExaminerDialog(points, num_atoms=6)
    grids = {float(math.prod(grid)): grid for grid in _COF2_KPOINT_SWEEP}
    points = {
        float(math.prod(grid)): _completed_state(_COF2_ENERGY_EV + energy, force)
        for grid, (energy, force) in _COF2_KPOINT_SWEEP.items()
    }
    return ConvergenceExaminerDialog(
        points,
        num_atoms=6,
        sweep_kind="kpoint",
        point_grids=grids,
        lattice_vectors=_COF2_LATTICE,
    )


def _plotted_points(axes) -> list[tuple[float, float]]:
    """Every data marker of ``axes`` in display coordinates."""

    points: list[tuple[float, float]] = []
    for line in axes.get_lines():
        if line.get_marker() in (None, "None", "", " "):
            continue
        for x, y in zip(line.get_xdata(), line.get_ydata(), strict=True):
            if math.isfinite(x) and math.isfinite(y):
                points.append(tuple(axes.transData.transform((x, y))))
    return points


def _assert_legends_clear(dialog: ConvergenceExaminerDialog) -> None:
    """No legend covers a data point, a label or the canvas edge, and no label is cut off."""

    dialog._canvas.draw()
    figure = dialog._figure
    renderer = dialog._canvas.get_renderer()
    legends = [axes.get_legend() for axes in figure.axes if axes.get_legend() is not None]
    assert len(legends) == 2
    canvas = figure.bbox
    for legend in legends:
        box = legend.get_tightbbox(renderer)
        assert box.x0 >= canvas.x0 - 1 and box.x1 <= canvas.x1 + 1
        assert box.y0 >= canvas.y0 - 1 and box.y1 <= canvas.y1 + 1
        for axes in figure.axes:
            for x, y in _plotted_points(axes):
                assert not box.contains(x, y), (legend, x, y)
            for label in [*axes.get_yticklabels(), *axes.get_xticklabels()]:
                if label.get_text():
                    assert not box.overlaps(label.get_window_extent(renderer)), label
            for text in (axes.yaxis.label, axes.xaxis.label, axes.title):
                if text.get_text():
                    assert not box.overlaps(text.get_window_extent(renderer)), text
    for axes in figure.axes:
        tight = axes.get_tightbbox(renderer)
        assert tight.x0 >= canvas.x0 - 1 and tight.x1 <= canvas.x1 + 1
        assert tight.y0 >= canvas.y0 - 1 and tight.y1 <= canvas.y1 + 1
        assert axes.yaxis.get_offset_text().get_text() == ""
        assert axes.xaxis.get_offset_text().get_text() == ""


@pytest.mark.parametrize("size", [(1080, 900), (880, 646)], ids=["default", "minimum"])
@pytest.mark.parametrize("ratio", [1.0, 2.0], ids=["dpr1", "dpr2"])
@pytest.mark.parametrize(
    ("kind", "stress", "narrow"),
    [
        ("cutoff", False, False),
        ("kpoint", False, False),
        ("cutoff", True, False),
        # Sub-unit thresholds give the widest tick labels (0.0005).
        ("cutoff", True, True),
        ("kpoint", False, True),
    ],
    ids=["cutoff", "kpoint", "cutoff-stress", "cutoff-stress-narrow", "kpoint-narrow"],
)
def test_convergence_examiner_legends_cover_no_point_or_label(
    qapp, size, ratio, kind, stress, narrow
) -> None:
    dialog = _cof2_examiner(kind, stress=stress)
    try:
        dialog.show()
        dialog.resize(*size)
        _settle(qapp)
        dialog._canvas._set_device_pixel_ratio(ratio)
        if narrow:
            dialog._energy_threshold.setValue(0.0005)
            dialog._force_threshold.setValue(0.0005)
        if stress:
            dialog._include_stress.setChecked(True)
            dialog._stress_threshold.setValue(0.0005 if narrow else 0.1)
        dialog._run_button.click()
        _settle(qapp)
        _assert_legends_clear(dialog)
    finally:
        dialog.close()


def _relative_axes(dialog: ConvergenceExaminerDialog) -> list:
    return [axes for axes in (dialog._energy_ax, dialog._force_ax, dialog._stress_ax) if axes]


def _threshold_of(axes) -> float:
    return axes.yaxis.get_transform().linthresh


def _plotted_offsets(axes) -> list[float]:
    values: list[float] = []
    for line in axes.get_lines():
        if line.get_marker() in (None, "None", "", " "):
            continue
        values += [float(y) for y in line.get_ydata() if math.isfinite(y)]
    return values


def _assert_tick_rule(dialog: ConvergenceExaminerDialog) -> None:
    """The tick rule of every relative axis, in its priority order.

    (a) no two y tick labels overlap; (b) 0 and the positive threshold are
    labelled; (c) each side the data passes the band on has a labelled tick at
    or beyond the decade of its outermost value.
    """

    dialog._canvas.draw()
    renderer = dialog._canvas.get_renderer()
    for axes in _relative_axes(dialog):
        if axes.get_yscale() != "symlog":
            continue
        low, high = sorted(axes.get_ylim())
        labels = [
            label
            for label in axes.get_yticklabels()
            if label.get_text() and low <= label.get_position()[1] <= high
        ]
        boxes = [label.get_window_extent(renderer) for label in labels]
        for first, second in itertools.combinations(boxes, 2):
            assert not first.overlaps(second), (axes.get_ylabel(), [t.get_text() for t in labels])
        values = [label.get_position()[1] for label in labels]
        threshold = _threshold_of(axes)
        assert 0.0 in values, axes.get_ylabel()
        assert any(math.isclose(value, threshold) for value in values), axes.get_ylabel()
        offsets = _plotted_offsets(axes)
        for sign in (1.0, -1.0):
            extent = max((sign * value for value in offsets), default=0.0)
            if extent <= threshold * 1.0001:
                continue
            decade = 10.0 ** math.floor(math.log10(extent))
            assert any(sign * value >= decade * (1 - 1e-9) for value in values), (
                axes.get_ylabel(),
                sign,
                extent,
                [t.get_text() for t in labels],
            )


def _run_examiner(dialog, qapp, size, ratio, **thresholds) -> None:
    dialog.show()
    dialog.resize(*size)
    _settle(qapp)
    dialog._canvas._set_device_pixel_ratio(ratio)
    if "energy" in thresholds:
        dialog._energy_threshold.setValue(thresholds["energy"])
    if "force" in thresholds:
        dialog._force_threshold.setValue(thresholds["force"])
    if thresholds.get("stress"):
        dialog._include_stress.setChecked(True)
    dialog._run_button.click()
    _settle(qapp)


@pytest.mark.parametrize("ratio", [1.0, 2.0], ids=["dpr1", "dpr2"])
@pytest.mark.parametrize("size", [(880, 646), (1080, 900)], ids=["minimum", "default"])
@pytest.mark.parametrize("stress", [False, True], ids=["no-stress", "stress"])
@pytest.mark.parametrize("energy_threshold", [1e-6, 1e-3, 1.0])
@pytest.mark.parametrize("kind", ["cutoff", "kpoint"])
def test_convergence_examiner_tick_rule_holds_at_every_size(
    qapp, kind, energy_threshold, stress, size, ratio
) -> None:
    dialog = _cof2_examiner(kind, stress=stress)
    try:
        _run_examiner(dialog, qapp, size, ratio, energy=energy_threshold, stress=stress)
        _assert_tick_rule(dialog)
        # The labels follow the panel's height when the dialog is resized.
        dialog.resize(980, 760)
        _settle(qapp)
        _assert_tick_rule(dialog)
    finally:
        dialog.close()


def test_convergence_examiner_tick_rule_on_the_kpoint_force_panel_at_a_tight_threshold(
    qapp,
) -> None:
    # The checker's case: max force 0.043 eV/Ang against a 0.02 threshold.
    dialog = _cof2_examiner("kpoint")
    try:
        _run_examiner(dialog, qapp, (1080, 900), 1.0, force=0.02)
        _assert_tick_rule(dialog)
        dialog._canvas.draw()
        labels = [label.get_text() for label in dialog._force_ax.get_yticklabels()]
        # The largest force is bracketed by the threshold and a round value above it.
        assert "0.05" in labels
    finally:
        dialog.close()


@pytest.mark.parametrize("size", [(880, 646), (1080, 900)], ids=["minimum", "default"])
def test_convergence_examiner_tick_rule_below_the_band(qapp, size) -> None:
    # Energies that rise towards the densest grid: every offset is negative.
    grids = {float(math.prod(grid)): grid for grid in _COF2_KPOINT_SWEEP}
    points = {
        float(math.prod(grid)): _completed_state(_COF2_ENERGY_EV - energy, force)
        for grid, (energy, force) in _COF2_KPOINT_SWEEP.items()
    }
    dialog = ConvergenceExaminerDialog(
        points,
        num_atoms=6,
        sweep_kind="kpoint",
        point_grids=grids,
        lattice_vectors=_COF2_LATTICE,
    )
    try:
        _run_examiner(dialog, qapp, size, 1.0)
        _assert_tick_rule(dialog)
    finally:
        dialog.close()


@pytest.mark.parametrize("kind", ["cutoff", "kpoint"])
@pytest.mark.parametrize("size", [(880, 646), (1080, 900)], ids=["minimum", "default"])
def test_convergence_examiner_tick_rule_with_two_points(qapp, kind, size) -> None:
    options: dict[str, object] = {}
    values = (300.0, 800.0)
    if kind == "kpoint":
        values = (12.0, 288.0)
        options = {
            "sweep_kind": "kpoint",
            "point_grids": {12.0: (2, 2, 3), 288.0: (6, 6, 8)},
            "lattice_vectors": _COF2_LATTICE,
        }
    points = {
        values[0]: SweepPointRunState(
            status="completed", total_energy=_COF2_ENERGY_EV + 1.84, max_force=0.62, max_stress=2.1
        ),
        values[1]: SweepPointRunState(
            status="completed", total_energy=_COF2_ENERGY_EV, max_force=0.011, max_stress=0.05
        ),
    }
    dialog = ConvergenceExaminerDialog(points, num_atoms=6, **options)
    try:
        _run_examiner(dialog, qapp, size, 1.0, stress=True)
        _assert_tick_rule(dialog)
    finally:
        dialog.close()


def test_convergence_examiner_drops_only_the_negative_threshold_label_when_crowded(qapp) -> None:
    dialog = _cof2_examiner("kpoint")
    try:
        _run_examiner(dialog, qapp, (880, 646), 2.0, energy=1e-6)
        dialog._canvas.draw()
        axes = dialog._energy_ax
        ticks = list(axes.get_yticks())
        labels = {
            round(label.get_position()[1], 12): label.get_text() for label in axes.get_yticklabels()
        }
        threshold = _threshold_of(axes)
        # The negative threshold keeps its tick (and the band's edge line) either way.
        assert any(math.isclose(tick, -threshold) for tick in ticks)
        assert labels[0.0] == "0"
        assert labels[round(threshold, 12)] == "0.001"
        _assert_tick_rule(dialog)
    finally:
        dialog.close()


def _marker_boxes(axes) -> list:
    """A box round every data marker and ring of ``axes``, in display coordinates."""

    boxes = []
    dpi = axes.figure.dpi
    for line in axes.get_lines():
        if line.get_marker() in (None, "None", "", " "):
            continue
        radius = line.get_markersize() * dpi / 72.0 / 2.0
        for x, y in zip(line.get_xdata(), line.get_ydata(), strict=True):
            if math.isfinite(x) and math.isfinite(y):
                cx, cy = axes.transData.transform((x, y))
                boxes.append(Bbox([[cx - radius, cy - radius], [cx + radius, cy + radius]]))
    return boxes


@pytest.mark.parametrize("size", [(880, 646), (980, 760), (1080, 900)])
@pytest.mark.parametrize("energy_threshold", [1.0, 0.002, 0.001])
def test_convergence_examiner_limit_label_covers_nothing(qapp, size, energy_threshold) -> None:
    dialog = _cof2_examiner("cutoff")
    try:
        _run_examiner(dialog, qapp, size, 1.0, energy=energy_threshold)
        dialog._canvas.draw()
        renderer = dialog._canvas.get_renderer()
        (label,) = [text for text in dialog._energy_ax.texts if text.get_text() == "E∞"]
        if not label.get_visible():
            return
        box = label.get_window_extent(renderer)
        for axes in dialog._figure.axes:
            for marker in _marker_boxes(axes):
                assert not box.overlaps(marker), (size, marker)
            for tick in [*axes.get_yticklabels(), *axes.get_xticklabels()]:
                if tick.get_text():
                    assert not box.overlaps(tick.get_window_extent(renderer)), tick
            legend = axes.get_legend()
            if legend is not None:
                assert not box.overlaps(legend.get_window_extent(renderer))
        assert dialog._energy_ax.bbox.contains(*box.p0) and dialog._energy_ax.bbox.contains(*box.p1)
    finally:
        dialog.close()


def test_convergence_examiner_force_legend_sits_right_of_the_stress_axis(qapp) -> None:
    dialog = _cof2_examiner("cutoff", stress=True)
    try:
        dialog.show()
        dialog.resize(880, 646)
        _settle(qapp)
        dialog._include_stress.setChecked(True)
        dialog._run_button.click()
        _settle(qapp)
        dialog._canvas.draw()
        renderer = dialog._canvas.get_renderer()

        legend = dialog._force_ax.get_legend()
        stress_axis = dialog._stress_ax.yaxis.get_tightbbox(renderer)
        assert legend.get_window_extent(renderer).x0 > stress_axis.x1
        labels = [text.get_text() for text in legend.get_texts()]
        assert labels == [
            "Thresholds ±0.05 eV/Ang,\n±0.1 GPa, log scale beyond",
            "Max force",
            "Force converged at 600 eV",
            "Max stress",
            "Stress converged at 600 eV",
        ]
    finally:
        dialog.close()


def _pop_out_examiner(kind: str, state: str) -> ConvergenceExaminerDialog:
    finished = {
        4.0: (-10.0, 0.30, 0.9),
        32.0: (-10.01, 0.20, 0.5),
        108.0: (-10.0105, 0.19, 0.45),
    }
    if state == "no points":
        points = {4.0: SweepPointRunState(status="failed")}
    elif state == "one point":
        points = {4.0: _completed_state(-10.0, 0.30)}
    else:
        points = {
            value: SweepPointRunState(
                status="completed", total_energy=energy, max_force=force, max_stress=stress
            )
            for value, (energy, force, stress) in finished.items()
        }
    options: dict[str, object] = {}
    if kind == "kpoint":
        options = {
            "sweep_kind": "kpoint",
            "point_grids": {4.0: (1, 1, 4), 32.0: (2, 2, 8), 108.0: (3, 3, 12)},
            "lattice_vectors": ((4.0, 0.0, 0.0), (0.0, 4.0, 0.0), (0.0, 0.0, 4.0)),
        }
    return ConvergenceExaminerDialog(points, num_atoms=2, **options)


@pytest.mark.parametrize("size", [(1080, 900), (880, 646)], ids=["default", "minimum"])
@pytest.mark.parametrize("kind", ["cutoff", "kpoint"])
@pytest.mark.parametrize("state", ["no points", "one point", "open", "analysed", "stress"])
def test_convergence_examiner_keeps_the_plot_clear_of_the_pop_out_button(
    qapp, size, kind, state
) -> None:
    dialog = _pop_out_examiner(kind, state)
    try:
        dialog.show()
        dialog.resize(*size)
        _settle(qapp)
        if state == "stress":
            dialog._include_stress.setChecked(True)
        if state in {"analysed", "stress"}:
            dialog._run_button.click()
        _settle(qapp)
        dialog._canvas.draw()

        button = dialog._canvas.parentWidget().pop_out_button
        corner = _bounds_in_dialog(dialog, button).translated(
            -_bounds_in_dialog(dialog, dialog._canvas).topLeft()
        )
        ratio = dialog._canvas.device_pixel_ratio
        height = dialog._canvas.height()
        renderer = dialog._canvas.get_renderer()
        artists = []
        for axes in dialog._figure.axes:
            artists += [axes, axes.title, axes.xaxis.label, axes.yaxis.label]
            if axes.get_legend() is not None:
                artists.append(axes.get_legend())
        for artist in artists:
            extent = artist.get_window_extent(renderer)
            if not extent.width or not extent.height:
                continue
            drawn = QRectF(
                extent.x0 / ratio,
                height - extent.y1 / ratio,
                extent.width / ratio,
                extent.height / ratio,
            )
            assert not drawn.intersects(QRectF(corner)), artist
    finally:
        dialog.close()


def test_convergence_examiner_gives_extra_height_to_the_plot_alone(qapp) -> None:
    dialog = _analysed_dialog(qapp, (880, 646))
    try:
        results_height = dialog._results.height()
        canvas_height = dialog._canvas.height()
        dialog.resize(880, 846)
        _settle(qapp)
        assert dialog._results.height() == results_height
        assert dialog._canvas.height() == canvas_height + 200
    finally:
        dialog.close()


def test_convergence_examiner_results_are_a_short_block_led_by_the_answer(qapp) -> None:
    dialog = _analysed_dialog(qapp, (1080, 900))
    try:
        assert dialog._results.height() == 120
        assert dialog._results.toPlainText().lstrip().startswith("Recommended cutoff\n")
    finally:
        dialog.close()


def test_convergence_examiner_shows_a_restored_recommendation_first(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {300.0: _completed_state(-10.0, 0.30), 400.0: _completed_state(-10.01, 0.20)},
        num_atoms=2,
    )
    state = dialog.state_snapshot()
    state["results_rows"] = [
        [["Thresholds", ["energy 0.001 eV/atom"]]],
        [["Recommended cutoff", ["400 eV"]]],
    ]
    restored = ConvergenceExaminerDialog(
        {300.0: _completed_state(-10.0, 0.30), 400.0: _completed_state(-10.01, 0.20)},
        num_atoms=2,
        initial_state=state,
    )
    assert restored._results.toPlainText().lstrip().startswith("Recommended cutoff\n400 eV")


def test_convergence_examiner_thresholds_share_one_compact_row(qapp) -> None:
    dialog = _analysed_dialog(qapp, (1080, 900))
    try:
        fields = (dialog._energy_threshold, dialog._force_threshold, dialog._stress_threshold)
        centres = {_bounds_in_dialog(dialog, field).center().y() for field in fields}
        assert len(centres) == 1
        stress_centre = _bounds_in_dialog(dialog, dialog._include_stress).center().y()
        assert abs(stress_centre - centres.pop()) <= 1
        for field in fields:
            assert field.width() <= 120
            assert field.fontMetrics().horizontalAdvance("0.001") < field.width() - 16
        assert dialog._energy_threshold.accessibleName() == "Energy threshold (eV/atom)"
        assert dialog._force_threshold.accessibleName() == "Force threshold (eV/Ang)"
        assert dialog._stress_threshold.accessibleName() == "Stress threshold (GPa)"
        assert dialog._include_stress.accessibleName() == "Use a stress threshold"
    finally:
        dialog.close()


def test_convergence_examiner_actions_sit_beside_the_thresholds_when_they_fit(qapp) -> None:
    dialog = _kpoint_dialog()
    try:
        dialog.show()
        dialog.resize(1600, 900)
        _settle(qapp)
        dialog._run_button.click()
        _settle(qapp)
        fields = _bounds_in_dialog(dialog, dialog._stress_threshold)
        buttons = _action_buttons(dialog)
        assert all(button.left() > fields.right() for button in buttons)
        assert all(button.top() <= fields.bottom() for button in buttons)

        # At the minimum the value-bearing adopt text does not fit: the actions
        # wrap under the thresholds, still side by side and nothing clipped.
        dialog.resize(880, 646)
        _settle(qapp)
        fields = _bounds_in_dialog(dialog, dialog._stress_threshold)
        buttons = _action_buttons(dialog)
        assert all(button.top() > fields.bottom() for button in buttons)
        assert buttons[0].right() < buttons[1].left() < buttons[1].right() < buttons[2].left()
        assert all(dialog.rect().contains(button) for button in buttons)
    finally:
        dialog.close()


def test_convergence_examiner_wraps_the_actions_when_the_adopt_text_grows(qapp) -> None:
    dialog = _kpoint_dialog()
    try:
        dialog.show()
        _settle(qapp)
        # Exactly wide enough for the actions while the adopt text is neutral.
        dialog.resize(dialog._header_width_needed(), 900)
        _settle(qapp)
        fields = _bounds_in_dialog(dialog, dialog._stress_threshold)
        assert all(button.left() > fields.right() for button in _action_buttons(dialog))

        dialog._run_button.click()
        _settle(qapp)
        assert dialog._adopt_button.text() == "Set 2 x 2 x 8 as Project Default K-Point Grid"
        fields = _bounds_in_dialog(dialog, dialog._stress_threshold)
        assert all(button.top() > fields.bottom() for button in _action_buttons(dialog))
    finally:
        dialog.close()


def test_convergence_examiner_close_button_sits_beside_adopt_and_closes(qapp) -> None:
    dialog = _analysed_dialog(qapp, (1080, 900))
    try:
        assert dialog._close_button.text() == "Close"
        assert dialog._adopt_button.text() == "Set 400 eV as Project Default Cutoff"
        adopt = _bounds_in_dialog(dialog, dialog._adopt_button)
        close = _bounds_in_dialog(dialog, dialog._close_button)
        assert close.left() > adopt.right()
        assert close.center().y() == adopt.center().y()

        dialog._close_button.click()
        _settle(qapp)
        assert dialog.isVisible() is False
    finally:
        dialog.close()


def test_convergence_examiner_return_analyses_in_a_field_and_closes_on_close(qapp) -> None:
    dialog = _analysed_dialog(qapp, (1080, 900))
    try:
        dialog._results.clear()
        QTest.keyClick(dialog._energy_threshold, Qt.Key.Key_Return)
        _settle(qapp)
        assert dialog.isVisible() is True
        assert "Recommended cutoff" in dialog._results.toPlainText()

        dialog._close_button.setFocus()
        _settle(qapp)
        QTest.keyClick(dialog._close_button, Qt.Key.Key_Return)
        _settle(qapp)
        assert dialog.isVisible() is False
    finally:
        dialog.close()


def test_convergence_examiner_shows_a_long_threshold_from_its_start(qapp) -> None:
    points = {300.0: _completed_state(-10.0, 0.30), 400.0: _completed_state(-10.01, 0.20)}
    state = ConvergenceExaminerDialog(points, num_atoms=2).state_snapshot()
    state["energy_threshold"] = 0.000001234567
    dialog = ConvergenceExaminerDialog(points, num_atoms=2, initial_state=state)
    assert dialog._energy_threshold.text() == "0.000001234567"
    assert dialog._energy_threshold.cursorPosition() == 0

    dialog._energy_threshold.setCursorPosition(14)
    dialog._energy_threshold.editingFinished.emit()
    assert dialog._energy_threshold.cursorPosition() == 0


def test_convergence_examiner_leads_with_the_recommendation_and_its_caveats(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.30),
            400.0: _completed_state(-10.01, 0.20),
            500.0: _completed_state(-10.0105, 0.19),
        },
        num_atoms=2,
        notes=("A grid was left out.",),
    )
    dialog._run_analysis()
    labels = [group[0][0] for group in dialog._result_groups]
    assert labels[:3] == ["Recommended cutoff", "Sweep length", "Notes"]
    assert labels[3] == "Thresholds"


def test_convergence_examiner_tab_order_follows_the_layout(qapp) -> None:
    dialog = _analysed_dialog(qapp, (1080, 900))
    try:
        order = [
            dialog._energy_threshold,
            dialog._force_threshold,
            dialog._include_stress,
            dialog._stress_threshold,
            dialog._run_button,
            dialog._adopt_button,
            dialog._close_button,
            dialog._results,
        ]
        for before, after in itertools.pairwise(order):
            assert before.nextInFocusChain() is after, (before, after)
    finally:
        dialog.close()


def test_convergence_examiner_has_no_method_picker(qapp) -> None:
    for kind in ("cutoff", "kpoint"):
        dialog = ConvergenceExaminerDialog(
            {
                8.0: _completed_state(-10.0, 0.3),
                27.0: _completed_state(-10.01, 0.2),
            },
            num_atoms=2,
            sweep_kind=kind,
        )

        assert dialog.findChildren(QComboBox) == []
        assert "Info" not in [button.text() for button in dialog.findChildren(QPushButton)]


def test_convergence_examiner_runs_analysis_and_enables_adopt(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.30),
            400.0: _completed_state(-10.01, 0.20),
            500.0: _completed_state(-10.0105, 0.19),
        },
        num_atoms=2,
    )

    dialog._run_analysis()

    assert "Recommended cutoff" in dialog._results.toPlainText()
    assert dialog._adopt_button.isEnabled() is True


def test_convergence_examiner_adopts_the_recommended_cutoff(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.30),
            400.0: _completed_state(-10.01, 0.20),
            500.0: _completed_state(-10.0105, 0.19),
        },
        num_atoms=2,
    )
    dialog._run_analysis()

    adopted: list[tuple[float, str, float, float, object]] = []
    dialog.default_adopted.connect(
        lambda value, method, energy_threshold, force_threshold, stress_threshold: adopted.append(
            (value, method, energy_threshold, force_threshold, stress_threshold)
        )
    )

    dialog._adopt_button.click()

    assert adopted
    # 400 eV is within 0.00025 eV/atom and 0.01 eV/Ang of 500 eV.
    assert adopted[-1][0] == pytest.approx(400.0)
    assert adopted[-1][1] == "converged"
    assert adopted[-1][2] == pytest.approx(0.001)
    assert adopted[-1][3] == pytest.approx(0.05)
    assert adopted[-1][4] is None


def test_convergence_examiner_restores_persisted_results_and_plot(qapp) -> None:
    first = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.30),
            400.0: _completed_state(-10.01, 0.20),
            500.0: _completed_state(-10.0105, 0.19),
        },
        num_atoms=2,
    )
    first._run_analysis()
    snapshot = to_jsonable(first.state_snapshot())

    second = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.30),
            400.0: _completed_state(-10.01, 0.20),
            500.0: _completed_state(-10.0105, 0.19),
        },
        num_atoms=2,
        initial_state=snapshot,
    )

    assert second._results.toPlainText() == first._results.toPlainText()
    assert "Recommended cutoff\n400 eV" in second._results.toPlainText()
    assert second._adopt_button.isEnabled() is True
    assert second._recommended_value == pytest.approx(first._recommended_value)


def test_convergence_examiner_discards_an_analysis_saved_by_an_earlier_version(qapp) -> None:
    states: list[dict[str, object]] = []
    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.30),
            400.0: _completed_state(-10.01, 0.20),
            500.0: _completed_state(-10.0105, 0.19),
        },
        num_atoms=2,
        initial_state={
            "energy_threshold": 0.002,
            "force_threshold": 0.07,
            "method": "asymptotic",
            "selected_method_for_recommendation": "asymptotic",
            "results_text": "Energy method: asymptotic\n\nRecommended cutoff: 500 eV",
            "recommended_value": 500.0,
            "energy_result": {"converged": True, "method": "asymptotic", "converged_value": 500.0},
            "force_result": {"converged": True, "method": "force-delta-threshold"},
            "plot_data": {
                "cutoffs": [300.0, 400.0, 500.0],
                "energies": [-10.0, -10.01, -10.0105],
                "forces": [0.30, 0.20, 0.19],
                "energy_threshold": 0.002,
                "force_threshold": 0.07,
            },
        },
    )
    dialog.state_changed.connect(states.append)

    text = dialog._results.toPlainText()
    assert "asymptotic" not in text
    assert "Completed points\n3" in text
    assert (
        "Earlier analysis\nset aside: it used an energy method this version no longer offers"
        "\n\nclick Analyse Convergence to judge the sweep again"
    ) in text
    assert dialog._recommended_value is None
    assert dialog._energy_result is None
    assert dialog._adopt_button.isEnabled() is False
    # The thresholds and the plotted points are kept.
    assert dialog._energy_threshold.value() == pytest.approx(0.002)
    assert dialog._force_threshold.value() == pytest.approx(0.07)
    assert dialog._last_plot_data["cutoffs"] == [300.0, 400.0, 500.0]

    dialog._run_analysis()

    assert "method" not in states[-1]
    assert "results_text" not in states[-1]
    assert "Earlier analysis" not in dialog._results.toPlainText()
    assert "selected_method_for_recommendation" not in states[-1]


def test_convergence_examiner_emits_state_changed_after_analysis(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.30),
            400.0: _completed_state(-10.01, 0.20),
            500.0: _completed_state(-10.0105, 0.19),
        },
        num_atoms=2,
    )

    states: list[dict[str, object]] = []
    dialog.state_changed.connect(lambda payload: states.append(payload))

    dialog._run_analysis()

    assert states
    assert states[-1].get("recommended_cutoff") == pytest.approx(400.0)


def test_convergence_examiner_kpoint_mode_emits_default(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            8.0: _completed_state(-10.0, 0.30),
            27.0: _completed_state(-10.01, 0.20),
            64.0: _completed_state(-10.0105, 0.19),
        },
        num_atoms=2,
        sweep_kind="kpoint",
        point_grids={8.0: (2, 2, 2), 27.0: (3, 3, 3), 64.0: (4, 4, 4)},
    )

    dialog._run_analysis()
    assert "Recommended k-point grid" in dialog._results.toPlainText()

    adopted: list[tuple[object, str, float, float, object]] = []
    dialog.default_adopted.connect(
        lambda key, method, energy_threshold, force_threshold, stress_threshold: adopted.append(
            (key, method, float(energy_threshold), float(force_threshold), stress_threshold)
        )
    )
    dialog._adopt_button.click()

    # The dialog emits the recommended point's grid, not its k-point count.
    assert adopted == [((3, 3, 3), "converged", pytest.approx(0.001), pytest.approx(0.05), None)]


def test_convergence_examiner_kpoint_adopt_needs_the_recommended_grid(qapp) -> None:
    sweep_data = {
        8.0: _completed_state(-10.0, 0.30),
        27.0: _completed_state(-10.01, 0.20),
        64.0: _completed_state(-10.0105, 0.19),
    }
    first = ConvergenceExaminerDialog(
        sweep_data,
        num_atoms=2,
        sweep_kind="kpoint",
        point_grids={8.0: (2, 2, 2), 27.0: (3, 3, 3), 64.0: (4, 4, 4)},
    )
    first._run_analysis()
    saved_state = first.state_snapshot()

    # Reopened on points whose recommended count (27) has no grid: there is
    # nothing to set, so the button stays disabled and nothing is emitted.
    dialog = ConvergenceExaminerDialog(
        sweep_data,
        num_atoms=2,
        sweep_kind="kpoint",
        point_grids={8.0: (2, 2, 2), 64.0: (4, 4, 4)},
        initial_state=saved_state,
    )
    adopted: list[tuple] = []
    dialog.default_adopted.connect(lambda *args: adopted.append(args))

    assert dialog._recommended_value == pytest.approx(27.0)
    assert dialog._adopt_button.isEnabled() is False
    assert dialog._adopt_button.text() == "Set 27 k-points as Project Default K-Point Grid"
    assert "Run the analysis again" in dialog._adopt_button.toolTip()
    dialog._emit_adopted_default()
    assert adopted == []


def test_convergence_examiner_kpoint_results_name_the_densest_grid_as_reference(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            8.0: _completed_state(-10.0, 0.30),
            27.0: _completed_state(-10.01, 0.20),
            64.0: _completed_state(-10.0105, 0.19),
        },
        num_atoms=2,
        sweep_kind="kpoint",
        point_grids={8.0: (2, 2, 2), 27.0: (3, 3, 3), 64.0: (4, 4, 4)},
    )

    dialog._run_analysis()

    text = dialog._results.toPlainText()
    assert (
        "Reference\n4 x 4 x 4, the densest grid in the sweep\n\n"
        "a sweep should extend beyond the grid it is meant to certify"
    ) in text
    assert "Energy\nconverged at 3 x 3 x 3\n" in text
    assert "within 0.00025 eV/atom of 4 x 4 x 4, as is every denser grid" in text
    assert "Recommended k-point grid\n3 x 3 x 3" in text
    assert "Sweep length" not in text
    assert "Grid:" not in text
    assert "Delta" not in text


def test_convergence_examiner_warns_when_the_cutoff_sweep_is_too_short(qapp) -> None:
    # Per-atom distance from the limit decays as Ecut**-1.5, 3 meV/atom at 600 eV.
    def energy(cutoff: float) -> float:
        return -1000.0 + 0.003 * 6 * (cutoff / 600.0) ** -1.5

    dialog = ConvergenceExaminerDialog(
        {cutoff: _completed_state(energy(cutoff), 0.01) for cutoff in (300.0, 400.0, 500.0, 600.0)},
        num_atoms=6,
    )

    dialog._run_analysis()

    text = dialog._results.toPlainText()
    assert (
        "Sweep length\ntoo short\n\n600 eV is 0.003 eV/atom from the fitted limit; "
        "extend the sweep to 1300 eV and analyse again"
    ) in text
    assert "Recommended cutoff\n500 eV" in text
    # The warning does not block adoption.
    assert dialog._adopt_button.isEnabled() is True
    assert theme_module.current_palette().warning.lower() in dialog._results.toHtml().lower()
    assert dialog._force_ax.get_title() == "Max force relative to 600 eV"
    legend = [text.get_text() for text in dialog._energy_ax.get_legend().get_texts()]
    assert "Power-law fit" in legend
    assert "Fitted limit, 3 meV/atom\nbelow the fitted curve at 600 eV" in legend


def test_convergence_examiner_reports_when_the_limit_cannot_be_estimated(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.30),
            400.0: _completed_state(-10.01, 0.20),
            500.0: _completed_state(-10.0105, 0.19),
        },
        num_atoms=2,
    )

    dialog._run_analysis()

    assert (
        "Sweep length\nlimit not estimated\n\nthe fit needs at least four cutoffs"
        in dialog._results.toPlainText()
    )


def test_convergence_examiner_kpoint_mode_adds_spacing_axis_when_lattice_provided(qapp) -> None:
    from mufinder.science.kpoint_grids import spacing_from_grid

    lattice = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    grids = [(1, 1, 1), (2, 2, 2), (3, 3, 3)]
    counts = [float(grid[0] * grid[1] * grid[2]) for grid in grids]
    sweep_data = {
        counts[0]: _completed_state(-10.0, 0.30),
        counts[1]: _completed_state(-10.02, 0.18),
        counts[2]: _completed_state(-10.025, 0.17),
    }

    dialog = ConvergenceExaminerDialog(
        sweep_data,
        num_atoms=2,
        sweep_kind="kpoint",
        point_grids=dict(zip(counts, grids, strict=True)),
        lattice_vectors=lattice,
    )

    assert dialog._point_grids == {counts[i]: grids[i] for i in range(3)}
    for count, grid in zip(counts, grids, strict=True):
        if grid == (1, 1, 1):
            # Gamma-only grids are excluded from _point_spacings because their
            # spacing is a meaningless boundary value, not a unique mapping.
            assert count not in dialog._point_spacings
            continue
        expected = spacing_from_grid(lattice, grid)
        assert dialog._point_spacings[count] == pytest.approx(expected)

    dialog._run_analysis()

    spacing_axes = [
        ax for ax in dialog._figure.axes if ax.get_xlabel() == "K-point spacing (1/Ang)"
    ]
    assert len(spacing_axes) == 1
    spacing_ax = spacing_axes[0]

    # Ticks are no longer pinned to data points; they sit at "nice" spacing
    # values (e.g. 0.1, 0.07, 0.05) placed via continuous interpolation of
    # CASTEP's spacing -> grid formula. Verify the axis exists with sensible
    # ticks within the data range and that the implied spacings decrease
    # monotonically as n_kpoints grows.
    tick_positions = list(spacing_ax.get_xticks())
    tick_labels = [tick.get_text() for tick in spacing_ax.get_xticklabels()]
    assert tick_positions, "expected at least one spacing tick"
    assert all(label for label in tick_labels), "expected non-empty tick labels"
    x_min, x_max = spacing_ax.get_xlim()
    for position in tick_positions:
        assert x_min <= position <= x_max
    # Tick positions read left-to-right; spacings read large-to-small.
    parsed = [float(label) for label in tick_labels]
    assert parsed == sorted(parsed, reverse=True)

    # The bottom n_kpoints axis and the spacing twin both use log scale so
    # small grids stop collapsing into the leftmost sliver of the plot.
    energy_ax = next(ax for ax in dialog._figure.axes if ax.get_ylabel() == "ΔE (meV/atom)")
    assert energy_ax.get_xscale() == "log"
    assert spacing_ax.get_xscale() == "log"


def test_convergence_examiner_kpoint_mode_omits_spacing_axis_without_lattice(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            8.0: _completed_state(-10.0, 0.30),
            27.0: _completed_state(-10.01, 0.20),
            64.0: _completed_state(-10.011, 0.19),
        },
        num_atoms=2,
        sweep_kind="kpoint",
        point_grids={8.0: (2, 2, 2), 27.0: (3, 3, 3), 64.0: (4, 4, 4)},
    )
    assert dialog._point_spacings == {}

    dialog._run_analysis()

    spacing_axes = [
        ax for ax in dialog._figure.axes if ax.get_xlabel() == "K-point spacing (1/Ang)"
    ]
    assert spacing_axes == []
    # Log scale applies even without lattice vectors — it's the n_kpoints
    # crowding fix, independent of whether spacing labels are drawn.
    energy_ax = next(ax for ax in dialog._figure.axes if ax.get_ylabel() == "ΔE (meV/atom)")
    assert energy_ax.get_xscale() == "log"


def test_convergence_examiner_cutoff_mode_has_no_spacing_axis(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.30),
            400.0: _completed_state(-10.01, 0.20),
            500.0: _completed_state(-10.011, 0.19),
        },
        num_atoms=2,
        sweep_kind="cutoff",
        lattice_vectors=[[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]],
    )

    dialog._run_analysis()

    spacing_axes = [
        ax for ax in dialog._figure.axes if ax.get_xlabel() == "K-point spacing (1/Ang)"
    ]
    assert spacing_axes == []
    assert dialog._point_spacings == {}
    # Cutoff sweeps keep the historical linear x-axis.
    energy_ax = next(ax for ax in dialog._figure.axes if ax.get_ylabel() == "ΔE (meV/atom)")
    assert energy_ax.get_xscale() == "linear"


def test_convergence_examiner_rejects_invalid_sweep_kind(qapp) -> None:
    with pytest.raises(ValueError, match="sweep_kind"):
        ConvergenceExaminerDialog({}, num_atoms=1, sweep_kind="invalid")


def test_convergence_examiner_handles_invalid_restore_payloads(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.30),
            400.0: _completed_state(-10.01, 0.20),
        },
        num_atoms=2,
        initial_state={
            "energy_threshold": "bad",
            "force_threshold": object(),
            "recommended_value": "bad",
            "energy_result": {
                "converged": True,
                "method": "delta-threshold",
                "details": "ok",
                "deltas": "bad",
            },
            "force_result": {
                "converged": True,
                "method": "delta-threshold",
                "details": "ok",
                "fitted_curve": "bad",
            },
            "plot_data": {"cutoffs": [300.0], "energies": [1.0, 2.0], "forces": [0.1]},
        },
    )

    assert dialog._recommended_value is None
    assert dialog._adopt_button.isEnabled() is False


def test_convergence_examiner_analysis_warns_when_backend_raises(
    qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.30),
            400.0: _completed_state(-10.01, 0.20),
        },
        num_atoms=2,
    )

    warnings: list[str] = []
    monkeypatch.setattr(
        convergence_examiner_module,
        "analyze_convergence",
        lambda **_kwargs: (_ for _ in ()).throw(ValueError("boom")),
    )
    monkeypatch.setattr(
        convergence_examiner_module.QMessageBox,
        "warning",
        lambda _parent, title, _message: warnings.append(title),
    )

    dialog._run_analysis()

    assert warnings == ["Convergence analysis failed"]


def test_convergence_examiner_requires_matching_energy_and_force_rows(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            300.0: convergence_examiner_module.SweepPointRunState(
                status="completed", total_energy=-10.0, max_force=None
            ),
            400.0: convergence_examiner_module.SweepPointRunState(
                status="completed", total_energy=-10.02, max_force=None
            ),
            500.0: convergence_examiner_module.SweepPointRunState(
                status="completed", total_energy=None, max_force=0.20
            ),
            600.0: convergence_examiner_module.SweepPointRunState(
                status="completed", total_energy=None, max_force=0.18
            ),
        },
        num_atoms=2,
    )

    dialog._run_analysis()

    assert "Need at least two matching completed rows" in dialog._results.toPlainText()
    assert dialog._adopt_button.isEnabled() is False


def test_convergence_examiner_hover_routes_twin_axes_to_logical_subplot(qapp) -> None:
    # Regression: with a spacing twiny on the energy subplot and a stress
    # twinx on the force subplot, hovering on either subplot used to dispatch
    # to whichever twin captured the event (e.g. stress would always fire on
    # the force region). Twin-axis hovers should now route to the underlying
    # subplot and show the matching tooltip.
    lattice = [[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 0.0, 5.0]]
    sweep_data = {
        1.0: SweepPointRunState(
            status="completed", total_energy=-10.0, max_force=0.30, max_stress=2.0
        ),
        8.0: SweepPointRunState(
            status="completed", total_energy=-10.02, max_force=0.18, max_stress=0.50
        ),
        27.0: SweepPointRunState(
            status="completed", total_energy=-10.025, max_force=0.17, max_stress=0.05
        ),
    }
    dialog = ConvergenceExaminerDialog(
        sweep_data,
        num_atoms=2,
        sweep_kind="kpoint",
        point_grids={1.0: (1, 1, 1), 8.0: (2, 2, 2), 27.0: (3, 3, 3)},
        lattice_vectors=lattice,
    )
    dialog._include_stress.setChecked(True)
    dialog._run_analysis()

    assert dialog._spacing_ax is not None
    assert dialog._stress_ax is not None

    # Hover on the spacing twin (top of the energy subplot) — should still
    # report an energy-subplot tooltip anchored on the energy axis.
    event = _move_mouse(dialog, dialog._energy_ax, 8.0)
    assert event.inaxes is dialog._spacing_ax
    assert dialog._hover_annotation is not None
    assert "Grid: 2 x 2 x 2" in dialog._hover_annotation.get_text()
    assert dialog._hover_annotation.axes is dialog._energy_ax

    # Hover on the stress twin (top of the force subplot) — should still
    # report a force-subplot tooltip anchored on the force axis (kpoint
    # mode shows just the grid label).
    event = _move_mouse(dialog, dialog._force_ax, 27.0)
    assert event.inaxes is dialog._stress_ax
    assert "Grid: 3 x 3 x 3" in dialog._hover_annotation.get_text()
    assert dialog._hover_annotation.axes is dialog._force_ax


def test_convergence_examiner_hover_combines_force_and_stress_in_cutoff_mode(qapp) -> None:
    # When the cursor is over the stress twin axis in cutoff mode, the
    # annotation should be anchored on the stress axis (so the arrow points
    # at the stress data point, not the force line) and show both metrics.
    sweep_data = {
        300.0: SweepPointRunState(
            status="completed", total_energy=-10.0, max_force=0.30, max_stress=1.5
        ),
        400.0: SweepPointRunState(
            status="completed", total_energy=-10.01, max_force=0.20, max_stress=0.4
        ),
        500.0: SweepPointRunState(
            status="completed", total_energy=-10.011, max_force=0.05, max_stress=0.08
        ),
    }
    dialog = ConvergenceExaminerDialog(sweep_data, num_atoms=2, sweep_kind="cutoff")
    dialog._include_stress.setChecked(True)
    dialog._run_analysis()

    assert dialog._stress_ax is not None
    event = _move_mouse(dialog, dialog._force_ax, 400.0)
    assert event.inaxes is dialog._stress_ax
    text = dialog._hover_annotation.get_text()
    assert "Cutoff: 400" in text
    assert "Max force" in text
    assert "Max stress" in text
    # Anchored on stress axis so the arrow points at the stress data point.
    assert dialog._hover_annotation.axes is dialog._stress_ax


def test_convergence_examiner_hover_tooltip_and_leave(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.30),
            400.0: _completed_state(-10.01, 0.20),
            500.0: _completed_state(-10.0105, 0.19),
        },
        num_atoms=2,
        sweep_kind="kpoint",
        point_grids={300.0: (3, 10, 10)},
    )
    dialog._run_analysis()

    energy_ax, force_ax = dialog._figure.axes
    _move_mouse(dialog, energy_ax, 300.0)
    assert dialog._hover_annotation is not None
    assert "Grid:" in dialog._hover_annotation.get_text()
    assert "Energy:" not in dialog._hover_annotation.get_text()

    _move_mouse(dialog, force_ax, 300.0)
    assert "Grid:" in dialog._hover_annotation.get_text()
    assert "Max force" not in dialog._hover_annotation.get_text()

    # Halfway between the 300 and 400 points on the log axis is far from both.
    _move_mouse(dialog, force_ax, (300.0 * 400.0) ** 0.5)
    assert dialog._hover_annotation.get_visible() is False


def test_convergence_examiner_plot_colors_follow_active_theme(qapp) -> None:
    theme_module.apply_theme(qapp, theme_module.THEME_DARK)
    try:
        palette = theme_module.current_palette()

        dialog = ConvergenceExaminerDialog(
            {
                300.0: _completed_state(-10.0, 0.30),
                400.0: _completed_state(-10.01, 0.20),
                500.0: _completed_state(-10.0105, 0.19),
            },
            num_atoms=2,
        )
        dialog._run_analysis()

        energy_ax, force_ax = dialog._figure.axes
        assert to_hex(dialog._figure.get_facecolor()).lower() == palette.bg_surface.lower()
        assert to_hex(energy_ax.get_facecolor()).lower() == palette.bg_surface_alt.lower()
        assert energy_ax.yaxis.label.get_color().lower() == palette.text_primary.lower()
        assert force_ax.xaxis.label.get_color().lower() == palette.text_primary.lower()

        energy_legend = energy_ax.get_legend()
        assert energy_legend is not None
        assert energy_legend.get_texts()[0].get_color().lower() == palette.text_primary.lower()
    finally:
        theme_module.apply_theme(qapp, theme_module.THEME_LIGHT)


def test_convergence_examiner_emit_adopted_uses_current_thresholds_when_not_cached(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.30),
            400.0: _completed_state(-10.01, 0.20),
        },
        num_atoms=2,
    )
    dialog._recommended_value = 400.0
    dialog._energy_threshold_for_recommendation = None
    dialog._force_threshold_for_recommendation = None
    dialog._energy_threshold.setValue(0.002)
    dialog._force_threshold.setValue(0.07)

    defaults: list[tuple] = []
    dialog.default_adopted.connect(
        lambda value, method, et, ft, st: defaults.append((value, method, et, ft, st))
    )

    dialog._offer_adoption()
    dialog._adopt_button.click()

    # One signal carries the adoption.
    assert defaults == [(400.0, "converged", pytest.approx(0.002), pytest.approx(0.07), None)]


def _completed_state_with_stress(
    energy: float, force: float, stress: float | None
) -> SweepPointRunState:
    return SweepPointRunState(
        status="completed", total_energy=energy, max_force=force, max_stress=stress
    )


def test_convergence_examiner_stress_tickbox_disabled_when_any_row_lacks_stress(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state_with_stress(-10.0, 0.30, 1.5),
            400.0: _completed_state_with_stress(-10.01, 0.20, None),
            500.0: _completed_state_with_stress(-10.0105, 0.19, 0.4),
        },
        num_atoms=2,
    )

    assert dialog._include_stress.isEnabled() is False
    assert dialog._stress_threshold.isEnabled() is False
    assert "CALCULATE_STRESS" in dialog._include_stress.toolTip()


def test_convergence_examiner_stress_tickbox_enables_when_all_rows_have_stress(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state_with_stress(-10.0, 0.30, 1.5),
            400.0: _completed_state_with_stress(-10.01, 0.20, 0.5),
            500.0: _completed_state_with_stress(-10.0105, 0.19, 0.42),
        },
        num_atoms=2,
    )

    assert dialog._include_stress.isEnabled() is True
    assert dialog._stress_threshold.isEnabled() is False  # off until ticked
    dialog._include_stress.setChecked(True)
    assert dialog._stress_threshold.isEnabled() is True


def test_convergence_examiner_runs_stress_and_passes_through_signal(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state_with_stress(-10.0, 0.30, 1.5),
            400.0: _completed_state_with_stress(-10.01, 0.20, 0.5),
            500.0: _completed_state_with_stress(-10.0105, 0.19, 0.42),
        },
        num_atoms=2,
    )
    dialog._include_stress.setChecked(True)
    dialog._stress_threshold.setValue(0.1)

    dialog._run_analysis()

    assert dialog._stress_result is not None
    assert "Stress\nconverged at 400 eV" in dialog._results.toPlainText()
    assert "stress 0.1 GPa" in dialog._results.toPlainText()
    assert dialog._force_ax.get_title() == "Max force and stress relative to 500 eV"

    adopted: list[tuple] = []
    dialog.default_adopted.connect(lambda *args: adopted.append(args))
    dialog._adopt_button.click()

    assert adopted
    assert adopted[-1][-1] == pytest.approx(0.1)


def test_convergence_examiner_withholds_recommendation_when_force_has_not_converged(qapp) -> None:
    """A converged energy alone must not be offered for adoption."""

    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.90),
            400.0: _completed_state(-10.0001, 0.60),
            500.0: _completed_state(-10.0002, 0.30),
        },
        num_atoms=2,
    )

    dialog._run_analysis()

    text = dialog._results.toPlainText()
    assert "Force\nnot converged within the sweep" in text
    assert (
        "Recommended cutoff\nnone\n\nthe force criterion did not converge within the sweep"
    ) in text
    assert dialog._adopt_button.isEnabled() is False


def test_convergence_examiner_kpoint_adopt_label_names_the_grid_once(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            8.0: _completed_state(-10.0, 0.30),
            27.0: _completed_state(-10.01, 0.20),
            64.0: _completed_state(-10.0105, 0.19),
        },
        num_atoms=2,
        sweep_kind="kpoint",
        point_grids={8.0: (2, 2, 2), 27.0: (3, 3, 3), 64.0: (4, 4, 4)},
        notes=("2 x 4 x 1 left out: same k-point count (8) as 2 x 2 x 2.",),
    )

    assert "Notes\n2 x 4 x 1 left out" in dialog._results.toPlainText()
    dialog._run_analysis()

    text = dialog._adopt_button.text()
    assert text.startswith("Set ") and text.endswith(" as Project Default K-Point Grid")
    assert "Grid:" not in text
    results = dialog._results.toPlainText()
    assert "Grid:" not in results
    assert "2 x 4 x 1 left out" in results


def _six_grid_kpoint_sweep() -> ConvergenceExaminerDialog:
    counts = [1.0, 8.0, 27.0, 64.0, 125.0, 216.0]
    return ConvergenceExaminerDialog(
        {
            count: _completed_state(-10.0 - 0.01 * (1.0 - 1.0 / count), 0.3 / count)
            for count in counts
        },
        num_atoms=2,
        sweep_kind="kpoint",
        point_grids={count: (round(count ** (1.0 / 3.0)),) * 3 for count in counts},
    )


def test_convergence_examiner_hover_measures_the_log_kpoint_axis_on_screen(qapp) -> None:
    """The note names the point nearest the cursor on screen, at either end of a log axis."""

    dialog = _six_grid_kpoint_sweep()
    dialog.resize(900, 700)
    dialog.show()
    dialog._run_analysis()
    qapp.processEvents()
    dialog._canvas.draw()
    axis = dialog._energy_ax
    assert axis.get_xscale() == "log"
    ratio = dialog._canvas.device_pixel_ratio

    def pixel(count: float) -> float:
        """Return the widget's x pixel of the k-point count ``count``."""

        return float(axis.transData.transform((count, 0.0))[0]) / ratio

    def hover_at(x: float) -> str | None:
        """Move the mouse over the middle of the energy panel at widget pixel ``x``."""

        y = dialog._canvas.height() - axis.bbox.y0 / ratio - axis.bbox.height / ratio / 2
        QTest.mouseMove(dialog._canvas, QPoint(round(x), round(y)))
        qapp.processEvents()
        note = dialog._hover_annotation
        return note.get_text() if note is not None and note.get_visible() else None

    # Halfway between 1 and 8 on screen is 3 k-points from 1: far from both on
    # screen, so no point is named.
    halfway = (pixel(1.0) + pixel(8.0)) / 2
    assert halfway - pixel(1.0) > 50
    assert hover_at(halfway + 5) is None
    # Ten pixels from a point names it at either end of the axis, although 10 px
    # left of the 216 point is about 10 k-points from it.
    assert hover_at(pixel(1.0) + 10) == "Grid: 1 x 1 x 1"
    assert hover_at(pixel(8.0) - 10) == "Grid: 2 x 2 x 2"
    assert hover_at(pixel(216.0) - 10) == "Grid: 6 x 6 x 6"
    assert hover_at(pixel(125.0) + 10) == "Grid: 5 x 5 x 5"


def test_convergence_examiner_hover_note_keeps_its_place_beside_the_point(qapp) -> None:
    """A note with room sits 10 points up and right of its point, at either end of the axis."""

    dialog = _six_grid_kpoint_sweep()
    dialog.show()
    dialog._run_analysis()
    plotted = dialog._plotted["energy"]
    for count in (1.0, 216.0):
        _move_mouse(dialog, dialog._energy_ax, count)
        dialog._canvas.draw()
        note = dialog._hover_annotation
        assert note.get_visible()
        assert note.axes is dialog._energy_ax
        assert note.xy == (count, plotted[count])
        assert note.get_anncoords() == "offset points"
        assert tuple(note.get_position()) == HOVER_NOTE_OFFSET == (10, 10)
        assert note.get_horizontalalignment() == "left"


def _hover_point(dialog: ConvergenceExaminerDialog, axes, x: float, y: float) -> None:
    """Move the mouse onto the data point ``(x, y)`` of ``axes``, as a real mouse move does."""

    dialog._canvas.draw()
    pixel_x, pixel_y = axes.transData.transform((x, y))
    MouseEvent("motion_notify_event", dialog._canvas, pixel_x, pixel_y)._process()


def _series_points(axes) -> list[tuple[object, int, int, float, float]]:
    """The line, index, point count and data point of every marked series point of ``axes``."""

    points = []
    for line in axes.get_lines():
        if line.get_marker() in (None, "None", "", " ") or len(line.get_xdata()) < 2:
            continue
        xs, ys = line.get_xdata(), line.get_ydata()
        points += [
            (line, index, len(xs), float(xs[index]), float(ys[index])) for index in range(len(xs))
        ]
    return points


def _assert_hover_notes_whole_beside_their_points(dialog: ConvergenceExaminerDialog) -> int:
    """Hover the first, middle and last point of every panel; check each note. Returns the count."""

    figure = dialog._figure
    checked = 0
    for axes in (dialog._energy_ax, dialog._force_ax, dialog._stress_ax):
        if axes is None:
            continue
        for line, index, count, x, y in _series_points(axes):
            if index not in {0, count // 2, count - 1}:
                continue
            _hover_point(dialog, axes, x, y)
            note = dialog._hover_annotation
            assert note is not None and note.get_visible(), (axes, index)
            dialog._canvas.draw()
            renderer = dialog._canvas.get_renderer()
            box = hover_note_box(note, renderer)
            point = note.axes.transData.transform(note.xy)
            # Whole inside the figure, drawn above everything (legends included).
            assert box_inside(box, figure.bbox), (note.get_text(), box, figure.bbox)
            assert note in figure.artists and not note.get_in_layout()
            assert note.get_zorder() > max(child.get_zorder() for child in figure.axes)
            # Clear of its own marker, and beside it.
            radius = line.get_markersize() * figure.dpi / 72 / 2
            assert distance_to_box(box, *point) > radius, (note.get_text(), box, point)
            assert distance_to_box(box, *point) <= math.hypot(*HOVER_NOTE_OFFSET) * figure.dpi / 72
            checked += 1
    return checked


@pytest.mark.parametrize("size", [(1080, 900), (880, 646)], ids=["default", "minimum"])
@pytest.mark.parametrize(
    ("kind", "stress"),
    [("cutoff", False), ("cutoff", True), ("kpoint", False)],
    ids=["cutoff", "cutoff-stress", "kpoint"],
)
def test_convergence_examiner_hover_note_stays_whole_beside_its_point(
    qapp, size, kind, stress
) -> None:
    dialog = _cof2_examiner(kind, stress=stress)
    try:
        dialog.show()
        dialog.resize(*size)
        _settle(qapp)
        dialog._include_stress.setChecked(stress)
        dialog._run_button.click()
        _settle(qapp)

        assert _assert_hover_notes_whole_beside_their_points(dialog) == (9 if stress else 6)
    finally:
        dialog.close()


@pytest.mark.parametrize("window_size", [(960, 680), (480, 320)], ids=["default", "smallest"])
def test_convergence_examiner_hover_note_stays_whole_in_the_pop_out(qapp, window_size) -> None:
    dialog = _cof2_examiner("cutoff", stress=True)
    try:
        dialog.show()
        _settle(qapp)
        dialog._include_stress.setChecked(True)
        dialog._run_button.click()
        _settle(qapp)
        frame = dialog._canvas.parentWidget()
        frame.pop_out()
        frame.popped_out_window.resize(*window_size)
        _settle(qapp)

        assert _assert_hover_notes_whole_beside_their_points(dialog) == 9
    finally:
        dialog.close()


@pytest.mark.parametrize(
    ("kind", "stress"),
    [("cutoff", True), ("cutoff", False), ("kpoint", False)],
    ids=["cutoff-stress", "cutoff", "kpoint"],
)
def test_convergence_examiner_pop_out_at_its_smallest_keeps_usable_panels(
    qapp, kind, stress
) -> None:
    # Below the smallest pop-out, the energy legend's overhang leaves no settled
    # layout and each redraw (a pointer move) shrinks the panels away.
    dialog = _cof2_examiner(kind, stress=stress)
    try:
        dialog.show()
        dialog.resize(880, 646)
        _settle(qapp)
        if stress:
            dialog._include_stress.setChecked(True)
        dialog._run_button.click()
        _settle(qapp)
        canvas = dialog._canvas
        frame = canvas.parentWidget()
        frame.pop_out()
        frame.popped_out_window.resize(100, 100)
        _settle(qapp)

        # The pop-out cannot be made shorter than its minimum.
        assert canvas.width() >= 480
        assert canvas.height() >= 366
        panels = (dialog._energy_ax, dialog._force_ax)
        heights = []
        for _ in range(50):
            canvas.draw()
            heights.append([axes.bbox.height for axes in panels])

        assert min(heights[-1]) >= 60
        for later in heights[-10:]:
            assert later == pytest.approx(heights[-1], abs=0.01)
    finally:
        dialog.close()


def test_convergence_examiner_at_its_minimum_size_keeps_usable_panels(
    qapp, designed_text_scale
) -> None:
    # With the app's theme, the docked plot at the dialog's minimum size gets at least the
    # height at which the layout settles; each redraw (a pointer move) leaves the panels be.
    dialog = _cof2_examiner("cutoff", stress=True)
    try:
        dialog.show()
        dialog.resize(100, 100)
        _settle(qapp)
        dialog._include_stress.setChecked(True)
        dialog._run_button.click()
        _settle(qapp)
        canvas = dialog._canvas
        assert canvas.height() >= 366
        assert canvas.geometry().bottom() < dialog.height()
        panels = (dialog._energy_ax, dialog._force_ax)
        heights = []
        for _ in range(200):
            canvas.draw()
            heights.append([axes.bbox.height for axes in panels])

        assert min(heights[-1]) >= 60
        for later in heights[-10:]:
            assert later == pytest.approx(heights[-1], abs=0.01)
        # The cutoff axis label stays on the canvas.
        label = dialog._force_ax.xaxis.label.get_window_extent(canvas.get_renderer())
        assert label.y0 >= 0
    finally:
        dialog.close()


def test_convergence_examiner_last_kpoint_note_is_drawn_over_the_legend(qapp, monkeypatch) -> None:
    """At the minimum size the finest grid's note reaches the legend and is drawn over it."""

    dialog = _cof2_examiner("kpoint")
    try:
        dialog.show()
        dialog.resize(880, 646)
        _settle(qapp)
        dialog._run_button.click()
        _settle(qapp)
        axes = dialog._energy_ax
        line, _index, _count, x, y = _series_points(axes)[-1]
        _hover_point(dialog, axes, x, y)
        note = dialog._hover_annotation
        legend = axes.get_legend()
        dialog._canvas.draw()
        renderer = dialog._canvas.get_renderer()
        assert hover_note_box(note, renderer).overlaps(legend.get_window_extent(renderer))

        drawn: list[object] = []
        for artist in (note, legend):
            original = artist.draw
            monkeypatch.setattr(
                artist,
                "draw",
                lambda renderer, _artist=artist, _draw=original: (
                    drawn.append(_artist),
                    _draw(renderer),
                ),
            )
        dialog._canvas.draw()

        assert drawn == [legend, note]
    finally:
        dialog.close()


@pytest.mark.parametrize(
    ("kind", "stress"),
    [("cutoff", False), ("cutoff", True), ("kpoint", False)],
    ids=["cutoff", "cutoff-stress", "kpoint"],
)
def test_convergence_examiner_hover_note_moves_no_panel(qapp, kind, stress) -> None:
    dialog = _cof2_examiner(kind, stress=stress)
    try:
        dialog.show()
        dialog.resize(1080, 900)
        _settle(qapp)
        dialog._include_stress.setChecked(stress)
        dialog._run_button.click()
        _settle(qapp)
        figure = dialog._figure
        for axes in (dialog._energy_ax, dialog._force_ax):
            _line, _index, _count, x, y = _series_points(axes)[-1]
            dialog._canvas.draw()
            before = panel_positions(figure)
            _hover_point(dialog, axes, x, y)
            dialog._canvas.draw()

            assert dialog._hover_annotation.get_visible()
            assert panel_positions(figure) == pytest.approx(before, abs=1e-12)
            dialog._on_plot_leave(None)
    finally:
        dialog.close()


def test_convergence_examiner_hover_survives_reanalysis(qapp) -> None:
    """Re-analysing clears the figure; the next hover must not touch the old annotation.

    ``Figure.clear`` detaches the cached hover annotation, and removing a
    detached artist raises ``NotImplementedError`` on every mouse move.
    """

    from matplotlib.backend_bases import MouseEvent

    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.30),
            400.0: _completed_state(-10.01, 0.20),
            500.0: _completed_state(-10.0105, 0.19),
        },
        num_atoms=2,
    )

    def hover() -> None:
        dialog._canvas.draw()
        # The force panel plots each force's distance from the finest point's.
        x, y = dialog._force_ax.transData.transform((400.0, 0.20 - 0.19))
        dialog._on_plot_hover(MouseEvent("motion_notify_event", dialog._canvas, x, y))

    dialog._run_analysis()
    hover()
    dialog._run_analysis()
    hover()

    annotation = dialog._hover_annotation
    assert annotation is not None
    assert annotation.axes is dialog._force_ax
    assert annotation.get_visible() is True


def test_convergence_examiner_accepts_tight_thresholds_as_typed(qapp) -> None:
    """Thresholds only need to be positive; there is no arbitrary floor."""

    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.30),
            400.0: _completed_state(-10.01, 0.20),
            500.0: _completed_state(-10.0105, 0.15),
        },
        num_atoms=2,
    )
    dialog._energy_threshold.setText("0.00001")
    dialog._force_threshold.setText("0.0001")

    dialog._run_analysis()

    assert dialog._energy_threshold.text() == "0.00001"
    assert dialog._force_threshold.text() == "0.0001"
    text = dialog._results.toPlainText()
    assert "Thresholds\nenergy 0.00001 eV/atom, force 0.0001 eV/Ang" in text


def test_convergence_examiner_floors_force_and_stress_at_castep_printed_precision(qapp) -> None:
    """CASTEP prints forces to 1e-5 eV/Ang and stresses to 1e-6 GPa.

    A finer threshold could only be met by values that print identically, so the
    fields stop at that precision and show the value the analysis used.
    """

    dialog = ConvergenceExaminerDialog(
        {
            300.0: SweepPointRunState(
                status="completed", total_energy=-10.0, max_force=0.30, max_stress=1.0
            ),
            400.0: SweepPointRunState(
                status="completed", total_energy=-10.01, max_force=0.20, max_stress=0.6
            ),
            500.0: SweepPointRunState(
                status="completed", total_energy=-10.0105, max_force=0.15, max_stress=0.4
            ),
        },
        num_atoms=2,
    )
    dialog._include_stress.setChecked(True)
    dialog._force_threshold.setText("0.000001")
    dialog._stress_threshold.setText("0.0000001")

    dialog._run_analysis()

    assert dialog._force_threshold.text() == "0.00001"
    assert dialog._stress_threshold.text() == "0.000001"
    text = dialog._results.toPlainText()
    assert "force 0.00001 eV/Ang, stress 0.000001 GPa" in text
    assert "0.00001 eV/Ang" in dialog._force_threshold.toolTip()


def test_convergence_examiner_raises_a_non_positive_threshold_to_the_smallest_positive_value(
    qapp,
) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.30),
            400.0: _completed_state(-10.01, 0.20),
        },
        num_atoms=2,
    )
    dialog._energy_threshold.setText("0")
    dialog._force_threshold.setText("0")

    dialog._run_analysis()

    assert dialog._energy_threshold.value() > 0.0
    assert dialog._force_threshold.text() == "0.00001"


def test_convergence_examiner_initial_plot_keeps_points_missing_a_force(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.30),
            400.0: SweepPointRunState(status="completed", total_energy=-10.01, max_force=None),
            500.0: _completed_state(-10.0105, 0.19),
        },
        num_atoms=2,
    )

    energy_line = dialog._energy_ax.lines[0]
    force_line = dialog._force_ax.lines[0]
    assert list(energy_line.get_xdata()) == [300.0, 400.0, 500.0]
    # Per atom from the finest point, in meV/atom.
    assert list(energy_line.get_ydata()) == pytest.approx([5.25, 0.25, 0.0])
    assert math.isnan(force_line.get_ydata()[1])


def test_convergence_examiner_saves_the_atom_count_and_restores_a_plot_without_it(qapp) -> None:
    points = {
        300.0: _completed_state(-10.0, 0.30),
        400.0: _completed_state(-10.01, 0.20),
        500.0: _completed_state(-10.0105, 0.19),
    }
    first = ConvergenceExaminerDialog(points, num_atoms=2)
    first._run_analysis()
    state = first.state_snapshot()
    assert state["plot_data"]["num_atoms"] == 2

    # A plot saved without the count is drawn per atom with the dialog's own count.
    del state["plot_data"]["num_atoms"]
    restored = ConvergenceExaminerDialog(points, num_atoms=2, initial_state=state)

    assert restored._energy_ax.get_ylabel() == "ΔE (meV/atom)"
    energy = next(line for line in restored._energy_ax.lines if line.get_label() == "Energy")
    assert list(energy.get_ydata()) == pytest.approx([5.25, 0.25, 0.0])
    assert restored.state_snapshot()["plot_data"]["num_atoms"] == 2


def test_convergence_examiner_hover_points_at_the_plotted_distance(qapp) -> None:
    from matplotlib.backend_bases import MouseEvent

    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.30),
            400.0: _completed_state(-10.01, 0.20),
            500.0: _completed_state(-10.0105, 0.19),
        },
        num_atoms=2,
    )
    dialog._run_analysis()
    dialog._canvas.draw()

    x, y = dialog._energy_ax.transData.transform((400.0, 0.25))
    dialog._on_plot_hover(MouseEvent("motion_notify_event", dialog._canvas, x, y))

    annotation = dialog._hover_annotation
    assert annotation is not None and annotation.get_visible()
    assert annotation.xy == pytest.approx((400.0, 0.25))
    assert "Energy: -10.01 eV" in annotation.get_text()
