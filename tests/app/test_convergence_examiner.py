"""Tests for cutoff convergence examiner dialog."""

from __future__ import annotations

import itertools
import math
from types import SimpleNamespace

from matplotlib.colors import to_hex
import pytest
from PySide6.QtCore import QPoint, QRect, QRectF, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QComboBox, QPushButton, QScrollArea, QWidget

import mufinder.app.convergence_examiner as convergence_examiner_module
from mufinder.app import theme as theme_module
from mufinder.app.convergence_examiner import ConvergenceExaminerDialog
from mufinder.app.convergence_types import SweepPointRunState
from mufinder.persistence.codec import to_jsonable


def _completed_state(energy: float, force: float) -> SweepPointRunState:
    return SweepPointRunState(status="completed", total_energy=energy, max_force=force)


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


@pytest.mark.parametrize("size", [(1080, 900), (880, 640)], ids=["default", "minimum"])
def test_convergence_examiner_shows_the_whole_plot_without_scrolling(qapp, size) -> None:
    dialog = _analysed_dialog(qapp, size)
    try:
        assert (dialog.minimumWidth(), dialog.minimumHeight()) == (880, 640)
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


def test_convergence_examiner_legends_stay_inside_their_panels_at_the_minimum_size(qapp) -> None:
    # The fullest legends: seven cutoffs fit a limit, and stress adds three force entries.
    dialog = ConvergenceExaminerDialog(
        {
            cutoff: SweepPointRunState(
                status="completed", total_energy=energy, max_force=force, max_stress=stress
            )
            for cutoff, energy, force, stress in (
                (300.0, -5620.2, 0.31, 0.9),
                (400.0, -5621.1, 0.2, 0.5),
                (500.0, -5621.5, 0.12, 0.3),
                (600.0, -5621.72, 0.07, 0.2),
                (700.0, -5621.78, 0.04, 0.12),
                (800.0, -5621.79, 0.035, 0.1),
                (900.0, -5621.795, 0.033, 0.09),
            )
        },
        num_atoms=6,
    )
    try:
        dialog.show()
        dialog.resize(880, 640)
        _settle(qapp)
        dialog._include_stress.setChecked(True)
        dialog._run_button.click()
        _settle(qapp)
        dialog._canvas.draw()
        renderer = dialog._canvas.get_renderer()
        assert len(dialog._stress_ax.get_legend().get_texts()) == 6
        force_legend = dialog._stress_ax.get_legend().get_window_extent(renderer)
        assert force_legend.height <= 0.45 * dialog._force_ax.get_window_extent(renderer).height
        for axes in (dialog._energy_ax, dialog._stress_ax):
            legend = axes.get_legend().get_window_extent(renderer)
            panel = axes.get_window_extent(renderer)
            assert legend.y0 >= panel.y0 - 1
            assert legend.y1 <= panel.y1 + 1
            assert legend.x0 >= panel.x0 - 1
            assert legend.x1 <= panel.x1 + 1
    finally:
        dialog.close()


def test_convergence_examiner_force_legend_draws_above_the_stress_axis(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            cutoff: SweepPointRunState(
                status="completed", total_energy=energy, max_force=force, max_stress=stress
            )
            for cutoff, energy, force, stress in (
                (300.0, -10.0, 0.30, 0.9),
                (400.0, -10.01, 0.20, 0.5),
                (500.0, -10.0105, 0.19, 0.45),
            )
        },
        num_atoms=2,
    )
    dialog._include_stress.setChecked(True)
    dialog._run_button.click()

    legend = dialog._stress_ax.get_legend()
    assert legend is not None
    assert dialog._force_ax.get_legend() is None
    # Axes draw in z-order, in figure order on a tie, so the legend's axes must come last.
    panels = sorted(dialog._figure.axes, key=lambda axes: axes.get_zorder())
    assert panels[-1] is dialog._stress_ax
    lines = dialog._stress_ax.get_lines()
    assert lines and legend.get_zorder() > max(line.get_zorder() for line in lines)
    # One row per quantity: filled by column, the force and stress entries alternate.
    labels = [text.get_text() for text in legend.get_texts()]
    assert labels[:2] == ["Max force", "Max stress"]


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


@pytest.mark.parametrize("size", [(1080, 900), (880, 640)], ids=["default", "minimum"])
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
    dialog = _analysed_dialog(qapp, (880, 640))
    try:
        results_height = dialog._results.height()
        canvas_height = dialog._canvas.height()
        dialog.resize(880, 840)
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
        dialog.resize(880, 640)
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
    assert dialog._force_ax.get_title() == "Force convergence (threshold 0.05 eV/Ang)"
    legend = [text.get_text() for text in dialog._energy_ax.get_legend().get_texts()]
    assert "Power-law fit" in legend
    assert "Fitted limit" in legend


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
    energy_ax = next(ax for ax in dialog._figure.axes if ax.get_ylabel() == "Energy (eV)")
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
    energy_ax = next(ax for ax in dialog._figure.axes if ax.get_ylabel() == "Energy (eV)")
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
    energy_ax = next(ax for ax in dialog._figure.axes if ax.get_ylabel() == "Energy (eV)")
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
    dialog._on_plot_hover(SimpleNamespace(inaxes=dialog._spacing_ax, xdata=8.0, ydata=-10.02))
    assert dialog._hover_annotation is not None
    assert "Grid: 2 x 2 x 2" in dialog._hover_annotation.get_text()
    assert dialog._hover_annotation.axes is dialog._energy_ax

    # Hover on the stress twin (top of the force subplot) — should still
    # report a force-subplot tooltip anchored on the force axis (kpoint
    # mode shows just the grid label).
    dialog._on_plot_hover(SimpleNamespace(inaxes=dialog._stress_ax, xdata=27.0, ydata=0.05))
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
    dialog._on_plot_hover(SimpleNamespace(inaxes=dialog._stress_ax, xdata=400.0, ydata=0.4))
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
    dialog._on_plot_hover(SimpleNamespace(inaxes=energy_ax, xdata=300.0, ydata=-10.0))
    assert dialog._hover_annotation is not None
    assert "Grid:" in dialog._hover_annotation.get_text()
    assert "Energy:" not in dialog._hover_annotation.get_text()

    dialog._on_plot_hover(SimpleNamespace(inaxes=force_ax, xdata=300.0, ydata=0.30))
    assert "Grid:" in dialog._hover_annotation.get_text()
    assert "Max force" not in dialog._hover_annotation.get_text()

    dialog._on_plot_hover(SimpleNamespace(inaxes=force_ax, xdata=10000.0, ydata=0.30))
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
    assert dialog._force_ax.get_title() == (
        "Force / stress convergence (thresholds 0.05 eV/Ang, 0.1 GPa)"
    )

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
        x, y = dialog._force_ax.transData.transform((400.0, 0.20))
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
    """CASTEP prints forces to 1e-5 eV/Ang and stresses to 1e-6 GPa (firstd.f90).

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
    assert list(energy_line.get_ydata()) == [-10.0, -10.01, -10.0105]
    assert math.isnan(force_line.get_ydata()[1])
