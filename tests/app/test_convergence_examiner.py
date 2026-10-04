"""Tests for cutoff convergence examiner dialog."""

from __future__ import annotations

from types import SimpleNamespace

from matplotlib.colors import to_hex
import pytest
from PySide6.QtWidgets import QScrollArea

import mufinder.app.convergence_examiner as convergence_examiner_module
from mufinder.app import theme as theme_module
from mufinder.app.convergence_examiner import ConvergenceExaminerDialog
from mufinder.app.convergence_types import CutoffRunState
from mufinder.persistence.codec import to_jsonable


def _completed_state(energy: float, force: float) -> CutoffRunState:
    return CutoffRunState(status="completed", total_energy=energy, max_force=force)


def test_convergence_examiner_disables_run_with_insufficient_points(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.3),
        },
        num_atoms=2,
    )

    assert dialog._run_button.isEnabled() is False
    assert "Need at least two" in dialog._results.toPlainText()


def test_convergence_examiner_uses_scrollable_layout_and_minimum_sizes(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.30),
            400.0: _completed_state(-10.01, 0.20),
            500.0: _completed_state(-10.0105, 0.19),
        },
        num_atoms=2,
    )

    scrolls = dialog.findChildren(QScrollArea)
    assert scrolls
    # Minimum size was lowered from 920x760 to 880x640 to fit a 13" MacBook's
    # usable desktop area; content still sits in a QScrollArea (asserted
    # above), so the dialog remains fully usable at the smaller minimum.
    assert dialog.minimumWidth() >= 880
    assert dialog.minimumHeight() >= 640
    assert dialog._canvas.minimumHeight() >= 520
    assert dialog._adopt_button.parentWidget() is not None


def test_convergence_examiner_shows_method_info(qapp, monkeypatch: pytest.MonkeyPatch) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.3),
            400.0: _completed_state(-10.01, 0.2),
        },
        num_atoms=2,
    )

    infos: list[str] = []
    monkeypatch.setattr(
        convergence_examiner_module.QMessageBox,
        "information",
        lambda _parent, _title, message: infos.append(message),
    )

    dialog._method_combo.setCurrentText("Derivative (dE/dEcut)")
    dialog._show_method_info()

    assert infos
    assert "dE/dEcut" in infos[-1]


def test_convergence_examiner_runs_analysis_and_enables_adopt(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.30),
            400.0: _completed_state(-10.01, 0.20),
            500.0: _completed_state(-10.0105, 0.19),
        },
        num_atoms=2,
    )

    dialog._method_combo.setCurrentText("Delta-Threshold")
    dialog._run_analysis()

    assert "Recommended cutoff" in dialog._results.toPlainText()
    assert dialog._adopt_button.isEnabled() is True


def test_convergence_examiner_emits_cutoff_adopted_signal(qapp) -> None:
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
    dialog.cutoff_adopted.connect(
        lambda value, method, energy_threshold, force_threshold, stress_threshold: adopted.append(
            (value, method, energy_threshold, force_threshold, stress_threshold)
        )
    )

    dialog._emit_adopted_cutoff()

    assert adopted
    assert adopted[-1][0] == pytest.approx(500.0)
    assert adopted[-1][1] == "delta-threshold"
    assert adopted[-1][2] == pytest.approx(0.001)
    assert adopted[-1][3] == pytest.approx(0.05)
    assert adopted[-1][4] is None


def test_convergence_examiner_disables_asymptotic_when_points_insufficient(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.3),
            400.0: _completed_state(-10.01, 0.2),
            500.0: _completed_state(-10.02, 0.18),
        },
        num_atoms=2,
    )

    index = dialog._method_combo.findText("Asymptotic Extrapolation")
    assert index >= 0
    assert dialog._method_combo.model().item(index).isEnabled() is False


def test_convergence_examiner_restores_persisted_results_and_plot(qapp) -> None:
    first = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.30),
            400.0: _completed_state(-10.01, 0.20),
            500.0: _completed_state(-10.0105, 0.19),
        },
        num_atoms=2,
    )
    first._method_combo.setCurrentText("Delta-Threshold")
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

    assert "Recommended cutoff" in second._results.toPlainText()
    assert second._adopt_button.isEnabled() is True
    assert second._recommended_cutoff == pytest.approx(first._recommended_cutoff)


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
    assert states[-1].get("recommended_cutoff") == pytest.approx(500.0)


def test_convergence_examiner_kpoint_mode_uses_delta_only_and_emits_default(qapp) -> None:
    dialog = ConvergenceExaminerDialog(
        {
            8.0: _completed_state(-10.0, 0.30),
            27.0: _completed_state(-10.01, 0.20),
            64.0: _completed_state(-10.0105, 0.19),
        },
        num_atoms=2,
        sweep_kind="kpoint",
        point_labels={8.0: "Grid: 2 x 2 x 2", 27.0: "Grid: 3 x 3 x 3", 64.0: "Grid: 4 x 4 x 4"},
    )

    derivative_index = dialog._method_combo.findText("Derivative (dE/dEcut)")
    asymptotic_index = dialog._method_combo.findText("Asymptotic Extrapolation")
    assert derivative_index >= 0
    assert asymptotic_index >= 0
    assert dialog._method_combo.model().item(derivative_index).isEnabled() is False
    assert dialog._method_combo.model().item(asymptotic_index).isEnabled() is False

    dialog._run_analysis()
    assert "Recommended k-point grid" in dialog._results.toPlainText()

    adopted: list[tuple[float, str, float, float, object]] = []
    dialog.default_adopted.connect(
        lambda value, method, energy_threshold, force_threshold, stress_threshold: adopted.append(
            (float(value), method, float(energy_threshold), float(force_threshold), stress_threshold)
        )
    )
    dialog._emit_adopted_cutoff()

    assert adopted


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
    labels = {
        counts[i]: f"Grid: {grids[i][0]} x {grids[i][1]} x {grids[i][2]}"
        for i in range(3)
    }

    dialog = ConvergenceExaminerDialog(
        sweep_data,
        num_atoms=2,
        sweep_kind="kpoint",
        point_labels=labels,
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
    energy_ax = next(
        ax for ax in dialog._figure.axes if ax.get_ylabel() == "Energy (eV)"
    )
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
        point_labels={8.0: "Grid: 2 x 2 x 2", 27.0: "Grid: 3 x 3 x 3", 64.0: "Grid: 4 x 4 x 4"},
    )
    assert dialog._point_spacings == {}

    dialog._run_analysis()

    spacing_axes = [
        ax for ax in dialog._figure.axes if ax.get_xlabel() == "K-point spacing (1/Ang)"
    ]
    assert spacing_axes == []
    # Log scale applies even without lattice vectors — it's the n_kpoints
    # crowding fix, independent of whether spacing labels are drawn.
    energy_ax = next(
        ax for ax in dialog._figure.axes if ax.get_ylabel() == "Energy (eV)"
    )
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
    energy_ax = next(
        ax for ax in dialog._figure.axes if ax.get_ylabel() == "Energy (eV)"
    )
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
            "energy_result": {"converged": True, "method": "delta-threshold", "details": "ok", "deltas": "bad"},
            "force_result": {"converged": True, "method": "delta-threshold", "details": "ok", "fitted_curve": "bad"},
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
            300.0: convergence_examiner_module.SweepPointRunState(status="completed", total_energy=-10.0, max_force=None),
            400.0: convergence_examiner_module.SweepPointRunState(status="completed", total_energy=-10.02, max_force=None),
            500.0: convergence_examiner_module.SweepPointRunState(status="completed", total_energy=None, max_force=0.20),
            600.0: convergence_examiner_module.SweepPointRunState(status="completed", total_energy=None, max_force=0.18),
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
        1.0: CutoffRunState(status="completed", total_energy=-10.0, max_force=0.30, max_stress=2.0),
        8.0: CutoffRunState(status="completed", total_energy=-10.02, max_force=0.18, max_stress=0.50),
        27.0: CutoffRunState(status="completed", total_energy=-10.025, max_force=0.17, max_stress=0.05),
    }
    dialog = ConvergenceExaminerDialog(
        sweep_data,
        num_atoms=2,
        sweep_kind="kpoint",
        point_labels={1.0: "Grid: 1 x 1 x 1", 8.0: "Grid: 2 x 2 x 2", 27.0: "Grid: 3 x 3 x 3"},
        lattice_vectors=lattice,
    )
    dialog._include_stress.setChecked(True)
    dialog._run_analysis()

    assert dialog._spacing_ax is not None
    assert dialog._stress_ax is not None

    # Hover on the spacing twin (top of the energy subplot) — should still
    # report an energy-subplot tooltip anchored on the energy axis.
    dialog._on_plot_hover(
        SimpleNamespace(inaxes=dialog._spacing_ax, xdata=8.0, ydata=-10.02)
    )
    assert dialog._hover_annotation is not None
    assert "Grid: 2 x 2 x 2" in dialog._hover_annotation.get_text()
    assert dialog._hover_annotation.axes is dialog._energy_ax

    # Hover on the stress twin (top of the force subplot) — should still
    # report a force-subplot tooltip anchored on the force axis (kpoint
    # mode shows just the grid label).
    dialog._on_plot_hover(
        SimpleNamespace(inaxes=dialog._stress_ax, xdata=27.0, ydata=0.05)
    )
    assert "Grid: 3 x 3 x 3" in dialog._hover_annotation.get_text()
    assert dialog._hover_annotation.axes is dialog._force_ax


def test_convergence_examiner_hover_combines_force_and_stress_in_cutoff_mode(qapp) -> None:
    # When the cursor is over the stress twin axis in cutoff mode, the
    # annotation should be anchored on the stress axis (so the arrow points
    # at the stress data point, not the force line) and show both metrics.
    sweep_data = {
        300.0: CutoffRunState(status="completed", total_energy=-10.0, max_force=0.30, max_stress=1.5),
        400.0: CutoffRunState(status="completed", total_energy=-10.01, max_force=0.20, max_stress=0.4),
        500.0: CutoffRunState(status="completed", total_energy=-10.011, max_force=0.05, max_stress=0.08),
    }
    dialog = ConvergenceExaminerDialog(sweep_data, num_atoms=2, sweep_kind="cutoff")
    dialog._include_stress.setChecked(True)
    dialog._run_analysis()

    assert dialog._stress_ax is not None
    dialog._on_plot_hover(
        SimpleNamespace(inaxes=dialog._stress_ax, xdata=400.0, ydata=0.4)
    )
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
        point_labels={300.0: "Grid: 3 x 10 x 10"},
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
    dialog._selected_method_for_recommendation = None
    dialog._energy_threshold_for_recommendation = None
    dialog._force_threshold_for_recommendation = None
    dialog._energy_threshold.setValue(0.002)
    dialog._force_threshold.setValue(0.07)

    defaults: list[tuple] = []
    cutoffs: list[tuple] = []
    dialog.default_adopted.connect(lambda value, method, et, ft, st: defaults.append((value, method, et, ft, st)))
    dialog.cutoff_adopted.connect(lambda value, method, et, ft, st: cutoffs.append((value, method, et, ft, st)))

    dialog._emit_adopted_cutoff()

    assert defaults == [(400.0, "delta-threshold", pytest.approx(0.002), pytest.approx(0.07), None)]
    assert cutoffs == [(400.0, "delta-threshold", pytest.approx(0.002), pytest.approx(0.07), None)]


def _completed_state_with_stress(energy: float, force: float, stress: float | None) -> CutoffRunState:
    return CutoffRunState(
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
    dialog._method_combo.setCurrentText("Delta-Threshold")

    dialog._run_analysis()

    assert dialog._stress_result is not None
    assert "Stress result" in dialog._results.toPlainText()

    adopted: list[tuple] = []
    dialog.cutoff_adopted.connect(lambda *args: adopted.append(args))
    dialog._emit_adopted_cutoff()

    assert adopted
    assert adopted[-1][-1] == pytest.approx(0.1)


# ---------------------------------------------------------------------------
# _compute_spacings — duplicate spacing suppression (frozen n=1 direction)
# ---------------------------------------------------------------------------


def test_compute_spacings_suppresses_duplicate_spacings_from_dominant_axis() -> None:
    # Hexagonal AlF3 (a=4.872, c=12.466 Å). Under the round-trip formula the
    # c-axis at N_c=3 contributes 1/(2*c)*1/(N_c-1)·|b_c| ≈ 0.0401, while the
    # ab axes only beat that bound once N_ab ≥ 7. So (2,2,3), (3,3,3), (4,4,3)
    # all report the same spacing — c dominates the min for each. The first
    # occurrence is kept; the rest are suppressed to avoid a non-decreasing
    # annotation series that would mislead about convergence.
    import math as _math

    a, c = 4.872, 12.466
    lattice = [[a, 0.0, 0.0], [-a / 2.0, a * _math.sqrt(3) / 2.0, 0.0], [0.0, 0.0, c]]
    grids = {12.0: (2, 2, 3), 27.0: (3, 3, 3), 48.0: (4, 4, 3)}

    spacings = ConvergenceExaminerDialog._compute_spacings(lattice, grids)

    # (2,2,3): first occurrence of the c-dominated bound → include
    assert 12.0 in spacings
    # (3,3,3) and (4,4,3): same bound, c still dominates → suppress
    assert 27.0 not in spacings
    assert 48.0 not in spacings


# ---------------------------------------------------------------------------
# recompute_spacings_in_snapshot — stale value replacement
# ---------------------------------------------------------------------------


def test_recompute_spacings_in_snapshot_replaces_stale_values() -> None:
    from mufinder.science.kpoint_grids import spacing_from_grid

    a = 4.0
    lattice = [[a, 0.0, 0.0], [0.0, a, 0.0], [0.0, 0.0, a]]
    snapshot: dict = {
        "point_labels": {"8.0": "Grid: 2 x 2 x 2", "27.0": "Grid: 3 x 3 x 3"},
        "plot_data": {
            "cutoffs": [8.0, 27.0],
            "energies": [-10.0, -10.01],
            # Deliberately wrong ("stale") spacing values:
            "point_spacings": {"8.0": 999.0, "27.0": 999.0},
        },
    }

    refreshed = ConvergenceExaminerDialog.recompute_spacings_in_snapshot(
        snapshot, lattice
    )

    expected_8 = spacing_from_grid(lattice, (2, 2, 2))
    expected_27 = spacing_from_grid(lattice, (3, 3, 3))
    assert refreshed["plot_data"]["point_spacings"][8.0] == pytest.approx(expected_8)
    assert refreshed["plot_data"]["point_spacings"][27.0] == pytest.approx(expected_27)
    # Original snapshot must not be mutated.
    assert snapshot["plot_data"]["point_spacings"] == {"8.0": 999.0, "27.0": 999.0}


def test_recompute_spacings_in_snapshot_falls_back_without_point_labels() -> None:
    # Snapshots that pre-date point_labels (or cutoff snapshots) are returned
    # as-is so the caller can still display them.
    snapshot: dict = {
        "plot_data": {
            "cutoffs": [300.0, 400.0],
            "point_spacings": {"300.0": 0.5},
        },
    }
    lattice = [[4.0, 0.0, 0.0], [0.0, 4.0, 0.0], [0.0, 0.0, 4.0]]

    result = ConvergenceExaminerDialog.recompute_spacings_in_snapshot(snapshot, lattice)

    # No point_labels → unchanged (but a copy, not the same object).
    assert result["plot_data"]["point_spacings"] == {"300.0": 0.5}
    assert result is not snapshot


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

    dialog._method_combo.setCurrentText("Delta-Threshold")
    dialog._run_analysis()

    text = dialog._results.toPlainText()
    assert "Recommended cutoff" not in text
    assert "No recommendation: the force criterion did not converge within the sweep" in text
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
        point_labels={8.0: "Grid: 2 x 2 x 2", 27.0: "Grid: 3 x 3 x 3", 64.0: "Grid: 4 x 4 x 4"},
        notes=("2 x 4 x 1 left out: same k-point count (8) as 2 x 2 x 2.",),
    )

    assert "Notes:\n- 2 x 4 x 1 left out" in dialog._results.toPlainText()
    dialog._run_analysis()

    text = dialog._adopt_button.text()
    assert text.startswith("Set ") and text.endswith(" as Project Default K-Point Grid")
    assert "Grid:" not in text
    results = dialog._results.toPlainText()
    assert "Recommended k-point grid: Grid:" not in results
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
    dialog._method_combo.setCurrentText("Delta-Threshold")

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


def test_convergence_examiner_shows_the_clamped_threshold_it_analysed_with(qapp) -> None:
    """A threshold typed below the minimum is used as the minimum, so show the minimum.

    Otherwise tightening an already-minimal threshold left ``0.0001`` on screen
    while the analysis ran with ``0.001`` and nothing visibly changed.
    """

    dialog = ConvergenceExaminerDialog(
        {
            300.0: _completed_state(-10.0, 0.30),
            400.0: _completed_state(-10.01, 0.20),
            500.0: _completed_state(-10.0105, 0.15),
        },
        num_atoms=2,
    )
    dialog._method_combo.setCurrentText("Delta-Threshold")
    dialog._force_threshold.setText("0.0001")

    dialog._run_analysis()

    assert dialog._force_threshold.text() == "0.001"
    assert "Force threshold: 0.001 eV/Ang" in dialog._results.toPlainText()
