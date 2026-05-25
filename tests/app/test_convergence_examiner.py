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
    assert dialog.minimumWidth() >= 920
    assert dialog.minimumHeight() >= 760
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
